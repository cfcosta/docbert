//! Hierarchical token pooling for multi-vector (ColBERT-style) document
//! embeddings.
//!
//! Direct implementation of the index-time vector-count reduction scheme
//! described in Clavié & Chaffin (2024),
//! [_"Reducing the Footprint of Multi-Vector Retrieval with Minimal
//! Performance Impact via Token Pooling"_][paper] (arXiv 2409.14683).
//! The paper shows that a large fraction of ColBERT's per-document
//! token vectors are redundant; averaging the most similar ones together
//! at indexing time can halve storage with **zero** average retrieval
//! degradation, and two-thirds of vectors can go with <3% NDCG loss on
//! most BEIR datasets.
//!
//! # Algorithm (paper, §2.1 "Hierarchical clustering based pooling")
//!
//! 1. Compute the pairwise cosine distance between the document's token
//!    embeddings. This module takes a precomputed dot-product matrix
//!    from the caller; the indexing pipeline fills it in with a single
//!    batched GPU matmul across the whole batch, which is dramatically
//!    faster than recomputing it per-document on CPU.
//! 2. Run hierarchical agglomerative clustering with **Ward's linkage**
//!    (the paper evaluates sequential, k-means, and hierarchical Ward;
//!    Ward wins on their ablations, Tables 1 and 2).
//! 3. Cut the dendrogram to yield at most
//!    `ceil(num_tokens / pooling_factor) + 1` clusters, per the paper's
//!    exact formula in §2.1.
//! 4. Replace each cluster by the **mean** of its constituent token
//!    vectors.
//!
//! # Scope
//!
//! This module exposes a single function,
//! [`pool_document_tokens`], that operates on one document's flat
//! `[num_tokens, dim]` `f32` buffer plus a matching dot-product
//! matrix, and returns a new, shorter buffer. The indexing pipeline
//! always calls it (at the factor picked by
//! `docbert_core::embedding::TOKEN_POOL_FACTOR`) between
//! `encode_documents_with_lengths` and the `EmbeddingDb::batch_store`
//! write.
//!
//! No model changes, no query-time cost: the pooled vectors are
//! stored in `embeddings.db` exactly like the original per-token
//! embeddings, and every downstream consumer (PLAID index build,
//! MaxSim scoring) sees them as normal ColBERT multi-vectors.
//!
//! [paper]: https://arxiv.org/abs/2409.14683

use std::num::NonZeroUsize;

use kodama::{Method, linkage};

/// Pool one document's per-token embeddings into a smaller set of
/// cluster-mean vectors, using a pre-computed pairwise dot-product
/// matrix.
///
/// # Inputs
///
/// - `tokens` — flat row-major `[num_tokens, dim]` buffer of `f32`
///   values. The paper assumes ColBERT-style L2-normalised token
///   embeddings (so cosine distance is well-defined), which is what
///   `docbert-pylate::ColBERT::encode` produces.
/// - `num_tokens` — number of rows in `tokens`. Must equal
///   `tokens.len() / dim`.
/// - `dim` — embedding dimensionality. Must be non-zero.
/// - `pooling_factor` — the paper's compression factor. A factor of 2
///   halves the vector count; a factor of 3 keeps ~⅓. The paper
///   evaluates 2, 3, 4, 5, 6, and 8. Factor `1` is a no-op and
///   short-circuits.
/// - `dots` — row-major `[num_tokens, dot_row_stride]` buffer holding
///   `⟨token_i, token_j⟩` at index `i * dot_row_stride + j`. Only the
///   upper triangle (`j > i`) is read. Entries outside the
///   `num_tokens × num_tokens` sub-matrix are ignored.
/// - `dot_row_stride` — row stride for `dots`. When the caller passes
///   a doc-sized `num_tokens × num_tokens` matrix this equals
///   `num_tokens`; when slicing out of a padded `[batch, padded, padded]`
///   tensor produced by the batched GPU matmul this equals `padded`.
///
/// # Output
///
/// Returns `(pooled, new_num_tokens)`. `pooled` is a new row-major
/// `[new_num_tokens, dim]` buffer; `new_num_tokens` is `at most`
/// `ceil(num_tokens / pooling_factor) + 1`, following the paper's
/// formula in §2.1.
///
/// # Behaviour at boundaries
///
/// - `pooling_factor == 1`: returns `tokens` unchanged.
/// - `num_tokens <= 1`: nothing to merge; returns unchanged.
/// - `num_tokens <= pooling_factor`: all tokens collapse to a single
///   mean vector.
///
/// # Determinism
///
/// Given the same `(tokens, num_tokens, dim, pooling_factor, dots,
/// dot_row_stride)` the output is bit-identical. Ward's agglomerative
/// scheme is deterministic, and cluster labels are re-assigned in a
/// canonical pass (first appearance of an original observation), so
/// cluster *ordering* is also stable.
///
/// # Cost
///
/// `O(num_tokens²)` reads to build the condensed distance matrix
/// (skips the O(num_tokens² · dim) scalar inner-product scan that
/// would otherwise dominate), `O(num_tokens²)` memory for the
/// condensed form, and `O(num_tokens²)` time for Ward's linkage via
/// `kodama`. At the `LateOn` default `document_length = 519` this is
/// under a millisecond per document — see the `embedding_compression`
/// bench for the exact number.
pub fn pool_document_tokens(
    tokens: &[f32],
    num_tokens: usize,
    dim: usize,
    pooling_factor: NonZeroUsize,
    dots: &[f32],
    dot_row_stride: usize,
) -> (Vec<f32>, u32) {
    assert_eq!(
        tokens.len(),
        num_tokens * dim,
        "pool_document_tokens: tokens.len() must equal num_tokens * dim"
    );
    assert!(dim > 0, "pool_document_tokens: dim must be non-zero");
    assert!(
        dot_row_stride >= num_tokens,
        "pool_document_tokens: dot_row_stride must be >= num_tokens"
    );
    if num_tokens > 0 {
        let last = (num_tokens - 1) * dot_row_stride + (num_tokens - 1);
        assert!(
            dots.len() > last,
            "pool_document_tokens: dots buffer is too short for the requested sub-matrix"
        );
    }

    let factor = pooling_factor.get();

    // Short-circuits (paper §2.1: "at most" formulas below only bite
    // when they would reduce the count).
    if factor == 1 || num_tokens <= 1 {
        return (tokens.to_vec(), num_tokens as u32);
    }

    // Collapse-to-one case: when the document is shorter than the
    // pooling factor, every token joins a single cluster and the
    // pooled output is the component-wise mean.
    if num_tokens <= factor {
        return (mean_pool_all(tokens, num_tokens, dim), 1);
    }

    // Paper §2.1: "this method will result in at most
    // `initial token count / Pooling Factor + 1` clusters".
    let target_clusters = num_tokens
        .div_ceil(factor)
        .saturating_add(1)
        .min(num_tokens);

    let mut condensed =
        condensed_cosine_distance_matrix(dots, num_tokens, dot_row_stride);

    // Ward's linkage. `kodama::linkage` consumes (mutates) the
    // condensed distance buffer as working space; we own it.
    let dendro = linkage(&mut condensed, num_tokens, Method::Ward);

    // Cut the dendrogram to `target_clusters` by replaying its first
    // `num_tokens - target_clusters` merges. Kodama labels original
    // observations `0..num_tokens` and assigns each merged cluster
    // label `num_tokens + step_index`.
    let assignments = cut_dendrogram(&dendro, num_tokens, target_clusters);

    mean_pool_by_assignment(tokens, num_tokens, dim, &assignments)
}

/// Collapse every token row into one mean vector. The short-document
/// branch of [`pool_document_tokens`].
fn mean_pool_all(tokens: &[f32], num_tokens: usize, dim: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; dim];
    let scale = 1.0 / num_tokens as f32;
    for row in tokens.chunks_exact(dim) {
        for (acc, &v) in mean.iter_mut().zip(row.iter()) {
            *acc += v;
        }
    }
    for v in &mut mean {
        *v *= scale;
    }
    mean
}

/// Build the upper-triangular, row-major condensed cosine-distance
/// matrix that `kodama::linkage` expects, from a precomputed dot-product
/// buffer with the given row stride.
///
/// For L2-normalised inputs (which ColBERT outputs are), cosine
/// distance reduces to `1 - ⟨x, y⟩`. We still clamp to `[0, 2]` to be
/// robust to rounding that can push a pair slightly negative when
/// vectors are near-identical.
fn condensed_cosine_distance_matrix(
    dots: &[f32],
    num_tokens: usize,
    dot_row_stride: usize,
) -> Vec<f64> {
    let n_pairs = num_tokens * (num_tokens - 1) / 2;
    let mut out = Vec::with_capacity(n_pairs);
    for i in 0..num_tokens {
        let row_base = i * dot_row_stride;
        for j in (i + 1)..num_tokens {
            let dot = dots[row_base + j];
            let dist = (1.0_f32 - dot).clamp(0.0, 2.0);
            out.push(dist as f64);
        }
    }
    out
}

/// Replay the first `num_tokens - target_clusters` merges of the
/// dendrogram to recover cluster membership. Returns, for each
/// original observation, its final cluster index in `0..target_clusters`,
/// numbered in first-appearance order so the output is stable across
/// platforms.
fn cut_dendrogram(
    dendro: &kodama::Dendrogram<f64>,
    num_tokens: usize,
    target_clusters: usize,
) -> Vec<usize> {
    // `members[label]` tracks which original observations belong to
    // cluster `label`. Original observations get labels 0..N, merged
    // clusters get labels N, N+1, ... matching kodama's convention.
    let mut members: Vec<Option<Vec<usize>>> =
        (0..num_tokens).map(|i| Some(vec![i])).collect();
    members.reserve(num_tokens.saturating_sub(1));

    let steps_to_apply = num_tokens.saturating_sub(target_clusters);
    for step in dendro.steps().iter().take(steps_to_apply) {
        let mut a = members[step.cluster1].take().expect(
            "kodama dendrogram step merged an already-consumed cluster",
        );
        let b = members[step.cluster2].take().expect(
            "kodama dendrogram step merged an already-consumed cluster",
        );
        a.extend(b);
        members.push(Some(a));
    }

    // Assign canonical labels in order of first original-observation
    // appearance so the same token set always maps to the same output
    // order, regardless of which merge path kodama picked.
    let mut out = vec![usize::MAX; num_tokens];
    let mut next_label = 0usize;
    for cluster in members.into_iter().flatten() {
        if cluster.is_empty() {
            continue;
        }
        let label = next_label;
        next_label += 1;
        for obs in cluster {
            out[obs] = label;
        }
    }
    debug_assert!(
        next_label <= target_clusters,
        "cut_dendrogram produced more clusters than requested"
    );
    out
}

/// Mean-pool the `tokens` rows into one output vector per cluster,
/// ordered by the cluster label assigned in `assignments`.
fn mean_pool_by_assignment(
    tokens: &[f32],
    num_tokens: usize,
    dim: usize,
    assignments: &[usize],
) -> (Vec<f32>, u32) {
    let num_clusters = assignments
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    let mut sums = vec![0.0f32; num_clusters * dim];
    let mut counts = vec![0u32; num_clusters];

    for (obs, &cluster) in assignments.iter().enumerate().take(num_tokens) {
        let row = &tokens[obs * dim..(obs + 1) * dim];
        let sink = &mut sums[cluster * dim..(cluster + 1) * dim];
        for (acc, &v) in sink.iter_mut().zip(row.iter()) {
            *acc += v;
        }
        counts[cluster] += 1;
    }

    for (c, &count) in counts.iter().enumerate() {
        debug_assert!(count > 0, "cluster {c} has no members");
        let scale = 1.0 / count as f32;
        let row = &mut sums[c * dim..(c + 1) * dim];
        for v in row {
            *v *= scale;
        }
    }

    (sums, num_clusters as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("non-zero")
    }

    /// Build the full `num_tokens × num_tokens` dot-product matrix on
    /// the CPU — the slow scalar oracle the GPU matmul replaces in
    /// production. Shared by every test that wants to call
    /// `pool_document_tokens` without writing the dot loop inline.
    fn naive_dots(tokens: &[f32], num_tokens: usize, dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; num_tokens * num_tokens];
        for i in 0..num_tokens {
            for j in 0..num_tokens {
                let dot: f32 = tokens[i * dim..(i + 1) * dim]
                    .iter()
                    .zip(tokens[j * dim..(j + 1) * dim].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                out[i * num_tokens + j] = dot;
            }
        }
        out
    }

    /// Convenience wrapper: build packed dots with
    /// `dot_row_stride == num_tokens` and call `pool_document_tokens`.
    /// Keeps each test focused on the clustering invariant it's
    /// actually asserting rather than the dot-product plumbing.
    fn pool_packed(
        tokens: &[f32],
        num_tokens: usize,
        dim: usize,
        factor: NonZeroUsize,
    ) -> (Vec<f32>, u32) {
        let dots = naive_dots(tokens, num_tokens, dim);
        pool_document_tokens(tokens, num_tokens, dim, factor, &dots, num_tokens)
    }

    #[test]
    fn factor_one_is_identity() {
        let tokens: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let (out, new_n) = pool_packed(&tokens, 4, 3, nz(1));
        assert_eq!(new_n, 4);
        assert_eq!(out, tokens);
    }

    #[test]
    fn single_token_is_unchanged() {
        let tokens = vec![1.0, 2.0, 3.0];
        let (out, new_n) = pool_packed(&tokens, 1, 3, nz(2));
        assert_eq!(new_n, 1);
        assert_eq!(out, tokens);
    }

    #[test]
    fn short_document_collapses_to_single_mean() {
        // Two tokens, pooling factor 3 — both tokens collapse.
        let tokens = vec![1.0_f32, 2.0, 3.0, 4.0];
        let (out, new_n) = pool_packed(&tokens, 2, 2, nz(3));
        assert_eq!(new_n, 1);
        // Component-wise mean of [(1,2), (3,4)] is (2, 3).
        assert_eq!(out, vec![2.0, 3.0]);
    }

    #[test]
    fn duplicated_tokens_produce_rows_matching_originals() {
        // Paper-grounded invariant: when the input contains duplicate
        // tokens (zero cosine distance), Ward's linkage merges those
        // pairs first and the cluster mean is the original vector.
        // So every output row must match *some* base row within
        // rounding — the output is a multiset sampled from `base`,
        // though not necessarily a permutation (the paper's "+1" in
        // the cluster ceiling means Ward can stop one merge short,
        // leaving a duplicate unmerged and that base row therefore
        // appearing twice in the output).
        let base: Vec<[f32; 2]> = vec![
            [0.6, 0.8],  // norm 1 token A
            [0.0, 1.0],  // norm 1 token B
            [-0.6, 0.8], // norm 1 token C
        ];
        let mut duplicated: Vec<f32> = Vec::new();
        for row in &base {
            duplicated.extend_from_slice(row);
        }
        for row in &base {
            duplicated.extend_from_slice(row);
        }

        let (pooled, n) = pool_packed(&duplicated, 6, 2, nz(2));
        // Paper §2.1 ceiling: ⌈6/2⌉ + 1 = 4 clusters at most.
        assert!(n <= 4, "cluster count should respect the paper ceiling");
        assert!(n >= 1);

        for chunk in pooled.chunks_exact(2) {
            let matched = base.iter().any(|b| {
                (chunk[0] - b[0]).abs() < 1e-5 && (chunk[1] - b[1]).abs() < 1e-5
            });
            assert!(
                matched,
                "pooled row {chunk:?} does not match any base token in {base:?}"
            );
        }
    }

    #[test]
    fn output_size_matches_paper_formula() {
        // Paper §2.1: "at most initial token count / Pooling Factor + 1".
        let tokens: Vec<f32> = (0..10 * 4).map(|i| (i as f32).sin()).collect();
        let (_, n) = pool_packed(&tokens, 10, 4, nz(2));
        // 10 / 2 + 1 = 6 is the paper's ceiling.
        assert!(n <= 6);
        assert!(n >= 1);
    }

    #[test]
    fn determinism_same_input_same_output() {
        // Bitwise-stable given the same input — required for
        // reproducible indexing runs.
        let tokens: Vec<f32> = (0..20 * 4).map(|i| (i as f32).cos()).collect();
        let (a, an) = pool_packed(&tokens, 20, 4, nz(3));
        let (b, bn) = pool_packed(&tokens, 20, 4, nz(3));
        assert_eq!(an, bn);
        assert_eq!(a, b);
    }

    /// When the dots buffer comes from a padded tensor (the GPU
    /// emits `[batch, padded, padded]`), the pool helper must read
    /// only the top-left `num_tokens × num_tokens` sub-matrix and
    /// produce output byte-identical to the packed (stride ==
    /// num_tokens) case. Garbage in the padding slots would surface
    /// as cluster-structure drift immediately.
    #[test]
    fn padded_dot_stride_matches_packed_stride() {
        let num_tokens = 16usize;
        let dim = 4usize;
        let padded = 32usize;
        let tokens: Vec<f32> = (0..num_tokens * dim)
            .map(|i| ((i as f32) * 0.21).sin())
            .collect();

        // Build a padded `padded × padded` dots matrix with real
        // values in the top-left quadrant and obvious garbage (-99.0)
        // everywhere else. If the pool helper ever read outside its
        // advertised sub-matrix, the garbage would blow up Ward
        // clustering immediately.
        let packed_dots = naive_dots(&tokens, num_tokens, dim);
        let mut padded_dots = vec![-99.0f32; padded * padded];
        for i in 0..num_tokens {
            for j in 0..num_tokens {
                padded_dots[i * padded + j] = packed_dots[i * num_tokens + j];
            }
        }

        let (packed_out, packed_n) = pool_document_tokens(
            &tokens,
            num_tokens,
            dim,
            nz(2),
            &packed_dots,
            num_tokens,
        );
        let (padded_out, padded_n) = pool_document_tokens(
            &tokens,
            num_tokens,
            dim,
            nz(2),
            &padded_dots,
            padded,
        );
        assert_eq!(packed_n, padded_n);
        assert_eq!(packed_out, padded_out);
    }

    /// Property: `pool_document_tokens` never grows the vector count.
    /// Paper §2.1 defines pooling factor as a compression factor; the
    /// output must be no larger than the input.
    #[hegel::test(test_cases = 200)]
    fn prop_pool_never_grows_vector_count(tc: hegel::TestCase) {
        use hegel::generators as gs;

        let dim: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
        let num_tokens: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(64));
        let factor: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
        let tokens: Vec<f32> = tc.draw(
            gs::vecs(gs::floats::<f32>().min_value(-1.0).max_value(1.0))
                .min_size(num_tokens * dim)
                .max_size(num_tokens * dim),
        );

        let (_, new_n) = pool_packed(&tokens, num_tokens, dim, nz(factor));
        assert!(new_n as usize <= num_tokens);
        // And the paper's explicit ceiling from §2.1.
        if factor > 1 && num_tokens > factor {
            let ceiling = num_tokens.div_ceil(factor).saturating_add(1);
            assert!((new_n as usize) <= ceiling);
        }
    }

    /// Property: every pooled row is a convex combination (uniform
    /// mean) of some subset of the input rows, so each component of
    /// every output row lies within the component-wise `[min, max]`
    /// range of the input rows.
    ///
    /// This is a consequence of Clavié & Chaffin's mean-pool step
    /// (paper §2.1 "apply mean pooling"): averaging a set of reals is
    /// always bounded by their extremes, regardless of which clusters
    /// Ward's linkage happened to form. Holds for *any* real input,
    /// so it's a safe oracle for the fuzzing ranges we use here (no
    /// dependence on cluster-count formulas, no dependence on whether
    /// ties in Ward distance get broken one way or the other).
    #[hegel::test(test_cases = 200)]
    fn prop_pool_output_is_bounded_by_input_range(tc: hegel::TestCase) {
        use hegel::generators as gs;

        let dim: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
        let num_tokens: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(24));
        let factor: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
        let tokens: Vec<f32> = tc.draw(
            gs::vecs(gs::floats::<f32>().min_value(-1.0).max_value(1.0))
                .min_size(num_tokens * dim)
                .max_size(num_tokens * dim),
        );

        let (pooled, n_clusters) =
            pool_packed(&tokens, num_tokens, dim, nz(factor));

        // Component-wise min/max across the input rows is the oracle
        // envelope. Output rows, being cluster-wise means, must
        // stay inside it (modulo tiny f32 rounding).
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];
        for row in tokens.chunks_exact(dim) {
            for (d, &v) in row.iter().enumerate() {
                mins[d] = mins[d].min(v);
                maxs[d] = maxs[d].max(v);
            }
        }

        assert_eq!(pooled.len(), (n_clusters as usize) * dim);
        for row in pooled.chunks_exact(dim) {
            for (d, &v) in row.iter().enumerate() {
                let eps = 1e-5 * (mins[d].abs().max(maxs[d].abs()).max(1.0));
                assert!(
                    v >= mins[d] - eps && v <= maxs[d] + eps,
                    "pooled value {v} at dim {d} escaped input range [{}, {}]",
                    mins[d],
                    maxs[d]
                );
            }
        }
    }
}
