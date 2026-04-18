//! Query-time search over a built [`Index`].
//!
//! The flow mirrors PLAID's reference implementation:
//!
//! 1. For every query token, find the `n_probe` top dot-product coarse
//!    centroids (matching `S_c,q = C · Q^T`). This is paper Stage 1.
//! 2. *Optional* centroid pruning: drop probed centroids whose best
//!    score across all query tokens sits below
//!    `centroid_score_threshold`. This is used both here (to shrink the
//!    reachable centroid set before postings are gathered) and below,
//!    at token level, when scoring `D̃`.
//! 3. Union the doc-level IVF postings for the surviving centroids to
//!    get a set of candidate documents.
//! 4. *Optional* centroid interaction. When `n_candidate_docs` is set,
//!    the cascade runs in two refinement passes:
//!    - Stage 2 (paper §4.3): approximate MaxSim with the pruned mask
//!      applied at token level; keep top `n_candidate_docs` (paper's
//!      `ndocs`).
//!    - Stage 3 (paper §4.2): approximate MaxSim without pruning on
//!      the Stage 2 survivors; keep top `n_candidate_docs / 4` (paper's
//!      `ndocs/4` empirical heuristic), clamped to at least `top_k`.
//! 5. Decode the survivors' stored residual codes into approximate
//!    embeddings and compute exact MaxSim against the query.
//! 6. Sort by exact MaxSim score and return the top-`top_k`.
//!
//! The pruning and interaction stages share a precomputed
//! `[n_query_tokens, n_centroids]` query-centroid score matrix that
//! step 1 would conceptually compute anyway, and together they
//! drastically shrink the set of candidates reaching the expensive
//! decode in step 5. Callers that want the legacy behaviour (every
//! probed candidate decodes, no pruning) can pass
//! `n_candidate_docs = None` and `centroid_score_threshold = None`.

use candle_core::Tensor;

use crate::{
    codec::DecodeTable,
    device::default_device,
    distance::dot,
    index::Index,
};

/// Tunable knobs for a single search call.
#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    /// Number of top-scoring documents to return.
    pub top_k: usize,
    /// Number of top dot-product centroids each query token probes.
    pub n_probe: usize,
    /// Maximum number of candidates surviving PLAID's
    /// centroid-interaction stage that proceed to full decode +
    /// exact MaxSim. `None` disables centroid interaction and sends
    /// every probed candidate straight to decode — correct but
    /// slower for large corpora.
    pub n_candidate_docs: Option<usize>,
    /// Minimum per-centroid score (max dot product against any query
    /// token) required for a centroid to contribute to candidate
    /// generation and centroid interaction. Centroids below this
    /// threshold are treated as unreachable for this query. `None`
    /// disables pruning, matching the legacy probe behaviour.
    pub centroid_score_threshold: Option<f32>,
}

/// One entry in a ranked search result list.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    pub doc_id: u64,
    pub score: f32,
}

/// Run a PLAID-style search over `index` and return the top-`top_k`
/// documents ranked by ColBERT MaxSim against `query_tokens`.
///
/// `query_tokens` is a flat row-major `n_query_tokens × dim` buffer.
/// Results are sorted by score, highest first. Ties are broken by
/// `doc_id` ascending to keep the output deterministic.
///
/// # Panics
///
/// Panics if `query_tokens.len() % index.params.dim != 0`, if
/// `params.top_k == 0`, or if `params.n_probe == 0`.
pub fn search(
    index: &Index,
    query_tokens: &[f32],
    params: SearchParams,
) -> Vec<SearchResult> {
    let dim = index.params.dim;
    assert!(params.top_k > 0, "search: top_k must be positive");
    assert!(params.n_probe > 0, "search: n_probe must be positive");
    assert!(
        query_tokens.len().is_multiple_of(dim),
        "search: query length {} is not a multiple of dim {}",
        query_tokens.len(),
        dim,
    );

    if query_tokens.is_empty() || index.num_documents() == 0 {
        return Vec::new();
    }

    let n_centroids = index.codec.num_centroids();
    let n_probe = params.n_probe.min(n_centroids);

    // Precompute the per-centroid max dot product against any query
    // token when either pruning or centroid interaction is active.
    // Pruning uses it to build a pruned-centroid bitmask for use in
    // both candidate generation and centroid interaction; centroid
    // interaction uses the full qc_scores matrix for approximate
    // per-doc scoring.
    let need_qc_scores = params.centroid_score_threshold.is_some()
        || params.n_candidate_docs.is_some();
    let qc_scores_opt: Option<Vec<f32>> = need_qc_scores.then(|| {
        query_centroid_score_matrix(query_tokens, &index.codec.centroids, dim)
    });
    let pruned_mask: Option<Vec<bool>> =
        params.centroid_score_threshold.map(|threshold| {
            let per_cent = per_centroid_max_scores(
                qc_scores_opt.as_ref().expect("qc_scores precomputed"),
                query_tokens.len() / dim,
                n_centroids,
            );
            per_cent.iter().map(|&s| s < threshold).collect()
        });

    // 1-2. Gather the union of candidate doc indices reachable via the
    //      probed centroids. IVF postings are already deduplicated per
    //      doc, so each centroid contributes at most one write per doc.
    //      Centroid pruning drops probed centroids whose best per-query
    //      score sits below the caller's threshold — skipping them
    //      both shrinks the candidate set and saves work later in
    //      centroid interaction.
    let mut candidate_docs: Vec<bool> = vec![false; index.num_documents()];
    for query_token in query_tokens.chunks_exact(dim) {
        for centroid_id in
            top_n_centroids(query_token, &index.codec.centroids, dim, n_probe)
        {
            if let Some(mask) = pruned_mask.as_ref()
                && mask[centroid_id]
            {
                continue;
            }
            for &doc_idx in index.ivf.docs_for_centroid(centroid_id) {
                candidate_docs[doc_idx as usize] = true;
            }
        }
    }

    let mut candidate_idxs: Vec<usize> = candidate_docs
        .iter()
        .enumerate()
        .filter_map(|(idx, &is_cand)| {
            (is_cand && !index.doc_tokens[idx].is_empty()).then_some(idx)
        })
        .collect();

    // 3. Centroid interaction: when the caller set `n_candidate_docs`,
    //    cheaply rank candidates via the precomputed query-centroid
    //    score matrix and keep only the top survivors before paying
    //    for full decode + exact MaxSim.
    if let Some(n_stage2) = params.n_candidate_docs {
        let qc_scores = qc_scores_opt
            .as_ref()
            .expect("qc_scores precomputed when n_candidate_docs is set");
        let n_q = query_tokens.len() / dim;
        let n_c = index.codec.num_centroids();

        // Stage 2 — cheap pruned centroid interaction. Keep top
        // `n_stage2` candidates (the paper's `ndocs`).
        candidate_idxs = shortlist_by_approx_score(
            candidate_idxs,
            index,
            qc_scores,
            n_q,
            n_c,
            pruned_mask.as_deref(),
            n_stage2,
        );

        // Stage 3 — unpruned centroid interaction. Refines the Stage 2
        // survivors down to `ndocs/4`, matching the paper's empirical
        // heuristic. Callers that want top-`k` results should set
        // `n_candidate_docs >= 4 * top_k` so Stage 3 doesn't undershoot
        // their requested result count. docbert-core's default already
        // does this (8 * top_k).
        let n_stage3 = n_stage2.div_ceil(4).max(1);
        candidate_idxs = shortlist_by_approx_score(
            candidate_idxs,
            index,
            qc_scores,
            n_q,
            n_c,
            None,
            n_stage3,
        );
    }

    // 5. Score every surviving candidate with one batched MaxSim matmul
    //    on decoded tokens.
    let mut scored: Vec<SearchResult> = if candidate_idxs.is_empty() {
        Vec::new()
    } else {
        batch_maxsim(query_tokens, &candidate_idxs, index, dim)
            .into_iter()
            .map(|(doc_idx, score)| SearchResult {
                doc_id: index.doc_ids[doc_idx],
                score,
            })
            .collect()
    };

    // 6. Rank by score (desc), tie-break by doc_id (asc) for determinism.
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    scored.truncate(params.top_k);
    scored
}

/// Indices of the `n` centroids with the highest dot-product against
/// `point`, most relevant first.
///
/// This is the multi-centroid counterpart to
/// [`crate::kmeans::nearest_centroid`]. It implements PLAID's
/// query-centroid scoring `S_c,q = C · Q^T` (one row per query token),
/// which is the ranking the paper uses for candidate generation.
///
/// Dot product is used here instead of squared L2 because the paper and
/// ColBERT's downstream MaxSim both operate on dot-product similarity.
/// For centroids with varying magnitudes the two metrics disagree, and
/// dot product is what keeps probe ordering consistent with the scorer.
/// Ties are broken toward the earlier centroid index for determinism.
///
/// The output vector has length `min(n, num_centroids)`.
///
/// # Panics
///
/// Panics on any shape mismatch between `point`, `centroids`, and `dim`.
pub fn top_n_centroids(
    point: &[f32],
    centroids: &[f32],
    dim: usize,
    n: usize,
) -> Vec<usize> {
    assert_eq!(
        point.len(),
        dim,
        "top_n_centroids: point length {} does not match dim {}",
        point.len(),
        dim,
    );
    assert!(dim > 0, "top_n_centroids: dim must be positive");
    assert!(
        !centroids.is_empty() && centroids.len().is_multiple_of(dim),
        "top_n_centroids: centroids length {} is not a positive multiple of dim {}",
        centroids.len(),
        dim,
    );

    let k = centroids.len() / dim;
    let mut scored: Vec<(usize, f32)> = centroids
        .chunks_exact(dim)
        .enumerate()
        .map(|(i, c)| (i, dot(point, c)))
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.into_iter().take(n.min(k)).map(|(i, _)| i).collect()
}

/// Compute the row-major `[n_q, n_centroids]` matrix of query-to-centroid
/// dot products used by PLAID's centroid-interaction stage.
///
/// Materialising the whole matrix once amortises the per-centroid work
/// across every candidate document — the same query-centroid entries
/// get hit by every doc that touches each centroid.
fn query_centroid_score_matrix(
    query_tokens: &[f32],
    centroids: &[f32],
    dim: usize,
) -> Vec<f32> {
    let n_q = query_tokens.len() / dim;
    let n_c = centroids.len() / dim;
    let mut out = vec![0.0f32; n_q * n_c];
    for (qi, q) in query_tokens.chunks_exact(dim).enumerate() {
        for (ci, c) in centroids.chunks_exact(dim).enumerate() {
            out[qi * n_c + ci] = dot(q, c);
        }
    }
    out
}

/// Rank `candidate_idxs` by approximate centroid-interaction score
/// and keep the top `limit` survivors.
///
/// A no-op when the input is already smaller than `limit`. Used by
/// both Stage 2 (pruned) and Stage 3 (unpruned) of the paper's
/// centroid-interaction cascade; the only difference between the two
/// callers is the `mask` argument.
fn shortlist_by_approx_score(
    candidate_idxs: Vec<usize>,
    index: &Index,
    qc_scores: &[f32],
    n_q: usize,
    n_centroids: usize,
    mask: Option<&[bool]>,
    limit: usize,
) -> Vec<usize> {
    if candidate_idxs.len() <= limit {
        return candidate_idxs;
    }
    let mut approx: Vec<(usize, f32)> = candidate_idxs
        .into_iter()
        .map(|doc_idx| {
            let score = approx_centroid_interaction_score(
                &index.doc_tokens[doc_idx],
                qc_scores,
                n_q,
                n_centroids,
                mask,
            );
            (doc_idx, score)
        })
        .collect();
    approx.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    approx.truncate(limit);
    approx.into_iter().map(|(idx, _)| idx).collect()
}

/// For every centroid, return the max dot-product score achieved by
/// any query token against that centroid.
///
/// PLAID's centroid pruning uses this "best any-query score" to cheaply
/// rank centroids by query relevance: a centroid whose best score
/// across all query tokens is still small cannot usefully participate
/// in MaxSim, so it can be dropped without hurting top-K.
fn per_centroid_max_scores(
    qc_scores: &[f32],
    n_q: usize,
    n_centroids: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NEG_INFINITY; n_centroids];
    for q in 0..n_q {
        let row = &qc_scores[q * n_centroids..(q + 1) * n_centroids];
        for (c, &s) in row.iter().enumerate() {
            if s > out[c] {
                out[c] = s;
            }
        }
    }
    out
}

/// Approximate MaxSim for one candidate doc using only the centroids
/// the doc touches. This is PLAID's centroid-interaction score:
///
/// `Σ_q max_{c ∈ unique_centroids(doc)} qc_scores[q, c]`
///
/// Duplicate centroid ids within the doc are filtered implicitly: a
/// centroid that appears twice can't beat itself in the max, so the
/// running-max shortcut is correct without an explicit dedup pass.
///
/// When `pruned_mask` is `Some`, tokens whose centroid is marked
/// pruned are skipped before the max, matching PLAID §4.3's token-level
/// pruning of `D̃`. If every token in the doc is pruned the result is
/// `f32::NEG_INFINITY`, which sorts below every non-pruned doc in the
/// centroid-interaction shortlist.
fn approx_centroid_interaction_score(
    doc: &[crate::codec::EncodedVector],
    qc_scores: &[f32],
    n_q: usize,
    n_centroids: usize,
    pruned_mask: Option<&[bool]>,
) -> f32 {
    if doc.is_empty() {
        return 0.0;
    }
    // If every token's centroid is pruned, the doc has no D̃ rows left;
    // report −∞ so the caller's ranking drops it instead of a
    // deceptive 0.0.
    if let Some(mask) = pruned_mask
        && doc.iter().all(|ev| mask[ev.centroid_id as usize])
    {
        return f32::NEG_INFINITY;
    }
    let mut total = 0.0f32;
    for q in 0..n_q {
        let row = &qc_scores[q * n_centroids..(q + 1) * n_centroids];
        let mut best = f32::NEG_INFINITY;
        for ev in doc {
            if let Some(mask) = pruned_mask
                && mask[ev.centroid_id as usize]
            {
                continue;
            }
            let s = row[ev.centroid_id as usize];
            if s > best {
                best = s;
            }
        }
        if best.is_finite() {
            total += best;
        }
    }
    total
}

/// Score a batch of candidate docs against `query_tokens` with a
/// padding-free packed MaxSim, matching PLAID §4.5.
///
/// All candidate tokens are decoded into a single `[total_tokens, dim]`
/// buffer concatenated along the doc axis — no padding between docs,
/// no mask. A single GEMM produces `[total_tokens, n_q]` scores, and
/// per-doc MaxSim is a scalar max-then-sum over the slice of rows
/// belonging to each candidate. On uniform-length corpora this costs
/// roughly the same as the old padded path, but on ragged corpora
/// (short + long docs mixed) it avoids the `max_len - doc_len` wasted
/// rows and the `[n_c, max_len, n_q]` masking broadcast the old path
/// had to materialise.
///
/// Final scoring still matches the ColBERT/PLAID reference definition
/// `S = Σ_i max_j q_i · d_j`. Decoding uses the `DecodeTable` lookup
/// so each packed byte lands in the output buffer as a single table
/// read plus the centroid add.
fn batch_maxsim(
    query_tokens: &[f32],
    candidate_idxs: &[usize],
    index: &Index,
    dim: usize,
) -> Vec<(usize, f32)> {
    let n_q = query_tokens.len() / dim;

    let decode_table = DecodeTable::new(&index.codec);
    let total_tokens_bound: usize = candidate_idxs
        .iter()
        .map(|&i| index.doc_tokens[i].len())
        .sum();
    // Packed (concatenated) decoded tokens and per-doc row offsets.
    let mut packed: Vec<f32> = Vec::with_capacity(total_tokens_bound * dim);
    let mut offsets: Vec<usize> = Vec::with_capacity(candidate_idxs.len() + 1);
    offsets.push(0);
    for &doc_idx in candidate_idxs {
        for ev in &index.doc_tokens[doc_idx] {
            packed.extend(
                index.codec.decode_vector_with_table(ev, &decode_table),
            );
        }
        offsets.push(packed.len() / dim);
    }
    let total_tokens = packed.len() / dim;
    if total_tokens == 0 || n_q == 0 {
        return candidate_idxs.iter().map(|&i| (i, 0.0)).collect();
    }

    // Single `[total_tokens, dim] × [dim, n_q]` GEMM. No padding, no
    // mask — the packed layout means every row is a real token.
    let device = default_device();
    let docs_t = Tensor::from_vec(packed, (total_tokens, dim), device)
        .expect("batch_maxsim: docs tensor allocation failed");
    let q_t = Tensor::from_slice(query_tokens, (n_q, dim), device)
        .expect("batch_maxsim: query tensor allocation failed");
    let q_transposed = q_t
        .t()
        .expect("query transpose")
        .contiguous()
        .expect("query contiguous");
    let scores_flat: Vec<f32> = docs_t
        .matmul(&q_transposed)
        .expect("batch_maxsim: docs × query.T matmul failed")
        .to_vec2::<f32>()
        .expect("scores to_vec2")
        .into_iter()
        .flatten()
        .collect();

    // Per-doc MaxSim reduction over the doc's slice of score rows.
    let mut out = Vec::with_capacity(candidate_idxs.len());
    for (i, &doc_idx) in candidate_idxs.iter().enumerate() {
        let start = offsets[i];
        let end = offsets[i + 1];
        if start == end {
            out.push((doc_idx, 0.0));
            continue;
        }
        let mut total = 0.0f32;
        for q in 0..n_q {
            let mut best = f32::NEG_INFINITY;
            for t in start..end {
                let s = scores_flat[t * n_q + q];
                if s > best {
                    best = s;
                }
            }
            if best.is_finite() {
                total += best;
            }
        }
        out.push((doc_idx, total));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        distance::dot,
        index::{DocumentTokens, IndexParams, build_index},
    };

    fn params() -> IndexParams {
        IndexParams {
            dim: 2,
            nbits: 2,
            k_centroids: 2,
            max_kmeans_iters: 50,
        }
    }

    /// Two well-separated clusters of **unit-norm** 2-D tokens in three
    /// documents. This mirrors the real-world ColBERT invariant that
    /// token embeddings live on the unit sphere, which is what makes
    /// the dot-product MaxSim meaningful. Doc 1 points east, doc 2
    /// points north, doc 3 has one token in each cluster.
    fn corpus() -> Vec<DocumentTokens> {
        // Unit vectors by construction.
        let east_a = [1.0f32, 0.0];
        let east_b = normalize([0.98, 0.2]);
        let east_c = normalize([0.97, -0.24]);
        let north_a = [0.0f32, 1.0];
        let north_b = normalize([0.2, 0.98]);
        let north_c = normalize([-0.24, 0.97]);

        let mut doc1 = Vec::new();
        doc1.extend_from_slice(&east_a);
        doc1.extend_from_slice(&east_b);
        doc1.extend_from_slice(&east_c);

        let mut doc2 = Vec::new();
        doc2.extend_from_slice(&north_a);
        doc2.extend_from_slice(&north_b);
        doc2.extend_from_slice(&north_c);

        let mut doc3 = Vec::new();
        doc3.extend_from_slice(&east_a);
        doc3.extend_from_slice(&north_a);

        vec![
            DocumentTokens {
                doc_id: 1,
                tokens: doc1,
                n_tokens: 3,
            },
            DocumentTokens {
                doc_id: 2,
                tokens: doc2,
                n_tokens: 3,
            },
            DocumentTokens {
                doc_id: 3,
                tokens: doc3,
                n_tokens: 2,
            },
        ]
    }

    fn normalize(v: [f32; 2]) -> [f32; 2] {
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        [v[0] / norm, v[1] / norm]
    }

    #[test]
    fn top_n_centroids_ranks_by_descending_dot_product() {
        // Three centroids on the +x axis at magnitudes 0, 5, and 12.
        // Query at (4, 0). Dot products are 0, 20, 48; PLAID ranks by
        // descending relevance (`S_c,q = C · Q^T`), so the biggest
        // dot product wins.
        let centroids = [0.0, 0.0, 5.0, 0.0, 12.0, 0.0];
        let out = top_n_centroids(&[4.0, 0.0], &centroids, 2, 3);
        assert_eq!(out, vec![2, 1, 0]);
    }

    #[test]
    fn top_n_centroids_breaks_ties_toward_earlier_index() {
        // Zero query makes every dot product 0. The deterministic
        // tie-break is "earliest index wins".
        let centroids = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let out = top_n_centroids(&[0.0, 0.0], &centroids, 2, 3);
        assert_eq!(out, vec![0, 1, 2]);
    }

    #[test]
    fn top_n_centroids_on_unit_norm_centroids_picks_most_aligned() {
        // All centroids and the query are unit-norm. Dot product then
        // reduces to cosine similarity, so the best-aligned centroid
        // wins. Here (0.6, 0.8) is the closest direction to (1, 0).
        let centroids = [
            0.6, 0.8, //
            0.0, 1.0, //
            -1.0, 0.0, //
            1.0, 0.0,
        ];
        let out = top_n_centroids(&[1.0, 0.0], &centroids, 2, 2);
        assert_eq!(out, vec![3, 0]);
    }

    #[test]
    fn top_n_centroids_caps_at_num_centroids() {
        let centroids = [0.0, 0.0, 5.0, 0.0];
        let out = top_n_centroids(&[0.0, 0.0], &centroids, 2, 10);
        assert_eq!(out, vec![0, 1]);
    }

    #[test]
    fn search_returns_empty_for_empty_query() {
        let index = build_index(&corpus(), params());
        let out = search(
            &index,
            &[],
            SearchParams {
                top_k: 3,
                n_probe: 2,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
        assert!(out.is_empty());
    }

    #[test]
    fn search_ranks_matching_cluster_highest() {
        // Unit-norm query pointing east. Doc 1 has three east tokens,
        // doc 3 has one east and one north token, doc 2 is entirely
        // north. MaxSim should prefer doc 1.
        let index = build_index(&corpus(), params());
        let out = search(
            &index,
            &[1.0, 0.0],
            SearchParams {
                top_k: 3,
                n_probe: 2,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
        assert!(!out.is_empty(), "should surface at least one doc");
        assert_eq!(out[0].doc_id, 1, "closest doc should rank first");
    }

    #[test]
    fn search_respects_top_k() {
        let index = build_index(&corpus(), params());
        let out = search(
            &index,
            &[1.0, 0.0],
            SearchParams {
                top_k: 1,
                n_probe: 2,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn search_scores_are_non_increasing() {
        let index = build_index(&corpus(), params());
        // Two query tokens, one east, one north.
        let out = search(
            &index,
            &[1.0, 0.0, 0.0, 1.0],
            SearchParams {
                top_k: 3,
                n_probe: 2,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
        for pair in out.windows(2) {
            assert!(
                pair[0].score >= pair[1].score,
                "scores must be descending: {pair:?}"
            );
        }
    }

    #[test]
    fn search_with_single_probe_still_finds_the_right_cluster() {
        // With n_probe=1, only one centroid is probed per query token.
        // A query firmly inside one cluster should still surface its
        // corresponding document first.
        let index = build_index(&corpus(), params());
        let out = search(
            &index,
            &[0.0, 1.0],
            SearchParams {
                top_k: 1,
                n_probe: 1,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
        assert_eq!(out[0].doc_id, 2);
    }

    #[test]
    fn search_skips_documents_with_no_tokens() {
        let mut docs = corpus();
        docs.push(DocumentTokens {
            doc_id: 42,
            tokens: vec![],
            n_tokens: 0,
        });
        let index = build_index(&docs, params());
        let out = search(
            &index,
            &[1.0, 0.0],
            SearchParams {
                top_k: 10,
                n_probe: 2,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
        assert!(
            out.iter().all(|r| r.doc_id != 42),
            "empty doc should not appear in results"
        );
    }

    #[test]
    fn search_score_equals_sum_of_maxsim_over_decoded_tokens() {
        // Sanity: the score we return should match a ground-truth
        // MaxSim computed with the same decoded tokens, confirming our
        // per-token similarity is dot product (not -squared_l2).
        let index = build_index(&corpus(), params());
        let query = [1.0f32, 0.0];

        let out = search(
            &index,
            &query,
            SearchParams {
                top_k: 1,
                n_probe: index.codec.num_centroids(),
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
        assert!(!out.is_empty());

        let top = out[0];
        let doc_idx = index.position_of(top.doc_id).expect("doc present");
        let decoded: Vec<Vec<f32>> = index.doc_tokens[doc_idx]
            .iter()
            .map(|ev| index.codec.decode_vector(ev))
            .collect();

        // Standard MaxSim: Σ max_j q_i · d_j.
        let mut expected = 0.0f32;
        for q in query.chunks_exact(2) {
            let best = decoded
                .iter()
                .map(|d| dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max);
            expected += best;
        }
        assert!(
            (expected - top.score).abs() < 1e-5,
            "score {} differs from ground-truth MaxSim {}",
            top.score,
            expected,
        );
    }

    #[test]
    fn search_works_with_four_bit_residuals() {
        // Build an index with 4-bit residual quantization (16 buckets)
        // and confirm the MaxSim-ranked top result is still the
        // matching-cluster doc.
        let params_4bit = IndexParams {
            dim: 2,
            nbits: 4,
            k_centroids: 2,
            max_kmeans_iters: 50,
        };
        let index = build_index(&corpus(), params_4bit);
        let out = search(
            &index,
            &[1.0, 0.0],
            SearchParams {
                top_k: 1,
                n_probe: 2,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
        assert_eq!(out[0].doc_id, 1);
    }

    #[test]
    fn search_survives_large_synthetic_corpus_on_both_clusters() {
        // Bigger synthetic corpus with two distinct unit-norm clusters.
        // An east query must top-rank an east document; same for north.
        let mut docs = Vec::new();
        for i in 0..20 {
            let jitter = i as f32 * 0.003;
            let east = normalize([1.0 - jitter, 0.05 + jitter]);
            docs.push(DocumentTokens {
                doc_id: 100 + i,
                tokens: east.to_vec(),
                n_tokens: 1,
            });
            let north = normalize([0.05 + jitter, 1.0 - jitter]);
            docs.push(DocumentTokens {
                doc_id: 200 + i,
                tokens: north.to_vec(),
                n_tokens: 1,
            });
        }
        let index = build_index(&docs, params());

        let east = search(
            &index,
            &[1.0, 0.0],
            SearchParams {
                top_k: 5,
                n_probe: 2,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
        for r in &east {
            assert!(
                (100..120).contains(&r.doc_id),
                "east query should only surface east docs: got {}",
                r.doc_id,
            );
        }

        let north = search(
            &index,
            &[0.0, 1.0],
            SearchParams {
                top_k: 5,
                n_probe: 2,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
        for r in &north {
            assert!(
                (200..220).contains(&r.doc_id),
                "north query should only surface north docs: got {}",
                r.doc_id,
            );
        }
    }

    #[test]
    #[should_panic(expected = "top_k must be positive")]
    fn search_panics_on_zero_top_k() {
        let index = build_index(&corpus(), params());
        let _ = search(
            &index,
            &[1.0, 0.0],
            SearchParams {
                top_k: 0,
                n_probe: 1,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
    }

    #[test]
    #[should_panic(expected = "n_probe must be positive")]
    fn search_panics_on_zero_n_probe() {
        let index = build_index(&corpus(), params());
        let _ = search(
            &index,
            &[1.0, 0.0],
            SearchParams {
                top_k: 1,
                n_probe: 0,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
    }

    #[test]
    #[should_panic(expected = "query length")]
    fn search_panics_on_ragged_query() {
        let index = build_index(&corpus(), params());
        let _ = search(
            &index,
            &[1.0, 0.0, 0.5],
            SearchParams {
                top_k: 1,
                n_probe: 1,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
    }

    #[test]
    fn centroid_interaction_shortlist_caps_candidates_surviving_to_decode() {
        // With only one candidate allowed past centroid interaction we
        // should get exactly one doc in the results, and it must be the
        // best-scoring one under the approximate stage.
        let index = build_index(&corpus(), params());
        let out = search(
            &index,
            &[1.0, 0.0],
            SearchParams {
                top_k: 5,
                n_probe: index.codec.num_centroids(),
                n_candidate_docs: Some(1),
                centroid_score_threshold: None,
            },
        );
        assert_eq!(out.len(), 1, "shortlist of 1 must cap output at 1");
        assert_eq!(
            out[0].doc_id, 1,
            "east query should surface the east doc as the survivor",
        );
    }

    #[test]
    fn centroid_interaction_with_large_shortlist_matches_no_shortlist() {
        // A shortlist size at least equal to the number of candidates
        // is effectively no shortlisting — the centroid-interaction
        // stage picks everything through, so results must agree with
        // the legacy `None` path byte-for-byte.
        let index = build_index(&corpus(), params());
        let query = [1.0, 0.0, 0.0, 1.0];

        let legacy = search(
            &index,
            &query,
            SearchParams {
                top_k: 3,
                n_probe: index.codec.num_centroids(),
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
        let shortlisted = search(
            &index,
            &query,
            SearchParams {
                top_k: 3,
                n_probe: index.codec.num_centroids(),
                n_candidate_docs: Some(1_000),
                centroid_score_threshold: None,
            },
        );
        assert_eq!(legacy, shortlisted);
    }

    #[test]
    fn centroid_pruning_with_low_threshold_matches_no_pruning() {
        // A threshold below every attainable centroid score is a
        // no-op. For unit-norm centroids the dot product ceiling is
        // ~1, so using −1.0 guarantees every centroid survives.
        let index = build_index(&corpus(), params());
        let query = [1.0, 0.0, 0.0, 1.0];
        let unpruned = search(
            &index,
            &query,
            SearchParams {
                top_k: 3,
                n_probe: index.codec.num_centroids(),
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        );
        let pruned = search(
            &index,
            &query,
            SearchParams {
                top_k: 3,
                n_probe: index.codec.num_centroids(),
                n_candidate_docs: None,
                centroid_score_threshold: Some(-1.0),
            },
        );
        assert_eq!(unpruned, pruned);
    }

    #[test]
    fn two_stage_interaction_caps_survivors_to_ndocs_div_four() {
        // Paper §4.2/§4.3: Stage 2 shortlists to `ndocs`, Stage 3
        // refines to `ndocs/4`. With 20 synthetic east/north docs, a
        // Stage 2 shortlist of 16, and `top_k` well above the Stage 3
        // cap, we should see at most `16/4 = 4` docs in the final
        // result — the Stage 3 bound, not `top_k` nor Stage 2's 16.
        let mut docs = Vec::new();
        for i in 0..20 {
            let jitter = i as f32 * 0.003;
            let east = normalize([1.0 - jitter, 0.05 + jitter]);
            docs.push(DocumentTokens {
                doc_id: 100 + i,
                tokens: east.to_vec(),
                n_tokens: 1,
            });
        }
        let index = build_index(&docs, params());
        let out = search(
            &index,
            &[1.0, 0.0],
            SearchParams {
                top_k: 20,
                n_probe: index.codec.num_centroids(),
                n_candidate_docs: Some(16),
                centroid_score_threshold: None,
            },
        );
        assert!(
            out.len() <= 4,
            "Stage 3 must cap the decoded set at ndocs/4 = 4; got {}",
            out.len(),
        );
        assert!(
            !out.is_empty(),
            "Stage 3 should still return at least one result"
        );
    }

    #[test]
    fn approx_score_skips_tokens_whose_centroid_is_pruned() {
        // Paper §4.3: D̃ must only be comprised of tokens whose centroid
        // passes the t_cs threshold. This directly exercises that
        // behaviour on a controlled qc_scores matrix.
        //
        // 3 centroids, 2 query tokens. Centroid 0 is strong for q0 only,
        // centroid 1 is strong for q1 only, centroid 2 is mediocre for
        // both. A doc with tokens in centroids [0, 2] would normally
        // pull qc_scores[2] into the q1 max (because centroid 0 is weak
        // for q1 and centroid 2 at least has 0.4). Pruning centroid 2
        // drops that contribution and only centroid 0's qc values
        // remain.
        use crate::codec::EncodedVector;
        let n_q = 2;
        let n_c = 3;
        // qc_scores layout is row-major [n_q, n_c].
        // q0 row: centroid 0 = 0.9, centroid 1 = 0.0, centroid 2 = 0.4.
        // q1 row: centroid 0 = 0.0, centroid 1 = 0.9, centroid 2 = 0.4.
        let qc = [0.9, 0.0, 0.4, 0.0, 0.9, 0.4];
        let doc = [
            EncodedVector {
                centroid_id: 0,
                codes: vec![0, 0],
            },
            EncodedVector {
                centroid_id: 2,
                codes: vec![0, 0],
            },
        ];

        let unpruned =
            approx_centroid_interaction_score(&doc, &qc, n_q, n_c, None);
        // q0: max(0.9, 0.4) = 0.9; q1: max(0.0, 0.4) = 0.4. Total 1.3.
        assert!((unpruned - 1.3).abs() < 1e-5, "unpruned was {unpruned}");

        // Prune centroid 2 (per-centroid max 0.4 < t_cs=0.5).
        let pruned_mask = [false, false, true];
        let pruned = approx_centroid_interaction_score(
            &doc,
            &qc,
            n_q,
            n_c,
            Some(&pruned_mask),
        );
        // q0: max over {centroid 0} = 0.9; q1: max over {centroid 0} =
        // 0.0. Total 0.9.
        assert!((pruned - 0.9).abs() < 1e-5, "pruned was {pruned}");
    }

    #[test]
    fn approx_score_on_all_pruned_doc_is_negative_infinity() {
        // Every centroid in the doc is masked. The scorer must return a
        // score low enough that the doc loses to any non-empty doc in
        // the centroid-interaction sort.
        use crate::codec::EncodedVector;
        let qc = [0.5, 0.5];
        let doc = [EncodedVector {
            centroid_id: 0,
            codes: vec![0, 0],
        }];
        let mask = [true];
        let score =
            approx_centroid_interaction_score(&doc, &qc, 1, 1, Some(&mask));
        assert!(
            score == f32::NEG_INFINITY,
            "all-pruned doc should score -inf, got {score}",
        );
    }

    #[test]
    fn centroid_pruning_excludes_docs_whose_only_centroid_is_below_threshold() {
        // Build an index with two clusters. An east-pointing query
        // scores the east centroid near 1.0 and the north centroid
        // near 0.0. A threshold of 0.5 prunes the north centroid,
        // so doc 2 (pure-north) must be filtered out — but doc 3
        // (mixed east + north) still has an east token and survives.
        let index = build_index(&corpus(), params());
        let out = search(
            &index,
            &[1.0f32, 0.0],
            SearchParams {
                top_k: 5,
                n_probe: index.codec.num_centroids(),
                n_candidate_docs: None,
                centroid_score_threshold: Some(0.5),
            },
        );
        let ids: Vec<u64> = out.iter().map(|r| r.doc_id).collect();
        assert!(
            !ids.contains(&2),
            "pure-north doc must be pruned when only the east centroid survives the threshold: got {ids:?}",
        );
        assert!(
            ids.contains(&1),
            "east-cluster doc must still rank when its centroid survives: got {ids:?}",
        );
    }

    #[test]
    fn centroid_pruning_with_unreachable_threshold_drops_every_candidate() {
        // No centroid can score higher than ~1.0 against a unit-norm
        // query on unit-norm centroids. A threshold of 10.0 prunes
        // everything, so no docs survive.
        let index = build_index(&corpus(), params());
        let out = search(
            &index,
            &[1.0, 0.0],
            SearchParams {
                top_k: 3,
                n_probe: index.codec.num_centroids(),
                n_candidate_docs: None,
                centroid_score_threshold: Some(10.0),
            },
        );
        assert!(
            out.is_empty(),
            "every centroid below threshold should yield empty results",
        );
    }

    #[test]
    fn centroid_interaction_preserves_exact_maxsim_on_surviving_candidates() {
        // The final ranking stage still runs exact decoded MaxSim.
        // With a shortlist that keeps everyone, the top-1 score must
        // equal the MaxSim over decoded tokens for that doc.
        let index = build_index(&corpus(), params());
        let query = [1.0f32, 0.0];
        let out = search(
            &index,
            &query,
            SearchParams {
                top_k: 1,
                n_probe: index.codec.num_centroids(),
                n_candidate_docs: Some(1_000),
                centroid_score_threshold: None,
            },
        );
        let top = out[0];
        let doc_idx = index.position_of(top.doc_id).unwrap();
        let decoded: Vec<Vec<f32>> = index.doc_tokens[doc_idx]
            .iter()
            .map(|ev| index.codec.decode_vector(ev))
            .collect();
        let expected: f32 = query
            .chunks_exact(2)
            .map(|q| {
                decoded
                    .iter()
                    .map(|d| dot(q, d))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum();
        assert!((expected - top.score).abs() < 1e-5);
    }
}
