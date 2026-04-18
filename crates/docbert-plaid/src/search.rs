//! Query-time search over a built [`Index`].
//!
//! The flow mirrors PLAID's reference implementation:
//!
//! 1. For every query token, find the `n_probe` nearest coarse centroids.
//! 2. Union the IVF lists for those centroids to get a set of candidate
//!    documents whose tokens landed near some query token.
//! 3. For each candidate document, decode its stored codes back into
//!    approximate embeddings and compute MaxSim against the query.
//! 4. Sort documents by MaxSim score and return the top-`top_k`.
//!
//! Step 3 is intentionally exact (decode then MaxSim) for now. It already
//! shrinks the search from "every document" to "documents reachable via
//! probed centroids", which is the majority of PLAID's speedup. Further
//! acceleration (centroid-lookup scoring, score-bounded early exit)
//! follows the same data structures and can be layered in later without
//! changing the public API.

use candle_core::Tensor;

use crate::{device::default_device, distance::dot, index::Index};

/// Tunable knobs for a single search call.
#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    /// Number of top-scoring documents to return.
    pub top_k: usize,
    /// Number of nearest centroids each query token probes.
    pub n_probe: usize,
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

    // 1-2. Gather the union of candidate doc indices whose tokens landed
    //      near some query token.
    let mut candidate_docs: Vec<bool> = vec![false; index.num_documents()];
    for query_token in query_tokens.chunks_exact(dim) {
        for centroid_id in
            top_n_centroids(query_token, &index.codec.centroids, dim, n_probe)
        {
            for tref in index.ivf.tokens_for_centroid(centroid_id) {
                candidate_docs[tref.doc_idx as usize] = true;
            }
        }
    }

    // 3. Score every candidate with one batched MaxSim matmul.
    let candidate_idxs: Vec<usize> = candidate_docs
        .iter()
        .enumerate()
        .filter_map(|(idx, &is_cand)| {
            (is_cand && !index.doc_tokens[idx].is_empty()).then_some(idx)
        })
        .collect();
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

    // 4. Rank by score (desc), tie-break by doc_id (asc) for determinism.
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

/// Score a batch of candidate docs against `query_tokens` with one
/// matmul instead of a per-doc Rust loop.
///
/// We pack all candidates into a single padded `[n_cands, max_doc_len,
/// dim]` tensor, materialise the query as `[n_q, dim]`, and compute
/// `docs.matmul(query.T)` once to get a `[n_cands, max_doc_len, n_q]`
/// score block. Padded positions get a large negative additive mask so
/// they can never win a max, then we max over `max_doc_len` (per-query
/// best doc token) and sum over `n_q` (Σ max — the standard MaxSim).
///
/// This matches the ColBERT/PLAID reference definition
/// (`S = Σ_i max_j q_i · d_j`) and is what fast-plaid does on its
/// `colbert_score_reduce` path. Decoding each doc's tokens stays
/// scalar (one centroid lookup + bucket-weight add per dim) — that
/// cost is small relative to the matmul and isn't the bottleneck.
fn batch_maxsim(
    query_tokens: &[f32],
    candidate_idxs: &[usize],
    index: &Index,
    dim: usize,
) -> Vec<(usize, f32)> {
    let n_q = query_tokens.len() / dim;

    // Decode every candidate's encoded tokens into a flat f32 vector,
    // record the per-doc lengths so we can build the mask. Decoding
    // stays scalar per-doc here: a tensor-batched decode trades the
    // per-token Rust loop for several extra `from_slice`/`to_vec1`
    // round trips and a u8→u32 conversion of the entire codes buffer,
    // which empirically (criterion bench) is 4–5× *slower* on both
    // CPU and CUDA backends for typical search candidate sizes.
    let mut decoded: Vec<Vec<f32>> = Vec::with_capacity(candidate_idxs.len());
    let mut max_len = 0usize;
    for &doc_idx in candidate_idxs {
        let mut tokens =
            Vec::with_capacity(index.doc_tokens[doc_idx].len() * dim);
        for ev in &index.doc_tokens[doc_idx] {
            tokens.extend(index.codec.decode_vector(ev));
        }
        max_len = max_len.max(tokens.len() / dim);
        decoded.push(tokens);
    }
    if max_len == 0 || n_q == 0 {
        return candidate_idxs.iter().map(|&i| (i, 0.0)).collect();
    }

    // Pack docs + mask. Padded slots stay at zero in `padded`, but the
    // mask drives them to −∞ in the score block before we take max.
    let n_c = candidate_idxs.len();
    let mut padded = vec![0.0f32; n_c * max_len * dim];
    let mut mask = vec![-1e9f32; n_c * max_len];
    for (i, tokens) in decoded.iter().enumerate() {
        let n_tok = tokens.len() / dim;
        let row_start = i * max_len * dim;
        padded[row_start..row_start + n_tok * dim].copy_from_slice(tokens);
        for j in 0..n_tok {
            mask[i * max_len + j] = 0.0;
        }
    }

    let device = default_device();
    // Allocate the docs as a 2-D `[n_c * max_len, dim]` tensor and the
    // query as `[n_q, dim]`. We then matmul against `query.T = [dim,
    // n_q]` to get a `[n_c * max_len, n_q]` score block, and reshape
    // back to `[n_c, max_len, n_q]`. Going through 2-D avoids candle's
    // matmul rank-broadcast restrictions and keeps the underlying GEMM
    // dispatch a single call.
    let docs_t = Tensor::from_vec(padded, (n_c * max_len, dim), device)
        .expect("batch_maxsim: docs tensor allocation failed");
    let q_t = Tensor::from_slice(query_tokens, (n_q, dim), device)
        .expect("batch_maxsim: query tensor allocation failed");
    let mask_t = Tensor::from_vec(mask, (n_c, max_len), device)
        .expect("batch_maxsim: mask tensor allocation failed");

    let q_transposed = q_t
        .t()
        .expect("query transpose")
        .contiguous()
        .expect("query contiguous");
    let scores_2d = docs_t
        .matmul(&q_transposed)
        .expect("batch_maxsim: docs × query.T matmul failed");
    let scores = scores_2d
        .reshape((n_c, max_len, n_q))
        .expect("batch_maxsim: score reshape failed");

    // Broadcast mask [n_c, max_len, 1] over the query dim and add it
    // to scores so padded positions can never win the next max.
    let masked = scores
        .broadcast_add(
            &mask_t
                .unsqueeze(2)
                .expect("mask unsqueeze")
                .broadcast_as((n_c, max_len, n_q))
                .expect("mask broadcast"),
        )
        .expect("batch_maxsim: mask broadcast_add failed");

    // max over doc tokens (dim 1) → [n_c, n_q]; then sum over q-tokens.
    let per_query_max = masked.max(1).expect("max over doc tokens");
    let totals = per_query_max.sum(1).expect("sum over query tokens");
    let totals_vec: Vec<f32> = totals.to_vec1().expect("totals to_vec1");

    candidate_idxs.iter().copied().zip(totals_vec).collect()
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
            },
        );
    }
}
