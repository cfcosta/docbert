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

use crate::{
    codec::EncodedVector,
    distance::{dot, squared_l2},
    index::Index,
};

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

    // 3. Score each candidate with exact MaxSim over decoded tokens.
    let mut scored: Vec<SearchResult> = Vec::new();
    for (doc_idx, is_candidate) in candidate_docs.iter().enumerate() {
        if !is_candidate {
            continue;
        }
        let encoded = &index.doc_tokens[doc_idx];
        if encoded.is_empty() {
            continue;
        }
        let score = maxsim_query_vs_encoded(query_tokens, encoded, index, dim);
        scored.push(SearchResult {
            doc_id: index.doc_ids[doc_idx],
            score,
        });
    }

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

/// Indices of the `n` centroids closest to `point`, nearest first.
///
/// This is the multi-centroid counterpart to
/// [`crate::kmeans::nearest_centroid`]. It's used by the search path to
/// build the probe set for each query token.
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
        .map(|(i, c)| (i, squared_l2(point, c)))
        .collect();
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.into_iter().take(n.min(k)).map(|(i, _)| i).collect()
}

/// MaxSim between a flat query tensor and a document's encoded tokens.
///
/// For every query token we find the maximum dot-product similarity
/// against any of the document's decoded tokens and sum those maxima.
/// This matches the ColBERT / PLAID reference definition
/// (`S = Σ_i max_j q_i · d_j`) so scores are directly comparable with
/// fast-plaid and the original ColBERT paper.
///
/// We use dot product rather than `-squared_l2`: for L2-normalized
/// query and doc tokens the two are rank-equivalent, but residual
/// quantization can push decoded doc tokens slightly off the unit
/// sphere, at which point only the dot product preserves the intended
/// scoring semantics.
fn maxsim_query_vs_encoded(
    query_tokens: &[f32],
    doc_encoded: &[EncodedVector],
    index: &Index,
    dim: usize,
) -> f32 {
    // Decode the document's token embeddings once so we don't redo the
    // work for every query token.
    let decoded: Vec<Vec<f32>> = doc_encoded
        .iter()
        .map(|e| index.codec.decode_vector(e))
        .collect();

    let mut total = 0.0f32;
    for q in query_tokens.chunks_exact(dim) {
        let mut best = f32::NEG_INFINITY;
        for d in &decoded {
            let sim = dot(q, d);
            if sim > best {
                best = sim;
            }
        }
        if best.is_finite() {
            total += best;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{DocumentTokens, IndexParams, build_index};

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
    fn top_n_centroids_returns_indices_in_ascending_distance_order() {
        // 3 centroids on a line. Point (4, 0) is closest to index 1, then
        // index 0, then index 2.
        let centroids = [0.0, 0.0, 5.0, 0.0, 12.0, 0.0];
        let out = top_n_centroids(&[4.0, 0.0], &centroids, 2, 3);
        assert_eq!(out, vec![1, 0, 2]);
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
