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

use crate::{codec::EncodedVector, distance::squared_l2, index::Index};

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
/// For every query token we find the maximum squared-similarity with
/// any of the document's decoded tokens and sum those maxima. We use
/// negative squared L2 as the similarity here because the embeddings
/// live on the same unit-norm sphere in practice (ColBERT applies L2
/// normalization), so ordering by `-||q-d||²` agrees with ordering by
/// cosine similarity without the extra divisions.
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
            let sim = -squared_l2(q, d);
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

    /// Two well-separated clusters of 2-D tokens in three documents:
    /// doc 1 lives near (0,0), doc 2 near (10,10), doc 3 straddles both.
    fn corpus() -> Vec<DocumentTokens> {
        vec![
            DocumentTokens {
                doc_id: 1,
                tokens: vec![0.0, 0.0, 0.1, 0.2, -0.1, 0.1],
                n_tokens: 3,
            },
            DocumentTokens {
                doc_id: 2,
                tokens: vec![10.0, 10.0, 10.2, 9.9, 9.8, 10.1],
                n_tokens: 3,
            },
            DocumentTokens {
                doc_id: 3,
                tokens: vec![0.3, -0.2, 9.7, 10.2],
                n_tokens: 2,
            },
        ]
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
        // Query near (0,0). Doc 1 sits exactly in that cluster, doc 3 has
        // a partial overlap, doc 2 is in the other cluster.
        let index = build_index(&corpus(), params());
        let out = search(
            &index,
            &[0.0, 0.0, 0.05, 0.1],
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
            &[0.0, 0.0],
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
        let out = search(
            &index,
            &[0.0, 0.0, 10.0, 10.0],
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
            &[10.0, 10.0],
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
            &[0.0, 0.0],
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
    #[should_panic(expected = "top_k must be positive")]
    fn search_panics_on_zero_top_k() {
        let index = build_index(&corpus(), params());
        let _ = search(
            &index,
            &[0.0, 0.0],
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
            &[0.0, 0.0],
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
            &[0.0, 0.0, 1.0],
            SearchParams {
                top_k: 1,
                n_probe: 1,
            },
        );
    }
}
