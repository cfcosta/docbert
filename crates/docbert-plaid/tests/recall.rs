//! Gold-standard recall checks for the PLAID index.
//!
//! These tests build a real PLAID index over a synthetic corpus, then
//! compare its top-K results against a brute-force exact MaxSim
//! computation that doesn't go through any of our index data
//! structures. The closer PLAID's top-K is to the exhaustive top-K,
//! the better the index quality.
//!
//! Two flavours of "exact":
//!
//! 1. **Exact-on-decoded**: brute-force MaxSim against the *decoded*
//!    tokens stored in the index. This isolates the IVF prune step:
//!    if PLAID returns the same top-K as a decoded-but-unprobed scan,
//!    the prune isn't dropping good candidates.
//!
//! 2. **Exact-on-original**: brute-force MaxSim against the *original*
//!    (pre-quantization) tokens. This bounds end-to-end recall after
//!    accounting for both IVF pruning and codec quantization error.
//!
//! Both tests assert recall@K ≥ a threshold. Thresholds are
//! intentionally loose for this synthetic corpus; the point is to
//! catch large regressions, not pin specific values that drift with
//! every algorithm tweak.

use docbert_plaid::{
    distance::dot,
    index::{DocumentTokens, IndexParams, build_index},
    search::{SearchParams, search},
};
use rand::{Rng, SeedableRng, rngs::StdRng};

const DIM: usize = 32;

fn normalize(row: &mut [f32]) {
    let n = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for v in row {
        *v /= n;
    }
}

fn random_unit_vectors(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data: Vec<f32> = (0..n * dim)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    for row in data.chunks_exact_mut(dim) {
        normalize(row);
    }
    data
}

fn build_corpus(
    seed: u64,
    n_docs: usize,
    tokens_per_doc: usize,
) -> Vec<DocumentTokens> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_docs)
        .map(|i| {
            let inner = rng.random::<u64>();
            DocumentTokens {
                doc_id: i as u64,
                tokens: random_unit_vectors(inner, tokens_per_doc, DIM),
                n_tokens: tokens_per_doc,
            }
        })
        .collect()
}

/// Exhaustive MaxSim against a slice of token tensors (each
/// `n_tokens × dim`, flat). Returns ranked `(doc_id, score)` pairs,
/// highest score first.
fn exhaustive_maxsim(
    query: &[f32],
    docs: &[(u64, Vec<f32>)],
    dim: usize,
) -> Vec<(u64, f32)> {
    let mut results: Vec<(u64, f32)> = docs
        .iter()
        .map(|(doc_id, tokens)| {
            let mut score = 0.0f32;
            for q in query.chunks_exact(dim) {
                let best = tokens
                    .chunks_exact(dim)
                    .map(|d| dot(q, d))
                    .fold(f32::NEG_INFINITY, f32::max);
                if best.is_finite() {
                    score += best;
                }
            }
            (*doc_id, score)
        })
        .collect();
    results.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    results
}

fn recall_at_k(plaid: &[u64], exact: &[u64]) -> f32 {
    if exact.is_empty() {
        return 1.0;
    }
    let plaid_set: std::collections::HashSet<u64> =
        plaid.iter().copied().collect();
    let hits = exact.iter().filter(|id| plaid_set.contains(id)).count();
    hits as f32 / exact.len() as f32
}

#[test]
fn plaid_search_recall_at_10_against_decoded_brute_force_is_high() {
    // Configuration: small enough to exhaustively brute-force, large
    // enough that the IVF actually prunes (k_centroids ≪ n_docs).
    let n_docs = 200;
    let tokens_per_doc = 40;
    let query_tokens = 16;
    let top_k = 10;
    let n_probe = 8;

    let corpus = build_corpus(0xCAFE, n_docs, tokens_per_doc);
    let index = build_index(
        &corpus,
        IndexParams {
            dim: DIM,
            nbits: 4,
            k_centroids: 32,
            max_kmeans_iters: 20,
        },
    )
    .unwrap();

    // Decoded-tokens snapshot: what the index "thinks" each doc looks
    // like after residual quantization.
    let decoded_docs: Vec<(u64, Vec<f32>)> = index
        .doc_ids
        .iter()
        .zip(index.doc_tokens.iter())
        .map(|(doc_id, encoded)| {
            let mut tokens = Vec::with_capacity(encoded.len() * DIM);
            for ev in encoded {
                tokens.extend(index.codec.decode_vector(ev).unwrap());
            }
            (*doc_id, tokens)
        })
        .collect();

    let mut total_recall = 0.0f32;
    let n_queries = 5;
    for q_seed in 0..n_queries {
        let query =
            random_unit_vectors(0xBEEF + q_seed as u64, query_tokens, DIM);
        let plaid_top: Vec<u64> = search(
            &index,
            &query,
            SearchParams {
                top_k,
                n_probe,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        )
        .unwrap()
        .into_iter()
        .map(|r| r.doc_id)
        .collect();

        let exact_top: Vec<u64> = exhaustive_maxsim(&query, &decoded_docs, DIM)
            .into_iter()
            .take(top_k)
            .map(|(doc_id, _)| doc_id)
            .collect();

        total_recall += recall_at_k(&plaid_top, &exact_top);
    }
    let avg = total_recall / n_queries as f32;
    assert!(
        avg >= 0.85,
        "recall@10 vs decoded brute-force should be ≥ 0.85; got {avg:.3}",
    );
}

#[test]
fn plaid_search_recall_at_10_against_original_brute_force_is_meaningful() {
    // End-to-end recall: PLAID's top-K vs a brute-force on the
    // *original* (pre-quantization) tokens. Quantization error eats a
    // little recall here so the bar is lower than the decoded test
    // above. This still catches "the index returned wildly wrong docs".
    let n_docs = 200;
    let tokens_per_doc = 40;
    let query_tokens = 16;
    let top_k = 10;
    let n_probe = 8;

    let corpus = build_corpus(0xF00D, n_docs, tokens_per_doc);
    let index = build_index(
        &corpus,
        IndexParams {
            dim: DIM,
            nbits: 4,
            k_centroids: 32,
            max_kmeans_iters: 20,
        },
    )
    .unwrap();

    // Snapshot of original tokens, indexed by doc_id.
    let original_docs: Vec<(u64, Vec<f32>)> = corpus
        .iter()
        .map(|d| (d.doc_id, d.tokens.clone()))
        .collect();

    let mut total_recall = 0.0f32;
    let n_queries = 5;
    for q_seed in 0..n_queries {
        let query =
            random_unit_vectors(0xDEADBEEF + q_seed as u64, query_tokens, DIM);
        let plaid_top: Vec<u64> = search(
            &index,
            &query,
            SearchParams {
                top_k,
                n_probe,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        )
        .unwrap()
        .into_iter()
        .map(|r| r.doc_id)
        .collect();

        let exact_top: Vec<u64> =
            exhaustive_maxsim(&query, &original_docs, DIM)
                .into_iter()
                .take(top_k)
                .map(|(doc_id, _)| doc_id)
                .collect();

        total_recall += recall_at_k(&plaid_top, &exact_top);
    }
    let avg = total_recall / n_queries as f32;
    assert!(
        avg >= 0.70,
        "end-to-end recall@10 should be ≥ 0.70; got {avg:.3}",
    );
}

#[test]
fn plaid_search_score_matches_recomputed_maxsim_on_decoded_tokens() {
    // For every doc PLAID returns, the score it reports must equal
    // the MaxSim we'd compute by hand against the index's own decoded
    // tokens. This pins the scorer to its specification independently
    // of the IVF-prune behaviour above.
    let corpus = build_corpus(0x1234, 50, 30);
    let index = build_index(
        &corpus,
        IndexParams {
            dim: DIM,
            nbits: 4,
            k_centroids: 16,
            max_kmeans_iters: 20,
        },
    )
    .unwrap();

    let query = random_unit_vectors(0x9999, 12, DIM);
    let results = search(
        &index,
        &query,
        SearchParams {
            top_k: 5,
            n_probe: 8,
            n_candidate_docs: None,
            centroid_score_threshold: None,
        },
    )
    .unwrap();

    for r in &results {
        let doc_idx = index.position_of(r.doc_id).expect("doc present");
        let mut decoded_tokens = Vec::new();
        for ev in &index.doc_tokens[doc_idx] {
            decoded_tokens.extend(index.codec.decode_vector(ev).unwrap());
        }
        let recomputed = {
            let mut score = 0.0f32;
            for q in query.chunks_exact(DIM) {
                let best = decoded_tokens
                    .chunks_exact(DIM)
                    .map(|d| dot(q, d))
                    .fold(f32::NEG_INFINITY, f32::max);
                if best.is_finite() {
                    score += best;
                }
            }
            score
        };
        assert!(
            (recomputed - r.score).abs() < 1e-4,
            "doc {} score {} differs from manual MaxSim {}",
            r.doc_id,
            r.score,
            recomputed,
        );
    }
}

#[test]
fn plaid_search_returns_results_sorted_descending_by_score() {
    // Independent assertion: regardless of corpus, results must come
    // back in non-increasing score order. This is the contract the
    // search API documents and downstream rankers (e.g. RRF in
    // docbert-core) rely on.
    let corpus = build_corpus(0x5050, 80, 25);
    let index = build_index(
        &corpus,
        IndexParams {
            dim: DIM,
            nbits: 2,
            k_centroids: 16,
            max_kmeans_iters: 10,
        },
    )
    .unwrap();

    let query = random_unit_vectors(0x7777, 10, DIM);
    let results = search(
        &index,
        &query,
        SearchParams {
            top_k: 20,
            n_probe: 4,
            n_candidate_docs: None,
            centroid_score_threshold: None,
        },
    )
    .unwrap();
    assert!(results.len() >= 2);
    for pair in results.windows(2) {
        assert!(
            pair[0].score >= pair[1].score,
            "scores not monotonic: {pair:?}",
        );
    }
}

#[test]
fn centroid_interaction_shortlist_keeps_recall_high_against_no_shortlist() {
    // Centroid-interaction is an approximate intermediate ranker:
    // each doc's MaxSim is approximated by scoring only against the
    // centroids its tokens landed in, skipping residual decode. On
    // uniform-random synthetic tokens the centroids carry less
    // per-doc signal than on real text (which clusters meaningfully),
    // so recall on this corpus is a floor — real documents should do
    // materially better. The test's job is to catch regressions, not
    // pin a production-grade number.
    let n_docs = 200;
    let tokens_per_doc = 40;
    let query_tokens = 16;
    let top_k = 10;
    let n_probe = 8;
    // The two-stage cascade shortlists to `ndocs` (Stage 2) then
    // `ndocs/4` (Stage 3). Asking for Stage 2 to keep ~200% of the
    // corpus makes Stage 2 a no-op, so Stage 3 is the effective cut
    // at `ndocs / 4`.
    let ndocs = top_k * 40;

    let corpus = build_corpus(0xBEEF, n_docs, tokens_per_doc);
    let index = build_index(
        &corpus,
        IndexParams {
            dim: DIM,
            nbits: 4,
            k_centroids: 64,
            max_kmeans_iters: 20,
        },
    )
    .unwrap();

    let mut total_recall = 0.0f32;
    let n_queries = 5;
    for q_seed in 0..n_queries {
        let query =
            random_unit_vectors(0xA11CE + q_seed as u64, query_tokens, DIM);
        let no_shortlist: Vec<u64> = search(
            &index,
            &query,
            SearchParams {
                top_k,
                n_probe,
                n_candidate_docs: None,
                centroid_score_threshold: None,
            },
        )
        .unwrap()
        .into_iter()
        .map(|r| r.doc_id)
        .collect();
        let shortlisted: Vec<u64> = search(
            &index,
            &query,
            SearchParams {
                top_k,
                n_probe,
                n_candidate_docs: Some(ndocs),
                centroid_score_threshold: None,
            },
        )
        .unwrap()
        .into_iter()
        .map(|r| r.doc_id)
        .collect();
        total_recall += recall_at_k(&shortlisted, &no_shortlist);
    }
    let avg = total_recall / n_queries as f32;
    assert!(
        avg >= 0.40,
        "centroid-interaction shortlist recall should be ≥ 0.40; got {avg:.3}",
    );
}

#[test]
fn maxsim_score_for_unit_norm_query_is_at_most_q_times_one() {
    // Per query token, max dot ≤ 1 when both query and doc tokens are
    // unit-norm. Sum over Q query tokens ≤ Q. PLAID's reported scores
    // must respect that ceiling on unit-norm inputs.
    let corpus = build_corpus(0x2222, 30, 20);
    let index = build_index(
        &corpus,
        IndexParams {
            dim: DIM,
            nbits: 4,
            k_centroids: 8,
            max_kmeans_iters: 20,
        },
    )
    .unwrap();

    let query_len = 8;
    let query = random_unit_vectors(0x3333, query_len, DIM);
    let results = search(
        &index,
        &query,
        SearchParams {
            top_k: 5,
            n_probe: 8,
            n_candidate_docs: None,
            centroid_score_threshold: None,
        },
    )
    .unwrap();

    // Quantization can push decoded tokens slightly off the unit
    // sphere, so we allow a small slack above the strict |Q| ceiling.
    let ceiling = query_len as f32 + 0.5;
    for r in &results {
        assert!(
            r.score <= ceiling,
            "doc {} score {} exceeds Q={query_len} ceiling",
            r.doc_id,
            r.score,
        );
    }
}
