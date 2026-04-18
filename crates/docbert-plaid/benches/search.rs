//! End-to-end search benchmarks against pre-built indexes.
//!
//! `search` is the latency-sensitive path: it runs every time the user
//! types a query. The probe step (centroid distances + IVF gather)
//! tends to be cheap; the dominant work is decoding candidate doc
//! tokens and the per-doc MaxSim matmul.

#[path = "shared.rs"]
mod shared;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use docbert_plaid::{
    index::{Index, IndexParams, build_index},
    search::{SearchParams, search, top_n_centroids},
};

const DIM: usize = 128;
const QUERY_TOKENS: usize = 32; // typical ColBERT query length

fn make_index(seed: u64, n_docs: usize, tokens_per_doc: usize) -> Index {
    let docs = shared::random_corpus(seed, n_docs, tokens_per_doc, DIM);
    let total_tokens = n_docs * tokens_per_doc;
    let k = (total_tokens as f32).sqrt().ceil() as usize;
    build_index(
        &docs,
        IndexParams {
            dim: DIM,
            nbits: 2,
            k_centroids: k.max(2),
            max_kmeans_iters: 5,
        },
    )
}

fn bench_top_n_centroids(c: &mut Criterion) {
    let mut group = c.benchmark_group("search/top_n_centroids");
    for &k in &[256usize, 1_024, 4_096] {
        let centroids = shared::random_unit_vectors(0xC047, k, DIM);
        let query = shared::random_unit_vectors(0x9, 1, DIM);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("k={k}")),
            &k,
            |b, _| {
                b.iter(|| {
                    top_n_centroids(
                        black_box(&query),
                        black_box(&centroids),
                        DIM,
                        8,
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search/end_to_end");
    for &(n_docs, tokens) in &[(100usize, 50usize), (1_000, 100), (5_000, 100)]
    {
        let index = make_index(0x1DEC, n_docs, tokens);
        let query = shared::random_unit_vectors(0x9, QUERY_TOKENS, DIM);
        let params = SearchParams {
            top_k: 10,
            n_probe: 8,
            n_candidate_docs: None,
            centroid_score_threshold: None,
        };
        group.sample_size(if n_docs >= 5_000 { 20 } else { 50 });
        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "docs={n_docs},tokens={tokens}"
            )),
            &(n_docs, tokens),
            |b, _| {
                b.iter(|| {
                    search(
                        black_box(&index),
                        black_box(&query),
                        black_box(params),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_top_n_centroids, bench_search);
criterion_main!(benches);
