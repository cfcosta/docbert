//! End-to-end index-build benchmark — the path that's currently slow
//! enough to matter (`docbert sync` was sitting at this step for many
//! minutes on a real corpus).
//!
//! The numbers here let us point the optimization work at the right
//! place. As of writing, build cost is dominated by k-means assignment
//! (`assign_points` × `max_kmeans_iters`); replacing the nested loop
//! with a vectorized matmul should move the needle the most.
//!
//! Each scenario is held to a small criterion sample size so the run
//! finishes in minutes rather than hours; the per-iteration timing is
//! still the meaningful number to compare across implementations.

#[path = "shared.rs"]
mod shared;

use std::hint::black_box;

use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
use docbert_plaid::index::{IndexParams, build_index};

const DIM: usize = 128;

fn bench_build_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_index");

    // (n_docs, tokens_per_doc, k_centroids, max_kmeans_iters)
    // Total tokens scales from 5k to 100k — small enough to bench in
    // reasonable time, large enough that the k-means inner loop
    // dominates and reflects production behaviour.
    let scenarios: &[(usize, usize, usize, usize)] =
        &[(100, 50, 64, 5), (500, 100, 128, 5), (1_000, 100, 256, 5)];

    for &(n_docs, tokens, k, iters) in scenarios {
        let docs = shared::random_corpus(0xB1D, n_docs, tokens, DIM);
        let total_tokens = n_docs * tokens;
        group.throughput(Throughput::Elements(total_tokens as u64));
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "docs={n_docs},tokens={tokens},k={k},iters={iters}"
            )),
            &(n_docs, tokens, k, iters),
            |b, _| {
                b.iter(|| {
                    build_index(
                        black_box(&docs),
                        IndexParams {
                            dim: DIM,
                            nbits: 2,
                            k_centroids: k,
                            max_kmeans_iters: iters,
                        },
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_build_index);
criterion_main!(benches);
