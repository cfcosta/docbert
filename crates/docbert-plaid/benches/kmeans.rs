//! Benchmarks for the k-means primitives.
//!
//! Scales are chosen to bracket realistic ColBERT workloads:
//!
//! - `dim = 128` matches ColBERT-v2 token embedding size.
//! - `k_centroids` of 256 / 1024 brackets fast-plaid's typical defaults.
//! - `n_points` scales from "tiny synthetic" up to "small personal
//!   corpus" (~50k tokens). Going much larger here makes a single
//!   sample take >1s and slows criterion to a crawl, but the trend the
//!   timings reveal — quadratic in `n × k × dim` — extrapolates
//!   directly to the millions-of-tokens regime where production
//!   indexes live.
//!
//! Use `cargo bench -p docbert-plaid --bench kmeans` to run.

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
use docbert_plaid::kmeans::{assign_points, fit, update_centroids};

const DIM: usize = 128;
const K: usize = 256;

fn bench_assign_points(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans/assign_points");
    let centroids = shared::random_unit_vectors(0xC047, K, DIM);
    for &n in &[1_000usize, 10_000, 50_000] {
        let points = shared::random_unit_vectors(0xDA7A, n, DIM);
        // Throughput in points: lets criterion print "points/s".
        group.throughput(Throughput::Elements(n as u64));
        group.sample_size(if n >= 50_000 { 10 } else { 50 });
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                assign_points(black_box(&points), black_box(&centroids), DIM)
            });
        });
    }
    group.finish();
}

fn bench_update_centroids(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans/update_centroids");
    let previous = shared::random_unit_vectors(0xC047, K, DIM);
    for &n in &[10_000usize, 50_000] {
        let points = shared::random_unit_vectors(0xDA7A, n, DIM);
        let centroids = shared::random_unit_vectors(0xC047, K, DIM);
        let assignments: Vec<usize> =
            assign_points(&points, &centroids, DIM).unwrap();
        group.throughput(Throughput::Elements(n as u64));
        group.sample_size(if n >= 50_000 { 10 } else { 30 });
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                update_centroids(
                    black_box(&points),
                    black_box(&assignments),
                    black_box(&previous),
                    DIM,
                )
            });
        });
    }
    group.finish();
}

fn bench_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans/fit");
    // `fit` runs assign + update for `max_iters` rounds. Even at 5k
    // points × K=256, a single sample is ~5 seconds, so we cap the
    // sample size aggressively here.
    for &(n, max_iters) in &[(1_000usize, 10usize), (5_000, 10)] {
        let points = shared::random_unit_vectors(0xF17, n, DIM);
        group.throughput(Throughput::Elements(n as u64 * max_iters as u64));
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::new("k=256", format!("n={n},iters={max_iters}")),
            &(n, max_iters),
            |b, &(_, _)| {
                b.iter(|| fit(black_box(&points), K, DIM, max_iters));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_assign_points,
    bench_update_centroids,
    bench_fit,
);
criterion_main!(benches);
