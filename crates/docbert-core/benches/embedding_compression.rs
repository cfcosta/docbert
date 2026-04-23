//! Side-by-side bench: baseline vs token pooling, with and without a
//! precomputed dot-product matrix.
//!
//! Answers two related questions about the embedding path:
//!
//! 1. What does Ward-pooling (Clavié & Chaffin, arXiv 2409.14683, §2.1)
//!    cost per document vs a `tokens.to_vec()` no-op?
//! 2. How much of that cost is the O(N²·D) cosine-distance scan — the
//!    part the `embed_documents_with` pipeline now computes once on
//!    the GPU via a batched `tokens @ tokens.T` matmul and then passes
//!    to [`pool_document_tokens_with_dots`]?
//!
//! Running all three arms on the same synthetic 519×128 document lets
//! us read the decomposition directly:
//!
//! - `baseline_memcpy` = trivial `tokens.to_vec()`.
//! - `pool_only/k` = raw CPU path (distance matrix + Ward + mean).
//! - `pool_with_dots/k` = post-GPU-matmul CPU work (Ward + mean only).
//!
//! `pool_only - pool_with_dots` is the distance-matrix cost the GPU
//! eliminates in production. `pool_with_dots - baseline_memcpy` is
//! what the embedding path actually pays per document.

use std::{hint::black_box, num::NonZeroUsize};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use docbert_core::token_pool::{
    pool_document_tokens,
    pool_document_tokens_with_dots,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

const NUM_TOKENS: usize = 519;
const DIM: usize = 128;

/// Generate a unit-norm random ColBERT-shaped document, seeded so
/// runs are comparable.
fn random_unit_document(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(NUM_TOKENS * DIM);
    for _ in 0..NUM_TOKENS {
        let mut row = vec![0.0f32; DIM];
        for v in &mut row {
            *v = rng.random::<f32>() * 2.0 - 1.0;
        }
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for v in &mut row {
            *v /= norm;
        }
        data.extend_from_slice(&row);
    }
    data
}

/// Build the full `NUM_TOKENS × NUM_TOKENS` dot-product matrix on the
/// CPU — what a batched GPU matmul would hand the pool helper in
/// production. Computed once outside the timed region so the
/// `pool_with_dots` arms measure only the post-matmul CPU work.
fn build_dots(tokens: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; NUM_TOKENS * NUM_TOKENS];
    for i in 0..NUM_TOKENS {
        for j in i..NUM_TOKENS {
            let dot: f32 = tokens[i * DIM..(i + 1) * DIM]
                .iter()
                .zip(tokens[j * DIM..(j + 1) * DIM].iter())
                .map(|(a, b)| a * b)
                .sum();
            out[i * NUM_TOKENS + j] = dot;
            out[j * NUM_TOKENS + i] = dot;
        }
    }
    out
}

fn bench_compression(c: &mut Criterion) {
    let doc = random_unit_document(0xCAFEBABE);
    let dots = build_dots(&doc);

    let mut group = c.benchmark_group("embedding_compression/519x128");
    group.sample_size(20);

    // --------------------------------------------------------------
    // Baseline: a direct `tokens.to_vec()`, the per-document cost
    // `embed_documents_with` would pay if pooling were disabled.
    // Every `pool_only` arm below adds work on top of this cost.
    // --------------------------------------------------------------
    group.bench_function("baseline_memcpy", |bencher| {
        bencher.iter(|| {
            let copy = black_box(&doc).to_vec();
            black_box(copy);
        });
    });

    // --------------------------------------------------------------
    // Pool only (Clavié & Chaffin §2.1). Shows the per-document cost
    // of the Ward-pooling primitive, swept over the paper-evaluated
    // factors. Factor 2 is the production default baked into
    // `embedding::TOKEN_POOL_FACTOR`; 3 and 4 are the next compression
    // settings the paper evaluates in §3.
    // --------------------------------------------------------------
    for &factor in &[2usize, 3, 4] {
        group.bench_with_input(
            BenchmarkId::new("pool_only", factor),
            &factor,
            |bencher, &factor| {
                let nz = NonZeroUsize::new(factor).unwrap();
                bencher.iter(|| {
                    let (pooled, n) = pool_document_tokens(
                        black_box(&doc),
                        black_box(NUM_TOKENS),
                        black_box(DIM),
                        black_box(nz),
                    );
                    black_box((pooled, n));
                });
            },
        );
    }

    // --------------------------------------------------------------
    // Pool with a precomputed dot-product matrix. This is what the
    // embedding pipeline pays in production: the GPU hands us the
    // dots via a batched `tokens @ tokens.T` matmul before pooling
    // runs on CPU, so the only CPU work left is building the
    // condensed distance matrix (N² reads), Ward linkage, dendrogram
    // cut, and mean-pool.
    // --------------------------------------------------------------
    for &factor in &[2usize, 3, 4] {
        group.bench_with_input(
            BenchmarkId::new("pool_with_dots", factor),
            &factor,
            |bencher, &factor| {
                let nz = NonZeroUsize::new(factor).unwrap();
                bencher.iter(|| {
                    let (pooled, n) = pool_document_tokens_with_dots(
                        black_box(&doc),
                        black_box(NUM_TOKENS),
                        black_box(DIM),
                        black_box(nz),
                        black_box(&dots),
                        black_box(NUM_TOKENS),
                    );
                    black_box((pooled, n));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_compression);
criterion_main!(benches);
