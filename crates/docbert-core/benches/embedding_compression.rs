//! Side-by-side bench: baseline vs token pooling.
//!
//! Answers the concrete question "what does the embedding path cost
//! with and without the compression primitive?" by running the
//! Clavié & Chaffin Ward-pooling scheme (arXiv 2409.14683, §2.1)
//! against a plain `tokens.to_vec()` baseline over the same
//! synthetic 519×128 ColBERT-shaped document.
//!
//! The `baseline_memcpy` arm is the per-document cost
//! `embed_documents_with` would pay if pooling were disabled — a
//! plain `tokens.to_vec()`. Every `pool_only` arm adds the pooling
//! work on top of that cost, so the delta is exactly what the
//! indexing path pays per document.

use std::{hint::black_box, num::NonZeroUsize};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use docbert_core::token_pool::pool_document_tokens;
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

fn bench_compression(c: &mut Criterion) {
    let doc = random_unit_document(0xCAFEBABE);

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

    group.finish();
}

criterion_group!(benches, bench_compression);
criterion_main!(benches);
