//! Micro-benchmarks for hierarchical Ward token pooling.
//!
//! Implements the method from Clavié & Chaffin (2024), *"Reducing the
//! Footprint of Multi-Vector Retrieval with Minimal Performance Impact
//! via Token Pooling"* (arXiv 2409.14683), and measures how long
//! [`docbert_core::token_pool::pool_document_tokens`] takes on
//! docbert-shaped inputs.
//!
//! The shape that matters: a single ColBERT document embedding tensor
//! at the `LateOn` default document length (519 tokens × 128 dims),
//! unit-normalised. The paper's §2.1 cost analysis hinges on the
//! per-document Ward-linkage step, so we sweep three pooling factors
//! (2, 3, 4) to see where the indexing-time overhead lands next to
//! today's "store-every-token" baseline.

use std::{hint::black_box, num::NonZeroUsize};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use docbert_core::token_pool::pool_document_tokens;
use rand::{RngExt, SeedableRng, rngs::StdRng};

/// Document-length dimensions drawn from the `LateOn` config.
const NUM_TOKENS: usize = 519;
const DIM: usize = 128;

/// Generate `NUM_TOKENS` L2-normalised random vectors of dimension
/// `DIM`, returned as a flat row-major `NUM_TOKENS × DIM` `f32` buffer.
/// Seeded so repeated runs compare like-for-like.
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

fn bench_token_pool(c: &mut Criterion) {
    let doc = random_unit_document(0xCAFE_BABE_u64);

    let mut group = c.benchmark_group("token_pool/519x128");
    group.sample_size(30);

    for &factor in &[2usize, 3, 4] {
        group.bench_with_input(
            BenchmarkId::from_parameter(factor),
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

criterion_group!(benches, bench_token_pool);
criterion_main!(benches);
