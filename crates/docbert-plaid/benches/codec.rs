//! Benchmarks for the residual quantization codec.
//!
//! `encode_vector` runs at every-token-of-every-document scale during
//! index build; `decode_vector` runs once per candidate token at search
//! time. `train_quantizer` runs once per index build but on every
//! residual, so its scaling matters too.

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
use docbert_plaid::codec::{EncodedVector, ResidualCodec, train_quantizer};

const DIM: usize = 128;
const K: usize = 256;

fn build_codec(nbits: u32) -> ResidualCodec {
    let centroids = shared::random_unit_vectors(0xC047, K, DIM);
    // Train cutoffs/weights on a representative sample of residuals.
    let sample: Vec<f32> = shared::random_unit_vectors(0xDA7A, 4096, DIM)
        .iter()
        .map(|v| v * 0.1)
        .collect();
    let (bucket_cutoffs, bucket_weights) = train_quantizer(sample, nbits);
    ResidualCodec {
        nbits,
        dim: DIM,
        centroids,
        bucket_cutoffs,
        bucket_weights,
    }
}

fn bench_encode_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec/encode_vector");
    let vectors = shared::random_unit_vectors(0xE1, 1024, DIM);
    for &nbits in &[2u32, 4] {
        let codec = build_codec(nbits);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("nbits={nbits}")),
            &nbits,
            |b, _| {
                let mut idx = 0usize;
                b.iter(|| {
                    let start = idx * DIM;
                    let v = &vectors[start..start + DIM];
                    idx = (idx + 1) % 1024;
                    codec.encode_vector(black_box(v))
                });
            },
        );
    }
    group.finish();
}

fn bench_decode_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec/decode_vector");
    let vectors = shared::random_unit_vectors(0xE2, 1024, DIM);
    for &nbits in &[2u32, 4] {
        let codec = build_codec(nbits);
        let encoded: Vec<EncodedVector> = vectors
            .chunks_exact(DIM)
            .map(|v| codec.encode_vector(v).unwrap())
            .collect();
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("nbits={nbits}")),
            &nbits,
            |b, _| {
                let mut idx = 0usize;
                b.iter(|| {
                    let ev = &encoded[idx];
                    idx = (idx + 1) % encoded.len();
                    codec.decode_vector(black_box(ev))
                });
            },
        );
    }
    group.finish();
}

fn bench_train_quantizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec/train_quantizer");
    for &n in &[1_024usize, 16_384, 131_072] {
        let residuals: Vec<f32> = shared::random_unit_vectors(0xE3, n, 1)
            .iter()
            .map(|v| v * 0.1)
            .collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| train_quantizer(black_box(residuals.clone()), 4));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_encode_vector,
    bench_decode_vector,
    bench_train_quantizer,
);
criterion_main!(benches);
