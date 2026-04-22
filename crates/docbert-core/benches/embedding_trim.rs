//! Compares two strategies for extracting per-doc embedding rows from
//! the padded `[batch, padded_tokens, dim]` tensor pylate returns:
//!
//! - `zero_scan`: the old implementation, which walked the tail of each
//!   doc's slice from the end and dropped rows whose every element was
//!   exactly `0.0`. Worst-case `O(padded_tokens · dim)` per doc.
//! - `length_slice`: the new implementation, which slices directly to
//!   `lengths[i] * dim` using per-doc token counts the encoder now
//!   reports. `O(lengths[i] · dim)` per doc, independent of padding.
//!
//! The workload mirrors what docbert actually embeds during `sync`: a
//! 128-doc submission batch, padded to the longest doc (519 tokens for
//! LateOn), 128-dim ColBERT output. A wide range of doc lengths is
//! generated so the tail-of-zeros the scan pays for is realistic — most
//! docs in a typical corpus are far shorter than the batch's longest.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::StdRng};

/// Representative docbert embedding batch shape.
const BATCH_SIZE: usize = 128;
const PADDED_TOKENS: usize = 519;
const DIM: usize = 128;

/// Build a synthetic `[BATCH_SIZE, PADDED_TOKENS, DIM]` buffer where each
/// doc has `lengths[b]` non-zero token rows followed by all-zero padding.
/// Returned along with the reference lengths so both strategies have the
/// same oracle.
fn build_padded_batch(seed: u64) -> (Vec<f32>, Vec<u32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0.0f32; BATCH_SIZE * PADDED_TOKENS * DIM];
    let mut lengths = Vec::with_capacity(BATCH_SIZE);
    for b in 0..BATCH_SIZE {
        // Mix short and long docs: ~70% of docs land in [20, 250] tokens,
        // the rest in [250, PADDED_TOKENS]. That gives the zero scan
        // plenty of padded tail to walk on most docs.
        let short = rng.random_range(0..10) < 7;
        let l = if short {
            rng.random_range(20..=250)
        } else {
            rng.random_range(250..=PADDED_TOKENS)
        };
        for t in 0..l {
            for d in 0..DIM {
                let base = b * PADDED_TOKENS * DIM + t * DIM + d;
                data[base] = (((b + t + d) as f32).sin()) * 0.25;
            }
        }
        lengths.push(l as u32);
    }
    (data, lengths)
}

/// Old: trim trailing all-zero rows per doc.
#[inline(never)]
fn trim_zero_scan(data: &[f32], dim: usize) -> Vec<Vec<f32>> {
    let doc_stride = PADDED_TOKENS * dim;
    (0..BATCH_SIZE)
        .map(|b| {
            let start = b * doc_stride;
            let end = start + doc_stride;
            let doc = &data[start..end];
            let mut cut = doc.len();
            while cut >= dim && doc[cut - dim..cut].iter().all(|&v| v == 0.0) {
                cut -= dim;
            }
            doc[..cut].to_vec()
        })
        .collect()
}

/// New: slice directly using reported lengths.
#[inline(never)]
fn trim_length_slice(
    data: &[f32],
    dim: usize,
    lengths: &[u32],
) -> Vec<Vec<f32>> {
    let doc_stride = PADDED_TOKENS * dim;
    (0..BATCH_SIZE)
        .map(|b| {
            let start = b * doc_stride;
            let take = usize::min(lengths[b] as usize, PADDED_TOKENS) * dim;
            data[start..start + take].to_vec()
        })
        .collect()
}

fn bench_trim(c: &mut Criterion) {
    let (data, lengths) = build_padded_batch(0x0D0C_BE71_u64);

    // Sanity check both strategies produce the same output on this
    // input. If the bench ever drifts away from correctness we want
    // criterion to fail fast at startup.
    let a = trim_zero_scan(&data, DIM);
    let b = trim_length_slice(&data, DIM, &lengths);
    assert_eq!(a, b, "bench setup: strategies disagree");

    let mut group = c.benchmark_group("embedding_trim");
    group.bench_function("zero_scan_128x519x128", |bencher| {
        bencher.iter(|| trim_zero_scan(black_box(&data), black_box(DIM)));
    });
    group.bench_function("length_slice_128x519x128", |bencher| {
        bencher.iter(|| {
            trim_length_slice(
                black_box(&data),
                black_box(DIM),
                black_box(&lengths),
            )
        });
    });
    group.finish();
}

criterion_group!(benches, bench_trim);
criterion_main!(benches);
