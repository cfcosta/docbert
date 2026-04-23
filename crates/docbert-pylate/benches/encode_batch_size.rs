//! Measures `ColBERT::encode` throughput at different inner batch
//! sizes on a real model (`lightonai/LateOn`), so the accelerator
//! batch-size default in `docbert-core` has an empirical footing.
//!
//! Target hardware is a 3060-class GPU with ~8 GB of VRAM usable
//! (12 GB card minus the desktop compositor and a bit of CUDA
//! runtime). The bench sweeps 32 / 64 / 96 / 128 so we can pick a
//! default that keeps throughput close to peak on this budget without
//! risking OOM in the wild.
//!
//! The bench is CUDA-only — without a GPU the comparison is moot,
//! and we don't want CI runs on CPU-only boxes spending minutes
//! hitting the CPU path. Skipped when `cuda` feature is off.

use std::{hint::black_box, time::Duration};

use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::Device;
    use docbert_pylate::ColBERT;

    use super::*;

    const MODEL_ID: &str = "lightonai/LateOn";
    const DOC_COUNT: usize = 192;
    const APPROX_TOKENS_PER_DOC: usize = 300;

    /// Build a corpus of synthetic documents. Each doc is a space-
    /// separated stream of pseudo-words whose tokenized length lands
    /// close to `APPROX_TOKENS_PER_DOC` so the model's activation
    /// memory per doc is representative of real usage.
    fn build_corpus() -> Vec<String> {
        // Roughly four chars per BPE token for English; padding on
        // both sides with a few extra words keeps us safely above
        // the target post-truncation.
        let target_chars = APPROX_TOKENS_PER_DOC * 4 + 64;
        let word = "docbert retrieval embedding bench corpus ";
        let repetitions = target_chars.div_ceil(word.len());
        let body = word.repeat(repetitions);
        (0..DOC_COUNT)
            .map(|i| format!("Document {i}. {body}"))
            .collect()
    }

    /// Load the model once with a given batch size. Fails fast if
    /// CUDA isn't actually available at runtime (e.g. a CI box that
    /// compiled the `cuda` feature but has no device).
    fn load_model(batch_size: usize) -> ColBERT {
        let device = Device::new_cuda(0).expect(
            "bench requires CUDA device 0; skip without --features cuda",
        );
        ColBERT::from(MODEL_ID)
            .with_device(device)
            .with_batch_size(batch_size)
            .try_into()
            .expect(
                "failed to load lightonai/LateOn; ensure the HF cache has it",
            )
    }

    pub fn bench_batch_size(c: &mut Criterion) {
        let corpus = build_corpus();

        let mut group = c.benchmark_group("encode_batch_size");
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(20));
        group.throughput(Throughput::Elements(corpus.len() as u64));

        for &batch in &[32usize, 64, 96, 128] {
            let mut model = load_model(batch);
            // Warm up CUDA kernels and the tokenizer path once before
            // criterion starts its sampling so the reported time
            // reflects steady-state throughput, not the first-call
            // initialisation cost.
            let _ = model
                .encode(&corpus[..corpus.len().min(32)], false)
                .expect("warmup encode");

            group.bench_with_input(
                BenchmarkId::from_parameter(batch),
                &batch,
                |bencher, _| {
                    bencher.iter(|| {
                        let out = model
                            .encode(black_box(&corpus), false)
                            .expect("encode");
                        black_box(out);
                    });
                },
            );
            drop(model);
        }

        group.finish();
    }
}

#[cfg(not(feature = "cuda"))]
fn bench_batch_size(_c: &mut Criterion) {
    eprintln!("encode_batch_size: skipping — cuda feature is off");
}

#[cfg(feature = "cuda")]
fn bench_batch_size(c: &mut Criterion) {
    cuda_impl::bench_batch_size(c);
}

criterion_group!(benches, bench_batch_size);
criterion_main!(benches);
