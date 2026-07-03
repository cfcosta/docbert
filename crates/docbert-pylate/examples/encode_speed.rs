//! Encoding speed + parity harness for optimization experiments.
//!
//! Measures `ColBERT::encode` document throughput on the production
//! model (`lightonai/GTE-ModernColBERT-v1`) with a deterministic
//! corpus of varied-length documents (the production indexing path is
//! varied-length, so this exercises the varlen/packed code paths, not
//! just the uniform fast path). Also supports dumping embeddings to a
//! file and comparing a later run against that dump, so optimization
//! experiments can prove numerical parity.
//!
//! Usage:
//!   encode_speed [--batch-size N] [--docs N] [--repeats N]
//!                [--uniform] [--queries]
//!                [--dump PATH | --compare PATH]
//!
//! Prints a single JSON object to stdout.

use std::{env, fs, time::Instant};

use candle_core::{Device, Tensor};
use docbert_pylate::ColBERT;

const MODEL_ID: &str = "lightonai/GTE-ModernColBERT-v1";

/// Deterministic xorshift so corpora are identical across runs/builds.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() as usize) % (hi - lo).max(1)
    }
}

const WORDS: &[&str] = &[
    "retrieval",
    "embedding",
    "tensor",
    "kernel",
    "attention",
    "gradient",
    "index",
    "query",
    "document",
    "cluster",
    "residual",
    "quantization",
    "throughput",
    "latency",
    "pipeline",
    "batch",
    "token",
    "vector",
    "similarity",
    "pooling",
];

/// Build `count` documents. Varied mode samples target token lengths
/// in [24, 320] (production chunks vary widely and often exceed the
/// model's 180-token document_length, exercising truncation); uniform
/// mode fixes ~300 tokens like the criterion bench.
fn build_corpus(count: usize, uniform: bool) -> Vec<String> {
    let mut rng = Rng(0x00C0FFEE);
    (0..count)
        .map(|i| {
            let target_tokens = if uniform { 300 } else { rng.range(24, 320) };
            // ~1.3 words per token for this vocab; overshoot slightly,
            // truncation trims the rest.
            let n_words = target_tokens * 3 / 4 + 4;
            let mut s = format!("Document {i}.");
            for _ in 0..n_words {
                s.push(' ');
                s.push_str(WORDS[rng.range(0, WORDS.len())]);
            }
            s
        })
        .collect()
}

fn build_queries(count: usize) -> Vec<String> {
    let mut rng = Rng(0xBADC0DE5);
    (0..count)
        .map(|i| {
            let mut s = format!("query {i}");
            for _ in 0..rng.range(3, 9) {
                s.push(' ');
                s.push_str(WORDS[rng.range(0, WORDS.len())]);
            }
            s
        })
        .collect()
}

struct Dump {
    dims: (usize, usize, usize),
    lengths: Vec<u32>,
    data: Vec<f32>,
}

fn write_dump(path: &str, emb: &Tensor, lengths: &[u32]) {
    let (b, t, d) = emb.dims3().expect("3d embeddings");
    let data: Vec<f32> = emb
        .to_dtype(candle_core::DType::F32)
        .expect("dtype")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("to_vec1");
    let mut bytes = Vec::with_capacity(24 + lengths.len() * 4 + data.len() * 4);
    for v in [b as u64, t as u64, d as u64] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for l in lengths {
        bytes.extend_from_slice(&l.to_le_bytes());
    }
    for v in &data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).expect("write dump");
}

fn read_dump(path: &str) -> Dump {
    let bytes = fs::read(path).expect("read dump");
    let u64_at = |o: usize| {
        u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap()) as usize
    };
    let (b, t, d) = (u64_at(0), u64_at(8), u64_at(16));
    let mut off = 24;
    let mut lengths = Vec::with_capacity(b);
    for _ in 0..b {
        lengths
            .push(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
        off += 4;
    }
    let mut data = Vec::with_capacity(b * t * d);
    while off < bytes.len() {
        data.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
        off += 4;
    }
    assert_eq!(data.len(), b * t * d, "dump payload size mismatch");
    Dump {
        dims: (b, t, d),
        lengths,
        data,
    }
}

/// Per-valid-token cosine similarity between current embeddings and a
/// reference dump, plus MaxSim score deltas for synthetic queries.
fn compare_with_dump(
    reference: &Dump,
    emb: &Tensor,
    lengths: &[u32],
    model: &ColBERT,
    queries: &Tensor,
) -> serde_json::Value {
    let (b, t, d) = emb.dims3().expect("3d embeddings");
    let (rb, rt, rd) = reference.dims;
    assert_eq!((b, d), (rb, rd), "batch/dim mismatch vs reference");
    let cur: Vec<f32> = emb
        .to_dtype(candle_core::DType::F32)
        .expect("dtype")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("to_vec1");

    let mut min_cos = f32::INFINITY;
    let mut sum_cos = 0f64;
    let mut max_abs = 0f32;
    let mut n_tokens = 0usize;
    for i in 0..b {
        assert_eq!(
            lengths[i], reference.lengths[i],
            "token length mismatch at doc {i}"
        );
        for j in 0..(lengths[i] as usize).min(t).min(rt) {
            let cur_row = &cur[(i * t + j) * d..(i * t + j + 1) * d];
            let ref_row =
                &reference.data[(i * rt + j) * d..(i * rt + j + 1) * d];
            let mut dot = 0f32;
            let mut na = 0f32;
            let mut nb = 0f32;
            for (a, r) in cur_row.iter().zip(ref_row) {
                dot += a * r;
                na += a * a;
                nb += r * r;
                max_abs = max_abs.max((a - r).abs());
            }
            let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
            min_cos = min_cos.min(cos);
            sum_cos += cos as f64;
            n_tokens += 1;
        }
    }

    // MaxSim deltas: score every query against every doc with both
    // embedding sets; report the largest absolute score difference and
    // whether any per-query ranking of the top document changed.
    let device = Device::Cpu;
    let ref_emb =
        Tensor::from_vec(reference.data.clone(), (rb, rt, rd), &device)
            .expect("ref tensor");
    let cur_emb = emb
        .to_dtype(candle_core::DType::F32)
        .expect("dtype")
        .to_device(&device)
        .expect("to cpu");
    let queries = queries
        .to_dtype(candle_core::DType::F32)
        .expect("dtype")
        .to_device(&device)
        .expect("to cpu");
    let ref_scores = model
        .similarity(&queries, &ref_emb)
        .expect("ref maxsim")
        .data;
    let cur_scores = model
        .similarity(&queries, &cur_emb)
        .expect("cur maxsim")
        .data;
    let mut max_score_delta = 0f32;
    let mut top1_changes = 0usize;
    for (rq, cq) in ref_scores.iter().zip(&cur_scores) {
        for (r, c) in rq.iter().zip(cq) {
            max_score_delta = max_score_delta.max((r - c).abs());
        }
        let top = |row: &[f32]| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap()
        };
        if top(rq) != top(cq) {
            top1_changes += 1;
        }
    }

    serde_json::json!({
        "compared_tokens": n_tokens,
        "min_token_cosine": min_cos,
        "mean_token_cosine": sum_cos / n_tokens.max(1) as f64,
        "max_abs_diff": max_abs,
        "max_maxsim_delta": max_score_delta,
        "top1_changes": top1_changes,
        "n_queries": ref_scores.len(),
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let has = |name: &str| args.iter().any(|a| a == name);

    let batch_size: usize = flag("--batch-size")
        .map(|v| v.parse().unwrap())
        .unwrap_or(64);
    let n_docs: usize =
        flag("--docs").map(|v| v.parse().unwrap()).unwrap_or(256);
    let repeats: usize =
        flag("--repeats").map(|v| v.parse().unwrap()).unwrap_or(5);
    let uniform = has("--uniform");
    let bench_queries = has("--queries");
    let dump_path = flag("--dump");
    let compare_path = flag("--compare");

    let device = Device::new_cuda(0).expect("CUDA device 0 required");
    let mut model: ColBERT = ColBERT::from(MODEL_ID)
        .with_device(device.clone())
        .with_batch_size(batch_size)
        .try_into()
        .expect("load model (needs GTE-ModernColBERT-v1 in HF cache)");

    let corpus = build_corpus(n_docs, uniform);
    let lengths = model
        .document_token_lengths(&corpus)
        .expect("token lengths");
    let total_tokens: u64 = lengths.iter().map(|&l| l as u64).sum();

    // Warmup (also builds cached masks/positions).
    let _ = model
        .encode(&corpus[..n_docs.min(64)], false)
        .expect("warmup");
    device.synchronize().expect("sync");

    let mut wall_ms = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let out = model.encode(&corpus, false).expect("encode");
        device.synchronize().expect("sync");
        wall_ms.push(start.elapsed().as_secs_f64() * 1e3);
        drop(out);
    }
    let best = wall_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = wall_ms.iter().sum::<f64>() / wall_ms.len() as f64;

    let mut result = serde_json::json!({
        "model": MODEL_ID,
        "batch_size": batch_size,
        "docs": n_docs,
        "uniform": uniform,
        "total_valid_tokens": total_tokens,
        "repeats": repeats,
        "wall_ms_best": best,
        "wall_ms_mean": mean,
        "docs_per_sec_best": n_docs as f64 / (best / 1e3),
        "tokens_per_sec_best": total_tokens as f64 / (best / 1e3),
    });

    if bench_queries {
        let queries = build_queries(32);
        // One-at-a-time, like interactive search.
        let _ = model.encode(&queries[..1], true).expect("warmup q");
        device.synchronize().expect("sync");
        let mut q_ms = Vec::with_capacity(queries.len());
        for q in &queries {
            let start = Instant::now();
            let out = model.encode(std::slice::from_ref(q), true).expect("q");
            device.synchronize().expect("sync");
            q_ms.push(start.elapsed().as_secs_f64() * 1e3);
            drop(out);
        }
        q_ms.sort_by(f64::total_cmp);
        result["query_ms_p50"] = q_ms[q_ms.len() / 2].into();
        result["query_ms_min"] = q_ms[0].into();
    }

    if dump_path.is_some() || compare_path.is_some() {
        // Fixed parity corpus, independent of --docs.
        let parity_corpus = build_corpus(64, false);
        let (emb, plens) = model
            .encode_documents_with_lengths(&parity_corpus)
            .expect("parity encode");
        if let Some(path) = dump_path {
            write_dump(&path, &emb, &plens);
            result["dumped"] = path.into();
        } else if let Some(path) = compare_path {
            let queries = model
                .encode(&build_queries(16), true)
                .expect("parity queries");
            let reference = read_dump(&path);
            result["parity"] =
                compare_with_dump(&reference, &emb, &plens, &model, &queries);
        }
    }

    println!("{}", serde_json::to_string_pretty(&result).expect("json"));
}
