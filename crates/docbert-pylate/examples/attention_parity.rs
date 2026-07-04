//! Path-isolation parity gate for the ModernBERT attention paths.
//!
//! Feeds identical random token ids through the eager masked
//! reference (`forward`) and each CUDA fast path (`forward_unmasked`,
//! `forward_varlen_padded`) and reports per-case deltas over the
//! valid tokens. Cases cover the uniform/ragged × short/long (sliding
//! window inactive/active) matrix.
//!
//! The F32 run is the semantic gate: attention round-trips through
//! F16 flash kernels but everything else stays F32, so min token
//! cosines land at ~0.999. Anything far below that means a path is
//! semantically wrong (wrong window, wrong rope positions, packing
//! bugs, ...). `--dtype bf16` compares two BF16 trunks whose rounding
//! differences amplify chaotically over 22 layers on these
//! out-of-distribution random ids (worst rows can drop to ~0.6 even
//! between correct paths), so BF16 numbers are only useful relative
//! to a same-dtype baseline, not as a correctness gate.
//!
//! Usage: attention_parity [--dtype f32|bf16]
//!        (requires CUDA and the HF model cache)

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use docbert_pylate::modernbert::{Config, ModernBert};

const MODEL_ID: &str = "lightonai/GTE-ModernColBERT-v1";

/// Deterministic xorshift so inputs are identical across runs/builds.
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

fn token_batch(
    rng: &mut Rng,
    lens: &[usize],
    device: &Device,
) -> (Tensor, Tensor) {
    let batch = lens.len();
    let seq_len = lens.iter().copied().max().unwrap_or(1);
    let mut ids = Vec::with_capacity(batch * seq_len);
    let mut mask = Vec::with_capacity(batch * seq_len);
    for &len in lens {
        for pos in 0..seq_len {
            // Stay inside the vocab, away from special tokens.
            ids.push(if pos < len {
                rng.range(1000, 20000) as u32
            } else {
                0
            });
            mask.push(u32::from(pos < len));
        }
    }
    (
        Tensor::from_vec(ids, (batch, seq_len), device).expect("ids"),
        Tensor::from_vec(mask, (batch, seq_len), device).expect("mask"),
    )
}

/// Max abs diff plus overall and per-row min token cosine over each
/// row's valid prefix. Per-row values pinpoint which sequence in a
/// ragged batch diverges when something is wrong.
fn compare(a: &Tensor, b: &Tensor, lens: &[usize]) -> (f32, f32, Vec<f32>) {
    let mut max_abs = 0f32;
    let mut row_min_cos = Vec::with_capacity(lens.len());
    for (row, &len) in lens.iter().enumerate() {
        let take = |t: &Tensor| -> Vec<Vec<f32>> {
            t.i(row)
                .expect("row")
                .narrow(0, 0, len)
                .expect("prefix")
                .to_dtype(DType::F32)
                .expect("f32")
                .to_vec2()
                .expect("vec2")
        };
        let mut min_cos = 1f32;
        for (ta, tb) in take(a).iter().zip(&take(b)) {
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (&x, &y) in ta.iter().zip(tb) {
                max_abs = max_abs.max((x - y).abs());
                dot += f64::from(x) * f64::from(y);
                na += f64::from(x) * f64::from(x);
                nb += f64::from(y) * f64::from(y);
            }
            min_cos =
                min_cos.min((dot / (na.sqrt() * nb.sqrt()).max(1e-30)) as f32);
        }
        row_min_cos.push(min_cos);
    }
    let min_cos = row_min_cos.iter().copied().fold(1f32, f32::min);
    (max_abs, min_cos, row_min_cos)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dtype = match args.iter().position(|a| a == "--dtype") {
        Some(i) => match args.get(i + 1).map(String::as_str) {
            Some("f32") => DType::F32,
            Some("bf16") => DType::BF16,
            other => panic!("--dtype must be f32 or bf16 (got {other:?})"),
        },
        None => DType::F32,
    };
    let device = Device::new_cuda(0).expect("CUDA device 0 required");
    let api = hf_hub::api::sync::Api::new().expect("hf hub api");
    let repo = api.model(MODEL_ID.to_string());
    let config_bytes = std::fs::read(repo.get("config.json").expect("config"))
        .expect("read config");
    let config: Config =
        serde_json::from_slice(&config_bytes).expect("parse config");
    let weights = repo.get("model.safetensors").expect("weights");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)
            .expect("varbuilder")
    };
    let model = ModernBert::load(vb, &config).expect("load model");

    // (name, per-row valid lengths). 299 > the 129-token full-attention
    // threshold, so "long" cases exercise the sliding window; "short"
    // cases stay under it.
    let cases: Vec<(&str, Vec<usize>)> = vec![
        ("uniform_short", vec![64; 8]),
        ("uniform_long", vec![299; 8]),
        ("ragged_short", vec![64, 61, 57, 52, 48, 44, 41, 40]),
        ("ragged_long", vec![299, 288, 270, 244, 210, 190, 165, 150]),
        (
            "ragged_cross_window",
            vec![299, 250, 170, 129, 96, 64, 40, 24],
        ),
        (
            "ragged_production_batch",
            vec![
                300, 297, 291, 284, 277, 271, 264, 250, 236, 229, 215, 201,
                188, 174, 161, 150,
            ],
        ),
    ];

    let mut rng = Rng(0xDEC0DE);
    let mut results = serde_json::Map::new();
    for (name, lens) in cases {
        let (ids, mask) = token_batch(&mut rng, &lens, &device);
        let reference = model.forward(&ids, &mask).expect("eager");
        let uniform = lens.iter().all(|&l| l == lens[0]);
        let fast = if uniform {
            model.forward_unmasked(&ids).expect("unmasked")
        } else {
            model.forward_varlen_padded(&ids, &lens).expect("varlen")
        };
        let (max_abs, min_cos, row_min_cos) = compare(&reference, &fast, &lens);
        results.insert(
            name.to_string(),
            serde_json::json!({
                "path": if uniform { "unmasked" } else { "varlen_padded" },
                "max_abs_diff": max_abs,
                "min_token_cosine": min_cos,
                "row_min_cosine": row_min_cos,
            }),
        );
    }

    // ColBERT query expansion: rows padded to a fixed 48 tokens where
    // padding rows still produce outputs (they feed MaxSim) but must
    // not act as keys. The eager mask only masks key columns, so it is
    // the exact semantic reference; compare ALL rows, padded included.
    let query_valid_lens = vec![9, 14, 21, 30, 39, 48, 5, 17];
    let query_seq_len = 48;
    let (ids, mask) = token_batch(
        &mut rng,
        &vec![query_seq_len; query_valid_lens.len()],
        &device,
    );
    let mut mask_rows = mask.to_vec2::<u32>().expect("mask rows");
    for (row, &len) in mask_rows.iter_mut().zip(&query_valid_lens) {
        for (pos, slot) in row.iter_mut().enumerate() {
            *slot = u32::from(pos < len);
        }
    }
    let mask = Tensor::from_vec(
        mask_rows.concat(),
        (query_valid_lens.len(), query_seq_len),
        &device,
    )
    .expect("query mask");
    let reference = model.forward(&ids, &mask).expect("eager query");
    let fast = model
        .forward_query_varlen(&ids, &query_valid_lens)
        .expect("query varlen");
    let all_rows = vec![query_seq_len; query_valid_lens.len()];
    let (max_abs, min_cos, row_min_cos) = compare(&reference, &fast, &all_rows);
    results.insert(
        "query_expansion".to_string(),
        serde_json::json!({
            "path": "query_varlen",
            "max_abs_diff": max_abs,
            "min_token_cosine": min_cos,
            "row_min_cosine": row_min_cos,
        }),
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::Value::Object(results))
            .expect("json")
    );
}
