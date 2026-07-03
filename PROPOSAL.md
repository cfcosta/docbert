# Proposal: run the embedding encoder in BF16

**TL;DR:** document embedding — the part of `docbert sync`/`rebuild` that
takes minutes — runs the ModernBERT trunk entirely in F32, and candle's F32
GEMMs default to `CUBLAS_COMPUTE_32F`: they never touch the GPU's tensor
cores. Loading the model in BF16 and deleting the per-layer F32→F16→F32
conversion churn around flash-attention is a small change (~60 lines, all
inside `docbert-pylate`) measured at **1.63× document-encoding throughput**
and **1.24× lower query latency** on the RTX 3080 Ti, with no retrieval
ranking changes (0 top-1 flips, max MaxSim score delta 0.019 on scores ~9.5).

Every claim below was verified experimentally on the production model
(`lightonai/GTE-ModernColBERT-v1`) with a new speed + numerical-parity
harness, `crates/docbert-pylate/examples/encode_speed.rs`. Two plausible
alternatives were tried and **rejected by measurement**: F16 (NaN embeddings
from activation overflow) and a fully-packed local-attention rewrite
(bit-exact but 15% _slower_; root cause identified, viable v2 noted as
follow-up).

## 1. Where the time goes today

The production encoder is ModernBERT-base: 22 layers, hidden 768, GeGLU MLPs
(Wi 768→2304, Wo 1152→768), 12 heads, RoPE, alternating attention — every
third layer global, the other 14 local with a ±64-token sliding window —
projected to 128-dim ColBERT token embeddings. Documents truncate to 300
tokens, queries pad to 48. Linear layers alone cost ≈ 220 MFLOPs/token.

Two facts, confirmed in candle 0.10.2's CUDA backend
(`candle-core-0.10.2/src/cuda_backend/mod.rs:2471-2682`):

1. **F32 matmul never uses tensor cores.** All F32 GEMMs run with compute
   type `CUBLAS_COMPUTE_32F` (plain CUDA-core FP32). TF32 is opt-in via
   `candle_core::cuda::set_gemm_reduced_precision_f32(true)`; nothing in
   docbert calls it. F16/BF16 GEMMs _do_ use tensor cores (~2× F32 peak on
   GA102 with F32 accumulation, ~4× with F16 accumulation).
2. **`ColBERT::new` hardcodes `DType::F32`** for the trunk and the Dense
   projection (`crates/docbert-pylate/src/model.rs`), while
   candle-flash-attn natively accepts **F16 and BF16** (FlashAttention-2,
   sm86-aware tiling; head_dim=64 is a first-class kernel). Because the
   trunk is F32 but flash-attn needs half precision, every attention call
   converts q/k/v F32→F16 and the output back — 4 full-tensor conversions ×
   22 layers × every forward.

So a 16-bit trunk simultaneously: moves all GEMMs onto tensor cores, halves
weight/activation memory traffic, and turns the conversion churn into no-ops.

This is the published standard, not a bet: the XTR replicability study
encodes ModernBERT-backbone ColBERT models in FP16 ("We load models and
encode queries and documents in FP16 precision"), Jina-ColBERT-v2 trains its
ColBERT in pure BF16, and the PLAID index downstream quantizes token
embeddings to 2 bits/dim — storage keeps far less precision than F16.

## 2. Experiments

Harness: deterministic varied-length corpus (24–320 target tokens — the
production path is varied-length; uniform ~300-token corpus as control),
best-of-5 wall time with explicit `device.synchronize()`, single-query p50
latency, and a dump/compare parity protocol: per-valid-token cosine and max
element diff against the F32 baseline's embeddings on a fixed 64-doc corpus,
plus MaxSim score deltas and top-1 ranking flips for 16 queries × 64 docs.

Hardware note: RTX 3080 Ti (SM86, 12 GB) shared with a live desktop session
(8–56% background utilization, ~7 GB VRAM in use — a game was running).
Absolute numbers are depressed; each experiment carries its own same-session
control, and the decisive effects (1.6×, NaN, −15%) dwarf the noise.

| experiment                              | varied 256 docs                       | speedup   | uniform 192 docs         | parity vs F32 (8,363 tokens)                                     |
| --------------------------------------- | ------------------------------------- | --------- | ------------------------ | ---------------------------------------------------------------- |
| F32 baseline (today)                    | 1299 ms · 197 docs/s · 27.7k tok/s    | 1.00×     | 1171 ms · 38.5k tok/s    | —                                                                |
| TF32 (`set_gemm_reduced_precision_f32`) | 1028 ms · 249 docs/s                  | 1.26×     | 903 ms · 50.0k tok/s     | min cos 0.99998 · max MaxSim Δ 0.0016 · 0 flips                  |
| **BF16 trunk**                          | **810 ms · 316 docs/s · 44.4k tok/s** | **1.63×** | **722 ms · 62.5k tok/s** | min cos 0.9851 · mean 0.99983 · max MaxSim Δ 0.019 · **0 flips** |
| F16 trunk                               | 813 ms                                | (1.62×)   | —                        | **NaN token embeddings — unusable**                              |
| F16 + f16 accumulation                  | 738 ms                                | (1.79×)   | —                        | NaN — unusable                                                   |
| packed local attention (F32)            | 1530 ms                               | **0.85×** | 1241 ms                  | bit-exact (max abs diff 0.0)                                     |
| BF16 + packed local                     | 956 ms                                | 1.38×     | 734 ms                   | same as BF16                                                     |

Single-query latency (p50, batch of 1): F32 5.24 ms → BF16 4.21 ms (1.24×).

Batch-size sweep (BF16, 512 varied docs): 64 → 284 docs/s, 128 → 255,
192 → 222. **Keep the default at 64** — the token-budget batching already
saturates the GPU, and larger budgets add padding within length-sorted
batches.

What the failures taught us:

- **F16 overflows.** Document embeddings come back NaN-contaminated
  (`mean_token_cosine: null` is serde's NaN; all 16 top-1 "flips" are NaN
  MaxSim scores winning `total_cmp`). F16's ±65504 range is exceeded
  somewhere in the trunk (transformer activation outliers — GeGLU
  intermediates or residual stream); reduced-precision accumulation makes it
  slightly worse. BF16 shares F32's exponent range and is clean. This
  empirically settles BF16 vs F16 — no quality study needed.
- **The packed-local rewrite is slower, and we know why.** The varlen path
  keeps hidden states packed `(total_tokens, hidden)` but unpacks to padded
  and re-packs q/k/v at each of the 14 local-attention layers
  (`unpack_varlen_bsd`/`pack_varlen_thd`, per-sequence `narrow`/`cat`
  loops). Calling `flash_attn_varlen_windowed` directly on the packed
  tensors (as the 8 global layers already do) removes thousands of tiny
  kernel launches and is bit-exact — but measured 15% _slower_. Root cause:
  the padded path applies RoPE with the **fused `rope_thd` kernel**
  (`apply_rotary_emb_thd`, one kernel per tensor), while the packed path
  (`apply_rotary_emb_packed`, modernbert.rs:142) composes rope from
  `broadcast_mul` + `rotate_half` (narrow/neg/cat) — ~7 kernels per tensor
  over ~15M elements, several on non-contiguous views. Extending that to 14
  more layers added more memory traffic than the unpack/repack it removed
  (length-sorted batches have little padding, so the padded compute isn't
  actually very wasteful). A v2 — gather per-token cos/sin rows by packed
  position, then call fused `rope_thd` on the packed tensor viewed as
  `(1, t, h, d)` — should get the launch-count win without the rope
  regression; it is follow-up material, not part of this proposal.

## 3. Proposed change

All inside `docbert-pylate` (~60 lines):

1. **Builder knob + defaults.** Add `with_dtype(DType)` to `ColBERTBuilder`;
   default **BF16 when the device is CUDA**, F32 on CPU/Metal. Mirror the
   existing batch-size env override with `DOCBERT_EMBEDDING_DTYPE=f32|f16|bf16`
   in `ModelManager` for zero-code rollback.
2. **Load in the chosen dtype.** Thread it through `ColBERT::new`
   (`VarBuilder::from_buffered_safetensors`) _and_ `build_dense_layers` so
   trunk and projection head match. Rope tables inherit the VarBuilder dtype.
3. **Conditional flash-attn conversion.** In the three CUDA attention paths,
   convert q/k/v to F16 only when the trunk dtype is F32 (BF16 passes
   through; keep the `orig_dtype` restore, which becomes a no-op clone).
4. **Keep the public output F32.** Cast the projected embeddings to F32
   immediately after the 128-dim projection, before L2-normalization: the
   normalize epsilon (1e-12) would underflow in half precision, and
   everything downstream — Ward pooling, PLAID, LMDB layout, `similarity`,
   tests reading `to_vec3::<f32>` — stays byte-compatible.
5. **Dtype-safe attention mask.** `prepare_4d_attention_mask` multiplies by
   `f32::MIN`, which becomes −∞/NaN territory when cast to half precision.
   Use a finite per-dtype min (−1e38 for BF16, −65504 for F16). This
   protects the masked query path.

Not proposed, and why:

- **TF32** (1.26×, one line) is strictly dominated by BF16 here; with a BF16
  trunk almost no F32 GEMMs remain on the hot path. Worth keeping in the
  back pocket if 16-bit ever has to be reverted.
- **F16 / f16-accumulation** — ruled out by the NaN result above.
- **Packed local attention** — ruled out as-implemented (0.85×); see v2 note.
- **Raising the batch budget** — measured slower (sweep above).

## 4. Correctness & risks

- **Parity evidence.** BF16: min per-token cosine 0.9851 (mean 0.99983),
  max MaxSim delta 0.019 on scores around 9.5 (~0.2%), 0/16 top-1 flips.
  TF32-level drift (0.0016) shows the harness resolves much smaller effects.
- **The pinned end-to-end test needs one adjustment.** `tests/tests.rs` pins
  a GTE-ModernColBERT query/doc similarity to 9.50805 ± 1e-2 and runs on
  CUDA when available; the observed BF16 MaxSim drift (up to ~0.019) makes
  that tolerance marginal. Either widen it to 5e-2 or pin the test to F32
  via the builder knob. All other test contracts (exact-zero padding,
  verbatim batch concatenation, MaxSim linearity) are dtype-agnostic or
  protected by the F32 output cast.
- **Embedding cache compatibility.** BF16 embeddings differ from cached F32
  ones at the 1e-3 level. Mixed old/new chunks retrieve fine (0 ranking
  flips), but for byte-reproducible indexes recommend a `rebuild` in the
  release notes; the LMDB format itself is unchanged (still F32).
- **Blast radius.** CPU and Metal paths keep F32 defaults and are untouched;
  `DOCBERT_EMBEDDING_DTYPE=f32` restores today's behavior exactly; the crate
  API still returns F32 tensors.
- **VRAM.** Halves weight (~300 MB → ~150 MB) and activation memory —
  extra headroom for the known unbounded rope/position-cache growth
  (follow-up 3) and for PLAID's build phase.

## 5. Follow-ups (ranked, separate changes)

1. **Fused no-bias LayerNorm.** ModernBERT sets `norm_bias=false`, and
   `candle_nn::LayerNorm` only uses its fused CUDA kernel when a bias
   exists — all ~45 LNs per forward take a ~10-kernel F32-upcast fallback.
   A persistent zeros(768) bias restores the fused path with identical
   numerics (the kernel accumulates in F32 internally).
2. **Route queries and small batches through flash-attention.** `is_query`
   and any batch ≤ `batch_size` docs run eager O(s²) masked attention with
   materialized 4D masks — interactive search and small incremental syncs
   never see the fast path today.
3. **Bound the rope/position caches.** `packed_cos_sin`/`varlen_positions`
   are keyed by the exact per-batch length vector and only ever insert —
   1–2 GB VRAM over a long indexing run (the reason
   `release_encoder_before_plaid` exists).
4. **Packed-local v2.** Per-token gathered cos/sin + fused `rope_thd` on the
   packed layout, then `flash_attn_varlen_windowed` for local layers — the
   launch-count win without the composed-rope regression. Re-measure.
5. **Pipeline overlap + tokenize once.** Each 128-chunk submission batch
   tokenizes twice (`document_token_lengths` then `encode`), blocks on two
   D2H copies (~65 MB, mostly padding — the `[b, 300, 300]` dots tensor),
   Ward-pools 128 docs sequentially on one CPU thread, then writes LMDB,
   all serial with the GPU idle. Double-buffering batches and returning
   lengths from `encode` itself stack multiplicatively with the BF16 win.
6. **Query-side fixed costs (search latency, not indexing).** Both search
   paths deserialize the entire PLAID index from disk _per query_ and the
   CLI reloads model weights per invocation — these dwarf the 4 ms encode.

## 6. Supporting literature (docbert collection)

- **A Replicability Study of XTR** (arXiv:2605.00646) — FP16 encoding is the
  published standard for ModernBERT-backbone ColBERT; same 300-token doc /
  48-token query budgets.
- **Jina-ColBERT-v2** (arXiv:2408.16672, #e64167) — flash-attention backbone
  as a "free performance improvement"; trained end-to-end in pure BF16.
- **ColBERTv2** (arXiv:2112.01488, #15d0b0) — 2-bit residual index storage;
  encoder output precision far exceeds what is kept.
- **PLAID** (arXiv:2205.09707, #88ae27) — padding waste and memory movement,
  not matmuls, dominate late-interaction profiles; packed-tensor kernels.
- **ColBERT** (arXiv:2004.12832, #4ca444) — length-sorted batching (already
  implemented in `encode`) and CPU-tokenize/GPU-encode overlap (follow-up 5).
- **Mistral 7B** (arXiv:2310.06825) — window-aware flash-attention gave a
  measured 2× over masked full attention; context for follow-up 4.
- **Longformer** (arXiv:2004.05150) — banded local attention should never be
  full attention plus a mask; context for follow-up 2.
- **WARP** (#a53066) — once retrieval is optimized, query _encoding_
  dominates end-to-end latency; motivates follow-ups 2 and 6.

## 7. Reproducing

```sh
nix develop -c cargo build --release -p docbert-pylate --features cuda \
  --example encode_speed

# NixOS: the real driver must shadow the toolkit's stub libcuda
export LD_LIBRARY_PATH=/run/opengl-driver/lib

target/release/examples/encode_speed --docs 256 --repeats 5      # throughput
target/release/examples/encode_speed --docs 32 --queries         # query p50
target/release/examples/encode_speed --dump    /tmp/ref_f32.bin  # on main
target/release/examples/encode_speed --compare /tmp/ref_f32.bin  # on branch
```

One JSON object per run: wall best/mean, docs/s, tokens/s, optional query
p50, and the parity block (`min/mean_token_cosine`, `max_abs_diff`,
`max_maxsim_delta`, `top1_changes`).
