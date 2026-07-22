# Proposal: Operate the embedding encoder in BF16

**Summary:** Document embedding is the part of `docbert sync`/`rebuild` that
operates for minutes. It operates the ModernBERT trunk fully in F32. The
candle F32 GEMMs use `CUBLAS_COMPUTE_32F` as the default. As a result, they
do not use the GPU tensor cores.

This proposal loads the model in BF16 and removes the per-layer F32→F16→F32
conversions around flash-attention. This is a small change of approximately
60 lines, all inside `docbert-pylate`. On the RTX 3080 Ti, the measurement
shows **1.63× document-encoding throughput** and **1.24× lower query
latency**. The retrieval ranking does not change (0 top-1 flips, max MaxSim
score delta 0.019 on scores ~9.5).

The experiments on the production model (`lightonai/GTE-ModernColBERT-v1`)
show that each claim that follows is correct. These experiments use a new speed and
numerical-parity harness, `crates/docbert-pylate/examples/encode_speed.rs`.
The measurements rejected two other possible alternatives. F16 gives NaN
embeddings from activation overflow. A fully-packed local-attention rewrite
is bit-exact but 15% _slower_. For the rewrite, the root cause is known, and
a possible v2 is a follow-up.

## Implementation status (2026-07-03): completed

Phase A and all six follow-ups from §5 are completed. We completed each
change, did a test on it, and committed it independently. For each step, we
did the full checks and a same-session ABAB benchmark against its parent
commit. The checks are fmt, clippy (with and without `--features cuda`), the
test suites, and the parity protocol that follows. The absolute docs/s values change
between sessions on the shared desktop GPU. Thus, we compare only these
paired ratios.

| commit                                             | measured effect (same-session ABAB)                            |
| -------------------------------------------------- | -------------------------------------------------------------- |
| BF16 trunk on CUDA (Phase A)                       | 1.64× document indexing, −19% query p50                        |
| fused LayerNorm with continuous zero bias (§5.1)   | 1.46× indexing, query p50 8.08 → 5.46 ms                       |
| one-slot GPU caches (§5.3)                         | removes the 1–2 GB VRAM leak, −0.7% on a warm-repeat synthetic |
| packed-local v2, fused `rope_thd` (§5.4)           | 1.50× indexing (549 → 825 docs/s), bit-identical numerics      |
| **fix: [PAD] keys attended in sorted varlen path** | correct results — refer to the text that follows               |
| flash routing for queries + small batches (§5.2)   | 32-doc batches 2.51× (310 → 778 docs/s), queries 1.07× p50     |
| tokenize once + pooling/store overlap (§5.5)       | 1.11× end-to-end embed→pool→store pipeline                     |
| PLAID index cache across queries (§5.6)            | ~790 ms fixed cost for each semantic query → ~0 ms warm        |

Together, the encode step measured approximately 3.6× the F32 baseline.
This is 1.64 × 1.46 × 1.50 for same-session pairs, one after the other. The pipeline
overlap adds 1.11× on top of that, end to end. Small incremental syncs are 2.51× faster. Searches on the production corpus (155k documents, 761 MB
index) no longer have approximately 790 ms of per-query index deserialization.

The model-weight reload for each CLI invocation (the other half of §5.6)
stays open. The MCP server path does not stop the process. Thus, this path
gets the full effect at this time.

**The implementation showed a problem that the initial harness cannot
find.** The sorted document path collected the per-row token lengths after
`pad_encodings` extended each row in place. Thus, flash varlen used each
padded width as a correct width. The [PAD] positions became attention keys in
each padded batch on the full GPU indexing path.

No check at that time found this problem. The integration tests encode 1–3
documents on the eager path. This proposal's initial parity protocol
compared against a GPU reference dump. That dump used the same code path as
the run under test. Thus, no test measured the varlen path against a
different ground truth.

The protocol (§7) has these changes at this time:

- The ground truth is an eager CPU F32 dump.
- The compare operations use `--batch-size 16` to select the multi-batch sorted varlen path.
- A new `attention_parity` example is the semantic check.

The `attention_parity` example supplies the same token ids to the eager
masked reference and to each CUDA fast path at F32. The min token cosine must
be ≥ 0.999. One BF16 trunk cannot make sure that a different trunk is correct. The rounding differences
become much larger and are not stable through the 22 layers.

With correct lengths, the change moved the batch-16 parity against CPU ground
truth. The min token cosine went from 0.9347 to 0.9699, and top-1 flips from
8 to ≤5. The remaining delta is attention-kernel rounding, not semantics.
The 768→128 projection makes this rounding larger.

## 1. Where the time goes

The production encoder is ModernBERT-base. It has 22 layers, hidden 768,
GeGLU MLPs (Wi 768→2304, Wo 1152→768), 12 heads, and RoPE. The attention layers
are global or local. Each third layer is global, and
the other 14 layers are local with a ±64-token sliding window. The model
projects to 128-dim ColBERT token embeddings.

Documents truncate to 300 tokens, and queries pad to 48. The linear layers only cost ≈ 220
MFLOPs/token.

The candle 0.10.2 CUDA backend
(`candle-core-0.10.2/src/cuda_backend/mod.rs:2471-2682`) shows two facts:

1. **F32 matmul does not use tensor cores.** All F32 GEMMs use the compute type
   `CUBLAS_COMPUTE_32F`, which is plain CUDA-core FP32. TF32 is available with
   `candle_core::cuda::set_gemm_reduced_precision_f32(true)`, but nothing in
   docbert uses it. F16 and BF16 GEMMs _do_ use tensor cores. On GA102, this
   is ~2× F32 peak with F32 accumulation, and ~4× with F16 accumulation.
2. **`ColBERT::new` sets `DType::F32` directly** for the trunk and the Dense
   projection (`crates/docbert-pylate/src/model.rs`). But candle-flash-attn
   accepts **F16 and BF16** as input. It uses FlashAttention-2 with sm86-aware
   tiling, and head_dim=64 is a first-class kernel. The trunk is F32, but
   flash-attn must have half precision. Thus, each attention call converts
   q/k/v from F32 to F16 and the output back. This is 4 full-tensor
   conversions × 22 layers × each forward.

Thus, a 16-bit trunk moves all GEMMs onto the tensor cores at the same time.
It also decreases the weight and activation memory traffic by half. The
conversions become no-ops.

This is the published standard, not a guess. The XTR replicability study
encodes ModernBERT-backbone ColBERT models in FP16. The study says, "We load
models and encode queries and documents in FP16 precision." Jina-ColBERT-v2
trains its ColBERT in pure BF16. Then, the PLAID index quantizes token
embeddings to 2 bits/dim. Thus, storage keeps far less precision than F16.

## 2. Experiments

The harness measures a deterministic varied-length corpus of 24–320 target
tokens. The production path is varied-length, and an equal-length ~300-token corpus
is the control. It measures best-of-5 wall time with an explicit
`device.synchronize()`, and single-query p50 latency. It also has a dump and
compare parity protocol. The protocol measures per-token cosine and
maximum element difference against the F32 baseline embeddings on a fixed
64-doc corpus. It also measures MaxSim score deltas and top-1 ranking flips
for 16 queries × 64 docs.

Hardware note: The RTX 3080 Ti (SM86, 12 GB) also operated a desktop session
at the same time. The other GPU load was 8–56%, and a game used ~7 GB VRAM in parallel. The absolute numbers are lower than
usual. But each experiment has its own same-session control. The important
effects (1.6×, NaN, −15%) are much larger than the measurement variation.

| experiment                              | varied 256 docs                       | speedup   | equal-length 192 docs    | parity vs F32 (8,363 tokens)                                     |
| --------------------------------------- | ------------------------------------- | --------- | ------------------------ | ---------------------------------------------------------------- |
| F32 baseline                            | 1299 ms · 197 docs/s · 27.7k tok/s    | 1.00×     | 1171 ms · 38.5k tok/s    | —                                                                |
| TF32 (`set_gemm_reduced_precision_f32`) | 1028 ms · 249 docs/s                  | 1.26×     | 903 ms · 50.0k tok/s     | min cos 0.99998 · max MaxSim Δ 0.0016 · 0 flips                  |
| **BF16 trunk**                          | **810 ms · 316 docs/s · 44.4k tok/s** | **1.63×** | **722 ms · 62.5k tok/s** | min cos 0.9851 · mean 0.99983 · max MaxSim Δ 0.019 · **0 flips** |
| F16 trunk                               | 813 ms                                | (1.62×)   | —                        | **NaN token embeddings — unusable**                              |
| F16 + f16 accumulation                  | 738 ms                                | (1.79×)   | —                        | NaN — unusable                                                   |
| packed local attention (F32)            | 1530 ms                               | **0.85×** | 1241 ms                  | bit-exact (max abs diff 0.0)                                     |
| BF16 + packed local                     | 956 ms                                | 1.38×     | 734 ms                   | same as BF16                                                     |

Single-query latency (p50, batch of 1): F32 5.24 ms → BF16 4.21 ms (1.24×).

Batch-size scan (BF16, 512 varied docs): 64 → 284 docs/s, 128 → 255,
192 → 222. **The default stays at 64.** The token-budget batching
fills the GPU fully. Larger budgets add padding in the length-sorted batches.

The failures show these facts:

- **F16 overflows.** The document embeddings have NaN values
  (`mean_token_cosine: null` is serde's NaN), and all 16 top-1 "flips" are NaN
  MaxSim scores that `total_cmp` puts first. The values are more than F16's ±65504
  range in the trunk, from transformer activation outliers (the GeGLU
  intermediates or the residual stream). Lower-precision accumulation makes
  it worse by a small quantity. BF16 has the same exponent range as F32 and does not
  overflow. The experiments show that BF16 is better than F16, and no quality
  study is necessary.
- **The packed-local rewrite is slower, and we know the cause.** The varlen
  path keeps hidden states packed `(total_tokens, hidden)`, but it unpacks to
  padded and re-packs q/k/v at each of the 14 local-attention layers
  (`unpack_varlen_bsd`/`pack_varlen_thd`, per-sequence `narrow`/`cat` loops).
  A direct call to `flash_attn_varlen_windowed` on the packed tensors (as the
  8 global layers do) removes thousands of small kernel launches and
  is bit-exact, but it measured 15% _slower_. The cause is RoPE, because the
  padded path uses the fused `rope_thd` kernel (`apply_rotary_emb_thd`, one
  kernel for each tensor). The packed path (`apply_rotary_emb_packed`,
  modernbert.rs:142) makes rope from `broadcast_mul` and `rotate_half`
  (narrow/neg/cat), which is ~7 kernels for each tensor on ~15M elements,
  some on non-contiguous
  views. The extension to 14 more layers added more memory traffic than the
  unpack and repack it removed, because the length-sorted batches have almost no
  padding.

## 3. Proposed change

All the changes are inside `docbert-pylate` (~60 lines):

1. **Builder knob and defaults.** This change adds `with_dtype(DType)` to
   `ColBERTBuilder`. The default is **BF16 when the device is CUDA**, and F32
   on CPU/Metal. It copies the batch-size env override with
   `DOCBERT_EMBEDDING_DTYPE=f32|f16|bf16` in `ModelManager` for zero-code
   rollback.
2. **Load in the chosen dtype.** The change sends the dtype to `ColBERT::new`
   (`VarBuilder::from_buffered_safetensors`) and to `build_dense_layers`, to make
   the trunk and the projection head agree. The rope tables use the
   VarBuilder dtype.
3. **Conditional flash-attn conversion.** In the three CUDA attention paths,
   the code converts q/k/v to F16 only when the trunk dtype is F32. BF16 goes
   through with no conversion. The `orig_dtype` cast stays and becomes a
   no-op clone.
4. **Keep the public output F32.** The code casts the projected embeddings to
   F32 immediately after the 128-dim projection, before L2-normalization. The
   normalize epsilon (1e-12) can underflow in half precision. Everything
   after this step (Ward pooling, PLAID, LMDB layout, `similarity`, and tests
   that read `to_vec3::<f32>`) stays byte-compatible.
5. **Dtype-safe attention mask.** `prepare_4d_attention_mask` multiplies by
   `f32::MIN`, which becomes −∞ or NaN in half precision. The code uses a
   finite min for each dtype (−1e38 for BF16, −65504 for F16). This gives
   protection to the masked query path.

This proposal does not include these changes:

- **TF32** (1.26×, one line). BF16 is better than TF32 here, because with a
  BF16 trunk almost no F32 GEMMs stay on the hot path. TF32 stays a good
  alternative for a subsequent change from 16-bit.
- **F16 and f16-accumulation.** The NaN result above removes these two alternatives.
- **Packed local attention.** The measurement (0.85×) removes it at this time. Refer to the v2 note.
- **A larger batch budget.** The scan above measured this as slower.

## 4. Correct results and risks

- **Parity indication.** For BF16, the min per-token cosine is 0.9851 (mean
  0.99983). The max MaxSim delta is 0.019 on scores around 9.5 (~0.2%), with
  0/16 top-1 flips. The TF32-level difference (0.0016) shows that the harness
  measures much smaller effects.
- **One change is necessary for the pinned end-to-end test.** `tests/tests.rs`
  pins a GTE-ModernColBERT query and doc similarity to 9.50805 ± 1e-2, and it
  operates on CUDA if there is a CUDA device. The BF16 MaxSim difference (up to
  ~0.019) makes that tolerance too small. One alternative increases the tolerance
  to 5e-2, and the other pins the test to F32 with the builder knob. All
  other test contracts (exact-zero padding, verbatim batch concatenation,
  MaxSim linearity) are dtype-agnostic, or the F32 output cast gives them protection.
- **Embedding cache compatibility.** BF16 embeddings are different from cached F32
  embeddings at the 1e-3 level. Mixed previous and new chunks retrieve correctly
  (0 ranking flips). But for byte-identical indexes, the release notes
  recommend a `rebuild`. The LMDB format itself does not change, and it
  continues to be F32.
- **Effect on other paths.** The CPU and Metal paths keep the F32 defaults
  and do not change. With `DOCBERT_EMBEDDING_DTYPE=f32`, the behavior is the
  same as before this change. The crate API continues to give F32 tensors.
- **VRAM.** BF16 decreases the weight memory (~300 MB → ~150 MB) and the
  activation memory by half. This gives more free memory for the known
  rope/position-cache growth that has no limit (follow-up 3), and for PLAID's
  build phase.

## 5. Follow-ups (ranked, different changes)

_We did all six changes. Refer to the implementation-status section at the
top for the measured results. The list below stays as written._

1. **Fused no-bias LayerNorm.** ModernBERT sets `norm_bias=false`, and
   `candle_nn::LayerNorm` uses its fused CUDA kernel only when there is a
   bias. Thus, all ~45 LNs for each forward use a ~10-kernel F32-upcast
   fallback. A continuous zeros(768) bias makes the fused path operate again
   with the same numerics (the kernel accumulates in F32 internally).
2. **Send queries and small batches through flash-attention.** `is_query`
   and any batch ≤ `batch_size` docs use eager O(s²) masked attention with 4D
   masks in memory. Thus, the query search and small incremental syncs never
   use the fast path at this time.
3. **Set a limit on the rope/position caches.** `packed_cos_sin`/`varlen_positions`
   use the exact per-batch length vector as the key, and they only add
   entries. This gives 1–2 GB VRAM during a long indexing run, and it is the
   cause of `release_encoder_before_plaid`.
4. **Packed-local v2.** This uses per-token gathered cos/sin and fused
   `rope_thd` on the packed tensor viewed as `(1, t, h, d)`, then
   `flash_attn_varlen_windowed` for the local layers. It gets the launch-count
   improvement without the composed-rope regression. A new measurement is
   necessary.
5. **Pipeline overlap and tokenize once.** Each 128-chunk submission batch
   tokenizes twice (`document_token_lengths` then `encode`). It waits for two
   D2H copies (~65 MB, mostly padding, the `[b, 300, 300]` dots tensor). It
   Ward-pools 128 docs one after another on one CPU thread, then writes LMDB, all serial with the GPU idle. Two more changes multiply with
   the BF16 improvement. These are double buffers for the batches, and
   lengths from `encode` itself.
6. **Query-side fixed costs (search latency, not indexing).** The two search
   paths deserialize the full PLAID index from disk _for each query_. The
   CLI reloads the model weights for each invocation. These costs are much
   larger than the 4 ms encode.

## 6. Supporting literature (docbert collection)

- **A Replicability Study of XTR** (arXiv:2605.00646) — FP16 encoding is the
  published standard for ModernBERT-backbone ColBERT, with the same 300-token
  doc and 48-token query budgets.
- **Jina-ColBERT-v2** (arXiv:2408.16672, #e64167) — the flash-attention
  backbone is a "free performance improvement", and the training is
  end-to-end in pure BF16.
- **ColBERTv2** (arXiv:2112.01488, #15d0b0) — 2-bit residual index storage.
  The encoder output precision is much more than the storage keeps.
- **PLAID** (arXiv:2205.09707, #88ae27) — padding waste and memory movement,
  not matmuls, are the primary cost in late-interaction profiles, with
  packed-tensor kernels.
- **ColBERT** (arXiv:2004.12832, #4ca444) — length-sorted batching (in
  `encode`) and CPU-tokenize/GPU-encode overlap (follow-up 5).
- **Mistral 7B** (arXiv:2310.06825) — window-aware flash-attention gave a
  measured 2× more than masked full attention. This is context for
  follow-up 4.
- **Longformer** (arXiv:2004.05150) — banded local attention must never be
  full attention plus a mask. This is context for follow-up 2.
- **WARP** (#a53066) — after retrieval is optimized, query _encoding_
  controls end-to-end latency. This is the cause of follow-ups 2 and 6.

## 7. Reproducing

```sh
CUDA_COMPUTE_CAP=80 nix develop -c cargo build --release \
  -p docbert-pylate --features cuda \
  --example encode_speed --example attention_parity

# NixOS: the real driver must shadow the toolkit's stub libcuda
export LD_LIBRARY_PATH=/run/opengl-driver/lib

target/release/examples/encode_speed --docs 256 --repeats 5      # throughput
target/release/examples/encode_speed --docs 32 --queries         # query p50

# Parity protocol (post-[PAD]-fix). Ground truth is an eager CPU F32
# dump — independent of every GPU code path:
target/release/examples/encode_speed --cpu --dump /tmp/ref_f32_cpu.bin
# batch 16 forces the multi-batch sorted varlen path; batch 64 covers
# the single-batch varlen path:
target/release/examples/encode_speed --compare /tmp/ref_f32_cpu.bin \
  --batch-size 16
target/release/examples/encode_speed --compare /tmp/ref_f32_cpu.bin \
  --batch-size 64

# Semantic gate: eager masked reference vs each CUDA fast path on
# identical token ids, at F32 (expect min token cosine >= 0.999 on
# every case, including query_expansion):
target/release/examples/attention_parity

# End-to-end embed→pool→store pipeline (docbert-core):
CUDA_COMPUTE_CAP=80 nix develop -c cargo build --release \
  -p docbert-core --features cuda --example index_speed
target/release/examples/index_speed --docs 512 --repeats 3
```

Each run gives one JSON object with the wall best and mean, docs/s, tokens/s,
an optional query p50, and the parity block (`min/mean_token_cosine`,
`max_abs_diff`, `max_maxsim_delta`, `top1_changes`).

One important point applies to a comparison of BF16 runs against
`/tmp/ref_f32_cpu.bin`. The expected values are ≈0.0736 max MaxSim delta and
0.9699 min token cosine at batch 16, and ≈0.0192 and 0.9958 at batch 64.
This is BF16-vs-F32 trunk drift, and the projection makes it larger. It is
stable for all the commits.

A check of semantics on those numbers alone does not find the [PAD]-key bug.
This is exactly the weakness from before. `attention_parity` at F32 is the
decisive check.
