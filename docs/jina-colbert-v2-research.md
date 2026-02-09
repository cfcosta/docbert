# Jina ColBERT v2 research notes

Date: 2026-02-06

## Scope

This document summarizes investigation into Jina ColBERT v2 ("jina-colbert-v2")
with emphasis on context window, chunking parameters, and how those relate to
pylate-rs usage in docbert.

## Key findings

### 1) Model context window vs ColBERT doc length

- The model card for `jinaai/jina-colbert-v2` claims an 8192-token input context.
- The HF `config.json` sets `max_position_embeddings` to 8194, and
  `tokenizer_config.json` sets `model_max_length` to 8194. This aligns with an
  ~8192-token usable context once special tokens are included.
- However, the ColBERT metadata shipped with the model (`artifact.metadata`)
  specifies `query_maxlen: 32` and `doc_maxlen: 300` and shows
  `attend_to_mask_tokens: false`.

Note: the `attend_to_mask_tokens` value has changed over time in the HF repo.
The initial upload set it to `true`, a later "fix the padding" commit set it to
`false`, and a subsequent PR ("revert the mask padding default setting")
switched it back to `true`. Verify the exact value in the artifact that is
exported before relying on it.

Implication: the base encoder supports long context, but the ColBERT defaults
suggest a 300-token document length unless explicitly overridden.

### 1b) PyLate export config (local)

Generated a local PyLate export using `scripts/prepare_jina_colbert_v2.py` and
inspected `config_sentence_transformers.json`. The exported config sets:

- `query_length`: 32
- `document_length`: 300
- `query_prefix`: `[QueryMarker]`
- `document_prefix`: `[DocumentMarker]`
- `attend_to_expansion_tokens`: true
- `do_query_expansion`: true

This confirms that even with the 8K-capable backbone, the PyLate export
defaults to a 300-token document length unless explicitly overridden.
Because pylate-rs reads `config_sentence_transformers.json`, these exported
values become the defaults used by docbert unless we override them at load
time.

The exported `config.json` and `tokenizer_config.json` still advertise the
8194 max positions / model max length, reinforcing that the backbone supports
longer sequences while the ColBERT config defaults remain shorter.

Export location used: `/tmp/jina-colbert-v2-export/` (removed after inspection;
rerun `scripts/prepare_jina_colbert_v2.py` to regenerate).

Export warnings observed:
- PyLate reported the tokenizer does not support resizing token embeddings,
  so prefix tokens were not added to vocabulary.
- HuggingFace dynamic modules from `jinaai/xlm-roberta-flash-implementation`
  were downloaded during export (expected with `trust_remote_code`).

### 2) Jina release guidance

- The Jina release post lists max query length 32 and max document length 8192,
  indicating the model is designed to handle long documents, and that inputs
  longer than the max will be truncated in their examples.
- The Jina model catalog page also lists 8K input length for the model.
- The Jina ColBERT v2 paper states they pretrained the backbone with a max
  sequence length of 8192 tokens (with RoPE), but do not explicitly focus on
  long-context retrieval in the paper.

Implication: Jina marketing/benchmarks emphasize 8K context, but the ColBERT
metadata suggests 300-token defaults for indexing.

### 3) pylate-rs behavior (lengths + truncation)

Findings below are from pylate-rs 1.0.4 source (Cargo.lock + ~/.cargo registry).

- pylate-rs reads `query_length`/`document_length` from
  `config_sentence_transformers.json` (if present), unless overridden via the
  builder; missing values fall back to 32 / 180.
- Tokenization always truncates to `max_length` using the tokenizer; there is
  no sliding-window implementation in pylate-rs.
- Queries are padded to a fixed `query_length` using the tokenizer mask token
  (ColBERT query expansion); documents are padded to the longest sequence in
  the batch (after truncation).

Implication: to actually use 8K documents with pylate-rs, we need to
explicitly set `document_length` to 8192 (or another desired value) in the
exported config and align docbert chunking accordingly. If a chunk exceeds
`document_length`, pylate-rs will silently truncate it.

### 4) HF repo artifacts are not pylate-rs-ready

- The `jinaai/jina-colbert-v2` repo does not include
  `config_sentence_transformers.json` or `1_Dense/` artifacts, which
  `pylate-rs` expects.
- Therefore, a PyLate export step is required to create a local model directory
  that pylate-rs can load.

Implication: docbert should use a local PyLate-exported model directory for
jina-colbert-v2. The helper script `scripts/prepare_jina_colbert_v2.py`
creates this export.

### 5) Related evidence from GTE-ModernColBERT

The GTE-ModernColBERT model card (LightOn) provides guidance that may be
relevant for long-doc settings in ColBERT models generally:

- Trained with document length 300; this explains default doc length.
- Claims ColBERT models can generalize to longer doc lengths; suggests
  setting `document_length=8192` for long-context use.
- Notes ModernBERT was trained with 8K context and recommends benchmarking
  if you push to larger lengths.

While this is for a different model, it is an official LightOn statement
about document_length scaling in ColBERT models.

## Docbert implications (no implementation)

1) Docbert now defaults to 1024-token chunking when no model config is available.
   This is a balance between chunk count and storage/latency cost, and does not
   match the 300-token ColBERT metadata defaults.
2) If we want true 8K context, we must:
   - export the model with an explicit `document_length` (e.g., 8192), and
   - increase docbert's chunk size to match. (pylate-rs does not implement
     sliding-window chunking.)
3) The cost increase from 300 to 8192 tokens is ~27x in token count, which has
   significant storage and latency implications for ColBERT late interaction.
4) With pylate-rs, any chunk longer than `document_length` will be truncated
   during tokenization, so docbert chunk sizes should not exceed the configured
   document length unless truncation is acceptable.

### Chunking recommendation matrix (draft)

Assumptions:
- Docbert uses ~4 chars/token for chunk sizing.
- ColBERT late interaction cost and storage scale roughly linearly with
  token count.

| doc_length (tokens) | approx chunk chars | relative cost vs 300 | Recommended when |
| ------------------- | ------------------ | -------------------- | ---------------- |
| 300                 | ~1200              | 1.0x                 | ColBERT metadata default; short docs or cost-sensitive usage. |
| 512                 | ~2048              | 1.7x                 | Slightly longer docs; modest recall gains. |
| 1024                | ~4096              | 3.4x                 | Docbert default; medium-long docs; want fewer chunks per doc. |
| 2048                | ~8192              | 6.8x                 | Long-form content; compute budget is healthy. |
| 4096                | ~16384             | 13.7x                | Conservative vs 8K context. |
| 8192                | ~32768             | 27.3x                | Full 8K context; only if storage/latency cost is acceptable. |

Notes:
- For long documents, larger chunks reduce total chunk count but increase
  per-chunk embedding size and rerank cost.
- If using 8K doc_length, consider reducing chunk overlap to limit growth.

## Open questions

- Should docbert expose a flag to override `document_length` at load time
  (builder) vs requiring a re-exported model directory?
- What explicit guidance (if any) does Jina provide for choosing doc_maxlen
  in retrieval settings beyond the 300-token ColBERT default?
- What is the recommended `doc_maxlen` for jina-colbert-v2 in official
  benchmarks, and how does that map to our use case?
- What is the current `attend_to_mask_tokens` value on `main`, and should
  docbert override it explicitly in the exported config / pylate-rs builder?

## Sources

- https://huggingface.co/jinaai/jina-colbert-v2
- https://huggingface.co/jinaai/jina-colbert-v2/blob/main/README.md
- https://huggingface.co/jinaai/jina-colbert-v2/blob/main/config.json
- https://huggingface.co/jinaai/jina-colbert-v2/blob/main/tokenizer_config.json
- https://huggingface.co/jinaai/jina-colbert-v2/blob/main/artifact.metadata
- https://huggingface.co/jinaai/jina-colbert-v2/tree/main
- https://huggingface.co/jinaai/jina-colbert-v2/blob/770cf58d8d9fb5a0e157077b9ed5c2cfac158e8e/artifact.metadata
- https://huggingface.co/jinaai/jina-colbert-v2/commit/1a87022789695ab00d96ba478ee1d177e4cb52e7
- https://huggingface.co/jinaai/jina-colbert-v2/blob/refs%2Fpr%2F1/artifact.metadata
- https://jina.ai/news/jina-colbert-v2-multilingual-late-interaction-retriever-for-embedding-and-reranking/
- https://jina.ai/models/jina-colbert-v2/
- https://aclanthology.org/2024.mrl-1.11.pdf
- https://huggingface.co/lightonai/GTE-ModernColBERT-v1
- local: ~/.cargo/registry/src/index.crates.io-*/pylate-rs-1.0.4/src/builder.rs
- local: ~/.cargo/registry/src/index.crates.io-*/pylate-rs-1.0.4/src/model.rs
