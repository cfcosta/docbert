# Jina ColBERT v2 research notes

Date: 2026-02-06

## Scope

This file collects notes from an investigation into `jinaai/jina-colbert-v2`, mainly around context window limits, chunking, and how the model fits into docbert through pylate-rs.

## Key findings

### 1) Model context window vs. ColBERT document length

- The model card for `jinaai/jina-colbert-v2` says the model accepts an 8192-token context.
- The Hugging Face `config.json` sets `max_position_embeddings` to 8194, and `tokenizer_config.json` sets `model_max_length` to 8194. That lines up with an effective context length of about 8192 once special tokens are counted.
- The ColBERT metadata shipped with the model (`artifact.metadata`) tells a different story. It sets `query_maxlen: 32`, `doc_maxlen: 300`, and shows `attend_to_mask_tokens: false`.

A detail worth watching: `attend_to_mask_tokens` has changed over time in the HF repo. The first upload set it to `true`, a later "fix the padding" commit changed it to `false`, and a later PR ("revert the mask padding default setting") switched it back to `true`. If we rely on that field, we should verify the exact exported artifact instead of assuming the current repo view matches it.

What this means: the backbone can handle long context, but the ColBERT defaults still look like a 300-token document setup unless we override them.

### 1b) PyLate export config (local)

I generated a local PyLate export with `scripts/prepare_jina_colbert_v2.py` and inspected `config_sentence_transformers.json`. The export set these values:

- `query_length`: 32
- `document_length`: 300
- `query_prefix`: `[QueryMarker]`
- `document_prefix`: `[DocumentMarker]`
- `attend_to_expansion_tokens`: true
- `do_query_expansion`: true

That lines up with the earlier finding: even with an 8K-capable backbone, the exported PyLate config still defaults to 300-token documents unless we change it.

Because pylate-rs reads `config_sentence_transformers.json`, these exported values become the defaults docbert sees unless we override them at load time.

The exported `config.json` and `tokenizer_config.json` still advertise 8194 max positions and model max length, which reinforces the split between what the backbone can do and what the ColBERT config asks it to do by default.

Export location used: `/tmp/jina-colbert-v2-export/` (removed after inspection; rerun `scripts/prepare_jina_colbert_v2.py` to regenerate).

Warnings seen during export:

- PyLate reported that the tokenizer does not support resizing token embeddings, so the prefix tokens were not added to the vocabulary.
- HuggingFace dynamic modules from `jinaai/xlm-roberta-flash-implementation` were downloaded during export, which is expected with `trust_remote_code`.

### 2) Jina release guidance

- The Jina release post lists max query length 32 and max document length 8192, which suggests the model is intended for long documents and that longer inputs are truncated in their examples.
- The Jina model catalog page also lists an 8K input length.
- The Jina ColBERT v2 paper says the backbone was pretrained with a maximum sequence length of 8192 tokens using RoPE, though long-context retrieval is not the main point of the paper.

What this means: Jina's public material leans hard on the 8K context story, while the ColBERT metadata still points to 300-token defaults for indexing.

### 3) pylate-rs behavior: lengths and truncation

The notes below come from pylate-rs `1.0.4` source code (`Cargo.lock` and `~/.cargo/registry`).

- pylate-rs reads `query_length` and `document_length` from `config_sentence_transformers.json` when that file exists, unless the builder overrides them. If those values are missing, it falls back to 32 and 180.
- Tokenization always truncates to `max_length` through the tokenizer. There is no sliding-window implementation in pylate-rs.
- Queries are padded to a fixed `query_length` with the tokenizer mask token for ColBERT query expansion. Documents are padded to the longest sequence in the batch after truncation.

What this means: if we really want 8K documents in pylate-rs, we need to set `document_length` to 8192, or another chosen value, in the exported config and then align docbert's chunking with it. If a chunk is longer than `document_length`, pylate-rs will silently truncate it.

### 4) Hugging Face repo artifacts are not ready for pylate-rs

- The `jinaai/jina-colbert-v2` repo does not include `config_sentence_transformers.json` or the `1_Dense/` artifacts that `pylate-rs` expects.
- Because of that, we need a PyLate export step to produce a local model directory that pylate-rs can load.

What this means: docbert should use a local PyLate-exported model directory for `jina-colbert-v2`. The helper script `scripts/prepare_jina_colbert_v2.py` already does that.

### 5) Related evidence from GTE-ModernColBERT

The LightOn model card for GTE-ModernColBERT includes a few notes that are probably useful here too:

- The model was trained with document length 300, which helps explain why that keeps showing up as a default.
- The card says ColBERT models can generalize to longer document lengths and suggests `document_length=8192` for long-context use.
- It also notes that ModernBERT was trained with 8K context and recommends benchmarking if you push to larger lengths.

This is a different model, so it is not direct evidence for Jina ColBERT v2. Still, it is an official LightOn statement about how ColBERT document length can scale.

## Docbert implications (no implementation)

1. docbert now defaults to 519-token chunking when no model config is available. That matches ColBERT-Zero's training length, but not the 300-token ColBERT metadata defaults discussed above.
2. If we want true 8K context, we need both of these:
   - a model export with an explicit `document_length` such as 8192
   - a matching docbert chunk size, since pylate-rs does not do sliding-window chunking
3. Moving from 300 to 8192 tokens increases token count by about 27x, which has a big effect on storage and MaxSim latency.
4. With pylate-rs, any chunk longer than `document_length` is truncated during tokenization. docbert chunk sizes should not exceed the configured document length unless truncation is acceptable.

### Chunking recommendation matrix (draft)

Assumptions:

- docbert uses roughly 4 characters per token when sizing chunks
- ColBERT late-interaction cost and storage grow roughly linearly with token count

| doc_length (tokens) | approx chunk chars | relative cost vs 300 | Recommended when                                                     |
| ------------------- | ------------------ | -------------------- | -------------------------------------------------------------------- |
| 300                 | ~1200              | 1.0x                 | ColBERT metadata default; short docs or cost-sensitive usage.        |
| 519                 | ~2076              | 1.7x                 | docbert default; matches ColBERT-Zero training length.               |
| 1024                | ~4096              | 3.4x                 | Medium-long docs; want fewer chunks per doc.                         |
| 2048                | ~8192              | 6.8x                 | Long-form content; compute budget is healthy.                        |
| 4096                | ~16384             | 13.7x                | A conservative step below full 8K context.                           |
| 8192                | ~32768             | 27.3x                | Full 8K context; only if the storage and latency cost is acceptable. |

Notes:

- Larger chunks reduce total chunk count for long documents, but each chunk costs more to store and rerank.
- If you use 8K `doc_length`, consider reducing chunk overlap to keep growth under control.

## Open questions

- Should docbert expose a flag that overrides `document_length` at load time through the builder, instead of requiring a re-exported model directory?
- What guidance, if any, does Jina give for choosing `doc_maxlen` beyond the 300-token ColBERT default?
- What `doc_maxlen` does Jina use in official benchmarks, and how does that map to docbert's use case?
- What is the current `attend_to_mask_tokens` value on `main`, and should docbert override it explicitly in the exported config or pylate-rs builder?

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
- local: `~/.cargo/registry/src/index.crates.io-*/pylate-rs-1.0.4/src/builder.rs`
- local: `~/.cargo/registry/src/index.crates.io-*/pylate-rs-1.0.4/src/model.rs`
