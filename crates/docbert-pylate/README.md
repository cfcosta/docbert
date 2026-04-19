# docbert-pylate

Rust library for late-interaction (ColBERT) model inference, used by
[`docbert`](../../) for query and document encoding.

This crate is a vendored, Rust-only fork of [`pylate-rs`](https://github.com/lightonai/pylate-rs).
The upstream Python, WebAssembly, and npm packaging layers have been removed —
`docbert-pylate` is consumed exclusively as a library from inside this workspace
and is not intended to be published as a standalone crate.

## What it provides

- A `ColBERT` model loaded from a Hugging Face repo or a local directory.
- BERT and ModernBERT backbones via [Candle](https://github.com/huggingface/candle).
- Query and document encoding with batched, rayon-parallel CPU execution and
  optional CUDA / Metal / MKL / Accelerate backends.
- Hierarchical token pooling for document embeddings.

## Acceleration features

| Feature      | Backend              |
| ------------ | -------------------- |
| _(default)_  | Standard CPU         |
| `accelerate` | Apple CPU (macOS)    |
| `mkl`        | Intel CPU (MKL)      |
| `metal`      | Apple GPU (M-series) |
| `cuda`       | NVIDIA GPU (CUDA)    |

Features are propagated from `docbert` / `docbert-core` — see the top-level
`docbert` crate for the user-facing build options.

## License

MIT — same as upstream `pylate-rs`.
