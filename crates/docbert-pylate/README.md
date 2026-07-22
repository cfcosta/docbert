# docbert-pylate

This crate is a Rust library for late-interaction (ColBERT) model inference.
The [`docbert`](../../) crate uses this library to encode queries and
documents.

This crate is a vendored, Rust-only fork of
[`pylate-rs`](https://github.com/lightonai/pylate-rs). We removed the upstream
Python, WebAssembly, and npm packaging layers. Only the crates in this
workspace use `docbert-pylate` as a library. We will not publish
`docbert-pylate` as a standalone crate.

## What this crate gives

This crate gives you:

- A `ColBERT` model that you can load from a Hugging Face repository or a
  local directory
- BERT and ModernBERT backbones that use
  [Candle](https://github.com/huggingface/candle)
- Query and document encoding with batched, rayon-parallel CPU execution
- Optional CUDA, Metal, MKL, and Accelerate backends
- Hierarchical token pooling for document embeddings.

## Acceleration features

| Feature      | Backend              |
| ------------ | -------------------- |
| _(default)_  | Standard CPU         |
| `accelerate` | Apple CPU (macOS)    |
| `mkl`        | Intel CPU (MKL)      |
| `metal`      | Apple GPU (M-series) |
| `cuda`       | NVIDIA GPU (CUDA)    |

The `docbert` and `docbert-core` crates give these features to
`docbert-pylate`. For the user-facing build options, refer to the top-level
`docbert` crate.

## License

This crate has the MIT license. The upstream `pylate-rs` has the same license.
