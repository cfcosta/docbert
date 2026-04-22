# Dependencies

This page tracks the **direct Cargo dependencies** declared in the current manifests:

- workspace root `Cargo.toml`
- `crates/docbert/Cargo.toml`
- `crates/docbert-core/Cargo.toml`
- `crates/docbert-plaid/Cargo.toml`
- `crates/docbert-pylate/Cargo.toml`

It focuses on what each direct dependency is for in the current codebase, plus the major feature relationships that matter when changing builds or runtime behavior.

## Workspace root

The workspace root currently declares **no direct Rust dependencies**.

It only defines:

- workspace members:
  - `crates/docbert`
  - `crates/docbert-core`
  - `crates/docbert-plaid`
  - `crates/docbert-pylate`
- resolver:
  - `resolver = "3"`

All dependency versions live in the crate manifests.

## `crates/docbert`

`docbert` is the application crate: CLI entrypoint, web runtime, MCP runtime, and higher-level indexing/runtime orchestration.

### Direct dependencies

| Dependency           | Version                | Role in current code                                                                            |
| -------------------- | ---------------------- | ----------------------------------------------------------------------------------------------- |
| `docbert-core`       | path `../docbert-core` | Shared storage, indexing, search, embedding, and model primitives used by the application crate |
| `axum`               | `0.8`                  | HTTP routing and handlers for `docbert web`                                                     |
| `base64`             | `0.22`                 | PDF upload encoding/decoding in web document routes                                             |
| `clap`               | `4.6.1`                | CLI parsing and command definitions in `src/cli.rs`                                             |
| `clap_complete`      | `4.6`                  | Generates shell completion scripts                                                              |
| `globset`            | `0.4`                  | Glob filtering for MCP resource handling and some search/file filtering paths                   |
| `include_dir`        | `0.7`                  | Embeds built UI assets for the web runtime                                                      |
| `kdam`               | `0.6.4`                | Progress bars/spinners for indexing and embedding work in CLI flows                             |
| `percent-encoding`   | `2`                    | URI/resource encoding helpers in the MCP layer                                                  |
| `rand`               | `0.10`                 | OAuth state / PKCE verifier generation for the ChatGPT Codex login flow                         |
| `reqwest`            | `0.13.2`               | Outbound HTTP client for the OAuth token exchange (JSON + form bodies, no default features)     |
| `rmcp`               | `1.5.0`                | MCP server implementation over stdio (`transport-io` feature)                                   |
| `schemars`           | `1.2.1`                | JSON schema generation for MCP tool/input shapes                                                |
| `serde`              | `1`                    | Serialization/deserialization for web and MCP request/response types                            |
| `serde_json`         | `1`                    | JSON values and serialization for web/MCP payloads                                              |
| `sha2`               | `0.11`                 | PKCE code-challenge hashing for the ChatGPT Codex OAuth flow                                    |
| `tantivy`            | `=0.26.0`              | Direct access to Tantivy writer/lock types in runtime resource handling                         |
| `tokio`              | `1`                    | Async runtime for web server and MCP runtime                                                    |
| `tracing`            | `0.1`                  | Runtime logging instrumentation                                                                 |
| `tracing-subscriber` | `0.3`                  | Logging initialization and env-filter support                                                   |
| `xdg`                | `3.0.0`                | Resolves the default data directory for the app                                                 |

### Direct dev-dependencies

| Dependency  | Version  | Role in current tests                                               |
| ----------- | -------- | ------------------------------------------------------------------- |
| `hegeltest` | `0.6`    | Property-based / parameterized test helpers                         |
| `pdf_oxide` | `0.3.35` | Test PDF generation/helpers for web document route tests            |
| `rmcp`      | `1.5.0`  | MCP client-side test support (`client` + `transport-child-process`) |
| `tempfile`  | `3`      | Temporary directories/files in tests                                |
| `tower`     | `0.5`    | Test utilities for Axum services                                    |

### `docbert` feature relationships

`docbert` does not define its own independent runtime backend matrix. Its feature flags are pass-throughs to `docbert-core`:

```toml
[features]
default = []
mkl = ["docbert-core/mkl"]
accelerate = ["docbert-core/accelerate"]
metal = ["docbert-core/metal"]
cuda = ["docbert-core/cuda"]
```

That means the application crate's acceleration/build choices are controlled by the core crate's model backend features.

### Notes on major application dependencies

#### `axum`

Used in `crates/docbert/src/web/*` for:

- route registration
- state extraction
- JSON request/response handling
- static/UI serving integration

#### `rmcp`

Used in `crates/docbert/src/mcp.rs` for:

- MCP server wiring
- stdio transport
- tool/prompt/resource definitions
- MCP request/response types and errors

#### `tantivy`

Even though `docbert-core` owns the main index abstraction, the application crate still depends on `tantivy` directly for:

- `IndexWriter`
- lock failure detection/classification in runtime resource retry logic

#### `tokio`

Used for:

- the web server runtime
- the MCP runtime
- async tests around the web/API surface

## `crates/docbert-core`

`docbert-core` is the reusable library crate: storage, search, indexing helpers, model management, chunking, and document preparation.

### Direct dependencies

| Dependency       | Version                                                                  | Role in current code                                                      |
| ---------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| `blake3`         | `1.8.4`                                                                  | Merkle snapshot hashing                                                   |
| `bytemuck`       | `1.25.0`                                                                 | Efficient `f32`/byte conversions in embedding storage                     |
| `candle-core`    | `0.10.2`                                                                 | Tensor representation and tensor operations for model/embedding work      |
| `docbert-plaid`  | workspace path `crates/docbert-plaid`                                    | PLAID multi-vector index used by the semantic leg of search               |
| `docbert-pylate` | workspace path `crates/docbert-pylate` (vendored from `pylate-rs` 1.0.4) | ColBERT model loading, query/document encoding, and similarity scoring    |
| `ignore`         | `0.4`                                                                    | Filesystem walking with optional Git-ignore-aware discovery               |
| `pdf_oxide`      | `0.3.35`                                                                 | PDF-to-markdown/text extraction during preparation                        |
| `rayon`          | `1.12.0`                                                                 | Parallel document loading/preparation work                                |
| `redb`           | `4.1.0`                                                                  | `config.db` and `embeddings.db` storage                                   |
| `rkyv`           | `0.8.15`                                                                 | Binary serialization for typed stored data                                |
| `serde`          | `1`                                                                      | Serialization support for public/config/runtime-facing data types         |
| `serde_json`     | `1`                                                                      | JSON values and parsing for metadata, settings, and conversation payloads |
| `tantivy`        | `0.26.0`                                                                 | Lexical indexing and BM25/fuzzy retrieval                                 |
| `thiserror`      | `2`                                                                      | Error definition for `docbert_core::Error`                                |

### Direct dev-dependencies

| Dependency  | Version | Role in current tests                                              |
| ----------- | ------- | ------------------------------------------------------------------ |
| `hegeltest` | `0.6`   | Property-based / parameterized test helpers used by the core tests |
| `tempfile`  | `3`     | Temporary directories/files in unit tests                          |

### `docbert-core` feature relationships

`docbert-core` owns the model-backend feature mapping:

```toml
[features]
default = []
mkl = ["docbert-pylate/mkl"]
accelerate = ["docbert-pylate/accelerate"]
metal = ["docbert-pylate/metal"]
cuda = ["docbert-pylate/cuda", "docbert-plaid/cuda"]
```

These are the main build-time switches for accelerated inference.

### Notes on major core dependencies

#### `docbert-pylate`

This is the main ColBERT integration layer.

Current uses include:

- model loading in `model_manager.rs`
- query/document encoding
- similarity computation used by reranking

`docbert-pylate` is vendored into the workspace under `crates/docbert-pylate` as a
Rust-only fork of [`pylate-rs`](https://github.com/lightonai/pylate-rs) (originally
based on upstream 1.0.4). The upstream Python, WebAssembly, and npm packaging
layers have been removed; the crate is consumed exclusively from inside the
workspace and tracks the workspace release version rather than upstream's.

#### `tantivy`

Used for:

- schema definition
- persistent or in-memory lexical indexes
- BM25 retrieval
- collection/path lookups
- fuzzy matching support

#### `redb`

Used for both major local databases:

- `config.db`
- `embeddings.db`

Current code relies on it for:

- collection/config storage
- document metadata
- conversations
- collection Merkle snapshots
- settings and JSON metadata blobs
- embedding matrix persistence

#### `rkyv`

Used for stable typed binary storage of structures such as:

- document metadata
- conversations
- stored JSON wrappers
- Merkle snapshot structures

#### `ignore`

Used in `walker.rs` for recursive discovery.

Current discovery behavior uses it to support:

- hidden-file filtering
- supported-extension filtering
- Git ignore handling when the collection root is itself a Git repo

#### `pdf_oxide`

Used in `preparation.rs` to:

- load PDF bytes
- convert PDFs to markdown when possible
- fall back to extracted text when markdown conversion is empty

## `crates/docbert-plaid`

`docbert-plaid` is the workspace-local crate that implements the PLAID multi-vector index used for ColBERT late-interaction retrieval. It has no dependency on `docbert-core`; `docbert-core` depends on it.

### Direct dependencies

| Dependency    | Version  | Role in current code                                                                  |
| ------------- | -------- | ------------------------------------------------------------------------------------- |
| `bytemuck`    | `1.25.0` | Efficient `f32`/byte conversions for on-disk index serialization                      |
| `candle-core` | `0.10.2` | Tensor ops for k-means assignment and MaxSim batch matmul (GPU under the `cuda` flag) |
| `rand`        | `0.10`   | Randomized centroid initialization                                                    |
| `thiserror`   | `2`      | Error definitions                                                                     |

### `docbert-plaid` feature relationships

```toml
[features]
default = []
cuda = ["candle-core/cuda"]
```

## `crates/docbert-pylate`

`docbert-pylate` is the Rust-only fork of [pylate-rs](https://github.com/lightonai/pylate-rs) (originally based on upstream 1.0.4) that has been vendored into the workspace. The upstream Python, WebAssembly, and npm packaging layers were removed; the crate tracks the workspace release version rather than upstream's.

It owns the ColBERT / LateOn model loading, query/document encoding, and token-level similarity work used by `docbert-core::ModelManager`.

### Direct dependencies

| Dependency            | Version   | Role in current code                                                     |
| --------------------- | --------- | ------------------------------------------------------------------------ |
| `candle-core`         | `0.10.2`  | Tensor representation and ops for inference                              |
| `candle-nn`           | `0.10.2`  | Neural-network primitives used by the model stack                        |
| `candle-transformers` | `0.10.2`  | Transformer building blocks (ModernBERT encoder, pooling, etc.)          |
| `candle-flash-attn`   | `0.10.2`  | Optional flash-attention kernel; enabled only through the `cuda` feature |
| `tokenizers`          | `0.22.2`  | HuggingFace tokenizer runtime (`onig` backend)                           |
| `serde`               | `1.0.228` | Model config / metadata deserialization                                  |
| `serde_json`          | `1.0.149` | JSON parsing for model configuration files                               |
| `safetensors`         | `0.7.0`   | Zero-copy safetensors loading for weights                                |
| `thiserror`           | `2.0.18`  | Error definitions                                                        |
| `anyhow`              | `1.0.102` | Loose error chaining internal to the crate                               |
| `hf-hub`              | `0.5.0`   | Downloads model weights and configs from HuggingFace (rustls + ureq)     |
| `kodama`              | `0.3.0`   | Clustering / tokenizer auxiliary support                                 |
| `rayon`               | `1.12.0`  | Parallelism in encoding/batching paths                                   |

### `docbert-pylate` feature relationships

```toml
[features]
default = []
metal      = ["candle-core/metal",      "candle-nn/metal",      "candle-transformers/metal"]
cuda       = ["candle-core/cuda",       "candle-nn/cuda",       "candle-transformers/cuda",
              "dep:candle-flash-attn"]
mkl        = ["candle-core/mkl",        "candle-nn/mkl",        "candle-transformers/mkl"]
accelerate = ["candle-core/accelerate", "candle-nn/accelerate", "candle-transformers/accelerate"]
```

These are the leaf flags that `docbert-core`'s `mkl`/`accelerate`/`metal`/`cuda` features ultimately enable.

## Cross-crate relationships

A few relationships matter more than the raw version list.

### `docbert` depends on `docbert-core`

The application crate reuses the core crate for:

- `ConfigDb`
- `DataDir`
- `EmbeddingDb`
- `SearchIndex`
- `ModelManager`
- search functions
- document preparation and indexing helpers

That is why most search/storage dependency weight lives in `docbert-core`, not `docbert`.

### Feature flags flow from app to core to `docbert-pylate`

The feature chain is:

```text
docbert feature -> docbert-core feature -> docbert-pylate backend feature
```

For example:

```text
cargo build -p docbert --features cuda
    -> enables docbert-core/cuda
    -> enables docbert-pylate/cuda
    -> enables docbert-plaid/cuda
```

### Some crates appear in both manifests for different reasons

- `tantivy`
  - core crate: main index abstraction and retrieval
  - app crate: runtime writer/lock handling
- `serde` / `serde_json`
  - core crate: stored/config/runtime data types
  - app crate: HTTP and MCP payload types
- `pdf_oxide`
  - core crate: actual PDF preparation support
  - app crate dev-dependency: test helpers for PDF upload coverage

## What changed relative to older dependency docs

This page intentionally reflects the current manifests and removes stale or incomplete framing.

Important current realities include:

- the workspace has four members, not two: `docbert`, `docbert-core`, `docbert-plaid`, `docbert-pylate`
- `docbert-core` depends directly on `docbert-plaid` (PLAID index) and `docbert-pylate` (ColBERT inference) — `hf-hub` is only pulled in transitively through `docbert-pylate`
- `docbert` now has direct runtime/web/MCP dependencies such as `axum`, `rmcp`, `tokio`, `schemars`, `include_dir`, `reqwest`, `sha2`, `rand`, and `tracing`
- `docbert-core` has direct dependencies for Merkle snapshots and PDF handling (`blake3`, `pdf_oxide`, `ignore`)
- feature mapping flows app → core → `docbert-pylate` / `docbert-plaid`, not just one top-level example
- the workspace root has no direct dependency list of its own

## Related references

- [`architecture.md`](./architecture.md)
- [`pipeline.md`](./pipeline.md)
- [`storage.md`](./storage.md)
- [`library-usage.md`](./library-usage.md)
