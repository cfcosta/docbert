# Dependencies

This page tracks the **direct Cargo dependencies** declared in the current manifests:

- workspace root `Cargo.toml`
- `crates/docbert/Cargo.toml`
- `crates/docbert-core/Cargo.toml`
- `crates/docbert-plaid/Cargo.toml`
- `crates/docbert-pylate/Cargo.toml`
- `crates/rustbert/Cargo.toml`

It focuses on what each direct dependency is for in the current codebase, plus the major feature relationships that matter when changing builds or runtime behavior.

## Workspace root

The workspace root currently declares **no direct Rust dependencies**.

It only defines:

- workspace members:
  - `crates/docbert`
  - `crates/docbert-core`
  - `crates/docbert-plaid`
  - `crates/docbert-pylate`
  - `crates/rustbert`
- resolver:
  - `resolver = "3"`

All dependency versions live in the crate manifests.

## `crates/docbert`

`docbert` is the application crate: CLI entrypoint, web runtime, MCP runtime, and higher-level indexing/runtime orchestration.

### Direct dependencies

| Dependency           | Version                | Role in current code                                                                               |
| -------------------- | ---------------------- | -------------------------------------------------------------------------------------------------- |
| `docbert-core`       | path `../docbert-core` | Shared storage, indexing, search, embedding, and model primitives used by the application crate    |
| `axum`               | `0.8`                  | HTTP routing and handlers for `docbert web`                                                        |
| `base64`             | `0.22`                 | PDF upload encoding/decoding in web document routes                                                |
| `clap`               | `4.6.1`                | CLI parsing and command definitions in `src/cli.rs`                                                |
| `clap_complete`      | `4.6`                  | Generates shell completion scripts                                                                 |
| `globset`            | `0.4`                  | Glob filtering for MCP resource handling and some search/file filtering paths                      |
| `include_dir`        | `0.7`                  | Embeds built UI assets for the web runtime                                                         |
| `kdam`               | `0.6.4`                | Progress bars/spinners for indexing and embedding work in CLI flows                                |
| `percent-encoding`   | `2`                    | URI/resource encoding helpers in the MCP layer                                                     |
| `rand`               | `0.10`                 | OAuth state / PKCE verifier generation for the ChatGPT Codex login flow                            |
| `reqwest`            | `0.13.2`               | Outbound HTTP client for the OAuth token exchange (`json`, `form`, `rustls` features; no defaults) |
| `rmcp`               | `1.5.0`                | MCP server implementation over stdio (`transport-io` feature)                                      |
| `schemars`           | `1.2.1`                | JSON schema generation for MCP tool/input shapes                                                   |
| `serde`              | `1`                    | Serialization/deserialization for web and MCP request/response types                               |
| `serde_json`         | `1`                    | JSON values and serialization for web/MCP payloads                                                 |
| `sha2`               | `0.11`                 | PKCE code-challenge hashing for the ChatGPT Codex OAuth flow                                       |
| `tantivy`            | `0.26.0`               | Direct access to Tantivy writer/lock types in runtime resource handling                            |
| `tokio`              | `1`                    | Async runtime for web server and MCP runtime                                                       |
| `tracing`            | `0.1`                  | Runtime logging instrumentation                                                                    |
| `tracing-subscriber` | `0.3`                  | Logging initialization and env-filter support                                                      |
| `xdg`                | `3.0.0`                | Resolves the default data directory for the app                                                    |

### Direct dev-dependencies

| Dependency  | Version  | Role in current tests                                               |
| ----------- | -------- | ------------------------------------------------------------------- |
| `hegeltest` | `0.8`    | Property-based / parameterized test helpers                         |
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

| Dependency       | Version                                                                  | Role in current code                                                                               |
| ---------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `blake3`         | `1.8.4`                                                                  | Merkle snapshot hashing                                                                            |
| `bytemuck`       | `1.25.0` (`derive`)                                                      | Efficient `f32`/byte conversions in embedding storage; `derive` feature for `Pod`/`Zeroable` impls |
| `candle-core`    | `0.10.2`                                                                 | Tensor representation and tensor operations for model/embedding work                               |
| `docbert-plaid`  | workspace path `crates/docbert-plaid`                                    | PLAID multi-vector index used by the semantic leg of search                                        |
| `docbert-pylate` | workspace path `crates/docbert-pylate` (vendored from `pylate-rs` 1.0.4) | ColBERT model loading, query/document encoding, and similarity scoring                             |
| `ignore`         | `0.4`                                                                    | Filesystem walking with optional Git-ignore-aware discovery                                        |
| `kodama`         | `0.3`                                                                    | Hierarchical Ward clustering for ColBERT token pooling                                             |
| `pdf_oxide`      | `0.3.35`                                                                 | PDF-to-markdown/text extraction during preparation                                                 |
| `rayon`          | `1.12.0`                                                                 | Parallel document loading/preparation work                                                         |
| `redb`           | `4.1.0`                                                                  | `config.db` and `embeddings.db` storage                                                            |
| `rkyv`           | `0.8.15`                                                                 | Binary serialization for typed stored data                                                         |
| `serde`          | `1`                                                                      | Serialization support for public/config/runtime-facing data types                                  |
| `serde_json`     | `1`                                                                      | JSON values and parsing for metadata, settings, and conversation payloads                          |
| `tantivy`        | `0.26.0`                                                                 | Lexical indexing and BM25/fuzzy retrieval                                                          |
| `thiserror`      | `2`                                                                      | Error definition for `docbert_core::Error`                                                         |

### Direct dev-dependencies

| Dependency  | Version | Role in current tests                                              |
| ----------- | ------- | ------------------------------------------------------------------ |
| `criterion` | `0.8`   | Benchmarks (`embedding_trim`, `embedding_compression`)             |
| `hegeltest` | `0.8`   | Property-based / parameterized test helpers used by the core tests |
| `rand`      | `0.10`  | Random data generation in unit tests and benches                   |
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

| Dependency    | Version             | Role in current code                                                                  |
| ------------- | ------------------- | ------------------------------------------------------------------------------------- |
| `bytemuck`    | `1.25.0` (`derive`) | Efficient `f32`/byte conversions for on-disk index serialization                      |
| `candle-core` | `0.10.2`            | Tensor ops for k-means assignment and MaxSim batch matmul (GPU under the `cuda` flag) |
| `rand`        | `0.10`              | Randomized centroid initialization                                                    |
| `thiserror`   | `2`                 | Error definitions                                                                     |

### Direct dev-dependencies

| Dependency  | Version | Role in current tests                            |
| ----------- | ------- | ------------------------------------------------ |
| `criterion` | `0.8`   | Benchmarks (`kmeans`)                            |
| `hegeltest` | `0.8`   | Property-based / parameterized test helpers      |
| `rand`      | `0.10`  | Random fixtures for kmeans/MaxSim tests          |
| `tempfile`  | `3`     | Temporary directories for index round-trip tests |

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
| `thiserror`           | `2.0.18`  | Error definitions                                                        |
| `hf-hub`              | `0.5.0`   | Downloads model weights and configs from HuggingFace (rustls + ureq)     |
| `kodama`              | `0.3.0`   | Hierarchical Ward clustering used by token-pooling encode paths          |
| `rayon`               | `1.12.0`  | Parallelism in encoding/batching paths                                   |

### Direct dev-dependencies

| Dependency  | Version   | Role in current tests                       |
| ----------- | --------- | ------------------------------------------- |
| `anyhow`    | `1.0.102` | Loose error chaining inside test helpers    |
| `criterion` | `0.8`     | Benchmarks (`encode_batch_size`)            |
| `hegeltest` | `0.8`     | Property-based / parameterized test helpers |

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

## `crates/rustbert`

`rustbert` is a separate binary that depends on `docbert-core` as a library and ships its own crates.io fetcher, parser, and MCP server. See [`rustbert.md`](./rustbert.md) for the full story; the manifest section there is the canonical dep list. The summary below mirrors `crates/rustbert/Cargo.toml`.

### Direct dependencies

| Dependency             | Version                                                 | Role in current code                                                                                                                  |
| ---------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `docbert-core`         | path `../docbert-core`                                  | Storage / index / search / model primitives                                                                                           |
| `cargo-lock`           | `10`                                                    | Parse `Cargo.lock` for `rustbert sync`                                                                                                |
| `clap`                 | `4.6` (`derive`, `env`)                                 | CLI parsing                                                                                                                           |
| `flate2`               | `1`                                                     | Gzip decode for crates.io tarballs                                                                                                    |
| `globset`              | `0.4`                                                   | `--exclude` glob filtering in `rustbert sync`                                                                                         |
| `proc-macro2`          | `1` (`span-locations`)                                  | Feature-only direct dep so `tt.span().start().line` returns real line numbers; cargo-machete is configured to ignore the absent `use` |
| `quote`                | `1`                                                     | Token-tree rendering for synthesized signatures                                                                                       |
| `reqwest`              | `0.13.2` (`rustls`, `stream`; no defaults)              | HTTP client for crates.io / docs.rs                                                                                                   |
| `semver`               | `1` (`serde`)                                           | Version resolution                                                                                                                    |
| `serde` / `serde_json` | `1`                                                     | crates.io API and metadata blobs                                                                                                      |
| `sha2`                 | `0.11`                                                  | Tarball checksum verification                                                                                                         |
| `syn`                  | `2` (`full`)                                            | Rust AST parsing                                                                                                                      |
| `tantivy`              | `0.26.0`                                                | Direct access to the lexical index used by the rustbert indexer                                                                       |
| `tar`                  | `0.4`                                                   | Tarball extraction                                                                                                                    |
| `thiserror`            | `2`                                                     | Error definitions                                                                                                                     |
| `tokio`                | `1` (`rt`, `rt-multi-thread`, `macros`, `time`, `sync`) | Async runtime                                                                                                                         |
| `toml`                 | `0.8`                                                   | Read crate `Cargo.toml` files extracted from tarballs                                                                                 |
| `tracing`              | `0.1`                                                   | Logging                                                                                                                               |
| `tracing-subscriber`   | `0.3` (`env-filter`)                                    | Logger init driven by `RUSTBERT_LOG`                                                                                                  |

### Direct dev-dependencies

| Dependency  | Version | Role in current tests                                  |
| ----------- | ------- | ------------------------------------------------------ |
| `flate2`    | `1`     | Build synthetic tarballs                               |
| `hegeltest` | `0.8`   | Property-based / parameterized test helpers            |
| `sha2`      | `0.11`  | Pre-compute checksums for fixture tarballs             |
| `tar`       | `0.4`   | Build synthetic tarballs                               |
| `tempfile`  | `3`     | Temp data dirs                                         |
| `tokio`     | `1`     | Async test harness (`rt`, `rt-multi-thread`, `macros`) |

### `rustbert` feature relationships

```toml
[features]
default    = []
mkl        = ["docbert-core/mkl"]
accelerate = ["docbert-core/accelerate"]
metal      = ["docbert-core/metal"]
cuda       = ["docbert-core/cuda"]
```

The MCP server is hand-rolled JSON-RPC over stdio; rustbert deliberately does not depend on `rmcp` or `schemars`. There is no `xdg` dep either — data-dir resolution is done in-tree against `RUSTBERT_DATA_DIR` and `XDG_DATA_HOME`.

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

- the workspace has five members: `docbert`, `docbert-core`, `docbert-plaid`, `docbert-pylate`, `rustbert`
- `docbert-core` depends directly on `docbert-plaid` (PLAID index), `docbert-pylate` (ColBERT inference), and `kodama` (Ward clustering for token pooling); `hf-hub` is only pulled in transitively through `docbert-pylate`
- `docbert` has direct runtime/web/MCP dependencies such as `axum`, `rmcp`, `tokio`, `schemars`, `include_dir`, `reqwest`, `sha2`, `rand`, `tracing`, and `tracing-subscriber`
- `docbert-core` has direct dependencies for Merkle snapshots and PDF handling (`blake3`, `pdf_oxide`, `ignore`)
- `rustbert` depends on `docbert-core` (path) plus its own fetch/parse stack (`reqwest`, `flate2`, `tar`, `cargo-lock`, `syn`, …); its MCP server is hand-rolled JSON-RPC, so unlike `docbert` it does **not** pull in `rmcp` or `schemars`
- feature mapping flows app → core → `docbert-pylate` / `docbert-plaid`; `rustbert` exposes the same accelerated-backend feature names and forwards them through `docbert-core`
- the workspace root has no direct dependency list of its own

## Related references

- [`architecture.md`](./architecture.md)
- [`pipeline.md`](./pipeline.md)
- [`storage.md`](./storage.md)
- [`library-usage.md`](./library-usage.md)
- [`rustbert.md`](./rustbert.md)
