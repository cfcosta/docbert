# Dependencies

This page tracks the direct Cargo dependencies declared in the current manifests:

- workspace root `Cargo.toml`
- `crates/docbert/Cargo.toml`
- `crates/docbert-core/Cargo.toml`
- `crates/docbert-web/Cargo.toml`
- `crates/docbert-webui/Cargo.toml`
- `crates/docbert-plaid/Cargo.toml`
- `crates/docbert-pylate/Cargo.toml`
- `crates/rustbert/Cargo.toml`

It focuses on what each direct dependency is for in the current codebase, plus the major feature relationships that matter when changing builds or runtime behavior.

## Workspace root

The workspace root declares no direct Rust dependencies.

It only defines:

- workspace members:
  - `crates/docbert`
  - `crates/docbert-core`
  - `crates/docbert-web`
  - `crates/docbert-webui`
  - `crates/docbert-plaid`
  - `crates/docbert-pylate`
  - `crates/rustbert`
- resolver:
  - `resolver = "3"`

All dependency versions live in the crate manifests.

## `crates/docbert`

`docbert` is the application crate: CLI entrypoint, MCP runtime, and higher-level indexing/runtime orchestration. The `web` subcommand delegates to `docbert-web` for the HTTP runtime.

### Direct dependencies

| Dependency           | Version                | Role in current code                                                                            |
| -------------------- | ---------------------- | ----------------------------------------------------------------------------------------------- |
| `docbert-core`       | path `../docbert-core` | Shared storage, indexing, search, embedding, and model primitives used by the application crate |
| `docbert-web`        | path `../docbert-web`  | Web server runtime that the `web` subcommand delegates to                                       |
| `clap`               | `4.6.1`                | CLI parsing and command definitions in `src/cli.rs`                                             |
| `clap_complete`      | `4.6`                  | Generates shell completion scripts                                                              |
| `globset`            | `0.4`                  | Glob filtering for MCP resource handling and some search/file filtering paths                   |
| `kdam`               | `0.6.4`                | Progress bars/spinners for indexing and embedding work in CLI flows                             |
| `percent-encoding`   | `2`                    | URI/resource encoding helpers in the MCP layer                                                  |
| `rmcp`               | `1.5.0`                | MCP server implementation over stdio (`transport-io` feature)                                   |
| `schemars`           | `1.2.1`                | JSON schema generation for MCP tool/input shapes                                                |
| `serde`              | `1`                    | Serialization/deserialization for MCP request/response types and CLI JSON output                |
| `serde_json`         | `1`                    | JSON values and serialization for MCP payloads and CLI JSON output                              |
| `tokio`              | `1`                    | Async runtime for the MCP server                                                                |
| `tracing`            | `0.1`                  | Runtime logging instrumentation                                                                 |
| `tracing-subscriber` | `0.3`                  | Logging initialization and env-filter support                                                   |
| `xdg`                | `3.0.0`                | Resolves the default data directory for the app                                                 |

### Direct dev-dependencies

| Dependency  | Version | Role in current tests                                               |
| ----------- | ------- | ------------------------------------------------------------------- |
| `hegeltest` | `0.10`  | Property-based / parameterized test helpers                         |
| `rmcp`      | `1.5.0` | MCP client-side test support (`client` + `transport-child-process`) |
| `tempfile`  | `3`     | Temporary directories/files in tests                                |

### `docbert` feature relationships

`docbert` does not define its own runtime backend matrix. Each of its feature flags forwards to both `docbert-core` and `docbert-web`:

```toml
[features]
default = []
mkl = ["docbert-core/mkl", "docbert-web/mkl"]
accelerate = ["docbert-core/accelerate", "docbert-web/accelerate"]
metal = ["docbert-core/metal", "docbert-web/metal"]
cuda = ["docbert-core/cuda", "docbert-web/cuda"]
```

That means the application crate's acceleration/build choices are controlled by the core crate's model backend features, reached both directly and through `docbert-web`.

### Notes on major application dependencies

#### `rmcp`

Used in `crates/docbert/src/mcp.rs` for:

- MCP server wiring
- stdio transport
- tool/prompt/resource definitions
- MCP request/response types and errors

#### `tokio`

Used for:

- the MCP runtime (`mcp.rs` builds a multi-thread runtime)
- async tests around the MCP surface

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
| `half`           | `2.7`                                                                    | `bf16` conversion for embedding storage (`embedding_db.rs`)                                        |
| `heed`           | `0.22`                                                                   | LMDB-backed `config.db` and `embeddings.db` storage with multi-process readers and writers         |
| `ignore`         | `0.4`                                                                    | Filesystem walking with optional Git-ignore-aware discovery                                        |
| `kodama`         | `0.3`                                                                    | Hierarchical Ward clustering for ColBERT token pooling                                             |
| `pdf_oxide`      | `0.3.35`                                                                 | PDF-to-markdown/text extraction during preparation                                                 |
| `rayon`          | `1.12.0`                                                                 | Parallel document loading/preparation work                                                         |
| `rkyv`           | `0.8.15`                                                                 | Binary serialization for typed stored data                                                         |
| `serde`          | `1`                                                                      | Serialization support for public/config/runtime-facing data types                                  |
| `serde_json`     | `1`                                                                      | JSON values and parsing for metadata, settings, and conversation payloads                          |
| `tantivy`        | `0.26.0`                                                                 | Lexical indexing and BM25/fuzzy retrieval                                                          |
| `thiserror`      | `2`                                                                      | Error definition for `docbert_core::Error`                                                         |
| `tracing`        | `0.1`                                                                    | Logging instrumentation for the storage and indexing paths                                         |

### Direct dev-dependencies

| Dependency  | Version | Role in current tests                                              |
| ----------- | ------- | ------------------------------------------------------------------ |
| `criterion` | `0.8`   | Benchmarks (`embedding_trim`, `embedding_compression`)             |
| `hegeltest` | `0.10`  | Property-based / parameterized test helpers used by the core tests |
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

Used for:

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

#### `heed`

Wraps [LMDB](https://www.symas.com/lmdb) for both major local databases:

- `config.db`
- `embeddings.db`

Used for:

- collection/config storage
- document metadata
- conversations
- collection Merkle snapshots
- settings and JSON metadata blobs
- chunk byte offsets
- embedding matrix persistence

LMDB's reader/writer locks let multiple `docbert mcp` / `docbert web` / CLI processes share one data dir without stepping on each other. Files still in the pre-1.0 redb format are refused on open; `docbert clean` resets them.

#### `rkyv`

Used for stable typed binary storage of structures such as:

- document metadata
- conversations
- stored JSON wrappers
- Merkle snapshot structures

#### `ignore`

Used in `walker.rs` for recursive discovery.

Discovery uses it for:

- hidden-file filtering
- supported-extension filtering
- Git ignore handling when the collection root is itself a Git repo

#### `pdf_oxide`

Used in `preparation.rs` to:

- load PDF bytes
- convert PDFs to markdown when possible
- fall back to extracted text when markdown conversion is empty

## `crates/docbert-web`

`docbert-web` is the web server crate behind `docbert web`: the Axum HTTP API, conversation/chat endpoints, document routes, and the ChatGPT Codex OAuth settings flow. It serves the embedded browser UI from `docbert-webui` through the default `webui` feature.

### Direct dependencies

| Dependency      | Version                            | Role in current code                                                                                                |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `docbert-core`  | path `../docbert-core`             | Storage, indexing, search, embedding, and model primitives behind the HTTP API                                      |
| `docbert-webui` | path `../docbert-webui` (optional) | Embedded browser UI assets and SPA fallback handler; enabled by the default `webui` feature                         |
| `axum`          | `0.8`                              | HTTP routing, state extraction, and JSON request/response handling                                                  |
| `base64`        | `0.22`                             | PDF upload decoding in document routes and URL-safe encoding in the ChatGPT Codex OAuth flow (`routes/settings.rs`) |
| `rand`          | `0.10`                             | OAuth state / PKCE verifier generation for the ChatGPT Codex login flow (`routes/settings.rs`)                      |
| `reqwest`       | `0.13.2`                           | Outbound HTTP client for the OAuth token exchange (`json`, `form`, `rustls` features; no defaults)                  |
| `serde`         | `1`                                | Serialization/deserialization for HTTP request/response types                                                       |
| `serde_json`    | `1`                                | JSON values and serialization for HTTP payloads                                                                     |
| `sha2`          | `0.11`                             | PKCE code-challenge hashing for the ChatGPT Codex OAuth flow (`routes/settings.rs`)                                 |
| `tantivy`       | `0.26.0`                           | Shared `IndexWriter` and lock-failure classification in the web runtime (`state.rs`, `runtime.rs`)                  |
| `tokio`         | `1`                                | Async runtime for the web server                                                                                    |
| `tracing`       | `0.1`                              | Runtime logging instrumentation                                                                                     |

### Direct dev-dependencies

| Dependency  | Version  | Role in current tests                                |
| ----------- | -------- | ---------------------------------------------------- |
| `pdf_oxide` | `0.3.35` | Test PDF generation/helpers for document route tests |
| `tempfile`  | `3`      | Temporary directories/files in tests                 |
| `tower`     | `0.5`    | Test utilities for Axum services                     |

### `docbert-web` feature relationships

```toml
[features]
default = ["webui"]
webui = ["dep:docbert-webui"]
mkl = ["docbert-core/mkl"]
accelerate = ["docbert-core/accelerate"]
metal = ["docbert-core/metal"]
cuda = ["docbert-core/cuda"]
```

The `webui` feature (on by default) pulls in the embedded browser UI; the acceleration flags forward to `docbert-core`.

## `crates/docbert-webui`

`docbert-webui` embeds the built browser UI. Its `build.rs` builds the frontend under `ui/` with bun (`bun install --frozen-lockfile` + `bun run build`, falling back to npm), and the crate embeds the resulting `ui/dist` output at compile time.

### Direct dependencies

| Dependency    | Version | Role in current code                                                     |
| ------------- | ------- | ------------------------------------------------------------------------ |
| `axum`        | `0.8`   | Request/response types for the fallback handler serving the embedded SPA |
| `include_dir` | `0.7`   | Embeds `ui/dist` into the binary at compile time                         |

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

| Dependency  | Version | Role in current tests                                   |
| ----------- | ------- | ------------------------------------------------------- |
| `criterion` | `0.8`   | Benchmarks (`kmeans`, `codec`, `search`, `build_index`) |
| `hegeltest` | `0.10`  | Property-based / parameterized test helpers             |
| `rand`      | `0.10`  | Random fixtures for kmeans/MaxSim tests                 |
| `tempfile`  | `3`     | Temporary directories for index round-trip tests        |

### `docbert-plaid` feature relationships

```toml
[features]
default = []
cuda = ["candle-core/cuda"]
```

## `crates/docbert-pylate`

`docbert-pylate` is the Rust-only fork of [pylate-rs](https://github.com/lightonai/pylate-rs) (originally based on upstream 1.0.4) that has been vendored into the workspace. The upstream Python, WebAssembly, and npm packaging layers were removed; the crate tracks the workspace release version rather than upstream's.

It owns the ColBERT-family late-interaction model loading, query/document encoding, and token-level similarity work used by `docbert-core::ModelManager`.

### Direct dependencies

| Dependency            | Version   | Role in current code                                                                     |
| --------------------- | --------- | ---------------------------------------------------------------------------------------- |
| `candle-core`         | `0.10.2`  | Tensor representation and ops for inference                                              |
| `candle-nn`           | `0.10.2`  | Neural-network primitives used by the model stack                                        |
| `candle-transformers` | `0.10.2`  | Transformer building blocks (ModernBERT encoder, pooling, etc.)                          |
| `candle-flash-attn`   | `0.10.2`  | Optional flash-attention kernel; enabled only through the `cuda` feature                 |
| `tokenizers`          | `0.23`    | HuggingFace tokenizer runtime (`onig` backend)                                           |
| `serde`               | `1.0.228` | Model config / metadata deserialization                                                  |
| `serde_json`          | `1.0.149` | JSON parsing for model configuration files                                               |
| `thiserror`           | `2.0.18`  | Error definitions                                                                        |
| `hf-hub`              | `0.5.0`   | Downloads model weights and configs from HuggingFace (`ureq`, `rustls-tls`; no defaults) |
| `kodama`              | `0.3.0`   | Hierarchical Ward clustering used by token-pooling encode paths                          |
| `rayon`               | `1.12.0`  | Parallelism in encoding/batching paths                                                   |

### Direct dev-dependencies

| Dependency  | Version   | Role in current tests                       |
| ----------- | --------- | ------------------------------------------- |
| `anyhow`    | `1.0.102` | Loose error chaining inside test helpers    |
| `criterion` | `0.8`     | Benchmarks (`encode_batch_size`)            |
| `hegeltest` | `0.10`    | Property-based / parameterized test helpers |

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
| `cargo-lock`           | `11`                                                    | Parse `Cargo.lock` for `rustbert sync`                                                                                                |
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
| `toml`                 | `1`                                                     | Read crate `Cargo.toml` files extracted from tarballs                                                                                 |
| `tracing`              | `0.1`                                                   | Logging                                                                                                                               |
| `tracing-subscriber`   | `0.3` (`env-filter`)                                    | Logger init driven by `RUSTBERT_LOG`                                                                                                  |

### Direct dev-dependencies

| Dependency  | Version | Role in current tests                                  |
| ----------- | ------- | ------------------------------------------------------ |
| `flate2`    | `1`     | Build synthetic tarballs                               |
| `hegeltest` | `0.10`  | Property-based / parameterized test helpers            |
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

The MCP server is hand-rolled JSON-RPC over stdio; rustbert deliberately does not depend on `rmcp` or `schemars`. There is no `xdg` dep either: data-dir resolution is done in-tree against `RUSTBERT_DATA_DIR` and `XDG_DATA_HOME`.

## Cross-crate relationships

A few relationships matter more than the raw version list.

### `docbert` depends on `docbert-core` and `docbert-web`

The application crate reuses the core crate for:

- `ConfigDb`
- `DataDir`
- `EmbeddingDb`
- `SearchIndex`
- `ModelManager`
- search functions
- document preparation and indexing helpers

That is why most search/storage dependency weight lives in `docbert-core`, not `docbert`. The web/HTTP dependency weight (`axum`, `reqwest`, `sha2`, `rand`, ...) lives in `docbert-web`, which the `web` subcommand delegates to.

### Feature flags flow from app to core to `docbert-pylate`

The feature chain is:

```text
docbert feature -> docbert-core / docbert-web features -> docbert-pylate backend feature
```

For example:

```text
cargo build -p docbert --features cuda
    -> enables docbert-core/cuda and docbert-web/cuda
    -> docbert-web/cuda forwards to docbert-core/cuda
    -> docbert-core/cuda enables docbert-pylate/cuda and docbert-plaid/cuda
```

### Some crates appear in several manifests for different reasons

- `tantivy`
  - core crate: main index abstraction and retrieval
  - web crate: shared index writer and lock-failure classification in the web runtime
  - app crate: `IndexWriter` handles obtained from `docbert-core`'s `SearchIndex` in CLI indexing flows
- `serde` / `serde_json`
  - core crate: stored/config/runtime data types
  - web crate: HTTP payload types
  - app crate: MCP payload types and CLI JSON output
- `pdf_oxide`
  - core crate: actual PDF preparation support
  - web crate dev-dependency: test helpers for PDF upload coverage

## Related references

- [`architecture.md`](./architecture.md)
- [`pipeline.md`](./pipeline.md)
- [`storage.md`](./storage.md)
- [`library-usage.md`](./library-usage.md)
- [`rustbert.md`](./rustbert.md)
