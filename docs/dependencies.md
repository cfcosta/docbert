# Dependencies

This page tracks the **direct Cargo dependencies** declared in the current manifests:

- workspace root `Cargo.toml`
- `crates/docbert/Cargo.toml`
- `crates/docbert-core/Cargo.toml`

It focuses on what each direct dependency is for in the current codebase, plus the major feature relationships that matter when changing builds or runtime behavior.

## Workspace root

The workspace root currently declares **no direct Rust dependencies**.

It only defines:

- workspace members:
  - `crates/docbert`
  - `crates/docbert-core`
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
| `clap`               | `4.5.57`               | CLI parsing and command definitions in `src/cli.rs`                                             |
| `include_dir`        | `0.7`                  | Embeds built UI assets for the web runtime                                                      |
| `clap_complete`      | `4.5`                  | Generates shell completion scripts                                                              |
| `globset`            | `0.4`                  | Glob filtering for MCP resource handling and some search/file filtering paths                   |
| `base64`             | `0.22`                 | PDF upload encoding/decoding in web document routes                                             |
| `kdam`               | `0.6.4`                | Progress bars/spinners for indexing and embedding work in CLI flows                             |
| `percent-encoding`   | `2`                    | URI/resource encoding helpers in the MCP layer                                                  |
| `rmcp`               | `1.2.0`                | MCP server implementation over stdio                                                            |
| `schemars`           | `1.2.1`                | JSON schema generation for MCP tool/input shapes                                                |
| `serde`              | `1`                    | Serialization/deserialization for web and MCP request/response types                            |
| `serde_json`         | `1`                    | JSON values and serialization for web/MCP payloads                                              |
| `tantivy`            | `0.25`                 | Direct access to Tantivy writer/lock types in runtime resource handling                         |
| `tokio`              | `1`                    | Async runtime for web server and MCP runtime                                                    |
| `tracing`            | `0.1`                  | Runtime logging instrumentation                                                                 |
| `tracing-subscriber` | `0.3`                  | Logging initialization and env-filter support                                                   |
| `xdg`                | `3.0.0`                | Resolves the default data directory for the app                                                 |

### Direct dev-dependencies

| Dependency  | Version  | Role in current tests                                    |
| ----------- | -------- | -------------------------------------------------------- |
| `pdf_oxide` | `0.3.21` | Test PDF generation/helpers for web document route tests |
| `rmcp`      | `1.2.0`  | MCP client-side test support                             |
| `tempfile`  | `3`      | Temporary directories/files in tests                     |
| `tower`     | `0.5`    | Test utilities for Axum services                         |

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

| Dependency    | Version                                                         | Role in current code                                                      |
| ------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `blake3`      | `1.8.2`                                                         | Merkle snapshot hashing                                                   |
| `bytemuck`    | `1.25.0`                                                        | Efficient `f32`/byte conversions in embedding storage                     |
| `candle-core` | `0.9.1`                                                         | Tensor representation and tensor operations for model/embedding work      |
| `hf-hub`      | `0.5.0`                                                         | Fetches remote model metadata such as `config_sentence_transformers.json` |
| `ignore`      | `0.4`                                                           | Filesystem walking with optional Git-ignore-aware discovery               |
| `pylate-rs`   | `1.0.4` from git rev `4a014da31ab13faef5155aefb92851100cca5035` | ColBERT model loading, query/document encoding, and similarity scoring    |
| `rayon`       | `1.11.0`                                                        | Parallel document loading/preparation work                                |
| `redb`        | `3.1.0`                                                         | `config.db` and `embeddings.db` storage                                   |
| `rkyv`        | `0.8.15`                                                        | Binary serialization for typed stored data                                |
| `pdf_oxide`   | `0.3.21`                                                        | PDF-to-markdown/text extraction during preparation                        |
| `serde`       | `1`                                                             | Serialization support for public/config/runtime-facing data types         |
| `serde_json`  | `1`                                                             | JSON values and parsing for metadata, settings, and conversation payloads |
| `tantivy`     | `0.25`                                                          | Lexical indexing and BM25/fuzzy retrieval                                 |
| `thiserror`   | `2`                                                             | Error definition for `docbert_core::Error`                                |

### Direct dev-dependencies

| Dependency | Version | Role in current tests                     |
| ---------- | ------- | ----------------------------------------- |
| `tempfile` | `3`     | Temporary directories/files in unit tests |

### `docbert-core` feature relationships

`docbert-core` owns the model-backend feature mapping:

```toml
[features]
default = []
mkl = ["pylate-rs/mkl"]
accelerate = ["pylate-rs/accelerate"]
metal = ["pylate-rs/metal"]
cuda = ["pylate-rs/cuda"]
```

These are the main build-time switches for accelerated inference.

### Notes on major core dependencies

#### `pylate-rs`

This is the main ColBERT integration layer.

Current uses include:

- model loading in `model_manager.rs`
- query/document encoding
- similarity computation used by reranking

Important current manifest detail: this is **not just a crates.io version pin**. The project depends on:

- version: `1.0.4`
- git source: `https://github.com/cfcosta/pylate-rs`
- rev: `4a014da31ab13faef5155aefb92851100cca5035`

That should stay documented because it affects reproducibility and debugging.

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

#### `hf-hub`

Used to resolve remote model metadata needed by the model manager, especially when docbert wants to read prompt/document-length-related configuration from Sentence Transformers exports.

The current manifest disables default features and enables `ureq`:

```toml
hf-hub = { version = "0.5.0", default-features = false, features = ["ureq"] }
```

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

### Feature flags flow from app to core to `pylate-rs`

The feature chain is:

```text
docbert feature -> docbert-core feature -> pylate-rs backend feature
```

For example:

```text
cargo build -p docbert --features cuda
    -> enables docbert-core/cuda
    -> enables pylate-rs/cuda
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

- `hf-hub` is `0.5.0`, not `0.4.3`
- `docbert` now has direct runtime/web/MCP dependencies such as `axum`, `rmcp`, `tokio`, `schemars`, `include_dir`, and `tracing`
- `docbert-core` has direct dependencies for Merkle snapshots and PDF handling (`blake3`, `pdf_oxide`, `ignore`)
- feature mapping is split across both crate manifests, not just one top-level example
- the workspace root has no direct dependency list of its own

## Related references

- [`architecture.md`](./architecture.md)
- [`pipeline.md`](./pipeline.md)
- [`storage.md`](./storage.md)
- [`library-usage.md`](./library-usage.md)
