# Dependencies

This page shows the direct Cargo dependencies in these manifests:

- workspace root `Cargo.toml`
- `crates/docbert/Cargo.toml`
- `crates/docbert-core/Cargo.toml`
- `crates/docbert-web/Cargo.toml`
- `crates/docbert-webui/Cargo.toml`
- `crates/docbert-plaid/Cargo.toml`
- `crates/docbert-pylate/Cargo.toml`
- `crates/rustbert/Cargo.toml`

This page shows what each direct dependency does in this codebase. It also shows the primary feature relationships. These relationships are important when you change the build or the runtime function.

## Workspace root

The workspace root has no direct Rust dependencies.

It has only these two items:

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

All dependency versions are in the crate manifests.

## `crates/docbert`

`docbert` is the application crate. It contains the CLI entrypoint, the MCP runtime, and the application-level control of indexing and runtime. The `web` subcommand uses `docbert-web` for the HTTP runtime.

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

`docbert` has no runtime backend matrix. Each `docbert` feature flag activates the same feature in `docbert-core` and `docbert-web`:

```toml
[features]
default = []
mkl = ["docbert-core/mkl", "docbert-web/mkl"]
accelerate = ["docbert-core/accelerate", "docbert-web/accelerate"]
metal = ["docbert-core/metal", "docbert-web/metal"]
cuda = ["docbert-core/cuda", "docbert-web/cuda"]
```

Thus the core crate's model backend features control the acceleration and build selections of the application crate. The application crate gets these features directly and through `docbert-web`.

### Notes on primary application dependencies

#### `rmcp`

`docbert` uses `rmcp` in `crates/docbert/src/mcp.rs` for these functions:

- The MCP server connections
- The stdio transport
- The tool, prompt, and resource definitions
- The MCP request and response types, and the errors.

#### `tokio`

`docbert` uses `tokio` for these functions:

- The MCP runtime (`mcp.rs` makes a multi-thread runtime)
- The async tests of the MCP surface.

## `crates/docbert-core`

`docbert-core` is the library crate that other crates use. It contains storage, search, indexing helpers, model management, chunking, and document preparation.

### Direct dependencies

| Dependency       | Version                                                                  | Role in current code                                                                               |
| ---------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `blake3`         | `1.8.4`                                                                  | Merkle snapshot hashing                                                                            |
| `bytemuck`       | `1.25.0` (`derive`)                                                      | Efficient `f32`/byte conversions in embedding storage. Feature `derive` for `Pod`/`Zeroable` impls |
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

`docbert-core` controls the model-backend feature mapping:

```toml
[features]
default = []
mkl = ["docbert-pylate/mkl"]
accelerate = ["docbert-pylate/accelerate"]
metal = ["docbert-pylate/metal"]
cuda = ["docbert-pylate/cuda", "docbert-plaid/cuda"]
```

These are the primary build-time flags for faster inference.

### Notes on primary core dependencies

#### `docbert-pylate`

`docbert-pylate` is the primary ColBERT integration layer.

`docbert-core` uses `docbert-pylate` for these functions:

- The model loading in `model_manager.rs`
- The query and document encoding
- The similarity computation for reranking.

The workspace has a copy of `docbert-pylate` under `crates/docbert-pylate`. `docbert-pylate` is a Rust-only fork of [`pylate-rs`](https://github.com/lightonai/pylate-rs). This fork starts from pylate-rs 1.0.4. The fork removes the upstream Python, WebAssembly, and npm packaging layers. Only code inside the workspace uses the crate. The crate uses the version number of the workspace release, not the upstream version number.

#### `tantivy`

`docbert-core` uses `tantivy` for these functions:

- The schema definition
- The lexical indexes on disk or in memory
- BM25 retrieval
- The collection and path lookups
- The fuzzy-matching support.

#### `heed`

`heed` is the [LMDB](https://www.symas.com/lmdb) wrapper for the two primary local databases:

- `config.db`
- `embeddings.db`

`docbert-core` uses `heed` to keep these items:

- The collection and config data
- The document metadata
- The conversations
- The collection Merkle snapshots
- The settings and JSON metadata blobs
- The chunk byte offsets
- The embedding matrix.

The LMDB reader and writer locks let many processes use one data dir at the same time. These processes include `docbert mcp`, `docbert web`, and the CLI. `docbert` does not open files in the pre-1.0 redb format. The `docbert clean` command sets these files to the initial state.

#### `rkyv`

`docbert-core` uses `rkyv` to keep these typed structures in a stable binary format:

- The document metadata
- The conversations
- The JSON wrappers
- The Merkle snapshot structures.

#### `ignore`

`docbert-core` uses `ignore` in `walker.rs` for the recursive search of files. This search has these functions:

- The hidden-file filtering
- The supported-extension filtering
- The Git-ignore rules, when the collection root is a Git repo.

#### `pdf_oxide`

`docbert-core` uses `pdf_oxide` in `preparation.rs`. `pdf_oxide` loads the PDF bytes and converts the PDFs to markdown when possible. When the markdown output is empty, `docbert-core` uses the extracted text.

## `crates/docbert-web`

`docbert-web` is the web server crate behind `docbert web`. It contains the Axum HTTP API, the conversation and chat endpoints, the document routes, and the OAuth settings flow for ChatGPT Codex. It supplies the embedded browser UI from `docbert-webui` through the default `webui` feature.

### Direct dependencies

| Dependency      | Version                            | Role in current code                                                                                                |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `docbert-core`  | path `../docbert-core`             | Storage, indexing, search, embedding, and model primitives behind the HTTP API                                      |
| `docbert-webui` | path `../docbert-webui` (optional) | Embedded browser UI assets and SPA fallback handler. The default `webui` feature includes them                      |
| `axum`          | `0.8`                              | HTTP routing, state extraction, and JSON request/response handling                                                  |
| `base64`        | `0.22`                             | PDF upload decoding in document routes and URL-safe encoding in the ChatGPT Codex OAuth flow (`routes/settings.rs`) |
| `rand`          | `0.10`                             | OAuth state / PKCE verifier generation for the ChatGPT Codex login flow (`routes/settings.rs`)                      |
| `reqwest`       | `0.13.2`                           | Outbound HTTP client for the OAuth token exchange (`json`, `form`, `rustls` features, no defaults)                  |
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

The `webui` feature (on by default) includes the embedded browser UI. The acceleration flags activate the same features in `docbert-core`.

## `crates/docbert-webui`

`docbert-webui` embeds the built browser UI. The `build.rs` script builds the frontend under `ui/` with bun. The commands are `bun install --frozen-lockfile` and `bun run build`. If bun is not available, `build.rs` uses npm. The crate then embeds the `ui/dist` output at compile time.

### Direct dependencies

| Dependency    | Version | Role in current code                                                     |
| ------------- | ------- | ------------------------------------------------------------------------ |
| `axum`        | `0.8`   | Request/response types for the fallback handler serving the embedded SPA |
| `include_dir` | `0.7`   | Embeds `ui/dist` into the binary at compile time                         |

## `crates/docbert-plaid`

`docbert-plaid` is the workspace-local crate for the PLAID multi-vector index. ColBERT late-interaction retrieval uses this index. `docbert-plaid` has no dependency on `docbert-core`. `docbert-core` has a dependency on `docbert-plaid`.

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

`docbert-pylate` is the Rust-only fork of [pylate-rs](https://github.com/lightonai/pylate-rs). This fork starts from pylate-rs 1.0.4. The workspace has a copy of it. The fork removes the upstream Python, WebAssembly, and npm packaging layers. The crate uses the version number of the workspace release, not the upstream version number.

`docbert-pylate` contains the model loading, the query and document encoding, and the token-level similarity work for the ColBERT-family late-interaction models. `docbert-core::ModelManager` uses this work.

### Direct dependencies

| Dependency            | Version   | Role in current code                                                                     |
| --------------------- | --------- | ---------------------------------------------------------------------------------------- |
| `candle-core`         | `0.10.2`  | Tensor representation and ops for inference                                              |
| `candle-nn`           | `0.10.2`  | Neural-network primitives used by the model stack                                        |
| `candle-transformers` | `0.10.2`  | Transformer building blocks (ModernBERT encoder, pooling, and other blocks)              |
| `candle-flash-attn`   | `0.10.2`  | Optional flash-attention kernel. It is active only with the `cuda` feature               |
| `tokenizers`          | `0.23`    | HuggingFace tokenizer runtime (`onig` backend)                                           |
| `serde`               | `1.0.228` | Model config / metadata deserialization                                                  |
| `serde_json`          | `1.0.149` | JSON parsing for model configuration files                                               |
| `thiserror`           | `2.0.18`  | Error definitions                                                                        |
| `hf-hub`              | `0.5.0`   | Downloads model weights and configs from HuggingFace (`ureq`, `rustls-tls`, no defaults) |
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

`docbert-core`'s `mkl`, `accelerate`, `metal`, and `cuda` features activate these leaf flags.

## `crates/rustbert`

`rustbert` is a binary that is not part of `docbert`. It has a dependency on `docbert-core` as a library. `rustbert` has a crates.io fetcher, a parser, and an MCP server. [`rustbert.md`](./rustbert.md) gives the full description. The manifest section there is the correct dependency list. The summary below shows the same data as `crates/rustbert/Cargo.toml`.

### Direct dependencies

| Dependency             | Version                                                 | Role in current code                                                                                                                  |
| ---------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `docbert-core`         | path `../docbert-core`                                  | Storage / index / search / model primitives                                                                                           |
| `cargo-lock`           | `11`                                                    | Parse `Cargo.lock` for `rustbert sync`                                                                                                |
| `clap`                 | `4.6` (`derive`, `env`)                                 | CLI parsing                                                                                                                           |
| `flate2`               | `1`                                                     | Gzip decode for crates.io tarballs                                                                                                    |
| `globset`              | `0.4`                                                   | `--exclude` glob filtering in `rustbert sync`                                                                                         |
| `proc-macro2`          | `1` (`span-locations`)                                  | Feature-only direct dep so `tt.span().start().line` returns real line numbers. Cargo-machete is configured to ignore the absent `use` |
| `quote`                | `1`                                                     | Token-tree rendering for synthesized signatures                                                                                       |
| `reqwest`              | `0.13.2` (`rustls`, `stream`, no defaults)              | HTTP client for crates.io / docs.rs                                                                                                   |
| `semver`               | `1` (`serde`)                                           | Version resolution                                                                                                                    |
| `serde` / `serde_json` | `1`                                                     | crates.io API and metadata blobs                                                                                                      |
| `sha2`                 | `0.11`                                                  | Tarball checksum check                                                                                                                |
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

The `rustbert` MCP server uses JSON-RPC over stdio without a library. Thus `rustbert` has no dependency on `rmcp` or `schemars`. `rustbert` also has no `xdg` dependency. `rustbert` finds the data dir in the source tree from `RUSTBERT_DATA_DIR` and `XDG_DATA_HOME`.

## Cross-crate relationships

A few relationships are more important than the version list.

### `docbert` uses `docbert-core` and `docbert-web`

The application crate uses the core crate for these items:

- `ConfigDb`
- `DataDir`
- `EmbeddingDb`
- `SearchIndex`
- `ModelManager`
- search functions
- document preparation and indexing helpers

Thus most search and storage dependencies are in `docbert-core`, not `docbert`. The web and HTTP dependencies (`axum`, `reqwest`, `sha2`, `rand`, and others) are in `docbert-web`. The `web` subcommand uses `docbert-web`.

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

### Some crates are in more than one manifest

- `tantivy`
  - core crate: primary index abstraction and retrieval
  - web crate: shared index writer and lock-failure classification in the web runtime
  - app crate: `IndexWriter` handles from `docbert-core`'s `SearchIndex` in the CLI indexing paths
- `serde` / `serde_json`
  - core crate: stored/config/runtime data types
  - web crate: HTTP payload types
  - app crate: MCP payload types and CLI JSON output
- `pdf_oxide`
  - core crate: PDF preparation support
  - web crate dev-dependency: test helpers for PDF upload coverage

## Related references

- [`architecture.md`](./architecture.md)
- [`pipeline.md`](./pipeline.md)
- [`storage.md`](./storage.md)
- [`library-usage.md`](./library-usage.md)
- [`rustbert.md`](./rustbert.md)
