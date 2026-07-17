# Architecture

## Overview

docbert is a local document retrieval system. The `docbert` binary has three user-facing entrypoints:

- the **CLI** (`docbert ...`)
- the **web runtime** (`docbert web`)
- the **MCP runtime** (`docbert mcp`)

All three sit on top of the same core storage and retrieval stack in `docbert-core`. The workspace also ships `rustbert`, a separate binary that reuses `docbert-core` for Rust crate docs lookup (see [`crates/rustbert`](#cratesrustbert) below).

At a high level, docbert:

1. registers one or more named filesystem collections
2. indexes Markdown, text, and PDF documents from those collections
3. stores lexical index data, metadata, embeddings, and settings locally
4. serves retrieval through the CLI, the web API/UI, or MCP tools/resources

docbert keeps state and files local: every runtime is a local process working over a local data directory.

## Product surfaces

### CLI

The CLI is the main operational interface for:

- collection registration
- context management
- search and retrieval
- indexing (`sync`, `rebuild`, `reindex`)
- maintenance and recovery (`clean` removes orphan and wrong-model embeddings and resets pre-1.0 legacy data)
- runtime inspection (`status`, `doctor`, `model show`)
- starting the web or MCP runtimes

The CLI parses commands in `crates/docbert/src/cli.rs` and dispatches them from `crates/docbert/src/main.rs` into command handlers.

### Web runtime

`docbert web` starts a single local process that serves:

- the browser UI
- the HTTP API documented in [`web-api.md`](./web-api.md)

The web runtime is implemented by the `docbert-web` crate. `main.rs` hands off to `docbert_web::run` (`crates/docbert-web/src/lib.rs`), which opens `config.db`, builds shared state through `state::init` (`crates/docbert-web/src/state.rs`), and starts the Axum server in `server::run` (`crates/docbert-web/src/server.rs`). The route surface lives in `crates/docbert-web/src/routes/*`.

### MCP runtime

`docbert mcp` starts a stdio-based MCP server for editor and agent integrations.

Its tools, prompt, and resource template are implemented in `crates/docbert/src/mcp.rs` and documented in [`mcp.md`](./mcp.md).

## Main crates and their roles

## `crates/docbert`

This is the application crate.

It owns:

- CLI parsing and top-level command dispatch
- indexing workflows and mutation orchestration
- MCP server setup and tool/resource handling
- handing `docbert web` off to the `docbert-web` crate
- runtime resource management around `config.db`, `embeddings.db`, `plaid.idx`, and Tantivy writers

Main modules:

- `src/main.rs`: process entrypoint, command dispatch, data-dir resolution, model resolution
- `src/cli.rs`: clap command surface
- `src/commands/*`: CLI behaviors
- `src/indexing.rs`: sync/rebuild planning and snapshot finalization
- `src/mcp.rs`: MCP runtime
- `src/runtime.rs`: blocking `config.db` open helper
- `src/snapshots.rs`: collection snapshot support around indexing/web mutations

## `crates/docbert-web`

This crate implements the Axum HTTP API and server behind `docbert web`. The `docbert` binary depends on it and enters through `docbert_web::run`.

It owns:

- shared web state construction (`src/state.rs`) and server startup (`src/server.rs`)
- the route surface in `src/routes/*` (`mod`, `search`, `documents`, `collections`, `conversations`, `settings`)
- per-document ingestion paths for web uploads (`src/ingest.rs`)
- collection snapshot refresh around web mutations (`src/snapshots.rs`)
- path resolution helpers (`src/paths.rs`) and blocking open/lock helpers (`src/runtime.rs`)

It depends on `docbert-core` for storage, search, and model management.

## `crates/docbert-webui`

This crate bundles the browser SPA. Its `build.rs` builds the React app in `ui/`, and `include_dir!` embeds the resulting `ui/dist` into the binary. At runtime it serves those assets as the fallback for every non-`/v1/` path.

It is an optional dependency of `docbert-web`, enabled by the default `webui` feature.

## `crates/docbert-core`

This is the shared library crate behind all entrypoints.

It owns:

- config and metadata persistence
- document ids and metadata models
- search/index abstractions
- embedding storage
- chunking and preparation
- ingestion helpers
- conversation storage types
- search execution and reranking logic
- filesystem walking/discovery rules

Public types re-exported by `docbert-core` include:

- `ConfigDb`
- `DataDir`
- `EmbeddingDb`
- `SearchIndex`
- `ModelManager`
- `Conversation`

## `crates/docbert-plaid`

This crate implements the PLAID multi-vector index used for semantic retrieval, following the pipeline described in "PLAID: An Efficient Engine for Late Interaction Retrieval" (Santhanam et al., 2022) and modelled on the `fast-plaid` reference implementation.

It owns:

- k-means centroid training over stored ColBERT token embeddings
- compressed codec for residual quantization
- MaxSim-based query evaluation against the compressed index
- on-disk `plaid.idx` file format
- CUDA-accelerated paths for k-means and MaxSim matmul when the `cuda` feature is enabled

`docbert-core::search::semantic` and the semantic leg of `docbert-core::search::run` both load this crate's index file from `DataDir::plaid_index()` and ask it to rank documents for an encoded query.

## `crates/docbert-pylate`

This crate is a Rust-only fork of [pylate-rs](https://github.com/lightonai/pylate-rs) (originally based on upstream 1.0.4), vendored into the workspace. The upstream Python, WebAssembly, and npm packaging layers have been removed; the crate tracks the workspace release version rather than upstream's.

It owns:

- ColBERT-family late-interaction model loading from HuggingFace or local paths
- query and document encoding
- token-level MaxSim similarity computation
- `candle`-backed inference with CPU, CUDA, Metal, MKL, and Accelerate backends selected via feature flags

`docbert-core::ModelManager` is a thin wrapper around this crate.

## `crates/rustbert`

`rustbert` is a separate sibling binary for Rust crate docs lookup: it fetches published crates and serves search over their public APIs, with its own CLI (`src/main.rs`) and MCP server (`src/mcp.rs`). It reuses `docbert-core` but is not wired into the `docbert` binary, and it is packaged independently (nix: `rustbert`, `rustbert-cuda`, `rustbert-metal`).

See [`rustbert.md`](./rustbert.md) for the full design.

## Core persistent state

docbert keeps local state under a resolved data directory managed by `DataDir`.

The major persistent pieces are:

- `config.db`
  - collections
  - contexts
  - document metadata
  - chunk byte offsets
  - conversations
  - collection Merkle snapshots
  - settings, including model and LLM-related values
- `embeddings.db`
  - ColBERT token embeddings
- `plaid.idx`
  - PLAID multi-vector index built over the embeddings
- `tantivy/`
  - lexical search index
- source collection directories on disk
  - treated as the authoritative document content for many read paths

For storage details, see [`storage.md`](./storage.md).

## Architectural layers

## 1. Collection and settings layer

Implemented primarily through `ConfigDb`.

Responsibilities:

- map collection names to filesystem roots
- store optional context strings
- store document metadata and user metadata
- store chunk byte offsets so search consumers can surface the byte range of a matching chunk
- store persisted conversation history
- store collection Merkle snapshots for sync/web mutation tracking
- store general settings such as model selection and persisted LLM settings

This layer is shared across CLI, web, and MCP surfaces.

## 2. Discovery and ingestion layer

Implemented across `walker`, `ingestion`, `preparation`, and `indexing`.

Responsibilities:

- discover eligible files from collection roots
- respect the walker rules, including Git ignore behavior for repo-backed collections
- load Markdown, text, and PDF files from disk
- convert PDFs into extracted Markdown/text for preview, search, and embeddings
- derive titles and metadata used by search and API responses
- produce chunk/embedding-ready document representations
- update collection snapshots after successful sync/rebuild or web mutations

## 3. Retrieval layer

Implemented around `SearchIndex`, `EmbeddingDb`, `search`, and `ModelManager`.

Responsibilities:

- lexical retrieval through Tantivy
- semantic scoring through ColBERT embeddings
- hybrid and semantic search modes
- reference resolution for `get`-style reads
- excerpt/snippet extraction from on-disk document content

This layer serves all three product surfaces.

## 4. Runtime surface layer

The `docbert` binary exposes the retrieval and storage layers through three runtimes:

- CLI handlers
- Axum HTTP routes for the web runtime
- RMCP tools/prompts/resources for the MCP runtime

Each runtime has its own request/response shape, but they share the same underlying data model and index state.

## Component map

```text
Collections on disk
        |
        v
  walker / ingestion / preparation
        |
        +-------------------+
        |                   |
        v                   v
   SearchIndex          EmbeddingDb
   (Tantivy)            (ColBERT vectors)
                            |
                            v
                       plaid.idx
                       (docbert-plaid multi-vector index)
        |                   |
        +---------+---------+
                  |
                  v
          docbert-core::search
          (BM25 + ColBERT/PLAID + RRF)
                  |
     +------------+-------------+
     |            |             |
     v            v             v
   CLI        Web runtime    MCP runtime
```

ModelManager (wrapping `docbert-pylate`) sits alongside `search` and is passed into every query path to encode the query tokens on demand.

## Runtime boundaries

## CLI runtime

`main.rs` resolves:

- data directory
- model configuration
- which command path to run

Not every command initializes the same resources.

Examples:

- `doctor` and `completions` are handled early, before the data directory is resolved
- `clean` is dispatched before the shared `ConfigDb::open`: it is the recovery path for data directories left in the pre-1.0 redb format, which `ConfigDb::open` refuses with `Error::LegacyDatabase` (an error whose message points at `docbert clean`)
- `web` and `mcp` resolve the model, then hand off into their long-lived runtimes
- most other commands open `ConfigDb` and run as short-lived operations

## Web runtime boundary

The web runtime is the `docbert web` process, implemented by the `docbert-web` crate.

Boundary details:

- one local process serves the SPA and the HTTP API
- the runtime is entered through `docbert_web::run`, which initializes shared state via `state::init` in the `docbert-web` crate
- route handlers open `config.db` / `embeddings.db` as needed
- document uploads and deletes mutate both source files and indexed state
- search reads use the shared in-process search index and model manager

For the concrete route contract, see [`web-api.md`](./web-api.md).

## MCP runtime boundary

The MCP runtime is a separate long-lived stdio server.

Boundary details:

- it keeps a shared `SearchIndex` and `ModelManager` in process state
- it reopens `config.db` and `embeddings.db` for calls and resource reads
- it exposes tools, one prompt, and one resource template
- retrieval tools can return plain text, structured JSON, resources, or a mix depending on the operation

For the concrete MCP contract, see [`mcp.md`](./mcp.md).

## Chat-related architecture

docbert's chat system is a browser-side agent that calls the docbert HTTP API and the configured LLM provider. It is built from:

- persisted conversation records in `config.db`
- LLM settings in `config.db`
- web API routes for conversations and settings
- UI/runtime orchestration in the browser client

Boundary details:

- conversation persistence and LLM settings are backend concerns
- chat orchestration strategy is largely a UI/runtime concern
- prompt instructions encourage multi-search and multi-file synthesis, but those are not independent backend guarantees

For the concrete persisted schema and route behavior, see [`chat-and-conversations.md`](./chat-and-conversations.md).

## Search architecture

The retrieval stack supports two main modes:

- **hybrid**: lexical retrieval plus ColBERT reranking
- **semantic**: semantic scoring path through ColBERT-only search

Both modes ultimately depend on the same local metadata, source files, embeddings, and model runtime.

Some response enrichment is pulled from disk at read time:

- search titles may be recomputed from the current document content on disk
- excerpts/snippets come from current file content on disk
- single-document reads also come directly from source files on disk

That means the source collection directories are part of the live read architecture as well as the indexing pipeline.

For pipeline details, see [`pipeline.md`](./pipeline.md).

## Concurrency and resource management

docbert relies on local-process concurrency rather than distributed coordination.

Patterns:

- Tantivy supplies read/write index primitives
- LMDB-backed stores are opened per operation where appropriate
- long-lived runtimes keep search/model state alive, but do not hold every storage handle permanently open
- mutation-heavy operations open writers around the work that needs them
- retry behavior is limited to the Tantivy index-writer lock: the web runtime retries writer acquisition on lock contention (`crates/docbert-web/src/runtime.rs`), while heed/LMDB database opens succeed without retries because LMDB supports concurrent handles

The exact persistence and lock-sensitive storage details are documented in [`storage.md`](./storage.md).

## Terminology guide

Use these terms consistently:

- **collection**: a named root directory registered in `config.db`
- **document**: one indexed source file within a collection
- **conversation**: persisted chat history record
- **web runtime** / **web server**: the process started by `docbert web`
- **MCP runtime** / **MCP server**: the process started by `docbert mcp`
- **hybrid search**: lexical retrieval plus semantic reranking
- **semantic search**: semantic-only retrieval path

## Related references

- [`cli.md`](./cli.md)
- [`web-api.md`](./web-api.md)
- [`chat-and-conversations.md`](./chat-and-conversations.md)
- [`mcp.md`](./mcp.md)
- [`pipeline.md`](./pipeline.md)
- [`storage.md`](./storage.md)
- [`rustbert.md`](./rustbert.md)
- [`library-usage.md`](./library-usage.md)
