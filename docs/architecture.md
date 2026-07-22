# Architecture

## Overview

docbert is a local document retrieval system. The `docbert` binary has three entrypoints for users:

- the **CLI** (`docbert ...`)
- the **web runtime** (`docbert web`)
- the **MCP runtime** (`docbert mcp`)

All three use the same core storage and retrieval stack in `docbert-core`. The workspace also includes `rustbert`. `rustbert` is a different binary that uses `docbert-core` for Rust crate docs lookup (refer to [`crates/rustbert`](#cratesrustbert) below).

docbert does these tasks:

1. It records one or more named filesystem collections.
2. It indexes Markdown, text, and PDF documents from these collections.
3. It keeps the lexical index data, metadata, embeddings, and settings locally.
4. It gives retrieval results through the CLI, the web API and UI, or the MCP tools and resources.

docbert keeps the state and files local. Each runtime is a local process that uses a local data directory.

## Product entrypoints

### CLI

The CLI does these primary operations:

- It records the collections.
- It controls the contexts.
- It does search and retrieval.
- It does indexing with `sync`, `rebuild`, and `reindex`.
- It does maintenance and repair. `clean` removes the embeddings with no document and the embeddings from a different model. `clean` also sets the pre-1.0 data to its start condition.
- It shows the runtime information with `status`, `doctor`, and `model show`.
- It starts the web or MCP runtimes.

The CLI parses the commands in `crates/docbert/src/cli.rs`. It sends them from `crates/docbert/src/main.rs` to the command handlers.

### Web runtime

`docbert web` starts one local process. This process supplies two items:

- the browser UI
- the HTTP API (refer to [`web-api.md`](./web-api.md))

The `docbert-web` crate makes the web runtime. `main.rs` gives control to `docbert_web::run` (`crates/docbert-web/src/lib.rs`). This function opens `config.db`. It prepares the state through `state::init` (`crates/docbert-web/src/state.rs`). Then it starts the Axum server in `server::run` (`crates/docbert-web/src/server.rs`). The routes are in `crates/docbert-web/src/routes/*`.

### MCP runtime

`docbert mcp` starts a stdio-based MCP server for editors and agents.

`crates/docbert/src/mcp.rs` contains its tools, prompt, and resource template. Refer to [`mcp.md`](./mcp.md) for more information.

## Primary crates and their functions

## `crates/docbert`

This is the application crate.

This crate does these tasks:

- It parses the CLI and sends the top-level commands to the handlers.
- It does the indexing operations and controls the data changes.
- It prepares the MCP server and controls the tools and resources.
- It gives control of `docbert web` to the `docbert-web` crate.
- It controls the runtime resources for `config.db`, `embeddings.db`, `plaid.idx`, and Tantivy writers.

Primary modules:

- `src/main.rs`: the process entrypoint. It sends commands to the handlers, and finds the data directory and the model.
- `src/cli.rs`: the clap commands.
- `src/commands/*`: the CLI operations.
- `src/indexing.rs`: prepares the sync and rebuild operations, and completes the collection snapshots.
- `src/mcp.rs`: the MCP runtime.
- `src/runtime.rs`: the function to open `config.db` (blocking).
- `src/snapshots.rs`: the collection snapshot support for the indexing and web changes.

## `crates/docbert-web`

This crate makes the Axum HTTP API and the server for `docbert web`. The `docbert` binary uses it, and starts it through `docbert_web::run`.

This crate does these tasks:

- It makes the web state that all routes use (`src/state.rs`), and starts the server (`src/server.rs`).
- It holds the routes in `src/routes/*` (`mod`, `search`, `documents`, `collections`, `conversations`, `settings`).
- It gives an ingestion path for each web upload (`src/ingest.rs`).
- It makes new collection snapshots after web changes (`src/snapshots.rs`).
- It gives the path functions (`src/paths.rs`) and the blocking open and lock functions (`src/runtime.rs`).

This crate uses `docbert-core` for storage, search, and model control.

## `crates/docbert-webui`

This crate includes the browser SPA. Its `build.rs` makes the React app in `ui/`. Then `include_dir!` embeds the `ui/dist` output into the binary. At runtime it gives these assets as the fallback for each non-`/v1/` path.

It is an optional dependency of `docbert-web`. The default `webui` feature makes it active.

## `crates/docbert-core`

All entrypoints use this library crate.

This crate does these tasks:

- It keeps the config and metadata.
- It has the document ids and the metadata models.
- It gives the search and index types.
- It keeps the embeddings.
- It chunks and prepares the documents.
- It gives the ingestion functions.
- It has the conversation storage types.
- It does the search and reranking operations.
- It has the rules to find files in the filesystem.

`docbert-core` re-exports these public types:

- `ConfigDb`
- `DataDir`
- `EmbeddingDb`
- `SearchIndex`
- `ModelManager`
- `Conversation`

## `crates/docbert-plaid`

This crate makes the PLAID multi-vector index for semantic retrieval. It uses the pipeline from "PLAID: An Efficient Engine for Late Interaction Retrieval" (Santhanam et al., 2022). It uses the `fast-plaid` reference implementation as a model.

This crate does these tasks:

- It does k-means centroid training on the ColBERT token embeddings.
- It has a compressed codec for residual quantization.
- It does the MaxSim query operations on the compressed index.
- It has the on-disk `plaid.idx` file format.
- It has CUDA-accelerated paths for k-means and MaxSim matmul when the `cuda` feature is active.

`docbert-core::search::semantic` and the semantic part of `docbert-core::search::run` read this crate's index file from `DataDir::plaid_index()`. Then they use the index to rank the documents for an encoded query.

## `crates/docbert-pylate`

This crate is a Rust-only fork of [pylate-rs](https://github.com/lightonai/pylate-rs), copied into the workspace (from upstream 1.0.4). This fork does not include the upstream Python, WebAssembly, and npm packaging layers. The crate uses the workspace release version, not the upstream version.

This crate does these tasks:

- It reads ColBERT-family late-interaction models from HuggingFace or local paths.
- It encodes queries and documents.
- It calculates the token-level MaxSim similarity.
- It does `candle`-backed inference with the CPU, CUDA, Metal, MKL, and Accelerate backends. Feature flags select the backend.

`docbert-core::ModelManager` is a thin layer on this crate.

## `crates/rustbert`

`rustbert` is a different binary for Rust crate docs lookup. It gets the released crates and lets users search their public APIs. `rustbert` has its own CLI (`src/main.rs`) and MCP server (`src/mcp.rs`). It uses `docbert-core`, but it is not part of the `docbert` binary. It has its own nix packages (`rustbert`, `rustbert-cuda`, `rustbert-metal`).

Refer to [`rustbert.md`](./rustbert.md) for more information.

## Core state on disk

docbert keeps the local state in a data directory. `DataDir` controls this directory.

The primary pieces on disk are:

- `config.db`
  - collections
  - contexts
  - document metadata
  - chunk byte offsets
  - conversations
  - collection Merkle snapshots
  - settings, with the model and LLM values
- `embeddings.db`
  - ColBERT token embeddings
- `plaid.idx`
  - PLAID multi-vector index on the embeddings
- `tantivy/`
  - lexical search index
- source collection directories on disk
  - the primary document content for many read paths

For storage information, refer to [`storage.md`](./storage.md).

## Architectural layers

## 1. Collection and settings layer

`ConfigDb` makes most of this layer.

This layer does these tasks:

- It connects the collection names to the filesystem roots.
- It keeps the optional context strings.
- It keeps the document metadata and the user metadata.
- It keeps the chunk byte offsets. A search uses these offsets to show the byte range of a chunk.
- It keeps the conversation history.
- It keeps the collection Merkle snapshots for the sync and web changes.
- It keeps the general settings, for example the model selection and the LLM settings.

The CLI, web, and MCP runtimes all use this layer.

## 2. Discovery and ingestion layer

This layer uses the `walker`, `ingestion`, `preparation`, and `indexing` modules.

This layer does these tasks:

- It finds the applicable files from the collection roots.
- It obeys the walker rules. These rules include the Git ignore rules for repo-backed collections.
- It reads Markdown, text, and PDF files from disk.
- It changes PDFs into Markdown or text for preview, search, and embeddings.
- It makes the titles and metadata that search and API responses use.
- It prepares the documents for the chunk and embedding steps.
- It makes new collection snapshots after docbert completes a sync, rebuild, or web change.

## 3. Retrieval layer

This layer uses `SearchIndex`, `EmbeddingDb`, `search`, and `ModelManager`.

This layer does these tasks:

- It does lexical retrieval through Tantivy.
- It does semantic scoring through the ColBERT embeddings.
- It gives the hybrid and semantic search modes.
- It finds the documents that `get`-style reads refer to.
- It gets the excerpts and snippets from the document content on disk.

The three runtimes all use this layer.

## 4. Runtime layer

The `docbert` binary makes the retrieval and storage layers available through three runtimes:

- CLI handlers
- Axum HTTP routes for the web runtime
- RMCP tools, prompts, and resources for the MCP runtime

Each runtime has its own request and response shape, but they use the same data model and index state.

## Component diagram

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

ModelManager (a layer on `docbert-pylate`) is adjacent to `search`. docbert gives it to each query path to encode the query tokens when necessary.

## Runtime boundaries

## CLI runtime

`main.rs` finds these items:

- the data directory
- the model configuration
- the command path to use

The commands do not all prepare the same resources.

Examples:

- docbert does `doctor` and `completions` before it finds the data directory
- docbert sends `clean` before the `ConfigDb::open` call. `clean` is the repair path for data directories in the pre-1.0 redb format. `ConfigDb::open` rejects this format with `Error::LegacyDatabase`. The error message points to `docbert clean`.
- `web` and `mcp` find the model. Then they start their continuous runtimes.
- most other commands open `ConfigDb` and do short operations

## Web runtime boundary

The web runtime is the `docbert web` process. The `docbert-web` crate makes it.

Boundary information:

- one local process gives the SPA and the HTTP API
- the runtime starts through `docbert_web::run`. This function prepares the state through `state::init` in the `docbert-web` crate.
- the route handlers open `config.db` and `embeddings.db` when necessary
- document uploads and deletes change the source files and the indexed state
- search reads use the same in-process search index and model manager

For the route information, refer to [`web-api.md`](./web-api.md).

## MCP runtime boundary

The MCP runtime is a different continuous stdio server.

Boundary information:

- it keeps one `SearchIndex` and one `ModelManager` in the process state
- it opens `config.db` and `embeddings.db` again for the calls and the resource reads
- it supplies tools, one prompt, and one resource template
- for each operation, the retrieval tools can give plain text, structured JSON, resources, or a mixture

For the MCP information, refer to [`mcp.md`](./mcp.md).

## Chat-related architecture

docbert's chat system is a browser-side agent. This agent uses the docbert HTTP API and the LLM provider from the settings. The chat system has these parts:

- the conversation records in `config.db`
- the LLM settings in `config.db`
- the web API routes for conversations and settings
- the UI and runtime control in the browser client

Boundary information:

- the conversation storage and the LLM settings are backend functions
- the UI and runtime do most of the chat control
- the prompt instructions recommend that the agent does many searches and uses information from many files. But the backend does not make sure that these operations occur.

For the schema on disk and the route operations, refer to [`chat-and-conversations.md`](./chat-and-conversations.md).

## Search architecture

The retrieval stack has two primary modes:

- **hybrid**: lexical retrieval and ColBERT reranking
- **semantic**: the semantic scoring path through ColBERT-only search

The two modes use the same local metadata, source files, embeddings, and model runtime.

docbert reads some response data from disk at read time:

- docbert can make the search titles again from the document content on disk.
- The excerpts and snippets come from the file content on disk.
- Reads of one document also come directly from the source files on disk.

Thus the source collection directories are part of the active read paths and the indexing pipeline.

For pipeline information, refer to [`pipeline.md`](./pipeline.md).

## Concurrency and resource control

docbert uses local-process concurrency, not distributed coordination.

docbert uses these methods:

- Tantivy supplies the read and write index operations.
- docbert opens LMDB-backed stores for each operation, where applicable.
- The continuous runtimes keep the search and model state active. But they do not hold all storage handles open all the time.
- The operations with many changes open the writers for that work.
- docbert tries again only for the Tantivy index-writer lock. When another writer holds the lock, the web runtime tries again to get it (`crates/docbert-web/src/runtime.rs`). The heed and LMDB database opens do not try again, because LMDB can have concurrent handles.

For the storage and lock information, refer to [`storage.md`](./storage.md).

## Terminology guide

Use these terms:

- **collection**: a named root directory that docbert records in `config.db`
- **document**: one indexed source file in a collection
- **conversation**: a chat history record on disk
- **web runtime** or **web server**: the process that `docbert web` starts
- **MCP runtime** or **MCP server**: the process that `docbert mcp` starts
- **hybrid search**: lexical retrieval and semantic reranking
- **semantic search**: the semantic-only retrieval path

## Related information

- [`cli.md`](./cli.md)
- [`web-api.md`](./web-api.md)
- [`chat-and-conversations.md`](./chat-and-conversations.md)
- [`mcp.md`](./mcp.md)
- [`pipeline.md`](./pipeline.md)
- [`storage.md`](./storage.md)
- [`rustbert.md`](./rustbert.md)
- [`library-usage.md`](./library-usage.md)
