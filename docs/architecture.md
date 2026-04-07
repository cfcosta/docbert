# Architecture

## Overview

docbert is a local document retrieval system with three user-facing entrypoints:

- the **CLI** (`docbert ...`)
- the **web runtime** (`docbert web`)
- the **MCP runtime** (`docbert mcp`)

All three sit on top of the same core storage and retrieval stack in `docbert-core`.

At a high level, docbert:

1. registers one or more named filesystem collections
2. indexes markdown/text documents from those collections
3. stores lexical index data, metadata, embeddings, and settings locally
4. serves retrieval through the CLI, the web API/UI, or MCP tools/resources

The current implementation is centered on **local state and local files**, not a remote service architecture.

## Product surfaces

### CLI

The CLI is the main operational interface for:

- collection registration
- context management
- search and retrieval
- indexing (`sync`, `rebuild`)
- runtime inspection (`status`, `doctor`, `model show`)
- starting the web or MCP runtimes

The CLI parses commands in `crates/docbert/src/cli.rs` and dispatches them from `crates/docbert/src/main.rs` into command handlers.

### Web runtime

`docbert web` starts a single local process that serves:

- the browser UI
- the HTTP API documented in [`web-api.md`](./web-api.md)

The web runtime is initialized in `crates/docbert/src/web/mod.rs` and its route surface lives in `crates/docbert/src/web/routes/*`.

### MCP runtime

`docbert mcp` starts a stdio-based MCP server for editor and agent integrations.

Its tools, prompt, and resource template are implemented in `crates/docbert/src/mcp.rs` and documented in [`mcp.md`](./mcp.md).

## Main crates and their roles

## `crates/docbert`

This is the application crate.

It owns:

- CLI parsing and top-level command dispatch
- indexing workflows and mutation orchestration
- web runtime setup and route wiring
- MCP server setup and tool/resource handling
- runtime resource management around `config.db`, `embeddings.db`, and Tantivy writers

Important modules:

- `src/main.rs` — process entrypoint, command dispatch, data-dir resolution, model resolution
- `src/cli.rs` — clap command surface
- `src/command_handlers/*` — CLI behaviors
- `src/indexing_workflow.rs` — sync/rebuild planning and snapshot finalization
- `src/web/*` — web runtime
- `src/mcp.rs` — MCP runtime
- `src/collection_snapshots.rs` — collection snapshot support around indexing/web mutations

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

Important public types re-exported by `docbert-core` include:

- `ConfigDb`
- `DataDir`
- `EmbeddingDb`
- `SearchIndex`
- `ModelManager`
- `Conversation`

## Core persistent state

docbert keeps local state under a resolved data directory managed by `DataDir`.

The major persistent pieces are:

- `config.db`
  - collections
  - contexts
  - document metadata
  - conversations
  - collection Merkle snapshots
  - settings, including model and LLM-related values
- `embeddings.db`
  - ColBERT token embeddings
- `tantivy/`
  - lexical search index
- source collection directories on disk
  - still treated as the authoritative document content for many read paths

For storage details, see [`storage.md`](./storage.md).

## Architectural layers

## 1. Collection and settings layer

Implemented primarily through `ConfigDb`.

Responsibilities:

- map collection names to filesystem roots
- store optional context strings
- store document metadata and user metadata
- store persisted conversation history
- store collection Merkle snapshots for sync/web mutation tracking
- store general settings such as model selection and persisted LLM settings

This layer is shared across CLI, web, and MCP surfaces.

## 2. Discovery and ingestion layer

Implemented across `walker`, `ingestion`, `preparation`, and `indexing_workflow`.

Responsibilities:

- discover eligible files from collection roots
- respect current walker rules, including Git ignore behavior for repo-backed collections
- load markdown/text files from disk
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

The application crate exposes the retrieval and storage layers through three runtimes:

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
        |                   |
        +---------+---------+
                  |
                  v
              search.rs
                  |
     +------------+-------------+
     |            |             |
     v            v             v
   CLI        Web runtime    MCP runtime
```

## Runtime boundaries

## CLI runtime

`main.rs` resolves:

- data directory
- model configuration
- which command path to run

Not every command initializes the same resources.

Examples:

- `doctor` and `completions` are handled early
- `web` and `mcp` resolve the model, then hand off into their long-lived runtimes
- most other commands open `ConfigDb` and run as short-lived operations

## Web runtime boundary

The web runtime is not a separate historical “docserver”; it is the current `docbert web` process.

Important boundary details from the current implementation:

- one local process serves the SPA and the HTTP API
- the runtime is initialized through `web::state::init(...)`
- route handlers open `config.db` / `embeddings.db` as needed
- document uploads and deletes mutate both source files and indexed state
- search reads use the shared in-process search index and model manager

For the concrete route contract, see [`web-api.md`](./web-api.md).

## MCP runtime boundary

The MCP runtime is a separate long-lived stdio server.

Important boundary details:

- it keeps a shared `SearchIndex` and `ModelManager` in process state
- it reopens `config.db` and `embeddings.db` for calls and resource reads
- it exposes tools, one prompt, and one resource template
- retrieval tools can return plain text, structured JSON, resources, or a mix depending on the operation

For the concrete MCP contract, see [`mcp.md`](./mcp.md).

## Chat-related architecture

docbert's chat system is not a separate backend service. It is built from:

- persisted conversation records in `config.db`
- LLM settings in `config.db`
- web API routes for conversations and settings
- UI/runtime orchestration in the browser client

Important boundary clarification:

- conversation persistence and LLM settings are backend concerns
- chat orchestration strategy is largely a UI/runtime concern
- prompt instructions encourage multi-search and multi-file synthesis, but those are not independent backend guarantees

For the concrete persisted schema and route behavior, see [`chat-and-conversations.md`](./chat-and-conversations.md).

## Search architecture

The current retrieval stack supports two main modes:

- **hybrid**
  - lexical retrieval plus ColBERT reranking
- **semantic**
  - semantic scoring path through ColBERT-only search

Both modes ultimately depend on the same local metadata, source files, embeddings, and model runtime.

One subtle but important detail is that some response enrichment is pulled from disk at read time:

- search titles may be recomputed from the current document content on disk
- excerpts/snippets come from current file content on disk
- single-document reads also come directly from source files on disk

That means the source collection directories remain part of the live read architecture, not just the indexing pipeline.

For pipeline details, see [`pipeline.md`](./pipeline.md).

## Concurrency and resource management

docbert is designed around local-process concurrency rather than distributed coordination.

Important patterns in the current implementation:

- Tantivy supplies read/write index primitives
- redb-backed stores are opened per operation where appropriate
- long-lived runtimes keep search/model state alive, but do not hold every storage handle permanently open
- mutation-heavy operations open writers around the work that needs them
- lock/contention retry behavior is part of runtime resource handling for long-lived web/MCP processes

The exact persistence and lock-sensitive storage details are documented in [`storage.md`](./storage.md).

## Terminology guide

Use these terms consistently in the current codebase:

- **collection** — a named root directory registered in `config.db`
- **document** — one indexed source file within a collection
- **conversation** — persisted chat history record
- **web runtime** / **web server** — the process started by `docbert web`
- **MCP runtime** / **MCP server** — the process started by `docbert mcp`
- **hybrid search** — lexical retrieval plus semantic reranking
- **semantic search** — semantic-only retrieval path

Avoid stale terminology like **docserver** when describing the current implementation.

## Related references

- [`cli.md`](./cli.md)
- [`web-api.md`](./web-api.md)
- [`chat-and-conversations.md`](./chat-and-conversations.md)
- [`mcp.md`](./mcp.md)
- [`pipeline.md`](./pipeline.md)
- [`storage.md`](./storage.md)
- [`library-usage.md`](./library-usage.md)
