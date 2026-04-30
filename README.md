# docbert

docbert is a local document retrieval tool with three main entrypoints:

- a **CLI** for registering collections, indexing them, and searching them
- a **local web runtime** with a browser UI and HTTP API
- an **MCP server** for editor and agent integrations

It uses a hybrid retrieval stack:

- **Tantivy/BM25** for fast lexical retrieval
- **ColBERT** for semantic reranking or semantic-only search

The current implementation works against local files and local state. Registered collection directories remain the source of truth for document content.

## What it does

- named collections backed by filesystem directories
- incremental indexing with collection snapshots
- hybrid search with BM25 + ColBERT reranking
- semantic-only search with `docbert ssearch`
- Markdown, plain text, and PDF ingestion
- local web UI and JSON API via `docbert web`
- persisted conversations and LLM settings for chat in the web UI, including ChatGPT Codex OAuth
- MCP tools, prompt, and `bert://...` resources via `docbert mcp`
- CPU, CUDA, Metal, Accelerate, and MKL build options through feature flags

## Quick start

```bash
# Register a directory as a collection
# This records the collection but does not index it yet.
docbert collection add ~/notes --name notes

# Index new/changed/deleted files
docbert sync

# Hybrid search (default CLI search path)
docbert search "rust ownership"

# Semantic-only search
docbert ssearch "memory management"

# Keyword-only/BM25-only search
docbert search "nginx config" --bm25-only

# JSON output for scripts
docbert search "release notes" --json

# Start the local web UI + HTTP API
docbert web --host 127.0.0.1 --port 3030

# Start the MCP server over stdio
docbert mcp
```

## Installation

### With Nix

```bash
# CPU build
nix build github:cfcosta/docbert

# CUDA build
nix build github:cfcosta/docbert#docbert-cuda

# Metal build (macOS)
nix build github:cfcosta/docbert#docbert-metal
```

### From source

```bash
git clone https://github.com/cfcosta/docbert
cd docbert

# CPU build
cargo build --release

# CUDA build
cargo build --release --features cuda

# Metal build (macOS)
cargo build --release --features metal
```

## Basic workflow

### 1. Register collections

```bash
docbert collection add /path/to/docs --name docs
docbert collection add /path/to/notes --name notes
docbert collection list
```

A collection is a named root directory stored in `config.db`.

Adding a collection does **not** index it. Run `sync` or `rebuild` after registration.

### 2. Index content

```bash
# Normal incremental update
docbert sync

# Sync one collection only
docbert sync -c notes

# Full rebuild
docbert rebuild

# Rebuild one collection
docbert rebuild -c docs
```

Current indexing behavior:

- discovers supported files under each collection root
- supports `.md`, `.txt`, and `.pdf`
- respects Git ignore rules only when the collection root is itself a Git repo
- uses collection Merkle snapshots to detect new, changed, and deleted files during `sync`
- stores lexical index data, embeddings, metadata, and snapshot state locally

If the active model no longer matches the stored embeddings, `sync` will refuse to proceed and tell you to run `docbert rebuild`.

### 3. Search

```bash
# Hybrid search
docbert search "query"

# Restrict to one collection
docbert search "query" -c notes

# More results
docbert search "query" -n 20

# Return all results above a threshold
docbert search "query" --all --min-score 0.2

# Disable fuzzy matching
docbert search "exact phrase" --no-fuzzy

# Print only matching file paths
docbert search "todo" --files

# Semantic-only search
docbert ssearch "same concept different wording"
```

### 4. Read documents

```bash
# By collection:path
docbert get notes:todo.md

# By short doc id
docbert get "#a1b2c3"

# JSON output
docbert get docs:api.md --json

# Multiple documents by glob
docbert multi-get "**/*.md" -c notes --files
```

## Web UI and HTTP API

`docbert web` starts one local process that serves:

- the browser UI
- the `/v1` HTTP API

Typical setup:

```bash
docbert collection add ~/notes --name notes
docbert sync
docbert web --host 127.0.0.1 --port 3030
```

The web runtime uses the same collection roots and local storage as the CLI.

Current highlights:

- search API under `/v1/search`
- document upload/delete routes that mutate source files on disk and keep indexed state in sync
- persisted conversations and LLM settings for chat
- one local process serving both the SPA and the API

More detail:

- [Web API reference](./docs/web-api.md)
- [Chat, conversations, and LLM settings](./docs/chat-and-conversations.md)

## Chat

The chat experience in the web UI is built from:

- persisted conversations in `config.db`
- persisted LLM settings in `config.db`
- web API routes for conversations and settings
- browser/runtime orchestration on top of docbert search and retrieval tools

Current auth options for chat include:

- API-key-backed providers such as OpenAI and Anthropic
- ChatGPT Plus/Pro via the `openai-codex` provider and local OAuth sign-in in Settings

Important boundary:

- conversation persistence and settings storage are backend behavior
- exact chat prompting/orchestration is runtime/UI behavior

See:

- [Chat, conversations, and LLM settings](./docs/chat-and-conversations.md)

## MCP server

`docbert mcp` starts a stdio MCP server for editor and agent integrations.

The current MCP surface includes:

- search tools
- retrieval tools
- status tool
- one prompt
- one `bert://{+path}` resource template

Example Claude Desktop config:

```json
{
  "mcpServers": {
    "docbert": {
      "command": "docbert",
      "args": ["mcp"]
    }
  }
}
```

See:

- [MCP reference](./docs/mcp.md)

## Model selection

Model resolution currently follows this priority order:

1. `--model <id-or-path>`
2. `DOCBERT_MODEL`
3. persisted `model_name` in `config.db`
4. built-in default model

Useful commands:

```bash
docbert model show
docbert model set /path/to/model
docbert model clear
```

For one-off overrides:

```bash
docbert --model /path/to/model search "query"
```

Useful environment variables:

- `DOCBERT_DATA_DIR`
- `DOCBERT_MODEL`
- `DOCBERT_LOG`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

## Data and storage

By default, docbert stores local state in the XDG data directory, typically:

```text
~/.local/share/docbert/
```

That state includes:

- `config.db`
- `embeddings.db`
- `plaid.idx`
- `tantivy/`

The collection roots themselves can live anywhere on disk.

See:

- [Storage reference](./docs/storage.md)

## How search works

Hybrid search runs BM25 and ColBERT in parallel and fuses the two rankings with Reciprocal Rank Fusion:

1. Tantivy produces up to 100 BM25 candidates (fuzzy matching on by default).
2. In parallel, the prebuilt PLAID semantic index produces up to 100 ColBERT MaxSim candidates for the same query.
3. The two ranked lists are combined with RRF (`k = 60`); each document contributes `1 / (k + rank_i)` from each list it appears in.
4. The top `--count` fused results are returned (or all results, with `--all`).

`--min-score` is ignored under RRF because fused scores are not on the BM25 scale; it only applies in `--bm25-only` mode, which skips the semantic leg entirely.

Semantic-only search (`docbert ssearch`, `POST /v1/search` with `mode=semantic`) skips the BM25 leg and ranks all stored documents directly against the PLAID index. Both modes require a prebuilt PLAID index — run `docbert sync` if the server returns `PlaidIndexMissing`.

See:

- [Pipeline reference](./docs/pipeline.md)
- [Architecture overview](./docs/architecture.md)

## Reference docs

- [CLI reference](./docs/cli.md)
- [Architecture overview](./docs/architecture.md)
- [Pipeline reference](./docs/pipeline.md)
- [Storage reference](./docs/storage.md)
- [Library usage (`docbert-core`)](./docs/library-usage.md)
- [Dependency reference](./docs/dependencies.md)
- [Web API reference](./docs/web-api.md)
- [Chat, conversations, and LLM settings](./docs/chat-and-conversations.md)
- [MCP reference](./docs/mcp.md)

## License

MIT OR Apache-2.0
