# docbert

docbert is a local document retrieval tool with three primary entrypoints:

- A **CLI** that records, indexes, and searches collections
- A **local web runtime** with a browser UI and an HTTP API
- An **MCP server** for editor and agent integrations

docbert uses a hybrid retrieval stack:

- **Tantivy/BM25** for fast lexical retrieval
- **ColBERT** for semantic reranking or semantic-only search

docbert uses local files and local data. The collection directories that you record are the primary source of the documents.

## What it does

docbert has these functions:

- Collections with names in filesystem directories
- Incremental indexing with collection snapshots (`docbert sync`), full index rebuilding (`docbert rebuild`), and PLAID-only reindexing of the embeddings (`docbert reindex`)
- Hybrid search with BM25 and ColBERT reranking
- Semantic-only search with `docbert ssearch`
- Markdown, plain text, and PDF files as document sources
- Context strings for each collection (`docbert context add/list/remove`) that the retrieval tools use
- Runtime diagnostics through `docbert doctor` (the available accelerators) and `docbert status`
- Local web UI and JSON API through `docbert web`
- Kept conversations and LLM settings for web UI chat, with ChatGPT Codex OAuth
- MCP tools, prompt, and `bert://...` resources through `docbert mcp`
- CPU, CUDA, Metal, Accelerate, and MKL backends through feature flags

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

## Basic steps

### 1. Add collections

```bash
docbert collection add /path/to/docs --name docs
docbert collection add /path/to/notes --name notes
docbert collection list
```

A collection is a root directory with a name. docbert keeps it in `config.db`.

When you add a collection, docbert does **not** index it. Use `docbert sync` or `docbert rebuild` after you add the collection.

### 2. Index documents

```bash
# Normal incremental update
docbert sync

# Sync one collection only
docbert sync -c notes

# Full rebuild
docbert rebuild

# Rebuild one collection
docbert rebuild -c docs

# Rebuild only the PLAID semantic index from existing embeddings,
# without re-encoding any documents
docbert reindex
```

docbert does these tasks when it indexes files:

- docbert finds the applicable files below each collection root.
- docbert reads `.md`, `.txt`, and `.pdf` files.
- docbert obeys the Git ignore rules only when the collection root is a Git repo.
- docbert uses the collection Merkle snapshots to find the new, changed, and removed files during `sync`.
- docbert keeps the lexical index data, the embeddings, the metadata, and the snapshot data on the local disk.

docbert keeps the embeddings for one model. If the active model is different, `sync` stops. Then `sync` tells you to use `docbert rebuild`.

### 3. Search

```bash
# Hybrid search
docbert search "query"

# Restrict to one collection
docbert search "query" -c notes

# More results
docbert search "query" -n 20

# Return all results instead of the top 10
docbert search "query" --all

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

The `docbert web` command starts one local process. This process supplies:

- The browser UI
- The `/v1` HTTP API

Typical steps:

```bash
docbert collection add ~/notes --name notes
docbert sync
docbert web --host 127.0.0.1 --port 3030
```

The web runtime uses the same collection roots and local storage as the CLI.

The web runtime includes these functions:

- A search API at `/v1/search`
- Document upload and delete endpoints that change the source files on disk and keep the indexed data in sync
- Kept conversations and LLM settings for chat
- One local process that supplies the SPA and the API

Refer to:

- [Web API reference](./docs/web-api.md)
- [Chat, conversations, and LLM settings](./docs/chat-and-conversations.md)

## Chat

The chat in the web UI uses these parts:

- The conversations in `config.db`
- The LLM settings in `config.db`
- The web API endpoints for conversations and settings
- The orchestration of the docbert search and retrieval tools, in the browser and the runtime

The chat authentication options include:

- The providers that use an API key, for example OpenAI and Anthropic
- ChatGPT Plus/Pro through the `openai-codex` provider, with local OAuth sign-in in Settings

docbert keeps the conversations and the settings in the backend. The runtime and the UI control the chat prompts and the tool orchestration.

Refer to:

- [Chat, conversations, and LLM settings](./docs/chat-and-conversations.md)

## MCP server

The `docbert mcp` command starts a stdio MCP server for editor and agent integrations.

The MCP server has these parts:

- Search tools
- Retrieval tools
- A status tool
- One prompt
- One `bert://{+path}` resource template

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

Refer to:

- [MCP reference](./docs/mcp.md)

## Model selection

docbert selects the model in this sequence:

1. The `--model <id-or-path>` option
2. The `DOCBERT_MODEL` variable
3. The `model_name` value in `config.db`
4. The built-in default model

You can use these commands:

```bash
docbert model show
docbert model set /path/to/model
docbert model clear
```

To set a different model one time only:

```bash
docbert --model /path/to/model search "query"
```

docbert uses these environment variables:

- `DOCBERT_DATA_DIR`
- `DOCBERT_MODEL`
- `DOCBERT_LOG`
- `DOCBERT_EMBEDDING_BATCH_SIZE`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

`DOCBERT_LOG` is the tracing filter for stderr logging. When you set this variable, it replaces the `-v` verbosity mapping. `DOCBERT_EMBEDDING_BATCH_SIZE` sets a different embedding batch size for indexing.

## Data and storage

By default, docbert keeps its local data in the XDG data directory. The usual path is:

```text
~/.local/share/docbert/
```

This data includes these files:

- `config.db` (an LMDB env, with the `config.db-lock` file adjacent to it)
- `embeddings.db` (an LMDB env, with the `embeddings.db-lock` file adjacent to it)
- `plaid.idx`
- `tantivy/`

The `config.db` and `embeddings.db` files use LMDB through [`heed`](https://docs.rs/heed). Thus, the `docbert mcp`, `docbert web`, and CLI processes can use one data dir at the same time.

docbert rejects data from releases before 1.0 and gives an error. This data includes redb-format files and `f32`-layout embeddings. You can use `docbert clean` to remove this data. Then you can use `docbert sync` to index it again.

The collection roots can be at different locations on the disk.

Refer to:

- [Storage reference](./docs/storage.md)

## How search operates

Hybrid search does a BM25 search and a ColBERT/PLAID search for the same query. Reciprocal Rank Fusion (RRF) then makes one ranking from the two result lists:

1. Tantivy gives a maximum of 100 BM25 candidates (docbert uses fuzzy matching by default).
2. The PLAID semantic index gives a maximum of 100 ColBERT MaxSim candidates for the same query.
3. RRF makes one ranking from the two ranked lists (`k = 60`). Each document gets a score of `1 / (k + rank_i)` from each list that contains it.
4. docbert shows the top `--count` results, or all results with `--all`.

RRF ignores `--min-score` because the RRF scores are not on the BM25 scale. docbert uses `--min-score` in `--bm25-only` mode and in semantic-only search (`docbert ssearch`, `POST /v1/search` with `mode=semantic`).

Semantic-only search does not do a BM25 search. This search ranks the documents directly with the PLAID index.

A PLAID index is necessary for the two modes. Without this index, search gives `Error::PlaidIndexMissing`, or HTTP 503 from the web API. You can make this index with `docbert sync`, `docbert rebuild`, or `docbert reindex`.

Refer to:

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
