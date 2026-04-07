# docbert

docbert is a CLI for searching local documents. It uses BM25 to find likely matches quickly, then reranks them with ColBERT.

Point it at one or more folders, sync the index, and search across Markdown or plain text files.

It can also serve a local web UI with `docbert web`, using those same CLI-managed collection folders as the source of truth.

## What it does

- two-stage search with BM25 and ColBERT
- semantic-only full scans with `docbert ssearch`
- named collections for grouping directories
- incremental indexing that only reprocesses changed files
- human-readable, JSON, or file-only output
- CUDA and Metal support for faster embedding work
- fuzzy matching for typo-tolerant queries
- Markdown and plain text support

## Quick start

```bash
# Add a collection of markdown notes
docbert collection add ~/notes --name notes

# Build or update the index
docbert sync

# Search across all collections
docbert search "how to configure nginx"

# Search with semantic reranking (default)
docbert search "memory management in systems programming"

# Run a semantic-only full scan (ColBERT only)
docbert ssearch "memory management in systems programming"

# Skip neural reranking and use BM25 only
docbert search "nginx config" --bm25-only

# Output JSON for scripts
docbert search "rust ownership" --json

# Print matching file paths
docbert search "todo" --files | xargs -I {} code {}

# Start the web UI on localhost:3030
# Collections must already be added with `docbert collection add`
docbert web --host 127.0.0.1 --port 3030
```

## Web UI

The web UI uses the same collection folders registered in the CLI.

Typical setup:

```bash
# Register a source folder first
docbert collection add ~/notes --name notes

# Index existing files
docbert sync

# Start the web UI
docbert web --host 127.0.0.1 --port 3030
```

Behavior:

- `GET /v1/collections` lists CLI-managed collections
- uploads write into collection folders on disk, then index/embed the new file
- document deletion removes the source file from the collection folder, then removes indexed state
- the long-running web process keeps the search index and model manager alive, but reopens `config.db`, `embeddings.db`, and a Tantivy writer only for the operation that needs them
- retryable database or writer lock contention is handled by waiting and retrying, so `docbert sync` can complete while `docbert web` stays up; non-lock errors still fail normally

## MCP server

docbert can also run as an MCP (Model Context Protocol) server for editors and AI tools.

The MCP process keeps the search index and model manager alive, but reopens `config.db` and `embeddings.db` for each tool call or resource read instead of holding redb handles for the full server lifetime. Retryable lock contention waits and retries, which lets `docbert sync` run against the same data directory while MCP is still connected.

Available tools:

- `docbert_search`: keyword + semantic search, with optional collection filters
- `semantic_search`: semantic-only search across all documents
- `docbert_get`: fetch a document by path or `#doc_id`
- `docbert_multi_get`: fetch multiple documents with a glob pattern
- `docbert_status`: show index health and collection summaries

### Claude Desktop config

File: `~/Library/Application Support/Claude/claude_desktop_config.json`

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

### Claude Code config

File: `~/.claude/settings.json`

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

## Installation

### With Nix

```bash
# CPU version
nix build github:cfcosta/docbert

# CUDA version (for NVIDIA GPUs)
nix build github:cfcosta/docbert#docbert-cuda

# Metal version (for Apple GPUs on macOS)
nix build github:cfcosta/docbert#docbert-metal
```

Shell completions for bash, zsh, and fish are installed with the Nix package.

### From source

```bash
git clone https://github.com/cfcosta/docbert
cd docbert

# CPU build
cargo build --release

# With CUDA support
cargo build --release --features cuda

# With Metal support (macOS)
cargo build --release --features metal
```

## Usage

### Manage collections

```bash
# Add a directory as a collection
docbert collection add /path/to/docs --name docs

# List all collections
docbert collection list

# Remove a collection
docbert collection remove docs
```

### Search

```bash
# Basic search (top 10 results)
docbert search "your query here"

# More results
docbert search "query" -n 20

# Search one collection
docbert search "query" -c notes

# Return all results above a score threshold
docbert search "query" --all --min-score 0.5

# Disable fuzzy matching
docbert search "exact phrase" --no-fuzzy

# Semantic-only full scan (slower on large corpora)
docbert ssearch "meaning of life"
```

### Retrieve documents

```bash
# Get a document by collection:path
docbert get notes:todo.md

# Get by document ID
docbert get "#a1b2c3"

# Output with metadata
docbert get notes:readme.md --json

# Get multiple documents with glob patterns
docbert multi-get "*.md" -c notes
```

### Maintenance

```bash
# Show system status
docbert status

# Sync changes incrementally
docbert sync

# Sync one collection
docbert sync -c notes

# Full rebuild
docbert rebuild

# Rebuild one collection
docbert rebuild -c notes
```

## How search works

Search happens in two steps:

1. Tantivy runs BM25 retrieval, optionally with fuzzy matching, and returns a candidate set.
2. pylate-rs reranks those candidates with ColBERT using `lightonai/ColBERT-Zero` by default.

That gives you fast keyword search without losing semantic ranking.

If you want pure semantic ranking, `docbert ssearch` skips BM25 and scores every stored embedding. That is slower on large collections, but it avoids BM25 and fuzzy-matching bias.

## Configuration

docbert stores its data in `~/.local/share/docbert/` or `$XDG_DATA_HOME/docbert/`.

Use `--data-dir` to override that:

```bash
docbert --data-dir /custom/path search "query"
```

Data directory resolution order:

1. `--data-dir` CLI flag
2. `DOCBERT_DATA_DIR` environment variable
3. XDG default: `$XDG_DATA_HOME/docbert/` or `~/.local/share/docbert/`

### Environment variables

- `DOCBERT_DATA_DIR`: override the data directory
- `DOCBERT_MODEL`: override the ColBERT model
- `DOCBERT_LOG`: set the log level, for example `debug`, `info`, or `warn`

### Model selection

Set a default model in `config.db`:

```bash
docbert model set /path/to/model
docbert model show
docbert model clear
```

Override it for a single command:

```bash
docbert --model /path/to/model search "query"
```

### Alternative models

The default model, `lightonai/ColBERT-Zero`, works out of the box. If you want to use another pylate-rs-compatible model:

```bash
docbert model set /path/to/model
# or
DOCBERT_MODEL=/path/to/model docbert search "query"
```

## Supported file types

- Markdown (`.md`)
- Plain text (`.txt`)

## Performance notes

- Use `--bm25-only` when keyword search is enough.
- The ColBERT model loads on the first semantic search.
- GPU support speeds up embedding generation.
- Incremental indexing only reprocesses changed files.

## License

MIT OR Apache-2.0
