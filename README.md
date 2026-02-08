# docbert

A blazing-fast semantic search CLI for your documents. Combines BM25 full-text search with ColBERT neural reranking to find exactly what you're looking for.

## Features

- **Two-stage search pipeline**: Fast BM25 retrieval followed by ColBERT semantic reranking
- **Semantic-only search**: ColBERT-only full scan when you want pure semantic ranking
- **Collection-based organization**: Group documents into named collections
- **Incremental indexing**: Only re-index changed files
- **Multiple output formats**: Human-readable, JSON, or plain file paths
- **GPU acceleration**: CUDA and Metal support for faster embeddings
- **Fuzzy matching**: Tolerates typos in search queries
- **Zero configuration**: Just point it at a directory and search

## Quick Start

```bash
# Add a collection of markdown notes
docbert collection add ~/notes --name notes

# Search across all collections
docbert search "how to configure nginx"

# Search with semantic understanding (default)
docbert search "memory management in systems programming"

# Semantic-only full scan (ColBERT only)
docbert ssearch "memory management in systems programming"

# Fast BM25-only search (no neural reranking)
docbert search "nginx config" --bm25-only

# Output as JSON for scripting
docbert search "rust ownership" --json

# Get file paths for piping to other tools
docbert search "todo" --files | xargs -I {} code {}
```

## MCP Server

docbert exposes an MCP (Model Context Protocol) server for AI agent integrations.

**Tools exposed:**
- `docbert_search` - Keyword + semantic search (supports collection filters)
- `semantic_search` - Semantic-only search across all documents
- `docbert_get` - Retrieve a document by path or `#doc_id`
- `docbert_multi_get` - Retrieve multiple documents by glob pattern
- `docbert_status` - Index health and collection summary

**Claude Desktop configuration** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

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

**Claude Code** (`~/.claude/settings.json`):

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

### With Nix (recommended)

```bash
# CPU version
nix build github:cfcosta/docbert

# CUDA version (for NVIDIA GPUs)
nix build github:cfcosta/docbert#docbert-cuda
```

Shell completions for bash, zsh, and fish are automatically installed with the Nix package.

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

### Managing Collections

```bash
# Add a directory as a collection
docbert collection add /path/to/docs --name docs

# List all collections
docbert collection list

# Remove a collection
docbert collection remove docs
```

### Searching

```bash
# Basic search (returns top 10 results)
docbert search "your query here"

# More results
docbert search "query" -n 20

# Search specific collection
docbert search "query" -c notes

# All results above a score threshold
docbert search "query" --all --min-score 0.5

# Disable fuzzy matching for exact searches
docbert search "exact phrase" --no-fuzzy

# Semantic-only full scan (no BM25 or fuzzy matching, slower for large corpora)
docbert ssearch "meaning of life"
```

### Retrieving Documents

```bash
# Get document by collection:path
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

# Sync changes incrementally (fast, only processes changed files)
docbert sync

# Sync specific collection
docbert sync -c notes

# Full rebuild (deletes everything and re-indexes from scratch)
docbert rebuild

# Rebuild specific collection
docbert rebuild -c notes
```

## How It Works

docbert uses a two-stage retrieval pipeline:

1. **BM25 Retrieval** (via [Tantivy](https://github.com/quickwit-oss/tantivy)): Fast full-text search with fuzzy matching retrieves the top 1000 candidates

2. **ColBERT Reranking** (via [pylate-rs](https://github.com/lightonai/pylate-rs)): Neural semantic scoring reranks candidates using the [jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2) model

This approach gives you the speed of traditional search with the semantic understanding of neural models.

For cases where you want pure semantic ranking, `docbert ssearch` (and the MCP `semantic_search` tool) skip Tantivy entirely and score every stored embedding. This is slower but avoids any BM25 or fuzzy matching influence.

## Configuration

docbert stores its data in `~/.local/share/docbert/` (or `$XDG_DATA_HOME/docbert/`).

Override with `--data-dir`:

```bash
docbert --data-dir /custom/path search "query"
```

### Environment Variables

- `DOCBERT_MODEL`: Override the ColBERT model (default: `jinaai/jina-colbert-v2`)
- `DOCBERT_LOG`: Set log level (e.g., `debug`, `info`, `warn`)

### Model Selection

Persist a default model in `config.db`:

```bash
docbert model set /path/to/model
docbert model show
docbert model clear
```

Override per command:

```bash
docbert --model /path/to/model search "query"
```

### Using Jina ColBERT v2 (default)

`jinaai/jina-colbert-v2` requires custom model code, so you must export it to a
local PyLate-compatible directory first.

```bash
pip install pylate
python scripts/prepare_jina_colbert_v2.py --output ~/.local/share/docbert/models/jina-colbert-v2
docbert model set ~/.local/share/docbert/models/jina-colbert-v2
docbert rebuild
```

Note: `jinaai/jina-colbert-v2` is released under CC-BY-NC-4.0 (non-commercial).

If you prefer to use another model without a local export step, set
`DOCBERT_MODEL` (or `docbert model set`) to a different pylate-rs-compatible
model directory.

## Supported File Types

- Markdown (`.md`)
- Plain text (`.txt`)

## Performance Tips

- Use `--bm25-only` for fast searches when semantic understanding isn't needed
- The ColBERT model is lazy-loaded on first semantic search
- GPU acceleration significantly speeds up embedding computation
- Incremental indexing means only changed files are re-processed

## License

MIT OR Apache-2.0
