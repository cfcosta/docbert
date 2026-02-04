# docbert

A blazing-fast semantic search CLI for your documents. Combines BM25 full-text search with ColBERT neural reranking to find exactly what you're looking for.

## Features

- **Two-stage search pipeline**: Fast BM25 retrieval followed by ColBERT semantic reranking
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

# Fast BM25-only search (no neural reranking)
docbert search "nginx config" --bm25-only

# Output as JSON for scripting
docbert search "rust ownership" --json

# Get file paths for piping to other tools
docbert search "todo" --files | xargs -I {} code {}
```

## Installation

### With Nix (recommended)

```bash
# CPU version
nix build github:cfcosta/docbert

# CUDA version (for NVIDIA GPUs)
nix build github:cfcosta/docbert#docbert-cuda
```

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

# Rebuild all indexes
docbert rebuild

# Rebuild specific collection
docbert rebuild -c notes

# Rebuild only embeddings (faster)
docbert rebuild --embeddings-only
```

## How It Works

docbert uses a two-stage retrieval pipeline:

1. **BM25 Retrieval** (via [Tantivy](https://github.com/quickwit-oss/tantivy)): Fast full-text search with fuzzy matching retrieves the top 1000 candidates

2. **ColBERT Reranking** (via [pylate-rs](https://github.com/lightonai/pylate-rs)): Neural semantic scoring reranks candidates using the [GTE-ModernColBERT](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) model

This approach gives you the speed of traditional search with the semantic understanding of neural models.

## Configuration

docbert stores its data in `~/.local/share/docbert/` (or `$XDG_DATA_HOME/docbert/`).

Override with `--data-dir`:

```bash
docbert --data-dir /custom/path search "query"
```

### Environment Variables

- `DOCBERT_MODEL`: Override the ColBERT model (default: `lightonai/GTE-ModernColBERT-v1`)
- `DOCBERT_LOG`: Set log level (e.g., `debug`, `info`, `warn`)

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
