# Architecture

## Overview

docbert is a hybrid document search tool. It keeps a Tantivy index for lexical retrieval and a redb store for ColBERT embeddings, then ties them together behind a CLI and a local web UI.

The main unit of organization is a collection: a named directory tree on disk. You register collections first, then run `docbert sync` or `docbert rebuild` when you want the index updated.

The web server is started with `docbert web --host 127.0.0.1 --port 3030` and serves both the SPA and a small JSON API. `GET /v1/collections` reflects the CLI-managed collections already stored in `config.db`.

## Main components

### 1. Collection manager

The collection manager keeps track of collection names and filesystem paths.

- Stores collection definitions (`name -> directory path`) in the redb config database
- Re-indexing only happens when the user runs `docbert sync` or `docbert rebuild`
- Can store user-defined context strings per collection for display and future LLM integration

### 2. Document ingester

The ingester walks each collection and turns files into indexable documents.

- Recursively scans the directory tree
- Reads file contents from plain text and Markdown files
- Assigns each document a stable internal ID, shown as a short hash like `#abc123`
- Tracks file modification times for incremental re-indexing

### 3. Tantivy index

Tantivy handles the first pass of search.

- Schema fields: document ID (STRING, STORED), collection name (STRING, STORED, FAST), relative file path (STRING, STORED), title (TEXT, STORED), body (TEXT), file modification time (`u64`, STORED, FAST)
- BM25 scoring uses Tantivy's defaults
- Fuzzy matching is available through `FuzzyTermQuery` with Levenshtein distance 1
- Returns the top 1000 candidates for reranking

### 4. Embedding store (redb)

The embedding store keeps precomputed ColBERT token embeddings for indexed documents.

- Key: document ID (`u64`); chunked documents use chunk-specific IDs for embeddings
- Value: serialized token embedding matrix (`[num_tokens, 128]` as a flat `&[u8]` array of little-endian `f32` values, prefixed with a `u32` token count)
- Uses `insert_reserve` for zero-copy writes during bulk indexing
- Lives in a single database file under the XDG data directory

### 5. ColBERT encoder (pylate-rs)

pylate-rs computes ColBERT embeddings using Candle under the hood.

- Model: `lightonai/ColBERT-Zero` by default, or a user-specified alternative
- Downloads model weights from HuggingFace Hub on first use and caches them locally
- Encodes queries with a `[Q]` prefix and `[MASK]` padding (`query_length` comes from `config_sentence_transformers.json`, default 32)
- Encodes documents with a `[D]` prefix; documents are padded to the longest sequence in the batch and truncated to `document_length` (default 519)
- Produces L2-normalized per-token embeddings with dimension 128

### 6. MaxSim reranker

The reranker scores each candidate returned by Tantivy.

- Loads precomputed document embeddings from redb
- Computes the MaxSim score by taking the best token match for each query token, then summing those values
- Can use MKL or Accelerate backends through pylate-rs and Candle feature flags
- Returns results sorted by MaxSim score

### 7. CLI interface (clap)

The CLI is the public face of the system.

- Subcommands: `collection`, `context`, `search`, `get`, `multi-get`, `web`, `mcp`
- Output formats: human-readable by default, or JSON with `--json`
- Collection filtering: `-c <name>` to search a single collection
- Score thresholding: `--min-score` to drop low-scoring results
- Pagination: `-n` to control result count

### 8. Web interface (`docbert web`)

The web interface serves a SPA and a local JSON API from the same process.

- Reads collection definitions from `config.db`
- Lists collections through `GET /v1/collections`
- Reads document titles/content from collection folders on disk
- Uploads write source files into collection folders, then index and embed them
- Deletes remove the source file from the collection folder, then remove indexed state

## Data flow

### Indexing pipeline

```text
Directory -> Document Ingester -> [document text, metadata]
                                       |
                    +------------------+------------------+
                    |                                     |
              Tantivy Index                      ColBERT Encoder
           (BM25 inverted index)            (per-token embeddings)
                    |                                     |
              Tantivy on disk                     redb on disk
           (XDG data dir)                      (XDG data dir)
```

### Search pipeline

```text
Query string
     |
     +---> Tantivy BM25 + Fuzzy -> top-1000 candidate doc IDs
     |
     +---> ColBERT encode query -> query token embeddings [query_length, 128]
                |
                v
     Load candidate doc embeddings from redb
                |
                v
     MaxSim reranking (query tokens x doc tokens)
                |
                v
     Sorted results by MaxSim score
                |
                v
     Format output (human / JSON)
```

## Storage layout

All persistent data lives under the XDG data directory (`$XDG_DATA_HOME/docbert/` or `~/.local/share/docbert/`):

```text
docbert/
  config.db          # Collection definitions, context strings, settings
  embeddings.db      # ColBERT token embeddings for all documents
  tantivy/           # Tantivy index directory (managed by Tantivy)
    meta.json
    <segment files>
```

## Concurrency model

- Tantivy allows concurrent readers and a single writer, enforced by its lock file
- redb allows concurrent readers and a single writer through MVCC
- Indexing writes run in a single thread, but document encoding can fan out across CPU cores through rayon inside pylate-rs
- Search is read-only and can serve concurrent queries
