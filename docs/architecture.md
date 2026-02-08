# Architecture

## Overview

docbert is a hybrid document search system that combines lexical retrieval (BM25 + fuzzy matching via Tantivy) with semantic reranking (ColBERT late interaction via pylate-rs). It operates as a CLI tool that indexes directories of documents into named collections and provides fast, high-quality search.

## Components

### 1. Collection Manager

Responsible for managing the mapping between named collections and filesystem directories. A collection is a named reference to a directory tree on disk.

- Stores collection definitions (name -> directory path) in the redb database
- Re-indexing is triggered explicitly via `docbert sync` or `docbert rebuild`
- Supports user-defined context strings per collection (for display purposes and future LLM integration)

### 2. Document Ingester

Reads files from collection directories and prepares them for indexing.

- Walks the directory tree recursively
- Reads file contents (initially plain text and markdown)
- Assigns each document a stable internal ID (a short hash suitable for display, e.g. `#abc123`)
- Tracks file modification times to support incremental re-indexing

### 3. Tantivy Index

Provides the first-stage retrieval using Tantivy's inverted index.

- Schema fields: document ID (STRING, STORED), collection name (STRING, STORED, FAST), relative file path (STRING, STORED), title (TEXT, STORED), body (TEXT), file modification time (u64, STORED, FAST)
- BM25 scoring is Tantivy's default and requires no configuration
- Fuzzy matching is available via FuzzyTermQuery (Levenshtein distance 1)
- Returns the top-1000 candidates for reranking

### 4. Embedding Store (redb)

Stores pre-computed ColBERT token-level embeddings for all indexed documents.

- Key: document ID (u64); chunked documents use chunk-specific IDs for embeddings
- Value: serialized token embedding matrix (`[num_tokens, 128]` as a flat `&[u8]` array of little-endian f32 values, prefixed with a u32 token count)
- Uses `insert_reserve` for zero-copy writes during bulk indexing
- Single database file stored in the XDG data directory

### 5. ColBERT Encoder (pylate-rs)

Computes ColBERT embeddings using the pylate-rs crate, which wraps the Candle ML framework.

- Model: `jinaai/jina-colbert-v2` (or user-configurable)
- Downloads model weights from HuggingFace Hub on first use, caches locally (for jina-colbert-v2, use a local PyLate export)
- Encodes queries with `[Q]` prefix and `[MASK]` padding (query_length from `config_sentence_transformers.json`, default 32)
- Encodes documents with `[D]` prefix; documents are padded to the longest sequence in the batch and truncated to `document_length`
- Output: per-token embeddings of dimension 128, L2-normalized

### 6. MaxSim Reranker

Performs ColBERT late interaction scoring on the candidate set from Tantivy.

- Loads pre-computed document embeddings from redb for each candidate
- Computes the MaxSim score: for each query token, find the maximum cosine similarity with any document token, then sum across all query tokens
- SIMD acceleration via the MKL or Accelerate backends (compile-time feature flags on pylate-rs/candle)
- Returns results sorted by MaxSim score

### 7. CLI Interface (clap)

Provides the user-facing command-line interface.

- Subcommands: `collection`, `context`, `search`, `get`, `multi-get`
- Output formats: human-readable (default), JSON (`--json`)
- Collection filtering: `-c <name>` to scope search to a single collection
- Score thresholding: `--min-score` to filter low-quality results
- Pagination: `-n` for result count

## Data Flow

### Indexing Pipeline

```
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

### Search Pipeline

```
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

## Storage Layout

All persistent data is stored under the XDG data directory (`$XDG_DATA_HOME/docbert/` or `~/.local/share/docbert/`):

```
docbert/
  config.db          # Collection definitions, context strings, settings
  embeddings.db      # ColBERT token embeddings for all documents
  tantivy/             # Tantivy index directory (managed by Tantivy)
    meta.json
    <segment files>
```

## Concurrency Model

- Tantivy supports concurrent readers with a single writer (lock-file enforced)
- redb supports concurrent readers with a single writer (MVCC)
- Indexing is single-threaded for writes but can parallelize document encoding across CPU cores via rayon (built into pylate-rs)
- Search is read-only and can serve concurrent queries
