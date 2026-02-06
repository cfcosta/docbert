# Storage Design

## Overview

docbert uses three storage systems, each chosen for its strengths:

| System                | Purpose                                             | Format                 |
| --------------------- | --------------------------------------------------- | ---------------------- |
| **redb** (config)     | Collection definitions, document metadata, settings | Typed key-value store  |
| **redb** (embeddings) | ColBERT per-token embedding matrices                | Binary key-value store |
| **Tantivy**           | Full-text search index (BM25 + fuzzy)               | Inverted index on disk |

All data lives under the XDG data directory: `$XDG_DATA_HOME/docbert/` (typically `~/.local/share/docbert/`).

## XDG Directory Layout

```
~/.local/share/docbert/
  config.db              # Configuration and metadata database
  embeddings.db          # ColBERT embedding vectors
  tantivy/                 # Tantivy index directory
    meta.json              # Tantivy metadata
    <uuid>.<ext>           # Tantivy segment files
```

The `xdg` crate (already a dependency) handles platform-appropriate directory resolution.

## config.db Schema

### Table: collections

Stores the mapping from collection names to filesystem paths.

- **Key**: `&str` (collection name, e.g. "notes")
- **Value**: `&str` (absolute directory path, e.g. "/home/user/notes")

### Table: contexts

Stores user-defined context strings for collections.

- **Key**: `&str` (collection URI, e.g. "bert://notes")
- **Value**: `&str` (context description, e.g. "Personal notes and ideas")

### Table: document_metadata

Stores per-document metadata for incremental indexing.

- **Key**: `u64` (document numeric ID)
- **Value**: `&[u8]` (serialized struct containing: collection name, relative path, mtime as u64, file size as u64, number of tokens as u32)

The serialization format for document metadata is a simple fixed-layout binary format:

- 4 bytes: collection name length (u32 LE)
- N bytes: collection name (UTF-8)
- 4 bytes: relative path length (u32 LE)
- N bytes: relative path (UTF-8)
- 8 bytes: mtime (u64 LE, seconds since epoch)
- 8 bytes: file size (u64 LE)
- 4 bytes: token count (u32 LE)

### Table: settings

Stores global configuration values.

- **Key**: `&str` (setting name)
- **Value**: `&str` (setting value)

Settings include:

- `model_name`: the HuggingFace model ID or local PyLate model path (default: `lightonai/GTE-ModernColBERT-v1`)
- `pool_factor`: hierarchical pooling factor (default: `1`)
- `index_version`: schema version for migration support

## embeddings.db Schema

### Table: embeddings

Stores pre-computed ColBERT token embeddings.

- **Key**: `u64` (document numeric ID, same as in document_metadata)
- **Value**: `&[u8]` (binary embedding data)

The binary format for embeddings:

- 4 bytes: number of tokens `T` (u32 LE)
- 4 bytes: embedding dimension `D` (u32 LE, always 128 currently)
- `T * D * 4` bytes: the embedding matrix as contiguous little-endian f32 values, row-major (token-major) order

Example: a document with 200 tokens at 128 dimensions:

- Header: 8 bytes
- Data: 200 _ 128 _ 4 = 102,400 bytes
- Total: 102,408 bytes (~100 KB)

### Why a separate redb file?

The embeddings database is kept separate from config.db because:

1. **Size**: Embeddings dominate storage. Keeping them separate allows independent compaction and backup strategies
2. **Access patterns**: During search, only embeddings for the ~1000 candidate documents are read. The config database is rarely accessed during search
3. **Rebuild independence**: The embedding database can be deleted and rebuilt from source documents without losing collection definitions or settings

## Tantivy Index Schema

### Fields

| Field        | Type   | Options      | Purpose                                                     |
| ------------ | ------ | ------------ | ----------------------------------------------------------- |
| `doc_id`     | STRING | STORED       | Short hash ID (e.g. "abc123") for display and `docbert get` |
| `doc_num_id` | u64    | STORED, FAST | Numeric ID matching redb keys                               |
| `collection` | STRING | STORED, FAST | Collection name for filtering                               |
| `path`       | STRING | STORED       | Relative file path within collection                        |
| `title`      | TEXT   | STORED       | Document title (first heading or filename)                  |
| `body`       | TEXT   | (not stored) | Full document text for search                               |
| `mtime`      | u64    | STORED, FAST | File modification time (epoch seconds)                      |

### Tokenizer Configuration

- `body` field: `en_stem` tokenizer (English stemming, lowercase, remove long tokens)
- `title` field: `en_stem` tokenizer
- `collection` field: STRING (exact match, not tokenized)
- `doc_id` field: STRING (exact match)

### Index Options

- body field uses `WithFreqsAndPositions` for phrase queries and BM25 scoring
- title field uses `WithFreqsAndPositions` with a 2x boost during search
- FAST fields on collection and mtime enable efficient filtering and sorting

## Document ID Generation

Each document gets two IDs:

1. **Numeric ID (u64)**: A hash of `(collection_name, relative_path)` using a stable hash function. Used as the key in redb tables. Collisions are handled by appending a counter suffix before re-hashing
2. **Short ID (string)**: The first 6 hex characters of the numeric ID (e.g., "abc123"). Used for human-readable display. If a collision occurs at the short-ID level, additional characters are added until unique

The numeric ID is deterministic given the same collection name and relative path, enabling idempotent re-indexing.

## Compaction and Maintenance

### redb compaction

redb's copy-on-write B-tree can accumulate dead pages over time. Running `Database::compact()` reclaims space. This should be exposed as a `docbert maintenance` command.

### Tantivy segment merging

Tantivy automatically merges segments during commits, but heavy indexing can leave many small segments. The `IndexWriter::merge()` API can force a merge. This is also a candidate for the maintenance command.

### Rebuilding from source

Since the source of truth is the filesystem, all indexes can be rebuilt:

```
docbert rebuild            # Re-index all collections from scratch
docbert rebuild -c notes   # Re-index only the "notes" collection
```

This deletes existing Tantivy entries and redb embeddings for the affected documents and re-processes them.
