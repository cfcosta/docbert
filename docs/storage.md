# Storage design

## Overview

docbert stores three different kinds of data, and each one lives in the place that fits it best.

| System                | Purpose                                             | Format                 |
| --------------------- | --------------------------------------------------- | ---------------------- |
| redb (config)         | Collection definitions, document metadata, settings | Typed key-value store  |
| redb (embeddings)     | ColBERT per-token embedding matrices                | Binary key-value store |
| Tantivy               | Full-text search index (BM25 + fuzzy)               | Inverted index on disk |

All of it lives under the XDG data directory: `$XDG_DATA_HOME/docbert/`, usually `~/.local/share/docbert/`.

## XDG directory layout

```text
~/.local/share/docbert/
  config.db              # configuration and metadata database
  embeddings.db          # ColBERT embedding vectors
  tantivy/               # Tantivy index directory
    meta.json            # Tantivy metadata
    <uuid>.<ext>         # Tantivy segment files
```

The `xdg` crate handles platform-appropriate directory resolution.

## `config.db` schema

### Table: `collections`

Maps collection names to filesystem paths.

- Key: `&str` such as `"notes"`
- Value: `&str` such as `"/home/user/notes"`

### Table: `contexts`

Stores user-defined context strings for collections.

- Key: `&str` such as `"bert://notes"`
- Value: `&str` such as `"Personal notes and ideas"`

### Table: `document_metadata`

Stores per-document metadata used by incremental indexing.

- Key: `u64` document numeric ID
- Value: `&[u8]` containing a UTF-8 string serialized as `collection\0relative_path\0mtime`, where `mtime` is seconds since the Unix epoch

### Table: `settings`

Stores global configuration values.

- Key: `&str` setting name
- Value: `&str` setting value

Current settings include:

- `model_name`: HuggingFace model ID or local PyLate model path; defaults to `lightonai/ColBERT-Zero`

At the moment, that is the only setting the codebase actually uses.

## `embeddings.db` schema

### Table: `embeddings`

Stores precomputed ColBERT token embeddings.

- Key: `u64` document numeric ID; chunked documents use a chunk-specific ID derived from the base document ID
- Value: `&[u8]` binary embedding payload

Binary layout:

- 4 bytes: token count `T` as little-endian `u32`
- 4 bytes: embedding dimension `D` as little-endian `u32` (currently 128)
- `T * D * 4` bytes: row-major `f32` values in little-endian order

Example for a document with 200 tokens at 128 dimensions:

- header: 8 bytes
- data: `200 * 128 * 4 = 102,400` bytes
- total: `102,408` bytes, or about 100 KB

### Why a separate redb file?

Embeddings live in their own database for a few practical reasons:

1. They dominate storage size, so it helps to keep them separate from small configuration data
2. Search usually reads embeddings for only the candidate set, while `config.db` is touched much less often
3. You can delete and rebuild the embedding database without losing collection definitions or settings

## Tantivy index schema

### Fields

| Field        | Type   | Options      | Purpose                                                             |
| ------------ | ------ | ------------ | ------------------------------------------------------------------- |
| `doc_id`     | STRING | STORED       | Short hash ID with `#` prefix, used for display and `docbert get`   |
| `doc_num_id` | `u64`  | STORED, FAST | Numeric ID matching redb keys                                       |
| `collection` | STRING | STORED, FAST | Collection name for filtering                                       |
| `path`       | STRING | STORED       | Relative file path within the collection                            |
| `title`      | TEXT   | STORED       | Document title from the first heading or the filename               |
| `body`       | TEXT   | not stored   | Full document text used for search                                  |
| `mtime`      | `u64`  | STORED, FAST | File modification time in epoch seconds                             |

### Tokenizer configuration

- `body` uses the `en_stem` tokenizer: lowercase, stemming, and long-token handling
- `title` also uses `en_stem`
- `collection` is `STRING`, so it is matched exactly and not tokenized
- `doc_id` is `STRING`, also matched exactly

### Index options

- `body` uses `WithFreqsAndPositions` for phrase queries and BM25 scoring
- `title` uses `WithFreqsAndPositions` and gets a 2x search-time boost
- FAST fields on `collection` and `mtime` make filtering and sorting cheaper

## Document ID generation

Each document gets two IDs:

1. Numeric ID (`u64`): a hash of `(collection_name, relative_path)` using Rust's `DefaultHasher`. This is the redb key.
2. Short ID (`string`): the first 6 hex characters of the numeric ID, usually shown with a leading `#` for display and lookup.

The numeric ID is deterministic for a given collection name and relative path, which makes re-indexing idempotent. There is no explicit collision handling in the current code; `docbert get #<id>` returns the first matching document.

Chunk IDs for embeddings are derived by XOR-ing the base numeric ID with `(chunk_index << 48)`. Chunk 0 uses the base ID unchanged.

## Compaction and maintenance

### redb compaction

redb uses a copy-on-write B-tree, so dead pages can build up over time. Running `Database::compact()` reclaims space. That should eventually be exposed as a `docbert maintenance` command.

### Tantivy segment merging

Tantivy merges segments automatically during commits, but heavy indexing can still leave many small segments behind. `IndexWriter::merge()` can force a merge. That is another good candidate for a maintenance command.

### Rebuilding from source

The filesystem is the source of truth, so indexes can always be rebuilt:

```text
docbert rebuild            # Re-index all collections from scratch
docbert rebuild -c notes   # Re-index only the "notes" collection
```

This removes existing Tantivy entries and embeddings for the affected scope, then indexes the source files again.
