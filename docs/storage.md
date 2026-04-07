# Storage

## Overview

docbert keeps its local state under one resolved data directory.

By default that is the XDG data path for `docbert`, usually:

```text
~/.local/share/docbert/
```

The actual location is resolved in this order:

1. `--data-dir <path>`
2. `DOCBERT_DATA_DIR`
3. the XDG data directory (`$XDG_DATA_HOME/docbert` or the platform equivalent)

Within that root, docbert currently uses four storage layers:

| Path / system | Role |
| --- | --- |
| `config.db` | collections, contexts, document metadata, conversations, collection snapshots, and settings |
| `embeddings.db` | stored ColBERT embedding matrices keyed by numeric document or chunk ID |
| `tantivy/` | lexical search index |
| collection roots on disk | source document content used for indexing, document reads, titles, and excerpts |

A key architectural point is that docbert is **not** purely index-backed. The source files in registered collection roots remain part of the live system.

## Data directory layout

`DataDir` resolves the following paths:

```text
<docbert-data-dir>/
  config.db
  embeddings.db
  tantivy/
```

`tantivy/` is created on demand when the search index is opened.

The collection roots themselves are **not** stored inside the data directory unless you explicitly register paths there. They can live anywhere on disk.

## Storage responsibilities by layer

## `config.db`

`config.db` is the main metadata and configuration database.

It is a local [redb](https://github.com/cberner/redb) database opened by `ConfigDb`.

It currently owns these tables:

- `collections`
- `contexts`
- `document_metadata`
- `conversations`
- `collection_merkle_snapshots`
- `settings`

This database is shared across the CLI, web runtime, and MCP runtime.

## `embeddings.db`

`embeddings.db` stores the semantic retrieval data used for ColBERT reranking and semantic search.

It is intentionally separate from `config.db` because embeddings are much larger and are often rebuilt independently from the lighter metadata/config state.

## `tantivy/`

The `tantivy/` directory stores the lexical search index.

It is used for:

- hybrid search candidate generation
- CLI and web retrieval paths that depend on BM25/fuzzy search
- collection-wide delete/rebuild operations that rewrite lexical state

## Collection roots on disk

The registered collection roots remain the source of truth for document content.

Current read/write behavior:

- `sync` and `rebuild` discover and load files from the collection roots
- web ingestion writes uploaded source files into those roots
- web deletion removes source files from those roots
- search result titles and excerpts are recomputed from current files on disk when possible
- document retrieval endpoints read current files from disk
- semantic search checks current on-disk content to decide whether a document has a non-empty semantic body

That means docbert stores **indexed state** locally, but still depends on the live filesystem for many reads.

## `config.db` details

## Encoding model

`config.db` is not a plain SQL schema.

Internally it is a redb key-value database with typed table definitions. Values are stored as binary blobs and decoded by `ConfigDb`.

Current encoding patterns:

- most typed structs use checked `rkyv` serialization
- plain settings strings are stored as encoded string values
- JSON settings are stored through `StoredJsonValue`

## Table: `collections`

Purpose:

- maps a collection name to its filesystem root

Shape:

- key: collection name (`&str`)
- value: encoded path string blob

Examples:

- `notes -> /home/user/notes`
- `docs -> /srv/wiki`

This table is used by all runtime surfaces to resolve where a document actually lives on disk.

## Table: `contexts`

Purpose:

- stores human-authored context descriptions for URIs such as `bert://notes`

Shape:

- key: URI string
- value: encoded description string blob

This is mainly used by retrieval and MCP-related flows that want collection/document context text.

## Table: `document_metadata`

Purpose:

- stores the core per-document metadata used to connect doc IDs back to collection paths
- supports lookup by short doc ID or relative path
- tracks file mtimes for metadata purposes

Shape:

- key: numeric document ID (`u64`)
- value: serialized `DocumentMetadata`

`DocumentMetadata` currently contains:

- `collection`
- `relative_path`
- `mtime`

Important notes:

- the numeric ID is derived deterministically from `(collection, relative_path)`
- `sync` change detection is now driven by Merkle snapshots, not just these mtimes
- the metadata is still required for document lookup, deletion, result decoration, and semantic-search candidate enumeration

## Table: `conversations`

Purpose:

- stores persisted chat conversations

Shape:

- key: conversation ID (`&str`)
- value: serialized `Conversation`

A stored conversation currently contains:

- conversation metadata such as `id`, `title`, `created_at`, `updated_at`
- message history
- per-message roles, actors, parts, and optional sources

This is the persistent backend state behind the web chat/conversation API.

For the full conversation model, see [`chat-and-conversations.md`](./chat-and-conversations.md).

## Table: `collection_merkle_snapshots`

Purpose:

- stores one Merkle snapshot per collection
- lets `sync` detect new, changed, and deleted files by comparing snapshots
- keeps web document mutations and indexing state aligned with the discovered collection contents

Shape:

- key: collection name (`&str`)
- value: serialized `CollectionMerkleSnapshot`

A snapshot currently includes:

- collection name
- root hash
- persisted directory nodes
- persisted file leaves

File and directory hashes are based on BLAKE3.

Important behavior:

- `sync` computes a fresh snapshot before doing work, but only stores it after success
- `rebuild` stores a fresh snapshot only after the rebuild succeeds
- web ingest/delete refreshes the snapshot only after mutation work succeeds end to end
- if the operation fails, docbert preserves the previous snapshot

## Table: `settings`

Purpose:

- stores general settings, persisted LLM settings, and document-scoped JSON metadata entries

Shape:

- key: string
- value: encoded string or encoded JSON blob, depending on the helper used

This is the most mixed-use table in `config.db`.

### Stable setting keys in current use

Current implementation-visible keys include:

- `model_name`
  - the stored default retrieval model selected by `docbert model set`
- `embedding_model`
  - the model ID last used to compute the current embeddings
  - used to block `sync` when the active model no longer matches the stored embeddings
- `llm_provider`
  - persisted chat/provider setting
- `llm_model`
  - persisted chat/model setting
- `llm_api_key`
  - persisted chat API key, if stored in docbert instead of coming from environment variables

### Document-scoped settings entries

The settings table also stores per-document JSON user metadata under generated keys:

- `doc_meta:{doc_id}`

These are used by the web document/search APIs to attach user metadata to documents.

### Compatibility / cleanup note

`ConfigDb::batch_remove_document_state` also removes keys with this prefix:

- `doc_content:{doc_id}`

The current code removes those keys for cleanup safety, but the present implementation does not actively write document content into `config.db`.

## Conversation and LLM settings persistence

Two user-visible features now persist state in `config.db` beyond classic indexing metadata.

### Conversations

Conversation history is stored in the `conversations` table.

That means conversations survive process restarts for:

- `docbert web`
- any future tooling that uses the same `ConfigDb`

### LLM settings

The web settings API persists provider/model/API-key choices into the `settings` table through the `PersistedLlmSettings` helper.

Stored keys:

- `llm_provider`
- `llm_model`
- `llm_api_key`

Read behavior is slightly broader than write behavior:

- if `llm_api_key` is stored, docbert returns that value
- if it is not stored and the provider is `openai`, docbert falls back to `OPENAI_API_KEY`
- if it is not stored and the provider is `anthropic`, docbert falls back to `ANTHROPIC_API_KEY`

So the persisted settings record and the effective runtime value are not always identical.

## Snapshot storage and change tracking

Merkle snapshots are now part of the normal storage model, not just an implementation detail.

They matter because they define how docbert decides what changed in a collection.

High-level flow:

1. discover the current supported files in a collection
2. build a deterministic Merkle snapshot from those files
3. compare that snapshot to the previously stored snapshot in `config.db`
4. classify paths as new, changed, or deleted
5. update `tantivy/`, `embeddings.db`, and metadata
6. replace the stored snapshot only if the operation succeeds

This keeps `config.db` as the canonical record of the last fully successful indexing view of each collection.

## `embeddings.db` details

`embeddings.db` stores embedding matrices keyed by numeric IDs.

Those IDs can represent:

- the base document ID
- chunk-specific IDs derived from the base document ID

That means one logical source document may correspond to multiple embedding rows.

Current storage responsibilities:

- hybrid search reranking
- semantic-only search
- web ingest replacement cleanup for stale chunk embeddings
- rebuild/sync removal of deleted document families

The binary layout is implementation-specific, but conceptually each entry stores:

- token count
- embedding dimension
- row-major `f32` vector data

Because embeddings are the largest stored artifact, `embeddings.db` is usually the main consumer of disk space in docbert.

## Tantivy storage details

The `tantivy/` directory stores the lexical index entries for each prepared document.

The current indexed fields include:

- document ID string
- numeric document ID
- collection
- relative path
- title
- body
- mtime

Important boundary:

- Tantivy stores enough to perform lexical retrieval and return candidate metadata
- it is **not** the sole source of returned titles, excerpts, or document content
- the web layer often rereads the source file from disk and recomputes title/excerpt information

## Storage lifecycle by operation

## `docbert collection add`

Writes:

- `config.db` `collections`

Does not write:

- `embeddings.db`
- `tantivy/`
- collection snapshots

## `docbert sync`

May update:

- `tantivy/`
- `embeddings.db`
- `config.db` `document_metadata`
- `config.db` `settings` via `embedding_model`
- `config.db` `collection_merkle_snapshots`

May remove:

- deleted documents from Tantivy
- deleted document families from `embeddings.db`
- deleted document metadata and document user metadata from `config.db`

## `docbert rebuild`

May clear and rebuild:

- `tantivy/`
- `embeddings.db`
- `config.db` `document_metadata`
- `config.db` `collection_merkle_snapshots`

Also updates:

- `config.db` `settings.embedding_model`

## Web document upload

Writes:

- source file into the collection root
- Tantivy entry for that document
- embedding rows for that document family
- `document_metadata`
- optional `doc_meta:{doc_id}` JSON metadata
- updated collection snapshot

## Web document delete

Removes:

- source file from the collection root
- Tantivy entry
- embedding family
- document metadata
- optional `doc_meta:{doc_id}` JSON metadata

Then refreshes:

- collection snapshot

## Web conversation/settings APIs

Write to:

- `conversations`
- `settings` (`llm_provider`, `llm_model`, `llm_api_key`)

## Operational notes

## Rebuildability

Because source files remain authoritative, docbert can rebuild most derived state from disk:

- Tantivy entries
- embeddings
- metadata
- collection snapshots

A rebuild does **not** recreate everything in `config.db` from documents alone, though. User-managed state such as collections, contexts, conversations, and persisted LLM settings still lives only in `config.db`.

## Model mismatch safety

`embedding_model` in `config.db` is used as a safety check.

If you switch to a different embedding model and then run `sync`, docbert refuses to mix old and new embeddings and tells you to run `rebuild` instead.

## Schema compatibility

On open, `ConfigDb` ensures the expected tables exist.

If redb reports a table type mismatch or incompatible schema definition, docbert surfaces that as a configuration error and instructs you to back up and reset `config.db`.

## Related references

- [`architecture.md`](./architecture.md)
- [`pipeline.md`](./pipeline.md)
- [`chat-and-conversations.md`](./chat-and-conversations.md)
- [`web-api.md`](./web-api.md)
