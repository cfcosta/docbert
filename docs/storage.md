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

Within that root, docbert currently uses five storage layers:

| Path / system                    | Role                                                                                                                             |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `config.db` (+ `config.db-lock`) | collections, contexts, document metadata, chunk manifests and chunk ownership, conversations, collection snapshots, and settings |
| `embeddings.db` (+ `…-lock`)     | stored ColBERT embedding matrices keyed by content-derived chunk ID                                                              |
| `tantivy/`                       | lexical search index                                                                                                             |
| `plaid.idx`                      | PLAID semantic index — compressed centroid assignments over the embeddings for fast MaxSim                                       |
| collection roots on disk         | source document content used for indexing, document reads, titles, and excerpts                                                  |

The `*.db` files are LMDB single-file environments (`NO_SUB_DIR`); the `*-lock` siblings are LMDB's inter-process lock files. Files still in the redb format used before 1.0 are refused on open (see [Legacy formats from releases before 1.0](#legacy-formats-from-releases-before-10)).

docbert is **not** purely index-backed. The source files in registered collection roots remain part of the live system.

## Data directory layout

`DataDir` resolves the following paths:

```text
<docbert-data-dir>/
  config.db
  embeddings.db
  plaid.idx
  tantivy/
```

`tantivy/` is created on demand when the search index is opened. `plaid.idx` is built by `sync`, `rebuild`, or `reindex` and is not present on a fresh data directory; both `semantic` and `hybrid` search fail with `PlaidIndexMissing` until one of those commands has run.

The collection roots themselves are **not** stored inside the data directory unless you explicitly register paths there. They can live anywhere on disk.

## Storage responsibilities by layer

## `config.db`

`config.db` is the main metadata and configuration database.

It is a local [LMDB](https://www.symas.com/lmdb) environment opened through the [`heed`](https://docs.rs/heed) crate by `ConfigDb`. LMDB gives docbert proper cross-process readers and writers, so multiple `docbert mcp` / `docbert web` / CLI invocations can share one data dir without stepping on each other. Alongside the data file, LMDB writes a `<path>-lock` sibling (`config.db-lock` next to `config.db`) for its inter-process lock; the lock file is small and gets recreated on demand.

It owns these named LMDB databases:

- `collections`
- `contexts`
- `document_metadata`
- `doc_chunks`
- `chunk_owners`
- `conversations`
- `collection_merkle_snapshots`
- `settings`

This database is shared across the CLI, web runtime, and MCP runtime.

## `embeddings.db`

`embeddings.db` stores the semantic retrieval data used for ColBERT reranking and semantic search.

It is intentionally separate from `config.db` because embeddings are much larger and are often rebuilt independently from the lighter metadata/config state.

## `tantivy/`

The `tantivy/` directory stores the lexical search index. It holds Tantivy's own on-disk format (segment files plus `meta.json`), created on demand when the index is first opened.

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

Internally it is an LMDB environment with eight named databases. Keys are typed (`&str` / `u64`); values are stored as binary blobs and decoded by `ConfigDb`.

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
- `sync` change detection is driven by Merkle snapshots, not these mtimes alone
- the metadata is required for document lookup, deletion, result decoration, and semantic-search candidate enumeration

## Table: `doc_chunks`

Purpose:

- stores each document's chunk manifest: the ordered list of chunks the document was split into, with the byte range each chunk occupies in the source file

Shape:

- key: numeric document ID (`u64`)
- value: ordered list of `DocChunkEntry` records

Each `DocChunkEntry` contains:

- `chunk_doc_id`: the chunk's content-derived ID, the same ID that keys the chunk's row in `embeddings.db`
- `start_byte`: byte offset where the chunk begins in the source document
- `byte_len`: byte length of the chunk in the source document

Important behavior:

- chunk IDs are content-derived, so the same chunk text can belong to many documents at different byte offsets; the manifest is the per-document record of those offsets
- the same chunk text appearing twice in one document produces two entries with the same ID and different byte ranges
- `ChunkByteOffset` values are derived from the manifest via `ConfigDb::get_chunk_offset_for_doc(doc_num_id, chunk_doc_id)`, not stored per chunk ID
- search consumers resolve a matching chunk's byte range by passing `FinalResult.best_chunk_doc_id` to `get_chunk_offset_for_doc`; a missing manifest or chunk entry falls back to no chunk-range information
- written during sync/rebuild and web ingestion; removed via `ConfigDb::remove_doc_chunks` / `batch_remove_doc_chunks`, which update `chunk_owners` in the same transaction and leave embedding rows in place

## Table: `chunk_owners`

Purpose:

- reverse index from chunk ID to the documents that contain it

Shape:

- key: chunk ID (`u64`)
- value: sorted, deduplicated list of owning numeric document IDs

Important behavior:

- maintained atomically with `doc_chunks`: every manifest write or removal adjusts the affected owners lists in the same LMDB write transaction, and an owners entry is dropped once its list is empty
- lets identical chunk text be shared across documents while staying attributable to each of them; semantic search uses it to fan a chunk hit in the PLAID index back out to every owning document
- `docbert clean` uses it for garbage collection: embedding rows whose chunk ID has no owners are orphans and get removed

## Table: `conversations`

Purpose:

- stores persisted chat conversations

Shape:

- key: conversation ID (`&str`)
- value: rkyv-encoded conversation record (`StoredConversation`)

A stored conversation contains:

- conversation metadata such as `id`, `title`, `created_at`, `updated_at`
- message history
- per-message roles, actors, parts, and optional sources

Records are accepted only in this rkyv format. A record that fails to decode is treated as absent: a lookup by ID returns nothing, listing skips the record (with a tracing warning), and it can still be deleted by ID.

This is the persistent backend state behind the web chat/conversation API.

For the full conversation model, see [`chat-and-conversations.md`](./chat-and-conversations.md).

## Table: `collection_merkle_snapshots`

Purpose:

- stores one Merkle snapshot per collection
- lets `sync` detect new, changed, and deleted files by comparing snapshots
- keeps web document mutations and indexing state aligned with the discovered collection contents

Shape:

- key: collection name (`&str`)
- value: serialized `Snapshot`

A snapshot includes:

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
- `llm_oauth:openai-codex`
  - structured JSON blob holding the local ChatGPT Plus/Pro (Codex) OAuth session when that provider is used

### Document-scoped settings entries

The settings table also stores per-document JSON user metadata under generated keys:

- `doc_meta:{doc_id}`

These are used by the web document/search APIs to attach user metadata to documents.

### Compatibility / cleanup note

`ConfigDb::batch_remove_document_state` also removes keys with this prefix:

- `doc_content:{doc_id}`

The current code removes those keys for cleanup safety, but the implementation does not write document content into `config.db`.

## Conversation and LLM settings persistence

Two user-visible features persist state in `config.db` beyond classic indexing metadata.

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
- `llm_oauth:openai-codex`
  - JSON-encoded OAuth credentials for the ChatGPT Plus/Pro (Codex) flow

Read behavior is broader than write behavior:

- if `llm_api_key` is stored, docbert returns that value for API-key-backed providers
- if it is not stored and the provider is `openai`, docbert falls back to `OPENAI_API_KEY`
- if it is not stored and the provider is `anthropic`, docbert falls back to `ANTHROPIC_API_KEY`
- if the provider is `openai-codex`, docbert resolves the separate OAuth blob instead of `llm_api_key`
- if that OAuth blob is close to expiry, docbert refreshes it before returning `/v1/settings/llm`

So the persisted settings record and the effective runtime value are not always identical.

## Snapshot storage and change tracking

Merkle snapshots are part of the storage model: they define how docbert decides what changed in a collection.

High-level flow:

1. discover the current supported files in a collection
2. build a deterministic Merkle snapshot from those files
3. compare that snapshot to the previously stored snapshot in `config.db`
4. classify paths as new, changed, or deleted
5. update `tantivy/`, `embeddings.db`, and metadata
6. replace the stored snapshot only if the operation succeeds

This keeps `config.db` as the canonical record of the last fully successful indexing view of each collection.

## `embeddings.db` details

`embeddings.db` stores embedding matrices keyed by numeric chunk IDs.

Chunk IDs are content-derived: each ID is a hash of the embedding model ID and the chunk text (`chunking::chunk_doc_id`). Identical chunk text produces the same ID in every document that contains it, so one row can be shared by many documents, and one source document usually maps to several rows through its `doc_chunks` manifest.

Current storage responsibilities:

- hybrid search reranking
- semantic-only search
- content-addressed embedding cache: rows are retained when documents change or disappear, so re-indexing identical chunk text is a cache hit instead of a re-encode

Orphaned rows (chunk IDs that no document references in `chunk_owners`) are garbage-collected only by `docbert clean`.

Each stored value is packed as:

- 4 bytes: token count (`u32`, little-endian)
- 4 bytes: embedding dimension (`u32`, little-endian)
- `token_count × dimension × 2` bytes: `bf16` values (little-endian, 2 bytes each) in row-major order

Components are stored at `bf16` precision: the encoder trunk computes in bf16 on CUDA, so the extra mantissa bits of an `f32` representation are noise below the model's own precision floor, and the PLAID index re-quantizes every token to 2 bits per dimension downstream. Halving the per-component width halves what is by far the largest file in the data directory.

Entries written by docbert releases before 1.0 carried row-major `f32` data instead. That layout is not decoded: reads fail with an error pointing at `docbert clean`, which drops the undecodable rows and clears the per-document state so the next `sync` re-embeds the affected documents (see [Legacy formats from releases before 1.0](#legacy-formats-from-releases-before-10)).

Because embeddings are the largest stored artifact, `embeddings.db` is usually the main consumer of disk space in docbert.

## Tantivy storage details

The `tantivy/` directory stores the lexical index entries for each prepared document.

The current schema includes:

- document ID string (stored)
- numeric document ID (stored, fast)
- collection (stored, fast)
- relative path (stored)
- title (stored, indexed with English stemming and 2x boost)
- body (indexed with English stemming, **not stored**)
- mtime (stored, fast)

Important boundary:

- Tantivy stores enough metadata for retrieval result decoration but not the body itself; body bytes only exist in the inverted index
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
- `plaid.idx` (rebuilt or incrementally updated for touched documents)
- `config.db` `document_metadata`
- `config.db` `doc_chunks` and `chunk_owners`
- `config.db` `settings` via `embedding_model`
- `config.db` `collection_merkle_snapshots`

May remove:

- deleted documents from Tantivy
- deleted documents' chunk manifests from `config.db` `doc_chunks`, with the matching `chunk_owners` entries
- deleted document metadata and document user metadata from `config.db`

Sync does not remove embedding rows: they stay in `embeddings.db` as a content-addressed cache. The only command that removes embedding rows is `docbert clean`.

## `docbert rebuild`

May clear and rebuild:

- `tantivy/`
- `plaid.idx`
- `config.db` `document_metadata`
- `config.db` `doc_chunks` and `chunk_owners`
- `config.db` `collection_merkle_snapshots`

Also updates:

- `config.db` `settings.embedding_model`

Rebuild does not clear `embeddings.db`. It removes chunk manifests and document artifacts, rewrites Tantivy state, and re-embeds the collections; because embedding rows are content-addressed, chunks whose text is unchanged resolve to existing rows as cache hits instead of being re-encoded.

## `docbert reindex`

Rewrites only:

- `plaid.idx`

Does not read or write source files, embeddings, Tantivy, or any other `config.db` table. The PLAID index is retrained over every embedding currently in `embeddings.db`. Use this when only the PLAID builder parameters changed (centroid count, codec bit-width, k-means iterations, …) and embeddings remain valid.

## Web document upload

Writes:

- source file into the collection root
- Tantivy entry for that document
- embedding rows for the document's chunks, keyed by content-derived chunk ID (rewriting a row for identical chunk text stores identical bytes)
- the document's chunk manifest in `doc_chunks`, with the matching `chunk_owners` entries
- `document_metadata`
- optional `doc_meta:{doc_id}` JSON metadata
- updated collection snapshot

Upload does not update `plaid.idx`; the PLAID index is built only by `sync`, `rebuild`, or `reindex`.

## Web document delete

Removes:

- source file from the collection root
- the document's chunk manifest from `doc_chunks`, with the matching `chunk_owners` entries
- document metadata
- optional `doc_meta:{doc_id}` JSON metadata
- Tantivy entry

Then refreshes:

- collection snapshot

Delete touches neither `plaid.idx` nor `embeddings.db`: embedding rows stay behind as content-addressed cache entries, searches skip chunks whose owners list is empty, and `docbert clean` garbage-collects rows no document owns.

## Web conversation/settings APIs

Write to:

- `conversations`
- `settings` (`llm_provider`, `llm_model`, `llm_api_key`, `llm_oauth:openai-codex`)

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

On open, `ConfigDb` ensures the expected named LMDB databases exist.

### Legacy formats from releases before 1.0

docbert 1.0 dropped every migration path for data written by older releases. `docbert clean` is the single recovery command; it detects each legacy artifact and resets exactly as much as necessary (use `--dry-run` to preview):

- **redb-format `config.db` / `embeddings.db`** (releases before 1.0): `ConfigDb::open` / `EmbeddingDb::open` sniff the first nine bytes and refuse redb files. `clean` handles them below the open layer: a legacy `embeddings.db` is deleted along with `plaid.idx` and the per-document state (collections stay registered; the next `sync` re-embeds everything); a legacy `config.db` resets the whole data dir, after which collections must be re-added.
- **`f32`-layout embedding rows** (pre-bf16 releases): reads refuse them; `clean` drops the rows and clears document state so `sync` re-embeds the affected documents. Rows already in the bf16 layout are kept and reused.
- **`plaid.idx` versions 1–2**: the loader rejects them; `docbert reindex` (or a full `rebuild`) regenerates the index from the stored embeddings.
- **conversation records from before 1.0** (payloads that are not current-format rkyv): reads treat them as absent; lookups return nothing and listings skip them with a warning. `docbert clean` does not touch them; deleting the affected conversation removes the record.

docbert never creates or touches `.redb-bak` files; if earlier releases left any behind, delete them manually.

### Hard schema breaks

If the LMDB env reports an unexpected database type or layout (very rare), docbert surfaces that as a configuration error and instructs you to back up and reset `config.db`.

## Related references

- [`architecture.md`](./architecture.md)
- [`pipeline.md`](./pipeline.md)
- [`chat-and-conversations.md`](./chat-and-conversations.md)
- [`web-api.md`](./web-api.md)
