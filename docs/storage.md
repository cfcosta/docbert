# Storage

## Overview

docbert keeps its local state in one data directory.

By default this is the XDG data path for `docbert`. Usually this is:

```text
~/.local/share/docbert/
```

docbert finds the location in this sequence:

1. `--data-dir <path>`
2. `DOCBERT_DATA_DIR`
3. The XDG data directory (`$XDG_DATA_HOME/docbert` or the platform equivalent)

In that root, docbert uses five storage layers:

| Path / system                    | Role                                                                                                                             |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `config.db` (+ `config.db-lock`) | collections, contexts, document metadata, chunk manifests and chunk ownership, conversations, collection snapshots, and settings |
| `embeddings.db` (+ `…-lock`)     | ColBERT embedding matrices, keyed by content-derived chunk ID                                                                    |
| `tantivy/`                       | lexical search index                                                                                                             |
| `plaid.idx`                      | PLAID semantic index — compressed centroid assignments over the embeddings for fast MaxSim                                       |
| collection roots on disk         | source document contents for indexing, document reads, titles, and excerpts                                                      |

The `*.db` files are LMDB single-file environments (`NO_SUB_DIR`). The `*-lock` files are LMDB's inter-process lock files. docbert rejects files that use the redb format from before 1.0 when it opens them (see [Legacy formats from releases before 1.0](#legacy-formats-from-releases-before-10)).

docbert does **not** use only the index. The source files in the collection roots that you add stay part of the active system.

## Data directory layout

`DataDir` finds these paths:

```text
<docbert-data-dir>/
  config.db
  embeddings.db
  plaid.idx
  tantivy/
```

docbert makes `tantivy/` when it first opens the search index. The `sync`, `rebuild`, or `reindex` commands make `plaid.idx`. A new data directory does not have `plaid.idx`. Thus `semantic` and `hybrid` search give a `PlaidIndexMissing` error until one of those commands runs.

docbert does **not** keep the collection roots in the data directory, unless you add paths there. The roots can be at other locations on disk.

## Storage functions by layer

## `config.db`

`config.db` is the primary metadata and configuration database.

`ConfigDb` opens a local [LMDB](https://www.symas.com/lmdb) environment through the [`heed`](https://docs.rs/heed) crate. LMDB gives docbert correct cross-process readers and writers. Thus you can safely run `docbert mcp`, `docbert web`, and CLI commands at the same time on one data directory. Next to the data file, LMDB writes a `<path>-lock` file (`config.db-lock` next to `config.db`) for its inter-process lock. The lock file is small, and docbert makes it again when necessary.

`config.db` has these named LMDB databases:

- `collections`
- `contexts`
- `document_metadata`
- `doc_chunks`
- `chunk_owners`
- `conversations`
- `collection_merkle_snapshots`
- `settings`

The CLI, web runtime, and MCP runtime all use this database.

## `embeddings.db`

`embeddings.db` keeps the semantic retrieval data for ColBERT reranking and semantic search.

docbert keeps `embeddings.db` apart from `config.db`. The embeddings use much more space than the smaller metadata and config state. docbert also makes the embeddings again frequently, apart from that state.

## `tantivy/`

The `tantivy/` directory keeps the lexical search index. It holds Tantivy's on-disk format (segment files and `meta.json`). docbert makes this directory when it first opens the index.

docbert uses it for:

- Hybrid-search candidate lists
- The CLI and web retrieval paths that use BM25 or fuzzy search
- Collection-wide delete or rebuild operations that rewrite the lexical state

## Collection roots on disk

The collection roots that you add stay the primary source for the document contents.

The read and write operations:

- The `sync` and `rebuild` commands find and read files from the collection roots
- Web ingestion writes the uploaded source files into those roots
- Web deletion removes source files from those roots
- docbert makes search result titles and excerpts again from the files on disk when possible
- Document retrieval endpoints read the files from disk
- Semantic search examines the on-disk contents to find if a document has a non-empty semantic body

So docbert keeps **indexed state** on the local disk. But docbert continues to use the active filesystem for many read operations.

## `config.db` details

## Encoding model

`config.db` does not use a SQL schema.

`config.db` is an LMDB environment with eight named databases. The keys have types (`&str` or `u64`). docbert keeps the values as binary blobs, and `ConfigDb` decodes them.

The encoding patterns:

- Most typed structs use checked `rkyv` serialization.
- docbert keeps the string settings as encoded string values.
- docbert keeps the JSON settings through `StoredJsonValue`.

## Table: `collections`

Purpose:

- Maps a collection name to its filesystem root.

Shape:

- Key: the collection name (`&str`)
- Value: the encoded path string blob

Examples:

- `notes -> /home/user/notes`
- `docs -> /srv/wiki`

All the runtime parts use this table to find where a document is on disk.

## Table: `contexts`

Purpose:

- Keeps the context descriptions that a person writes for URIs, for example `bert://notes`.

Shape:

- Key: the URI string
- Value: the encoded description string blob

In most cases, retrieval and MCP flows use this for the collection or document context text.

## Table: `document_metadata`

Purpose:

- Keeps the primary per-document metadata that connects doc IDs to collection paths.
- Gives lookup by short doc ID or relative path.
- Records the file mtimes for the metadata.

Shape:

- Key: the numeric document ID (`u64`)
- Value: the serialized `DocumentMetadata`

`DocumentMetadata` contains:

- `collection`
- `relative_path`
- `mtime`

Important notes:

- docbert calculates the numeric ID from `(collection, relative_path)`. The same values always give the same ID.
- Merkle snapshots control `sync` change detection, and not only these mtimes.
- docbert uses the metadata to find and delete documents, to add information to the results, and to find the semantic-search candidates.

## Table: `doc_chunks`

Purpose:

- Keeps each document's chunk manifest. The manifest is the list of the chunks in sequence. docbert divided the document into these chunks. Each entry has the byte range that the chunk uses in the source file.

Shape:

- Key: the numeric document ID (`u64`)
- Value: the list of `DocChunkEntry` records in sequence

Each `DocChunkEntry` contains:

- `chunk_doc_id`: the content-derived ID of the chunk, and the same ID that keys the chunk's row in `embeddings.db`
- `start_byte`: the byte offset where the chunk starts in the source document
- `byte_len`: the byte length of the chunk in the source document

Important notes:

- Chunk IDs are content-derived. Thus the same chunk text can be in many documents at different byte offsets. The manifest is the per-document record of those offsets.
- The same chunk text can appear twice in one document. This makes two entries with the same ID and different byte ranges.
- docbert calculates `ChunkByteOffset` values from the manifest with `ConfigDb::get_chunk_offset_for_doc(doc_num_id, chunk_doc_id)`. docbert does not keep them for each chunk ID.
- The search code finds a chunk's byte range. It gives `FinalResult.best_chunk_doc_id` to `get_chunk_offset_for_doc`. If a manifest or chunk entry is missing, docbert gives no chunk-range information.
- docbert writes this during `sync`, `rebuild`, and web ingestion. `ConfigDb::remove_doc_chunks` and `batch_remove_doc_chunks` remove it. These update `chunk_owners` in the same transaction and keep the embedding rows.

## Table: `chunk_owners`

Purpose:

- A reverse index from chunk ID to the documents that contain it.

Shape:

- Key: the chunk ID (`u64`)
- Value: the numeric document IDs of the documents that contain the chunk, in numeric sequence. Each ID is in the list one time only.

Important notes:

- docbert updates this and `doc_chunks` in the same LMDB write transaction. Each manifest write or removal changes the applicable `chunk_owners` lists. docbert removes a `chunk_owners` entry when its list is empty.
- This lets documents use the same chunk text. docbert keeps a record of each document that uses the chunk. Semantic search uses this to map a chunk hit in the PLAID index back to each document that contains the chunk.
- `docbert clean` uses it for garbage collection. If a chunk ID has no document in `chunk_owners`, its embedding row is an orphan. `docbert clean` removes the orphan rows.

## Table: `conversations`

Purpose:

- Keeps the chat conversations.

Shape:

- Key: the conversation ID (`&str`)
- Value: the rkyv-encoded conversation record (`StoredConversation`)

A conversation record contains:

- The conversation metadata, for example `id`, `title`, `created_at`, and `updated_at`
- The message history
- The per-message roles, actors, parts, and optional sources

docbert accepts records only in this rkyv format. If a record does not decode, docbert ignores it. A lookup by ID gives nothing. A list request ignores the record and writes a tracing warning. docbert can delete the record by ID.

This is the backend state that docbert keeps for the web chat and conversation API.

The file [`chat-and-conversations.md`](./chat-and-conversations.md) gives the full conversation model.

## Table: `collection_merkle_snapshots`

Purpose:

- Keeps one Merkle snapshot for each collection.
- Lets `sync` compare snapshots to find new, changed, and deleted files.
- Keeps web document mutations and indexing state aligned with the collection contents that `sync` finds.

Shape:

- Key: the collection name (`&str`)
- Value: the serialized `Snapshot`

A snapshot includes:

- The collection name
- The root hash
- The directory nodes
- The file leaves

File and directory hashes use BLAKE3.

Important notes:

- `sync` calculates a new snapshot before it does the work. `sync` keeps the snapshot only after the work completes correctly.
- `rebuild` keeps a new snapshot only after it completes correctly.
- Web ingest and delete refresh the snapshot only after the mutation work completes fully.
- If the operation does not complete correctly, docbert keeps the previous snapshot.

## Table: `settings`

Purpose:

- Keeps general settings, the LLM settings, and document-scoped JSON metadata entries.

Shape:

- Key: a string
- Value: an encoded string or an encoded JSON blob. The helper selects which type docbert uses.

This table has the most different functions in `config.db`.

### Stable setting keys in use

The implementation shows these keys:

- `model_name`
  - The default retrieval model that `docbert model set` selects.
- `embedding_model`
  - The model ID that docbert last used to calculate the embeddings.
  - docbert uses this to stop `sync` when the active model does not agree with the embeddings in `embeddings.db`.
- `llm_provider`
  - The chat provider setting.
- `llm_model`
  - The chat model setting.
- `llm_api_key`
  - The chat API key, if docbert keeps it and it does not come from environment variables.
- `llm_oauth:openai-codex`
  - A JSON blob that holds the local ChatGPT Plus/Pro (Codex) OAuth session. docbert uses this blob for that provider.

### Document-scoped settings entries

The settings table also keeps per-document JSON user metadata under these keys:

- `doc_meta:{doc_id}`

The web document and search APIs use these to attach user metadata to documents.

### Compatibility and cleanup note

`ConfigDb::batch_remove_document_state` also removes keys with this prefix:

- `doc_content:{doc_id}`

The code removes those keys for cleanup safety. But the implementation does not write document contents into `config.db`.

## Conversation and LLM settings storage

You see two features that keep state in `config.db`. This state is more than the usual indexing metadata.

### Conversations

docbert keeps the conversation history in the `conversations` table.

So the conversations continue when these restart:

- `docbert web`
- Other tools that will use the same `ConfigDb`

### LLM settings

The web settings API keeps the provider, model, and API-key selections in the `settings` table. It uses the `PersistedLlmSettings` helper.

These keys:

- `llm_provider`
- `llm_model`
- `llm_api_key`
- `llm_oauth:openai-codex`
  - The JSON-encoded OAuth credentials for the ChatGPT Plus/Pro (Codex) flow.

docbert reads the LLM settings with this method:

- If docbert has `llm_api_key`, docbert gives that value for API-key-backed providers.
- If docbert does not have it and the provider is `openai`, docbert uses `OPENAI_API_KEY`.
- If docbert does not have it and the provider is `anthropic`, docbert uses `ANTHROPIC_API_KEY`.
- If the provider is `openai-codex`, docbert uses the different OAuth blob and not `llm_api_key`.
- If that OAuth blob is near its end time, docbert refreshes it before it gives `/v1/settings/llm`.

So the settings record on disk and the runtime value are not always the same.

## Snapshot storage and change records

Merkle snapshots are part of the storage model. They give the method that docbert uses to find the changes in a collection.

docbert uses this general flow:

1. It finds the applicable files in a collection.
2. It makes a deterministic Merkle snapshot from those files.
3. It compares that snapshot to the snapshot in `config.db`.
4. It identifies each path as new, changed, or deleted.
5. It updates `tantivy/`, `embeddings.db`, and the metadata.
6. It replaces the snapshot on disk only if the operation completes correctly.

This keeps `config.db` as the primary record of the last indexing view of each collection that fully completed.

## `embeddings.db` details

`embeddings.db` keeps embedding matrices, keyed by numeric chunk IDs.

Chunk IDs are content-derived. Each ID is a hash of the embedding model ID and the chunk text (`chunking::chunk_doc_id`). The same chunk text makes the same ID in each document that contains it. Thus many documents can use one row. One source document usually maps to some rows through its `doc_chunks` manifest.

The storage functions:

- Hybrid search reranking
- Semantic-only search
- Content-addressed embedding cache. docbert keeps the rows when a document changes and when docbert removes the document. Thus a re-index of the same chunk text is a cache hit and not a re-encode.

Only `docbert clean` removes the orphan rows. An orphan row has a chunk ID with no document in `chunk_owners`.

docbert writes each value in this format:

- 4 bytes: the token count (`u32`, little-endian)
- 4 bytes: the embedding dimension (`u32`, little-endian)
- `token_count × dimension × 2` bytes: the `bf16` values (little-endian, 2 bytes each) in row-major sequence

docbert keeps components at `bf16` precision. The encoder trunk calculates in bf16 on CUDA. Thus an `f32` layout has more mantissa bits. These bits are noise below the model's precision floor. Downstream, the PLAID index re-quantizes each token to 2 bits for each dimension.

A bf16 component uses 2 bytes. An f32 component uses 4 bytes. `embeddings.db` uses much more space than the other files in the data directory. Thus the bf16 layout makes `embeddings.db` smaller.

Entries from docbert releases before 1.0 had row-major `f32` data. docbert does not decode that layout. Reads give an error that points to `docbert clean`. `docbert clean` removes the rows that do not decode. It also removes the per-document state. Thus the next `sync` re-embeds the applicable documents (see [Legacy formats from releases before 1.0](#legacy-formats-from-releases-before-10)).

The embeddings use the most space of all docbert data. Thus `embeddings.db` usually uses the most disk space in docbert.

## Tantivy storage details

The `tantivy/` directory keeps the lexical index entries for each prepared document.

The schema includes:

- The document ID string (kept)
- The numeric document ID (kept, fast)
- The collection (kept, fast)
- The relative path (kept)
- The title (kept, indexed with English stemming and 2x boost)
- The body (indexed with English stemming, **not kept**)
- The mtime (kept, fast)

Important notes:

- Tantivy keeps sufficient metadata to add information to retrieval results. Tantivy does not keep the body. The body bytes are only in the inverted index.
- Tantivy is **not** the only source of the titles, excerpts, or document contents that docbert gives.
- The web layer frequently reads the source file from disk again. It then makes the title and excerpt information again.

## Storage lifecycle by operation

## `docbert collection add`

`docbert collection add` writes to these:

- `config.db` `collections`

`docbert collection add` does not write to these:

- `embeddings.db`
- `tantivy/`
- collection snapshots

## `docbert sync`

`sync` can update these:

- `tantivy/`
- `embeddings.db`
- `plaid.idx` (docbert makes it again, or updates it for the touched documents)
- `config.db` `document_metadata`
- `config.db` `doc_chunks` and `chunk_owners`
- `config.db` `settings` through `embedding_model`
- `config.db` `collection_merkle_snapshots`

`sync` can remove these:

- deleted documents from Tantivy
- the chunk manifests of deleted documents from `config.db` `doc_chunks`, with the applicable `chunk_owners` entries
- deleted document metadata and document user metadata from `config.db`

`sync` does not remove embedding rows. They stay in `embeddings.db` as a content-addressed cache. Only `docbert clean` removes embedding rows.

## `docbert rebuild`

`rebuild` can remove these and make them again:

- `tantivy/`
- `plaid.idx`
- `config.db` `document_metadata`
- `config.db` `doc_chunks` and `chunk_owners`
- `config.db` `collection_merkle_snapshots`

`rebuild` also updates this:

- `config.db` `settings.embedding_model`

`rebuild` does not remove `embeddings.db`. It removes chunk manifests and document artifacts. It rewrites the Tantivy state and re-embeds the collections. Embedding rows are content-addressed. Thus a chunk with text that did not change maps to a row that docbert keeps. This is a cache hit, and docbert does not re-encode the chunk.

## `docbert reindex`

`reindex` rewrites only this:

- `plaid.idx`

`reindex` does not read or write the source files, embeddings, Tantivy, or other `config.db` tables. `reindex` retrains the PLAID index over all embeddings in `embeddings.db`. Use `reindex` only when the PLAID builder parameters changed (centroid count, codec bit-width, k-means iterations) and the embeddings stay correct.

## Web document upload

The upload writes these:

- the source file into the collection root
- a Tantivy entry for that document
- the embedding rows for the document's chunks, keyed by content-derived chunk ID (a rewrite of a row for the same chunk text keeps the same bytes)
- the document's chunk manifest in `doc_chunks`, with the applicable `chunk_owners` entries
- `document_metadata`
- the optional `doc_meta:{doc_id}` JSON metadata
- the updated collection snapshot

The upload does not update `plaid.idx`. Only `sync`, `rebuild`, or `reindex` make the PLAID index.

## Web document delete

The delete removes these:

- the source file from the collection root
- the document's chunk manifest from `doc_chunks`, with the applicable `chunk_owners` entries
- the document metadata
- the optional `doc_meta:{doc_id}` JSON metadata
- the Tantivy entry

Then the delete refreshes this:

- the collection snapshot

The delete does not touch `plaid.idx` or `embeddings.db`. The embedding rows stay as content-addressed cache entries. Searches ignore the chunks with an empty `chunk_owners` list. `docbert clean` removes the rows with no document in `chunk_owners`.

## Web conversation and settings APIs

These APIs write to:

- `conversations`
- `settings` (`llm_provider`, `llm_model`, `llm_api_key`, `llm_oauth:openai-codex`)

## Operational notes

## Rebuildability

The source files stay the primary source. Thus docbert can make most of these again from disk:

- The Tantivy entries
- The embeddings
- The metadata
- The collection snapshots

A rebuild does **not** make all the data in `config.db` again from the documents. docbert keeps some state only in `config.db`. This state includes the collections, contexts, conversations, and LLM settings.

## Model mismatch safety

docbert uses `embedding_model` in `config.db` as a safety check.

You can change to a different embedding model. If you then run `sync`, docbert does not mix the previous and new embeddings. docbert tells you to run `rebuild`.

## Schema compatibility

On open, `ConfigDb` makes sure that `config.db` contains the necessary named LMDB databases.

### Legacy formats from releases before 1.0

docbert 1.0 removed all migration paths for data from previous releases. `docbert clean` is the only command that repairs legacy data. It finds each legacy artifact and repairs only the necessary data. To see the changes first, use `--dry-run`. `docbert clean` repairs these legacy artifacts:

- **redb-format `config.db` or `embeddings.db`** (releases before 1.0): `ConfigDb::open` and `EmbeddingDb::open` read the first nine bytes and reject redb files. `clean` repairs them below the open layer. For a legacy `embeddings.db`, `clean` deletes it, `plaid.idx`, and the per-document state. The collections stay in the database, and the next `sync` re-embeds all the documents. For a legacy `config.db`, `clean` removes all data in the data directory. Then you must add the collections again.
- **`f32`-layout embedding rows** (pre-bf16 releases): reads reject them. `clean` removes the rows and removes the document state. Thus `sync` re-embeds the applicable documents. `clean` keeps and reuses the rows that are in the bf16 layout.
- **`plaid.idx` versions 1–2**: docbert rejects them. `docbert reindex` (or a full `rebuild`) makes the index again from the embeddings in `embeddings.db`.
- **conversation records from before 1.0** (payloads that do not use the rkyv format of docbert 1.0): reads ignore them. Lookups give nothing, and list requests ignore them with a warning. `docbert clean` does not touch them. To remove the record, delete the applicable conversation.

docbert does not make or touch `.redb-bak` files. If previous releases made `.redb-bak` files, delete them by hand.

### Hard schema breaks

The LMDB env can report a database type or layout that is not correct. This does not occur frequently. Then docbert gives a configuration error. docbert tells you to make a backup of `config.db` and then to remove it.

## Related references

- [`architecture.md`](./architecture.md)
- [`pipeline.md`](./pipeline.md)
- [`chat-and-conversations.md`](./chat-and-conversations.md)
- [`web-api.md`](./web-api.md)
