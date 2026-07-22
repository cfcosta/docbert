# Pipeline

## What the pipeline does

docbert's pipeline has two related parts:

- **indexing** finds the files, prepares the searchable body, updates Tantivy, writes the embeddings, and records the metadata.
- **retrieval** changes a query into a sequence of results. Then it adds the titles and the excerpts, and reads the documents from disk.

The CLI, the web runtime, and the MCP runtime use the same pipeline. But each interface has small differences in the request format and the output format.

## Source files and indexed data

A collection is a filesystem root with a name. docbert keeps this data in `config.db`.

The pipeline uses these types of data:

- **Source files on disk** in the collection roots
- **`tantivy/`** for lexical search
- **`embeddings.db`** for ColBERT embeddings
- **`plaid.idx`** for the PLAID multi-vector index (the storage for the semantic part)
- **`config.db`** for the collection data, document metadata, chunk byte offsets, configuration, and collection Merkle snapshots.

The source files on disk are part of the read path. docbert reads the file data from disk when it adds data to the results. docbert also reads the file data when it gets a document. docbert does not use the index as the only data source.

## Indexing pipeline

## When indexing occurs

When you add a collection, docbert does not index it.

Indexing occurs through these operations:

- `docbert sync`
- `docbert rebuild`
- `docbert reindex`
- Document add and delete operations through `docbert web`.

`docbert reindex` is PLAID-only. It rebuilds the semantic index from the embeddings and does not encode again.

The CLI and web runtime use the same operations to find files, prepare documents, write metadata, and write snapshots.

## Stage 1: collection discovery

Discovery starts in `docbert_core::walker::discover_files`. This function does these steps:

- It examines the collection root and all subdirectories.
- It ignores the files and directories with names that start with `.`.
- It includes only these file extensions: `.md`, `.txt`, and `.pdf`.
- It puts the file modification time into `DiscoveredFile.mtime`.
- It puts the file list in sequence by relative path.
- It uses file symlinks when they point to files with these extensions.
- It ignores the symlinks that point to a missing file. It also prevents directory-cycle problems.

### Git ignore rules

Discovery uses Git rules, but only when the collection root is a Git repo.

If there is a `collection_root/.git` directory, discovery obeys these rules:

- `.gitignore`
- Git ignore files in subdirectories
- `.git/info/exclude`
- Git global excludes.

If the collection root is not a Git repo, a `.gitignore` file has no effect on indexing.

docbert obeys these rules in all indexing operations.

## Stage 2: which work to do

The step to select the work is different for `sync`, `rebuild`, and web document changes.

### `docbert sync`

`sync` selects the changed files in `crates/docbert/src/indexing.rs`. `sync` does these steps:

1. It finds the files with `walker::discover_files`.
2. It calculates a new collection Merkle snapshot from these files.
3. It compares this snapshot with the previous snapshot in `config.db`.
4. It divides the paths into three groups: new files, changed files, and deleted files.
5. It changes the deleted paths into stable document IDs.
6. It indexes and embeds only the new and changed files, and removes the data for the deleted files.

The Merkle snapshot diff controls which files docbert selects. docbert keeps `mtime` in `DocumentMetadata` as metadata for each document. docbert does not use `mtime` to find the changed files.

### `docbert rebuild`

`rebuild` does not select only the changes. `rebuild` does these steps:

1. It finds the files that docbert can index in the specified collection(s).
2. It removes the previous indexed data for these collections (the flags control this).
3. It reads the files again.
4. It re-indexes the files, re-embeds them, or does the two operations.
5. It writes a new collection snapshot only after docbert completes the rebuild.

The rebuild modes are:

- The default rebuild re-indexes and re-embeds the files.
- `--index-only` removes and rebuilds Tantivy and the document metadata. It does not remove the embeddings and does not make them again.
- `--embeddings-only` removes and rebuilds the embeddings and the document metadata. It does not write the Tantivy entries again.

If you set `--index-only` and `--embeddings-only` together, docbert does not read the documents. It updates only the metadata and the snapshots for these files.

### Web document add and delete

The web runtime changes one document at a time. It does not examine the full collection for each change.

For `POST /v1/documents`, docbert does these steps:

1. It makes sure that the collection is available.
2. It writes the uploaded source file into the collection root.
3. It reads and prepares that one document from disk.
4. It calculates its embeddings.
5. It updates Tantivy, the embeddings, the metadata, and the user metadata.
6. It updates the collection snapshot.

For `DELETE /v1/documents/{collection}/{path}`, docbert does these steps:

1. It makes sure that the document is in the `config.db` metadata.
2. It removes the source file from disk.
3. It removes the Tantivy data.
4. It removes all embeddings for the document.
5. It removes the document metadata and the user metadata.
6. It updates the collection snapshot.

The snapshot update occurs only after docbert completes the change work.

## Stage 3: how docbert reads and prepares documents

docbert reads files through `docbert_core::ingestion::load_documents`. This function reads the files at the same time. It keeps the files with load errors and the files that it reads correctly in different groups.

Each file that docbert reads correctly becomes a `SearchDocument` with these parts:

- A stable document ID from the collection and relative path
- The relative path
- The title
- The searchable body
- Optional raw data or metadata, for some sources
- The `mtime`.

### How docbert prepares Markdown and text

For markdown and text data, docbert does these steps:

- It removes the YAML frontmatter from the searchable body.
- It gets the title from the first Markdown `# ` heading, if the file has one.
- It uses the filename without the extension if the file has no `# ` heading.

### How docbert prepares PDF files

docbert can index PDF files. The PDF steps are:

- docbert changes the PDF bytes to markdown when possible.
- If the change to markdown gives no text, docbert uses the raw text from the PDF.
- docbert then uses this text for the title and the embeddings, the same as the other document data.

### Load errors

Files that docbert cannot read or prepare do not stop discovery. docbert records these files as load errors. The files that docbert reads correctly continue through indexing and embedding.

## Stage 4: lexical indexing with Tantivy

Lexical indexing writes the prepared document body into `SearchIndex`.

For each prepared document, docbert keeps these fields:

- The full string of the document ID
- The numeric document ID
- The collection name
- The relative path
- The title
- The searchable body
- The `mtime`.

docbert commits a batch after it adds the prepared documents.

During `sync` and `rebuild`:

- docbert removes the deleted documents from Tantivy before it commits new work.
- `rebuild` can remove a full collection from Tantivy before it adds documents again.

During a web add operation:

- One document write updates the index in the same change operation as the metadata and the embeddings.

## Stage 5: chunking and embedding

docbert embeds ColBERT-style document chunks. docbert makes these chunks from the prepared searchable body.

docbert makes the chunks with these parameters:

- The chunk size is in characters, not in tokens.
- docbert calculates the default chunk size from the model's `document_length` in `config_sentence_transformers.json` (approximately `document_length * 4` characters). docbert uses `300` tokens when `document_length` is not available.
- The overlap default is `0`.
- If the selected model path is local and has `config_sentence_transformers.json`, docbert reads `document_length` and calculates the chunk size from it.

docbert makes the chunk IDs from the base document ID:

- Chunk `0` keeps the base document ID.
- The subsequent chunks get chunk-specific numeric IDs from the base document ID.

As a result, one source document can have more than one embedding row in `embeddings.db`.

### Chunk byte offsets

docbert writes the byte offset and length of each chunk in the source document to the `doc_chunks` manifest in `config.db`. docbert keeps this manifest with the embeddings. The manifest holds one entry for each document.

The search code gets these offsets through `ConfigDb::get_chunk_offset_for_doc`. This function shows the correct byte range of the best chunk to the callers (see `FinalResult.best_chunk_doc_id`).

### Empty semantic bodies

docbert makes no embedding chunks if the prepared searchable body is empty.

This property is important for frontmatter-only documents. These documents can have metadata and a lexical entry, but they give no semantic embeddings.

### Embedding storage

CLI indexing writes the embeddings in batches. CLI indexing does these steps:

1. It collects the chunk text for all prepared documents.
2. It encodes the chunks with the active model.
3. It writes the chunk embeddings to `embeddings.db`.
4. It updates the `config.db` metadata for each file that it indexes correctly.

Web add operations use the same steps, but for one document at a time.

When docbert replaces a document through the web runtime, it also removes the remaining chunk embeddings from the previous chunks.

## Stage 6: metadata and snapshot storage

After docbert indexes and embeds the files correctly, docbert writes the metadata and the collection data.

### Document metadata

`config.db` keeps `DocumentMetadata` with a key of the numeric document ID:

- The collection
- The relative path
- The `mtime`.

For web uploads, docbert also keeps optional user metadata isolated from `DocumentMetadata`.

### Collection snapshots

Collection Merkle snapshots record the files in each collection. docbert uses them to select files in a subsequent `sync`.

The snapshots have these properties:

- `sync` calculates the new snapshot first, but keeps it only after docbert completes all the work.
- `rebuild` calculates and keeps a new snapshot only after docbert completes the rebuild.
- Web add and delete operations update the snapshot only after docbert completes the full change.

If docbert does not complete the indexing or the change step, docbert keeps the previous snapshot.

## Retrieval pipeline

## Search modes

docbert has two primary search modes:

- **hybrid**: a BM25 part on Tantivy and a ColBERT/PLAID part, with Reciprocal Rank Fusion
- **semantic**: only ColBERT/PLAID retrieval through all the documents.

The two search modes must have a PLAID index. If the index is missing, the search gives `Error::PlaidIndexMissing`. The web runtime shows this as `503 Service Unavailable`. The commands `docbert sync`, `docbert rebuild`, and `docbert reindex` make the index.

The interfaces select different defaults:

- CLI `search` uses hybrid search.
- CLI `ssearch` uses semantic-only search.
- The web `/v1/search` API uses `semantic` as the default. It uses hybrid only when the caller sends `"mode": "hybrid"`.

## Hybrid search sequence

docbert does hybrid search in `docbert_core::search::run`.

### Step 1: BM25 part

The BM25 part queries Tantivy. It gives a maximum of `100` candidates (the `RRF_CANDIDATE_LIMIT` constant).

The BM25 part has these properties:

- It can keep only the results from one collection.
- It uses fuzzy matching as the default.
- A CLI-only `--no-fuzzy` path uses BM25 retrieval without fuzzy matching.
- A CLI-only `--bm25-only` path does not use the semantic part. It gives the BM25 results directly, with a `min_score` filter.

### Step 2: semantic part

The semantic part uses the same PLAID query pipeline as `search::semantic`:

1. It reads the PLAID index from `plaid.idx` (this gives `PlaidIndexMissing` if the index is missing).
2. It reads the document metadata from `config.db`. It can keep only the metadata for the specified collection.
3. It encodes the query with the active ColBERT model through `model.encode_query(...)`.
4. It gets more candidates than the count (`max(count * 8, 64)`) from `plaid::search`.
5. It keeps one entry for each base document, with the id of the best chunk.
6. It keeps a maximum of `100` candidates by score.

### Step 3: Reciprocal Rank Fusion

Reciprocal Rank Fusion puts the two candidate lists together:

- Each document adds `1 / (k + rank_i)` from each list that it is in (`k = 60`, the `RRF_K` constant).
- A document that is not in a list does not add a score from that list.
- docbert puts the results in sequence by the RRF score, from high to low.

The fusion metadata uses the BM25 part when a document is in the two lists. Thus the titles from Tantivy stay. For semantic-only entries, docbert reads the titles again from disk.

### Step 4: result limits

After fusion, docbert does these steps:

- It uses the specified number of results unless you set `--all`.
- It gives each result a 1-based rank.

docbert ignores `min_score` in RRF mode, because the RRF scores do not use the BM25 score range. `min_score` applies in `--bm25-only` mode and in semantic-only search. In semantic-only search, docbert applies `min_score` to the PLAID MaxSim score.

## Semantic-only search sequence

docbert does semantic-only search in `docbert_core::search::semantic`. docbert does these steps:

1. It reads the PLAID index (this gives `PlaidIndexMissing` if the index is missing).
2. It reads all document metadata from `config.db`.
3. It can keep only one collection.
4. It encodes the query with the active ColBERT model.
5. It gets more candidates than the count (`max(count * 8, 64)`) from `plaid::search`.
6. It keeps one entry for each base document, with the id and score of the best chunk.
7. It removes the results below the `min_score` value.
8. It keeps a maximum of `count` results, unless you set `all`.
9. It reads the titles from the file data on disk.

## Result data and document reads

A retrieval result is not the last data that docbert gives to the user.

### Titles

docbert reads the search result titles again from the on-disk data when possible.

As a result, the titles can show the on-disk file. docbert uses the on-disk title, not the previous fallback title from the first-stage index result.

### Excerpts

The web search API adds excerpts after it puts the results in sequence.

For each result, the route handler does these steps:

1. It finds the source file from the collection-relative path.
2. It reads the data from disk.
3. It gets the title again from the disk data.
4. It gets a maximum of three excerpts, with line ranges, from the query text.

If the document does not contain the query text, docbert can use the first lines of the document.

### Document reads

`GET /v1/documents/{collection}/{path}` also reads directly from the source file on disk. It gets the title from the disk data.

Thus the filesystem is part of the retrieval path and the indexing path.

## End-to-end steps for each interface

## CLI sync/rebuild

For the CLI collection pipeline, docbert does these steps:

1. It finds the specified collection(s).
2. It finds the files that it can index.
3. It selects the `sync` or `rebuild` work.
4. It reads and prepares the correct files.
5. It updates Tantivy.
6. It updates the embeddings.
7. It writes the metadata.
8. It writes the collection snapshot.

## Web search and document reads

For the web retrieval pipeline, docbert does these steps:

1. It parses the JSON request.
2. It selects `semantic` or `hybrid`.
3. It opens `config.db` and `embeddings.db`.
4. It uses the same search pipeline.
5. It reads the files on disk and adds titles, metadata, and excerpts to the results.
6. It gives JSON.

## Web add/delete

For the web change pipeline, docbert does these steps:

1. It changes the source file on disk.
2. It updates the indexed data and the embeddings.
3. It updates the metadata.
4. It updates the collection snapshot.
5. It gives JSON or a status.

## MCP tools

The MCP runtime uses the same search and retrieval operations. But it gives them as MCP tools and resources, not as HTTP or terminal output.

Refer to [`mcp.md`](./mcp.md) for the MCP response formats.

## Important properties

These pipeline properties are important when you operate docbert:

- When you add a collection, docbert does not index it. You must use `sync` or `rebuild` to index it.
- Git ignore rules are important only when the collection root is a Git repo.
- docbert finds and prepares PDFs in the pipeline.
- `sync` uses collection snapshots to find new, changed, and deleted files.
- Hybrid search and semantic search must have a PLAID index. On a new data directory, the search gives `PlaidIndexMissing` until you use `docbert sync` (or `docbert rebuild` or `docbert reindex`).
- Search results and document reads can show the on-disk data after indexing. docbert reads the titles and excerpts again from disk at retrieval time.
- After you change the embedding model, you must do a rebuild before `sync` can continue safely.

## Related documents

- [`architecture.md`](./architecture.md)
- [`storage.md`](./storage.md)
- [`web-api.md`](./web-api.md)
- [`mcp.md`](./mcp.md)
