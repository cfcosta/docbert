# Pipeline

## Overview

docbert's pipeline has two related halves:

- **indexing**, which discovers files, prepares searchable content, updates Tantivy, stores embeddings, and records metadata
- **retrieval**, which resolves queries into ranked results and then enriches them with titles, excerpts, and document reads from disk

The current implementation is shared across the CLI, the web runtime, and the MCP runtime, but each surface exposes a slightly different request shape and output format.

## Source material and indexed state

A registered collection is just a named filesystem root stored in `config.db`.

The pipeline then works across four kinds of state:

- **source files on disk** inside collection roots
- **`tantivy/`** for lexical search
- **`embeddings.db`** for ColBERT embeddings
- **`config.db`** for collection registration, document metadata, settings, and collection Merkle snapshots

One important architectural detail: docbert does **not** treat the index as the only source of truth for reads. Search result enrichment and document retrieval still read current file contents from disk.

## Indexing pipeline

## When indexing runs

Collection registration does not index anything by itself.

Indexing happens through:

- `docbert sync`
- `docbert rebuild`
- web document ingestion and deletion via `docbert web`

The CLI and web runtime use the same shared discovery, preparation, metadata, and snapshot primitives.

## Stage 1: collection discovery

Discovery starts in `docbert_core::walker::discover_files`.

Current behavior:

- recursively walks the collection root
- skips hidden files and hidden directories
- includes only supported extensions:
  - `.md`
  - `.txt`
  - `.pdf`
- resolves file modification time into `DiscoveredFile.mtime`
- sorts the final file list by relative path
- supports file symlinks when they resolve to supported files
- skips broken symlinks and avoids directory-cycle problems

### Git ignore behavior

Discovery is now Git-aware, but only when the **collection root itself is a Git repo**.

If `collection_root/.git` exists, discovery respects:

- `.gitignore`
- nested Git ignore files
- `.git/info/exclude`
- Git global excludes

If the collection root is **not** a Git repo, a stray `.gitignore` file does **not** affect indexing.

That distinction is intentional and is part of the current indexing contract.

## Stage 2: deciding what work to do

The exact planning step depends on whether docbert is running `sync`, `rebuild`, or a web mutation.

### `docbert sync`

`sync` performs incremental selection in `crates/docbert/src/indexing_workflow.rs`.

The flow is:

1. discover files with `walker::discover_files`
2. compute a fresh collection Merkle snapshot from the discovered files
3. compare that snapshot with the previously stored snapshot from `config.db`
4. classify paths into:
   - new files
   - changed files
   - deleted files
5. convert deleted paths into deterministic document IDs
6. process only new and changed files, and remove state for deleted files

This is an important current behavior change: **incremental sync is snapshot-based, not just mtime-based**.

`mtime` is still stored in `DocumentMetadata`, but change selection is driven by the Merkle snapshot diff.

### `docbert rebuild`

`rebuild` does not do incremental selection. It:

1. discovers the current supported files in the target collection(s)
2. clears prior indexed state for those collections, depending on flags
3. reloads the current files
4. re-indexes and/or re-embeds them
5. writes a fresh collection snapshot only after success

Current rebuild modes:

- default rebuild: re-index and re-embed
- `--index-only`: clear and rebuild Tantivy plus document metadata, but do not remove or regenerate embeddings
- `--embeddings-only`: clear and rebuild embeddings plus document metadata, but do not rewrite Tantivy entries

If both `--index-only` and `--embeddings-only` are set, docbert skips document loading and only refreshes metadata and snapshots for the discovered files.

### Web document ingestion and deletion

The web runtime uses narrower per-document mutation paths rather than a full collection walk for every change.

For `POST /v1/documents`:

1. validate the collection exists
2. write the uploaded source file into the collection root
3. load and prepare that one document from disk
4. compute its embeddings
5. update Tantivy, embeddings, metadata, and user metadata
6. refresh the collection snapshot

For `DELETE /v1/documents/{collection}/{path}`:

1. confirm the document exists in stored metadata
2. remove the source file from disk
3. remove Tantivy state
4. remove the entire embedding family for the document
5. remove document metadata and user metadata
6. refresh the collection snapshot

The snapshot update is intentionally kept behind successful mutation work.

## Stage 3: loading and preparing documents

File loading happens through `docbert_core::ingestion::load_documents`, which parallelizes reads and keeps failures separate from successfully loaded files.

Each successful file becomes a `SearchDocument` with:

- stable document ID derived from collection + relative path
- relative path
- title
- searchable body
- optional raw content / metadata, depending on source
- mtime

### Markdown and text preparation

For markdown and text content, docbert currently:

- strips YAML frontmatter from the searchable body
- derives the title from the first Markdown `# ` heading when present
- otherwise falls back to the filename stem

### PDF preparation

PDF files are now part of the supported indexing pipeline.

Current behavior:

- PDF bytes are converted to markdown when possible
- if markdown conversion yields nothing useful, docbert falls back to extracted text
- the resulting text is then treated like other document content for title extraction and embedding

### Load failures

Unreadable or unconvertible files do not abort discovery. They are tracked as load failures and logged, while successfully loaded files continue through indexing and embedding.

## Stage 4: lexical indexing with Tantivy

Lexical indexing writes the prepared document body into `SearchIndex`.

For each prepared document, docbert stores:

- full document ID string
- numeric document ID
- collection name
- relative path
- title
- searchable body
- mtime

A batch is committed after the prepared documents are added.

During sync and rebuild:

- deleted documents are removed from Tantivy before new work is committed
- rebuild can clear an entire collection from Tantivy before re-adding documents

During web ingestion:

- a single document write updates the index in the same mutation flow as metadata and embeddings

## Stage 5: chunking and embedding

Embedding uses ColBERT-style document chunks derived from the prepared searchable body.

Current chunking behavior:

- chunking operates on characters, not tokens
- default chunk size is based on docbert's default document length budget (`519` tokens approximated as `519 * 4` characters)
- overlap defaults to `0`
- if the selected model path is local and has `config_sentence_transformers.json`, docbert reads `document_length` and derives the chunk size from it

Chunk IDs are document-family aware:

- chunk `0` keeps the base document ID
- later chunks get chunk-specific numeric IDs derived from the base document ID

That means one source document can correspond to multiple embedding rows in `embeddings.db`.

### Empty semantic bodies

If a document's searchable body is empty after preparation, chunk generation returns no embedding chunks.

This matters for frontmatter-only documents: they may still have metadata and a lexical entry, but they do not contribute semantic embeddings.

### Embedding storage

CLI indexing uses batched embedding storage.

The flow is:

1. collect chunk text for all prepared documents
2. encode chunks with the current model
3. store chunk embeddings in `embeddings.db`
4. keep metadata in `config.db` in sync with successfully processed files

Web ingestion follows the same conceptual flow, but at one-document granularity.

When an existing document is replaced through the web runtime, docbert also removes stale chunk embeddings that no longer belong to the current chunk set.

## Stage 6: metadata and snapshot persistence

After successful indexing/embedding work, docbert persists metadata and collection state.

### Document metadata

`config.db` stores `DocumentMetadata` keyed by numeric document ID:

- collection
- relative path
- mtime

For web uploads, optional user metadata is also stored separately.

### Collection snapshots

Collection Merkle snapshots record the discovered file set for each collection and are used to drive later sync selection.

Current snapshot behavior:

- `sync` computes the current snapshot up front but only stores it after all planned work succeeds
- `rebuild` computes and stores a fresh snapshot only after the rebuild succeeds
- web ingestion and deletion refresh the snapshot only after the mutation succeeds end to end

If the indexing or mutation step fails, docbert keeps the previous snapshot.

## Retrieval pipeline

## Search modes

docbert currently exposes two main search modes:

- **hybrid**: BM25 first-stage retrieval followed by ColBERT reranking
- **semantic**: ColBERT-only retrieval over the stored document set

Different surfaces choose different defaults:

- CLI `search` uses hybrid search
- CLI `ssearch` uses semantic-only search
- the web `/v1/search` API defaults to `semantic` unless the caller passes `"mode": "hybrid"`

## Hybrid search flow

Hybrid search is implemented in `docbert_core::search::execute_search`.

### Step 1: BM25 candidate generation

The first stage queries Tantivy.

Current behavior:

- fetch up to `1000` candidates
- optionally filter to one collection
- use fuzzy matching by default
- allow a CLI-only `--no-fuzzy` path that uses plain BM25 retrieval instead
- allow a CLI-only `--bm25-only` path that skips semantic reranking entirely

If BM25 returns no candidates, the pipeline ends there.

### Step 2: query embedding

If reranking is enabled, docbert encodes the query with the active ColBERT model via `model.encode_query(...)`.

### Step 3: ColBERT reranking

docbert then reranks the BM25 candidates using stored embeddings from `embeddings.db`.

Conceptually, the reranker:

- loads embeddings for the candidate document IDs
- scores them with ColBERT MaxSim-style similarity
- returns ranked document IDs with semantic scores

The final hybrid result list keeps the candidate identity and path information from BM25, but uses the reranked semantic score.

### Step 4: filtering and limiting

After reranking, docbert:

- drops results below `min_score`
- applies the requested count unless `--all` is set
- assigns final 1-based ranks

## Semantic-only search flow

Semantic-only search is implemented in `docbert_core::search::execute_semantic_search`.

Current behavior:

1. load all stored document metadata from `config.db`
2. optionally filter to one collection
3. discard documents whose current on-disk content has no semantic body after frontmatter stripping
4. encode the query
5. rerank across all remaining document IDs using stored embeddings
6. filter by `min_score`
7. limit to `count` unless `all` is set
8. populate titles from current file contents on disk

This is broader and typically more expensive than hybrid search because it is not narrowed by a BM25 candidate stage first.

## Result enrichment and document reads

A retrieval result is not the final user-visible payload yet.

### Titles

Search result titles are refreshed from current on-disk content when possible.

That means titles can reflect the source file currently on disk even if an older fallback title existed in the first-stage index result.

### Excerpts

The web search API adds excerpts after search ranking.

For each result, the route handler:

1. resolves the collection-relative path back to a source file
2. reads the current content from disk
3. recomputes the title from disk content
4. extracts up to three excerpts with line ranges based on the query text

If the literal query text does not appear, excerpt generation can fall back to the first lines of the document.

### Document reads

`GET /v1/documents/{collection}/{path}` also reads directly from the source file on disk and derives the returned title from the current content.

This is why the filesystem remains part of the live retrieval path, not just the indexing path.

## End-to-end flow by surface

## CLI sync/rebuild

The CLI collection pipeline is:

1. resolve target collection(s)
2. discover supported files
3. plan sync/rebuild work
4. load and prepare successful files
5. update Tantivy
6. update embeddings
7. persist metadata
8. persist the collection snapshot

## Web search and document reads

The web retrieval pipeline is:

1. parse the JSON request
2. choose `semantic` or `hybrid`
3. open `config.db` and `embeddings.db`
4. run the shared search pipeline
5. enrich results from current files on disk with titles, metadata, and excerpts
6. return JSON

## Web ingest/delete

The web mutation pipeline is:

1. mutate the source file on disk
2. update indexed state and embeddings
3. update metadata
4. refresh the collection snapshot
5. return JSON or status

## MCP tools

The MCP runtime uses the same underlying search and retrieval primitives, but wraps them as MCP tools/resources instead of HTTP or terminal output.

See [`mcp.md`](./mcp.md) for the concrete MCP response shapes.

## Practical implications

A few pipeline details matter when operating docbert in practice:

- adding a collection does not index it; run `sync` or `rebuild`
- Git ignore rules only matter when the collection root is itself a Git repo
- PDFs are part of the current discovery and preparation pipeline
- `sync` uses collection snapshots to detect new/changed/deleted files
- semantic search depends on stored embeddings and current readable file content
- search results and document reads can reflect current on-disk content even after indexing, because titles and excerpts are refreshed from disk at retrieval time
- changing embedding models requires a rebuild before sync will proceed safely

## Related references

- [`architecture.md`](./architecture.md)
- [`storage.md`](./storage.md)
- [`web-api.md`](./web-api.md)
- [`mcp.md`](./mcp.md)
