# docserver

## Overview

docserver is an HTTP server that wraps `docbert-core` to expose document ingestion and search over a REST API. It is a separate binary crate in the workspace, intended for services that need to index and query documents programmatically without the CLI.

Storage is configured through environment variables. The server manages one data directory containing all collections, indexes, and embeddings.

Document ingestion accepts Markdown files for now. The API includes a `content_type` field so new formats can be added later without breaking existing clients.

## Configuration

All configuration is through environment variables.

| Variable             | Required | Default                  | Description                                                  |
| -------------------- | -------- | ------------------------ | ------------------------------------------------------------ |
| `DOCSERVER_DATA_DIR` | yes      | —                        | Root storage path for config.db, embeddings.db, and tantivy/ |
| `DOCSERVER_HOST`     | no       | `127.0.0.1`              | Bind address                                                 |
| `DOCSERVER_PORT`     | no       | `3030`                   | Bind port                                                    |
| `DOCSERVER_MODEL`    | no       | `lightonai/ColBERT-Zero` | ColBERT model ID or local path                               |
| `DOCSERVER_LOG`      | no       | `warn`                   | Log level filter (`trace`, `debug`, `info`, `warn`, `error`) |

The server creates `DOCSERVER_DATA_DIR` and its subdirectories on startup if they do not exist.

## API

All request and response bodies are JSON. The API is versioned under `/v1/`.

### Collections

#### Create a collection

```
POST /v1/collections
```

```json
{ "name": "notes" }
```

Returns `201 Created`:

```json
{ "name": "notes" }
```

Returns `409 Conflict` if the collection already exists.

#### List collections

```
GET /v1/collections
```

Returns `200 OK`:

```json
[{ "name": "notes" }, { "name": "docs" }]
```

#### Delete a collection

```
DELETE /v1/collections/:name
```

Removes the collection definition, all Tantivy entries for documents in that collection, and all embeddings for those documents.

Returns `204 No Content` on success, `404 Not Found` if the collection does not exist.

### Documents

#### Ingest documents

```
POST /v1/documents
```

```json
{
  "collection": "notes",
  "documents": [
    {
      "path": "meeting-notes/2025-03-26.md",
      "content": "# Meeting Notes\n\nDiscussed project timeline.",
      "content_type": "text/markdown",
      "metadata": {
        "author": "alice",
        "tags": ["meeting", "q1-planning"],
        "source": "confluence"
      }
    }
  ]
}
```

Fields:

- `collection` — target collection name; must already exist
- `documents` — array of documents to ingest
  - `path` — logical path within the collection, used together with the collection name to compute a stable document ID; not a filesystem path
  - `content` — full document text
  - `content_type` — MIME type; only `text/markdown` is accepted initially
  - `metadata` (optional) — arbitrary JSON object attached to the document; stored alongside the document and returned in search results and document retrieval, but not indexed for search

The server processes each document through the full pipeline:

1. Validate `content_type`
2. Strip YAML frontmatter, extract title from first `# ` heading
3. Add to the Tantivy index
4. Chunk if needed, compute ColBERT embeddings, store in the embedding database
5. Store document metadata (including the `metadata` JSON object) for future lookups and retrieval

If a document with the same `(collection, path)` already exists, it is replaced.

Returns `200 OK`:

```json
{
  "ingested": 1,
  "documents": [
    {
      "doc_id": "#a1b2c3",
      "path": "meeting-notes/2025-03-26.md",
      "title": "Meeting Notes",
      "metadata": {
        "author": "alice",
        "tags": ["meeting", "q1-planning"],
        "source": "confluence"
      }
    }
  ]
}
```

Returns `400 Bad Request` if the collection does not exist, a content type is unsupported, or required fields are missing.

#### Get a document

```
GET /v1/documents/:collection/:path
```

The `:path` segment is the URL-encoded logical path (e.g., `meeting-notes%2F2025-03-26.md`).

Returns `200 OK`:

```json
{
  "doc_id": "#a1b2c3",
  "collection": "notes",
  "path": "meeting-notes/2025-03-26.md",
  "title": "Meeting Notes",
  "content": "# Meeting Notes\n\nDiscussed project timeline.",
  "metadata": {
    "author": "alice",
    "tags": ["meeting", "q1-planning"],
    "source": "confluence"
  }
}
```

Returns `404 Not Found` if the document does not exist.

#### Delete a document

```
DELETE /v1/documents/:collection/:path
```

Removes the document from the Tantivy index, embedding database, and metadata store.

Returns `204 No Content` on success, `404 Not Found` if the document does not exist.

### Search

```
POST /v1/search
```

```json
{
  "query": "project timeline",
  "mode": "semantic",
  "collection": "notes",
  "count": 10,
  "min_score": 0.0
}
```

Fields:

- `query` (required) — search query string
- `mode` (optional, default `"semantic"`) — search mode:
  - `"semantic"` — ColBERT-only search; scores every stored embedding with MaxSim. Best for meaning-based queries where wording may differ. Slower on large corpora since it scans all embeddings.
  - `"hybrid"` — BM25 first-pass with fuzzy matching, then ColBERT reranking of the top candidates. Faster on large corpora and good when the query shares keywords with the target documents.
- `collection` (optional) — restrict search to one collection
- `count` (optional, default 10) — number of results to return
- `min_score` (optional, default 0.0) — minimum score threshold

Returns `200 OK`:

```json
{
  "query": "project timeline",
  "mode": "semantic",
  "result_count": 1,
  "results": [
    {
      "rank": 1,
      "score": 0.847,
      "doc_id": "#a1b2c3",
      "collection": "notes",
      "path": "project-ideas.md",
      "title": "Project Timeline and Milestones",
      "metadata": {
        "author": "alice"
      }
    }
  ]
}
```

`"semantic"` mode maps to `docbert_core::search::execute_semantic_search`. `"hybrid"` mode maps to `docbert_core::search::execute_search` with fuzzy matching enabled.

## Content type extensibility

The `content_type` field on ingestion requests is the extension point for supporting new document formats. Adding a new format requires:

1. Register the MIME type in a supported-types list
2. Implement a content processor that takes raw content and produces:
   - Searchable plain text (for Tantivy indexing and ColBERT embedding)
   - A document title
3. The rest of the pipeline is format-agnostic

For Markdown, the processor reuses `docbert-core`'s existing logic: `text_util::strip_yaml_frontmatter` and `ingestion::extract_title`.

## Error responses

All errors return JSON with a human-readable message and a machine-readable code:

```json
{
  "error": "collection not found: nonexistent",
  "code": "NOT_FOUND"
}
```

| HTTP status | Code             | When                                                |
| ----------- | ---------------- | --------------------------------------------------- |
| 400         | `BAD_REQUEST`    | Missing required fields, unsupported content type   |
| 404         | `NOT_FOUND`      | Collection or document does not exist               |
| 409         | `CONFLICT`       | Collection already exists                           |
| 500         | `INTERNAL_ERROR` | Unexpected failure (index error, model error, etc.) |

## Architecture

```
docserver binary
  HTTP layer (axum)
    routes/collections.rs  — POST/GET/DELETE /v1/collections
    routes/documents.rs    — POST/GET/DELETE /v1/documents
    routes/search.rs       — POST /v1/search (semantic and hybrid modes)
    error.rs               — error type to JSON response mapping

  App state (shared via Arc)
    DataDir
    ConfigDb
    SearchIndex
    EmbeddingDb
    ModelManager (behind Mutex for lazy loading)

  content.rs — content type validation and processing

  depends on: docbert-core
```

Handlers call `docbert-core` functions directly. The `SearchIndex` writer is behind a `Mutex` since Tantivy allows only one writer at a time.

## Crate structure

```
crates/docserver/
  Cargo.toml
  src/
    main.rs          — env var parsing, state initialization, server startup
    state.rs         — AppState struct holding docbert-core resources
    routes/
      mod.rs         — router assembly
      collections.rs — collection CRUD handlers
      documents.rs   — document ingestion, retrieval, deletion
      search.rs      — hybrid and semantic search handlers
    error.rs         — ApiError type and IntoResponse impl
    content.rs       — content type registry and text extraction
```

Dependencies:

- `docbert-core` — search, indexing, embeddings, storage
- `axum` — HTTP framework
- `tokio` — async runtime
- `serde`, `serde_json` — request/response serialization
- `tracing`, `tracing-subscriber` — structured logging

## Concurrency

- The `SearchIndex` reader supports concurrent queries. The writer is behind a `Mutex` and acquired only during ingestion.
- `ConfigDb` and `EmbeddingDb` use redb, which supports concurrent readers and a single writer through MVCC.
- `ModelManager` is behind a `Mutex` because model loading mutates internal state.
- Search is fully read-only and can serve concurrent requests.
- Ingestion acquires the writer lock, so concurrent ingestion requests are serialized.
