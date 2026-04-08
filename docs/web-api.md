# Web API reference

This page documents the HTTP API implemented by `docbert web`.

It covers the current server behavior in `crates/docbert/src/web/routes/*`. When the browser client contains helper functions for routes the server does not implement, this page follows the server, not the client.

## Base path

All API routes live under:

```text
/v1
```

The web process also serves the browser UI, but UI routes are not part of this API reference.

## Format and conventions

- Request and response bodies are JSON unless otherwise noted.
- Successful `DELETE` responses typically return `204 No Content`.
- Most unexpected storage or indexing failures return `500 Internal Server Error`.
- Route parameters are path-decoded by Axum after URL encoding.
- The UI client sends `Content-Type: application/json` for API requests.

## Route summary

| Method   | Route                                        | Purpose                                                                  |
| -------- | -------------------------------------------- | ------------------------------------------------------------------------ |
| `GET`    | `/v1/collections`                            | List registered collection names.                                        |
| `GET`    | `/v1/conversations`                          | List saved conversations.                                                |
| `POST`   | `/v1/conversations`                          | Create a conversation.                                                   |
| `GET`    | `/v1/conversations/{id}`                     | Get one conversation.                                                    |
| `PUT`    | `/v1/conversations/{id}`                     | Replace one conversation.                                                |
| `DELETE` | `/v1/conversations/{id}`                     | Delete one conversation.                                                 |
| `POST`   | `/v1/documents`                              | Upload and ingest Markdown or PDF documents into an existing collection. |
| `GET`    | `/v1/collections/{name}/documents`           | List documents in one collection.                                        |
| `GET`    | `/v1/documents/{collection}/{*path}`         | Read one document and its stored metadata.                               |
| `DELETE` | `/v1/documents/{collection}/{*path}`         | Delete one document from disk and from indexed state.                    |
| `POST`   | `/v1/search`                                 | Run semantic or hybrid search.                                           |
| `GET`    | `/v1/settings/llm`                           | Read persisted LLM settings, including effective auth state.             |
| `PUT`    | `/v1/settings/llm`                           | Update persisted LLM settings.                                           |
| `POST`   | `/v1/settings/llm/oauth/openai-codex/start`  | Start ChatGPT Plus/Pro (Codex) OAuth login.                              |
| `POST`   | `/v1/settings/llm/oauth/openai-codex/logout` | Clear the stored ChatGPT Plus/Pro (Codex) OAuth session.                 |

## Unsupported and absent routes

These behaviors matter because the UI client has helpers for some of them, but the current server does not implement them:

- `POST /v1/collections` is **not implemented** and returns `404 Not Found`.
- `DELETE /v1/collections/{name}` is **not implemented** and returns `404 Not Found`.
- There is no HTTP route for `PUT /v1/documents/...`; upload uses `POST /v1/documents` instead.
- There is no separate collection-create API. Collections are created through the CLI with `docbert collection add`.

## Collections

### `GET /v1/collections`

Return the names of collections already registered in `config.db`.

Response body:

```json
[{ "name": "docs" }, { "name": "notes" }]
```

Notes:

- The server intentionally returns only `name`, not the collection filesystem path.
- Items are derived from the CLI-managed collection registry.

Status codes:

- `200 OK` on success
- `500 Internal Server Error` if the config database cannot be opened or read

## Conversations

The conversations API persists complete conversation objects in `config.db`.

### Conversation shape

`GET`, `POST`, and `PUT` routes use the `docbert_core::Conversation` shape:

```json
{
  "id": "conv-1",
  "title": "New conversation",
  "created_at": 1715000000000,
  "updated_at": 1715000000000,
  "messages": [
    {
      "id": "msg-1",
      "role": "assistant",
      "actor": { "type": "parent" },
      "parts": [
        { "type": "text", "text": "Hello" },
        { "type": "thinking", "text": "Reasoning text" },
        {
          "type": "tool_call",
          "name": "docbert_search",
          "args": { "query": "rust" },
          "result": "...",
          "is_error": false
        }
      ],
      "sources": [
        {
          "collection": "notes",
          "path": "rust.md",
          "title": "Rust"
        }
      ]
    }
  ]
}
```

Important details:

- `role` is lowercase: `"user"` or `"assistant"`.
- `actor` is tagged with `type`, for example `{"type":"parent"}` or `{"type":"subagent", ...}`.
- `parts` is tagged with `type`, for example `text`, `thinking`, or `tool_call`.
- `sources` is optional.
- Legacy stored payloads may be normalized on read, but the route responses use the current normalized shape.

### `GET /v1/conversations`

List saved conversations in descending `updated_at` order.

Response body:

```json
[
  {
    "id": "conv-2",
    "title": "Recent conversation",
    "created_at": 1715000000000,
    "updated_at": 1715000100000,
    "message_count": 4
  },
  {
    "id": "conv-1",
    "title": "Older conversation",
    "created_at": 1714000000000,
    "updated_at": 1714000050000,
    "message_count": 1
  }
]
```

Status codes:

- `200 OK`
- `500 Internal Server Error`

### `POST /v1/conversations`

Create a conversation record.

Request body:

```json
{
  "id": "conv-1",
  "title": "Optional title"
}
```

Behavior:

- `title` is optional.
- If `title` is omitted, the server stores `"New conversation"`.
- `created_at` and `updated_at` are assigned by the server using the current Unix time in milliseconds.
- `messages` starts as an empty list.

Response body (`201 Created`):

```json
{
  "id": "conv-1",
  "title": "New conversation",
  "created_at": 1715000000000,
  "updated_at": 1715000000000,
  "messages": []
}
```

Status codes:

- `201 Created`
- `500 Internal Server Error`

### `GET /v1/conversations/{id}`

Read one conversation by id.

Status codes:

- `200 OK`
- `404 Not Found` if the conversation does not exist
- `500 Internal Server Error`

### `PUT /v1/conversations/{id}`

Replace one conversation.

Request body:

- Must be a full `Conversation` object.
- The server ignores any mismatched body `id` and replaces it with the `{id}` path parameter.
- The server refreshes `updated_at` at write time.

Example request body:

```json
{
  "id": "ignored-if-different",
  "title": "Updated title",
  "created_at": 1715000000000,
  "updated_at": 0,
  "messages": []
}
```

Example response:

```json
{
  "id": "conv-1",
  "title": "Updated title",
  "created_at": 1715000000000,
  "updated_at": 1715000200000,
  "messages": []
}
```

Status codes:

- `200 OK`
- `404 Not Found` if the conversation does not exist
- `500 Internal Server Error`

### `DELETE /v1/conversations/{id}`

Delete one conversation.

Status codes:

- `204 No Content`
- `404 Not Found` if the conversation does not exist
- `500 Internal Server Error`

## Documents

The documents API writes into collection folders on disk and keeps the search index, embeddings, metadata, and stored snapshots in sync.

### Ingest request shape

Uploads use `POST /v1/documents` with this request shape:

```json
{
  "collection": "notes",
  "documents": [
    {
      "path": "nested/hello.md",
      "content": "# Uploaded\n\nBody",
      "content_type": "text/markdown",
      "metadata": { "topic": "rust" }
    }
  ]
}
```

Important limitations:

- `collection` must already exist in the CLI-managed collection registry.
- `content_type` must be either `text/markdown` or `application/pdf`.
- For `text/markdown`, `content` is the raw Markdown text.
- For `application/pdf`, `content` is a base64-encoded PDF payload.
- The server writes the uploaded file into the collection root on disk before ingesting it.
- Uploaded PDFs are stored as `.pdf` files on disk; indexing and preview use extracted Markdown/text.
- Nested paths are allowed.

### `POST /v1/documents`

Upload and ingest one or more Markdown or PDF documents into an existing collection.

Response body:

```json
{
  "ingested": 1,
  "documents": [
    {
      "doc_id": "#abc123",
      "path": "nested/hello.md",
      "title": "Uploaded",
      "metadata": { "topic": "rust" }
    }
  ]
}
```

Behavior:

- The returned `title` is derived from document content and path.
- For PDFs, the title comes from extracted Markdown/text content, while the original PDF remains on disk.
- `metadata` is optional and is stored as document user metadata.
- Existing files at the same path are overwritten.
- Ingest also updates the collection snapshot state.

Status codes:

- `200 OK`
- `400 Bad Request` for unsupported content types, invalid base64/PDF payloads, or invalid collection/path resolution
- `500 Internal Server Error`

### `GET /v1/collections/{name}/documents`

List stored documents for one collection.

Response body:

```json
[
  {
    "doc_id": "#abc123",
    "path": "nested/hello.md",
    "title": "Uploaded"
  }
]
```

Behavior:

- The route verifies that the collection exists.
- `title` is recomputed from the document currently on disk, not just from indexed state.
- For PDFs, the title is derived from extracted Markdown/text preview content.
- Results are sorted by `path`.

Status codes:

- `200 OK`
- `404 Not Found` if the collection does not exist
- `400 Bad Request` if a stored PDF for that collection cannot be parsed
- `500 Internal Server Error`

### `GET /v1/documents/{collection}/{*path}`

Read one document and its stored metadata.

For Markdown documents, `content` is the stored source text. For PDFs, `content` is extracted Markdown/text preview content rather than raw PDF bytes.

Response body:

```json
{
  "doc_id": "notes:hello.md",
  "collection": "notes",
  "path": "hello.md",
  "title": "Uploaded",
  "content": "# Uploaded\n\nBody",
  "metadata": { "topic": "rust" }
}
```

Important note about `doc_id`:

- The response uses `DocumentId::to_string()`, which yields the full collection/path-style identifier.
- This differs from some other endpoints, such as document listing and search results, which use the short display id.

Status codes:

- `200 OK`
- `404 Not Found` if the document metadata does not exist or the file cannot be read from disk
- `400 Bad Request` if the collection/path cannot be resolved or the current PDF content cannot be parsed
- `500 Internal Server Error`

### `DELETE /v1/documents/{collection}/{*path}`

Delete one document from disk and from indexed state.

Behavior:

- The route first checks that stored metadata exists for the requested document.
- It then deletes the source file from disk.
- After that it removes indexed state, embeddings, and metadata.
- The collection snapshot is updated as part of the deletion flow.

Status codes:

- `204 No Content`
- `404 Not Found` if the document metadata is missing or the file cannot be removed from disk
- `400 Bad Request` if the collection/path cannot be resolved
- `500 Internal Server Error`

## Search

### Search request shape

`POST /v1/search` accepts:

```json
{
  "query": "rust ownership",
  "mode": "semantic",
  "collection": "notes",
  "count": 10,
  "min_score": 0.0
}
```

Fields:

- `query` — required string
- `mode` — optional, defaults to `"semantic"`
- `collection` — optional collection filter
- `count` — optional, defaults to `10`
- `min_score` — optional, defaults to `0.0`

Supported modes:

- `semantic`
- `hybrid`

Any other mode returns `400 Bad Request`.

### `POST /v1/search`

Run semantic or hybrid search and return enriched results.

Response body:

```json
{
  "query": "rust ownership",
  "mode": "semantic",
  "result_count": 1,
  "results": [
    {
      "rank": 1,
      "score": 0.95,
      "doc_id": "#abc123",
      "collection": "notes",
      "path": "rust.md",
      "title": "Rust Ownership",
      "metadata": { "topic": "rust" },
      "excerpts": [
        {
          "text": "...excerpt text...",
          "start_line": 12,
          "end_line": 18
        }
      ]
    }
  ]
}
```

Behavior notes:

- The server defaults to `semantic` mode, not `hybrid`.
- `title` is loaded from the current file on disk when possible.
- `metadata` comes from stored document user metadata.
- `excerpts` are derived from the current file content using the query text and may be empty.
- The server returns `result_count` as the actual number of returned items.

Status codes:

- `200 OK`
- `400 Bad Request` for an unknown `mode`
- `500 Internal Server Error`

## LLM settings

### Settings response shape

`GET /v1/settings/llm` and the response from `PUT /v1/settings/llm` use this JSON shape:

```json
{
  "provider": "openai",
  "model": "gpt-4.1",
  "api_key": "sk-...",
  "oauth_connected": false
}
```

`provider`, `model`, and `api_key` may also be `null`.

`oauth_connected` is always present and is `true` only when the current provider is using a live OAuth-backed ChatGPT Codex session.

`oauth_expires_at` is omitted unless an OAuth-backed ChatGPT Codex session is currently available.

### `GET /v1/settings/llm`

Read persisted LLM settings.

Behavior notes:

- `provider` and `model` come from persisted settings when present.
- For API-key-backed providers, `api_key` comes from persisted settings when present.
- If a stored API key is absent, the server may substitute an environment variable based on `provider`:
  - `openai` → `OPENAI_API_KEY`
  - `anthropic` → `ANTHROPIC_API_KEY`
- Unknown API-key providers do not get environment fallback.
- For `provider = "openai-codex"`, the route does not use `llm_api_key` or env fallback. Instead it resolves a stored OAuth session, refreshes it when needed, and returns the current access token as `api_key`.
- If no valid ChatGPT Codex OAuth session is available, `oauth_connected` is `false` and `api_key` is `null`.

Example response:

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet",
  "api_key": "env-or-stored-key",
  "oauth_connected": false
}
```

Example response for a connected ChatGPT Codex session:

```json
{
  "provider": "openai-codex",
  "model": "gpt-5.1-codex-mini",
  "api_key": "oauth-access-token",
  "oauth_connected": true,
  "oauth_expires_at": 1715003600000
}
```

Status codes:

- `200 OK`
- `500 Internal Server Error`

### `PUT /v1/settings/llm`

Persist LLM settings.

Request body:

```json
{
  "provider": "openai",
  "model": "gpt-4.1",
  "api_key": "stored-key"
}
```

Behavior notes:

- Empty-string `api_key` is stored as absent in the persisted settings.
- `provider` and `model` may be cleared by sending `null`.
- If `provider = "openai-codex"`, the server ignores any supplied `api_key` field and persists only the provider/model selection. OAuth state is managed separately.
- The HTTP response returns the normalized effective settings shape, including `oauth_connected`.

Example request that clears settings:

```json
{
  "provider": null,
  "model": null,
  "api_key": ""
}
```

Status codes:

- `200 OK`
- `500 Internal Server Error`

### `POST /v1/settings/llm/oauth/openai-codex/start`

Start the ChatGPT Plus/Pro (Codex) OAuth flow.

Response body:

```json
{
  "authorization_url": "https://auth.openai.com/oauth/authorize?..."
}
```

Behavior notes:

- The route spins up a temporary localhost callback listener on `http://localhost:1455/auth/callback`.
- If that callback port is already busy, the route returns `409 Conflict`.
- The returned URL is intended to be opened in the user's browser.

Status codes:

- `200 OK`
- `409 Conflict` when the temporary callback listener cannot bind to port `1455`
- `500 Internal Server Error`

### `POST /v1/settings/llm/oauth/openai-codex/logout`

Remove the stored ChatGPT Codex OAuth session.

Behavior notes:

- This clears the stored OAuth credential blob but leaves the selected `provider` and `model` unchanged.

Status codes:

- `204 No Content`
- `500 Internal Server Error`

## Notes for integrators

- Use the CLI to create collections; do not assume an HTTP collection-create route exists.
- Uploads support both Markdown and PDF documents.
- PDF uploads send base64-encoded bytes in the request, but document reads return extracted Markdown/text content.
- Search defaults to semantic mode unless you explicitly send `"mode": "hybrid"`.
- Document and search endpoints do not use the same `doc_id` format everywhere:
  - search/list responses use short ids
  - single-document responses use the full `DocumentId::to_string()` form
- If you are consuming both the web UI client and the server directly, treat this page and the route implementation as the source of truth for what the server actually supports.
