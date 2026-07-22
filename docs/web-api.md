# Web API reference

This page gives information about the HTTP API of `docbert web`.

This page shows how the server operates. The code is in `crates/docbert-web/src/routes/*`. If the browser client has functions for routes that the server does not have, this page agrees with the server, not the client.

## Base path

All API routes use this base path:

```text
/v1
```

The web process also supplies the browser UI. But this page does not include the UI routes.

## Format and conventions

- Request and response bodies are JSON, unless this page shows something different.
- A correct `DELETE` request usually gives a `204 No Content` response.
- Most unusual storage or indexing failures give `500 Internal Server Error`.
- Axum path-decodes the route parameters after URL encoding.
- The UI client sends `Content-Type: application/json` for API requests.

## Route summary

| Method   | Route                                        | Purpose                                                                          |
| -------- | -------------------------------------------- | -------------------------------------------------------------------------------- |
| `GET`    | `/v1/collections`                            | Shows the collection names in the registry.                                      |
| `GET`    | `/v1/conversations`                          | Shows the conversations in storage.                                              |
| `POST`   | `/v1/conversations`                          | Makes a conversation.                                                            |
| `GET`    | `/v1/conversations/{id}`                     | Gets one conversation.                                                           |
| `PUT`    | `/v1/conversations/{id}`                     | Replaces one conversation.                                                       |
| `DELETE` | `/v1/conversations/{id}`                     | Deletes one conversation.                                                        |
| `POST`   | `/v1/documents`                              | Uploads and ingests Markdown or PDF documents into a collection in the registry. |
| `GET`    | `/v1/collections/{name}/documents`           | Shows the documents in one collection.                                           |
| `GET`    | `/v1/documents/{collection}/{*path}`         | Reads one document and its metadata in storage.                                  |
| `DELETE` | `/v1/documents/{collection}/{*path}`         | Deletes one document from disk and from the indexed state.                       |
| `POST`   | `/v1/search`                                 | Does semantic or hybrid search.                                                  |
| `GET`    | `/v1/settings/llm`                           | Reads the LLM settings in storage, with the auth state in use.                   |
| `PUT`    | `/v1/settings/llm`                           | Updates the LLM settings in storage.                                             |
| `POST`   | `/v1/settings/llm/oauth/openai-codex/start`  | Starts ChatGPT Plus/Pro (Codex) OAuth login.                                     |
| `POST`   | `/v1/settings/llm/oauth/openai-codex/logout` | Removes the ChatGPT Plus/Pro (Codex) OAuth session in storage.                   |

## Routes that the server does not have

The UI client has functions for some of these routes, but the server does not have them:

- `POST /v1/collections` is **not available** and gives `404 Not Found`.
- `DELETE /v1/collections/{name}` is **not available** and gives `404 Not Found`.
- There is no HTTP route for `PUT /v1/documents/...`. Upload uses `POST /v1/documents`.
- docbert has no API to make a collection. The `docbert collection add` command in the CLI makes collections.

## Collections

### `GET /v1/collections`

This route gives the names of the collections in the registry in `config.db`.

Response body:

```json
[{ "name": "docs" }, { "name": "notes" }]
```

Notes:

- The server gives only `name`, not the collection filesystem path.
- The items are from the collection registry that the CLI keeps.

Status codes:

- `200 OK` for a correct request
- `500 Internal Server Error` if the server cannot open or read the config database

## Conversations

The conversations API keeps full conversation objects in `config.db`.

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
          "name": "search",
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

Notes:

- `role` is lowercase: `"user"` or `"assistant"`.
- `actor` has a `type` tag, for example `{"type":"parent"}` or `{"type":"subagent", ...}`.
- `parts` has a `type` tag, for example `text`, `thinking`, or `tool_call`.
- `sources` is optional.
- The server ignores and discards unknown fields in request bodies. It does not reject them.
- The server keeps the conversation records with rkyv encoding in this shape. If the server cannot decode a record, the record is not available. Then `GET /v1/conversations/{id}` gives `404`, and `GET /v1/conversations` does not include the record. For such a record, the server records a warning. `DELETE` still removes the record.

### `GET /v1/conversations`

This route gives the conversations in storage in `updated_at` sequence, with the highest value first.

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

This route makes a conversation record.

Request body:

```json
{
  "id": "conv-1",
  "title": "Optional title"
}
```

Notes:

- `title` is optional.
- If the request has no `title`, the server keeps `"New conversation"`.
- The server gives `created_at` and `updated_at` the Unix time in milliseconds.
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
- `409 Conflict` if a conversation with this `id` is in storage. A POST request does not replace it, but a PUT request updates it.
- `500 Internal Server Error`

### `GET /v1/conversations/{id}`

This route reads one conversation by `id`.

Status codes:

- `200 OK`
- `404 Not Found` if there is no conversation with this `id`
- `500 Internal Server Error`

### `PUT /v1/conversations/{id}`

This route replaces one conversation.

Request body:

- The body must be a full `Conversation` object.
- The server ignores a body `id` that is different from the `{id}` path parameter, and uses the `{id}` value.
- The server sets `updated_at` to the write time.
- The server keeps `created_at` from the request body without change. The server does not keep `created_at` from the record in storage.

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
- `404 Not Found` if there is no conversation with this `id`
- `500 Internal Server Error`

### `DELETE /v1/conversations/{id}`

This route deletes one conversation.

Status codes:

- `204 No Content`
- `404 Not Found` if there is no conversation with this `id`
- `500 Internal Server Error`

## Documents

The documents API writes into collection folders on disk. It also updates the search index, the embeddings, the metadata, and the snapshots in storage.

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

Limits:

- The `collection` must be in the collection registry that the CLI keeps.
- The `content_type` must be `text/markdown` or `application/pdf`.
- For `text/markdown`, `content` is the Markdown source text.
- For `application/pdf`, `content` is the PDF data with base64 encoding.
- The server writes the uploaded file into the collection root on disk before it ingests the file.
- The server keeps uploaded PDFs as `.pdf` files on disk. The index and preview use the Markdown or text that the server gets from the PDF.
- A path can have subdirectories.

### `POST /v1/documents`

This route uploads and ingests one or more Markdown or PDF documents into a collection in the registry.

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

Notes:

- The server makes the `title` in the response from the document content and path.
- For PDFs, the `title` is from the Markdown or text content. The uploaded PDF stays on disk.
- `metadata` is optional. The server keeps it as document user metadata.
- The server replaces a file at the same path.
- When the server ingests a document, it also updates the collection snapshot state.

Status codes:

- `200 OK`
- `400 Bad Request` for a content type that the server cannot use, or for base64 or PDF data that is not correct
- `404 Not Found` if the `collection` is not in the registry
- `500 Internal Server Error`

### `GET /v1/collections/{name}/documents`

This route gives the documents in storage for one collection.

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

Notes:

- The route makes sure that the collection is in the registry.
- The server makes the `title` again from the document on disk, not only from the indexed state.
- For PDFs, the `title` is from the Markdown or text preview.
- The server gives the results in `path` sequence.

Status codes:

- `200 OK`
- `404 Not Found` if the collection is not in the registry
- `400 Bad Request` if the server cannot parse a PDF in storage for that collection
- `500 Internal Server Error`

### `GET /v1/documents/{collection}/{*path}`

This route reads one document and its metadata in storage.

For Markdown documents, `content` is the source text in storage. For PDFs, `content` is the Markdown or text preview, not the PDF bytes.

Optional range query parameters (camelCase) let callers get a part of the response from the server:

| Query param | Type    | Notes                        |
| ----------- | ------- | ---------------------------- |
| `startLine` | `usize` | 1-based inclusive first line |
| `endLine`   | `usize` | 1-based inclusive last line  |
| `startByte` | `u64`   | 0-based inclusive first byte |
| `endByte`   | `u64`   | 0-based inclusive last byte  |

You cannot use a line range and a byte range at the same time. If you send `startLine` or `endLine` with `startByte` or `endByte`, the server gives `400 Bad Request`. If you send none of the four parameters, the server gives the full document.

Response body:

```json
{
  "doc_id": "#abc123",
  "collection": "notes",
  "path": "hello.md",
  "title": "Uploaded",
  "content": "# Uploaded\n\nBody",
  "metadata": { "topic": "rust" },
  "line_count": 3,
  "byte_count": 18
}
```

Field notes:

- `doc_id` is the short hex form (for example, `#abc123`). The `disambiguated_short_id` function makes it. If that function is not available, the server uses the `DocumentId::Display` form. `doc_id` agrees with the form that the document list and the search give.
- The server does not include `metadata` when the document has no user metadata in storage.
- `line_count` and `byte_count` give the dimensions of the full document. Callers can use them to make a range request without a second request. The server does not include the two values when it cannot read the file.

Status codes:

- `200 OK`
- `400 Bad Request` if you send a line range and a byte range at the same time, or if the server cannot parse the PDF content
- `404 Not Found` if there is no document metadata, or if the server cannot read the file from disk
- `500 Internal Server Error`

### `DELETE /v1/documents/{collection}/{*path}`

This route deletes one document from disk and from the indexed state.

Notes:

- First, the route makes sure that the metadata for the document is in storage.
- Then the route deletes the source file from disk.
- Then the route removes the indexed state, the embeddings, the chunk offsets, and the metadata.
- The route also updates the collection snapshot as part of this procedure.

Status codes:

- `204 No Content`
- `400 Bad Request` if the server cannot find the collection or the path
- `404 Not Found` if the document metadata is missing, or if the server cannot remove the file from disk
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

- `query`: a string. This field is necessary.
- `mode`: optional. The default is `"semantic"`.
- `collection`: an optional collection filter.
- `count`: optional. The default is `10`.
- `min_score`: optional. The default is `0.0`.

The modes are:

- `semantic`
- `hybrid`
- `bm25`

Any mode that is not one of these three gives `400 Bad Request`. The `bm25` mode uses only the Tantivy full-text index. The PLAID semantic index is not necessary for `bm25`.

### `POST /v1/search`

This route does semantic, hybrid, or BM25 search. It gives results with more fields.

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
      ],
      "line_count": 42,
      "byte_count": 1380,
      "match_chunk": {
        "start_byte": 320,
        "end_byte": 612
      }
    }
  ]
}
```

Notes:

- The default mode is `semantic`, not `hybrid`.
- The server reads the `title` from the file on disk, if possible.
- `metadata` is from the document user metadata in storage. The server does not include it when there is no metadata in storage.
- The server makes the `excerpts` from the file content with the query text. The `excerpts` can be empty. Then the server does not include the field in the JSON.
- `line_count` and `byte_count` give the dimensions of the document on disk. The server does not include the two values when it cannot read the file.
- `match_chunk` has the byte range of the chunk with the best score from the semantic search. The server makes sure that this byte range is not more than the number of bytes in the file. The server does not include `match_chunk` for a BM25-only hit with no chunk-level score. The server also does not include it when it did not record the chunk offsets, or when it cannot read the document.
- The server sets `result_count` to the number of items in the response.

Status codes:

- `200 OK`
- `400 Bad Request` for an unknown `mode`
- `503 Service Unavailable` if the server does not have the PLAID semantic index. The PLAID semantic index is necessary for the `semantic` and `hybrid` modes. The `bm25` mode does not use it, and never gives this status. The server records the query and gives an empty body. The `docbert sync` command makes the index.
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

`provider`, `model`, and `api_key` can also be `null`.

`oauth_connected` is always in the response. It is `true` only when the provider in use has an available OAuth session for ChatGPT Codex.

The server does not include `oauth_expires_at`, unless an OAuth session for ChatGPT Codex is available.

### `GET /v1/settings/llm`

This route reads the LLM settings in storage.

Notes:

- `provider` and `model` are from the settings in storage, when the settings have them.
- For a provider that uses an API key, `api_key` is from the settings in storage, when the settings have it.
- If no API key is in storage, the server can use an environment variable for the `provider`:
  - `openai` → `OPENAI_API_KEY`
  - `anthropic` → `ANTHROPIC_API_KEY`
- An unknown API-key provider does not get an environment variable.
- For `provider = "openai-codex"`, the route does not use `llm_api_key` or an environment variable. The route finds the OAuth session in storage and refreshes it when necessary. Then the route gives the access token as `api_key`.
- If no OAuth session for ChatGPT Codex is available, `oauth_connected` is `false` and `api_key` is `null`.

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

This route keeps the LLM settings in storage.

Request body:

```json
{
  "provider": "openai",
  "model": "gpt-4.1",
  "api_key": "stored-key"
}
```

Notes:

- If `api_key` is an empty string, the server keeps it as no value in the settings.
- You can remove `provider` and `model` when you send `null`.
- If `provider = "openai-codex"`, the server ignores an `api_key` field in the request. The server keeps only the selected `provider` and `model`. The server keeps the OAuth state independently.
- The HTTP response gives the settings shape in use, and includes `oauth_connected`.

Example request that removes settings:

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

This route starts the ChatGPT Plus/Pro (Codex) OAuth login.

Response body:

```json
{
  "authorization_url": "https://auth.openai.com/oauth/authorize?..."
}
```

Notes:

- The route starts a temporary listener for the localhost callback on `http://localhost:1455/auth/callback`.
- If a different program uses the callback port, the route gives `409 Conflict`.
- The user opens the `authorization_url` in a browser.

Status codes:

- `200 OK`
- `409 Conflict` when the temporary callback listener cannot use port `1455`
- `500 Internal Server Error`

### `POST /v1/settings/llm/oauth/openai-codex/logout`

This route removes the OAuth session for ChatGPT Codex from storage.

Notes:

- This route removes the OAuth credential in storage. It keeps the selected `provider` and `model` without change.

Status codes:

- `204 No Content`
- `500 Internal Server Error`

## Notes for integrators

- Use the CLI to make collections. Do not think that docbert has an HTTP route to make a collection.
- You can upload Markdown documents and PDF documents.
- PDF uploads send base64-encoded bytes in the request. Document reads give the Markdown or text content.
- Search uses `semantic` mode as the default. To use a different mode, send `"mode": "hybrid"` or `"mode": "bm25"`.
- All document and search endpoints give `doc_id` as the short hex form (for example, `#abc123`). There is no `collection:path` form in the response.
- Use this page and the route source code to know what the server can do. Do not use only the web UI client.
