# Chat, conversations, and LLM settings

This page documents the backend-facing parts of docbert's chat system:

- how conversations are persisted
- what the conversation HTTP routes actually do
- how LLM settings are stored and resolved
- which chat behaviors are real backend guarantees versus prompt- or UI-driven runtime behavior

It is intentionally narrower than a UI walkthrough. The source of truth here is the current implementation in:

- `crates/docbert/src/web/routes/conversations.rs`
- `crates/docbert/src/web/routes/settings.rs`
- `crates/docbert-core/src/config_db.rs`
- `crates/docbert-core/src/conversation.rs`
- `crates/docbert/ui/src/pages/chat-agent-runtime.ts`

## What is persisted

The chat system persists two different kinds of state in `config.db`:

1. **Conversations**
   - stored in the `conversations` table
   - keyed by conversation id
   - contain title, timestamps, and the full message list

2. **Persisted LLM settings**
   - stored in the shared `settings` table
   - use these keys:
     - `llm_provider`
     - `llm_model`
     - `llm_api_key`

Those are separate concerns:

- conversation persistence controls chat history
- persisted LLM settings control which provider/model/key the chat UI can use

## Conversation lifecycle

The current conversation lifecycle is implemented entirely through the web API routes under `/v1/conversations`.

### Create

Route:

```text
POST /v1/conversations
```

Request body:

```json
{
  "id": "conv-1",
  "title": "Optional title"
}
```

Behavior:

- `id` is required.
- `title` is optional.
- if `title` is omitted, the server stores `"New conversation"`
- `created_at` and `updated_at` are set by the server to the current Unix time in milliseconds
- `messages` starts empty

Example response:

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
- `409 Conflict` if a conversation with the supplied `id` already exists (POST never overwrites — use PUT to update)
- `500 Internal Server Error`

### List

Route:

```text
GET /v1/conversations
```

Response shape:

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

Behavior:

- conversations are returned as summaries, not full message transcripts
- results are sorted by `updated_at` descending
- `message_count` is derived from the stored `messages.len()`

Status codes:

- `200 OK`
- `500 Internal Server Error`

### Read one conversation

Route:

```text
GET /v1/conversations/{id}
```

Response shape:

- returns the full stored `Conversation` record

Example:

```json
{
  "id": "conv-1",
  "title": "Project notes",
  "created_at": 1715000000000,
  "updated_at": 1715000200000,
  "messages": [
    {
      "id": "msg-1",
      "role": "user",
      "actor": { "type": "parent" },
      "parts": [{ "type": "text", "text": "What changed?" }]
    },
    {
      "id": "msg-2",
      "role": "assistant",
      "actor": { "type": "parent" },
      "parts": [
        { "type": "thinking", "text": "Searching relevant files" },
        { "type": "text", "text": "Here is what I found..." },
        {
          "type": "tool_call",
          "name": "search_hybrid",
          "args": { "query": "project changes" },
          "result": "...",
          "is_error": false
        }
      ]
    }
  ]
}
```

Status codes:

- `200 OK`
- `404 Not Found` if the conversation id does not exist
- `500 Internal Server Error`

### Update

Route:

```text
PUT /v1/conversations/{id}
```

Request body:

- must be a full `Conversation` object

Behavior:

- the server checks that the conversation already exists
- the server overwrites `body.id` with the `{id}` path parameter
- the server refreshes `updated_at` on write
- the rest of the conversation body is stored as provided

Example request:

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
  "updated_at": 1715000300000,
  "messages": []
}
```

Status codes:

- `200 OK`
- `404 Not Found` if the conversation id does not exist
- `500 Internal Server Error`

### Delete

Route:

```text
DELETE /v1/conversations/{id}
```

Behavior:

- deletes the stored conversation record if present

Status codes:

- `204 No Content`
- `404 Not Found` if the conversation id does not exist
- `500 Internal Server Error`

## Conversation message format

Stored conversations use `docbert_core::Conversation` and related chat types.

### Conversation shape

```json
{
  "id": "conv-1",
  "title": "Chat",
  "created_at": 1715000000000,
  "updated_at": 1715000100000,
  "messages": []
}
```

### Message shape

Each message has:

- `id`
- `role`
- optional `actor`
- `parts`
- optional `sources`

#### `role`

Serialized as lowercase:

- `"user"`
- `"assistant"`

#### `actor`

`actor` is optional and tagged by `type`.

Parent example:

```json
{ "type": "parent" }
```

Subagent example:

```json
{
  "type": "subagent",
  "id": "sub-1",
  "collection": "notes",
  "path": "rust.md",
  "status": "running"
}
```

Subagent status values:

- `queued`
- `running`
- `done`
- `error`

#### `parts`

`parts` is a tagged list. Supported part types are:

- `text`
- `thinking`
- `tool_call`

Examples:

```json
{ "type": "text", "text": "Answer text" }
```

```json
{ "type": "thinking", "text": "Intermediate reasoning" }
```

```json
{
  "type": "tool_call",
  "name": "search_hybrid",
  "args": { "query": "rust" },
  "result": "...",
  "is_error": false
}
```

#### `sources`

When present, each source has:

```json
{
  "collection": "notes",
  "path": "rust.md",
  "title": "Rust"
}
```

## Legacy normalization behavior

`docbert_core::conversation` still knows how to deserialize older stored payloads that used legacy fields like:

- `content`
- `content_parts`
- `tool_calls`

When those legacy payloads are read, they are normalized into the current `parts`-based structure.

That means:

- old stored data may still load successfully
- current API responses use the normalized modern shape
- new clients should target `parts`, not the legacy fields

## Persisted LLM settings

The chat UI depends on LLM settings served from `/v1/settings/llm`.

### HTTP shape

`GET /v1/settings/llm` and the response from `PUT /v1/settings/llm` use this shape:

```json
{
  "provider": "openai",
  "model": "gpt-4.1",
  "api_key": "sk-...",
  "oauth_connected": false
}
```

`provider`, `model`, and `api_key` may also be `null`.

`oauth_connected` is always present and indicates whether the current provider is backed by an active ChatGPT Codex OAuth session.

`oauth_expires_at` appears only when a live ChatGPT Codex OAuth session is currently available.

### Storage mapping

The backend stores the base provider/model/API-key values in `config.db` via `PersistedLlmSettings` using these keys:

- `llm_provider`
- `llm_model`
- `llm_api_key`

The ChatGPT Codex OAuth session, when present, is stored separately as structured JSON under:

- `llm_oauth:openai-codex`

### `GET /v1/settings/llm`

Behavior:

- loads persisted provider/model values from `config.db`
- for API-key-backed providers, if a stored API key exists, it is returned directly
- if no stored API key exists, the server may fall back to an environment variable based on `provider`
- for `provider = "openai-codex"`, the server resolves a stored OAuth session instead of using `llm_api_key`
- if the stored ChatGPT Codex session is close to expiry, the server refreshes it before returning settings
- if no valid ChatGPT Codex session exists, `oauth_connected` is `false` and `api_key` is `null`

Current environment fallback rules:

- `provider = "anthropic"` → `ANTHROPIC_API_KEY`
- `provider = "openai"` → `OPENAI_API_KEY`
- unknown providers → no fallback

Example response using a stored key:

```json
{
  "provider": "openai",
  "model": "gpt-4.1",
  "api_key": "stored-key",
  "oauth_connected": false
}
```

Example response using env fallback:

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet",
  "api_key": "env-key",
  "oauth_connected": false
}
```

Example response using a connected ChatGPT Codex session:

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

Behavior:

- replaces the persisted provider/model/API-key settings in one write transaction
- empty-string `api_key` is normalized to absent in storage
- `provider` and `model` can be cleared by sending `null`
- if `provider = "openai-codex"`, any supplied `api_key` is ignored and OAuth state remains managed separately
- the HTTP response returns the normalized effective settings shape, including `oauth_connected`

Example request:

```json
{
  "provider": "openai",
  "model": "gpt-4.1",
  "api_key": "stored-key"
}
```

Example clear request:

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

Starts the ChatGPT Plus/Pro (Codex) OAuth flow and returns an authorization URL.

The server temporarily binds `http://localhost:1455/auth/callback` and expects the browser to complete the OAuth redirect there.

### `POST /v1/settings/llm/oauth/openai-codex/logout`

Clears the stored ChatGPT Codex OAuth session without removing the selected provider/model.

## What the backend guarantees

These are stable backend-level behaviors documented by the current implementation:

- conversations are persisted in `config.db`
- listing conversations returns summaries sorted by descending `updated_at`
- creating a conversation defaults the title to `New conversation` when omitted
- updating a conversation overwrites the body id with the path id and refreshes `updated_at`
- deleting a conversation removes the stored record
- LLM settings are persisted separately from conversation history
- stored API keys can fall back to provider-specific environment variables when absent

## What is runtime guidance, not a backend guarantee

The chat runtime also contains prompt and orchestration logic in `crates/docbert/ui/src/pages/chat-agent-runtime.ts`. That file is important context, but it should not be confused with a stable backend contract.

Current runtime guidance includes:

- the parent agent should start with search tools
- it should not stop after a single search or a single file when synthesis is needed
- it should analyze multiple promising files before answering
- it should prefer evidence-backed synthesis across documents
- file-analysis subagents are prompted to stay within one file and return structured markdown sections

Those behaviors are **prompt-driven and runtime-driven**, not enforced by the conversation routes themselves.

In practice, that means:

- a stored conversation may contain parent-agent and subagent messages that reflect this orchestration
- the backend conversation schema supports those messages
- the backend does **not** independently enforce that the model will always search multiple times, analyze multiple files, or structure answers exactly as prompted

## Boundaries between UI behavior and backend behavior

To avoid over-documenting frontend details as server guarantees:

### Backend behavior

Documented here:

- persistence format
- conversation CRUD routes
- LLM settings persistence and env fallback
- current serialized message schema

### UI/runtime behavior

Not guaranteed by the backend:

- exact conversation title generation rules in the browser beyond what the server stores
- placeholder messages used while streaming
- local message-update behavior while tokens arrive
- how the transcript visually renders thinking/tool-call/subagent parts
- whether the model follows the current system prompt perfectly on every run

## Practical integration notes

- If you need durable history, use the conversation routes; they are the persistence boundary.
- If you need to inspect whether chat can run at all, check `/v1/settings/llm` and whether provider/model plus either an API key or `oauth_connected = true` are effectively available.
- If you are building tooling against stored conversation data, target the modern `parts`-based schema and support optional `actor` and `sources`.
- If you are reasoning about answer quality, distinguish between:
  - what the backend stores and returns
  - what the prompt currently encourages the model to do
