# Chat, conversations, and LLM settings

This page gives information about the backend parts of the docbert chat system:

- How the server stores conversations
- What the conversation HTTP routes do
- How the server stores and finds the LLM settings
- Which chat functions the backend makes sure of, and which functions come from the prompt or the UI at runtime.

This page does not try to give full UI information. This information comes from the implementation in these files:

- `crates/docbert-web/src/routes/conversations.rs`
- `crates/docbert-web/src/routes/settings.rs`
- `crates/docbert-core/src/config_db.rs`
- `crates/docbert-core/src/conversation.rs`
- `crates/docbert-webui/ui/src/pages/chat-agent-runtime.ts`

The chat agent operates fully in the browser. It connects to the configured LLM. Then it uses the docbert MCP tools (`search`, `semantic_search`, `get`, `multi_get`, `status`) for retrieval. The backend has **no** `/v1/chat` endpoint. The only chat-adjacent HTTP surfaces are `/v1/conversations` (conversation persistence) and `/v1/settings/llm` (provider and key configuration).

## What the server stores

The chat system stores two types of data in `config.db`:

1. **Conversations**
   - The server stores them in the `conversations` table.
   - The server uses the conversation id as the key.
   - Each record has the title, the timestamps, and all the messages.

2. **Persisted LLM settings**
   - The server stores them in the `settings` table. Other settings also use this table.
   - The settings use these keys:
     - `llm_provider`
     - `llm_model`
     - `llm_api_key`

The two are different:

- Conversation persistence controls the stored conversations.
- The persisted LLM settings control the provider, model, and key that the chat UI can use.

## Conversation lifecycle

The web API routes under `/v1/conversations` do the full conversation lifecycle.

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

Function:

- The `id` field is necessary.
- The `title` field is optional.
- If the request has no `title`, the server stores `"New conversation"`.
- When it creates the conversation, the server sets `created_at` and `updated_at` to the Unix time in milliseconds.
- The `messages` list starts empty.

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
- `409 Conflict` if a conversation has this `id`. POST does not replace a record. PUT changes a stored record.
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

Function:

- The server gives each conversation as a short record, without all the messages.
- The server gives the results by `updated_at`, most recent first.
- The server calculates `message_count` from the stored `messages.len()`.

Status codes:

- `200 OK`
- `500 Internal Server Error`

### Read one conversation

Route:

```text
GET /v1/conversations/{id}
```

Response shape:

- The server gives the full stored `Conversation` record.

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
          "name": "search",
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
- `404 Not Found` if no conversation has this id
- `500 Internal Server Error`

### Update

Route:

```text
PUT /v1/conversations/{id}
```

Request body:

- The body must be a full `Conversation` object.
- The `messages` field is necessary. It has no `serde(default)`.
- A body without `messages` does not deserialize. The server gives `422 Unprocessable Entity`.
- If the JSON syntax is incorrect, the server gives `400 Bad Request`.

Function:

- The server makes sure that the conversation is in the database.
- The server replaces `body.id` with the `{id}` path parameter.
- When the server writes the record, it sets a new `updated_at`.
- The server stores `created_at` from the request body without a change. The server does not keep `created_at` from the record in the database.
- The server stores the remaining fields of the conversation body without changes.

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
- `404 Not Found` if no conversation has this id
- `500 Internal Server Error`

### Delete

Route:

```text
DELETE /v1/conversations/{id}
```

Function:

- The server deletes the stored conversation record, if it is in the database.

Status codes:

- `204 No Content`
- `404 Not Found` if no conversation has this id
- `500 Internal Server Error`

## Conversation message format

Stored conversations use `docbert_core::Conversation` and the related chat types.

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

- The `id` field
- The `role` field
- The optional `actor` field
- The `parts` field
- The optional `sources` field.

The server ignores and discards unknown input fields. For example, a client can send an unknown `content` field on a message. The server discards it and does not give an error.

#### `role`

The server serializes `role` in lowercase:

- `"user"`
- `"assistant"`

#### `actor`

The `actor` field is optional. It has a `type` tag.

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

The `parts` field is a tagged list. The permitted part types are:

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
  "name": "search",
  "args": { "query": "rust" },
  "result": "...",
  "is_error": false
}
```

The chat runtime selects the string in the `name` field to identify the call. The backend does not do a check of this string. At this time, this string is the same as one of the docbert MCP tool names: `search`, `semantic_search`, `get`, `multi_get`, or `status`.

#### `sources`

When a source is available, it has these fields:

```json
{
  "collection": "notes",
  "path": "rust.md",
  "title": "Rust"
}
```

## Stored records that do not decode

The server encodes stored conversation records with rkyv, in this format. A stored record that does not decode has the same result as a missing record:

- `GET /v1/conversations/{id}` gives `404 Not Found`.
- `GET /v1/conversations` does not include the record. The server records a warning.
- `DELETE /v1/conversations/{id}` removes the record.

## Persisted LLM settings

The chat UI uses the LLM settings from `/v1/settings/llm`.

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

The `provider`, `model`, and `api_key` fields can also be `null`.

The response always includes the `oauth_connected` field. This field shows if the provider has an active ChatGPT Codex OAuth session.

The response includes `oauth_expires_at` only when an active ChatGPT Codex OAuth session is available.

### Storage mapping

The backend stores the primary provider, model, and API-key values in `config.db` with `PersistedLlmSettings`. The server uses these keys:

- `llm_provider`
- `llm_model`
- `llm_api_key`

When a ChatGPT Codex OAuth session is available, the server stores it independently. The server uses JSON with a specified structure, under this key:

- `llm_oauth:openai-codex`

### `GET /v1/settings/llm`

Function:

- The server reads the stored provider and model values from `config.db`.
- For a provider that uses an API key, the server gives a stored API key if there is one.
- If there is no stored API key, the server can use an environment variable for the `provider`.
- For `provider = "openai-codex"`, the server uses a stored OAuth session and not `llm_api_key`.
- The stored ChatGPT Codex session has an end time. If this time is near, the server refreshes the session before it gives the settings.
- If the server has no active ChatGPT Codex session, `oauth_connected` is `false` and `api_key` is `null`.

These are the environment fallback rules:

- `provider = "anthropic"` → `ANTHROPIC_API_KEY`
- `provider = "openai"` → `OPENAI_API_KEY`
- unknown providers → no fallback

Example response with a stored key:

```json
{
  "provider": "openai",
  "model": "gpt-4.1",
  "api_key": "stored-key",
  "oauth_connected": false
}
```

Example response with env fallback:

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet",
  "api_key": "env-key",
  "oauth_connected": false
}
```

Example response with a connected ChatGPT Codex session:

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

Function:

- The server replaces the stored provider, model, and API-key settings in one write transaction.
- For an empty-string `api_key`, the server stores no value.
- You can remove `provider` and `model` when you send `null`.
- For `provider = "openai-codex"`, the server ignores the `api_key` in the request. The server keeps the OAuth state independently.
- The HTTP response gives the settings shape, with `oauth_connected`. This shape shows the settings after the write.

Example request:

```json
{
  "provider": "openai",
  "model": "gpt-4.1",
  "api_key": "stored-key"
}
```

Example request to remove values:

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

This route starts the ChatGPT Plus/Pro (Codex) OAuth flow. It gives an authorization URL.

For a short time, the server binds `http://localhost:1455/auth/callback`. The browser completes the OAuth redirect at this address.

### `POST /v1/settings/llm/oauth/openai-codex/logout`

This route removes the stored ChatGPT Codex OAuth session. It does not remove the selected provider and model.

## What the backend makes sure of

These backend functions are stable in this implementation:

- The server stores conversations in `config.db`.
- The list route gives short records, by `updated_at`, most recent first.
- When a create request has no title, the server sets the title to `New conversation`.
- An update replaces the body id with the path id. The update sets a new `updated_at`.
- A delete request removes the stored record.
- The server stores the LLM settings independently from the stored conversations.
- When a stored API key is missing, the server can use a provider-specific environment variable.

## Runtime information, not a stable backend function

The chat runtime also has prompt and orchestration code in `crates/docbert-webui/ui/src/pages/chat-agent-runtime.ts`. That file gives good information. But it is not a stable backend contract.

At this time, the chat runtime prompt recommends that the model:

- Starts with the search tools
- Does not stop after one search or one file, when more information is necessary
- Examines more than one file before it gives a result
- Uses data from more than one document for its result

The prompt also tells the file-analysis subagents to stay in one file. These subagents give markdown sections with a specified structure.

These functions come from the prompt and the runtime. The conversation routes do **not** make sure of them.

As a result:

- A stored conversation can contain parent-agent and subagent messages. These messages show this orchestration.
- The backend conversation schema includes these messages.
- The backend does not make sure that the model always searches more than one time or examines more than one file.
- The backend does not make sure that the model gives results with the structure that the prompt shows.

## The difference between UI function and backend function

This section shows the difference between the backend functions and the frontend functions.

### Backend function

This page includes:

- The persistence format
- The conversation CRUD routes
- The LLM settings persistence and env fallback
- The serialized message schema.

### UI and runtime function

The backend does not make sure of these functions:

- The conversation title rules in the browser that the server does not store
- The temporary messages during the token stream
- The local message-update function during the token stream
- How the transcript shows the thinking, tool-call, and subagent parts
- Whether the model obeys the system prompt fully each time it operates.

## Integration notes

- If you must keep conversations, use the conversation routes. These routes store the conversations.
- To find if chat can operate, examine `/v1/settings/llm`. Make sure that a provider and a model are available. Make sure that there is an API key, or that `oauth_connected` is `true`.
- If you make tools for the stored conversation data, use the `parts`-based schema. Your tools must accept the optional `actor` and `sources` fields.
- If you think about result quality, know the difference between the two:
  - What the backend stores and gives
  - What the prompt tells the model to do.
