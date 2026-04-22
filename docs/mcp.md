# MCP reference

This page documents the MCP server started by:

```bash
docbert mcp
```

The implementation lives in `crates/docbert/src/mcp.rs` and is the source of truth for the tools, prompt, resource template, parameter handling, and response shapes described here.

## Overview

The MCP server exposes:

- **tools** for search, document retrieval, and status
- **one prompt** that explains how to use those tools
- **one resource template** for reading indexed documents directly as MCP resources

At startup, the server:

- opens the Tantivy search index
- initializes a `ModelManager`
- serves over stdio

For each tool call or resource read, it reopens the config and embedding databases as needed rather than keeping those handles permanently attached to a single transaction.

## Available MCP surfaces

### Tools

| Name                | Purpose                                                                                       | Returns                                                                  |
| ------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `docbert_search`    | Hybrid/BM25-oriented search with optional collection filtering and optional snippet previews. | Plain text summary + structured JSON content.                            |
| `semantic_search`   | Semantic-only ColBERT search across all documents.                                            | Plain text summary + structured JSON content.                            |
| `docbert_get`       | Read one document by reference, optionally slicing by line range.                             | Resource content (`text/markdown`).                                      |
| `docbert_multi_get` | Read multiple documents by glob pattern with per-file size/line limits.                       | One or more resource contents, plus plain text skip notices when needed. |
| `docbert_status`    | Show index/data-dir/model/collection/document summary.                                        | Plain text summary + structured JSON content.                            |

### Prompt

| Name            | Purpose                                                                                             |
| --------------- | --------------------------------------------------------------------------------------------------- |
| `docbert_query` | A short usage guide telling clients to start with search, then fetch documents or status as needed. |

### Resource template

| URI template     | Name               | MIME type       |
| ---------------- | ------------------ | --------------- |
| `bert://{+path}` | `docbert-document` | `text/markdown` |

## Server metadata and instructions

The MCP server reports these high-level instructions to clients:

- start with `docbert_search` or `semantic_search`
- use `docbert_get` or `docbert_multi_get` once you know what to read
- use `docbert_status` for index health and collection summary

Those instructions are advisory metadata, not separate executable behavior.

## Tool details

## `docbert_search`

Search indexed documents using the normal search stack.

### Parameters

```json
{
  "query": "rust ownership",
  "limit": 10,
  "minScore": 0.0,
  "collection": "notes",
  "bm25Only": false,
  "noFuzzy": false,
  "all": false,
  "includeSnippet": true
}
```

Fields:

- `query` — required string
- `limit` — optional maximum number of results; default `10`
- `minScore` — optional minimum score threshold; ignored under RRF fusion, only applied when `bm25Only` is true; default `0.0`
- `collection` — optional collection filter
- `bm25Only` — optional, skip the semantic leg and return BM25 results directly
- `noFuzzy` — optional, disable fuzzy matching in the BM25 leg
- `all` — optional, return all results
- `includeSnippet` — optional, defaults to `true`

### Behavior

- Uses `search::run(...)`, which runs BM25 and semantic retrieval in parallel and fuses them with Reciprocal Rank Fusion unless `bm25Only` is set.
- Opens `config.db` and `embeddings.db` for the call.
- Locks the shared `ModelManager` while searching.
- If `includeSnippet` is true, the server tries to read the matching file from disk and extract a snippet for the query.
- If a collection-level or document-level context exists in `config.db`, it is included in each structured result item as `context`.

### Tool output

The tool returns:

1. **plain text summary** as normal MCP text content
2. **structured JSON** in `structured_content`

Plain text example:

```text
Found 2 results for "rust ownership":
#abc123 0.950 notes/rust.md
#def456 0.811 docs/memory.md
```

Structured example:

```json
{
  "query": "rust ownership",
  "resultCount": 2,
  "results": [
    {
      "docId": "#abc123",
      "collection": "notes",
      "path": "rust.md",
      "file": "notes/rust.md",
      "title": "Rust",
      "score": 0.95,
      "context": "Personal notes",
      "snippet": "1: Rust is fast.\n2: Ownership keeps memory safe.",
      "lineCount": 2,
      "byteCount": 43
    }
  ]
}
```

### Notes

- `docId` is normalized through `format_document_ref(...)`, so it has a single leading `#`.
- The structured JSON uses camelCase field names like `resultCount` and `docId`.
- No snippet is included when `includeSnippet` is false or when the file cannot be read.
- `lineCount` and `byteCount` describe the preview content the document returns through `docbert_get`, so callers can pick a `startLine`/`endLine` or `startByte`/`endByte` without a second round-trip. Both are `null` when the file cannot be read.

## `semantic_search`

Run semantic-only search.

### Parameters

```json
{
  "query": "same concept different wording",
  "limit": 10,
  "minScore": 0.0,
  "all": false,
  "includeSnippet": true
}
```

Fields:

- `query` — required string
- `limit` — optional maximum number of results; default `10`
- `minScore` — optional minimum score threshold; default `0.0`
- `all` — optional, return all results above threshold
- `includeSnippet` — optional, defaults to `true`

### Behavior

- Uses `search::semantic(...)`.
- Does **not** accept a collection parameter in the MCP schema.
- Shares the same result formatting path as `docbert_search`.

### Tool output

Like `docbert_search`, this returns:

- a plain text summary
- structured JSON with `query`, `resultCount`, and `results`

If the index is effectively empty or nothing matches, the structured `results` array is empty.

## `docbert_get`

Fetch one document by reference.

### Parameters

```json
{
  "reference": "notes:rust.md",
  "startLine": 10,
  "endLine": 60,
  "lineNumbers": true
}
```

Fields:

- `reference` — required; accepted forms are:
  - `collection:path`
  - `#doc_id`
  - plain path
- `startLine` — optional 1-based inclusive first line
- `endLine` — optional 1-based inclusive last line
- `startByte` — optional 0-based inclusive first byte
- `endByte` — optional 0-based inclusive last byte
- `lineNumbers` — optional boolean; when true, adds line numbers

Line and byte ranges are mutually exclusive. Supplying any of `startLine`/`endLine` alongside any of `startByte`/`endByte` returns an MCP `invalid_params` error.

### Reference parsing detail

There is one extra convenience behavior:

- if no range fields are given and `reference` ends in `:<digits>`, the server interprets that suffix as the starting line

For example:

- `notes:rust.md:25` is parsed as document `notes:rust.md` with `startLine = 25`

### Behavior

- Resolves the document reference via `search::resolve_reference(...)`.
- Resolves the collection root from `config.db`.
- Reads the file from disk.
- Slices the content: `text_util::apply_line_range(...)` for line ranges, `text_util::apply_byte_range(...)` for byte ranges. Byte offsets landing inside a multi-byte UTF-8 character round down to the previous character boundary.
- When the requested range omits trailing content, a `[... N more lines remaining]` or `[... N more bytes remaining]` footer is appended.
- Adds line numbers when requested. For byte ranges, numbering restarts at 1 since byte offsets don't map to line numbers.
- Prepends an HTML comment context header if collection- or document-level context exists.

### Tool output

Unlike the search and status tools, `docbert_get` returns a **resource**, not plain text JSON.

Example resource shape conceptually:

```json
{
  "uri": "bert://notes/rust%2Emd",
  "mimeType": "text/markdown",
  "text": "<!-- Context: Personal notes -->\n\n10: fn main() { ... }"
}
```

### Important difference from search tools

- search/status tools return plain text summary plus structured JSON
- `docbert_get` returns a `Content::resource(...)` payload

### Failure behavior

- missing document reference → MCP resource-not-found style error
- missing collection root → MCP resource-not-found style error
- read failure → internal error
- line range and byte range both supplied → `invalid_params` error

## `docbert_multi_get`

Fetch multiple documents by glob pattern.

### Parameters

```json
{
  "pattern": "**/*.md",
  "collection": "notes",
  "startLine": 1,
  "endLine": 50,
  "lineNumbers": true
}
```

Fields:

- `pattern` — required glob against relative paths
- `collection` — optional collection filter
- `startLine` — optional inclusive per-file first line
- `endLine` — optional inclusive per-file last line
- `startByte` — optional inclusive per-file first byte
- `endByte` — optional inclusive per-file last byte
- `lineNumbers` — optional boolean

Line and byte ranges are mutually exclusive, as in `docbert_get`.

### Behavior

- Uses `globset::Glob` to match relative paths against stored document metadata.
- Collects matching `(collection, path)` pairs and sorts them.
- For each match:
  - resolves the full file path
  - reads file content from disk
  - applies the configured line or byte range
  - prepends the context HTML comment if present
  - emits the file as a resource

### Tool output

This tool may return a **mixed content list**:

- `Content::resource(...)` entries for successfully read files
- plain text entries for files that could not be resolved or read

Example skip text:

```text
[SKIPPED: notes:large.md - failed to read]
```

### No-match behavior

If nothing matches, the tool returns an error-style text result:

```text
No documents match '*.md'
```

### Important difference from `docbert_get`

- `docbert_get` returns one resource or an error-like text result
- `docbert_multi_get` can return several resources plus skip notices in the same tool result

## `docbert_status`

Return index and collection summary information.

### Parameters

None.

### Behavior

- Reads collection registrations from `config.db`
- Reads the list of document ids and metadata from `config.db`
- Uses `model_name` from settings, falling back to `DEFAULT_MODEL_ID`
- Counts documents per collection

### Tool output

Like the search tools, this returns:

1. plain text summary
2. structured JSON

Plain text example:

```text
Docbert index status:
  Data dir: /home/user/.local/share/docbert
  Model: lightonai/LateOn
  Documents: 42
  Collections: 2
    - notes (30 docs) /home/user/notes
    - docs (12 docs) /home/user/docs
```

Structured example:

```json
{
  "dataDir": "/home/user/.local/share/docbert",
  "model": "lightonai/LateOn",
  "documents": 42,
  "collections": [
    {
      "name": "notes",
      "path": "/home/user/notes",
      "documents": 30
    }
  ]
}
```

## Prompt: `docbert_query`

The server publishes one prompt named `docbert_query`.

Purpose:

- explain the available tools
- encourage clients to search first, then fetch documents
- mention useful parameters like `min_score`, `bm25_only`, and line-range support

This prompt is advisory content for MCP clients. It does not change the underlying tool behavior.

## Resource template: `bert://{+path}`

The server also exposes one resource template:

- URI template: `bert://{+path}`
- name: `docbert-document`
- MIME type: `text/markdown`

### Reading a resource

The server supports direct resource reads via URIs like:

```text
bert://notes/rust%2Emd
bert://notes/subdir/file.md
bert://notes/space%20name.md
```

### URI behavior

- the part after `bert://` is decoded segment-by-segment
- the first segment is the collection name
- the rest becomes the relative path
- path segments are percent-decoded

Invalid cases include:

- unsupported URI schemes
- empty collection name
- empty path

### Resource contents

Reading a `bert://...` resource returns a markdown text resource with:

- full file content from disk
- line numbers always added starting at line 1
- prepended HTML comment context if context is available

Conceptual example:

```json
{
  "uri": "bert://notes/space%20name.md",
  "mimeType": "text/markdown",
  "text": "<!-- Context: Personal notes -->\n\n1: Hello world"
}
```

## Context behavior

Several MCP surfaces can include collection/document context from `config.db`.

Lookup order:

1. document-level URI context for `bert://{collection}/{path}`
2. collection-level URI context for `bert://{collection}`

Where it appears:

- search results: as the structured `context` field
- `docbert_get`: prepended as `<!-- Context: ... -->`
- `docbert_multi_get`: prepended the same way for each returned resource
- direct resource reads via `bert://...`: prepended the same way

## Defaults and limits

Important defaults from the implementation:

- default search limit: `10`
- search snippets are included by default
- neither `docbert_get` nor `docbert_multi_get` impose a size cap — callers slice explicitly with `startLine`/`endLine` or `startByte`/`endByte`

## Error-handling notes

The MCP server uses a mix of MCP-style errors and successful tool results containing text when appropriate.

### MCP-style error cases

Typical examples:

- failed to open config DB
- failed to open embedding DB
- search failure
- invalid glob pattern
- missing document reference
- invalid or unsupported `bert://...` URI

### Successful tool results with text instead of resource/structured payloads

Typical examples:

- `docbert_multi_get` skip notices for unreadable files or missing collections
- `docbert_multi_get` with no matches

This distinction matters when building clients:

- do not assume every retrieval tool always returns a resource
- do not assume every failure comes back as a transport-level error

## Integration tips

- Start with `docbert_search` or `semantic_search`, then use `docbert_get` or `docbert_multi_get` for full text.
- Prefer `docbert_search` when you want BM25 controls like `bm25Only` or `noFuzzy`.
- Use `semantic_search` when you want semantic-only retrieval and do not need a collection parameter.
- Handle both plain text and resource content in retrieval tools.
- If you store or compare identifiers, note that search results return normalized `#...` document references, while direct resource reads use `bert://...` URIs.
