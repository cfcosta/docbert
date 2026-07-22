# MCP reference

This page gives information about the MCP server. This command starts the server:

```bash
docbert mcp
```

The `crates/docbert/src/mcp.rs` file has the implementation. This file is the correct source for the tools, prompt, resource template, parameters, and response shapes on this page.

## General information

The MCP server has these surfaces:

- Six tools for search, document retrieval, and status
- One prompt that shows how to use these tools
- One resource template to read indexed documents directly as MCP resources

When the server starts, it:

- Opens the Tantivy search index
- Starts a `ModelManager`
- Uses the stdio transport

For each tool call or resource read, the server opens the config and embedding databases again as necessary. It does not keep those handles attached to one transaction permanently.

## Available MCP surfaces

### Tools

| Name              | Purpose                                                                                                                                                                        | Returns                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| `search`          | Hybrid search. This tool fuses BM25 keyword matching with ColBERT semantic reranking through RRF. `bm25_search` is better for terms that are in the documents without changes. | Plain text summary + structured JSON content.                              |
| `semantic_search` | Semantic-only ColBERT search in all documents.                                                                                                                                 | Plain text summary + structured JSON content.                              |
| `bm25_search`     | Keyword-only BM25 search with an optional collection filter and optional snippet previews.                                                                                     | Plain text summary + structured JSON content.                              |
| `get`             | Reads one document by reference, with an optional slice by line range.                                                                                                         | Resource content (`text/markdown`).                                        |
| `multi_get`       | Reads multiple documents by glob pattern, with size and line limits for each file.                                                                                             | One or more resource contents, and plain text skip notices when necessary. |
| `status`          | Shows a summary of the index, data-dir, model, collections, and documents.                                                                                                     | Plain text summary + structured JSON content.                              |

### Prompt

| Name    | Purpose                                                                                           |
| ------- | ------------------------------------------------------------------------------------------------- |
| `query` | Short information for all six tools, tool selection by signal, and document retrieval and status. |

### Resource template

| URI template     | Name               | MIME type       |
| ---------------- | ------------------ | --------------- |
| `bert://{+path}` | `docbert-document` | `text/markdown` |

## Server metadata and instructions

The MCP server shows these instructions to clients:

- Select a search tool by signal. Use `bm25_search` for terms, identifiers, and strings that are in the documents without changes. Use `semantic_search` for general concepts. Use `search` (hybrid) when the query has the two signals.
- When you find the correct document, use `get` or `multi_get` to read it.
- Use `status` to see the general condition of the index.

These instructions give information only. They do not add separate behavior.

## Tool details

## `search`

This tool does a search of the indexed documents with the standard search stack.

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

- `query`: required string
- `limit`: optional maximum number of results. The default is `10`.
- `minScore`: optional minimum score threshold. The server uses it only when `bm25Only` is true. Under RRF fusion, the server ignores it. The default is `0.0`.
- `collection`: optional collection filter
- `bm25Only`: optional. This flag stops the semantic leg and gives the BM25 results directly.
- `noFuzzy`: optional. This flag stops fuzzy matching in the BM25 leg.
- `all`: optional. This flag gives all the results.
- `includeSnippet`: optional. The default is `true`.

### Behavior

- This tool uses `search::run(...)`. That function does a BM25 leg and a semantic leg. The semantic leg uses the PLAID index. Then the function fuses the two legs with Reciprocal Rank Fusion, unless `bm25Only` is set.
- The tool opens `config.db` and `embeddings.db` for the call.
- The tool locks the shared `ModelManager` during the search.
- If `includeSnippet` is true, the tool tries to read the result file from disk and get a snippet for the query.
- If a collection-level or document-level context exists in `config.db`, the tool includes it in each structured result item as `context`.

### Tool output

This tool gives:

1. A plain text summary as standard MCP text content
2. Structured JSON in `structured_content`

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

- The `format_document_ref(...)` function normalizes `docId` to one `#` prefix.
- The structured JSON uses camelCase field names, for example `resultCount` and `docId`.
- The tool includes no snippet when `includeSnippet` is false, or when it cannot read the file.
- `lineCount` and `byteCount` give information about the preview content that `get` gives for the document. Thus callers can select a `startLine`/`endLine` or `startByte`/`endByte` without a second round-trip. `lineCount` and `byteCount` are `null` when the tool cannot read the file.

## `semantic_search`

This tool does a semantic-only search.

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

- `query`: required string
- `limit`: optional maximum number of results. The default is `10`.
- `minScore`: optional minimum score threshold. The tool uses this on PLAID MaxSim scores. The default is `0.0`.
- `all`: optional. This flag gives all results above the threshold.
- `includeSnippet`: optional. The default is `true`.

### Behavior

- This tool uses `search::semantic(...)`. That function reads the PLAID index and ranks the documents against it.
- The tool gives an MCP error if the PLAID index does not exist yet (refer to `Error::PlaidIndexMissing`).
- The MCP schema for this tool has no collection parameter.
- This tool formats its results in the same way as `search`.

### Tool output

This tool gives the same two things as `search`:

- A plain text summary
- Structured JSON with `query`, `resultCount`, and `results`

If the index is empty, or if nothing matches, the structured `results` array is empty.

## `bm25_search`

This tool does a keyword-only BM25 search.

### Parameters

```json
{
  "query": "PlaidIndexMissing",
  "limit": 10,
  "minScore": 0.0,
  "collection": "notes",
  "noFuzzy": false,
  "all": false,
  "includeSnippet": true
}
```

Fields:

- `query`: required string
- `limit`: optional maximum number of results. The default is `10`.
- `minScore`: optional minimum score threshold. The tool uses this directly on BM25 scores. The default is `0.0`.
- `collection`: optional collection filter
- `noFuzzy`: optional. This flag stops fuzzy matching in the BM25 leg.
- `all`: optional. This flag gives all results above the threshold.
- `includeSnippet`: optional. The default is `true`.

### Behavior

- This tool uses `search::run(...)` with the BM25-only flag always set. Thus the semantic leg does not operate, and the tool does not use the PLAID index. This tool has no `bm25Only` parameter, because it is always BM25.
- This tool is for terms, identifiers, symbols, file names, error strings, and other queries. The words of these queries are in the documents without changes.
- The BM25 leg uses fuzzy matching by default. `noFuzzy` stops it.
- The tool opens `config.db` for the call. It locks the shared `ModelManager`, the same as the other search tools, but BM25 scoring does not use the model.
- This tool formats its results in the same way as `search`. This includes the `includeSnippet` behavior and the structured `context` field.

### Tool output

This tool gives the same two things as `search`:

- A plain text summary
- Structured JSON with `query`, `resultCount`, and `results`

The result items have the same shape as `search` results (`docId`, `collection`, `path`, `file`, `title`, `score`, `context`, `snippet`, `lineCount`, `byteCount`).

## `get`

This tool gets one document by reference.

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

- `reference`: required. The forms are:
  - `collection:path`
  - `#doc_id`
  - plain path
- `startLine`: optional 1-based inclusive first line
- `endLine`: optional 1-based inclusive last line
- `startByte`: optional 0-based inclusive first byte
- `endByte`: optional 0-based inclusive last byte
- `lineNumbers`: optional boolean. When this is true, the tool adds line numbers.

You cannot use line ranges and byte ranges together. If you supply `startLine` or `endLine` with `startByte` or `endByte`, the tool gives an MCP `invalid_params` error.

### Reference parsing

There is one more behavior that helps you:

- If you give no range fields, and `reference` ends in `:<digits>`, the tool uses that suffix as the first line.

For example:

- The tool parses `notes:rust.md:25` as document `notes:rust.md` with `startLine = 25`.

### Behavior

- The tool resolves the document reference with `search::resolve_reference(...)`.
- The tool resolves the collection root from `config.db`.
- The tool reads the file from disk.
- The tool slices the content. It uses `text_util::apply_line_range(...)` for line ranges, and `text_util::apply_byte_range(...)` for byte ranges. A byte offset inside a multi-byte UTF-8 character moves down to the character boundary before it.
- When more content exists after the range, the tool adds a footer. The footer is `[... N more lines remaining]` or `[... N more bytes remaining]`.
- The tool adds line numbers when `lineNumbers` is true. For byte ranges, the line numbers restart at `1`, because byte offsets do not map to line numbers.
- The tool adds a context header before the content, as an HTML comment, if collection-level or document-level context exists.

### Tool output

The search and status tools give a plain text summary and structured JSON. But `get` gives a resource, not plain text JSON.

This example shows the resource shape:

```json
{
  "uri": "bert://notes/rust%2Emd",
  "mimeType": "text/markdown",
  "text": "<!-- Context: Personal notes -->\n\n10: fn main() { ... }"
}
```

### Important difference from search tools

- The search and status tools give a plain text summary and structured JSON.
- `get` gives a `Content::resource(...)` payload.

### Failure behavior

- Missing document reference → MCP resource-not-found style error
- Missing collection root → MCP resource-not-found style error
- Read failure → internal error
- Line range and byte range supplied together → `invalid_params` error

## `multi_get`

This tool gets multiple documents by glob pattern.

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

- `pattern`: required glob for relative paths
- `collection`: optional collection filter
- `startLine`: optional inclusive first line for each file
- `endLine`: optional inclusive last line for each file
- `startByte`: optional inclusive first byte for each file
- `endByte`: optional inclusive last byte for each file
- `lineNumbers`: optional boolean

You cannot use line ranges and byte ranges together, the same as in `get`.

### Behavior

- The tool uses `globset::Glob` to match the relative paths against the document metadata.
- The tool collects the `(collection, path)` pairs that match, and sorts them.
- For each match, the tool:
  - Resolves the full file path
  - Reads the file content from disk
  - Applies the line or byte range
  - Adds the context before the content as an HTML comment, if context exists
  - Gives the file as a resource

### Tool output

This tool can give a content list with two types of entries:

- `Content::resource(...)` entries for the files that the tool reads
- Plain text entries for files that the tool cannot resolve or read

Example skip text:

```text
[SKIPPED: notes:large.md - failed to read]
```

### No-match behavior

If nothing matches, the tool gives an error-style text result (`CallToolResult::error`) with this text:

```text
No documents match '*.md'
```

### Important difference from `get`

- `get` gives one resource, or an error-style text result.
- `multi_get` can give more than one resource, and skip notices, in the same tool result.

## `status`

This tool gives a summary of the index and the collections.

### Parameters

None.

### Behavior

- The tool reads the collection records from `config.db`.
- The tool reads the list of document ids and metadata from `config.db`.
- The tool shows the `model_name` from the settings. If no `model_name` exists, the tool uses `DEFAULT_MODEL_ID`. This tool does not use the `--model` CLI flag or `DOCBERT_MODEL`. This behavior is correct, because the MCP server shows the value in the settings, not the CLI-time model.
- The tool counts the documents for each collection.

### Tool output

This tool gives the same two things as the search tools:

1. A plain text summary
2. Structured JSON

Plain text example:

```text
Docbert index status:
  Data dir: /home/user/.local/share/docbert
  Model: lightonai/GTE-ModernColBERT-v1
  Documents: 42
  Collections: 2
    - notes (30 docs) /home/user/notes
    - docs (12 docs) /home/user/docs
```

Structured example:

```json
{
  "dataDir": "/home/user/.local/share/docbert",
  "model": "lightonai/GTE-ModernColBERT-v1",
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

## Prompt: `query`

The server has one prompt with the name `query`.

This prompt:

- Gives the names of all six tools (`bm25_search`, `semantic_search`, `search`, `get`, `multi_get`, `status`)
- Gives information about tool selection by signal. It recommends `bm25_search` for terms, identifiers, and strings that are in the documents without changes. It recommends `semantic_search` for general concepts with different words. It recommends `search` for queries with the two signals. It also recommends the tool that is best for the query.
- Gives information for use of the tools. `min_score` removes low-confidence results. `search` also has a `bm25_only` flag. That flag gives the same result as `bm25_search`, but the `bm25_search` tool is better. With `get`, you can use `startLine`/`endLine` or `startByte`/`endByte` (inclusive), and optional line numbers.

This prompt gives information to MCP clients. It does not change the tool behavior.

## Resource template: `bert://{+path}`

The server also has one resource template:

- URI template: `bert://{+path}`
- Name: `docbert-document`
- MIME type: `text/markdown`

### Read a resource

You can read resources directly through URIs. For example:

```text
bert://notes/rust%2Emd
bert://notes/subdir/file.md
bert://notes/space%20name.md
```

### URI behavior

- The server decodes the part after `bert://` one segment at a time.
- The first segment is the collection name.
- The other segments become the relative path.
- The server percent-decodes the path segments.

The server rejects these cases:

- A URI scheme that is not `bert`
- An empty collection name
- An empty path

### Resource contents

When you read a `bert://...` resource, the tool gives a markdown text resource with these parts:

- The full file content from disk
- Line numbers that always start at line `1`
- The context before the content, as an HTML comment, if context exists

This example shows the resource:

```json
{
  "uri": "bert://notes/space%20name.md",
  "mimeType": "text/markdown",
  "text": "<!-- Context: Personal notes -->\n\n1: Hello world"
}
```

## Context behavior

Some MCP surfaces can include collection and document context from `config.db`.

Lookup order:

1. The document-level URI context for `bert://{collection}/{path}`
2. The collection-level URI context for `bert://{collection}`

The context is in these locations:

- Search results: the structured `context` field
- `get`: before the content, as `<!-- Context: ... -->`
- `multi_get`: before the content, for each resource it gives
- Resource reads through `bert://...`: before the content

## Defaults and limits

The implementation has these defaults:

- The default search limit is `10`.
- The tool includes search snippets by default.
- `get` and `multi_get` do not have a size cap. Callers slice the content with `startLine`/`endLine` or `startByte`/`endByte`.

## Error-handling notes

The MCP server uses two types of results. The first type is an MCP-style error. The second type is a good tool result that contains text.

### MCP-style error cases

Typical examples:

- The server cannot open the config database
- The server cannot open the embedding database
- A search failure
- A glob pattern that is not correct
- A missing document reference
- A `bert://...` URI that the server cannot use

### Tool results with text, not resource or structured payloads

Typical examples:

- `multi_get` skip notices for files it cannot read, or for missing collections. The tool includes these notices and the resource entries in the same tool result.

This difference is important when you build clients:

- Do not think that every retrieval tool always gives a resource.
- Do not think that every failure is a transport-level error.

## Integration

- Select a search tool by signal. Use `bm25_search` for terms and identifiers. Use `semantic_search` for general concepts. Use `search` for queries that have the two signals. Then use `get` or `multi_get` for the full text.
- Use `bm25_search` for keyword-only search directly. `search` with `bm25Only` gives the same result. You can use `noFuzzy` with the two tools.
- Use `semantic_search` when you want semantic-only retrieval, and a collection parameter is not necessary.
- Make sure that your client can read plain text and resource content from retrieval tools.
- If you keep or compare identifiers, be careful. Search results give normalized `#...` document references. But resource reads use `bert://...` URIs.
