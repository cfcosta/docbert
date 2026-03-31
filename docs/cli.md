# CLI design

## Command structure

```text
docbert <subcommand> [options]
```

## Global options

| Option              | Description                             |
| ------------------- | --------------------------------------- |
| `--data-dir <path>` | Override the XDG data directory         |
| `--model <id>`      | Override the ColBERT model ID or path   |
| `--verbose` / `-v`  | Increase log verbosity; can be repeated |

## Subcommands

### `docbert collection`

Manage document collections.

#### `docbert collection add <path> --name <name>`

Register a directory as a named collection.

- `<path>`: directory path, resolved to an absolute path
- `--name <name>`: collection name

This command only records metadata. It does not index files.

Behavior:

1. Check that the directory exists and can be read
2. Store the collection definition in `config.db`
3. Print a confirmation message

If the collection already exists, the command fails. Use `docbert sync` or `docbert rebuild` to index or re-index it.

#### `docbert collection remove <name>`

Remove a collection and its indexed data.

- Deletes the collection definition from `config.db`
- Deletes Tantivy entries for documents in that collection
- Deletes embeddings for documents in that collection
- Leaves context strings alone for now

#### `docbert collection list`

List registered collections and their paths.

Human output:

```text
notes       ~/notes
meetings    ~/Documents/meetings
docs        ~/work/docs
```

JSON output (`--json`):

```json
[
  { "name": "notes", "path": "/home/user/notes" },
  { "name": "docs", "path": "/home/user/docs" }
]
```

### `docbert context`

Manage context descriptions for collections.

#### `docbert context add <uri> <description>`

Add or update a context string for a collection.

- `<uri>`: collection URI in the form `bert://<name>`
- `<description>`: free-text description

Context strings live in `config.db` and show up in search results so users and AI agents can tell what a collection contains.

#### `docbert context remove <uri>`

Remove the context string for a collection.

#### `docbert context list`

List all stored context strings.

### `docbert search <query>`

Search all collections, or one collection if `-c` is set.

#### Options

| Option                | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| `-n <count>`          | Number of results to return (default: 10)                  |
| `-c <collection>`     | Search only this collection                                |
| `--json`              | Output results as JSON                                     |
| `--all`               | Return all results above the score threshold; ignores `-n` |
| `--files`             | Output only file paths, one per line                       |
| `--min-score <float>` | Minimum MaxSim score threshold (default: 0.0)              |
| `--bm25-only`         | Skip ColBERT reranking and return BM25 results             |
| `--no-fuzzy`          | Disable fuzzy matching in the first stage                  |

#### Behavior

1. Run the search pipeline described in `pipeline.md`
2. Format and print the results

Human output:

```text
  1. [0.847] notes:project-ideas.md #a1b2c3
     Project Timeline and Milestones

  2. [0.812] docs:roadmap.md #d4e5f6
     Q1 2025 Roadmap
```

JSON output:

```json
{
  "query": "project timeline",
  "result_count": 1,
  "results": [
    {
      "rank": 1,
      "score": 0.847,
      "doc_id": "#a1b2c3",
      "collection": "notes",
      "path": "project-ideas.md",
      "title": "Project Timeline and Milestones"
    }
  ]
}
```

### `docbert ssearch <query>`

Run semantic-only search across all collections. This skips BM25 and fuzzy matching, then scores every indexed document family directly.

That makes it useful when wording differs a lot from the query, but it is slower on large corpora.

#### Options

| Option                | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| `-n <count>`          | Number of results to return (default: 10)                  |
| `--json`              | Output results as JSON                                     |
| `--all`               | Return all results above the score threshold; ignores `-n` |
| `--files`             | Output only file paths, one per line                       |
| `--min-score <float>` | Minimum MaxSim score threshold (default: 0.0)              |

#### Behavior

1. Encode the query with ColBERT
2. Score every indexed document family against all stored chunk embeddings for that document
3. Keep the best chunk score as the document score
4. Apply `--min-score` and `-n`, unless `--all` is set
5. Format the output the same way as `docbert search`

### `docbert get <reference>`

Retrieve a document's full content.

#### Reference formats

- Path: `meetings/2024-01-15.md` looks up a relative path across all collections
- Doc ID: `#abc123` looks up a short document ID
- Qualified path: `notes:project-ideas.md` scopes the lookup to one collection

#### Options

| Option   | Description                                                       |
| -------- | ----------------------------------------------------------------- |
| `--full` | Print full document content; this is currently the default anyway |
| `--json` | Output JSON with metadata                                         |
| `--meta` | Print only metadata: collection, path, and full file path         |

### `docbert multi-get <pattern>`

Retrieve multiple documents that match a glob pattern.

- `<pattern>` applies to relative paths, for example `journals/2025-05*.md`
- Search spans all collections unless `-c` is set

#### Options

| Option            | Description                   |
| ----------------- | ----------------------------- |
| `-c <collection>` | Restrict to one collection    |
| `--json`          | Output as a JSON array        |
| `--files`         | Output only file paths        |
| `--full`          | Include full document content |

### `docbert sync`

Incrementally sync collections with source files. Only new, changed, or deleted files are processed.

#### Options

| Option            | Description               |
| ----------------- | ------------------------- |
| `-c <collection>` | Sync only this collection |

Use this for normal updates. It is much faster than `rebuild` because it only touches files that changed since the last sync.

### `docbert rebuild`

Rebuild indexes from source files.

This deletes existing data for the affected scope and indexes everything again.

#### Options

| Option              | Description                       |
| ------------------- | --------------------------------- |
| `-c <collection>`   | Rebuild only this collection      |
| `--embeddings-only` | Only recompute ColBERT embeddings |
| `--index-only`      | Only rebuild the Tantivy index    |

Use this when you need a clean pass, for example after index corruption or a model change.

### `docbert model`

Manage the default ColBERT model stored in `config.db`.

#### Subcommands

| Command                     | Description                                    |
| --------------------------- | ---------------------------------------------- |
| `docbert model show`        | Show the resolved model and where it came from |
| `docbert model set <model>` | Persist a HuggingFace model ID or local path   |
| `docbert model clear`       | Clear the stored model setting                 |

You can still override the model per command with `--model`.

### `docbert status`

Show system status and a few basic counts.

Example output:

```text
Data directory: ~/.local/share/docbert/
Model: lightonai/ColBERT-Zero
Model source: default
Collections: 3
  notes: /path/to/notes
  meetings: /path/to/meetings
  docs: /path/to/docs
Documents: 1070
```

### `docbert mcp`

Start the MCP (Model Context Protocol) server for AI agents.

The server exposes these tools over stdio:

- `docbert_search` for BM25 + ColBERT search, with optional collection filters
- `semantic_search` for ColBERT-only search across all documents
- `docbert_get`
- `docbert_multi_get`
- `docbert_status`

Example configuration for Claude Desktop or Claude Code:

```json
{
  "mcpServers": {
    "docbert": {
      "command": "docbert",
      "args": ["mcp"]
    }
  }
}
```

## Exit codes

| Code | Meaning                             |
| ---- | ----------------------------------- |
| 0    | Success                             |
| 1    | General error; used for any failure |

## Environment variables

| Variable           | Description                                                    |
| ------------------ | -------------------------------------------------------------- |
| `DOCBERT_DATA_DIR` | Override the XDG data directory                                |
| `DOCBERT_MODEL`    | Override the default model name; lower priority than `--model` |
| `DOCBERT_LOG`      | Log level: `trace`, `debug`, `info`, `warn`, or `error`        |
| `HF_HOME`          | HuggingFace Hub cache directory used by pylate-rs              |
| `NO_COLOR`         | Disable colored output                                         |
