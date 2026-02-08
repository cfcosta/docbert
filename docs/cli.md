# CLI Design

## Command Structure

```
docbert <subcommand> [options]
```

### Global Options

| Option              | Description                              |
| ------------------- | ---------------------------------------- |
| `--data-dir <path>` | Override the XDG data directory          |
| `--model <id>`      | Override the ColBERT model ID or local path |
| `--verbose` / `-v`  | Increase log verbosity (can be repeated) |

## Subcommands

### `docbert collection`

Manage document collections.

#### `docbert collection add <path> --name <name>`

Register a directory as a named collection.

- `<path>`: Path to the directory (resolved to absolute)
- `--name <name>`: Human-readable collection name (required)

Behavior:

1. Validate the directory exists and is readable
2. Store the collection definition in config.db
3. Print a confirmation message

If the collection already exists, this returns an error. Use `docbert sync` or
`docbert rebuild` to (re)index collections.

#### `docbert collection remove <name>`

Remove a collection and all its indexed data.

- Removes the collection definition from config.db
- Deletes all Tantivy entries for documents in this collection
- Deletes all embeddings for documents in this collection
Note: context strings are not removed automatically.

#### `docbert collection list`

List all registered collections with their paths.

Output format (human):

```
notes       ~/notes
meetings    ~/Documents/meetings
docs        ~/work/docs
```

Output format (JSON, with `--json`):

```json
[
  {"name": "notes", "path": "/home/user/notes"},
  ...
]
```

### `docbert context`

Manage context descriptions for collections.

#### `docbert context add <uri> <description>`

Add or update a context string for a collection.

- `<uri>`: Collection URI in the form `bert://<name>` (e.g., `bert://notes`)
- `<description>`: Free-text description

Context strings are stored in config.db and displayed in search results to help users (and AI agents) understand what a collection contains.

#### `docbert context remove <uri>`

Remove the context string for a collection.

#### `docbert context list`

List all context strings.

### `docbert search <query>`

Search across all collections (or a specific one).

#### Options

| Option                | Description                                                 |
| --------------------- | ----------------------------------------------------------- |
| `-n <count>`          | Number of results to return (default: 10)                   |
| `-c <collection>`     | Search only within this collection                          |
| `--json`              | Output results as JSON                                      |
| `--all`               | Return all results above the score threshold (ignores `-n`) |
| `--files`             | Output only file paths (one per line)                       |
| `--min-score <float>` | Minimum MaxSim score threshold (default: 0.0)               |
| `--bm25-only`         | Skip ColBERT reranking, return BM25 results directly        |
| `--no-fuzzy`          | Disable fuzzy matching in the first stage                   |

#### Behavior

1. Run the search pipeline (see pipeline.md)
2. Format and display results

Human output format:

```
  1. [0.847] notes:project-ideas.md ##a1b2c3
     Project Timeline and Milestones

  2. [0.812] docs:roadmap.md ##d4e5f6
     Q1 2025 Roadmap
```

JSON output format:

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

Semantic-only search across all collections (ColBERT only, no BM25 or fuzzy matching).
This performs a full embedding scan, so it is slower on large corpora.

#### Options

| Option                | Description                                                 |
| --------------------- | ----------------------------------------------------------- |
| `-n <count>`          | Number of results to return (default: 10)                   |
| `--json`              | Output results as JSON                                      |
| `--all`               | Return all results above the score threshold (ignores `-n`) |
| `--files`             | Output only file paths (one per line)                       |
| `--min-score <float>` | Minimum MaxSim score threshold (default: 0.0)               |

#### Behavior

1. Encode the query with ColBERT
2. Score every stored document embedding (chunk 0 only)
3. Filter by `--min-score` and limit to `-n` (unless `--all`)
4. Format and display results

Human output format is the same as `docbert search`.

### `docbert get <reference>`

Retrieve a document's full content.

#### Reference formats

- **Path**: `meetings/2024-01-15.md` -- looks up by relative path across all collections
- **Doc ID**: `#abc123` -- looks up by short document ID
- **Qualified path**: `notes:project-ideas.md` -- collection-qualified path

#### Options

| Option   | Description                                                |
| -------- | ---------------------------------------------------------- |
| `--full` | Print full document content (default behavior; flag is currently redundant) |
| `--json` | Output as JSON with metadata                               |
| `--meta` | Print only metadata (collection, path, full file path) |

### `docbert multi-get <pattern>`

Retrieve multiple documents matching a glob pattern.

- `<pattern>`: Glob pattern applied to relative paths (e.g., `journals/2025-05*.md`)
- Searches across all collections

#### Options

| Option            | Description                                  |
| ----------------- | -------------------------------------------- |
| `-c <collection>` | Restrict to a specific collection            |
| `--json`          | Output as JSON array                         |
| `--files`         | Output only file paths                       |
| `--full`          | Include full document content (can be large) |

### `docbert sync`

Incrementally sync collections with source files. Only processes new, changed, or deleted files based on modification time.

#### Options

| Option            | Description                 |
| ----------------- | --------------------------- |
| `-c <collection>` | Sync only this collection   |

This is the recommended command for regular updates. It's much faster than `rebuild` because it only processes files that have changed since the last sync.

### `docbert rebuild`

Full rebuild of indexes from source files. Deletes all existing data and re-indexes everything from scratch.

#### Options

| Option              | Description                       |
| ------------------- | --------------------------------- |
| `-c <collection>`   | Rebuild only this collection      |
| `--embeddings-only` | Only recompute ColBERT embeddings |
| `--index-only`      | Only rebuild the Tantivy index    |

Use this when you need to force a complete re-index (e.g., after index corruption or model changes).

### `docbert model`

Manage the default ColBERT model stored in `config.db`.

#### Subcommands

| Command | Description |
| ------- | ----------- |
| `docbert model show` | Show the resolved model (and source) |
| `docbert model set <model>` | Persist a HuggingFace model ID or local path |
| `docbert model clear` | Clear the stored model setting |

You can also override the model per command with `--model`.

### `docbert status`

Show system status and statistics.

Output:

```
Data directory: ~/.local/share/docbert/
Model: jinaai/jina-colbert-v2
Model source: default
Collections: 3
  notes: /path/to/notes
  meetings: /path/to/meetings
  docs: /path/to/docs
Documents: 1070
```

### `docbert mcp`

Start the MCP (Model Context Protocol) server for AI agents.

The MCP server exposes tools for search and retrieval over stdio:

- `docbert_search` (BM25 + ColBERT, supports collection filters)
- `semantic_search` (ColBERT-only, scans all documents)
- `docbert_get`
- `docbert_multi_get`
- `docbert_status`

Example configuration (Claude Desktop / Claude Code):

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

## Exit Codes

| Code | Meaning                                      |
| ---- | -------------------------------------------- |
| 0    | Success                                      |
| 1    | General error (default on any failure)       |

## Environment Variables

| Variable           | Description                                         |
| ------------------ | --------------------------------------------------- |
| `DOCBERT_DATA_DIR` | Override XDG data directory                         |
| `DOCBERT_MODEL`    | Override default model name (lower priority than `--model`) |
| `DOCBERT_LOG`      | Log level (trace, debug, info, warn, error)         |
| `HF_HOME`          | HuggingFace Hub cache directory (used by pylate-rs) |
| `NO_COLOR`         | Disable colored output (standard)                   |
