# CLI reference

`docbert` is the command-line entrypoint for registering collections, indexing them, querying them, inspecting runtime configuration, and starting the local web or MCP servers.

This page is a command reference. For product overview and setup, use the top-level `README.md`.

## Command shape

```text
docbert [GLOBAL OPTIONS] <COMMAND>
```

## Global options

| Option                 | Description                                                                     |
| ---------------------- | ------------------------------------------------------------------------------- |
| `--data-dir <path>`    | Override the resolved data directory for this invocation.                       |
| `--model <id-or-path>` | Override the resolved ColBERT model for this invocation.                        |
| `-v`, `-vv`, `-vvv`    | Enable logging to stderr. `-v` = info, `-vv` = debug, `-vvv` and above = trace. |

### Data directory resolution

When a command needs storage, docbert resolves the data directory in this order:

1. `--data-dir <path>`
2. `DOCBERT_DATA_DIR`
3. the XDG data directory for `docbert` (typically `~/.local/share/docbert/`)

The directory is created on demand.

### Commands that do not open the data directory

Two commands are handled before storage initialization:

- `docbert doctor`
- `docbert completions <shell>`

That means they do not require an existing data directory.

## Commands

### `docbert collection`

Manage registered document collections.

#### `docbert collection add <path> --name <name>`

Register a directory as a named collection.

- `<path>` must exist and be a directory.
- The path is canonicalized before being stored.
- The command records collection metadata only. It does **not** index files.

Example:

```bash
docbert collection add ~/notes --name notes
```

After adding a collection, run `docbert sync` or `docbert rebuild` to index it.

#### `docbert collection remove <name>`

Remove a collection and its indexed state.

This command:

- removes the collection registration
- removes Tantivy index entries for that collection
- removes stored embeddings for that collection
- removes stored document metadata and user metadata for that collection

It does **not** delete the source directory on disk.

#### `docbert collection list`

List registered collections.

Options:

| Option   | Description                              |
| -------- | ---------------------------------------- |
| `--json` | Emit JSON instead of tab-separated text. |

Behavior notes:

- Human output is `name<TAB>path`.
- If no collections are registered, human output is `No collections registered.`

### `docbert context`

Manage free-text context strings for collections.

Context strings help users and agents understand what a collection contains.

#### `docbert context add <uri> <description>`

Add or replace a context string.

- `<uri>` is typically a collection URI like `bert://notes`
- `<description>` is free text

#### `docbert context remove <uri>`

Remove a stored context string.

#### `docbert context list`

List stored context strings.

Options:

| Option   | Description                              |
| -------- | ---------------------------------------- |
| `--json` | Emit JSON instead of tab-separated text. |

Behavior notes:

- Human output is `uri<TAB>description`.
- If no contexts are defined, human output is `No contexts defined.`

### `docbert search <query>`

Run the normal search path across all collections or a single collection.

This is the default search command. It uses the hybrid search path unless certain flags force the more general search executor.

Options:

| Option                    | Description                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| `-n, --count <count>`     | Number of results to return. Default: `10`.                                                         |
| `-c, --collection <name>` | Restrict search to one collection.                                                                  |
| `--json`                  | Emit JSON output.                                                                                   |
| `--all`                   | Return all results above `--min-score`.                                                             |
| `--files`                 | Print only matching file paths.                                                                     |
| `--min-score <score>`     | Minimum score threshold. Only applied with `--bm25-only`; ignored under RRF fusion. Default: `0.0`. |
| `--bm25-only`             | Skip the semantic leg and return BM25 results directly.                                             |
| `--no-fuzzy`              | Disable fuzzy matching in the BM25 leg.                                                             |

Behavior notes:

- By default, docbert runs BM25 and semantic retrieval in parallel and fuses them with Reciprocal Rank Fusion (`SearchMode::Hybrid`).
- If any of `--bm25-only`, `--no-fuzzy`, or `--all` are set, docbert calls `search::run` directly with the corresponding parameters.
- Output mode is chosen in this order:
  1. `--json`
  2. `--files`
  3. human-readable formatted results
- `--all` changes result selection behavior but does not suppress `--count` parsing; it simply tells the search layer to return all results above the score threshold.

Examples:

```bash
docbert search "vector search"
docbert search "release notes" -c docs --files
docbert search "gpu fallback" --json --min-score 0.2
docbert search "roadmap" --bm25-only --no-fuzzy
```

### `docbert ssearch <query>`

Run semantic-only search.

Options:

| Option                | Description                                 |
| --------------------- | ------------------------------------------- |
| `-n, --count <count>` | Number of results to return. Default: `10`. |
| `--json`              | Emit JSON output.                           |
| `--all`               | Return all results above `--min-score`.     |
| `--files`             | Print only matching file paths.             |
| `--min-score <score>` | Minimum score threshold. Default: `0.0`.    |

Behavior notes:

- This command does not accept `--collection`; it currently searches semantically across the configured corpus through the semantic-search path.
- Output mode selection is the same as for `docbert search`.
- It initializes the model runtime for every invocation and logs runtime details to stderr.

Example:

```bash
docbert ssearch "same concept different wording" -n 20
```

### `docbert get <reference>`

Retrieve a single document by reference.

Accepted reference forms:

- relative path across collections, for example `notes/meeting.md`
- short document id, for example `#abc123`
- qualified reference, for example `notes:meeting.md`

Options:

| Option   | Description                                                                                                   |
| -------- | ------------------------------------------------------------------------------------------------------------- |
| `--json` | Emit JSON with metadata and content.                                                                          |
| `--meta` | Print only collection/path/file metadata.                                                                     |
| `--full` | Accepted, but currently not required because the default non-JSON, non-meta mode already prints full content. |

Behavior notes:

- Human mode prints the file content directly.
- `--meta` wins over `--json` because the command checks `meta` first.
- `--json` includes the resolved full file path and content.

Examples:

```bash
docbert get notes:roadmap.md
docbert get #abc123 --json
docbert get docs/api.md --meta
```

### `docbert multi-get <pattern>`

Retrieve multiple documents by glob pattern against relative paths.

Options:

| Option                    | Description                          |
| ------------------------- | ------------------------------------ |
| `-c, --collection <name>` | Restrict matches to one collection.  |
| `--json`                  | Emit a JSON array.                   |
| `--files`                 | Print only full file paths.          |
| `--full`                  | Print full contents for all matches. |

Behavior notes:

- The pattern is compiled as a glob and matched against stored relative paths.
- Human output mode depends on flags:
  - `--json`: JSON array
  - `--files`: one full path per line
  - `--full`: each document preceded by `--- collection:path ---`
  - default: `collection:path` lines followed by a match count
- If there are no matches and no output-mode flag is set, docbert prints `No documents match '<pattern>'`.

Examples:

```bash
docbert multi-get "journals/2025-05*.md"
docbert multi-get "**/*.md" -c notes --files
docbert multi-get "specs/*.md" --json
```

### `docbert sync`

Incrementally sync registered collections with source files.

Options:

| Option                    | Description               |
| ------------------------- | ------------------------- |
| `-c, --collection <name>` | Sync only one collection. |

Behavior notes:

- Sync processes new, changed, and deleted files only.
- If a collection path no longer exists, the command warns and skips that collection.
- If a collection is already current, docbert prints `Collection '<name>' is up to date.`
- If no collections are registered for the requested scope, docbert prints `No collections to sync.`
- Sync refuses to run if the stored `embedding_model` differs from the currently resolved model. In that case it tells you to run `docbert rebuild`.
- On success, sync stores the current model id as the embedding model.
- File discovery now respects Git ignore rules when the collection root itself is a Git repository.

Use `sync` for normal updates.

Example:

```bash
docbert sync
docbert sync -c notes
```

### `docbert rebuild`

Rebuild indexed state from source files.

Options:

| Option                    | Description                                                |
| ------------------------- | ---------------------------------------------------------- |
| `-c, --collection <name>` | Rebuild only one collection.                               |
| `--embeddings-only`       | Recompute embeddings without rebuilding the Tantivy index. |
| `--index-only`            | Rebuild the Tantivy index without recomputing embeddings.  |

Behavior notes:

- If no collections are registered for the requested scope, docbert prints `No collections to rebuild.`
- Before rebuilding a collection, docbert removes existing indexed state for that collection.
- If a collection path no longer exists, the command warns and skips that collection.
- Rebuild updates the stored embedding model on success.
- File discovery uses the same walker as sync, including Git-ignore-aware discovery for repo-backed collections.

Use rebuild when you need a clean indexing pass or when changing models.

Examples:

```bash
docbert rebuild
docbert rebuild -c notes
docbert rebuild --embeddings-only
docbert rebuild --index-only
```

### `docbert reindex`

Rebuild the PLAID semantic index from the embeddings already stored in `embeddings.db`, without re-encoding any documents.

Behavior notes:

- Reindex does not walk collection roots, does not read source files, and does not call the model.
- It reads every stored embedding, retrains the PLAID centroids/codec, and replaces the on-disk PLAID file at `<data-dir>/plaid.idx`.
- Typical use is after a PLAID builder change (centroid count, codec bit-width, k-means iterations, …) where `rebuild` would unnecessarily re-embed every document against the unchanged model.
- If you changed the embedding model itself, run `docbert rebuild` instead — reindex won't regenerate embeddings.

This command takes no flags.

Example:

```bash
docbert reindex
```

### `docbert status`

Show the resolved runtime model, collection count, and document count.

Options:

| Option   | Description                               |
| -------- | ----------------------------------------- |
| `--json` | Emit JSON instead of human-readable text. |

Behavior notes:

- Human output includes:
  - data directory
  - resolved model id
  - model source
  - embedding model state
  - collection count and collection paths
  - document count
- If the stored embedding model differs from the currently resolved model, status prints:
  - `Embedding model: <stored> (MISMATCH -- run \`docbert rebuild\`)`
- JSON output includes `data_dir`, `model`, `model_source`, `embedding_model`, `collections`, and `documents`.

Example:

```bash
docbert status
docbert status --json
```

### `docbert doctor`

Inspect accelerator/runtime availability without opening the normal data directory.

Options:

| Option   | Description                     |
| -------- | ------------------------------- |
| `--json` | Emit the doctor report as JSON. |

Behavior notes:

- Human output reports the selected device plus CUDA and Metal compile/use status.
- When compiled support exists but runtime use fails, the error is printed.
- A fallback note is printed when relevant.

Example:

```bash
docbert doctor
docbert doctor --json
```

### `docbert model`

Manage the persisted default model setting.

#### `docbert model show`

Show the resolved model and where it came from.

Options:

| Option   | Description                               |
| -------- | ----------------------------------------- |
| `--json` | Emit JSON instead of human-readable text. |

Behavior notes:

- Human output includes the resolved model, source, and any CLI/env/config contributors.
- JSON output includes the resolved model plus the optional CLI/env/config values.

#### `docbert model set <model>`

Persist a default model id or local path in `config.db`.

Behavior notes:

- This stores the value under `model_name`.
- If `<model>` is a local directory and it lacks `config_sentence_transformers.json`, docbert warns that `docbert-pylate` may not load it.
- Changing the default model does not re-embed existing documents. You usually need `docbert rebuild` afterward.

#### `docbert model clear`

Remove the persisted default model setting.

After clearing it, model resolution falls back to CLI override, `DOCBERT_MODEL`, or the built-in default.

Examples:

```bash
docbert model show
docbert model show --json
docbert model set answerdotai/answerai-colbert-small-v1
docbert model clear
```

### `docbert web`

Start the web UI server.

Options:

| Option          | Description                         |
| --------------- | ----------------------------------- |
| `--host <addr>` | Bind address. Default: `127.0.0.1`. |
| `--port <port>` | Bind port. Default: `3030`.         |

Behavior notes:

- `web` resolves the model before starting the server.
- The command opens `config.db` only long enough to resolve the current model, then starts the web runtime.
- It serves the local web application and API from one process.

Example:

```bash
docbert web
docbert web --host 127.0.0.1 --port 3030
```

### `docbert mcp`

Start the MCP server for agent integrations.

Behavior notes:

- `mcp` resolves the model before starting the stdio server.
- The command opens `config.db` only long enough to resolve the current model, then starts the MCP runtime.
- Tool and resource details are documented separately in the MCP docs; this CLI command has no additional flags.

Example:

```bash
docbert mcp
```

### `docbert completions <shell>`

Generate shell completion scripts.

This command is hidden from normal help output but is intentionally available.

Supported shells come from `clap_complete::Shell` and include the standard shells supported by clap-complete.

Example:

```bash
docbert completions bash > ~/.local/share/bash-completion/completions/docbert
```

## Model resolution summary

The resolved model used by commands is chosen in this priority order:

1. `--model <id-or-path>`
2. `DOCBERT_MODEL`
3. persisted `model_name` in `config.db`
4. the built-in default model

`docbert status` and `docbert model show` are the easiest ways to inspect the resolved model and its source.

## Environment variables

| Variable           | Description                                                                             |
| ------------------ | --------------------------------------------------------------------------------------- |
| `DOCBERT_DATA_DIR` | Override the data directory when `--data-dir` is not provided.                          |
| `DOCBERT_MODEL`    | Override the resolved model when `--model` is not provided.                             |
| `DOCBERT_LOG`      | Logging filter used when tracing is initialized. If set, it overrides the `-v` mapping. |

## Exit behavior

The CLI returns success on `0`. Failures are reported through the shared error path and terminate the command with a non-zero exit status.
