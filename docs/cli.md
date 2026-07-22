# CLI reference

`docbert` is the command-line entrypoint. It registers collections, indexes them, and queries them. It also shows the runtime configuration and starts the local web or MCP servers.

This page is a command reference. For information about docbert and how to install it, refer to the top-level `README.md`.

## Command shape

```text
docbert [GLOBAL OPTIONS] <COMMAND>
```

## Global options

| Option                 | Description                                                                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `--data-dir <path>`    | Overrides the resolved data directory for this invocation.                                                                                 |
| `--model <id-or-path>` | Overrides the resolved ColBERT model for this invocation.                                                                                  |
| `-v`, `-vv`            | Increases the stderr log verbosity. docbert always writes log output. No flag gives info, `-v` gives debug, and `-vv` and more give trace. |

### Data directory resolution

When storage is necessary for a command, docbert resolves the data directory in this sequence:

1. `--data-dir <path>`
2. `DOCBERT_DATA_DIR`
3. The XDG data directory for `docbert` (usually `~/.local/share/docbert/`).

docbert makes the directory when necessary.

### Commands that do not open the data directory

docbert starts two commands before it initializes storage:

- `docbert doctor`
- `docbert completions <shell>`

Thus, a data directory is not necessary for these two commands.

docbert also starts `docbert clean` before the standard database open. `docbert clean` resolves the data directory like any storage command. But it examines the database files at the filesystem level before it opens anything. Its task is to reset the stores that docbert wrote in the pre-1.0 format. The standard open path rejects those files.

## Commands

### `docbert collection`

The `docbert collection` subcommands add, remove, and show the registered document collections.

#### `docbert collection add <path> --name <name>`

This command registers a directory as a named collection.

- `<path>` must be available and must be a directory.
- docbert canonicalizes the path before storage.
- This command records collection metadata only. It does **not** index the files.

Example:

```bash
docbert collection add ~/notes --name notes
```

After you add a collection, use `docbert sync` or `docbert rebuild` to index it.

#### `docbert collection remove <name>`

This command removes a collection and its indexed state.

This command removes these items for that collection:

- The collection registration and its Merkle snapshot
- The Tantivy index entries
- The chunk manifests, document metadata, and user metadata.

The embedding rows stay. docbert keeps them in a content-addressed cache. Thus, when docbert re-indexes the same content, it gets cache hits and does not use the encoder again. Use `docbert clean` to remove the rows that nothing refers to.

This command does **not** delete the source directory on disk.

#### `docbert collection list`

This command shows the registered collections.

Options:

| Option   | Description                         |
| -------- | ----------------------------------- |
| `--json` | Shows JSON, not tab-separated text. |

Behavior notes:

- The human output is `name<TAB>path`.
- If no collections are registered, the human output is `No collections registered.`

### `docbert context`

The `docbert context` subcommands add, remove, and show the free-text context strings for collections.

Context strings help users and agents to know what a collection contains.

#### `docbert context add <uri> <description>`

This command adds or replaces a context string.

- `<uri>` is usually a collection URI, for example `bert://notes`.
- `<description>` is free text.

#### `docbert context remove <uri>`

This command removes a context string.

#### `docbert context list`

This command shows the context strings.

Options:

| Option   | Description                         |
| -------- | ----------------------------------- |
| `--json` | Shows JSON, not tab-separated text. |

Behavior notes:

- The human output is `uri<TAB>description`.
- If there are no contexts, the human output is `No contexts defined.`

### `docbert search <query>`

This command uses the standard search path in all collections, or in a single collection.

This is the default search command. It uses the hybrid search path. But some flags make docbert use the more general search executor.

Options:

| Option                    | Description                                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `-n, --count <count>`     | The number of results to show. Default: `10`.                                                                             |
| `-c, --collection <name>` | Searches in one collection only.                                                                                          |
| `--json`                  | Shows JSON output.                                                                                                        |
| `--all`                   | Shows all the results, not only the top `--count`.                                                                        |
| `--files`                 | Shows only the file paths of the matches.                                                                                 |
| `--min-score <score>`     | The minimum score threshold. docbert applies it with `--bm25-only`. docbert ignores it during RRF fusion. Default: `0.0`. |
| `--bm25-only`             | Does not use the semantic leg. Shows only the BM25 results.                                                               |
| `--no-fuzzy`              | Prevents fuzzy matching in the BM25 leg.                                                                                  |

Behavior notes:

- By default, docbert uses the BM25 leg and the semantic leg. It fuses them with Reciprocal Rank Fusion (`SearchMode::Hybrid`). The two legs use the same `search::run` path. `--no-fuzzy` and `--all` adjust that path. They do not select a different path.
- `--bm25-only` does not use the semantic leg. It shows only the BM25 results. This is the only mode where `--min-score` filters the results.
- docbert selects the output mode in this sequence:
  1. `--json`
  2. `--files`
  3. The human-readable formatted results.
- `--all` changes how docbert selects the results. docbert parses `--count`, and `--all` does not prevent this. `--all` tells the search layer to show all the results. The search layer does not stop at `--count`. During RRF fusion, no score filter applies. With `--bm25-only`, docbert removes the results below `--min-score`.

Examples:

```bash
docbert search "vector search"
docbert search "release notes" -c docs --files
docbert search "gpu fallback" --json
docbert search "roadmap" --bm25-only --no-fuzzy --min-score 0.2
```

### `docbert ssearch <query>`

This command does the semantic-only search.

Options:

| Option                | Description                                   |
| --------------------- | --------------------------------------------- |
| `-n, --count <count>` | The number of results to show. Default: `10`. |
| `--json`              | Shows JSON output.                            |
| `--all`               | Shows all the results above `--min-score`.    |
| `--files`             | Shows only the file paths of the matches.     |
| `--min-score <score>` | The minimum score threshold. Default: `0.0`.  |

Behavior notes:

- This command does not accept `--collection`. It searches semantically in the configured corpus through the semantic-search path.
- The output mode selection is the same as for `docbert search`.
- It initializes the model runtime for each invocation. It writes the runtime information to stderr.

Example:

```bash
docbert ssearch "same concept different wording" -n 20
```

### `docbert get <reference>`

This command gets a single document by reference.

docbert accepts these references:

- A relative path in the collections, for example `notes/meeting.md`
- A short document id, for example `#abc123`
- A qualified reference, for example `notes:meeting.md`.

Options:

| Option   | Description                                                                                                    |
| -------- | -------------------------------------------------------------------------------------------------------------- |
| `--json` | Shows JSON with metadata and content.                                                                          |
| `--meta` | Shows only the collection, path, and file metadata.                                                            |
| `--full` | docbert accepts this flag, but it does nothing. The default non-JSON and non-meta mode shows the full content. |

Behavior notes:

- The human mode shows the file content.
- If you give `--meta` and `--json` together, the command uses `--meta`. The command examines `meta` first.
- `--json` includes the resolved full file path and the content.

Examples:

```bash
docbert get notes:roadmap.md
docbert get #abc123 --json
docbert get docs/api.md --meta
```

### `docbert multi-get <pattern>`

This command gets multiple documents with a glob pattern on the relative paths.

Options:

| Option                    | Description                              |
| ------------------------- | ---------------------------------------- |
| `-c, --collection <name>` | Gives matches from one collection only.  |
| `--json`                  | Shows a JSON array.                      |
| `--files`                 | Shows only the full file paths.          |
| `--full`                  | Shows the full contents for all matches. |

Behavior notes:

- docbert compiles the pattern as a glob. It compares the glob with the recorded relative paths.
- The human output mode changes with the flags:
  - `--json`: A JSON array
  - `--files`: One full path for each line
  - `--full`: docbert shows `--- collection:path ---` before each document
  - default: `collection:path` lines, and then a match count.
- If there are no matches, and you do not set an output-mode flag, docbert shows `No documents match '<pattern>'`.

Examples:

```bash
docbert multi-get "journals/2025-05*.md"
docbert multi-get "**/*.md" -c notes --files
docbert multi-get "specs/*.md" --json
```

### `docbert sync`

This command syncs the registered collections with the source files.

Options:

| Option                    | Description                |
| ------------------------- | -------------------------- |
| `-c, --collection <name>` | Syncs one collection only. |

Behavior notes:

- Sync operates only on the new, changed, and deleted files.
- If a collection path is not available, the command shows a warning and ignores that collection.
- If a collection is up to date, docbert shows `Collection '<name>' is up to date.`
- If no collections are registered for the given scope, docbert shows `No collections to sync.`
- Sync does not start if the recorded `embedding_model` is different from the resolved model. Then, it tells you to use `docbert rebuild`.
- If sync does not have an error, it records the resolved model id as the embedding model.
- When the collection root is a Git repository, the file walker obeys the Git ignore rules.

Use `sync` for the standard updates.

Example:

```bash
docbert sync
docbert sync -c notes
```

### `docbert rebuild`

This command rebuilds the indexed state from the source files.

Options:

| Option                    | Description                                                        |
| ------------------------- | ------------------------------------------------------------------ |
| `-c, --collection <name>` | Rebuilds one collection only.                                      |
| `--embeddings-only`       | Recomputes the embeddings, but does not rebuild the Tantivy index. |
| `--index-only`            | Rebuilds the Tantivy index, but does not recompute the embeddings. |

Behavior notes:

- If no collections are registered for the given scope, docbert shows `No collections to rebuild.`
- Before docbert rebuilds a collection, it removes the indexed state for that collection.
- If a collection path is not available, the command shows a warning and ignores that collection.
- If rebuild does not have an error, it changes the recorded embedding model.
- Rebuild uses the same file walker as sync. For repo-backed collections, the walker obeys the Git ignore rules.

Use rebuild when a clean indexing pass is necessary, or when you change models.

Examples:

```bash
docbert rebuild
docbert rebuild -c notes
docbert rebuild --embeddings-only
docbert rebuild --index-only
```

### `docbert reindex`

This command rebuilds the PLAID semantic index from the embeddings in `embeddings.db`. It does not re-encode any documents.

Behavior notes:

- Reindex does not examine the collection roots, does not read the source files, and does not use the model.
- Reindex reads each recorded embedding. It retrains the PLAID centroids and codec. It replaces the on-disk PLAID file at `<data-dir>/plaid.idx`.
- You usually use reindex after a change to the PLAID builder. Examples are a change to the centroid count, the codec bit-width, or the k-means iterations. The model does not change. A `rebuild` re-embeds each document with the same model, and this is not necessary.
- If you changed the embedding model, use `docbert rebuild`. Reindex does not recompute the embeddings.

This command has no flags.

Example:

```bash
docbert reindex
```

### `docbert clean`

This command removes the data that is not necessary in the store. It also resets the data from docbert releases before 1.0.

The pre-1.0 error messages tell you to use this command. If `config.db` or `embeddings.db` is in the pre-1.0 redb format, the standard open paths reject it. `docbert clean` then resets it.

Options:

| Option      | Description                                            |
| ----------- | ------------------------------------------------------ |
| `--dry-run` | Shows what docbert can remove, but does not remove it. |
| `--json`    | Shows a machine-readable report, not text.             |

Behavior notes:

- docbert starts clean before the standard database open (refer to "Commands that do not open the data directory" above). Thus, clean operates on the stores that the other commands do not open.
- The legacy pre-pass operates at the filesystem level, before any database opens:
  - If `config.db` is in the pre-1.0 redb format, clean deletes `config.db`, `config.db-lock`, `embeddings.db`, `embeddings.db-lock`, `plaid.idx`, and the `tantivy/` directory. This is a full reset. You must add the collections again with `docbert collection add`. Then, you must use `docbert sync`.
  - If only `embeddings.db` is in the pre-1.0 redb format, clean deletes `embeddings.db`, `embeddings.db-lock`, and `plaid.idx`. Then, clean removes the state for each document. Thus, the next `docbert sync` re-embeds everything. The collections stay registered.
  - When docbert finds legacy-format files, this reset is the full clean operation. With `--dry-run`, docbert shows the reset but does not do it.
- The standard pass operates when docbert finds no legacy-format databases:
  - It removes the orphan embeddings (the rows that no chunk manifest refers to).
  - It removes the wrong-model embeddings. This occurs if the recorded `embedding_model` is different from the model for this invocation. A `--model` override can cause this difference. Then, each recorded embedding is a wrong-model embedding. docbert removes the document state, and the next `docbert sync` re-embeds.
  - It removes the rows that are in the pre-1.0 `f32` embedding layout. These rows also cause docbert to remove the document state. Then, `sync` re-embeds these documents.
  - After the removals, clean rebuilds `plaid.idx` from the remaining embeddings. Clean deletes `plaid.idx` when there are no embeddings.
- `--dry-run` shows what each pass can remove. It does not delete anything.
- `--json` shows a machine-readable report. For the standard pass, the report gives the counts of the total, orphan, wrong-model, and legacy-layout embeddings. It also gives the number of removed embeddings, and the model mismatch information. For the legacy pre-pass, the report gives the database files that docbert found in the legacy format. It also gives the paths that docbert removed.
- If clean removed anything, use `docbert sync` to re-index.

Examples:

```bash
docbert clean --dry-run
docbert clean
docbert clean --json
```

### `docbert status`

This command shows the resolved runtime model, the collection count, and the document count.

Options:

| Option   | Description                          |
| -------- | ------------------------------------ |
| `--json` | Shows JSON, not human-readable text. |

Behavior notes:

- The human output includes these items:
  - The data directory
  - The resolved model id
  - The model source
  - The embedding model state
  - The collection count and the collection paths
  - The document count.
- If the recorded embedding model is different from the resolved model, status shows this text:
  - `Embedding model: <stored> (MISMATCH -- run \`docbert rebuild\`)`
- The JSON output includes `data_dir`, `model`, `model_source`, `embedding_model`, `documents`, and `collections`. The JSON `collections` field is a count (`usize`). It is not the path list from the human output.

Example:

```bash
docbert status
docbert status --json
```

### `docbert doctor`

This command examines the accelerator and runtime availability. It does not open the standard data directory.

Options:

| Option   | Description                      |
| -------- | -------------------------------- |
| `--json` | Shows the doctor report as JSON. |

Behavior notes:

- The human output shows the selected device. It also shows the CUDA and Metal compile status and use status.
- docbert can have compiled support, but it cannot use the support at runtime. Then, docbert shows the error.
- docbert can show a fallback note.

Example:

```bash
docbert doctor
docbert doctor --json
```

### `docbert model`

The `docbert model` subcommands show, set, and remove the default model in `config.db`.

#### `docbert model show`

This command shows the resolved model and its source.

Options:

| Option   | Description                          |
| -------- | ------------------------------------ |
| `--json` | Shows JSON, not human-readable text. |

Behavior notes:

- The human output includes the resolved model, the source, and the CLI, env, and config values.
- The JSON output includes the resolved model and the optional CLI, env, and config values.

#### `docbert model set <model>`

This command writes a default model id or local path to `config.db`.

Behavior notes:

- This command records the value with the `model_name` key.
- `<model>` can be a local directory. If this directory does not have `config_sentence_transformers.json`, docbert shows a warning. The warning tells you that `docbert-pylate` possibly cannot load the model.
- A change to the default model does not re-embed the indexed documents. Usually, `docbert rebuild` is necessary after this.

#### `docbert model clear`

This command removes the default model from `config.db`.

After you clear it, the model resolution uses the `--model` override, `DOCBERT_MODEL`, or the built-in default.

Examples:

```bash
docbert model show
docbert model show --json
docbert model set answerdotai/answerai-colbert-small-v1
docbert model clear
```

### `docbert web`

This command starts the web UI server.

Options:

| Option          | Description                             |
| --------------- | --------------------------------------- |
| `--host <addr>` | The bind address. Default: `127.0.0.1`. |
| `--port <port>` | The bind port. Default: `3030`.         |

Behavior notes:

- `web` resolves the model before it starts the server.
- The command opens `config.db` only to resolve the model. Then, it starts the web runtime.
- It gives the local web UI and the API from one process.

Example:

```bash
docbert web
docbert web --host 127.0.0.1 --port 3030
```

### `docbert mcp`

This command starts the MCP server for agent integrations.

Behavior notes:

- `mcp` resolves the model before it starts the stdio server.
- The command opens `config.db` only to resolve the model. Then, it starts the MCP runtime.
- The MCP docs give the tool and resource information. This CLI command has no flags of its own.

Example:

```bash
docbert mcp
```

### `docbert completions <shell>`

This command makes the shell completion scripts.

The standard help output does not show this command. But the command is available.

The supported shells are in `clap_complete::Shell`. They include the standard shells that clap-complete supports.

Example:

```bash
docbert completions bash > ~/.local/share/bash-completion/completions/docbert
```

## Model resolution summary

docbert selects the resolved model for commands in this sequence:

1. `--model <id-or-path>`
2. `DOCBERT_MODEL`
3. The `model_name` value in `config.db`
4. The built-in default model.

`docbert status` and `docbert model show` are the easiest commands to examine the resolved model and its source.

## Environment variables

| Variable                       | Description                                                                                                                                                                                                                                                |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DOCBERT_DATA_DIR`             | Overrides the data directory when you do not give `--data-dir`.                                                                                                                                                                                            |
| `DOCBERT_MODEL`                | Overrides the resolved model when you do not give `--model`.                                                                                                                                                                                               |
| `DOCBERT_LOG`                  | The tracing filter for the stderr log output (tracing-subscriber `EnvFilter` syntax). When you set it, `-v` does not change the verbosity. When you do not set it, the verbosity is `info` by default, `debug` with `-v`, and `trace` with `-vv` and more. |
| `DOCBERT_EMBEDDING_BATCH_SIZE` | Overrides the embedding batch size that docbert uses when it encodes documents (default: 32 on CPU, 64 on CUDA/Metal). On accelerated devices, the value is a token-budget limit (`batch_size × document_length`). It is not a fixed document count.       |

## Exit behavior

If a command does not have an error, the exit status of the CLI is `0`. docbert shows the errors through the shared error path. The errors stop the command with a non-zero exit status.
