# rustbert — Rust crate docs lookup

## Why this file exists

The docbert pipeline uses _local collections_. A local collection is a directory that the user adds with `collection add`, then syncs and searches. This model is correct for "my notes" or "this repo's docs". But it is not correct for the rustbert question:

> "What does `serde::Serializer::serialize_struct` actually look like on `serde 1.0.219`, and what's its docstring?"

The user does not want to clone serde, add it as a collection, sync, and search. The user wants a tool (MCP-shaped, equivalent to the `rust_docs` server) with a crate name and version as input. This tool gives the result when necessary. It uses semantic search and lexical search of the public API of the crate.

`rustbert` is that tool. It is a different binary with its CLI and MCP server. It fetches Rust crate sources from the primary sources (crates.io, with more docstrings from docs.rs JSON). rustbert parses them and gives item-level results.

rustbert also has a `rustbert sync` command. This command reads the `Cargo.lock` of a Rust project. It fetches each dependency before a search. As a result, the cache has a set of crates before a search.

## Primary functions

1. rustbert finds a crate's public API by `(crate_name, version)`. You do not add the crate first. The first call to `rustbert search serde@1.0.219 "Serializer"` must give a result immediately.
2. rustbert shows item-level results. Each result is for one `fn`, `struct`, `trait`, `impl`, or `mod`. It gives the signature, docstring, qualified path, and source span.
3. rustbert gives semantic search (ColBERT) and lexical search (Tantivy) for a fetched crate. It uses the same retrieval stack as docbert.
4. **`rustbert sync` fetches each dependency of a Rust project before a search.** It reads the project `Cargo.lock` to find them. As a result, work in a Rust project does not wait for the first search of each dependency.
5. rustbert uses the cache to the maximum. It fetches and parses a given `(crate, version)` one time only, for all uses. The cache keeps the data after rustbert starts again.
6. rustbert resolves `latest` (or a semver pattern) to a specified published version. It caches that resolution.
7. rustbert is one binary with an MCP server and a CLI. It uses `docbert-core` as a library.

## Functions not included

1. **`rustbert sync` does not index the project source.** `rustbert sync` indexes the dependencies of your project, and not the project source. A different command, `rustbert index` (see §2.1), indexes your project directly. `sync` does not do this.
2. **rustbert does not index all crates on crates.io.** rustbert fetches one crate at a time when necessary, or the crates in a project lockfile (`rustbert sync`). rustbert does not fetch all crates at one time.
3. **No type resolution or cross-crate references.** rustbert follows `pub use` chains, but not all of them. It records re-exports as metadata at the path of the initial item. But rustbert does not include generic-bound resolution or full cross-crate type inference.
4. **No macro expansion.** Source-level parsing cannot find the items that macros make. rustbert does not index these items.
5. **No source edit or rewrite.** rustbert is read-only. It downloads, parses, and gives results.

## 1. The `rustbert` tool

rustbert is a different binary in this workspace. It has its CLI, its MCP server, and its data directory. It is **not** a docbert subcommand, and it does not use the same storage as docbert. rustbert uses `docbert-core` as a library for storage, search, and embedding:

```text
rustbert (binary)
   └──► docbert-core (library: SearchDocument, SearchIndex, EmbeddingDb, ConfigDb, ModelManager)
        └──► docbert-plaid, docbert-pylate
```

The user model:

- The user uses `rustbert`, and not `docbert rustbert <subcommand>`.
- The two MCP servers (`rustbert mcp` and `docbert mcp`) operate independently. The user installs each one in the editor or agent.
- There is no cross-tool routing. The docbert chat does not find Rust API questions and send them to rustbert. rustbert does not use docbert's collections.

`docbert-core` is the only part that rustbert and docbert use together, and only as a library dependency.

### 1.1 Data directory

rustbert uses its data directory, adjacent to the docbert data directory:

```text
~/.local/share/rustbert/        # (or $XDG_DATA_HOME/rustbert/)
├── config.db                   # synthetic-collection metadata, sync history
├── embeddings.db               # ColBERT token embeddings per cached item
├── plaid.idx                   # PLAID multi-vector index
├── tantivy/                    # lexical index
└── crate-cache/
    ├── serde-1.0.219.crate     # raw downloaded tarball
    ├── serde-1.0.219/          # extracted source tree
    └── tokio-1.45.0/
```

You can override the standard directories with `RUSTBERT_DATA_DIR` or the `--data-dir` CLI flag. rustbert does not use the same data directory as docbert. The different data directories make sure that rustbert does not mix the user-prose results with the Rust-API results.

## 2. User-visible surface

### 2.1 CLI

```bash
# Pre-fetch every dep of a Rust project by walking its Cargo.lock
rustbert sync
rustbert sync --lock /path/to/Cargo.lock
rustbert sync --jobs 8 --force        # re-fetch even if cached, 8-way parallel
rustbert sync --dry-run               # show the plan without fetching
rustbert sync --exclude 'serde*'      # skip crates by glob (repeatable)

# Index the host project itself (the case sync deliberately skips)
rustbert index                        # current dir
rustbert index /path/to/project       # explicit project root or workspace root

# One-off lookup of a specific crate (also fetches if not cached)
rustbert search serde@1.0.219 "serialize a struct with a custom field name"
rustbert search serde "Serializer" --kind trait     # version defaults to latest
rustbert get serde@1.0.219 serde::Serializer::serialize_struct
rustbert list serde@1.0.219 --kind trait --module serde::de

# Cache management
rustbert status                       # all cached crates with item counts + fetched_at
rustbert status serde                 # all versions of one crate
rustbert evict serde@1.0.0            # drop one entry
rustbert evict --all                  # nuke the cache

# Pre-warm a single crate without searching
rustbert fetch serde@1.0.219

# Re-resolve cached "latest" / semver-pattern entries against upstream
rustbert refresh                              # all
rustbert refresh serde                        # one crate
rustbert refresh --older-than 604800          # only entries older than N seconds (here: 7 days)

# Long-lived runtime
rustbert mcp                          # stdio MCP server
```

### 2.2 MCP tools

rustbert gives all four tools to an LLM caller as **Rust documentation lookup**. The agent uses them when it writes, examines, or debugs Rust code. At these times the agent must have correct API information from a published crate, and not information from training data.

```jsonc
// search
{
  "name": "search",
  "description": "Look up Rust crate documentation: search a published crate's public API for items matching a query.",
  "input": {
    "crate": "serde",
    "version": "1.0.219",           // or "latest", or a semver req like "^1.0"; default "latest"
    "query": "serialize a struct with a custom field name",
    "kind": "fn",                   // optional: fn|struct|enum|trait|impl|mod|const|type|macro
    "module_prefix": "serde::de",   // optional path prefix to scope results
    "limit": 10                     // optional, default 10
  }
}

// get
{
  "name": "get",
  "description": "Read the full rustdoc entry — signature, doc comment, source location — for one item by qualified path.",
  "input": {
    "crate": "serde",
    "version": "1.0.219",           // optional, default "latest"
    "path": "serde::Serializer::serialize_struct"
  }
}

// list
{
  "name": "list",
  "description": "Browse a published crate's public API by listing items, optionally filtered by kind or module prefix.",
  "input": {
    "crate": "serde",
    "version": "latest",            // optional, default "latest"
    "kind": "trait",                // optional
    "module_prefix": "serde::de",   // optional
    "limit": 50                     // optional, default 50
  }
}

// status
{
  "name": "status",
  "description": "Report which Rust crates and versions are cached locally for documentation lookup.",
  "input": { "crate": "serde" }     // optional; filters to one crate
}
```

The MCP server uses JSON-RPC through stdio. It gives tool schemas as `serde_json` literals (no `rmcp` or `schemars` runtime).

`sync` is **CLI-only**. rustbert does not make it an MCP tool. A read of a `Cargo.lock` and a fetch of many crates can continue for some minutes. This operation is incorrect for an MCP request/response and for an LLM caller.

Resource template:

```
rustbert://<crate>/<version>/<qualified_path>
```

A read of `rustbert://serde/1.0.219/serde::Serializer::serialize_struct` gives the item with its signature, docstring, and source slice.

## 3. `rustbert sync` — dependency fetch before search

This is the primary command for users who work in a Rust project.

### 3.1 Operation

```text
1. Locate the lockfile
        ├── --lock PATH          → use that file
        └── default              → walk up from cwd until Cargo.lock is found

2. Parse Cargo.lock with the `cargo-lock` crate
        └── enumerate every [[package]] entry

3. Filter
        ├── skip entries with `source = "..."` other than crates.io  (path/git deps)
        ├── skip entries already cached (unless --force)
        └── apply --exclude / --depth filters

4. Plan
        ├── compute fetch order (no ordering constraints — they're independent)
        └── show a summary before fetching: "<N> crates queued (<M> already cached)"

5. Fetch + index in parallel (default: --jobs 4)
        for each (crate, version):
            ├── crates.io tarball download (with checksum verification)
            ├── flate2 + tar extraction → crate-cache/<crate>-<version>/
            ├── syn parse + module discovery
            ├── lower items → SearchDocument
            ├── ColBERT embed + Tantivy/PLAID index
            └── mark cache entry as `ready` in config.db

6. Report
        ├── per-crate: succeeded / failed / skipped (with reason)
        └── final summary + total time + cache size delta
```

### 3.2 CLI flags

| Flag             | Standard | Effect                                                        |
| ---------------- | -------- | ------------------------------------------------------------- |
| `--lock PATH`    | found    | Uses a specified `Cargo.lock`, and not a search up from cwd   |
| `--jobs N`       | `4`      | Number of fetches at the same time                            |
| `--force`        | off      | Re-fetches a `(crate, version)` pair, also when it is cached  |
| `--dry-run`      | off      | Prints the fetch steps, but does not fetch                    |
| `--exclude GLOB` | none     | Ignores crates by a glob pattern (repeatable, with `globset`) |

Embedding is part of `sync`. rustbert fetches, parses, indexes (Tantivy + ColBERT), and keeps each `(crate, version)` before `sync` completes. As a result, you can search all of the cache when `sync` completes.

### 3.3 Fetch count, retries, and rate limits

- The standard value is 4 fetches at the same time. Crates.io does not show a maximum request rate, but rustbert keeps the rate low. Each fetch downloads not more than a small number of MB. This is a low request rate.
- A 429 or 503 response causes exponential backoff with jitter, and not more than 3 retries.
- A failure does not stop `sync`. The final report shows the result for each crate. When you use `rustbert sync` again, it retries the entries with a failure. A different command, [`rustbert refresh`](#36-refreshing-latest-entries), re-resolves `latest`-style entries.

### 3.4 The limits of `sync`

- **Does not index the source of your project.** `rustbert index` does this (see §2.1). `sync` is only about dependencies on crates.io.
- **Does not fetch git or path dependencies.** These are not on crates.io. When `sync` selects the crates.io entries from the package list, it ignores the git and path entries without output.
- **Does not write to `Cargo.toml` or `Cargo.lock`.** `sync` is read-only.
- **Does not hold the lockfile open** during the fetch. As a result, you can start `cargo build` at the same time.
- **Does not make a different entry for each feature-flag set of a crate version.** rustbert keeps a crate one time for each resolved version. This does not change with the features in different parts of your tree.

### 3.5 Workspaces

For workspace projects (many `Cargo.toml` members below one `Cargo.lock`):

- `rustbert sync` reads each dependency in the resolved lockfile. This does not change with the workspace member that uses the dependency.
- rustbert reads many lockfiles in the same tree (this is not frequent) one at a time. You give `--lock` for each lockfile.

### 3.6 Refreshing `latest` entries

`rustbert sync` does not change the `latest`-resolved entries in the cache. You cannot change specified versions, and the cache does not remove entries without a command.

rustbert has a different command for new versions that become available on crates.io:

```bash
rustbert refresh                       # re-resolve every cached "latest" / semver-pattern entry
rustbert refresh serde                 # only that crate
rustbert refresh --older-than 604800   # only entries older than N seconds (here: 7 days)
```

rustbert keeps refresh as a different command. A `sync` command that also re-resolves versions does a different quantity of work each time. The quantity changes with the number of out-of-date cache entries. Users do not want this from a pre-fetch command.

## 4. Fetch pipeline (one crate)

`rustbert sync` (for each package) and an on-demand search miss use this same path.

### 4.1 Source

| Source            | URL pattern                                           | Format              | Pros                                                  | Cons                                             |
| ----------------- | ----------------------------------------------------- | ------------------- | ----------------------------------------------------- | ------------------------------------------------ |
| crates.io tarball | `https://crates.io/api/v1/crates/{c}/{v}/download`    | `.crate` (gzip tar) | Always available. Small. You cannot change a version. | Source only. A syn parse is necessary.           |
| docs.rs JSON      | `https://docs.rs/crate/{c}/{v}/json` (when available) | rustdoc-types JSON  | Full trait and type information. doc links.           | Not all crates have it. The format has versions. |

**Primary source:** crates.io tarball and `syn`. This is always available.

**docs.rs docstrings:** rustbert can add docstrings from docs.rs JSON to the syn parse. It does this when docs.rs JSON is available for the resolved `(crate, version)`, with `rustdoc_merge::merge_rustdoc_docs`. rustbert tries this step. If a failure occurs, rustbert records it, ignores it, and uses the syn-only result with no change.

### 4.2 Version resolution

```text
"latest"  → GET https://crates.io/api/v1/crates/{name}
            → pick max stable, non-yanked version
            → cache (name, "latest", resolved_version, fetched_at)

"^1.0"    → resolve via semver against the same JSON
"1.0.*"   → ditto
"1.0.219" → use as-is; 404 → clean error to caller
```

You cannot change a specified version. rustbert caches `latest` and semver-pattern resolutions on the first lookup. It uses them again until the user starts `rustbert refresh` (or `rustbert evict`). There is no TTL.

### 4.3 Tarball handling

```text
1. Download                                  reqwest (rustls, no default features)
2. Verify Content-Length + SHA-256           checksum from the crates.io index
3. Extract                                   flate2 + tar → crate-cache/<crate>-<version>/
4. Run cargo_metadata --offline              enumerate package(s) and src roots
5. Parse .rs files with syn                  per the data model in §5
6. Lower items → SearchDocument              with synthetic collection name
7. ColBERT embed + Tantivy/PLAID index       reusing docbert-core
8. Mark cache entry `ready`                  in rustbert's config.db
```

If a step has a failure, rustbert records the entry as `failed`, with a cause. Calls after this retry when necessary.

### 4.4 Module file lookup

rustbert finds the file for a `mod foo;` declaration in this sequence:

1. The `#[path = "..."]` attribute, if it is available
2. `parent_dir/foo.rs`
3. `parent_dir/foo/mod.rs`
4. If rustbert finds no file, it records a load failure for the module and continues.

For an inline `mod foo { … }`, rustbert reads the AST directly, without a filesystem lookup.

## 5. Data model

### 5.1 `RustItem`

```rust
pub struct RustItem {
    pub kind: RustItemKind,           // Fn, Struct, Enum, Trait, Impl, Mod, Const, Static, TypeAlias, Macro
    pub crate_name: String,
    pub crate_version: String,
    pub module_path: Vec<String>,
    pub name: Option<String>,
    pub qualified_path: String,       // "serde::Serializer::serialize_struct"
    pub signature: String,            // canonicalized via prettyplease
    pub doc_markdown: String,
    pub source_file: PathBuf,         // cache-relative, e.g. "src/ser/mod.rs"
    pub byte_span: Range<usize>,
    pub line_span: Range<u32>,
    pub visibility: Visibility,
    pub attrs: Vec<String>,           // pre-rendered (#[deprecated], #[cfg(unix)], …)
}
```

### 5.2 Lowering to `SearchDocument`

```text
SearchDocument.did             = DocumentId::new(synthetic_collection, qualified_path)
SearchDocument.relative_path   = "<source_file>#L<start>-L<end>"
SearchDocument.title           = qualified_path
SearchDocument.searchable_body = format!(
    "{kind} {qualified_path}\n\n{signature}\n\n{doc_markdown}"
)
SearchDocument.raw_content     = Some(item source slice)
SearchDocument.metadata        = Some(json!({
    "kind": kind,
    "crate": crate_name,
    "version": crate_version,
    "module_path": module_path,
    "visibility": visibility,
    "attrs": attrs,
    "source_file": source_file,
    "line_span": [start, end],
}))
```

### 5.3 Synthetic collection naming

rustbert keeps each cached `(crate, version)` as a synthetic collection:

```
rustbert:<crate>@<resolved_version>
```

rustbert has its data directory. As a result, a synthetic collection name cannot be the same as a docbert collection name. A person can see the name in a low-level check. But the rustbert CLI and MCP do not tell the user to type one. The user always uses `(crate, version)`.

### 5.4 `cfg` and re-exports

- `#[cfg(...)]` items: rustbert indexes each one and records the predicate in `attrs`. No filter uses them at this time. Each `cfg`-gated item is searchable.
- `pub use` re-exports: rustbert indexes these only at their _initial_ path. rustbert examines private modules during the lowering. As a result, rustbert follows a `pub use` chain to the initial item, also through non-public modules. rustbert records alias paths as metadata, not as different items.

## 6. Cache invariants and eviction

### 6.1 Layout

```text
~/.local/share/rustbert/
├── config.db                       # synthetic-collection metadata
│   ├── crate_versions              (crate, requested, resolved, fetched_at, status)
│   ├── crate_items                 (synthetic_collection_id, qualified_path → metadata blob)
│   └── sync_runs                   (lockfile_path, started_at, finished_at, summary)
├── embeddings.db                   # ColBERT vectors
├── plaid.idx                       # PLAID multi-vector index
├── tantivy/                        # lexical index
└── crate-cache/
    ├── serde-1.0.219.crate
    └── serde-1.0.219/
```

### 6.2 Invariants

- You cannot change a specified version. rustbert does not re-fetch it without `--force`.
- `latest` and semver patterns resolve to a specified version. rustbert caches that version. A re-resolution after the TTL can make a new entry. rustbert keeps the previous entry.
- Cache entries with `status = failed` retry on the next access.

### 6.3 Eviction

rustbert v1 has no eviction without a command:

- `rustbert evict <crate>[@<version>]` removes a specified entry.
- `rustbert evict --all` removes all entries.

A subsequent LRU policy with a `cache.max_bytes` limit is possible, if the cache becomes too large.

## 7. Dependencies

### 7.1 Source parsing

- **`syn` (`full`)** — the primary AST source. It uses the stable toolchain and gives a full AST.
- **rustdoc JSON through docs.rs** — more docstrings that rustbert adds to the syn parse (`docs_rs.rs`, `rustdoc_merge.rs`). It has a schema version. rustbert ignores it when it is not available.
- **`tree-sitter-rust`** — rustbert does not use it. It is not better than syn.
- **`ra_ap_*`** — rustbert does not use these. They are IDE tools, not a search-index dependency.

### 7.2 Lockfile parsing

`cargo-lock` (the RustSec crate) reads `Cargo.lock` into a typed model. This model has package, version, source, dependencies, and checksum fields. `rustbert sync` uses `cargo-lock` to read the lockfile. For each crate, `toml` reads the manifest. rustbert does not use `cargo_metadata`. A build environment is not necessary.

### 7.3 Network and archive

- `reqwest` (rustls, no default features, with the `stream` feature for download streams) — for HTTP.
- `flate2` — gzip.
- `tar` — tarball extraction.
- `semver` (`serde` feature) — version resolution.
- `sha2` — checksum check.

### 7.4 Other

- `serde` and `serde_json` — for the crates.io API and metadata blobs.
- `proc-macro2` (`span-locations`) — byte spans for rustbert. rustbert records it directly in `[dependencies]` only to set the `span-locations` feature to on. This feature applies to the `proc-macro2` that syn and quote include (cargo-machete ignores the missing `use` statement).
- `quote` — makes the signature token-tree text.
- `tokio` (`rt`, `rt-multi-thread`, `macros`, `time`, `sync`) — the async runtime.
- `clap` (`derive`, `env`) — parses the CLI arguments.
- `tracing` and `tracing-subscriber` (`env-filter`) — for the log messages.
- `globset` — for the `--exclude` glob patterns.
- `toml` — reads crate `Cargo.toml` files that rustbert extracts from tarballs.
- `tantivy` — a direct dependency. rustbert can open the lexical index without the docbert-core re-export, for low-level access.
- `thiserror` — for error definitions.

### 7.5 Manifest

```toml
[package]
name    = "rustbert"
version = "0.1.0"
edition = "2024"

[dependencies]
docbert-core = { path = "../docbert-core" }

# CLI + runtime
clap            = { version = "4.6", features = ["derive", "env"] }
tokio           = { version = "1", features = ["rt", "rt-multi-thread", "macros", "time", "sync"] }
tracing         = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Network + archive
reqwest         = { version = "0.13.2", default-features = false, features = ["rustls", "stream"] }
flate2          = "1"
tar             = "0.4"
semver          = { version = "1", features = ["serde"] }
sha2            = "0.11"

# Cargo / Rust parsing
cargo-lock      = "10"
syn             = { version = "2", features = ["full"] }
proc-macro2     = { version = "1", features = ["span-locations"] }
quote           = "1"
toml            = "0.8"

# Index + filesystem helpers
tantivy         = "0.26.0"
globset         = "0.4"

# Serialization + errors
serde           = { version = "1", features = ["derive"] }
serde_json      = "1"
thiserror       = "2"

[features]
default    = []
mkl        = ["docbert-core/mkl"]
accelerate = ["docbert-core/accelerate"]
metal      = ["docbert-core/metal"]
cuda       = ["docbert-core/cuda"]
```

The MCP server is a hand-written JSON-RPC server through stdio. There is no `rmcp` or `schemars` runtime. There is no `xdg` dependency. rustbert does the data-dir resolution in-tree, with `RUSTBERT_DATA_DIR` and `XDG_DATA_HOME`.

## 8. Use of `docbert-core`

rustbert uses `docbert-core` as a library, and does not add to core:

- `SearchDocument`, `DocumentId`, `ChunkPlan` — rustbert lowers items to these.
- `SearchIndex`, `EmbeddingDb`, `ConfigDb`, `DataDir` — rustbert uses these for storage.
- `ModelManager` — rustbert uses this for ColBERT inference.
- `docbert_core::search::run` — rustbert uses this for the search backend. rustbert applies the `kind` and `module_prefix` filters **post-rank**, on the cached `RustItem` records. (rustbert does not push the `kind` field on the lowered metadata into a Tantivy field.) If the post-rank filter is too slow with many items, a follow-up can add a typed filter in `docbert-core::search::run`. The post-rank filter is satisfactory.

## 9. Failure modes

- **Crate does not exist (404):** rustbert gives a clear `CrateNotFound { name }` error.
- **Version does not exist:** rustbert gives `VersionNotFound { name, requested, resolvable }`.
- **Yanked version:** rustbert accepts it but shows it in the response metadata. Queries for it continue to operate.
- **Network failure:** rustbert retries with backoff. After the retries, rustbert gives `FetchFailed`.
- **The tarball checksum does not agree:** rustbert stops. It does not parse source that is possibly changed.
- **Parse failure for one file:** rustbert records a load failure and continues with the other files.
- **Damaged cache:** rustbert finds this through the `status` column. It evicts the not-completed entries on the next access.
- **Disk full during a fetch:** rustbert removes the not-completed entry and gives `CacheWriteFailed`.
- **`Cargo.lock` has an incorrect format (sync):** rustbert stops the `sync` with a clear error before a fetch.
- **`Cargo.lock` not found (sync):** rustbert gives a clear error and recommends `--lock`.

## 10. Status

**Completed (v0.1.0):**

- The `rustbert` binary in this workspace, with the manifest in §7.5.
- CLI: `search`, `get`, `list`, `status`, `evict`, `fetch`, `sync`, `refresh`, `index`, `mcp`. The `--data-dir` flag for all commands, with the `RUSTBERT_DATA_DIR` env variable as an alternative.
- MCP tools: `search`, `get`, `list`, `status`. Hand-written JSON-RPC through stdio, with no `rmcp` runtime.
- crates.io tarball download through `reqwest`, `flate2`, and `tar`. rustbert makes sure that the checksum agrees with the crates.io index.
- rustbert reads `Cargo.lock` through `cargo-lock`.
- Synthetic-collection storage in the rustbert data dir.
- More docs.rs JSON docstrings that rustbert adds to the syn parse.
- `rustbert index` indexes your project, and this includes workspaces. rustbert indexes each member as its synthetic collection.

**Subsequent work:**

- rustbert can push the `kind` and `module_prefix` filters into `docbert-core::search::run`, if the post-filter cost becomes too large.
- A `rustbert web` UI or HTTP interface (there is no `Web` subcommand at this time).
- More work on the `rustbert://` resource template and MCP resource API.
- Custom registries, sparse protocols, and git protocols for non-crates.io sources.

## 11. Resolved questions

These questions were open during the design. They are resolved at this time:

- **Embedding is a standard part of `rustbert sync`.** `sync` gives search immediately after it completes.
- **Re-resolution of `latest` is a different command.** `rustbert sync` retries the work with a failure. `rustbert refresh` re-resolves `latest` and semver-pattern entries through crates.io.
- **rustbert and docbert are different projects.** There is no cross-routing between the docbert chat agent and the rustbert MCP tools. Users install the two MCP servers independently if they want the two.

## 12. Open questions

1. **Fetch count limit.** `--jobs 4` is a low value. A test on a `Cargo.lock` with approximately 150 crates can set the correct value.
2. **Cache changes for `latest`.** There is no TTL at this time. Users start `rustbert refresh` (with the optional `--older-than <seconds>`) when they want a re-resolution. rustbert can examine this again, because the manual step is a possible problem.
3. **Data dirs for each project.** The `--data-dir` flag overrides the data directory for one command. A continuous cache for each project is also possible with a config file or an env variable.
4. **Filter for dev-dependencies.** `Cargo.lock` does not show the difference between dev-dependencies and runtime dependencies. As a result, `sync` indexes all entries in the lockfile. A subsequent opt-out can use `cargo metadata` and not the lockfile.
5. **`cfg`-gated items.** rustbert at this time indexes each item (it records the predicate in `attrs`). If search gives too many platform-specific items, a configuration flag can select only the items for the local platform.

## 13. Risks

- **Network dependency for new lookups.** Without a network connection, only cached crates operate. To decrease this risk, rustbert uses the maximum cache, `RUSTBERT_OFFLINE=1`, and `rustbert sync` (to fetch crates before a search).
- **crates.io rate limits.** rustbert is a tool for one user, with a low request rate and a low fetch count. rustbert obeys `Retry-After` and uses backoff after a 429.
- **`sync` time on large lockfiles.** A 300-crate lockfile with full embedding can continue for some minutes. rustbert can decrease this with progress bars, `--no-embed`, `--depth`, and more fetches at the same time.
- **A large cache.** Tarballs are usually a small number of MB. The embeddings can be very large. A subsequent LRU eviction policy can decrease this.
- **rustdoc-types schema changes (Phase 2).** The format has versions. rustbert holds it at one version and puts the JSON docstrings behind a feature flag.
- **An overlap with docbert.** `rustbert mcp` and `docbert mcp` operate at the same time. Users of the two install the two MCP servers in their editor. The names and tool descriptions must show that the two are different. As a result, an LLM caller selects the correct one.

## Related references

- [`architecture.md`](./architecture.md)
- [`pipeline.md`](./pipeline.md)
- [`dependencies.md`](./dependencies.md)
- [`storage.md`](./storage.md)
- [`mcp.md`](./mcp.md)
