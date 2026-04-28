# rustbert — Rust crate docs lookup

**Status:** design proposal, not yet implemented

**Date:** 2026-04-27

## Why this file exists

docbert's existing pipeline is built around _local collections_: a directory the user registers with `collection add`, syncs, and searches. That model is great for "my notes" or "this repo's docs" but is the wrong shape for the question this design solves:

> "What does `serde::Serializer::serialize_struct` actually look like on `serde 1.0.219`, and what's its docstring?"

The user does not want to clone serde, register it as a collection, sync, and search. They want a tool — MCP-shaped, like the existing `rust_docs` server — that takes a **crate name and version** and returns the answer on demand, with semantic + lexical search across the crate's public API.

This document proposes a new project, **`rustbert`** — a separate binary with its own CLI and MCP server — that fetches Rust crate sources from canonical remotes (crates.io, optionally docs.rs), parses them, and serves item-level answers. It also exposes a `rustbert sync` command that walks a Rust project's `Cargo.lock` and proactively pre-fetches every dependency, so a working set of crates is hot in the cache before any search runs.

## Goals

1. Look up a crate's public API by `(crate_name, version)` with **no registration step**. First call to `rustbert search serde@1.0.219 "Serializer"` should just work.
2. Surface item-level results: one hit per `fn` / `struct` / `trait` / `impl` / `mod`, with signature, docstring, qualified path, and source span.
3. Support semantic search (ColBERT) and lexical search (Tantivy) over a fetched crate, reusing docbert's existing retrieval stack underneath.
4. **`rustbert sync` proactively fetches every dependency of a Rust project** by walking its `Cargo.lock`, so working in a real codebase doesn't pay first-search latency for every dep.
5. Cache aggressively. A given `(crate, version)` is fetched and parsed at most once across runs; the cache survives restarts.
6. Resolve `latest` (or a semver pattern) to a concrete published version and cache that resolution.
7. Ship as a standalone binary (`rustbert`) with both an MCP server and a CLI; reuse `docbert-core` under the hood.

## Non-goals

1. **Indexing local Cargo projects' source.** `rustbert sync` indexes your project's _dependencies_, not the project itself. A "look at my own crate" feature can layer on later — it's out of scope for v1.
2. **Indexing the entire crates.io corpus.** Fetches are demand-driven (one crate at a time) or scoped to a project's lockfile (`rustbert sync`). No mass ingestion.
3. **Type resolution / cross-crate references.** Following `pub use` chains, resolving generic bounds, or mapping across crate boundaries is out of scope for v1.
4. **Macro expansion.** Items synthesized by macros are invisible to source-level parsing; we accept that gap.
5. **Editing or rewriting source.** Read-only. We download, parse, and serve.

## 1. What `rustbert` is

rustbert is its own project — a separate binary with its own CLI, its own MCP server, and its own data directory. It is **not** a docbert subcommand and does not share storage with docbert. Architecturally, it depends on `docbert-core` as a library for storage / search / embedding primitives:

```text
rustbert (binary)
   └──► docbert-core (library: SearchDocument, SearchIndex, EmbeddingDb, ConfigDb, ModelManager)
        └──► docbert-plaid, docbert-pylate
```

Whether rustbert ships from its own repository or lives as a sibling crate next to docbert in a Cargo workspace is a deployment choice that doesn't affect the user model. Either way:

- the user runs `rustbert` directly — never `docbert rustbert <subcommand>`
- the two MCP servers (`rustbert mcp` and `docbert mcp`) are independent and the user wires them up separately in their editor / agent config
- there is no cross-tool routing: docbert's chat doesn't auto-detect Rust API questions and forward them to rustbert, and rustbert doesn't reach into docbert's collections

`docbert-core` is the only shared surface, and only as a library dependency.

### 1.1 Data directory

rustbert uses its own data directory, parallel to docbert's:

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

Defaults are overrideable with `RUSTBERT_DATA_DIR`. Sharing docbert's data dir is _not_ supported in v1 — keeping them separate avoids accidental cross-contamination of search results between user prose and Rust APIs.

## 2. User-visible surface

### 2.1 CLI

```bash
# Pre-fetch every dep of a Rust project (default: cwd, follows Cargo.lock)
rustbert sync
rustbert sync /path/to/project
rustbert sync --lock /path/to/Cargo.lock
rustbert sync --jobs 8 --force        # re-fetch even if cached, 8-way parallel

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
rustbert refresh                      # all
rustbert refresh serde                # one crate
rustbert refresh --older-than 7d      # only stale entries

# Long-lived runtimes
rustbert mcp                          # stdio MCP server
rustbert web --port 3031              # phase 3+, see §10
```

### 2.2 MCP tools

All four tools are framed for an LLM caller as **Rust documentation lookup** — the agent should reach for them whenever it is writing, reviewing, or debugging Rust code and needs ground-truth API information from a published crate rather than relying on training data.

```jsonc
// rustdocs_search
{
  "name": "rustdocs_search",
  "description": "Look up Rust crate documentation: search a published crate's public API for items matching a query.",
  "input": {
    "crate": "serde",
    "version": "1.0.219",          // or "latest", or a semver req like "^1.0"
    "query": "serialize a struct with a custom field name",
    "kind": "fn",                   // optional: fn|struct|enum|trait|impl|mod|const|type|macro
    "limit": 10                     // optional, default 10
  }
}

// rustdocs_get
{
  "name": "rustdocs_get",
  "description": "Read the full rustdoc entry — signature, doc comment, source location — for one item by qualified path.",
  "input": {
    "crate": "serde",
    "version": "1.0.219",
    "path": "serde::Serializer::serialize_struct"
  }
}

// rustdocs_list
{
  "name": "rustdocs_list",
  "description": "Browse a published crate's public API by listing items, optionally filtered by kind or module prefix.",
  "input": {
    "crate": "serde",
    "version": "latest",
    "kind": "trait",                // optional
    "module_prefix": "serde::de"    // optional
  }
}

// rustdocs_status
{
  "name": "rustdocs_status",
  "description": "Report which Rust crates and versions are cached locally for documentation lookup.",
  "input": { "crate": "serde", "version": "latest" }   // both optional
}
```

`sync` is **CLI-only**, not exposed as an MCP tool. Walking a `Cargo.lock` and fetching dozens-to-hundreds of crates can run for minutes; that's the wrong shape for an MCP request/response and would be a poor experience for an LLM caller.

Resource template:

```
rustbert://<crate>/<version>/<qualified_path>
```

A direct read of `rustbert://serde/1.0.219/serde::Serializer::serialize_struct` returns the item with signature + docstring + source slice.

## 3. `rustbert sync` — proactive dependency fetch

This is the headline command for users embedded in a real Rust project.

### 3.1 What it does

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

| Flag               | Default    | Effect                                                                                                                                                                                          |
| ------------------ | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--lock PATH`      | discovered | Use a specific `Cargo.lock` instead of walking up from cwd                                                                                                                                      |
| `--jobs N`         | `4`        | Parallel fetch concurrency. Capped to crates.io's polite ceiling                                                                                                                                |
| `--force`          | off        | Re-fetch even cached `(crate, version)` pairs                                                                                                                                                   |
| `--no-embed`       | off        | Escape hatch: download + parse + index but skip ColBERT embedding (deferred to first search). Embedding is part of `sync` by default — the whole point is "instant search after sync finishes." |
| `--depth N`        | unlimited  | Only fetch packages at depth ≤ N in the dep graph (1 = direct deps)                                                                                                                             |
| `--exclude GLOB`   | none       | Skip crates matching the pattern (repeatable)                                                                                                                                                   |
| `--include-dev`    | on         | Include dev-dependencies (mirrors what's in `Cargo.lock`)                                                                                                                                       |
| `--dry-run`        | off        | Print the plan without fetching                                                                                                                                                                 |
| `--manifest PATH`  | none       | Alternative to `--lock`: parse `Cargo.toml`, run `cargo metadata --offline` to enumerate                                                                                                        |
| `--registry URL`   | crates.io  | Custom registry base URL (matches Cargo's mirror story)                                                                                                                                         |
| `--workspace-only` | off        | For workspaces, only sync deps of workspace members, not deps of every package found                                                                                                            |

### 3.3 Concurrency, retries, rate limits

- Default concurrency is 4. Crates.io publishes no hard limit, but we keep it polite: 4 parallel downloads, each ≤ a few MB, is well under any reasonable threshold.
- 429 / 503 → exponential backoff with jitter, up to 3 retries.
- Failures don't abort the run. The final report lists per-crate outcomes and `rustbert sync --resume` retries only the failures (it does **not** re-resolve `latest`-style entries — see [`rustbert refresh`](#36-refreshing-latest-entries) for that).
- Network is gated by `RUSTBERT_OFFLINE=1`: in that mode, sync uses only what's already cached and reports the gap.

### 3.4 What sync deliberately doesn't do

- **Doesn't index your project's own source.** That's a future feature (Phase 4). v1 sync is strictly about deps on crates.io.
- **Doesn't follow git/path deps.** They aren't on crates.io. The plan summary lists them as "skipped — non-crates.io source" so users see what's missing.
- **Doesn't write to `Cargo.toml` or `Cargo.lock`.** Read-only.
- **Doesn't hold the lockfile open** during the fetch, so it's safe to run `cargo build` concurrently.
- **Doesn't dedupe across feature flags.** A crate appears once per resolved version regardless of which features are active in different parts of your tree.

### 3.5 Workspace handling

For workspace projects (multiple `Cargo.toml` members under one `Cargo.lock`):

- `rustbert sync` defaults to "every dep in the lockfile, regardless of which member uses it."
- `rustbert sync --workspace-only` excludes deps that are themselves workspace members.
- Multiple lockfiles in the same tree (rare) are handled one at a time — pass `--lock` explicitly.

### 3.6 Refreshing `latest` entries

`rustbert sync` and `rustbert sync --resume` both leave previously cached `latest`-resolved entries alone — concrete versions are immutable, and a resume run is for retrying _failed_ work, not for chasing newer upstream releases.

A separate command handles the "newer versions may exist upstream" case:

```bash
rustbert refresh                 # re-resolve every cached "latest" / semver-pattern entry
rustbert refresh serde           # only that crate
rustbert refresh --older-than 7d # only entries older than the cutoff
```

Refresh is its own command on purpose. Mixing version-rolling into `sync` would make the command's blast radius depend on how stale the cache happens to be, which is the opposite of what users want from "pre-fetch the deps of this project."

## 4. Fetch pipeline (single-crate)

This is the path used both by `rustbert sync` (per-package) and by an on-demand search miss.

### 4.1 Source

| Source            | URL pattern                                         | Format              | Pros                                           | Cons                              |
| ----------------- | --------------------------------------------------- | ------------------- | ---------------------------------------------- | --------------------------------- |
| crates.io tarball | `https://crates.io/api/v1/crates/{c}/{v}/download`  | `.crate` (gzip tar) | Always available; small; immutable per version | Source-only; needs syn parsing    |
| docs.rs JSON      | `https://docs.rs/crate/{c}/{v}/json` (when shipped) | rustdoc-types JSON  | Fully resolved trait/type info; doc links      | Coverage uneven; format versioned |

**v1 default:** crates.io tarball + `syn`. Robust, always works.

**Phase 2 enrichment:** when docs.rs JSON is available, layer it on top of the syn parse for trait-impl edges and intra-doc links.

### 4.2 Version resolution

```text
"latest"  → GET https://crates.io/api/v1/crates/{name}
            → pick max stable, non-yanked version
            → cache (name, "latest", resolved_version, fetched_at)
            → re-resolve after `latest_ttl` (default: 24h)

"^1.0"    → resolve via semver against the same JSON
"1.0.*"   → ditto
"1.0.219" → use as-is; 404 → clean error to caller
```

Concrete versions are immutable; "latest" / semver-pattern resolution is what's TTL-gated.

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

If any step fails, the entry is recorded as `failed` with a reason; subsequent calls retry on demand.

### 4.4 Module discovery

For each `mod foo;` declaration:

1. `#[path = "..."]`, if present.
2. `parent_dir/foo.rs`.
3. `parent_dir/foo/mod.rs`.
4. Otherwise log a per-module load failure and continue.

`mod foo { … }` recurses into the AST without filesystem lookup.

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

Each cached `(crate, version)` is stored as a synthetic collection:

```
rustbert:<crate>@<resolved_version>
```

rustbert owns its own data directory, so there's no collision risk with docbert's user-facing collections. The name is still visible in low-level inspections, but rustbert's CLI/MCP surface never asks the user to type one — the user always works with `(crate, version)` directly.

### 5.4 `cfg` and re-exports

- `#[cfg(...)]` items: indexed unconditionally; the predicate is captured in `attrs` for future filtering.
- `pub use` re-exports: indexed only at their _original_ path; the alias path is recorded as metadata, not as a separate item.

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

- Concrete versions are immutable. Never re-fetched without `--force`.
- `latest` / semver patterns resolve to a concrete version, which is what's cached. Re-resolution past TTL may produce a new entry; the old one isn't invalidated.
- Cache entries with `status = failed` retry on next access.

### 6.3 Eviction

v1 has no automatic eviction:

- `rustbert evict <crate>[@<version>]` removes a specific entry.
- `rustbert evict --all` clears everything.

A future LRU policy gated by `cache.max_bytes` can layer on if the cache grows uncomfortably large in practice.

## 7. Dependency survey

### 7.1 Source parsing

- **`syn` (with `full`)**: v1 baseline. Stable toolchain, rich AST, robust. Already in the docbert lockfile transitively.
- **`rustdoc-types`**: Phase 2 enrichment when docs.rs JSON is available.
- **`tree-sitter-rust`**: skipped. No win over syn.
- **`ra_ap_*`**: skipped. IDE infrastructure, not a search-index dep.

### 7.2 Lockfile parsing

`cargo-lock` (the official RustSec crate) reads `Cargo.lock` into a typed model with package, version, source, dependencies, and checksum fields. It's the right tool for `rustbert sync`'s discovery phase.

`cargo_metadata` is still useful per-crate after extraction (to discover the `src/lib.rs` entry point in a downloaded tarball), but for the high-level dependency walk we go straight from `Cargo.lock` because it's faster and doesn't require a working build environment.

### 7.3 Network + archive

- `reqwest` (rustls, no default features) — HTTP.
- `flate2` — gzip.
- `tar` — tarball extraction.
- `semver` — version resolution.
- `sha2` — checksum verification.

### 7.4 Other

- `serde` / `serde_json` — crates.io API, metadata blobs.
- `proc-macro2` (`span-locations`) — byte spans.
- `quote` + `prettyplease` — signature rendering.
- `pulldown-cmark` — doc-comment markdown parsing.
- `tokio` — async runtime (HTTP, parallelism).
- `clap` — CLI parsing.
- `rmcp` — MCP server (matches docbert's choice).
- `indicatif` — progress bars for `rustbert sync`.
- `tracing` / `tracing-subscriber` — logging.
- `xdg` — data dir resolution.
- `thiserror` — error definitions.

### 7.5 Recommended manifest

```toml
[package]
name    = "rustbert"
version = "0.1.0"
edition = "2024"

[dependencies]
# `docbert-core` reference is deployment-dependent: a `path = "../docbert-core"`
# entry if rustbert is a workspace sibling, a `git = "..."` entry if it lives in
# its own repo without a published core crate, or a `version = "..."` entry once
# `docbert-core` is published. The choice is up to whoever ships rustbert.
docbert-core    = { path = "../docbert-core" }

# CLI + runtime
clap            = { version = "4.6", features = ["derive"] }
tokio           = { version = "1", features = ["rt-multi-thread", "fs", "macros"] }
rmcp            = { version = "1.5", features = ["transport-io", "server"] }
tracing         = "0.1"
tracing-subscriber = "0.3"
xdg             = "3"
indicatif       = "0.18"

# Network + archive
reqwest         = { version = "0.13", default-features = false, features = ["rustls-tls", "json", "stream"] }
flate2          = "1"
tar             = "0.4"
semver          = "1"
sha2            = "0.11"

# Cargo / Rust parsing
cargo-lock      = "10"
cargo_metadata  = "0.20"
syn             = { version = "2", features = ["full", "extra-traits", "visit"] }
proc-macro2     = { version = "1", features = ["span-locations"] }
quote           = "1"
prettyplease    = "0.2"
pulldown-cmark  = "0.13"

# Serialization + errors
serde           = { version = "1", features = ["derive"] }
serde_json      = "1"
schemars        = "1.2"             # MCP tool input schemas, matches docbert
thiserror       = "2"

[features]
default = []
mkl        = ["docbert-core/mkl"]
accelerate = ["docbert-core/accelerate"]
metal      = ["docbert-core/metal"]
cuda       = ["docbert-core/cuda"]
```

`syn` / `proc-macro2` / `quote` are already in the workspace lockfile transitively, so the marginal compile cost is small.

## 8. Integration with `docbert-core`

rustbert reuses `docbert-core` as a library and contributes nothing back into core for v1:

- `SearchDocument`, `DocumentId`, `ChunkPlan` — used as the lowering target.
- `SearchIndex`, `EmbeddingDb`, `ConfigDb`, `DataDir` — used for storage.
- `ModelManager` — used for ColBERT inference.
- `docbert_core::search::run` — used for the search backend, with a `kind` filter applied on the metadata blob. The filter is implemented in rustbert (post-search rerank/filter), not in core, so docbert itself stays unchanged.

If post-search filtering proves expensive at scale, a follow-up could land a `kind` filter inside `docbert-core::search::run` that operates on a Tantivy field. For v1, post-filter is fine.

## 9. Failure modes

- **Crate doesn't exist (404):** clean `CrateNotFound { name }` error.
- **Version doesn't exist:** `VersionNotFound { name, requested, resolvable }`.
- **Yanked version:** allowed but flagged in response metadata. Forensic queries still work.
- **Network failure:** retry with backoff; eventually `FetchFailed`.
- **Tarball checksum mismatch:** hard failure. We do not parse possibly-tampered source.
- **Per-file parse failure:** record as a load failure; continue siblings.
- **Cache corruption:** detect via the `status` column; partial entries are evicted on next access.
- **Disk full mid-fetch:** roll back the partial entry; surface `CacheWriteFailed`.
- **`Cargo.lock` malformed (sync):** abort the sync with a clear error before any fetch.
- **`Cargo.lock` not found (sync):** clean error, suggest `--lock`.

## 10. Phasing

**Phase 1 — v1 MVP**

- New crate `rustbert` with the manifest in §7.5; deployment topology (separate repo or sibling workspace member) is open and doesn't affect the user model.
- CLI: `search`, `get`, `list`, `status`, `evict`, `fetch`, `sync`, `refresh`, `mcp`.
- MCP tools: `rustdocs_search`, `rustdocs_get`, `rustdocs_list`, `rustdocs_status`.
- crates.io tarball ingestion via `reqwest` + `flate2` + `tar`.
- `Cargo.lock` walking via `cargo-lock`.
- Synthetic collection storage in rustbert's own data dir.

**Phase 2 — docs.rs enrichment**

- Optional rustdoc JSON merge atop the syn parse.
- Trait-impl edges and intra-doc link resolution.
- Falls back to syn-only when JSON is unavailable.

**Phase 3 — query ergonomics**

- `kind` filter pushed into `docbert-core::search::run` if post-filter cost matters.
- Web UI / API at `rustbert web`.
- `rustbert://` MCP resource template implementation polish.

**Phase 4 — your own crate too**

- Optional: index the host project's own source (the case rustbert deliberately skips in v1) so a single search hits both your project and its deps.
- Custom registries / sparse / git protocols for non-crates.io sources.

## 11. Resolved decisions

These were on the table during design discussion and are now baked in:

- **Embedding is part of `rustbert sync` by default.** The whole point of sync is "instant search after it finishes." `--no-embed` exists as an escape hatch for users who explicitly want to defer the embedding cost, but it is not the default.
- **`rustbert sync --resume` does not refresh `latest` entries.** Resume is for retrying _failed_ work in the previous run. Re-resolving `latest` / semver-pattern entries against upstream is a separate command, `rustbert refresh` (see §3.6).
- **rustbert and docbert are separate projects.** No automatic cross-routing between docbert's chat agent and rustbert's MCP tools. Users wire up the two MCP servers independently if they want both.

## 12. Open questions

1. **Concurrency cap.** `--jobs 4` is conservative. Tune after benchmarking on a real `Cargo.lock` with ~150 crates.
2. **`latest` TTL.** 24h is the proposed default for "latest" / semver-pattern resolution. Open to tuning based on how stale results feel in practice.
3. **Per-project data dirs.** rustbert defaults to `~/.local/share/rustbert/`. Should we offer a `--data-dir` flag for per-project caches? Useful for strict isolation; complicates cross-project sharing.
4. **Should sync index dev-dependencies?** Default is yes (mirrors `Cargo.lock`). `--include-dev=false` opts out. Maybe the default should flip if dev-deps add too much noise.
5. **`cfg`-gated items.** v1 indexes everything. If search drowns in platform-specific items, add a config knob to filter to the host platform.

## 13. Risks

- **Network dependency for new lookups.** No internet → only cached crates work. Mitigation: aggressive caching, `RUSTBERT_OFFLINE=1`, `rustbert sync` to pre-warm.
- **crates.io rate limits.** Single-user tool, low volume, polite concurrency. Respect `Retry-After` and back off on 429.
- **Sync run time on large lockfiles.** A 300-crate lockfile with full embedding can be measured in minutes. Mitigation: progress bars, `--no-embed`, `--depth`, parallelism.
- **Cache size growth.** Tarballs are typically a few MB; embeddings can dominate. A future LRU eviction policy mitigates.
- **rustdoc-types schema churn (Phase 2).** Format is versioned; pin it and feature-gate JSON enrichment.
- **Surface duplication with docbert.** `rustbert mcp` and `docbert mcp` coexist; users running both expose two MCP servers to their editor. Naming and tool descriptions must make the split obvious so an LLM caller picks the right one.

## Related references

- [`architecture.md`](./architecture.md)
- [`pipeline.md`](./pipeline.md)
- [`dependencies.md`](./dependencies.md)
- [`storage.md`](./storage.md)
- [`mcp.md`](./mcp.md)
