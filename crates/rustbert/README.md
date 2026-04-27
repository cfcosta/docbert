# rustbert

Rust crate docs lookup — fetch and search published Rust APIs without registering anything.

## What it does

Give it a crate name and an optional version. It fetches the source from crates.io, parses every public item with `syn`, and serves item-level search and retrieval through:

- a **CLI** for one-off lookups and `Cargo.lock`-driven pre-fetching
- an **MCP server** for editor and agent integrations

Every operation is scoped to a single `(crate, version)`; there is no global cross-crate index.

## Quick start

```bash
# Fetch and parse a specific version
rustbert fetch serde@1.0.219

# Search inside a crate (auto-fetches on first hit)
rustbert search serde "Serializer" --kind trait

# Print one item by qualified path
rustbert get serde "serde::Serializer::serialize_struct"

# List items, filtered by kind / module
rustbert list serde --kind trait --module "serde::de"

# Pre-fetch every crates.io dep of a Rust project (parallel)
rustbert sync --jobs 8

# Re-resolve cached `latest` entries against upstream
rustbert refresh

# Cache state
rustbert status
rustbert evict serde@1.0.0
rustbert evict --all

# MCP server on stdio
rustbert mcp
```

Spec strings parse the same shape every time: `name`, `name@1.2.3`, `name@^1.0`, `name@latest`, `name@*`. `latest` and `*` are sentinels for "max stable, non-yanked version".

## What's in scope

1. Lookups by `(crate, version)` with **no registration step**.
2. Item-level results: one hit per `fn` / `struct` / `enum` / `union` / `trait` / `impl` / `mod` / `const` / `static` / `type alias` / `macro_rules!`, with signature, docstring, qualified path, and source span.
3. Search ranks by token overlap weighted toward the qualified path; case-insensitive substring matching.
4. `rustbert sync` walks a `Cargo.lock` and pre-fetches every crates.io dep in parallel so the working set is hot before the first search runs.
5. `rustbert refresh` re-resolves cached `latest` entries against upstream without re-downloading.

## What's out of scope (for v1)

- Indexing your project's own source. `rustbert sync` indexes the project's _dependencies_, not the project itself. A "look at my own crate" feature can layer on later — it's a separate scope.
- Indexing the whole crates.io corpus. Fetches are demand-driven or scoped to a project's lockfile.
- Type resolution / cross-crate `pub use` chasing.
- Macro expansion. Items synthesized by macros are invisible to source-level parsing.

## How it stores things

```text
~/.local/share/rustbert/                 # or $RUSTBERT_DATA_DIR
├── registry.json                        # cache bookkeeping + resolved-version pins
├── items/<crate>-<version>.json         # parsed RustItem list per crate
└── crate-cache/
    ├── <crate>-<version>.crate          # raw downloaded tarball
    └── <crate>-<version>/               # extracted source tree
```

Concrete versions (`serde@1.0.219`) are immutable — never re-fetched without `--force`. `latest` resolutions live in the registry; `rustbert refresh` re-checks them on demand.

## MCP tools

The `rustbert mcp` server speaks JSON-RPC 2.0 on stdio and exposes four tools:

- `rustbert_search(crate, version?, query, kind?, module_prefix?, limit?)`
- `rustbert_get(crate, version?, path)`
- `rustbert_list(crate, version?, kind?, module_prefix?, limit?)`
- `rustbert_status(crate?)`

Plus the standard `initialize` / `tools/list` / `tools/call` lifecycle. `rustbert_sync` is intentionally CLI-only — lockfile walks can run for minutes, which is the wrong shape for an MCP request.

Wire it up in your editor / agent config the same way you'd wire any other stdio MCP server (point it at the `rustbert` binary with `mcp` as the only arg).

## Configuration

| Variable                 | Purpose                                                  |
| ------------------------ | -------------------------------------------------------- |
| `RUSTBERT_DATA_DIR`      | Override the data directory                              |
| `RUSTBERT_LOG`           | tracing-subscriber filter (default `warn,rustbert=info`) |
| `XDG_DATA_HOME` / `HOME` | Standard XDG fallbacks                                   |

CLI flag `--data-dir <path>` takes precedence over both.

## Status

v1 implementation — lexical search only. The design (`docs/rustbert.md`) sketches a Phase 2 that layers ColBERT semantic ranking via `docbert-core` on top of the existing in-memory search; the lowering layer (`lowering::SearchDocument`) is already shaped to match `docbert_core::SearchDocument` so the swap is mechanical.

## License

MIT OR Apache-2.0
