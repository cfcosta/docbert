# rustbert

rustbert is a lookup tool for Rust crate docs. It fetches and searches Rust APIs from crates.io. It is not necessary to prepare a crate first.

## What it does

You give rustbert a crate name and an optional version. It fetches the source from crates.io. It parses each public item with `syn`. Then it gives item-level search and retrieval through these two:

- A **CLI** for one-off lookups and `Cargo.lock`-driven pre-fetching
- An **MCP server** for editor and agent integration.

Each operation uses only one `(crate, version)`. There is no cross-crate index.

## Quick start

```bash
# Fetch and parse a specific version
rustbert fetch serde@1.0.219

# Search inside a crate (auto-fetches on first hit, BM25 + ColBERT hybrid)
rustbert search serde "Serializer" --kind trait

# Print one item by qualified path
rustbert get serde "serde::Serializer::serialize_struct"

# List items, filtered by kind / module
rustbert list serde --kind trait --module "serde::de"

# Pre-fetch every crates.io dep of a Rust project (parallel + embed + PLAID rebuild)
rustbert sync --jobs 8

# Lexical-only sync (skip ColBERT embedding, faster but no semantic ranking)
rustbert sync --no-embed

# Re-resolve cached `latest` entries against upstream
rustbert refresh

# Index a local Cargo project's source
rustbert index .

# Cache state
rustbert status
rustbert evict serde@1.0.0
rustbert evict --all

# MCP server on stdio
rustbert mcp
```

Spec strings parse to the same shape each time: `name`, `name@1.2.3`, `name@^1.0`, `name@latest`, `name@*`. `latest` and `*` select the maximum stable version that is not yanked.

## What is in scope

1. Lookups by `(crate, version)`. It is not necessary to prepare a crate first.
2. rustbert gives item-level results. Each result is one hit for a `fn`, `struct`, `enum`, `union`, `trait`, `impl`, `mod`, `const`, `static`, `type alias`, or `macro_rules!`. Each hit has the signature, docstring, qualified path, and source span.
3. Search ranks the items by token overlap, and overlap in the qualified path is more important. The substring search is case-insensitive.
4. `rustbert sync` reads a `Cargo.lock`. It pre-fetches all the crates.io dependencies at the same time. Thus the working set is available before the first search.
5. `rustbert refresh` resolves the cached `latest` entries again. It uses the upstream registry, and does not download the crates again.

## What is out of scope

- rustbert does not index the full crates.io corpus. It fetches only when necessary, or for the crates in a project lockfile.
- rustbert does not do type resolution. It does not resolve cross-crate `pub use` re-exports.
- rustbert does not do macro expansion. Source-level parsing cannot find the items that macros make.

## How it stores things

```text
~/.local/share/rustbert/                 # or $RUSTBERT_DATA_DIR
├── registry.json                        # cache bookkeeping + resolved-version pins
├── items/<crate>-<version>.json         # parsed RustItem list per crate
└── crate-cache/
    ├── <crate>-<version>.crate          # raw downloaded tarball
    └── <crate>-<version>/               # extracted source tree
```

A version number (for example, `serde@1.0.219`) does not change. Without `--force`, rustbert does not fetch it again. The registry holds the `latest` resolutions. `rustbert refresh` examines them again when necessary.

## MCP tools

The `rustbert mcp` server uses JSON-RPC 2.0 on stdio. It gives four tools:

- `search(crate, version?, query, kind?, module_prefix?, limit?)` — searches a crate's public API by query.
- `get(crate, version?, path)` — gives the full rustdoc for one item by qualified path.
- `list(crate, version?, kind?, module_prefix?, limit?)` — shows the items in a crate.
- `status(crate?)` — shows which crates and versions are in the local cache.

The server also has the standard `initialize`, `tools/list`, and `tools/call` lifecycle. `sync` is CLI-only. The lockfile operation can continue for minutes. This is not correct for an MCP request.

Configure it in your editor or agent config, the same as other stdio MCP servers. Set the config to the `rustbert` binary. Use `mcp` as the only argument.

## Configuration

| Variable                 | Purpose                                                  |
| ------------------------ | -------------------------------------------------------- |
| `RUSTBERT_DATA_DIR`      | Override the data directory                              |
| `RUSTBERT_LOG`           | tracing-subscriber filter (default `warn,rustbert=info`) |
| `XDG_DATA_HOME` / `HOME` | Standard XDG fallbacks                                   |

The CLI flag `--data-dir <path>` overrides the two variables.

## What is in scope (continued)

- rustbert can index your project source through `rustbert index <path>`. rustbert makes the indexed package into a collection (`<name>@<version>`). Then `search`, `get`, and `list` operate on it the same as on fetched crates. rustbert cannot index workspace roots that have no `[package]` table. As an alternative, the member directories are the correct targets.
- rustbert can also add the rustdoc JSON data from docs.rs, as a best-effort operation. When docs.rs has a JSON build for the `(crate, version)`, the JSON goes to `<data_dir>/items/<crate>-<version>.rustdoc.json` for downstream tools. rustbert ignores 404 or 410 responses without an error message. This response occurs for most crates, because they have no rustdoc JSON. The syn-only parsing stays the primary data source.

## How search works

`rustbert search` uses the same hybrid retrieval stack as `docbert search`:

- The lexical part uses **BM25** through Tantivy, with English stemming and fuzzy search.
- The semantic part uses **ColBERT** through `docbert-pylate` (the workspace fork of `pylate-rs`).
- **Reciprocal Rank Fusion** mixes the two ranked lists at `k=60`.
- **PLAID** compresses the ColBERT vectors for fast MaxSim retrieval.

rustbert ranks the items first. Then it uses the `--kind` and `--module` filters on the cached items. Thus the CLI gives the same results each time.

The first time you use `rustbert sync`, it starts a ColBERT model download from HuggingFace. For a fast lexical-only sync, use the `--no-embed` flag (no model download, no PLAID rebuild).

## License

MIT OR Apache-2.0
