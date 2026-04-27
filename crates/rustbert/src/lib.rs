//! rustbert — Rust crate docs lookup.
//!
//! See `docs/rustbert.md` in the repository root for the design document.
//! This crate is in early implementation; only foundational pure-logic
//! modules are wired up so far.

//! rustbert — Rust crate docs lookup.
//!
//! A standalone tool that fetches published Rust crates from
//! crates.io, parses their public APIs with `syn`, and serves
//! item-level lookup over a CLI and an MCP server.
//!
//! See `docs/rustbert.md` in the repository root for the full design.
//!
//! ## Module map
//!
//! - [`crate_ref`] — parse `name[@version]` user input.
//! - [`crates_io`] — typed crates.io API client.
//! - [`fetcher`] — async HTTP fetcher trait + in-memory fake.
//! - [`reqwest_fetcher`] — production reqwest-backed fetcher.
//! - [`download`] — verified `.crate` tarball download.
//! - [`extract`] — gzipped-tar extraction with path-traversal safety.
//! - [`resolver`] — `VersionSpec` → concrete `semver::Version`.
//! - [`item`] — the [`item::RustItem`] data model.
//! - [`parse`] — per-file syn item visitor.
//! - [`module_discovery`] — `mod foo;` → file resolution.
//! - [`crate_walker`] — top-level "crate root → items" walker.
//! - [`lockfile`] — extract crates.io packages from `Cargo.lock`.
//! - [`data_dir`] — XDG-aware data directory + cache layout.
//! - [`cache`] — JSON-backed item cache + registry.
//! - [`ingestion`] — fetch + parse + store orchestrator.
//! - [`search`] — in-memory search/get/list against cached items.
//! - [`sync`] — Cargo.lock walker + parallel batch ingest.
//! - [`mcp`] — JSON-RPC MCP server over stdio.
//! - [`lowering`] — `RustItem` → `SearchDocument`-shaped record
//!   (mirrors `docbert_core::SearchDocument`).
//! - [`collection`] — synthetic `rustbert:<crate>@<version>` naming.

pub mod cache;
pub mod collection;
pub mod crate_ref;
pub mod crate_walker;
pub mod crates_io;
pub mod data_dir;
pub mod docs_rs;
pub mod download;
pub mod error;
pub mod extract;
pub mod fetcher;
pub mod host_project;
pub mod indexer;
pub mod ingestion;
pub mod item;
pub mod lockfile;
pub mod lowering;
pub mod mcp;
pub mod module_discovery;
pub mod parse;
pub mod reqwest_fetcher;
pub mod resolver;
pub mod rustdoc_merge;
pub mod search;
pub mod sync;

pub use collection::SyntheticCollection;
pub use crate_ref::{CrateRef, VersionSpec};
pub use crates_io::{CrateMetadata, CratesIoApi, PublishedVersion};
pub use error::{Error, Result};
pub use fetcher::{FakeFetcher, Fetcher};
pub use item::{RustItem, RustItemKind, Visibility};
pub use lockfile::{LockedCrate, crates_io_packages_from_str};
pub use reqwest_fetcher::ReqwestFetcher;
pub use resolver::{Resolution, resolve};
