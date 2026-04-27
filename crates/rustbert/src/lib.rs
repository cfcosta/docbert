//! rustbert — Rust crate docs lookup.
//!
//! See `docs/rustbert.md` in the repository root for the design document.
//! This crate is in early implementation; only foundational pure-logic
//! modules are wired up so far.

pub mod cache;
pub mod collection;
pub mod crate_ref;
pub mod crate_walker;
pub mod crates_io;
pub mod data_dir;
pub mod download;
pub mod error;
pub mod extract;
pub mod fetcher;
pub mod ingestion;
pub mod item;
pub mod lockfile;
pub mod lowering;
pub mod module_discovery;
pub mod parse;
pub mod reqwest_fetcher;
pub mod resolver;
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
