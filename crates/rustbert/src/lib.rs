//! rustbert — Rust crate docs lookup.
//!
//! See `docs/rustbert.md` in the repository root for the design document.
//! This crate is in early implementation; only foundational pure-logic
//! modules are wired up so far.

pub mod collection;
pub mod crate_ref;
pub mod crates_io;
pub mod error;
pub mod fetcher;
pub mod lockfile;

pub use collection::SyntheticCollection;
pub use crate_ref::{CrateRef, VersionSpec};
pub use crates_io::{CrateMetadata, CratesIoApi, PublishedVersion};
pub use error::{Error, Result};
pub use fetcher::{FakeFetcher, Fetcher};
pub use lockfile::{LockedCrate, crates_io_packages_from_str};
