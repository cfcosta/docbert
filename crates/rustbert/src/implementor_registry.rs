//! Workspace-wide registry of `impl Trait for Type` sites.
//!
//! Each ingest contributes its crate's [`TraitImplementor`] records;
//! the registry stores them in a single JSON file at the data-dir root
//! and lets readers look up "who implements this trait" at trait-page
//! display time. That's how cross-crate impls show up on a trait's
//! page even when the trait lives in a different crate (`candle-core`
//! defines `Module`, `docbert-pylate` implements it — the registry
//! joins them).
//!
//! Storage format is a flat list of `(crate, version, implementor)`
//! triples — small enough that the few-KB file rewrites on every
//! ingest stay cheap, and uniform enough that tracing through it is
//! trivial. Replacing a crate's contributions is a partition by
//! `(crate, version)`.
//!
//! Lookup is by exact trait path first, then by trait-name fallback
//! (last `::`-separated segment). The fallback covers the common case
//! where `impl Display for Foo` was authored against `std::fmt::Display`
//! but the trait the page belongs to is registered under the canonical
//! path the trait's own crate produced — both spellings end up in
//! play because external code refers to traits by their `use`-able
//! path while internal code uses the file path.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{error::Result, parse::TraitImplementor};

const REGISTRY_FILE: &str = "implementors.json";

/// One `(crate, version, implementor)` row in the registry. The
/// crate identity is stored alongside each implementor so a re-index
/// of the same crate-version can drop its prior contributions
/// before adding new ones.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisteredImplementor {
    pub from_crate: String,
    pub from_version: semver::Version,
    pub implementor: TraitImplementor,
}

/// On-disk layout. A flat list keyed by trait path keeps the JSON
/// small and roundtrips cleanly even when readers don't know every
/// trait the index has seen.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct OnDisk {
    /// `trait_path` (resolved or as-authored when no `use` statement
    /// applied) → records.
    entries: std::collections::HashMap<String, Vec<RegisteredImplementor>>,
}

#[derive(Debug, Clone)]
pub struct ImplementorRegistry {
    path: PathBuf,
    state: OnDisk,
}

impl ImplementorRegistry {
    /// Open (or create) the registry rooted at `data_dir`. Creates
    /// the parent directory if missing.
    pub fn open(data_dir: &Path) -> Result<Self> {
        fs::create_dir_all(data_dir)?;
        let path = data_dir.join(REGISTRY_FILE);
        let state = if path.exists() {
            let bytes = fs::read(&path)?;
            serde_json::from_slice::<OnDisk>(&bytes).map_err(|e| {
                crate::error::Error::Cache(format!(
                    "implementor registry parse: {e}"
                ))
            })?
        } else {
            OnDisk::default()
        };
        Ok(Self { path, state })
    }

    /// Replace every record contributed by `(crate_name, version)`
    /// with the supplied list, then persist. Idempotent on re-index:
    /// the same crate's prior contributions don't accumulate.
    pub fn set_for_crate(
        &mut self,
        crate_name: &str,
        version: &semver::Version,
        implementors: &[TraitImplementor],
    ) -> Result<()> {
        // Drop prior contributions from this exact `(crate, version)`
        // pair across every key, then strip empty buckets so the file
        // doesn't grow lopsided over time.
        for records in self.state.entries.values_mut() {
            records.retain(|r| {
                !(r.from_crate == crate_name && &r.from_version == version)
            });
        }
        self.state.entries.retain(|_, v| !v.is_empty());

        for impl_ in implementors {
            let key = impl_.trait_path.clone();
            self.state.entries.entry(key).or_default().push(
                RegisteredImplementor {
                    from_crate: crate_name.to_string(),
                    from_version: version.clone(),
                    implementor: impl_.clone(),
                },
            );
        }
        self.persist()
    }

    /// Look up implementors for a trait, matching by full path first
    /// and falling back to last-segment match so a trait re-exported
    /// under a different path (e.g. `serde::Serialize` aliasing
    /// `serde::ser::Serialize`) still finds its impls.
    pub fn lookup(&self, trait_full_path: &str) -> Vec<&RegisteredImplementor> {
        let mut out: Vec<&RegisteredImplementor> = Vec::new();
        if let Some(records) = self.state.entries.get(trait_full_path) {
            out.extend(records.iter());
        }
        let last_segment = trait_full_path
            .rsplit_once("::")
            .map(|(_, t)| t)
            .unwrap_or(trait_full_path);
        for (key, records) in &self.state.entries {
            if key == trait_full_path {
                continue;
            }
            let key_last = key.rsplit_once("::").map(|(_, t)| t).unwrap_or(key);
            if key_last == last_segment {
                out.extend(records.iter());
            }
        }
        out
    }

    fn persist(&self) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(&self.state).map_err(|e| {
            crate::error::Error::Cache(format!(
                "implementor registry serialise: {e}"
            ))
        })?;
        // Atomic rewrite: write to a temp file then rename so a
        // crash mid-write can't leave the registry truncated.
        let tmp = self.path.with_extension("json.tmp");
        fs::write(&tmp, bytes)?;
        fs::rename(&tmp, &self.path)?;
        Ok(())
    }
}

/// Render the rustdoc-style "Implementors" block that gets appended
/// to a trait item's body. Empty implementors list returns `None` so
/// the caller can skip the section entirely rather than write a
/// header followed by nothing.
pub fn render_implementors_block(
    implementors: &[&RegisteredImplementor],
) -> Option<String> {
    if implementors.is_empty() {
        return None;
    }
    let mut out = String::new();
    out.push_str("\n\n# Implementors\n\n");
    for record in implementors {
        out.push_str(&format!(
            "- `{}` (in `{}@{}`)\n",
            record.implementor.impl_signature,
            record.from_crate,
            record.from_version
        ));
        for sig in &record.implementor.method_signatures {
            out.push_str(&format!("    - `{sig}`\n"));
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;

    fn implementor(trait_path: &str, self_type: &str) -> TraitImplementor {
        TraitImplementor {
            trait_path: trait_path.to_string(),
            self_type: self_type.to_string(),
            impl_signature: format!("impl {trait_path} for {self_type}"),
            method_signatures: vec![format!("fn fmt(&self)")],
            source_file: PathBuf::from("src/lib.rs"),
            line_start: 1,
            line_end: 5,
        }
    }

    #[test]
    fn set_for_crate_then_lookup_by_full_path() {
        let tmp = TempDir::new().unwrap();
        let mut reg = ImplementorRegistry::open(tmp.path()).unwrap();
        let v = semver::Version::new(1, 0, 0);
        reg.set_for_crate(
            "candle-core",
            &v,
            &[implementor("std::fmt::Display", "Tensor")],
        )
        .unwrap();
        let hits = reg.lookup("std::fmt::Display");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].implementor.self_type, "Tensor");
    }

    #[test]
    fn lookup_falls_back_to_last_segment() {
        let tmp = TempDir::new().unwrap();
        let mut reg = ImplementorRegistry::open(tmp.path()).unwrap();
        let v = semver::Version::new(1, 0, 0);
        // Stored under `std::fmt::Display` (the resolved path), but
        // the trait page lives in a re-exported alias path; the
        // last-segment fallback picks it up regardless.
        reg.set_for_crate(
            "demo",
            &v,
            &[implementor("std::fmt::Display", "Foo")],
        )
        .unwrap();
        let hits = reg.lookup("alt::path::Display");
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn re_indexing_replaces_prior_contributions() {
        let tmp = TempDir::new().unwrap();
        let mut reg = ImplementorRegistry::open(tmp.path()).unwrap();
        let v = semver::Version::new(1, 0, 0);

        reg.set_for_crate(
            "demo",
            &v,
            &[implementor("std::fmt::Display", "Foo")],
        )
        .unwrap();
        // Same crate-version contributes a different set on re-index.
        reg.set_for_crate("demo", &v, &[implementor("std::fmt::Debug", "Bar")])
            .unwrap();

        let display_hits = reg.lookup("std::fmt::Display");
        let debug_hits = reg.lookup("std::fmt::Debug");
        assert_eq!(
            display_hits.len(),
            0,
            "old contribution should be dropped on re-index",
        );
        assert_eq!(debug_hits.len(), 1);
    }

    #[test]
    fn re_indexing_other_crate_does_not_disturb_first() {
        let tmp = TempDir::new().unwrap();
        let mut reg = ImplementorRegistry::open(tmp.path()).unwrap();
        let v = semver::Version::new(1, 0, 0);
        reg.set_for_crate(
            "alpha",
            &v,
            &[implementor("std::fmt::Display", "A")],
        )
        .unwrap();
        reg.set_for_crate("beta", &v, &[implementor("std::fmt::Display", "B")])
            .unwrap();

        let hits = reg.lookup("std::fmt::Display");
        assert_eq!(hits.len(), 2);
        let crates: Vec<&str> =
            hits.iter().map(|h| h.from_crate.as_str()).collect();
        assert!(crates.contains(&"alpha"));
        assert!(crates.contains(&"beta"));
    }

    #[test]
    fn round_trip_through_disk() {
        let tmp = TempDir::new().unwrap();
        {
            let mut reg = ImplementorRegistry::open(tmp.path()).unwrap();
            let v = semver::Version::new(1, 0, 0);
            reg.set_for_crate(
                "demo",
                &v,
                &[implementor("std::fmt::Display", "Foo")],
            )
            .unwrap();
        }
        // New process opens the same dir and sees the prior writes.
        let reg = ImplementorRegistry::open(tmp.path()).unwrap();
        let hits = reg.lookup("std::fmt::Display");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].implementor.self_type, "Foo");
    }

    #[test]
    fn render_implementors_block_formats_rustdoc_style() {
        let tmp = TempDir::new().unwrap();
        let mut reg = ImplementorRegistry::open(tmp.path()).unwrap();
        let v = semver::Version::new(1, 0, 0);
        reg.set_for_crate(
            "demo",
            &v,
            &[implementor("std::fmt::Display", "Foo")],
        )
        .unwrap();
        let hits = reg.lookup("std::fmt::Display");
        let block = render_implementors_block(&hits).unwrap();
        assert!(block.contains("# Implementors"));
        assert!(block.contains("impl std::fmt::Display for Foo"));
        assert!(block.contains("fn fmt(&self)"));
        assert!(block.contains("demo@1.0.0"));
    }

    #[test]
    fn render_returns_none_for_empty_implementors() {
        assert!(render_implementors_block(&[]).is_none());
    }
}
