//! On-disk cache for parsed crate items.
//!
//! Each `(crate, resolved_version)` pair has its parsed `Vec<RustItem>`
//! stored as JSON under `<data_dir>/items/<crate>-<version>.json`. The
//! tarball and extracted source tree live under `<data_dir>/crate-cache/`
//! (managed by [`crate::data_dir`] and [`crate::extract`]).
//!
//! A small `<data_dir>/registry.json` carries cache-wide bookkeeping:
//! one entry per cached `(crate, version)` with `fetched_at` (epoch
//! seconds), `status` (`Ready` / `Failed`), and an optional resolved-
//! version mapping for `latest` lookups.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::{
    collection::SyntheticCollection,
    error::{Error, Result},
    item::RustItem,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheStatus {
    Ready,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub crate_name: String,
    pub version: semver::Version,
    pub fetched_at: u64,
    pub status: CacheStatus,
    pub item_count: usize,
}

/// Resolved-version cache for `latest` and semver-pattern lookups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedRequest {
    pub crate_name: String,
    pub requested: String,
    pub resolved_version: semver::Version,
    pub resolved_at: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RegistryFile {
    #[serde(default)]
    entries: BTreeMap<String, CacheEntry>,
    #[serde(default)]
    resolved: BTreeMap<String, ResolvedRequest>,
}

#[derive(Clone)]
pub struct CrateCache {
    data_dir: PathBuf,
}

impl CrateCache {
    pub fn new(data_dir: impl Into<PathBuf>) -> Result<Self> {
        let data_dir = data_dir.into();
        fs::create_dir_all(items_dir(&data_dir))?;
        if !registry_path(&data_dir).exists() {
            write_registry(&data_dir, &RegistryFile::default())?;
        }
        Ok(Self { data_dir })
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    pub fn store(
        &self,
        collection: &SyntheticCollection,
        items: &[RustItem],
    ) -> Result<()> {
        let path = self.items_path(collection);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_vec_pretty(items).map_err(json_err)?;
        fs::write(&path, json)?;

        let mut registry = self.read_registry()?;
        registry.entries.insert(
            entry_key(collection),
            CacheEntry {
                crate_name: collection.crate_name.clone(),
                version: collection.version.clone(),
                fetched_at: now(),
                status: CacheStatus::Ready,
                item_count: items.len(),
            },
        );
        self.write_registry(&registry)
    }

    pub fn load(
        &self,
        collection: &SyntheticCollection,
    ) -> Result<Vec<RustItem>> {
        let path = self.items_path(collection);
        let bytes = fs::read(&path)?;
        let mut items: Vec<RustItem> =
            serde_json::from_slice(&bytes).map_err(json_err)?;
        // Older caches stored the crates.io spelling (`candle-core::…`)
        // in `qualified_path`. Normalise on read so callers always
        // see the canonical Rust form regardless of when the entry was
        // written. New writes already produce the canonical form via
        // `RustItem::build_qualified_path`.
        for item in &mut items {
            item.qualified_path =
                crate::item::normalize_qualified_path(&item.qualified_path);
        }
        merge_implementors_into_traits(&self.data_dir, &mut items);
        Ok(items)
    }

    pub fn has(&self, collection: &SyntheticCollection) -> bool {
        self.items_path(collection).exists()
    }

    pub fn remove(&self, collection: &SyntheticCollection) -> Result<()> {
        let items = self.items_path(collection);
        if items.exists() {
            fs::remove_file(&items)?;
        }
        let extracted = crate::data_dir::extracted_crate_dir(
            &self.data_dir,
            &collection.crate_name,
            &collection.version,
        );
        if extracted.exists() {
            fs::remove_dir_all(&extracted)?;
        }
        let tarball = crate::data_dir::crate_tarball_path(
            &self.data_dir,
            &collection.crate_name,
            &collection.version,
        );
        if tarball.exists() {
            fs::remove_file(&tarball)?;
        }
        let mut registry = self.read_registry()?;
        registry.entries.remove(&entry_key(collection));
        self.write_registry(&registry)?;
        Ok(())
    }

    pub fn entries(&self) -> Result<Vec<CacheEntry>> {
        Ok(self.read_registry()?.entries.into_values().collect())
    }

    pub fn record_resolved(
        &self,
        crate_name: &str,
        requested: &str,
        resolved: &semver::Version,
    ) -> Result<()> {
        let mut registry = self.read_registry()?;
        registry.resolved.insert(
            resolved_key(crate_name, requested),
            ResolvedRequest {
                crate_name: crate_name.to_string(),
                requested: requested.to_string(),
                resolved_version: resolved.clone(),
                resolved_at: now(),
            },
        );
        self.write_registry(&registry)
    }

    pub fn resolved(
        &self,
        crate_name: &str,
        requested: &str,
    ) -> Result<Option<ResolvedRequest>> {
        Ok(self
            .read_registry()?
            .resolved
            .get(&resolved_key(crate_name, requested))
            .cloned())
    }

    fn items_path(&self, collection: &SyntheticCollection) -> PathBuf {
        items_dir(&self.data_dir).join(format!(
            "{}-{}.json",
            collection.crate_name, collection.version,
        ))
    }

    fn read_registry(&self) -> Result<RegistryFile> {
        read_registry(&self.data_dir)
    }

    fn write_registry(&self, registry: &RegistryFile) -> Result<()> {
        write_registry(&self.data_dir, registry)
    }
}

fn items_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("items")
}

fn registry_path(data_dir: &Path) -> PathBuf {
    data_dir.join("registry.json")
}

fn entry_key(collection: &SyntheticCollection) -> String {
    format!("{}@{}", collection.crate_name, collection.version)
}

fn resolved_key(crate_name: &str, requested: &str) -> String {
    format!("{crate_name}@{requested}")
}

fn read_registry(data_dir: &Path) -> Result<RegistryFile> {
    let path = registry_path(data_dir);
    if !path.exists() {
        return Ok(RegistryFile::default());
    }
    let bytes = fs::read(&path)?;
    serde_json::from_slice(&bytes).map_err(json_err)
}

fn write_registry(data_dir: &Path, registry: &RegistryFile) -> Result<()> {
    let path = registry_path(data_dir);
    let json = serde_json::to_vec_pretty(registry).map_err(json_err)?;
    fs::write(&path, json)?;
    Ok(())
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Append the workspace-wide implementor records to each `Trait`
/// item's body so it reads like a rustdoc trait page. Errors are
/// swallowed: the registry is informational, not load-blocking — a
/// missing or corrupt file just means traits render without the
/// "Implementors" section.
fn merge_implementors_into_traits(data_dir: &Path, items: &mut [RustItem]) {
    let Ok(registry) =
        crate::implementor_registry::ImplementorRegistry::open(data_dir)
    else {
        return;
    };
    for item in items {
        if item.kind != crate::item::RustItemKind::Trait {
            continue;
        }
        let hits = registry.lookup(&item.qualified_path);
        if let Some(block) =
            crate::implementor_registry::render_implementors_block(&hits)
        {
            item.body.push_str(&block);
        }
    }
}

fn json_err(e: serde_json::Error) -> Error {
    Error::Cache(format!("json: {e}"))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;
    use crate::item::{RustItemKind, Visibility};

    fn collection(name: &str, version: (u64, u64, u64)) -> SyntheticCollection {
        SyntheticCollection {
            crate_name: name.to_string(),
            version: semver::Version::new(version.0, version.1, version.2),
        }
    }

    fn sample_item(qpath: &str) -> RustItem {
        RustItem {
            kind: RustItemKind::Fn,
            crate_name: "x".to_string(),
            crate_version: semver::Version::new(1, 0, 0),
            module_path: vec![],
            name: Some("f".to_string()),
            qualified_path: qpath.to_string(),
            signature: "pub fn f()".to_string(),
            doc_markdown: String::new(),
            body: "pub fn f () { }".to_string(),
            source_file: PathBuf::from("src/lib.rs"),
            byte_start: 0,
            byte_len: 0,
            line_start: 1,
            line_end: 1,
            visibility: Visibility::Public,
            attrs: vec![],
        }
    }

    #[test]
    fn store_then_load_round_trips() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let coll = collection("x", (1, 0, 0));
        let items = vec![sample_item("x::a"), sample_item("x::b")];

        cache.store(&coll, &items).unwrap();
        let loaded = cache.load(&coll).unwrap();
        assert_eq!(loaded, items);
    }

    #[test]
    fn has_reports_presence() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let coll = collection("x", (1, 0, 0));
        assert!(!cache.has(&coll));
        cache.store(&coll, &[]).unwrap();
        assert!(cache.has(&coll));
    }

    #[test]
    fn remove_drops_items_and_registry_entry() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let coll = collection("x", (1, 0, 0));
        cache.store(&coll, &[sample_item("x::a")]).unwrap();
        cache.remove(&coll).unwrap();
        assert!(!cache.has(&coll));
        assert!(cache.entries().unwrap().is_empty());
    }

    #[test]
    fn registry_lists_all_stored_crates() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        cache
            .store(&collection("a", (1, 0, 0)), &[sample_item("a::x")])
            .unwrap();
        cache
            .store(&collection("b", (2, 0, 0)), &[sample_item("b::x")])
            .unwrap();

        let entries = cache.entries().unwrap();
        let names: Vec<_> =
            entries.iter().map(|e| e.crate_name.as_str()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
        for e in &entries {
            assert_eq!(e.status, CacheStatus::Ready);
            assert_eq!(e.item_count, 1);
        }
    }

    #[test]
    fn record_and_lookup_resolved_versions() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let v = semver::Version::new(1, 0, 219);
        cache.record_resolved("serde", "latest", &v).unwrap();

        let r = cache.resolved("serde", "latest").unwrap().unwrap();
        assert_eq!(r.resolved_version, v);
        assert_eq!(r.requested, "latest");
        assert!(r.resolved_at > 0);

        assert!(cache.resolved("serde", "missing").unwrap().is_none());
    }

    #[test]
    fn store_overwrites_existing_items() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let coll = collection("x", (1, 0, 0));
        cache.store(&coll, &[sample_item("x::a")]).unwrap();
        cache
            .store(&coll, &[sample_item("x::b"), sample_item("x::c")])
            .unwrap();
        let loaded = cache.load(&coll).unwrap();
        assert_eq!(loaded.len(), 2);
        let entries = cache.entries().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].item_count, 2);
    }
}
