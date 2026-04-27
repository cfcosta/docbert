//! Walk an extracted crate and emit every Rust item.
//!
//! Composes [`crate::parse`] (per-file syn visitor) with
//! [`crate::module_discovery`] (filesystem `mod foo;` resolution).
//!
//! The walker is resilient: a syn parse error in one file is recorded
//! as a [`LoadFailure`] and the rest of the crate continues to ingest.
//! Missing module files are also load failures (the user's library is
//! published broken, but we still surface what we can).

use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use crate::{
    error::{Error, Result},
    item::RustItem,
    module_discovery,
    parse::{self, PendingModule},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadFailure {
    pub source_file: PathBuf,
    pub reason: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WalkOutcome {
    pub items: Vec<RustItem>,
    pub failures: Vec<LoadFailure>,
}

/// Walk an extracted crate at `crate_root` and emit every `RustItem`
/// reachable from `lib.rs` / `main.rs`.
///
/// `crate_name` and `crate_version` flow into every produced item.
/// Returns aggregate items and per-file load failures.
pub fn walk_extracted_crate(
    crate_root: &Path,
    crate_name: &str,
    crate_version: &semver::Version,
) -> Result<WalkOutcome> {
    let entry = locate_entry_point(crate_root)?;
    let mut walker = CrateWalker {
        crate_root: crate_root.to_path_buf(),
        crate_name: crate_name.to_string(),
        crate_version: crate_version.clone(),
        visited: HashSet::new(),
        out: WalkOutcome::default(),
    };
    walker.walk_file(&entry, &[]);
    Ok(walker.out)
}

fn locate_entry_point(crate_root: &Path) -> Result<PathBuf> {
    // Prefer Cargo.toml-declared paths (`[lib].path`, `[[bin]].path`).
    // Some published crates use a flat layout (`lib.rs` at the root)
    // and configure it via `[lib]`; without honoring that, walking
    // them fails with "no entry point".
    if let Some(path) = manifest_entry_point(crate_root) {
        return Ok(path);
    }

    // Standard Cargo layout fallback.
    for candidate in ["src/lib.rs", "src/main.rs"] {
        let abs = crate_root.join(candidate);
        if abs.is_file() {
            return Ok(PathBuf::from(candidate));
        }
    }

    // Last-ditch: a `lib.rs` at the project root with no manifest
    // declaration (very old crates).
    if crate_root.join("lib.rs").is_file() {
        return Ok(PathBuf::from("lib.rs"));
    }

    Err(Error::NoEntryPoint {
        path: crate_root.display().to_string(),
    })
}

fn manifest_entry_point(crate_root: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(crate_root.join("Cargo.toml")).ok()?;
    let value: toml::Value = text.parse().ok()?;

    // [lib] path = "..."
    if let Some(p) = value
        .get("lib")
        .and_then(|t| t.get("path"))
        .and_then(toml::Value::as_str)
    {
        let candidate = PathBuf::from(p);
        if crate_root.join(&candidate).is_file() {
            return Some(candidate);
        }
    }

    // [[bin]] path = "..." — first bin wins.
    if let Some(bins) = value.get("bin").and_then(toml::Value::as_array)
        && let Some(p) = bins
            .first()
            .and_then(|b| b.get("path"))
            .and_then(toml::Value::as_str)
    {
        let candidate = PathBuf::from(p);
        if crate_root.join(&candidate).is_file() {
            return Some(candidate);
        }
    }

    None
}

struct CrateWalker {
    crate_root: PathBuf,
    crate_name: String,
    crate_version: semver::Version,
    visited: HashSet<PathBuf>,
    out: WalkOutcome,
}

impl CrateWalker {
    fn walk_file(&mut self, source_file: &Path, module_path: &[String]) {
        if !self.visited.insert(source_file.to_path_buf()) {
            return; // already walked
        }

        let abs = self.crate_root.join(source_file);
        let source_text = match fs::read_to_string(&abs) {
            Ok(s) => s,
            Err(e) => {
                self.out.failures.push(LoadFailure {
                    source_file: source_file.to_path_buf(),
                    reason: format!("read error: {e}"),
                });
                return;
            }
        };

        let outcome = match parse::parse_file(
            &self.crate_name,
            &self.crate_version,
            source_file,
            module_path,
            &source_text,
        ) {
            Ok(o) => o,
            Err(e) => {
                self.out.failures.push(LoadFailure {
                    source_file: source_file.to_path_buf(),
                    reason: e.to_string(),
                });
                return;
            }
        };

        self.out.items.extend(outcome.items);

        // Recurse into resolved external modules.
        for pending in outcome.pending_modules {
            self.resolve_and_walk(&pending);
        }
    }

    fn resolve_and_walk(&mut self, pending: &PendingModule) {
        let Some(resolved) =
            module_discovery::resolve(&self.crate_root, pending)
        else {
            self.out.failures.push(LoadFailure {
                source_file: pending.source_file.clone(),
                reason: format!(
                    "could not resolve `mod {}` from {}",
                    pending.name,
                    pending.source_file.display(),
                ),
            });
            return;
        };

        let mut child_module_path = pending.parent_module_path.clone();
        child_module_path.push(pending.name.clone());

        self.walk_file(&resolved, &child_module_path);
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::item::RustItemKind;

    fn write(root: &Path, rel: &str, contents: &str) {
        let abs = root.join(rel);
        if let Some(parent) = abs.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(abs, contents).unwrap();
    }

    fn version() -> semver::Version {
        semver::Version::new(0, 1, 0)
    }

    #[test]
    fn walks_lib_rs_only() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "src/lib.rs", "pub fn root() {}");

        let out = walk_extracted_crate(tmp.path(), "x", &version()).unwrap();
        assert_eq!(out.items.len(), 1);
        assert_eq!(out.items[0].qualified_path, "x::root");
        assert!(out.failures.is_empty());
    }

    #[test]
    fn walks_into_external_module() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "src/lib.rs", "pub mod foo;");
        write(tmp.path(), "src/foo.rs", "pub fn ping() {}");

        let out = walk_extracted_crate(tmp.path(), "x", &version()).unwrap();
        let paths: Vec<_> =
            out.items.iter().map(|i| i.qualified_path.clone()).collect();
        assert!(paths.contains(&"x::foo".to_string()));
        assert!(paths.contains(&"x::foo::ping".to_string()));
    }

    #[test]
    fn walks_three_levels_deep() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "src/lib.rs", "pub mod a;");
        write(tmp.path(), "src/a.rs", "pub mod b;");
        write(tmp.path(), "src/a/b.rs", "pub mod c;");
        write(tmp.path(), "src/a/b/c.rs", "pub fn deep() {}");

        let out = walk_extracted_crate(tmp.path(), "x", &version()).unwrap();
        let paths: Vec<_> =
            out.items.iter().map(|i| i.qualified_path.clone()).collect();
        assert!(paths.contains(&"x::a::b::c::deep".to_string()));
    }

    #[test]
    fn missing_module_becomes_load_failure() {
        let tmp = TempDir::new().unwrap();
        write(
            tmp.path(),
            "src/lib.rs",
            "pub mod missing; pub fn here() {}",
        );

        let out = walk_extracted_crate(tmp.path(), "x", &version()).unwrap();
        // The fn is still indexed.
        let paths: Vec<_> =
            out.items.iter().map(|i| i.qualified_path.clone()).collect();
        assert!(paths.contains(&"x::here".to_string()));
        // The missing mod surfaces as a failure.
        assert_eq!(out.failures.len(), 1);
        assert!(
            out.failures[0]
                .reason
                .contains("could not resolve `mod missing`"),
            "got {:?}",
            out.failures[0].reason,
        );
    }

    #[test]
    fn syntax_error_in_one_file_does_not_abort_others() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "src/lib.rs", "pub mod good; pub mod broken;");
        write(tmp.path(), "src/good.rs", "pub fn ok_fn() {}");
        write(tmp.path(), "src/broken.rs", "fn busted("); // syntax error

        let out = walk_extracted_crate(tmp.path(), "x", &version()).unwrap();
        let names: Vec<_> =
            out.items.iter().filter_map(|i| i.name.clone()).collect();
        assert!(names.contains(&"ok_fn".to_string()));
        assert_eq!(out.failures.len(), 1);
        assert_eq!(out.failures[0].source_file, PathBuf::from("src/broken.rs"),);
    }

    #[test]
    fn falls_back_to_main_rs_for_binaries() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "src/main.rs", "fn main() {}");

        let out = walk_extracted_crate(tmp.path(), "x", &version()).unwrap();
        assert!(out.items.iter().any(|i| i.kind == RustItemKind::Fn));
    }

    #[test]
    fn no_entry_point_errors() {
        let tmp = TempDir::new().unwrap();
        let err =
            walk_extracted_crate(tmp.path(), "x", &version()).unwrap_err();
        assert!(matches!(err, Error::NoEntryPoint { .. }));
    }

    #[test]
    fn flat_layout_with_explicit_lib_path() {
        // Mimics fnv@1.0.7: `lib.rs` at the root, `[lib] path = "lib.rs"`.
        let tmp = TempDir::new().unwrap();
        write(
            tmp.path(),
            "Cargo.toml",
            "[package]\nname = \"x\"\nversion = \"0.1.0\"\n\n[lib]\npath = \"lib.rs\"\n",
        );
        write(tmp.path(), "lib.rs", "pub fn root() {}");

        let out = walk_extracted_crate(tmp.path(), "x", &version()).unwrap();
        assert!(out.items.iter().any(|i| i.qualified_path == "x::root"));
    }

    #[test]
    fn flat_layout_without_lib_section_falls_back_to_root_lib_rs() {
        let tmp = TempDir::new().unwrap();
        write(
            tmp.path(),
            "Cargo.toml",
            "[package]\nname = \"x\"\nversion = \"0.1.0\"\n",
        );
        write(tmp.path(), "lib.rs", "pub fn root() {}");

        let out = walk_extracted_crate(tmp.path(), "x", &version()).unwrap();
        assert!(out.items.iter().any(|i| i.qualified_path == "x::root"));
    }

    #[test]
    fn manifest_bin_path_is_honored() {
        let tmp = TempDir::new().unwrap();
        write(
            tmp.path(),
            "Cargo.toml",
            "[package]\nname = \"x\"\nversion = \"0.1.0\"\n\n[[bin]]\nname = \"x\"\npath = \"runner.rs\"\n",
        );
        write(tmp.path(), "runner.rs", "pub fn run() {}");

        let out = walk_extracted_crate(tmp.path(), "x", &version()).unwrap();
        assert!(out.items.iter().any(|i| i.qualified_path == "x::run"));
    }

    #[test]
    fn does_not_recurse_into_already_visited_files() {
        // Construct a degenerate case: a #[path] attribute that points
        // a child module back at its own parent. The visited-set
        // protects against infinite recursion.
        let tmp = TempDir::new().unwrap();
        write(
            tmp.path(),
            "src/lib.rs",
            "#[path = \"lib.rs\"]\npub mod loop_;\npub fn ok() {}",
        );

        let out = walk_extracted_crate(tmp.path(), "x", &version()).unwrap();
        // Should terminate (not hang) and produce at least the fn.
        assert!(out.items.iter().any(|i| i.qualified_path == "x::ok"));
    }
}
