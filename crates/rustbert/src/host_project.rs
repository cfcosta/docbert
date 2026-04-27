//! Index a local Cargo project's source the same way published
//! crates are indexed. The Phase-4 follow-up from the design.
//!
//! Reads the project's `Cargo.toml` for the package name and version,
//! walks `src/` via [`crate::crate_walker`], lowers items into
//! `RustItem`s, and stores them under a synthetic collection
//! `<name>@<version>`. Subsequent `rustbert search`, `get`, and
//! `list` calls work identically against the local project.
//!
//! Limitations (intentional for v1):
//!
//! - Workspaces: only the package at the given path is indexed,
//!   not every workspace member. Pass `--manifest <path>` per member
//!   to index more.
//! - No `cargo build` / `cargo metadata` invocation. The package name
//!   and version come from a minimal TOML parse of `[package]`.
//! - Path / git deps in the project's tree are not auto-fetched —
//!   `rustbert sync` is the path for those.

use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::{
    cache::CrateCache,
    collection::SyntheticCollection,
    crate_walker,
    error::{Error, Result},
    indexer::Indexer,
};

#[derive(Debug, Clone)]
pub struct ProjectInfo {
    pub name: String,
    pub version: semver::Version,
}

/// Read minimal `[package] name = "..."  version = "..."` from a
/// project root's `Cargo.toml`. Returns an error if the file is
/// missing or doesn't expose a `[package]` table.
///
/// Handles workspace inheritance: when `version.workspace = true`,
/// walks up directories to find the workspace root's
/// `[workspace.package].version` and uses that.
pub fn read_project_info(project_root: &Path) -> Result<ProjectInfo> {
    let manifest_path = project_root.join("Cargo.toml");
    let text = fs::read_to_string(&manifest_path).map_err(|e| {
        Error::Cache(format!(
            "Cargo.toml read at {}: {e}",
            manifest_path.display()
        ))
    })?;

    let value: toml::Value = text.parse().map_err(|e: toml::de::Error| {
        Error::Cache(format!("Cargo.toml parse: {e}"))
    })?;

    let package = value
        .get("package")
        .ok_or_else(|| {
            Error::Cache(format!(
                "Cargo.toml at {} has no [package] table — workspace roots aren't supported",
                manifest_path.display()
            ))
        })?;

    let name = package
        .get("name")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| {
            Error::Cache("Cargo.toml [package] missing `name`".to_string())
        })?
        .to_string();

    let version_str = resolve_field(
        package.get("version"),
        project_root,
        "version",
    )?
    .ok_or_else(|| {
        Error::Cache(
            "Cargo.toml [package] missing `version` (and no inheritable workspace.package.version found)".to_string(),
        )
    })?;
    let version = semver::Version::parse(&version_str).map_err(|e| {
        Error::Cache(format!("Cargo.toml [package].version: {e}"))
    })?;

    Ok(ProjectInfo { name, version })
}

/// Resolve a `[package]` field value, honoring `workspace = true`
/// inheritance. Returns `Ok(None)` when the field is absent and
/// has no resolvable workspace inheritance.
fn resolve_field(
    field: Option<&toml::Value>,
    project_root: &Path,
    field_name: &str,
) -> Result<Option<String>> {
    let Some(field) = field else { return Ok(None) };

    // Direct string: `version = "1.0.0"`.
    if let Some(s) = field.as_str() {
        return Ok(Some(s.to_string()));
    }

    // Workspace inheritance: `version.workspace = true`.
    if let Some(table) = field.as_table()
        && table.get("workspace").and_then(toml::Value::as_bool) == Some(true)
    {
        let ws = find_workspace_root(project_root)?;
        let ws_text = fs::read_to_string(ws.join("Cargo.toml"))?;
        let ws_value: toml::Value =
            ws_text.parse().map_err(|e: toml::de::Error| {
                Error::Cache(format!("workspace Cargo.toml parse: {e}"))
            })?;
        return Ok(ws_value
            .get("workspace")
            .and_then(|w| w.get("package"))
            .and_then(|p| p.get(field_name))
            .and_then(toml::Value::as_str)
            .map(String::from));
    }

    Ok(None)
}

/// Walk up from `start` looking for a `Cargo.toml` with a
/// `[workspace]` table.
fn find_workspace_root(start: &Path) -> Result<PathBuf> {
    let mut current =
        start.canonicalize().unwrap_or_else(|_| start.to_path_buf());
    loop {
        let manifest = current.join("Cargo.toml");
        if manifest.is_file()
            && let Ok(text) = fs::read_to_string(&manifest)
            && let Ok(value) = text.parse::<toml::Value>()
            && value.get("workspace").is_some()
        {
            return Ok(current);
        }
        if !current.pop() {
            break;
        }
    }
    Err(Error::Cache(
        "no [workspace] root found above project — `workspace = true` won't resolve".to_string(),
    ))
}

/// Walk a local Cargo project, index its items, and store them as a
/// synthetic collection `<name>@<version>`. Mirrors the pipeline
/// `ingestion::ingest` runs against fetched crates, minus the network
/// fetch / tarball extraction steps.
pub fn index_project(
    project_root: &Path,
    cache: &CrateCache,
    indexer: &Indexer,
) -> Result<(SyntheticCollection, usize, usize)> {
    let info = read_project_info(project_root)?;
    let collection = SyntheticCollection {
        crate_name: info.name.clone(),
        version: info.version.clone(),
    };

    let walked = crate_walker::walk_extracted_crate(
        project_root,
        &info.name,
        &info.version,
    )?;
    let item_count = walked.items.len();
    let failure_count = walked.failures.len();

    cache.store(&collection, &walked.items)?;
    indexer.index_lexical(&collection, &walked.items)?;

    Ok((collection, item_count, failure_count))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    fn write_project(root: &Path, name: &str, version: &str, lib_rs: &str) {
        fs::write(
            root.join("Cargo.toml"),
            format!(
                "[package]\nname = \"{name}\"\nversion = \"{version}\"\nedition = \"2021\"\n"
            ),
        )
        .unwrap();
        let src = root.join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("lib.rs"), lib_rs).unwrap();
    }

    #[test]
    fn reads_package_name_and_version() {
        let tmp = TempDir::new().unwrap();
        write_project(tmp.path(), "myproj", "1.2.3", "");
        let info = read_project_info(tmp.path()).unwrap();
        assert_eq!(info.name, "myproj");
        assert_eq!(info.version, semver::Version::new(1, 2, 3));
    }

    #[test]
    fn rejects_workspace_root_without_package_table() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"a\", \"b\"]\n",
        )
        .unwrap();
        let err = read_project_info(tmp.path()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("[package]"), "got {msg:?}");
    }

    #[test]
    fn resolves_workspace_inherited_version() {
        let tmp = TempDir::new().unwrap();
        // Workspace root with [workspace.package].version
        fs::write(
            tmp.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/foo\"]\n\n[workspace.package]\nversion = \"2.5.0\"\n",
        )
        .unwrap();
        let member = tmp.path().join("crates/foo");
        fs::create_dir_all(member.join("src")).unwrap();
        fs::write(
            member.join("Cargo.toml"),
            "[package]\nname = \"foo\"\nversion.workspace = true\n",
        )
        .unwrap();
        fs::write(member.join("src/lib.rs"), "pub fn x() {}").unwrap();

        let info = read_project_info(&member).unwrap();
        assert_eq!(info.name, "foo");
        assert_eq!(info.version, semver::Version::new(2, 5, 0));
    }

    #[test]
    fn workspace_inheritance_without_root_errors() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("Cargo.toml"),
            "[package]\nname = \"orphan\"\nversion.workspace = true\n",
        )
        .unwrap();
        let err = read_project_info(tmp.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("workspace") || msg.contains("[workspace]"),
            "got {msg:?}"
        );
    }

    #[test]
    fn missing_manifest_errors() {
        let tmp = TempDir::new().unwrap();
        let err = read_project_info(tmp.path()).unwrap_err();
        assert!(err.to_string().contains("Cargo.toml read"));
    }

    #[test]
    fn index_project_stores_items_in_cache_and_indexer() {
        let tmp_proj = TempDir::new().unwrap();
        write_project(
            tmp_proj.path(),
            "demo",
            "0.1.0",
            "/// Greet someone.\npub fn greet() {}\n\npub struct Holder;\n",
        );

        let tmp_cache = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp_cache.path()).unwrap();
        let indexer = Indexer::open(tmp_cache.path()).unwrap();

        let (coll, items, failures) =
            index_project(tmp_proj.path(), &cache, &indexer).unwrap();

        assert_eq!(coll.crate_name, "demo");
        assert_eq!(coll.version, semver::Version::new(0, 1, 0));
        assert!(items >= 2, "got {items} items");
        assert_eq!(failures, 0);

        let cached = cache.load(&coll).unwrap();
        assert!(cached.iter().any(|i| i.qualified_path == "demo::greet"));
        assert!(cached.iter().any(|i| i.qualified_path == "demo::Holder"));
    }
}
