//! Resolve a `mod foo;` declaration to a source file on disk.
//!
//! Cargo's resolution rules, in order:
//!
//! 1. `#[path = "..."]` attribute, if present, taken relative to the
//!    parent file's directory.
//! 2. Sibling `<name>.rs` next to the parent file's "module dir".
//! 3. Subdirectory `<name>/mod.rs` next to the parent file's "module
//!    dir".
//!
//! The "module dir" is `<crate_root>/src` for `lib.rs` / `main.rs` and
//! `<parent_file_dir>/<file_stem>` for any other parent file. (i.e.
//! `src/foo.rs` looks for children under `src/foo/`; `src/foo/mod.rs`
//! looks under `src/foo/`.)

use std::path::{Path, PathBuf};

use crate::parse::PendingModule;

/// Resolve `pending` against `crate_root`.
///
/// Returns the source file path **relative to** `crate_root` if found,
/// or `None` if neither candidate exists. The returned path is what
/// the parser uses as `RustItem::source_file` for the inner module.
pub fn resolve(crate_root: &Path, pending: &PendingModule) -> Option<PathBuf> {
    let parent_dir = pending.source_file.parent().unwrap_or(Path::new(""));
    let parent_stem = pending
        .source_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    // Module-content directory: where `mod foo;` looks for children.
    let module_dir: PathBuf =
        if is_crate_root_file(&pending.source_file) || parent_stem == "mod" {
            parent_dir.to_path_buf()
        } else {
            parent_dir.join(parent_stem)
        };

    // 1. #[path = "..."] override.
    if let Some(path_attr) = &pending.path_attr {
        let candidate = parent_dir.join(path_attr);
        if crate_root.join(&candidate).is_file() {
            return Some(candidate);
        }
        // #[path] always wins even when it doesn't exist — the higher
        // walker logs a load failure for this file. Returning None here
        // matches "neither candidate found", which is the correct shape.
        return None;
    }

    // 2. Sibling `<name>.rs`.
    let name_rs = module_dir.join(format!("{}.rs", pending.name));
    if crate_root.join(&name_rs).is_file() {
        return Some(name_rs);
    }

    // 3. Subdirectory `<name>/mod.rs`.
    let name_mod_rs = module_dir.join(&pending.name).join("mod.rs");
    if crate_root.join(&name_mod_rs).is_file() {
        return Some(name_mod_rs);
    }

    None
}

fn is_crate_root_file(path: &Path) -> bool {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    matches!(stem, "lib" | "main")
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    fn pending(
        name: &str,
        source_file: &str,
        parent_module_path: Vec<String>,
    ) -> PendingModule {
        PendingModule {
            name: name.to_string(),
            path_attr: None,
            parent_module_path,
            source_file: PathBuf::from(source_file),
        }
    }

    #[test]
    fn lib_rs_finds_sibling_rs_file() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/lib.rs"), "mod foo;").unwrap();
        fs::write(tmp.path().join("src/foo.rs"), "// foo").unwrap();

        let resolved =
            resolve(tmp.path(), &pending("foo", "src/lib.rs", vec![]));
        assert_eq!(resolved, Some(PathBuf::from("src/foo.rs")));
    }

    #[test]
    fn lib_rs_finds_subdir_mod_rs() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src/foo")).unwrap();
        fs::write(tmp.path().join("src/lib.rs"), "mod foo;").unwrap();
        fs::write(tmp.path().join("src/foo/mod.rs"), "// foo").unwrap();

        let resolved =
            resolve(tmp.path(), &pending("foo", "src/lib.rs", vec![]));
        assert_eq!(resolved, Some(PathBuf::from("src/foo/mod.rs")));
    }

    #[test]
    fn sibling_rs_wins_over_subdir_mod_rs() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src/foo")).unwrap();
        fs::write(tmp.path().join("src/lib.rs"), "mod foo;").unwrap();
        fs::write(tmp.path().join("src/foo.rs"), "// foo flat").unwrap();
        fs::write(tmp.path().join("src/foo/mod.rs"), "// foo nested").unwrap();

        let resolved =
            resolve(tmp.path(), &pending("foo", "src/lib.rs", vec![]));
        assert_eq!(resolved, Some(PathBuf::from("src/foo.rs")));
    }

    #[test]
    fn nested_module_under_subdirectory() {
        // src/lib.rs -> mod foo;  (resolved to src/foo.rs)
        // src/foo.rs -> mod bar;  (should resolve to src/foo/bar.rs)
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src/foo")).unwrap();
        fs::write(tmp.path().join("src/lib.rs"), "mod foo;").unwrap();
        fs::write(tmp.path().join("src/foo.rs"), "mod bar;").unwrap();
        fs::write(tmp.path().join("src/foo/bar.rs"), "// bar").unwrap();

        let resolved = resolve(
            tmp.path(),
            &pending("bar", "src/foo.rs", vec!["foo".into()]),
        );
        assert_eq!(resolved, Some(PathBuf::from("src/foo/bar.rs")));
    }

    #[test]
    fn nested_module_under_mod_rs_uses_same_dir() {
        // src/foo/mod.rs is the parent. mod bar; should look in
        // src/foo/, not src/foo/foo/.
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src/foo")).unwrap();
        fs::write(tmp.path().join("src/foo/mod.rs"), "mod bar;").unwrap();
        fs::write(tmp.path().join("src/foo/bar.rs"), "// bar").unwrap();

        let resolved = resolve(
            tmp.path(),
            &pending("bar", "src/foo/mod.rs", vec!["foo".into()]),
        );
        assert_eq!(resolved, Some(PathBuf::from("src/foo/bar.rs")));
    }

    #[test]
    fn path_attribute_overrides_default_lookup() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/lib.rs"), "mod foo;").unwrap();
        fs::write(tmp.path().join("src/renamed.rs"), "// renamed").unwrap();

        let mut p = pending("foo", "src/lib.rs", vec![]);
        p.path_attr = Some("renamed.rs".to_string());

        let resolved = resolve(tmp.path(), &p);
        assert_eq!(resolved, Some(PathBuf::from("src/renamed.rs")));
    }

    #[test]
    fn path_attribute_pointing_to_missing_file_returns_none() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/lib.rs"), "mod foo;").unwrap();

        let mut p = pending("foo", "src/lib.rs", vec![]);
        p.path_attr = Some("does_not_exist.rs".to_string());

        let resolved = resolve(tmp.path(), &p);
        assert_eq!(resolved, None);
    }

    #[test]
    fn missing_module_returns_none() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/lib.rs"), "mod foo;").unwrap();

        let resolved =
            resolve(tmp.path(), &pending("foo", "src/lib.rs", vec![]));
        assert_eq!(resolved, None);
    }

    #[test]
    fn main_rs_treated_like_lib_rs() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.rs"), "mod foo;").unwrap();
        fs::write(tmp.path().join("src/foo.rs"), "// foo").unwrap();

        let resolved =
            resolve(tmp.path(), &pending("foo", "src/main.rs", vec![]));
        assert_eq!(resolved, Some(PathBuf::from("src/foo.rs")));
    }
}
