#![allow(dead_code)]

use std::path::PathBuf;

use docbert_core::{ConfigDb, error, path_safety};

pub(crate) fn resolve_collection_root(
    config_db: &ConfigDb,
    collection: &str,
) -> error::Result<PathBuf> {
    let root = config_db.get_collection(collection)?.ok_or_else(|| {
        error::Error::NotFound {
            kind: "collection",
            name: collection.to_string(),
        }
    })?;

    if root.is_empty() {
        return Err(error::Error::Config(format!(
            "collection '{collection}' does not have a filesystem path"
        )));
    }

    let path = PathBuf::from(root).canonicalize()?;
    if !path.is_dir() {
        return Err(error::Error::Config(format!(
            "collection '{collection}' path is not a directory: {}",
            path.display()
        )));
    }

    Ok(path)
}

pub(crate) fn resolve_document_path(
    config_db: &ConfigDb,
    collection: &str,
    relative_path: &str,
) -> error::Result<PathBuf> {
    let root = resolve_collection_root(config_db, collection)?;
    path_safety::resolve_safe_document_path(&root, relative_path)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    fn test_db() -> (tempfile::TempDir, ConfigDb) {
        let tmp = tempfile::tempdir().unwrap();
        let db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        (tmp, db)
    }

    #[test]
    fn web_paths_accept_nested_relative_paths() {
        let (tmp, db) = test_db();
        let root = tmp.path().join("notes");
        fs::create_dir_all(&root).unwrap();
        db.set_collection("notes", root.to_str().unwrap()).unwrap();

        let resolved =
            resolve_document_path(&db, "notes", "nested/deep/file.md").unwrap();

        let canonical_root = root.canonicalize().unwrap();
        assert_eq!(resolved, canonical_root.join("nested/deep/file.md"));
    }

    #[test]
    fn web_paths_reject_parent_dir_traversal() {
        let (tmp, db) = test_db();
        let root = tmp.path().join("notes");
        fs::create_dir_all(&root).unwrap();
        db.set_collection("notes", root.to_str().unwrap()).unwrap();

        let err =
            resolve_document_path(&db, "notes", "../secret.md").unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("parent traversal") || msg.contains("absolute"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn web_paths_resolution_stays_under_collection_root() {
        let (tmp, db) = test_db();
        let root = tmp.path().join("notes");
        fs::create_dir_all(root.join("nested")).unwrap();
        db.set_collection("notes", root.to_str().unwrap()).unwrap();

        let resolved =
            resolve_document_path(&db, "notes", "nested/file.md").unwrap();

        assert!(resolved.starts_with(root.canonicalize().unwrap()));
    }

    #[cfg(unix)]
    #[test]
    fn web_paths_overwrite_targets_resolve_only_inside_registered_collection_path()
     {
        use std::os::unix::fs::symlink;

        let (tmp, db) = test_db();
        let root = tmp.path().join("notes");
        let outside = tmp.path().join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        let outside_file = outside.join("escape.md");
        fs::write(&outside_file, "escape").unwrap();
        symlink(&outside_file, root.join("linked.md")).unwrap();
        db.set_collection("notes", root.to_str().unwrap()).unwrap();

        let err = resolve_document_path(&db, "notes", "linked.md").unwrap_err();

        assert!(err.to_string().contains("outside the collection root"));
    }
}
