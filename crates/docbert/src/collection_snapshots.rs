use std::path::Path;

use docbert_core::{
    ConfigDb,
    error,
    incremental::{MerkleDiffResult, diff_collection_snapshots},
    merkle::{CollectionMerkleSnapshot, build_collection_snapshot},
    walker::{self, DiscoveredFile},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CollectionSnapshotChange {
    pub(crate) previous_snapshot: Option<CollectionMerkleSnapshot>,
    pub(crate) current_snapshot: CollectionMerkleSnapshot,
    pub(crate) diff: MerkleDiffResult,
}

pub(crate) fn load_collection_snapshot(
    config_db: &ConfigDb,
    collection: &str,
) -> error::Result<Option<CollectionMerkleSnapshot>> {
    config_db.get_collection_merkle_snapshot(collection)
}

pub(crate) fn compute_collection_snapshot(
    collection: &str,
    root: &Path,
) -> error::Result<CollectionMerkleSnapshot> {
    let discovered = walker::discover_files(root)?;
    build_collection_snapshot(collection, &discovered)
}

pub(crate) fn compute_collection_snapshot_change_for_discovered(
    config_db: &ConfigDb,
    collection: &str,
    discovered: &[DiscoveredFile],
) -> error::Result<CollectionSnapshotChange> {
    let previous_snapshot = load_collection_snapshot(config_db, collection)?;
    let current_snapshot = build_collection_snapshot(collection, discovered)?;
    let diff = diff_collection_snapshots(
        previous_snapshot.as_ref(),
        &current_snapshot,
    );

    Ok(CollectionSnapshotChange {
        previous_snapshot,
        current_snapshot,
        diff,
    })
}

pub(crate) fn compute_collection_snapshot_change(
    config_db: &ConfigDb,
    collection: &str,
    root: &Path,
) -> error::Result<CollectionSnapshotChange> {
    let discovered = walker::discover_files(root)?;
    compute_collection_snapshot_change_for_discovered(
        config_db,
        collection,
        &discovered,
    )
}

pub(crate) fn replace_collection_snapshot(
    config_db: &ConfigDb,
    snapshot: &CollectionMerkleSnapshot,
) -> error::Result<()> {
    config_db.set_collection_merkle_snapshot(&snapshot.collection, snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_collection() -> (tempfile::TempDir, ConfigDb, std::path::PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        (tmp, config_db, root)
    }

    #[test]
    fn compute_collection_snapshot_change_is_read_only_until_replace() {
        let (_tmp, config_db, root) = setup_collection();
        std::fs::write(root.join("a.md"), "alpha").unwrap();

        let original = compute_collection_snapshot("notes", &root).unwrap();
        config_db
            .set_collection_merkle_snapshot("notes", &original)
            .unwrap();

        std::fs::write(root.join("a.md"), "beta").unwrap();

        let change =
            compute_collection_snapshot_change(&config_db, "notes", &root)
                .unwrap();

        assert_eq!(
            config_db.get_collection_merkle_snapshot("notes").unwrap(),
            Some(original.clone())
        );
        assert_eq!(change.previous_snapshot, Some(original));
        assert_eq!(change.diff.changed_paths, vec!["a.md".to_string()]);
    }

    #[test]
    fn replace_collection_snapshot_mutates_db_only_when_called() {
        let (_tmp, config_db, root) = setup_collection();
        std::fs::write(root.join("a.md"), "alpha").unwrap();

        let first = compute_collection_snapshot("notes", &root).unwrap();
        replace_collection_snapshot(&config_db, &first).unwrap();
        assert_eq!(
            config_db.get_collection_merkle_snapshot("notes").unwrap(),
            Some(first.clone())
        );

        std::fs::write(root.join("b.md"), "bravo").unwrap();
        let second = compute_collection_snapshot("notes", &root).unwrap();

        assert_eq!(
            config_db.get_collection_merkle_snapshot("notes").unwrap(),
            Some(first.clone())
        );

        replace_collection_snapshot(&config_db, &second).unwrap();
        assert_eq!(
            config_db.get_collection_merkle_snapshot("notes").unwrap(),
            Some(second)
        );
    }
}
