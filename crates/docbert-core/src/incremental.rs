use crate::{
    config_db::ConfigDb,
    doc_id::DocumentId,
    error::Result,
    merkle::Snapshot,
    storage_codec::{decode_bytes, encode_bytes},
    walker::DiscoveredFile,
};

/// Metadata docbert stores for each indexed document in `config.db`.
///
/// The serialized form is a `rkyv`-encoded binary payload.
///
/// # Examples
///
/// ```
/// use docbert_core::incremental::DocumentMetadata;
///
/// let meta = DocumentMetadata {
///     collection: "notes".to_string(),
///     relative_path: "hello.md".to_string(),
///     mtime: 1700000000,
/// };
/// let bytes = meta.serialize().unwrap();
/// let restored = DocumentMetadata::deserialize(&bytes).unwrap();
/// assert_eq!(meta, restored);
/// ```
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct DocumentMetadata {
    pub collection: String,
    pub relative_path: String,
    pub mtime: u64,
}

impl DocumentMetadata {
    /// Serialize to a byte vector for storage in the config database.
    ///
    /// The format is a checked `rkyv` archive. Use [`deserialize`](Self::deserialize)
    /// to recover the struct.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Rkyv`] if the rkyv encoder fails (effectively
    /// only on allocator failure).
    ///
    /// [`Error::Rkyv`]: crate::Error::Rkyv
    pub fn serialize(&self) -> crate::Result<Vec<u8>> {
        encode_bytes(self)
    }

    /// Deserialize from bytes. Returns `None` if the archive is invalid.
    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        decode_bytes(bytes).ok()
    }
}

/// What changed between two Merkle snapshots for the same collection.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MerkleDiffResult {
    /// Paths present in the current snapshot but not in the previous snapshot.
    pub new_paths: Vec<String>,
    /// Paths present in both snapshots whose leaf hash changed.
    pub changed_paths: Vec<String>,
    /// Paths present in the previous snapshot but not in the current snapshot.
    pub deleted_paths: Vec<String>,
    /// Deterministic document IDs derived from deleted paths.
    pub deleted_ids: Vec<u64>,
}

/// Compare a previously stored collection snapshot with a newly computed one.
///
/// This classifies file paths as new, changed, or deleted using Merkle leaf
/// hashes instead of filesystem modification times.
pub fn diff_snapshots(
    previous: Option<&Snapshot>,
    current: &Snapshot,
) -> MerkleDiffResult {
    let mut result = MerkleDiffResult::default();

    let collection = previous
        .map(|snapshot| snapshot.collection.as_str())
        .unwrap_or(current.collection.as_str());

    let previous_files: std::collections::BTreeMap<&str, _> = previous
        .map(|snapshot| {
            snapshot
                .files
                .iter()
                .map(|file| (file.relative_path.as_str(), file.leaf_hash))
                .collect()
        })
        .unwrap_or_default();
    let current_files: std::collections::BTreeMap<&str, _> = current
        .files
        .iter()
        .map(|file| (file.relative_path.as_str(), file.leaf_hash))
        .collect();

    for (path, current_hash) in &current_files {
        match previous_files.get(path) {
            None => result.new_paths.push((*path).to_string()),
            Some(previous_hash) if previous_hash != current_hash => {
                result.changed_paths.push((*path).to_string())
            }
            Some(_) => {}
        }
    }

    for path in previous_files.keys() {
        if !current_files.contains_key(path) {
            let deleted_path = (*path).to_string();
            result
                .deleted_ids
                .push(DocumentId::new(collection, &deleted_path).numeric);
            result.deleted_paths.push(deleted_path);
        }
    }

    result
}

/// Write metadata for one document after it has been indexed.
///
/// This recomputes the [`DocumentId`] from the collection name and relative
/// path, then serializes and stores the metadata.
pub fn store_metadata(
    config_db: &ConfigDb,
    collection: &str,
    file: &DiscoveredFile,
) -> Result<()> {
    let rel_path = file.relative_path.to_string_lossy().to_string();
    let doc_id = DocumentId::new(collection, &rel_path);
    let meta = DocumentMetadata {
        collection: collection.to_string(),
        relative_path: rel_path,
        mtime: file.mtime,
    };
    config_db.set_document_metadata_typed(doc_id.numeric, &meta)?;
    Ok(())
}

/// Write metadata for several documents in one transaction.
///
/// This is cheaper than calling [`store_metadata`] in a loop because every
/// write shares the same database transaction.
pub fn batch_store_metadata(
    config_db: &ConfigDb,
    collection: &str,
    files: &[DiscoveredFile],
) -> Result<()> {
    let entries: Vec<(u64, DocumentMetadata)> = files
        .iter()
        .map(|file| {
            let rel_path = file.relative_path.to_string_lossy().to_string();
            let doc_id = DocumentId::new(collection, &rel_path);
            let meta = DocumentMetadata {
                collection: collection.to_string(),
                relative_path: rel_path,
                mtime: file.mtime,
            };
            (doc_id.numeric, meta)
        })
        .collect();
    config_db.batch_set_document_metadata_typed(&entries)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::*;
    use crate::merkle::build_snapshot;

    fn write_snapshot(
        root: &tempfile::TempDir,
        files: &[(&str, &str, u64)],
    ) -> Snapshot {
        for (relative_path, content, _) in files {
            let full_path = root.path().join(relative_path);
            if let Some(parent) = full_path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&full_path, content).unwrap();
        }

        let discovered: Vec<DiscoveredFile> = files
            .iter()
            .map(|(relative_path, _content, mtime)| DiscoveredFile {
                relative_path: PathBuf::from(relative_path),
                absolute_path: root.path().join(relative_path),
                mtime: *mtime,
            })
            .collect();

        build_snapshot("notes", &discovered).unwrap()
    }

    #[test]
    fn document_metadata_rkyv_roundtrips() {
        let meta = DocumentMetadata {
            collection: "notes".to_string(),
            relative_path: "hello.md".to_string(),
            mtime: 12345,
        };
        let bytes = meta.serialize().unwrap();
        let restored = DocumentMetadata::deserialize(&bytes).unwrap();
        assert_eq!(meta, restored);
    }

    #[test]
    fn document_metadata_invalid_bytes_return_none() {
        let mut bytes = DocumentMetadata {
            collection: "notes".to_string(),
            relative_path: "hello.md".to_string(),
            mtime: 12345,
        }
        .serialize()
        .unwrap();
        bytes.truncate(bytes.len() / 2);
        assert!(DocumentMetadata::deserialize(&bytes).is_none());
    }

    #[test]
    fn merkle_diff_reports_added_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let current = write_snapshot(&tmp, &[("a.md", "hello", 1)]);

        let diff = diff_snapshots(None, &current);

        assert_eq!(
            diff,
            MerkleDiffResult {
                new_paths: vec!["a.md".to_string()],
                changed_paths: vec![],
                deleted_paths: vec![],
                deleted_ids: vec![],
            }
        );
    }

    #[test]
    fn merkle_diff_reports_changed_paths() {
        let previous_tmp = tempfile::tempdir().unwrap();
        let current_tmp = tempfile::tempdir().unwrap();
        let previous = write_snapshot(&previous_tmp, &[("a.md", "hello", 1)]);
        let current = write_snapshot(&current_tmp, &[("a.md", "hello!", 1)]);

        let diff = diff_snapshots(Some(&previous), &current);

        assert_eq!(
            diff,
            MerkleDiffResult {
                new_paths: vec![],
                changed_paths: vec!["a.md".to_string()],
                deleted_paths: vec![],
                deleted_ids: vec![],
            }
        );
    }

    #[test]
    fn merkle_diff_reports_deleted_paths() {
        let previous_tmp = tempfile::tempdir().unwrap();
        let current_tmp = tempfile::tempdir().unwrap();
        let previous = write_snapshot(&previous_tmp, &[("a.md", "hello", 1)]);
        let current = write_snapshot(&current_tmp, &[]);

        let diff = diff_snapshots(Some(&previous), &current);

        assert_eq!(
            diff,
            MerkleDiffResult {
                new_paths: vec![],
                changed_paths: vec![],
                deleted_paths: vec!["a.md".to_string()],
                deleted_ids: vec![DocumentId::new("notes", "a.md").numeric],
            }
        );
    }

    #[test]
    fn merkle_diff_ignores_mtime_when_content_is_unchanged() {
        let previous_tmp = tempfile::tempdir().unwrap();
        let current_tmp = tempfile::tempdir().unwrap();
        let previous = write_snapshot(&previous_tmp, &[("a.md", "hello", 1)]);
        let current =
            write_snapshot(&current_tmp, &[("a.md", "hello", 999_999)]);

        let diff = diff_snapshots(Some(&previous), &current);

        assert_eq!(diff, MerkleDiffResult::default());
    }

    #[test]
    fn merkle_diff_deleted_ids_match_document_id_derivation_and_ordering() {
        let previous_tmp = tempfile::tempdir().unwrap();
        let current_tmp = tempfile::tempdir().unwrap();
        let previous = write_snapshot(
            &previous_tmp,
            &[
                ("b.md", "bee", 1),
                ("nested/a.md", "aye", 1),
                ("nested/c.md", "see", 1),
            ],
        );
        let current = write_snapshot(&current_tmp, &[]);

        let diff = diff_snapshots(Some(&previous), &current);

        assert_eq!(
            diff.deleted_paths,
            vec![
                "b.md".to_string(),
                "nested/a.md".to_string(),
                "nested/c.md".to_string(),
            ]
        );
        assert_eq!(
            diff.deleted_ids,
            diff.deleted_paths
                .iter()
                .map(|path| DocumentId::new("notes", path).numeric)
                .collect::<Vec<_>>()
        );
    }
}
