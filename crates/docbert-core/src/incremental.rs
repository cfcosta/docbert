use std::collections::HashMap;

use crate::{
    config_db::ConfigDb,
    doc_id::DocumentId,
    error::Result,
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
/// let bytes = meta.serialize();
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
    pub fn serialize(&self) -> Vec<u8> {
        encode_bytes(self)
            .expect("DocumentMetadata serialization should succeed")
    }

    /// Deserialize from bytes. Returns `None` if the archive is invalid.
    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        decode_bytes(bytes).ok()
    }
}

/// What changed in a collection since the last indexing pass.
///
/// Returned by [`diff_collection`]. Files are grouped as new, changed, or
/// deleted relative to the last stored metadata.
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// use docbert_core::ConfigDb;
/// use docbert_core::incremental::{diff_collection, store_metadata};
/// use docbert_core::walker::DiscoveredFile;
/// use std::path::PathBuf;
///
/// let db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
///
/// let file = DiscoveredFile {
///     relative_path: PathBuf::from("hello.md"),
///     absolute_path: PathBuf::from("/abs/hello.md"),
///     mtime: 100,
/// };
/// store_metadata(&db, "notes", &file).unwrap();
///
/// // Same file with updated mtime -> changed
/// let updated = DiscoveredFile { mtime: 200, ..file };
/// let diff = diff_collection(&db, "notes", &[updated]).unwrap();
/// assert_eq!(diff.changed_files.len(), 1);
/// assert!(diff.new_files.is_empty());
/// assert!(diff.deleted_ids.is_empty());
/// ```
#[derive(Debug, Default)]
pub struct DiffResult {
    /// Files that are new (not in metadata).
    pub new_files: Vec<DiscoveredFile>,
    /// Files that have changed (mtime differs).
    pub changed_files: Vec<DiscoveredFile>,
    /// Document IDs that were in metadata but no longer on disk.
    pub deleted_ids: Vec<u64>,
}

/// Compare the files you just discovered with the metadata already on disk.
///
/// For the given collection, this walks stored metadata, compares mtimes, and
/// returns a [`DiffResult`] describing what is new, changed, or gone.
pub fn diff_collection(
    config_db: &ConfigDb,
    collection: &str,
    discovered: &[DiscoveredFile],
) -> Result<DiffResult> {
    // Build a map of all known documents for this collection.
    let mut known: HashMap<String, (u64, u64)> = HashMap::new(); // path -> (doc_id, mtime)

    for (doc_id, meta) in config_db.list_all_document_metadata_typed()? {
        if meta.collection == collection {
            known.insert(meta.relative_path.clone(), (doc_id, meta.mtime));
        }
    }

    let mut result = DiffResult::default();

    // Track which known docs we've seen in the discovered set.
    let mut seen_paths = std::collections::HashSet::new();

    for file in discovered {
        let rel_path = file.relative_path.to_string_lossy().to_string();
        seen_paths.insert(rel_path.clone());

        match known.get(&rel_path) {
            None => {
                result.new_files.push(file.clone());
            }
            Some((_doc_id, stored_mtime)) => {
                if file.mtime != *stored_mtime {
                    result.changed_files.push(file.clone());
                }
                // If the mtime matches, the file is unchanged, so skip it.
            }
        }
    }

    // Find deleted documents (in metadata but not on disk).
    for (path, (doc_id, _)) in &known {
        if !seen_paths.contains(path) {
            result.deleted_ids.push(*doc_id);
        }
    }

    Ok(result)
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
    use std::path::PathBuf;

    use super::*;

    fn test_db() -> (tempfile::TempDir, ConfigDb) {
        let tmp = tempfile::tempdir().unwrap();
        let db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        (tmp, db)
    }

    fn make_file(name: &str, mtime: u64) -> DiscoveredFile {
        DiscoveredFile {
            relative_path: PathBuf::from(name),
            absolute_path: PathBuf::from(format!("/abs/{name}")),
            mtime,
        }
    }

    #[test]
    fn document_metadata_rkyv_roundtrips() {
        let meta = DocumentMetadata {
            collection: "notes".to_string(),
            relative_path: "hello.md".to_string(),
            mtime: 12345,
        };
        let bytes = meta.serialize();
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
        .serialize();
        bytes.truncate(bytes.len() / 2);
        assert!(DocumentMetadata::deserialize(&bytes).is_none());
    }

    #[test]
    fn all_new_files() {
        let (_tmp, db) = test_db();
        let files = vec![make_file("a.md", 100), make_file("b.md", 200)];
        let diff = diff_collection(&db, "notes", &files).unwrap();

        assert_eq!(diff.new_files.len(), 2);
        assert!(diff.changed_files.is_empty());
        assert!(diff.deleted_ids.is_empty());
    }

    #[test]
    fn unchanged_files() {
        let (_tmp, db) = test_db();
        let file = make_file("a.md", 100);
        store_metadata(&db, "notes", &file).unwrap();

        let diff = diff_collection(&db, "notes", &[file]).unwrap();
        assert!(diff.new_files.is_empty());
        assert!(diff.changed_files.is_empty());
        assert!(diff.deleted_ids.is_empty());
    }

    #[test]
    fn changed_file_detected() {
        let (_tmp, db) = test_db();
        let file = make_file("a.md", 100);
        store_metadata(&db, "notes", &file).unwrap();

        // Same file, different mtime
        let updated = make_file("a.md", 200);
        let diff = diff_collection(&db, "notes", &[updated]).unwrap();
        assert!(diff.new_files.is_empty());
        assert_eq!(diff.changed_files.len(), 1);
        assert!(diff.deleted_ids.is_empty());
    }

    #[test]
    fn deleted_file_detected() {
        let (_tmp, db) = test_db();
        let file = make_file("a.md", 100);
        store_metadata(&db, "notes", &file).unwrap();

        // Empty discovery = file was deleted
        let diff = diff_collection(&db, "notes", &[]).unwrap();
        assert!(diff.new_files.is_empty());
        assert!(diff.changed_files.is_empty());
        assert_eq!(diff.deleted_ids.len(), 1);
    }

    #[test]
    fn ignores_other_collections() {
        let (_tmp, db) = test_db();
        let file = make_file("a.md", 100);
        store_metadata(&db, "other", &file).unwrap();

        // Diffing "notes" should not see "other"'s files
        let diff = diff_collection(&db, "notes", &[]).unwrap();
        assert!(diff.deleted_ids.is_empty());
    }
}
