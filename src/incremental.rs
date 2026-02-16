use std::collections::HashMap;

use crate::{
    config_db::ConfigDb,
    doc_id::DocumentId,
    error::Result,
    walker::DiscoveredFile,
};

/// Metadata stored per document in config.db.
///
/// Serialized as: `"collection\0relative_path\0mtime"`.
///
/// # Examples
///
/// ```
/// use docbert::incremental::DocumentMetadata;
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentMetadata {
    pub collection: String,
    pub relative_path: String,
    pub mtime: u64,
}

impl DocumentMetadata {
    /// Serialize to a byte vector for storage in the config database.
    pub fn serialize(&self) -> Vec<u8> {
        format!(
            "{}\0{}\0{}",
            self.collection, self.relative_path, self.mtime
        )
        .into_bytes()
    }

    /// Deserialize from bytes. Returns `None` if the format is invalid.
    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        let s = std::str::from_utf8(bytes).ok()?;
        let mut parts = s.splitn(3, '\0');
        let collection = parts.next()?.to_string();
        let relative_path = parts.next()?.to_string();
        let mtime = parts.next()?.parse().ok()?;
        Some(Self {
            collection,
            relative_path,
            mtime,
        })
    }
}

/// Result of comparing discovered files against stored metadata.
#[derive(Debug, Default)]
pub struct DiffResult {
    /// Files that are new (not in metadata).
    pub new_files: Vec<DiscoveredFile>,
    /// Files that have changed (mtime differs).
    pub changed_files: Vec<DiscoveredFile>,
    /// Document IDs that were in metadata but no longer on disk.
    pub deleted_ids: Vec<u64>,
}

/// Compare discovered files against stored document metadata.
///
/// Returns which files are new, changed, or deleted.
pub fn diff_collection(
    config_db: &ConfigDb,
    collection: &str,
    discovered: &[DiscoveredFile],
) -> Result<DiffResult> {
    // Build a map of all known documents for this collection.
    let mut known: HashMap<String, (u64, u64)> = HashMap::new(); // path -> (doc_id, mtime)

    for (doc_id, bytes) in config_db.list_all_document_metadata()? {
        if let Some(meta) = DocumentMetadata::deserialize(&bytes)
            && meta.collection == collection
        {
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
                // If mtime matches, it's unchanged — skip.
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

/// Store metadata for a document after successful indexing.
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
    config_db.set_document_metadata(doc_id.numeric, &meta.serialize())?;
    Ok(())
}

/// Store metadata for multiple documents in a single transaction.
pub fn batch_store_metadata(
    config_db: &ConfigDb,
    collection: &str,
    files: &[DiscoveredFile],
) -> Result<()> {
    let entries: Vec<(u64, Vec<u8>)> = files
        .iter()
        .map(|file| {
            let rel_path = file.relative_path.to_string_lossy().to_string();
            let doc_id = DocumentId::new(collection, &rel_path);
            let meta = DocumentMetadata {
                collection: collection.to_string(),
                relative_path: rel_path,
                mtime: file.mtime,
            };
            (doc_id.numeric, meta.serialize())
        })
        .collect();
    config_db.batch_set_document_metadata(&entries)
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
    fn metadata_roundtrip() {
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
