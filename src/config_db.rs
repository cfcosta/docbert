use std::path::Path;

use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};

use crate::error::Result;

const COLLECTIONS: TableDefinition<&str, &str> =
    TableDefinition::new("collections");
const CONTEXTS: TableDefinition<&str, &str> = TableDefinition::new("contexts");
const DOCUMENT_METADATA: TableDefinition<u64, &[u8]> =
    TableDefinition::new("document_metadata");
const SETTINGS: TableDefinition<&str, &str> = TableDefinition::new("settings");

/// Persistent key-value store for docbert configuration.
///
/// Manages four tables:
/// - **collections**: maps collection names to filesystem paths
/// - **contexts**: maps URIs to human-readable descriptions
/// - **document_metadata**: maps numeric document IDs to serialized metadata
/// - **settings**: general key-value settings (e.g., `model_name`)
///
/// Backed by [redb](https://github.com/cberner/redb), an embedded ACID database.
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// use docbert::ConfigDb;
///
/// let db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
///
/// // Register a collection
/// db.set_collection("notes", "/home/user/notes").unwrap();
/// assert_eq!(db.get_collection("notes").unwrap(), Some("/home/user/notes".to_string()));
///
/// // Store a setting
/// db.set_setting("model_name", "custom/model").unwrap();
/// assert_eq!(db.get_setting("model_name").unwrap(), Some("custom/model".to_string()));
/// ```
pub struct ConfigDb {
    db: Database,
}

impl ConfigDb {
    /// Open or create a config database at the given path.
    ///
    /// Creates all required tables on first open.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// use docbert::ConfigDb;
    /// let db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// ```
    pub fn open(path: &Path) -> Result<Self> {
        let db = Database::create(path)?;

        // Ensure all tables exist by opening them in a write transaction.
        let txn = db.begin_write()?;
        txn.open_table(COLLECTIONS)?;
        txn.open_table(CONTEXTS)?;
        txn.open_table(DOCUMENT_METADATA)?;
        txn.open_table(SETTINGS)?;
        txn.commit()?;

        Ok(Self { db })
    }

    // -- Collections --

    /// Register a named collection pointing to a filesystem directory.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_collection("notes", "/home/user/notes").unwrap();
    /// ```
    pub fn set_collection(&self, name: &str, path: &str) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(COLLECTIONS)?;
            table.insert(name, path)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Look up a collection's filesystem path by name.
    ///
    /// Returns `None` if the collection is not registered.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// assert_eq!(db.get_collection("nonexistent").unwrap(), None);
    /// db.set_collection("notes", "/path").unwrap();
    /// assert_eq!(db.get_collection("notes").unwrap(), Some("/path".to_string()));
    /// ```
    pub fn get_collection(&self, name: &str) -> Result<Option<String>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(COLLECTIONS)?;
        Ok(table.get(name)?.map(|v| v.value().to_string()))
    }

    /// Remove a collection by name. Returns `true` if it existed.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_collection("notes", "/notes").unwrap();
    /// assert!(db.remove_collection("notes").unwrap());
    /// assert!(!db.remove_collection("notes").unwrap()); // already gone
    /// ```
    pub fn remove_collection(&self, name: &str) -> Result<bool> {
        let txn = self.db.begin_write()?;
        let removed = {
            let mut table = txn.open_table(COLLECTIONS)?;
            table.remove(name)?.is_some()
        };
        txn.commit()?;
        Ok(removed)
    }

    /// List all registered collections as `(name, path)` pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_collection("notes", "/notes").unwrap();
    /// db.set_collection("docs", "/docs").unwrap();
    /// let collections = db.list_collections().unwrap();
    /// assert_eq!(collections.len(), 2);
    /// ```
    pub fn list_collections(&self) -> Result<Vec<(String, String)>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(COLLECTIONS)?;
        let mut result = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            result.push((k.value().to_string(), v.value().to_string()));
        }
        Ok(result)
    }

    // -- Contexts --

    /// Attach a human-readable context description to a URI.
    ///
    /// Contexts are prepended as `<!-- Context: ... -->` comments when
    /// documents are served via MCP.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_context("bert://notes", "Personal research notes").unwrap();
    /// ```
    pub fn set_context(&self, uri: &str, description: &str) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(CONTEXTS)?;
            table.insert(uri, description)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Get the context description for a URI, if one exists.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// assert_eq!(db.get_context("bert://notes").unwrap(), None);
    /// db.set_context("bert://notes", "Personal notes").unwrap();
    /// assert_eq!(
    ///     db.get_context("bert://notes").unwrap(),
    ///     Some("Personal notes".to_string()),
    /// );
    /// ```
    pub fn get_context(&self, uri: &str) -> Result<Option<String>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(CONTEXTS)?;
        Ok(table.get(uri)?.map(|v| v.value().to_string()))
    }

    /// Remove a context by URI. Returns `true` if it existed.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_context("bert://notes", "Notes").unwrap();
    /// assert!(db.remove_context("bert://notes").unwrap());
    /// assert!(!db.remove_context("bert://notes").unwrap()); // already gone
    /// ```
    pub fn remove_context(&self, uri: &str) -> Result<bool> {
        let txn = self.db.begin_write()?;
        let removed = {
            let mut table = txn.open_table(CONTEXTS)?;
            table.remove(uri)?.is_some()
        };
        txn.commit()?;
        Ok(removed)
    }

    /// List all contexts as `(uri, description)` pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_context("bert://a", "A description").unwrap();
    /// db.set_context("bert://b", "B description").unwrap();
    /// let contexts = db.list_contexts().unwrap();
    /// assert_eq!(contexts.len(), 2);
    /// ```
    pub fn list_contexts(&self) -> Result<Vec<(String, String)>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(CONTEXTS)?;
        let mut result = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            result.push((k.value().to_string(), v.value().to_string()));
        }
        Ok(result)
    }

    // -- Document Metadata --

    /// Store serialized metadata for a document by its numeric ID.
    ///
    /// The bytes are opaque to `ConfigDb`; callers typically use
    /// [`DocumentMetadata::serialize`](crate::incremental::DocumentMetadata::serialize).
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_document_metadata(42, b"collection\0path\01000").unwrap();
    /// let bytes = db.get_document_metadata(42).unwrap().unwrap();
    /// assert_eq!(bytes, b"collection\0path\01000");
    /// ```
    pub fn set_document_metadata(
        &self,
        doc_id: u64,
        data: &[u8],
    ) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(DOCUMENT_METADATA)?;
            table.insert(doc_id, data)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Retrieve serialized metadata for a document. Returns `None` if not found.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// assert!(db.get_document_metadata(999).unwrap().is_none());
    /// db.set_document_metadata(42, b"data").unwrap();
    /// assert_eq!(db.get_document_metadata(42).unwrap().unwrap(), b"data");
    /// ```
    pub fn get_document_metadata(
        &self,
        doc_id: u64,
    ) -> Result<Option<Vec<u8>>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(DOCUMENT_METADATA)?;
        Ok(table.get(doc_id)?.map(|v| v.value().to_vec()))
    }

    /// Remove a document's metadata. Returns `true` if it existed.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_document_metadata(42, b"data").unwrap();
    /// assert!(db.remove_document_metadata(42).unwrap());
    /// assert!(!db.remove_document_metadata(42).unwrap());
    /// ```
    pub fn remove_document_metadata(&self, doc_id: u64) -> Result<bool> {
        let txn = self.db.begin_write()?;
        let removed = {
            let mut table = txn.open_table(DOCUMENT_METADATA)?;
            table.remove(doc_id)?.is_some()
        };
        txn.commit()?;
        Ok(removed)
    }

    /// Remove multiple document metadata entries in a single transaction.
    ///
    /// More efficient than calling [`remove_document_metadata`](Self::remove_document_metadata)
    /// in a loop because all removals share one write transaction.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_document_metadata(1, b"a").unwrap();
    /// db.set_document_metadata(2, b"b").unwrap();
    /// db.batch_remove_document_metadata(&[1, 2]).unwrap();
    /// assert!(db.get_document_metadata(1).unwrap().is_none());
    /// assert!(db.get_document_metadata(2).unwrap().is_none());
    /// ```
    pub fn batch_remove_document_metadata(
        &self,
        doc_ids: &[u64],
    ) -> Result<()> {
        if doc_ids.is_empty() {
            return Ok(());
        }
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(DOCUMENT_METADATA)?;
            for &doc_id in doc_ids {
                table.remove(doc_id)?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Set multiple document metadata entries in a single transaction.
    ///
    /// More efficient than calling [`set_document_metadata`](Self::set_document_metadata)
    /// in a loop because all writes share one transaction.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.batch_set_document_metadata(&[
    ///     (1, b"meta_a".to_vec()),
    ///     (2, b"meta_b".to_vec()),
    /// ]).unwrap();
    /// assert_eq!(db.get_document_metadata(1).unwrap().unwrap(), b"meta_a");
    /// assert_eq!(db.get_document_metadata(2).unwrap().unwrap(), b"meta_b");
    /// ```
    pub fn batch_set_document_metadata(
        &self,
        entries: &[(u64, Vec<u8>)],
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(DOCUMENT_METADATA)?;
            for (doc_id, data) in entries {
                table.insert(*doc_id, data.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// List all stored document numeric IDs.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_document_metadata(10, b"a").unwrap();
    /// db.set_document_metadata(20, b"b").unwrap();
    /// let mut ids = db.list_document_ids().unwrap();
    /// ids.sort();
    /// assert_eq!(ids, vec![10, 20]);
    /// ```
    pub fn list_document_ids(&self) -> Result<Vec<u64>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(DOCUMENT_METADATA)?;
        let mut result = Vec::new();
        for entry in table.iter()? {
            let (k, _v) = entry?;
            result.push(k.value());
        }
        Ok(result)
    }

    /// Return all `(doc_id, metadata_bytes)` pairs in a single read transaction.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_document_metadata(1, b"a").unwrap();
    /// db.set_document_metadata(2, b"b").unwrap();
    /// let all = db.list_all_document_metadata().unwrap();
    /// assert_eq!(all.len(), 2);
    /// ```
    pub fn list_all_document_metadata(&self) -> Result<Vec<(u64, Vec<u8>)>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(DOCUMENT_METADATA)?;
        let mut result = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            result.push((k.value(), v.value().to_vec()));
        }
        Ok(result)
    }

    // -- Settings --

    /// Store a key-value setting.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_setting("model_name", "lightonai/ColBERT-Zero").unwrap();
    /// ```
    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(SETTINGS)?;
            table.insert(key, value)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Get a setting value by key. Returns `None` if not set.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// assert!(db.get_setting("model_name").unwrap().is_none());
    /// db.set_setting("model_name", "custom/model").unwrap();
    /// assert_eq!(db.get_setting("model_name").unwrap().unwrap(), "custom/model");
    /// ```
    pub fn get_setting(&self, key: &str) -> Result<Option<String>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(SETTINGS)?;
        Ok(table.get(key)?.map(|v| v.value().to_string()))
    }

    /// Remove a setting by key. Returns `true` if it existed.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_setting("key", "value").unwrap();
    /// assert!(db.remove_setting("key").unwrap());
    /// assert!(!db.remove_setting("key").unwrap());
    /// ```
    pub fn remove_setting(&self, key: &str) -> Result<bool> {
        let txn = self.db.begin_write()?;
        let removed = {
            let mut table = txn.open_table(SETTINGS)?;
            table.remove(key)?.is_some()
        };
        txn.commit()?;
        Ok(removed)
    }

    /// Get a setting, returning the default if not set.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// assert_eq!(db.get_setting_or("model_name", "default/model").unwrap(), "default/model");
    /// db.set_setting("model_name", "custom/model").unwrap();
    /// assert_eq!(db.get_setting_or("model_name", "default/model").unwrap(), "custom/model");
    /// ```
    pub fn get_setting_or(&self, key: &str, default: &str) -> Result<String> {
        Ok(self
            .get_setting(key)?
            .unwrap_or_else(|| default.to_string()))
    }
}

impl std::fmt::Debug for ConfigDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigDb").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> (tempfile::TempDir, ConfigDb) {
        let tmp = tempfile::tempdir().unwrap();
        let db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        (tmp, db)
    }

    #[test]
    fn collections_crud() {
        let (_tmp, db) = test_db();

        assert_eq!(db.list_collections().unwrap(), vec![]);
        assert_eq!(db.get_collection("notes").unwrap(), None);

        db.set_collection("notes", "/home/user/notes").unwrap();
        assert_eq!(
            db.get_collection("notes").unwrap(),
            Some("/home/user/notes".to_string())
        );

        let collections = db.list_collections().unwrap();
        assert_eq!(collections.len(), 1);
        assert_eq!(collections[0].0, "notes");

        assert!(db.remove_collection("notes").unwrap());
        assert!(!db.remove_collection("notes").unwrap());
        assert_eq!(db.get_collection("notes").unwrap(), None);
    }

    #[test]
    fn contexts_crud() {
        let (_tmp, db) = test_db();

        db.set_context("bert://notes", "Personal notes").unwrap();
        assert_eq!(
            db.get_context("bert://notes").unwrap(),
            Some("Personal notes".to_string())
        );

        db.set_context("bert://notes", "Updated description")
            .unwrap();
        assert_eq!(
            db.get_context("bert://notes").unwrap(),
            Some("Updated description".to_string())
        );

        let contexts = db.list_contexts().unwrap();
        assert_eq!(contexts.len(), 1);

        assert!(db.remove_context("bert://notes").unwrap());
        assert_eq!(db.list_contexts().unwrap(), vec![]);
    }

    #[test]
    fn document_metadata_crud() {
        let (_tmp, db) = test_db();

        let data = b"test metadata bytes";
        db.set_document_metadata(42, data).unwrap();

        let retrieved = db.get_document_metadata(42).unwrap().unwrap();
        assert_eq!(retrieved, data);

        let ids = db.list_document_ids().unwrap();
        assert_eq!(ids, vec![42]);

        assert!(db.remove_document_metadata(42).unwrap());
        assert_eq!(db.get_document_metadata(42).unwrap(), None);
    }

    #[test]
    fn settings_crud() {
        let (_tmp, db) = test_db();

        assert_eq!(db.get_setting("model_name").unwrap(), None);
        assert_eq!(
            db.get_setting_or("model_name", "default-model").unwrap(),
            "default-model"
        );

        db.set_setting("model_name", "custom-model").unwrap();
        assert_eq!(
            db.get_setting("model_name").unwrap(),
            Some("custom-model".to_string())
        );
        assert_eq!(
            db.get_setting_or("model_name", "default-model").unwrap(),
            "custom-model"
        );

        assert!(db.remove_setting("model_name").unwrap());
        assert_eq!(db.get_setting("model_name").unwrap(), None);
    }

    #[test]
    fn not_found_returns_none() {
        let (_tmp, db) = test_db();

        assert_eq!(db.get_collection("nonexistent").unwrap(), None);
        assert_eq!(db.get_context("nonexistent").unwrap(), None);
        assert_eq!(db.get_document_metadata(999).unwrap(), None);
        assert_eq!(db.get_setting("nonexistent").unwrap(), None);
    }

    #[test]
    fn reopen_preserves_data() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.db");

        {
            let db = ConfigDb::open(&path).unwrap();
            db.set_collection("notes", "/path/to/notes").unwrap();
            db.set_setting("version", "1").unwrap();
        }

        {
            let db = ConfigDb::open(&path).unwrap();
            assert_eq!(
                db.get_collection("notes").unwrap(),
                Some("/path/to/notes".to_string())
            );
            assert_eq!(
                db.get_setting("version").unwrap(),
                Some("1".to_string())
            );
        }
    }

    #[test]
    fn collection_not_found_returns_none() {
        let (_tmp, db) = test_db();
        assert!(db.get_collection("ghost").unwrap().is_none());
    }
}
