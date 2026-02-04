use std::path::Path;

use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};

use crate::error::Result;

const COLLECTIONS: TableDefinition<&str, &str> =
    TableDefinition::new("collections");
const CONTEXTS: TableDefinition<&str, &str> = TableDefinition::new("contexts");
const DOCUMENT_METADATA: TableDefinition<u64, &[u8]> =
    TableDefinition::new("document_metadata");
const SETTINGS: TableDefinition<&str, &str> = TableDefinition::new("settings");

pub struct ConfigDb {
    db: Database,
}

impl ConfigDb {
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

    pub fn set_collection(&self, name: &str, path: &str) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(COLLECTIONS)?;
            table.insert(name, path)?;
        }
        txn.commit()?;
        Ok(())
    }

    pub fn get_collection(&self, name: &str) -> Result<Option<String>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(COLLECTIONS)?;
        Ok(table.get(name)?.map(|v| v.value().to_string()))
    }

    pub fn remove_collection(&self, name: &str) -> Result<bool> {
        let txn = self.db.begin_write()?;
        let removed = {
            let mut table = txn.open_table(COLLECTIONS)?;
            table.remove(name)?.is_some()
        };
        txn.commit()?;
        Ok(removed)
    }

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

    pub fn set_context(&self, uri: &str, description: &str) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(CONTEXTS)?;
            table.insert(uri, description)?;
        }
        txn.commit()?;
        Ok(())
    }

    pub fn get_context(&self, uri: &str) -> Result<Option<String>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(CONTEXTS)?;
        Ok(table.get(uri)?.map(|v| v.value().to_string()))
    }

    pub fn remove_context(&self, uri: &str) -> Result<bool> {
        let txn = self.db.begin_write()?;
        let removed = {
            let mut table = txn.open_table(CONTEXTS)?;
            table.remove(uri)?.is_some()
        };
        txn.commit()?;
        Ok(removed)
    }

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

    pub fn get_document_metadata(
        &self,
        doc_id: u64,
    ) -> Result<Option<Vec<u8>>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(DOCUMENT_METADATA)?;
        Ok(table.get(doc_id)?.map(|v| v.value().to_vec()))
    }

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

    /// Return all (doc_id, metadata_bytes) pairs in a single read transaction.
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

    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(SETTINGS)?;
            table.insert(key, value)?;
        }
        txn.commit()?;
        Ok(())
    }

    pub fn get_setting(&self, key: &str) -> Result<Option<String>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(SETTINGS)?;
        Ok(table.get(key)?.map(|v| v.value().to_string()))
    }

    /// Get a setting, returning the default if not set.
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
