use std::path::Path;

use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};

use crate::{
    Conversation,
    Error,
    error::Result,
    incremental::DocumentMetadata,
    storage_codec::{decode_bytes, encode_bytes},
    stored_json::StoredJsonValue,
};

const COLLECTIONS: TableDefinition<&str, &[u8]> =
    TableDefinition::new("collections");
const CONTEXTS: TableDefinition<&str, &[u8]> = TableDefinition::new("contexts");
const DOCUMENT_METADATA: TableDefinition<u64, &[u8]> =
    TableDefinition::new("document_metadata");
const CONVERSATIONS: TableDefinition<&str, &[u8]> =
    TableDefinition::new("conversations");
const SETTINGS: TableDefinition<&str, &[u8]> = TableDefinition::new("settings");

const KEY_LLM_PROVIDER: &str = "llm_provider";
const KEY_LLM_MODEL: &str = "llm_model";
const KEY_LLM_API_KEY: &str = "llm_api_key";

/// redb-backed store for collections, settings, and document metadata.
///
/// It keeps four tables:
/// - **collections**: collection names to filesystem paths
/// - **contexts**: URIs to human-readable descriptions
/// - **document_metadata**: numeric document IDs to serialized metadata
/// - **settings**: general key-value settings such as `model_name`
///
/// The database itself is [redb](https://github.com/cberner/redb), so everything stays local.
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// use docbert_core::ConfigDb;
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PersistedLlmSettings {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
}

fn encode_string(value: &str) -> Result<Vec<u8>> {
    encode_bytes(&value.to_string())
}

fn decode_aligned<T>(bytes: &[u8]) -> Result<T>
where
    T: rkyv::Archive,
    T::Archived: for<'a> rkyv::bytecheck::CheckBytes<
            rkyv::api::high::HighValidator<'a, rkyv::rancor::Error>,
        > + rkyv::Deserialize<
            T,
            rkyv::api::high::HighDeserializer<rkyv::rancor::Error>,
        >,
{
    let mut aligned = rkyv::util::AlignedVec::<16>::new();
    aligned.extend_from_slice(bytes);
    decode_bytes(&aligned)
}

fn decode_string(bytes: &[u8]) -> Result<String> {
    decode_aligned(bytes)
}

fn document_content_key(doc_id: u64) -> String {
    format!("doc_content:{doc_id}")
}

fn document_user_metadata_key(doc_id: u64) -> String {
    format!("doc_meta:{doc_id}")
}

fn map_schema_error(err: redb::TableError) -> Error {
    match err {
        redb::TableError::TableTypeMismatch { .. }
        | redb::TableError::TypeDefinitionChanged { .. } => Error::Config(
            "config.db uses an older incompatible schema; back up the file and remove or reset config.db before restarting".to_string(),
        ),
        other => other.into(),
    }
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
    /// use docbert_core::ConfigDb;
    /// let db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// ```
    pub fn open(path: &Path) -> Result<Self> {
        let db = Database::create(path)?;

        // Ensure all tables exist by opening them in a write transaction.
        let txn = db.begin_write()?;
        txn.open_table(COLLECTIONS).map_err(map_schema_error)?;
        txn.open_table(CONTEXTS).map_err(map_schema_error)?;
        txn.open_table(CONVERSATIONS).map_err(map_schema_error)?;
        txn.open_table(DOCUMENT_METADATA)
            .map_err(map_schema_error)?;
        txn.open_table(SETTINGS).map_err(map_schema_error)?;
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
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_collection("notes", "/home/user/notes").unwrap();
    /// ```
    pub fn set_collection(&self, name: &str, path: &str) -> Result<()> {
        let encoded = encode_string(path)?;
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(COLLECTIONS)?;
            table.insert(name, encoded.as_slice())?;
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
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// assert_eq!(db.get_collection("nonexistent").unwrap(), None);
    /// db.set_collection("notes", "/path").unwrap();
    /// assert_eq!(db.get_collection("notes").unwrap(), Some("/path".to_string()));
    /// ```
    pub fn get_collection(&self, name: &str) -> Result<Option<String>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(COLLECTIONS)?;
        table
            .get(name)?
            .map(|v| decode_string(v.value()))
            .transpose()
    }

    /// Remove a collection by name. Returns `true` if it existed.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
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
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
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
            result.push((k.value().to_string(), decode_string(v.value())?));
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
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_context("bert://notes", "Personal research notes").unwrap();
    /// ```
    pub fn set_context(&self, uri: &str, description: &str) -> Result<()> {
        let encoded = encode_string(description)?;
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(CONTEXTS)?;
            table.insert(uri, encoded.as_slice())?;
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
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
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
        table
            .get(uri)?
            .map(|v| decode_string(v.value()))
            .transpose()
    }

    /// Remove a context by URI. Returns `true` if it existed.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
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
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
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
            result.push((k.value().to_string(), decode_string(v.value())?));
        }
        Ok(result)
    }

    // -- Conversations --

    /// Store a typed conversation keyed by its ID.
    pub fn set_conversation_typed(
        &self,
        conversation: &Conversation,
    ) -> Result<()> {
        let data = conversation.serialize()?;
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(CONVERSATIONS)?;
            table.insert(conversation.id.as_str(), data.as_slice())?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Retrieve a typed conversation by ID. Returns `None` if not found.
    pub fn get_conversation_typed(
        &self,
        id: &str,
    ) -> Result<Option<Conversation>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(CONVERSATIONS)?;
        table
            .get(id)?
            .map(|bytes| {
                let mut aligned = rkyv::util::AlignedVec::<16>::new();
                aligned.extend_from_slice(bytes.value());
                Conversation::deserialize(&aligned)
            })
            .transpose()
    }

    /// Remove a conversation by ID. Returns `true` if it existed.
    pub fn remove_conversation(&self, id: &str) -> Result<bool> {
        let txn = self.db.begin_write()?;
        let removed = {
            let mut table = txn.open_table(CONVERSATIONS)?;
            table.remove(id)?.is_some()
        };
        txn.commit()?;
        Ok(removed)
    }

    /// List all conversations as typed records.
    pub fn list_conversations_typed(&self) -> Result<Vec<Conversation>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(CONVERSATIONS)?;
        let mut result = Vec::new();
        for entry in table.iter()? {
            let (_id, bytes) = entry?;
            let mut aligned = rkyv::util::AlignedVec::<16>::new();
            aligned.extend_from_slice(bytes.value());
            result.push(Conversation::deserialize(&aligned)?);
        }
        Ok(result)
    }

    // -- Document Metadata --

    /// Remove a document's metadata. Returns `true` if it existed.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// let meta = docbert_core::incremental::DocumentMetadata {
    ///     collection: "notes".to_string(),
    ///     relative_path: "a.md".to_string(),
    ///     mtime: 1,
    /// };
    /// db.set_document_metadata_typed(42, &meta).unwrap();
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
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// let first = docbert_core::incremental::DocumentMetadata {
    ///     collection: "notes".to_string(),
    ///     relative_path: "a.md".to_string(),
    ///     mtime: 1,
    /// };
    /// let second = docbert_core::incremental::DocumentMetadata {
    ///     collection: "notes".to_string(),
    ///     relative_path: "b.md".to_string(),
    ///     mtime: 2,
    /// };
    /// db.set_document_metadata_typed(1, &first).unwrap();
    /// db.set_document_metadata_typed(2, &second).unwrap();
    /// db.batch_remove_document_metadata(&[1, 2]).unwrap();
    /// assert!(db.get_document_metadata_typed(1).unwrap().is_none());
    /// assert!(db.get_document_metadata_typed(2).unwrap().is_none());
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

    /// Store typed metadata for a document by its numeric ID.
    pub fn set_document_metadata_typed(
        &self,
        doc_id: u64,
        metadata: &DocumentMetadata,
    ) -> Result<()> {
        let data = metadata.serialize();
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(DOCUMENT_METADATA)?;
            table.insert(doc_id, data.as_slice())?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Retrieve typed metadata for a document. Returns `None` if not found.
    pub fn get_document_metadata_typed(
        &self,
        doc_id: u64,
    ) -> Result<Option<DocumentMetadata>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(DOCUMENT_METADATA)?;
        table
            .get(doc_id)?
            .map(|bytes| decode_aligned::<DocumentMetadata>(bytes.value()))
            .transpose()
    }

    /// Set multiple typed document metadata entries in a single transaction.
    pub fn batch_set_document_metadata_typed(
        &self,
        entries: &[(u64, DocumentMetadata)],
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(DOCUMENT_METADATA)?;
            for (doc_id, metadata) in entries {
                let data = metadata.serialize();
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
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// let first = docbert_core::incremental::DocumentMetadata {
    ///     collection: "notes".to_string(),
    ///     relative_path: "a.md".to_string(),
    ///     mtime: 1,
    /// };
    /// let second = docbert_core::incremental::DocumentMetadata {
    ///     collection: "notes".to_string(),
    ///     relative_path: "b.md".to_string(),
    ///     mtime: 2,
    /// };
    /// db.set_document_metadata_typed(10, &first).unwrap();
    /// db.set_document_metadata_typed(20, &second).unwrap();
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

    /// Return all `(doc_id, metadata)` pairs in a single read transaction.
    pub fn list_all_document_metadata_typed(
        &self,
    ) -> Result<Vec<(u64, DocumentMetadata)>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(DOCUMENT_METADATA)?;
        let mut result = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            result.push((
                k.value(),
                decode_aligned::<DocumentMetadata>(v.value())?,
            ));
        }
        Ok(result)
    }

    /// Look up a document by its short hex display id.
    pub fn find_document_by_short_id(
        &self,
        short_id: &str,
    ) -> Result<Option<(u64, DocumentMetadata)>> {
        let entries = self.list_all_document_metadata_typed()?;
        Ok(entries.into_iter().find(|(_doc_id, meta)| {
            let did = crate::doc_id::DocumentId::new(
                &meta.collection,
                &meta.relative_path,
            );
            did.short == short_id
        }))
    }

    /// Look up a document by its relative path across all collections.
    pub fn find_document_by_path(
        &self,
        path: &str,
    ) -> Result<Option<(u64, DocumentMetadata)>> {
        let entries = self.list_all_document_metadata_typed()?;
        Ok(entries
            .into_iter()
            .find(|(_doc_id, meta)| meta.relative_path == path))
    }

    // -- Settings --

    /// Store a key-value setting.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// db.set_setting("model_name", "lightonai/ColBERT-Zero").unwrap();
    /// ```
    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        let encoded = encode_string(value)?;
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(SETTINGS)?;
            table.insert(key, encoded.as_slice())?;
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
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// assert!(db.get_setting("model_name").unwrap().is_none());
    /// db.set_setting("model_name", "custom/model").unwrap();
    /// assert_eq!(db.get_setting("model_name").unwrap().unwrap(), "custom/model");
    /// ```
    pub fn get_setting(&self, key: &str) -> Result<Option<String>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(SETTINGS)?;
        table
            .get(key)?
            .map(|v| decode_string(v.value()))
            .transpose()
    }

    /// Remove a setting by key. Returns `true` if it existed.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
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
    /// # let db = docbert_core::ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// assert_eq!(db.get_setting_or("model_name", "default/model").unwrap(), "default/model");
    /// db.set_setting("model_name", "custom/model").unwrap();
    /// assert_eq!(db.get_setting_or("model_name", "default/model").unwrap(), "custom/model");
    /// ```
    pub fn get_setting_or(&self, key: &str, default: &str) -> Result<String> {
        Ok(self
            .get_setting(key)?
            .unwrap_or_else(|| default.to_string()))
    }

    /// Load the persisted LLM settings stored in config.db.
    pub fn get_persisted_llm_settings(&self) -> Result<PersistedLlmSettings> {
        Ok(PersistedLlmSettings {
            provider: self.get_setting(KEY_LLM_PROVIDER)?,
            model: self.get_setting(KEY_LLM_MODEL)?,
            api_key: self.get_setting(KEY_LLM_API_KEY)?,
        })
    }

    /// Replace the persisted LLM settings in a single write transaction.
    pub fn set_persisted_llm_settings(
        &self,
        settings: &PersistedLlmSettings,
    ) -> Result<()> {
        let provider = settings
            .provider
            .as_deref()
            .map(encode_string)
            .transpose()?;
        let model = settings.model.as_deref().map(encode_string).transpose()?;
        let api_key =
            settings.api_key.as_deref().map(encode_string).transpose()?;

        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(SETTINGS)?;
            match provider.as_deref() {
                Some(bytes) => {
                    table.insert(KEY_LLM_PROVIDER, bytes)?;
                }
                None => {
                    table.remove(KEY_LLM_PROVIDER)?;
                }
            }
            match model.as_deref() {
                Some(bytes) => {
                    table.insert(KEY_LLM_MODEL, bytes)?;
                }
                None => {
                    table.remove(KEY_LLM_MODEL)?;
                }
            }
            match api_key.as_deref() {
                Some(bytes) => {
                    table.insert(KEY_LLM_API_KEY, bytes)?;
                }
                None => {
                    table.remove(KEY_LLM_API_KEY)?;
                }
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Store a structured JSON value under a settings key.
    pub fn set_json_setting(
        &self,
        key: &str,
        value: &serde_json::Value,
    ) -> Result<()> {
        let encoded = encode_bytes(&StoredJsonValue::from(value.clone()))?;
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(SETTINGS)?;
            table.insert(key, encoded.as_slice())?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Load a structured JSON value from a settings key.
    pub fn get_json_setting(
        &self,
        key: &str,
    ) -> Result<Option<serde_json::Value>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(SETTINGS)?;
        table
            .get(key)?
            .map(|v| {
                decode_aligned::<StoredJsonValue>(v.value()).map(Into::into)
            })
            .transpose()
    }

    /// Remove a structured JSON setting by key. Returns `true` if it existed.
    pub fn remove_json_setting(&self, key: &str) -> Result<bool> {
        self.remove_setting(key)
    }

    /// Remove multiple documents' metadata, stored content, and optional user
    /// metadata in a single write transaction.
    pub fn batch_remove_document_state(
        &self,
        doc_ids: &[u64],
    ) -> Result<()> {
        if doc_ids.is_empty() {
            return Ok(());
        }

        let txn = self.db.begin_write()?;
        {
            let mut metadata_table = txn.open_table(DOCUMENT_METADATA)?;
            for &doc_id in doc_ids {
                metadata_table.remove(doc_id)?;
            }
        }
        {
            let mut settings_table = txn.open_table(SETTINGS)?;
            for &doc_id in doc_ids {
                let content_key = document_content_key(doc_id);
                let user_metadata_key = document_user_metadata_key(doc_id);
                settings_table.remove(content_key.as_str())?;
                settings_table.remove(user_metadata_key.as_str())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Store user-provided document metadata in the shared settings table.
    pub fn set_document_user_metadata(
        &self,
        doc_id: u64,
        value: &serde_json::Value,
    ) -> Result<()> {
        let key = document_user_metadata_key(doc_id);
        self.set_json_setting(&key, value)
    }

    /// Load user-provided document metadata from the shared settings table.
    pub fn get_document_user_metadata(
        &self,
        doc_id: u64,
    ) -> Result<Option<serde_json::Value>> {
        let key = document_user_metadata_key(doc_id);
        self.get_json_setting(&key)
    }

    /// Remove user-provided document metadata from the shared settings table.
    pub fn remove_document_user_metadata(&self, doc_id: u64) -> Result<bool> {
        let key = document_user_metadata_key(doc_id);
        self.remove_json_setting(&key)
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
    fn filesystem_collections_only_roundtrip_collection_paths() {
        let (tmp, db) = test_db();
        db.set_collection("notes", "/tmp/notes").unwrap();
        drop(db);

        let reopened = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        assert_eq!(
            reopened.get_collection("notes").unwrap(),
            Some("/tmp/notes".to_string())
        );
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
        assert_eq!(db.get_document_metadata_typed(999).unwrap(), None);
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

    #[test]
    fn open_rejects_legacy_string_tables_with_reset_message() {
        const LEGACY_COLLECTIONS: TableDefinition<&str, &str> =
            TableDefinition::new("collections");
        const LEGACY_CONTEXTS: TableDefinition<&str, &str> =
            TableDefinition::new("contexts");
        const LEGACY_SETTINGS: TableDefinition<&str, &str> =
            TableDefinition::new("settings");

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.db");
        let db = Database::create(&path).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut collections = txn.open_table(LEGACY_COLLECTIONS).unwrap();
            collections.insert("notes", "/tmp/notes").unwrap();
        }
        {
            let mut contexts = txn.open_table(LEGACY_CONTEXTS).unwrap();
            contexts.insert("bert://notes", "legacy context").unwrap();
        }
        {
            let mut settings = txn.open_table(LEGACY_SETTINGS).unwrap();
            settings.insert("model_name", "legacy-model").unwrap();
        }
        txn.commit().unwrap();
        drop(db);

        let err = ConfigDb::open(&path).unwrap_err();
        assert!(
            err.to_string().contains("remove or reset config.db"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn json_setting_roundtrips() {
        let (_tmp, db) = test_db();
        let value = serde_json::json!({
            "enabled": true,
            "score": 0.75,
            "tags": ["rust", "storage"],
            "limits": { "top_k": u64::MAX },
            "note": null,
        });

        db.set_json_setting("search_config", &value).unwrap();
        assert_eq!(db.get_json_setting("search_config").unwrap(), Some(value));
        assert!(db.remove_json_setting("search_config").unwrap());
        assert!(db.get_json_setting("search_config").unwrap().is_none());
    }

    #[test]
    fn persisted_llm_settings_roundtrip_all_fields() {
        let (_tmp, db) = test_db();
        let settings = PersistedLlmSettings {
            provider: Some("openai".to_string()),
            model: Some("gpt-4.1".to_string()),
            api_key: Some("secret-key".to_string()),
        };

        db.set_persisted_llm_settings(&settings).unwrap();

        assert_eq!(db.get_persisted_llm_settings().unwrap(), settings);
    }

    #[test]
    fn persisted_llm_settings_clears_absent_fields() {
        let (_tmp, db) = test_db();

        db.set_persisted_llm_settings(&PersistedLlmSettings {
            provider: Some("openai".to_string()),
            model: Some("gpt-4.1".to_string()),
            api_key: Some("secret-key".to_string()),
        })
        .unwrap();
        db.set_persisted_llm_settings(&PersistedLlmSettings {
            provider: None,
            model: None,
            api_key: None,
        })
        .unwrap();

        assert_eq!(
            db.get_persisted_llm_settings().unwrap(),
            PersistedLlmSettings::default()
        );
    }

    #[test]
    fn document_user_metadata_roundtrips() {
        let (_tmp, db) = test_db();
        let value = serde_json::json!({
            "title": "Hello",
            "priority": 3,
            "archived": false,
            "attrs": { "lang": "en" },
        });

        db.set_document_user_metadata(7, &value).unwrap();
        assert_eq!(db.get_document_user_metadata(7).unwrap(), Some(value));
        assert!(db.remove_document_user_metadata(7).unwrap());
        assert!(db.get_document_user_metadata(7).unwrap().is_none());
    }

    #[test]
    fn typed_document_metadata_crud() {
        let (_tmp, db) = test_db();
        let metadata = DocumentMetadata {
            collection: "notes".to_string(),
            relative_path: "hello.md".to_string(),
            mtime: 12345,
        };

        db.set_document_metadata_typed(42, &metadata).unwrap();

        let loaded = db
            .get_document_metadata_typed(42)
            .unwrap()
            .expect("document metadata should exist");
        assert_eq!(loaded, metadata);

        let listed = db.list_all_document_metadata_typed().unwrap();
        assert_eq!(listed, vec![(42, metadata.clone())]);

        assert!(db.remove_document_metadata(42).unwrap());
        assert!(db.get_document_metadata_typed(42).unwrap().is_none());
    }

    #[test]
    fn typed_document_metadata_batch_set_roundtrips() {
        let (_tmp, db) = test_db();
        let entries = vec![
            (
                1,
                DocumentMetadata {
                    collection: "notes".to_string(),
                    relative_path: "a.md".to_string(),
                    mtime: 100,
                },
            ),
            (
                2,
                DocumentMetadata {
                    collection: "docs".to_string(),
                    relative_path: "b.md".to_string(),
                    mtime: 200,
                },
            ),
        ];

        db.batch_set_document_metadata_typed(&entries).unwrap();

        let mut listed = db.list_all_document_metadata_typed().unwrap();
        listed.sort_by_key(|(doc_id, _)| *doc_id);
        assert_eq!(listed, entries);
    }

    #[test]
    fn typed_conversation_crud() {
        let (_tmp, db) = test_db();
        let conversation = Conversation {
            id: "conv-1".to_string(),
            title: "Chat".to_string(),
            created_at: 1,
            updated_at: 2,
            messages: vec![crate::conversation::ChatMessage {
                id: "msg-1".to_string(),
                role: crate::conversation::ChatRole::Assistant,
                actor: Some(crate::conversation::ChatActor::Parent),
                parts: vec![crate::conversation::ChatPart::ToolCall {
                    name: "search_hybrid".to_string(),
                    args: serde_json::json!({
                        "query": "rust",
                        "top_k": 5,
                        "filters": { "collection": "notes" }
                    }),
                    result: Some("[]".to_string()),
                    is_error: false,
                }],
                sources: Some(vec![crate::conversation::ChatSource {
                    collection: "notes".to_string(),
                    path: "rust.md".to_string(),
                    title: "Rust".to_string(),
                }]),
            }],
        };

        db.set_conversation_typed(&conversation).unwrap();

        let loaded = db
            .get_conversation_typed("conv-1")
            .unwrap()
            .expect("conversation should exist");
        assert_eq!(loaded.id, conversation.id);
        assert_eq!(loaded.title, conversation.title);
        assert_eq!(loaded.messages.len(), conversation.messages.len());
        assert_eq!(loaded.messages[0].id, conversation.messages[0].id);
        assert_eq!(loaded.messages[0].role, conversation.messages[0].role);
        assert_eq!(loaded.messages[0].actor, conversation.messages[0].actor);
        assert_eq!(loaded.messages[0].parts, conversation.messages[0].parts);
        assert_eq!(
            loaded.messages[0].sources,
            conversation.messages[0].sources
        );

        let listed = db.list_conversations_typed().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, conversation.id);
        assert_eq!(listed[0].messages.len(), conversation.messages.len());
        assert_eq!(listed[0].messages[0].parts, conversation.messages[0].parts);

        assert!(db.remove_conversation("conv-1").unwrap());
        assert!(db.get_conversation_typed("conv-1").unwrap().is_none());
    }
}
