use std::path::Path;

use heed::{
    Database,
    Env,
    byteorder::BigEndian,
    types::{Bytes, Str, U64},
};

use crate::{
    Conversation,
    error::Result,
    incremental::DocumentMetadata,
    merkle::Snapshot,
    redb_migration::{self, CONFIG_MAX_DBS},
    storage_codec::{decode_bytes, encode_bytes},
    stored_json::StoredJsonValue,
};

const COLLECTIONS_DB: &str = "collections";
const CONTEXTS_DB: &str = "contexts";
const DOCUMENT_METADATA_DB: &str = "document_metadata";
const CONVERSATIONS_DB: &str = "conversations";
const COLLECTION_MERKLE_SNAPSHOTS_DB: &str = "collection_merkle_snapshots";
const SETTINGS_DB: &str = "settings";
/// Per-document chunk manifest.
///
/// Keyed by `doc_num_id` and storing the ordered list of chunks the
/// document was split into. Each entry carries the chunk's
/// content-derived id (see [`crate::chunking::chunk_doc_id`]) plus the
/// byte range it occupies in the source document.
///
/// Because chunk ids are now content-derived, the same chunk text can
/// belong to many documents at different byte offsets — the manifest
/// is the per-doc record of those offsets, complementing
/// [`CHUNK_OWNERS_DB`] which records ownership in the reverse
/// direction.
const DOC_CHUNKS_DB: &str = "doc_chunks";
/// Reverse index: chunk id → documents that contain it.
///
/// Keyed by `chunk_doc_id` and storing a sorted, deduplicated list of
/// `doc_num_id` values that include this chunk. The semantic search
/// path uses this to fan out a chunk hit in the PLAID index back to
/// every document that owns it.
const CHUNK_OWNERS_DB: &str = "chunk_owners";

const MAP_SIZE: usize = 1024 * 1024 * 1024; // 1 GiB

const KEY_LLM_PROVIDER: &str = "llm_provider";
const KEY_LLM_MODEL: &str = "llm_model";
const KEY_LLM_API_KEY: &str = "llm_api_key";

/// Local store for collections, settings, and document metadata.
///
/// It keeps eight named LMDB databases inside one
/// [`heed::Env`](https://docs.rs/heed):
///
/// - **collections**: collection names to filesystem paths
/// - **contexts**: URIs to human-readable descriptions
/// - **document_metadata**: numeric document IDs to serialized metadata
/// - **conversations**: conversation IDs to serialized chat history
/// - **collection_merkle_snapshots**: collection name to last snapshot
/// - **settings**: general key-value settings such as `model_name`
/// - **doc_chunks**: numeric document ID to its ordered chunk manifest
///   (each entry pairs a content-derived chunk id with its byte range)
/// - **chunk_owners**: chunk id to the set of documents that contain
///   it — populated atomically alongside `doc_chunks`
///
/// LMDB gives us proper cross-process readers and writers, so several
/// `docbert mcp` / `docbert web` / CLI processes can share the same
/// data dir. Legacy redb-formatted files are migrated on first open;
/// see [`crate::redb_migration`].
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
    env: Env,
    collections: Database<Str, Bytes>,
    contexts: Database<Str, Bytes>,
    document_metadata: Database<U64<BigEndian>, Bytes>,
    conversations: Database<Str, Bytes>,
    collection_merkle_snapshots: Database<Str, Bytes>,
    settings: Database<Str, Bytes>,
    doc_chunks: Database<U64<BigEndian>, Bytes>,
    chunk_owners: Database<U64<BigEndian>, Bytes>,
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

impl ConfigDb {
    /// Open or create a config database at the given path.
    ///
    /// Creates all required named databases on first open. If the file
    /// at `path` is still in the legacy redb format, this transparently
    /// migrates it to the heed/LMDB format before returning. The
    /// original is preserved as `<path>.redb-bak`.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// use docbert_core::ConfigDb;
    /// let db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
    /// ```
    pub fn open(path: &Path) -> Result<Self> {
        redb_migration::ensure_config_db_migrated(path)?;
        let env =
            redb_migration::open_heed_env(path, MAP_SIZE, CONFIG_MAX_DBS)?;
        let mut wtxn = env.write_txn()?;
        let collections =
            env.create_database(&mut wtxn, Some(COLLECTIONS_DB))?;
        let contexts = env.create_database(&mut wtxn, Some(CONTEXTS_DB))?;
        let document_metadata =
            env.create_database(&mut wtxn, Some(DOCUMENT_METADATA_DB))?;
        let conversations =
            env.create_database(&mut wtxn, Some(CONVERSATIONS_DB))?;
        let collection_merkle_snapshots = env
            .create_database(&mut wtxn, Some(COLLECTION_MERKLE_SNAPSHOTS_DB))?;
        let settings = env.create_database(&mut wtxn, Some(SETTINGS_DB))?;
        let doc_chunks = env.create_database(&mut wtxn, Some(DOC_CHUNKS_DB))?;
        let chunk_owners =
            env.create_database(&mut wtxn, Some(CHUNK_OWNERS_DB))?;
        wtxn.commit()?;
        Ok(Self {
            env,
            collections,
            contexts,
            document_metadata,
            conversations,
            collection_merkle_snapshots,
            settings,
            doc_chunks,
            chunk_owners,
        })
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
        let mut wtxn = self.env.write_txn()?;
        self.collections.put(&mut wtxn, name, encoded.as_slice())?;
        wtxn.commit()?;
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
        let rtxn = self.env.read_txn()?;
        self.collections
            .get(&rtxn, name)?
            .map(decode_string)
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
        let mut wtxn = self.env.write_txn()?;
        let removed = self.collections.delete(&mut wtxn, name)?;
        wtxn.commit()?;
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
        let rtxn = self.env.read_txn()?;
        let mut result = Vec::new();
        for entry in self.collections.iter(&rtxn)? {
            let (k, v) = entry?;
            result.push((k.to_string(), decode_string(v)?));
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
        let mut wtxn = self.env.write_txn()?;
        self.contexts.put(&mut wtxn, uri, encoded.as_slice())?;
        wtxn.commit()?;
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
        let rtxn = self.env.read_txn()?;
        self.contexts
            .get(&rtxn, uri)?
            .map(decode_string)
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
        let mut wtxn = self.env.write_txn()?;
        let removed = self.contexts.delete(&mut wtxn, uri)?;
        wtxn.commit()?;
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
        let rtxn = self.env.read_txn()?;
        let mut result = Vec::new();
        for entry in self.contexts.iter(&rtxn)? {
            let (k, v) = entry?;
            result.push((k.to_string(), decode_string(v)?));
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
        let mut wtxn = self.env.write_txn()?;
        self.conversations.put(
            &mut wtxn,
            conversation.id.as_str(),
            data.as_slice(),
        )?;
        wtxn.commit()?;
        Ok(())
    }

    /// Retrieve a typed conversation by ID. Returns `None` if not found.
    pub fn get_conversation_typed(
        &self,
        id: &str,
    ) -> Result<Option<Conversation>> {
        let rtxn = self.env.read_txn()?;
        let Some(bytes) = self.conversations.get(&rtxn, id)? else {
            return Ok(None);
        };
        let mut aligned = rkyv::util::AlignedVec::<16>::new();
        aligned.extend_from_slice(bytes);
        Conversation::deserialize(&aligned).map(Some)
    }

    /// Remove a conversation by ID. Returns `true` if it existed.
    pub fn remove_conversation(&self, id: &str) -> Result<bool> {
        let mut wtxn = self.env.write_txn()?;
        let removed = self.conversations.delete(&mut wtxn, id)?;
        wtxn.commit()?;
        Ok(removed)
    }

    /// List all conversations as typed records.
    pub fn list_conversations_typed(&self) -> Result<Vec<Conversation>> {
        let rtxn = self.env.read_txn()?;
        let mut result = Vec::new();
        for entry in self.conversations.iter(&rtxn)? {
            let (_id, bytes) = entry?;
            let mut aligned = rkyv::util::AlignedVec::<16>::new();
            aligned.extend_from_slice(bytes);
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
        let mut wtxn = self.env.write_txn()?;
        let removed = self.document_metadata.delete(&mut wtxn, &doc_id)?;
        wtxn.commit()?;
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
        let mut wtxn = self.env.write_txn()?;
        for &doc_id in doc_ids {
            self.document_metadata.delete(&mut wtxn, &doc_id)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Store typed metadata for a document by its numeric ID.
    pub fn set_document_metadata_typed(
        &self,
        doc_id: u64,
        metadata: &DocumentMetadata,
    ) -> Result<()> {
        let data = metadata.serialize()?;
        let mut wtxn = self.env.write_txn()?;
        self.document_metadata
            .put(&mut wtxn, &doc_id, data.as_slice())?;
        wtxn.commit()?;
        Ok(())
    }

    /// Retrieve typed metadata for a document. Returns `None` if not found.
    pub fn get_document_metadata_typed(
        &self,
        doc_id: u64,
    ) -> Result<Option<DocumentMetadata>> {
        let rtxn = self.env.read_txn()?;
        self.document_metadata
            .get(&rtxn, &doc_id)?
            .map(decode_aligned::<DocumentMetadata>)
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

        let mut wtxn = self.env.write_txn()?;
        for (doc_id, metadata) in entries {
            let data = metadata.serialize()?;
            self.document_metadata
                .put(&mut wtxn, doc_id, data.as_slice())?;
        }
        wtxn.commit()?;
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
        let rtxn = self.env.read_txn()?;
        let mut result = Vec::new();
        for entry in self.document_metadata.iter(&rtxn)? {
            let (k, _) = entry?;
            result.push(k);
        }
        Ok(result)
    }

    /// Return all `(doc_id, metadata)` pairs in a single read transaction.
    pub fn list_all_document_metadata_typed(
        &self,
    ) -> Result<Vec<(u64, DocumentMetadata)>> {
        let rtxn = self.env.read_txn()?;
        let mut result = Vec::new();
        for entry in self.document_metadata.iter(&rtxn)? {
            let (k, v) = entry?;
            result.push((k, decode_aligned::<DocumentMetadata>(v)?));
        }
        Ok(result)
    }

    /// Compute the shortest unique hex prefix for a document ID.
    ///
    /// Returns a `#`-prefixed string with at least 6 hex chars, extended
    /// as needed to avoid collisions with other documents in the corpus.
    pub fn disambiguated_short_id(
        &self,
        did: &crate::doc_id::DocumentId,
    ) -> Result<String> {
        let target_hex = did.full_hex();
        let all = self.list_all_document_metadata_typed()?;
        let mut min_len = 6usize;

        for (_doc_id, meta) in &all {
            let other = crate::doc_id::DocumentId::new(
                &meta.collection,
                &meta.relative_path,
            );
            if other.numeric == did.numeric {
                continue;
            }
            let other_hex = other.full_hex();
            let shared = target_hex
                .chars()
                .zip(other_hex.chars())
                .take_while(|(a, b)| a == b)
                .count();
            if shared >= min_len {
                min_len = (shared + 1).min(64);
            }
        }

        Ok(crate::doc_id::format_document_ref(&target_hex[..min_len]))
    }

    /// Look up a document by a hex prefix of its ID (git-style).
    ///
    /// Returns `Some` when exactly one document matches the prefix.
    /// Returns `None` when zero or more than one document matches
    /// (ambiguous prefix).
    pub fn find_document_by_short_id(
        &self,
        short_id: &str,
    ) -> Result<Option<(u64, DocumentMetadata)>> {
        let entries = self.list_all_document_metadata_typed()?;
        let mut matches: Vec<(u64, DocumentMetadata)> = entries
            .into_iter()
            .filter(|(_doc_id, meta)| {
                let did = crate::doc_id::DocumentId::new(
                    &meta.collection,
                    &meta.relative_path,
                );
                did.full_hex().starts_with(short_id)
            })
            .collect();
        if matches.len() == 1 {
            Ok(matches.pop())
        } else {
            Ok(None)
        }
    }

    /// Look up a document by its relative path across all collections.
    ///
    /// Returns `Some` when exactly one document matches. Returns `None`
    /// when zero or more than one collection contains the path
    /// (ambiguous — the caller should require `collection:path`).
    pub fn find_document_by_path(
        &self,
        path: &str,
    ) -> Result<Option<(u64, DocumentMetadata)>> {
        let entries = self.list_all_document_metadata_typed()?;
        let mut matches: Vec<(u64, DocumentMetadata)> = entries
            .into_iter()
            .filter(|(_doc_id, meta)| meta.relative_path == path)
            .collect();
        if matches.len() == 1 {
            Ok(matches.pop())
        } else {
            Ok(None)
        }
    }

    // -- Collection Merkle Snapshots --

    /// Store a full Merkle snapshot blob for one collection.
    pub fn set_collection_merkle_snapshot(
        &self,
        collection: &str,
        snapshot: &Snapshot,
    ) -> Result<()> {
        let data = snapshot.serialize()?;
        let mut wtxn = self.env.write_txn()?;
        self.collection_merkle_snapshots.put(
            &mut wtxn,
            collection,
            data.as_slice(),
        )?;
        wtxn.commit()?;
        Ok(())
    }

    /// Load the persisted Merkle snapshot blob for one collection.
    pub fn get_collection_merkle_snapshot(
        &self,
        collection: &str,
    ) -> Result<Option<Snapshot>> {
        let rtxn = self.env.read_txn()?;
        self.collection_merkle_snapshots
            .get(&rtxn, collection)?
            .map(decode_aligned::<Snapshot>)
            .transpose()
    }

    /// Remove a collection Merkle snapshot. Returns `true` if it existed.
    pub fn remove_collection_merkle_snapshot(
        &self,
        collection: &str,
    ) -> Result<bool> {
        let mut wtxn = self.env.write_txn()?;
        let removed = self
            .collection_merkle_snapshots
            .delete(&mut wtxn, collection)?;
        wtxn.commit()?;
        Ok(removed)
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
        let mut wtxn = self.env.write_txn()?;
        self.settings.put(&mut wtxn, key, encoded.as_slice())?;
        wtxn.commit()?;
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
        let rtxn = self.env.read_txn()?;
        self.settings
            .get(&rtxn, key)?
            .map(decode_string)
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
        let mut wtxn = self.env.write_txn()?;
        let removed = self.settings.delete(&mut wtxn, key)?;
        wtxn.commit()?;
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

        let mut wtxn = self.env.write_txn()?;
        match provider.as_deref() {
            Some(bytes) => {
                self.settings.put(&mut wtxn, KEY_LLM_PROVIDER, bytes)?;
            }
            None => {
                self.settings.delete(&mut wtxn, KEY_LLM_PROVIDER)?;
            }
        }
        match model.as_deref() {
            Some(bytes) => {
                self.settings.put(&mut wtxn, KEY_LLM_MODEL, bytes)?;
            }
            None => {
                self.settings.delete(&mut wtxn, KEY_LLM_MODEL)?;
            }
        }
        match api_key.as_deref() {
            Some(bytes) => {
                self.settings.put(&mut wtxn, KEY_LLM_API_KEY, bytes)?;
            }
            None => {
                self.settings.delete(&mut wtxn, KEY_LLM_API_KEY)?;
            }
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Store a structured JSON value under a settings key.
    pub fn set_json_setting(
        &self,
        key: &str,
        value: &serde_json::Value,
    ) -> Result<()> {
        let encoded = encode_bytes(&StoredJsonValue::from(value.clone()))?;
        let mut wtxn = self.env.write_txn()?;
        self.settings.put(&mut wtxn, key, encoded.as_slice())?;
        wtxn.commit()?;
        Ok(())
    }

    /// Load a structured JSON value from a settings key.
    pub fn get_json_setting(
        &self,
        key: &str,
    ) -> Result<Option<serde_json::Value>> {
        let rtxn = self.env.read_txn()?;
        self.settings
            .get(&rtxn, key)?
            .map(|v| decode_aligned::<StoredJsonValue>(v).map(Into::into))
            .transpose()
    }

    /// Remove a structured JSON setting by key. Returns `true` if it existed.
    pub fn remove_json_setting(&self, key: &str) -> Result<bool> {
        self.remove_setting(key)
    }

    /// Remove multiple documents' metadata, stored content, and optional user
    /// metadata in a single write transaction.
    pub fn batch_remove_document_state(&self, doc_ids: &[u64]) -> Result<()> {
        if doc_ids.is_empty() {
            return Ok(());
        }

        let mut wtxn = self.env.write_txn()?;
        for &doc_id in doc_ids {
            self.document_metadata.delete(&mut wtxn, &doc_id)?;
        }
        for &doc_id in doc_ids {
            let content_key = document_content_key(doc_id);
            let user_metadata_key = document_user_metadata_key(doc_id);
            self.settings.delete(&mut wtxn, content_key.as_str())?;
            self.settings
                .delete(&mut wtxn, user_metadata_key.as_str())?;
        }
        wtxn.commit()?;
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

    // -- Per-document chunk manifest --

    /// Replace a document's chunk manifest, atomically updating the
    /// reverse `chunk_owners` index in the same transaction.
    ///
    /// `manifest` is the new ordered list of chunks for `doc_num_id`.
    /// Any chunk previously associated with the document but no longer
    /// referenced is removed from that chunk's owners list (and the
    /// owners entry is dropped entirely once empty). Newly-referenced
    /// chunks are added.
    ///
    /// Passing an empty manifest is equivalent to
    /// [`remove_doc_chunks`](Self::remove_doc_chunks).
    pub fn set_doc_chunks(
        &self,
        doc_num_id: u64,
        manifest: &[DocChunkEntry],
    ) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;

        let previous_unique = match self.doc_chunks.get(&wtxn, &doc_num_id)? {
            Some(bytes) => {
                let entries: Vec<DocChunkEntry> = decode_aligned(bytes)?;
                unique_chunk_ids(&entries)
            }
            None => Vec::new(),
        };
        let new_unique = unique_chunk_ids(manifest);

        let previous: std::collections::HashSet<u64> =
            previous_unique.iter().copied().collect();
        let current: std::collections::HashSet<u64> =
            new_unique.iter().copied().collect();

        for &chunk_doc_id in &previous {
            if !current.contains(&chunk_doc_id) {
                remove_owner_in_txn(
                    &self.chunk_owners,
                    &mut wtxn,
                    chunk_doc_id,
                    doc_num_id,
                )?;
            }
        }
        for &chunk_doc_id in &current {
            if !previous.contains(&chunk_doc_id) {
                add_owner_in_txn(
                    &self.chunk_owners,
                    &mut wtxn,
                    chunk_doc_id,
                    doc_num_id,
                )?;
            }
        }

        if manifest.is_empty() {
            self.doc_chunks.delete(&mut wtxn, &doc_num_id)?;
        } else {
            let manifest_vec: Vec<DocChunkEntry> = manifest.to_vec();
            let bytes = encode_bytes(&manifest_vec)?;
            self.doc_chunks
                .put(&mut wtxn, &doc_num_id, bytes.as_slice())?;
        }

        wtxn.commit()?;
        Ok(())
    }

    /// Load a document's chunk manifest. Returns `None` if the document
    /// has no recorded chunks.
    pub fn get_doc_chunks(
        &self,
        doc_num_id: u64,
    ) -> Result<Option<Vec<DocChunkEntry>>> {
        let rtxn = self.env.read_txn()?;
        let Some(bytes) = self.doc_chunks.get(&rtxn, &doc_num_id)? else {
            return Ok(None);
        };
        Ok(Some(decode_aligned(bytes)?))
    }

    /// Look up the byte range of `chunk_doc_id` within `doc_num_id`.
    ///
    /// Because chunk ids are content-derived, the same chunk text can
    /// land at different byte offsets in different documents — this is
    /// the per-document lookup. Returns the first occurrence when a
    /// chunk repeats inside one document.
    pub fn get_chunk_offset_for_doc(
        &self,
        doc_num_id: u64,
        chunk_doc_id: u64,
    ) -> Result<Option<ChunkByteOffset>> {
        let Some(manifest) = self.get_doc_chunks(doc_num_id)? else {
            return Ok(None);
        };
        Ok(manifest
            .into_iter()
            .find(|entry| entry.chunk_doc_id == chunk_doc_id)
            .map(|entry| ChunkByteOffset {
                start_byte: entry.start_byte,
                byte_len: entry.byte_len,
            }))
    }

    /// Remove a document's chunk manifest and the document's
    /// contribution to every referenced chunk's owners list.
    ///
    /// The embedding entries themselves are left in place so the cache
    /// stays warm for future indexes that re-derive the same chunk
    /// content. Returns `true` when a manifest was present.
    pub fn remove_doc_chunks(&self, doc_num_id: u64) -> Result<bool> {
        let mut wtxn = self.env.write_txn()?;
        let Some(bytes) = self.doc_chunks.get(&wtxn, &doc_num_id)? else {
            return Ok(false);
        };
        let entries: Vec<DocChunkEntry> = decode_aligned(bytes)?;
        for chunk_doc_id in unique_chunk_ids(&entries) {
            remove_owner_in_txn(
                &self.chunk_owners,
                &mut wtxn,
                chunk_doc_id,
                doc_num_id,
            )?;
        }
        self.doc_chunks.delete(&mut wtxn, &doc_num_id)?;
        wtxn.commit()?;
        Ok(true)
    }

    /// Remove many document manifests in one write transaction.
    pub fn batch_remove_doc_chunks(&self, doc_num_ids: &[u64]) -> Result<()> {
        if doc_num_ids.is_empty() {
            return Ok(());
        }
        let mut wtxn = self.env.write_txn()?;
        for &doc_num_id in doc_num_ids {
            let Some(bytes) = self.doc_chunks.get(&wtxn, &doc_num_id)? else {
                continue;
            };
            let entries: Vec<DocChunkEntry> = decode_aligned(bytes)?;
            for chunk_doc_id in unique_chunk_ids(&entries) {
                remove_owner_in_txn(
                    &self.chunk_owners,
                    &mut wtxn,
                    chunk_doc_id,
                    doc_num_id,
                )?;
            }
            self.doc_chunks.delete(&mut wtxn, &doc_num_id)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Sorted, deduplicated list of documents that contain
    /// `chunk_doc_id`. Empty when no document references it.
    pub fn get_chunk_owners(&self, chunk_doc_id: u64) -> Result<Vec<u64>> {
        let rtxn = self.env.read_txn()?;
        let Some(bytes) = self.chunk_owners.get(&rtxn, &chunk_doc_id)? else {
            return Ok(Vec::new());
        };
        decode_aligned(bytes)
    }
}

/// Distinct chunk ids referenced by a manifest, preserving first-seen
/// order so deterministic ordering survives any iteration.
fn unique_chunk_ids(manifest: &[DocChunkEntry]) -> Vec<u64> {
    let mut seen = std::collections::HashSet::with_capacity(manifest.len());
    let mut out = Vec::with_capacity(manifest.len());
    for entry in manifest {
        if seen.insert(entry.chunk_doc_id) {
            out.push(entry.chunk_doc_id);
        }
    }
    out
}

fn add_owner_in_txn(
    owners: &Database<U64<BigEndian>, Bytes>,
    wtxn: &mut heed::RwTxn<'_>,
    chunk_doc_id: u64,
    doc_num_id: u64,
) -> Result<()> {
    let mut current: Vec<u64> = match owners.get(wtxn, &chunk_doc_id)? {
        Some(bytes) => decode_aligned(bytes)?,
        None => Vec::new(),
    };
    if let Err(insert_at) = current.binary_search(&doc_num_id) {
        current.insert(insert_at, doc_num_id);
        let bytes = encode_bytes(&current)?;
        owners.put(wtxn, &chunk_doc_id, bytes.as_slice())?;
    }
    Ok(())
}

fn remove_owner_in_txn(
    owners: &Database<U64<BigEndian>, Bytes>,
    wtxn: &mut heed::RwTxn<'_>,
    chunk_doc_id: u64,
    doc_num_id: u64,
) -> Result<()> {
    let Some(bytes) = owners.get(wtxn, &chunk_doc_id)? else {
        return Ok(());
    };
    let mut current: Vec<u64> = decode_aligned(bytes)?;
    if let Ok(pos) = current.binary_search(&doc_num_id) {
        current.remove(pos);
        if current.is_empty() {
            owners.delete(wtxn, &chunk_doc_id)?;
        } else {
            let bytes = encode_bytes(&current)?;
            owners.put(wtxn, &chunk_doc_id, bytes.as_slice())?;
        }
    }
    Ok(())
}

/// Where a stored chunk lives in its source document.
///
/// Recorded once at index time and looked up at search time so a
/// matching chunk can be surfaced as an inclusive byte range — letting
/// the agent jump to the relevant slice of a large file instead of
/// re-reading the whole document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkByteOffset {
    /// Byte offset where the chunk begins in the original document.
    pub start_byte: u64,
    /// Byte length of the chunk in the original document.
    pub byte_len: u64,
}

impl ChunkByteOffset {
    /// Inclusive end byte (`start_byte + byte_len - 1`).
    ///
    /// Returns `None` for an empty chunk so callers don't accidentally
    /// hand `apply_byte_range` a backwards range.
    pub fn inclusive_end(self) -> Option<u64> {
        if self.byte_len == 0 {
            None
        } else {
            Some(self.start_byte + self.byte_len - 1)
        }
    }
}

/// One entry in a document's chunk manifest.
///
/// Pairs a content-derived `chunk_doc_id` with the byte range the chunk
/// occupies in the source document. Same chunk text appearing twice in
/// one document produces two entries with the same id but different
/// byte ranges.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct DocChunkEntry {
    /// Content-derived chunk identifier.
    pub chunk_doc_id: u64,
    /// Byte offset where the chunk begins in the source document.
    pub start_byte: u64,
    /// Byte length of the chunk in the source document.
    pub byte_len: u64,
}

impl DocChunkEntry {
    /// Convert a chunk entry into its byte range.
    pub fn byte_offset(&self) -> ChunkByteOffset {
        ChunkByteOffset {
            start_byte: self.start_byte,
            byte_len: self.byte_len,
        }
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

    fn test_snapshot(collection: &str, byte: u8) -> Snapshot {
        Snapshot::new(
            collection,
            [byte; crate::merkle::MERKLE_HASH_LEN],
            vec![crate::merkle::MerkleDirectoryNode::new(
                "",
                [byte.wrapping_add(1); crate::merkle::MERKLE_HASH_LEN],
                vec![crate::merkle::MerkleChildEntry::file(
                    "note.md",
                    [byte.wrapping_add(2); crate::merkle::MERKLE_HASH_LEN],
                )],
            )],
            vec![crate::merkle::MerkleFileLeaf::new(
                "note.md",
                [byte.wrapping_add(3); crate::merkle::MERKLE_HASH_LEN],
                [byte.wrapping_add(2); crate::merkle::MERKLE_HASH_LEN],
            )],
        )
    }

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
    fn collection_merkle_snapshot_roundtrips() {
        let (_tmp, db) = test_db();
        let snapshot = test_snapshot("notes", 7);

        db.set_collection_merkle_snapshot("notes", &snapshot)
            .unwrap();

        let loaded = db
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist");
        assert_eq!(loaded, snapshot);
    }

    #[test]
    fn collection_merkle_snapshot_replaces_existing_value() {
        let (_tmp, db) = test_db();
        let first = test_snapshot("notes", 1);
        let second = test_snapshot("notes", 9);

        db.set_collection_merkle_snapshot("notes", &first).unwrap();
        db.set_collection_merkle_snapshot("notes", &second).unwrap();

        assert_eq!(
            db.get_collection_merkle_snapshot("notes").unwrap(),
            Some(second)
        );
    }

    #[test]
    fn collection_merkle_snapshot_remove_reports_presence() {
        let (_tmp, db) = test_db();
        let snapshot = test_snapshot("notes", 3);
        db.set_collection_merkle_snapshot("notes", &snapshot)
            .unwrap();

        assert!(db.remove_collection_merkle_snapshot("notes").unwrap());
        assert!(!db.remove_collection_merkle_snapshot("notes").unwrap());
        assert!(
            db.get_collection_merkle_snapshot("notes")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn collection_merkle_snapshot_table_is_created_on_open() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.db");

        {
            let db = ConfigDb::open(&path).unwrap();
            let snapshot = test_snapshot("notes", 5);
            db.set_collection_merkle_snapshot("notes", &snapshot)
                .unwrap();
        }

        let reopened = ConfigDb::open(&path).unwrap();
        assert_eq!(
            reopened.get_collection_merkle_snapshot("notes").unwrap(),
            Some(test_snapshot("notes", 5))
        );
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
                    name: "docbert_search".to_string(),
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

    #[test]
    fn chunk_byte_offset_inclusive_end_handles_empty_chunks() {
        let offset = ChunkByteOffset {
            start_byte: 1024,
            byte_len: 256,
        };
        assert_eq!(offset.inclusive_end(), Some(1024 + 256 - 1));

        let empty = ChunkByteOffset {
            start_byte: 0,
            byte_len: 0,
        };
        assert_eq!(empty.inclusive_end(), None);
    }

    fn entry(chunk_doc_id: u64, start: u64, len: u64) -> DocChunkEntry {
        DocChunkEntry {
            chunk_doc_id,
            start_byte: start,
            byte_len: len,
        }
    }

    #[test]
    fn doc_chunks_roundtrip_preserves_order_and_repeats() {
        let (_tmp, db) = test_db();
        let manifest = vec![entry(1, 0, 10), entry(2, 10, 20), entry(1, 30, 5)];

        db.set_doc_chunks(42, &manifest).unwrap();

        let loaded = db.get_doc_chunks(42).unwrap().unwrap();
        assert_eq!(loaded, manifest);
    }

    #[test]
    fn doc_chunks_offset_lookup_returns_first_occurrence() {
        let (_tmp, db) = test_db();
        let manifest = vec![entry(7, 0, 10), entry(7, 100, 10)];

        db.set_doc_chunks(1, &manifest).unwrap();

        let offset = db.get_chunk_offset_for_doc(1, 7).unwrap().unwrap();
        assert_eq!(offset.start_byte, 0);
        assert_eq!(offset.byte_len, 10);

        assert!(db.get_chunk_offset_for_doc(1, 99).unwrap().is_none());
        assert!(db.get_chunk_offset_for_doc(2, 7).unwrap().is_none());
    }

    #[test]
    fn chunk_owners_track_documents_and_dedup() {
        let (_tmp, db) = test_db();

        // doc 1 owns chunk 7 once even when it appears twice in the
        // manifest.
        db.set_doc_chunks(1, &[entry(7, 0, 10), entry(7, 100, 10)])
            .unwrap();
        assert_eq!(db.get_chunk_owners(7).unwrap(), vec![1u64]);

        // Adding doc 2 grows the owners list.
        db.set_doc_chunks(2, &[entry(7, 50, 5)]).unwrap();
        assert_eq!(db.get_chunk_owners(7).unwrap(), vec![1u64, 2]);
    }

    #[test]
    fn doc_chunks_replace_drops_orphans_from_owners_index() {
        let (_tmp, db) = test_db();
        // Initial: doc 1 owns chunks {7, 8}.
        db.set_doc_chunks(1, &[entry(7, 0, 10), entry(8, 10, 5)])
            .unwrap();
        assert_eq!(db.get_chunk_owners(7).unwrap(), vec![1u64]);
        assert_eq!(db.get_chunk_owners(8).unwrap(), vec![1u64]);

        // Replacement: doc 1 only references chunk 7 now. Chunk 8's
        // owners list must drop doc 1, leaving the entry empty.
        db.set_doc_chunks(1, &[entry(7, 0, 10)]).unwrap();
        assert_eq!(db.get_chunk_owners(7).unwrap(), vec![1u64]);
        assert!(db.get_chunk_owners(8).unwrap().is_empty());
    }

    #[test]
    fn remove_doc_chunks_clears_manifest_and_owner_entries() {
        let (_tmp, db) = test_db();
        db.set_doc_chunks(1, &[entry(7, 0, 10), entry(8, 20, 5)])
            .unwrap();
        db.set_doc_chunks(2, &[entry(7, 50, 5)]).unwrap();

        assert!(db.remove_doc_chunks(1).unwrap());
        assert!(db.get_doc_chunks(1).unwrap().is_none());
        // Chunk 7 remains because doc 2 still owns it; chunk 8 is gone.
        assert_eq!(db.get_chunk_owners(7).unwrap(), vec![2u64]);
        assert!(db.get_chunk_owners(8).unwrap().is_empty());

        // Removing again is a no-op.
        assert!(!db.remove_doc_chunks(1).unwrap());
    }

    #[test]
    fn batch_remove_doc_chunks_clears_many() {
        let (_tmp, db) = test_db();
        db.set_doc_chunks(1, &[entry(7, 0, 10)]).unwrap();
        db.set_doc_chunks(2, &[entry(7, 5, 10)]).unwrap();
        db.set_doc_chunks(3, &[entry(8, 0, 10)]).unwrap();

        db.batch_remove_doc_chunks(&[1, 2]).unwrap();

        assert!(db.get_doc_chunks(1).unwrap().is_none());
        assert!(db.get_doc_chunks(2).unwrap().is_none());
        assert!(db.get_doc_chunks(3).unwrap().is_some());
        assert!(db.get_chunk_owners(7).unwrap().is_empty());
        assert_eq!(db.get_chunk_owners(8).unwrap(), vec![3u64]);
    }
}
