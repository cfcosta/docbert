#![allow(dead_code)]

use std::path::Path;

use docbert_core::{
    DocChunkEntry,
    DocumentId,
    error,
    incremental,
    preparation::{self, SearchDocument},
};

use crate::{snapshots, web::state::AppState};

pub(crate) type EmbeddingEntry = (u64, u32, u32, Vec<f32>);

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct IngestedDocument {
    pub(crate) doc_id: String,
    pub(crate) path: String,
    pub(crate) title: String,
    pub(crate) metadata: Option<serde_json::Value>,
}

pub(crate) fn load_document(
    collection: &str,
    relative_path: &str,
    full_path: &Path,
    metadata: Option<serde_json::Value>,
    mtime: u64,
) -> error::Result<SearchDocument> {
    let raw_content =
        preparation::load_preview_content(Path::new(relative_path), full_path)?;

    Ok(preparation::uploaded(
        collection,
        relative_path,
        &raw_content,
        metadata,
        mtime,
    ))
}

pub(crate) fn ingest_prepared_document(
    state: &AppState,
    collection: &str,
    document: &SearchDocument,
    embedding_entries: &[EmbeddingEntry],
    manifest: &[DocChunkEntry],
) -> error::Result<IngestedDocument> {
    let config_db = state.open_config_db_blocking()?;
    let embedding_db = state.open_embedding_db_blocking()?;
    let collection_root = collection_root(&config_db, collection)?;
    let previous_snapshot =
        snapshots::load_collection_snapshot(&config_db, collection)?;
    let previous_manifest = config_db
        .get_doc_chunks(document.did.numeric)?
        .unwrap_or_default();

    let previous_metadata =
        config_db.get_document_metadata_typed(document.did.numeric)?;
    let previous_user_metadata =
        config_db.get_document_user_metadata(document.did.numeric)?;

    let mut writer = state.open_index_writer_blocking(50_000_000)?;

    state.search_index.add_document(
        &writer,
        &document.did.full_hex(),
        document.did.numeric,
        collection,
        &document.relative_path,
        &document.title,
        &document.searchable_body,
        document.mtime,
    )?;

    // The embedding store is treated as a content-addressed cache: we
    // freely overwrite any chunk_doc_id with the freshly-computed
    // matrix (the value is content-derived so the new bytes match the
    // old ones bit-for-bit). Past embeddings are never deleted so the
    // cache stays warm for future documents that re-derive the same
    // chunk text.
    if let Err(err) = embedding_db.batch_store(embedding_entries) {
        let _ = writer.rollback();
        return Err(err);
    }

    if let Err(err) = config_db.set_doc_chunks(document.did.numeric, manifest) {
        let _ = writer.rollback();
        restore_previous_manifest(
            &config_db,
            document.did.numeric,
            &previous_manifest,
        )?;
        return Err(err);
    }

    if let Err(err) = persist_metadata(&config_db, collection, document) {
        let _ = writer.rollback();
        restore_previous_manifest(
            &config_db,
            document.did.numeric,
            &previous_manifest,
        )?;
        restore_previous_metadata(
            &config_db,
            document.did.numeric,
            previous_metadata.as_ref(),
            previous_user_metadata.as_ref(),
        )?;
        return Err(err);
    }

    if let Err(err) = writer.commit() {
        restore_previous_manifest(
            &config_db,
            document.did.numeric,
            &previous_manifest,
        )?;
        restore_previous_metadata(
            &config_db,
            document.did.numeric,
            previous_metadata.as_ref(),
            previous_user_metadata.as_ref(),
        )?;
        return Err(err.into());
    }

    // The collection snapshot must move in lockstep with successful ingest
    // side effects. If snapshot refresh fails, keep the previous snapshot.
    if let Err(err) = refresh_collection_snapshot(
        &config_db,
        collection,
        &collection_root,
        previous_snapshot.as_ref(),
    ) {
        restore_previous_manifest(
            &config_db,
            document.did.numeric,
            &previous_manifest,
        )?;
        restore_previous_metadata(
            &config_db,
            document.did.numeric,
            previous_metadata.as_ref(),
            previous_user_metadata.as_ref(),
        )?;
        return Err(err);
    }

    // Embedding entries themselves are never removed on overwrite
    // (the cache is immortal). Suppress the unused variable warning
    // without dropping the field, since callers still rely on the
    // signature.
    let _ = &embedding_db;

    Ok(IngestedDocument {
        doc_id: document.did.to_string(),
        path: document.relative_path.clone(),
        title: document.title.clone(),
        metadata: document.metadata.clone(),
    })
}

/// Saved state of a document before a batch ingest overwrote it.
///
/// Collected per-document so that a batch-level rollback can restore
/// the previous metadata, chunk manifest, and Tantivy entry. Note that
/// embeddings themselves are intentionally not captured: they live in
/// a content-addressed cache that is never destructively cleared on
/// overwrite, so there is nothing to "restore" beyond what is already
/// on disk.
#[derive(Debug)]
pub(crate) struct PreviousDocumentState {
    pub(crate) did: DocumentId,
    pub(crate) metadata: Option<incremental::DocumentMetadata>,
    pub(crate) user_metadata: Option<serde_json::Value>,
    pub(crate) manifest: Vec<DocChunkEntry>,
}

/// Capture the full pre-existing state of a document before overwriting it.
///
/// Returns `None` when the document is entirely new (no metadata stored).
pub(crate) fn capture_previous_state(
    state: &AppState,
    collection: &str,
    relative_path: &str,
) -> error::Result<Option<PreviousDocumentState>> {
    let config_db = state.open_config_db_blocking()?;
    let did = DocumentId::new(collection, relative_path);

    let metadata = config_db.get_document_metadata_typed(did.numeric)?;
    if metadata.is_none() {
        return Ok(None);
    }

    let user_metadata = config_db.get_document_user_metadata(did.numeric)?;
    let manifest = config_db.get_doc_chunks(did.numeric)?.unwrap_or_default();

    Ok(Some(PreviousDocumentState {
        did,
        metadata,
        user_metadata,
        manifest,
    }))
}

/// Roll back a single document to its previous state after a batch failure.
///
/// If `previous` is `Some`, the document's metadata, embeddings, and Tantivy
/// entry are restored to their pre-ingest values. If `None`, the document
/// is deleted outright (it was newly created by this batch).
pub(crate) fn rollback_document(
    state: &AppState,
    collection: &str,
    relative_path: &str,
    previous: Option<&PreviousDocumentState>,
) -> error::Result<()> {
    match previous {
        None => delete_document(state, collection, relative_path),
        Some(prev) => {
            let config_db = state.open_config_db_blocking()?;
            let collection_root = collection_root(&config_db, collection)?;
            let previous_snapshot =
                snapshots::load_collection_snapshot(&config_db, collection)?;

            // Embeddings are an immortal content-addressed cache. On
            // rollback we only need to restore the manifest — no
            // embedding side effects to undo.
            restore_previous_manifest(
                &config_db,
                prev.did.numeric,
                &prev.manifest,
            )?;

            // Restore previous metadata.
            restore_previous_metadata(
                &config_db,
                prev.did.numeric,
                prev.metadata.as_ref(),
                prev.user_metadata.as_ref(),
            )?;

            // Re-index in Tantivy: delete the current entry and, if
            // metadata existed, re-add the old version. If there was
            // previous metadata the document was already in Tantivy and
            // we need to restore it.
            let mut writer = state.open_index_writer_blocking(50_000_000)?;
            state
                .search_index
                .delete_document(&writer, &prev.did.full_hex())?;
            if let Some(meta) = &prev.metadata {
                let full_path = std::path::Path::new(
                    collection_root.to_str().unwrap_or(""),
                )
                .join(&meta.relative_path);
                let title = if full_path.exists() {
                    let content = preparation::load_preview_content(
                        std::path::Path::new(&meta.relative_path),
                        &full_path,
                    )
                    .unwrap_or_default();
                    docbert_core::ingestion::extract_title(
                        &content,
                        std::path::Path::new(&meta.relative_path),
                    )
                } else {
                    docbert_core::ingestion::extract_title(
                        "",
                        std::path::Path::new(&meta.relative_path),
                    )
                };
                let searchable_body = if full_path.exists() {
                    preparation::load_preview_content(
                        std::path::Path::new(&meta.relative_path),
                        &full_path,
                    )
                    .unwrap_or_default()
                } else {
                    String::new()
                };
                state.search_index.add_document(
                    &writer,
                    &prev.did.full_hex(),
                    prev.did.numeric,
                    &meta.collection,
                    &meta.relative_path,
                    &title,
                    &searchable_body,
                    meta.mtime,
                )?;
            }
            writer.commit()?;

            refresh_collection_snapshot(
                &config_db,
                collection,
                &collection_root,
                previous_snapshot.as_ref(),
            )?;

            Ok(())
        }
    }
}

pub(crate) fn delete_document(
    state: &AppState,
    collection: &str,
    relative_path: &str,
) -> error::Result<()> {
    let config_db = state.open_config_db_blocking()?;
    let collection_root = collection_root(&config_db, collection)?;
    let previous_snapshot =
        snapshots::load_collection_snapshot(&config_db, collection)?;
    let did = DocumentId::new(collection, relative_path);

    // Drop the manifest and metadata first (cheap, idempotent). The
    // manifest removal also takes the document out of every chunk's
    // owners list inside the same LMDB write transaction, leaving the
    // embedding cache untouched. Subsequent searches will simply skip
    // any chunk whose owners list goes empty — it's harmless to leave
    // the embedding bytes around for a future re-indexer to reuse.
    config_db.remove_doc_chunks(did.numeric)?;
    config_db.remove_document_metadata(did.numeric)?;
    config_db.remove_document_user_metadata(did.numeric)?;

    // Commit the Tantivy deletion last — it's the visible "point of no
    // return". All metadata is already gone, so no orphan state is
    // possible on Tantivy failure.
    let mut writer = state.open_index_writer_blocking(50_000_000)?;
    state
        .search_index
        .delete_document(&writer, &did.full_hex())?;
    writer.commit()?;

    refresh_collection_snapshot(
        &config_db,
        collection,
        &collection_root,
        previous_snapshot.as_ref(),
    )?;

    Ok(())
}

fn collection_root(
    config_db: &docbert_core::ConfigDb,
    collection: &str,
) -> error::Result<std::path::PathBuf> {
    let root = config_db.get_collection(collection)?.ok_or_else(|| {
        error::Error::NotFound {
            kind: "collection",
            name: collection.to_string(),
        }
    })?;
    Ok(std::path::PathBuf::from(root))
}

fn persist_metadata(
    config_db: &docbert_core::ConfigDb,
    collection: &str,
    document: &SearchDocument,
) -> error::Result<()> {
    let metadata = incremental::DocumentMetadata {
        collection: collection.to_string(),
        relative_path: document.relative_path.clone(),
        mtime: document.mtime,
    };
    config_db.set_document_metadata_typed(document.did.numeric, &metadata)?;

    match document.metadata.as_ref() {
        Some(value) => {
            config_db.set_document_user_metadata(document.did.numeric, value)?
        }
        None => {
            config_db.remove_document_user_metadata(document.did.numeric)?;
        }
    }

    Ok(())
}

fn refresh_collection_snapshot(
    config_db: &docbert_core::ConfigDb,
    collection: &str,
    collection_root: &Path,
    previous_snapshot: Option<&docbert_core::merkle::Snapshot>,
) -> error::Result<()> {
    let current_snapshot =
        snapshots::compute_collection_snapshot(collection, collection_root)?;
    if let Err(err) =
        snapshots::replace_collection_snapshot(config_db, &current_snapshot)
    {
        restore_previous_collection_snapshot(
            config_db,
            collection,
            previous_snapshot,
        )?;
        return Err(err);
    }
    Ok(())
}

fn restore_previous_collection_snapshot(
    config_db: &docbert_core::ConfigDb,
    collection: &str,
    previous_snapshot: Option<&docbert_core::merkle::Snapshot>,
) -> error::Result<()> {
    match previous_snapshot {
        Some(snapshot) => {
            snapshots::replace_collection_snapshot(config_db, snapshot)
        }
        None => {
            config_db.remove_collection_merkle_snapshot(collection)?;
            Ok(())
        }
    }
}

fn restore_previous_manifest(
    config_db: &docbert_core::ConfigDb,
    doc_num_id: u64,
    previous_manifest: &[DocChunkEntry],
) -> error::Result<()> {
    config_db.set_doc_chunks(doc_num_id, previous_manifest)?;
    Ok(())
}

fn restore_previous_metadata(
    config_db: &docbert_core::ConfigDb,
    doc_id: u64,
    previous_metadata: Option<&incremental::DocumentMetadata>,
    previous_user_metadata: Option<&serde_json::Value>,
) -> error::Result<()> {
    match previous_metadata {
        Some(metadata) => {
            config_db.set_document_metadata_typed(doc_id, metadata)?
        }
        None => {
            config_db.remove_document_metadata(doc_id)?;
        }
    }

    match previous_user_metadata {
        Some(value) => config_db.set_document_user_metadata(doc_id, value)?,
        None => {
            config_db.remove_document_user_metadata(doc_id)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        sync::{Arc, Mutex},
    };

    use docbert_core::{ConfigDb, ModelManager, SearchIndex};

    use super::*;
    use crate::web::state::Inner;

    fn test_state() -> (tempfile::TempDir, AppState) {
        let tmp = tempfile::tempdir().unwrap();
        let state = Arc::new(Inner {
            data_dir: docbert_core::DataDir::new(tmp.path()),
            search_index: SearchIndex::open_in_ram().unwrap(),
            model: Mutex::new(ModelManager::new()),
            model_id: "test-model".to_string(),
        });
        (tmp, state)
    }

    fn test_config_db(state: &AppState) -> ConfigDb {
        ConfigDb::open(&state.data_dir.config_db()).unwrap()
    }

    fn test_embedding_db(state: &AppState) -> docbert_core::EmbeddingDb {
        docbert_core::EmbeddingDb::open(&state.data_dir.embeddings_db())
            .unwrap()
    }

    fn seed_collection_root(
        tmp: &tempfile::TempDir,
        state: &AppState,
        collection: &str,
    ) -> PathBuf {
        let root = tmp.path().join(collection);
        std::fs::create_dir_all(&root).unwrap();
        test_config_db(state)
            .set_collection(collection, root.to_str().unwrap())
            .unwrap();
        root
    }

    fn write_markdown(
        root: &Path,
        relative_path: &str,
        content: &str,
    ) -> PathBuf {
        let full_path = root.join(relative_path);
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&full_path, content).unwrap();
        full_path
    }

    /// Synthesize a stable per-document chunk id. Tests don't go through
    /// the real chunker, so we just shift the document id by the chunk
    /// index — the actual scheme is irrelevant for these tests as long
    /// as ids stay distinct within and across documents.
    fn synthetic_chunk_id(doc_id: u64, chunk_index: usize) -> u64 {
        doc_id.wrapping_add(chunk_index as u64).wrapping_add(0xC0DE)
    }

    fn fake_embedding_entries(
        doc_id: u64,
        chunk_count: usize,
    ) -> Vec<EmbeddingEntry> {
        (0..chunk_count)
            .map(|chunk_index| {
                (
                    synthetic_chunk_id(doc_id, chunk_index),
                    1,
                    2,
                    vec![chunk_index as f32 + 1.0, 9.0],
                )
            })
            .collect()
    }

    /// Build a synthetic chunk manifest that mirrors the embedding ids
    /// `fake_embedding_entries` produces. Each chunk gets a distinct,
    /// monotonically increasing byte range so tests can assert which
    /// entry survives a round-trip without ambiguity.
    fn fake_manifest(doc_id: u64, chunk_count: usize) -> Vec<DocChunkEntry> {
        (0..chunk_count)
            .map(|chunk_index| DocChunkEntry {
                chunk_doc_id: synthetic_chunk_id(doc_id, chunk_index),
                start_byte: (chunk_index as u64) * 100,
                byte_len: 50,
            })
            .collect()
    }

    #[test]
    fn web_ingest_initial_ingest_stores_index_embeddings_and_metadata() {
        let (tmp, state) = test_state();
        let root = seed_collection_root(&tmp, &state, "notes");
        let full_path = write_markdown(&root, "hello.md", "# Hello\n\nBody");
        let document = load_document(
            "notes",
            "hello.md",
            &full_path,
            Some(serde_json::json!({"topic": "rust"})),
            7,
        )
        .unwrap();

        let ingested = ingest_prepared_document(
            &state,
            "notes",
            &document,
            &fake_embedding_entries(document.did.numeric, 2),
            &fake_manifest(document.did.numeric, 2),
        )
        .unwrap();

        let snapshot = test_config_db(&state)
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after ingest");

        assert_eq!(ingested.doc_id, document.did.to_string());
        assert_eq!(ingested.title, "Hello");
        assert_eq!(
            test_config_db(&state)
                .get_document_metadata_typed(document.did.numeric)
                .unwrap()
                .unwrap()
                .mtime,
            7
        );
        assert_eq!(
            test_config_db(&state)
                .get_document_user_metadata(document.did.numeric)
                .unwrap(),
            Some(serde_json::json!({"topic": "rust"}))
        );
        assert!(
            test_embedding_db(&state)
                .load(synthetic_chunk_id(document.did.numeric, 0))
                .unwrap()
                .is_some()
        );
        assert!(
            test_embedding_db(&state)
                .load(synthetic_chunk_id(document.did.numeric, 1))
                .unwrap()
                .is_some()
        );
        let indexed = state
            .search_index
            .find_by_collection_path("notes", "hello.md")
            .unwrap()
            .unwrap();
        assert_eq!(indexed.title, "Hello");
        assert_eq!(snapshot.files.len(), 1);
        assert_eq!(snapshot.files[0].relative_path, "hello.md");
    }

    #[test]
    fn web_ingest_overwrite_replacement_updates_metadata_and_index() {
        let (tmp, state) = test_state();
        let root = seed_collection_root(&tmp, &state, "notes");
        let full_path = write_markdown(&root, "hello.md", "# First\n\nBody");
        let first =
            load_document("notes", "hello.md", &full_path, None, 1).unwrap();
        ingest_prepared_document(
            &state,
            "notes",
            &first,
            &fake_embedding_entries(first.did.numeric, 2),
            &fake_manifest(first.did.numeric, 2),
        )
        .unwrap();
        let first_snapshot = test_config_db(&state)
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after first ingest");

        write_markdown(&root, "hello.md", "# Updated\n\nBody v2");
        let second = load_document(
            "notes",
            "hello.md",
            &full_path,
            Some(serde_json::json!({"version": 2})),
            9,
        )
        .unwrap();
        ingest_prepared_document(
            &state,
            "notes",
            &second,
            &fake_embedding_entries(second.did.numeric, 2),
            &fake_manifest(second.did.numeric, 2),
        )
        .unwrap();

        let second_snapshot = test_config_db(&state)
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after replacement ingest");

        let metadata = test_config_db(&state)
            .get_document_metadata_typed(second.did.numeric)
            .unwrap()
            .unwrap();
        assert_eq!(metadata.mtime, 9);
        assert_eq!(
            test_config_db(&state)
                .get_document_user_metadata(second.did.numeric)
                .unwrap(),
            Some(serde_json::json!({"version": 2}))
        );
        let indexed = state
            .search_index
            .find_by_collection_path("notes", "hello.md")
            .unwrap()
            .unwrap();
        assert_eq!(indexed.title, "Updated");
        assert_ne!(first_snapshot.root_hash, second_snapshot.root_hash);
    }

    #[test]
    fn web_ingest_replacement_drops_stale_chunks_from_manifest_but_keeps_cache()
    {
        let (tmp, state) = test_state();
        let root = seed_collection_root(&tmp, &state, "notes");
        let full_path = write_markdown(&root, "hello.md", "# Hello\n\nBody");
        let document =
            load_document("notes", "hello.md", &full_path, None, 1).unwrap();
        ingest_prepared_document(
            &state,
            "notes",
            &document,
            &fake_embedding_entries(document.did.numeric, 3),
            &fake_manifest(document.did.numeric, 3),
        )
        .unwrap();

        ingest_prepared_document(
            &state,
            "notes",
            &document,
            &fake_embedding_entries(document.did.numeric, 2),
            &fake_manifest(document.did.numeric, 2),
        )
        .unwrap();

        // The manifest no longer references the removed chunk, and the
        // chunk_owners reverse index drops the document from that
        // chunk's owner list.
        let manifest = test_config_db(&state)
            .get_doc_chunks(document.did.numeric)
            .unwrap()
            .unwrap();
        assert_eq!(manifest.len(), 2);
        let stale_chunk = synthetic_chunk_id(document.did.numeric, 2);
        assert!(
            test_config_db(&state)
                .get_chunk_owners(stale_chunk)
                .unwrap()
                .is_empty()
        );

        // Embedding entries themselves stick around — the cache is
        // intentionally immortal so a future re-indexer can reuse the
        // same content without recomputing.
        assert!(
            test_embedding_db(&state)
                .load(stale_chunk)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn web_ingest_snapshot_failure_preserves_previous_snapshot() {
        let (tmp, state) = test_state();
        let root = seed_collection_root(&tmp, &state, "notes");
        let full_path = write_markdown(&root, "hello.md", "# Hello\n\nBody");
        let document =
            load_document("notes", "hello.md", &full_path, None, 1).unwrap();
        ingest_prepared_document(
            &state,
            "notes",
            &document,
            &fake_embedding_entries(document.did.numeric, 1),
            &fake_manifest(document.did.numeric, 1),
        )
        .unwrap();
        let original_snapshot = test_config_db(&state)
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after first ingest");

        write_markdown(&root, "hello.md", "# Hello\n\nUpdated");
        let updated =
            load_document("notes", "hello.md", &full_path, None, 2).unwrap();
        std::fs::remove_dir_all(&root).unwrap();

        assert!(
            ingest_prepared_document(
                &state,
                "notes",
                &updated,
                &fake_embedding_entries(updated.did.numeric, 1),
                &fake_manifest(updated.did.numeric, 1),
            )
            .is_err()
        );
        assert_eq!(
            test_config_db(&state)
                .get_collection_merkle_snapshot("notes")
                .unwrap(),
            Some(original_snapshot)
        );
    }

    #[test]
    fn web_delete_snapshot_failure_preserves_previous_snapshot() {
        let (tmp, state) = test_state();
        let root = seed_collection_root(&tmp, &state, "notes");
        let full_path = write_markdown(&root, "hello.md", "# Hello\n\nBody");
        let document =
            load_document("notes", "hello.md", &full_path, None, 1).unwrap();
        ingest_prepared_document(
            &state,
            "notes",
            &document,
            &fake_embedding_entries(document.did.numeric, 1),
            &fake_manifest(document.did.numeric, 1),
        )
        .unwrap();
        let original_snapshot = test_config_db(&state)
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after ingest");

        std::fs::remove_dir_all(&root).unwrap();

        assert!(delete_document(&state, "notes", "hello.md").is_err());
        assert_eq!(
            test_config_db(&state)
                .get_collection_merkle_snapshot("notes")
                .unwrap(),
            Some(original_snapshot)
        );
    }

    #[test]
    fn capture_previous_state_returns_none_for_new_document() {
        let (_tmp, state) = test_state();
        let result =
            capture_previous_state(&state, "notes", "nonexistent.md").unwrap();
        assert!(
            result.is_none(),
            "new document should have no previous state"
        );
    }

    #[test]
    fn capture_previous_state_returns_full_state_for_existing_document() {
        let (tmp, state) = test_state();
        let root = seed_collection_root(&tmp, &state, "notes");
        let full_path = write_markdown(&root, "hello.md", "# Hello\n\nBody");

        let document = load_document(
            "notes",
            "hello.md",
            &full_path,
            Some(serde_json::json!({"tag": "important"})),
            7,
        )
        .unwrap();
        ingest_prepared_document(
            &state,
            "notes",
            &document,
            &fake_embedding_entries(document.did.numeric, 2),
            &fake_manifest(document.did.numeric, 2),
        )
        .unwrap();

        let prev = capture_previous_state(&state, "notes", "hello.md").unwrap();
        let prev = prev.expect("existing document should have previous state");
        assert!(prev.metadata.is_some());
        assert_eq!(
            prev.user_metadata,
            Some(serde_json::json!({"tag": "important"}))
        );
        assert_eq!(
            prev.manifest.len(),
            2,
            "should capture the existing chunk manifest"
        );
    }

    #[test]
    fn rollback_document_restores_metadata_and_manifest_for_overwrite() {
        let (tmp, state) = test_state();
        let root = seed_collection_root(&tmp, &state, "notes");
        let full_path = write_markdown(&root, "hello.md", "# Original\n\nBody");

        let original = load_document(
            "notes",
            "hello.md",
            &full_path,
            Some(serde_json::json!({"version": 1})),
            5,
        )
        .unwrap();
        let original_embeddings =
            fake_embedding_entries(original.did.numeric, 2);
        let original_manifest = fake_manifest(original.did.numeric, 2);
        ingest_prepared_document(
            &state,
            "notes",
            &original,
            &original_embeddings,
            &original_manifest,
        )
        .unwrap();

        let prev = capture_previous_state(&state, "notes", "hello.md").unwrap();

        write_markdown(&root, "hello.md", "# Updated\n\nNew body");
        let updated = load_document(
            "notes",
            "hello.md",
            &full_path,
            Some(serde_json::json!({"version": 2})),
            10,
        )
        .unwrap();
        ingest_prepared_document(
            &state,
            "notes",
            &updated,
            &fake_embedding_entries(updated.did.numeric, 3),
            &fake_manifest(updated.did.numeric, 3),
        )
        .unwrap();

        assert_eq!(
            test_config_db(&state)
                .get_document_user_metadata(original.did.numeric)
                .unwrap(),
            Some(serde_json::json!({"version": 2}))
        );

        std::fs::write(&full_path, "# Original\n\nBody").unwrap();
        rollback_document(&state, "notes", "hello.md", prev.as_ref()).unwrap();

        let restored_meta = test_config_db(&state)
            .get_document_metadata_typed(original.did.numeric)
            .unwrap()
            .expect("metadata should be restored");
        assert_eq!(restored_meta.mtime, 5);
        assert_eq!(
            test_config_db(&state)
                .get_document_user_metadata(original.did.numeric)
                .unwrap(),
            Some(serde_json::json!({"version": 1}))
        );

        let restored_manifest = test_config_db(&state)
            .get_doc_chunks(original.did.numeric)
            .unwrap()
            .expect("manifest should be restored");
        assert_eq!(restored_manifest, original_manifest);

        // Embedding cache is immortal — both the original and the
        // overwrite chunks remain on disk, ready to be reused.
        assert!(
            test_embedding_db(&state)
                .load(synthetic_chunk_id(original.did.numeric, 0))
                .unwrap()
                .is_some(),
        );
        assert!(
            test_embedding_db(&state)
                .load(synthetic_chunk_id(original.did.numeric, 2))
                .unwrap()
                .is_some(),
        );
    }

    #[test]
    fn rollback_document_deletes_new_document() {
        let (tmp, state) = test_state();
        let root = seed_collection_root(&tmp, &state, "notes");
        let full_path = write_markdown(&root, "new.md", "# New\n\nBody");

        let document =
            load_document("notes", "new.md", &full_path, None, 1).unwrap();
        ingest_prepared_document(
            &state,
            "notes",
            &document,
            &fake_embedding_entries(document.did.numeric, 1),
            &fake_manifest(document.did.numeric, 1),
        )
        .unwrap();

        // Rollback with no previous state (new document)
        rollback_document(&state, "notes", "new.md", None).unwrap();

        // Document metadata + manifest are gone, but the embedding
        // cache stays warm. The chunk's owners list collapses to empty.
        assert!(
            test_config_db(&state)
                .get_document_metadata_typed(document.did.numeric)
                .unwrap()
                .is_none(),
            "metadata should be deleted"
        );
        assert!(
            test_config_db(&state)
                .get_doc_chunks(document.did.numeric)
                .unwrap()
                .is_none(),
            "manifest should be gone"
        );
        let only_chunk = synthetic_chunk_id(document.did.numeric, 0);
        assert!(
            test_config_db(&state)
                .get_chunk_owners(only_chunk)
                .unwrap()
                .is_empty(),
            "chunk owners list should be empty after deletion"
        );
        assert!(
            test_embedding_db(&state)
                .load(only_chunk)
                .unwrap()
                .is_some(),
            "embedding cache stays warm — no destructive cleanup",
        );
    }

    #[test]
    fn web_ingest_load_document_reads_from_disk() {
        let (tmp, state) = test_state();
        let root = seed_collection_root(&tmp, &state, "notes");
        let full_path = write_markdown(
            &root,
            "nested/hello.md",
            "---\ntitle: ignored\n---\n# From Disk\n\nBody",
        );

        let document =
            load_document("notes", "nested/hello.md", &full_path, None, 5)
                .unwrap();

        assert_eq!(document.relative_path, "nested/hello.md");
        assert_eq!(document.title, "From Disk");
        assert_eq!(document.searchable_body, "# From Disk\n\nBody");
        assert_eq!(document.mtime, 5);
    }
}
