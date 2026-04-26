#![allow(dead_code)]

use std::path::Path;

use docbert_core::{
    ChunkByteOffset,
    DocumentId,
    chunking::document_family_key,
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
    chunk_offsets: &[(u64, ChunkByteOffset)],
) -> error::Result<IngestedDocument> {
    let config_db = state.open_config_db_blocking()?;
    let embedding_db = state.open_embedding_db_blocking()?;
    let collection_root = collection_root(&config_db, collection)?;
    let previous_snapshot =
        snapshots::load_collection_snapshot(&config_db, collection)?;
    let existing_embeddings =
        load_document_family_embeddings(&embedding_db, document.did.numeric)?;
    let existing_embedding_ids: Vec<u64> = existing_embeddings
        .iter()
        .map(|(doc_id, _, _, _)| *doc_id)
        .collect();
    let current_embedding_ids: Vec<u64> = embedding_entries
        .iter()
        .map(|(doc_id, _, _, _)| *doc_id)
        .collect();
    let existing_chunk_offsets = load_document_family_chunk_offsets(
        &config_db,
        &existing_embedding_ids,
    )?;

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

    if let Err(err) = embedding_db.batch_store(embedding_entries) {
        let _ = writer.rollback();
        return Err(err);
    }

    // Persist chunk byte offsets in lockstep with the embedding store. We
    // wipe any prior family entries first so a replacement that produces
    // fewer chunks doesn't leak offsets for chunk indexes that no longer
    // exist; then write the new offsets.
    if let Err(err) =
        persist_chunk_offsets(&config_db, document.did.numeric, chunk_offsets)
    {
        let _ = writer.rollback();
        let _ = embedding_db.batch_remove(&current_embedding_ids);
        restore_previous_embeddings(
            &embedding_db,
            &existing_embeddings,
            &current_embedding_ids,
        )?;
        restore_previous_chunk_offsets(
            &config_db,
            document.did.numeric,
            &existing_chunk_offsets,
        )?;
        return Err(err);
    }

    if let Err(err) = persist_metadata(&config_db, collection, document) {
        let _ = writer.rollback();
        restore_previous_embeddings(
            &embedding_db,
            &existing_embeddings,
            &current_embedding_ids,
        )?;
        restore_previous_chunk_offsets(
            &config_db,
            document.did.numeric,
            &existing_chunk_offsets,
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
        restore_previous_embeddings(
            &embedding_db,
            &existing_embeddings,
            &current_embedding_ids,
        )?;
        restore_previous_chunk_offsets(
            &config_db,
            document.did.numeric,
            &existing_chunk_offsets,
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
        restore_previous_embeddings(
            &embedding_db,
            &existing_embeddings,
            &current_embedding_ids,
        )?;
        restore_previous_chunk_offsets(
            &config_db,
            document.did.numeric,
            &existing_chunk_offsets,
        )?;
        restore_previous_metadata(
            &config_db,
            document.did.numeric,
            previous_metadata.as_ref(),
            previous_user_metadata.as_ref(),
        )?;
        return Err(err);
    }

    remove_stale_previous_embeddings(
        &embedding_db,
        &existing_embedding_ids,
        &current_embedding_ids,
    )?;

    Ok(IngestedDocument {
        doc_id: document.did.to_string(),
        path: document.relative_path.clone(),
        title: document.title.clone(),
        metadata: document.metadata.clone(),
    })
}

/// Saved state of a document before a batch ingest overwrote it.
///
/// Collected per-document so that a batch-level rollback can restore the
/// previous metadata, embeddings, and Tantivy entry instead of
/// destructively deleting everything.
#[derive(Debug)]
pub(crate) struct PreviousDocumentState {
    pub(crate) did: DocumentId,
    pub(crate) metadata: Option<incremental::DocumentMetadata>,
    pub(crate) user_metadata: Option<serde_json::Value>,
    pub(crate) embeddings: Vec<EmbeddingEntry>,
    pub(crate) chunk_offsets: Vec<(u64, ChunkByteOffset)>,
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
    let embedding_db = state.open_embedding_db_blocking()?;
    let embeddings =
        load_document_family_embeddings(&embedding_db, did.numeric)?;
    let embedding_ids: Vec<u64> =
        embeddings.iter().map(|(doc_id, _, _, _)| *doc_id).collect();
    let chunk_offsets =
        load_document_family_chunk_offsets(&config_db, &embedding_ids)?;

    Ok(Some(PreviousDocumentState {
        did,
        metadata,
        user_metadata,
        embeddings,
        chunk_offsets,
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
            let embedding_db = state.open_embedding_db_blocking()?;
            let collection_root = collection_root(&config_db, collection)?;
            let previous_snapshot =
                snapshots::load_collection_snapshot(&config_db, collection)?;

            // Remove the current (bad) embeddings for this document family.
            embedding_db.remove_document_family(prev.did.numeric)?;

            // Restore previous embeddings.
            if !prev.embeddings.is_empty() {
                embedding_db.batch_store(&prev.embeddings)?;
            }

            // Wipe and restore the chunk byte offsets the same way: the
            // overwrite that just failed already wrote new offsets, so we
            // clear the family and replay the captured pre-overwrite set.
            config_db.batch_remove_chunk_offsets_for_document_families(&[
                prev.did.numeric,
            ])?;
            if !prev.chunk_offsets.is_empty() {
                config_db.batch_set_chunk_offsets(&prev.chunk_offsets)?;
            }

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

    // Remove embeddings and metadata first (cheap, idempotent).
    // If this fails the document is still intact and can be retried.
    let embedding_db = state.open_embedding_db_blocking()?;
    embedding_db.remove_document_family(did.numeric)?;
    config_db.remove_document_metadata(did.numeric)?;
    config_db.remove_document_user_metadata(did.numeric)?;
    config_db
        .batch_remove_chunk_offsets_for_document_families(&[did.numeric])?;

    // Commit the Tantivy deletion last — it's the visible "point of no
    // return".  All metadata/embeddings are already gone, so no orphan
    // state is possible on Tantivy failure.
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

fn load_document_family_embeddings(
    embedding_db: &docbert_core::EmbeddingDb,
    base_doc_id: u64,
) -> error::Result<Vec<EmbeddingEntry>> {
    let family_key = document_family_key(base_doc_id);
    let doc_ids: Vec<u64> = embedding_db
        .list_ids()?
        .into_iter()
        .filter(|doc_id| document_family_key(*doc_id) == family_key)
        .collect();

    let loaded = embedding_db.batch_load(&doc_ids)?;
    let mut entries = Vec::with_capacity(loaded.len());
    for (doc_id, matrix) in loaded {
        let Some(matrix) = matrix else {
            continue;
        };
        entries.push((
            doc_id,
            matrix.num_tokens,
            matrix.dimension,
            matrix.data,
        ));
    }
    Ok(entries)
}

fn restore_previous_embeddings(
    embedding_db: &docbert_core::EmbeddingDb,
    previous_embeddings: &[EmbeddingEntry],
    current_embedding_ids: &[u64],
) -> error::Result<()> {
    let previous_ids: std::collections::HashSet<u64> = previous_embeddings
        .iter()
        .map(|(doc_id, _, _, _)| *doc_id)
        .collect();
    let stale_new_ids: Vec<u64> = current_embedding_ids
        .iter()
        .copied()
        .filter(|doc_id| !previous_ids.contains(doc_id))
        .collect();
    if !stale_new_ids.is_empty() {
        embedding_db.batch_remove(&stale_new_ids)?;
    }
    if !previous_embeddings.is_empty() {
        embedding_db.batch_store(previous_embeddings)?;
    }
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

fn persist_chunk_offsets(
    config_db: &docbert_core::ConfigDb,
    base_doc_id: u64,
    chunk_offsets: &[(u64, ChunkByteOffset)],
) -> error::Result<()> {
    // Wipe the family before writing. A re-ingest that produces fewer
    // chunks (e.g. the document shrank) would otherwise leave offsets
    // behind for chunk indexes that no longer exist, and search would
    // happily surface those stale ranges.
    config_db
        .batch_remove_chunk_offsets_for_document_families(&[base_doc_id])?;
    config_db.batch_set_chunk_offsets(chunk_offsets)?;
    Ok(())
}

fn restore_previous_chunk_offsets(
    config_db: &docbert_core::ConfigDb,
    base_doc_id: u64,
    previous_chunk_offsets: &[(u64, ChunkByteOffset)],
) -> error::Result<()> {
    // The just-failed ingest already wrote (and may have wiped) the
    // family's offsets — clear what is there now and replay the pre-ingest
    // set so the table matches the embeddings + metadata we are
    // restoring.
    config_db
        .batch_remove_chunk_offsets_for_document_families(&[base_doc_id])?;
    config_db.batch_set_chunk_offsets(previous_chunk_offsets)?;
    Ok(())
}

fn load_document_family_chunk_offsets(
    config_db: &docbert_core::ConfigDb,
    chunk_doc_ids: &[u64],
) -> error::Result<Vec<(u64, ChunkByteOffset)>> {
    let mut entries = Vec::with_capacity(chunk_doc_ids.len());
    for &chunk_doc_id in chunk_doc_ids {
        if let Some(offset) = config_db.get_chunk_offset(chunk_doc_id)? {
            entries.push((chunk_doc_id, offset));
        }
    }
    Ok(entries)
}

fn remove_stale_previous_embeddings(
    embedding_db: &docbert_core::EmbeddingDb,
    previous_embedding_ids: &[u64],
    current_embedding_ids: &[u64],
) -> error::Result<()> {
    let current_ids: std::collections::HashSet<u64> =
        current_embedding_ids.iter().copied().collect();
    let stale_previous_ids: Vec<u64> = previous_embedding_ids
        .iter()
        .copied()
        .filter(|doc_id| !current_ids.contains(doc_id))
        .collect();
    if !stale_previous_ids.is_empty() {
        embedding_db.batch_remove(&stale_previous_ids)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        sync::{Arc, Mutex},
    };

    use docbert_core::{
        ConfigDb,
        ModelManager,
        SearchIndex,
        chunking::chunk_doc_id,
    };

    use super::*;
    use crate::web::state::Inner;

    fn test_state() -> (tempfile::TempDir, AppState) {
        let tmp = tempfile::tempdir().unwrap();
        let state = Arc::new(Inner {
            data_dir: docbert_core::DataDir::new(tmp.path()),
            search_index: SearchIndex::open_in_ram().unwrap(),
            model: Mutex::new(ModelManager::new()),
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

    fn fake_embedding_entries(
        doc_id: u64,
        chunk_count: usize,
    ) -> Vec<EmbeddingEntry> {
        let mut entries = Vec::new();
        for chunk_index in 0..chunk_count {
            let embedding_id = if chunk_index == 0 {
                doc_id
            } else {
                chunk_doc_id(doc_id, chunk_index)
            };
            entries.push((
                embedding_id,
                1,
                2,
                vec![chunk_index as f32 + 1.0, 9.0],
            ));
        }
        entries
    }

    /// Build a synthetic chunk-offset table that mirrors the embedding
    /// IDs `fake_embedding_entries` would produce. Each chunk gets a
    /// distinct, monotonically increasing byte range so tests can assert
    /// that the right entry survives roundtripping without colliding.
    fn fake_chunk_offsets(
        doc_id: u64,
        chunk_count: usize,
    ) -> Vec<(u64, ChunkByteOffset)> {
        (0..chunk_count)
            .map(|chunk_index| {
                let id = if chunk_index == 0 {
                    doc_id
                } else {
                    chunk_doc_id(doc_id, chunk_index)
                };
                (
                    id,
                    ChunkByteOffset {
                        start_byte: (chunk_index as u64) * 100,
                        byte_len: 50,
                    },
                )
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
            &fake_chunk_offsets(document.did.numeric, 2),
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
                .load(document.did.numeric)
                .unwrap()
                .is_some()
        );
        assert!(
            test_embedding_db(&state)
                .load(chunk_doc_id(document.did.numeric, 1))
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
            &fake_chunk_offsets(first.did.numeric, 2),
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
            &fake_chunk_offsets(second.did.numeric, 2),
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
    fn web_ingest_replacement_removes_stale_chunk_embeddings() {
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
            &fake_chunk_offsets(document.did.numeric, 3),
        )
        .unwrap();

        ingest_prepared_document(
            &state,
            "notes",
            &document,
            &fake_embedding_entries(document.did.numeric, 2),
            &fake_chunk_offsets(document.did.numeric, 2),
        )
        .unwrap();

        assert!(
            test_embedding_db(&state)
                .load(document.did.numeric)
                .unwrap()
                .is_some()
        );
        assert!(
            test_embedding_db(&state)
                .load(chunk_doc_id(document.did.numeric, 1))
                .unwrap()
                .is_some()
        );
        assert!(
            test_embedding_db(&state)
                .load(chunk_doc_id(document.did.numeric, 2))
                .unwrap()
                .is_none()
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
            &fake_chunk_offsets(document.did.numeric, 1),
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
                &fake_chunk_offsets(updated.did.numeric, 1),
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
            &fake_chunk_offsets(document.did.numeric, 1),
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
            &fake_chunk_offsets(document.did.numeric, 2),
        )
        .unwrap();

        let prev = capture_previous_state(&state, "notes", "hello.md").unwrap();
        let prev = prev.expect("existing document should have previous state");
        assert!(prev.metadata.is_some());
        assert_eq!(
            prev.user_metadata,
            Some(serde_json::json!({"tag": "important"}))
        );
        assert!(
            !prev.embeddings.is_empty(),
            "should capture existing embeddings"
        );
    }

    #[test]
    fn rollback_document_restores_metadata_and_embeddings_for_overwrite() {
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
        let original_chunk_offsets =
            fake_chunk_offsets(original.did.numeric, 2);
        ingest_prepared_document(
            &state,
            "notes",
            &original,
            &original_embeddings,
            &original_chunk_offsets,
        )
        .unwrap();

        // Capture state before overwrite
        let prev = capture_previous_state(&state, "notes", "hello.md").unwrap();

        // Overwrite
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
            &fake_chunk_offsets(updated.did.numeric, 3),
        )
        .unwrap();

        // Verify updated state
        assert_eq!(
            test_config_db(&state)
                .get_document_user_metadata(original.did.numeric)
                .unwrap(),
            Some(serde_json::json!({"version": 2}))
        );

        // Restore original file bytes
        std::fs::write(&full_path, "# Original\n\nBody").unwrap();

        // Rollback
        rollback_document(&state, "notes", "hello.md", prev.as_ref()).unwrap();

        // Metadata should be restored
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

        // Embeddings should be restored (original had 2 chunks)
        assert!(
            test_embedding_db(&state)
                .load(original.did.numeric)
                .unwrap()
                .is_some(),
            "base embedding should be restored"
        );
        assert!(
            test_embedding_db(&state)
                .load(chunk_doc_id(original.did.numeric, 1))
                .unwrap()
                .is_some(),
            "chunk embedding should be restored"
        );
        // Third chunk from the overwrite should be gone
        assert!(
            test_embedding_db(&state)
                .load(chunk_doc_id(original.did.numeric, 2))
                .unwrap()
                .is_none(),
            "overwrite chunk should be removed"
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
            &fake_chunk_offsets(document.did.numeric, 1),
        )
        .unwrap();

        // Rollback with no previous state (new document)
        rollback_document(&state, "notes", "new.md", None).unwrap();

        // Document should be fully deleted
        assert!(
            test_config_db(&state)
                .get_document_metadata_typed(document.did.numeric)
                .unwrap()
                .is_none(),
            "metadata should be deleted"
        );
        assert!(
            test_embedding_db(&state)
                .load(document.did.numeric)
                .unwrap()
                .is_none(),
            "embeddings should be deleted"
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
