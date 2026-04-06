#![allow(dead_code)]

use std::path::Path;

use docbert_core::{
    DocumentId,
    chunking::document_family_key,
    error,
    incremental,
    preparation::{self, SearchDocument},
};

use crate::{collection_snapshots, web::state::AppState};

pub(crate) type EmbeddingEntry = (u64, u32, u32, Vec<f32>);

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct IngestedDocument {
    pub(crate) doc_id: String,
    pub(crate) path: String,
    pub(crate) title: String,
    pub(crate) metadata: Option<serde_json::Value>,
}

pub(crate) fn load_markdown_document(
    collection: &str,
    relative_path: &str,
    full_path: &Path,
    metadata: Option<serde_json::Value>,
    mtime: u64,
) -> error::Result<SearchDocument> {
    let raw_markdown = std::fs::read_to_string(full_path)?;
    let prepared =
        preparation::prepare_markdown(Path::new(relative_path), &raw_markdown);
    let did = docbert_core::DocumentId::new(collection, relative_path);

    Ok(SearchDocument {
        did,
        relative_path: relative_path.to_string(),
        title: prepared.title,
        searchable_body: prepared.searchable_body,
        raw_content: None,
        metadata,
        mtime,
    })
}

pub(crate) fn ingest_prepared_document(
    state: &AppState,
    collection: &str,
    document: &SearchDocument,
    embedding_entries: &[EmbeddingEntry],
) -> error::Result<IngestedDocument> {
    let collection_root = collection_root(state, collection)?;
    let previous_snapshot = collection_snapshots::load_collection_snapshot(
        &state.config_db,
        collection,
    )?;
    let existing_embeddings =
        load_document_family_embeddings(state, document.did.numeric)?;
    let existing_embedding_ids: Vec<u64> = existing_embeddings
        .iter()
        .map(|(doc_id, _, _, _)| *doc_id)
        .collect();
    let current_embedding_ids: Vec<u64> = embedding_entries
        .iter()
        .map(|(doc_id, _, _, _)| *doc_id)
        .collect();

    let previous_metadata = state
        .config_db
        .get_document_metadata_typed(document.did.numeric)?;
    let previous_user_metadata = state
        .config_db
        .get_document_user_metadata(document.did.numeric)?;

    let mut writer = state.writer.lock().map_err(|_| {
        error::Error::Config("failed to lock tantivy writer".to_string())
    })?;

    state.search_index.add_document(
        &writer,
        &document.did.to_string(),
        document.did.numeric,
        collection,
        &document.relative_path,
        &document.title,
        &document.searchable_body,
        document.mtime,
    )?;

    if let Err(err) = state.embedding_db.batch_store(embedding_entries) {
        let _ = writer.rollback();
        return Err(err);
    }

    if let Err(err) = persist_metadata(state, collection, document) {
        let _ = writer.rollback();
        restore_previous_embeddings(
            state,
            &existing_embeddings,
            &current_embedding_ids,
        )?;
        restore_previous_metadata(
            state,
            document.did.numeric,
            previous_metadata.as_ref(),
            previous_user_metadata.as_ref(),
        )?;
        return Err(err);
    }

    if let Err(err) = writer.commit() {
        restore_previous_embeddings(
            state,
            &existing_embeddings,
            &current_embedding_ids,
        )?;
        restore_previous_metadata(
            state,
            document.did.numeric,
            previous_metadata.as_ref(),
            previous_user_metadata.as_ref(),
        )?;
        return Err(err.into());
    }

    // The collection snapshot must move in lockstep with successful ingest
    // side effects. If snapshot refresh fails, keep the previous snapshot.
    if let Err(err) = refresh_collection_snapshot(
        state,
        collection,
        &collection_root,
        previous_snapshot.as_ref(),
    ) {
        restore_previous_embeddings(
            state,
            &existing_embeddings,
            &current_embedding_ids,
        )?;
        restore_previous_metadata(
            state,
            document.did.numeric,
            previous_metadata.as_ref(),
            previous_user_metadata.as_ref(),
        )?;
        return Err(err);
    }

    remove_stale_previous_embeddings(
        state,
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

pub(crate) fn delete_document(
    state: &AppState,
    collection: &str,
    relative_path: &str,
) -> error::Result<()> {
    let collection_root = collection_root(state, collection)?;
    let previous_snapshot = collection_snapshots::load_collection_snapshot(
        &state.config_db,
        collection,
    )?;
    let did = DocumentId::new(collection, relative_path);
    let mut writer = state.writer.lock().map_err(|_| {
        error::Error::Config("failed to lock tantivy writer".to_string())
    })?;

    state
        .search_index
        .delete_document(&writer, &did.to_string());
    writer.commit()?;

    state.embedding_db.remove_document_family(did.numeric)?;
    state.config_db.remove_document_metadata(did.numeric)?;
    state.config_db.remove_document_user_metadata(did.numeric)?;
    // Delete updates the stored collection snapshot only after index and
    // metadata cleanup succeeds end to end.
    refresh_collection_snapshot(
        state,
        collection,
        &collection_root,
        previous_snapshot.as_ref(),
    )?;

    Ok(())
}

fn collection_root(
    state: &AppState,
    collection: &str,
) -> error::Result<std::path::PathBuf> {
    let root =
        state.config_db.get_collection(collection)?.ok_or_else(|| {
            error::Error::NotFound {
                kind: "collection",
                name: collection.to_string(),
            }
        })?;
    Ok(std::path::PathBuf::from(root))
}

fn persist_metadata(
    state: &AppState,
    collection: &str,
    document: &SearchDocument,
) -> error::Result<()> {
    let metadata = incremental::DocumentMetadata {
        collection: collection.to_string(),
        relative_path: document.relative_path.clone(),
        mtime: document.mtime,
    };
    state
        .config_db
        .set_document_metadata_typed(document.did.numeric, &metadata)?;

    match document.metadata.as_ref() {
        Some(value) => state
            .config_db
            .set_document_user_metadata(document.did.numeric, value)?,
        None => {
            state
                .config_db
                .remove_document_user_metadata(document.did.numeric)?;
        }
    }

    Ok(())
}

fn refresh_collection_snapshot(
    state: &AppState,
    collection: &str,
    collection_root: &Path,
    previous_snapshot: Option<&docbert_core::merkle::CollectionMerkleSnapshot>,
) -> error::Result<()> {
    let current_snapshot = collection_snapshots::compute_collection_snapshot(
        collection,
        collection_root,
    )?;
    if let Err(err) = collection_snapshots::replace_collection_snapshot(
        &state.config_db,
        &current_snapshot,
    ) {
        restore_previous_collection_snapshot(
            state,
            collection,
            previous_snapshot,
        )?;
        return Err(err);
    }
    Ok(())
}

fn restore_previous_collection_snapshot(
    state: &AppState,
    collection: &str,
    previous_snapshot: Option<&docbert_core::merkle::CollectionMerkleSnapshot>,
) -> error::Result<()> {
    match previous_snapshot {
        Some(snapshot) => collection_snapshots::replace_collection_snapshot(
            &state.config_db,
            snapshot,
        ),
        None => {
            state
                .config_db
                .remove_collection_merkle_snapshot(collection)?;
            Ok(())
        }
    }
}

fn load_document_family_embeddings(
    state: &AppState,
    base_doc_id: u64,
) -> error::Result<Vec<EmbeddingEntry>> {
    let family_key = document_family_key(base_doc_id);
    let doc_ids: Vec<u64> = state
        .embedding_db
        .list_ids()?
        .into_iter()
        .filter(|doc_id| document_family_key(*doc_id) == family_key)
        .collect();

    let loaded = state.embedding_db.batch_load(&doc_ids)?;
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
    state: &AppState,
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
        state.embedding_db.batch_remove(&stale_new_ids)?;
    }
    if !previous_embeddings.is_empty() {
        state.embedding_db.batch_store(previous_embeddings)?;
    }
    Ok(())
}

fn restore_previous_metadata(
    state: &AppState,
    doc_id: u64,
    previous_metadata: Option<&incremental::DocumentMetadata>,
    previous_user_metadata: Option<&serde_json::Value>,
) -> error::Result<()> {
    match previous_metadata {
        Some(metadata) => state
            .config_db
            .set_document_metadata_typed(doc_id, metadata)?,
        None => {
            state.config_db.remove_document_metadata(doc_id)?;
        }
    }

    match previous_user_metadata {
        Some(value) => {
            state.config_db.set_document_user_metadata(doc_id, value)?
        }
        None => {
            state.config_db.remove_document_user_metadata(doc_id)?;
        }
    }

    Ok(())
}

fn remove_stale_previous_embeddings(
    state: &AppState,
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
        state.embedding_db.batch_remove(&stale_previous_ids)?;
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
        EmbeddingDb,
        ModelManager,
        SearchIndex,
        chunking::chunk_doc_id,
    };

    use super::*;
    use crate::web::state::Inner;

    fn test_state() -> (tempfile::TempDir, AppState) {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let search_index = SearchIndex::open_in_ram().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let writer = search_index.writer(15_000_000).unwrap();
        let state = Arc::new(Inner {
            config_db,
            search_index,
            embedding_db,
            model: Mutex::new(ModelManager::new()),
            writer: Mutex::new(writer),
        });
        (tmp, state)
    }

    fn seed_collection_root(
        tmp: &tempfile::TempDir,
        state: &AppState,
        collection: &str,
    ) -> PathBuf {
        let root = tmp.path().join(collection);
        std::fs::create_dir_all(&root).unwrap();
        state
            .config_db
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

    #[test]
    fn web_ingest_initial_ingest_stores_index_embeddings_and_metadata() {
        let (tmp, state) = test_state();
        let root = seed_collection_root(&tmp, &state, "notes");
        let full_path = write_markdown(&root, "hello.md", "# Hello\n\nBody");
        let document = load_markdown_document(
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
        )
        .unwrap();

        let snapshot = state
            .config_db
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after ingest");

        assert_eq!(ingested.doc_id, document.did.to_string());
        assert_eq!(ingested.title, "Hello");
        assert_eq!(
            state
                .config_db
                .get_document_metadata_typed(document.did.numeric)
                .unwrap()
                .unwrap()
                .mtime,
            7
        );
        assert_eq!(
            state
                .config_db
                .get_document_user_metadata(document.did.numeric)
                .unwrap(),
            Some(serde_json::json!({"topic": "rust"}))
        );
        assert!(
            state
                .embedding_db
                .load(document.did.numeric)
                .unwrap()
                .is_some()
        );
        assert!(
            state
                .embedding_db
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
            load_markdown_document("notes", "hello.md", &full_path, None, 1)
                .unwrap();
        ingest_prepared_document(
            &state,
            "notes",
            &first,
            &fake_embedding_entries(first.did.numeric, 2),
        )
        .unwrap();
        let first_snapshot = state
            .config_db
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after first ingest");

        write_markdown(&root, "hello.md", "# Updated\n\nBody v2");
        let second = load_markdown_document(
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
        )
        .unwrap();

        let second_snapshot = state
            .config_db
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after replacement ingest");

        let metadata = state
            .config_db
            .get_document_metadata_typed(second.did.numeric)
            .unwrap()
            .unwrap();
        assert_eq!(metadata.mtime, 9);
        assert_eq!(
            state
                .config_db
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
            load_markdown_document("notes", "hello.md", &full_path, None, 1)
                .unwrap();
        ingest_prepared_document(
            &state,
            "notes",
            &document,
            &fake_embedding_entries(document.did.numeric, 3),
        )
        .unwrap();

        ingest_prepared_document(
            &state,
            "notes",
            &document,
            &fake_embedding_entries(document.did.numeric, 2),
        )
        .unwrap();

        assert!(
            state
                .embedding_db
                .load(document.did.numeric)
                .unwrap()
                .is_some()
        );
        assert!(
            state
                .embedding_db
                .load(chunk_doc_id(document.did.numeric, 1))
                .unwrap()
                .is_some()
        );
        assert!(
            state
                .embedding_db
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
            load_markdown_document("notes", "hello.md", &full_path, None, 1)
                .unwrap();
        ingest_prepared_document(
            &state,
            "notes",
            &document,
            &fake_embedding_entries(document.did.numeric, 1),
        )
        .unwrap();
        let original_snapshot = state
            .config_db
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after first ingest");

        write_markdown(&root, "hello.md", "# Hello\n\nUpdated");
        let updated =
            load_markdown_document("notes", "hello.md", &full_path, None, 2)
                .unwrap();
        std::fs::remove_dir_all(&root).unwrap();

        assert!(
            ingest_prepared_document(
                &state,
                "notes",
                &updated,
                &fake_embedding_entries(updated.did.numeric, 1),
            )
            .is_err()
        );
        assert_eq!(
            state
                .config_db
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
            load_markdown_document("notes", "hello.md", &full_path, None, 1)
                .unwrap();
        ingest_prepared_document(
            &state,
            "notes",
            &document,
            &fake_embedding_entries(document.did.numeric, 1),
        )
        .unwrap();
        let original_snapshot = state
            .config_db
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after ingest");

        std::fs::remove_dir_all(&root).unwrap();

        assert!(delete_document(&state, "notes", "hello.md").is_err());
        assert_eq!(
            state
                .config_db
                .get_collection_merkle_snapshot("notes")
                .unwrap(),
            Some(original_snapshot)
        );
    }

    #[test]
    fn web_ingest_load_markdown_document_reads_from_disk() {
        let (tmp, state) = test_state();
        let root = seed_collection_root(&tmp, &state, "notes");
        let full_path = write_markdown(
            &root,
            "nested/hello.md",
            "---\ntitle: ignored\n---\n# From Disk\n\nBody",
        );

        let document = load_markdown_document(
            "notes",
            "nested/hello.md",
            &full_path,
            None,
            5,
        )
        .unwrap();

        assert_eq!(document.relative_path, "nested/hello.md");
        assert_eq!(document.title, "From Disk");
        assert_eq!(document.searchable_body, "# From Disk\n\nBody");
        assert_eq!(document.mtime, 5);
    }
}
