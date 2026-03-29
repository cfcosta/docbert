use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};

use crate::{error::ApiError, state::AppState};

#[derive(Deserialize)]
pub struct CreateRequest {
    name: String,
}

#[derive(Serialize)]
pub struct CollectionItem {
    name: String,
}

pub async fn create(
    State(state): State<AppState>,
    Json(body): Json<CreateRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let existing = state.config_db.get_collection(&body.name)?;
    if existing.is_some() {
        return Err(ApiError::Conflict(format!(
            "collection already exists: {}",
            body.name
        )));
    }

    // Server-managed collections don't map to a filesystem directory.
    state.config_db.set_managed_collection(&body.name)?;

    Ok((
        StatusCode::CREATED,
        Json(CollectionItem { name: body.name }),
    ))
}

pub async fn list(
    State(state): State<AppState>,
) -> Result<Json<Vec<CollectionItem>>, ApiError> {
    let collections = state.config_db.list_collections()?;
    let items = collections
        .into_iter()
        .map(|(name, _path)| CollectionItem { name })
        .collect();
    Ok(Json(items))
}

pub async fn delete(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let existing = state.config_db.get_collection(&name)?;
    if existing.is_none() {
        return Err(ApiError::NotFound(format!(
            "collection not found: {name}"
        )));
    }

    // Delete Tantivy entries for this collection.
    {
        let mut writer = state.writer.lock().map_err(ApiError::internal)?;
        state.search_index.delete_collection(&writer, &name);
        writer.commit().map_err(ApiError::internal)?;
    }

    // Delete embeddings for documents in this collection.
    let all_meta = state.config_db.list_all_document_metadata_typed()?;
    let mut ids_to_remove = Vec::new();
    for (doc_id, meta) in &all_meta {
        if meta.collection == name {
            ids_to_remove.push(*doc_id);
        }
    }
    if !ids_to_remove.is_empty() {
        state
            .embedding_db
            .batch_remove_document_families(&ids_to_remove)?;
        state
            .config_db
            .batch_remove_document_artifacts(&ids_to_remove)?;
    }

    state.config_db.remove_collection(&name)?;

    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use docbert_core::{
        ConfigDb,
        DocumentId,
        EmbeddingDb,
        SearchIndex,
        chunking::chunk_doc_id,
        incremental,
    };

    use super::*;
    use crate::state::Inner;

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
            model: Mutex::new(docbert_core::ModelManager::new()),
            writer: Mutex::new(writer),
        });

        (tmp, state)
    }

    fn seed_collection_document(
        state: &AppState,
        collection: &str,
        path: &str,
    ) -> u64 {
        state.config_db.set_managed_collection(collection).unwrap();
        let did = DocumentId::new(collection, path);
        let metadata = incremental::DocumentMetadata {
            collection: collection.to_string(),
            relative_path: path.to_string(),
            mtime: 0,
        };
        state
            .config_db
            .put_document_artifacts(
                did.numeric,
                &metadata,
                "# Hello\nBody",
                Some(&serde_json::json!({ "topic": "rust" })),
            )
            .unwrap();
        state
            .embedding_db
            .store(did.numeric, 1, 2, &[1.0, 2.0])
            .unwrap();
        state
            .embedding_db
            .store(chunk_doc_id(did.numeric, 1), 1, 2, &[3.0, 4.0])
            .unwrap();

        did.numeric
    }

    #[tokio::test]
    async fn collection_delete_removes_document_artifacts_for_collection_docs()
    {
        let (_tmp, state) = test_state();
        let doc_id = seed_collection_document(&state, "notes", "hello.md");

        let status = delete(State(state.clone()), Path("notes".to_string()))
            .await
            .unwrap()
            .into_response()
            .status();
        assert_eq!(status, StatusCode::NO_CONTENT);
        assert!(state.config_db.get_collection("notes").unwrap().is_none());
        assert!(
            state
                .config_db
                .get_document_metadata_typed(doc_id)
                .unwrap()
                .is_none()
        );
        assert!(
            state
                .config_db
                .get_document_content(doc_id)
                .unwrap()
                .is_none()
        );
        assert!(
            state
                .config_db
                .get_document_user_metadata(doc_id)
                .unwrap()
                .is_none()
        );
        assert!(state.embedding_db.load(doc_id).unwrap().is_none());
        assert!(
            state
                .embedding_db
                .load(chunk_doc_id(doc_id, 1))
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn collection_delete_keeps_other_collection_chunk_families() {
        let (_tmp, state) = test_state();
        let notes_doc_id =
            seed_collection_document(&state, "notes", "hello.md");
        let docs_doc_id = seed_collection_document(&state, "docs", "guide.md");

        let status = delete(State(state.clone()), Path("notes".to_string()))
            .await
            .unwrap()
            .into_response()
            .status();
        assert_eq!(status, StatusCode::NO_CONTENT);

        assert!(state.embedding_db.load(notes_doc_id).unwrap().is_none());
        assert!(
            state
                .embedding_db
                .load(chunk_doc_id(notes_doc_id, 1))
                .unwrap()
                .is_none()
        );
        assert!(state.embedding_db.load(docs_doc_id).unwrap().is_some());
        assert!(
            state
                .embedding_db
                .load(chunk_doc_id(docs_doc_id, 1))
                .unwrap()
                .is_some()
        );
    }
}
