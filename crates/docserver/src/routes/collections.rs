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
    // Store an empty path to mark it as existing.
    state.config_db.set_collection(&body.name, "")?;

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
        let mut writer =
            state.writer.lock().map_err(|e| ApiError::internal(e))?;
        state.search_index.delete_collection(&writer, &name);
        writer.commit().map_err(ApiError::internal)?;
    }

    // Delete embeddings for documents in this collection.
    let all_meta = state.config_db.list_all_document_metadata()?;
    let mut ids_to_remove = Vec::new();
    for (doc_id, bytes) in &all_meta {
        if let Some(meta) =
            docbert_core::incremental::DocumentMetadata::deserialize(bytes)
        {
            if meta.collection == name {
                ids_to_remove.push(*doc_id);
            }
        }
    }
    if !ids_to_remove.is_empty() {
        state.embedding_db.batch_remove(&ids_to_remove)?;
        state
            .config_db
            .batch_remove_document_metadata(&ids_to_remove)?;
    }

    state.config_db.remove_collection(&name)?;

    Ok(StatusCode::NO_CONTENT)
}
