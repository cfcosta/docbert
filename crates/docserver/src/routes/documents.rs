use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use docbert_core::{DocumentId, chunking, embedding, incremental};
use serde::{Deserialize, Serialize};

use crate::{content, error::ApiError, state::AppState};

#[derive(Deserialize)]
pub struct IngestRequest {
    collection: String,
    documents: Vec<IngestDocument>,
}

#[derive(Deserialize)]
pub struct IngestDocument {
    path: String,
    content: String,
    content_type: String,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct IngestResponse {
    ingested: usize,
    documents: Vec<IngestedDoc>,
}

#[derive(Serialize)]
pub struct IngestedDoc {
    doc_id: String,
    path: String,
    title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}

pub async fn ingest(
    State(state): State<AppState>,
    Json(body): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, ApiError> {
    // Validate collection exists.
    let existing = state.config_db.get_collection(&body.collection)?;
    if existing.is_none() {
        return Err(ApiError::BadRequest(format!(
            "collection not found: {}",
            body.collection
        )));
    }

    // Validate all content types up front.
    for doc in &body.documents {
        if !content::is_supported(&doc.content_type) {
            return Err(ApiError::BadRequest(format!(
                "unsupported content type: {}",
                doc.content_type
            )));
        }
    }

    let mut ingested_docs = Vec::with_capacity(body.documents.len());
    let mut docs_to_embed: Vec<(u64, String)> = Vec::new();

    // Acquire the Tantivy writer for the duration of this ingestion.
    let mut writer = state.writer.lock().map_err(|e| ApiError::internal(e))?;

    for doc in &body.documents {
        let processed =
            content::process(&doc.content_type, &doc.path, &doc.content);
        let did = DocumentId::new(&body.collection, &doc.path);

        // Delete any existing entry for this document.
        state
            .search_index
            .delete_document(&writer, &did.to_string());

        // Add to Tantivy.
        state.search_index.add_document(
            &writer,
            &did.to_string(),
            did.numeric,
            &body.collection,
            &doc.path,
            &processed.title,
            &processed.body,
            0, // mtime not meaningful for API-ingested documents
        )?;

        // Prepare chunks for embedding.
        let chunks = chunking::chunk_text(
            &processed.body,
            chunking::DEFAULT_CHUNK_SIZE,
            chunking::DEFAULT_CHUNK_OVERLAP,
        );
        for chunk in chunks {
            let chunk_id = chunking::chunk_doc_id(did.numeric, chunk.index);
            docs_to_embed.push((chunk_id, chunk.text));
        }

        // Build metadata for storage: collection\0path\0mtime format
        // used by docbert-core's incremental module.
        let doc_meta = incremental::DocumentMetadata {
            collection: body.collection.clone(),
            relative_path: doc.path.clone(),
            mtime: 0,
        };
        state
            .config_db
            .set_document_metadata(did.numeric, &doc_meta.serialize())?;

        // Store raw content so GET /documents can return it.
        let content_key = format!("doc_content:{}", did.numeric);
        state.config_db.set_setting(&content_key, &doc.content)?;

        // Store user metadata as a setting keyed by doc ID if provided.
        if let Some(ref user_meta) = doc.metadata {
            let meta_key = format!("doc_meta:{}", did.numeric);
            let meta_val = serde_json::to_string(user_meta)
                .map_err(|e| ApiError::internal(e))?;
            state.config_db.set_setting(&meta_key, &meta_val)?;
        }

        ingested_docs.push(IngestedDoc {
            doc_id: did.to_string(),
            path: doc.path.clone(),
            title: processed.title,
            metadata: doc.metadata.clone(),
        });
    }

    writer.commit().map_err(ApiError::internal)?;

    // Compute embeddings (this can be slow, runs after Tantivy commit).
    if !docs_to_embed.is_empty() {
        let mut model =
            state.model.lock().map_err(|e| ApiError::internal(e))?;
        embedding::embed_and_store(
            &mut model,
            &state.embedding_db,
            docs_to_embed,
        )?;
    }

    Ok(Json(IngestResponse {
        ingested: ingested_docs.len(),
        documents: ingested_docs,
    }))
}

#[derive(Serialize)]
pub struct DocumentListItem {
    doc_id: String,
    path: String,
    title: String,
}

pub async fn list_by_collection(
    State(state): State<AppState>,
    Path(collection): Path<String>,
) -> Result<Json<Vec<DocumentListItem>>, ApiError> {
    let existing = state.config_db.get_collection(&collection)?;
    if existing.is_none() {
        return Err(ApiError::NotFound(format!(
            "collection not found: {collection}"
        )));
    }

    let all_meta = state.config_db.list_all_document_metadata()?;
    let mut items = Vec::new();
    for (doc_id, bytes) in &all_meta {
        if let Some(meta) = incremental::DocumentMetadata::deserialize(bytes) {
            if meta.collection == collection {
                let short_id = docbert_core::search::short_doc_id(*doc_id);
                // Try to get title from Tantivy search results.
                let title = state
                    .search_index
                    .search(&format!("\"{short_id}\""), 1)
                    .ok()
                    .and_then(|r| r.into_iter().next())
                    .map(|r| r.title)
                    .unwrap_or_else(|| meta.relative_path.clone());
                items.push(DocumentListItem {
                    doc_id: short_id,
                    path: meta.relative_path,
                    title,
                });
            }
        }
    }
    items.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(Json(items))
}

#[derive(Serialize)]
pub struct DocumentResponse {
    doc_id: String,
    collection: String,
    path: String,
    title: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}

pub async fn get(
    State(state): State<AppState>,
    Path((collection, path)): Path<(String, String)>,
) -> Result<Json<DocumentResponse>, ApiError> {
    let did = DocumentId::new(&collection, &path);

    // Check that the document metadata exists.
    let meta_bytes = state
        .config_db
        .get_document_metadata(did.numeric)?
        .ok_or_else(|| {
            ApiError::NotFound(format!(
                "document not found: {collection}:{path}"
            ))
        })?;

    let meta = incremental::DocumentMetadata::deserialize(&meta_bytes)
        .ok_or_else(|| {
            ApiError::internal("failed to deserialize document metadata")
        })?;

    // Retrieve the document content from Tantivy.
    // Search by the doc_id to find it.
    let results = state
        .search_index
        .search(&format!("\"{}\"", did.to_string()), 1)
        .unwrap_or_default();

    // Fall back: we have metadata but might not find it via search.
    // Try a direct path-based lookup.
    let title = if let Some(r) = results.first() {
        r.title.clone()
    } else {
        path.clone()
    };

    // Load stored content.
    let content_key = format!("doc_content:{}", did.numeric);
    let content = state
        .config_db
        .get_setting(&content_key)?
        .unwrap_or_default();

    // Load user metadata from settings.
    let user_meta = load_user_metadata(&state, did.numeric);

    Ok(Json(DocumentResponse {
        doc_id: did.to_string(),
        collection: meta.collection,
        path: meta.relative_path,
        title,
        content,
        metadata: user_meta,
    }))
}

pub async fn delete(
    State(state): State<AppState>,
    Path((collection, path)): Path<(String, String)>,
) -> Result<impl IntoResponse, ApiError> {
    let did = DocumentId::new(&collection, &path);

    let exists = state
        .config_db
        .get_document_metadata(did.numeric)?
        .is_some();
    if !exists {
        return Err(ApiError::NotFound(format!(
            "document not found: {collection}:{path}"
        )));
    }

    // Delete from Tantivy.
    {
        let mut writer =
            state.writer.lock().map_err(|e| ApiError::internal(e))?;
        state
            .search_index
            .delete_document(&writer, &did.to_string());
        writer.commit().map_err(ApiError::internal)?;
    }

    // Delete embeddings (base + chunks).
    let _ = state.embedding_db.remove(did.numeric);

    // Delete metadata.
    state.config_db.remove_document_metadata(did.numeric)?;

    // Delete stored content and user metadata.
    let content_key = format!("doc_content:{}", did.numeric);
    let _ = state.config_db.remove_setting(&content_key);
    let meta_key = format!("doc_meta:{}", did.numeric);
    let _ = state.config_db.remove_setting(&meta_key);

    Ok(StatusCode::NO_CONTENT)
}

fn load_user_metadata(
    state: &AppState,
    doc_numeric_id: u64,
) -> Option<serde_json::Value> {
    let meta_key = format!("doc_meta:{doc_numeric_id}");
    state
        .config_db
        .get_setting(&meta_key)
        .ok()
        .flatten()
        .and_then(|s| serde_json::from_str(&s).ok())
}
