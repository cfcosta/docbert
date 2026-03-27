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

#[derive(Debug)]
struct PreparedDocument {
    did: DocumentId,
    path: String,
    title: String,
    body: String,
    content: String,
    metadata: Option<serde_json::Value>,
    embedding_chunks: Vec<(u64, String)>,
}

fn prepare_documents(
    collection: &str,
    documents: &[IngestDocument],
) -> Result<Vec<PreparedDocument>, ApiError> {
    let mut prepared = Vec::with_capacity(documents.len());

    for doc in documents {
        let processed =
            content::process(&doc.content_type, &doc.path, &doc.content);
        let did = DocumentId::new(collection, &doc.path);
        let chunks = chunking::chunk_text(
            &processed.body,
            chunking::DEFAULT_CHUNK_SIZE,
            chunking::DEFAULT_CHUNK_OVERLAP,
        );

        if chunks.is_empty() {
            return Err(ApiError::BadRequest(format!(
                "document has no embeddable content after preprocessing: {}",
                doc.path
            )));
        }

        tracing::debug!(
            doc_id = %did,
            path = %doc.path,
            title = %processed.title,
            body_len = processed.body.len(),
            chunks = chunks.len(),
            "prepared document for ingestion",
        );

        let embedding_chunks = chunks
            .into_iter()
            .map(|chunk| {
                (chunking::chunk_doc_id(did.numeric, chunk.index), chunk.text)
            })
            .collect();

        prepared.push(PreparedDocument {
            did,
            path: doc.path.clone(),
            title: processed.title,
            body: processed.body,
            content: doc.content.clone(),
            metadata: doc.metadata.clone(),
            embedding_chunks,
        });
    }

    Ok(prepared)
}

fn rollback_writer(writer: &mut tantivy::IndexWriter) {
    if let Err(e) = writer.rollback() {
        tracing::warn!(error = %e, "failed to rollback tantivy writer");
    }
}

fn cleanup_embeddings(state: &AppState, embedding_ids: &[u64], context: &str) {
    if embedding_ids.is_empty() {
        return;
    }

    if let Err(e) = state.embedding_db.batch_remove(embedding_ids) {
        tracing::warn!(error = %e, ids = embedding_ids.len(), %context, "failed to cleanup embeddings");
    }
}

fn cleanup_persisted_documents(
    state: &AppState,
    doc_ids: &[u64],
    context: &str,
) {
    for &doc_id in doc_ids {
        if let Err(e) = state.config_db.remove_document_metadata(doc_id) {
            tracing::warn!(error = %e, doc_id, %context, "failed to cleanup document metadata");
        }

        let content_key = format!("doc_content:{doc_id}");
        if let Err(e) = state.config_db.remove_setting(&content_key) {
            tracing::warn!(error = %e, doc_id, %context, "failed to cleanup stored content");
        }

        let meta_key = format!("doc_meta:{doc_id}");
        if let Err(e) = state.config_db.remove_setting(&meta_key) {
            tracing::warn!(error = %e, doc_id, %context, "failed to cleanup stored metadata");
        }
    }
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

    let doc_count = body.documents.len();
    tracing::info!(
        collection = %body.collection,
        documents = doc_count,
        "starting ingestion",
    );

    let prepared = prepare_documents(&body.collection, &body.documents)?;
    let docs_to_embed: Vec<(u64, String)> = prepared
        .iter()
        .flat_map(|doc| doc.embedding_chunks.iter().cloned())
        .collect();

    tracing::info!(
        collection = %body.collection,
        chunks = docs_to_embed.len(),
        "computing embeddings",
    );
    let embedding_entries = {
        let mut model = state.model.lock().map_err(ApiError::internal)?;
        embedding::embed_documents(&mut model, docs_to_embed)
            .map_err(ApiError::internal)?
    };
    let embedding_ids: Vec<u64> = embedding_entries
        .iter()
        .map(|(doc_id, _, _, _)| *doc_id)
        .collect();
    tracing::info!(
        collection = %body.collection,
        stored = embedding_entries.len(),
        "embeddings computed",
    );

    // Acquire the Tantivy writer for the duration of this ingestion.
    let mut writer = state.writer.lock().map_err(ApiError::internal)?;

    // Phase 1: Delete any existing entries for documents being re-ingested.
    // We must commit deletes before adding new versions because Tantivy's
    // delete_term applies to all documents matching the term in the same
    // commit, including freshly added ones with the same doc_id.
    if !prepared.is_empty() {
        for doc in &prepared {
            state
                .search_index
                .delete_document(&writer, &doc.did.to_string());
        }
        writer.commit().map_err(ApiError::internal)?;
    }

    // Phase 2: Stage all documents in Tantivy. Nothing is visible until commit.
    for doc in &prepared {
        if let Err(e) = state.search_index.add_document(
            &writer,
            &doc.did.to_string(),
            doc.did.numeric,
            &body.collection,
            &doc.path,
            &doc.title,
            &doc.body,
            0, // mtime not meaningful for API-ingested documents
        ) {
            rollback_writer(&mut writer);
            return Err(ApiError::from(e));
        }
    }

    // Phase 3: Persist embeddings. Ingestion must fail if embeddings cannot be
    // stored, so we do this before committing Tantivy.
    if let Err(e) = state.embedding_db.batch_store(&embedding_entries) {
        rollback_writer(&mut writer);
        return Err(ApiError::internal(e));
    }
    tracing::info!(
        collection = %body.collection,
        stored = embedding_entries.len(),
        "embeddings stored",
    );

    // Phase 4: Persist metadata and raw content. If this fails, remove the
    // freshly stored embeddings and discard staged Tantivy changes.
    let mut persisted_doc_ids = Vec::with_capacity(prepared.len());
    let mut ingested_docs = Vec::with_capacity(prepared.len());
    for doc in &prepared {
        let persist_result: Result<(), ApiError> = (|| {
            let doc_meta = incremental::DocumentMetadata {
                collection: body.collection.clone(),
                relative_path: doc.path.clone(),
                mtime: 0,
            };
            state.config_db.set_document_metadata(
                doc.did.numeric,
                &doc_meta.serialize(),
            )?;

            let content_key = format!("doc_content:{}", doc.did.numeric);
            state.config_db.set_setting(&content_key, &doc.content)?;

            let meta_key = format!("doc_meta:{}", doc.did.numeric);
            match &doc.metadata {
                Some(user_meta) => {
                    let meta_val = serde_json::to_string(user_meta)
                        .map_err(ApiError::internal)?;
                    state.config_db.set_setting(&meta_key, &meta_val)?;
                }
                None => {
                    let _ = state.config_db.remove_setting(&meta_key);
                }
            }

            Ok(())
        })();

        if let Err(e) = persist_result {
            rollback_writer(&mut writer);
            cleanup_embeddings(
                &state,
                &embedding_ids,
                "metadata persistence failed",
            );
            cleanup_persisted_documents(
                &state,
                &persisted_doc_ids,
                "metadata persistence failed",
            );
            return Err(e);
        }

        persisted_doc_ids.push(doc.did.numeric);
        ingested_docs.push(IngestedDoc {
            doc_id: doc.did.to_string(),
            path: doc.path.clone(),
            title: doc.title.clone(),
            metadata: doc.metadata.clone(),
        });
    }
    tracing::debug!(
        collection = %body.collection,
        documents = persisted_doc_ids.len(),
        "metadata and content persisted",
    );

    // Phase 5: Make Tantivy changes visible. If this fails, clean up the other
    // persisted state so the request is rejected rather than partially accepted.
    if let Err(e) = writer.commit() {
        cleanup_embeddings(&state, &embedding_ids, "tantivy commit failed");
        cleanup_persisted_documents(
            &state,
            &persisted_doc_ids,
            "tantivy commit failed",
        );
        return Err(ApiError::internal(e));
    }
    tracing::info!(
        collection = %body.collection,
        documents = prepared.len(),
        "tantivy index committed",
    );

    tracing::info!(
        collection = %body.collection,
        ingested = ingested_docs.len(),
        "ingestion complete",
    );

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
        if let Some(meta) = incremental::DocumentMetadata::deserialize(bytes)
            && meta.collection == collection
        {
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
        .search(&format!("\"{did}\""), 1)
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
        let mut writer = state.writer.lock().map_err(ApiError::internal)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_documents_rejects_empty_body() {
        let documents = vec![IngestDocument {
            path: "empty.md".to_string(),
            content: "   \n\t".to_string(),
            content_type: "text/markdown".to_string(),
            metadata: None,
        }];

        let err = prepare_documents("notes", &documents).unwrap_err();
        match err {
            ApiError::BadRequest(message) => {
                assert!(message.contains("empty.md"));
                assert!(message.contains("no embeddable content"));
            }
            other => panic!("expected bad request, got {other:?}"),
        }
    }

    #[test]
    fn prepare_documents_rejects_frontmatter_only_body() {
        let documents = vec![IngestDocument {
            path: "frontmatter-only.md".to_string(),
            content: "---\ntitle: Hidden\n---\n".to_string(),
            content_type: "text/markdown".to_string(),
            metadata: None,
        }];

        let err = prepare_documents("notes", &documents).unwrap_err();
        match err {
            ApiError::BadRequest(message) => {
                assert!(message.contains("frontmatter-only.md"));
                assert!(message.contains("no embeddable content"));
            }
            other => panic!("expected bad request, got {other:?}"),
        }
    }

    #[test]
    fn prepare_documents_builds_embedding_chunks_for_markdown_body() {
        let documents = vec![IngestDocument {
            path: "alpha.md".to_string(),
            content: "# Alpha\n\nRust ownership and borrowing.".to_string(),
            content_type: "text/markdown".to_string(),
            metadata: None,
        }];

        let prepared = prepare_documents("notes", &documents).unwrap();
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].title, "Alpha");
        assert_eq!(prepared[0].embedding_chunks.len(), 1);
        assert_eq!(prepared[0].embedding_chunks[0].0, prepared[0].did.numeric);
        assert!(prepared[0].embedding_chunks[0].1.contains("ownership"));
    }
}
