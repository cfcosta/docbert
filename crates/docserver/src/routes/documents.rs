use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use docbert_core::{DocumentId, chunking, embedding};
use serde::{Deserialize, Serialize};

use crate::{
    content,
    document_ingest::{
        AppStateIngestionBackend,
        DocumentIngestCommand,
        EmbeddingEntry,
        PersistedIngestDocument,
        PreparedIngestDocument,
    },
    error::ApiError,
    state::AppState,
};

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

fn prepare_documents(
    collection: &str,
    documents: &[IngestDocument],
) -> Result<Vec<PreparedIngestDocument>, ApiError> {
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

        prepared.push(PreparedIngestDocument {
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
    let embedding_entries: Vec<EmbeddingEntry> = {
        let mut model = state.model.lock().map_err(ApiError::internal)?;
        embedding::embed_documents(&mut model, docs_to_embed)
            .map_err(ApiError::internal)?
    };
    tracing::info!(
        collection = %body.collection,
        stored = embedding_entries.len(),
        "embeddings computed",
    );

    let mut backend = AppStateIngestionBackend::new(&state)?;
    let ingested_documents = DocumentIngestCommand {
        backend: &mut backend,
        collection: &body.collection,
        documents: &prepared,
        embedding_entries: &embedding_entries,
    }
    .execute()?;
    tracing::info!(
        collection = %body.collection,
        documents = prepared.len(),
        "ingestion complete",
    );

    Ok(Json(IngestResponse {
        ingested: ingested_documents.len(),
        documents: ingested_documents
            .into_iter()
            .map(|document: PersistedIngestDocument| IngestedDoc {
                doc_id: document.doc_id,
                path: document.path,
                title: document.title,
                metadata: document.metadata,
            })
            .collect(),
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

    let all_meta = state.config_db.list_all_document_metadata_typed()?;
    let mut items = Vec::new();
    for (doc_id, meta) in &all_meta {
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
                path: meta.relative_path.clone(),
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
    let meta = state
        .config_db
        .get_document_metadata_typed(did.numeric)?
        .ok_or_else(|| {
            ApiError::NotFound(format!(
                "document not found: {collection}:{path}"
            ))
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
    let content = state
        .config_db
        .get_document_content(did.numeric)?
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
        .get_document_metadata_typed(did.numeric)?
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
    let _ = state.embedding_db.remove_document_family(did.numeric)?;

    // Delete stored metadata and document artifacts.
    let _ = state.config_db.remove_document_artifacts(did.numeric)?;

    Ok(StatusCode::NO_CONTENT)
}

fn load_user_metadata(
    state: &AppState,
    doc_numeric_id: u64,
) -> Option<serde_json::Value> {
    state
        .config_db
        .get_document_user_metadata(doc_numeric_id)
        .ok()
        .flatten()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use docbert_core::{
        ConfigDb,
        EmbeddingDb,
        ModelManager,
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
            model: Mutex::new(ModelManager::new()),
            writer: Mutex::new(writer),
        });

        (tmp, state)
    }

    fn seed_document(
        state: &AppState,
        collection: &str,
        path: &str,
        content: &str,
        metadata: Option<serde_json::Value>,
    ) -> DocumentId {
        state.config_db.set_managed_collection(collection).unwrap();
        let did = DocumentId::new(collection, path);
        let stored_metadata = incremental::DocumentMetadata {
            collection: collection.to_string(),
            relative_path: path.to_string(),
            mtime: 0,
        };
        state
            .config_db
            .put_document_artifacts(
                did.numeric,
                &stored_metadata,
                content,
                metadata.as_ref(),
            )
            .unwrap();
        {
            let mut writer = state.writer.lock().unwrap();
            state
                .search_index
                .add_document(
                    &writer,
                    &did.to_string(),
                    did.numeric,
                    collection,
                    path,
                    "Hello",
                    content,
                    0,
                )
                .unwrap();
            writer.commit().unwrap();
        }
        did
    }

    fn seed_document_embeddings(
        state: &AppState,
        did: DocumentId,
        chunk_indices: &[usize],
    ) {
        state
            .embedding_db
            .store(did.numeric, 1, 2, &[1.0, 2.0])
            .unwrap();
        for &chunk_index in chunk_indices {
            state
                .embedding_db
                .store(
                    chunk_doc_id(did.numeric, chunk_index),
                    1,
                    2,
                    &[3.0, 4.0],
                )
                .unwrap();
        }
    }

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

    #[tokio::test]
    async fn ingest_rejects_missing_collection_before_embedding() {
        let (_tmp, state) = test_state();

        let result = ingest(
            State(state),
            Json(IngestRequest {
                collection: "notes".to_string(),
                documents: vec![IngestDocument {
                    path: "hello.md".to_string(),
                    content: "# Hello\nBody".to_string(),
                    content_type: "text/markdown".to_string(),
                    metadata: None,
                }],
            }),
        )
        .await;

        match result {
            Err(ApiError::BadRequest(message)) => {
                assert!(message.contains("collection not found"));
            }
            Err(other) => panic!("expected bad request, got {other:?}"),
            Ok(_) => panic!("expected ingest to fail for a missing collection"),
        }
    }

    #[tokio::test]
    async fn ingest_rejects_unsupported_content_type_before_embedding() {
        let (_tmp, state) = test_state();
        state.config_db.set_managed_collection("notes").unwrap();

        let result = ingest(
            State(state),
            Json(IngestRequest {
                collection: "notes".to_string(),
                documents: vec![IngestDocument {
                    path: "hello.bin".to_string(),
                    content: "not markdown".to_string(),
                    content_type: "application/octet-stream".to_string(),
                    metadata: None,
                }],
            }),
        )
        .await;

        match result {
            Err(ApiError::BadRequest(message)) => {
                assert!(message.contains("unsupported content type"));
            }
            Err(other) => panic!("expected bad request, got {other:?}"),
            Ok(_) => panic!(
                "expected ingest to fail for an unsupported content type"
            ),
        }
    }

    #[tokio::test]
    async fn documents_route_delete_removes_atomic_artifacts() {
        let (_tmp, state) = test_state();
        let did = seed_document(
            &state,
            "notes",
            "hello.md",
            "# Hello\nBody",
            Some(serde_json::json!({ "topic": "rust" })),
        );
        seed_document_embeddings(&state, did.clone(), &[1]);

        let document = get(
            State(state.clone()),
            Path(("notes".to_string(), "hello.md".to_string())),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(document.doc_id, did.to_string());
        assert_eq!(document.content, "# Hello\nBody");
        assert_eq!(
            document.metadata,
            Some(serde_json::json!({ "topic": "rust" }))
        );

        let status = delete(
            State(state.clone()),
            Path(("notes".to_string(), "hello.md".to_string())),
        )
        .await
        .unwrap()
        .into_response()
        .status();
        assert_eq!(status, StatusCode::NO_CONTENT);

        assert!(
            state
                .config_db
                .get_document_metadata_typed(did.numeric)
                .unwrap()
                .is_none()
        );
        assert!(
            state
                .config_db
                .get_document_content(did.numeric)
                .unwrap()
                .is_none()
        );
        assert!(
            state
                .config_db
                .get_document_user_metadata(did.numeric)
                .unwrap()
                .is_none()
        );
        assert!(state.embedding_db.load(did.numeric).unwrap().is_none());
        assert!(
            state
                .embedding_db
                .load(chunk_doc_id(did.numeric, 1))
                .unwrap()
                .is_none()
        );

        match get(
            State(state),
            Path(("notes".to_string(), "hello.md".to_string())),
        )
        .await
        {
            Err(ApiError::NotFound(message)) => {
                assert!(message.contains("document not found"));
            }
            Err(other) => panic!("expected not found, got {other:?}"),
            Ok(_) => panic!("expected get to fail after document deletion"),
        }
    }
}
