use std::{path::Path, time::SystemTime};

use axum::{
    Json,
    extract::{Path as AxumPath, State},
    http::StatusCode,
};
use docbert_core::{
    DocumentId,
    chunking,
    embedding,
    ingestion,
    preparation::{self, SearchDocument},
};
use serde::{Deserialize, Serialize};

use crate::web::{
    ingest::{self, EmbeddingEntry},
    paths,
    state::AppState,
};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct DocumentListItem {
    pub(crate) doc_id: String,
    pub(crate) path: String,
    pub(crate) title: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct DocumentResponse {
    pub(crate) doc_id: String,
    pub(crate) collection: String,
    pub(crate) path: String,
    pub(crate) title: String,
    pub(crate) content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct IngestRequest {
    pub(crate) collection: String,
    pub(crate) documents: Vec<IngestDocument>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct IngestDocument {
    pub(crate) path: String,
    pub(crate) content: String,
    pub(crate) content_type: String,
    #[serde(default)]
    pub(crate) metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct IngestResponse {
    pub(crate) ingested: usize,
    pub(crate) documents: Vec<IngestedDoc>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct IngestedDoc {
    pub(crate) doc_id: String,
    pub(crate) path: String,
    pub(crate) title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) metadata: Option<serde_json::Value>,
}

fn map_error(err: docbert_core::Error) -> StatusCode {
    match err {
        docbert_core::Error::NotFound { .. } => StatusCode::NOT_FOUND,
        docbert_core::Error::Config(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

const TEST_FAKE_EMBEDDINGS_ENV: &str = "DOCBERT_WEB_TEST_FAKE_EMBEDDINGS";

fn title_from_disk(relative_path: &str, content: &str) -> String {
    ingestion::extract_title(content, Path::new(relative_path))
}

fn upload_chunking_config() -> chunking::ChunkingConfig {
    chunking::ChunkingConfig {
        chunk_size: chunking::DEFAULT_CHUNK_SIZE,
        overlap: chunking::DEFAULT_CHUNK_OVERLAP,
        document_length: None,
    }
}

fn document_mtime(full_path: &Path) -> std::io::Result<u64> {
    Ok(std::fs::metadata(full_path)?
        .modified()
        .unwrap_or(SystemTime::UNIX_EPOCH)
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs())
}

fn compute_embedding_entries(
    state: &AppState,
    document: &SearchDocument,
) -> Result<Vec<EmbeddingEntry>, StatusCode> {
    let docs_to_embed =
        preparation::embedding_chunks(document, upload_chunking_config());
    if docs_to_embed.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    if std::env::var_os(TEST_FAKE_EMBEDDINGS_ENV).is_some() {
        return Ok(docs_to_embed
            .into_iter()
            .map(|(doc_id, _)| (doc_id, 1, 2, vec![1.0, 0.0]))
            .collect());
    }

    let mut model = state
        .model
        .lock()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    embedding::embed_documents(&mut model, docs_to_embed)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

pub(crate) async fn ingest(
    State(state): State<AppState>,
    Json(body): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, StatusCode> {
    {
        let config_db = state.open_config_db_blocking().map_err(map_error)?;
        paths::resolve_collection_root(&config_db, &body.collection)
            .map_err(map_error)?;
    }

    let mut ingested = Vec::with_capacity(body.documents.len());
    for uploaded in &body.documents {
        if uploaded.content_type != "text/markdown" {
            return Err(StatusCode::BAD_REQUEST);
        }

        let full_path = {
            let config_db = state.open_config_db_blocking().map_err(map_error)?;
            paths::resolve_document_path(
                &config_db,
                &body.collection,
                &uploaded.path,
            )
            .map_err(map_error)?
        };
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }
        std::fs::write(&full_path, &uploaded.content)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let mtime = document_mtime(&full_path)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let document = ingest::load_markdown_document(
            &body.collection,
            &uploaded.path,
            &full_path,
            uploaded.metadata.clone(),
            mtime,
        )
        .map_err(map_error)?;
        let embedding_entries = compute_embedding_entries(&state, &document)?;
        let result = ingest::ingest_prepared_document(
            &state,
            &body.collection,
            &document,
            &embedding_entries,
        )
        .map_err(map_error)?;
        ingested.push(IngestedDoc {
            doc_id: result.doc_id,
            path: result.path,
            title: result.title,
            metadata: result.metadata,
        });
    }

    Ok(Json(IngestResponse {
        ingested: ingested.len(),
        documents: ingested,
    }))
}

pub(crate) async fn list_by_collection(
    State(state): State<AppState>,
    AxumPath(collection): AxumPath<String>,
) -> Result<Json<Vec<DocumentListItem>>, StatusCode> {
    let config_db = state.open_config_db().map_err(map_error)?;
    paths::resolve_collection_root(&config_db, &collection).map_err(map_error)?;

    let all_meta = config_db
        .list_all_document_metadata_typed()
        .map_err(map_error)?;
    let mut items = Vec::new();
    for (doc_id, meta) in &all_meta {
        if meta.collection != collection {
            continue;
        }

        let full_path = paths::resolve_document_path(
            &config_db,
            &meta.collection,
            &meta.relative_path,
        )
        .map_err(map_error)?;
        let content = std::fs::read_to_string(&full_path)
            .map_err(|_| StatusCode::NOT_FOUND)?;
        items.push(DocumentListItem {
            doc_id: docbert_core::search::short_doc_id(*doc_id),
            path: meta.relative_path.clone(),
            title: title_from_disk(&meta.relative_path, &content),
        });
    }
    items.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(Json(items))
}

pub(crate) async fn delete(
    State(state): State<AppState>,
    AxumPath((collection, path)): AxumPath<(String, String)>,
) -> Result<StatusCode, StatusCode> {
    let did = DocumentId::new(&collection, &path);
    let full_path = {
        let config_db = state.open_config_db_blocking().map_err(map_error)?;
        config_db
            .get_document_metadata_typed(did.numeric)
            .map_err(map_error)?
            .ok_or(StatusCode::NOT_FOUND)?;

        paths::resolve_document_path(&config_db, &collection, &path)
            .map_err(map_error)?
    };
    std::fs::remove_file(&full_path).map_err(|_| StatusCode::NOT_FOUND)?;
    ingest::delete_document(&state, &collection, &path).map_err(map_error)?;

    Ok(StatusCode::NO_CONTENT)
}

pub(crate) async fn get(
    State(state): State<AppState>,
    AxumPath((collection, path)): AxumPath<(String, String)>,
) -> Result<Json<DocumentResponse>, StatusCode> {
    let config_db = state.open_config_db().map_err(map_error)?;
    let did = DocumentId::new(&collection, &path);
    config_db
        .get_document_metadata_typed(did.numeric)
        .map_err(map_error)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let full_path =
        paths::resolve_document_path(&config_db, &collection, &path)
            .map_err(map_error)?;
    let content = std::fs::read_to_string(&full_path)
        .map_err(|_| StatusCode::NOT_FOUND)?;
    let metadata = config_db
        .get_document_user_metadata(did.numeric)
        .map_err(map_error)?;

    Ok(Json(DocumentResponse {
        doc_id: did.to_string(),
        collection,
        path: path.clone(),
        title: title_from_disk(&path, &content),
        content,
        metadata,
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, OnceLock};

    use axum::{
        Router,
        body::{Body, to_bytes},
        http::{Request, StatusCode},
        routing,
    };
    use docbert_core::{ConfigDb, ModelManager, SearchIndex, incremental};
    use tower::util::ServiceExt;

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
        docbert_core::EmbeddingDb::open(&state.data_dir.embeddings_db()).unwrap()
    }

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn documents_router(state: AppState) -> Router {
        Router::new()
            .route("/v1/documents", routing::post(ingest))
            .route(
                "/v1/collections/{name}/documents",
                routing::get(list_by_collection),
            )
            .route(
                "/v1/documents/{collection}/{*path}",
                routing::get(get).delete(delete),
            )
            .with_state(state)
    }

    fn seed_filesystem_document(
        state: &AppState,
        root: &Path,
        collection: &str,
        relative_path: &str,
        content: &str,
    ) -> DocumentId {
        std::fs::create_dir_all(
            root.join(
                Path::new(relative_path)
                    .parent()
                    .unwrap_or_else(|| Path::new("")),
            ),
        )
        .unwrap();
        std::fs::write(root.join(relative_path), content).unwrap();
        test_config_db(&state)
            .set_collection(collection, root.to_str().unwrap())
            .unwrap();
        let did = DocumentId::new(collection, relative_path);
        test_config_db(&state)
            .set_document_metadata_typed(
                did.numeric,
                &incremental::DocumentMetadata {
                    collection: collection.to_string(),
                    relative_path: relative_path.to_string(),
                    mtime: 1,
                },
            )
            .unwrap();
        did
    }

    #[tokio::test]
    async fn web_documents_upload_writes_file_to_collection_folder() {
        let _guard = env_lock();
        unsafe {
            std::env::set_var(TEST_FAKE_EMBEDDINGS_ENV, "1");
        }
        let (tmp, state) = test_state();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        test_config_db(&state)
            .set_collection("notes", root.to_str().unwrap())
            .unwrap();

        let response = documents_router(state.clone())
            .oneshot(
                Request::builder()
                    .uri("/v1/documents")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r##"{"collection":"notes","documents":[{"path":"hello.md","content":"# Uploaded\n\nBody","content_type":"text/markdown"}]}"##,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        unsafe {
            std::env::remove_var(TEST_FAKE_EMBEDDINGS_ENV);
        }
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            std::fs::read_to_string(root.join("hello.md")).unwrap(),
            "# Uploaded\n\nBody"
        );
        let snapshot = test_config_db(&state)
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after upload");
        assert_eq!(snapshot.files.len(), 1);
        assert_eq!(snapshot.files[0].relative_path, "hello.md");
    }

    #[tokio::test]
    async fn web_documents_upload_preserves_nested_paths() {
        let _guard = env_lock();
        unsafe {
            std::env::set_var(TEST_FAKE_EMBEDDINGS_ENV, "1");
        }
        let (tmp, state) = test_state();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        test_config_db(&state)
            .set_collection("notes", root.to_str().unwrap())
            .unwrap();

        let response = documents_router(state)
            .oneshot(
                Request::builder()
                    .uri("/v1/documents")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r##"{"collection":"notes","documents":[{"path":"nested/deep/hello.md","content":"# Nested\n\nBody","content_type":"text/markdown"}]}"##,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        unsafe {
            std::env::remove_var(TEST_FAKE_EMBEDDINGS_ENV);
        }
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            std::fs::read_to_string(root.join("nested/deep/hello.md")).unwrap(),
            "# Nested\n\nBody"
        );
    }

    #[tokio::test]
    async fn web_documents_upload_overwrite_replaces_contents() {
        let _guard = env_lock();
        unsafe {
            std::env::set_var(TEST_FAKE_EMBEDDINGS_ENV, "1");
        }
        let (tmp, state) = test_state();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("hello.md"), "old").unwrap();
        test_config_db(&state)
            .set_collection("notes", root.to_str().unwrap())
            .unwrap();
        let previous_snapshot = test_config_db(&state)
            .get_collection_merkle_snapshot("notes")
            .unwrap();

        let response = documents_router(state.clone())
            .oneshot(
                Request::builder()
                    .uri("/v1/documents")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r##"{"collection":"notes","documents":[{"path":"hello.md","content":"# Updated\n\nBody v2","content_type":"text/markdown"}]}"##,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        unsafe {
            std::env::remove_var(TEST_FAKE_EMBEDDINGS_ENV);
        }
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            std::fs::read_to_string(root.join("hello.md")).unwrap(),
            "# Updated\n\nBody v2"
        );
        let updated_snapshot = test_config_db(&state)
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after overwrite upload");
        if let Some(previous_snapshot) = previous_snapshot {
            assert_ne!(previous_snapshot.root_hash, updated_snapshot.root_hash);
        }
    }

    #[tokio::test]
    async fn web_documents_delete_removes_source_file_and_metadata() {
        let _guard = env_lock();
        unsafe {
            std::env::set_var(TEST_FAKE_EMBEDDINGS_ENV, "1");
        }
        let (tmp, state) = test_state();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        test_config_db(&state)
            .set_collection("notes", root.to_str().unwrap())
            .unwrap();

        let upload_response = documents_router(state.clone())
            .oneshot(
                Request::builder()
                    .uri("/v1/documents")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r##"{"collection":"notes","documents":[{"path":"hello.md","content":"# Uploaded\n\nBody","content_type":"text/markdown","metadata":{"topic":"rust"}}]}"##,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(upload_response.status(), StatusCode::OK);

        let did = DocumentId::new("notes", "hello.md");
        let previous_snapshot = test_config_db(&state)
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after upload");
        let response = documents_router(state.clone())
            .oneshot(
                Request::builder()
                    .uri("/v1/documents/notes/hello.md")
                    .method("DELETE")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        unsafe {
            std::env::remove_var(TEST_FAKE_EMBEDDINGS_ENV);
        }
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert!(!root.join("hello.md").exists());
        assert!(
            test_config_db(&state)
                .get_document_metadata_typed(did.numeric)
                .unwrap()
                .is_none()
        );
        assert!(
            test_config_db(&state)
                .get_document_user_metadata(did.numeric)
                .unwrap()
                .is_none()
        );
        let updated_snapshot = test_config_db(&state)
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after delete");
        assert!(updated_snapshot.files.is_empty());
        assert_ne!(previous_snapshot.root_hash, updated_snapshot.root_hash);
    }

    #[tokio::test]
    async fn web_documents_delete_removes_tantivy_entry_and_embeddings() {
        let _guard = env_lock();
        unsafe {
            std::env::set_var(TEST_FAKE_EMBEDDINGS_ENV, "1");
        }
        let (tmp, state) = test_state();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        test_config_db(&state)
            .set_collection("notes", root.to_str().unwrap())
            .unwrap();

        let upload_response = documents_router(state.clone())
            .oneshot(
                Request::builder()
                    .uri("/v1/documents")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r##"{"collection":"notes","documents":[{"path":"hello.md","content":"# Uploaded\n\nBody","content_type":"text/markdown"}]}"##,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(upload_response.status(), StatusCode::OK);

        let did = DocumentId::new("notes", "hello.md");
        let previous_snapshot = test_config_db(&state)
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after upload");
        let response = documents_router(state.clone())
            .oneshot(
                Request::builder()
                    .uri("/v1/documents/notes/hello.md")
                    .method("DELETE")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        unsafe {
            std::env::remove_var(TEST_FAKE_EMBEDDINGS_ENV);
        }
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert!(
            state
                .search_index
                .find_by_collection_path("notes", "hello.md")
                .unwrap()
                .is_none()
        );
        assert!(test_embedding_db(&state).load(did.numeric).unwrap().is_none());
        let updated_snapshot = test_config_db(&state)
            .get_collection_merkle_snapshot("notes")
            .unwrap()
            .expect("snapshot should exist after delete");
        assert!(updated_snapshot.files.is_empty());
        assert_ne!(previous_snapshot.root_hash, updated_snapshot.root_hash);
    }

    #[tokio::test]
    async fn web_documents_delete_returns_not_found_for_missing() {
        let (tmp, state) = test_state();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        test_config_db(&state)
            .set_collection("notes", root.to_str().unwrap())
            .unwrap();

        let response = documents_router(state)
            .oneshot(
                Request::builder()
                    .uri("/v1/documents/notes/missing.md")
                    .method("DELETE")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn web_documents_get_lists_collection_documents_with_titles_from_disk()
     {
        let (tmp, state) = test_state();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        let did = seed_filesystem_document(
            &state,
            &root,
            "notes",
            "nested/hello.md",
            "# Disk Title\n\nBody",
        );
        let mut writer = state.open_index_writer_blocking(15_000_000).unwrap();
        state
            .search_index
            .add_document(
                &writer,
                &did.to_string(),
                did.numeric,
                "notes",
                "nested/hello.md",
                "Index Title",
                "index body",
                1,
            )
            .unwrap();
        writer.commit().unwrap();
        drop(writer);

        let response = documents_router(state)
            .oneshot(
                Request::builder()
                    .uri("/v1/collections/notes/documents")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let items: Vec<DocumentListItem> =
            serde_json::from_slice(&body).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].path, "nested/hello.md");
        assert_eq!(items[0].title, "Disk Title");
    }

    #[tokio::test]
    async fn web_documents_get_returns_document_content_and_title_from_disk() {
        let (tmp, state) = test_state();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        let did = seed_filesystem_document(
            &state,
            &root,
            "notes",
            "hello.md",
            "# Disk Title\n\nDisk body",
        );
        test_config_db(&state)
            .set_document_user_metadata(
                did.numeric,
                &serde_json::json!({ "topic": "rust" }),
            )
            .unwrap();

        let response = documents_router(state)
            .oneshot(
                Request::builder()
                    .uri("/v1/documents/notes/hello.md")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let item: DocumentResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(item.title, "Disk Title");
        assert_eq!(item.content, "# Disk Title\n\nDisk body");
        assert_eq!(item.metadata, Some(serde_json::json!({ "topic": "rust" })));
    }
}
