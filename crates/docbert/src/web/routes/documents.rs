use std::path::Path;

use axum::{
    Json,
    extract::{Path as AxumPath, State},
    http::StatusCode,
};
use docbert_core::{DocumentId, ingestion};
use serde::{Deserialize, Serialize};

use crate::web::{paths, state::AppState};

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

fn map_error(err: docbert_core::Error) -> StatusCode {
    match err {
        docbert_core::Error::NotFound { .. } => StatusCode::NOT_FOUND,
        docbert_core::Error::Config(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn title_from_disk(relative_path: &str, content: &str) -> String {
    ingestion::extract_title(content, Path::new(relative_path))
}

pub(crate) async fn list_by_collection(
    State(state): State<AppState>,
    AxumPath(collection): AxumPath<String>,
) -> Result<Json<Vec<DocumentListItem>>, StatusCode> {
    paths::resolve_collection_root(&state.config_db, &collection)
        .map_err(map_error)?;

    let all_meta = state
        .config_db
        .list_all_document_metadata_typed()
        .map_err(map_error)?;
    let mut items = Vec::new();
    for (doc_id, meta) in &all_meta {
        if meta.collection != collection {
            continue;
        }

        let full_path = paths::resolve_document_path(
            &state.config_db,
            &meta.collection,
            &meta.relative_path,
        )
        .map_err(map_error)?;
        let content = std::fs::read_to_string(&full_path).map_err(|_| StatusCode::NOT_FOUND)?;
        items.push(DocumentListItem {
            doc_id: docbert_core::search::short_doc_id(*doc_id),
            path: meta.relative_path.clone(),
            title: title_from_disk(&meta.relative_path, &content),
        });
    }
    items.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(Json(items))
}

pub(crate) async fn get(
    State(state): State<AppState>,
    AxumPath((collection, path)): AxumPath<(String, String)>,
) -> Result<Json<DocumentResponse>, StatusCode> {
    let did = DocumentId::new(&collection, &path);
    state
        .config_db
        .get_document_metadata_typed(did.numeric)
        .map_err(map_error)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let full_path = paths::resolve_document_path(&state.config_db, &collection, &path)
        .map_err(map_error)?;
    let content = std::fs::read_to_string(&full_path).map_err(|_| StatusCode::NOT_FOUND)?;
    let metadata = state
        .config_db
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
    use std::sync::{Arc, Mutex};

    use axum::{
        Router,
        body::{Body, to_bytes},
        http::{Request, StatusCode},
        routing,
    };
    use docbert_core::{
        ConfigDb, EmbeddingDb, ModelManager, SearchIndex, incremental,
    };
    use tower::util::ServiceExt;

    use super::*;
    use crate::web::state::Inner;

    fn test_state() -> (tempfile::TempDir, AppState) {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let search_index = SearchIndex::open_in_ram().unwrap();
        let embedding_db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
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

    fn documents_router(state: AppState) -> Router {
        Router::new()
            .route(
                "/v1/collections/{name}/documents",
                routing::get(list_by_collection),
            )
            .route(
                "/v1/documents/{collection}/{*path}",
                routing::get(get),
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
        std::fs::create_dir_all(root.join(Path::new(relative_path).parent().unwrap_or_else(|| Path::new("")))).unwrap();
        std::fs::write(root.join(relative_path), content).unwrap();
        state
            .config_db
            .set_collection(collection, root.to_str().unwrap())
            .unwrap();
        let did = DocumentId::new(collection, relative_path);
        state
            .config_db
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
    async fn web_documents_get_lists_collection_documents_with_titles_from_disk() {
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
        let mut writer = state.writer.lock().unwrap();
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
        let items: Vec<DocumentListItem> = serde_json::from_slice(&body).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].path, "nested/hello.md");
        assert_eq!(items[0].title, "Disk Title");
    }

    #[tokio::test]
    async fn web_documents_get_returns_document_content_and_title_from_disk_not_config() {
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
        state
            .config_db
            .set_document_content(did.numeric, "Stored content")
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
    }
}
