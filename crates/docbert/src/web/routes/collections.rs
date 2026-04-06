use axum::{Json, extract::State, http::StatusCode};
use serde::{Deserialize, Serialize};

use crate::web::state::AppState;

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct CollectionItem {
    pub(crate) name: String,
}

pub(crate) async fn list(
    State(state): State<AppState>,
) -> Result<Json<Vec<CollectionItem>>, StatusCode> {
    let collections = state
        .config_db
        .list_collections()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let items = collections
        .into_iter()
        .map(|(name, _path)| CollectionItem { name })
        .collect();
    Ok(Json(items))
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
    use docbert_core::{ConfigDb, EmbeddingDb, ModelManager, SearchIndex};
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

    fn collections_router(state: AppState) -> Router {
        Router::new()
            .route(
                "/v1/collections",
                routing::get(list).fallback(|| async { StatusCode::NOT_FOUND }),
            )
            .with_state(state)
    }

    #[tokio::test]
    async fn web_collections_list_returns_cli_registered_filesystem_collections() {
        let (tmp, state) = test_state();
        let notes = tmp.path().join("notes");
        let docs = tmp.path().join("docs");
        std::fs::create_dir_all(&notes).unwrap();
        std::fs::create_dir_all(&docs).unwrap();
        state
            .config_db
            .set_collection("notes", notes.to_str().unwrap())
            .unwrap();
        state
            .config_db
            .set_collection("docs", docs.to_str().unwrap())
            .unwrap();

        let response = collections_router(state)
            .oneshot(
                Request::builder()
                    .uri("/v1/collections")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let mut items: Vec<CollectionItem> = serde_json::from_slice(&body).unwrap();
        items.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(
            items,
            vec![
                CollectionItem {
                    name: "docs".to_string(),
                },
                CollectionItem {
                    name: "notes".to_string(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn web_collections_post_returns_404() {
        let (_tmp, state) = test_state();

        let response = collections_router(state)
            .oneshot(
                Request::builder()
                    .uri("/v1/collections")
                    .method("POST")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn web_collections_delete_returns_404() {
        let (_tmp, state) = test_state();

        let response = collections_router(state)
            .oneshot(
                Request::builder()
                    .uri("/v1/collections/notes")
                    .method("DELETE")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
