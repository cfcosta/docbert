use axum::{Json, extract::State};
use docbert_core::{
    search::{self, SearchMode, SearchRequestCore},
    search_results,
};
use serde::{Deserialize, Serialize};

use crate::{error::ApiError, state::AppState};

#[derive(Deserialize)]
pub struct SearchRequest {
    query: String,
    #[serde(default = "default_mode")]
    mode: String,
    collection: Option<String>,
    #[serde(default = "default_count")]
    count: usize,
    #[serde(default)]
    min_score: f32,
}

fn default_mode() -> String {
    SearchMode::Semantic.as_str().to_string()
}

fn default_count() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    query: String,
    mode: String,
    result_count: usize,
    results: Vec<SearchResultItem>,
}

#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    rank: usize,
    score: f32,
    doc_id: String,
    collection: String,
    path: String,
    title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}

pub async fn search(
    State(state): State<AppState>,
    Json(body): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let mode = SearchMode::parse(&body.mode).ok_or_else(|| {
        ApiError::BadRequest(format!(
            "unknown search mode: {}; expected \"semantic\" or \"hybrid\"",
            body.mode
        ))
    })?;
    let request = SearchRequestCore {
        query: body.query.clone(),
        collection: body.collection.clone(),
        count: body.count,
        min_score: body.min_score,
    };
    let mut model = state.model.lock().map_err(ApiError::internal)?;
    let results = search::execute_search_mode(
        mode,
        &request,
        &state.search_index,
        &state.config_db,
        &state.embedding_db,
        &mut model,
    )?;

    let items = search_results::enrich(results, |doc_id| {
        load_user_metadata(&state, doc_id)
    })
    .into_iter()
    .map(|r| SearchResultItem {
        rank: r.rank,
        score: r.score,
        doc_id: r.doc_id,
        collection: r.collection,
        path: r.path,
        title: r.title,
        metadata: r.metadata,
    })
    .collect::<Vec<_>>();

    let result_count = items.len();
    Ok(Json(SearchResponse {
        query: body.query,
        mode: mode.as_str().to_string(),
        result_count,
        results: items,
    }))
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

    use docbert_core::{ConfigDb, EmbeddingDb, ModelManager, SearchIndex};

    use super::*;
    use crate::state::Inner;

    fn test_state() -> (tempfile::TempDir, AppState) {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let search_index = SearchIndex::open_in_ram().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let writer = search_index.writer(15_000_000).unwrap();
        (
            tmp,
            Arc::new(Inner {
                config_db,
                search_index,
                embedding_db,
                model: Mutex::new(ModelManager::new()),
                writer: Mutex::new(writer),
            }),
        )
    }

    #[test]
    fn default_mode_is_semantic() {
        assert_eq!(default_mode(), "semantic");
    }

    #[test]
    fn load_user_metadata_returns_stored_metadata() {
        let (_tmp, state) = test_state();
        state
            .config_db
            .set_document_user_metadata(
                7,
                &serde_json::json!({"topic": "rust"}),
            )
            .unwrap();

        assert_eq!(
            load_user_metadata(&state, 7),
            Some(serde_json::json!({"topic": "rust"}))
        );
    }

    #[tokio::test]
    async fn search_rejects_unknown_mode() {
        let (_tmp, state) = test_state();

        let error = search(
            State(state),
            Json(SearchRequest {
                query: "rust".to_string(),
                mode: "bm25".to_string(),
                collection: None,
                count: 10,
                min_score: 0.0,
            }),
        )
        .await
        .unwrap_err();

        match error {
            ApiError::BadRequest(message) => {
                assert!(message.contains("unknown search mode"));
            }
            other => panic!("expected bad request, got {other:?}"),
        }
    }
}
