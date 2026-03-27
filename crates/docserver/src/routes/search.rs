use axum::{Json, extract::State};
use docbert_core::search::{
    self,
    FinalResult,
    SearchParams,
    SemanticSearchParams,
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
    "semantic".to_string()
}

fn default_count() -> usize {
    10
}

#[derive(Serialize)]
pub struct SearchResponse {
    query: String,
    mode: String,
    result_count: usize,
    results: Vec<SearchResultItem>,
}

#[derive(Serialize)]
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
    let results = match body.mode.as_str() {
        "semantic" => {
            let params = SemanticSearchParams {
                query: body.query.clone(),
                collection: body.collection.clone(),
                count: body.count,
                min_score: body.min_score,
                all: false,
            };
            let mut model =
                state.model.lock().map_err(|e| ApiError::internal(e))?;
            search::execute_semantic_search(
                &params,
                &state.config_db,
                &state.embedding_db,
                &mut model,
            )?
        }
        "hybrid" => {
            let params = SearchParams {
                query: body.query.clone(),
                count: body.count,
                collection: body.collection.clone(),
                min_score: body.min_score,
                bm25_only: false,
                no_fuzzy: false,
                all: false,
            };
            let mut model =
                state.model.lock().map_err(|e| ApiError::internal(e))?;
            search::execute_search(
                &params,
                &state.search_index,
                &state.embedding_db,
                &mut model,
            )?
        }
        other => {
            return Err(ApiError::BadRequest(format!(
                "unknown search mode: {other}; expected \"semantic\" or \"hybrid\""
            )));
        }
    };

    let items = results
        .into_iter()
        .map(|r| to_result_item(&state, r))
        .collect::<Vec<_>>();

    let result_count = items.len();
    Ok(Json(SearchResponse {
        query: body.query,
        mode: body.mode,
        result_count,
        results: items,
    }))
}

fn to_result_item(state: &AppState, r: FinalResult) -> SearchResultItem {
    let metadata = load_user_metadata(state, r.doc_num_id);
    SearchResultItem {
        rank: r.rank,
        score: r.score,
        doc_id: r.doc_id,
        collection: r.collection,
        path: r.path,
        title: r.title,
        metadata,
    }
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
