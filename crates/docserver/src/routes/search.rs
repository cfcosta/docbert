use axum::{Json, extract::State};
use docbert_core::{
    search::{self, SearchMode, SearchQuery},
    text_util,
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

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct SearchExcerpt {
    text: String,
    start_line: usize,
    end_line: usize,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    excerpts: Vec<SearchExcerpt>,
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
    let request = SearchQuery {
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
    drop(model);

    let items = results
        .into_iter()
        .map(|result| build_search_result_item(&state, result, &body.query))
        .collect::<Vec<_>>();

    let result_count = items.len();
    Ok(Json(SearchResponse {
        query: body.query,
        mode: mode.as_str().to_string(),
        result_count,
        results: items,
    }))
}

fn build_search_result_item(
    state: &AppState,
    result: search::FinalResult,
    query: &str,
) -> SearchResultItem {
    let metadata = load_user_metadata(state, result.doc_num_id);
    let excerpts = load_excerpts(state, result.doc_num_id, query);

    SearchResultItem {
        rank: result.rank,
        score: result.score,
        doc_id: result.doc_id,
        collection: result.collection,
        path: result.path,
        title: result.title,
        metadata,
        excerpts,
    }
}

fn load_excerpts(
    state: &AppState,
    doc_numeric_id: u64,
    query: &str,
) -> Vec<SearchExcerpt> {
    state
        .config_db
        .get_document_content(doc_numeric_id)
        .ok()
        .flatten()
        .map(|content| {
            text_util::extract_excerpts(&content, query, 3)
                .into_iter()
                .map(|excerpt| SearchExcerpt {
                    text: excerpt.text,
                    start_line: excerpt.start_line,
                    end_line: excerpt.end_line,
                })
                .collect()
        })
        .unwrap_or_default()
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
        DocumentId,
        EmbeddingDb,
        ModelManager,
        SearchIndex,
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

    fn seed_stored_document(
        state: &AppState,
        collection: &str,
        path: &str,
        content: &str,
        user_metadata: Option<serde_json::Value>,
    ) -> DocumentId {
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
                content,
                user_metadata.as_ref(),
            )
            .unwrap();
        did
    }

    fn final_result(did: &DocumentId, title: &str) -> search::FinalResult {
        search::FinalResult {
            rank: 1,
            score: 0.95,
            doc_id: did.short.clone(),
            doc_num_id: did.numeric,
            collection: "notes".to_string(),
            path: "rust.md".to_string(),
            title: title.to_string(),
        }
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

    #[test]
    fn build_search_result_item_preserves_metadata_and_adds_excerpts() {
        let (_tmp, state) = test_state();
        let did = seed_stored_document(
            &state,
            "notes",
            "rust.md",
            "line1\nRust ownership\nline3\nline4",
            Some(serde_json::json!({"topic": "rust"})),
        );

        let item = build_search_result_item(
            &state,
            final_result(&did, "Rust"),
            "ownership",
        );

        assert_eq!(item.metadata, Some(serde_json::json!({"topic": "rust"})));
        assert_eq!(item.excerpts.len(), 1);
        assert_eq!(
            item.excerpts[0],
            SearchExcerpt {
                text: "line1\nRust ownership\nline3\nline4".to_string(),
                start_line: 1,
                end_line: 4,
            }
        );
    }

    #[test]
    fn build_search_result_item_uses_first_lines_when_query_has_no_literal_match()
     {
        let (_tmp, state) = test_state();
        let did = seed_stored_document(
            &state,
            "notes",
            "rust.md",
            "line1\nline2\nline3\nline4\nline5\nline6\nline7",
            None,
        );

        let item = build_search_result_item(
            &state,
            final_result(&did, "Semantic result"),
            "memory management",
        );

        assert_eq!(item.excerpts.len(), 1);
        assert_eq!(
            item.excerpts[0],
            SearchExcerpt {
                text: "line1\nline2\nline3\nline4\nline5\nline6".to_string(),
                start_line: 1,
                end_line: 6,
            }
        );
    }

    #[test]
    fn build_search_result_item_caps_excerpt_count_at_three() {
        let (_tmp, state) = test_state();
        let did = seed_stored_document(
            &state,
            "notes",
            "rust.md",
            &[
                "ownership one",
                "line2",
                "line3",
                "line4",
                "line5",
                "line6",
                "line7",
                "ownership two",
                "line9",
                "line10",
                "line11",
                "line12",
                "line13",
                "line14",
                "ownership three",
                "line16",
                "line17",
                "line18",
                "line19",
                "line20",
                "line21",
                "ownership four",
            ]
            .join("\n"),
            None,
        );

        let item = build_search_result_item(
            &state,
            final_result(&did, "Rust"),
            "ownership",
        );

        assert_eq!(item.excerpts.len(), 3);
        assert!(
            !item
                .excerpts
                .iter()
                .any(|excerpt| excerpt.text.contains("ownership four"))
        );
    }

    #[test]
    fn search_result_item_serialization_keeps_existing_fields_and_adds_excerpts()
     {
        let (_tmp, state) = test_state();
        let did = seed_stored_document(
            &state,
            "notes",
            "rust.md",
            "line1\nRust ownership\nline3",
            Some(serde_json::json!({"topic": "rust"})),
        );

        let value = serde_json::to_value(build_search_result_item(
            &state,
            final_result(&did, "Rust"),
            "ownership",
        ))
        .unwrap();

        assert_eq!(value.get("rank").and_then(|v| v.as_u64()), Some(1));
        assert_eq!(
            value.get("doc_id").and_then(|v| v.as_str()),
            Some(did.short.as_str())
        );
        assert_eq!(
            value.get("collection").and_then(|v| v.as_str()),
            Some("notes")
        );
        assert_eq!(value.get("path").and_then(|v| v.as_str()), Some("rust.md"));
        assert_eq!(value.get("title").and_then(|v| v.as_str()), Some("Rust"));
        assert_eq!(
            value.get("metadata"),
            Some(&serde_json::json!({"topic": "rust"}))
        );
        let excerpts = value
            .get("excerpts")
            .and_then(|v| v.as_array())
            .expect("excerpts array");
        assert_eq!(excerpts.len(), 1);
        assert_eq!(
            excerpts[0],
            serde_json::json!({
                "text": "line1\nRust ownership\nline3",
                "start_line": 1,
                "end_line": 3,
            })
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
