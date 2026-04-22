use std::path::Path;

use axum::{Json, extract::State, http::StatusCode};
use docbert_core::{
    search::{self, SearchMode, SearchQuery},
    text,
};
use serde::{Deserialize, Serialize};

use crate::web::{paths, routes::log_internal_error, state::AppState};

#[derive(Debug, Deserialize)]
pub(crate) struct SearchRequest {
    pub(crate) query: String,
    #[serde(default = "default_mode")]
    pub(crate) mode: String,
    pub(crate) collection: Option<String>,
    #[serde(default = "default_count")]
    pub(crate) count: usize,
    #[serde(default)]
    pub(crate) min_score: f32,
}

fn default_mode() -> String {
    SearchMode::Semantic.as_str().to_string()
}

fn default_count() -> usize {
    10
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct SearchResponse {
    pub(crate) query: String,
    pub(crate) mode: String,
    pub(crate) result_count: usize,
    pub(crate) results: Vec<SearchResultItem>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SearchExcerpt {
    pub(crate) text: String,
    pub(crate) start_line: usize,
    pub(crate) end_line: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct SearchResultItem {
    pub(crate) rank: usize,
    pub(crate) score: f32,
    pub(crate) doc_id: String,
    pub(crate) collection: String,
    pub(crate) path: String,
    pub(crate) title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) metadata: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) excerpts: Vec<SearchExcerpt>,
}

pub(crate) async fn search(
    State(state): State<AppState>,
    Json(body): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, StatusCode> {
    let mode = SearchMode::parse(&body.mode).ok_or(StatusCode::BAD_REQUEST)?;
    let request = SearchQuery {
        query: body.query.clone(),
        collection: body.collection.clone(),
        count: body.count,
        min_score: body.min_score,
    };

    let config_db = state.open_config_db().map_err(|err| {
        log_internal_error(err, "search::search open config db")
    })?;
    // Recover from a poisoned model mutex instead of surfacing 500s
    // forever. A poisoned lock means a prior request panicked while
    // holding the mutex; the ModelManager's state is still intact
    // enough for subsequent calls, and leaving it wedged would be
    // worse than the bug that caused the original panic.
    let mut model = state.model.lock().unwrap_or_else(|poisoned| {
        tracing::warn!(
            "search recovered from poisoned model mutex (a prior search panicked)"
        );
        poisoned.into_inner()
    });
    let mut results = search::by_mode(
        mode,
        &request,
        &state.search_index,
        &config_db,
        &state.data_dir,
        &mut model,
    )
    .map_err(|err| match err {
        // "PLAID index not built yet" is a caller-actionable state, not
        // a server fault: surface it as 503 so clients can distinguish
        // from a real internal error.
        docbert_core::Error::PlaidIndexMissing => {
            tracing::info!(
                query = %body.query,
                mode = %body.mode,
                "search rejected: PLAID index missing (run `docbert sync`)"
            );
            StatusCode::SERVICE_UNAVAILABLE
        }
        other => {
            tracing::error!(
                error = %other,
                ?other,
                query = %body.query,
                mode = %body.mode,
                collection = ?body.collection,
                "search::search failed",
            );
            StatusCode::INTERNAL_SERVER_ERROR
        }
    })?;
    drop(model);

    search::disambiguate_doc_ids(&mut results, &config_db);

    let items: Vec<SearchResultItem> = results
        .into_iter()
        .map(|result| {
            build_search_result_item(&state, &config_db, result, &body.query)
        })
        .collect();

    Ok(Json(SearchResponse {
        query: body.query,
        mode: mode.as_str().to_string(),
        result_count: items.len(),
        results: items,
    }))
}

fn build_search_result_item(
    _state: &AppState,
    config_db: &docbert_core::ConfigDb,
    result: search::FinalResult,
    query: &str,
) -> SearchResultItem {
    let metadata = load_user_metadata(config_db, result.doc_num_id);
    let (title, excerpts) = load_title_and_excerpts(
        config_db,
        &result.collection,
        &result.path,
        query,
        &result.title,
    );

    SearchResultItem {
        rank: result.rank,
        score: result.score,
        doc_id: result.doc_id,
        collection: result.collection,
        path: result.path,
        title,
        metadata,
        excerpts,
    }
}

fn load_title_and_excerpts(
    config_db: &docbert_core::ConfigDb,
    collection: &str,
    path: &str,
    query: &str,
    fallback_title: &str,
) -> (String, Vec<SearchExcerpt>) {
    let Ok(full_path) =
        paths::resolve_document_path(config_db, collection, path)
    else {
        return (fallback_title.to_string(), Vec::new());
    };
    let Ok(content) = docbert_core::preparation::load_preview_content(
        Path::new(path),
        &full_path,
    ) else {
        return (fallback_title.to_string(), Vec::new());
    };

    let title =
        docbert_core::ingestion::extract_title(&content, Path::new(path));
    let excerpts = text::extract_excerpts(&content, query, 3)
        .into_iter()
        .map(|excerpt| SearchExcerpt {
            text: excerpt.text,
            start_line: excerpt.start_line,
            end_line: excerpt.end_line,
        })
        .collect();

    (title, excerpts)
}

fn load_user_metadata(
    config_db: &docbert_core::ConfigDb,
    doc_numeric_id: u64,
) -> Option<serde_json::Value> {
    config_db
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
        ModelManager,
        SearchIndex,
        incremental,
    };

    use super::*;
    use crate::web::state::Inner;

    fn test_state() -> (tempfile::TempDir, AppState) {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = docbert_core::DataDir::new(tmp.path());
        (
            tmp,
            Arc::new(Inner {
                data_dir,
                search_index: SearchIndex::open_in_ram().unwrap(),
                model: Mutex::new(ModelManager::new()),
            }),
        )
    }

    fn seed_filesystem_document(
        state: &AppState,
        collection: &str,
        path: &str,
        content: &str,
        user_metadata: Option<serde_json::Value>,
    ) -> DocumentId {
        let root = tempfile::tempdir().unwrap();
        let collection_root = root.keep();
        std::fs::create_dir_all(
            collection_root.join(
                Path::new(path).parent().unwrap_or_else(|| Path::new("")),
            ),
        )
        .unwrap();
        std::fs::write(collection_root.join(path), content).unwrap();
        ConfigDb::open(&state.data_dir.config_db())
            .unwrap()
            .set_collection(collection, collection_root.to_str().unwrap())
            .unwrap();
        let did = DocumentId::new(collection, path);
        ConfigDb::open(&state.data_dir.config_db())
            .unwrap()
            .set_document_metadata_typed(
                did.numeric,
                &incremental::DocumentMetadata {
                    collection: collection.to_string(),
                    relative_path: path.to_string(),
                    mtime: 1,
                },
            )
            .unwrap();
        if let Some(metadata) = user_metadata.as_ref() {
            ConfigDb::open(&state.data_dir.config_db())
                .unwrap()
                .set_document_user_metadata(did.numeric, metadata)
                .unwrap();
        }
        did
    }

    fn final_result(
        did: &DocumentId,
        title: &str,
        path: &str,
    ) -> search::FinalResult {
        search::FinalResult {
            rank: 1,
            score: 0.95,
            doc_id: did.short.clone(),
            doc_num_id: did.numeric,
            collection: "notes".to_string(),
            path: path.to_string(),
            title: title.to_string(),
        }
    }

    #[test]
    fn web_search_default_mode_is_semantic() {
        assert_eq!(default_mode(), "semantic");
    }

    #[test]
    fn web_search_result_item_reads_title_and_excerpts_from_disk() {
        let (_tmp, state) = test_state();
        let did = seed_filesystem_document(
            &state,
            "notes",
            "rust.md",
            "line1\n# Disk Rust\nRust ownership\nline4",
            Some(serde_json::json!({"topic": "rust"})),
        );

        let config_db = state.open_config_db().unwrap();
        let item = build_search_result_item(
            &state,
            &config_db,
            final_result(&did, "Index Rust", "rust.md"),
            "ownership",
        );

        assert_eq!(item.title, "Disk Rust");
        assert_eq!(item.metadata, Some(serde_json::json!({"topic": "rust"})));
        assert_eq!(item.excerpts.len(), 1);
        assert_eq!(
            item.excerpts[0],
            SearchExcerpt {
                text: "line1\n# Disk Rust\nRust ownership\nline4".to_string(),
                start_line: 1,
                end_line: 4,
            }
        );
    }

    #[test]
    fn web_search_result_item_uses_first_lines_when_query_has_no_literal_match()
    {
        let (_tmp, state) = test_state();
        let did = seed_filesystem_document(
            &state,
            "notes",
            "rust.md",
            "line1\nline2\nline3\nline4\nline5\nline6\nline7",
            None,
        );

        let config_db = state.open_config_db().unwrap();
        let item = build_search_result_item(
            &state,
            &config_db,
            final_result(&did, "Semantic result", "rust.md"),
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
    fn web_search_result_item_caps_excerpt_count_at_three() {
        let (_tmp, state) = test_state();
        let did = seed_filesystem_document(
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

        let config_db = state.open_config_db().unwrap();
        let item = build_search_result_item(
            &state,
            &config_db,
            final_result(&did, "Rust", "rust.md"),
            "ownership",
        );

        assert_eq!(item.excerpts.len(), 3);
        assert!(
            !item
                .excerpts
                .iter()
                .any(|e| e.text.contains("ownership four"))
        );
    }

    #[tokio::test]
    async fn web_search_rejects_unknown_mode() {
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

        assert_eq!(error, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn web_search_without_plaid_index_returns_service_unavailable() {
        // Fresh data dir → no PLAID index → semantic leg cannot run.
        // The handler converts PlaidIndexMissing to 503 so clients can
        // tell this apart from a real internal error and prompt the
        // user to run `docbert sync`.
        let (_tmp, state) = test_state();

        let status = search(
            State(state),
            Json(SearchRequest {
                query: "rust".to_string(),
                mode: "hybrid".to_string(),
                collection: None,
                count: 10,
                min_score: 0.0,
            }),
        )
        .await
        .expect_err("expected 503 for missing PLAID index");
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }
}
