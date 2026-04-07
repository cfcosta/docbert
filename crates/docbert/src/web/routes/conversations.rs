use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use docbert_core::Conversation;
use serde::{Deserialize, Serialize};

use crate::web::state::AppState;

#[derive(Debug, Serialize, PartialEq, Eq)]
pub(crate) struct ConversationSummary {
    id: String,
    title: String,
    created_at: u64,
    updated_at: u64,
    message_count: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CreateRequest {
    id: String,
    title: Option<String>,
}

pub(crate) async fn list(
    State(state): State<AppState>,
) -> Result<Json<Vec<ConversationSummary>>, StatusCode> {
    let config_db = state
        .open_config_db()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let mut summaries: Vec<ConversationSummary> = config_db
        .list_conversations_typed()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .into_iter()
        .map(|c| ConversationSummary {
            id: c.id,
            title: c.title,
            created_at: c.created_at,
            updated_at: c.updated_at,
            message_count: c.messages.len(),
        })
        .collect();
    summaries.sort_by_key(|summary| std::cmp::Reverse(summary.updated_at));
    Ok(Json(summaries))
}

pub(crate) async fn create(
    State(state): State<AppState>,
    Json(body): Json<CreateRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let now = now_millis();
    let conv = Conversation {
        id: body.id,
        title: body.title.unwrap_or_else(|| "New conversation".to_string()),
        created_at: now,
        updated_at: now,
        messages: vec![],
    };
    state
        .open_config_db()
        .and_then(|config_db| config_db.set_conversation_typed(&conv))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok((StatusCode::CREATED, Json(conv)))
}

pub(crate) async fn get(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Conversation>, StatusCode> {
    let config_db = state
        .open_config_db()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let conv = config_db
        .get_conversation_typed(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(conv))
}

pub(crate) async fn update(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(mut body): Json<Conversation>,
) -> Result<Json<Conversation>, StatusCode> {
    let config_db = state
        .open_config_db()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    config_db
        .get_conversation_typed(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    body.id = id;
    body.updated_at = now_millis();

    config_db
        .set_conversation_typed(&body)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(body))
}

pub(crate) async fn delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let existed = state
        .open_config_db()
        .and_then(|config_db| config_db.remove_conversation(&id))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    if !existed {
        return Err(StatusCode::NOT_FOUND);
    }
    Ok(StatusCode::NO_CONTENT)
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::response::IntoResponse;
    use docbert_core::{
        ChatMessage,
        ConfigDb,
        Conversation,
        ModelManager,
        SearchIndex,
        conversation::{ChatActor, ChatPart, ChatRole},
    };

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

    fn conversation(id: &str, updated_at: u64) -> Conversation {
        Conversation {
            id: id.to_string(),
            title: format!("Conversation {id}"),
            created_at: updated_at.saturating_sub(10),
            updated_at,
            messages: vec![ChatMessage {
                id: format!("msg-{id}"),
                role: ChatRole::Assistant,
                actor: Some(ChatActor::Parent),
                parts: vec![ChatPart::Text {
                    text: "hello".to_string(),
                }],
                sources: None,
            }],
        }
    }

    #[tokio::test]
    async fn web_conversations_list_sorts_by_updated_at_desc() {
        let (_tmp, state) = test_state();
        {
            let config_db = ConfigDb::open(&state.data_dir.config_db()).unwrap();
            config_db
                .set_conversation_typed(&conversation("older", 10))
                .unwrap();
            config_db
                .set_conversation_typed(&conversation("newer", 20))
                .unwrap();
        }

        let response = list(State(state)).await.unwrap().0;

        assert_eq!(response.len(), 2);
        assert_eq!(response[0].id, "newer");
        assert_eq!(response[1].id, "older");
        assert_eq!(response[0].message_count, 1);
    }

    #[tokio::test]
    async fn web_conversations_create_defaults_title() {
        let (_tmp, state) = test_state();

        let response = create(
            State(state.clone()),
            Json(CreateRequest {
                id: "conv-1".to_string(),
                title: None,
            }),
        )
        .await
        .unwrap()
        .into_response();

        assert_eq!(response.status(), StatusCode::CREATED);
        let stored = ConfigDb::open(&state.data_dir.config_db())
            .unwrap()
            .get_conversation_typed("conv-1")
            .unwrap()
            .unwrap();
        assert_eq!(stored.title, "New conversation");
        assert_eq!(stored.messages.len(), 0);
    }

    #[tokio::test]
    async fn web_conversations_get_returns_not_found_for_missing() {
        let (_tmp, state) = test_state();

        let status = get(State(state), Path("missing".to_string()))
            .await
            .unwrap_err();

        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn web_conversations_update_overwrites_path_id_and_refreshes_timestamp()
     {
        let (_tmp, state) = test_state();
        ConfigDb::open(&state.data_dir.config_db())
            .unwrap()
            .set_conversation_typed(&conversation("conv-1", 10))
            .unwrap();

        let body = Conversation {
            id: "other-id".to_string(),
            title: "Updated title".to_string(),
            created_at: 1,
            updated_at: 2,
            messages: vec![],
        };

        let response = update(
            State(state.clone()),
            Path("conv-1".to_string()),
            Json(body),
        )
        .await
        .unwrap()
        .0;

        assert_eq!(response.id, "conv-1");
        assert_eq!(response.title, "Updated title");
        assert!(response.updated_at >= 10);
        let stored = ConfigDb::open(&state.data_dir.config_db())
            .unwrap()
            .get_conversation_typed("conv-1")
            .unwrap()
            .unwrap();
        assert_eq!(stored.id, "conv-1");
        assert_eq!(stored.title, "Updated title");
    }

    #[tokio::test]
    async fn web_conversations_delete_returns_no_content_for_existing() {
        let (_tmp, state) = test_state();
        ConfigDb::open(&state.data_dir.config_db())
            .unwrap()
            .set_conversation_typed(&conversation("conv-1", 10))
            .unwrap();

        let response = delete(State(state.clone()), Path("conv-1".to_string()))
            .await
            .unwrap()
            .into_response();

        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert!(
            ConfigDb::open(&state.data_dir.config_db())
                .unwrap()
                .get_conversation_typed("conv-1")
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn web_conversations_delete_returns_not_found_for_missing() {
        let (_tmp, state) = test_state();

        match delete(State(state), Path("missing".to_string())).await {
            Err(status) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => panic!("expected not found"),
        }
    }
}
