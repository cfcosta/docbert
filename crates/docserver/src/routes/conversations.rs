use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use docbert_core::Conversation;
use serde::{Deserialize, Serialize};

use crate::{error::ApiError, state::AppState};

#[derive(Serialize)]
pub struct ConversationSummary {
    id: String,
    title: String,
    created_at: u64,
    updated_at: u64,
    message_count: usize,
}

#[derive(Deserialize)]
pub struct CreateRequest {
    id: String,
    title: Option<String>,
}

pub async fn list(
    State(state): State<AppState>,
) -> Result<Json<Vec<ConversationSummary>>, ApiError> {
    let raw = state.config_db.list_conversations()?;
    let mut summaries: Vec<ConversationSummary> = raw
        .into_iter()
        .filter_map(|(_, data)| {
            Conversation::deserialize(&data)
                .ok()
                .map(|c| ConversationSummary {
                    id: c.id,
                    title: c.title,
                    created_at: c.created_at,
                    updated_at: c.updated_at,
                    message_count: c.messages.len(),
                })
        })
        .collect();
    summaries.sort_by_key(|summary| std::cmp::Reverse(summary.updated_at));
    Ok(Json(summaries))
}

pub async fn create(
    State(state): State<AppState>,
    Json(body): Json<CreateRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let now = now_millis();
    let conv = Conversation {
        id: body.id,
        title: body.title.unwrap_or_else(|| "New conversation".to_string()),
        created_at: now,
        updated_at: now,
        messages: vec![],
    };
    let data = conv.serialize().map_err(ApiError::internal)?;
    state.config_db.set_conversation(&conv.id, &data)?;
    Ok((StatusCode::CREATED, Json(conv)))
}

pub async fn get(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Conversation>, ApiError> {
    let data = state.config_db.get_conversation(&id)?.ok_or_else(|| {
        ApiError::NotFound(format!("conversation not found: {id}"))
    })?;
    let conv = Conversation::deserialize(&data).map_err(ApiError::internal)?;
    Ok(Json(conv))
}

pub async fn update(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(mut body): Json<Conversation>,
) -> Result<Json<Conversation>, ApiError> {
    state.config_db.get_conversation(&id)?.ok_or_else(|| {
        ApiError::NotFound(format!("conversation not found: {id}"))
    })?;

    body.id = id;
    body.updated_at = now_millis();

    let data = body.serialize().map_err(ApiError::internal)?;
    state.config_db.set_conversation(&body.id, &data)?;
    Ok(Json(body))
}

pub async fn delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let existed = state.config_db.remove_conversation(&id)?;
    if !existed {
        return Err(ApiError::NotFound(format!(
            "conversation not found: {id}"
        )));
    }
    Ok(StatusCode::NO_CONTENT)
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
