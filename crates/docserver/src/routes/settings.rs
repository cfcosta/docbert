use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};

use crate::{error::ApiError, state::AppState};

const KEY_LLM_PROVIDER: &str = "llm_provider";
const KEY_LLM_MODEL: &str = "llm_model";

#[derive(Serialize, Deserialize)]
pub struct LlmSettings {
    provider: Option<String>,
    model: Option<String>,
}

pub async fn get(State(state): State<AppState>) -> Result<Json<LlmSettings>, ApiError> {
    let provider = state.config_db.get_setting(KEY_LLM_PROVIDER)?;
    let model = state.config_db.get_setting(KEY_LLM_MODEL)?;
    Ok(Json(LlmSettings { provider, model }))
}

pub async fn update(
    State(state): State<AppState>,
    Json(body): Json<LlmSettings>,
) -> Result<Json<LlmSettings>, ApiError> {
    match &body.provider {
        Some(p) => state.config_db.set_setting(KEY_LLM_PROVIDER, p)?,
        None => {
            let _ = state.config_db.remove_setting(KEY_LLM_PROVIDER);
        }
    }
    match &body.model {
        Some(m) => state.config_db.set_setting(KEY_LLM_MODEL, m)?,
        None => {
            let _ = state.config_db.remove_setting(KEY_LLM_MODEL);
        }
    }
    Ok(Json(body))
}
