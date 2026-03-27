use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};

use crate::{error::ApiError, state::AppState};

const KEY_LLM_PROVIDER: &str = "llm_provider";
const KEY_LLM_MODEL: &str = "llm_model";
const KEY_LLM_API_KEY: &str = "llm_api_key";

#[derive(Serialize, Deserialize)]
pub struct LlmSettings {
    provider: Option<String>,
    model: Option<String>,
    api_key: Option<String>,
}

/// Resolve the API key: stored value first, then fall back to env var
/// based on provider.
fn resolve_api_key(
    stored: Option<String>,
    provider: Option<&str>,
) -> Option<String> {
    if stored.is_some() {
        return stored;
    }

    let env_var = match provider? {
        "anthropic" => "ANTHROPIC_API_KEY",
        "openai" => "OPENAI_API_KEY",
        _ => return None,
    };

    std::env::var(env_var).ok()
}

pub async fn get(
    State(state): State<AppState>,
) -> Result<Json<LlmSettings>, ApiError> {
    let provider = state.config_db.get_setting(KEY_LLM_PROVIDER)?;
    let model = state.config_db.get_setting(KEY_LLM_MODEL)?;
    let stored_key = state.config_db.get_setting(KEY_LLM_API_KEY)?;
    let api_key = resolve_api_key(stored_key, provider.as_deref());

    Ok(Json(LlmSettings {
        provider,
        model,
        api_key,
    }))
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
    match &body.api_key {
        Some(k) if !k.is_empty() => {
            state.config_db.set_setting(KEY_LLM_API_KEY, k)?
        }
        _ => {
            let _ = state.config_db.remove_setting(KEY_LLM_API_KEY);
        }
    }
    Ok(Json(body))
}
