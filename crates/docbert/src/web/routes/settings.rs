use axum::{Json, extract::State, http::StatusCode};
use docbert_core::PersistedLlmSettings;
use serde::{Deserialize, Serialize};

use crate::web::state::AppState;

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LlmSettings {
    pub(crate) provider: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) api_key: Option<String>,
}

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

pub(crate) async fn get(
    State(state): State<AppState>,
) -> Result<Json<LlmSettings>, StatusCode> {
    let config_db = state
        .open_config_db()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let stored = config_db
        .get_persisted_llm_settings()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let api_key = resolve_api_key(stored.api_key, stored.provider.as_deref());

    Ok(Json(LlmSettings {
        provider: stored.provider,
        model: stored.model,
        api_key,
    }))
}

pub(crate) async fn update(
    State(state): State<AppState>,
    Json(body): Json<LlmSettings>,
) -> Result<Json<LlmSettings>, StatusCode> {
    let stored = PersistedLlmSettings {
        provider: body.provider.clone(),
        model: body.model.clone(),
        api_key: body.api_key.clone().filter(|key| !key.is_empty()),
    };
    state
        .open_config_db()
        .and_then(|config_db| config_db.set_persisted_llm_settings(&stored))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(body))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, OnceLock};

    use docbert_core::{ConfigDb, ModelManager, SearchIndex};

    use super::*;
    use crate::web::state::Inner;

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn test_state() -> (tempfile::TempDir, AppState) {
        let tmp = tempfile::tempdir().unwrap();
        let state = Arc::new(Inner {
            data_dir: docbert_core::DataDir::new(tmp.path()),
            search_index: SearchIndex::open_in_ram().unwrap(),
            model: Mutex::new(ModelManager::new()),
        });

        (tmp, state)
    }

    #[test]
    fn web_settings_get_prefers_stored_api_key_over_env() {
        let _guard = env_lock();
        let (_tmp, state) = test_state();
        ConfigDb::open(&state.data_dir.config_db())
            .unwrap()
            .set_persisted_llm_settings(&PersistedLlmSettings {
                provider: Some("openai".to_string()),
                model: Some("gpt-4.1".to_string()),
                api_key: Some("stored-key".to_string()),
            })
            .unwrap();
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "env-key");
        }

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let response = runtime.block_on(get(State(state))).unwrap().0;

        assert_eq!(response.provider.as_deref(), Some("openai"));
        assert_eq!(response.model.as_deref(), Some("gpt-4.1"));
        assert_eq!(response.api_key.as_deref(), Some("stored-key"));

        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
        }
    }

    #[test]
    fn web_settings_get_uses_env_api_key_when_stored_key_missing() {
        let _guard = env_lock();
        let (_tmp, state) = test_state();
        ConfigDb::open(&state.data_dir.config_db())
            .unwrap()
            .set_persisted_llm_settings(&PersistedLlmSettings {
                provider: Some("anthropic".to_string()),
                model: Some("claude-sonnet".to_string()),
                api_key: None,
            })
            .unwrap();
        unsafe {
            std::env::set_var("ANTHROPIC_API_KEY", "env-key");
        }

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let response = runtime.block_on(get(State(state))).unwrap().0;

        assert_eq!(response.provider.as_deref(), Some("anthropic"));
        assert_eq!(response.model.as_deref(), Some("claude-sonnet"));
        assert_eq!(response.api_key.as_deref(), Some("env-key"));

        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
        }
    }

    #[tokio::test]
    async fn web_settings_update_clears_empty_api_key_and_absent_provider_model()
     {
        let (_tmp, state) = test_state();
        ConfigDb::open(&state.data_dir.config_db())
            .unwrap()
            .set_persisted_llm_settings(&PersistedLlmSettings {
                provider: Some("openai".to_string()),
                model: Some("gpt-4.1".to_string()),
                api_key: Some("stored-key".to_string()),
            })
            .unwrap();

        let response = update(
            State(state.clone()),
            Json(LlmSettings {
                provider: None,
                model: None,
                api_key: Some(String::new()),
            }),
        )
        .await
        .unwrap()
        .0;

        assert_eq!(response.provider, None);
        assert_eq!(response.model, None);
        assert_eq!(response.api_key, Some(String::new()));
        assert_eq!(
            ConfigDb::open(&state.data_dir.config_db())
                .unwrap()
                .get_persisted_llm_settings()
                .unwrap(),
            PersistedLlmSettings::default()
        );
    }

    #[tokio::test]
    async fn web_settings_update_preserves_http_shape() {
        let (_tmp, state) = test_state();
        let body = LlmSettings {
            provider: Some("openai".to_string()),
            model: Some("gpt-4.1".to_string()),
            api_key: Some("stored-key".to_string()),
        };

        let response =
            update(State(state.clone()), Json(body)).await.unwrap().0;

        assert_eq!(response.provider.as_deref(), Some("openai"));
        assert_eq!(response.model.as_deref(), Some("gpt-4.1"));
        assert_eq!(response.api_key.as_deref(), Some("stored-key"));
        assert_eq!(
            ConfigDb::open(&state.data_dir.config_db())
                .unwrap()
                .get_persisted_llm_settings()
                .unwrap(),
            PersistedLlmSettings {
                provider: Some("openai".to_string()),
                model: Some("gpt-4.1".to_string()),
                api_key: Some("stored-key".to_string()),
            }
        );
    }
}
