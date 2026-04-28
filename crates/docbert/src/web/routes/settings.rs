use std::{
    io::ErrorKind,
    sync::{Arc, OnceLock},
    time::Duration,
};

use axum::{
    Json,
    Router,
    extract::{Query, State},
    http::StatusCode,
    response::Html,
    routing,
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use docbert_core::{ConfigDb, PersistedLlmSettings};
use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::oneshot;

use crate::web::{routes::log_internal_error, state::AppState};

const KEY_OPENAI_CODEX_OAUTH: &str = "llm_oauth:openai-codex";
const OPENAI_CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const OPENAI_CODEX_AUTHORIZE_URL: &str =
    "https://auth.openai.com/oauth/authorize";
const OPENAI_CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const OPENAI_CODEX_REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const OPENAI_CODEX_SCOPE: &str = "openid profile email offline_access";
const OPENAI_CODEX_CALLBACK_BIND_ADDR: &str = "127.0.0.1:1455";
const OAUTH_REFRESH_SKEW_MS: u64 = 60_000;
const OAUTH_CALLBACK_SERVER_TTL_SECS: u64 = 600;
const OAUTH_HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// Shared `reqwest::Client` for OAuth calls. Built once so the
/// connection pool and TLS session are reused across token
/// exchanges, and so a stalled token endpoint can't hang the
/// request handler past the configured timeout.
fn oauth_http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .user_agent(concat!("docbert/", env!("CARGO_PKG_VERSION")))
            .timeout(OAUTH_HTTP_TIMEOUT)
            .build()
            .expect("oauth reqwest client should build")
    })
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LlmSettings {
    pub(crate) provider: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) api_key: Option<String>,
    #[serde(default)]
    pub(crate) oauth_connected: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) oauth_expires_at: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LlmSettingsUpdate {
    pub(crate) provider: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) api_key: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct OAuthStartResponse {
    pub(crate) authorization_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct OpenAICodexOAuthCredentials {
    access_token: String,
    refresh_token: String,
    expires_at: u64,
}

#[derive(Debug, Deserialize)]
struct OpenAICodexCallbackQuery {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAICodexTokenResponse {
    access_token: Option<String>,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
}

fn provider_uses_oauth(provider: Option<&str>) -> bool {
    matches!(provider, Some("openai-codex"))
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

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn create_oauth_random_bytes(len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    rand::rng().fill_bytes(&mut bytes);
    bytes
}

fn create_oauth_state() -> String {
    URL_SAFE_NO_PAD.encode(create_oauth_random_bytes(16))
}

fn create_code_verifier() -> String {
    URL_SAFE_NO_PAD.encode(create_oauth_random_bytes(32))
}

fn create_code_challenge(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    URL_SAFE_NO_PAD.encode(digest)
}

fn build_openai_codex_authorization_url(
    state: &str,
    challenge: &str,
) -> Result<String, StatusCode> {
    let mut url = reqwest::Url::parse(OPENAI_CODEX_AUTHORIZE_URL)
        .map_err(|err| log_internal_error(err, "oauth parse authorize url"))?;
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", OPENAI_CODEX_CLIENT_ID)
        .append_pair("redirect_uri", OPENAI_CODEX_REDIRECT_URI)
        .append_pair("scope", OPENAI_CODEX_SCOPE)
        .append_pair("code_challenge", challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", state)
        .append_pair("id_token_add_organizations", "true")
        .append_pair("codex_cli_simplified_flow", "true")
        .append_pair("originator", "pi");
    Ok(url.to_string())
}

fn oauth_callback_html(title: &str, message: &str, success: bool) -> String {
    let status = if success { "success" } else { "error" };
    let close_script = if success {
        "<script>setTimeout(() => window.close(), 1200);</script>"
    } else {
        ""
    };

    format!(
        r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      :root {{ color-scheme: light dark; }}
      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #11111b;
        color: #cdd6f4;
        display: grid;
        min-height: 100vh;
        place-items: center;
        padding: 24px;
      }}
      .card {{
        width: min(32rem, 100%);
        border: 1px solid #313244;
        border-radius: 16px;
        padding: 24px;
        background: #181825;
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.35);
      }}
      h1 {{ margin: 0 0 12px; font-size: 1.2rem; }}
      p {{ margin: 0; line-height: 1.6; color: #bac2de; }}
      .status {{
        display: inline-flex;
        align-items: center;
        margin-bottom: 14px;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
      }}
      .status.success {{ background: rgba(166, 227, 161, 0.14); color: #a6e3a1; border: 1px solid rgba(166, 227, 161, 0.35); }}
      .status.error {{ background: rgba(243, 139, 168, 0.14); color: #f38ba8; border: 1px solid rgba(243, 139, 168, 0.35); }}
    </style>
    {close_script}
  </head>
  <body>
    <main class="card">
      <div class="status {status}">{title}</div>
      <h1>{title}</h1>
      <p>{message}</p>
    </main>
  </body>
</html>"#,
        title = title,
        message = message,
        status = status,
        close_script = close_script,
    )
}

fn load_openai_codex_oauth(
    config_db: &ConfigDb,
) -> Result<Option<OpenAICodexOAuthCredentials>, StatusCode> {
    config_db
        .get_json_setting(KEY_OPENAI_CODEX_OAUTH)
        .map_err(|err| log_internal_error(err, "oauth load read"))?
        .map(serde_json::from_value)
        .transpose()
        .map_err(|err| log_internal_error(err, "oauth load parse"))
}

fn store_openai_codex_oauth(
    config_db: &ConfigDb,
    credentials: &OpenAICodexOAuthCredentials,
) -> Result<(), StatusCode> {
    let value = serde_json::to_value(credentials)
        .map_err(|err| log_internal_error(err, "oauth store serialize"))?;
    config_db
        .set_json_setting(KEY_OPENAI_CODEX_OAUTH, &value)
        .map_err(|err| log_internal_error(err, "oauth store write"))
}

fn clear_openai_codex_oauth(config_db: &ConfigDb) -> Result<(), StatusCode> {
    config_db
        .remove_json_setting(KEY_OPENAI_CODEX_OAUTH)
        .map(|_| ())
        .map_err(|err| log_internal_error(err, "oauth clear"))
}

async fn exchange_openai_codex_authorization_code(
    code: &str,
    verifier: &str,
) -> Result<OpenAICodexOAuthCredentials, String> {
    let response = oauth_http_client()
        .post(OPENAI_CODEX_TOKEN_URL)
        .form(&[
            ("grant_type", "authorization_code"),
            ("client_id", OPENAI_CODEX_CLIENT_ID),
            ("code", code),
            ("code_verifier", verifier),
            ("redirect_uri", OPENAI_CODEX_REDIRECT_URI),
        ])
        .send()
        .await
        .map_err(|error| error.to_string())?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        tracing::warn!(
            status = %status,
            body = %text,
            "openai-codex authorization_code exchange failed"
        );
        return Err("Could not complete ChatGPT sign-in.".to_string());
    }

    let body = response
        .json::<OpenAICodexTokenResponse>()
        .await
        .map_err(|error| error.to_string())?;

    let access_token = body.access_token.ok_or_else(|| {
        "Missing access token in ChatGPT sign-in response.".to_string()
    })?;
    let refresh_token = body.refresh_token.ok_or_else(|| {
        "Missing refresh token in ChatGPT sign-in response.".to_string()
    })?;
    let expires_in = body.expires_in.ok_or_else(|| {
        "Missing token expiry in ChatGPT sign-in response.".to_string()
    })?;

    Ok(OpenAICodexOAuthCredentials {
        access_token,
        refresh_token,
        expires_at: now_ms() + expires_in * 1000,
    })
}

async fn refresh_openai_codex_access_token(
    refresh_token: &str,
) -> Result<OpenAICodexOAuthCredentials, String> {
    let response = oauth_http_client()
        .post(OPENAI_CODEX_TOKEN_URL)
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", OPENAI_CODEX_CLIENT_ID),
            ("refresh_token", refresh_token),
        ])
        .send()
        .await
        .map_err(|error| error.to_string())?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        tracing::warn!(
            status = %status,
            body = %text,
            "openai-codex token refresh failed"
        );
        return Err("Could not refresh the ChatGPT session.".to_string());
    }

    let body = response
        .json::<OpenAICodexTokenResponse>()
        .await
        .map_err(|error| error.to_string())?;

    let access_token = body.access_token.ok_or_else(|| {
        "Missing access token in ChatGPT refresh response.".to_string()
    })?;
    let expires_in = body.expires_in.ok_or_else(|| {
        "Missing token expiry in ChatGPT refresh response.".to_string()
    })?;

    Ok(OpenAICodexOAuthCredentials {
        access_token,
        refresh_token: body
            .refresh_token
            .unwrap_or_else(|| refresh_token.to_string()),
        expires_at: now_ms() + expires_in * 1000,
    })
}

async fn resolve_openai_codex_access_token(
    state: &AppState,
) -> Result<Option<OpenAICodexOAuthCredentials>, StatusCode> {
    let config_db = state.open_config_db().map_err(|err| {
        log_internal_error(err, "settings resolve oauth open db")
    })?;
    let Some(credentials) = load_openai_codex_oauth(&config_db)? else {
        return Ok(None);
    };

    if credentials.expires_at > now_ms() + OAUTH_REFRESH_SKEW_MS {
        return Ok(Some(credentials));
    }

    match refresh_openai_codex_access_token(&credentials.refresh_token).await {
        Ok(refreshed) => {
            store_openai_codex_oauth(&config_db, &refreshed)?;
            Ok(Some(refreshed))
        }
        Err(error) => {
            tracing::warn!(%error, "openai-codex session refresh failed; clearing stored oauth state");
            clear_openai_codex_oauth(&config_db)?;
            Ok(None)
        }
    }
}

async fn build_llm_settings_response(
    state: &AppState,
    stored: PersistedLlmSettings,
) -> Result<LlmSettings, StatusCode> {
    if provider_uses_oauth(stored.provider.as_deref()) {
        if let Some(credentials) =
            resolve_openai_codex_access_token(state).await?
        {
            return Ok(LlmSettings {
                provider: stored.provider,
                model: stored.model,
                api_key: Some(credentials.access_token),
                oauth_connected: true,
                oauth_expires_at: Some(credentials.expires_at),
            });
        }

        return Ok(LlmSettings {
            provider: stored.provider,
            model: stored.model,
            api_key: None,
            oauth_connected: false,
            oauth_expires_at: None,
        });
    }

    let api_key = resolve_api_key(stored.api_key, stored.provider.as_deref());
    Ok(LlmSettings {
        provider: stored.provider,
        model: stored.model,
        api_key,
        oauth_connected: false,
        oauth_expires_at: None,
    })
}

pub(crate) async fn get(
    State(state): State<AppState>,
) -> Result<Json<LlmSettings>, StatusCode> {
    let stored = {
        let config_db = state
            .open_config_db()
            .map_err(|err| log_internal_error(err, "settings::get open db"))?;
        config_db
            .get_persisted_llm_settings()
            .map_err(|err| log_internal_error(err, "settings::get read"))?
    };
    let resolved = build_llm_settings_response(&state, stored).await?;
    Ok(Json(resolved))
}

pub(crate) async fn update(
    State(state): State<AppState>,
    Json(body): Json<LlmSettingsUpdate>,
) -> Result<Json<LlmSettings>, StatusCode> {
    let stored = PersistedLlmSettings {
        provider: body.provider.clone(),
        model: body.model.clone(),
        api_key: if provider_uses_oauth(body.provider.as_deref()) {
            None
        } else {
            body.api_key.clone().filter(|key| !key.is_empty())
        },
    };
    state
        .open_config_db()
        .and_then(|config_db| config_db.set_persisted_llm_settings(&stored))
        .map_err(|err| log_internal_error(err, "settings::update persist"))?;
    let resolved = build_llm_settings_response(&state, stored).await?;
    Ok(Json(resolved))
}

fn send_oauth_shutdown(
    shutdown: &Arc<std::sync::Mutex<Option<oneshot::Sender<()>>>>,
) {
    if let Ok(mut guard) = shutdown.lock()
        && let Some(sender) = guard.take()
    {
        let _ = sender.send(());
    }
}

async fn handle_openai_codex_callback(
    state: AppState,
    query: OpenAICodexCallbackQuery,
    verifier: String,
    expected_state: String,
    shutdown: Arc<std::sync::Mutex<Option<oneshot::Sender<()>>>>,
) -> (StatusCode, Html<String>) {
    let response = if let Some(error) = query.error {
        (
            StatusCode::BAD_REQUEST,
            Html(oauth_callback_html(
                "Sign-in failed",
                &format!("ChatGPT returned an error: {error}"),
                false,
            )),
        )
    } else if query.state.as_deref() != Some(expected_state.as_str()) {
        (
            StatusCode::BAD_REQUEST,
            Html(oauth_callback_html(
                "Sign-in failed",
                "The ChatGPT sign-in state did not match. Please try again.",
                false,
            )),
        )
    } else if let Some(code) = query.code {
        match exchange_openai_codex_authorization_code(&code, &verifier).await {
            Ok(credentials) => {
                let persisted = state
                    .open_config_db()
                    .map_err(|err| {
                        log_internal_error(err, "oauth callback open db")
                    })
                    .and_then(|config_db| {
                        store_openai_codex_oauth(&config_db, &credentials)
                    });

                match persisted {
                    Ok(()) => (
                        StatusCode::OK,
                        Html(oauth_callback_html(
                            "Sign-in complete",
                            "ChatGPT is now connected to docbert. You can return to the Settings page.",
                            true,
                        )),
                    ),
                    Err(_) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Html(oauth_callback_html(
                            "Sign-in failed",
                            "docbert could not save the ChatGPT session locally.",
                            false,
                        )),
                    ),
                }
            }
            Err(error) => (
                StatusCode::BAD_REQUEST,
                Html(oauth_callback_html("Sign-in failed", &error, false)),
            ),
        }
    } else {
        (
            StatusCode::BAD_REQUEST,
            Html(oauth_callback_html(
                "Sign-in failed",
                "ChatGPT did not return an authorization code.",
                false,
            )),
        )
    };

    send_oauth_shutdown(&shutdown);
    response
}

async fn spawn_openai_codex_callback_server(
    state: AppState,
    verifier: String,
    expected_state: String,
) -> Result<(), StatusCode> {
    let listener =
        tokio::net::TcpListener::bind(OPENAI_CODEX_CALLBACK_BIND_ADDR)
            .await
            .map_err(|error| {
                if error.kind() == ErrorKind::AddrInUse {
                    StatusCode::CONFLICT
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                }
            })?;

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let shutdown = Arc::new(std::sync::Mutex::new(Some(shutdown_tx)));

    let app = Router::new().route(
        "/auth/callback",
        routing::get({
            let state = state.clone();
            let verifier = verifier.clone();
            let expected_state = expected_state.clone();
            let shutdown = shutdown.clone();
            move |Query(query): Query<OpenAICodexCallbackQuery>| {
                let state = state.clone();
                let verifier = verifier.clone();
                let expected_state = expected_state.clone();
                let shutdown = shutdown.clone();
                async move {
                    handle_openai_codex_callback(
                        state,
                        query,
                        verifier,
                        expected_state,
                        shutdown,
                    )
                    .await
                }
            }
        }),
    );

    tokio::spawn(async move {
        let shutdown_future = async move {
            tokio::select! {
                _ = async {
                    let _ = shutdown_rx.await;
                } => {}
                _ = tokio::time::sleep(Duration::from_secs(OAUTH_CALLBACK_SERVER_TTL_SECS)) => {}
            }
        };

        if let Err(error) = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_future)
            .await
        {
            tracing::warn!(%error, "openai-codex callback server stopped with error");
        }
    });

    Ok(())
}

pub(crate) async fn start_openai_codex_oauth(
    State(state): State<AppState>,
) -> Result<Json<OAuthStartResponse>, StatusCode> {
    let verifier = create_code_verifier();
    let challenge = create_code_challenge(&verifier);
    let oauth_state = create_oauth_state();

    spawn_openai_codex_callback_server(state, verifier, oauth_state.clone())
        .await?;

    Ok(Json(OAuthStartResponse {
        authorization_url: build_openai_codex_authorization_url(
            &oauth_state,
            &challenge,
        )?,
    }))
}

pub(crate) async fn logout_openai_codex_oauth(
    State(state): State<AppState>,
) -> Result<StatusCode, StatusCode> {
    let config_db = state
        .open_config_db()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    clear_openai_codex_oauth(&config_db)?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, OnceLock};

    use docbert_core::{ModelManager, SearchIndex};

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

    fn store_codex_credentials(state: &AppState, expires_at: u64) {
        let config_db = ConfigDb::open(&state.data_dir.config_db()).unwrap();
        store_openai_codex_oauth(
            &config_db,
            &OpenAICodexOAuthCredentials {
                access_token: "oauth-access".to_string(),
                refresh_token: "oauth-refresh".to_string(),
                expires_at,
            },
        )
        .unwrap();
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
        assert!(!response.oauth_connected);
        assert_eq!(response.oauth_expires_at, None);

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
        assert!(!response.oauth_connected);
        assert_eq!(response.oauth_expires_at, None);

        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
        }
    }

    #[test]
    fn web_settings_get_uses_openai_codex_oauth_when_available() {
        let (_tmp, state) = test_state();
        ConfigDb::open(&state.data_dir.config_db())
            .unwrap()
            .set_persisted_llm_settings(&PersistedLlmSettings {
                provider: Some("openai-codex".to_string()),
                model: Some("gpt-5.1-codex-mini".to_string()),
                api_key: None,
            })
            .unwrap();
        let expires_at = now_ms() + 120_000;
        store_codex_credentials(&state, expires_at);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let response = runtime.block_on(get(State(state))).unwrap().0;

        assert_eq!(response.provider.as_deref(), Some("openai-codex"));
        assert_eq!(response.model.as_deref(), Some("gpt-5.1-codex-mini"));
        assert_eq!(response.api_key.as_deref(), Some("oauth-access"));
        assert!(response.oauth_connected);
        assert_eq!(response.oauth_expires_at, Some(expires_at));
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
            Json(LlmSettingsUpdate {
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
        assert_eq!(response.api_key, None);
        assert!(!response.oauth_connected);
        assert_eq!(response.oauth_expires_at, None);
        assert_eq!(
            ConfigDb::open(&state.data_dir.config_db())
                .unwrap()
                .get_persisted_llm_settings()
                .unwrap(),
            PersistedLlmSettings::default()
        );
    }

    #[tokio::test]
    async fn web_settings_update_ignores_manual_api_key_for_openai_codex() {
        let (_tmp, state) = test_state();

        let response = update(
            State(state.clone()),
            Json(LlmSettingsUpdate {
                provider: Some("openai-codex".to_string()),
                model: Some("gpt-5.1-codex-mini".to_string()),
                api_key: Some("should-not-store".to_string()),
            }),
        )
        .await
        .unwrap()
        .0;

        assert_eq!(response.provider.as_deref(), Some("openai-codex"));
        assert_eq!(response.model.as_deref(), Some("gpt-5.1-codex-mini"));
        assert_eq!(response.api_key, None);
        assert!(!response.oauth_connected);
        assert_eq!(response.oauth_expires_at, None);
        assert_eq!(
            ConfigDb::open(&state.data_dir.config_db())
                .unwrap()
                .get_persisted_llm_settings()
                .unwrap(),
            PersistedLlmSettings {
                provider: Some("openai-codex".to_string()),
                model: Some("gpt-5.1-codex-mini".to_string()),
                api_key: None,
            }
        );
    }

    #[tokio::test]
    async fn web_settings_logout_openai_codex_oauth_clears_stored_session() {
        let (_tmp, state) = test_state();
        store_codex_credentials(&state, now_ms() + 120_000);

        let status = logout_openai_codex_oauth(State(state.clone()))
            .await
            .unwrap();

        assert_eq!(status, StatusCode::NO_CONTENT);
        assert!(
            ConfigDb::open(&state.data_dir.config_db())
                .unwrap()
                .get_json_setting(KEY_OPENAI_CODEX_OAUTH)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn openai_codex_authorization_url_contains_expected_fields() {
        let url =
            build_openai_codex_authorization_url("state-123", "challenge-456")
                .unwrap();
        let parsed = reqwest::Url::parse(&url).unwrap();
        let query = parsed
            .query_pairs()
            .collect::<std::collections::HashMap<_, _>>();

        assert!(parsed.as_str().starts_with(OPENAI_CODEX_AUTHORIZE_URL));
        assert_eq!(
            query.get("client_id"),
            Some(&OPENAI_CODEX_CLIENT_ID.into())
        );
        assert_eq!(
            query.get("redirect_uri"),
            Some(&OPENAI_CODEX_REDIRECT_URI.into())
        );
        assert_eq!(query.get("state"), Some(&"state-123".into()));
        assert_eq!(query.get("code_challenge"), Some(&"challenge-456".into()));
        assert_eq!(query.get("originator"), Some(&"pi".into()));
    }
}
