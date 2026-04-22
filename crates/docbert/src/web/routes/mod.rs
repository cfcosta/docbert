use axum::{Router, http::StatusCode, routing};

use super::{state::AppState, ui};

pub(crate) mod collections;
pub(crate) mod conversations;
pub(crate) mod documents;
pub(crate) mod search;
pub(crate) mod settings;

/// Logs a handler error at `ERROR` level and returns `500 Internal
/// Server Error`.
///
/// Every route used to convert errors with `map_err(|_| INTERNAL_...)`,
/// which swallowed the original error entirely — a 500 in the client
/// left no trace server-side. Route handlers now funnel their 500 paths
/// through this helper so every internal error lands in the tracing
/// output with its `Display` and `Debug` representation plus a short
/// call-site-provided `context` string.
pub(crate) fn log_internal_error<E>(err: E, context: &'static str) -> StatusCode
where
    E: std::fmt::Display + std::fmt::Debug,
{
    tracing::error!(
        error = %err,
        ?err,
        context,
        "handler returning 500",
    );
    StatusCode::INTERNAL_SERVER_ERROR
}

pub(crate) fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/v1/collections",
            routing::get(collections::list)
                .fallback(|| async { StatusCode::NOT_FOUND }),
        )
        .route(
            "/v1/conversations",
            routing::get(conversations::list).post(conversations::create),
        )
        .route(
            "/v1/conversations/{id}",
            routing::get(conversations::get)
                .put(conversations::update)
                .delete(conversations::delete),
        )
        .route("/v1/documents", routing::post(documents::ingest))
        .route(
            "/v1/collections/{name}/documents",
            routing::get(documents::list_by_collection),
        )
        .route(
            "/v1/documents/{collection}/{*path}",
            routing::get(documents::get).delete(documents::delete),
        )
        .route("/v1/search", routing::post(search::search))
        .route("/v1/settings/llm", routing::get(settings::get))
        .route("/v1/settings/llm", routing::put(settings::update))
        .route(
            "/v1/settings/llm/oauth/openai-codex/start",
            routing::post(settings::start_openai_codex_oauth),
        )
        .route(
            "/v1/settings/llm/oauth/openai-codex/logout",
            routing::post(settings::logout_openai_codex_oauth),
        )
        .fallback(ui::serve)
}
