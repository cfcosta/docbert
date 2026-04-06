use axum::{Router, http::StatusCode, routing};

use super::state::AppState;

pub(crate) mod collections;
pub(crate) mod conversations;
pub(crate) mod settings;

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
        .route("/v1/settings/llm", routing::get(settings::get))
        .route("/v1/settings/llm", routing::put(settings::update))
}
