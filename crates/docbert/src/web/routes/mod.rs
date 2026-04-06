use axum::{Router, http::StatusCode, routing};

use super::{state::AppState, ui};

pub(crate) mod collections;
pub(crate) mod conversations;
pub(crate) mod documents;
pub(crate) mod search;
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
        .fallback(ui::serve)
}
