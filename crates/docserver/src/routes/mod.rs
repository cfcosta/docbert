use axum::{Router, routing};

use crate::{state::AppState, ui};

mod collections;
mod conversations;
mod documents;
mod search;
mod settings;

pub fn router() -> Router<AppState> {
    let api = Router::new()
        .route("/v1/collections", routing::post(collections::create))
        .route("/v1/collections", routing::get(collections::list))
        .route(
            "/v1/collections/{name}",
            routing::delete(collections::delete),
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
            routing::get(documents::get),
        )
        .route(
            "/v1/documents/{collection}/{*path}",
            routing::delete(documents::delete),
        )
        .route("/v1/search", routing::post(search::search))
        .route("/v1/settings/llm", routing::get(settings::get))
        .route("/v1/settings/llm", routing::put(settings::update));

    api.fallback(ui::serve)
}
