use axum::{Router, routing};

use crate::{state::AppState, ui};

mod collections;
mod documents;
mod search;

pub fn router() -> Router<AppState> {
    let api = Router::new()
        .route("/v1/collections", routing::post(collections::create))
        .route("/v1/collections", routing::get(collections::list))
        .route(
            "/v1/collections/{name}",
            routing::delete(collections::delete),
        )
        .route("/v1/documents", routing::post(documents::ingest))
        .route(
            "/v1/documents/{collection}/{*path}",
            routing::get(documents::get),
        )
        .route(
            "/v1/documents/{collection}/{*path}",
            routing::delete(documents::delete),
        )
        .route("/v1/search", routing::post(search::search));

    api.fallback(ui::serve)
}
