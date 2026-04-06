use axum::{Router, routing};

use super::state::AppState;

pub(crate) mod settings;

pub(crate) fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/settings/llm", routing::get(settings::get))
        .route("/v1/settings/llm", routing::put(settings::update))
}
