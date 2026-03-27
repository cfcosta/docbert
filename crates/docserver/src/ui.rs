use axum::{
    extract::Request,
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
};
use rust_embed::Embed;

#[derive(Embed)]
#[folder = "ui/dist/"]
struct Assets;

pub async fn serve(req: Request) -> Response {
    let path = req.uri().path().trim_start_matches('/');

    // Try the exact path first (e.g. "assets/index-abc123.js").
    if let Some(file) = Assets::get(path) {
        return file_response(path, &file);
    }

    // For any path that doesn't match a static file, serve index.html
    // so the SPA's client-side router can handle it.
    match Assets::get("index.html") {
        Some(file) => Html(file.data.to_vec()).into_response(),
        None => (StatusCode::NOT_FOUND, "UI not built").into_response(),
    }
}

fn file_response(path: &str, file: &rust_embed::EmbeddedFile) -> Response {
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    ([(header::CONTENT_TYPE, mime.as_ref())], file.data.clone()).into_response()
}
