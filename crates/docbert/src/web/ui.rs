use std::path::{Path, PathBuf};

use axum::{
    extract::Request,
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
};

const UI_DIST_RELATIVE: &str = "../docserver/ui/dist";

pub(crate) async fn serve(req: Request) -> Response {
    let path = req.uri().path().trim_start_matches('/');

    if path.starts_with("v1/") {
        return StatusCode::NOT_FOUND.into_response();
    }

    if !path.is_empty() {
        let asset_path = ui_dist_dir().join(path);
        if asset_path.is_file() {
            return match std::fs::read(&asset_path) {
                Ok(bytes) => file_response(path, bytes),
                Err(_) => StatusCode::NOT_FOUND.into_response(),
            };
        }
    }

    match std::fs::read_to_string(ui_dist_dir().join("index.html")) {
        Ok(html) => Html(html).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "UI not built").into_response(),
    }
}

fn ui_dist_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(UI_DIST_RELATIVE)
}

fn file_response(path: &str, bytes: Vec<u8>) -> Response {
    let content_type = match Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
    {
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("html") => "text/html; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        Some("ttf") => "font/ttf",
        _ => "application/octet-stream",
    };

    ([(header::CONTENT_TYPE, content_type)], bytes).into_response()
}
