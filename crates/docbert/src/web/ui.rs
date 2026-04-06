use std::{borrow::Cow, path::Path};

use axum::{
    extract::Request,
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
};
use include_dir::{Dir, include_dir};

static UI_DIST: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/ui/dist");

pub(crate) async fn serve(req: Request) -> Response {
    let path = req.uri().path().trim_start_matches('/');

    if path.starts_with("v1/") {
        return StatusCode::NOT_FOUND.into_response();
    }

    if !path.is_empty()
        && let Some(file) = UI_DIST.get_file(path)
    {
        return file_response(path, Cow::Borrowed(file.contents()));
    }

    match UI_DIST.get_file("index.html") {
        Some(file) => {
            Html(String::from_utf8_lossy(file.contents()).into_owned())
                .into_response()
        }
        None => (StatusCode::NOT_FOUND, "UI not built").into_response(),
    }
}

fn file_response(path: &str, bytes: Cow<'_, [u8]>) -> Response {
    let content_type =
        match Path::new(path).extension().and_then(|ext| ext.to_str()) {
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

    ([(header::CONTENT_TYPE, content_type)], bytes.into_owned()).into_response()
}
