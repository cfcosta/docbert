use std::path::Path;

use docbert_core::document_preparation;

pub struct ProcessedContent {
    pub title: String,
    pub body: String,
}

const SUPPORTED_CONTENT_TYPES: &[&str] = &["text/markdown"];

pub fn is_supported(content_type: &str) -> bool {
    SUPPORTED_CONTENT_TYPES.contains(&content_type)
}

pub fn process(content_type: &str, path: &str, raw: &str) -> ProcessedContent {
    match content_type {
        "text/markdown" => {
            let prepared =
                document_preparation::prepare_markdown(Path::new(path), raw);
            ProcessedContent {
                title: prepared.title,
                body: prepared.searchable_body,
            }
        }
        _ => ProcessedContent {
            title: Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("untitled")
                .to_string(),
            body: raw.to_string(),
        },
    }
}
