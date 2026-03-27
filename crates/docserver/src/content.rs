use std::path::Path;

use docbert_core::{ingestion, text_util};

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
            let body = text_util::strip_yaml_frontmatter(raw).to_string();
            let title = ingestion::extract_title(&body, Path::new(path));
            ProcessedContent { title, body }
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
