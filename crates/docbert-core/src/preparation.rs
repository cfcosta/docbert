use std::{fs, path::Path};

use pdf_oxide::{converters::ConversionOptions, document::PdfDocument};

use crate::{
    chunking::{self, ChunkingConfig},
    doc_id::DocumentId,
    ingestion,
    text_util,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkdownBody {
    pub title: String,
    pub searchable_body: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchDocument {
    pub did: DocumentId,
    pub relative_path: String,
    pub title: String,
    pub searchable_body: String,
    pub raw_content: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub mtime: u64,
}

pub fn prepare_markdown(
    relative_path: &Path,
    raw_markdown: &str,
) -> MarkdownBody {
    let searchable_body =
        text_util::strip_yaml_frontmatter(raw_markdown).to_string();
    let title = ingestion::extract_title(&searchable_body, relative_path);

    MarkdownBody {
        title,
        searchable_body,
    }
}

pub fn prepare_uploaded(
    collection: &str,
    relative_path: &str,
    raw_markdown: &str,
    metadata: Option<serde_json::Value>,
    mtime: u64,
) -> SearchDocument {
    let prepared = prepare_markdown(Path::new(relative_path), raw_markdown);
    let did = DocumentId::new(collection, relative_path);

    SearchDocument {
        did,
        relative_path: relative_path.to_string(),
        title: prepared.title,
        searchable_body: prepared.searchable_body,
        raw_content: Some(raw_markdown.to_string()),
        metadata,
        mtime,
    }
}

pub fn prepare_filesystem(
    collection: &str,
    relative_path: &Path,
    raw_markdown: &str,
    mtime: u64,
) -> SearchDocument {
    let prepared = prepare_markdown(relative_path, raw_markdown);
    let relative_path = relative_path.to_string_lossy().to_string();
    let did = DocumentId::new(collection, &relative_path);

    SearchDocument {
        did,
        relative_path,
        title: prepared.title,
        searchable_body: prepared.searchable_body,
        raw_content: None,
        metadata: None,
        mtime,
    }
}

pub fn load_preview_content(
    relative_path: &Path,
    full_path: &Path,
) -> crate::Result<String> {
    if is_pdf(relative_path) {
        return extract_pdf_markdown(&fs::read(full_path)?);
    }

    Ok(fs::read_to_string(full_path)?)
}

pub fn prepare_supported_filesystem(
    collection: &str,
    relative_path: &Path,
    full_path: &Path,
    mtime: u64,
) -> crate::Result<SearchDocument> {
    let raw_content = load_preview_content(relative_path, full_path)?;
    Ok(prepare_filesystem(
        collection,
        relative_path,
        &raw_content,
        mtime,
    ))
}

pub fn extract_pdf_markdown(pdf_bytes: &[u8]) -> crate::Result<String> {
    let mut doc = PdfDocument::from_bytes(pdf_bytes.to_vec())?;
    let page_count = doc.page_count()?;
    let options = ConversionOptions::default();
    let mut pages = Vec::with_capacity(page_count);

    for page_index in 0..page_count {
        let markdown = doc.to_markdown(page_index, &options)?;
        let trimmed = markdown.trim();
        if !trimmed.is_empty() {
            pages.push(trimmed.to_string());
        }
    }

    if !pages.is_empty() {
        return Ok(normalize_pdf_markdown(&pages.join("\n\n")));
    }

    let mut text_pages = Vec::with_capacity(page_count);
    for page_index in 0..page_count {
        let text = doc.extract_text(page_index)?;
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            text_pages.push(trimmed.to_string());
        }
    }

    Ok(normalize_pdf_markdown(&text_pages.join("\n\n")))
}

fn normalize_pdf_markdown(content: &str) -> String {
    let trimmed = content.trim();
    if trimmed.is_empty()
        || trimmed
            .lines()
            .any(|line| line.trim_start().starts_with("# "))
    {
        return trimmed.to_string();
    }

    let Some(first_line) = trimmed.lines().find_map(|line| {
        let candidate = line.trim();
        (!candidate.is_empty()).then_some(candidate)
    }) else {
        return String::new();
    };

    let title = first_line.trim_start_matches('#').trim();
    if title.is_empty() {
        return trimmed.to_string();
    }

    let rest = trimmed
        .strip_prefix(first_line)
        .unwrap_or(trimmed)
        .trim_start_matches(['\r', '\n']);
    if rest.is_empty() {
        format!("# {title}")
    } else {
        format!("# {title}\n\n{rest}")
    }
}

fn is_pdf(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
}

pub fn embedding_chunks(
    document: &SearchDocument,
    chunking_config: ChunkingConfig,
) -> Vec<(u64, String)> {
    chunking::chunk_text(
        &document.searchable_body,
        chunking_config.chunk_size,
        chunking_config.overlap,
    )
    .into_iter()
    .map(|chunk| {
        (
            chunking::chunk_doc_id(document.did.numeric, chunk.index),
            chunk.text,
        )
    })
    .collect()
}

pub fn collect_chunks<F>(
    documents: &[SearchDocument],
    chunking_config: ChunkingConfig,
    mut on_document_processed: F,
) -> Vec<(u64, String)>
where
    F: FnMut(usize),
{
    let mut docs_to_embed = Vec::new();

    for (i, document) in documents.iter().enumerate() {
        docs_to_embed.extend(embedding_chunks(document, chunking_config));
        on_document_processed(i + 1);
    }

    docs_to_embed
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn prepare_markdown_strips_frontmatter() {
        let prepared = prepare_markdown(
            Path::new("note.md"),
            "---\ntitle: ignored\n---\n# Real Title\n\nBody",
        );

        assert_eq!(prepared.title, "Real Title");
        assert_eq!(prepared.searchable_body, "# Real Title\n\nBody");
    }

    #[test]
    fn prepare_markdown_uses_first_h1_as_title() {
        let prepared =
            prepare_markdown(Path::new("note.md"), "# Hello\n\n# Later");

        assert_eq!(prepared.title, "Hello");
    }

    #[test]
    fn prepare_markdown_falls_back_to_filename() {
        let prepared =
            prepare_markdown(Path::new("docs/note.md"), "plain body");

        assert_eq!(prepared.title, "note");
        assert_eq!(prepared.searchable_body, "plain body");
    }

    #[test]
    fn prepare_markdown_frontmatter_only_yields_empty_body() {
        let prepared =
            prepare_markdown(Path::new("note.md"), "---\ntitle: Hidden\n---\n");

        assert_eq!(prepared.title, "note");
        assert!(prepared.searchable_body.is_empty());
    }

    #[test]
    fn prepare_uploaded_preserves_raw_content() {
        let prepared = prepare_uploaded(
            "notes",
            "note.md",
            "# Hello\n\nBody",
            Some(serde_json::json!({ "topic": "rust" })),
            0,
        );

        assert_eq!(prepared.raw_content, Some("# Hello\n\nBody".to_string()));
        assert_eq!(
            prepared.metadata,
            Some(serde_json::json!({ "topic": "rust" }))
        );
    }

    #[test]
    fn prepare_filesystem_omits_raw_content() {
        let prepared = prepare_filesystem(
            "notes",
            Path::new("note.md"),
            "# Hello\n\nBody",
            42,
        );

        assert_eq!(prepared.raw_content, None);
        assert_eq!(prepared.mtime, 42);
    }

    #[test]
    fn uploaded_and_filesystem_documents_share_title_and_searchable_body() {
        let uploaded = prepare_uploaded(
            "notes",
            "note.md",
            "---\ntitle: ignored\n---\n# Hello\n\nBody",
            None,
            0,
        );
        let filesystem = prepare_filesystem(
            "notes",
            Path::new("note.md"),
            "---\ntitle: ignored\n---\n# Hello\n\nBody",
            10,
        );

        assert_eq!(uploaded.did.numeric, filesystem.did.numeric);
        assert_eq!(uploaded.title, filesystem.title);
        assert_eq!(uploaded.searchable_body, filesystem.searchable_body);
    }

    #[test]
    fn embedding_chunks_uses_chunk_doc_ids() {
        let document = prepare_filesystem(
            "notes",
            Path::new("note.md"),
            &"a".repeat(DEFAULT_TEST_CHUNK_SIZE * 3),
            1,
        );
        let chunks = embedding_chunks(&document, test_chunking_config());

        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].0, document.did.numeric);
        assert_eq!(
            chunks[1].0,
            chunking::chunk_doc_id(document.did.numeric, 1)
        );
    }

    #[test]
    fn embedding_chunks_returns_empty_for_empty_searchable_body() {
        let document = prepare_filesystem(
            "notes",
            Path::new("note.md"),
            "---\ntitle: ignored\n---\n",
            1,
        );
        let chunks = embedding_chunks(&document, test_chunking_config());

        assert!(chunks.is_empty());
    }

    #[test]
    fn collect_chunks_preserves_document_order() {
        let first =
            prepare_filesystem("notes", Path::new("a.md"), "# A\n\nAlpha", 1);
        let second =
            prepare_filesystem("notes", Path::new("b.md"), "# B\n\nBeta", 2);
        let mut processed = Vec::new();

        let chunks = collect_chunks(
            &[first.clone(), second.clone()],
            test_chunking_config(),
            |count| processed.push(count),
        );

        assert_eq!(processed, vec![1, 2]);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].0, first.did.numeric);
        assert_eq!(chunks[1].0, second.did.numeric);
    }

    const DEFAULT_TEST_CHUNK_SIZE: usize = 100;

    fn test_chunking_config() -> ChunkingConfig {
        ChunkingConfig {
            chunk_size: DEFAULT_TEST_CHUNK_SIZE,
            overlap: 0,
            document_length: None,
        }
    }
}
