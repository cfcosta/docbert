use std::path::Path;

use rayon::prelude::*;
use tantivy::IndexWriter;

use crate::{
    doc_id::DocumentId,
    error::Result,
    tantivy_index::SearchIndex,
    walker::DiscoveredFile,
};

/// A fully loaded document ready for indexing and embedding.
///
/// Produced by [`load_documents`]. Holds all data derived from a discovered
/// file so downstream stages can reuse it without re-reading from disk.
#[derive(Debug, Clone)]
pub struct LoadedDocument {
    /// Short display document ID (e.g. `#a1b2c3`).
    pub doc_id: String,
    /// Numeric document ID used as database key.
    pub doc_num_id: u64,
    /// Relative path within the collection.
    pub relative_path: String,
    /// Extracted title (first `# ` heading or filename fallback).
    pub title: String,
    /// Full file content.
    pub content: String,
    /// Last modification time (seconds since Unix epoch).
    pub mtime: u64,
}

/// Extract a title from file content.
///
/// Looks for the first markdown heading (line starting with `# `).
/// Falls back to the filename without extension.
pub(crate) fn extract_title(content: &str, file_path: &Path) -> String {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(heading) = trimmed.strip_prefix("# ") {
            let title = heading.trim();
            if !title.is_empty() {
                return title.to_string();
            }
        }
    }

    // Fallback: filename without extension
    file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled")
        .to_string()
}

/// Load discovered files into memory and derive indexable metadata.
///
/// Reads files in parallel, extracts titles, and computes stable document IDs.
/// Files that cannot be read are skipped.
pub fn load_documents(
    collection: &str,
    files: &[DiscoveredFile],
) -> Vec<LoadedDocument> {
    files
        .par_iter()
        .filter_map(|file| {
            let content = std::fs::read_to_string(&file.absolute_path).ok()?;
            let title = extract_title(&content, &file.relative_path);
            let relative_path =
                file.relative_path.to_string_lossy().to_string();
            let did = DocumentId::new(collection, &relative_path);
            Some(LoadedDocument {
                doc_id: did.to_string(),
                doc_num_id: did.numeric,
                relative_path,
                title,
                content,
                mtime: file.mtime,
            })
        })
        .collect()
}

/// Ingest preloaded documents into the Tantivy search index.
///
/// This is useful when the caller needs to reuse loaded content for additional
/// processing (e.g. embedding) without reading files twice.
pub fn ingest_loaded_documents(
    index: &SearchIndex,
    writer: &mut IndexWriter,
    collection: &str,
    documents: &[LoadedDocument],
) -> Result<usize> {
    for doc in documents {
        index.add_document(
            writer,
            &doc.doc_id,
            doc.doc_num_id,
            collection,
            &doc.relative_path,
            &doc.title,
            &doc.content,
            doc.mtime,
        )?;
    }

    writer.commit()?;
    Ok(documents.len())
}

/// Ingest a batch of discovered files into the Tantivy search index.
///
/// For each file: reads content, extracts title, generates document IDs,
/// and adds to the index. Commits the batch at the end.
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// # std::fs::write(tmp.path().join("hello.md"), "# Hello\n\nWorld").unwrap();
/// use docbert::{SearchIndex, ingestion, walker};
///
/// let files = walker::discover_files(tmp.path()).unwrap();
/// let index = SearchIndex::open_in_ram().unwrap();
/// let mut writer = index.writer(15_000_000).unwrap();
///
/// let count = ingestion::ingest_files(&index, &mut writer, "test", &files).unwrap();
/// assert_eq!(count, 1);
///
/// let results = index.search("hello", 10).unwrap();
/// assert_eq!(results[0].title, "Hello");
/// ```
pub fn ingest_files(
    index: &SearchIndex,
    writer: &mut IndexWriter,
    collection: &str,
    files: &[DiscoveredFile],
) -> Result<usize> {
    let loaded = load_documents(collection, files);
    ingest_loaded_documents(index, writer, collection, &loaded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_title_from_heading() {
        let content = "# My Document\n\nSome body text.";
        assert_eq!(extract_title(content, Path::new("file.md")), "My Document");
    }

    #[test]
    fn extract_title_skips_empty_heading() {
        let content = "# \n\nSome text with no real heading.";
        assert_eq!(extract_title(content, Path::new("notes.md")), "notes");
    }

    #[test]
    fn extract_title_fallback_to_filename() {
        let content = "No heading here, just plain text.";
        assert_eq!(
            extract_title(content, Path::new("my-notes.md")),
            "my-notes"
        );
    }

    #[test]
    fn extract_title_empty_content() {
        assert_eq!(extract_title("", Path::new("file.md")), "file");
    }

    #[test]
    fn extract_title_multiple_headings_takes_first() {
        let content = "# First\n\n# Second\n";
        assert_eq!(extract_title(content, Path::new("f.md")), "First");
    }

    #[test]
    fn extract_title_h2_not_matched() {
        let content = "## Sub Heading Only\n\nSome text.";
        assert_eq!(extract_title(content, Path::new("doc.md")), "doc");
    }

    #[test]
    fn extract_title_heading_with_leading_whitespace() {
        let content = "  # Indented Title\n";
        assert_eq!(extract_title(content, Path::new("f.md")), "Indented Title");
    }

    #[test]
    fn extract_title_no_extension() {
        let content = "no heading";
        assert_eq!(extract_title(content, Path::new("README")), "README");
    }

    #[test]
    fn extract_title_nested_path() {
        let content = "no heading";
        assert_eq!(extract_title(content, Path::new("a/b/c/file.md")), "file");
    }

    #[test]
    fn ingest_and_search() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("hello.md"),
            "# Hello World\n\nThis is about greeting people.",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("rust.txt"),
            "Rust is a systems programming language.",
        )
        .unwrap();

        let files = crate::walker::discover_files(tmp.path()).unwrap();
        let index = SearchIndex::open_in_ram().unwrap();
        let mut writer = index.writer(15_000_000).unwrap();

        let count = ingest_files(&index, &mut writer, "test", &files).unwrap();
        assert_eq!(count, 2);

        let results = index.search("hello", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].title, "Hello World");
        assert_eq!(results[0].collection, "test");
    }

    #[test]
    fn load_documents_builds_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("note.md"),
            "# Title\n\nBody content for loading.",
        )
        .unwrap();

        let files = crate::walker::discover_files(tmp.path()).unwrap();
        let loaded = load_documents("notes", &files);

        assert_eq!(loaded.len(), 1);
        let doc = &loaded[0];
        assert_eq!(doc.relative_path, "note.md");
        assert_eq!(doc.title, "Title");
        assert!(doc.doc_id.starts_with('#'));
        assert!(doc.doc_num_id > 0);
        assert!(doc.content.contains("Body content"));
    }

    #[test]
    fn load_documents_skips_unreadable_files() {
        use std::path::PathBuf;

        let tmp = tempfile::tempdir().unwrap();
        let valid_path = tmp.path().join("ok.md");
        std::fs::write(&valid_path, "# Ok\n\nReadable").unwrap();

        let files = vec![
            DiscoveredFile {
                relative_path: PathBuf::from("ok.md"),
                absolute_path: valid_path,
                mtime: 1,
            },
            DiscoveredFile {
                relative_path: PathBuf::from("missing.md"),
                absolute_path: tmp.path().join("missing.md"),
                mtime: 1,
            },
        ];

        let loaded = load_documents("notes", &files);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].relative_path, "ok.md");
    }

    #[test]
    fn ingest_loaded_documents_indexes_content() {
        let index = SearchIndex::open_in_ram().unwrap();
        let mut writer = index.writer(15_000_000).unwrap();

        let docs = vec![LoadedDocument {
            doc_id: "#abc123".to_string(),
            doc_num_id: 123,
            relative_path: "x.md".to_string(),
            title: "Rust Guide".to_string(),
            content: "Rust ownership and borrowing.".to_string(),
            mtime: 1000,
        }];

        let count =
            ingest_loaded_documents(&index, &mut writer, "notes", &docs)
                .unwrap();
        assert_eq!(count, 1);

        let results = index.search("ownership", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].collection, "notes");
        assert_eq!(results[0].path, "x.md");
    }

    #[test]
    fn ingest_updates_existing() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("doc.md");
        std::fs::write(&file_path, "# Original\n\nOriginal content.").unwrap();

        let index = SearchIndex::open_in_ram().unwrap();
        let mut writer = index.writer(15_000_000).unwrap();

        let files = crate::walker::discover_files(tmp.path()).unwrap();
        ingest_files(&index, &mut writer, "test", &files).unwrap();

        // Update the file
        std::fs::write(&file_path, "# Updated\n\nNew content.").unwrap();
        let files = crate::walker::discover_files(tmp.path()).unwrap();
        ingest_files(&index, &mut writer, "test", &files).unwrap();

        let results = index.search("content", 10).unwrap();
        // Should only have one doc (the updated version)
        let doc_ids: Vec<_> = results.iter().map(|r| &r.doc_id).collect();
        let unique_ids: std::collections::HashSet<_> = doc_ids.iter().collect();
        assert_eq!(unique_ids.len(), 1);
    }
}
