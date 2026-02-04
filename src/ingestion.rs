use std::path::Path;

use rayon::prelude::*;
use tantivy::IndexWriter;

use crate::{
    doc_id::DocumentId,
    error::Result,
    tantivy_index::SearchIndex,
    walker::DiscoveredFile,
};

/// Extract a title from file content.
///
/// Looks for the first markdown heading (line starting with `# `).
/// Falls back to the filename without extension.
fn extract_title(content: &str, file_path: &Path) -> String {
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

/// Ingest a batch of discovered files into the Tantivy search index.
///
/// For each file: reads content, extracts title, generates document IDs,
/// and adds to the index. Commits the batch at the end.
pub fn ingest_files(
    index: &SearchIndex,
    writer: &mut IndexWriter,
    collection: &str,
    files: &[DiscoveredFile],
) -> Result<usize> {
    // Read files in parallel, then index sequentially (IndexWriter is not Send).
    let loaded: Vec<_> = files
        .par_iter()
        .filter_map(|file| {
            let content = std::fs::read_to_string(&file.absolute_path).ok()?;
            let title = extract_title(&content, &file.relative_path);
            let rel_path_str = file.relative_path.to_string_lossy().to_string();
            let doc_id = DocumentId::new(collection, &rel_path_str);
            Some((doc_id, rel_path_str, title, content, file.mtime))
        })
        .collect();

    for (doc_id, rel_path_str, title, content, mtime) in &loaded {
        index.add_document(
            writer,
            &doc_id.to_string(),
            doc_id.numeric,
            collection,
            rel_path_str,
            title,
            content,
            *mtime,
        )?;
    }

    writer.commit()?;
    Ok(loaded.len())
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
