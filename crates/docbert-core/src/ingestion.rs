use std::path::Path;

use rayon::prelude::*;
use tantivy::IndexWriter;

use crate::{
    error::Result,
    preparation::{self, SearchDocument},
    tantivy_index::SearchIndex,
    walker::DiscoveredFile,
};

/// A file that could not be read during document loading.
#[derive(Debug, Clone)]
pub struct LoadFailure {
    /// The file that failed to load.
    pub file: DiscoveredFile,
    /// The I/O error message returned while reading the file.
    pub error: String,
}

/// Result of loading discovered files into memory.
#[derive(Debug, Clone, Default)]
pub struct LoadDocumentsResult {
    /// Successfully loaded documents in file order.
    pub documents: Vec<SearchDocument>,
    /// Discovered files that were loaded successfully.
    pub loaded_files: Vec<DiscoveredFile>,
    /// Files that could not be read.
    pub failures: Vec<LoadFailure>,
}

/// Pick a title from file content.
///
/// If the file has a Markdown heading that starts with `# `, the first one wins.
/// Otherwise the filename stem becomes the title.
pub fn extract_title(content: &str, file_path: &Path) -> String {
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

/// Read discovered files into memory and derive the metadata needed for indexing.
///
/// This runs in parallel, extracts titles, and computes stable document IDs.
/// Files that cannot be read or converted are reported in
/// [`LoadDocumentsResult::failures`] so callers can avoid marking them as
/// successfully processed.
pub fn load_documents(
    collection: &str,
    files: &[DiscoveredFile],
) -> LoadDocumentsResult {
    enum LoadOutcome {
        Loaded {
            file: DiscoveredFile,
            document: SearchDocument,
        },
        Failed(LoadFailure),
    }

    let outcomes: Vec<_> = files
        .par_iter()
        .map(|file| {
            match preparation::prepare_supported_filesystem(
                collection,
                &file.relative_path,
                &file.absolute_path,
                file.mtime,
            ) {
                Ok(document) => LoadOutcome::Loaded {
                    file: file.clone(),
                    document,
                },
                Err(error) => LoadOutcome::Failed(LoadFailure {
                    file: file.clone(),
                    error: error.to_string(),
                }),
            }
        })
        .collect();

    let mut result = LoadDocumentsResult::default();
    for outcome in outcomes {
        match outcome {
            LoadOutcome::Loaded { file, document } => {
                result.loaded_files.push(file);
                result.documents.push(document);
            }
            LoadOutcome::Failed(failure) => result.failures.push(failure),
        }
    }

    result
}

/// Write preloaded documents into the Tantivy index.
///
/// Use this when you want to reuse the loaded content for another step, such as
/// embedding, instead of reading every file twice.
pub fn ingest_prepared_documents(
    index: &SearchIndex,
    writer: &mut IndexWriter,
    collection: &str,
    documents: &[SearchDocument],
) -> Result<usize> {
    for doc in documents {
        index.add_document(
            writer,
            &doc.did.full_hex(),
            doc.did.numeric,
            collection,
            &doc.relative_path,
            &doc.title,
            &doc.searchable_body,
            doc.mtime,
        )?;
    }

    writer.commit()?;
    Ok(documents.len())
}

/// Read a batch of discovered files and add them to the Tantivy index.
///
/// For each file, this reads the content, extracts a title, generates document
/// IDs, and writes the result to the index. The batch is committed at the end.
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// # std::fs::write(tmp.path().join("hello.md"), "# Hello\n\nWorld").unwrap();
/// use docbert_core::{SearchIndex, ingestion, walker};
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
    ingest_prepared_documents(index, writer, collection, &loaded.documents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConfigDb, DocumentId, incremental};

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

        assert_eq!(loaded.documents.len(), 1);
        assert_eq!(loaded.loaded_files.len(), 1);
        assert!(loaded.failures.is_empty());

        let doc = &loaded.documents[0];
        assert_eq!(doc.relative_path, "note.md");
        assert_eq!(doc.title, "Title");
        assert!(doc.did.to_string().starts_with('#'));
        assert!(doc.did.numeric > 0);
        assert!(doc.searchable_body.contains("Body content"));
        assert!(doc.raw_content.is_none());
    }

    #[test]
    fn load_documents_strips_frontmatter_before_indexing() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("note.md"),
            "---\ntitle: ignored\ntags:\n  - test\n---\n# Real Title\n\nBody content.",
        )
        .unwrap();

        let files = crate::walker::discover_files(tmp.path()).unwrap();
        let loaded = load_documents("notes", &files);

        let doc = &loaded.documents[0];
        assert_eq!(doc.title, "Real Title");
        assert_eq!(doc.searchable_body, "# Real Title\n\nBody content.");
    }

    #[test]
    fn load_documents_matches_prepare_markdown() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("note.md"),
            "---\ntitle: ignored\n---\n# Heading\n\nBody content.",
        )
        .unwrap();

        let files = crate::walker::discover_files(tmp.path()).unwrap();
        let loaded = load_documents("notes", &files);
        let prepared = preparation::prepare_markdown(
            Path::new("note.md"),
            "---\ntitle: ignored\n---\n# Heading\n\nBody content.",
        );

        let doc = &loaded.documents[0];
        assert_eq!(doc.title, prepared.title);
        assert_eq!(doc.searchable_body, prepared.searchable_body);
    }

    #[test]
    fn load_documents_extracts_pdf_content() {
        let tmp = tempfile::tempdir().unwrap();
        let pdf = pdf_oxide::api::Pdf::from_markdown("# PDF Title\n\nPDF body")
            .unwrap();
        std::fs::write(tmp.path().join("paper.pdf"), pdf.into_bytes()).unwrap();

        let files = crate::walker::discover_files(tmp.path()).unwrap();
        let loaded = load_documents("notes", &files);

        assert_eq!(loaded.documents.len(), 1);
        assert!(loaded.failures.is_empty());
        assert_eq!(loaded.documents[0].relative_path, "paper.pdf");
        assert!(!loaded.documents[0].title.is_empty());
        assert!(loaded.documents[0].searchable_body.contains("PDF"));
        assert!(loaded.documents[0].searchable_body.contains("body"));
    }

    #[test]
    fn invalid_pdf_becomes_a_load_failure() {
        use std::path::PathBuf;

        let tmp = tempfile::tempdir().unwrap();
        let file = DiscoveredFile {
            relative_path: PathBuf::from("broken.pdf"),
            absolute_path: tmp.path().join("broken.pdf"),
            mtime: 1,
        };
        std::fs::write(&file.absolute_path, b"not a pdf").unwrap();

        let loaded = load_documents("notes", &[file]);

        assert!(loaded.documents.is_empty());
        assert!(loaded.loaded_files.is_empty());
        assert_eq!(loaded.failures.len(), 1);
        assert!(!loaded.failures[0].error.is_empty());
    }

    #[test]
    fn unreadable_file_does_not_block_other_readable_files_from_loading() {
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
        assert_eq!(loaded.documents.len(), 1);
        assert_eq!(loaded.documents[0].relative_path, "ok.md");
        assert_eq!(loaded.loaded_files.len(), 1);
        assert_eq!(
            loaded.loaded_files[0].relative_path,
            PathBuf::from("ok.md")
        );
        assert_eq!(loaded.failures.len(), 1);
        assert_eq!(
            loaded.failures[0].file.relative_path,
            PathBuf::from("missing.md")
        );
    }

    #[test]
    fn sync_does_not_advance_metadata_for_unreadable_changed_file() {
        use std::path::PathBuf;

        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let stored = DiscoveredFile {
            relative_path: PathBuf::from("note.md"),
            absolute_path: tmp.path().join("note.md"),
            mtime: 10,
        };
        incremental::store_metadata(&config_db, "notes", &stored).unwrap();

        let changed = DiscoveredFile {
            relative_path: PathBuf::from("note.md"),
            absolute_path: tmp.path().join("missing-note.md"),
            mtime: 20,
        };
        let loaded = load_documents("notes", &[changed]);
        assert!(loaded.documents.is_empty());
        assert!(loaded.loaded_files.is_empty());
        assert_eq!(loaded.failures.len(), 1);

        incremental::batch_store_metadata(
            &config_db,
            "notes",
            &loaded.loaded_files,
        )
        .unwrap();

        let doc_id = DocumentId::new("notes", "note.md");
        let metadata = config_db
            .get_document_metadata_typed(doc_id.numeric)
            .unwrap()
            .unwrap();
        assert_eq!(metadata.mtime, 10);
    }

    #[test]
    fn rebuild_does_not_store_metadata_for_unreadable_file() {
        use std::path::PathBuf;

        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let unreadable = DiscoveredFile {
            relative_path: PathBuf::from("note.md"),
            absolute_path: tmp.path().join("missing-note.md"),
            mtime: 20,
        };
        let loaded = load_documents("notes", &[unreadable]);
        assert!(loaded.documents.is_empty());
        assert!(loaded.loaded_files.is_empty());
        assert_eq!(loaded.failures.len(), 1);

        incremental::batch_store_metadata(
            &config_db,
            "notes",
            &loaded.loaded_files,
        )
        .unwrap();

        let doc_id = DocumentId::new("notes", "note.md");
        assert!(
            config_db
                .get_document_metadata_typed(doc_id.numeric)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn ingest_prepared_documents_indexes_content() {
        let index = SearchIndex::open_in_ram().unwrap();
        let mut writer = index.writer(15_000_000).unwrap();

        let docs = vec![SearchDocument {
            did: DocumentId::new("notes", "x.md"),
            relative_path: "x.md".to_string(),
            title: "Rust Guide".to_string(),
            searchable_body: "Rust ownership and borrowing.".to_string(),
            raw_content: None,
            metadata: None,
            mtime: 1000,
        }];

        let count =
            ingest_prepared_documents(&index, &mut writer, "notes", &docs)
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
