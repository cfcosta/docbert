use docbert_core::{
    ConfigDb,
    chunking::{self, ChunkingConfig},
    error,
    ingestion::{self, LoadFailure, LoadedDocument},
    walker::DiscoveredFile,
};

use crate::cli;

#[derive(Debug, Default)]
pub(crate) struct DocumentLoadBatch {
    pub documents: Vec<LoadedDocument>,
    pub metadata_files: Vec<DiscoveredFile>,
    pub failures: Vec<LoadFailure>,
}

pub(crate) fn resolve_target_collections(
    config_db: &ConfigDb,
    collection: Option<&str>,
) -> error::Result<Vec<(String, String)>> {
    if let Some(name) = collection {
        let path = config_db.get_collection(name)?.ok_or_else(|| {
            error::Error::NotFound {
                kind: "collection",
                name: name.to_string(),
            }
        })?;
        Ok(vec![(name.to_string(), path)])
    } else {
        config_db.list_collections()
    }
}

pub(crate) fn load_rebuild_batch(
    collection: &str,
    files: &[DiscoveredFile],
    args: &cli::RebuildArgs,
) -> DocumentLoadBatch {
    if args.embeddings_only && args.index_only {
        DocumentLoadBatch {
            metadata_files: files.to_vec(),
            ..DocumentLoadBatch::default()
        }
    } else {
        let result = ingestion::load_documents(collection, files);
        DocumentLoadBatch {
            documents: result.documents,
            metadata_files: result.loaded_files,
            failures: result.failures,
        }
    }
}

pub(crate) fn load_sync_batch(
    collection: &str,
    files: &[DiscoveredFile],
) -> DocumentLoadBatch {
    let result = ingestion::load_documents(collection, files);
    DocumentLoadBatch {
        documents: result.documents,
        metadata_files: result.loaded_files,
        failures: result.failures,
    }
}

pub(crate) fn chunk_documents_for_embedding<F>(
    documents: &[LoadedDocument],
    chunking_config: ChunkingConfig,
    mut on_document_processed: F,
) -> Vec<(u64, String)>
where
    F: FnMut(usize),
{
    let mut docs_to_embed = Vec::new();

    for (i, document) in documents.iter().enumerate() {
        let chunks = chunking::chunk_text(
            &document.content,
            chunking_config.chunk_size,
            chunking_config.overlap,
        );
        for chunk in chunks {
            let chunk_id =
                chunking::chunk_doc_id(document.doc_num_id, chunk.index);
            docs_to_embed.push((chunk_id, chunk.text));
        }
        on_document_processed(i + 1);
    }

    docs_to_embed
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use docbert_core::{doc_id::DocumentId, incremental};

    use super::*;

    fn rebuild_args(
        index_only: bool,
        embeddings_only: bool,
    ) -> cli::RebuildArgs {
        cli::RebuildArgs {
            collection: Some("notes".to_string()),
            embeddings_only,
            index_only,
        }
    }

    #[test]
    fn rebuild_stores_metadata_for_all_discovered_files() {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.md"), "# A\n\nFirst").unwrap();
        std::fs::write(root.join("b.md"), "# B\n\nSecond").unwrap();

        let files = docbert_core::walker::discover_files(&root).unwrap();
        let batch =
            load_rebuild_batch("notes", &files, &rebuild_args(false, false));

        incremental::batch_store_metadata(
            &config_db,
            "notes",
            &batch.metadata_files,
        )
        .unwrap();

        let a_id = DocumentId::new("notes", "a.md");
        let b_id = DocumentId::new("notes", "b.md");
        assert!(
            config_db
                .get_document_metadata_typed(a_id.numeric)
                .unwrap()
                .is_some()
        );
        assert!(
            config_db
                .get_document_metadata_typed(b_id.numeric)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn sync_stores_metadata_only_for_processed_files_and_deletes_removed_ids() {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();

        let stored_a = DiscoveredFile {
            relative_path: PathBuf::from("a.md"),
            absolute_path: tmp.path().join("a.md"),
            mtime: 10,
        };
        let stored_b = DiscoveredFile {
            relative_path: PathBuf::from("b.md"),
            absolute_path: tmp.path().join("b.md"),
            mtime: 15,
        };
        let stored_deleted = DiscoveredFile {
            relative_path: PathBuf::from("deleted.md"),
            absolute_path: tmp.path().join("deleted.md"),
            mtime: 20,
        };

        incremental::store_metadata(&config_db, "notes", &stored_a).unwrap();
        incremental::store_metadata(&config_db, "notes", &stored_b).unwrap();
        incremental::store_metadata(&config_db, "notes", &stored_deleted)
            .unwrap();

        std::fs::write(tmp.path().join("a.md"), "# A\n\nUpdated").unwrap();
        let changed_a = DiscoveredFile {
            relative_path: PathBuf::from("a.md"),
            absolute_path: tmp.path().join("a.md"),
            mtime: 99,
        };

        let batch = load_sync_batch("notes", &[changed_a]);
        incremental::batch_store_metadata(
            &config_db,
            "notes",
            &batch.metadata_files,
        )
        .unwrap();

        let deleted_id = DocumentId::new("notes", "deleted.md");
        config_db
            .batch_remove_document_metadata(&[deleted_id.numeric])
            .unwrap();

        let updated_a = config_db
            .get_document_metadata_typed(DocumentId::new("notes", "a.md").numeric)
            .unwrap()
            .unwrap();
        assert_eq!(updated_a.mtime, 99);

        let unchanged_b = config_db
            .get_document_metadata_typed(DocumentId::new("notes", "b.md").numeric)
            .unwrap()
            .unwrap();
        assert_eq!(unchanged_b.mtime, 15);

        assert!(
            config_db
                .get_document_metadata_typed(deleted_id.numeric)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn chunk_documents_for_embedding_skips_frontmatter_only_docs() {
        let docs = vec![LoadedDocument {
            doc_id: "#abc123".to_string(),
            doc_num_id: 123,
            relative_path: "note.md".to_string(),
            title: "note".to_string(),
            content: String::new(),
            mtime: 1,
        }];

        let chunking_config = ChunkingConfig {
            chunk_size: 100,
            overlap: 0,
            document_length: None,
        };
        let mut processed = 0;
        let chunks =
            chunk_documents_for_embedding(&docs, chunking_config, |_| {
                processed += 1;
            });

        assert!(chunks.is_empty());
        assert_eq!(processed, 1);
    }

    #[test]
    fn rebuild_flag_matrix_preserves_index_only_and_embeddings_only_behavior() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("note.md"), "# Note\n\nBody").unwrap();
        let files = docbert_core::walker::discover_files(&root).unwrap();

        let normal =
            load_rebuild_batch("notes", &files, &rebuild_args(false, false));
        assert_eq!(normal.documents.len(), 1);
        assert_eq!(normal.metadata_files.len(), 1);

        let index_only =
            load_rebuild_batch("notes", &files, &rebuild_args(true, false));
        assert_eq!(index_only.documents.len(), 1);
        assert_eq!(index_only.metadata_files.len(), 1);

        let embeddings_only =
            load_rebuild_batch("notes", &files, &rebuild_args(false, true));
        assert_eq!(embeddings_only.documents.len(), 1);
        assert_eq!(embeddings_only.metadata_files.len(), 1);

        let skip_loading =
            load_rebuild_batch("notes", &files, &rebuild_args(true, true));
        assert!(skip_loading.documents.is_empty());
        assert_eq!(skip_loading.metadata_files.len(), 1);
        assert!(skip_loading.failures.is_empty());
    }
}
