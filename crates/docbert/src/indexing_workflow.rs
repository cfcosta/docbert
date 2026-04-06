use std::{collections::HashSet, path::Path};

use docbert_core::{
    ConfigDb,
    error,
    ingestion::{self, LoadFailure},
    preparation::SearchDocument,
    walker::{self, DiscoveredFile},
};

use crate::{cli, collection_snapshots};

#[derive(Debug, Default)]
pub(crate) struct DocumentLoadBatch {
    pub documents: Vec<SearchDocument>,
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

#[derive(Debug, Default)]
pub(crate) struct SyncSelection {
    pub new_files: Vec<DiscoveredFile>,
    pub changed_files: Vec<DiscoveredFile>,
    pub deleted_ids: Vec<u64>,
}

pub(crate) fn select_sync_work(
    config_db: &ConfigDb,
    collection: &str,
    root: &Path,
) -> error::Result<SyncSelection> {
    let discovered = walker::discover_files(root)?;
    let change = collection_snapshots::compute_collection_snapshot_change_for_discovered(
        config_db,
        collection,
        &discovered,
    )?;

    let new_paths: HashSet<&str> =
        change.diff.new_paths.iter().map(String::as_str).collect();
    let changed_paths: HashSet<&str> = change
        .diff
        .changed_paths
        .iter()
        .map(String::as_str)
        .collect();

    let mut selection = SyncSelection {
        deleted_ids: change.diff.deleted_ids,
        ..SyncSelection::default()
    };

    for file in discovered {
        let relative_path = file.relative_path.to_string_lossy();
        if new_paths.contains(relative_path.as_ref()) {
            selection.new_files.push(file);
        } else if changed_paths.contains(relative_path.as_ref()) {
            selection.changed_files.push(file);
        }
    }

    Ok(selection)
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
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();

        std::fs::write(root.join("a.md"), "# A\n\nOriginal").unwrap();
        std::fs::write(root.join("b.md"), "# B\n\nKeep").unwrap();
        std::fs::write(root.join("deleted.md"), "# Gone\n\nDelete me").unwrap();

        let initial_files =
            docbert_core::walker::discover_files(&root).unwrap();
        incremental::batch_store_metadata(&config_db, "notes", &initial_files)
            .unwrap();
        let initial_snapshot = docbert_core::merkle::build_collection_snapshot(
            "notes",
            &initial_files,
        )
        .unwrap();
        config_db
            .set_collection_merkle_snapshot("notes", &initial_snapshot)
            .unwrap();

        std::thread::sleep(std::time::Duration::from_secs(1));
        std::fs::write(root.join("a.md"), "# A\n\nUpdated").unwrap();
        std::fs::remove_file(root.join("deleted.md")).unwrap();

        let selection = select_sync_work(&config_db, "notes", &root).unwrap();
        assert_eq!(selection.new_files.len(), 0);
        assert_eq!(selection.changed_files.len(), 1);
        assert_eq!(
            selection.changed_files[0].relative_path,
            PathBuf::from("a.md")
        );
        let deleted_id = DocumentId::new("notes", "deleted.md");
        assert_eq!(selection.deleted_ids, vec![deleted_id.numeric]);

        let batch = load_sync_batch("notes", &selection.changed_files);
        incremental::batch_store_metadata(
            &config_db,
            "notes",
            &batch.metadata_files,
        )
        .unwrap();
        config_db
            .batch_remove_document_metadata(&selection.deleted_ids)
            .unwrap();

        let updated_a = config_db
            .get_document_metadata_typed(
                DocumentId::new("notes", "a.md").numeric,
            )
            .unwrap()
            .unwrap();
        assert!(updated_a.mtime > 0);

        let unchanged_b = config_db
            .get_document_metadata_typed(
                DocumentId::new("notes", "b.md").numeric,
            )
            .unwrap()
            .unwrap();
        assert_eq!(unchanged_b.relative_path, "b.md");

        assert!(
            config_db
                .get_document_metadata_typed(deleted_id.numeric)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn collect_chunks_skips_frontmatter_only_docs() {
        let docs = vec![SearchDocument {
            did: DocumentId::new("notes", "note.md"),
            relative_path: "note.md".to_string(),
            title: "note".to_string(),
            searchable_body: String::new(),
            raw_content: None,
            metadata: None,
            mtime: 1,
        }];

        let chunking_config = docbert_core::chunking::ChunkingConfig {
            chunk_size: 100,
            overlap: 0,
            document_length: None,
        };
        let mut processed = 0;
        let chunks = docbert_core::preparation::collect_chunks(
            &docs,
            chunking_config,
            |_| {
                processed += 1;
            },
        );

        assert!(chunks.is_empty());
        assert_eq!(processed, 1);
    }

    #[test]
    fn load_rebuild_batch_returns_prepared_search_documents() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("note.md"), "# Note\n\nBody").unwrap();
        let files = docbert_core::walker::discover_files(&root).unwrap();

        let batch =
            load_rebuild_batch("notes", &files, &rebuild_args(false, false));

        assert_eq!(batch.documents.len(), 1);
        assert_eq!(batch.documents[0].relative_path, "note.md");
        assert_eq!(batch.documents[0].title, "Note");
        assert_eq!(batch.documents[0].searchable_body, "# Note\n\nBody");
        assert!(batch.documents[0].raw_content.is_none());
    }

    #[test]
    fn load_sync_batch_returns_prepared_search_documents() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("note.md"), "# Note\n\nBody").unwrap();
        let files = docbert_core::walker::discover_files(tmp.path()).unwrap();

        let batch = load_sync_batch("notes", &files);

        assert_eq!(batch.documents.len(), 1);
        assert_eq!(batch.documents[0].relative_path, "note.md");
        assert_eq!(batch.documents[0].title, "Note");
        assert_eq!(batch.documents[0].searchable_body, "# Note\n\nBody");
        assert!(batch.documents[0].raw_content.is_none());
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
