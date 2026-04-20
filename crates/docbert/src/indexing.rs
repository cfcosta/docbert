use std::{collections::HashSet, path::Path};

use docbert_core::{
    ConfigDb,
    error,
    ingestion::{self, LoadFailure},
    merkle::Snapshot,
    preparation::SearchDocument,
    walker::{self, DiscoveredFile},
};

use crate::{cli, snapshots};

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

#[derive(Debug)]
pub(crate) struct SyncSelection {
    pub new_files: Vec<DiscoveredFile>,
    pub changed_files: Vec<DiscoveredFile>,
    pub deleted_ids: Vec<u64>,
    pub current_snapshot: Snapshot,
}

pub(crate) fn select_sync_work(
    config_db: &ConfigDb,
    collection: &str,
    root: &Path,
) -> error::Result<SyncSelection> {
    let discovered = walker::discover_files(root)?;
    let change = snapshots::compute_collection_snapshot_change_for_discovered(
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
        new_files: Vec::new(),
        changed_files: Vec::new(),
        deleted_ids: change.diff.deleted_ids,
        current_snapshot: change.current_snapshot,
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

pub(crate) fn finalize_sync_snapshot(
    config_db: &ConfigDb,
    selection: &SyncSelection,
    sync_result: error::Result<()>,
) -> error::Result<()> {
    sync_result?;
    snapshots::replace_collection_snapshot(
        config_db,
        &selection.current_snapshot,
    )
}

pub(crate) fn finalize_rebuild_snapshot(
    config_db: &ConfigDb,
    collection: &str,
    root: &Path,
    rebuild_result: error::Result<()>,
) -> error::Result<()> {
    rebuild_result?;
    let snapshot = snapshots::compute_collection_snapshot(collection, root)?;
    snapshots::replace_collection_snapshot(config_db, &snapshot)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use docbert_core::{doc_id::DocumentId, incremental};

    use super::*;

    fn seed_snapshot(
        config_db: &ConfigDb,
        collection: &str,
        root: &std::path::Path,
    ) -> docbert_core::merkle::Snapshot {
        let files = docbert_core::walker::discover_files(root).unwrap();
        let snapshot =
            docbert_core::merkle::build_snapshot(collection, &files).unwrap();
        config_db
            .set_collection_merkle_snapshot(collection, &snapshot)
            .unwrap();
        snapshot
    }

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

    fn rebuild_mode_cases() -> [(&'static str, cli::RebuildArgs, usize); 3] {
        [
            ("full", rebuild_args(false, false), 1),
            ("index-only", rebuild_args(true, false), 1),
            ("embeddings-only", rebuild_args(false, true), 1),
        ]
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
        let initial_snapshot = seed_snapshot(&config_db, "notes", &root);

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
        assert_eq!(
            config_db.get_collection_merkle_snapshot("notes").unwrap(),
            Some(initial_snapshot)
        );
    }

    #[test]
    fn finalize_sync_snapshot_replaces_snapshot_on_success() {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.md"), "original").unwrap();

        let original_snapshot = seed_snapshot(&config_db, "notes", &root);

        std::thread::sleep(std::time::Duration::from_secs(1));
        std::fs::write(root.join("a.md"), "updated").unwrap();

        let selection = select_sync_work(&config_db, "notes", &root).unwrap();
        assert_ne!(selection.current_snapshot, original_snapshot);

        finalize_sync_snapshot(&config_db, &selection, Ok(())).unwrap();

        assert_eq!(
            config_db.get_collection_merkle_snapshot("notes").unwrap(),
            Some(selection.current_snapshot)
        );
    }

    #[test]
    fn finalize_sync_snapshot_preserves_snapshot_on_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.md"), "original").unwrap();

        let original_snapshot = seed_snapshot(&config_db, "notes", &root);

        std::thread::sleep(std::time::Duration::from_secs(1));
        std::fs::write(root.join("a.md"), "updated").unwrap();

        let selection = select_sync_work(&config_db, "notes", &root).unwrap();
        let failure = Err(error::Error::Config("sync failed".to_string()));

        assert!(
            finalize_sync_snapshot(&config_db, &selection, failure).is_err()
        );
        assert_eq!(
            config_db.get_collection_merkle_snapshot("notes").unwrap(),
            Some(original_snapshot)
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

        for (mode_name, args, expected_documents) in rebuild_mode_cases() {
            let batch = load_rebuild_batch("notes", &files, &args);
            assert_eq!(
                batch.documents.len(),
                expected_documents,
                "unexpected document count for {mode_name}"
            );
            assert_eq!(
                batch.metadata_files.len(),
                1,
                "unexpected metadata file count for {mode_name}"
            );
        }

        let skip_loading =
            load_rebuild_batch("notes", &files, &rebuild_args(true, true));
        assert!(skip_loading.documents.is_empty());
        assert_eq!(skip_loading.metadata_files.len(), 1);
        assert!(skip_loading.failures.is_empty());
    }

    #[test]
    fn finalize_rebuild_snapshot_replaces_snapshot_for_all_successful_modes() {
        for (mode_name, args, expected_documents) in rebuild_mode_cases() {
            let tmp = tempfile::tempdir().unwrap();
            let config_db =
                ConfigDb::open(&tmp.path().join("config.db")).unwrap();
            let root = tmp.path().join("notes");
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join("note.md"), "# Note\n\nOriginal").unwrap();

            let original_snapshot = seed_snapshot(&config_db, "notes", &root);

            std::thread::sleep(std::time::Duration::from_secs(1));
            std::fs::write(root.join("note.md"), "# Note\n\nUpdated").unwrap();
            let files = docbert_core::walker::discover_files(&root).unwrap();
            let batch = load_rebuild_batch("notes", &files, &args);
            assert_eq!(
                batch.documents.len(),
                expected_documents,
                "unexpected document count for {mode_name}"
            );
            assert_eq!(batch.metadata_files.len(), 1);

            finalize_rebuild_snapshot(&config_db, "notes", &root, Ok(()))
                .unwrap();

            let stored_snapshot = config_db
                .get_collection_merkle_snapshot("notes")
                .unwrap()
                .unwrap();
            assert_ne!(
                stored_snapshot, original_snapshot,
                "snapshot did not change for {mode_name}"
            );
        }
    }

    #[test]
    fn finalize_rebuild_snapshot_preserves_snapshot_for_all_failed_modes() {
        for (mode_name, args, expected_documents) in rebuild_mode_cases() {
            let tmp = tempfile::tempdir().unwrap();
            let config_db =
                ConfigDb::open(&tmp.path().join("config.db")).unwrap();
            let root = tmp.path().join("notes");
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join("note.md"), "# Note\n\nOriginal").unwrap();

            let original_snapshot = seed_snapshot(&config_db, "notes", &root);

            std::thread::sleep(std::time::Duration::from_secs(1));
            std::fs::write(root.join("note.md"), "# Note\n\nUpdated").unwrap();
            let files = docbert_core::walker::discover_files(&root).unwrap();
            let batch = load_rebuild_batch("notes", &files, &args);
            assert_eq!(
                batch.documents.len(),
                expected_documents,
                "unexpected document count for {mode_name}"
            );
            assert_eq!(batch.metadata_files.len(), 1);

            let failure = Err(error::Error::Config(format!(
                "rebuild failed in {mode_name}"
            )));
            assert!(
                finalize_rebuild_snapshot(&config_db, "notes", &root, failure)
                    .is_err()
            );
            assert_eq!(
                config_db.get_collection_merkle_snapshot("notes").unwrap(),
                Some(original_snapshot),
                "snapshot changed unexpectedly for {mode_name}"
            );
        }
    }

    #[test]
    fn removing_snapshot_makes_sync_rediscover_all_files() {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.md"), "# A\n\nAlpha").unwrap();
        std::fs::write(root.join("b.md"), "# B\n\nBeta").unwrap();

        // Seed metadata + snapshot as if a sync already happened
        let initial_files =
            docbert_core::walker::discover_files(&root).unwrap();
        incremental::batch_store_metadata(&config_db, "notes", &initial_files)
            .unwrap();
        seed_snapshot(&config_db, "notes", &root);

        // Simulate what collection_remove does: clear metadata + snapshot
        let doc_ids: Vec<u64> = config_db
            .list_all_document_metadata_typed()
            .unwrap()
            .into_iter()
            .filter_map(|(doc_id, meta)| {
                (meta.collection == "notes").then_some(doc_id)
            })
            .collect();
        config_db.batch_remove_document_state(&doc_ids).unwrap();
        config_db
            .remove_collection_merkle_snapshot("notes")
            .unwrap();

        // Now sync should see all files as new (not "up to date")
        let selection = select_sync_work(&config_db, "notes", &root).unwrap();
        assert_eq!(
            selection.new_files.len(),
            2,
            "sync should rediscover all files after snapshot removal"
        );
        assert!(selection.changed_files.is_empty());
        assert!(selection.deleted_ids.is_empty());
    }

    #[test]
    fn stale_snapshot_without_fix_would_skip_sync() {
        // Demonstrates the bug: if the snapshot is NOT removed, sync
        // sees no work even though all metadata is gone.
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.md"), "# A\n\nAlpha").unwrap();

        let initial_files =
            docbert_core::walker::discover_files(&root).unwrap();
        incremental::batch_store_metadata(&config_db, "notes", &initial_files)
            .unwrap();
        seed_snapshot(&config_db, "notes", &root);

        // Clear metadata but KEEP the stale snapshot (the bug scenario)
        let doc_ids: Vec<u64> = config_db
            .list_all_document_metadata_typed()
            .unwrap()
            .into_iter()
            .map(|(doc_id, _)| doc_id)
            .collect();
        config_db.batch_remove_document_state(&doc_ids).unwrap();
        // NOTE: NOT removing snapshot here — simulating the old buggy code

        let selection = select_sync_work(&config_db, "notes", &root).unwrap();
        // With the stale snapshot, sync thinks nothing changed
        assert!(
            selection.new_files.is_empty()
                && selection.changed_files.is_empty()
                && selection.deleted_ids.is_empty(),
            "stale snapshot should cause sync to see no work (demonstrating the bug)"
        );
    }

    #[hegel::test(test_cases = 30)]
    fn prop_removing_snapshot_always_causes_full_rediscovery(
        tc: hegel::TestCase,
    ) {
        use hegel::generators as gs;

        let file_count: u8 =
            tc.draw(gs::integers().min_value(1_u8).max_value(8));

        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();

        for i in 0..file_count {
            std::fs::write(
                root.join(format!("doc_{i}.md")),
                format!("# Doc {i}\n\nContent {i}"),
            )
            .unwrap();
        }

        let files = docbert_core::walker::discover_files(&root).unwrap();
        incremental::batch_store_metadata(&config_db, "notes", &files).unwrap();
        seed_snapshot(&config_db, "notes", &root);

        // Simulate collection removal: clear state + snapshot
        let doc_ids: Vec<u64> = config_db
            .list_all_document_metadata_typed()
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        config_db.batch_remove_document_state(&doc_ids).unwrap();
        config_db
            .remove_collection_merkle_snapshot("notes")
            .unwrap();

        let selection = select_sync_work(&config_db, "notes", &root).unwrap();
        assert_eq!(
            selection.new_files.len(),
            file_count as usize,
            "sync should rediscover all {file_count} files"
        );
    }
}
