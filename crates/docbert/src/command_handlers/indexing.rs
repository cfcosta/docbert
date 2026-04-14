use docbert_core::{
    ConfigDb,
    DataDir,
    EmbeddingDb,
    ModelManager,
    SearchIndex,
    chunking,
    embedding,
    error,
    incremental,
    ingestion,
    search,
    walker,
};
use kdam::{BarExt, Spinner, tqdm};

use super::model::{EMBEDDING_MODEL_KEY, log_model_runtime};
use crate::{cli, indexing_workflow};

pub(super) fn remove_document_embeddings_for_ids(
    embedding_db: &EmbeddingDb,
    doc_ids: &[u64],
) -> error::Result<()> {
    embedding_db
        .batch_remove_document_families(doc_ids)
        .map(|_| ())
}

pub(super) fn remove_document_artifacts_for_ids(
    config_db: &ConfigDb,
    doc_ids: &[u64],
) -> error::Result<()> {
    config_db.batch_remove_document_state(doc_ids)
}

fn log_load_failures(failures: &[ingestion::LoadFailure]) {
    for failure in failures {
        eprintln!(
            "  Warning: failed to read {}: {}",
            failure.file.relative_path.display(),
            failure.error
        );
    }
}

fn create_progress_bar(total: usize, desc: &str) -> kdam::Bar {
    tqdm!(
        total = total,
        ncols = 80,
        force_refresh = true,
        desc = desc,
        bar_format = "{desc suffix=' '}|{animation}| {spinner} {count}/{total} [{percentage:.0}%] in {elapsed human=true} ({rate:.1}/s, eta: {remaining human=true})",
        spinner = Spinner::new(
            &[
                "▁▂▃",
                "▂▃▄",
                "▃▄▅",
                "▄▅▆",
                "▅▆▇",
                "▆▇█",
                "▇█▇",
                "█▇▆",
                "▇▆▅",
                "▆▅▄",
                "▅▄▃",
                "▄▃▂",
                "▃▂▁"
            ],
            30.0,
            1.0,
        )
    )
}

fn finish_progress_bar(pb: &mut kdam::Bar) {
    let _ = pb.set_bar_format(
        "{desc suffix=' '}|{animation}| {count}/{total} [{percentage:.0}%] in {elapsed human=true} ({rate:.1}/s)",
    );
    let _ = pb.clear();
    let _ = pb.refresh();
    eprintln!();
}

struct IndexingRuntime {
    search_index: SearchIndex,
    embedding_db: EmbeddingDb,
    model: ModelManager,
    chunking_config: chunking::ChunkingConfig,
}

fn initialize_indexing_runtime(
    data_dir: &DataDir,
    model_id: &str,
) -> error::Result<IndexingRuntime> {
    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
    let mut model = ModelManager::with_model_id(model_id.to_string());
    log_model_runtime(&mut model)?;
    let chunking_config = chunking::resolve_chunking_config(model_id);
    if let Some(doc_len) = chunking_config.document_length {
        eprintln!(
            "Using document_length {doc_len} from config_sentence_transformers.json (chunk size ~{} chars).",
            chunking_config.chunk_size
        );
    }

    Ok(IndexingRuntime {
        search_index,
        embedding_db,
        model,
        chunking_config,
    })
}

fn process_document_batch(
    config_db: &ConfigDb,
    runtime: &mut IndexingRuntime,
    collection: &str,
    document_batch: &indexing_workflow::DocumentLoadBatch,
    index_documents: bool,
    embed_documents: bool,
) -> error::Result<()> {
    log_load_failures(&document_batch.failures);

    if index_documents {
        let mut writer = runtime.search_index.writer(15_000_000)?;
        let count = ingestion::ingest_prepared_documents(
            &runtime.search_index,
            &mut writer,
            collection,
            &document_batch.documents,
        )?;
        eprintln!("  Indexed {count} documents");
    }

    if embed_documents {
        let mut pb =
            create_progress_bar(document_batch.documents.len(), "Chunking");
        let docs_to_embed = docbert_core::preparation::collect_chunks(
            &document_batch.documents,
            runtime.chunking_config,
            |processed_count| {
                let _ = pb.update_to(processed_count);
            },
        );
        finish_progress_bar(&mut pb);

        if !docs_to_embed.is_empty() {
            let total_chunks = docs_to_embed.len();
            let mut pb = create_progress_bar(total_chunks, "Embedding");
            embedding::embed_and_store_in_batches(
                &mut runtime.model,
                &runtime.embedding_db,
                docs_to_embed,
                embedding::EMBEDDING_SUBMISSION_BATCH_SIZE,
                |embedded_count| {
                    let _ = pb.update_to(embedded_count);
                },
            )?;
            finish_progress_bar(&mut pb);
        }
    }

    incremental::batch_store_metadata(
        config_db,
        collection,
        &document_batch.metadata_files,
    )?;

    Ok(())
}

pub(crate) fn cmd_rebuild(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    args: &cli::RebuildArgs,
    model_id: &str,
) -> error::Result<()> {
    let collections = indexing_workflow::resolve_target_collections(
        config_db,
        args.collection.as_deref(),
    )?;

    if collections.is_empty() {
        eprintln!("No collections to rebuild.");
        return Ok(());
    }

    let mut runtime = initialize_indexing_runtime(data_dir, model_id)?;

    for (name, path) in &collections {
        let root = std::path::Path::new(path);
        if !root.is_dir() {
            eprintln!(
                "Warning: collection '{name}' path does not exist: {path}"
            );
            continue;
        }

        eprintln!("Rebuilding collection '{name}'...");

        if !args.embeddings_only {
            let mut writer = runtime.search_index.writer(15_000_000)?;
            runtime.search_index.delete_collection(&writer, name);
            writer.commit()?;
        }

        let old_doc_ids: Vec<u64> = config_db
            .list_all_document_metadata_typed()?
            .into_iter()
            .filter_map(|(doc_id, meta)| {
                (meta.collection == *name).then_some(doc_id)
            })
            .collect();
        if !args.index_only {
            remove_document_embeddings_for_ids(
                &runtime.embedding_db,
                &old_doc_ids,
            )?;
        }
        if !args.embeddings_only {
            remove_document_artifacts_for_ids(config_db, &old_doc_ids)?;
        }

        let files = walker::discover_files(root)?;
        eprintln!("  Found {} files", files.len());

        if !args.embeddings_only || !args.index_only {
            eprintln!("  Loading {} files...", files.len());
        }
        let document_batch =
            indexing_workflow::load_rebuild_batch(name, &files, args);
        // Rebuild also advances the stored Merkle snapshot only after the
        // collection has been processed successfully.
        let rebuild_result = process_document_batch(
            config_db,
            &mut runtime,
            name,
            &document_batch,
            !args.embeddings_only,
            !args.index_only,
        );
        indexing_workflow::finalize_rebuild_snapshot(
            config_db,
            name,
            root,
            rebuild_result,
        )?;

        eprintln!("  Done.");
    }

    config_db.set_setting(EMBEDDING_MODEL_KEY, model_id)?;

    eprintln!("Rebuild complete.");
    Ok(())
}

pub(crate) fn cmd_sync(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    args: &cli::SyncArgs,
    model_id: &str,
) -> error::Result<()> {
    let collections = indexing_workflow::resolve_target_collections(
        config_db,
        args.collection.as_deref(),
    )?;

    if collections.is_empty() {
        eprintln!("No collections to sync.");
        return Ok(());
    }

    if let Some(prev_model) = config_db.get_setting(EMBEDDING_MODEL_KEY)?
        && prev_model != model_id
    {
        eprintln!(
            "Warning: embeddings were computed with '{prev_model}', but current model is '{model_id}'."
        );
        eprintln!(
            "Mixing embeddings from different models produces invalid results."
        );
        eprintln!(
            "Run `docbert rebuild` to re-embed all documents with the new model."
        );
        return Err(error::Error::Config(format!(
            "model mismatch: embeddings use '{prev_model}', current is '{model_id}'"
        )));
    }

    let mut runtime = initialize_indexing_runtime(data_dir, model_id)?;

    for (name, path) in &collections {
        let root = std::path::Path::new(path);
        if !root.is_dir() {
            eprintln!(
                "Warning: collection '{name}' path does not exist: {path}"
            );
            continue;
        }

        let selection =
            indexing_workflow::select_sync_work(config_db, name, root)?;

        if selection.new_files.is_empty()
            && selection.changed_files.is_empty()
            && selection.deleted_ids.is_empty()
        {
            eprintln!("Collection '{name}' is up to date.");
            continue;
        }

        eprintln!("Syncing collection '{name}'...");
        eprintln!(
            "  {} new, {} changed, {} deleted",
            selection.new_files.len(),
            selection.changed_files.len(),
            selection.deleted_ids.len()
        );

        // Keep the stored Merkle snapshot behind the indexed state: plan and
        // execute sync work first, then advance the snapshot only if the work
        // succeeded.
        let sync_result = (|| {
            if !selection.deleted_ids.is_empty() {
                let mut writer = runtime.search_index.writer(15_000_000)?;
                for &doc_id in &selection.deleted_ids {
                    let display = search::short_doc_id(doc_id);
                    runtime.search_index.delete_document(&writer, &display);
                }
                writer.commit()?;

                remove_document_embeddings_for_ids(
                    &runtime.embedding_db,
                    &selection.deleted_ids,
                )?;
                remove_document_artifacts_for_ids(
                    config_db,
                    &selection.deleted_ids,
                )?;
                eprintln!(
                    "  Removed {} documents",
                    selection.deleted_ids.len()
                );
            }

            let files_to_process: Vec<_> = selection
                .new_files
                .iter()
                .chain(selection.changed_files.iter())
                .cloned()
                .collect();

            if !files_to_process.is_empty() {
                eprintln!("  Loading {} files...", files_to_process.len());
                let document_batch =
                    indexing_workflow::load_sync_batch(name, &files_to_process);
                process_document_batch(
                    config_db,
                    &mut runtime,
                    name,
                    &document_batch,
                    true,
                    true,
                )?;
            }

            Ok(())
        })();

        indexing_workflow::finalize_sync_snapshot(
            config_db,
            &selection,
            sync_result,
        )?;

        eprintln!("  Done.");
    }

    config_db.set_setting(EMBEDDING_MODEL_KEY, model_id)?;

    eprintln!("Sync complete.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use docbert_core::{ConfigDb, DocumentId, incremental};

    use super::*;

    /// Simulate the metadata deletion logic from cmd_rebuild for a given mode.
    /// Returns whether document metadata and user metadata survived.
    fn simulate_rebuild_metadata_lifecycle(
        embeddings_only: bool,
    ) -> (bool, bool) {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();

        // Pre-seed document metadata and user metadata
        let did = DocumentId::new("notes", "hello.md");
        let meta = incremental::DocumentMetadata {
            collection: "notes".to_string(),
            relative_path: "hello.md".to_string(),
            mtime: 42,
        };
        config_db
            .set_document_metadata_typed(did.numeric, &meta)
            .unwrap();
        config_db
            .set_document_user_metadata(
                did.numeric,
                &serde_json::json!({"topic": "rust"}),
            )
            .unwrap();

        let old_doc_ids = vec![did.numeric];

        // This is the rebuild logic under test:
        if !embeddings_only {
            remove_document_artifacts_for_ids(&config_db, &old_doc_ids)
                .unwrap();
        }

        let doc_metadata_survives = config_db
            .get_document_metadata_typed(did.numeric)
            .unwrap()
            .is_some();
        let user_metadata_survives = config_db
            .get_document_user_metadata(did.numeric)
            .unwrap()
            .is_some();

        (doc_metadata_survives, user_metadata_survives)
    }

    #[test]
    fn embeddings_only_rebuild_preserves_document_metadata() {
        let (doc_meta, user_meta) = simulate_rebuild_metadata_lifecycle(true);
        assert!(
            doc_meta,
            "embeddings-only rebuild must not delete document metadata"
        );
        assert!(
            user_meta,
            "embeddings-only rebuild must not delete user metadata"
        );
    }

    #[test]
    fn full_rebuild_deletes_document_metadata() {
        let (doc_meta, user_meta) = simulate_rebuild_metadata_lifecycle(false);
        assert!(!doc_meta, "full rebuild must delete document metadata");
        assert!(!user_meta, "full rebuild must delete user metadata");
    }

    #[hegel::test(test_cases = 50)]
    fn prop_embeddings_only_never_deletes_metadata(tc: hegel::TestCase) {
        use hegel::generators as gs;

        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();

        // Generate a random number of documents with metadata
        let doc_count: u8 =
            tc.draw(gs::integers().min_value(1_u8).max_value(10));

        let mut doc_ids = Vec::new();
        for i in 0..doc_count {
            let path = format!("doc_{i}.md");
            let did = DocumentId::new("notes", &path);
            let meta = incremental::DocumentMetadata {
                collection: "notes".to_string(),
                relative_path: path,
                mtime: i as u64 + 1,
            };
            config_db
                .set_document_metadata_typed(did.numeric, &meta)
                .unwrap();
            config_db
                .set_document_user_metadata(
                    did.numeric,
                    &serde_json::json!({"index": i}),
                )
                .unwrap();
            doc_ids.push(did.numeric);
        }

        // embeddings_only = true: do NOT call remove_document_artifacts_for_ids
        // (this is the fix we're testing)

        // Verify all metadata survived
        for &doc_id in &doc_ids {
            assert!(
                config_db
                    .get_document_metadata_typed(doc_id)
                    .unwrap()
                    .is_some(),
                "document metadata was deleted during embeddings-only rebuild"
            );
            assert!(
                config_db
                    .get_document_user_metadata(doc_id)
                    .unwrap()
                    .is_some(),
                "user metadata was deleted during embeddings-only rebuild"
            );
        }
    }
}
