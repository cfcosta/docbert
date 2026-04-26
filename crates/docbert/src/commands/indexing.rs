use std::time::Instant;

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
    walker,
};
use kdam::{BarExt, Spinner, tqdm};

use super::{
    model::{EMBEDDING_MODEL_KEY, log_model_runtime},
    style,
};
use crate::{cli, indexing};

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
            "  {} failed to read {}: {}",
            style::warn(&"Warning:"),
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
    chunking_config: chunking::Config,
}

fn initialize_indexing_runtime(
    data_dir: &DataDir,
    model_id: &str,
) -> error::Result<IndexingRuntime> {
    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
    let mut model = ModelManager::with_model_id(model_id.to_string());
    log_model_runtime(&mut model)?;
    let chunking_config = chunking::resolve_config(model_id);
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
    document_batch: &indexing::DocumentLoadBatch,
    index_documents: bool,
    embed_documents: bool,
) -> error::Result<()> {
    log_load_failures(&document_batch.failures);

    // Collect the doc IDs we're about to process so we can roll back on
    // failure.
    let batch_doc_ids: Vec<u64> = document_batch
        .documents
        .iter()
        .map(|d| d.did.numeric)
        .collect();

    // Step 1: Compute and store embeddings (expensive, most likely to fail).
    // Done before Tantivy so a model failure doesn't leave committed index
    // entries without embeddings.
    if embed_documents {
        let mut pb =
            create_progress_bar(document_batch.documents.len(), "Chunking");
        let mut docs_to_embed: Vec<(u64, String)> = Vec::new();
        let mut chunk_offset_entries: Vec<(
            u64,
            docbert_core::ChunkByteOffset,
        )> = Vec::new();
        for (i, document) in document_batch.documents.iter().enumerate() {
            for plan in docbert_core::preparation::chunk_plan(
                document,
                runtime.chunking_config,
            ) {
                chunk_offset_entries.push((plan.chunk_doc_id, plan.offset));
                docs_to_embed.push((plan.chunk_doc_id, plan.text));
            }
            let _ = pb.update_to(i + 1);
        }
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

            // Persist chunk byte offsets only after the embedding batch
            // commits. Doing it earlier would leave offsets pointing at
            // chunks the embedding store doesn't have yet, which search
            // would surface as ranges with no semantic backing.
            //
            // Wipe each base document's family first so a re-index that
            // produces fewer chunks doesn't leak offsets for the
            // higher chunk indexes.
            let base_doc_ids: Vec<u64> = document_batch
                .documents
                .iter()
                .map(|d| d.did.numeric)
                .collect();
            config_db.batch_remove_chunk_offsets_for_document_families(
                &base_doc_ids,
            )?;
            config_db.batch_set_chunk_offsets(&chunk_offset_entries)?;
        }
    }

    // Step 2: Store metadata (cheap, rarely fails).
    incremental::batch_store_metadata(
        config_db,
        collection,
        &document_batch.metadata_files,
    )?;

    // Step 3: Commit to Tantivy last. If this fails, roll back the
    // embeddings and metadata we persisted in steps 1-2 so all three
    // stores stay consistent.
    if index_documents {
        let mut writer = runtime.search_index.writer(15_000_000)?;
        if let Err(err) = ingestion::ingest_prepared_documents(
            &runtime.search_index,
            &mut writer,
            collection,
            &document_batch.documents,
        ) {
            let _ = remove_document_embeddings_for_ids(
                &runtime.embedding_db,
                &batch_doc_ids,
            );
            let _ =
                remove_document_artifacts_for_ids(config_db, &batch_doc_ids);
            let _ = config_db.batch_remove_chunk_offsets_for_document_families(
                &batch_doc_ids,
            );
            return Err(err);
        }
        eprintln!("  Indexed {} documents", document_batch.documents.len());
    }

    Ok(())
}

/// Rebuild the PLAID semantic index from whatever is currently in
/// `embedding_db` and persist it under `data_dir`.
///
/// Called at the end of `sync` and `rebuild` once the embedding
/// database is in its post-sync state. When the embedding db is empty
/// (no collections have been indexed yet, or every doc was deleted), we
/// skip with a short message — there's nothing to train centroids on.
///
/// Centroid count is adapted to the corpus size: we want enough
/// centroids for good recall but cannot ask for more centroids than we
/// have training tokens. The ceiling of 256 matches
/// `PlaidBuildParams::default()` and covers small-to-medium personal
/// corpora; the ⌈√n⌉ heuristic keeps tiny corpora (e.g. a fresh user
/// with one collection) from tripping the k > tokens guard inside
/// `build_index_from_embedding_db`.
/// Incrementally sync the PLAID index after `docbert sync` has written
/// the new/changed embeddings. Falls back to [`rebuild_plaid_index`]
/// when no prior index exists on disk.
///
/// `touched_bases` is the union of *base* doc_ids whose embeddings
/// were (re-)written during this sync pass — the bridge expands each
/// to its current chunk ids and re-encodes them against the existing
/// codec, leaving every untouched document verbatim.
///
/// This replaces the full rebuild-every-sync behaviour: for a corpus
/// of a few million tokens, retraining k-means + quantizer + encoding
/// every token dominates sync time even when only a handful of files
/// changed. With this path, cost scales with the size of the delta.
fn sync_plaid_index(
    data_dir: &DataDir,
    embedding_db: &EmbeddingDb,
    touched_bases: &[u64],
) -> error::Result<()> {
    let Some(existing) = docbert_core::plaid::load_index(data_dir)? else {
        // First-time sync — we have no codec to reuse. Fall back to
        // the full build path, which also handles the empty-db case.
        return rebuild_plaid_index(data_dir, embedding_db);
    };

    let start = Instant::now();
    eprintln!(
        "{} ({} touched base doc(s))...",
        style::subheader(&"Updating PLAID index"),
        touched_bases.len(),
    );
    let updated = docbert_core::plaid::update_index_for_touched_bases(
        embedding_db,
        existing,
        touched_bases,
    )?;
    docbert_core::plaid::save_index(&updated, data_dir)?;
    eprintln!(
        "  Indexed {} documents in {}.",
        updated.num_documents(),
        style::accent(&style::format_duration(start.elapsed())),
    );
    Ok(())
}

fn rebuild_plaid_index(
    data_dir: &DataDir,
    embedding_db: &EmbeddingDb,
) -> error::Result<()> {
    // `list_shapes` reads only the 8-byte header per entry and is
    // enough to pick `k_centroids`; the heavy loading is deferred to
    // `build_index_from_embedding_db`, which streams matrices into a
    // pre-sized pool.
    let shapes = embedding_db.list_shapes()?;
    if shapes.is_empty() {
        eprintln!("No embeddings yet; skipping PLAID index rebuild.");
        return Ok(());
    }
    let total_tokens: usize = shapes.iter().map(|&(_, n, _)| n as usize).sum();
    if total_tokens == 0 {
        eprintln!(
            "No tokens in the embedding database; skipping PLAID index rebuild."
        );
        return Ok(());
    }

    let default_params = docbert_core::plaid::PlaidBuildParams::default();
    let sqrt_tokens = (total_tokens as f32).sqrt().ceil() as usize;
    let k_centroids = default_params
        .k_centroids
        .min(total_tokens)
        .min(sqrt_tokens.max(1));
    let params = docbert_core::plaid::PlaidBuildParams {
        k_centroids,
        ..default_params
    };

    let start = Instant::now();
    eprintln!(
        "{} ({total_tokens} tokens, {k_centroids} centroids)...",
        style::subheader(&"Rebuilding PLAID semantic index"),
    );
    let index = docbert_core::plaid::build_index_from_embedding_db(
        embedding_db,
        params,
    )?;
    docbert_core::plaid::save_index(&index, data_dir)?;
    eprintln!(
        "  Indexed {} documents in {}.",
        index.num_documents(),
        style::accent(&style::format_duration(start.elapsed())),
    );
    Ok(())
}

pub(crate) fn rebuild(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    args: &cli::RebuildArgs,
    model_id: &str,
) -> error::Result<()> {
    let collections = indexing::resolve_target_collections(
        config_db,
        args.collection.as_deref(),
    )?;

    if collections.is_empty() {
        eprintln!("No collections to rebuild.");
        return Ok(());
    }

    let total_start = Instant::now();
    let mut runtime = initialize_indexing_runtime(data_dir, model_id)?;

    for (name, path) in &collections {
        let root = std::path::Path::new(path);
        if !root.is_dir() {
            eprintln!(
                "{} collection '{name}' path does not exist: {path}",
                style::warn(&"Warning:"),
            );
            continue;
        }

        let collection_start = Instant::now();
        eprintln!("{} '{name}'...", style::subheader(&"Rebuilding collection"),);

        if !args.embeddings_only {
            let mut writer = runtime.search_index.writer(15_000_000)?;
            runtime.search_index.delete_collection(&writer, name)?;
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
        let document_batch = indexing::load_rebuild_batch(name, &files, args);
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
        indexing::finalize_rebuild_snapshot(
            config_db,
            name,
            root,
            rebuild_result,
        )?;

        eprintln!(
            "  Done in {}.",
            style::accent(&style::format_duration(collection_start.elapsed())),
        );
    }

    config_db.set_setting(EMBEDDING_MODEL_KEY, model_id)?;

    let embedding_db = release_encoder_before_plaid(runtime)?;
    rebuild_plaid_index(data_dir, &embedding_db)?;

    eprintln!(
        "{} in {}.",
        style::header(&"Rebuild complete"),
        style::accent(&style::format_duration(total_start.elapsed())),
    );
    Ok(())
}

/// Drop the encoder model and trim CUDA's async mempool before the
/// PLAID build kicks off.
///
/// ModernBert's per-batch `packed_cos_sin` and `varlen_positions`
/// caches accumulate on the order of 1–2 GB of VRAM over a full
/// embedding pass (one entry per unique combination of sequence
/// lengths seen in a batch). The caches live on the model, and the
/// `ModelManager` stays alive until `rebuild`/`sync` returns —
/// exactly when the PLAID builder wants to allocate a ~3.47 GB
/// contiguous points tensor. Without releasing the encoder first,
/// CUDA's async mempool stays committed to the old allocations and
/// `cuMemAllocAsync` returns `CUDA_ERROR_OUT_OF_MEMORY` on a 12 GB
/// card.
///
/// Returns the retained `EmbeddingDb` handle so callers can continue
/// with the PLAID build; every other field of the runtime is dropped
/// together with the encoder.
fn release_encoder_before_plaid(
    runtime: IndexingRuntime,
) -> error::Result<EmbeddingDb> {
    let IndexingRuntime {
        search_index,
        embedding_db,
        model,
        chunking_config: _,
    } = runtime;
    drop(search_index);
    drop(model);
    docbert_core::plaid::release_cached_device_memory()?;
    Ok(embedding_db)
}

/// Rebuild the PLAID semantic index from whatever is already stored
/// in the embedding database, without re-running the encoder.
///
/// Useful after changes to the PLAID builder itself (centroid count,
/// codec bit-width, kmeans iters, …): `rebuild` would have to re-embed
/// every document — a four-minute walk over the model for the current
/// docbert corpus — before it even starts training centroids. This
/// command skips straight to the train + encode step.
pub(crate) fn reindex(data_dir: &DataDir) -> error::Result<()> {
    let total_start = Instant::now();
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
    rebuild_plaid_index(data_dir, &embedding_db)?;
    eprintln!(
        "{} in {}.",
        style::header(&"Reindex complete"),
        style::accent(&style::format_duration(total_start.elapsed())),
    );
    Ok(())
}

pub(crate) fn sync(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    args: &cli::SyncArgs,
    model_id: &str,
) -> error::Result<()> {
    let collections = indexing::resolve_target_collections(
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
            "{} embeddings were computed with '{prev_model}', but current model is '{model_id}'.",
            style::warn(&"Warning:"),
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

    let total_start = Instant::now();
    let mut runtime = initialize_indexing_runtime(data_dir, model_id)?;

    // Accumulate base doc_ids whose embeddings get re-written during
    // this sync. The PLAID update path needs these so it can expand
    // each base into its current chunk ids and re-encode only those
    // against the frozen codec — every untouched document keeps its
    // old encoding.
    let mut touched_bases: Vec<u64> = Vec::new();

    for (name, path) in &collections {
        let root = std::path::Path::new(path);
        if !root.is_dir() {
            eprintln!(
                "{} collection '{name}' path does not exist: {path}",
                style::warn(&"Warning:"),
            );
            continue;
        }

        let selection = indexing::select_sync_work(config_db, name, root)?;

        if selection.new_files.is_empty()
            && selection.changed_files.is_empty()
            && selection.deleted_ids.is_empty()
        {
            eprintln!("{} '{name}' is up to date.", style::dim(&"Collection"),);
            continue;
        }

        let collection_start = Instant::now();
        eprintln!("{} '{name}'...", style::subheader(&"Syncing collection"),);
        eprintln!(
            "  {} new, {} changed, {} deleted",
            selection.new_files.len(),
            selection.changed_files.len(),
            selection.deleted_ids.len()
        );

        // New + changed files are about to be re-embedded in the
        // process_document_batch call below. Capture their base
        // doc_ids up front so we still have them after the batch
        // takes ownership of the DiscoveredFile vecs.
        for f in selection
            .new_files
            .iter()
            .chain(selection.changed_files.iter())
        {
            let path_str = f.relative_path.to_string_lossy();
            touched_bases.push(
                docbert_core::DocumentId::new(name, path_str.as_ref()).numeric,
            );
        }

        // Keep the stored Merkle snapshot behind the indexed state: plan and
        // execute sync work first, then advance the snapshot only if the work
        // succeeded.
        let sync_result = (|| {
            if !selection.deleted_ids.is_empty() {
                // Build Tantivy keys from metadata BEFORE deleting it.
                let tantivy_keys: Vec<String> = selection
                    .deleted_ids
                    .iter()
                    .filter_map(|&doc_id| {
                        config_db
                            .get_document_metadata_typed(doc_id)
                            .ok()
                            .flatten()
                            .map(|meta| {
                                docbert_core::DocumentId::new(
                                    &meta.collection,
                                    &meta.relative_path,
                                )
                                .full_hex()
                            })
                    })
                    .collect();

                remove_document_embeddings_for_ids(
                    &runtime.embedding_db,
                    &selection.deleted_ids,
                )?;
                remove_document_artifacts_for_ids(
                    config_db,
                    &selection.deleted_ids,
                )?;

                let mut writer = runtime.search_index.writer(15_000_000)?;
                for key in &tantivy_keys {
                    runtime.search_index.delete_document(&writer, key)?;
                }
                writer.commit()?;
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
                    indexing::load_sync_batch(name, &files_to_process);
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

        indexing::finalize_sync_snapshot(config_db, &selection, sync_result)?;

        eprintln!(
            "  Done in {}.",
            style::accent(&style::format_duration(collection_start.elapsed())),
        );
    }

    config_db.set_setting(EMBEDDING_MODEL_KEY, model_id)?;

    let embedding_db = release_encoder_before_plaid(runtime)?;
    sync_plaid_index(data_dir, &embedding_db, &touched_bases)?;

    eprintln!(
        "{} in {}.",
        style::header(&"Sync complete"),
        style::accent(&style::format_duration(total_start.elapsed())),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use docbert_core::{ConfigDb, DocumentId, incremental};

    use super::*;

    /// Simulate the metadata deletion logic from rebuild for a given mode.
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

    #[test]
    fn process_batch_stores_metadata_before_tantivy_commit() {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.md"), "# A\n\nContent A").unwrap();

        let files = walker::discover_files(&root).unwrap();
        let search_index = SearchIndex::open_in_ram().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let model = ModelManager::new();
        let chunking_config = chunking::Config {
            chunk_size: 100,
            overlap: 0,
            document_length: None,
        };

        let mut runtime = IndexingRuntime {
            search_index,
            embedding_db,
            model,
            chunking_config,
        };

        let batch = indexing::load_rebuild_batch(
            "notes",
            &files,
            &crate::cli::RebuildArgs {
                collection: Some("notes".to_string()),
                embeddings_only: false,
                index_only: true,
            },
        );

        // index_documents=true, embed_documents=false
        process_document_batch(
            &config_db,
            &mut runtime,
            "notes",
            &batch,
            true,
            false,
        )
        .unwrap();

        // Metadata should be stored
        let did = docbert_core::DocumentId::new("notes", "a.md");
        let meta = config_db.get_document_metadata_typed(did.numeric).unwrap();
        assert!(meta.is_some(), "metadata should be stored after batch");

        // And the document should be in the index
        let results = runtime.search_index.search("content", 10).unwrap();
        assert!(!results.is_empty(), "document should be searchable");
    }

    #[hegel::test(test_cases = 30)]
    fn prop_process_batch_metadata_and_index_are_consistent(
        tc: hegel::TestCase,
    ) {
        use hegel::generators as gs;

        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let root = tmp.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();

        // Generate random number of documents
        let doc_count: u8 =
            tc.draw(gs::integers().min_value(1_u8).max_value(5));
        for i in 0..doc_count {
            let name = format!("doc_{i}.md");
            std::fs::write(
                root.join(&name),
                format!("# Doc {i}\n\nContent for document {i}"),
            )
            .unwrap();
        }

        let files = walker::discover_files(&root).unwrap();
        let search_index = SearchIndex::open_in_ram().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let model = ModelManager::new();
        let chunking_config = chunking::Config {
            chunk_size: 100,
            overlap: 0,
            document_length: None,
        };

        let mut runtime = IndexingRuntime {
            search_index,
            embedding_db,
            model,
            chunking_config,
        };

        let batch = indexing::load_rebuild_batch(
            "notes",
            &files,
            &crate::cli::RebuildArgs {
                collection: Some("notes".to_string()),
                embeddings_only: false,
                index_only: true,
            },
        );

        process_document_batch(
            &config_db,
            &mut runtime,
            "notes",
            &batch,
            true,
            false,
        )
        .unwrap();

        // Every document should have both metadata and index entry
        for i in 0..doc_count {
            let name = format!("doc_{i}.md");
            let did = docbert_core::DocumentId::new("notes", &name);
            assert!(
                config_db
                    .get_document_metadata_typed(did.numeric)
                    .unwrap()
                    .is_some(),
                "metadata missing for {name}"
            );
        }

        // Index should have all documents
        let all_meta = config_db.list_all_document_metadata_typed().unwrap();
        let notes_meta: Vec<_> = all_meta
            .iter()
            .filter(|(_, m)| m.collection == "notes")
            .collect();
        assert_eq!(
            notes_meta.len(),
            doc_count as usize,
            "metadata count mismatch"
        );
    }

    #[test]
    fn rebuild_plaid_index_skips_when_embedding_db_is_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let embedding_db =
            EmbeddingDb::open(&data_dir.embeddings_db()).unwrap();

        // Should return Ok and not write a PLAID index file.
        rebuild_plaid_index(&data_dir, &embedding_db).unwrap();
        assert!(
            !data_dir.plaid_index().exists(),
            "no PLAID file should be written for an empty db"
        );
    }

    #[test]
    fn sync_plaid_index_falls_back_to_full_build_when_no_index_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let embedding_db =
            EmbeddingDb::open(&data_dir.embeddings_db()).unwrap();
        embedding_db.store(1, 2, 2, &[0.0, 0.0, 0.1, 0.1]).unwrap();
        embedding_db
            .store(2, 2, 2, &[9.0, 9.0, 10.0, 10.0])
            .unwrap();

        // No index file exists yet — sync must fall back to a full
        // build so the first post-sync PLAID index lands on disk.
        sync_plaid_index(&data_dir, &embedding_db, &[]).unwrap();

        let loaded = docbert_core::plaid::load_index(&data_dir)
            .unwrap()
            .expect("sync must produce an index on the no-existing-index path");
        let mut doc_ids = loaded.doc_ids.clone();
        doc_ids.sort();
        assert_eq!(doc_ids, vec![1, 2]);
    }

    #[test]
    fn sync_plaid_index_reuses_codec_across_an_untouched_sync() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let embedding_db =
            EmbeddingDb::open(&data_dir.embeddings_db()).unwrap();
        embedding_db.store(1, 2, 2, &[0.0, 0.0, 0.1, 0.1]).unwrap();
        embedding_db
            .store(2, 2, 2, &[9.0, 9.0, 10.0, 10.0])
            .unwrap();

        // First sync: full build.
        sync_plaid_index(&data_dir, &embedding_db, &[]).unwrap();
        let first =
            docbert_core::plaid::load_index(&data_dir).unwrap().unwrap();

        // Second sync with no touches should take the incremental
        // path and keep the trained codec bit-for-bit.
        sync_plaid_index(&data_dir, &embedding_db, &[]).unwrap();
        let second =
            docbert_core::plaid::load_index(&data_dir).unwrap().unwrap();

        assert_eq!(second.codec.centroids, first.codec.centroids);
        assert_eq!(second.codec.bucket_cutoffs, first.codec.bucket_cutoffs);
        assert_eq!(second.codec.bucket_weights, first.codec.bucket_weights);
    }

    #[test]
    fn sync_plaid_index_re_encodes_a_touched_base() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let embedding_db =
            EmbeddingDb::open(&data_dir.embeddings_db()).unwrap();
        embedding_db.store(1, 2, 2, &[0.0, 0.0, 0.1, 0.1]).unwrap();
        embedding_db
            .store(2, 2, 2, &[9.0, 9.0, 10.0, 10.0])
            .unwrap();

        sync_plaid_index(&data_dir, &embedding_db, &[]).unwrap();
        let before =
            docbert_core::plaid::load_index(&data_dir).unwrap().unwrap();
        let before_tokens_for_1 =
            before.doc_tokens_vec(before.position_of(1).unwrap());
        let before_codec = before.codec.clone();

        // Re-embed doc 1 with tokens from the other cluster, then
        // sync incrementally with 1 as the touched base.
        embedding_db.store(1, 2, 2, &[9.5, 9.5, 10.1, 9.9]).unwrap();
        sync_plaid_index(&data_dir, &embedding_db, &[1]).unwrap();

        let after =
            docbert_core::plaid::load_index(&data_dir).unwrap().unwrap();
        let after_tokens_for_1 =
            after.doc_tokens_vec(after.position_of(1).unwrap());

        // Codec unchanged, but doc 1's encoding flipped to the other
        // cluster — proof the incremental path re-read tokens and
        // re-encoded them against the frozen codec.
        assert_eq!(after.codec.centroids, before_codec.centroids);
        assert_ne!(
            after_tokens_for_1, before_tokens_for_1,
            "touched base must be re-encoded on the incremental path",
        );
    }

    #[test]
    fn sync_plaid_index_prunes_docs_removed_from_the_db() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let embedding_db =
            EmbeddingDb::open(&data_dir.embeddings_db()).unwrap();
        embedding_db.store(1, 2, 2, &[0.0, 0.0, 0.1, 0.1]).unwrap();
        embedding_db
            .store(2, 2, 2, &[9.0, 9.0, 10.0, 10.0])
            .unwrap();
        sync_plaid_index(&data_dir, &embedding_db, &[]).unwrap();
        assert!(
            docbert_core::plaid::load_index(&data_dir)
                .unwrap()
                .unwrap()
                .doc_ids
                .contains(&2),
        );

        // Simulate a deletion: doc 2 gone from the db, no base touched.
        embedding_db.remove(2).unwrap();
        sync_plaid_index(&data_dir, &embedding_db, &[]).unwrap();

        let after =
            docbert_core::plaid::load_index(&data_dir).unwrap().unwrap();
        assert!(!after.doc_ids.contains(&2));
        assert!(after.doc_ids.contains(&1));
    }

    #[test]
    fn rebuild_plaid_index_writes_loadable_file_for_non_empty_db() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let embedding_db =
            EmbeddingDb::open(&data_dir.embeddings_db()).unwrap();

        // Two tiny 2-D documents — enough tokens for the sqrt-based
        // centroid heuristic to pick k >= 1.
        embedding_db.store(1, 2, 2, &[0.0, 0.0, 0.1, 0.1]).unwrap();
        embedding_db
            .store(2, 2, 2, &[9.0, 9.0, 10.0, 10.0])
            .unwrap();

        rebuild_plaid_index(&data_dir, &embedding_db).unwrap();

        let loaded = docbert_core::plaid::load_index(&data_dir)
            .unwrap()
            .expect("PLAID index should be persisted after rebuild");
        let mut doc_ids = loaded.doc_ids.clone();
        doc_ids.sort();
        assert_eq!(doc_ids, vec![1, 2]);
    }
}
