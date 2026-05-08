use std::time::Instant;

use docbert_core::{ConfigDb, DataDir, EmbeddingDb, SearchIndex, error};
use serde::Serialize;

use super::{
    indexing::{
        rebuild_plaid_index,
        remove_chunk_manifests_for_ids,
        remove_document_artifacts_for_ids,
    },
    model::EMBEDDING_MODEL_KEY,
    style,
};
use crate::cli;

#[derive(Serialize)]
struct CleanReport<'a> {
    total_embeddings: usize,
    orphan_embeddings: usize,
    wrong_model_embeddings: usize,
    removed_embeddings: usize,
    model_mismatch: bool,
    stored_model: Option<&'a str>,
    current_model: &'a str,
    dry_run: bool,
}

fn report_json(report: &CleanReport<'_>) -> error::Result<String> {
    serde_json::to_string(report).map_err(|e| {
        error::Error::Config(format!("failed to serialize clean report: {e}"))
    })
}

/// Drop the Tantivy entries and the merkle snapshots that reference
/// every collection. The collections themselves stay registered so a
/// subsequent `sync` re-walks the same source paths.
fn wipe_collections_index_state(
    config_db: &ConfigDb,
    data_dir: &DataDir,
) -> error::Result<()> {
    let collections = config_db.list_collections()?;
    if collections.is_empty() {
        return Ok(());
    }

    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let mut writer = search_index.writer(15_000_000)?;
    for (name, _) in &collections {
        search_index.delete_collection(&writer, name)?;
    }
    writer.commit()?;

    for (name, _) in &collections {
        config_db.remove_collection_merkle_snapshot(name)?;
    }
    Ok(())
}

pub(crate) fn run(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    args: &cli::CleanArgs,
    model_id: &str,
) -> error::Result<()> {
    let start = Instant::now();
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
    let stored_model = config_db.get_setting(EMBEDDING_MODEL_KEY)?;
    let model_mismatch = matches!(
        stored_model.as_deref(),
        Some(stored) if stored != model_id
    );

    let all_ids = embedding_db.list_ids()?;

    // Partition the embedding ids. Wrong-model wins when the global
    // embedding-model setting disagrees with the current model — every
    // stored chunk_doc_id was hashed against the prior model_id, so
    // none of them can be referenced by a manifest under the new model.
    // In that case the chunk_owners check is moot and we treat the
    // whole table as wrong-model.
    let (wrong_model_ids, orphan_ids): (Vec<u64>, Vec<u64>) = if model_mismatch
    {
        (all_ids.clone(), Vec::new())
    } else {
        let mut orphans = Vec::new();
        for &id in &all_ids {
            if config_db.get_chunk_owners(id)?.is_empty() {
                orphans.push(id);
            }
        }
        (Vec::new(), orphans)
    };

    let mut to_remove: Vec<u64> = wrong_model_ids
        .iter()
        .copied()
        .chain(orphan_ids.iter().copied())
        .collect();
    to_remove.sort_unstable();
    to_remove.dedup();

    if !args.dry_run && !to_remove.is_empty() {
        embedding_db.batch_remove(&to_remove)?;

        if model_mismatch {
            // Manifests, document state, and Tantivy entries all
            // reference chunk_doc_ids that no longer exist. Wipe them
            // so a future sync/rebuild starts on a clean slate.
            let all_doc_ids: Vec<u64> = config_db
                .list_all_document_metadata_typed()?
                .into_iter()
                .map(|(doc_id, _)| doc_id)
                .collect();
            remove_chunk_manifests_for_ids(config_db, &all_doc_ids)?;
            remove_document_artifacts_for_ids(config_db, &all_doc_ids)?;
            wipe_collections_index_state(config_db, data_dir)?;
            config_db.remove_setting(EMBEDDING_MODEL_KEY)?;
        }

        let plaid_path = data_dir.plaid_index();
        if embedding_db.list_ids()?.is_empty() {
            if plaid_path.exists() {
                std::fs::remove_file(&plaid_path)?;
            }
        } else {
            rebuild_plaid_index(data_dir, &embedding_db)?;
        }
    }

    let removed_embeddings = if args.dry_run { 0 } else { to_remove.len() };
    let report = CleanReport {
        total_embeddings: all_ids.len(),
        orphan_embeddings: orphan_ids.len(),
        wrong_model_embeddings: wrong_model_ids.len(),
        removed_embeddings,
        model_mismatch,
        stored_model: stored_model.as_deref(),
        current_model: model_id,
        dry_run: args.dry_run,
    };

    if args.json {
        println!("{}", report_json(&report)?);
        return Ok(());
    }

    eprintln!("{}", style::header(&"Clean"));
    eprintln!("  Current model: {model_id}");
    match stored_model.as_deref() {
        Some(stored) => eprintln!("  Stored embedding model: {stored}"),
        None => eprintln!("  Stored embedding model: (not set)"),
    }
    eprintln!("  Total embeddings: {}", all_ids.len());
    eprintln!("  Orphan embeddings: {}", orphan_ids.len());
    eprintln!("  Wrong-model embeddings: {}", wrong_model_ids.len());
    if args.dry_run {
        eprintln!(
            "  {} {} embeddings would be removed (dry run)",
            style::dim(&"->"),
            to_remove.len(),
        );
    } else {
        eprintln!("  Removed {} embeddings", to_remove.len());
        if model_mismatch {
            eprintln!(
                "  {} cleared document state — run `docbert sync` to re-embed.",
                style::warn(&"Note:"),
            );
        }
    }
    eprintln!(
        "{} in {}",
        style::header(&"Clean complete"),
        style::accent(&style::format_duration(start.elapsed())),
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use docbert_core::{
        ConfigDb,
        DataDir,
        DocChunkEntry,
        DocumentId,
        EmbeddingDb,
        incremental,
    };

    use super::*;

    fn fixture() -> (
        tempfile::TempDir,
        DataDir,
        ConfigDb,
        EmbeddingDb,
        cli::CleanArgs,
    ) {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let config_db = ConfigDb::open(&data_dir.config_db()).unwrap();
        let embedding_db =
            EmbeddingDb::open(&data_dir.embeddings_db()).unwrap();
        let args = cli::CleanArgs {
            dry_run: false,
            json: false,
        };
        (tmp, data_dir, config_db, embedding_db, args)
    }

    fn seed_doc_with_chunk(
        config_db: &ConfigDb,
        embedding_db: &EmbeddingDb,
        collection: &str,
        path: &str,
        chunk_doc_id: u64,
    ) -> DocumentId {
        let did = DocumentId::new(collection, path);
        let metadata = incremental::DocumentMetadata {
            collection: collection.to_string(),
            relative_path: path.to_string(),
            mtime: 0,
        };
        config_db
            .set_document_metadata_typed(did.numeric, &metadata)
            .unwrap();
        embedding_db.store(chunk_doc_id, 1, 2, &[1.0, 2.0]).unwrap();
        config_db
            .set_doc_chunks(
                did.numeric,
                &[DocChunkEntry {
                    chunk_doc_id,
                    start_byte: 0,
                    byte_len: 10,
                }],
            )
            .unwrap();
        did
    }

    #[test]
    fn clean_removes_orphan_embeddings_only() {
        let (_tmp, data_dir, config_db, embedding_db, args) = fixture();
        config_db
            .set_setting(EMBEDDING_MODEL_KEY, "model-a")
            .unwrap();
        let _live = seed_doc_with_chunk(
            &config_db,
            &embedding_db,
            "notes",
            "live.md",
            100,
        );

        // Pure-orphan embedding: stored, but no manifest references it.
        embedding_db.store(200, 1, 2, &[3.0, 4.0]).unwrap();

        run(&config_db, &data_dir, &args, "model-a").unwrap();

        assert!(
            embedding_db.load(100).unwrap().is_some(),
            "live embedding stays"
        );
        assert!(
            embedding_db.load(200).unwrap().is_none(),
            "orphan embedding removed"
        );
        // Stored model setting is intact when the model still matches.
        assert_eq!(
            config_db
                .get_setting(EMBEDDING_MODEL_KEY)
                .unwrap()
                .as_deref(),
            Some("model-a"),
        );
    }

    #[test]
    fn clean_dry_run_keeps_orphans_in_place() {
        let (_tmp, data_dir, config_db, embedding_db, mut args) = fixture();
        config_db
            .set_setting(EMBEDDING_MODEL_KEY, "model-a")
            .unwrap();
        embedding_db.store(200, 1, 2, &[3.0, 4.0]).unwrap();
        args.dry_run = true;

        run(&config_db, &data_dir, &args, "model-a").unwrap();

        assert!(
            embedding_db.load(200).unwrap().is_some(),
            "dry run must not delete anything",
        );
    }

    #[test]
    fn clean_wipes_everything_when_model_mismatches() {
        let (_tmp, data_dir, config_db, embedding_db, args) = fixture();
        config_db.set_collection("notes", "/tmp/notes").unwrap();
        config_db
            .set_setting(EMBEDDING_MODEL_KEY, "model-old")
            .unwrap();
        let live = seed_doc_with_chunk(
            &config_db,
            &embedding_db,
            "notes",
            "live.md",
            100,
        );

        run(&config_db, &data_dir, &args, "model-new").unwrap();

        assert!(
            embedding_db.load(100).unwrap().is_none(),
            "wrong-model embedding removed",
        );
        assert!(
            config_db
                .get_document_metadata_typed(live.numeric)
                .unwrap()
                .is_none(),
            "document metadata wiped after model-mismatch clean",
        );
        assert!(
            config_db.get_doc_chunks(live.numeric).unwrap().is_none(),
            "manifests wiped after model-mismatch clean",
        );
        assert!(
            config_db
                .get_setting(EMBEDDING_MODEL_KEY)
                .unwrap()
                .is_none(),
            "stored embedding-model setting cleared",
        );
        // Collection registration survives so the user doesn't have to
        // re-add it before running sync.
        assert!(config_db.get_collection("notes").unwrap().is_some());
    }

    #[test]
    fn clean_is_a_no_op_when_db_is_already_clean() {
        let (_tmp, data_dir, config_db, embedding_db, args) = fixture();
        config_db
            .set_setting(EMBEDDING_MODEL_KEY, "model-a")
            .unwrap();
        seed_doc_with_chunk(&config_db, &embedding_db, "notes", "a.md", 1);

        run(&config_db, &data_dir, &args, "model-a").unwrap();

        assert!(embedding_db.load(1).unwrap().is_some());
    }

    #[test]
    fn clean_handles_missing_stored_model_setting() {
        let (_tmp, data_dir, config_db, embedding_db, args) = fixture();
        // Orphan with no EMBEDDING_MODEL_KEY set: no mismatch (the
        // stored value is None), but orphans should still be cleaned.
        embedding_db.store(7, 1, 2, &[1.0, 2.0]).unwrap();

        run(&config_db, &data_dir, &args, "model-a").unwrap();

        assert!(embedding_db.load(7).unwrap().is_none());
    }
}
