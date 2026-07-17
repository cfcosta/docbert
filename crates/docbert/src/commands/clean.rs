use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use docbert_core::{
    ConfigDb,
    DataDir,
    EmbeddingDb,
    SearchIndex,
    error,
    is_legacy_redb_file,
    model_manager::resolve_model,
};
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
    legacy_embeddings: usize,
    removed_embeddings: usize,
    model_mismatch: bool,
    stored_model: Option<&'a str>,
    current_model: &'a str,
    dry_run: bool,
}

/// Report for the filesystem-level reset of databases still in the
/// redb format written by docbert releases before 1.0.
#[derive(Serialize)]
struct LegacyResetReport {
    legacy_config_db: bool,
    legacy_embeddings_db: bool,
    removed_paths: Vec<PathBuf>,
    dry_run: bool,
}

/// Sibling lock file heed/LMDB creates next to a `NO_SUB_DIR` env.
fn lock_file(db_path: &Path) -> PathBuf {
    let mut os = db_path.as_os_str().to_os_string();
    os.push("-lock");
    PathBuf::from(os)
}

/// Remove `path` (file or directory) if it exists, recording it in
/// `removed`. In dry-run mode only the recording happens.
fn remove_path(
    path: PathBuf,
    dry_run: bool,
    removed: &mut Vec<PathBuf>,
) -> error::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    if !dry_run {
        if path.is_dir() {
            std::fs::remove_dir_all(&path)?;
        } else {
            std::fs::remove_file(&path)?;
        }
    }
    removed.push(path);
    Ok(())
}

/// Wipe every piece of per-document state: chunk manifests, document
/// metadata and content, Tantivy entries, merkle snapshots, and the
/// stored embedding-model marker. Collections stay registered so the
/// next `sync` re-walks the same source paths.
fn wipe_document_state(
    config_db: &ConfigDb,
    data_dir: &DataDir,
) -> error::Result<()> {
    let all_doc_ids: Vec<u64> = config_db
        .list_all_document_metadata_typed()?
        .into_iter()
        .map(|(doc_id, _)| doc_id)
        .collect();
    remove_chunk_manifests_for_ids(config_db, &all_doc_ids)?;
    remove_document_artifacts_for_ids(config_db, &all_doc_ids)?;
    wipe_collections_index_state(config_db, data_dir)?;
    config_db.remove_setting(EMBEDDING_MODEL_KEY)?;
    Ok(())
}

/// Handle databases still in the redb format written before 1.0.
///
/// Those files can't be opened by this version (`ConfigDb::open` /
/// `EmbeddingDb::open` refuse them), so clean deals with them at the
/// filesystem level before touching any store. A legacy embeddings.db
/// is deleted together with the plaid index derived from it, and the
/// per-document state is wiped so the next `sync` re-embeds every
/// collection. A legacy config.db takes the whole index state with it
/// — every other file references it — leaving a pristine data dir.
///
/// Returns `true` when legacy files were found; the reset (or its
/// dry-run preview) is the whole clean run in that case.
fn reset_legacy_databases(
    data_dir: &DataDir,
    args: &cli::CleanArgs,
) -> error::Result<bool> {
    let config_path = data_dir.config_db();
    let embeddings_path = data_dir.embeddings_db();
    let legacy_config = is_legacy_redb_file(&config_path)?;
    let legacy_embeddings = is_legacy_redb_file(&embeddings_path)?;
    if !legacy_config && !legacy_embeddings {
        return Ok(false);
    }

    let mut removed = Vec::new();
    remove_path(embeddings_path.clone(), args.dry_run, &mut removed)?;
    remove_path(lock_file(&embeddings_path), args.dry_run, &mut removed)?;
    remove_path(data_dir.plaid_index(), args.dry_run, &mut removed)?;
    if legacy_config {
        remove_path(config_path.clone(), args.dry_run, &mut removed)?;
        remove_path(lock_file(&config_path), args.dry_run, &mut removed)?;
        remove_path(data_dir.tantivy_dir()?, args.dry_run, &mut removed)?;
    } else if !args.dry_run {
        // config.db is healthy, but its manifests and document state
        // reference embeddings that no longer exist. Wipe them so the
        // next `sync` re-walks and re-embeds instead of skipping
        // "already indexed" documents whose embeddings are gone.
        let config_db = ConfigDb::open(&config_path)?;
        wipe_document_state(&config_db, data_dir)?;
    }

    if args.json {
        let report = LegacyResetReport {
            legacy_config_db: legacy_config,
            legacy_embeddings_db: legacy_embeddings,
            removed_paths: removed,
            dry_run: args.dry_run,
        };
        println!("{}", serde_json::to_string(&report)?);
        return Ok(true);
    }

    eprintln!("{}", style::header(&"Clean"));
    if legacy_config {
        eprintln!(
            "  config.db is in the pre-1.0 redb format; resetting all \
             index state.",
        );
    } else {
        eprintln!(
            "  embeddings.db is in the pre-1.0 redb format; resetting \
             embeddings and semantic index.",
        );
    }
    let verb = if args.dry_run {
        "would remove"
    } else {
        "removed"
    };
    for path in &removed {
        eprintln!("  {} {}", style::dim(&verb), path.display());
    }
    if !args.dry_run {
        if legacy_config {
            eprintln!(
                "  {} re-add collections (`docbert collection add`), \
                 then run `docbert sync` to re-index.",
                style::warn(&"Next:"),
            );
        } else {
            eprintln!(
                "  {} run `docbert sync` to re-index.",
                style::warn(&"Next:"),
            );
        }
    }
    Ok(true)
}

/// CLI entry point. Runs the legacy-format pre-pass before opening any
/// database — clean must stay usable when the stores are in the
/// pre-1.0 redb format that the normal open paths refuse.
pub(crate) fn run_standalone(
    data_dir: &DataDir,
    args: &cli::CleanArgs,
    model_override: Option<&str>,
) -> error::Result<()> {
    if reset_legacy_databases(data_dir, args)? {
        return Ok(());
    }
    let config_db = ConfigDb::open(&data_dir.config_db())?;
    let model_id = resolve_model(&config_db, model_override)?.model_id;
    run(&config_db, data_dir, args, &model_id)
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
    // Rows still in the pre-1.0 f32 layout can't be decoded any more;
    // clean is the documented way to get rid of them.
    let legacy_ids = embedding_db.list_legacy_ids()?;

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
        .chain(legacy_ids.iter().copied())
        .collect();
    to_remove.sort_unstable();
    to_remove.dedup();

    let wipe_state = model_mismatch || !legacy_ids.is_empty();

    if !args.dry_run && !to_remove.is_empty() {
        embedding_db.batch_remove(&to_remove)?;

        if wipe_state {
            // Manifests, document state, and Tantivy entries reference
            // chunk_doc_ids whose embeddings just got removed (all of
            // them on a model switch, the undecodable pre-1.0 rows
            // otherwise). Wipe that state so a future sync re-walks
            // and re-embeds instead of skipping documents it believes
            // are already indexed.
            wipe_document_state(config_db, data_dir)?;
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
        legacy_embeddings: legacy_ids.len(),
        removed_embeddings,
        model_mismatch,
        stored_model: stored_model.as_deref(),
        current_model: model_id,
        dry_run: args.dry_run,
    };

    if args.json {
        println!("{}", serde_json::to_string(&report)?);
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
    eprintln!("  Legacy-format embeddings: {}", legacy_ids.len());
    if args.dry_run {
        eprintln!(
            "  {} {} embeddings would be removed (dry run)",
            style::dim(&"->"),
            to_remove.len(),
        );
    } else {
        eprintln!("  Removed {} embeddings", to_remove.len());
        if wipe_state {
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

    /// First bytes of a redb file, as written by docbert < 1.0.
    const REDB_MAGIC: &[u8] = b"redb\x1A\x0A\xA9\x0D\x0A";

    fn write_fake_redb_file(path: &Path) {
        let mut bytes = REDB_MAGIC.to_vec();
        bytes.extend_from_slice(b"junk payload");
        std::fs::write(path, bytes).unwrap();
    }

    /// Serialize an embedding value in the pre-1.0 layout: same 8-byte
    /// header, body of little-endian f32s instead of bf16s.
    fn legacy_f32_bytes(
        num_tokens: u32,
        dimension: u32,
        data: &[f32],
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&num_tokens.to_le_bytes());
        bytes.extend_from_slice(&dimension.to_le_bytes());
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn clean_drops_legacy_f32_rows_and_wipes_document_state() {
        let (_tmp, data_dir, config_db, embedding_db, args) = fixture();
        config_db.set_collection("notes", "/tmp/notes").unwrap();
        config_db
            .set_setting(EMBEDDING_MODEL_KEY, "model-a")
            .unwrap();
        let live = seed_doc_with_chunk(
            &config_db,
            &embedding_db,
            "notes",
            "live.md",
            100,
        );
        embedding_db
            .insert_raw(300, &legacy_f32_bytes(1, 2, &[0.1, 0.2]))
            .unwrap();

        run(&config_db, &data_dir, &args, "model-a").unwrap();

        assert!(
            embedding_db.list_legacy_ids().unwrap().is_empty(),
            "legacy rows removed",
        );
        assert!(
            embedding_db.load(100).unwrap().is_some(),
            "current-layout rows stay for reuse on the next sync",
        );
        assert!(
            config_db.get_doc_chunks(live.numeric).unwrap().is_none(),
            "manifests wiped so sync re-embeds the affected documents",
        );
        assert!(
            config_db
                .get_document_metadata_typed(live.numeric)
                .unwrap()
                .is_none(),
            "document state wiped so sync re-walks the collection",
        );
    }

    #[test]
    fn standalone_clean_resets_legacy_redb_embeddings_file() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let args = cli::CleanArgs {
            dry_run: false,
            json: false,
        };

        // Healthy config with a registered collection and one synced
        // doc; the embeddings file itself is still redb-formatted, so
        // it can only be handled below the EmbeddingDb::open layer.
        {
            let config_db = ConfigDb::open(&data_dir.config_db()).unwrap();
            config_db.set_collection("notes", "/tmp/notes").unwrap();
            config_db
                .set_setting(EMBEDDING_MODEL_KEY, "model-a")
                .unwrap();
            let did = DocumentId::new("notes", "live.md");
            let metadata = incremental::DocumentMetadata {
                collection: "notes".to_string(),
                relative_path: "live.md".to_string(),
                mtime: 0,
            };
            config_db
                .set_document_metadata_typed(did.numeric, &metadata)
                .unwrap();
            config_db
                .set_doc_chunks(
                    did.numeric,
                    &[DocChunkEntry {
                        chunk_doc_id: 100,
                        start_byte: 0,
                        byte_len: 10,
                    }],
                )
                .unwrap();
        }
        write_fake_redb_file(&data_dir.embeddings_db());
        std::fs::write(data_dir.plaid_index(), b"stale").unwrap();

        run_standalone(&data_dir, &args, None).unwrap();

        assert!(!data_dir.embeddings_db().exists(), "legacy file removed");
        assert!(!data_dir.plaid_index().exists(), "derived index removed");

        let config_db = ConfigDb::open(&data_dir.config_db()).unwrap();
        assert!(
            config_db.get_collection("notes").unwrap().is_some(),
            "collections survive an embeddings-only reset",
        );
        let did = DocumentId::new("notes", "live.md");
        assert!(
            config_db.get_doc_chunks(did.numeric).unwrap().is_none(),
            "manifests wiped so sync re-embeds",
        );
        assert!(
            config_db
                .get_setting(EMBEDDING_MODEL_KEY)
                .unwrap()
                .is_none(),
        );
    }

    #[test]
    fn standalone_clean_resets_legacy_redb_config_file() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let args = cli::CleanArgs {
            dry_run: false,
            json: false,
        };
        write_fake_redb_file(&data_dir.config_db());
        write_fake_redb_file(&data_dir.embeddings_db());
        std::fs::write(data_dir.plaid_index(), b"stale").unwrap();
        let tantivy = data_dir.tantivy_dir().unwrap();
        std::fs::write(tantivy.join("meta.json"), b"{}").unwrap();

        run_standalone(&data_dir, &args, None).unwrap();

        assert!(!data_dir.config_db().exists());
        assert!(!data_dir.embeddings_db().exists());
        assert!(!data_dir.plaid_index().exists());
        assert!(!tantivy.exists());

        // A fresh open starts from scratch.
        let config_db = ConfigDb::open(&data_dir.config_db()).unwrap();
        assert!(config_db.list_collections().unwrap().is_empty());
    }

    #[test]
    fn standalone_clean_dry_run_keeps_legacy_files_in_place() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let args = cli::CleanArgs {
            dry_run: true,
            json: false,
        };
        write_fake_redb_file(&data_dir.embeddings_db());

        run_standalone(&data_dir, &args, None).unwrap();

        assert!(
            data_dir.embeddings_db().exists(),
            "dry run must not delete anything",
        );
    }
}
