use std::path::Path;

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
    model_manager::{MODEL_ENV_VAR, ModelResolution},
    search,
    walker,
};
use kdam::{BarExt, Spinner, tqdm};
use serde::Serialize;

use crate::{cli, indexing_workflow};

fn serialize_json<T: Serialize + ?Sized>(value: &T, error_context: &str) -> error::Result<String> {
    serde_json::to_string(value).map_err(|e| error::Error::Config(format!("{error_context}: {e}")))
}

#[derive(Serialize)]
struct CollectionListItem<'a> {
    name: &'a str,
    path: &'a str,
}

fn collection_list_json_string(collections: &[(String, String)]) -> error::Result<String> {
    let items: Vec<_> = collections
        .iter()
        .map(|(name, path)| CollectionListItem { name, path })
        .collect();
    serialize_json(&items, "failed to serialize collection list")
}

#[derive(Serialize)]
struct ContextListItem<'a> {
    uri: &'a str,
    description: &'a str,
}

fn context_list_json_string(contexts: &[(String, String)]) -> error::Result<String> {
    let items: Vec<_> = contexts
        .iter()
        .map(|(uri, description)| ContextListItem { uri, description })
        .collect();
    serialize_json(&items, "failed to serialize context list")
}

#[derive(Serialize)]
struct GetJsonOutput<'a> {
    collection: &'a str,
    path: &'a str,
    file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
}

fn get_json_string(
    collection: &str,
    path: &str,
    full_path: &Path,
    content: Option<&str>,
) -> error::Result<String> {
    serialize_json(
        &GetJsonOutput {
            collection,
            path,
            file: full_path.display().to_string(),
            content,
        },
        "failed to serialize get response",
    )
}

#[derive(Serialize)]
struct MultiGetJsonItem {
    collection: String,
    path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

fn multi_get_json_string(items: &[MultiGetJsonItem]) -> error::Result<String> {
    serialize_json(items, "failed to serialize multi-get response")
}

#[derive(Serialize)]
struct StatusJsonOutput<'a> {
    data_dir: String,
    model: &'a str,
    model_source: &'a str,
    embedding_model: Option<&'a str>,
    collections: usize,
    documents: usize,
}

fn status_json_string(
    data_dir: &DataDir,
    model_resolution: &ModelResolution,
    embedding_model: Option<&str>,
    collection_count: usize,
    doc_count: usize,
) -> error::Result<String> {
    serialize_json(
        &StatusJsonOutput {
            data_dir: data_dir.root().display().to_string(),
            model: &model_resolution.model_id,
            model_source: model_resolution.source.as_str(),
            embedding_model,
            collections: collection_count,
            documents: doc_count,
        },
        "failed to serialize status response",
    )
}

#[derive(Serialize)]
struct ModelShowJsonOutput<'a> {
    resolved: &'a str,
    source: &'a str,
    cli: Option<&'a str>,
    env: Option<&'a str>,
    config: Option<&'a str>,
}

fn model_show_json_string(model_resolution: &ModelResolution) -> error::Result<String> {
    serialize_json(
        &ModelShowJsonOutput {
            resolved: &model_resolution.model_id,
            source: model_resolution.source.as_str(),
            cli: model_resolution.cli_model.as_deref(),
            env: model_resolution.env_model.as_deref(),
            config: model_resolution.config_model.as_deref(),
        },
        "failed to serialize model resolution",
    )
}

pub(crate) fn run_search(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    model_resolution: &ModelResolution,
    args: &cli::SearchArgs,
) -> error::Result<()> {
    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
    let mut model = ModelManager::with_model_id(model_resolution.model_id.clone());
    if !args.bm25_only {
        log_model_runtime(&mut model)?;
    }

    let params = search::SearchParams {
        query: args.query.clone(),
        count: args.count,
        collection: args.collection.clone(),
        min_score: args.min_score,
        bm25_only: args.bm25_only,
        no_fuzzy: args.no_fuzzy,
        all: args.all,
    };

    let results = search::execute_search(&params, &search_index, &embedding_db, &mut model)?;

    if args.json {
        search::format_json(&results, &args.query);
    } else if args.files {
        search::format_files(&results, config_db);
    } else {
        search::format_human(&results);
    }
    Ok(())
}

pub(crate) fn run_semantic_search(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    model_resolution: &ModelResolution,
    args: &cli::SemanticSearchArgs,
) -> error::Result<()> {
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
    let mut model = ModelManager::with_model_id(model_resolution.model_id.clone());
    log_model_runtime(&mut model)?;

    let params = search::SemanticSearchParams {
        query: args.query.clone(),
        count: args.count,
        min_score: args.min_score,
        all: args.all,
    };

    let results = search::execute_semantic_search(&params, config_db, &embedding_db, &mut model)?;

    if args.json {
        search::format_json(&results, &args.query);
    } else if args.files {
        search::format_files(&results, config_db);
    } else {
        search::format_human(&results);
    }
    Ok(())
}

pub(crate) fn collection_add(
    config_db: &ConfigDb,
    path: &std::path::Path,
    name: &str,
) -> error::Result<()> {
    // Validate the directory exists and is readable
    if !path.exists() {
        return Err(error::Error::Config(format!(
            "directory does not exist: {}",
            path.display()
        )));
    }
    if !path.is_dir() {
        return Err(error::Error::Config(format!(
            "path is not a directory: {}",
            path.display()
        )));
    }

    // Resolve to absolute path
    let abs_path = path
        .canonicalize()
        .map_err(|e| error::Error::Config(format!("cannot resolve path {}: {e}", path.display())))?;

    // Check for duplicate collection name
    if config_db.get_collection(name)?.is_some() {
        return Err(error::Error::Config(format!(
            "collection '{name}' already exists"
        )));
    }

    // Store collection definition
    config_db.set_collection(name, &abs_path.to_string_lossy())?;

    println!("Added collection '{name}' -> {}", abs_path.display());
    Ok(())
}

pub(crate) fn collection_remove(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    name: &str,
) -> error::Result<()> {
    // Verify collection exists
    if config_db.get_collection(name)?.is_none() {
        return Err(error::Error::NotFound {
            kind: "collection",
            name: name.to_string(),
        });
    }

    // Delete from Tantivy index
    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let mut writer = search_index.writer(15_000_000)?;
    search_index.delete_collection(&writer, name);
    writer.commit()?;

    // Delete embeddings and metadata for all documents in this collection
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
    let doc_ids: Vec<u64> = config_db
        .list_all_document_metadata()?
        .into_iter()
        .filter_map(|(doc_id, bytes)| {
            let meta = incremental::DocumentMetadata::deserialize(&bytes)?;
            (meta.collection == name).then_some(doc_id)
        })
        .collect();
    embedding_db.batch_remove(&doc_ids)?;
    config_db.batch_remove_document_metadata(&doc_ids)?;

    // Remove collection definition
    config_db.remove_collection(name)?;

    println!("Removed collection '{name}'");
    Ok(())
}

pub(crate) fn collection_list(config_db: &ConfigDb, json: bool) -> error::Result<()> {
    let collections = config_db.list_collections()?;

    if json {
        println!("{}", collection_list_json_string(&collections)?);
    } else if collections.is_empty() {
        println!("No collections registered.");
    } else {
        for (name, path) in &collections {
            println!("{name}\t{path}");
        }
    }
    Ok(())
}

pub(crate) fn context_add(config_db: &ConfigDb, uri: &str, description: &str) -> error::Result<()> {
    config_db.set_context(uri, description)?;
    println!("Added context for '{uri}'");
    Ok(())
}

pub(crate) fn context_remove(config_db: &ConfigDb, uri: &str) -> error::Result<()> {
    if !config_db.remove_context(uri)? {
        return Err(error::Error::NotFound {
            kind: "context",
            name: uri.to_string(),
        });
    }
    println!("Removed context for '{uri}'");
    Ok(())
}

pub(crate) fn context_list(config_db: &ConfigDb, json: bool) -> error::Result<()> {
    let contexts = config_db.list_contexts()?;

    if json {
        println!("{}", context_list_json_string(&contexts)?);
    } else if contexts.is_empty() {
        println!("No contexts defined.");
    } else {
        for (uri, desc) in &contexts {
            println!("{uri}\t{desc}");
        }
    }
    Ok(())
}

pub(crate) fn cmd_get(config_db: &ConfigDb, args: &cli::GetArgs) -> error::Result<()> {
    let reference = &args.reference;

    // Try to resolve the reference
    let (collection, path) = if let Some(stripped) = reference.strip_prefix('#') {
        // Doc ID reference: look up in metadata
        search::resolve_by_doc_id(config_db, stripped).ok_or_else(|| error::Error::NotFound {
            kind: "document",
            name: format!("#{stripped}"),
        })?
    } else if let Some((coll, path)) = reference.split_once(':') {
        (coll.to_string(), path.to_string())
    } else {
        // Plain path: search all collections
        search::resolve_by_path(config_db, reference).ok_or_else(|| error::Error::NotFound {
            kind: "document",
            name: reference.to_string(),
        })?
    };

    let collection_path =
        config_db
            .get_collection(&collection)?
            .ok_or_else(|| error::Error::NotFound {
                kind: "collection",
                name: collection.clone(),
            })?;

    let full_path = std::path::Path::new(&collection_path).join(&path);

    if args.meta {
        println!("collection: {collection}");
        println!("path: {path}");
        println!("file: {}", full_path.display());
    } else if args.json {
        let content = std::fs::read_to_string(&full_path)?;
        println!(
            "{}",
            get_json_string(&collection, &path, &full_path, Some(&content))?
        );
    } else {
        let content = std::fs::read_to_string(&full_path)?;
        print!("{content}");
    }

    Ok(())
}

pub(crate) fn cmd_multi_get(config_db: &ConfigDb, args: &cli::MultiGetArgs) -> error::Result<()> {
    let glob = globset::Glob::new(&args.pattern)
        .map_err(|e| error::Error::Config(format!("invalid glob pattern: {e}")))?
        .compile_matcher();

    // Collect matching documents
    let mut matches: Vec<(String, String)> = Vec::new(); // (collection, relative_path)

    for (_doc_id, bytes) in config_db.list_all_document_metadata()? {
        if let Some(meta) = incremental::DocumentMetadata::deserialize(&bytes) {
            // Filter by collection if specified
            if let Some(ref coll) = args.collection
                && meta.collection != *coll
            {
                continue;
            }

            if glob.is_match(&meta.relative_path) {
                matches.push((meta.collection, meta.relative_path));
            }
        }
    }

    matches.sort();

    if args.json {
        let mut items = Vec::with_capacity(matches.len());
        for (collection, path) in &matches {
            let collection_path = config_db.get_collection(collection)?;
            let (file, content) = if let Some(ref cp) = collection_path {
                let full_path = std::path::Path::new(cp).join(path);
                let content = if args.full {
                    std::fs::read_to_string(&full_path).ok()
                } else {
                    None
                };
                (Some(full_path.to_string_lossy().to_string()), content)
            } else {
                (None, None)
            };

            items.push(MultiGetJsonItem {
                collection: collection.clone(),
                path: path.clone(),
                file,
                content,
            });
        }

        println!("{}", multi_get_json_string(&items)?);
    } else if args.files {
        for (collection, path) in &matches {
            if let Ok(Some(collection_path)) = config_db.get_collection(collection) {
                let full_path = std::path::Path::new(&collection_path).join(path);
                println!("{}", full_path.display());
            }
        }
    } else if args.full {
        for (collection, path) in &matches {
            if let Ok(Some(collection_path)) = config_db.get_collection(collection) {
                let full_path = std::path::Path::new(&collection_path).join(path);
                println!("--- {collection}:{path} ---");
                if let Ok(content) = std::fs::read_to_string(&full_path) {
                    print!("{content}");
                    if !content.ends_with('\n') {
                        println!();
                    }
                }
            }
        }
    } else if matches.is_empty() {
        println!("No documents match '{}'", args.pattern);
    } else {
        for (collection, path) in &matches {
            println!("{collection}:{path}");
        }
        println!("\n{} match(es)", matches.len());
    }

    Ok(())
}

pub(crate) fn cmd_doctor(json: bool) -> error::Result<()> {
    let report = docbert_core::model_manager::doctor_report();

    if json {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|e| {
                error::Error::Config(format!("failed to serialize doctor report: {e}"))
            })?
        );
        return Ok(());
    }

    println!("Selected device: {}", report.selected_device);
    println!(
        "CUDA support: {}",
        if report.cuda.compiled {
            "compiled in"
        } else {
            "not compiled in"
        }
    );
    if report.cuda.compiled {
        println!(
            "CUDA usable: {}",
            if report.cuda.usable { "yes" } else { "no" }
        );
        if let Some(err) = report.cuda.error {
            println!("CUDA error: {err}");
        }
    }
    println!(
        "Metal support: {}",
        if report.metal.compiled {
            "compiled in"
        } else {
            "not compiled in"
        }
    );
    if report.metal.compiled {
        println!(
            "Metal usable: {}",
            if report.metal.usable { "yes" } else { "no" }
        );
        if let Some(err) = report.metal.error {
            println!("Metal error: {err}");
        }
    }
    if let Some(note) = report.fallback_note {
        println!("Note: {note}");
    }

    Ok(())
}

pub(crate) fn cmd_status(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    model_resolution: &ModelResolution,
    json: bool,
) -> error::Result<()> {
    let collections = config_db.list_collections()?;
    let doc_count = config_db.list_document_ids()?.len();
    let model_name = &model_resolution.model_id;
    let embedding_model = config_db.get_setting(EMBEDDING_MODEL_KEY)?;

    if json {
        println!(
            "{}",
            status_json_string(
                data_dir,
                model_resolution,
                embedding_model.as_deref(),
                collections.len(),
                doc_count,
            )?
        );
    } else {
        println!("Data directory: {}", data_dir.root().display());
        println!("Model: {model_name}");
        println!("Model source: {}", model_resolution.source.as_str());
        if let Some(ref emb) = embedding_model {
            if emb != model_name {
                println!("Embedding model: {emb} (MISMATCH -- run `docbert rebuild`)");
            } else {
                println!("Embedding model: {emb}");
            }
        } else {
            println!("Embedding model: (not set)");
        }
        println!("Collections: {}", collections.len());
        for (name, path) in &collections {
            println!("  {name}: {path}");
        }
        println!("Documents: {doc_count}");
    }
    Ok(())
}

pub(crate) fn cmd_model_show(model_resolution: &ModelResolution, json: bool) {
    if json {
        println!(
            "{}",
            model_show_json_string(model_resolution)
                .expect("model resolution JSON serialization should succeed")
        );
    } else {
        println!("Resolved model: {}", model_resolution.model_id);
        println!("Source: {}", model_resolution.source.as_str());
        if let Some(cli) = model_resolution.cli_model.as_deref() {
            println!("CLI override: {cli}");
        }
        if let Some(env) = model_resolution.env_model.as_deref() {
            println!("{MODEL_ENV_VAR}: {env}");
        }
        if let Some(cfg) = model_resolution.config_model.as_deref() {
            println!("Config setting: {cfg}");
        } else {
            println!("Config setting: (unset)");
        }
    }
}

pub(crate) fn cmd_model_set(config_db: &ConfigDb, model: &str) -> error::Result<()> {
    config_db.set_setting("model_name", model)?;

    let model_path = Path::new(model);
    if model_path.is_dir() {
        let st_config = model_path.join("config_sentence_transformers.json");
        if !st_config.exists() {
            eprintln!(
                "Warning: {} is missing config_sentence_transformers.json; pylate-rs may not load this model.",
                model_path.display()
            );
        }
    }

    println!("Stored model_name: {model}");
    Ok(())
}

pub(crate) fn cmd_model_clear(config_db: &ConfigDb) -> error::Result<()> {
    if config_db.remove_setting("model_name")? {
        println!("Cleared model_name setting.");
    } else {
        println!("model_name setting was already unset.");
    }
    Ok(())
}

/// Settings key used to record which model produced the stored embeddings.
const EMBEDDING_MODEL_KEY: &str = "embedding_model";

fn log_model_runtime(model: &mut ModelManager) -> error::Result<()> {
    let runtime = model.runtime_config()?;
    eprintln!(
        "Embedding runtime: device={}, document_length={}, pylate_batch_size={}",
        runtime.device, runtime.document_length, runtime.embedding_batch_size
    );
    if let Some(note) = runtime.fallback_note {
        eprintln!("Warning: {note}");
    }
    Ok(())
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

/// Create a progress bar with docbert's standard styling.
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

/// Finish a progress bar and leave the terminal in a clean state.
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

fn initialize_indexing_runtime(data_dir: &DataDir, model_id: &str) -> error::Result<IndexingRuntime> {
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
        let count = ingestion::ingest_loaded_documents(
            &runtime.search_index,
            &mut writer,
            collection,
            &document_batch.documents,
        )?;
        eprintln!("  Indexed {count} documents");
    }

    if embed_documents {
        let mut pb = create_progress_bar(document_batch.documents.len(), "Chunking");
        let docs_to_embed = indexing_workflow::chunk_documents_for_embedding(
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

    incremental::batch_store_metadata(config_db, collection, &document_batch.metadata_files)?;

    Ok(())
}

pub(crate) fn cmd_rebuild(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    args: &cli::RebuildArgs,
    model_id: &str,
) -> error::Result<()> {
    let collections =
        indexing_workflow::resolve_target_collections(config_db, args.collection.as_deref())?;

    if collections.is_empty() {
        eprintln!("No collections to rebuild.");
        return Ok(());
    }

    let mut runtime = initialize_indexing_runtime(data_dir, model_id)?;

    for (name, path) in &collections {
        let root = std::path::Path::new(path);
        if !root.is_dir() {
            eprintln!("Warning: collection '{name}' path does not exist: {path}");
            continue;
        }

        eprintln!("Rebuilding collection '{name}'...");

        // Delete existing Tantivy entries for this collection
        if !args.embeddings_only {
            let mut writer = runtime.search_index.writer(15_000_000)?;
            runtime.search_index.delete_collection(&writer, name);
            writer.commit()?;
        }

        // Delete existing embeddings and metadata for this collection
        let old_doc_ids: Vec<u64> = config_db
            .list_all_document_metadata()?
            .into_iter()
            .filter_map(|(doc_id, bytes)| {
                let meta = incremental::DocumentMetadata::deserialize(&bytes)?;
                (meta.collection == *name).then_some(doc_id)
            })
            .collect();
        if !args.index_only {
            runtime.embedding_db.batch_remove(&old_doc_ids)?;
        }
        config_db.batch_remove_document_metadata(&old_doc_ids)?;

        // Discover files
        let files = walker::discover_files(root)?;
        eprintln!("  Found {} files", files.len());

        if !args.embeddings_only || !args.index_only {
            eprintln!("  Loading {} files...", files.len());
        }
        let document_batch = indexing_workflow::load_rebuild_batch(name, &files, args);
        process_document_batch(
            config_db,
            &mut runtime,
            name,
            &document_batch,
            !args.embeddings_only,
            !args.index_only,
        )?;

        eprintln!("  Done.");
    }

    // Record which model produced these embeddings
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
    let collections =
        indexing_workflow::resolve_target_collections(config_db, args.collection.as_deref())?;

    if collections.is_empty() {
        eprintln!("No collections to sync.");
        return Ok(());
    }

    // Check if the model changed since last embed
    if let Some(prev_model) = config_db.get_setting(EMBEDDING_MODEL_KEY)?
        && prev_model != model_id
    {
        eprintln!(
            "Warning: embeddings were computed with '{prev_model}', but current model is '{model_id}'."
        );
        eprintln!("Mixing embeddings from different models produces invalid results.");
        eprintln!("Run `docbert rebuild` to re-embed all documents with the new model.");
        return Err(error::Error::Config(format!(
            "model mismatch: embeddings use '{prev_model}', current is '{model_id}'"
        )));
    }

    let mut runtime = initialize_indexing_runtime(data_dir, model_id)?;

    for (name, path) in &collections {
        let root = std::path::Path::new(path);
        if !root.is_dir() {
            eprintln!("Warning: collection '{name}' path does not exist: {path}");
            continue;
        }

        // Discover files on disk
        let files = walker::discover_files(root)?;

        // Diff against stored metadata
        let diff = incremental::diff_collection(config_db, name, &files)?;

        if diff.new_files.is_empty() && diff.changed_files.is_empty() && diff.deleted_ids.is_empty() {
            eprintln!("Collection '{name}' is up to date.");
            continue;
        }

        eprintln!("Syncing collection '{name}'...");
        eprintln!(
            "  {} new, {} changed, {} deleted",
            diff.new_files.len(),
            diff.changed_files.len(),
            diff.deleted_ids.len()
        );

        // Handle deleted documents
        if !diff.deleted_ids.is_empty() {
            let mut writer = runtime.search_index.writer(15_000_000)?;
            for &doc_id in &diff.deleted_ids {
                let display = search::short_doc_id(doc_id);
                runtime.search_index.delete_document(&writer, &display);
            }
            writer.commit()?;

            runtime.embedding_db.batch_remove(&diff.deleted_ids)?;
            config_db.batch_remove_document_metadata(&diff.deleted_ids)?;
            eprintln!("  Removed {} documents", diff.deleted_ids.len());
        }

        // Combine new and changed files for processing
        let files_to_process: Vec<_> = diff
            .new_files
            .into_iter()
            .chain(diff.changed_files)
            .collect();

        if !files_to_process.is_empty() {
            eprintln!("  Loading {} files...", files_to_process.len());
            let document_batch = indexing_workflow::load_sync_batch(name, &files_to_process);
            process_document_batch(config_db, &mut runtime, name, &document_batch, true, true)?;
        }

        eprintln!("  Done.");
    }

    // Record which model produced these embeddings
    config_db.set_setting(EMBEDDING_MODEL_KEY, model_id)?;

    eprintln!("Sync complete.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use docbert_core::model_manager::ModelSource;

    use super::*;

    #[test]
    fn collection_list_json_snapshot() {
        let json = collection_list_json_string(&[
            ("notes".to_string(), "/tmp/notes".to_string()),
            ("docs".to_string(), "/tmp/docs".to_string()),
        ])
        .unwrap();

        assert_eq!(
            json,
            "[{\"name\":\"notes\",\"path\":\"/tmp/notes\"},{\"name\":\"docs\",\"path\":\"/tmp/docs\"}]"
        );
    }

    #[test]
    fn context_list_json_snapshot() {
        let json =
            context_list_json_string(&[("bert://notes".to_string(), "Personal notes".to_string())])
                .unwrap();

        assert_eq!(
            json,
            "[{\"uri\":\"bert://notes\",\"description\":\"Personal notes\"}]"
        );
    }

    #[test]
    fn get_json_snapshot() {
        let json = get_json_string(
            "notes",
            "hello.md",
            Path::new("/tmp/notes/hello.md"),
            Some("Hello\nWorld\n"),
        )
        .unwrap();

        assert_eq!(
            json,
            "{\"collection\":\"notes\",\"path\":\"hello.md\",\"file\":\"/tmp/notes/hello.md\",\"content\":\"Hello\\nWorld\\n\"}"
        );
    }

    #[test]
    fn multi_get_json_snapshot() {
        let json = multi_get_json_string(&[
            MultiGetJsonItem {
                collection: "notes".to_string(),
                path: "hello.md".to_string(),
                file: Some("/tmp/notes/hello.md".to_string()),
                content: Some("Hello\n".to_string()),
            },
            MultiGetJsonItem {
                collection: "docs".to_string(),
                path: "missing.md".to_string(),
                file: None,
                content: None,
            },
        ])
        .unwrap();

        assert_eq!(
            json,
            "[{\"collection\":\"notes\",\"path\":\"hello.md\",\"file\":\"/tmp/notes/hello.md\",\"content\":\"Hello\\n\"},{\"collection\":\"docs\",\"path\":\"missing.md\"}]"
        );
    }

    #[test]
    fn status_json_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let model_resolution = ModelResolution {
            model_id: "lightonai/ColBERT-Zero".to_string(),
            source: ModelSource::Config,
            env_model: None,
            config_model: Some("lightonai/ColBERT-Zero".to_string()),
            cli_model: None,
        };

        let with_embedding = status_json_string(
            &data_dir,
            &model_resolution,
            Some("lightonai/ColBERT-Zero"),
            2,
            15,
        )
        .unwrap();
        assert_eq!(
            with_embedding,
            format!(
                "{{\"data_dir\":\"{}\",\"model\":\"lightonai/ColBERT-Zero\",\"model_source\":\"config\",\"embedding_model\":\"lightonai/ColBERT-Zero\",\"collections\":2,\"documents\":15}}",
                data_dir.root().display()
            )
        );

        let without_embedding =
            status_json_string(&data_dir, &model_resolution, None, 2, 15).unwrap();
        assert_eq!(
            without_embedding,
            format!(
                "{{\"data_dir\":\"{}\",\"model\":\"lightonai/ColBERT-Zero\",\"model_source\":\"config\",\"embedding_model\":null,\"collections\":2,\"documents\":15}}",
                data_dir.root().display()
            )
        );
    }

    #[test]
    fn model_show_json_snapshot() {
        let resolution = ModelResolution {
            model_id: "cli/model".to_string(),
            source: ModelSource::Cli,
            cli_model: Some("cli/model".to_string()),
            env_model: Some("env/model".to_string()),
            config_model: None,
        };

        let json = model_show_json_string(&resolution).unwrap();
        assert_eq!(
            json,
            "{\"resolved\":\"cli/model\",\"source\":\"cli\",\"cli\":\"cli/model\",\"env\":\"env/model\",\"config\":null}"
        );
    }
}
