use std::path::Path;

use clap::Parser;
use docbert::{
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
    mcp,
    model_manager::{MODEL_ENV_VAR, ModelResolution, resolve_model},
    search,
    walker,
};
use kdam::{BarExt, Spinner, tqdm};
use tracing_subscriber::EnvFilter;

mod cli;

use cli::{Cli, CollectionAction, Command, ContextAction};

fn init_tracing(verbose: u8) {
    let filter = if let Ok(env) = std::env::var("DOCBERT_LOG") {
        EnvFilter::new(env)
    } else {
        match verbose {
            1 => EnvFilter::new("info"),
            2 => EnvFilter::new("debug"),
            _ => EnvFilter::new("trace"),
        }
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .without_time()
        .init();
}

fn main() -> error::Result<()> {
    let cli = Cli::parse();

    if cli.verbose > 0 {
        init_tracing(cli.verbose);
    }

    // Handle completions early (doesn't need data_dir or config_db)
    if let Command::Completions(args) = &cli.command {
        args.generate();
        return Ok(());
    }

    let data_dir = DataDir::resolve(cli.data_dir.as_deref())?;
    let config_db = ConfigDb::open(&data_dir.config_db())?;
    let model_resolution = resolve_model(&config_db, cli.model.as_deref())?;

    match cli.command {
        Command::Completions(_) => unreachable!(), // Handled above
        Command::Collection { action } => match action {
            CollectionAction::Add { path, name } => {
                collection_add(&config_db, &path, &name)?;
            }
            CollectionAction::Remove { name } => {
                collection_remove(&config_db, &data_dir, &name)?;
            }
            CollectionAction::List { json } => {
                collection_list(&config_db, json)?;
            }
        },
        Command::Context { action } => match action {
            ContextAction::Add { uri, description } => {
                context_add(&config_db, &uri, &description)?;
            }
            ContextAction::Remove { uri } => {
                context_remove(&config_db, &uri)?;
            }
            ContextAction::List { json } => {
                context_list(&config_db, json)?;
            }
        },
        Command::Search(args) => {
            let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
            let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
            let mut model =
                ModelManager::with_model_id(model_resolution.model_id.clone());

            let params = search::SearchParams {
                query: args.query.clone(),
                count: args.count,
                collection: args.collection.clone(),
                min_score: args.min_score,
                bm25_only: args.bm25_only,
                no_fuzzy: args.no_fuzzy,
                all: args.all,
            };

            let results = search::execute_search(
                &params,
                &search_index,
                &embedding_db,
                &mut model,
            )?;

            if args.json {
                search::format_json(&results, &args.query);
            } else if args.files {
                search::format_files(&results, &config_db);
            } else {
                search::format_human(&results);
            }
        }
        Command::Ssearch(args) => {
            let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
            let mut model =
                ModelManager::with_model_id(model_resolution.model_id.clone());

            let params = search::SemanticSearchParams {
                query: args.query.clone(),
                count: args.count,
                min_score: args.min_score,
                all: args.all,
            };

            let results = search::execute_semantic_search(
                &params,
                &config_db,
                &embedding_db,
                &mut model,
            )?;

            if args.json {
                search::format_json(&results, &args.query);
            } else if args.files {
                search::format_files(&results, &config_db);
            } else {
                search::format_human(&results);
            }
        }
        Command::Get(args) => {
            cmd_get(&config_db, &args)?;
        }
        Command::MultiGet(args) => {
            cmd_multi_get(&config_db, &args)?;
        }
        Command::Rebuild(args) => {
            cmd_rebuild(
                &config_db,
                &data_dir,
                &args,
                &model_resolution.model_id,
            )?;
        }
        Command::Sync(args) => {
            cmd_sync(&config_db, &data_dir, &args, &model_resolution.model_id)?;
        }
        Command::Status(args) => {
            cmd_status(&config_db, &data_dir, &model_resolution, args.json)?;
        }
        Command::Mcp => {
            mcp::run_mcp(data_dir, config_db, model_resolution.model_id)?;
        }
        Command::Model { action } => match action {
            cli::ModelAction::Show { json } => {
                cmd_model_show(&model_resolution, json);
            }
            cli::ModelAction::Set { model } => {
                cmd_model_set(&config_db, &model)?;
            }
            cli::ModelAction::Clear => {
                cmd_model_clear(&config_db)?;
            }
        },
    }

    Ok(())
}

fn collection_add(
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
    let abs_path = path.canonicalize().map_err(|e| {
        error::Error::Config(format!(
            "cannot resolve path {}: {e}",
            path.display()
        ))
    })?;

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

fn collection_remove(
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

fn collection_list(config_db: &ConfigDb, json: bool) -> error::Result<()> {
    let collections = config_db.list_collections()?;

    if json {
        print!("[");
        for (i, (name, path)) in collections.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{{\"name\":\"{name}\",\"path\":\"{path}\"}}");
        }
        println!("]");
    } else if collections.is_empty() {
        println!("No collections registered.");
    } else {
        for (name, path) in &collections {
            println!("{name}\t{path}");
        }
    }
    Ok(())
}

fn context_add(
    config_db: &ConfigDb,
    uri: &str,
    description: &str,
) -> error::Result<()> {
    config_db.set_context(uri, description)?;
    println!("Added context for '{uri}'");
    Ok(())
}

fn context_remove(config_db: &ConfigDb, uri: &str) -> error::Result<()> {
    if !config_db.remove_context(uri)? {
        return Err(error::Error::NotFound {
            kind: "context",
            name: uri.to_string(),
        });
    }
    println!("Removed context for '{uri}'");
    Ok(())
}

fn context_list(config_db: &ConfigDb, json: bool) -> error::Result<()> {
    let contexts = config_db.list_contexts()?;

    if json {
        print!("[");
        for (i, (uri, desc)) in contexts.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{{\"uri\":\"{uri}\",\"description\":\"{desc}\"}}");
        }
        println!("]");
    } else if contexts.is_empty() {
        println!("No contexts defined.");
    } else {
        for (uri, desc) in &contexts {
            println!("{uri}\t{desc}");
        }
    }
    Ok(())
}

fn cmd_get(config_db: &ConfigDb, args: &cli::GetArgs) -> error::Result<()> {
    let reference = &args.reference;

    // Try to resolve the reference
    let (collection, path) = if let Some(stripped) = reference.strip_prefix('#')
    {
        // Doc ID reference — look up in metadata
        search::resolve_by_doc_id(config_db, stripped).ok_or_else(|| {
            error::Error::NotFound {
                kind: "document",
                name: format!("#{stripped}"),
            }
        })?
    } else if let Some((coll, path)) = reference.split_once(':') {
        (coll.to_string(), path.to_string())
    } else {
        // Plain path — search all collections
        search::resolve_by_path(config_db, reference).ok_or_else(|| {
            error::Error::NotFound {
                kind: "document",
                name: reference.to_string(),
            }
        })?
    };

    let collection_path =
        config_db.get_collection(&collection)?.ok_or_else(|| {
            error::Error::NotFound {
                kind: "collection",
                name: collection.clone(),
            }
        })?;

    let full_path = std::path::Path::new(&collection_path).join(&path);

    if args.meta {
        println!("collection: {collection}");
        println!("path: {path}");
        println!("file: {}", full_path.display());
    } else if args.json {
        print!(
            "{{\"collection\":\"{collection}\",\"path\":\"{path}\",\"file\":\"{}\"",
            full_path.display()
        );
        if !args.meta {
            let content = std::fs::read_to_string(&full_path)?;
            print!(",\"content\":{}", search::json_escape(&content));
        }
        println!("}}");
    } else {
        let content = std::fs::read_to_string(&full_path)?;
        print!("{content}");
    }

    Ok(())
}

fn cmd_multi_get(
    config_db: &ConfigDb,
    args: &cli::MultiGetArgs,
) -> error::Result<()> {
    let glob = globset::Glob::new(&args.pattern)
        .map_err(|e| {
            error::Error::Config(format!("invalid glob pattern: {e}"))
        })?
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
        print!("[");
        for (i, (collection, path)) in matches.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            let collection_path = config_db.get_collection(collection)?;
            print!(
                "{{\"collection\":{},\"path\":{}",
                search::json_escape(collection),
                search::json_escape(path),
            );
            if let Some(ref cp) = collection_path {
                let full_path = std::path::Path::new(cp).join(path);
                print!(
                    ",\"file\":{}",
                    search::json_escape(&full_path.to_string_lossy()),
                );
                if args.full
                    && let Ok(content) = std::fs::read_to_string(&full_path)
                {
                    print!(",\"content\":{}", search::json_escape(&content));
                }
            }
            print!("}}");
        }
        println!("]");
    } else if args.files {
        for (collection, path) in &matches {
            if let Ok(Some(collection_path)) =
                config_db.get_collection(collection)
            {
                let full_path =
                    std::path::Path::new(&collection_path).join(path);
                println!("{}", full_path.display());
            }
        }
    } else if args.full {
        for (collection, path) in &matches {
            if let Ok(Some(collection_path)) =
                config_db.get_collection(collection)
            {
                let full_path =
                    std::path::Path::new(&collection_path).join(path);
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

fn cmd_status(
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
        let emb_model_json = match &embedding_model {
            Some(m) => search::json_escape(m),
            None => "null".to_string(),
        };
        println!(
            "{{\"data_dir\":\"{}\",\"model\":\"{model_name}\",\"model_source\":\"{}\",\"embedding_model\":{emb_model_json},\"collections\":{},\"documents\":{doc_count}}}",
            data_dir.root().display(),
            model_resolution.source.as_str(),
            collections.len()
        );
    } else {
        println!("Data directory: {}", data_dir.root().display());
        println!("Model: {model_name}");
        println!("Model source: {}", model_resolution.source.as_str());
        if let Some(ref emb) = embedding_model {
            if emb != model_name {
                println!(
                    "Embedding model: {emb} (MISMATCH -- run `docbert rebuild`)"
                );
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

fn json_optional(value: Option<&str>) -> String {
    match value {
        Some(v) => search::json_escape(v),
        None => "null".to_string(),
    }
}

fn cmd_model_show(model_resolution: &ModelResolution, json: bool) {
    if json {
        println!(
            "{{\"resolved\":{},\"source\":{},\"cli\":{},\"env\":{},\"config\":{}}}",
            search::json_escape(&model_resolution.model_id),
            search::json_escape(model_resolution.source.as_str()),
            json_optional(model_resolution.cli_model.as_deref()),
            json_optional(model_resolution.env_model.as_deref()),
            json_optional(model_resolution.config_model.as_deref()),
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

fn cmd_model_set(config_db: &ConfigDb, model: &str) -> error::Result<()> {
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

fn cmd_model_clear(config_db: &ConfigDb) -> error::Result<()> {
    if config_db.remove_setting("model_name")? {
        println!("Cleared model_name setting.");
    } else {
        println!("model_name setting was already unset.");
    }
    Ok(())
}

/// Settings key for tracking which model produced the stored embeddings.
const EMBEDDING_MODEL_KEY: &str = "embedding_model";

/// Batch size for embedding operations (balances progress granularity vs overhead)
const EMBEDDING_BATCH_SIZE: usize = 32;

/// Create a progress bar with consistent styling.
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

/// Finalize a progress bar (clear spinner, show final state).
fn finish_progress_bar(pb: &mut kdam::Bar) {
    let _ = pb.set_bar_format(
        "{desc suffix=' '}|{animation}| {count}/{total} [{percentage:.0}%] in {elapsed human=true} ({rate:.1}/s)",
    );
    let _ = pb.clear();
    let _ = pb.refresh();
    eprintln!();
}

fn cmd_rebuild(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    args: &cli::RebuildArgs,
    model_id: &str,
) -> error::Result<()> {
    let collections: Vec<(String, String)> =
        if let Some(ref name) = args.collection {
            let path = config_db.get_collection(name)?.ok_or_else(|| {
                error::Error::NotFound {
                    kind: "collection",
                    name: name.clone(),
                }
            })?;
            vec![(name.clone(), path)]
        } else {
            config_db.list_collections()?
        };

    if collections.is_empty() {
        eprintln!("No collections to rebuild.");
        return Ok(());
    }

    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
    let mut model = ModelManager::with_model_id(model_id.to_string());
    let chunking_config = chunking::resolve_chunking_config(model_id);
    if let Some(doc_len) = chunking_config.document_length {
        eprintln!(
            "Using document_length {doc_len} from config_sentence_transformers.json (chunk size ~{} chars).",
            chunking_config.chunk_size
        );
    }

    for (name, path) in &collections {
        let root = std::path::Path::new(path);
        if !root.is_dir() {
            eprintln!(
                "Warning: collection '{name}' path does not exist: {path}"
            );
            continue;
        }

        eprintln!("Rebuilding collection '{name}'...");

        // Delete existing Tantivy entries for this collection
        if !args.embeddings_only {
            let mut writer = search_index.writer(15_000_000)?;
            search_index.delete_collection(&writer, name);
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
            embedding_db.batch_remove(&old_doc_ids)?;
        }
        config_db.batch_remove_document_metadata(&old_doc_ids)?;

        // Discover files
        let files = walker::discover_files(root)?;
        eprintln!("  Found {} files", files.len());

        let loaded_docs = if args.embeddings_only && args.index_only {
            Vec::new()
        } else {
            eprintln!("  Loading {} files...", files.len());
            ingestion::load_documents(name, &files)
        };

        // Re-index into Tantivy
        if !args.embeddings_only {
            let mut writer = search_index.writer(15_000_000)?;
            let count = ingestion::ingest_loaded_documents(
                &search_index,
                &mut writer,
                name,
                &loaded_docs,
            )?;
            eprintln!("  Indexed {count} documents");
        }

        // Re-compute embeddings with chunking for long documents
        if !args.index_only {
            // Chunk documents with progress
            let mut pb = create_progress_bar(loaded_docs.len(), "Chunking");
            let mut docs_to_embed: Vec<(u64, String)> = Vec::new();
            for doc in loaded_docs {
                let chunks = chunking::chunk_text(
                    &doc.content,
                    chunking_config.chunk_size,
                    chunking_config.overlap,
                );
                for chunk in chunks {
                    let chunk_id =
                        chunking::chunk_doc_id(doc.doc_num_id, chunk.index);
                    docs_to_embed.push((chunk_id, chunk.text));
                }
                let _ = pb.update(1);
            }
            finish_progress_bar(&mut pb);

            // Embed documents in batches with progress
            if !docs_to_embed.is_empty() {
                let total_chunks = docs_to_embed.len();
                let mut pb = create_progress_bar(total_chunks, "Embedding");
                let mut embedded_count = 0;

                while !docs_to_embed.is_empty() {
                    let take = docs_to_embed.len().min(EMBEDDING_BATCH_SIZE);
                    let batch_vec: Vec<(u64, String)> =
                        docs_to_embed.drain(..take).collect();
                    let count = embedding::embed_and_store(
                        &mut model,
                        &embedding_db,
                        batch_vec,
                    )?;
                    embedded_count += count;
                    let _ = pb.update_to(embedded_count);
                }
                finish_progress_bar(&mut pb);
            }
        }

        // Store metadata for all files
        incremental::batch_store_metadata(config_db, name, &files)?;

        eprintln!("  Done.");
    }

    // Record which model produced these embeddings
    config_db.set_setting(EMBEDDING_MODEL_KEY, model_id)?;

    eprintln!("Rebuild complete.");
    Ok(())
}

fn cmd_sync(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    args: &cli::SyncArgs,
    model_id: &str,
) -> error::Result<()> {
    let collections: Vec<(String, String)> =
        if let Some(ref name) = args.collection {
            let path = config_db.get_collection(name)?.ok_or_else(|| {
                error::Error::NotFound {
                    kind: "collection",
                    name: name.clone(),
                }
            })?;
            vec![(name.clone(), path)]
        } else {
            config_db.list_collections()?
        };

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

    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
    let mut model = ModelManager::with_model_id(model_id.to_string());
    let chunking_config = chunking::resolve_chunking_config(model_id);
    if let Some(doc_len) = chunking_config.document_length {
        eprintln!(
            "Using document_length {doc_len} from config_sentence_transformers.json (chunk size ~{} chars).",
            chunking_config.chunk_size
        );
    }

    for (name, path) in &collections {
        let root = std::path::Path::new(path);
        if !root.is_dir() {
            eprintln!(
                "Warning: collection '{name}' path does not exist: {path}"
            );
            continue;
        }

        // Discover files on disk
        let files = walker::discover_files(root)?;

        // Diff against stored metadata
        let diff = incremental::diff_collection(config_db, name, &files)?;

        if diff.new_files.is_empty()
            && diff.changed_files.is_empty()
            && diff.deleted_ids.is_empty()
        {
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
            let mut writer = search_index.writer(15_000_000)?;
            for &doc_id in &diff.deleted_ids {
                let display = search::short_doc_id(doc_id);
                search_index.delete_document(&writer, &display);
            }
            writer.commit()?;

            embedding_db.batch_remove(&diff.deleted_ids)?;
            config_db.batch_remove_document_metadata(&diff.deleted_ids)?;
            eprintln!("  Removed {} documents", diff.deleted_ids.len());
        }

        // Combine new and changed files for processing
        let files_to_process: Vec<_> = diff
            .new_files
            .into_iter()
            .chain(diff.changed_files.into_iter())
            .collect();

        if !files_to_process.is_empty() {
            eprintln!("  Loading {} files...", files_to_process.len());
            let loaded_docs =
                ingestion::load_documents(name, &files_to_process);

            // Index into Tantivy (add_document handles delete-before-add)
            let mut writer = search_index.writer(15_000_000)?;
            let count = ingestion::ingest_loaded_documents(
                &search_index,
                &mut writer,
                name,
                &loaded_docs,
            )?;
            eprintln!("  Indexed {count} documents");

            // Chunk documents with progress
            let mut pb = create_progress_bar(loaded_docs.len(), "Chunking");
            let mut docs_to_embed: Vec<(u64, String)> = Vec::new();
            for doc in loaded_docs.into_iter() {
                let chunks = chunking::chunk_text(
                    &doc.content,
                    chunking_config.chunk_size,
                    chunking_config.overlap,
                );
                for chunk in chunks {
                    let chunk_id =
                        chunking::chunk_doc_id(doc.doc_num_id, chunk.index);
                    docs_to_embed.push((chunk_id, chunk.text));
                }
                let _ = pb.update(1);
            }
            finish_progress_bar(&mut pb);

            // Embed documents in batches with progress
            if !docs_to_embed.is_empty() {
                let total_chunks = docs_to_embed.len();
                let mut pb = create_progress_bar(total_chunks, "Embedding");
                let mut embedded_count = 0;

                while !docs_to_embed.is_empty() {
                    let take = docs_to_embed.len().min(EMBEDDING_BATCH_SIZE);
                    let batch_vec: Vec<(u64, String)> =
                        docs_to_embed.drain(..take).collect();
                    let count = embedding::embed_and_store(
                        &mut model,
                        &embedding_db,
                        batch_vec,
                    )?;
                    embedded_count += count;
                    let _ = pb.update_to(embedded_count);
                }
                finish_progress_bar(&mut pb);
            }

            // Store metadata for processed files
            incremental::batch_store_metadata(
                config_db,
                name,
                &files_to_process,
            )?;
        }

        eprintln!("  Done.");
    }

    // Record which model produced these embeddings
    config_db.set_setting(EMBEDDING_MODEL_KEY, model_id)?;

    eprintln!("Sync complete.");
    Ok(())
}
