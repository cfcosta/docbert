use clap::Parser;

pub mod cli;
pub mod config_db;
pub mod data_dir;
pub mod doc_id;
pub mod embedding;
pub mod embedding_db;
pub mod error;
pub mod incremental;
pub mod ingestion;
pub mod model_manager;
pub mod reranker;
pub mod search;
pub mod tantivy_index;
pub mod walker;

use cli::{Cli, CollectionAction, Command, ContextAction};
use config_db::ConfigDb;
use data_dir::DataDir;
use embedding_db::EmbeddingDb;
use model_manager::ModelManager;
use tantivy_index::SearchIndex;

fn main() -> error::Result<()> {
    let cli = Cli::parse();
    let data_dir = DataDir::resolve(cli.data_dir.as_deref())?;
    let config_db = ConfigDb::open(&data_dir.config_db())?;

    match cli.command {
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
            let mut model = ModelManager::default();

            let results = search::execute_search(
                &args,
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
        Command::Get(args) => {
            cmd_get(&config_db, &args)?;
        }
        Command::MultiGet(args) => {
            eprintln!("TODO: multi-get {}", args.pattern);
        }
        Command::Rebuild(_args) => {
            eprintln!("TODO: rebuild");
        }
        Command::Status(args) => {
            cmd_status(&config_db, &data_dir, args.json)?;
        }
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

    // Delete embeddings for all documents in this collection
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
    for doc_id in config_db.list_document_ids()? {
        if let Some(bytes) = config_db.get_document_metadata(doc_id)?
            && let Some(meta) =
                incremental::DocumentMetadata::deserialize(&bytes)
            && meta.collection == name
        {
            embedding_db.remove(doc_id)?;
            config_db.remove_document_metadata(doc_id)?;
        }
    }

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
        resolve_by_doc_id(config_db, stripped)?
    } else if let Some((coll, path)) = reference.split_once(':') {
        (coll.to_string(), path.to_string())
    } else {
        // Plain path — search all collections
        resolve_by_path(config_db, reference)?
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
            print!(",\"content\":");
            search::print_json_string_pub(&content);
        }
        println!("}}");
    } else {
        let content = std::fs::read_to_string(&full_path)?;
        print!("{content}");
    }

    Ok(())
}

fn resolve_by_doc_id(
    config_db: &ConfigDb,
    short_id: &str,
) -> error::Result<(String, String)> {
    for doc_id in config_db.list_document_ids()? {
        if let Some(bytes) = config_db.get_document_metadata(doc_id)?
            && let Some(meta) =
                incremental::DocumentMetadata::deserialize(&bytes)
        {
            let did =
                doc_id::DocumentId::new(&meta.collection, &meta.relative_path);
            if did.to_string().contains(short_id) || did.short == short_id {
                return Ok((meta.collection, meta.relative_path));
            }
        }
    }
    Err(error::Error::NotFound {
        kind: "document",
        name: format!("#{short_id}"),
    })
}

fn resolve_by_path(
    config_db: &ConfigDb,
    path: &str,
) -> error::Result<(String, String)> {
    for doc_id in config_db.list_document_ids()? {
        if let Some(bytes) = config_db.get_document_metadata(doc_id)?
            && let Some(meta) = incremental::DocumentMetadata::deserialize(&bytes)
            && meta.relative_path == path
        {
            return Ok((meta.collection, meta.relative_path));
        }
    }
    Err(error::Error::NotFound {
        kind: "document",
        name: path.to_string(),
    })
}

fn cmd_status(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    json: bool,
) -> error::Result<()> {
    let collections = config_db.list_collections()?;
    let doc_count = config_db.list_document_ids()?.len();
    let model_name = config_db
        .get_setting_or("model_name", "lightonai/GTE-ModernColBERT-v1")?;

    if json {
        println!(
            "{{\"data_dir\":\"{}\",\"model\":\"{model_name}\",\"collections\":{},\"documents\":{doc_count}}}",
            data_dir.root().display(),
            collections.len()
        );
    } else {
        println!("Data directory: {}", data_dir.root().display());
        println!("Model: {model_name}");
        println!("Collections: {}", collections.len());
        for (name, path) in &collections {
            println!("  {name}: {path}");
        }
        println!("Documents: {doc_count}");
    }
    Ok(())
}
