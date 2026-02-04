use clap::Parser;

pub mod cli;
pub mod config_db;
pub mod data_dir;
pub mod doc_id;
pub mod embedding_db;
pub mod error;
pub mod ingestion;
pub mod model_manager;
pub mod tantivy_index;
pub mod walker;

use cli::{Cli, CollectionAction, Command, ContextAction};
use config_db::ConfigDb;
use data_dir::DataDir;

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
                eprintln!("TODO: collection remove {name}");
            }
            CollectionAction::List { json: _ } => {
                eprintln!("TODO: collection list");
            }
        },
        Command::Context { action } => match action {
            ContextAction::Add { uri, description } => {
                eprintln!("TODO: context add {uri} {description}");
            }
            ContextAction::Remove { uri } => {
                eprintln!("TODO: context remove {uri}");
            }
            ContextAction::List { json: _ } => {
                eprintln!("TODO: context list");
            }
        },
        Command::Search(args) => {
            eprintln!("TODO: search {}", args.query);
        }
        Command::Get(args) => {
            eprintln!("TODO: get {}", args.reference);
        }
        Command::MultiGet(args) => {
            eprintln!("TODO: multi-get {}", args.pattern);
        }
        Command::Rebuild(_args) => {
            eprintln!("TODO: rebuild");
        }
        Command::Status(_args) => {
            eprintln!("TODO: status");
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
