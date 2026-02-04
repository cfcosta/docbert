use clap::Parser;

pub mod cli;
pub mod config_db;
pub mod data_dir;
pub mod doc_id;
pub mod embedding_db;
pub mod error;
pub mod tantivy_index;

use cli::{Cli, CollectionAction, Command, ContextAction};
use data_dir::DataDir;

fn main() -> error::Result<()> {
    let cli = Cli::parse();
    let _data_dir = DataDir::resolve(cli.data_dir.as_deref())?;

    match cli.command {
        Command::Collection { action } => match action {
            CollectionAction::Add { path, name } => {
                eprintln!("TODO: collection add {name} -> {}", path.display());
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
