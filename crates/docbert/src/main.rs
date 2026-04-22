use std::path::{Path, PathBuf};

use clap::Parser;
use docbert_core::{ConfigDb, DataDir, error, model_manager::resolve_model};
use tracing_subscriber::EnvFilter;

mod cli;
mod commands;
mod indexing;
mod mcp;
mod runtime;
mod snapshots;
mod web;

use cli::{Cli, CollectionAction, Command, ContextAction};

/// Resolve the data directory using this priority order:
/// 1. an explicit path, such as `--data-dir`
/// 2. the `DOCBERT_DATA_DIR` environment variable
/// 3. the XDG data directory (`~/.local/share/docbert/`)
///
/// Creates the directory, along with any missing parents, if needed.
fn resolve_data_dir(explicit: Option<&Path>) -> error::Result<DataDir> {
    let root = if let Some(path) = explicit {
        path.to_path_buf()
    } else if let Ok(val) = std::env::var("DOCBERT_DATA_DIR") {
        PathBuf::from(val)
    } else {
        xdg::BaseDirectories::with_prefix("docbert")
            .get_data_home()
            .ok_or_else(|| {
                error::Error::Config(
                    "could not determine XDG data home directory".into(),
                )
            })?
    };

    std::fs::create_dir_all(&root)
        .map_err(|_| error::Error::DataDir(root.clone()))?;

    Ok(DataDir::new(root))
}

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

    // Handle commands that don't need data_dir or config_db early.
    if let Command::Completions(args) = &cli.command {
        args.generate();
        return Ok(());
    }
    if let Command::Doctor(args) = &cli.command {
        commands::model::doctor(args.json)?;
        return Ok(());
    }

    let data_dir = resolve_data_dir(cli.data_dir.as_deref())?;

    if let Command::Mcp = &cli.command {
        let model_id = {
            let config_db = ConfigDb::open(&data_dir.config_db())?;
            resolve_model(&config_db, cli.model.as_deref())?.model_id
        };
        mcp::run_mcp(data_dir, model_id)?;
        return Ok(());
    }

    if let Command::Web(args) = &cli.command {
        let model_id = {
            let config_db = ConfigDb::open(&data_dir.config_db())?;
            resolve_model(&config_db, cli.model.as_deref())?.model_id
        };
        web::run(args, data_dir, model_id)?;
        return Ok(());
    }

    let config_db = ConfigDb::open(&data_dir.config_db())?;
    let model_resolution = resolve_model(&config_db, cli.model.as_deref())?;

    match cli.command {
        Command::Completions(_) => unreachable!(), // Handled above
        Command::Doctor(_) => unreachable!(),      // Handled above
        Command::Collection { action } => match action {
            CollectionAction::Add { path, name } => {
                commands::collections::add(&config_db, &path, &name)?;
            }
            CollectionAction::Remove { name } => {
                commands::collections::remove(&config_db, &data_dir, &name)?;
            }
            CollectionAction::List { json } => {
                commands::collections::list(&config_db, json)?;
            }
        },
        Command::Context { action } => match action {
            ContextAction::Add { uri, description } => {
                commands::contexts::add(&config_db, &uri, &description)?;
            }
            ContextAction::Remove { uri } => {
                commands::contexts::remove(&config_db, &uri)?;
            }
            ContextAction::List { json } => {
                commands::contexts::list(&config_db, json)?;
            }
        },
        Command::Search(args) => {
            commands::search::run(
                &config_db,
                &data_dir,
                &model_resolution,
                &args,
            )?;
        }
        Command::Ssearch(args) => {
            commands::search::semantic(
                &config_db,
                &data_dir,
                &model_resolution,
                &args,
            )?;
        }
        Command::Get(args) => {
            commands::search::get(&config_db, &args)?;
        }
        Command::MultiGet(args) => {
            commands::search::multi_get(&config_db, &args)?;
        }
        Command::Rebuild(args) => {
            commands::indexing::rebuild(
                &config_db,
                &data_dir,
                &args,
                &model_resolution.model_id,
            )?;
        }
        Command::Reindex => {
            commands::indexing::reindex(&data_dir)?;
        }
        Command::Sync(args) => {
            commands::indexing::sync(
                &config_db,
                &data_dir,
                &args,
                &model_resolution.model_id,
            )?;
        }
        Command::Status(args) => {
            commands::model::status(
                &config_db,
                &data_dir,
                &model_resolution,
                args.json,
            )?;
        }
        Command::Mcp => unreachable!(), // Handled above
        Command::Web(_) => unreachable!(), // Handled above
        Command::Model { action } => match action {
            cli::ModelAction::Show { json } => {
                commands::model::show(&model_resolution, json)?;
            }
            cli::ModelAction::Set { model } => {
                commands::model::set(&config_db, &model)?;
            }
            cli::ModelAction::Clear => {
                commands::model::clear(&config_db)?;
            }
        },
    }

    Ok(())
}
