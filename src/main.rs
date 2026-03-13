use clap::Parser;
use docbert::{ConfigDb, DataDir, error, mcp, model_manager::resolve_model};
use tracing_subscriber::EnvFilter;

mod cli;
mod command_handlers;
mod indexing_workflow;

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

    // Handle commands that don't need data_dir or config_db early.
    if let Command::Completions(args) = &cli.command {
        args.generate();
        return Ok(());
    }
    if let Command::Doctor(args) = &cli.command {
        command_handlers::cmd_doctor(args.json)?;
        return Ok(());
    }

    let data_dir = DataDir::resolve(cli.data_dir.as_deref())?;
    let config_db = ConfigDb::open(&data_dir.config_db())?;
    let model_resolution = resolve_model(&config_db, cli.model.as_deref())?;

    match cli.command {
        Command::Completions(_) => unreachable!(), // Handled above
        Command::Doctor(_) => unreachable!(),      // Handled above
        Command::Collection { action } => match action {
            CollectionAction::Add { path, name } => {
                command_handlers::collection_add(&config_db, &path, &name)?;
            }
            CollectionAction::Remove { name } => {
                command_handlers::collection_remove(
                    &config_db, &data_dir, &name,
                )?;
            }
            CollectionAction::List { json } => {
                command_handlers::collection_list(&config_db, json)?;
            }
        },
        Command::Context { action } => match action {
            ContextAction::Add { uri, description } => {
                command_handlers::context_add(&config_db, &uri, &description)?;
            }
            ContextAction::Remove { uri } => {
                command_handlers::context_remove(&config_db, &uri)?;
            }
            ContextAction::List { json } => {
                command_handlers::context_list(&config_db, json)?;
            }
        },
        Command::Search(args) => {
            command_handlers::run_search(
                &config_db,
                &data_dir,
                &model_resolution,
                &args,
            )?;
        }
        Command::Ssearch(args) => {
            command_handlers::run_semantic_search(
                &config_db,
                &data_dir,
                &model_resolution,
                &args,
            )?;
        }
        Command::Get(args) => {
            command_handlers::cmd_get(&config_db, &args)?;
        }
        Command::MultiGet(args) => {
            command_handlers::cmd_multi_get(&config_db, &args)?;
        }
        Command::Rebuild(args) => {
            command_handlers::cmd_rebuild(
                &config_db,
                &data_dir,
                &args,
                &model_resolution.model_id,
            )?;
        }
        Command::Sync(args) => {
            command_handlers::cmd_sync(
                &config_db,
                &data_dir,
                &args,
                &model_resolution.model_id,
            )?;
        }
        Command::Status(args) => {
            command_handlers::cmd_status(
                &config_db,
                &data_dir,
                &model_resolution,
                args.json,
            )?;
        }
        Command::Mcp => {
            mcp::run_mcp(data_dir, config_db, model_resolution.model_id)?;
        }
        Command::Model { action } => match action {
            cli::ModelAction::Show { json } => {
                command_handlers::cmd_model_show(&model_resolution, json);
            }
            cli::ModelAction::Set { model } => {
                command_handlers::cmd_model_set(&config_db, &model)?;
            }
            cli::ModelAction::Clear => {
                command_handlers::cmd_model_clear(&config_db)?;
            }
        },
    }

    Ok(())
}
