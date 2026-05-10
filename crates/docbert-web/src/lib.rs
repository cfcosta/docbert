use docbert_core::{ConfigDb, DataDir, error};

mod ingest;
mod paths;
mod routes;
mod runtime;
mod server;
mod snapshots;
mod state;
mod ui;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebArgs {
    pub host: String,
    pub port: u16,
}

pub fn run(
    args: &WebArgs,
    data_dir: DataDir,
    model_id: String,
) -> error::Result<()> {
    let config_db = ConfigDb::open(&data_dir.config_db())?;
    let state = state::init(config_db, data_dir, model_id)?;
    server::run(&args.host, args.port, state)
}
