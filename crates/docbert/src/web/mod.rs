use docbert_core::{ConfigDb, DataDir, error};

use crate::cli::WebArgs;

mod routes;
mod server;
mod state;
mod ui;

pub(crate) fn run(
    args: &WebArgs,
    data_dir: DataDir,
    config_db: ConfigDb,
    model_id: String,
) -> error::Result<()> {
    let state = state::init(config_db, data_dir, model_id)?;
    server::run(&args.host, args.port, state)
}
