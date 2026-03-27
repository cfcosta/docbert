use std::path::PathBuf;

use docbert_core::DataDir;
use tracing_subscriber::EnvFilter;

mod content;
mod error;
mod routes;
mod state;
mod ui;

fn main() {
    let log_filter =
        std::env::var("DOCSERVER_LOG").unwrap_or_else(|_| "warn".into());
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(log_filter))
        .with_writer(std::io::stderr)
        .without_time()
        .init();

    let data_path = std::env::var("DOCSERVER_DATA_DIR").unwrap_or_else(|_| {
        eprintln!("error: DOCSERVER_DATA_DIR is required");
        std::process::exit(1);
    });
    let host =
        std::env::var("DOCSERVER_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port =
        std::env::var("DOCSERVER_PORT").unwrap_or_else(|_| "3030".into());
    let model_id = std::env::var("DOCSERVER_MODEL").ok();

    let root = PathBuf::from(&data_path);
    std::fs::create_dir_all(&root).unwrap_or_else(|e| {
        eprintln!("error: failed to create data directory {data_path}: {e}");
        std::process::exit(1);
    });

    let data_dir = DataDir::new(root);
    let app_state = state::init(data_dir, model_id).unwrap_or_else(|e| {
        eprintln!("error: failed to initialize state: {e}");
        std::process::exit(1);
    });

    let app = routes::router().with_state(app_state);
    let addr = format!("{host}:{port}");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| {
            eprintln!("error: failed to start tokio runtime: {e}");
            std::process::exit(1);
        });

    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .unwrap_or_else(|e| {
                eprintln!("error: failed to bind to {addr}: {e}");
                std::process::exit(1);
            });
        tracing::info!("docserver listening on {addr}");
        axum::serve(listener, app).await.unwrap_or_else(|e| {
            eprintln!("error: server error: {e}");
            std::process::exit(1);
        });
    });
}
