use axum::Router;
use docbert_core::error;

use super::state::AppState;

fn router() -> Router<AppState> {
    Router::new()
}

pub(crate) fn run(host: &str, port: u16, state: AppState) -> error::Result<()> {
    let addr = format!("{host}:{port}");
    let app = router().with_state(state);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        tracing::info!("docbert web listening on {addr}");
        axum::serve(listener, app).await?;
        Ok(())
    })
}
