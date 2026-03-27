use std::sync::{Arc, Mutex};

use docbert_core::{ConfigDb, DataDir, EmbeddingDb, ModelManager, SearchIndex};
use tantivy::IndexWriter;

pub struct Inner {
    pub config_db: ConfigDb,
    pub search_index: SearchIndex,
    pub embedding_db: EmbeddingDb,
    pub model: Mutex<ModelManager>,
    pub writer: Mutex<IndexWriter>,
}

pub type AppState = Arc<Inner>;

pub fn init(
    data_dir: DataDir,
    model_id: Option<String>,
) -> Result<AppState, docbert_core::Error> {
    let config_db = ConfigDb::open(&data_dir.config_db())?;
    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;

    let mut model = match model_id {
        Some(id) => ModelManager::with_model_id(id),
        None => ModelManager::new(),
    };

    // Eagerly load the ColBERT model so the first ingestion doesn't block.
    tracing::info!("loading ColBERT model: {}", model.model_id());
    match model.runtime_config() {
        Ok(cfg) => {
            tracing::info!(
                device = %cfg.device,
                batch_size = cfg.embedding_batch_size,
                document_length = cfg.document_length,
                "model loaded: {}",
                model.model_id(),
            );
            if let Some(note) = &cfg.fallback_note {
                tracing::warn!("{note}");
            }
        }
        Err(e) => tracing::warn!(
            "failed to preload model (will retry on first use): {e}"
        ),
    }

    let writer = search_index.writer(50_000_000)?;

    Ok(Arc::new(Inner {
        config_db,
        search_index,
        embedding_db,
        model: Mutex::new(model),
        writer: Mutex::new(writer),
    }))
}
