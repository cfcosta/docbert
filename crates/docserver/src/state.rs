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

pub fn init(data_dir: DataDir, model_id: Option<String>) -> Result<AppState, docbert_core::Error> {
    let config_db = ConfigDb::open(&data_dir.config_db())?;
    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;

    let model = match model_id {
        Some(id) => ModelManager::with_model_id(id),
        None => ModelManager::new(),
    };

    let writer = search_index.writer(50_000_000)?;

    Ok(Arc::new(Inner {
        config_db,
        search_index,
        embedding_db,
        model: Mutex::new(model),
        writer: Mutex::new(writer),
    }))
}
