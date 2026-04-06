use std::sync::{Arc, Mutex};

use docbert_core::{ConfigDb, DataDir, EmbeddingDb, ModelManager, SearchIndex, error};
use tantivy::IndexWriter;

#[allow(dead_code)]
pub(crate) struct Inner {
    pub(crate) config_db: ConfigDb,
    pub(crate) search_index: SearchIndex,
    pub(crate) embedding_db: EmbeddingDb,
    pub(crate) model: Mutex<ModelManager>,
    pub(crate) writer: Mutex<IndexWriter>,
}

pub(crate) type AppState = Arc<Inner>;

pub(crate) fn init(
    config_db: ConfigDb,
    data_dir: DataDir,
    model_id: String,
) -> error::Result<AppState> {
    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
    let model = ModelManager::with_model_id(model_id);
    let writer = search_index.writer(50_000_000)?;

    Ok(Arc::new(Inner {
        config_db,
        search_index,
        embedding_db,
        model: Mutex::new(model),
        writer: Mutex::new(writer),
    }))
}
