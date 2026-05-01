use std::sync::{Arc, Mutex};

use docbert_core::{
    ConfigDb,
    DataDir,
    EmbeddingDb,
    ModelManager,
    SearchIndex,
    error,
};
use tantivy::IndexWriter;

use crate::runtime;

pub(crate) struct Inner {
    pub(crate) data_dir: DataDir,
    pub(crate) search_index: SearchIndex,
    pub(crate) model: Mutex<ModelManager>,
    pub(crate) model_id: String,
}

pub(crate) type AppState = Arc<Inner>;

impl Inner {
    pub(crate) fn open_config_db(&self) -> error::Result<ConfigDb> {
        runtime::open_config_db_blocking(&self.data_dir)
    }

    pub(crate) fn open_config_db_blocking(&self) -> error::Result<ConfigDb> {
        runtime::open_config_db_blocking(&self.data_dir)
    }

    pub(crate) fn open_embedding_db_blocking(
        &self,
    ) -> error::Result<EmbeddingDb> {
        runtime::open_embedding_db_blocking(&self.data_dir)
    }

    pub(crate) fn open_index_writer_blocking(
        &self,
        memory_budget: usize,
    ) -> error::Result<IndexWriter> {
        runtime::open_index_writer_blocking(&self.search_index, memory_budget)
    }
}

pub(crate) fn init(
    _config_db: ConfigDb,
    data_dir: DataDir,
    model_id: String,
) -> error::Result<AppState> {
    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let model = ModelManager::with_model_id(model_id.clone());

    Ok(Arc::new(Inner {
        data_dir,
        search_index,
        model: Mutex::new(model),
        model_id,
    }))
}
