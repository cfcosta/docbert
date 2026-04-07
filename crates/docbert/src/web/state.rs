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

use crate::runtime_resources;

#[allow(dead_code)]
pub(crate) struct Inner {
    pub(crate) data_dir: DataDir,
    pub(crate) config_db: ConfigDb,
    pub(crate) search_index: SearchIndex,
    pub(crate) embedding_db: EmbeddingDb,
    pub(crate) model: Mutex<ModelManager>,
    pub(crate) writer: Mutex<IndexWriter>,
}

pub(crate) type AppState = Arc<Inner>;

impl Inner {
    #[cfg(test)]
    pub(crate) fn open_config_db(&self) -> error::Result<&ConfigDb> {
        Ok(&self.config_db)
    }

    #[cfg(not(test))]
    pub(crate) fn open_config_db(&self) -> error::Result<ConfigDb> {
        runtime_resources::open_config_db_blocking(&self.data_dir)
    }

    #[cfg(test)]
    pub(crate) fn open_embedding_db(&self) -> error::Result<&EmbeddingDb> {
        Ok(&self.embedding_db)
    }

    #[cfg(not(test))]
    pub(crate) fn open_embedding_db(&self) -> error::Result<EmbeddingDb> {
        runtime_resources::open_embedding_db_blocking(&self.data_dir)
    }

    #[cfg(test)]
    pub(crate) fn open_config_db_blocking(&self) -> error::Result<&ConfigDb> {
        Ok(&self.config_db)
    }

    #[cfg(not(test))]
    pub(crate) fn open_config_db_blocking(&self) -> error::Result<ConfigDb> {
        runtime_resources::open_config_db_blocking(&self.data_dir)
    }

    #[cfg(test)]
    pub(crate) fn open_embedding_db_blocking(
        &self,
    ) -> error::Result<&EmbeddingDb> {
        Ok(&self.embedding_db)
    }

    #[cfg(not(test))]
    pub(crate) fn open_embedding_db_blocking(
        &self,
    ) -> error::Result<EmbeddingDb> {
        runtime_resources::open_embedding_db_blocking(&self.data_dir)
    }

    pub(crate) fn open_index_writer_blocking(
        &self,
        memory_budget: usize,
    ) -> error::Result<IndexWriter> {
        runtime_resources::open_index_writer_blocking(
            &self.search_index,
            memory_budget,
        )
    }
}

pub(crate) fn init(
    _config_db: ConfigDb,
    data_dir: DataDir,
    model_id: String,
) -> error::Result<AppState> {
    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let placeholders_dir = data_dir.root().join(".runtime-placeholders");
    std::fs::create_dir_all(&placeholders_dir)?;
    let placeholder_data_dir = DataDir::new(placeholders_dir);
    let config_db = ConfigDb::open(&placeholder_data_dir.config_db())?;
    let embedding_db = EmbeddingDb::open(&placeholder_data_dir.embeddings_db())?;
    let model = ModelManager::with_model_id(model_id);
    let placeholder_index = SearchIndex::open_in_ram()?;
    let writer = placeholder_index.writer(50_000_000)?;

    Ok(Arc::new(Inner {
        data_dir,
        config_db,
        search_index,
        embedding_db,
        model: Mutex::new(model),
        writer: Mutex::new(writer),
    }))
}
