//! docbert - a local document search engine combining BM25 and ColBERT reranking.
//!
//! docbert indexes collections of markdown and text files, providing fast keyword
//! search via [Tantivy](https://github.com/quickwit-oss/tantivy) with optional
//! neural reranking via [ColBERT](https://github.com/stanford-futuredata/ColBERT).
//!
//! # Quick start
//!
//! ```no_run
//! use docbert::{ConfigDb, DataDir, SearchIndex, EmbeddingDb, ModelManager};
//! use docbert::search::{self, SearchParams};
//!
//! let data_dir = DataDir::resolve(None).unwrap();
//! let config_db = ConfigDb::open(&data_dir.config_db()).unwrap();
//! let search_index = SearchIndex::open(&data_dir.tantivy_dir().unwrap()).unwrap();
//! let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db()).unwrap();
//! let mut model = ModelManager::new();
//!
//! let params = SearchParams {
//!     query: "rust programming".to_string(),
//!     count: 10,
//!     collection: None,
//!     min_score: 0.0,
//!     bm25_only: true,
//!     no_fuzzy: false,
//!     all: false,
//! };
//!
//! let results = search::execute_search(&params, &search_index, &embedding_db, &mut model)
//!     .unwrap();
//! for r in &results {
//!     println!("{}:{} (score: {:.3})", r.collection, r.path, r.score);
//! }
//! ```

pub mod chunking;
pub mod config_db;
pub mod data_dir;
pub mod doc_id;
pub mod embedding;
pub mod embedding_db;
pub mod error;
pub mod incremental;
pub mod ingestion;
pub mod mcp;
pub mod model_manager;
pub mod reranker;
pub mod search;
pub mod tantivy_index;
pub mod text_util;
pub mod walker;

pub use config_db::ConfigDb;
pub use data_dir::DataDir;
pub use doc_id::DocumentId;
pub use embedding_db::EmbeddingDb;
pub use error::{Error, Result};
pub use model_manager::ModelManager;
pub use tantivy_index::SearchIndex;
