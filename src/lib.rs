//! docbert -- a local document search engine combining BM25 and ColBERT reranking.
//!
//! docbert indexes collections of markdown and text files, providing fast keyword
//! search via [Tantivy](https://github.com/quickwit-oss/tantivy) with optional
//! neural reranking via [ColBERT](https://github.com/stanford-futuredata/ColBERT)
//! (specifically the GTE-ModernColBERT model via pylate-rs).
//!
//! # Architecture
//!
//! The search pipeline has two stages:
//!
//! 1. **BM25 retrieval** -- Tantivy indexes documents with English stemming and
//!    retrieves the top 1000 candidates for a query. Optionally includes fuzzy
//!    matching (Levenshtein distance 1).
//!
//! 2. **ColBERT reranking** -- each candidate's per-token embedding matrix is
//!    compared against the query embedding using MaxSim scoring, producing a
//!    semantic relevance score that captures meaning beyond keyword overlap.
//!
//! # Storage
//!
//! All data is stored locally in three databases managed by [`DataDir`]:
//!
//! - **`config.db`** ([`ConfigDb`]) -- collections, contexts, document metadata, settings
//! - **`embeddings.db`** ([`EmbeddingDb`]) -- ColBERT per-token embedding matrices
//! - **`tantivy/`** ([`SearchIndex`]) -- BM25 full-text search index
//!
//! # Quick start
//!
//! ```no_run
//! use docbert::{ConfigDb, DataDir, SearchIndex, EmbeddingDb, ModelManager};
//! use docbert::search::{self, SearchParams};
//!
//! // Open databases
//! let data_dir = DataDir::resolve(None).unwrap();
//! let config_db = ConfigDb::open(&data_dir.config_db()).unwrap();
//! let search_index = SearchIndex::open(&data_dir.tantivy_dir().unwrap()).unwrap();
//! let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db()).unwrap();
//! let mut model = ModelManager::new();
//!
//! // Search with BM25 only (no model download required)
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
//!     println!("{}: {}:{} (score: {:.3})", r.rank, r.collection, r.path, r.score);
//! }
//! ```
//!
//! # Indexing documents
//!
//! ```no_run
//! use docbert::{ConfigDb, SearchIndex, EmbeddingDb, ModelManager, DocumentId};
//! use docbert::{walker, ingestion, embedding};
//!
//! # let tmp = tempfile::tempdir().unwrap();
//! // Discover files in a directory
//! let files = walker::discover_files(tmp.path()).unwrap();
//!
//! // Index into Tantivy
//! let index = SearchIndex::open_in_ram().unwrap();
//! let mut writer = index.writer(15_000_000).unwrap();
//! let count = ingestion::ingest_files(&index, &mut writer, "notes", &files).unwrap();
//!
//! // Optionally compute ColBERT embeddings (downloads model on first use)
//! let emb_db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
//! let mut model = ModelManager::new();
//! // embedding::embed_and_store(&mut model, &emb_db, docs).unwrap();
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
