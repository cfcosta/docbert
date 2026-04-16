//! docbert-core is the library behind docbert, a local document search engine
//! that fuses BM25 with ColBERT semantic retrieval.
//!
//! Point it at one or more folders, index them, and search them locally. Tantivy
//! handles the fast keyword pass. ColBERT adds a semantic retrieval pass, and
//! the two are combined with Reciprocal Rank Fusion.
//!
//! # How it works
//!
//! Search runs two retrievers in parallel:
//!
//! 1. **BM25 retrieval** - Tantivy indexes documents with English stemming and
//!    returns up to 100 keyword candidates. Fuzzy matching is optional.
//!
//! 2. **ColBERT semantic retrieval** - docbert scores the query embedding
//!    against every stored document embedding with MaxSim and keeps the top
//!    100. That helps when the wording is different but the meaning is close.
//!
//! The two ranked lists are fused with Reciprocal Rank Fusion. A `bm25_only`
//! flag skips the semantic leg when you want keyword-only results.
//!
//! # Storage
//!
//! docbert keeps its local state in three places managed by [`DataDir`]:
//!
//! - **`config.db`** ([`ConfigDb`]) - collections, contexts, document metadata, and settings
//! - **`embeddings.db`** ([`EmbeddingDb`]) - ColBERT token embedding matrices
//! - **`tantivy/`** ([`SearchIndex`]) - BM25 full-text index
//!
//! # Quick start
//!
//! ```no_run
//! use std::path::Path;
//! use docbert_core::{ConfigDb, DataDir, SearchIndex, ModelManager};
//! use docbert_core::search::{self, SearchParams};
//!
//! // Open the local databases (the caller decides the root path).
//! let data_dir = DataDir::new(Path::new("/home/user/.local/share/docbert"));
//! let config_db = ConfigDb::open(&data_dir.config_db()).unwrap();
//! let search_index = SearchIndex::open(&data_dir.tantivy_dir().unwrap()).unwrap();
//! let mut model = ModelManager::new();
//!
//! // BM25-only search does not need a PLAID index or a ColBERT model.
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
//! let results = search::execute_search(
//!     &params,
//!     &search_index,
//!     &config_db,
//!     &data_dir,
//!     &mut model,
//! )
//! .unwrap();
//! for r in &results {
//!     println!("{}: {}:{} (score: {:.3})", r.rank, r.collection, r.path, r.score);
//! }
//! ```
//!
//! # Indexing documents
//!
//! ```no_run
//! use docbert_core::{ConfigDb, SearchIndex, EmbeddingDb, ModelManager, DocumentId};
//! use docbert_core::{walker, ingestion, embedding};
//!
//! # let tmp = tempfile::tempdir().unwrap();
//! // Find files in a directory.
//! let files = walker::discover_files(tmp.path()).unwrap();
//!
//! // Add them to Tantivy.
//! let index = SearchIndex::open_in_ram().unwrap();
//! let mut writer = index.writer(15_000_000).unwrap();
//! let count = ingestion::ingest_files(&index, &mut writer, "notes", &files).unwrap();
//!
//! // You can also compute ColBERT embeddings. The first call downloads the model.
//! let emb_db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
//! let mut model = ModelManager::new();
//! // embedding::embed_and_store(&mut model, &emb_db, docs).unwrap();
//! ```

pub mod chunking;
pub mod config_db;
pub mod conversation;
pub mod data_dir;
pub mod doc_id;
pub mod embedding;
pub mod embedding_db;
pub mod error;
pub mod incremental;
pub mod ingestion;
pub mod merkle;
pub mod model_manager;
pub mod path_safety;
pub mod plaid;
pub mod preparation;
pub mod reranker;
pub mod results;
pub mod search;
pub mod storage_codec;
pub mod stored_json;
pub mod tantivy_index;
pub mod text_util;
pub mod walker;

pub use config_db::{ConfigDb, PersistedLlmSettings};
pub use conversation::{ChatMessage, Conversation};
pub use data_dir::DataDir;
pub use doc_id::DocumentId;
pub use embedding_db::EmbeddingDb;
pub use error::{Error, Result};
pub use model_manager::ModelManager;
pub use tantivy_index::SearchIndex;
