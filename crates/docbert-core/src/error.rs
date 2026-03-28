use std::path::PathBuf;

/// Convenience alias for `std::result::Result<T, docbert_core::Error>`.
///
/// # Examples
///
/// ```
/// fn do_work() -> docbert_core::Result<()> {
///     // ... operations that may fail ...
///     Ok(())
/// }
/// ```
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level error type for docbert.
///
/// Most variants wrap errors from lower layers such as redb, Tantivy, or
/// the model stack.
///
/// # Examples
///
/// ```
/// use docbert_core::Error;
///
/// let err = Error::Config("missing collection".to_string());
/// assert!(err.to_string().contains("missing collection"));
///
/// let err = Error::NotFound { kind: "document", name: "#abc123".to_string() };
/// assert!(err.to_string().contains("document not found"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("database error: {0}")]
    Redb(#[from] redb::Error),

    #[error("database error: {0}")]
    RedbDatabase(#[from] redb::DatabaseError),

    #[error("database storage error: {0}")]
    RedbStorage(#[from] redb::StorageError),

    #[error("database transaction error: {0}")]
    RedbTransaction(#[from] redb::TransactionError),

    #[error("database table error: {0}")]
    RedbTable(#[from] redb::TableError),

    #[error("database commit error: {0}")]
    RedbCommit(#[from] redb::CommitError),

    #[error("search index error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),

    #[error("query parse error: {0}")]
    QueryParse(#[from] tantivy::query::QueryParserError),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("{kind} not found: {name}")]
    NotFound { kind: &'static str, name: String },

    #[error("data directory does not exist and could not be created: {0}")]
    DataDir(PathBuf),

    #[error("model error: {0}")]
    Colbert(#[from] pylate_rs::ColbertError),

    #[error("tensor error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("archive serialization error: {0}")]
    Rkyv(#[from] rkyv::rancor::Error),
}
