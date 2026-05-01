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
/// Most variants wrap errors from lower layers such as the LMDB-backed
/// key-value store, Tantivy, or the model stack.
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
    Heed(#[from] heed::Error),

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
    Colbert(#[from] docbert_pylate::ColbertError),

    #[error("tensor error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("plaid index error: {0}")]
    Plaid(#[from] docbert_plaid::PlaidError),

    #[error("serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("pdf error: {0}")]
    Pdf(#[from] pdf_oxide::error::Error),

    #[error("archive serialization error: {0}")]
    Rkyv(#[from] rkyv::rancor::Error),

    #[error(
        "PLAID semantic index is not built yet; run `docbert sync` or `docbert rebuild` to build it"
    )]
    PlaidIndexMissing,
}
