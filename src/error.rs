use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("database error: {0}")]
    Redb(#[from] redb::Error),

    #[error("database storage error: {0}")]
    RedbStorage(#[from] redb::StorageError),

    #[error("database transaction error: {0}")]
    RedbTransaction(#[from] redb::TransactionError),

    #[error("database table error: {0}")]
    RedbTable(#[from] redb::TableError),

    #[error("database commit error: {0}")]
    RedbCommit(#[from] redb::CommitError),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("{kind} not found: {name}")]
    NotFound { kind: &'static str, name: String },

    #[error("data directory does not exist and could not be created: {0}")]
    DataDir(PathBuf),
}
