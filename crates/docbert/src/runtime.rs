use std::{thread, time::Duration};

use docbert_core::{ConfigDb, DataDir, EmbeddingDb, SearchIndex, error};
use tantivy::{IndexWriter, TantivyError};

const OPEN_RETRY_DELAY: Duration = Duration::from_millis(50);

#[allow(dead_code)]
pub(crate) fn open_config_db_blocking(
    data_dir: &DataDir,
) -> error::Result<ConfigDb> {
    loop {
        match ConfigDb::open(&data_dir.config_db()) {
            Ok(db) => return Ok(db),
            Err(err) if is_retryable_redb_open_contention(&err) => {
                thread::sleep(OPEN_RETRY_DELAY);
            }
            Err(err) => return Err(err),
        }
    }
}

#[allow(dead_code)]
pub(crate) fn open_embedding_db_blocking(
    data_dir: &DataDir,
) -> error::Result<EmbeddingDb> {
    loop {
        match EmbeddingDb::open(&data_dir.embeddings_db()) {
            Ok(db) => return Ok(db),
            Err(err) if is_retryable_redb_open_contention(&err) => {
                thread::sleep(OPEN_RETRY_DELAY);
            }
            Err(err) => return Err(err),
        }
    }
}

#[allow(dead_code)]
pub(crate) fn open_index_writer_blocking(
    search_index: &SearchIndex,
    memory_budget: usize,
) -> error::Result<IndexWriter> {
    loop {
        match search_index.writer(memory_budget) {
            Ok(writer) => return Ok(writer),
            Err(err) if is_retryable_tantivy_lock(&err) => {
                thread::sleep(OPEN_RETRY_DELAY);
            }
            Err(err) => return Err(err),
        }
    }
}

fn is_retryable_redb_open_contention(err: &error::Error) -> bool {
    matches!(
        err,
        error::Error::RedbDatabase(inner)
            if inner.to_string().contains("already open")
                || inner.to_string().contains("Cannot acquire lock")
    )
}

fn is_retryable_tantivy_lock(err: &error::Error) -> bool {
    matches!(err, error::Error::Tantivy(TantivyError::LockFailure(_, _)))
}

#[cfg(test)]
mod tests {
    use docbert_core::Error;
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn runtime_retries_redb_open_contention_classification() {
        let tmp = tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let _first = ConfigDb::open(&data_dir.config_db()).unwrap();

        let err = match ConfigDb::open(&data_dir.config_db()) {
            Ok(_) => panic!("expected config db open contention error"),
            Err(err) => err,
        };

        assert!(is_retryable_redb_open_contention(&err));
        assert!(matches!(err, Error::RedbDatabase(_)));
        assert!(err.to_string().contains("already open"));
    }

    #[test]
    fn runtime_retries_tantivy_lock_classification() {
        let tmp = tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let search_index =
            SearchIndex::open(&data_dir.tantivy_dir().unwrap()).unwrap();
        let _first = search_index.writer(15_000_000).unwrap();

        let err = match search_index.writer(15_000_000) {
            Ok(_) => panic!("expected tantivy writer lock error"),
            Err(err) => err,
        };

        assert!(is_retryable_tantivy_lock(&err));
        assert!(matches!(
            err,
            Error::Tantivy(TantivyError::LockFailure(_, _))
        ));
    }
}
