use std::{thread, time::Duration};

use docbert_core::{ConfigDb, DataDir, EmbeddingDb, SearchIndex, error};
use tantivy::{IndexWriter, TantivyError};

const OPEN_RETRY_DELAY: Duration = Duration::from_millis(50);

/// Open `config.db`. With the heed/LMDB backend a second concurrent
/// open from the same process or another process is supported via
/// LMDB's MVCC + reader/writer locks, so this no longer needs the
/// retry loop the redb-backed version had.
pub(crate) fn open_config_db_blocking(
    data_dir: &DataDir,
) -> error::Result<ConfigDb> {
    ConfigDb::open(&data_dir.config_db())
}

/// Open `embeddings.db`. See [`open_config_db_blocking`] for the
/// no-retry rationale.
pub(crate) fn open_embedding_db_blocking(
    data_dir: &DataDir,
) -> error::Result<EmbeddingDb> {
    EmbeddingDb::open(&data_dir.embeddings_db())
}

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

fn is_retryable_tantivy_lock(err: &error::Error) -> bool {
    matches!(err, error::Error::Tantivy(TantivyError::LockFailure(_, _)))
}

#[cfg(test)]
mod tests {
    use docbert_core::Error;
    use tempfile::tempdir;

    use super::*;

    /// Two concurrent `ConfigDb::open` calls used to deadlock the
    /// runtime under redb because redb refused a second handle until
    /// the first one was dropped. LMDB lets multiple readers and
    /// writers coexist on the same file, so opening the same path
    /// twice in one process should just work.
    #[test]
    fn open_config_db_blocking_supports_concurrent_handles() {
        let tmp = tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let first = open_config_db_blocking(&data_dir).unwrap();
        let second = open_config_db_blocking(&data_dir).unwrap();
        // Use both handles so the compiler can't optimise either out
        // before the second open lands.
        first.set_collection("notes", "/tmp/n").unwrap();
        assert_eq!(
            second.get_collection("notes").unwrap(),
            Some("/tmp/n".to_string())
        );
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
