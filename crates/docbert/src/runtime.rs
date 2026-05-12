use docbert_core::{ConfigDb, DataDir, error};

/// Open `config.db`. With the heed/LMDB backend a second concurrent
/// open from the same process or another process is supported via
/// LMDB's MVCC + reader/writer locks, so this no longer needs the
/// retry loop the redb-backed version had.
pub(crate) fn open_config_db_blocking(
    data_dir: &DataDir,
) -> error::Result<ConfigDb> {
    ConfigDb::open(&data_dir.config_db())
}

#[cfg(test)]
mod tests {
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
}
