use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Root directory for docbert's on-disk state.
///
/// It contains:
/// - `config.db`: collections and metadata
/// - `embeddings.db`: ColBERT embedding matrices
/// - `tantivy/`: the Tantivy search index
///
/// `DataDir` is a thin wrapper around a [`PathBuf`]. It does not resolve
/// default locations or create directories on its own — use
/// [`DataDir::new`] with an already-resolved path.
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// use docbert_core::DataDir;
///
/// let dir = DataDir::new(tmp.path());
/// assert_eq!(dir.config_db(), tmp.path().join("config.db"));
/// ```
#[derive(Debug, Clone)]
pub struct DataDir {
    root: PathBuf,
}

impl DataDir {
    /// Wrap an existing path as a data directory.
    ///
    /// This does **not** create the directory or resolve defaults.  The
    /// caller is responsible for ensuring the path exists.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// use docbert_core::DataDir;
    ///
    /// let dir = DataDir::new(tmp.path());
    /// assert_eq!(dir.root(), tmp.path());
    /// ```
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Returns the root path of the data directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path to the config database file (`config.db`).
    pub fn config_db(&self) -> PathBuf {
        self.root.join("config.db")
    }

    /// Path to the embeddings database file (`embeddings.db`).
    pub fn embeddings_db(&self) -> PathBuf {
        self.root.join("embeddings.db")
    }

    /// Return the path to the Tantivy index directory (`tantivy/`).
    ///
    /// Creates the directory first if it does not exist yet.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// use docbert_core::DataDir;
    ///
    /// let dir = DataDir::new(tmp.path());
    /// let tantivy = dir.tantivy_dir().unwrap();
    /// assert!(tantivy.exists());
    /// assert_eq!(tantivy, tmp.path().join("tantivy"));
    /// ```
    pub fn tantivy_dir(&self) -> Result<PathBuf> {
        let path = self.root.join("tantivy");
        std::fs::create_dir_all(&path)
            .map_err(|_| Error::DataDir(path.clone()))?;
        Ok(path)
    }

    /// Path to the optional PLAID index file (`plaid.idx`).
    ///
    /// The file may not exist yet — the index is built on demand and
    /// callers should treat a missing file as "no PLAID index" rather
    /// than an error.
    pub fn plaid_index(&self) -> PathBuf {
        self.root.join("plaid.idx")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_wraps_path() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::new(tmp.path());

        assert_eq!(dir.root(), tmp.path());
        assert_eq!(dir.config_db(), tmp.path().join("config.db"));
        assert_eq!(dir.embeddings_db(), tmp.path().join("embeddings.db"));
    }

    #[test]
    fn tantivy_dir_is_created() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::new(tmp.path());
        let tantivy = dir.tantivy_dir().unwrap();

        assert!(tantivy.exists());
        assert_eq!(tantivy, tmp.path().join("tantivy"));
    }

    #[test]
    fn tantivy_dir_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::new(tmp.path());
        let first = dir.tantivy_dir().unwrap();
        let second = dir.tantivy_dir().unwrap();
        assert_eq!(first, second);
        assert!(first.exists());
    }
}
