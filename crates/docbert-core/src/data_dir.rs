use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Root directory for docbert's on-disk state.
///
/// It contains:
/// - `config.db`: collections and metadata
/// - `embeddings.db`: ColBERT embedding matrices
/// - `tantivy/`: the Tantivy search index
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// use docbert_core::DataDir;
///
/// let dir = DataDir::resolve(Some(tmp.path())).unwrap();
/// assert!(dir.root().exists());
/// assert_eq!(dir.config_db(), tmp.path().join("config.db"));
/// ```
#[derive(Debug, Clone)]
pub struct DataDir {
    root: PathBuf,
}

impl DataDir {
    /// Pick the data directory using this priority order:
    /// 1. an explicit path, such as `--data-dir`
    /// 2. the `DOCBERT_DATA_DIR` environment variable
    /// 3. the XDG data directory (`~/.local/share/docbert/`)
    ///
    /// Creates the directory, along with any missing parents, if needed.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// use docbert_core::DataDir;
    ///
    /// let dir = DataDir::resolve(Some(tmp.path())).unwrap();
    /// assert!(dir.root().exists());
    /// ```
    pub fn resolve(explicit: Option<&Path>) -> Result<Self> {
        let root = if let Some(path) = explicit {
            path.to_path_buf()
        } else if let Ok(val) = std::env::var("DOCBERT_DATA_DIR") {
            PathBuf::from(val)
        } else {
            xdg::BaseDirectories::with_prefix("docbert")
                .get_data_home()
                .ok_or_else(|| Error::Config("could not determine XDG data home directory".into()))?
        };

        std::fs::create_dir_all(&root).map_err(|_| Error::DataDir(root.clone()))?;

        Ok(Self { root })
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
    /// let dir = DataDir::resolve(Some(tmp.path())).unwrap();
    /// let tantivy = dir.tantivy_dir().unwrap();
    /// assert!(tantivy.exists());
    /// assert_eq!(tantivy, tmp.path().join("tantivy"));
    /// ```
    pub fn tantivy_dir(&self) -> Result<PathBuf> {
        let path = self.root.join("tantivy");
        std::fs::create_dir_all(&path).map_err(|_| Error::DataDir(path.clone()))?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_with_explicit_path() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::resolve(Some(tmp.path())).unwrap();

        assert_eq!(dir.root(), tmp.path());
        assert_eq!(dir.config_db(), tmp.path().join("config.db"));
        assert_eq!(dir.embeddings_db(), tmp.path().join("embeddings.db"));
    }

    #[test]
    fn tantivy_dir_is_created() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::resolve(Some(tmp.path())).unwrap();
        let tantivy = dir.tantivy_dir().unwrap();

        assert!(tantivy.exists());
        assert_eq!(tantivy, tmp.path().join("tantivy"));
    }

    #[test]
    fn resolve_creates_nonexistent_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("a").join("b").join("c");
        let dir = DataDir::resolve(Some(&nested)).unwrap();
        assert!(dir.root().exists());
        assert_eq!(dir.root(), nested);
    }

    #[test]
    fn tantivy_dir_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::resolve(Some(tmp.path())).unwrap();
        let first = dir.tantivy_dir().unwrap();
        let second = dir.tantivy_dir().unwrap();
        assert_eq!(first, second);
        assert!(first.exists());
    }
}
