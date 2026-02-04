use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct DataDir {
    root: PathBuf,
}

impl DataDir {
    /// Resolve the data directory from, in order of priority:
    /// 1. An explicit path (from --data-dir)
    /// 2. The DOCBERT_DATA_DIR environment variable
    /// 3. The XDG data directory (~/.local/share/docbert/)
    pub fn resolve(explicit: Option<&Path>) -> Result<Self> {
        let root = if let Some(path) = explicit {
            path.to_path_buf()
        } else if let Ok(val) = std::env::var("DOCBERT_DATA_DIR") {
            PathBuf::from(val)
        } else {
            xdg::BaseDirectories::with_prefix("docbert")
                .get_data_home()
                .ok_or_else(|| {
                    Error::Config(
                        "could not determine XDG data home directory".into(),
                    )
                })?
        };

        std::fs::create_dir_all(&root)
            .map_err(|_| Error::DataDir(root.clone()))?;

        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config_db(&self) -> PathBuf {
        self.root.join("config.redb")
    }

    pub fn embeddings_db(&self) -> PathBuf {
        self.root.join("embeddings.redb")
    }

    pub fn tantivy_dir(&self) -> Result<PathBuf> {
        let path = self.root.join("tantivy");
        std::fs::create_dir_all(&path)
            .map_err(|_| Error::DataDir(path.clone()))?;
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
        assert_eq!(dir.config_db(), tmp.path().join("config.redb"));
        assert_eq!(dir.embeddings_db(), tmp.path().join("embeddings.redb"));
    }

    #[test]
    fn tantivy_dir_is_created() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::resolve(Some(tmp.path())).unwrap();
        let tantivy = dir.tantivy_dir().unwrap();

        assert!(tantivy.exists());
        assert_eq!(tantivy, tmp.path().join("tantivy"));
    }
}
