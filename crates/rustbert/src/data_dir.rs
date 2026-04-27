//! Data directory resolution and cache path helpers.
//!
//! rustbert keeps its own data dir parallel to docbert's. Resolution
//! order:
//!
//! 1. `$RUSTBERT_DATA_DIR` (explicit override).
//! 2. `$XDG_DATA_HOME/rustbert/`.
//! 3. `$HOME/.local/share/rustbert/`.
//!
//! Cache layout:
//!
//! ```text
//! <data_dir>/
//! ├── config.db            # synthetic-collection metadata, sync runs
//! ├── embeddings.db
//! ├── plaid.idx
//! ├── tantivy/
//! └── crate-cache/
//!     ├── <name>-<version>.crate    # raw downloaded tarball
//!     └── <name>-<version>/         # extracted source tree
//! ```

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Resolve the data directory rustbert should use, honoring
/// `$RUSTBERT_DATA_DIR` first, then XDG conventions.
pub fn data_dir() -> Result<PathBuf> {
    data_dir_with_env(EnvLookup::Process)
}

#[derive(Debug, Clone, Copy)]
enum EnvLookup {
    Process,
    /// For tests: ignore process env entirely.
    #[cfg(test)]
    None,
}

fn data_dir_with_env(lookup: EnvLookup) -> Result<PathBuf> {
    if matches!(lookup, EnvLookup::Process) {
        if let Ok(d) = std::env::var("RUSTBERT_DATA_DIR")
            && !d.is_empty()
        {
            return Ok(PathBuf::from(d));
        }
        if let Ok(d) = std::env::var("XDG_DATA_HOME")
            && !d.is_empty()
        {
            return Ok(PathBuf::from(d).join("rustbert"));
        }
        if let Ok(home) = std::env::var("HOME")
            && !home.is_empty()
        {
            return Ok(PathBuf::from(home).join(".local/share/rustbert"));
        }
    }
    Err(Error::DataDirUnknown)
}

/// Path to the per-crate cache directory under `data_dir`.
pub fn crate_cache_root(data_dir: &Path) -> PathBuf {
    data_dir.join("crate-cache")
}

/// Path to the extracted source tree for `(name, version)`.
pub fn extracted_crate_dir(
    data_dir: &Path,
    name: &str,
    version: &semver::Version,
) -> PathBuf {
    crate_cache_root(data_dir).join(format!("{name}-{version}"))
}

/// Path to the raw `.crate` tarball for `(name, version)`.
pub fn crate_tarball_path(
    data_dir: &Path,
    name: &str,
    version: &semver::Version,
) -> PathBuf {
    crate_cache_root(data_dir).join(format!("{name}-{version}.crate"))
}

/// `tantivy/` subdirectory.
pub fn tantivy_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("tantivy")
}

/// `config.db` path.
pub fn config_db_path(data_dir: &Path) -> PathBuf {
    data_dir.join("config.db")
}

/// `embeddings.db` path.
pub fn embeddings_db_path(data_dir: &Path) -> PathBuf {
    data_dir.join("embeddings.db")
}

/// `plaid.idx` path.
pub fn plaid_index_path(data_dir: &Path) -> PathBuf {
    data_dir.join("plaid.idx")
}

/// Create the per-data-dir scaffolding (`crate-cache/`, `tantivy/`).
/// Idempotent.
pub fn ensure_layout(data_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(crate_cache_root(data_dir))?;
    std::fs::create_dir_all(tantivy_dir(data_dir))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version() -> semver::Version {
        semver::Version::new(1, 0, 219)
    }

    #[test]
    fn extracted_dir_format() {
        let p = extracted_crate_dir(Path::new("/data"), "serde", &version());
        assert_eq!(p, PathBuf::from("/data/crate-cache/serde-1.0.219"));
    }

    #[test]
    fn tarball_path_format() {
        let p = crate_tarball_path(Path::new("/data"), "serde", &version());
        assert_eq!(p, PathBuf::from("/data/crate-cache/serde-1.0.219.crate"));
    }

    #[test]
    fn subsystem_paths_use_known_filenames() {
        let d = Path::new("/data");
        assert_eq!(config_db_path(d), PathBuf::from("/data/config.db"));
        assert_eq!(embeddings_db_path(d), PathBuf::from("/data/embeddings.db"));
        assert_eq!(plaid_index_path(d), PathBuf::from("/data/plaid.idx"));
        assert_eq!(tantivy_dir(d), PathBuf::from("/data/tantivy"));
        assert_eq!(crate_cache_root(d), PathBuf::from("/data/crate-cache"));
    }

    #[test]
    fn ensure_layout_creates_subdirs() {
        let tmp = tempfile::TempDir::new().unwrap();
        ensure_layout(tmp.path()).unwrap();
        assert!(tmp.path().join("crate-cache").is_dir());
        assert!(tmp.path().join("tantivy").is_dir());
    }

    #[test]
    fn ensure_layout_is_idempotent() {
        let tmp = tempfile::TempDir::new().unwrap();
        ensure_layout(tmp.path()).unwrap();
        ensure_layout(tmp.path()).unwrap();
        assert!(tmp.path().join("crate-cache").is_dir());
    }

    #[test]
    fn process_env_unset_returns_known_error() {
        // Run resolution with the process env hidden — i.e. construct
        // an EnvLookup::None call. Confirms the error variant rather
        // than relying on the test process's environment being set.
        let err = data_dir_with_env(EnvLookup::None).unwrap_err();
        assert!(matches!(err, Error::DataDirUnknown));
    }
}
