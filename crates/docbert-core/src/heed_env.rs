//! Shared heed/LMDB environment plumbing for `config.db` and
//! `embeddings.db`.
//!
//! Both stores are single-file LMDB environments opened through
//! [`open_heed_env`], which layers a process-wide handle cache over
//! heed's one-env-per-path rule.
//!
//! Files written by docbert releases before 1.0 may still be in the
//! legacy redb format. The transparent redb→heed migration was removed
//! for 1.0; [`ensure_not_redb`] detects such files by their magic bytes
//! and refuses to open them, pointing the user at `docbert clean`.

use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use heed::{Env, EnvFlags, EnvOpenOptions};

use crate::error::Result;

/// First nine bytes of every redb file. Matches `redb::MAGICNUMBER`.
pub(crate) const REDB_MAGIC: &[u8] = b"redb\x1A\x0A\xA9\x0D\x0A";

/// Number of named heed databases the config env can hold. Keep in
/// sync with the named databases opened by [`crate::config_db`].
pub(crate) const CONFIG_MAX_DBS: u32 = 8;

/// Number of named heed databases the embeddings env can hold.
pub(crate) const EMBEDDINGS_MAX_DBS: u32 = 2;

/// Returns `true` if `path` exists and starts with the redb magic
/// bytes — i.e. it was written by a docbert release before 1.0 and
/// this version can no longer open it. Returns `false` for every
/// other case (missing file, empty file, heed/LMDB file, anything
/// else).
///
/// Public so `docbert clean` can spot legacy files without going
/// through the rejecting open path.
pub fn is_legacy_redb_file(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let mut file = File::open(path)?;
    let mut header = [0u8; 9];
    let read = file.read(&mut header)?;
    Ok(read == header.len() && header == REDB_MAGIC)
}

/// Refuse to open a legacy redb-formatted database file.
///
/// docbert < 1.0 stored `config.db` / `embeddings.db` in the redb
/// format and later releases migrated them transparently; 1.0 dropped
/// that migration, so the only remaining path is resetting the file
/// via `docbert clean`.
pub(crate) fn ensure_not_redb(path: &Path) -> Result<()> {
    if is_legacy_redb_file(path)? {
        return Err(crate::Error::LegacyDatabase {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

/// Open a heed environment at `path` with the standard docbert flags
/// (single-file via `NO_SUB_DIR`, generous map size, configurable
/// `max_dbs`). The path becomes the data file; heed creates a
/// `<path>-lock` sibling for the LMDB lock.
///
/// LMDB only allows one [`Env`] per path per process. Heed enforces
/// this with an internal registry that returns
/// [`heed::Error::EnvAlreadyOpened`] on the second open. docbert calls
/// `ConfigDb::open` / `EmbeddingDb::open` from many places (every web
/// handler, the MCP runtime, the CLI commands), so this helper layers
/// a process-wide cache on top of heed's API: the first call opens
/// the env and inserts it, subsequent calls clone the cached handle.
/// `Env` is internally reference-counted, so cloning is cheap.
pub(crate) fn open_heed_env(
    path: &Path,
    map_size: usize,
    max_dbs: u32,
) -> Result<Env> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    // Touch the data file so the lock-file creation step is unambiguous —
    // LMDB uses the data file's mode bits as the template.
    if !path.exists() {
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(path)?;
    }

    let key = canonical_env_key(path);
    let mut cache = env_cache().lock().unwrap();
    if let Some(env) = cache.get(&key) {
        return Ok(env.clone());
    }

    let env = unsafe {
        let mut opts = EnvOpenOptions::new();
        opts.map_size(map_size);
        opts.max_dbs(max_dbs);
        opts.flags(EnvFlags::NO_SUB_DIR);
        opts.open(path)?
    };
    cache.insert(key, env.clone());
    Ok(env)
}

fn env_cache() -> &'static Mutex<HashMap<PathBuf, Env>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Env>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Canonical-path key for [`env_cache`]. Mirrors how heed itself keys
/// its internal `OPENED_ENV` registry: prefer `canonicalize(path)`, but
/// fall back to canonicalizing the parent and joining the filename
/// when the file doesn't exist yet.
fn canonical_env_key(path: &Path) -> PathBuf {
    if let Ok(canonical) = std::fs::canonicalize(path) {
        return canonical;
    }
    if let Some((dir, file_name)) = path.parent().zip(path.file_name())
        && let Ok(dir) = std::fs::canonicalize(dir)
    {
        return dir.join(file_name);
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    /// Minimal stand-in for a redb file: the magic bytes plus junk.
    pub(crate) fn write_fake_redb_file(path: &Path) {
        let mut bytes = REDB_MAGIC.to_vec();
        bytes.extend_from_slice(b"junk payload");
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn detects_redb_magic_only() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("config.db");
        write_fake_redb_file(&p);
        assert!(is_legacy_redb_file(&p).unwrap());
    }

    #[test]
    fn returns_false_for_missing_or_empty_or_heed_files() {
        let dir = tempdir().unwrap();
        // missing
        let missing = dir.path().join("nope");
        assert!(!is_legacy_redb_file(&missing).unwrap());

        // empty file
        let empty = dir.path().join("empty");
        std::fs::write(&empty, b"").unwrap();
        assert!(!is_legacy_redb_file(&empty).unwrap());

        // a heed env at the same name
        let heed = dir.path().join("heed");
        let env = open_heed_env(&heed, 1024 * 1024, 2).unwrap();
        drop(env);
        assert!(!is_legacy_redb_file(&heed).unwrap());
    }

    #[test]
    fn ensure_not_redb_rejects_legacy_files_with_clean_instructions() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("config.db");
        write_fake_redb_file(&p);

        let err = ensure_not_redb(&p).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("docbert clean"), "unhelpful error: {msg}");
    }

    #[test]
    fn ensure_not_redb_accepts_missing_and_heed_files() {
        let dir = tempdir().unwrap();
        ensure_not_redb(&dir.path().join("missing.db")).unwrap();

        let heed = dir.path().join("data.db");
        let env = open_heed_env(&heed, 1024 * 1024, 2).unwrap();
        drop(env);
        ensure_not_redb(&heed).unwrap();
    }

    #[test]
    fn reopening_an_env_returns_the_cached_handle() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("data.db");
        for _ in 0..2 {
            let env = open_heed_env(&p, 1024 * 1024, 2).unwrap();
            let rtxn = env.read_txn().unwrap();
            drop(rtxn);
        }
    }
}
