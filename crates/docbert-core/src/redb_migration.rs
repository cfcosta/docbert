//! On-disk migration from the legacy redb-backed `config.db` /
//! `embeddings.db` files to the heed/LMDB-backed format.
//!
//! redb is single-process — opening the same file from two `docbert mcp`
//! instances racing the same data dir is unsupported. Switching to LMDB
//! (via heed) gives us proper cross-process locking and MVCC so multiple
//! mcp / web / CLI processes can share one data dir.
//!
//! The migration runs **transparently on first open**: every call to
//! [`ConfigDb::open`] / [`EmbeddingDb::open`] checks the file header,
//! and if it still smells like redb, copies the data into a fresh heed
//! env, backs the old file up to `<path>.redb-bak`, and atomically
//! swaps. On a fresh data dir, on a re-run after a successful
//! migration, or against a heed file the call is a no-op.
//!
//! [`ConfigDb::open`]: crate::ConfigDb::open
//! [`EmbeddingDb::open`]: crate::EmbeddingDb::open

use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::Instant,
};

use heed::{
    Database,
    Env,
    EnvFlags,
    EnvOpenOptions,
    byteorder::BigEndian,
    types::{Bytes, Str, U64},
};

use crate::error::Result;

/// First nine bytes of every redb file. Matches `redb::MAGICNUMBER`.
const REDB_MAGIC: &[u8] = b"redb\x1A\x0A\xA9\x0D\x0A";

/// Map size for migration-time heed environments. Generous because the
/// migration writes everything in a single transaction; the runtime
/// `ConfigDb` / `EmbeddingDb` open paths use their own sizes.
const MIGRATION_CONFIG_MAP_SIZE: usize = 1024 * 1024 * 1024; // 1 GiB

/// Embeddings can be huge. 64 GiB sparse-mapped is fine on typical
/// filesystems (the file only consumes the actual byte count on disk).
const MIGRATION_EMBEDDING_MAP_SIZE: usize = 64 * 1024 * 1024 * 1024; // 64 GiB

/// Number of named heed databases the config env can hold.
/// Keep in sync with [`crate::config_db`].
pub(crate) const CONFIG_MAX_DBS: u32 = 8;

/// Number of named heed databases the embeddings env can hold.
pub(crate) const EMBEDDINGS_MAX_DBS: u32 = 2;

/// Schema descriptor for one redb table that needs migrating.
enum RedbKey {
    Str,
    U64,
}

struct RedbTable {
    name: &'static str,
    key: RedbKey,
}

const CONFIG_TABLES: &[RedbTable] = &[
    RedbTable {
        name: "collections",
        key: RedbKey::Str,
    },
    RedbTable {
        name: "contexts",
        key: RedbKey::Str,
    },
    RedbTable {
        name: "document_metadata",
        key: RedbKey::U64,
    },
    RedbTable {
        name: "conversations",
        key: RedbKey::Str,
    },
    RedbTable {
        name: "collection_merkle_snapshots",
        key: RedbKey::Str,
    },
    RedbTable {
        name: "settings",
        key: RedbKey::Str,
    },
    RedbTable {
        name: "chunk_offsets",
        key: RedbKey::U64,
    },
];

const EMBEDDING_TABLES: &[RedbTable] = &[RedbTable {
    name: "embeddings",
    key: RedbKey::U64,
}];

/// Returns `true` if `path` exists and starts with the redb magic
/// bytes. Returns `false` for every other case (missing file, empty
/// file, heed/LMDB file, anything else) — callers treat that as
/// "no migration needed".
pub fn is_redb_file(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let mut file = File::open(path)?;
    let mut header = [0u8; 9];
    let read = file.read(&mut header)?;
    Ok(read == header.len() && header == REDB_MAGIC)
}

/// Migrate a redb-backed `config.db` at `path` to a heed env at the
/// same path. No-op if the file isn't a redb file.
pub fn ensure_config_db_migrated(path: &Path) -> Result<()> {
    ensure_migrated(
        path,
        CONFIG_TABLES,
        MIGRATION_CONFIG_MAP_SIZE,
        CONFIG_MAX_DBS,
    )
}

/// Migrate a redb-backed `embeddings.db` at `path` to a heed env at the
/// same path. No-op if the file isn't a redb file.
pub fn ensure_embedding_db_migrated(path: &Path) -> Result<()> {
    ensure_migrated(
        path,
        EMBEDDING_TABLES,
        MIGRATION_EMBEDDING_MAP_SIZE,
        EMBEDDINGS_MAX_DBS,
    )
}

fn ensure_migrated(
    path: &Path,
    tables: &[RedbTable],
    map_size: usize,
    max_dbs: u32,
) -> Result<()> {
    if !is_redb_file(path)? {
        return Ok(());
    }

    let started = Instant::now();
    tracing::info!(
        path = %path.display(),
        "redb→heed migration: detected legacy redb file, starting migration",
    );

    let backup_path = backup_path_for(path);
    let tmp_path = tmp_path_for(path);

    // Sanity: leftover artifacts from a previous failed run would
    // confuse rename steps. Surface the situation instead of silently
    // overwriting either of them.
    if backup_path.exists() {
        return Err(crate::Error::Config(format!(
            "redb→heed migration: a backup already exists at {}; \
             move or remove it before retrying",
            backup_path.display()
        )));
    }
    if tmp_path.exists() {
        // Stale heed-tmp from a previous interrupted run. Safe to
        // remove since the swap step hasn't happened yet (otherwise
        // `path` would already be heed-formatted).
        let _ = std::fs::remove_file(&tmp_path);
        let _ = std::fs::remove_file(tmp_path.with_extension("heed-tmp-lock"));
    }

    let counts = copy_redb_to_heed(path, &tmp_path, tables, map_size, max_dbs)?;

    // Atomic swap: keep `path` valid at all times by going through a
    // sibling rename rather than truncate-in-place. The backup file is
    // intentionally kept on disk for the user to recover from if the
    // migrated env later turns out to be broken.
    std::fs::rename(path, &backup_path)?;
    if let Err(err) = std::fs::rename(&tmp_path, path) {
        // Roll back the original rename so the next run can retry.
        let _ = std::fs::rename(&backup_path, path);
        return Err(err.into());
    }
    // The temp lock file becomes orphaned after we rename the data
    // file; heed will recreate `<path>-lock` on next open.
    let _ = std::fs::remove_file(tmp_path.with_extension("heed-tmp-lock"));

    let elapsed = started.elapsed();
    tracing::info!(
        path = %path.display(),
        backup = %backup_path.display(),
        tables = counts.len(),
        total_entries = counts.iter().map(|(_, n)| n).sum::<u64>(),
        elapsed_ms = elapsed.as_millis() as u64,
        "redb→heed migration: completed",
    );
    for (name, n) in &counts {
        tracing::debug!(
            table = name,
            entries = *n,
            "redb→heed migration: copied table",
        );
    }

    Ok(())
}

/// Copy every entry of every named table from a redb file at `src`
/// into a fresh heed env at `dst`. Returns one (table_name,
/// entry_count) per source table so the caller can log a summary.
fn copy_redb_to_heed(
    src: &Path,
    dst: &Path,
    tables: &[RedbTable],
    map_size: usize,
    max_dbs: u32,
) -> Result<Vec<(&'static str, u64)>> {
    use redb::{ReadableDatabase, ReadableTable as _};

    let src_db = redb::Database::open(src).map_err(redb_error_to_config)?;
    let read_txn = src_db.begin_read().map_err(redb_error_to_config)?;

    // Don't leave a half-written destination file behind on failure.
    if dst.exists() {
        let _ = std::fs::remove_file(dst);
    }
    let env = open_heed_env(dst, map_size, max_dbs)?;
    let mut wtxn = env.write_txn()?;

    let mut counts = Vec::with_capacity(tables.len());
    for table in tables {
        let n = match table.key {
            RedbKey::Str => {
                let def: redb::TableDefinition<&str, &[u8]> =
                    redb::TableDefinition::new(table.name);
                let src_table = match read_txn.open_table(def) {
                    Ok(t) => t,
                    Err(redb::TableError::TableDoesNotExist(_)) => continue,
                    Err(e) => return Err(redb_error_to_config(e)),
                };
                let dst_db: Database<Str, Bytes> =
                    env.create_database(&mut wtxn, Some(table.name))?;
                let mut count = 0u64;
                for entry in src_table.iter().map_err(redb_error_to_config)? {
                    let (k, v) = entry.map_err(redb_error_to_config)?;
                    dst_db.put(&mut wtxn, k.value(), v.value())?;
                    count += 1;
                }
                count
            }
            RedbKey::U64 => {
                let def: redb::TableDefinition<u64, &[u8]> =
                    redb::TableDefinition::new(table.name);
                let src_table = match read_txn.open_table(def) {
                    Ok(t) => t,
                    Err(redb::TableError::TableDoesNotExist(_)) => continue,
                    Err(e) => return Err(redb_error_to_config(e)),
                };
                let dst_db: Database<U64<BigEndian>, Bytes> =
                    env.create_database(&mut wtxn, Some(table.name))?;
                let mut count = 0u64;
                for entry in src_table.iter().map_err(redb_error_to_config)? {
                    let (k, v) = entry.map_err(redb_error_to_config)?;
                    dst_db.put(&mut wtxn, &k.value(), v.value())?;
                    count += 1;
                }
                count
            }
        };
        counts.push((table.name, n));
    }

    wtxn.commit()?;
    drop(env);
    drop(read_txn);
    drop(src_db);

    Ok(counts)
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

fn backup_path_for(path: &Path) -> PathBuf {
    append_extension(path, "redb-bak")
}

fn tmp_path_for(path: &Path) -> PathBuf {
    append_extension(path, "heed-tmp")
}

fn append_extension(path: &Path, extra: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".");
    s.push(extra);
    PathBuf::from(s)
}

fn redb_error_to_config<E: std::error::Error>(err: E) -> crate::Error {
    crate::Error::Config(format!("redb→heed migration: {err}"))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use tempfile::tempdir;

    use super::*;

    fn write_redb_config_fixture(
        path: &Path,
        collections: &[(String, Vec<u8>)],
        document_metadata: &[(u64, Vec<u8>)],
    ) {
        let db = redb::Database::create(path).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let collections_def: redb::TableDefinition<&str, &[u8]> =
                redb::TableDefinition::new("collections");
            let mut t = txn.open_table(collections_def).unwrap();
            for (name, payload) in collections {
                t.insert(name.as_str(), payload.as_slice()).unwrap();
            }
        }
        {
            let metadata_def: redb::TableDefinition<u64, &[u8]> =
                redb::TableDefinition::new("document_metadata");
            let mut t = txn.open_table(metadata_def).unwrap();
            for (id, payload) in document_metadata {
                t.insert(id, payload.as_slice()).unwrap();
            }
        }
        // Touch the other str-keyed tables so they exist and get
        // migrated cleanly.
        for name in [
            "contexts",
            "conversations",
            "collection_merkle_snapshots",
            "settings",
        ] {
            let def: redb::TableDefinition<&str, &[u8]> =
                redb::TableDefinition::new(name);
            txn.open_table(def).unwrap();
        }
        // chunk_offsets is u64-keyed, like document_metadata — keep
        // the schema honest in the fixture so the migration doesn't
        // refuse to open it for a TableTypeMismatch reason.
        let chunk_offsets_def: redb::TableDefinition<u64, &[u8]> =
            redb::TableDefinition::new("chunk_offsets");
        txn.open_table(chunk_offsets_def).unwrap();
        txn.commit().unwrap();
    }

    fn write_redb_embedding_fixture(path: &Path, entries: &[(u64, Vec<u8>)]) {
        let db = redb::Database::create(path).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let def: redb::TableDefinition<u64, &[u8]> =
                redb::TableDefinition::new("embeddings");
            let mut t = txn.open_table(def).unwrap();
            for (id, payload) in entries {
                t.insert(id, payload.as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    #[test]
    fn detects_redb_magic_only() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("config.db");
        write_redb_config_fixture(
            &p,
            &[("notes".into(), b"/tmp".to_vec())],
            &[],
        );
        assert!(is_redb_file(&p).unwrap());
    }

    #[test]
    fn returns_false_for_missing_or_empty_or_heed_files() {
        let dir = tempdir().unwrap();
        // missing
        let missing = dir.path().join("nope");
        assert!(!is_redb_file(&missing).unwrap());

        // empty file
        let empty = dir.path().join("empty");
        std::fs::write(&empty, b"").unwrap();
        assert!(!is_redb_file(&empty).unwrap());

        // a heed env at the same name
        let heed = dir.path().join("heed");
        let env = open_heed_env(&heed, 1024 * 1024, 2).unwrap();
        drop(env);
        assert!(!is_redb_file(&heed).unwrap());
    }

    #[test]
    fn migration_is_noop_for_fresh_path() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("missing.db");
        ensure_config_db_migrated(&p).unwrap();
        assert!(!p.exists(), "migration should not touch a missing path");
    }

    #[test]
    fn migration_is_noop_for_already_heed_file() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("config.db");
        let env = open_heed_env(&p, 1024 * 1024, CONFIG_MAX_DBS).unwrap();
        drop(env);

        // No exception, no backup file written.
        ensure_config_db_migrated(&p).unwrap();
        assert!(!backup_path_for(&p).exists());
    }

    #[test]
    fn migration_copies_config_db_round_trip() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("config.db");
        let collections =
            vec![("notes".to_string(), b"/home/user/notes".to_vec())];
        let metadata = vec![(42u64, b"metadata-blob".to_vec())];
        write_redb_config_fixture(&p, &collections, &metadata);

        ensure_config_db_migrated(&p).unwrap();

        // Old file moved aside.
        assert!(backup_path_for(&p).exists());
        // New file is heed.
        assert!(!is_redb_file(&p).unwrap());

        // Read the migrated env back and compare.
        let env = open_heed_env(&p, MIGRATION_CONFIG_MAP_SIZE, CONFIG_MAX_DBS)
            .unwrap();
        let rtxn = env.read_txn().unwrap();
        let collections_db: Database<Str, Bytes> = env
            .open_database(&rtxn, Some("collections"))
            .unwrap()
            .unwrap();
        assert_eq!(
            collections_db.get(&rtxn, "notes").unwrap(),
            Some(b"/home/user/notes".as_slice())
        );
        let meta_db: Database<U64<BigEndian>, Bytes> = env
            .open_database(&rtxn, Some("document_metadata"))
            .unwrap()
            .unwrap();
        assert_eq!(
            meta_db.get(&rtxn, &42u64).unwrap(),
            Some(b"metadata-blob".as_slice())
        );
    }

    #[test]
    fn migration_copies_embedding_db_round_trip() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("embeddings.db");
        let entries =
            vec![(1u64, b"first".to_vec()), (u64::MAX, b"last".to_vec())];
        write_redb_embedding_fixture(&p, &entries);

        ensure_embedding_db_migrated(&p).unwrap();

        assert!(!is_redb_file(&p).unwrap());
        let env =
            open_heed_env(&p, MIGRATION_EMBEDDING_MAP_SIZE, EMBEDDINGS_MAX_DBS)
                .unwrap();
        let rtxn = env.read_txn().unwrap();
        let db: Database<U64<BigEndian>, Bytes> = env
            .open_database(&rtxn, Some("embeddings"))
            .unwrap()
            .unwrap();
        assert_eq!(db.get(&rtxn, &1u64).unwrap(), Some(b"first".as_slice()));
        assert_eq!(db.get(&rtxn, &u64::MAX).unwrap(), Some(b"last".as_slice()));
    }

    #[test]
    fn migration_refuses_when_backup_already_exists() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("config.db");
        write_redb_config_fixture(&p, &[], &[]);
        std::fs::write(backup_path_for(&p), b"prior backup").unwrap();

        let err = ensure_config_db_migrated(&p).unwrap_err();
        assert!(
            err.to_string().contains("backup already exists"),
            "expected stale-backup error, got: {err}"
        );
    }

    #[test]
    fn migration_recovers_stale_heed_tmp() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("config.db");
        write_redb_config_fixture(&p, &[], &[]);
        // Pretend a previous run crashed mid-write.
        std::fs::write(tmp_path_for(&p), b"junk").unwrap();

        ensure_config_db_migrated(&p).unwrap();
        // Migration succeeded and the post-swap data file is heed.
        assert!(!is_redb_file(&p).unwrap());
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 32,
            ..ProptestConfig::default()
        })]

        /// Any set of (str-keyed, bytes-valued) entries written to the
        /// `collections` table of a redb file should be readable
        /// byte-for-byte from the heed-backed env after migration.
        #[test]
        fn proptest_collections_round_trip(
            entries in proptest::collection::vec(
                (
                    "[a-z][a-z0-9_-]{0,31}",
                    proptest::collection::vec(any::<u8>(), 0..256),
                ),
                0..16,
            ),
        ) {
            let dir = tempdir().unwrap();
            let p = dir.path().join("config.db");

            // Dedup keys so the fixture is deterministic.
            let mut by_key = std::collections::BTreeMap::new();
            for (k, v) in entries {
                by_key.insert(k, v);
            }
            let entries: Vec<_> = by_key.into_iter().collect();
            write_redb_config_fixture(&p, &entries, &[]);

            ensure_config_db_migrated(&p).unwrap();
            let env =
                open_heed_env(&p, MIGRATION_CONFIG_MAP_SIZE, CONFIG_MAX_DBS)
                    .unwrap();
            let rtxn = env.read_txn().unwrap();
            let db: Database<Str, Bytes> = env
                .open_database(&rtxn, Some("collections"))
                .unwrap()
                .unwrap();
            for (k, expected) in &entries {
                let actual = db.get(&rtxn, k.as_str()).unwrap();
                prop_assert_eq!(actual, Some(expected.as_slice()));
            }
        }

        /// Same property for u64-keyed embedding rows. Covers the
        /// little-endian (redb) → big-endian (heed) key re-encoding so
        /// `u64::MAX` and small integer keys still round-trip.
        #[test]
        fn proptest_embeddings_round_trip(
            entries in proptest::collection::vec(
                (
                    any::<u64>(),
                    proptest::collection::vec(any::<u8>(), 0..512),
                ),
                0..16,
            ),
        ) {
            let dir = tempdir().unwrap();
            let p = dir.path().join("embeddings.db");

            let mut by_key = std::collections::BTreeMap::new();
            for (k, v) in entries {
                by_key.insert(k, v);
            }
            let entries: Vec<_> = by_key.into_iter().collect();
            write_redb_embedding_fixture(&p, &entries);

            ensure_embedding_db_migrated(&p).unwrap();
            let env = open_heed_env(
                &p,
                MIGRATION_EMBEDDING_MAP_SIZE,
                EMBEDDINGS_MAX_DBS,
            )
            .unwrap();
            let rtxn = env.read_txn().unwrap();
            let db: Database<U64<BigEndian>, Bytes> =
                env.open_database(&rtxn, Some("embeddings")).unwrap().unwrap();
            for (k, expected) in &entries {
                let actual = db.get(&rtxn, k).unwrap();
                prop_assert_eq!(actual, Some(expected.as_slice()));
            }
        }
    }

    #[test]
    fn round_trip_preserves_a_legacy_redb_table_after_re_open() {
        // Belt-and-braces test: write redb, migrate, then re-open
        // through `open_heed_env` (the production path) twice. The
        // second open used to be a hot spot for heed lockfile
        // confusion when migration left orphan `*-lock` files behind.
        let dir = tempdir().unwrap();
        let p = dir.path().join("config.db");
        write_redb_config_fixture(
            &p,
            &[("k".into(), vec![1, 2, 3])],
            &[(7, vec![9, 9])],
        );
        ensure_config_db_migrated(&p).unwrap();

        for _ in 0..2 {
            let env =
                open_heed_env(&p, MIGRATION_CONFIG_MAP_SIZE, CONFIG_MAX_DBS)
                    .unwrap();
            let rtxn = env.read_txn().unwrap();
            let db: Database<Str, Bytes> = env
                .open_database(&rtxn, Some("collections"))
                .unwrap()
                .unwrap();
            assert_eq!(
                db.get(&rtxn, "k").unwrap(),
                Some(vec![1, 2, 3].as_slice())
            );
        }
    }
}
