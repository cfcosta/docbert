//! Top-level error type for rustbert.
//!
//! Each variant wraps a single source error via `#[from]` so call sites
//! can use `?` instead of `map_err`.

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid crate reference: {0}")]
    InvalidCrateRef(String),

    #[error("invalid synthetic collection name: {0}")]
    InvalidCollectionName(String),

    #[error("invalid version specifier: {0}")]
    InvalidVersion(#[from] semver::Error),

    #[error("Cargo.lock parse error: {0}")]
    CargoLock(#[from] cargo_lock::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP transport error fetching {url}: {message}")]
    HttpTransport { url: String, message: String },

    #[error("HTTP {status} fetching {url}")]
    HttpStatus { url: String, status: u16 },

    #[error("crate not found on registry: {name}")]
    CrateNotFound { name: String },

    #[error("crates.io API error: {0}")]
    CratesIoApi(String),

    #[error("no version of `{name}` matches `{spec}`")]
    NoMatchingVersion { name: String, spec: String },

    #[error(
        "checksum mismatch for {name}@{version}: expected {expected}, got {actual}"
    )]
    ChecksumMismatch {
        name: String,
        version: semver::Version,
        expected: String,
        actual: String,
    },

    #[error("tarball entry escapes extraction root: {path}")]
    UnsafeTarballEntry { path: String },

    #[error("tarball is empty or has no top-level directory")]
    EmptyTarball,

    #[error("syn parse error in {path}: {source}")]
    Syn {
        path: String,
        #[source]
        source: syn::Error,
    },

    #[error("no crate entry point (src/lib.rs or src/main.rs) found in {path}")]
    NoEntryPoint { path: String },

    #[error(
        "could not resolve a data directory: set $RUSTBERT_DATA_DIR or $HOME"
    )]
    DataDirUnknown,

    #[error("cache error: {0}")]
    Cache(String),

    #[error("Cargo.lock not found in cwd or any ancestor")]
    LockfileNotFound,

    #[error("Tantivy error: {0}")]
    Tantivy(#[from] tantivy::error::TantivyError),

    #[error("invalid glob pattern: {0}")]
    Globset(#[from] globset::Error),
}
