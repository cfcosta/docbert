//! docs.rs rustdoc JSON enrichment.
//!
//! docs.rs serves rustdoc's JSON output at
//! `/crate/{name}/{version}/json` for crates whose builds completed
//! successfully on a JSON-capable target. The format is the
//! `rustdoc-types` schema; coverage is uneven (not every published
//! crate has it).
//!
//! v1 behavior: best-effort fetch + cache. The raw JSON is stored
//! alongside the extracted source under
//! `<data_dir>/items/<crate>-<version>.rustdoc.json`. Callers that
//! want richer enrichment (trait-impl edges, intra-doc link
//! resolution) can deserialize and merge — `rustdoc-types` is a
//! moving target so this layer stays format-version-agnostic.

use std::path::{Path, PathBuf};

use crate::{
    error::{Error, Result},
    fetcher::Fetcher,
};

const DEFAULT_BASE_URL: &str = "https://docs.rs";

/// HTTP client for docs.rs JSON endpoints.
pub struct DocsRsClient<F: Fetcher + Clone> {
    fetcher: F,
    base_url: String,
}

impl<F: Fetcher + Clone> DocsRsClient<F> {
    pub fn new(fetcher: F) -> Self {
        Self {
            fetcher,
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    pub fn with_base_url(fetcher: F, base_url: impl Into<String>) -> Self {
        Self {
            fetcher,
            base_url: base_url.into(),
        }
    }

    /// URL of the rustdoc JSON for `(name, version)`.
    pub fn rustdoc_url(&self, name: &str, version: &semver::Version) -> String {
        format!("{}/crate/{}/{}/json", self.base_url, name, version)
    }

    /// Fetch the rustdoc JSON. Returns `Ok(None)` for 404 (the crate
    /// has no JSON build on docs.rs); other errors propagate.
    pub async fn fetch_rustdoc_json(
        &self,
        name: &str,
        version: &semver::Version,
    ) -> Result<Option<Vec<u8>>> {
        let url = self.rustdoc_url(name, version);
        match self.fetcher.get_bytes(&url).await {
            Ok(bytes) => Ok(Some(bytes)),
            Err(Error::HttpStatus { status: 404, .. }) => Ok(None),
            Err(Error::HttpStatus { status: 410, .. }) => Ok(None),
            Err(other) => Err(other),
        }
    }
}

/// Path inside the data dir where rustdoc JSON for a (crate, version) lands.
pub fn rustdoc_json_path(
    data_dir: &Path,
    name: &str,
    version: &semver::Version,
) -> PathBuf {
    data_dir
        .join("items")
        .join(format!("{name}-{version}.rustdoc.json"))
}

/// Persist `bytes` to the rustdoc JSON cache path.
pub fn write_rustdoc_json(
    data_dir: &Path,
    name: &str,
    version: &semver::Version,
    bytes: &[u8],
) -> Result<()> {
    let path = rustdoc_json_path(data_dir, name, version);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Read the cached rustdoc JSON, if any.
pub fn read_rustdoc_json(
    data_dir: &Path,
    name: &str,
    version: &semver::Version,
) -> Result<Option<Vec<u8>>> {
    let path = rustdoc_json_path(data_dir, name, version);
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(std::fs::read(path)?))
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::fetcher::FakeFetcher;

    fn version() -> semver::Version {
        semver::Version::new(1, 0, 219)
    }

    #[tokio::test]
    async fn rustdoc_url_uses_default_base() {
        let client = DocsRsClient::new(FakeFetcher::new());
        assert_eq!(
            client.rustdoc_url("serde", &version()),
            "https://docs.rs/crate/serde/1.0.219/json"
        );
    }

    #[tokio::test]
    async fn rustdoc_url_honors_configured_base() {
        let client = DocsRsClient::with_base_url(
            FakeFetcher::new(),
            "https://my-mirror.example",
        );
        assert_eq!(
            client.rustdoc_url("serde", &version()),
            "https://my-mirror.example/crate/serde/1.0.219/json"
        );
    }

    #[tokio::test]
    async fn fetch_returns_some_for_200() {
        let url = "https://docs.rs/crate/serde/1.0.219/json";
        let fetcher = FakeFetcher::new()
            .with_bytes(url, b"{\"format_version\":1}".to_vec());
        let client = DocsRsClient::new(fetcher);
        let result = client
            .fetch_rustdoc_json("serde", &version())
            .await
            .unwrap();
        assert_eq!(result.as_deref(), Some(&b"{\"format_version\":1}"[..]));
    }

    #[tokio::test]
    async fn fetch_returns_none_for_404() {
        let url = "https://docs.rs/crate/serde/1.0.219/json";
        let fetcher = FakeFetcher::new().with_status(url, 404);
        let client = DocsRsClient::new(fetcher);
        let result = client
            .fetch_rustdoc_json("serde", &version())
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn fetch_returns_none_for_410_gone() {
        // docs.rs sometimes returns 410 for crates whose builds were
        // explicitly removed.
        let url = "https://docs.rs/crate/serde/1.0.219/json";
        let fetcher = FakeFetcher::new().with_status(url, 410);
        let client = DocsRsClient::new(fetcher);
        let result = client
            .fetch_rustdoc_json("serde", &version())
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn fetch_propagates_other_status_errors() {
        let url = "https://docs.rs/crate/serde/1.0.219/json";
        let fetcher = FakeFetcher::new().with_status(url, 500);
        let client = DocsRsClient::new(fetcher);
        let err = client
            .fetch_rustdoc_json("serde", &version())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::HttpStatus { status: 500, .. }));
    }

    #[test]
    fn write_then_read_round_trips() {
        let tmp = TempDir::new().unwrap();
        let bytes = b"{\"k\":\"v\"}".to_vec();
        write_rustdoc_json(tmp.path(), "demo", &version(), &bytes).unwrap();
        let read = read_rustdoc_json(tmp.path(), "demo", &version())
            .unwrap()
            .unwrap();
        assert_eq!(read, bytes);
    }

    #[test]
    fn read_returns_none_when_missing() {
        let tmp = TempDir::new().unwrap();
        let read = read_rustdoc_json(tmp.path(), "demo", &version()).unwrap();
        assert!(read.is_none());
    }

    #[test]
    fn rustdoc_json_path_format() {
        let p = rustdoc_json_path(Path::new("/data"), "serde", &version());
        assert_eq!(p, PathBuf::from("/data/items/serde-1.0.219.rustdoc.json"));
    }
}
