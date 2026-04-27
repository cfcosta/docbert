//! crates.io API client.
//!
//! Wraps a [`Fetcher`] and exposes typed endpoints for the bits
//! rustbert needs: `GET /api/v1/crates/{name}` to list versions
//! and `/api/v1/crates/{name}/{version}/download` for tarballs.
//!
//! Base URL is configurable so the same client can target a private
//! registry mirror via `rustbert sync --registry URL`.

use serde::Deserialize;

use crate::{
    error::{Error, Result},
    fetcher::Fetcher,
};

const DEFAULT_BASE_URL: &str = "https://crates.io";

#[derive(Debug, Clone)]
pub struct CrateMetadata {
    pub name: String,
    pub versions: Vec<PublishedVersion>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedVersion {
    pub num: semver::Version,
    pub yanked: bool,
    pub checksum: String,
}

pub struct CratesIoApi<F: Fetcher> {
    fetcher: F,
    base_url: String,
}

impl<F: Fetcher> CratesIoApi<F> {
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

    /// Borrow the underlying fetcher. Used by code paths that need
    /// to issue additional requests (tarball downloads, etc.) through
    /// the same client.
    pub fn fetcher(&self) -> &F {
        &self.fetcher
    }

    /// URL for `GET /api/v1/crates/{name}` against the configured base.
    pub fn metadata_url(&self, name: &str) -> String {
        format!("{}/api/v1/crates/{}", self.base_url, name)
    }

    /// URL for the tarball download endpoint for `(name, version)`.
    pub fn download_url(
        &self,
        name: &str,
        version: &semver::Version,
    ) -> String {
        format!(
            "{}/api/v1/crates/{}/{}/download",
            self.base_url, name, version,
        )
    }

    /// List all published versions for a crate.
    ///
    /// Returns [`Error::CrateNotFound`] for a 404 response and
    /// [`Error::CratesIoApi`] for malformed payloads.
    pub async fn crate_metadata(&self, name: &str) -> Result<CrateMetadata> {
        let url = self.metadata_url(name);
        let bytes = match self.fetcher.get_bytes(&url).await {
            Ok(b) => b,
            Err(Error::HttpStatus { status: 404, .. }) => {
                return Err(Error::CrateNotFound {
                    name: name.to_string(),
                });
            }
            Err(other) => return Err(other),
        };

        let response: CrateMetadataResponse = serde_json::from_slice(&bytes)
            .map_err(|e| Error::CratesIoApi(format!("decode {url}: {e}")))?;

        let mut versions = Vec::with_capacity(response.versions.len());
        for v in response.versions {
            let num = semver::Version::parse(&v.num).map_err(|e| {
                Error::CratesIoApi(format!(
                    "version {:?} for {name} is not valid semver: {e}",
                    v.num
                ))
            })?;
            // Yanked versions on crates.io may have a null checksum
            // (rare, but defensible); treat the missing case as empty.
            versions.push(PublishedVersion {
                num,
                yanked: v.yanked,
                checksum: v.checksum.unwrap_or_default(),
            });
        }

        Ok(CrateMetadata {
            name: response.crate_field.name,
            versions,
        })
    }
}

#[derive(Deserialize)]
struct CrateMetadataResponse {
    #[serde(rename = "crate")]
    crate_field: CrateField,
    versions: Vec<VersionField>,
}

#[derive(Deserialize)]
struct CrateField {
    name: String,
}

#[derive(Deserialize)]
struct VersionField {
    num: String,
    yanked: bool,
    #[serde(default)]
    checksum: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetcher::FakeFetcher;

    fn metadata_payload() -> Vec<u8> {
        serde_json::json!({
            "crate": {
                "id": "serde",
                "name": "serde",
                "max_version": "1.0.219",
            },
            "versions": [
                {
                    "num": "1.0.219",
                    "yanked": false,
                    "checksum": "1111111111111111111111111111111111111111111111111111111111111111",
                },
                {
                    "num": "1.0.218",
                    "yanked": false,
                    "checksum": "2222222222222222222222222222222222222222222222222222222222222222",
                },
                {
                    "num": "1.0.0-rc.1",
                    "yanked": true,
                    "checksum": "3333333333333333333333333333333333333333333333333333333333333333",
                },
            ],
        })
        .to_string()
        .into_bytes()
    }

    #[tokio::test]
    async fn returns_parsed_versions() {
        let fetcher = FakeFetcher::new().with_bytes(
            "https://crates.io/api/v1/crates/serde",
            metadata_payload(),
        );
        let api = CratesIoApi::new(fetcher);

        let metadata = api.crate_metadata("serde").await.unwrap();
        assert_eq!(metadata.name, "serde");
        assert_eq!(metadata.versions.len(), 3);
        assert_eq!(metadata.versions[0].num, semver::Version::new(1, 0, 219));
        assert!(!metadata.versions[0].yanked);
    }

    #[tokio::test]
    async fn yanked_flag_is_propagated() {
        let fetcher = FakeFetcher::new().with_bytes(
            "https://crates.io/api/v1/crates/serde",
            metadata_payload(),
        );
        let api = CratesIoApi::new(fetcher);

        let metadata = api.crate_metadata("serde").await.unwrap();
        let prerelease = metadata
            .versions
            .iter()
            .find(|v| v.num.pre.as_str() == "rc.1")
            .expect("payload contains 1.0.0-rc.1");
        assert!(prerelease.yanked);
    }

    #[tokio::test]
    async fn checksum_is_propagated() {
        let fetcher = FakeFetcher::new().with_bytes(
            "https://crates.io/api/v1/crates/serde",
            metadata_payload(),
        );
        let api = CratesIoApi::new(fetcher);
        let metadata = api.crate_metadata("serde").await.unwrap();
        assert_eq!(
            metadata.versions[0].checksum,
            "1111111111111111111111111111111111111111111111111111111111111111"
        );
    }

    #[tokio::test]
    async fn missing_crate_404_becomes_crate_not_found() {
        let fetcher = FakeFetcher::new()
            .with_status("https://crates.io/api/v1/crates/no-such-crate", 404);
        let api = CratesIoApi::new(fetcher);

        let err = api.crate_metadata("no-such-crate").await.unwrap_err();
        assert!(
            matches!(err, Error::CrateNotFound { name } if name == "no-such-crate")
        );
    }

    #[tokio::test]
    async fn other_http_status_is_propagated() {
        let fetcher = FakeFetcher::new()
            .with_status("https://crates.io/api/v1/crates/serde", 503);
        let api = CratesIoApi::new(fetcher);

        let err = api.crate_metadata("serde").await.unwrap_err();
        assert!(matches!(err, Error::HttpStatus { status: 503, .. }));
    }

    #[tokio::test]
    async fn malformed_json_becomes_api_error() {
        let fetcher = FakeFetcher::new().with_bytes(
            "https://crates.io/api/v1/crates/serde",
            b"not json".to_vec(),
        );
        let api = CratesIoApi::new(fetcher);

        let err = api.crate_metadata("serde").await.unwrap_err();
        assert!(matches!(err, Error::CratesIoApi(_)));
    }

    #[tokio::test]
    async fn invalid_semver_in_payload_becomes_api_error() {
        let bad = serde_json::json!({
            "crate": { "id": "x", "name": "x", "max_version": "potato" },
            "versions": [{ "num": "potato", "yanked": false, "checksum": "abc" }]
        })
        .to_string()
        .into_bytes();
        let fetcher = FakeFetcher::new()
            .with_bytes("https://crates.io/api/v1/crates/x", bad);
        let api = CratesIoApi::new(fetcher);

        let err = api.crate_metadata("x").await.unwrap_err();
        assert!(matches!(err, Error::CratesIoApi(_)));
    }

    #[tokio::test]
    async fn metadata_url_uses_configured_base() {
        let api = CratesIoApi::with_base_url(
            FakeFetcher::new(),
            "https://my-mirror.example",
        );
        assert_eq!(
            api.metadata_url("serde"),
            "https://my-mirror.example/api/v1/crates/serde"
        );
    }

    #[tokio::test]
    async fn download_url_uses_configured_base() {
        let api = CratesIoApi::with_base_url(
            FakeFetcher::new(),
            "https://my-mirror.example",
        );
        let v = semver::Version::new(1, 0, 219);
        assert_eq!(
            api.download_url("serde", &v),
            "https://my-mirror.example/api/v1/crates/serde/1.0.219/download"
        );
    }
}
