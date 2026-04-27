//! Download a `.crate` tarball and verify its SHA-256 against the
//! checksum published in the crates.io metadata.
//!
//! A mismatch is a hard failure — possibly-tampered source must never
//! reach the parser.

use sha2::{Digest, Sha256};

use crate::{
    crates_io::CratesIoApi,
    error::{Error, Result},
    fetcher::Fetcher,
};

/// Download a tarball and verify its SHA-256 checksum.
///
/// `expected_checksum` is the lowercase hex digest as published in the
/// crates.io metadata. An empty `expected_checksum` is rejected — the
/// caller never wants to skip the integrity check silently.
pub async fn download_verified<F: Fetcher>(
    api: &CratesIoApi<F>,
    fetcher: &F,
    name: &str,
    version: &semver::Version,
    expected_checksum: &str,
) -> Result<Vec<u8>> {
    if expected_checksum.is_empty() {
        return Err(Error::ChecksumMismatch {
            name: name.to_string(),
            version: version.clone(),
            expected: "<missing>".to_string(),
            actual: "<unknown>".to_string(),
        });
    }

    let url = api.download_url(name, version);
    let bytes = fetcher.get_bytes(&url).await?;
    let actual = sha256_hex(&bytes);

    if !actual.eq_ignore_ascii_case(expected_checksum) {
        return Err(Error::ChecksumMismatch {
            name: name.to_string(),
            version: version.clone(),
            expected: expected_checksum.to_string(),
            actual,
        });
    }

    Ok(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(out, "{b:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetcher::FakeFetcher;

    fn payload() -> Vec<u8> {
        b"# crate contents".to_vec()
    }

    fn known_checksum() -> String {
        sha256_hex(&payload())
    }

    fn api(fetcher: FakeFetcher) -> CratesIoApi<FakeFetcher> {
        CratesIoApi::new(fetcher)
    }

    fn version() -> semver::Version {
        semver::Version::new(1, 0, 219)
    }

    fn url() -> String {
        "https://crates.io/api/v1/crates/serde/1.0.219/download".to_string()
    }

    #[tokio::test]
    async fn matching_checksum_returns_bytes() {
        let fetcher = FakeFetcher::new().with_bytes(url(), payload());
        let api = CratesIoApi::new(fetcher);
        // CratesIoApi consumed the fetcher; build a parallel one for the
        // download (the real call site shares a single fetcher reference).
        let fetcher2 = FakeFetcher::new().with_bytes(url(), payload());
        let bytes = download_verified(
            &api,
            &fetcher2,
            "serde",
            &version(),
            &known_checksum(),
        )
        .await
        .unwrap();
        assert_eq!(bytes, payload());
    }

    #[tokio::test]
    async fn checksum_mismatch_errors() {
        let fetcher = FakeFetcher::new().with_bytes(url(), payload());
        let api = api(FakeFetcher::new().with_bytes(url(), payload()));
        let err = download_verified(
            &api,
            &fetcher,
            "serde",
            &version(),
            "0000000000000000000000000000000000000000000000000000000000000000",
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::ChecksumMismatch { .. }));
    }

    #[tokio::test]
    async fn empty_checksum_is_rejected() {
        let fetcher = FakeFetcher::new().with_bytes(url(), payload());
        let api = api(FakeFetcher::new().with_bytes(url(), payload()));
        let err = download_verified(&api, &fetcher, "serde", &version(), "")
            .await
            .unwrap_err();
        assert!(matches!(err, Error::ChecksumMismatch { .. }));
    }

    #[tokio::test]
    async fn checksum_comparison_is_case_insensitive() {
        let fetcher = FakeFetcher::new().with_bytes(url(), payload());
        let api = api(FakeFetcher::new().with_bytes(url(), payload()));
        let upper = known_checksum().to_uppercase();
        let bytes =
            download_verified(&api, &fetcher, "serde", &version(), &upper)
                .await
                .unwrap();
        assert_eq!(bytes, payload());
    }

    #[tokio::test]
    async fn http_404_propagates() {
        let fetcher = FakeFetcher::new().with_status(url(), 404);
        let api = api(FakeFetcher::new().with_bytes(url(), payload()));
        let err = download_verified(
            &api,
            &fetcher,
            "serde",
            &version(),
            &known_checksum(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::HttpStatus { status: 404, .. }));
    }

    #[test]
    fn sha256_hex_for_empty_input_is_known_value() {
        // sha256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[hegel::test(test_cases = 30)]
    fn prop_matching_checksum_round_trips(tc: hegel::TestCase) {
        use hegel::generators as gs;

        let bytes: Vec<u8> =
            tc.draw(gs::vecs(gs::integers::<u8>()).max_size(256));
        let checksum = sha256_hex(&bytes);

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        runtime.block_on(async {
            let fetcher = FakeFetcher::new().with_bytes(url(), bytes.clone());
            let api = api(FakeFetcher::new().with_bytes(url(), bytes.clone()));
            let result = download_verified(
                &api,
                &fetcher,
                "serde",
                &version(),
                &checksum,
            )
            .await
            .unwrap();
            assert_eq!(result, bytes);
        });
    }
}
