//! End-to-end ingestion: `(crate, version_spec) → cached items`.
//!
//! Composes [`crate::crates_io`] (metadata), [`crate::resolver`]
//! (Latest/Req → concrete), [`crate::download`] (verified download),
//! [`crate::extract`] (tarball → source tree), [`crate::crate_walker`]
//! (source tree → items), and [`crate::cache`] (persist).
//!
//! Idempotent: if `(crate, resolved_version)` is already cached and
//! `force` is false, the function returns immediately with
//! `IngestionReport::AlreadyCached`.

use std::path::Path;

use crate::{
    cache::CrateCache,
    collection::SyntheticCollection,
    crate_ref::{CrateRef, VersionSpec},
    crate_walker,
    crates_io::CratesIoApi,
    data_dir,
    docs_rs::DocsRsClient,
    download,
    error::Result,
    extract,
    fetcher::Fetcher,
    indexer::Indexer,
    resolver,
};

#[derive(Debug, Clone)]
pub enum IngestionReport {
    AlreadyCached {
        collection: SyntheticCollection,
        item_count: usize,
    },
    Fetched {
        collection: SyntheticCollection,
        item_count: usize,
        load_failures: Vec<String>,
        was_yanked: bool,
    },
}

impl IngestionReport {
    pub fn collection(&self) -> &SyntheticCollection {
        match self {
            Self::AlreadyCached { collection, .. }
            | Self::Fetched { collection, .. } => collection,
        }
    }

    pub fn item_count(&self) -> usize {
        match self {
            Self::AlreadyCached { item_count, .. }
            | Self::Fetched { item_count, .. } => *item_count,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct IngestionOptions {
    pub force: bool,
}

/// Ingest one crate. Resolves the version against crates.io if
/// necessary, downloads the verified tarball, extracts, walks, and
/// stores the item set in the cache + lexical index.
///
/// Embedding is intentionally deferred to [`crate::sync`] (which
/// handles batch embedding + PLAID rebuild after a full sync run) —
/// single-crate auto-fetch from `search`/`get`/`list` shouldn't pay
/// the embedding cost.
#[tracing::instrument(skip(fetcher, api, cache, indexer), fields(crate_name = %crate_ref.name))]
pub async fn ingest<F: Fetcher + Clone>(
    fetcher: &F,
    api: &CratesIoApi<F>,
    cache: &CrateCache,
    indexer: &Indexer,
    crate_ref: &CrateRef,
    options: IngestionOptions,
) -> Result<IngestionReport> {
    let report =
        ingest_to_cache(fetcher, api, cache, crate_ref, options).await?;
    let items = cache.load(report.collection())?;
    indexer.index_lexical(report.collection(), &items)?;
    Ok(report)
}

/// Cache-only ingest path. Use this from parallel runners where each
/// task can't share an `Indexer` mutably — index sequentially after
/// the parallel fetch completes via [`Indexer::index_lexical`].
pub async fn ingest_to_cache<F: Fetcher + Clone>(
    fetcher: &F,
    api: &CratesIoApi<F>,
    cache: &CrateCache,
    crate_ref: &CrateRef,
    options: IngestionOptions,
) -> Result<IngestionReport> {
    // Concrete-version short-circuit: if the user asked for an exact
    // version and we already have it cached (from a fetch or from
    // `rustbert index`), skip the crates.io round-trip entirely.
    if !options.force
        && let VersionSpec::Concrete(ref v) = crate_ref.version
    {
        let collection = SyntheticCollection {
            crate_name: crate_ref.name.clone(),
            version: v.clone(),
        };
        if cache.has(&collection) {
            let count = cache.load(&collection)?.len();
            return Ok(IngestionReport::AlreadyCached {
                collection,
                item_count: count,
            });
        }
    }

    // Latest / req short-circuit: if we have ANY cached version of
    // this crate, surface the highest one. That's not strictly the
    // upstream "latest", but it lets `rustbert search myproj` work
    // against a host-indexed project without hitting crates.io.
    if !options.force && matches!(crate_ref.version, VersionSpec::Latest) {
        let mut best: Option<SyntheticCollection> = None;
        for entry in cache.entries()? {
            if entry.crate_name != crate_ref.name {
                continue;
            }
            best = match best {
                None => Some(SyntheticCollection {
                    crate_name: entry.crate_name,
                    version: entry.version,
                }),
                Some(prev) if entry.version > prev.version => {
                    Some(SyntheticCollection {
                        crate_name: entry.crate_name,
                        version: entry.version,
                    })
                }
                Some(prev) => Some(prev),
            };
        }
        if let Some(coll) = best {
            let count = cache.load(&coll)?.len();
            return Ok(IngestionReport::AlreadyCached {
                collection: coll,
                item_count: count,
            });
        }
    }

    let metadata = api.crate_metadata(&crate_ref.name).await?;
    let resolution = resolver::resolve(&crate_ref.version, &metadata)?;
    let collection = SyntheticCollection {
        crate_name: crate_ref.name.clone(),
        version: resolution.version.clone(),
    };
    tracing::info!(version = %resolution.version, "resolved");

    // Pin "latest"/semver-pattern requests so subsequent calls don't
    // hit crates.io for the resolution alone.
    if !matches!(crate_ref.version, VersionSpec::Concrete(_)) {
        let requested = match &crate_ref.version {
            VersionSpec::Latest => "latest".to_string(),
            VersionSpec::Req(r) => r.to_string(),
            VersionSpec::Concrete(_) => unreachable!(),
        };
        cache.record_resolved(
            &crate_ref.name,
            &requested,
            &resolution.version,
        )?;
    }

    if !options.force && cache.has(&collection) {
        let count = cache.load(&collection)?.len();
        tracing::debug!(item_count = count, "already cached");
        return Ok(IngestionReport::AlreadyCached {
            collection,
            item_count: count,
        });
    }
    tracing::info!("fetching tarball");

    let bytes = download::download_verified(
        api,
        fetcher,
        &crate_ref.name,
        &resolution.version,
        &resolution.checksum,
    )
    .await?;

    let extract_dest = data_dir::extracted_crate_dir(
        cache.data_dir(),
        &collection.crate_name,
        &collection.version,
    );
    if extract_dest.exists() {
        std::fs::remove_dir_all(&extract_dest)?;
    }
    extract::extract_crate_tarball(&bytes, &extract_dest)?;

    let walked = crate_walker::walk_extracted_crate(
        &extract_dest,
        &collection.crate_name,
        &collection.version,
    )?;

    tracing::info!(
        item_count = walked.items.len(),
        load_failures = walked.failures.len(),
        "walk complete"
    );
    cache.store(&collection, &walked.items)?;

    // Best-effort rustdoc JSON enrichment. docs.rs publishes JSON
    // for many but not all crate builds; a 404 isn't fatal —
    // we just skip the enrichment and rely on syn output.
    let docs_client = DocsRsClient::new(fetcher.clone());
    if let Ok(Some(json_bytes)) = docs_client
        .fetch_rustdoc_json(&collection.crate_name, &collection.version)
        .await
    {
        let _ = crate::docs_rs::write_rustdoc_json(
            cache.data_dir(),
            &collection.crate_name,
            &collection.version,
            &json_bytes,
        );
        tracing::info!(
            bytes = json_bytes.len(),
            "rustdoc JSON cached for enrichment"
        );
    }

    let load_failures: Vec<String> = walked
        .failures
        .iter()
        .map(|f| format!("{}: {}", f.source_file.display(), f.reason))
        .collect();

    Ok(IngestionReport::Fetched {
        collection,
        item_count: walked.items.len(),
        load_failures,
        was_yanked: resolution.yanked,
    })
}

/// Convenience: write the raw tarball alongside the extracted tree.
/// Some debug paths want both. Optional.
pub fn write_tarball(
    cache_dir: &Path,
    name: &str,
    version: &semver::Version,
    bytes: &[u8],
) -> Result<()> {
    let path = data_dir::crate_tarball_path(cache_dir, name, version);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::{Compression, write::GzEncoder};
    use tempfile::TempDir;

    use super::*;
    use crate::fetcher::FakeFetcher;

    fn build_tarball(top: &str, entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut tar_bytes = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut tar_bytes);
            for (path, body) in entries {
                let full = format!("{top}/{path}");
                let mut header = tar::Header::new_gnu();
                header.set_path(&full).unwrap();
                header.set_size(body.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                builder.append(&header, *body).unwrap();
            }
            builder.finish().unwrap();
        }
        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        gz.write_all(&tar_bytes).unwrap();
        gz.finish().unwrap()
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::Digest;
        let digest = sha2::Sha256::digest(bytes);
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn metadata_payload(version: &str, checksum: &str) -> Vec<u8> {
        serde_json::json!({
            "crate": { "id": "demo", "name": "demo", "max_version": version },
            "versions": [
                { "num": version, "yanked": false, "checksum": checksum }
            ],
        })
        .to_string()
        .into_bytes()
    }

    fn setup() -> (TempDir, CrateCache, Indexer) {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let indexer = Indexer::open(tmp.path()).unwrap();
        (tmp, cache, indexer)
    }

    #[tokio::test]
    async fn happy_path_fetches_parses_and_caches() {
        let tarball = build_tarball(
            "demo-1.0.0",
            &[("src/lib.rs", b"/// hi\npub fn greet() {}")],
        );
        let checksum = sha256_hex(&tarball);
        let meta = metadata_payload("1.0.0", &checksum);

        let fetcher = FakeFetcher::new()
            .with_bytes("https://crates.io/api/v1/crates/demo", meta)
            .with_bytes(
                "https://crates.io/api/v1/crates/demo/1.0.0/download",
                tarball,
            );
        let api = CratesIoApi::new(FakeFetcher::new().with_bytes(
            "https://crates.io/api/v1/crates/demo",
            metadata_payload("1.0.0", &checksum),
        ));

        let (_tmp, cache, indexer) = setup();
        let report = ingest(
            &fetcher,
            &api,
            &cache,
            &indexer,
            &CrateRef::parse("demo").unwrap(),
            IngestionOptions::default(),
        )
        .await
        .unwrap();

        assert!(matches!(report, IngestionReport::Fetched { .. }));
        assert_eq!(report.item_count(), 1);
        assert_eq!(report.collection().crate_name, "demo");
        assert_eq!(report.collection().version, semver::Version::new(1, 0, 0));

        // The cache now has the items.
        let items = cache.load(report.collection()).unwrap();
        assert_eq!(items[0].qualified_path, "demo::greet");
        assert_eq!(items[0].doc_markdown, "hi");
    }

    #[tokio::test]
    async fn idempotent_repeat_call_returns_already_cached() {
        let tarball = build_tarball(
            "demo-1.0.0",
            &[("src/lib.rs", b"pub fn greet() {}")],
        );
        let checksum = sha256_hex(&tarball);
        let meta = metadata_payload("1.0.0", &checksum);

        let fetcher = FakeFetcher::new()
            .with_bytes("https://crates.io/api/v1/crates/demo", meta.clone())
            .with_bytes(
                "https://crates.io/api/v1/crates/demo/1.0.0/download",
                tarball,
            );
        let api = CratesIoApi::new(
            FakeFetcher::new()
                .with_bytes("https://crates.io/api/v1/crates/demo", meta),
        );

        let (_tmp, cache, indexer) = setup();
        let r1 = ingest(
            &fetcher,
            &api,
            &cache,
            &indexer,
            &CrateRef::parse("demo").unwrap(),
            IngestionOptions::default(),
        )
        .await
        .unwrap();
        assert!(matches!(r1, IngestionReport::Fetched { .. }));

        // Second call — same input, should not re-fetch.
        let api2 = CratesIoApi::new(FakeFetcher::new().with_bytes(
            "https://crates.io/api/v1/crates/demo",
            metadata_payload("1.0.0", &checksum),
        ));
        let r2 = ingest(
            &FakeFetcher::new(), // empty fetcher — would error if used
            &api2,
            &cache,
            &indexer,
            &CrateRef::parse("demo").unwrap(),
            IngestionOptions::default(),
        )
        .await
        .unwrap();
        assert!(matches!(r2, IngestionReport::AlreadyCached { .. }));
    }

    #[tokio::test]
    async fn force_re_fetches_even_when_cached() {
        let tarball = build_tarball(
            "demo-1.0.0",
            &[("src/lib.rs", b"pub fn greet() {}")],
        );
        let checksum = sha256_hex(&tarball);
        let meta = metadata_payload("1.0.0", &checksum);

        let (_tmp, cache, indexer) = setup();
        // Pre-populate
        let coll = SyntheticCollection {
            crate_name: "demo".to_string(),
            version: semver::Version::new(1, 0, 0),
        };
        cache.store(&coll, &[]).unwrap();

        let fetcher = FakeFetcher::new()
            .with_bytes("https://crates.io/api/v1/crates/demo", meta.clone())
            .with_bytes(
                "https://crates.io/api/v1/crates/demo/1.0.0/download",
                tarball,
            );
        let api = CratesIoApi::new(
            FakeFetcher::new()
                .with_bytes("https://crates.io/api/v1/crates/demo", meta),
        );

        let report = ingest(
            &fetcher,
            &api,
            &cache,
            &indexer,
            &CrateRef::parse("demo").unwrap(),
            IngestionOptions { force: true },
        )
        .await
        .unwrap();
        assert!(matches!(report, IngestionReport::Fetched { .. }));
        assert_eq!(report.item_count(), 1);
    }

    #[tokio::test]
    async fn latest_request_is_recorded_in_cache() {
        let tarball = build_tarball(
            "demo-1.0.0",
            &[("src/lib.rs", b"pub fn greet() {}")],
        );
        let checksum = sha256_hex(&tarball);
        let meta = metadata_payload("1.0.0", &checksum);

        let fetcher = FakeFetcher::new()
            .with_bytes("https://crates.io/api/v1/crates/demo", meta.clone())
            .with_bytes(
                "https://crates.io/api/v1/crates/demo/1.0.0/download",
                tarball,
            );
        let api = CratesIoApi::new(
            FakeFetcher::new()
                .with_bytes("https://crates.io/api/v1/crates/demo", meta),
        );

        let (_tmp, cache, indexer) = setup();
        ingest(
            &fetcher,
            &api,
            &cache,
            &indexer,
            &CrateRef::parse("demo").unwrap(),
            IngestionOptions::default(),
        )
        .await
        .unwrap();

        let resolved = cache.resolved("demo", "latest").unwrap().unwrap();
        assert_eq!(resolved.resolved_version, semver::Version::new(1, 0, 0));
    }
}
