//! End-to-end integration test: fetch a synthetic crate via the
//! `FakeFetcher`, parse it, store it in the cache, and verify search
//! and retrieval surface the expected items.
//!
//! Exercises every layer except the real network and the actual
//! binary spawn — that's covered by manual smoke testing against
//! crates.io.

use std::io::Write;

use flate2::{Compression, write::GzEncoder};
use rustbert::{
    cache::CrateCache,
    crate_ref::CrateRef,
    crates_io::CratesIoApi,
    fetcher::FakeFetcher,
    indexer::Indexer,
    ingestion::{self, IngestionOptions, IngestionReport},
    item::RustItemKind,
    search::{self, SearchOptions},
};
use tempfile::TempDir;

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
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn metadata(versions: &[(&str, &str, bool)]) -> Vec<u8> {
    let versions_json: Vec<_> = versions
        .iter()
        .map(|(num, checksum, yanked)| {
            serde_json::json!({
                "num": num,
                "yanked": yanked,
                "checksum": checksum,
            })
        })
        .collect();
    serde_json::json!({
        "crate": { "id": "demo", "name": "demo", "max_version": versions[0].0 },
        "versions": versions_json,
    })
    .to_string()
    .into_bytes()
}

const SAMPLE_LIB: &[u8] = b"\
//! Demo crate.

/// Greet someone by name.
pub fn greet(name: &str) -> String {
    format!(\"hello, {name}\")
}

/// A holder for arbitrary data.
pub struct Holder<T> {
    inner: T,
}

impl<T> Holder<T> {
    /// Wrap a value.
    pub fn new(inner: T) -> Self { Self { inner } }
}

pub mod nested;
";

const SAMPLE_NESTED: &[u8] = b"\
/// A trait describing things that can ping back.
pub trait Pingable {
    fn ping(&self) -> &'static str;
}

pub const MAX_PINGS: u32 = 100;
";

#[tokio::test]
async fn fetch_parse_index_and_search_roundtrip() {
    // Build a synthetic crate.
    let tarball = build_tarball(
        "demo-1.0.0",
        &[("src/lib.rs", SAMPLE_LIB), ("src/nested.rs", SAMPLE_NESTED)],
    );
    let checksum = sha256_hex(&tarball);

    let fetcher = FakeFetcher::new()
        .with_bytes(
            "https://crates.io/api/v1/crates/demo",
            metadata(&[("1.0.0", &checksum, false)]),
        )
        .with_bytes(
            "https://crates.io/api/v1/crates/demo/1.0.0/download",
            tarball,
        );
    let api = CratesIoApi::new(fetcher.clone());

    let tmp = TempDir::new().unwrap();
    let cache = CrateCache::new(tmp.path()).unwrap();
    let mut indexer = Indexer::open(tmp.path()).unwrap();

    // Step 1: ingest from `latest` — exercises metadata fetch, version
    // resolution, tarball download, checksum verification, extraction,
    // walking, and cache storage.
    let report = ingestion::ingest(
        &fetcher,
        &api,
        &cache,
        &mut indexer,
        &CrateRef::parse("demo").unwrap(),
        IngestionOptions::default(),
    )
    .await
    .unwrap();
    assert!(matches!(report, IngestionReport::Fetched { .. }));
    let coll = report.collection().clone();
    assert_eq!(coll.crate_name, "demo");
    assert_eq!(coll.version, semver::Version::new(1, 0, 0));
    assert!(
        report.item_count() >= 5,
        "expected at least 5 items, got {}",
        report.item_count()
    );

    // Step 2: load items from the cache.
    let items = cache.load(&coll).unwrap();

    // Step 3: search hits the function we documented as "Greet".
    let hits = search::search(&items, "greet", &SearchOptions::default());
    assert!(!hits.is_empty());
    assert_eq!(hits[0].item.qualified_path, "demo::greet");

    // Step 4: search the nested module finds the trait.
    let hits = search::search(
        &items,
        "ping",
        &SearchOptions {
            kind: Some(RustItemKind::Trait),
            ..Default::default()
        },
    );
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].item.qualified_path, "demo::nested::Pingable");

    // Step 5: get by exact qualified path.
    let item = search::get(&items, "demo::Holder").unwrap();
    assert_eq!(item.kind, RustItemKind::Struct);

    // Step 6: list with kind filter returns alphabetical order.
    let listed = search::list(
        &items,
        &SearchOptions {
            kind: Some(RustItemKind::Const),
            ..Default::default()
        },
    );
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].qualified_path, "demo::nested::MAX_PINGS");

    // Step 7: a second ingest is idempotent.
    let report2 = ingestion::ingest(
        &fetcher,
        &api,
        &cache,
        &mut indexer,
        &CrateRef::parse("demo").unwrap(),
        IngestionOptions::default(),
    )
    .await
    .unwrap();
    assert!(matches!(report2, IngestionReport::AlreadyCached { .. }));

    // Step 8: latest resolution was recorded.
    let resolved = cache.resolved("demo", "latest").unwrap().unwrap();
    assert_eq!(resolved.resolved_version, semver::Version::new(1, 0, 0));

    // Step 9: eviction cleans up.
    cache.remove(&coll).unwrap();
    assert!(!cache.has(&coll));
    assert!(cache.entries().unwrap().is_empty());
}

#[tokio::test]
async fn yanked_version_is_flagged_in_report() {
    let tarball =
        build_tarball("demo-2.0.0", &[("src/lib.rs", b"pub fn x() {}")]);
    let checksum = sha256_hex(&tarball);

    // Both 2.0.0 (yanked) and 1.0.0 (good). Latest resolution should
    // pick 1.0.0; explicit @2.0.0 should fetch but mark yanked.
    let payload =
        metadata(&[("2.0.0", &checksum, true), ("1.0.0", &checksum, false)]);

    let fetcher = FakeFetcher::new()
        .with_bytes("https://crates.io/api/v1/crates/demo", payload)
        .with_bytes(
            "https://crates.io/api/v1/crates/demo/2.0.0/download",
            tarball,
        );
    let api = CratesIoApi::new(fetcher.clone());
    let tmp = TempDir::new().unwrap();
    let cache = CrateCache::new(tmp.path()).unwrap();
    let mut indexer = Indexer::open(tmp.path()).unwrap();

    let report = ingestion::ingest(
        &fetcher,
        &api,
        &cache,
        &mut indexer,
        &CrateRef::parse("demo@2.0.0").unwrap(),
        IngestionOptions::default(),
    )
    .await
    .unwrap();
    let IngestionReport::Fetched { was_yanked, .. } = report else {
        panic!("expected Fetched")
    };
    assert!(was_yanked);
}
