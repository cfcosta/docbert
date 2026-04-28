//! `rustbert sync` — walk a project's `Cargo.lock` and proactively
//! ingest every crates.io dependency.
//!
//! Three pieces:
//!
//! 1. [`discover_lockfile`] walks up from `cwd` looking for `Cargo.lock`.
//! 2. [`build_plan`] turns a lockfile (string content) into a list of
//!    `(crate, version)` pairs to fetch, with classification reasons
//!    for everything that's filtered out.
//! 3. [`execute_plan`] runs the planned ingestions through the
//!    configured fetcher with bounded concurrency and reports per-
//!    crate outcomes.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use globset::{Glob, GlobSet, GlobSetBuilder};
use tokio::{sync::Semaphore, task::JoinSet};

use crate::{
    cache::CrateCache,
    crate_ref::{CrateRef, VersionSpec},
    crates_io::CratesIoApi,
    error::{Error, Result},
    fetcher::Fetcher,
    indexer::Indexer,
    ingestion::{self, IngestionOptions, IngestionReport},
    lockfile,
};

/// Walk up from `start` looking for `Cargo.lock`.
pub fn discover_lockfile(start: &Path) -> Result<PathBuf> {
    let mut current = if start.is_absolute() {
        start.to_path_buf()
    } else {
        std::env::current_dir()?.join(start)
    };
    loop {
        let candidate = current.join("Cargo.lock");
        if candidate.is_file() {
            return Ok(candidate);
        }
        if !current.pop() {
            break;
        }
    }
    Err(Error::LockfileNotFound)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    NotOnCratesIo,
    AlreadyCached,
    Excluded,
}

#[derive(Debug, Clone)]
pub struct SyncPlan {
    pub queued: Vec<(String, semver::Version)>,
    pub skipped: Vec<(String, semver::Version, SkipReason)>,
}

#[derive(Debug, Clone, Default)]
pub struct SyncOptions {
    pub force: bool,
    pub jobs: usize,
    pub exclude: Vec<String>,
    pub dry_run: bool,
}

impl SyncOptions {
    pub fn jobs(&self) -> usize {
        if self.jobs == 0 { 4 } else { self.jobs }
    }
}

#[derive(Debug, Clone)]
pub struct SyncOutcome {
    pub plan: SyncPlan,
    pub successes: Vec<(String, semver::Version, usize)>,
    pub failures: Vec<(String, semver::Version, String)>,
}

/// Build a [`SyncPlan`] from a lockfile's string contents.
///
/// `cache` is consulted to mark already-cached pairs as skipped, and
/// `options.exclude` glob-matches against `crate_name` and `name@version`.
pub fn build_plan(
    lockfile_text: &str,
    cache: &CrateCache,
    options: &SyncOptions,
) -> Result<SyncPlan> {
    let crates = lockfile::crates_io_packages_from_str(lockfile_text)?;
    let exclude = build_exclude_set(&options.exclude)?;

    let mut plan = SyncPlan {
        queued: Vec::new(),
        skipped: Vec::new(),
    };

    for c in crates {
        let collection = crate::collection::SyntheticCollection {
            crate_name: c.name.clone(),
            version: c.version.clone(),
        };
        if exclude.is_match(&c.name)
            || exclude.is_match(format!("{}@{}", c.name, c.version))
        {
            plan.skipped.push((c.name, c.version, SkipReason::Excluded));
            continue;
        }
        if !options.force && cache.has(&collection) {
            plan.skipped
                .push((c.name, c.version, SkipReason::AlreadyCached));
            continue;
        }
        plan.queued.push((c.name, c.version));
    }
    Ok(plan)
}

fn build_exclude_set(globs: &[String]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for g in globs {
        builder.add(Glob::new(g)?);
    }
    Ok(builder.build()?)
}

/// Execute a plan: fetch each queued crate via the supplied fetcher
/// with bounded concurrency, then sequentially write each successful
/// crate's items into the lexical index, embed them, and rebuild the
/// PLAID index once at the end. Failures don't abort the run.
#[tracing::instrument(skip_all, fields(queued = plan.queued.len(), jobs = options.jobs()))]
pub async fn execute_plan<F>(
    plan: SyncPlan,
    fetcher: F,
    api: CratesIoApi<F>,
    cache: CrateCache,
    indexer: &mut Indexer,
    options: &SyncOptions,
) -> Result<SyncOutcome>
where
    F: Fetcher + Clone + 'static,
{
    if options.dry_run {
        return Ok(SyncOutcome {
            plan,
            successes: Vec::new(),
            failures: Vec::new(),
        });
    }

    let semaphore = Arc::new(Semaphore::new(options.jobs()));
    let mut tasks: JoinSet<(String, semver::Version, Result<IngestionReport>)> =
        JoinSet::new();

    for (name, version) in plan.queued.clone() {
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| Error::Cache(format!("semaphore: {e}")))?;
        let f = fetcher.clone();
        let a = api.clone();
        let c = cache.clone();
        let force = options.force;
        let task_name = name.clone();
        let task_version = version.clone();
        tasks.spawn(async move {
            let _permit = permit;
            let crate_ref = CrateRef {
                name: task_name.clone(),
                version: VersionSpec::Concrete(task_version.clone()),
            };
            let result = ingestion::ingest_to_cache(
                &f,
                &a,
                &c,
                &crate_ref,
                IngestionOptions { force },
            )
            .await;
            (task_name, task_version, result)
        });
    }

    let mut outcome = SyncOutcome {
        plan,
        successes: Vec::new(),
        failures: Vec::new(),
    };
    while let Some(joined) = tasks.join_next().await {
        let (name, version, result) = match joined {
            Ok(triple) => triple,
            Err(e) => {
                outcome.failures.push((
                    "<unknown>".to_string(),
                    semver::Version::new(0, 0, 0),
                    format!("join error: {e}"),
                ));
                continue;
            }
        };
        match result {
            Ok(report) => {
                outcome.successes.push((name, version, report.item_count()));
            }
            Err(e) => {
                outcome.failures.push((name, version, e.to_string()));
            }
        }
    }

    // Sequential index + embed pass over every crate that
    // successfully landed in the cache. Lexical and ColBERT indexing
    // are coupled inside `index_items` — either both succeed or the
    // crate's lexical entries are rolled back. Crates whose indexing
    // fails get reclassified as failures so the report stays honest.
    let mut indexed = 0usize;
    let mut still_successful = Vec::new();
    let mut promoted_failures: Vec<(String, semver::Version, String)> =
        Vec::new();
    for (name, version, items) in std::mem::take(&mut outcome.successes) {
        let coll = crate::collection::SyntheticCollection {
            crate_name: name.clone(),
            version: version.clone(),
        };
        match cache.load(&coll) {
            Ok(loaded) => match indexer.index_items(&coll, &loaded) {
                Ok(_) => {
                    indexed += 1;
                    still_successful.push((name, version, items));
                }
                Err(e) => promoted_failures.push((
                    name,
                    version,
                    format!("index_items: {e}"),
                )),
            },
            Err(e) => promoted_failures.push((
                name,
                version,
                format!("cache.load: {e}"),
            )),
        }
    }
    outcome.successes = still_successful;
    outcome.failures.extend(promoted_failures);

    if indexed > 0
        && let Err(e) = indexer.rebuild_plaid()
    {
        tracing::warn!(
            error = %e,
            "PLAID rebuild failed — search will fall back to BM25-only",
        );
    }
    tracing::info!(
        indexed = indexed,
        successes = outcome.successes.len(),
        failures = outcome.failures.len(),
        "sync complete",
    );
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    fn lockfile_with_one_crates_io_dep() -> String {
        r#"version = 4

[[package]]
name = "demo"
version = "1.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "1111111111111111111111111111111111111111111111111111111111111111"

[[package]]
name = "myapp"
version = "0.1.0"

[[package]]
name = "from-git"
version = "0.2.0"
source = "git+https://github.com/foo/bar?rev=abc#abc123def4567890123456789012345678901234"
"#
        .to_string()
    }

    #[test]
    fn discovers_lockfile_in_cwd() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("Cargo.lock"), "version = 4").unwrap();
        let found = discover_lockfile(tmp.path()).unwrap();
        assert_eq!(found, tmp.path().join("Cargo.lock"));
    }

    #[test]
    fn discovers_lockfile_in_parent() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("Cargo.lock"), "version = 4").unwrap();
        let nested = tmp.path().join("a/b/c");
        fs::create_dir_all(&nested).unwrap();
        let found = discover_lockfile(&nested).unwrap();
        assert_eq!(found, tmp.path().join("Cargo.lock"));
    }

    #[test]
    fn missing_lockfile_errors() {
        let tmp = TempDir::new().unwrap();
        let err = discover_lockfile(tmp.path()).unwrap_err();
        assert!(matches!(err, Error::LockfileNotFound));
    }

    #[test]
    fn plan_queues_only_crates_io_packages() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let plan = build_plan(
            &lockfile_with_one_crates_io_dep(),
            &cache,
            &SyncOptions::default(),
        )
        .unwrap();

        let queued: Vec<_> =
            plan.queued.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(queued, vec!["demo"]);
        // Path/git deps don't appear in skipped because they don't make
        // it through `crates_io_packages_from_str` in the first place.
        // The skipped list is for crates we *could* fetch but won't.
        assert!(plan.skipped.is_empty());
    }

    #[test]
    fn plan_marks_already_cached_crates_as_skipped() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let coll = crate::collection::SyntheticCollection {
            crate_name: "demo".to_string(),
            version: semver::Version::new(1, 0, 0),
        };
        cache.store(&coll, &[]).unwrap();

        let plan = build_plan(
            &lockfile_with_one_crates_io_dep(),
            &cache,
            &SyncOptions::default(),
        )
        .unwrap();

        assert!(plan.queued.is_empty());
        assert_eq!(
            plan.skipped[0],
            (
                "demo".to_string(),
                semver::Version::new(1, 0, 0),
                SkipReason::AlreadyCached
            )
        );
    }

    #[test]
    fn plan_force_includes_already_cached_crates() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let coll = crate::collection::SyntheticCollection {
            crate_name: "demo".to_string(),
            version: semver::Version::new(1, 0, 0),
        };
        cache.store(&coll, &[]).unwrap();

        let opts = SyncOptions {
            force: true,
            ..Default::default()
        };
        let plan =
            build_plan(&lockfile_with_one_crates_io_dep(), &cache, &opts)
                .unwrap();
        assert_eq!(plan.queued.len(), 1);
    }

    #[test]
    fn plan_excludes_glob_matches() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let opts = SyncOptions {
            exclude: vec!["dem*".to_string()],
            ..Default::default()
        };
        let plan =
            build_plan(&lockfile_with_one_crates_io_dep(), &cache, &opts)
                .unwrap();
        assert!(plan.queued.is_empty());
        assert_eq!(plan.skipped[0].2, SkipReason::Excluded);
    }

    #[test]
    fn dry_run_returns_plan_without_fetching() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let plan = build_plan(
            &lockfile_with_one_crates_io_dep(),
            &cache,
            &SyncOptions::default(),
        )
        .unwrap();

        let opts = SyncOptions {
            dry_run: true,
            ..Default::default()
        };
        // (the dry-run path doesn't open the indexer for index_items)
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let mut indexer = Indexer::open(tmp.path()).unwrap();
        let outcome = runtime.block_on(execute_plan(
            plan.clone(),
            crate::FakeFetcher::new(),
            CratesIoApi::new(crate::FakeFetcher::new()),
            cache,
            &mut indexer,
            &opts,
        ));
        let outcome = outcome.unwrap();
        assert!(outcome.successes.is_empty());
        assert!(outcome.failures.is_empty());
        assert_eq!(outcome.plan.queued.len(), 1);
    }
}
