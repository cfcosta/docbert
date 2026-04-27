//! `rustbert` CLI entry point.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use rustbert::{
    cache::{CacheStatus, CrateCache},
    crate_ref::CrateRef,
    crates_io::CratesIoApi,
    data_dir,
    error::{Error, Result},
    fetcher::Fetcher,
    ingestion::{self, IngestionOptions, IngestionReport},
    item::{RustItem, RustItemKind},
    reqwest_fetcher::ReqwestFetcher,
    resolver,
    search::{self, SearchOptions},
    sync as sync_mod,
};

#[derive(Parser, Debug)]
#[command(name = "rustbert", version, about = "Rust crate docs lookup")]
struct Cli {
    /// Override the data directory (defaults to $RUSTBERT_DATA_DIR or
    /// $XDG_DATA_HOME/rustbert).
    #[arg(long, env = "RUSTBERT_DATA_DIR", global = true)]
    data_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Pre-warm the cache for one crate.
    Fetch {
        /// Crate spec: `name`, `name@version`, `name@^1.0`, `name@latest`.
        spec: String,
        /// Force a re-fetch even if cached.
        #[arg(long)]
        force: bool,
    },

    /// Search a crate's items by query.
    Search {
        /// Crate spec.
        spec: String,
        /// Query terms.
        query: Vec<String>,
        /// Filter to one item kind.
        #[arg(long)]
        kind: Option<String>,
        /// Filter to a module prefix.
        #[arg(long)]
        module: Option<String>,
        /// Maximum results.
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },

    /// Print one item by qualified path.
    Get {
        /// Crate spec.
        spec: String,
        /// Fully qualified path, e.g. `serde::Serializer::serialize_struct`.
        path: String,
    },

    /// List items in a crate.
    List {
        /// Crate spec.
        spec: String,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        module: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },

    /// Cache state report.
    Status {
        /// Optional crate name to filter on.
        crate_name: Option<String>,
    },

    /// Drop a cached entry.
    Evict {
        /// Crate spec to evict, or `--all` to clear everything.
        spec: Option<String>,
        #[arg(long)]
        all: bool,
    },

    /// Walk a project's Cargo.lock and pre-fetch every crates.io
    /// dependency in parallel.
    Sync {
        /// Path to a specific Cargo.lock (default: walk up from cwd).
        #[arg(long)]
        lock: Option<PathBuf>,
        /// Concurrency for parallel fetches.
        #[arg(long, default_value_t = 4)]
        jobs: usize,
        /// Re-fetch even if cached.
        #[arg(long)]
        force: bool,
        /// Show the plan without fetching.
        #[arg(long)]
        dry_run: bool,
        /// Skip crates matching this glob (repeatable).
        #[arg(long)]
        exclude: Vec<String>,
        /// Skip ColBERT embedding and the PLAID rebuild. Lexical
        /// search still works; semantic ranking is unavailable.
        #[arg(long)]
        no_embed: bool,
    },

    /// Re-resolve cached `latest`/semver-pattern entries against
    /// upstream. Does not re-download; use `sync --force` for that.
    Refresh {
        /// Optional crate name to refresh (default: all).
        crate_name: Option<String>,
        /// Only refresh entries older than this many seconds (default:
        /// always refresh).
        #[arg(long)]
        older_than: Option<u64>,
    },

    /// Index a local Cargo project's source as a synthetic
    /// collection. Subsequent `search`/`get`/`list` calls work
    /// against the local project the same way they do against
    /// fetched crates.
    Index {
        /// Path to the Cargo project root (default: current dir).
        path: Option<PathBuf>,
    },

    /// Run the MCP server on stdio.
    Mcp,
}

fn main() -> std::process::ExitCode {
    init_tracing();
    // syn's recursive parser blows tokio's default 2 MB worker stack on
    // crates with deeply nested types or macro expansions. Bump every
    // worker to 8 MB (matching the OS-thread default) so realistic
    // crates parse cleanly under sync's parallel runner.
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(8 * 1024 * 1024)
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("rustbert: tokio runtime: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match runtime.block_on(real_main()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rustbert: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

async fn real_main() -> Result<()> {
    let cli = Cli::parse();
    let data_dir = resolve_data_dir(cli.data_dir)?;
    data_dir::ensure_layout(&data_dir)?;
    let cache = CrateCache::new(&data_dir)?;

    match cli.command {
        Command::Fetch { spec, force } => cmd_fetch(&cache, &spec, force).await,
        Command::Search {
            spec,
            query,
            kind,
            module,
            limit,
        } => {
            cmd_search(&cache, &spec, &query.join(" "), kind, module, limit)
                .await
        }
        Command::Get { spec, path } => cmd_get(&cache, &spec, &path).await,
        Command::List {
            spec,
            kind,
            module,
            limit,
        } => cmd_list(&cache, &spec, kind, module, limit).await,
        Command::Status { crate_name } => cmd_status(&cache, crate_name),
        Command::Evict { spec, all } => cmd_evict(&cache, spec, all),
        Command::Sync {
            lock,
            jobs,
            force,
            dry_run,
            exclude,
            no_embed,
        } => {
            cmd_sync(&cache, lock, jobs, force, dry_run, exclude, no_embed)
                .await
        }
        Command::Refresh {
            crate_name,
            older_than,
        } => cmd_refresh(&cache, crate_name, older_than).await,
        Command::Index { path } => cmd_index(&cache, path),
        Command::Mcp => rustbert::mcp::serve(cache.clone()).await,
    }
}

fn cmd_index(cache: &CrateCache, path: Option<PathBuf>) -> Result<()> {
    let project_root = match path {
        Some(p) => p,
        None => std::env::current_dir()?,
    };
    let indexer = rustbert::indexer::Indexer::open(cache.data_dir())?;

    if rustbert::host_project::is_workspace_root(&project_root) {
        let outcomes = rustbert::host_project::index_workspace(
            &project_root,
            cache,
            &indexer,
        )?;
        let mut succeeded = 0usize;
        let mut failed = 0usize;
        let mut total_items = 0usize;
        for outcome in &outcomes {
            match &outcome.result {
                Ok(member) => {
                    println!(
                        "  ✓ {name}@{version}  {items} items ({failures} failures)",
                        name = member.collection.crate_name,
                        version = member.collection.version,
                        items = member.item_count,
                        failures = member.failure_count,
                    );
                    succeeded += 1;
                    total_items += member.item_count;
                }
                Err(e) => {
                    eprintln!("  ✗ {}: {}", outcome.path.display(), e);
                    failed += 1;
                }
            }
        }
        println!(
            "\nindexed {succeeded} workspace members ({total_items} items total, {failed} failed)",
        );
        return Ok(());
    }

    let (coll, items, failures) =
        rustbert::host_project::index_project(&project_root, cache, &indexer)?;
    println!(
        "indexed {} items from {}@{} ({} failures)",
        items, coll.crate_name, coll.version, failures,
    );
    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_env("RUSTBERT_LOG")
        .unwrap_or_else(|_| EnvFilter::new("warn,rustbert=info"));
    let _ = fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

fn resolve_data_dir(cli_override: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(d) = cli_override {
        return Ok(d);
    }
    data_dir::data_dir()
}

fn parse_kind(s: Option<String>) -> Result<Option<RustItemKind>> {
    match s {
        None => Ok(None),
        Some(s) => RustItemKind::parse(&s).map(Some).ok_or_else(|| {
            Error::InvalidCrateRef(format!("unknown kind: {s}"))
        }),
    }
}

async fn cmd_fetch(cache: &CrateCache, spec: &str, force: bool) -> Result<()> {
    let crate_ref = CrateRef::parse(spec)?;
    let fetcher = ReqwestFetcher::new()?;
    let api = CratesIoApi::new(fetcher.clone());
    let indexer = rustbert::indexer::Indexer::open(cache.data_dir())?;
    let report = ingestion::ingest(
        &fetcher,
        &api,
        cache,
        &indexer,
        &crate_ref,
        IngestionOptions { force },
    )
    .await?;
    print_report(&report);
    Ok(())
}

async fn ensure_cached<F: Fetcher + Clone>(
    cache: &CrateCache,
    indexer: &rustbert::indexer::Indexer,
    fetcher: &F,
    api: &CratesIoApi<F>,
    crate_ref: &CrateRef,
) -> Result<rustbert::collection::SyntheticCollection> {
    let report = ingestion::ingest(
        fetcher,
        api,
        cache,
        indexer,
        crate_ref,
        IngestionOptions::default(),
    )
    .await?;
    Ok(report.collection().clone())
}

async fn cmd_search(
    cache: &CrateCache,
    spec: &str,
    query: &str,
    kind: Option<String>,
    module: Option<String>,
    limit: usize,
) -> Result<()> {
    let crate_ref = CrateRef::parse(spec)?;
    let fetcher = ReqwestFetcher::new()?;
    let api = CratesIoApi::new(fetcher.clone());
    let mut indexer = rustbert::indexer::Indexer::open(cache.data_dir())?;
    let coll =
        ensure_cached(cache, &indexer, &fetcher, &api, &crate_ref).await?;

    // BM25 + (when available) ColBERT hybrid via the docbert-core
    // stack. Kind / module filters are applied post-rank against the
    // cached items.
    let kind_filter = parse_kind(kind)?;
    let module_filter = module;
    let params = docbert_core::search::SearchParams {
        query: query.to_string(),
        count: limit * 4, // overfetch to leave room for post-filtering
        collection: Some(coll.to_string()),
        min_score: 0.0,
        bm25_only: false,
        no_fuzzy: false,
        all: false,
    };
    let results = indexer.search(params)?;
    let items = cache.load(&coll)?;

    let mut shown = 0;
    for r in results {
        let Some(item) = items.iter().find(|i| i.qualified_path == r.title)
        else {
            continue;
        };
        if let Some(k) = kind_filter
            && item.kind != k
        {
            continue;
        }
        if let Some(prefix) = &module_filter
            && !item.qualified_path.starts_with(prefix)
        {
            continue;
        }
        print_item_one_line(item, Some((r.score * 1000.0) as u32));
        shown += 1;
        if shown >= limit {
            break;
        }
    }
    if shown == 0 {
        println!("(no matches)");
    }
    Ok(())
}

async fn cmd_get(cache: &CrateCache, spec: &str, path: &str) -> Result<()> {
    let crate_ref = CrateRef::parse(spec)?;
    let fetcher = ReqwestFetcher::new()?;
    let api = CratesIoApi::new(fetcher.clone());
    let indexer = rustbert::indexer::Indexer::open(cache.data_dir())?;
    let coll =
        ensure_cached(cache, &indexer, &fetcher, &api, &crate_ref).await?;
    let items = cache.load(&coll)?;
    let item =
        search::get(&items, path).ok_or_else(|| Error::NoMatchingVersion {
            name: format!("item not found: {path}"),
            spec: coll.to_string(),
        })?;
    print_item_full(item);
    Ok(())
}

async fn cmd_list(
    cache: &CrateCache,
    spec: &str,
    kind: Option<String>,
    module: Option<String>,
    limit: usize,
) -> Result<()> {
    let crate_ref = CrateRef::parse(spec)?;
    let fetcher = ReqwestFetcher::new()?;
    let api = CratesIoApi::new(fetcher.clone());
    let indexer = rustbert::indexer::Indexer::open(cache.data_dir())?;
    let coll =
        ensure_cached(cache, &indexer, &fetcher, &api, &crate_ref).await?;
    let items = cache.load(&coll)?;
    let opts = SearchOptions {
        kind: parse_kind(kind)?,
        module_prefix: module,
        limit: Some(limit),
    };
    let listed = search::list(&items, &opts);
    for item in listed {
        print_item_one_line(item, None);
    }
    Ok(())
}

fn cmd_status(cache: &CrateCache, name_filter: Option<String>) -> Result<()> {
    let entries = cache.entries()?;
    let filtered: Vec<_> = match name_filter.as_deref() {
        Some(name) => entries
            .into_iter()
            .filter(|e| e.crate_name == name)
            .collect(),
        None => entries,
    };
    if filtered.is_empty() {
        println!("(no cached crates)");
        return Ok(());
    }
    for e in filtered {
        let status = match e.status {
            CacheStatus::Ready => "ready",
            CacheStatus::Failed => "failed",
        };
        println!(
            "{name}@{version}\t{count} items\t{status}\t fetched_at={ts}",
            name = e.crate_name,
            version = e.version,
            count = e.item_count,
            ts = e.fetched_at,
        );
    }
    Ok(())
}

fn cmd_evict(
    cache: &CrateCache,
    spec: Option<String>,
    all: bool,
) -> Result<()> {
    let indexer = rustbert::indexer::Indexer::open(cache.data_dir())?;

    if all {
        for e in cache.entries()? {
            let coll = rustbert::collection::SyntheticCollection {
                crate_name: e.crate_name,
                version: e.version,
            };
            indexer.remove_collection(&coll)?;
            cache.remove(&coll)?;
        }
        println!("evicted everything");
        return Ok(());
    }
    let spec = spec.ok_or_else(|| {
        Error::InvalidCrateRef("evict requires a <spec> or `--all`".to_string())
    })?;
    let crate_ref = CrateRef::parse(&spec)?;
    let entries = cache.entries()?;
    let mut removed = 0;
    for e in entries {
        if e.crate_name != crate_ref.name {
            continue;
        }
        match &crate_ref.version {
            rustbert::crate_ref::VersionSpec::Concrete(v)
                if &e.version != v =>
            {
                continue;
            }
            _ => {}
        }
        let coll = rustbert::collection::SyntheticCollection {
            crate_name: e.crate_name.clone(),
            version: e.version.clone(),
        };
        indexer.remove_collection(&coll)?;
        cache.remove(&coll)?;
        removed += 1;
    }
    println!("evicted {removed} entries");
    Ok(())
}

async fn cmd_sync(
    cache: &CrateCache,
    lock: Option<PathBuf>,
    jobs: usize,
    force: bool,
    dry_run: bool,
    exclude: Vec<String>,
    no_embed: bool,
) -> Result<()> {
    let lockfile = match lock {
        Some(p) => p,
        None => sync_mod::discover_lockfile(&std::env::current_dir()?)?,
    };
    println!("using lockfile: {}", lockfile.display());

    // Index the host project too. The lockfile's parent is either a
    // workspace root or a single-package root; either way the user
    // expects `sync` to leave their own crates searchable alongside
    // the deps.
    if !dry_run && let Some(project_root) = lockfile.parent() {
        let host_indexer = rustbert::indexer::Indexer::open(cache.data_dir())?;
        if rustbert::host_project::is_workspace_root(project_root) {
            println!("indexing host workspace members:");
            let outcomes = rustbert::host_project::index_workspace(
                project_root,
                cache,
                &host_indexer,
            )?;
            for o in &outcomes {
                match &o.result {
                    Ok(m) => println!(
                        "  ✓ {}@{}  {} items",
                        m.collection.crate_name,
                        m.collection.version,
                        m.item_count
                    ),
                    Err(e) => eprintln!("  ✗ {}: {}", o.path.display(), e),
                }
            }
        } else if project_root.join("Cargo.toml").is_file() {
            match rustbert::host_project::index_project(
                project_root,
                cache,
                &host_indexer,
            ) {
                Ok((coll, items, _)) => println!(
                    "indexed host project {}@{}  {} items",
                    coll.crate_name, coll.version, items
                ),
                Err(e) => eprintln!("host project skipped: {e}"),
            }
        }
    }

    let text = std::fs::read_to_string(&lockfile)?;
    let opts = sync_mod::SyncOptions {
        force,
        jobs,
        exclude,
        dry_run,
        no_embed,
    };
    let plan = sync_mod::build_plan(&text, cache, &opts)?;
    println!(
        "{} crates queued, {} skipped",
        plan.queued.len(),
        plan.skipped.len(),
    );
    for (name, version, reason) in &plan.skipped {
        let r = match reason {
            sync_mod::SkipReason::AlreadyCached => "already cached",
            sync_mod::SkipReason::NotOnCratesIo => "not on crates.io",
            sync_mod::SkipReason::Excluded => "excluded",
        };
        println!("  - {name}@{version} ({r})");
    }
    if dry_run {
        println!("(dry run; no fetches performed)");
        return Ok(());
    }
    if plan.queued.is_empty() {
        println!("nothing to do");
        return Ok(());
    }

    let fetcher = ReqwestFetcher::new()?;
    let api = CratesIoApi::new(fetcher.clone());
    let mut indexer = rustbert::indexer::Indexer::open(cache.data_dir())?;
    let outcome = sync_mod::execute_plan(
        plan,
        fetcher,
        api,
        cache.clone(),
        &mut indexer,
        &opts,
    )
    .await?;
    println!(
        "\ndone: {} succeeded, {} failed",
        outcome.successes.len(),
        outcome.failures.len(),
    );
    for (name, version, items) in &outcome.successes {
        println!("  ✓ {name}@{version}  {items} items");
    }
    if !outcome.failures.is_empty() {
        eprintln!();
        for (name, version, error) in &outcome.failures {
            eprintln!("  ✗ {name}@{version}: {error}");
        }
    }
    Ok(())
}

async fn cmd_refresh(
    cache: &CrateCache,
    crate_filter: Option<String>,
    older_than: Option<u64>,
) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Walk the registry's resolved-version table and re-resolve each
    // non-concrete request.
    let entries = cache.entries()?;
    let mut checked = 0usize;
    let mut changed = 0usize;

    for entry in entries {
        if let Some(filter) = &crate_filter
            && &entry.crate_name != filter
        {
            continue;
        }
        // We re-resolve via the latest record only; concrete versions
        // never refresh.
        let Some(resolved) = cache.resolved(&entry.crate_name, "latest")?
        else {
            continue;
        };
        if let Some(threshold) = older_than {
            let age = now.saturating_sub(resolved.resolved_at);
            if age < threshold {
                continue;
            }
        }
        checked += 1;

        let fetcher = ReqwestFetcher::new()?;
        let api = CratesIoApi::new(fetcher);
        let metadata = api.crate_metadata(&entry.crate_name).await?;
        let new_resolution = resolver::resolve(
            &rustbert::crate_ref::VersionSpec::Latest,
            &metadata,
        )?;
        if new_resolution.version != resolved.resolved_version {
            cache.record_resolved(
                &entry.crate_name,
                "latest",
                &new_resolution.version,
            )?;
            println!(
                "{}@latest: {} → {}",
                entry.crate_name,
                resolved.resolved_version,
                new_resolution.version,
            );
            changed += 1;
        } else {
            println!(
                "{}@latest: {} (unchanged)",
                entry.crate_name, resolved.resolved_version,
            );
        }
    }
    println!("\nchecked {checked} entries, {changed} updated");
    Ok(())
}

fn print_report(report: &IngestionReport) {
    let coll = report.collection();
    match report {
        IngestionReport::AlreadyCached { item_count, .. } => {
            println!(
                "{}@{} already cached ({} items)",
                coll.crate_name, coll.version, item_count
            );
        }
        IngestionReport::Fetched {
            item_count,
            load_failures,
            was_yanked,
            ..
        } => {
            let yank = if *was_yanked { " (yanked!)" } else { "" };
            println!(
                "fetched {}@{}: {} items{yank}",
                coll.crate_name, coll.version, item_count
            );
            if !load_failures.is_empty() {
                eprintln!("{} files failed to parse:", load_failures.len());
                for f in load_failures {
                    eprintln!("  {f}");
                }
            }
        }
    }
}

fn print_item_one_line(item: &RustItem, score: Option<u32>) {
    let score = match score {
        Some(s) => format!("[{s:>3}] "),
        None => String::new(),
    };
    println!(
        "{score}{kind:<8} {path}  ({file}:{start}-{end})",
        kind = item.kind.as_str(),
        path = item.qualified_path,
        file = item.source_file.display(),
        start = item.line_start,
        end = item.line_end,
    );
}

fn print_item_full(item: &RustItem) {
    println!(
        "{kind} {path}",
        kind = item.kind.as_str(),
        path = item.qualified_path
    );
    println!("  {}", item.signature);
    println!(
        "  source: {file}:{start}-{end}",
        file = item.source_file.display(),
        start = item.line_start,
        end = item.line_end,
    );
    println!("  visibility: {}", item.visibility.as_str());
    if !item.attrs.is_empty() {
        println!("  attrs: {}", item.attrs.join(" "));
    }
    if !item.doc_markdown.is_empty() {
        println!();
        for line in item.doc_markdown.lines() {
            println!("    {line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_fetch() {
        let cli =
            Cli::try_parse_from(["rustbert", "fetch", "serde@1.0.0"]).unwrap();
        let Command::Fetch { spec, force } = cli.command else {
            panic!("expected fetch")
        };
        assert_eq!(spec, "serde@1.0.0");
        assert!(!force);
    }

    #[test]
    fn cli_parses_fetch_with_force() {
        let cli =
            Cli::try_parse_from(["rustbert", "fetch", "serde", "--force"])
                .unwrap();
        let Command::Fetch { force, .. } = cli.command else {
            panic!()
        };
        assert!(force);
    }

    #[test]
    fn cli_parses_search_with_options() {
        let cli = Cli::try_parse_from([
            "rustbert",
            "search",
            "serde",
            "Serializer",
            "trait",
            "--kind",
            "trait",
            "--module",
            "serde::ser",
            "--limit",
            "20",
        ])
        .unwrap();
        let Command::Search {
            spec,
            query,
            kind,
            module,
            limit,
        } = cli.command
        else {
            panic!()
        };
        assert_eq!(spec, "serde");
        assert_eq!(query, vec!["Serializer", "trait"]);
        assert_eq!(kind.as_deref(), Some("trait"));
        assert_eq!(module.as_deref(), Some("serde::ser"));
        assert_eq!(limit, 20);
    }

    #[test]
    fn cli_parses_evict_all() {
        let cli = Cli::try_parse_from(["rustbert", "evict", "--all"]).unwrap();
        let Command::Evict { all, spec } = cli.command else {
            panic!()
        };
        assert!(all);
        assert!(spec.is_none());
    }

    #[test]
    fn cli_parses_status_no_filter() {
        let cli = Cli::try_parse_from(["rustbert", "status"]).unwrap();
        let Command::Status { crate_name } = cli.command else {
            panic!()
        };
        assert!(crate_name.is_none());
    }

    #[test]
    fn cli_top_level_data_dir_override_works() {
        let cli = Cli::try_parse_from([
            "rustbert",
            "--data-dir",
            "/tmp/rustbert-test",
            "status",
        ])
        .unwrap();
        assert_eq!(cli.data_dir, Some(PathBuf::from("/tmp/rustbert-test")));
    }

    #[test]
    fn parse_kind_validates() {
        assert_eq!(parse_kind(None).unwrap(), None);
        assert_eq!(
            parse_kind(Some("fn".into())).unwrap(),
            Some(RustItemKind::Fn)
        );
        assert!(parse_kind(Some("nonsense".into())).is_err());
    }
}
