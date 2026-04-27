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
    search::{self, SearchOptions},
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
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    init_tracing();
    match real_main().await {
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
    }
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
    let report = ingestion::ingest(
        &fetcher,
        &api,
        cache,
        &crate_ref,
        IngestionOptions { force },
    )
    .await?;
    print_report(&report);
    Ok(())
}

async fn ensure_cached<F: Fetcher + Clone>(
    cache: &CrateCache,
    fetcher: &F,
    api: &CratesIoApi<F>,
    crate_ref: &CrateRef,
) -> Result<rustbert::collection::SyntheticCollection> {
    let report = ingestion::ingest(
        fetcher,
        api,
        cache,
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
    let coll = ensure_cached(cache, &fetcher, &api, &crate_ref).await?;
    let items = cache.load(&coll)?;

    let opts = SearchOptions {
        kind: parse_kind(kind)?,
        module_prefix: module,
        limit: Some(limit),
    };
    let hits = search::search(&items, query, &opts);
    if hits.is_empty() {
        println!("(no matches)");
        return Ok(());
    }
    for hit in hits {
        print_item_one_line(hit.item, Some(hit.score));
    }
    Ok(())
}

async fn cmd_get(cache: &CrateCache, spec: &str, path: &str) -> Result<()> {
    let crate_ref = CrateRef::parse(spec)?;
    let fetcher = ReqwestFetcher::new()?;
    let api = CratesIoApi::new(fetcher.clone());
    let coll = ensure_cached(cache, &fetcher, &api, &crate_ref).await?;
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
    let coll = ensure_cached(cache, &fetcher, &api, &crate_ref).await?;
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
    if all {
        for e in cache.entries()? {
            let coll = rustbert::collection::SyntheticCollection {
                crate_name: e.crate_name,
                version: e.version,
            };
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
        cache.remove(&coll)?;
        removed += 1;
    }
    println!("evicted {removed} entries");
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
