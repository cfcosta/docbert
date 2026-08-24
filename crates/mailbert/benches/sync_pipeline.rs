//! The pipeline of a sync, measured against the fake server. (§10.5)
//!
//! The sync of today runs its stages one after the other. It downloads
//! every folder, then it writes the index, then it plans the model, and
//! then it embeds. This bench gives each stage its own number, so the
//! work of §10.5 can show which stage it moved.
//!
//! Two shapes of mailbox run, because the two accounts of a real
//! configuration have a different shape:
//!
//! * `one_folder` holds every message in one folder. The work of a sync
//!   splits by folder, so one connection carries the whole download and
//!   the other connections stay idle. This is the shape of Gmail.
//! * `many_folders` holds the same messages across eight folders. Eight
//!   connections read at the same time, and they meet at the one writer
//!   of the store. This is the shape of Fastmail.
//!
//! No bench loads a model. The `give` seam of [`semantic::run`] takes a
//! stub that counts the passages and writes nothing, so the bench
//! measures the store and the plan alone.
//!
//! `MAILBERT_BENCH_MESSAGES` and `MAILBERT_BENCH_SIZE` change how much
//! mail the fake server holds.

use std::{
    hint::black_box,
    sync::{Arc, OnceLock},
};

use criterion::{
    BatchSize,
    BenchmarkId,
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
use docbert_core::EmbeddingDb;
use mailbert::{
    pass,
    semantic,
    sync::{self, How},
};
use mailbert_core::{Store, config::Account, index::MailIndex};
use mailbert_imap::{
    Pool,
    Server,
    fake::{FakeFolder, FakeServer, Plan as Mailbox},
};
use tempfile::TempDir;
use tokio::runtime::Runtime;

/// How many messages the mailbox of a bench holds.
const MESSAGES: u32 = 500;

/// About how many bytes one message takes. A message of a real mailbox
/// is larger, but a bench that holds 500 of them must still run fast.
const SIZE: usize = 4096;

/// How many folders the mailbox of the `many_folders` shape holds.
const FOLDERS: u32 = 8;

/// How many connections one account opens.
const CONNECTIONS: usize = 8;

/// The model that a plan names. No bench loads it, because the plan
/// only reads the name to make its fingerprint. (§6.2)
const MODEL: &str = "lightonai/GTE-ModernColBERT-v1";

/// The number that an environment variable holds, or `fallback`.
fn from_env<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(fallback)
}

/// Everything that one run of the pipeline reads and writes.
///
/// The directory comes last, because it must drop after the store and
/// after the databases that hold its files open.
struct Bed {
    /// How many messages the server holds. Every stage must meet it.
    count: u32,
    store: Arc<Store>,
    index: MailIndex,
    db: &'static EmbeddingDb,
    account: Account,
    pool: Arc<Pool>,
    _server: FakeServer,
    _dir: TempDir,
}

/// One database of embeddings for the whole run.
///
/// The stub of the model writes no vector, so this database stays
/// empty, and no measurement reads it. One database for each iteration
/// leaks an LMDB environment, because docbert keeps every environment
/// that it opens, and a long run then has no address space left.
fn shared_db() -> &'static EmbeddingDb {
    static DB: OnceLock<EmbeddingDb> = OnceLock::new();

    DB.get_or_init(|| {
        let dir = tempfile::tempdir().expect("a directory");
        let db = EmbeddingDb::open(&dir.path().join("embeddings.db"))
            .expect("a database of embeddings");

        // The environment maps the file, and the run needs it to the
        // end, so the directory must outlive this call.
        std::mem::forget(dir);

        db
    })
}

/// A store with no message, and a server that holds `count` of them.
///
/// `folders` says across how many folders the mail spreads. The last
/// folder takes the remainder, so the server always holds `count`.
async fn fresh(folders: u32, count: u32, size: usize) -> Bed {
    let dir = tempfile::tempdir().expect("a directory");
    let store = Arc::new(Store::open(dir.path()).expect("a store"));
    let index = MailIndex::open_in_ram().expect("an index");
    let db = shared_db();

    // Each folder takes its own range of UIDs, so no two folders hold
    // one message. The store keeps one entry for one set of bytes, and
    // a mailbox of shared mail would measure less work than it says.
    let mut plan = Mailbox::new();
    let mut first = 1;

    for number in 1..=folders {
        let share = match number == folders {
            true => count + 1 - first,
            false => count / folders,
        };
        plan = plan.with(FakeFolder::filled_from(
            &format!("INBOX{number}"),
            first,
            share,
            size,
        ));
        first += share;
    }

    let server = FakeServer::start(plan).await.expect("a server");
    let port = server.port();
    let names: Vec<String> = (1..=folders)
        .map(|number| format!("INBOX{number}"))
        .collect();

    // The fake server speaks no TLS, so the pool of a bench cannot come
    // from `sync::server`. Only a bench opens a connection like this.
    let pool = Arc::new(Pool::new(
        Server::at("127.0.0.1", port, false)
            .with_login("me", "secret")
            .with_connections(CONNECTIONS),
    ));

    // A stage measures the work of one whole mailbox. A store that
    // still held the mail of the run before would measure nothing.
    assert!(
        store.all().expect("the store reads").is_empty(),
        "a run started from a store that already held mail"
    );

    Bed {
        count,
        store,
        index,
        db,
        account: Account {
            name: "bench".to_string(),
            host: "127.0.0.1".to_string(),
            user: "me".to_string(),
            port,
            password_command: None,
            password_file: None,
            password: Some("secret".to_string()),
            folders: names,
            exclude: Vec::new(),
            footers: Vec::new(),
            all_folders: false,
            connections: CONNECTIONS,
        },
        pool,
        _server: server,
        _dir: dir,
    }
}

/// Fetch every message, parse it, and write it to the store. (§3.4)
async fn download(bed: &Bed) -> usize {
    let report = sync::one(
        Arc::clone(&bed.store),
        Arc::clone(&bed.pool),
        &bed.account,
        How::default(),
    )
    .await
    .expect("the sync reads the fake server");

    // A sync that kept nothing still reports a time. A folder name that
    // the server does not know gives that empty run, and the number of
    // the bench would then mean nothing.
    assert_eq!(
        report.counts.kept,
        u64::from(bed.count),
        "the download did not keep the mail of the server"
    );

    report.touched.len()
}

/// Thread the store, and write what the index is behind on. (§6.1)
fn index_pass(bed: &Bed) -> usize {
    let touched = bed.store.all().expect("the store reads");
    let touched = touched.iter().map(|one| one.id).collect();

    let wrote = pass::after_sync(&bed.store, &bed.index, &touched)
        .expect("the index takes the mail");

    assert_eq!(
        wrote.messages, bed.count as usize,
        "the index pass wrote fewer messages than the store holds"
    );

    wrote.messages
}

/// Read every passage that the model has not seen. (§6.2)
fn plan(bed: &Bed) -> semantic::Plan {
    let work = semantic::plan(&bed.store, MODEL).expect("the store reads");

    assert!(!work.work.is_empty(), "the plan of a full store is empty");

    work
}

/// Walk the plan, with a stub in the place of the model.
///
/// The stub counts the passages that it took, and gives no vector back,
/// so the bench measures the walk of the plan and never a model.
// The stub gives no error back, so the size of the error of the crate
// never reaches a caller. The lint cannot see that.
#[allow(clippy::result_large_err)]
fn embed(bed: &Bed, plan: &semantic::Plan) -> usize {
    let mut passages = 0;
    let mut give = |batch: Vec<(u64, String)>| {
        passages += batch.len();
        Ok(batch.len())
    };

    semantic::run(&bed.store, bed.db, plan, |_| {}, &mut give)
        .expect("the plan walks");

    assert!(passages > 0, "the walk gave the model no passage");

    passages
}

/// The four stages, each measured on its own, for one shape of mailbox.
fn stages(c: &mut Criterion, shape: &str, folders: u32) {
    let count: u32 = from_env("MAILBERT_BENCH_MESSAGES", MESSAGES);
    let size: usize = from_env("MAILBERT_BENCH_SIZE", SIZE);
    let rt = Runtime::new().expect("a runtime");

    let mut group = c.benchmark_group(format!("sync_pipeline/{shape}"));
    group.throughput(Throughput::Elements(u64::from(count)));
    group.sample_size(10);

    let name = BenchmarkId::from_parameter(format!(
        "messages={count},size={size},folders={folders}"
    ));

    group.bench_with_input(name.clone(), &count, |b, _| {
        b.iter_batched(
            || rt.block_on(fresh(folders, count, size)),
            |bed| rt.block_on(download(black_box(&bed))),
            BatchSize::PerIteration,
        );
    });
    group.finish();

    // The three stages after the download all read a full store, so
    // each one downloads first, outside of what the bench times.
    let mut group = c.benchmark_group(format!("index_pass/{shape}"));
    group.throughput(Throughput::Elements(u64::from(count)));
    group.sample_size(10);
    group.bench_with_input(name.clone(), &count, |b, _| {
        b.iter_batched(
            || {
                let bed = rt.block_on(fresh(folders, count, size));
                rt.block_on(download(&bed));
                bed
            },
            |bed| index_pass(black_box(&bed)),
            BatchSize::PerIteration,
        );
    });
    group.finish();

    let mut group = c.benchmark_group(format!("embed_plan/{shape}"));
    group.throughput(Throughput::Elements(u64::from(count)));
    group.sample_size(10);
    group.bench_with_input(name.clone(), &count, |b, _| {
        b.iter_batched(
            || {
                let bed = rt.block_on(fresh(folders, count, size));
                rt.block_on(download(&bed));
                bed
            },
            |bed| plan(black_box(&bed)),
            BatchSize::PerIteration,
        );
    });
    group.finish();

    let mut group = c.benchmark_group(format!("embed_walk/{shape}"));
    group.throughput(Throughput::Elements(u64::from(count)));
    group.sample_size(10);
    group.bench_with_input(name.clone(), &count, |b, _| {
        b.iter_batched(
            || {
                let bed = rt.block_on(fresh(folders, count, size));
                rt.block_on(download(&bed));
                let work = plan(&bed);
                (bed, work)
            },
            |(bed, work)| embed(black_box(&bed), black_box(&work)),
            BatchSize::PerIteration,
        );
    });
    group.finish();

    // This is the number that §10.5 must move. The stages above say
    // where the time went, and this one says how much of it went.
    let mut group = c.benchmark_group(format!("whole/{shape}"));
    group.throughput(Throughput::Elements(u64::from(count)));
    group.sample_size(10);
    group.bench_with_input(name, &count, |b, _| {
        b.iter_batched(
            || rt.block_on(fresh(folders, count, size)),
            |bed| {
                rt.block_on(download(&bed));
                index_pass(&bed);
                let work = plan(&bed);
                embed(&bed, &work)
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

/// Every message in one folder. One connection does the whole download.
fn one_folder(c: &mut Criterion) {
    stages(c, "one_folder", 1);
}

/// The mail across eight folders. Eight connections read at one time.
fn many_folders(c: &mut Criterion) {
    stages(c, "many_folders", FOLDERS);
}

criterion_group!(benches, one_folder, many_folders);
criterion_main!(benches);
