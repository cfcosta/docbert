//! One pass of `sync` over the accounts of the configuration. (§2.1)
//!
//! A sync reads what the server lists, keeps the folders that the
//! account names, and plans one job for each of them. (§3.3) The jobs
//! then run at the same time, one connection each, because a folder is
//! the only unit that a plan can split. (§3.1)
//!
//! The sync writes nothing to the server. It sends `EXAMINE` and never
//! `SELECT`, and it reads a body with `BODY.PEEK[]`, so no message
//! takes a `\Seen` flag that the reader did not give it. (§3)

use std::{
    collections::BTreeSet,
    future::Future,
    io::{self, Write},
    sync::Arc,
    time::{Duration, Instant},
};

use mailbert_core::{
    Listed,
    MessageId,
    Store,
    config::Account,
    index::MailIndex,
};
use mailbert_imap::{
    Backoff,
    FolderState,
    Held,
    Job,
    Pool,
    Server,
    plan,
    resume,
    sync::again,
};
use regex::Regex;
use serde::Serialize;
use tokio::{task::JoinSet, time::sleep};
use tracing::{Instrument, instrument::WithSubscriber};

use crate::{
    Tool,
    cli,
    error::{Error, Result},
    pass::{self, Wrote},
    semantic::{Brain, Embedded},
    settings,
    sink::{Counts, Sink, sync_state},
};

/// How many UIDs one `FETCH` asks for. (§3.2)
pub const BATCH: u32 = 500;

/// How many messages the sweep embeds between two lines of progress.
pub const SAY_EVERY: usize = 500;

/// How one sync runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct How {
    /// Forget the saved state, and read each folder from its first UID.
    pub full: bool,

    /// Say what the sync would ask for, and write nothing.
    pub dry: bool,

    /// How long the sync waits after a connection breaks. (§3.4)
    pub back: Backoff,
}

/// What the sync of one folder did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FolderReport {
    /// The name that the server gave the folder.
    pub folder: String,

    /// How many messages the job asks the server for.
    pub asked: u64,

    /// True when the folder starts again, because `UIDVALIDITY` moved.
    pub restart: bool,
}

/// What the sync of one account did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    /// The name that the configuration gives the account.
    pub account: String,

    /// One line for each folder that the sync opened.
    pub folders: Vec<FolderReport>,

    /// The folders that the account names, and the server does not
    /// hold. A name with a typo lands here. (§2.1)
    pub missing: Vec<String>,

    /// What the sink of this account wrote.
    pub counts: Counts,

    /// The messages that moved, for the threading pass. (§5.5)
    pub touched: BTreeSet<MessageId>,
}

impl Report {
    /// How many messages the sync asked the server for.
    pub fn asked(&self) -> u64 {
        self.folders.iter().map(|folder| folder.asked).sum()
    }
}

/// The server that an account names. (§1.2)
///
/// The connection always takes TLS, because a password goes over it.
pub fn server(account: &Account, password: &str) -> Server {
    Server::at(&account.host, account.port, true)
        .with_login(&account.user, password)
        .with_connections(account.connections)
}

/// Sync one account on the connections of one pool. (§2.1)
///
/// # Errors
///
/// The function fails if the server refuses a command, if a footer
/// pattern is broken, or if the store cannot take a message.
#[tracing::instrument(
    skip_all,
    fields(account = %account.name),
    name = "account"
)]
pub async fn one(
    store: Arc<Store>,
    pool: Arc<Pool>,
    account: &Account,
    how: How,
) -> Result<Report> {
    let start = Instant::now();
    let footers = account.footer_patterns()?;
    let (jobs, missing) = prepare(&store, &pool, account, how).await?;

    let mut report = Report {
        account: account.name.clone(),
        folders: jobs.iter().map(line).collect(),
        missing,
        counts: Counts::default(),
        touched: BTreeSet::new(),
    };

    if how.dry {
        tracing::info!(asked = report.asked(), "a dry run asks for this");

        return Ok(report);
    }

    let (counts, touched) =
        work(&store, &pool, account, jobs, &footers, how.back).await?;
    report.counts = counts;
    report.touched = touched;

    tracing::info!(
        folders = report.folders.len(),
        kept = counts.kept,
        moved = counts.moved,
        gone = counts.gone,
        broken = counts.broken,
        bytes = counts.bytes,
        ms = start.elapsed().as_millis(),
        "the account is done"
    );

    Ok(report)
}

/// What one job says before it runs.
fn line(job: &Job) -> FolderReport {
    FolderReport {
        folder: job.folder.clone(),
        asked: job.count(),
        restart: job.restart,
    }
}

/// The jobs of one account, with a new connection after a break.
///
/// A connection that waited in the pool may be closed already, and the
/// client learns that only when it writes. Every step here reads, and
/// a read runs again without harm. (§3.4)
async fn prepare(
    store: &Store,
    pool: &Pool,
    account: &Account,
    how: How,
) -> Result<(Vec<Job>, Vec<String>)> {
    let mut failures = 0;

    loop {
        let error = match ready(store, pool, account, how).await {
            Ok(ready) => return Ok(ready),
            Err(error) => error,
        };

        failures += 1;
        if failures >= how.back.tries.max(1) || !broken(&error) {
            return Err(error);
        }

        sleep(how.back.wait(failures)).await;
    }
}

/// True when a new connection can do what this error stopped.
fn broken(error: &Error) -> bool {
    matches!(error, Error::Imap(inner) if again(inner))
}

/// One try: what the server lists, and one job for each folder.
async fn ready(
    store: &Store,
    pool: &Pool,
    account: &Account,
    how: How,
) -> Result<(Vec<Job>, Vec<String>)> {
    let mut held = pool.take().await?;

    match look(store, &mut held, account, how).await {
        Ok(ready) => Ok(ready),
        Err(error) => {
            // A connection that broke in the middle of an answer holds
            // bytes that belong to no command. It never goes back.
            held.retire();

            Err(error)
        }
    }
}

/// The folders to sync, and one job for each of them.
///
/// A full sync gives each folder an empty state, so `UIDVALIDITY`
/// differs and the plan asks for every UID again. (§3.3)
async fn look(
    store: &Store,
    held: &mut Held<'_>,
    account: &Account,
    how: How,
) -> Result<(Vec<Job>, Vec<String>)> {
    let listed = held.folders().await?;

    // A folder that holds no mail cannot answer an `EXAMINE`, so it
    // never reaches a plan. (§3.1)
    //
    // The attributes travel with the name, because an `exclude` entry
    // can name an attribute of RFC 6154. (§1.2)
    let available: Vec<Listed> = listed
        .iter()
        .filter(|folder| folder.holds_mail())
        .map(Listed::from)
        .collect();

    let chosen = account.select_folders(&available);
    let mut jobs = Vec::with_capacity(chosen.len());

    tracing::info!(
        listed = listed.len(),
        holding = available.len(),
        chosen = chosen.len(),
        "listed the folders"
    );

    for folder in &chosen {
        let view = held.examine(folder).await?;
        let saved = match how.full {
            true => FolderState::default(),
            false => store
                .state(&account.name, folder)?
                .as_ref()
                .map(sync_state)
                .unwrap_or_default(),
        };

        let job = plan(folder, &saved, &view, BATCH);
        tracing::debug!(
            folder,
            asked = job.count(),
            restart = job.restart,
            "planned a folder"
        );
        jobs.push(job);
    }

    Ok((jobs, account.missing_folders(&available)))
}

/// Run every job, and join what the sinks wrote.
///
/// Each folder takes its own sink and its own connection, so a slow
/// folder never holds up another one. The store takes the writes in
/// the order that they arrive, because one write transaction runs at
/// a time. (§4.2)
async fn work(
    store: &Arc<Store>,
    pool: &Arc<Pool>,
    account: &Account,
    jobs: Vec<Job>,
    footers: &[Regex],
    back: Backoff,
) -> Result<(Counts, BTreeSet<MessageId>)> {
    let mut tasks = JoinSet::new();

    for job in jobs {
        let store = Arc::clone(store);
        let pool = Arc::clone(pool);
        let name = account.name.clone();
        let footers = footers.to_vec();
        let span = tracing::info_span!("folder", folder = %job.folder);

        // The span and the subscriber both travel with the task. A
        // folder runs on its own thread, and neither one crosses a
        // `spawn` on its own. (§10.5)
        tasks.spawn(
            async move {
                let start = Instant::now();
                let asked = job.count();
                let mut sink = Sink::new(store, &name).with_footers(footers);
                let ended = resume(&pool, &job, &mut sink, back).await;
                let counts = sink.counts();

                tracing::info!(
                    asked,
                    kept = counts.kept,
                    moved = counts.moved,
                    gone = counts.gone,
                    broken = counts.broken,
                    bytes = counts.bytes,
                    ms = start.elapsed().as_millis(),
                    ok = ended.is_ok(),
                    "the folder is done"
                );

                (counts, sink.touched().clone(), ended)
            }
            .instrument(span)
            .with_current_subscriber(),
        );
    }

    let mut counts = Counts::default();
    let mut touched = BTreeSet::new();
    let mut broke = None;

    // Every job joins, even after one of them failed, so the mail that
    // the other folders read still reaches the report. (§3.4)
    while let Some(joined) = tasks.join_next().await {
        let (one, wrote, ended) = joined.expect("the sink did not panic");

        counts = counts.and(one);
        touched.extend(wrote);

        if let Err(error) = ended
            && broke.is_none()
        {
            broke = Some(error);
        }
    }

    match broke {
        Some(error) => Err(error.into()),
        None => Ok((counts, touched)),
    }
}

// ---------------------------------------------------------------------
// The watch. (§3.1)
// ---------------------------------------------------------------------

/// How long one `IDLE` holds a connection open, when nothing arrives.
///
/// RFC 2177 tells a client to send the command again before the server
/// stops waiting, and 2 minutes is well inside every limit.
pub const PAUSE: Duration = Duration::from_secs(120);

/// One account, and the connections that reach it.
pub struct Watched<'a> {
    /// The account, as the configuration names it.
    pub account: &'a Account,

    /// The connections of that account. (§3.1)
    pub pool: Arc<Pool>,
}

/// A pool for each account, with the password that it needs. (§1.2)
///
/// # Errors
///
/// The function fails if a password source gives nothing.
pub fn pools<'a>(accounts: &[&'a Account]) -> Result<Vec<Watched<'a>>> {
    accounts
        .iter()
        .map(|account| {
            let password = settings::secret(account)?;

            Ok(Watched {
                account,
                pool: Arc::new(Pool::new(server(account, &password))),
            })
        })
        .collect()
}

/// What one sync writes.
///
/// The store keeps the messages, the index of §6.1 answers `ksearch`,
/// and the brain of §6.2 answers `search`. A sync with no brain writes
/// the first two, and that is what `--dry-run` and a test both do.
pub struct Books<'a> {
    /// The messages, the threads, and the sync state. (§4.2)
    pub store: &'a Arc<Store>,

    /// The lexical index of §6.1.
    pub index: &'a MailIndex,

    /// The model, the embeddings, and the PLAID index of §6.2.
    pub brain: Option<&'a mut Brain>,
}

/// Sync again each time a server says that something changed. (§3.1)
///
/// The first round always runs. Each round after it waits on `IDLE`,
/// so a quiet mailbox costs one open connection and no command.
///
/// With more than one account, the round starts when every wait ended,
/// so `pause` is the longest that mail waits behind another account.
/// One account never waits: its `IDLE` ends the moment mail arrives.
///
/// # Errors
///
/// The function fails if a server refuses a command, or if the store
/// or the index cannot take a write. A connection that breaks is not
/// a failure: the next round opens another one.
pub async fn watching<S: Future<Output = ()>>(
    books: &mut Books<'_>,
    watched: &[Watched<'_>],
    how: How,
    pause: Duration,
    stop: S,
    out: &mut dyn Write,
) -> Result<usize> {
    let mut rounds = 0;
    tokio::pin!(stop);

    loop {
        let mut reports = Vec::with_capacity(watched.len());
        let mut folders = Vec::with_capacity(watched.len());
        let mut touched = BTreeSet::new();

        for one in watched {
            let report = self::one(
                Arc::clone(books.store),
                Arc::clone(&one.pool),
                one.account,
                how,
            )
            .await?;

            touched.extend(report.touched.iter().copied());
            folders.push(inbox(&report));
            reports.push(report);
        }

        let (wrote, embedded) = match how.dry {
            true => (Wrote::default(), Embedded::default()),
            false => (
                pass::after_sync(books.store, books.index, &touched)?,
                sweep(books.store, books.brain.as_deref_mut(), out)?,
            ),
        };
        rounds += 1;

        say(&reports, wrote, embedded, how.dry, false, out)?;

        tokio::select! {
            () = &mut stop => return Ok(rounds),
            ended = listen(watched, &folders, pause) => ended?,
        }
    }
}

/// The folder to hold open. Mail arrives in the inbox. (§3.1)
fn inbox(report: &Report) -> Option<String> {
    let names: Vec<&str> = report
        .folders
        .iter()
        .map(|one| one.folder.as_str())
        .collect();

    names
        .iter()
        .find(|name| name.eq_ignore_ascii_case("INBOX"))
        .or_else(|| names.first())
        .map(|name| (*name).to_string())
}

/// Wait until one account reports a change, or until the wait ends.
async fn listen(
    watched: &[Watched<'_>],
    folders: &[Option<String>],
    pause: Duration,
) -> Result<()> {
    let mut tasks = JoinSet::new();

    for (one, folder) in watched.iter().zip(folders) {
        let Some(folder) = folder.clone() else {
            continue;
        };
        let pool = Arc::clone(&one.pool);

        tasks.spawn(async move { wait(&pool, &folder, pause).await });
    }

    // An account with no folder gives the loop nothing to wait on, so
    // the wait runs here instead and the round comes back later.
    if tasks.is_empty() {
        sleep(pause).await;

        return Ok(());
    }

    while let Some(joined) = tasks.join_next().await {
        // A connection that broke ends the wait, and the next round
        // opens another one. It is not a failure of the sync. (§3.4)
        match joined.expect("the watch did not panic") {
            Ok(_) => {}
            Err(error) if broken(&error) => return Ok(()),
            Err(error) => return Err(error),
        }
    }

    Ok(())
}

/// Hold one folder open, and say whether anything arrived. (§3.1)
async fn wait(pool: &Pool, folder: &str, most: Duration) -> Result<bool> {
    let mut held = pool.take().await?;
    let idle = held.can("IDLE");

    let news = match open_and_idle(&mut held, folder, most).await {
        Ok(news) => news,
        Err(error) => {
            // A connection that broke in the middle of an answer holds
            // bytes that belong to no command. It never goes back.
            held.retire();

            return Err(error.into());
        }
    };

    // A server with no `IDLE` answers at once. Without a wait here the
    // loop would ask it again and again, as fast as the network goes.
    if !idle {
        drop(held);
        sleep(most).await;
    }

    Ok(news)
}

/// Open the folder, and then wait on it.
async fn open_and_idle(
    held: &mut Held<'_>,
    folder: &str,
    most: Duration,
) -> mailbert_imap::Result<bool> {
    if held.selected() != Some(folder) {
        held.examine(folder).await?;
    }

    held.idle(most).await
}

// ---------------------------------------------------------------------
// The command. (§2.1)
// ---------------------------------------------------------------------

/// Do the work of `mailbert sync`. (§2.1)
///
/// # Errors
///
/// The function fails if an account name is not in the configuration,
/// if a server refuses a command, or if the index cannot take a write.
pub fn command(tool: &Tool, args: &cli::Sync) -> Result<()> {
    let config = tool.config()?;
    let wanted = settings::accounts(&config, args.account.as_deref())?;
    let store = tool.store()?;
    let index = tool.index()?;

    // A dry run asks the server for nothing, so it embeds nothing and
    // needs no model. (§2.1)
    let mut brain = match args.dry_run {
        true => None,
        false => Some(tool.brain()?),
    };

    let how = How {
        full: args.full,
        dry: args.dry_run,
        back: Backoff::default(),
    };

    crate::block_on(async {
        let watched = pools(&wanted)?;
        let mut books = Books {
            store: &store,
            index: &index,
            brain: brain.as_mut(),
        };

        let ended = match args.watch {
            true => watching(
                &mut books,
                &watched,
                how,
                PAUSE,
                quiet(tokio::signal::ctrl_c()),
                &mut io::stdout(),
            )
            .await
            .map(|_| ()),
            false => {
                once(&mut books, &watched, how, args.json, &mut io::stdout())
                    .await
            }
        };

        // The connections say goodbye, even after the sync failed.
        for one in &watched {
            one.pool.close().await;
        }

        ended
    })
}

/// One pass over every account, and the index write after it. (§2.1)
async fn once(
    books: &mut Books<'_>,
    watched: &[Watched<'_>],
    how: How,
    json: bool,
    out: &mut dyn Write,
) -> Result<()> {
    let mut reports = Vec::with_capacity(watched.len());

    tracing::info!(
        accounts = watched.len(),
        full = how.full,
        dry = how.dry,
        "the sync starts"
    );

    for account in watched {
        reports.push(
            one(
                Arc::clone(books.store),
                Arc::clone(&account.pool),
                account.account,
                how,
            )
            .await?,
        );
    }

    // A dry run asks for nothing, so no thread moves and the index
    // stays as it was. (§2.1)
    let (wrote, embedded) = match how.dry {
        true => (Wrote::default(), Embedded::default()),
        false => {
            let touched: BTreeSet<MessageId> = reports
                .iter()
                .flat_map(|report| report.touched.iter().copied())
                .collect();

            let start = Instant::now();
            let wrote = pass::after_sync(books.store, books.index, &touched)?;
            tracing::info!(
                touched = touched.len(),
                messages = wrote.messages,
                threads = wrote.threads,
                ms = start.elapsed().as_millis(),
                "wrote the index"
            );

            (wrote, sweep(books.store, books.brain.as_deref_mut(), out)?)
        }
    };

    say(&reports, wrote, embedded, how.dry, json, out)
}

/// Embed what a sync changed, and write the PLAID index. (§6.2)
///
/// A sync with no brain embeds nothing. `--dry-run` is that case, and
/// so is a test that has no model.
///
/// The sweep says how far it is, because a first pass over a mailbox
/// of 100000 messages takes long enough to look stopped.
fn sweep(
    store: &Store,
    brain: Option<&mut Brain>,
    out: &mut dyn Write,
) -> Result<Embedded> {
    let Some(brain) = brain else {
        return Ok(Embedded::default());
    };

    let start = Instant::now();
    let mut said = 0;
    let embedded = brain.sweep(store, |done| {
        if done >= said + SAY_EVERY {
            said = done;
            tracing::info!(done, "embedding");
            let _ = writeln!(out, "embedded {done} messages");
        }
    })?;

    tracing::info!(
        messages = embedded.messages,
        passages = embedded.passages,
        dropped = embedded.dropped,
        ms = start.elapsed().as_millis(),
        "embedded what the sync changed"
    );

    Ok(embedded)
}

/// Drop whatever a future gives, so it can be a stop signal.
async fn quiet<F: Future>(future: F) {
    let _ = future.await;
}

/// What one account did, in the shape that `--json` writes.
#[derive(Debug, Serialize)]
struct Summary<'a> {
    account: &'a str,
    folders: usize,
    asked: u64,
    kept: u64,
    moved: u64,
    gone: u64,
    broken: u64,
    bytes: u64,
    missing: &'a [String],
}

/// The whole answer of one `sync`, in the shape that `--json` writes.
#[derive(Debug, Serialize)]
struct Answer<'a> {
    dry_run: bool,
    accounts: Vec<Summary<'a>>,
    messages: usize,
    threads: usize,
    embedded: usize,
    passages: usize,
}

/// Write what the sync did.
///
/// # Errors
///
/// The function fails if the output does not take the text.
fn say(
    reports: &[Report],
    wrote: Wrote,
    embedded: Embedded,
    dry: bool,
    json: bool,
    out: &mut dyn Write,
) -> Result<()> {
    let accounts: Vec<Summary<'_>> = reports.iter().map(summary).collect();

    if json {
        let answer = Answer {
            dry_run: dry,
            accounts,
            messages: wrote.messages,
            threads: wrote.threads,
            embedded: embedded.messages,
            passages: embedded.passages,
        };

        writeln!(out, "{}", serde_json::to_string_pretty(&answer)?)?;

        return Ok(());
    }

    for (report, one) in reports.iter().zip(&accounts) {
        match dry {
            true => writeln!(
                out,
                "{}: would ask for {} messages in {} folders",
                one.account, one.asked, one.folders
            )?,
            false => writeln!(
                out,
                "{}: {} new, {} changed, {} gone, {} in {} folders",
                one.account,
                one.kept,
                one.moved,
                one.gone,
                size(one.bytes),
                one.folders
            )?,
        }

        for name in &report.missing {
            writeln!(out, "  no folder `{name}` on the server")?;
        }

        if report.counts.broken > 0 {
            writeln!(
                out,
                "  {} messages that mailbert cannot read",
                report.counts.broken
            )?;
        }
    }

    if !dry {
        writeln!(
            out,
            "indexed {} messages in {} threads",
            wrote.messages, wrote.threads
        )?;
    }

    if embedded.messages > 0 {
        writeln!(
            out,
            "embedded {} messages in {} passages",
            embedded.messages, embedded.passages
        )?;
    }

    Ok(())
}

/// One line of the report of an account.
fn summary(report: &Report) -> Summary<'_> {
    Summary {
        account: &report.account,
        folders: report.folders.len(),
        asked: report.asked(),
        kept: report.counts.kept,
        moved: report.counts.moved,
        gone: report.counts.gone,
        broken: report.counts.broken,
        bytes: report.counts.bytes,
        missing: &report.missing,
    }
}

/// A count of bytes that a person reads.
fn size(bytes: u64) -> String {
    const STEP: f64 = 1024.0;
    const NAMES: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];

    let mut left = bytes as f64;
    let mut at = 0;

    while left >= STEP && at + 1 < NAMES.len() {
        left /= STEP;
        at += 1;
    }

    match at {
        0 => format!("{bytes} B"),
        _ => format!("{left:.1} {}", NAMES[at]),
    }
}

#[cfg(test)]
mod tests {
    //! # Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | A sync writes every message that the server holds | The plan of the fake server | A message that the sync drops never comes back. (§3.3) |
    //! | A second sync asks for nothing | The count of the report | An incremental sync is what makes 100k messages work. (§3.3) |
    //! | A sync never writes to the server | `FakeServer::writes` | mailbert is a download-only mirror. (§3) |

    use std::{future::Future, time::Duration};

    use hegel::{TestCase, generators as gs};
    use mailbert_core::{config::Account, index::MailIndex};
    use mailbert_imap::fake::{FakeFolder, FakeMessage, FakeServer, Plan};
    use serde_json::{Value, json};
    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::trace::pen::{Pen, capture, open};

    /// Run one future, on a runtime that gives the folders real threads.
    fn run_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .expect("the machine gives a runtime")
            .block_on(future)
    }

    /// What a sync writes, with no brain. A test has no model.
    fn books<'a>(store: &'a Arc<Store>, index: &'a MailIndex) -> Books<'a> {
        Books {
            store,
            index,
            brain: None,
        }
    }

    fn open_at(dir: &TempDir) -> Arc<Store> {
        Arc::new(Store::open(dir.path()).expect("a store"))
    }

    /// One message that mailbert can read.
    fn raw(number: u32) -> String {
        format!(
            "Message-ID: <m{number}@example.com>\r\n\
             From: Ann <ann@example.com>\r\n\
             To: Bo <bo@example.com>\r\n\
             Subject: number {number}\r\n\
             Date: Mon, 1 Jan 2024 00:00:00 +0000\r\n\
             \r\n\
             The body of message {number}.\r\n"
        )
    }

    /// A folder with `count` messages, from UID 1.
    fn a_folder(name: &str, count: u32) -> FakeFolder {
        (1..=count).fold(FakeFolder::new(name), |folder, uid| {
            folder.with(FakeMessage::new(uid, &raw(uid)))
        })
    }

    /// An account that reads one fake server, and names its folders.
    fn an_account(port: u16, folders: &[&str]) -> Account {
        Account {
            name: "work".to_string(),
            host: "127.0.0.1".to_string(),
            user: "me".to_string(),
            port,
            password_command: None,
            password_file: None,
            password: Some("secret".to_string()),
            folders: folders.iter().map(|name| (*name).to_string()).collect(),
            exclude: Vec::new(),
            footers: Vec::new(),
            all_folders: false,
            connections: 3,
        }
    }

    /// The pool that reaches the fake server. It takes no TLS.
    fn a_pool(port: u16) -> Arc<Pool> {
        Arc::new(Pool::new(
            Server::at("127.0.0.1", port, false)
                .with_login("me", "secret")
                .with_connections(3),
        ))
    }

    /// A sync that waits milliseconds, and not seconds, after a break.
    fn quickly() -> How {
        How {
            back: Backoff {
                tries: 5,
                first: Duration::from_millis(2),
                most: Duration::from_millis(20),
            },
            ..How::default()
        }
    }

    /// Sync one account, and give what it did.
    async fn sync(
        store: &Arc<Store>,
        pool: &Arc<Pool>,
        account: &Account,
        how: How,
    ) -> Report {
        one(Arc::clone(store), Arc::clone(pool), account, how)
            .await
            .expect("the fake server answers")
    }

    /// The command lines that hold this word.
    fn saw(server: &FakeServer, word: &str) -> Vec<String> {
        server
            .seen()
            .commands
            .into_iter()
            .filter(|line| line.contains(word))
            .collect()
    }

    // -----------------------------------------------------------------
    // The log of a sync. (§10.5)
    // -----------------------------------------------------------------

    /// Run one sync, and give back what its log says.
    ///
    /// The subscriber travels on the future, and not on the thread.
    /// The tests run at the same time, and a thread-local subscriber
    /// does not reach every task of a sync. (§10.5)
    fn log_of(store: &Arc<Store>, folders: &[(&str, u32)]) -> String {
        open();

        let pen = Pen::default();
        let held = tracing::Dispatch::new(capture(pen.clone()));

        run_on(
            async {
                let plan =
                    folders.iter().fold(Plan::new(), |plan, (name, count)| {
                        plan.with(a_folder(name, *count))
                    });
                let server = FakeServer::start(plan).await.expect("a server");
                let names: Vec<&str> =
                    folders.iter().map(|(name, _)| *name).collect();
                let account = an_account(server.port(), &names);

                sync(store, &a_pool(server.port()), &account, How::default())
                    .await
            }
            .with_subscriber(held),
        );

        pen.text()
    }

    /// A reader who watches a sync must see which account, and which
    /// folder, the work is in. Nothing else says where the time goes.
    #[test]
    fn the_log_of_a_sync_names_the_account_and_each_folder() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let log = log_of(&store, &[("INBOX", 3), ("Sent", 2)]);

        // The span of the folder sits inside the span of the account,
        // so every line says where the work is. (§10.5)
        assert!(
            log.contains("account{account=work}:folder{folder=INBOX}:"),
            "{log}"
        );
        assert!(
            log.contains("account{account=work}:folder{folder=Sent}:"),
            "{log}"
        );
    }

    #[test]
    fn the_log_of_a_sync_counts_what_each_folder_kept() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let log = log_of(&store, &[("INBOX", 3)]);
        let done = log
            .lines()
            .find(|line| line.contains("the folder is done"))
            .unwrap_or_else(|| panic!("no folder ended: {log}"));

        assert!(done.contains("folder=INBOX"), "{done}");
        assert!(done.contains("kept=3"), "{done}");
        assert!(done.contains("ms="), "{done}");
    }

    /// §10.5 says how long a step took, because a reader who waits
    /// wants to know whether the server or the store is slow.
    #[test]
    fn the_log_of_a_sync_says_how_many_folders_it_chose() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let log = log_of(&store, &[("INBOX", 1), ("Sent", 1)]);
        let listed = log
            .lines()
            .find(|line| line.contains("listed the folders"))
            .unwrap_or_else(|| panic!("no listing: {log}"));

        assert!(listed.contains("chosen=2"), "{listed}");
    }

    // -----------------------------------------------------------------
    // What one sync writes. (§2.1, §3.3)
    // -----------------------------------------------------------------

    #[test]
    fn a_sync_writes_every_message_of_a_folder_that_the_config_names() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let kept = run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 3)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);

            sync(&store, &a_pool(server.port()), &account, How::default()).await
        });

        assert_eq!(kept.counts.kept, 3);
        assert_eq!(store.all().expect("a read").len(), 3);
    }

    #[test]
    fn a_sync_names_the_account_of_every_copy_that_it_wrote() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 2)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);

            sync(&store, &a_pool(server.port()), &account, How::default()).await
        });

        for message in store.all().expect("a read") {
            for at in &message.locations {
                assert_eq!(at.account, "work", "{:?}", message.id);
            }
        }
    }

    /// §3: mailbert is a download-only mirror. It writes nothing.
    #[test]
    fn a_sync_never_writes_to_the_server() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let writes = run_on(async {
            let server = FakeServer::start(
                Plan::new()
                    .with(a_folder("INBOX", 4))
                    .with(a_folder("Archive", 2)),
            )
            .await
            .expect("a server");
            let account = an_account(server.port(), &["INBOX", "Archive"]);

            sync(&store, &a_pool(server.port()), &account, How::default())
                .await;

            server.writes()
        });

        assert!(writes.is_empty(), "{writes:?}");
    }

    /// §3: a body arrives with `BODY.PEEK[]`, so it takes no `\Seen`.
    #[test]
    fn a_sync_reads_a_body_and_leaves_the_seen_flag_alone() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let commands = run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 2)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);

            sync(&store, &a_pool(server.port()), &account, How::default())
                .await;

            server.seen().commands
        });

        assert!(commands.iter().any(|line| line.contains("BODY.PEEK[]")));
        assert!(!commands.iter().any(|line| line.contains("SELECT ")));
    }

    #[test]
    fn a_sync_opens_no_folder_that_the_config_leaves_out() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let (report, opened) = run_on(async {
            let server = FakeServer::start(
                Plan::new()
                    .with(a_folder("INBOX", 1))
                    .with(a_folder("Spam", 5)),
            )
            .await
            .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);

            let report =
                sync(&store, &a_pool(server.port()), &account, How::default())
                    .await;

            (report, saw(&server, "Spam"))
        });

        assert_eq!(report.counts.kept, 1);
        assert!(opened.is_empty(), "{opened:?}");
    }

    #[test]
    fn an_excluded_folder_never_reaches_a_plan() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let report = run_on(async {
            let server = FakeServer::start(
                Plan::new()
                    .with(a_folder("INBOX", 1))
                    .with(a_folder("Spam", 5)),
            )
            .await
            .expect("a server");

            let mut account = an_account(server.port(), &[]);
            account.all_folders = true;
            account.exclude = vec!["Spam".to_string()];

            sync(&store, &a_pool(server.port()), &account, How::default()).await
        });

        let names: Vec<&str> = report
            .folders
            .iter()
            .map(|one| one.folder.as_str())
            .collect();

        assert_eq!(names, vec!["INBOX"]);
    }

    #[test]
    fn an_account_that_takes_them_all_reads_every_folder_of_the_server() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let report = run_on(async {
            let server = FakeServer::start(
                Plan::new()
                    .with(a_folder("INBOX", 2))
                    .with(a_folder("Archive", 3)),
            )
            .await
            .expect("a server");

            let mut account = an_account(server.port(), &[]);
            account.all_folders = true;

            sync(&store, &a_pool(server.port()), &account, How::default()).await
        });

        assert_eq!(report.folders.len(), 2);
        assert_eq!(report.counts.kept, 5);
    }

    /// Gmail translates the name of its Trash folder, so a name in
    /// `exclude` cannot keep that folder out. The attribute can. (§1.2)
    #[test]
    fn an_exclude_that_names_an_attribute_keeps_a_folder_out() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let report = run_on(async {
            let plan = Plan::new()
                .with(a_folder("INBOX", 2))
                .with(a_folder("Lixeira", 3).with_attribute("\\Trash"));
            let server = FakeServer::start(plan).await.expect("a server");

            let mut account = an_account(server.port(), &[]);
            account.all_folders = true;
            account.exclude = vec!["\\Trash".to_string()];

            sync(&store, &a_pool(server.port()), &account, How::default()).await
        });

        // The two of INBOX arrive, and the three of the trash do not.
        assert_eq!(report.counts.kept, 2);
    }

    #[test]
    fn a_folder_that_the_server_does_not_hold_is_named_in_the_report() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let report = run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 1)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX", "Archve"]);

            sync(&store, &a_pool(server.port()), &account, How::default()).await
        });

        assert_eq!(report.missing, vec!["Archve"]);
        assert_eq!(report.counts.kept, 1, "the good folder still arrives");
    }

    // -----------------------------------------------------------------
    // The second sync. (§3.3)
    // -----------------------------------------------------------------

    #[test]
    fn a_second_sync_asks_the_server_for_nothing() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let (first, second) = run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 3)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);
            let pool = a_pool(server.port());

            let first = sync(&store, &pool, &account, How::default()).await;
            let second = sync(&store, &pool, &account, How::default()).await;

            (first, second)
        });

        assert_eq!(first.asked(), 3);
        assert_eq!(second.asked(), 0);
        assert_eq!(second.counts.kept, 0);
        assert_eq!(store.all().expect("a read").len(), 3);
    }

    #[test]
    fn a_second_sync_reads_only_the_message_that_arrived() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let second = run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 2)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);
            let pool = a_pool(server.port());

            sync(&store, &pool, &account, How::default()).await;

            server.change(|plan| {
                let folder = plan.folder_mut("INBOX").expect("the folder");
                folder
                    .messages
                    .push(FakeMessage::new(3, &raw(3)).with_mod_seq(9));
            });

            sync(&store, &pool, &account, How::default()).await
        });

        assert_eq!(second.asked(), 1);
        assert_eq!(second.counts.kept, 1);
        assert_eq!(store.all().expect("a read").len(), 3);
    }

    #[test]
    fn a_full_sync_asks_for_every_message_again() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let again = run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 3)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);
            let pool = a_pool(server.port());

            sync(&store, &pool, &account, How::default()).await;

            sync(
                &store,
                &pool,
                &account,
                How {
                    full: true,
                    ..How::default()
                },
            )
            .await
        });

        assert_eq!(again.asked(), 3);
        assert!(again.folders[0].restart);
        assert_eq!(store.all().expect("a read").len(), 3, "no message doubles");
    }

    #[test]
    fn a_uid_validity_that_moved_reads_the_folder_again() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let second = run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 2)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);
            let pool = a_pool(server.port());

            sync(&store, &pool, &account, How::default()).await;

            server.change(|plan| {
                plan.folder_mut("INBOX").expect("the folder").uid_validity = 77;
            });

            sync(&store, &pool, &account, How::default()).await
        });

        assert!(second.folders[0].restart);
        assert_eq!(second.asked(), 2);
        assert_eq!(store.all().expect("a read").len(), 2);
    }

    // -----------------------------------------------------------------
    // What a dry run does. (§2.1)
    // -----------------------------------------------------------------

    #[test]
    fn a_dry_run_says_what_it_would_ask_and_writes_nothing() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let (report, fetches) = run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 4)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);

            let report = sync(
                &store,
                &a_pool(server.port()),
                &account,
                How {
                    dry: true,
                    ..How::default()
                },
            )
            .await;

            (report, saw(&server, "FETCH"))
        });

        assert_eq!(report.asked(), 4);
        assert_eq!(report.counts, Counts::default());
        assert!(fetches.is_empty(), "{fetches:?}");
        assert!(store.all().expect("a read").is_empty());
    }

    #[test]
    fn a_dry_run_leaves_the_state_of_the_folder_alone() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 2)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);

            sync(
                &store,
                &a_pool(server.port()),
                &account,
                How {
                    dry: true,
                    ..How::default()
                },
            )
            .await;
        });

        assert_eq!(store.state("work", "INBOX").expect("a read"), None);
    }

    // -----------------------------------------------------------------
    // The flags, and the messages that go away. (§3.3)
    // -----------------------------------------------------------------

    #[test]
    fn a_flag_that_the_server_set_reaches_the_store() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 2)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);
            let pool = a_pool(server.port());

            sync(&store, &pool, &account, How::default()).await;

            server.change(|plan| {
                let folder = plan.folder_mut("INBOX").expect("the folder");
                folder.messages[0].flags = vec!["\\Seen".to_string()];
                folder.messages[0].mod_seq = 40;
            });

            sync(&store, &pool, &account, How::default()).await;
        });

        let read = store
            .placed("work", "INBOX", 1)
            .expect("a read")
            .and_then(|id| store.get(&id).expect("a read"))
            .expect("the message");

        assert!(read.flags.contains("\\seen"), "{:?}", read.flags);
    }

    #[test]
    fn a_flag_that_the_server_dropped_leaves_the_store() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        run_on(async {
            let mut folder = FakeFolder::new("INBOX");
            folder = folder
                .with(FakeMessage::new(1, &raw(1)).with_flags(&["\\Seen"]));

            let server = FakeServer::start(Plan::new().with(folder))
                .await
                .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);
            let pool = a_pool(server.port());

            sync(&store, &pool, &account, How::default()).await;

            server.change(|plan| {
                let folder = plan.folder_mut("INBOX").expect("the folder");
                folder.messages[0].flags = Vec::new();
                folder.messages[0].mod_seq = 40;
            });

            sync(&store, &pool, &account, How::default()).await;
        });

        let read = store
            .placed("work", "INBOX", 1)
            .expect("a read")
            .and_then(|id| store.get(&id).expect("a read"))
            .expect("the message");

        assert!(read.flags.is_empty(), "{:?}", read.flags);
    }

    #[test]
    fn a_message_that_the_server_lost_leaves_that_folder() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 2)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);
            let pool = a_pool(server.port());

            sync(&store, &pool, &account, How::default()).await;

            server.change(|plan| {
                let folder = plan.folder_mut("INBOX").expect("the folder");
                folder.messages.retain(|message| message.uid != 1);
                folder.gone.push(1);
                folder.messages[0].mod_seq = 40;
            });

            sync(&store, &pool, &account, How::default()).await;
        });

        assert_eq!(store.placed("work", "INBOX", 1).expect("a read"), None);
        assert!(store.placed("work", "INBOX", 2).expect("a read").is_some());
    }

    // -----------------------------------------------------------------
    // What the command writes. (§2.1)
    // -----------------------------------------------------------------

    /// A report with these counts, and no folder that went missing.
    fn a_report(name: &str, folders: usize, counts: Counts) -> Report {
        Report {
            account: name.to_string(),
            folders: (0..folders)
                .map(|at| FolderReport {
                    folder: format!("Box{at}"),
                    asked: 2,
                    restart: false,
                })
                .collect(),
            missing: Vec::new(),
            counts,
            touched: BTreeSet::new(),
        }
    }

    fn text_of(reports: &[Report], wrote: Wrote, dry: bool) -> String {
        let mut out = Vec::new();
        say(reports, wrote, Embedded::default(), dry, false, &mut out)
            .expect("a writable buffer");

        String::from_utf8(out).expect("the report is text")
    }

    fn json_of(reports: &[Report], wrote: Wrote, dry: bool) -> Value {
        let mut out = Vec::new();
        say(reports, wrote, Embedded::default(), dry, true, &mut out)
            .expect("a writable buffer");

        serde_json::from_slice(&out).expect("the report is JSON")
    }

    fn embedded(messages: usize, passages: usize) -> Embedded {
        Embedded {
            messages,
            passages,
            dropped: 0,
        }
    }

    /// §6.2: a sync embeds what it fetched, and says so.
    #[test]
    fn the_report_names_what_the_sweep_embedded() {
        let mut out = Vec::new();
        let reports = [a_report("work", 1, Counts::default())];
        say(
            &reports,
            Wrote::default(),
            embedded(2, 5),
            false,
            false,
            &mut out,
        )
        .expect("a writable buffer");

        let text = String::from_utf8(out).expect("the report is text");

        assert!(text.contains("embedded 2 messages in 5 passages"), "{text}");
    }

    /// A sync that embedded nothing says nothing about the model.
    #[test]
    fn a_report_of_no_embedding_says_nothing_about_it() {
        let text = text_of(
            &[a_report("work", 1, Counts::default())],
            Wrote::default(),
            false,
        );

        assert!(!text.contains("embedded"), "{text}");
    }

    #[test]
    fn the_json_holds_what_the_sweep_embedded() {
        let mut out = Vec::new();
        let reports = [a_report("work", 1, Counts::default())];
        say(
            &reports,
            Wrote::default(),
            embedded(2, 5),
            false,
            true,
            &mut out,
        )
        .expect("a writable buffer");

        let answer: Value =
            serde_json::from_slice(&out).expect("the report is JSON");

        assert_eq!(answer["embedded"], 2);
        assert_eq!(answer["passages"], 5);
    }

    /// A `--dry-run` makes no brain, and a test has none either. The
    /// sweep must then do nothing at all, and load no model.
    #[test]
    fn a_sweep_with_no_brain_embeds_nothing() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut out = Vec::new();

        let done = sweep(&store, None, &mut out).expect("a sweep");

        assert_eq!(done, Embedded::default());
        assert!(out.is_empty());
    }

    /// §1.2: the password goes over TLS, whatever the port says.
    #[test]
    fn the_server_of_an_account_always_takes_tls() {
        let account = an_account(143, &["INBOX"]);

        let server = server(&account, "hunter2");

        assert!(server.tls);
        assert_eq!(server.port, 143);
        assert_eq!(server.user, "me");
        assert_eq!(server.password, "hunter2");
        assert_eq!(server.connections, 3);
    }

    #[test]
    fn the_report_of_a_sync_says_what_each_account_did() {
        let counts = Counts {
            kept: 12,
            moved: 3,
            gone: 1,
            broken: 0,
            bytes: 2048,
        };

        let text = text_of(
            &[a_report("work", 2, counts)],
            Wrote {
                messages: 12,
                threads: 5,
            },
            false,
        );

        assert!(text.contains("work: 12 new, 3 changed, 1 gone"), "{text}");
        assert!(text.contains("2.0 KiB in 2 folders"), "{text}");
        assert!(text.contains("indexed 12 messages in 5 threads"), "{text}");
    }

    #[test]
    fn the_report_of_a_dry_run_says_that_it_wrote_nothing() {
        let text = text_of(
            &[a_report("work", 3, Counts::default())],
            Wrote::default(),
            true,
        );

        assert!(
            text.contains("would ask for 6 messages in 3 folders"),
            "{text}"
        );
        assert!(!text.contains("indexed"), "{text}");
    }

    #[test]
    fn the_report_names_a_folder_that_the_server_does_not_hold() {
        let mut report = a_report("work", 1, Counts::default());
        report.missing = vec!["Archve".to_string()];

        let text = text_of(&[report], Wrote::default(), false);

        assert!(text.contains("no folder `Archve` on the server"), "{text}");
    }

    #[test]
    fn the_report_counts_the_messages_that_mailbert_cannot_read() {
        let counts = Counts {
            broken: 2,
            ..Counts::default()
        };

        let text =
            text_of(&[a_report("work", 1, counts)], Wrote::default(), false);

        assert!(
            text.contains("2 messages that mailbert cannot read"),
            "{text}"
        );
    }

    #[test]
    fn the_json_of_a_sync_holds_the_counts_of_every_account() {
        let counts = Counts {
            kept: 4,
            moved: 1,
            gone: 0,
            broken: 0,
            bytes: 90,
        };

        let json = json_of(
            &[
                a_report("work", 1, counts),
                a_report("home", 2, Counts::default()),
            ],
            Wrote {
                messages: 4,
                threads: 2,
            },
            false,
        );

        assert_eq!(json["dry_run"], json!(false));
        assert_eq!(json["messages"], json!(4));
        assert_eq!(json["threads"], json!(2));
        assert_eq!(json["accounts"][0]["account"], json!("work"));
        assert_eq!(json["accounts"][0]["kept"], json!(4));
        assert_eq!(json["accounts"][0]["asked"], json!(2));
        assert_eq!(json["accounts"][1]["account"], json!("home"));
        assert_eq!(json["accounts"][1]["folders"], json!(2));
    }

    #[test]
    fn a_size_under_a_kibibyte_reads_in_bytes() {
        assert_eq!(size(0), "0 B");
        assert_eq!(size(1023), "1023 B");
        assert_eq!(size(1024), "1.0 KiB");
        assert_eq!(size(1024 * 1024 * 3), "3.0 MiB");
    }

    #[hegel::test(test_cases = 60)]
    fn prop_a_size_never_loses_its_unit(tc: TestCase) {
        let bytes =
            tc.draw(gs::integers::<u64>().min_value(0).max_value(u64::MAX / 2));

        let text = size(bytes);
        let unit = text.split(' ').nth(1).expect("a unit after the number");

        assert!(["B", "KiB", "MiB", "GiB", "TiB"].contains(&unit), "{text}");
    }

    // -----------------------------------------------------------------
    // The watch. (§3.1)
    // -----------------------------------------------------------------

    /// One account on a fake server, with the pool that reaches it.
    fn a_watch<'a>(account: &'a Account, port: u16) -> Vec<Watched<'a>> {
        vec![Watched {
            account,
            pool: a_pool(port),
        }]
    }

    /// A signal that arrives after this long.
    async fn after(wait: Duration) {
        tokio::time::sleep(wait).await;
    }

    /// A signal for when the store holds `count` messages.
    ///
    /// The wait has an end, so a watch that never comes back fails the
    /// test instead of holding it open.
    async fn until(store: &Store, count: usize, most: Duration) {
        let end = tokio::time::Instant::now() + most;

        while store.all().expect("a read").len() < count
            && tokio::time::Instant::now() < end
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    #[test]
    fn a_watch_syncs_once_before_it_waits() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = MailIndex::open_in_ram().expect("an index");
        let mut out = Vec::new();

        let rounds = run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 2)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);
            let watched = a_watch(&account, server.port());

            watching(
                &mut books(&store, &index),
                &watched,
                quickly(),
                Duration::from_millis(80),
                after(Duration::from_millis(30)),
                &mut out,
            )
            .await
            .expect("a watch that stops well")
        });

        assert_eq!(rounds, 1);
        assert_eq!(store.all().expect("a read").len(), 2);

        let text = String::from_utf8(out).expect("the report is text");
        assert!(text.contains("work: 2 new"), "{text}");
    }

    #[test]
    fn a_watch_syncs_again_when_a_message_arrives() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = MailIndex::open_in_ram().expect("an index");
        let mut out = Vec::new();

        let rounds = run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 2)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);
            let watched = a_watch(&account, server.port());

            let deliver = async {
                // The message must arrive while the connection idles.
                // The server logs `IDLE` the moment it reads it, and a
                // short wait after that lets the fake read its count.
                while saw(&server, "IDLE").is_empty() {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;

                server.change(|plan| {
                    plan.folder_mut("INBOX")
                        .expect("the folder")
                        .messages
                        .push(FakeMessage::new(3, &raw(3)).with_mod_seq(9));
                });
            };

            let mut held = books(&store, &index);
            let (rounds, ()) = tokio::join!(
                watching(
                    &mut held,
                    &watched,
                    quickly(),
                    Duration::from_secs(20),
                    until(&store, 3, Duration::from_secs(5)),
                    &mut out,
                ),
                deliver,
            );

            rounds.expect("a watch that stops well")
        });

        assert!(rounds >= 2, "the watch never woke up: {rounds}");
        assert_eq!(store.all().expect("a read").len(), 3);
    }

    #[test]
    fn a_watch_that_sees_nothing_asks_the_server_for_nothing() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = MailIndex::open_in_ram().expect("an index");
        let mut out = Vec::new();

        let (rounds, fetches, writes) = run_on(async {
            let server =
                FakeServer::start(Plan::new().with(a_folder("INBOX", 2)))
                    .await
                    .expect("a server");
            let account = an_account(server.port(), &["INBOX"]);
            let watched = a_watch(&account, server.port());

            let rounds = watching(
                &mut books(&store, &index),
                &watched,
                quickly(),
                Duration::from_secs(5),
                after(Duration::from_millis(200)),
                &mut out,
            )
            .await
            .expect("a watch that stops well");

            (rounds, saw(&server, "FETCH").len(), server.writes())
        });

        assert_eq!(rounds, 1, "a quiet mailbox never syncs again");
        assert_eq!(fetches, 1, "the first round asks, and nothing else does");
        assert!(writes.is_empty(), "{writes:?}");
    }

    #[test]
    fn a_watch_of_a_server_with_no_idle_still_stops() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = MailIndex::open_in_ram().expect("an index");
        let mut out = Vec::new();

        let rounds = run_on(async {
            let plan = Plan::new()
                .with(a_folder("INBOX", 1))
                .with_capabilities(&["IMAP4rev1"]);
            let server = FakeServer::start(plan).await.expect("a server");
            let account = an_account(server.port(), &["INBOX"]);
            let watched = a_watch(&account, server.port());

            watching(
                &mut books(&store, &index),
                &watched,
                quickly(),
                Duration::from_millis(50),
                after(Duration::from_millis(220)),
                &mut out,
            )
            .await
            .expect("a watch that stops well")
        });

        assert!(rounds >= 2, "a poll must come back: {rounds}");
        assert_eq!(store.all().expect("a read").len(), 1);
    }

    #[test]
    fn the_folder_to_hold_open_is_the_inbox() {
        let mut report = a_report("work", 0, Counts::default());
        for name in ["Archive", "INBOX", "Spam"] {
            report.folders.push(FolderReport {
                folder: name.to_string(),
                asked: 0,
                restart: false,
            });
        }

        assert_eq!(inbox(&report), Some("INBOX".to_string()));
    }

    #[test]
    fn an_account_with_no_inbox_holds_its_first_folder_open() {
        let report = a_report("work", 2, Counts::default());

        assert_eq!(inbox(&report), Some("Box0".to_string()));
    }

    #[test]
    fn an_account_with_no_folder_holds_nothing_open() {
        let report = a_report("work", 0, Counts::default());

        assert_eq!(inbox(&report), None);
    }

    // -----------------------------------------------------------------
    // The properties.
    // -----------------------------------------------------------------

    /// A folder holds from 0 to 6 messages.
    #[hegel::composite]
    fn some_counts(tc: TestCase) -> Vec<u32> {
        tc.draw(
            gs::vecs(gs::integers::<u32>().min_value(0).max_value(6))
                .min_size(1)
                .max_size(3),
        )
    }

    #[hegel::test(test_cases = 24)]
    fn prop_a_sync_writes_every_message_that_the_server_holds(tc: TestCase) {
        let counts = tc.draw(some_counts());
        let names: Vec<String> =
            (0..counts.len()).map(|at| format!("Box{at}")).collect();

        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let report = run_on(async {
            let mut plan = Plan::new();
            for (name, count) in names.iter().zip(&counts) {
                plan = plan.with(a_folder(name, *count));
            }

            let server = FakeServer::start(plan).await.expect("a server");
            let mut account = an_account(server.port(), &[]);
            account.all_folders = true;

            sync(&store, &a_pool(server.port()), &account, How::default()).await
        });

        // Each folder holds the same message identities, so the store
        // joins them into the count of the largest folder. (§4.2)
        let most = counts.iter().copied().max().unwrap_or(0) as usize;
        let total: u32 = counts.iter().sum();

        assert_eq!(report.counts.kept, u64::from(total));
        assert_eq!(store.all().expect("a read").len(), most);
    }

    #[hegel::test(test_cases = 24)]
    fn prop_a_second_sync_asks_for_nothing(tc: TestCase) {
        let counts = tc.draw(some_counts());
        let names: Vec<String> =
            (0..counts.len()).map(|at| format!("Box{at}")).collect();

        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let (second, writes) = run_on(async {
            let mut plan = Plan::new();
            for (name, count) in names.iter().zip(&counts) {
                plan = plan.with(a_folder(name, *count));
            }

            let server = FakeServer::start(plan).await.expect("a server");
            let mut account = an_account(server.port(), &[]);
            account.all_folders = true;
            let pool = a_pool(server.port());

            sync(&store, &pool, &account, How::default()).await;
            let second = sync(&store, &pool, &account, How::default()).await;

            (second, server.writes())
        });

        assert_eq!(second.asked(), 0);
        assert_eq!(second.counts.kept, 0);
        assert!(writes.is_empty(), "{writes:?}");
    }

    #[hegel::test(test_cases = 16)]
    fn prop_a_sync_that_stopped_reads_the_rest(tc: TestCase) {
        let count = tc.draw(gs::integers::<u32>().min_value(1).max_value(6));
        let cut = tc.draw(gs::integers::<usize>().min_value(4).max_value(9));

        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        run_on(async {
            let plan =
                Plan::new().with(a_folder("INBOX", count)).cut_after(cut);
            let server = FakeServer::start(plan).await.expect("a server");
            let account = an_account(server.port(), &["INBOX"]);
            let pool = a_pool(server.port());

            // The first sync may break in the middle of the folder.
            let _ =
                one(Arc::clone(&store), Arc::clone(&pool), &account, quickly())
                    .await;

            server.change(|plan| plan.cut_after = None);

            sync(&store, &pool, &account, quickly()).await;
        });

        assert_eq!(store.all().expect("a read").len(), count as usize);
    }
}
