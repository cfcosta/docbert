//! Incremental sync of one folder. (§3.3, §3.4)
//!
//! A second sync reads only what changed. The state of a folder holds
//! `UIDVALIDITY`, `UIDNEXT`, and `HIGHESTMODSEQ`, and the UIDs that the
//! server has and the store does not. That last set is what makes a
//! sync that stops continue where it stopped. (§3.4)

use std::{
    collections::VecDeque,
    sync::{Mutex, MutexGuard},
    time::Duration,
};

use futures_util::{StreamExt, stream::FuturesUnordered};
use tokio::{sync::mpsc, time::sleep};

use crate::{
    connection::{Batch, Connection, View},
    error::{Error, Result},
    pool::{Held, Pool},
    sequence::UidSet,
};

/// Hold the queue of batches. A lock that broke means a task died with
/// the queue open, and no connection can trust the queue after that.
fn hold(queue: &Mutex<VecDeque<usize>>) -> MutexGuard<'_, VecDeque<usize>> {
    queue.lock().unwrap_or_else(|held| held.into_inner())
}

/// What mailbert keeps for one folder of one account. (§3.3)
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FolderState {
    pub uid_validity: u32,
    pub uid_next: u32,
    pub highest_mod_seq: u64,
    /// The UIDs that the server has, and the store does not. (§3.4)
    pub pending: UidSet,
}

impl FolderState {
    /// True when no sync of this folder ended well.
    pub fn is_new(&self) -> bool {
        self.uid_validity == 0
    }
}

/// What a sync of one folder must do. (§3.3)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Job {
    pub folder: String,
    /// True when the UIDs changed meaning, so the folder starts again.
    pub restart: bool,
    /// The batches to fetch, newest first. (§3.2)
    pub batches: Vec<UidSet>,
    /// The UIDs that arrived before, and may have changed. (§3.3)
    pub changed: Option<UidSet>,
    /// The `MODSEQ` above which the server reports a change.
    pub since: Option<u64>,
    /// The state to keep before the first batch. (§3.4)
    pub start: FolderState,
    /// The state to keep when every batch arrived. (§3.4)
    pub done: FolderState,
    /// How many UIDs one batch holds. (§3.1)
    pub size: u32,
}

impl Job {
    /// True when the folder needs no work.
    pub fn is_empty(&self) -> bool {
        self.batches.is_empty() && self.changed.is_none()
    }

    /// True when the plan asks for far more UIDs than the folder has.
    /// (§3.2)
    ///
    /// The plan asks for every UID between the last sync and `UIDNEXT`.
    /// `exists` is what `EXAMINE` said the folder holds, so the plan
    /// asks for at least `count() - exists` UIDs that went away long
    /// ago. Those UIDs are holes, and a batch of holes costs a round
    /// trip and brings no mail.
    ///
    /// [`Connection::uids`] names the mail and cuts the holes away, but
    /// it costs a round trip of its own, and the answer names every UID
    /// of the folder. It therefore only pays when it saves a batch or
    /// more. The common sync owes a few messages and no batch of holes,
    /// and it must not pay for a search.
    ///
    /// [`Connection::uids`]: crate::Connection::uids
    pub fn mostly_holes(&self, exists: u32) -> bool {
        let holes = self.count().saturating_sub(u64::from(exists));

        holes >= u64::from(self.size)
    }

    /// How many messages the job must fetch.
    pub fn count(&self) -> u64 {
        self.batches.iter().map(UidSet::count).sum()
    }

    /// What is left of this job, from the state that the store holds.
    ///
    /// A sync that stopped continues from here, and asks for no
    /// message that arrived before it stopped. (§3.4)
    pub fn left(&self, state: &FolderState) -> Self {
        plan(&self.folder, state, &self.view(), self.size)
    }

    /// The folder, as the server showed it when the job started.
    fn view(&self) -> View {
        View {
            name: self.folder.clone(),
            exists: 0,
            uid_validity: self.done.uid_validity,
            uid_next: self.done.uid_next,
            highest_mod_seq: self.done.highest_mod_seq,
        }
    }

    /// Ask only for the UIDs that the server says it holds. (§3.2)
    ///
    /// A plan asks for every UID between the last sync and `UIDNEXT`.
    /// A folder that lost mail long ago spans a wide range of UIDs and
    /// holds little mail, so most of that range is holes. A batch of
    /// holes costs a round trip and brings nothing.
    ///
    /// `held` comes from [`Connection::uids`]. It is a filter, and
    /// never a source of work: a UID that the plan never wanted stays
    /// out of it.
    ///
    /// The place of the folder does not move. `done` still names the
    /// `UIDNEXT` that the server showed, so the next sync starts from
    /// there and not from the last UID that this one read. (§3.4)
    ///
    /// `changed` stays whole. It asks which flags moved on the mail
    /// that the store already holds, and that question is about the
    /// store, and not about what a fetch can bring. (§3.3)
    pub fn only(mut self, held: &UidSet) -> Self {
        let owed = self.start.pending.and(held);

        // A plan that the cut left empty is a folder that is done.
        // `run` marks `start` and runs no batch, so `start` must
        // carry the place that the server showed, exactly as a plan
        // that owed nothing from the beginning does. (§3.3)
        if owed.is_empty() {
            self.start.highest_mod_seq = self.done.highest_mod_seq;
        }

        self.batches = owed.split(self.size);
        self.start.pending = owed;

        self
    }

    /// The state to keep after this batch arrived. (§3.4)
    ///
    /// The sequence of the folder moves only when the debt is paid.
    /// A sync that stops before that reads the rest the next time.
    pub fn after(&self, held: &FolderState, batch: &UidSet) -> FolderState {
        let pending = held.pending.without(batch);
        let highest_mod_seq = if pending.is_empty() {
            self.done.highest_mod_seq
        } else {
            held.highest_mod_seq
        };

        FolderState {
            uid_validity: self.done.uid_validity,
            uid_next: self.done.uid_next,
            highest_mod_seq,
            pending,
        }
    }
}

/// What one folder needs, from its state and what the server says.
///
/// A `UIDVALIDITY` that changed makes every old UID meaningless, so
/// the folder starts again. The blobs stay, because the identity of a
/// message does not come from its UID. (§3.3, §4.1)
pub fn plan(folder: &str, saved: &FolderState, view: &View, size: u32) -> Job {
    let restart = saved.uid_validity != view.uid_validity;
    let top = view.uid_next.saturating_sub(1);

    let owed = if restart {
        span(1, top)
    } else {
        saved.pending.union(&span(saved.uid_next.max(1), top))
    };

    // The messages that the store holds already. Only these can have
    // a flag that changed, or go away. (§3.3)
    let moved = !restart && view.highest_mod_seq > saved.highest_mod_seq;
    let known = span(1, saved.uid_next.saturating_sub(1)).without(&owed);
    let ask = moved && saved.highest_mod_seq > 0 && !known.is_empty();

    Job {
        folder: folder.to_string(),
        restart,
        batches: owed.split(size),
        changed: ask.then_some(known),
        since: ask.then_some(saved.highest_mod_seq),
        start: FolderState {
            uid_validity: view.uid_validity,
            uid_next: view.uid_next,
            highest_mod_seq: if owed.is_empty() {
                view.highest_mod_seq
            } else {
                saved.highest_mod_seq
            },
            pending: owed.clone(),
        },
        done: FolderState {
            uid_validity: view.uid_validity,
            uid_next: view.uid_next,
            highest_mod_seq: view.highest_mod_seq,
            pending: UidSet::new(),
        },
        size,
    }
}

/// The UIDs from `low` to `high`, and nothing when there are none.
///
/// `UidSet::range` orders its two ends, which is right for a set that
/// a user wrote. Here an end below the start means an empty folder,
/// or a folder that did not grow.
fn span(low: u32, high: u32) -> UidSet {
    if low > high || high == 0 {
        return UidSet::new();
    }

    UidSet::range(low, high)
}

/// What a sync does with each batch that it reads. (§3.4)
pub trait Keep {
    /// Keep the state of a folder, and nothing else.
    fn mark(
        &mut self,
        folder: &str,
        state: &FolderState,
    ) -> impl Future<Output = Result<()>> + Send;

    /// Keep a batch of messages, and the state after it.
    ///
    /// The sync waits for this, so the state on disk never runs ahead
    /// of the mail that the store holds. (§3.4)
    fn batch(
        &mut self,
        folder: &str,
        batch: Batch,
        state: &FolderState,
    ) -> impl Future<Output = Result<()>> + Send;

    /// The state that the store holds for this folder.
    ///
    /// A sync that starts again reads this, and not its own memory,
    /// because the memory of a program that stopped is gone. (§3.4)
    fn state(
        &mut self,
        folder: &str,
    ) -> impl Future<Output = Result<FolderState>> + Send;
}

/// Do one job on one connection. (§3.3)
///
/// The state goes to the sink before the first batch, and again after
/// each one. A sync that stops loses only the batch that was in the
/// air. (§3.4)
pub async fn run<K: Keep>(
    connection: &mut Connection,
    job: &Job,
    keep: &mut K,
) -> Result<FolderState> {
    if connection.selected() != Some(job.folder.as_str()) {
        connection.examine(&job.folder).await?;
    }

    let mut state = job.start.clone();
    keep.mark(&job.folder, &state).await?;

    // The batch that the store has not taken yet. The socket reads the
    // batch that follows while the store writes this one, so the disk
    // and the server never wait for each other. (§3.4)
    let mut waiting: Option<(Batch, FolderState)> = None;

    for set in &job.batches {
        state = job.after(&state, set);

        let batch =
            read(connection, keep, &job.folder, set, None, waiting).await?;

        // The state travels with its own batch, and not with the batch
        // that the socket read ahead. (§3.4)
        waiting = Some((batch, state.clone()));
    }

    // The messages that the store holds already: a flag that changed,
    // or a message that went away. (§3.3)
    if let (Some(changed), Some(since)) = (&job.changed, job.since) {
        let batch =
            read(connection, keep, &job.folder, changed, Some(since), waiting)
                .await?;

        waiting = Some((batch, state.clone()));
    }

    if let Some((batch, at)) = waiting {
        keep.batch(&job.folder, batch, &at).await?;
    }

    Ok(state)
}

/// Run a job on every connection that the pool can spare. (§3.1)
///
/// One folder of a mailbox can hold most of its mail. Gmail keeps
/// every message in `All Mail`, so a sync of eight folders on eight
/// connections leaves seven of them idle while one reads 60000
/// messages. The batches of a plan are independent, so any connection
/// can read any of them.
///
/// The connections take batches from one queue, and send what they
/// read back to this task. This task owns the state of the folder, and
/// it is the only one that writes. The batches therefore land in the
/// order that the servers answer, and the state still moves one batch
/// at a time. `Job::after` takes the UIDs of a batch out of the debt,
/// and that does not depend on the order.
///
/// One connection is enough. It reads the next batch while this task
/// writes the last one, exactly as [`run`] does. (§3.4)
///
/// `hands` is a wish. The folder takes one connection and waits for
/// it, because a folder with no connection can do nothing. It asks for
/// the rest and does not wait, because a connection that another
/// folder needs is not free. (§3.1)
///
/// A folder that still owes batches asks again, and waits. The folders
/// that end give their connections back, and a big folder takes them.
/// Gmail keeps every message in `All Mail`, so the folder that reads
/// longest is the one that needs the connections of the others. (§3.1)
pub async fn spread<K: Keep>(
    pool: &Pool,
    job: &Job,
    keep: &mut K,
    hands: usize,
) -> Result<FolderState> {
    let mut state = job.start.clone();
    keep.mark(&job.folder, &state).await?;

    // A connection with no batch to read is a connection that another
    // folder needs. (§3.1)
    let want = hands.min(job.batches.len()).max(1);

    let queue = Mutex::new((0..job.batches.len()).collect::<VecDeque<usize>>());
    let (give, mut take) = mpsc::channel::<Result<(usize, Batch)>>(want);

    // The block owns the queue and the sender, so both go away when
    // every connection ends. A sender that lives in `spread` never
    // goes away, and the writer then waits for a batch for ever.
    let reading = async move {
        let mut readers = FuturesUnordered::new();

        // The first connection that comes back stays for the flags of
        // §3.3. A connection that broke comes back as `None`, and the
        // writer gives the error of it before the flags run.
        let mut spare = None;
        let mut asking = true;

        readers.push(reader(pool.take().await?, job, &queue, give.clone()));

        while readers.len() < want {
            match pool.try_take().await {
                Ok(Some(one)) => {
                    readers.push(reader(one, job, &queue, give.clone()));
                }
                Ok(None) => break,
                Err(problem) => {
                    tracing::debug!(%problem, "the pool spared no connection");

                    break;
                }
            }
        }

        tracing::debug!(
            folder = job.folder,
            hands = readers.len(),
            wanted = want,
            batches = job.batches.len(),
            "spread a folder over the connections"
        );

        loop {
            if readers.is_empty() {
                break;
            }

            // A folder that still owes batches asks for another
            // connection, and waits for it. The folders that end give
            // theirs back, and this is where they land. (§3.1)
            //
            // A folder that failed asks for nothing, because no batch
            // of it will land now. No test sees this: the readers all
            // find the closed queue in one wake, and the loop ends
            // before it asks again. A slow server makes the difference
            // that a fake one cannot. (§3.4)
            let grow = asking
                && !give.is_closed()
                && readers.len() < want
                && !hold(&queue).is_empty();

            if !grow {
                spare = spare.or(readers.next().await.flatten());

                continue;
            }

            tokio::select! {
                biased;

                // One connection stays for the flags of §3.3. The
                // others go back to the pool now, and not at the end,
                // for the folders that still read. (§3.1)
                done = readers.next() => spare = spare.or(done.flatten()),

                got = pool.take() => match got {
                    Ok(one) => {
                        readers.push(reader(one, job, &queue, give.clone()));
                    }
                    // A pool that is only busy makes `take` wait, so
                    // this is a pool that broke. The folder stops
                    // asking, because a second ask breaks the same way
                    // and the loop would spin. (§3.1)
                    Err(problem) => {
                        tracing::debug!(
                            %problem,
                            "the pool spared no connection"
                        );

                        asking = false;
                    }
                },
            }
        }

        Ok::<Option<Held<'_>>, Error>(spare)
    };

    let writing = async {
        let mut state = job.start.clone();

        while let Some(got) = take.recv().await {
            let problem = match got {
                Err(problem) => Some(problem),
                Ok((at, batch)) => {
                    // The state names the batches that landed, and
                    // never one that a connection is still reading.
                    // (§3.4)
                    state = job.after(&state, &job.batches[at]);
                    keep.batch(&job.folder, batch, &state).await.err()
                }
            };

            let Some(problem) = problem else {
                continue;
            };

            // Nothing will take another batch. The connections must
            // learn that, or they wait for a reader that never comes.
            // The batches that already arrived go nowhere, and the
            // folder owes them again. (§3.4)
            take.close();
            while take.recv().await.is_some() {}

            return Err(problem);
        }

        Ok::<FolderState, Error>(state)
    };

    let (held, wrote) = tokio::join!(reading, writing);
    let mut spare = held?;
    state = wrote?;

    // The messages that the store holds already: a flag that changed,
    // or a message that went away. One connection is enough, because
    // the plan asks this one time. (§3.3)
    if let (Some(changed), Some(since)) = (&job.changed, job.since) {
        // Every reader gave its connection back, because the folder
        // read every batch of the queue.
        let one = spare.as_mut().expect("a connection that read the folder");
        let batch = match hand(one, &job.folder, changed, Some(since)).await {
            Ok(batch) => batch,
            Err(problem) => {
                one.retire();

                return Err(problem);
            }
        };

        keep.batch(&job.folder, batch, &state).await?;
    }

    Ok(state)
}

/// Read batches from the queue, until the queue is empty. (§3.1)
///
/// The connection comes back, so the folder can use it again for the
/// flags of §3.3. A connection that broke goes nowhere. (§3.4)
async fn reader<'a>(
    mut one: Held<'a>,
    job: &Job,
    queue: &Mutex<VecDeque<usize>>,
    give: mpsc::Sender<Result<(usize, Batch)>>,
) -> Option<Held<'a>> {
    loop {
        // The lock never crosses an await, so a connection that waits
        // for the server holds up no other one.
        let next = hold(queue).pop_front();
        let Some(at) = next else {
            break;
        };

        match hand(&mut one, &job.folder, &job.batches[at], None).await {
            Ok(batch) => {
                if give.send(Ok((at, batch))).await.is_err() {
                    break;
                }
            }
            Err(problem) => {
                // The answer stopped in the middle, so bytes that
                // belong to no command may be on the socket. It never
                // goes back to the pool.
                one.retire();
                let _ = give.send(Err(problem)).await;

                return None;
            }
        }
    }

    Some(one)
}

/// Read one set on one connection, and open the folder if it is not.
async fn hand(
    one: &mut Held<'_>,
    folder: &str,
    set: &UidSet,
    since: Option<u64>,
) -> Result<Batch> {
    if one.selected() != Some(folder) {
        one.examine(folder).await?;
    }

    one.fetch(set, since).await
}

/// Read one set, while the store writes the batch that waits. (§3.4)
///
/// The two run at the same time, and both run to the end. A write that
/// fails therefore never leaves bytes of a half-read answer on the
/// socket, and the connection stays whole for the pool.
///
/// The write goes first when it fails, because a batch that the store
/// refused is a batch that the folder still owes. The mail that the
/// read brought is then dropped, and the next sync asks for it again.
async fn read<K: Keep>(
    connection: &mut Connection,
    keep: &mut K,
    folder: &str,
    set: &UidSet,
    since: Option<u64>,
    waiting: Option<(Batch, FolderState)>,
) -> Result<Batch> {
    let Some((batch, at)) = waiting else {
        return connection.fetch(set, since).await;
    };

    let (got, wrote) = tokio::join!(
        connection.fetch(set, since),
        keep.batch(folder, batch, &at),
    );
    wrote?;

    got
}

/// How long a sync waits before it tries again. (§3.4)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Backoff {
    /// How many times a sync of one folder starts. Always one or more.
    pub tries: usize,
    /// The wait after the first failure.
    pub first: Duration,
    /// The longest wait.
    pub most: Duration,
}

impl Default for Backoff {
    fn default() -> Self {
        Self {
            tries: 4,
            first: Duration::from_secs(1),
            most: Duration::from_secs(30),
        }
    }
}

impl Backoff {
    /// The wait after this many failures. Each wait is twice the last.
    pub fn wait(&self, failures: usize) -> Duration {
        let steps =
            u32::try_from(failures.saturating_sub(1)).unwrap_or(u32::MAX);
        let times = 1u32.checked_shl(steps).unwrap_or(u32::MAX);
        let Some(wait) = self.first.checked_mul(times) else {
            return self.most;
        };

        wait.min(self.most)
    }
}

/// True when a sync that tries again could do better. (§3.4)
///
/// A server that says NO to a command says it again the next time.
/// A connection that broke is worth another try.
pub fn again(error: &Error) -> bool {
    matches!(
        error,
        Error::Io(_) | Error::Closed | Error::Refused(_) | Error::Malformed(_)
    )
}

/// Do one job, and start again when a connection breaks. (§3.4)
///
/// Each new try reads the state out of the store, so it asks for no
/// message that arrived before the failure.
pub async fn resume<K: Keep>(
    pool: &Pool,
    job: &Job,
    keep: &mut K,
    back: Backoff,
) -> Result<FolderState> {
    let mut job = job.clone();
    let mut failures = 0;

    loop {
        let error = match once(pool, &job, keep).await {
            Ok(state) => return Ok(state),
            Err(error) => error,
        };

        failures += 1;
        if failures >= back.tries.max(1) || !again(&error) {
            return Err(error);
        }

        sleep(back.wait(failures)).await;
        job = job.left(&keep.state(&job.folder).await?);
    }
}

/// One try of one job, on one connection of the pool.
async fn once<K: Keep>(
    pool: &Pool,
    job: &Job,
    keep: &mut K,
) -> Result<FolderState> {
    // The folder asks for the whole pool, and takes what is free. A
    // mailbox that keeps its mail in one folder then reads that folder
    // on every connection. A mailbox of many folders gives one
    // connection to each, because each folder asks first and the pool
    // is empty when the next one asks. (§3.1)
    spread(pool, job, keep, pool.limit()).await
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_plan_asks_for_every_uid_that_is_missing` | model-based | A plan that drops a UID drops a message, and the user never learns of it. |
    //! | `prop_a_plan_never_asks_twice_for_a_uid` | algebraic | §3.1 counts its time in bytes. A UID that arrives twice is a message that is paid for twice. |
    //! | `prop_a_second_sync_of_the_same_folder_asks_for_nothing` | metamorphic | §3.3 exists so a sync of a folder that did not change costs nothing. |
    //! | `prop_a_uid_validity_that_changed_asks_for_the_whole_folder` | metamorphic | RFC 3501 §2.3.1.1 makes the old UIDs meaningless. A plan that keeps them reads the wrong mail. |
    //! | `prop_a_folder_only_finishes_when_every_batch_arrived` | model-based | §3.4 resumes from the state on disk. A state that finishes early loses every batch that never came. |
    //! | `prop_a_sync_that_stops_anywhere_still_reads_every_message` | metamorphic | §3.4 promises that a sync which stops continues. A message that a failure hides is a message the user never sees. |
    //! | `prop_a_cut_plan_asks_for_every_uid_that_the_server_holds` | model-based | §3.2 cuts the plan down to the mail that the server has. A cut that drops a UID the server holds loses mail, and one that keeps a UID the server lost pays for a round trip that brings nothing. |
    //! | `prop_cutting_a_plan_twice_is_cutting_it_one_time` | algebraic | A plan that shrinks each time somebody reads it loses mail after a retry. |
    //! | `prop_the_order_of_the_batches_never_changes_the_state` | algebraic | §3.1 gives one folder to every connection, so the batches land in the order that the server answers. A fold that depends on that order would leave a different state each run. |
    //! | `prop_the_state_after_a_batch_never_moves_again` | algebraic | §3.4 reads the next batch while the store writes this one, so the state of a batch is computed once and carried. A state that moved when somebody read it twice would land on the wrong batch. |

    use hegel::{TestCase, generators as gs};

    use super::*;
    use crate::{
        fake::{FakeFolder, FakeMessage, FakeServer, Plan},
        pool::{Pool, Server},
    };

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    const SIZE: u32 = 4;

    fn a_view(validity: u32, next: u32, mod_seq: u64) -> View {
        View {
            name: "INBOX".to_string(),
            exists: next.saturating_sub(1),
            uid_validity: validity,
            uid_next: next,
            highest_mod_seq: mod_seq,
        }
    }

    fn whole(sets: &[UidSet]) -> UidSet {
        sets.iter().fold(UidSet::new(), |all, set| all.union(set))
    }

    /// A sink that keeps every call, and can fail on demand. (§3.4)
    #[derive(Debug, Default)]
    struct Recorder {
        marks: Vec<FolderState>,
        batches: Vec<Batch>,
        states: Vec<FolderState>,
        stop_after: Option<usize>,
        held: FolderState,
    }

    impl Keep for Recorder {
        async fn mark(
            &mut self,
            _folder: &str,
            state: &FolderState,
        ) -> Result<()> {
            self.marks.push(state.clone());

            Ok(())
        }

        async fn batch(
            &mut self,
            _folder: &str,
            batch: Batch,
            state: &FolderState,
        ) -> Result<()> {
            if self.stop_after == Some(self.batches.len()) {
                return Err(Error::Closed);
            }
            self.batches.push(batch);
            self.states.push(state.clone());
            self.held = state.clone();

            Ok(())
        }

        async fn state(&mut self, _folder: &str) -> Result<FolderState> {
            Ok(self.held.clone())
        }
    }

    /// A store that holds what arrived, and fails once. (§3.4)
    #[derive(Debug, Default)]
    struct Store {
        held: FolderState,
        uids: Vec<u32>,
        gone: Vec<u32>,
        calls: usize,
        fail_at: Option<usize>,
    }

    impl Keep for Store {
        async fn mark(
            &mut self,
            _folder: &str,
            state: &FolderState,
        ) -> Result<()> {
            self.held = state.clone();

            Ok(())
        }

        async fn batch(
            &mut self,
            _folder: &str,
            batch: Batch,
            state: &FolderState,
        ) -> Result<()> {
            let at = self.calls;
            self.calls += 1;
            if self.fail_at == Some(at) {
                return Err(Error::Closed);
            }

            self.uids.extend(batch.messages.iter().map(|held| held.uid));
            self.gone.extend(&batch.gone);
            self.held = state.clone();

            Ok(())
        }

        async fn state(&mut self, _folder: &str) -> Result<FolderState> {
            Ok(self.held.clone())
        }
    }

    fn quick() -> Backoff {
        Backoff {
            tries: 4,
            first: Duration::from_millis(1),
            most: Duration::from_millis(4),
        }
    }

    fn a_folder(count: u32) -> FakeFolder {
        a_folder_named("INBOX", count)
    }

    fn a_folder_named(name: &str, count: u32) -> FakeFolder {
        let mut folder = FakeFolder::new(name).with_uid_validity(77);

        for uid in 1..=count {
            folder = folder.with(
                FakeMessage::new(
                    uid,
                    &format!("Subject: {uid}\r\n\r\nbody\r\n"),
                )
                .with_mod_seq(u64::from(uid)),
            );
        }

        folder
    }

    async fn a_pool(server: &FakeServer) -> Pool {
        Pool::new(
            Server::at("127.0.0.1", server.port(), false)
                .with_login("me", "secret"),
        )
    }

    fn run_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    // -----------------------------------------------------------------
    // The plan, cut down to the mail that the server holds. (§3.2)
    // -----------------------------------------------------------------

    /// The point of T33. A folder that lost most of its mail spans a
    /// wide range of UIDs, and a batch of the holes costs a round trip
    /// that brings nothing.
    #[test]
    fn a_plan_asks_only_for_the_uids_that_the_server_holds() {
        let held = UidSet::parse("2:3,90").unwrap();
        let job =
            plan("INBOX", &FolderState::default(), &a_view(77, 100, 9), 2)
                .only(&held);

        assert_eq!(whole(&job.batches), held);
        assert_eq!(job.count(), 3);
        assert_eq!(job.start.pending, held);
    }

    /// The state must still say where the folder got to, or the next
    /// sync reads the whole folder again. (§3.4)
    #[test]
    fn a_plan_that_was_cut_down_keeps_the_place_of_the_folder() {
        let whole_plan =
            plan("INBOX", &FolderState::default(), &a_view(77, 100, 9), SIZE);
        let job = whole_plan.clone().only(&UidSet::parse("2:3").unwrap());

        assert_eq!(job.done, whole_plan.done);
        assert_eq!(job.start.uid_next, whole_plan.start.uid_next);
        assert_eq!(job.start.uid_validity, whole_plan.start.uid_validity);
        assert_eq!(job.restart, whole_plan.restart);
    }

    #[test]
    fn a_server_that_holds_nothing_leaves_no_batch() {
        let job =
            plan("INBOX", &FolderState::default(), &a_view(77, 100, 9), SIZE)
                .only(&UidSet::new());

        assert!(job.is_empty());
        assert_eq!(job.count(), 0);
        assert!(job.start.pending.is_empty());
    }

    /// A plan that the server cut down to nothing is a folder that is
    /// done. `run` marks `start` and never runs a batch, so `start`
    /// must carry the place that the server showed. A folder that
    /// keeps the old `MODSEQ` asks the same question every sync. (§3.3)
    #[test]
    fn a_plan_cut_down_to_nothing_says_the_folder_is_done() {
        let saved = FolderState {
            uid_validity: 77,
            uid_next: 11,
            highest_mod_seq: 9,
            pending: UidSet::new(),
        };
        let job = plan("INBOX", &saved, &a_view(77, 14, 20), SIZE)
            .only(&UidSet::new());

        // The job still asks which flags moved, so it is not empty.
        // It fetches no body, and `run` marks `start` as it is.
        assert!(job.batches.is_empty());
        assert_eq!(
            job.start.highest_mod_seq, job.done.highest_mod_seq,
            "the folder never moves past this MODSEQ"
        );
    }

    /// A plan that still holds a batch must not move ahead of it.
    #[test]
    fn a_plan_that_still_owes_a_batch_keeps_the_old_place() {
        let saved = FolderState {
            uid_validity: 77,
            uid_next: 11,
            highest_mod_seq: 9,
            pending: UidSet::new(),
        };
        let job = plan("INBOX", &saved, &a_view(77, 14, 20), SIZE)
            .only(&UidSet::parse("12").unwrap());

        assert!(!job.batches.is_empty());
        assert_eq!(job.start.highest_mod_seq, 9);
    }

    /// A folder that lost most of its mail is the one case that pays
    /// for the search. (§3.2)
    #[test]
    fn a_folder_of_holes_is_worth_a_search() {
        // 60000 UIDs, and 100 messages left.
        let job = plan(
            "INBOX",
            &FolderState::default(),
            &a_view(77, 60_001, 9),
            500,
        );

        assert!(job.mostly_holes(100));
    }

    /// The common sync owes a few messages, and the search would cost
    /// more than the batches that it saves.
    #[test]
    fn a_folder_that_owes_a_little_is_not_worth_a_search() {
        let saved = FolderState {
            uid_validity: 77,
            uid_next: 11,
            highest_mod_seq: 9,
            pending: UidSet::new(),
        };
        let job = plan("INBOX", &saved, &a_view(77, 14, 9), 500);

        assert!(!job.mostly_holes(13));
    }

    /// A first sync of a folder that lost nothing has no holes, so the
    /// search would name every UID and change no batch.
    #[test]
    fn a_folder_with_no_holes_is_not_worth_a_search() {
        let job = plan(
            "INBOX",
            &FolderState::default(),
            &a_view(77, 60_001, 9),
            500,
        );

        assert!(!job.mostly_holes(60_000));
    }

    /// The saving must be a batch or more, or the search pays nothing.
    #[test]
    fn a_folder_with_a_few_holes_is_not_worth_a_search() {
        let job =
            plan("INBOX", &FolderState::default(), &a_view(77, 1_001, 9), 500);

        assert!(!job.mostly_holes(900));

        // Exactly one batch of holes pays for the search.
        assert!(job.mostly_holes(500));
        assert!(job.mostly_holes(400));
    }

    /// A server that says a folder holds nothing leaves the plan whole
    /// only when the plan is already empty.
    #[test]
    fn an_empty_plan_is_not_worth_a_search() {
        let job =
            plan("INBOX", &FolderState::default(), &a_view(77, 1, 0), 500);

        assert!(!job.mostly_holes(0));
    }

    /// A server that holds every UID must leave the plan alone.
    #[test]
    fn a_server_that_holds_everything_changes_no_plan() {
        let job =
            plan("INBOX", &FolderState::default(), &a_view(77, 11, 9), SIZE);
        let cut = job.clone().only(&UidSet::parse("1:10").unwrap());

        assert_eq!(whole(&cut.batches), whole(&job.batches));
        assert_eq!(cut.start.pending, job.start.pending);
    }

    /// A UID that the plan never wanted must not join it. The answer
    /// of the server is a filter, and never a source of work.
    #[test]
    fn a_uid_that_the_plan_never_wanted_stays_out() {
        let saved = FolderState {
            uid_validity: 77,
            uid_next: 11,
            highest_mod_seq: 9,
            pending: UidSet::new(),
        };
        let job = plan("INBOX", &saved, &a_view(77, 14, 9), SIZE)
            .only(&UidSet::parse("1:13").unwrap());

        assert_eq!(whole(&job.batches), UidSet::parse("11:13").unwrap());
    }

    /// §3.3 reads the flags of the mail that the store already holds.
    /// That question is about the store, and not about what a fetch
    /// can bring, so a cut plan must still ask it.
    #[test]
    fn a_cut_plan_still_asks_which_flags_moved() {
        let saved = FolderState {
            uid_validity: 77,
            uid_next: 11,
            highest_mod_seq: 9,
            pending: UidSet::new(),
        };
        let job = plan("INBOX", &saved, &a_view(77, 14, 20), SIZE);
        let cut = job
            .clone()
            .only(&UidSet::parse("13".to_string().as_str()).unwrap());

        assert_eq!(cut.changed, job.changed);
        assert_eq!(cut.since, job.since);
    }

    // -----------------------------------------------------------------
    // The plan. (§3.3)
    // -----------------------------------------------------------------

    #[test]
    fn a_first_sync_asks_for_every_message() {
        let job =
            plan("INBOX", &FolderState::default(), &a_view(77, 11, 9), SIZE);

        assert!(job.restart);
        assert_eq!(whole(&job.batches), UidSet::parse("1:10").unwrap());
        assert_eq!(job.changed, None);
        assert_eq!(job.start.uid_next, 11);
        assert_eq!(job.start.pending, UidSet::parse("1:10").unwrap());
        assert_eq!(job.done.highest_mod_seq, 9);
        assert!(job.done.pending.is_empty());
    }

    #[test]
    fn a_first_sync_of_an_empty_folder_asks_for_nothing() {
        let job =
            plan("INBOX", &FolderState::default(), &a_view(77, 1, 0), SIZE);

        assert!(job.is_empty());
        assert_eq!(job.count(), 0);
    }

    #[test]
    fn a_second_sync_asks_only_for_the_new_messages() {
        let saved = FolderState {
            uid_validity: 77,
            uid_next: 11,
            highest_mod_seq: 9,
            pending: UidSet::new(),
        };
        let job = plan("INBOX", &saved, &a_view(77, 14, 9), SIZE);

        assert!(!job.restart);
        assert_eq!(whole(&job.batches), UidSet::parse("11:13").unwrap());
    }

    #[test]
    fn a_second_sync_asks_again_for_what_never_arrived() {
        let saved = FolderState {
            uid_validity: 77,
            uid_next: 11,
            highest_mod_seq: 9,
            pending: UidSet::parse("2:4").unwrap(),
        };
        let job = plan("INBOX", &saved, &a_view(77, 14, 9), SIZE);

        assert_eq!(whole(&job.batches), UidSet::parse("2:4,11:13").unwrap());
    }

    #[test]
    fn a_sync_of_a_folder_that_did_not_change_asks_for_nothing() {
        let saved = FolderState {
            uid_validity: 77,
            uid_next: 11,
            highest_mod_seq: 9,
            pending: UidSet::new(),
        };
        let job = plan("INBOX", &saved, &a_view(77, 11, 9), SIZE);

        assert!(job.is_empty());
        assert_eq!(job.start, saved);
    }

    #[test]
    fn a_uid_validity_that_changed_starts_the_folder_again() {
        let saved = FolderState {
            uid_validity: 77,
            uid_next: 11,
            highest_mod_seq: 9,
            pending: UidSet::parse("3").unwrap(),
        };
        let job = plan("INBOX", &saved, &a_view(78, 6, 2), SIZE);

        assert!(job.restart);
        assert_eq!(whole(&job.batches), UidSet::parse("1:5").unwrap());
        assert_eq!(job.changed, None, "the old MODSEQ means nothing now");
        assert_eq!(job.done.uid_validity, 78);
    }

    #[test]
    fn a_higher_mod_seq_asks_about_the_messages_that_arrived_before() {
        let saved = FolderState {
            uid_validity: 77,
            uid_next: 11,
            highest_mod_seq: 9,
            pending: UidSet::new(),
        };
        let job = plan("INBOX", &saved, &a_view(77, 14, 20), SIZE);

        assert_eq!(job.changed, Some(UidSet::parse("1:10").unwrap()));
        assert_eq!(job.since, Some(9));
    }

    #[test]
    fn a_mod_seq_that_did_not_move_asks_about_nothing() {
        let saved = FolderState {
            uid_validity: 77,
            uid_next: 11,
            highest_mod_seq: 9,
            pending: UidSet::new(),
        };
        let job = plan("INBOX", &saved, &a_view(77, 14, 9), SIZE);

        assert_eq!(job.changed, None);
        assert_eq!(job.since, None);
    }

    #[test]
    fn a_server_without_condstore_asks_about_nothing() {
        let saved = FolderState {
            uid_validity: 77,
            uid_next: 11,
            highest_mod_seq: 0,
            pending: UidSet::new(),
        };
        let job = plan("INBOX", &saved, &a_view(77, 14, 0), SIZE);

        assert_eq!(job.changed, None);
        assert_eq!(whole(&job.batches), UidSet::parse("11:13").unwrap());
    }

    #[test]
    fn the_batches_of_a_plan_hold_no_more_than_the_size() {
        let job =
            plan("INBOX", &FolderState::default(), &a_view(77, 31, 0), SIZE);

        assert!(job.batches.iter().all(|set| set.count() <= u64::from(SIZE)));
        assert_eq!(job.count(), 30);
    }

    // -----------------------------------------------------------------
    // The state after a batch. (§3.4)
    // -----------------------------------------------------------------

    #[test]
    fn a_batch_that_arrived_leaves_the_state_with_less_to_do() {
        let job =
            plan("INBOX", &FolderState::default(), &a_view(77, 11, 9), SIZE);
        let after = job.after(&job.start, &job.batches[0]);

        assert_eq!(after.pending, UidSet::parse("1:6").unwrap());
        assert_eq!(after.uid_next, 11);
        assert_eq!(
            after.highest_mod_seq, 0,
            "the sequence moves only when the folder is whole"
        );
    }

    #[test]
    fn the_last_batch_of_a_folder_finishes_it() {
        let job =
            plan("INBOX", &FolderState::default(), &a_view(77, 11, 9), SIZE);
        let mut state = job.start.clone();

        for batch in &job.batches {
            state = job.after(&state, batch);
        }

        assert_eq!(state, job.done);
    }

    // -----------------------------------------------------------------
    // Run a job. (§3.3)
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn a_job_reads_every_message_of_the_folder() {
        let server = FakeServer::start(Plan::new().with(a_folder(10)))
            .await
            .unwrap();
        let pool = a_pool(&server).await;
        let mut held = pool.take().await.unwrap();

        let view = held.examine("INBOX").await.unwrap();
        let job = plan("INBOX", &FolderState::default(), &view, SIZE);
        let mut keep = Recorder::default();
        let state = run(&mut held, &job, &mut keep).await.unwrap();

        let uids: Vec<u32> = keep
            .batches
            .iter()
            .flat_map(|batch| batch.messages.iter().map(|held| held.uid))
            .collect();

        assert_eq!(uids.len(), 10);
        assert_eq!(state, job.done);
        assert_eq!(keep.marks.len(), 1);
        assert_eq!(keep.marks[0], job.start);
    }

    #[tokio::test]
    async fn a_job_opens_the_folder_when_the_connection_did_not() {
        let server = FakeServer::start(Plan::new().with(a_folder(3)))
            .await
            .unwrap();
        let pool = a_pool(&server).await;
        let mut held = pool.take().await.unwrap();
        let job =
            plan("INBOX", &FolderState::default(), &a_view(77, 4, 3), SIZE);
        let mut keep = Recorder::default();
        run(&mut held, &job, &mut keep).await.unwrap();

        assert_eq!(held.selected(), Some("INBOX"));
        assert_eq!(keep.batches[0].messages.len(), 3);
    }

    #[tokio::test]
    async fn a_job_with_nothing_to_do_reads_nothing() {
        let server = FakeServer::start(Plan::new().with(a_folder(3)))
            .await
            .unwrap();
        let pool = a_pool(&server).await;
        let mut held = pool.take().await.unwrap();
        let view = held.examine("INBOX").await.unwrap();
        let saved = FolderState {
            uid_validity: view.uid_validity,
            uid_next: view.uid_next,
            highest_mod_seq: view.highest_mod_seq,
            pending: UidSet::new(),
        };
        let job = plan("INBOX", &saved, &view, SIZE);
        let mut keep = Recorder::default();
        run(&mut held, &job, &mut keep).await.unwrap();

        assert!(keep.batches.is_empty());
        assert_eq!(keep.marks.len(), 1);
    }

    #[tokio::test]
    async fn a_job_that_stops_keeps_what_arrived() {
        let server = FakeServer::start(Plan::new().with(a_folder(10)))
            .await
            .unwrap();
        let pool = a_pool(&server).await;
        let mut held = pool.take().await.unwrap();
        let view = held.examine("INBOX").await.unwrap();
        let job = plan("INBOX", &FolderState::default(), &view, SIZE);

        let mut keep = Recorder {
            stop_after: Some(1),
            ..Recorder::default()
        };
        assert!(run(&mut held, &job, &mut keep).await.is_err());

        // The one batch that arrived is out of the debt, and the rest
        // of the folder is still owed. (§3.4)
        let held_state = keep.states.last().unwrap().clone();
        assert_eq!(held_state.pending, UidSet::parse("1:6").unwrap());

        let next = plan("INBOX", &held_state, &view, SIZE);
        assert_eq!(whole(&next.batches), UidSet::parse("1:6").unwrap());
    }

    // -----------------------------------------------------------------
    // The socket reads while the store writes. (§3.4)
    // -----------------------------------------------------------------

    /// How many batches the server was asked for.
    fn fetches(server: &FakeServer) -> usize {
        server
            .seen()
            .commands
            .iter()
            .filter(|line| line.contains("UID FETCH"))
            .count()
    }

    /// A sink that will not finish a write before the socket asked for
    /// the batch that follows. (§3.4)
    ///
    /// A sync that reads and writes one after the other can never
    /// satisfy it. The fetch waits for the write, and the write waits
    /// for the fetch. The sink gives up after a moment, and `run` then
    /// gives an error.
    struct Overlapping<'a> {
        server: &'a FakeServer,
        writes: usize,
        of: usize,
    }

    impl Keep for Overlapping<'_> {
        async fn mark(
            &mut self,
            _folder: &str,
            _state: &FolderState,
        ) -> Result<()> {
            Ok(())
        }

        async fn batch(
            &mut self,
            _folder: &str,
            _batch: Batch,
            _state: &FolderState,
        ) -> Result<()> {
            self.writes += 1;

            // No read follows the last write, so it waits for nothing.
            if self.writes >= self.of {
                return Ok(());
            }

            let want = self.writes + 1;
            for _ in 0..400 {
                if fetches(self.server) >= want {
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }

            Err(Error::Closed)
        }

        async fn state(&mut self, _folder: &str) -> Result<FolderState> {
            Ok(FolderState::default())
        }
    }

    #[tokio::test]
    async fn a_job_reads_the_next_batch_while_the_store_writes_the_last() {
        let server = FakeServer::start(Plan::new().with(a_folder(10)))
            .await
            .unwrap();
        let pool = a_pool(&server).await;
        let mut held = pool.take().await.unwrap();

        let view = held.examine("INBOX").await.unwrap();
        let job = plan("INBOX", &FolderState::default(), &view, SIZE);
        assert_eq!(job.batches.len(), 3, "the test wants more than one batch");

        let mut keep = Overlapping {
            server: &server,
            writes: 0,
            of: job.batches.len(),
        };

        run(&mut held, &job, &mut keep).await.expect(
            "the socket never asked for a batch while the store wrote one",
        );
    }

    /// The state that lands with a batch must be the state after that
    /// batch, and never the state after the batch that follows. A read
    /// that runs ahead must not carry the state ahead with it. (§3.4)
    #[tokio::test]
    async fn the_state_that_lands_with_a_batch_names_that_batch() {
        let server = FakeServer::start(Plan::new().with(a_folder(10)))
            .await
            .unwrap();
        let pool = a_pool(&server).await;
        let mut held = pool.take().await.unwrap();

        let view = held.examine("INBOX").await.unwrap();
        let job = plan("INBOX", &FolderState::default(), &view, SIZE);
        let mut keep = Recorder::default();
        run(&mut held, &job, &mut keep).await.unwrap();

        for (at, batch) in keep.batches.iter().enumerate() {
            let state = &keep.states[at];

            for message in &batch.messages {
                assert!(
                    !state.pending.holds(message.uid),
                    "the state still owes {} of its own batch",
                    message.uid
                );
            }

            for later in &keep.batches[at + 1..] {
                for message in &later.messages {
                    assert!(
                        state.pending.holds(message.uid),
                        "the state gave up {} before it arrived",
                        message.uid
                    );
                }
            }
        }
    }

    /// A sync that stops loses only what was in the air. The mail that
    /// the store wrote is out of the debt for good. (§3.4)
    #[tokio::test]
    async fn a_job_that_stops_never_asks_again_for_what_the_store_wrote() {
        let server = FakeServer::start(Plan::new().with(a_folder(10)))
            .await
            .unwrap();
        let pool = a_pool(&server).await;
        let mut held = pool.take().await.unwrap();

        let view = held.examine("INBOX").await.unwrap();
        let job = plan("INBOX", &FolderState::default(), &view, SIZE);

        let mut keep = Recorder {
            stop_after: Some(2),
            ..Recorder::default()
        };
        assert!(run(&mut held, &job, &mut keep).await.is_err());
        assert_eq!(keep.batches.len(), 2, "the store took two batches");

        let last = keep.states.last().unwrap().clone();
        let asks = whole(&plan("INBOX", &last, &view, SIZE).batches);

        for batch in &keep.batches {
            for message in &batch.messages {
                assert!(
                    !asks.holds(message.uid),
                    "the sync asks again for {}, and the store has it",
                    message.uid
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // Every connection reads the same folder. (§3.1)
    // -----------------------------------------------------------------

    /// A folder of 40 messages, in batches of 4, is ten batches. One
    /// connection reads them one at a time, and four read them four at
    /// a time.
    async fn a_spread(
        server: &FakeServer,
        hands: usize,
    ) -> (Pool, Job, Recorder) {
        let pool = a_pool_of(server, hands);
        let mut held = pool.take().await.unwrap();
        let view = held.examine("INBOX").await.unwrap();
        drop(held);

        let job = plan("INBOX", &FolderState::default(), &view, SIZE);

        (pool, job, Recorder::default())
    }

    fn a_pool_of(server: &FakeServer, connections: usize) -> Pool {
        Pool::new(
            Server::at("127.0.0.1", server.port(), false)
                .with_login("me", "secret")
                .with_connections(connections),
        )
    }

    #[tokio::test]
    async fn a_folder_that_spreads_reads_every_message() {
        let server = FakeServer::start(Plan::new().with(a_folder(40)))
            .await
            .unwrap();
        let (pool, job, mut keep) = a_spread(&server, 4).await;
        assert_eq!(job.batches.len(), 10, "the test wants many batches");

        let state = spread(&pool, &job, &mut keep, 4).await.unwrap();

        let mut uids: Vec<u32> = keep
            .batches
            .iter()
            .flat_map(|batch| batch.messages.iter().map(|held| held.uid))
            .collect();
        uids.sort_unstable();

        assert_eq!(uids, (1..=40).collect::<Vec<u32>>());
        assert_eq!(state, job.done, "the folder never finished");
    }

    #[tokio::test]
    async fn a_folder_that_spreads_reads_on_every_connection() {
        let server = FakeServer::start(Plan::new().with(a_folder(40)))
            .await
            .unwrap();
        let (pool, job, mut keep) = a_spread(&server, 4).await;

        spread(&pool, &job, &mut keep, 4).await.unwrap();

        assert_eq!(
            server.seen().most_open,
            4,
            "the folder left connections idle"
        );
    }

    /// The state must move only for a batch that landed, whatever
    /// order the connections answer in. (§3.1)
    #[tokio::test]
    async fn a_folder_that_spreads_owes_every_batch_that_never_landed() {
        let server = FakeServer::start(Plan::new().with(a_folder(40)))
            .await
            .unwrap();
        let (pool, job, _) = a_spread(&server, 4).await;

        let mut keep = Recorder {
            stop_after: Some(3),
            ..Recorder::default()
        };
        assert!(spread(&pool, &job, &mut keep, 4).await.is_err());

        let last = keep.states.last().unwrap().clone();
        let wrote: Vec<u32> = keep
            .batches
            .iter()
            .flat_map(|batch| batch.messages.iter().map(|held| held.uid))
            .collect();

        for uid in 1..=40 {
            assert_eq!(
                !last.pending.holds(uid),
                wrote.contains(&uid),
                "the state and the store disagree about {uid}"
            );
        }
    }

    #[tokio::test]
    async fn a_folder_that_spreads_marks_its_start_one_time() {
        let server = FakeServer::start(Plan::new().with(a_folder(40)))
            .await
            .unwrap();
        let (pool, job, mut keep) = a_spread(&server, 4).await;

        spread(&pool, &job, &mut keep, 4).await.unwrap();

        assert_eq!(keep.marks.len(), 1);
        assert_eq!(keep.marks[0], job.start);
    }

    /// True when the server answered a fetch already.
    fn saw_a_fetch(server: &FakeServer) -> bool {
        server
            .seen()
            .commands
            .iter()
            .any(|line| line.contains("FETCH"))
    }

    /// A job for one folder of the fake server.
    async fn a_job(pool: &Pool, folder: &str) -> Job {
        let mut held = pool.take().await.unwrap();
        let view = held.examine(folder).await.unwrap();
        drop(held);

        plan(folder, &FolderState::default(), &view, SIZE)
    }

    /// §3.1: a folder that still owes batches must ask for a
    /// connection again. A folder that asks one time reads the rest of
    /// a big mailbox on the connection that it took at the start,
    /// while the connections of the folders that ended sit idle.
    #[tokio::test]
    async fn a_folder_that_spreads_asks_for_a_connection_again() {
        let server = FakeServer::start(
            Plan::new()
                .with(a_folder(48))
                .slow(Duration::from_millis(20)),
        )
        .await
        .unwrap();
        let (pool, job, mut keep) = a_spread(&server, 2).await;
        assert_eq!(job.batches.len(), 12, "the test wants many batches");

        // The pool holds two connections, and this task holds one of
        // them. The folder starts with the other one, and it can take
        // no second connection.
        let mut keeper = pool.take().await.unwrap();

        // The connection never goes back to the pool, so a folder that
        // asks again opens a new one, and the server counts it.
        keeper.retire();

        let (state, ()) =
            tokio::join!(spread(&pool, &job, &mut keep, 2), async {
                // The connection comes free after the folder took what
                // there was to take. The folder reads by then, so what it
                // opens after this it opened because it asked again.
                while !saw_a_fetch(&server) {
                    sleep(Duration::from_millis(1)).await;
                }

                drop(keeper);
            });

        state.unwrap();
        assert_eq!(
            server.seen().connections,
            3,
            "the folder never took the connection that came free"
        );
    }

    /// §3.1: a folder that ends gives its connections back, and a
    /// folder that still reads must take them. A big folder that reads
    /// alone costs the whole mailbox on one connection.
    #[tokio::test]
    async fn a_folder_that_ends_gives_its_connection_to_one_that_runs() {
        let wait = Duration::from_millis(30);
        let server = FakeServer::start(
            Plan::new()
                .with(a_folder(48))
                .with(a_folder_named("Sent", 4))
                .slow(wait),
        )
        .await
        .unwrap();
        let pool = a_pool_of(&server, 2);
        let big = a_job(&pool, "INBOX").await;
        let small = a_job(&pool, "Sent").await;
        let mut one = Recorder::default();
        let mut two = Recorder::default();

        assert_eq!(big.batches.len(), 12, "the test wants many batches");
        assert_eq!(small.batches.len(), 1, "the test wants one batch");

        // The small folder takes a connection first, and the big
        // folder then starts with the one that is left. A big folder
        // that starts first takes them both and starves the other.
        let start = std::time::Instant::now();
        let (read_small, read_big) = tokio::join!(
            spread(&pool, &small, &mut two, 2),
            spread(&pool, &big, &mut one, 2),
        );
        let took = start.elapsed();

        read_big.unwrap();
        read_small.unwrap();

        // Twelve batches on one connection cost 360 ms. The small
        // folder ends after one batch, and the big folder reads the
        // rest of its own on two connections.
        assert!(
            took < wait * 9,
            "the big folder read alone, and it took {took:?}"
        );
    }

    /// §3.4: a folder that failed asks for no more connections, and it
    /// reads no more batches. The connections belong to the folders
    /// that still read.
    #[tokio::test]
    async fn a_folder_that_failed_reads_no_more_batches() {
        let server = FakeServer::start(Plan::new().with(a_folder(200)))
            .await
            .unwrap();
        let (pool, job, _) = a_spread(&server, 4).await;
        let mut keep = Recorder {
            stop_after: Some(1),
            ..Recorder::default()
        };
        assert_eq!(job.batches.len(), 50, "the test wants many batches");

        assert!(spread(&pool, &job, &mut keep, 4).await.is_err());

        let fetches = server
            .seen()
            .commands
            .iter()
            .filter(|line| line.contains("FETCH"))
            .count();

        assert!(fetches < 20, "the folder read {fetches} batches");
    }

    /// A folder that asks for more connections than the pool holds
    /// takes what there is, and still reads every message. (§3.1)
    #[tokio::test]
    async fn a_folder_that_spreads_takes_the_connections_that_there_are() {
        let server = FakeServer::start(Plan::new().with(a_folder(40)))
            .await
            .unwrap();
        let (pool, job, mut keep) = a_spread(&server, 2).await;

        spread(&pool, &job, &mut keep, 8).await.unwrap();

        assert_eq!(server.seen().most_open, 2, "the pool holds two");
        assert_eq!(
            keep.batches
                .iter()
                .map(|batch| batch.messages.len())
                .sum::<usize>(),
            40
        );
    }

    /// One connection must still read the next batch while the store
    /// writes the last one. A fan-out of one is a pipeline. (§3.4)
    #[tokio::test]
    async fn one_connection_that_spreads_still_reads_while_the_store_writes() {
        let server = FakeServer::start(Plan::new().with(a_folder(40)))
            .await
            .unwrap();
        let (pool, job, _) = a_spread(&server, 1).await;

        let mut keep = Overlapping {
            server: &server,
            writes: 0,
            of: job.batches.len(),
        };

        spread(&pool, &job, &mut keep, 1).await.expect(
            "the socket never asked for a batch while the store wrote one",
        );
    }

    /// Cut the connection of the pool at its first fetch. (§3.1)
    ///
    /// The connection is open and holds INBOX already, so the fetch is
    /// the next command that it sends. The count of the plan is the
    /// count of one connection, and only one is open here.
    fn cut_at_the_first_fetch(server: &FakeServer) {
        let sent = server.seen().commands.len();
        server.change(|plan| plan.cut_after = Some(sent + 1));
    }

    #[tokio::test]
    async fn a_folder_that_spreads_gives_the_error_of_a_broken_connection() {
        let server = FakeServer::start(Plan::new().with(a_folder(40)))
            .await
            .unwrap();
        let (pool, job, mut keep) = a_spread(&server, 1).await;
        cut_at_the_first_fetch(&server);

        let problem = spread(&pool, &job, &mut keep, 1).await;

        assert!(problem.is_err(), "the sync hid a connection that broke");
    }

    /// The answer stopped in the middle, so bytes that belong to no
    /// command may be on the socket. The next command to read them
    /// would take the wrong answer. (§3.1)
    #[tokio::test]
    async fn a_spread_that_broke_leaves_no_connection_in_the_pool() {
        let server = FakeServer::start(Plan::new().with(a_folder(40)))
            .await
            .unwrap();
        let (pool, job, mut keep) = a_spread(&server, 1).await;
        cut_at_the_first_fetch(&server);

        spread(&pool, &job, &mut keep, 1).await.unwrap_err();

        assert_eq!(pool.idle(), 0, "a connection that broke waits in the pool");
    }

    /// A store that gives an error takes no more batches. The
    /// connections must stop, because a batch that nobody takes is a
    /// round trip that brings nothing. (§3.1)
    /// The pool holds eight, and the folder asks for two. The other
    /// six belong to the folders that did not ask yet. (§3.1)
    #[tokio::test]
    async fn a_folder_that_spreads_takes_no_more_hands_than_it_asked_for() {
        let server = FakeServer::start(Plan::new().with(a_folder(40)))
            .await
            .unwrap();
        let (pool, job, mut keep) = a_spread(&server, 8).await;
        assert_eq!(job.batches.len(), 10, "the test wants a batch for each");

        spread(&pool, &job, &mut keep, 2).await.unwrap();

        assert_eq!(server.seen().most_open, 2, "the folder took too many");
    }

    /// A sync of one folder must use the pool, and not one connection
    /// of it. This is what [`spread`] is for. (§3.1)
    #[tokio::test]
    async fn a_folder_that_resumes_reads_on_every_connection() {
        let server = FakeServer::start(Plan::new().with(a_folder(40)))
            .await
            .unwrap();
        let pool = a_pool_of(&server, 4);
        let mut held = pool.take().await.unwrap();
        let view = held.examine("INBOX").await.unwrap();
        drop(held);

        let job = plan("INBOX", &FolderState::default(), &view, SIZE);
        let mut store = Store::default();
        resume(&pool, &job, &mut store, quick()).await.unwrap();

        assert_eq!(server.seen().most_open, 4, "the sync left a hand idle");
        assert_eq!(store.uids.len(), 40, "the sync lost a message");
    }

    #[tokio::test]
    async fn a_folder_that_spreads_stops_reading_when_the_store_stops() {
        let server = FakeServer::start(Plan::new().with(a_folder(160)))
            .await
            .unwrap();
        let (pool, job, _) = a_spread(&server, 2).await;
        assert_eq!(job.batches.len(), 40, "the test wants many batches");
        assert_eq!(fetches(&server), 0, "the plan asked for no mail yet");

        let mut keep = Recorder {
            stop_after: Some(3),
            ..Recorder::default()
        };
        spread(&pool, &job, &mut keep, 2).await.unwrap_err();

        // Two connections and a queue of two hold a few batches more
        // than the store took. They never hold forty.
        let asked = fetches(&server);
        assert!(
            asked < job.batches.len(),
            "the connections read {asked} batches after the store stopped"
        );
    }

    #[tokio::test]
    async fn a_folder_that_spreads_reads_the_flags_that_changed() {
        let server = FakeServer::start(Plan::new().with(a_folder(6)))
            .await
            .unwrap();
        let pool = a_pool_of(&server, 2);
        let mut held = pool.take().await.unwrap();
        let view = held.examine("INBOX").await.unwrap();
        drop(held);

        let saved = FolderState {
            uid_validity: view.uid_validity,
            uid_next: 5,
            highest_mod_seq: 2,
            pending: UidSet::new(),
        };
        let job = plan("INBOX", &saved, &view, SIZE);
        assert!(!job.batches.is_empty() && job.changed.is_some());

        let mut keep = Recorder::default();
        spread(&pool, &job, &mut keep, 2).await.unwrap();

        let uids: Vec<u32> = keep
            .batches
            .iter()
            .flat_map(|batch| batch.messages.iter().map(|held| held.uid))
            .collect();

        for uid in [5, 6] {
            assert!(uids.contains(&uid), "the new message {uid} never landed");
        }

        // The flags of UIDs 3 and 4 moved after the sequence 2, and the
        // flags of 1 and 2 did not. A fetch that forgets `CHANGEDSINCE`
        // brings all four, and the sync then writes mail it has. (§3.3)
        for uid in [3, 4] {
            assert!(uids.contains(&uid), "the flags of {uid} never landed");
        }
        for uid in [1, 2] {
            assert!(!uids.contains(&uid), "the flags of {uid} never moved");
        }
    }

    /// A folder can hold new mail and changed flags at the same time.
    /// The read of the changed set runs while the store writes the
    /// last batch of new mail, so that batch must still land. (§3.4)
    #[tokio::test]
    async fn a_job_reads_the_new_mail_and_the_flags_that_changed() {
        let server = FakeServer::start(Plan::new().with(a_folder(6)))
            .await
            .unwrap();
        let pool = a_pool(&server).await;
        let mut held = pool.take().await.unwrap();
        let view = held.examine("INBOX").await.unwrap();

        // UIDs 5 and 6 are new, and the flags of 1 thru 4 may have
        // moved since the sequence 2.
        let saved = FolderState {
            uid_validity: view.uid_validity,
            uid_next: 5,
            highest_mod_seq: 2,
            pending: UidSet::new(),
        };
        let job = plan("INBOX", &saved, &view, SIZE);
        assert!(!job.batches.is_empty(), "the test wants new mail");
        assert!(job.changed.is_some(), "the test wants a changed set");

        let mut keep = Recorder::default();
        run(&mut held, &job, &mut keep).await.unwrap();

        let uids: Vec<u32> = keep
            .batches
            .iter()
            .flat_map(|batch| batch.messages.iter().map(|held| held.uid))
            .collect();

        for uid in [5, 6] {
            assert!(uids.contains(&uid), "the new message {uid} never landed");
        }

        // The flags of UIDs 3 and 4 moved after the sequence 2, and the
        // flags of 1 and 2 did not. A fetch that forgets `CHANGEDSINCE`
        // brings all four, and the sync then writes mail it has. (§3.3)
        for uid in [3, 4] {
            assert!(uids.contains(&uid), "the flags of {uid} never landed");
        }
        for uid in [1, 2] {
            assert!(!uids.contains(&uid), "the flags of {uid} never moved");
        }
    }

    #[tokio::test]
    async fn a_job_reads_the_messages_that_changed() {
        let server = FakeServer::start(Plan::new().with(a_folder(6)))
            .await
            .unwrap();
        let pool = a_pool(&server).await;
        let mut held = pool.take().await.unwrap();
        let view = held.examine("INBOX").await.unwrap();
        let saved = FolderState {
            uid_validity: view.uid_validity,
            uid_next: view.uid_next,
            highest_mod_seq: 4,
            pending: UidSet::new(),
        };
        let job = plan("INBOX", &saved, &view, SIZE);
        let mut keep = Recorder::default();
        run(&mut held, &job, &mut keep).await.unwrap();

        let uids: Vec<u32> = keep
            .batches
            .iter()
            .flat_map(|batch| batch.messages.iter().map(|held| held.uid))
            .collect();

        assert_eq!(uids, vec![5, 6], "only the messages above the sequence");
    }

    #[tokio::test]
    async fn a_job_never_writes_to_the_server() {
        let server = FakeServer::start(Plan::new().with(a_folder(9)))
            .await
            .unwrap();
        let pool = a_pool(&server).await;
        let mut held = pool.take().await.unwrap();
        let view = held.examine("INBOX").await.unwrap();
        let job = plan("INBOX", &FolderState::default(), &view, SIZE);
        run(&mut held, &job, &mut Recorder::default())
            .await
            .unwrap();

        assert!(server.writes().is_empty(), "{:?}", server.writes());
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::composite]
    fn a_state(tc: TestCase) -> (FolderState, View, u32) {
        let validity = tc.draw(gs::integers::<u32>().min_value(1).max_value(3));
        let seen = tc.draw(gs::integers::<u32>().min_value(1).max_value(60));
        let more = tc.draw(gs::integers::<u32>().min_value(0).max_value(40));
        let mod_seq = tc.draw(gs::integers::<u64>().min_value(0).max_value(50));
        let grew = tc.draw(gs::integers::<u64>().min_value(0).max_value(50));
        let owed = tc.draw(
            gs::vecs(gs::integers::<u32>().min_value(1).max_value(seen))
                .min_size(0)
                .max_size(8),
        );
        let size = tc.draw(gs::integers::<u32>().min_value(1).max_value(9));

        let saved = FolderState {
            uid_validity: validity,
            uid_next: seen + 1,
            highest_mod_seq: mod_seq,
            pending: UidSet::of(&owed),
        };
        let view = View {
            name: "INBOX".to_string(),
            exists: seen + more,
            uid_validity: tc
                .draw(gs::sampled_from(vec![validity, validity + 1])),
            uid_next: seen + more + 1,
            highest_mod_seq: mod_seq + grew,
        };

        (saved, view, size)
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_plan_asks_for_every_uid_that_is_missing(tc: TestCase) {
        let (saved, view, size) = tc.draw(a_state());
        let job = plan("INBOX", &saved, &view, size);
        let asked = whole(&job.batches);

        let owed: Vec<u32> = (1..view.uid_next)
            .filter(|uid| {
                job.restart
                    || *uid >= saved.uid_next
                    || saved.pending.holds(*uid)
            })
            .collect();
        let owed = UidSet::of(&owed);

        assert_eq!(asked, owed);
        assert_eq!(job.start.pending, owed);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_cut_plan_asks_for_every_uid_that_the_server_holds(tc: TestCase) {
        let (saved, view, size) = tc.draw(a_state());
        let some: Vec<u32> = tc.draw(
            gs::vecs(gs::integers::<u32>().min_value(1).max_value(120))
                .min_size(0)
                .max_size(20),
        );
        let held = UidSet::of(&some);

        let job = plan("INBOX", &saved, &view, size);
        let owed = job.start.pending.clone();
        let cut = job.only(&held);
        let asked = whole(&cut.batches);

        for uid in 1..view.uid_next {
            let wanted = owed.holds(uid) && held.holds(uid);

            assert_eq!(asked.holds(uid), wanted, "the cut is wrong at {uid}");
        }
        assert_eq!(cut.start.pending, asked, "the state lost a UID");
    }

    /// A cut of a cut must give the same plan as one cut of both. A
    /// plan that shrinks each time it is read loses mail. (§3.2)
    #[hegel::test(test_cases = 150)]
    fn prop_cutting_a_plan_twice_is_cutting_it_one_time(tc: TestCase) {
        let (saved, view, size) = tc.draw(a_state());
        let one = UidSet::of(
            &tc.draw(
                gs::vecs(gs::integers::<u32>().min_value(1).max_value(120))
                    .min_size(0)
                    .max_size(20),
            ),
        );
        let two = UidSet::of(
            &tc.draw(
                gs::vecs(gs::integers::<u32>().min_value(1).max_value(120))
                    .min_size(0)
                    .max_size(20),
            ),
        );

        let twice = plan("INBOX", &saved, &view, size).only(&one).only(&two);
        let once = plan("INBOX", &saved, &view, size).only(&one.and(&two));

        assert_eq!(twice.start, once.start);
        assert_eq!(whole(&twice.batches), whole(&once.batches));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_plan_never_asks_twice_for_a_uid(tc: TestCase) {
        let (saved, view, size) = tc.draw(a_state());
        let job = plan("INBOX", &saved, &view, size);

        let total: u64 = job.batches.iter().map(UidSet::count).sum();
        assert_eq!(total, whole(&job.batches).count());

        for set in &job.batches {
            assert!(set.count() <= u64::from(size));
        }
        if let Some(changed) = &job.changed {
            assert!(changed.without(&whole(&job.batches)) == *changed);
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_second_sync_of_the_same_folder_asks_for_nothing(tc: TestCase) {
        let (saved, view, size) = tc.draw(a_state());
        let job = plan("INBOX", &saved, &view, size);
        let again = plan("INBOX", &job.done, &view, size);

        assert!(again.is_empty(), "{again:?}");
        assert_eq!(again.count(), 0);
    }

    /// Many connections read one folder, so the batches land in the
    /// order that the server answers, and not in the order of the
    /// plan. The state must not depend on that order. (§3.1)
    #[hegel::test(test_cases = 200)]
    fn prop_the_order_of_the_batches_never_changes_the_state(tc: TestCase) {
        let (saved, view, size) = tc.draw(a_state());
        let job = plan("INBOX", &saved, &view, size);

        if job.batches.is_empty() {
            return;
        }

        let fold = |order: &[usize]| {
            order.iter().fold(job.start.clone(), |state, at| {
                job.after(&state, &job.batches[*at])
            })
        };

        let straight: Vec<usize> = (0..job.batches.len()).collect();
        let mut mixed = straight.clone();
        let steps = tc.draw(gs::integers::<usize>().min_value(0).max_value(8));
        for _ in 0..steps {
            let here = tc.draw(
                gs::integers::<usize>()
                    .min_value(0)
                    .max_value(mixed.len() - 1),
            );
            let there = tc.draw(
                gs::integers::<usize>()
                    .min_value(0)
                    .max_value(mixed.len() - 1),
            );
            mixed.swap(here, there);
        }

        assert_eq!(fold(&straight), fold(&mixed), "the order moved the state");
        assert_eq!(fold(&straight), job.done);
    }

    /// `run` reads the next batch while the store writes this one, so
    /// it computes the state of a batch one time and carries it to the
    /// write. That is only safe while the state holds still. (§3.4)
    #[hegel::test(test_cases = 200)]
    fn prop_the_state_after_a_batch_never_moves_again(tc: TestCase) {
        let (saved, view, size) = tc.draw(a_state());
        let job = plan("INBOX", &saved, &view, size);
        let mut state = job.start.clone();

        for batch in &job.batches {
            let once = job.after(&state, batch);
            let twice = job.after(&once, batch);

            assert_eq!(once, twice, "the state moved on the second read");
            state = once;
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_uid_validity_that_changed_asks_for_the_whole_folder(
        tc: TestCase,
    ) {
        let (saved, mut view, size) = tc.draw(a_state());
        view.uid_validity = saved.uid_validity + 1;
        let job = plan("INBOX", &saved, &view, size);

        let all: Vec<u32> = (1..view.uid_next).collect();

        assert!(job.restart);
        assert_eq!(whole(&job.batches), UidSet::of(&all));
        assert_eq!(job.changed, None);
        assert_eq!(job.since, None);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_folder_only_finishes_when_every_batch_arrived(tc: TestCase) {
        let (saved, view, size) = tc.draw(a_state());
        let job = plan("INBOX", &saved, &view, size);
        let mut state = job.start.clone();

        for (at, batch) in job.batches.iter().enumerate() {
            state = job.after(&state, batch);
            let last = at + 1 == job.batches.len();

            assert_eq!(state.pending.is_empty(), last);
            assert_eq!(state == job.done, last, "the state finished early");
        }

        assert_eq!(state, job.done);
    }

    #[hegel::test(test_cases = 60)]
    fn prop_a_job_reads_every_message_of_the_folder(tc: TestCase) {
        let count = tc.draw(gs::integers::<u32>().min_value(0).max_value(25));
        let size = tc.draw(gs::integers::<u32>().min_value(1).max_value(6));

        let (uids, state, done) = run_on(async move {
            let server = FakeServer::start(Plan::new().with(a_folder(count)))
                .await
                .unwrap();
            let pool = a_pool(&server).await;
            let mut held = pool.take().await.unwrap();
            let view = held.examine("INBOX").await.unwrap();
            let job = plan("INBOX", &FolderState::default(), &view, size);
            let mut keep = Recorder::default();
            let state = run(&mut held, &job, &mut keep).await.unwrap();

            let mut uids: Vec<u32> = keep
                .batches
                .iter()
                .flat_map(|batch| batch.messages.iter().map(|held| held.uid))
                .collect();
            uids.sort_unstable();

            (uids, state, job.done)
        });

        assert_eq!(uids, (1..=count).collect::<Vec<u32>>());
        assert_eq!(state, done);
    }
    // -----------------------------------------------------------------
    // Failure, and the resume after it. (§3.4)
    // -----------------------------------------------------------------

    #[test]
    fn each_wait_is_twice_the_last_one() {
        let back = Backoff {
            tries: 5,
            first: Duration::from_millis(10),
            most: Duration::from_millis(100),
        };

        assert_eq!(back.wait(1), Duration::from_millis(10));
        assert_eq!(back.wait(2), Duration::from_millis(20));
        assert_eq!(back.wait(3), Duration::from_millis(40));
        assert_eq!(back.wait(4), Duration::from_millis(80));
    }

    #[test]
    fn a_wait_never_grows_above_the_longest_one() {
        let back = Backoff {
            tries: 40,
            first: Duration::from_millis(10),
            most: Duration::from_millis(100),
        };

        assert_eq!(back.wait(9), Duration::from_millis(100));
        assert_eq!(back.wait(9_000), Duration::from_millis(100));
    }

    #[test]
    fn a_connection_that_broke_is_worth_another_try() {
        assert!(again(&Error::Closed));
        assert!(again(&Error::Refused("busy".into())));
        assert!(again(&Error::Io(std::io::Error::other("gone"))));
    }

    #[test]
    fn a_server_that_says_no_is_not_worth_another_try() {
        assert!(!again(&Error::No("no such folder".into())));
        assert!(!again(&Error::Bad("what?".into())));
    }

    #[tokio::test]
    async fn a_sync_that_loses_the_connection_reads_the_rest() {
        // The server cuts each connection after a few commands, so
        // the first try never finishes the folder.
        let server =
            FakeServer::start(Plan::new().with(a_folder(20)).cut_after(7))
                .await
                .unwrap();
        let pool = a_pool(&server).await;
        let mut held = pool.take().await.unwrap();
        let view = held.examine("INBOX").await.unwrap();
        drop(held);

        let job = plan("INBOX", &FolderState::default(), &view, SIZE);
        let mut store = Store::default();
        let state = resume(&pool, &job, &mut store, quick()).await.unwrap();

        store.uids.sort_unstable();
        assert_eq!(store.uids, (1..=20).collect::<Vec<u32>>());
        assert_eq!(state, job.done);
        assert!(server.seen().connections > 1, "it never tried again");
    }

    #[tokio::test]
    async fn a_sync_that_keeps_failing_gives_up() {
        let server =
            FakeServer::start(Plan::new().with(a_folder(20)).cut_after(1))
                .await
                .unwrap();
        let pool = a_pool(&server).await;
        let job =
            plan("INBOX", &FolderState::default(), &a_view(77, 21, 20), SIZE);
        let mut store = Store::default();
        let error = resume(&pool, &job, &mut store, quick()).await;

        assert!(error.is_err(), "{error:?}");
        assert_eq!(server.seen().connections, 4, "one connection for a try");
    }

    #[tokio::test]
    async fn a_folder_that_is_not_there_never_tries_again() {
        let server = FakeServer::start(Plan::new().with(a_folder(3)))
            .await
            .unwrap();
        let pool = a_pool(&server).await;
        let job =
            plan("Nowhere", &FolderState::default(), &a_view(77, 4, 3), SIZE);
        let mut store = Store::default();
        let error = resume(&pool, &job, &mut store, quick()).await;

        assert!(matches!(error, Err(Error::No(_))), "{error:?}");
        assert_eq!(server.seen().connections, 1);
    }

    #[tokio::test]
    async fn a_store_that_fails_once_loses_no_message() {
        let server = FakeServer::start(Plan::new().with(a_folder(12)))
            .await
            .unwrap();
        let pool = a_pool(&server).await;
        let mut held = pool.take().await.unwrap();
        let view = held.examine("INBOX").await.unwrap();
        drop(held);

        let job = plan("INBOX", &FolderState::default(), &view, SIZE);
        let mut store = Store {
            fail_at: Some(1),
            ..Store::default()
        };
        resume(&pool, &job, &mut store, quick()).await.unwrap();

        store.uids.sort_unstable();
        assert_eq!(store.uids, (1..=12).collect::<Vec<u32>>());
    }

    #[tokio::test]
    async fn a_connection_that_broke_never_goes_back_to_the_pool() {
        let server =
            FakeServer::start(Plan::new().with(a_folder(20)).cut_after(4))
                .await
                .unwrap();
        let pool = a_pool(&server).await;
        let job =
            plan("INBOX", &FolderState::default(), &a_view(77, 21, 20), SIZE);
        let mut store = Store::default();
        let _ = resume(&pool, &job, &mut store, quick()).await;

        assert_eq!(pool.idle(), 0, "a broken connection went back");
    }

    #[tokio::test]
    async fn a_sync_that_starts_again_asks_for_no_message_twice() {
        let server = FakeServer::start(Plan::new().with(a_folder(12)))
            .await
            .unwrap();
        let pool = a_pool(&server).await;
        let mut held = pool.take().await.unwrap();
        let view = held.examine("INBOX").await.unwrap();
        drop(held);

        let job = plan("INBOX", &FolderState::default(), &view, SIZE);
        let mut store = Store {
            fail_at: Some(2),
            ..Store::default()
        };
        resume(&pool, &job, &mut store, quick()).await.unwrap();

        let mut once = store.uids.clone();
        once.sort_unstable();
        once.dedup();
        assert_eq!(once.len(), store.uids.len(), "a message arrived twice");
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_sync_that_stops_anywhere_still_reads_every_message(tc: TestCase) {
        let count = tc.draw(gs::integers::<u32>().min_value(1).max_value(20));
        let size = tc.draw(gs::integers::<u32>().min_value(1).max_value(5));
        let fail_at =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(5));

        let uids = run_on(async move {
            let server = FakeServer::start(Plan::new().with(a_folder(count)))
                .await
                .unwrap();
            let pool = a_pool(&server).await;
            let mut held = pool.take().await.unwrap();
            let view = held.examine("INBOX").await.unwrap();
            drop(held);

            let job = plan("INBOX", &FolderState::default(), &view, size);
            let mut store = Store {
                fail_at: Some(fail_at),
                ..Store::default()
            };
            resume(&pool, &job, &mut store, quick()).await.unwrap();

            let mut uids = store.uids;
            uids.sort_unstable();

            uids
        });

        assert_eq!(uids, (1..=count).collect::<Vec<u32>>());
    }
}
