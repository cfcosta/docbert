//! Incremental sync of one folder. (§3.3, §3.4)
//!
//! A second sync reads only what changed. The state of a folder holds
//! `UIDVALIDITY`, `UIDNEXT`, and `HIGHESTMODSEQ`, and the UIDs that the
//! server has and the store does not. That last set is what makes a
//! sync that stops continue where it stopped. (§3.4)

use std::time::Duration;

use tokio::time::sleep;

use crate::{
    connection::{Batch, Connection, View},
    error::{Error, Result},
    pool::Pool,
    sequence::UidSet,
};

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

    for set in &job.batches {
        let batch = connection.fetch(set, None).await?;
        state = job.after(&state, set);
        keep.batch(&job.folder, batch, &state).await?;
    }

    // The messages that the store holds already: a flag that changed,
    // or a message that went away. (§3.3)
    if let (Some(changed), Some(since)) = (&job.changed, job.since) {
        let batch = connection.fetch(changed, Some(since)).await?;
        keep.batch(&job.folder, batch, &state).await?;
    }

    Ok(state)
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
    let mut held = pool.take().await?;

    match run(&mut held, job, keep).await {
        Ok(state) => Ok(state),
        Err(error) => {
            // A connection that broke in the middle of an answer holds
            // bytes that belong to no command. It never goes back.
            held.retire();

            Err(error)
        }
    }
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
        let mut folder = FakeFolder::new("INBOX").with_uid_validity(77);

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
