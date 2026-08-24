//! The one writer of a sync. (§4.2)
//!
//! A sync runs a folder for each connection, and every one of them has
//! mail to keep. LMDB lets one writer into a database at a time, and a
//! commit asks the disk for a flush. Eight folders that commit on
//! their own therefore pay for eight flushes, one after the other.
//!
//! This module gives every folder one handle to one writer. The writer
//! takes the change that waits, and every other change that already
//! waits behind it, and gives the whole group to [`Store::apply`]. The
//! group then costs one flush. A folder that arrives while a commit
//! runs joins the next group, so the queue never grows without bound
//! and no folder waits for a timer.

use std::sync::Arc;

use mailbert_core::{
    Error,
    Result,
    store::{Applied, Change, Store},
};
use tokio::sync::{mpsc, oneshot};

/// How many changes one transaction takes.
///
/// A bigger group costs less for each change, and holds more mail in
/// memory. A sync has one folder for each connection, so a group of
/// this size holds every folder of any account of §2.1.
const MOST: usize = 64;

/// How many changes wait for the writer before a folder blocks.
const QUEUE: usize = 256;

/// What a folder sends, and where the answer goes.
struct Handoff {
    change: Change,
    back: oneshot::Sender<std::result::Result<Applied, String>>,
}

/// A handle on the one writer. Every folder of a sync holds one.
///
/// The writer stops when the last handle goes away.
#[derive(Clone)]
pub struct Writer {
    give: mpsc::Sender<Handoff>,
}

impl Writer {
    /// Start the writer on the store.
    pub fn new(store: Arc<Store>) -> Self {
        Self::with(move |group: Vec<Change>| store.apply(&group))
    }

    /// Start the writer on any write.
    ///
    /// [`new`] gives it the store. A test gives it a write that fails,
    /// so it can watch what a folder hears then.
    fn with<W>(write: W) -> Self
    where
        W: Fn(Vec<Change>) -> Result<Vec<Applied>> + Send + Sync + 'static,
    {
        let (give, take) = mpsc::channel(QUEUE);
        tokio::spawn(serve(Arc::new(write), take));

        Self { give }
    }

    /// Give the writer one change, and wait for what it left.
    pub async fn send(&self, change: Change) -> Result<Applied> {
        let (back, wait) = oneshot::channel();

        if self.give.send(Handoff { change, back }).await.is_err() {
            return Err(Error::Write(STOPPED.to_string()));
        }

        match wait.await {
            Ok(Ok(applied)) => Ok(applied),
            Ok(Err(said)) => Err(Error::Write(said)),
            Err(_) => Err(Error::Write(STOPPED.to_string())),
        }
    }
}

/// What a folder hears when the writer is no longer there.
const STOPPED: &str = "the writer of the sync is no longer there";

/// Take the group that waits, and write it in one transaction.
async fn serve<W>(write: Arc<W>, mut take: mpsc::Receiver<Handoff>)
where
    W: Fn(Vec<Change>) -> Result<Vec<Applied>> + Send + Sync + 'static,
{
    while let Some(first) = take.recv().await {
        let (changes, backs): (Vec<Change>, Vec<_>) = drain(first, &mut take)
            .into_iter()
            .map(|one| (one.change, one.back))
            .unzip();

        // The store speaks to the disk, so it never runs on the thread
        // that the folders share.
        let held = Arc::clone(&write);
        let done = tokio::task::spawn_blocking(move || held(changes)).await;

        match done {
            Ok(Ok(applied)) if applied.len() == backs.len() => {
                for (back, one) in backs.into_iter().zip(applied) {
                    let _ = back.send(Ok(one));
                }
            }
            Ok(Ok(_)) => tell(backs, "the store answered the wrong group"),
            Ok(Err(problem)) => tell(backs, &problem.to_string()),
            Err(problem) => tell(backs, &problem.to_string()),
        }
    }
}

/// Take the change that waits, and every change behind it. (§4.2)
///
/// The group goes into one transaction, so the flush of the disk serves
/// all of it. A folder that arrives while that commit runs joins the
/// next group, so the queue never grows without bound, and no folder
/// waits for a timer.
fn drain(first: Handoff, take: &mut mpsc::Receiver<Handoff>) -> Vec<Handoff> {
    let mut group = vec![first];

    while group.len() < MOST {
        match take.try_recv() {
            Ok(next) => group.push(next),
            Err(_) => break,
        }
    }

    group
}

/// Say the same thing to every folder of a group that failed.
fn tell(
    backs: Vec<oneshot::Sender<std::result::Result<Applied, String>>>,
    said: &str,
) {
    for back in backs {
        let _ = back.send(Err(said.to_string()));
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_the_writer_leaves_what_the_store_leaves` | differential | §4.2. The writer exists to cut the flushes of the disk, and for no other reason. A store that it leaves differently loses mail, or a place, or a state. |

    use std::collections::BTreeSet;

    use hegel::{TestCase, generators as gs};
    use mailbert_core::{
        message::{Location, Message, SEEN},
        mime,
        store::SyncState,
    };
    use tempfile::{TempDir, tempdir};

    use super::*;

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    fn open_at(dir: &TempDir) -> Arc<Store> {
        Arc::new(Store::open(dir.path()).expect("a store"))
    }

    fn raw_bytes(key: &str) -> Vec<u8> {
        format!(
            "From: Alice Smith <alice@example.test>\r\n\
             To: bob@example.test\r\n\
             Subject: Deposit\r\n\
             Date: Fri, 14 Aug 2026 09:30:00 +0000\r\n\
             Message-ID: <{key}@x.test>\r\n\
             \r\n\
             The deposit is due.\r\n"
        )
        .into_bytes()
    }

    fn a_state(uid_next: u32) -> SyncState {
        SyncState {
            uid_validity: 77,
            uid_next,
            highest_mod_seq: 900,
            pending: String::new(),
            synced_at: 1_755_820_800,
        }
    }

    /// The change that one folder brings.
    fn a_change(folder: &str, mail: &[(&str, u32)]) -> Change {
        Change {
            account: "work".to_string(),
            folder: folder.to_string(),
            writes: mail
                .iter()
                .map(|(key, uid)| {
                    let raw = raw_bytes(key);
                    let message = Message::new(
                        mime::parse(&raw).expect("a message"),
                        Location {
                            account: "work".to_string(),
                            folder: folder.to_string(),
                            uid: *uid,
                            uid_validity: 77,
                            received: 8_640_000,
                            flags: BTreeSet::new(),
                        },
                        [SEEN],
                    );

                    (message, raw)
                })
                .collect(),
            gone: Vec::new(),
            state: Some(a_state(mail.len() as u32 + 1)),
        }
    }

    // -----------------------------------------------------------------
    // The writer.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn the_writer_keeps_what_a_folder_gave_it() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let writer = Writer::new(Arc::clone(&store));

        let done = writer
            .send(a_change("INBOX", &[("a", 1), ("b", 2)]))
            .await
            .expect("a write");

        assert_eq!(done.kept.len(), 2, "the writer lost a message");
        assert_eq!(store.all().expect("a read").len(), 2);
        assert!(store.placed("work", "INBOX", 1).expect("a read").is_some());
        assert_eq!(
            store.state("work", "INBOX").expect("a read"),
            Some(a_state(3))
        );
    }

    /// The reason that this module exists. Every folder must reach the
    /// store, and each of them must hear about its own mail. (§4.2)
    #[tokio::test]
    async fn the_writer_takes_the_folders_that_wait_together() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let writer = Writer::new(Arc::clone(&store));

        let names = ["INBOX", "Sent", "Drafts", "Work", "Home", "Spam"];
        let mut tasks = Vec::new();

        for (at, name) in names.iter().enumerate() {
            let writer = writer.clone();
            let key = format!("k{at}");
            let name = name.to_string();

            tasks.push(tokio::spawn(async move {
                let uid = at as u32 + 1;
                let done = writer
                    .send(a_change(&name, &[(&key, uid)]))
                    .await
                    .expect("a write");

                (name, done)
            }));
        }

        for task in tasks {
            let (name, done) = task.await.expect("a folder");

            assert_eq!(done.kept.len(), 1, "{name} heard the wrong answer");
            assert_eq!(
                done.kept[0].locations[0].folder, name,
                "a folder heard the answer of another folder"
            );
        }

        assert_eq!(store.all().expect("a read").len(), names.len());
        for name in names {
            assert!(
                store.state("work", name).expect("a read").is_some(),
                "the writer lost the state of {name}"
            );
        }
    }

    #[tokio::test]
    async fn the_writer_takes_the_copies_that_went_away() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let writer = Writer::new(Arc::clone(&store));
        let held = writer
            .send(a_change("INBOX", &[("a", 1), ("b", 2)]))
            .await
            .expect("the first write");

        let mut gone = a_change("INBOX", &[]);
        gone.gone = vec![1];
        let done = writer.send(gone).await.expect("a write");

        assert_eq!(done.vanished, vec![held.kept[0].id]);
        assert_eq!(store.placed("work", "INBOX", 1).expect("a read"), None);
    }

    /// A folder must learn that its mail did not land, and never think
    /// that it did. (§3.4)
    #[tokio::test]
    async fn a_folder_learns_that_the_writer_is_gone() {
        // A channel with nobody at the other end.
        let (give, take) = mpsc::channel(1);
        drop(take);
        let writer = Writer { give };

        assert!(
            matches!(
                writer.send(a_change("INBOX", &[("a", 1)])).await,
                Err(Error::Write(_))
            ),
            "a folder thought that its mail landed"
        );
    }

    /// A folder that shares a group with a folder that failed must hear
    /// about it, and never wait for ever. (§3.4)
    #[tokio::test]
    async fn a_folder_learns_that_its_mail_did_not_land() {
        let writer = Writer::with(|_| {
            Err(Error::Write("the disk has no room".to_string()))
        });

        let said = writer.send(a_change("INBOX", &[("a", 1)])).await;

        assert!(
            matches!(&said, Err(Error::Write(said))
                if said.contains("the disk has no room")),
            "a folder heard {said:?}"
        );
    }

    /// A store that answers a group of another size gives every folder
    /// the answer of another folder. Nobody hears an answer then.
    #[tokio::test]
    async fn a_folder_hears_nothing_of_another_group() {
        let writer = Writer::with(|_| Ok(Vec::new()));

        let said = writer.send(a_change("INBOX", &[("a", 1)])).await;

        assert!(
            matches!(&said, Err(Error::Write(said))
                if said.contains("wrong group")),
            "a folder heard {said:?}"
        );
    }

    // -----------------------------------------------------------------
    // The group that one transaction takes. (§4.2)
    // -----------------------------------------------------------------

    /// A handoff whose answer goes nowhere.
    fn a_handoff(folder: &str) -> Handoff {
        let (back, _) = oneshot::channel();

        Handoff {
            change: a_change(folder, &[]),
            back,
        }
    }

    /// Fill a channel, and take the first change out of it.
    fn queued(count: usize) -> (Handoff, mpsc::Receiver<Handoff>) {
        let (give, mut take) = mpsc::channel(count.max(1));

        for at in 0..count {
            give.try_send(a_handoff(&format!("f{at}"))).expect("a slot");
        }
        drop(give);

        (take.try_recv().expect("the first change"), take)
    }

    /// The reason that this module exists. One flush of the disk must
    /// serve every change that already waits. (§4.2)
    #[test]
    fn a_group_takes_every_change_that_waits() {
        let (first, mut take) = queued(5);

        assert_eq!(drain(first, &mut take).len(), 5);
    }

    #[test]
    fn a_change_that_waits_alone_goes_alone() {
        let (first, mut take) = queued(1);

        assert_eq!(drain(first, &mut take).len(), 1);
    }

    /// A group holds mail in memory, so one transaction has a limit.
    #[test]
    fn a_group_stops_at_the_most_that_one_transaction_takes() {
        let (first, mut take) = queued(MOST + 10);

        assert_eq!(drain(first, &mut take).len(), MOST);
    }

    /// The changes that the limit left behind must go in the next group,
    /// and never away.
    #[test]
    fn the_changes_over_the_limit_wait_for_the_next_group() {
        let (first, mut take) = queued(MOST + 10);
        drain(first, &mut take);

        // The group took `MOST`, and the first change was already out
        // of the channel, so ten of them are still there.
        let next = take.try_recv().expect("a change that waits");

        assert_eq!(drain(next, &mut take).len(), 10);
    }

    #[hegel::test(test_cases = 20)]
    fn prop_the_writer_leaves_what_the_store_leaves(tc: TestCase) {
        let folders: Vec<String> = tc.draw(
            gs::vecs(gs::sampled_from(vec![
                "INBOX".to_string(),
                "Sent".to_string(),
                "Drafts".to_string(),
            ]))
            .min_size(1)
            .max_size(5),
        );

        let group: Vec<Change> = folders
            .iter()
            .enumerate()
            .map(|(at, folder)| {
                let key = format!("k{at}");
                a_change(folder, &[(&key, at as u32 + 1)])
            })
            .collect();

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime");

        let through = tempdir().expect("a directory");
        let through = open_at(&through);
        let direct = tempdir().expect("a directory");
        let direct = open_at(&direct);

        runtime.block_on(async {
            let writer = Writer::new(Arc::clone(&through));

            for change in &group {
                writer.send(change.clone()).await.expect("a write");
            }
        });

        for change in &group {
            direct.apply(std::slice::from_ref(change)).expect("a write");
        }

        assert_eq!(
            through.all().expect("a read"),
            direct.all().expect("a read"),
            "the writer left another store"
        );
        for folder in &folders {
            assert_eq!(
                through.state("work", folder).expect("a read"),
                direct.state("work", folder).expect("a read"),
                "the writer left another state"
            );
        }
    }
}
