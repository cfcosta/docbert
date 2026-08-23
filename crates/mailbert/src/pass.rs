//! The pass that follows a sync: threading, and the index. (§5.5, §6.1)
//!
//! A sync writes mail into the store, and the store alone answers no
//! search. This pass threads what the store holds, and writes each
//! message into the index with the thread that it belongs to.
//!
//! The pass reads the whole store, because §5.5 threads a message
//! against every other message. It writes only the threads that moved,
//! because a mailbox of 100000 messages must not go through the index
//! again for one new message.

use std::collections::BTreeSet;

use mailbert_core::{
    MailIndex,
    Message,
    MessageId,
    Store,
    ThreadId,
    Threading,
    threading,
};

use crate::error::Result;

/// How much memory the index writer may hold before it flushes.
const BUDGET: usize = 64 * 1024 * 1024;

/// What one pass wrote.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Wrote {
    /// The messages that went into the index.
    pub messages: usize,

    /// The threads that the store holds, after the pass.
    pub threads: usize,
}

/// Thread the store, and write every message into the index.
///
/// This is the pass of a first sync, and of a rebuild.
pub fn everything(store: &Store, index: &MailIndex) -> Result<Wrote> {
    let messages = store.all()?;
    let threading = thread_all(&messages);

    write(store, index, &messages, &threading, None)
}

/// Thread the store, and write the threads that a sync moved.
///
/// A message that arrives can join two threads into one, and that moves
/// every message of both. Each of those messages is in the thread of a
/// message that the sync touched, so this writes them all.
///
/// The pass also writes the mail that the index is behind on. A sync
/// that stops after the download, and before this pass, leaves mail
/// that no later sync touches. Without this repair that mail stays out
/// of the index for ever, and no search finds it. (§6.1)
pub fn after_sync(
    store: &Store,
    index: &MailIndex,
    touched: &BTreeSet<MessageId>,
) -> Result<Wrote> {
    let messages = store.all()?;
    let threading = thread_all(&messages);
    let behind = behind(index, &messages)?;

    if !behind.is_empty() {
        tracing::info!(
            messages = behind.len(),
            "the index was behind the store"
        );
    }

    let mut want = touched.clone();
    want.extend(behind);

    if want.is_empty() {
        return Ok(Wrote {
            messages: 0,
            threads: threading.len(),
        });
    }

    write(store, index, &messages, &threading, Some(&want))
}

/// The mail that the store holds, and that the index lacks.
fn behind(
    index: &MailIndex,
    messages: &[Message],
) -> Result<BTreeSet<MessageId>> {
    let held = index.held()?;

    Ok(messages
        .iter()
        .map(|message| message.id)
        .filter(|id| !held.contains(&id.numeric()))
        .collect())
}

/// Thread every message that the store holds. (§5.5)
fn thread_all(messages: &[Message]) -> Threading {
    let inputs: Vec<_> = messages.iter().map(Message::thread_input).collect();

    threading::thread(&inputs)
}

/// The messages to write: the threads of `touched`, or everything.
fn wanted(
    threading: &Threading,
    touched: Option<&BTreeSet<MessageId>>,
) -> Option<BTreeSet<MessageId>> {
    let touched = touched?;
    let mut found = BTreeSet::new();

    for id in touched {
        match threading.thread_of(*id) {
            Some(thread) => {
                found.extend(threading.members(thread).iter().copied())
            }
            // A message that is in no thread is a message that the
            // store does not hold. The pass writes what it can.
            None => {
                found.insert(*id);
            }
        }
    }

    Some(found)
}

/// Write the index, either whole or for the threads that moved.
fn write(
    store: &Store,
    index: &MailIndex,
    messages: &[Message],
    threading: &Threading,
    touched: Option<&BTreeSet<MessageId>>,
) -> Result<Wrote> {
    let wanted = wanted(threading, touched);

    let mut writer = index.writer(BUDGET)?;
    let mut count = 0;

    // A whole pass replaces the index, so a message that left the store
    // leaves the index with it.
    if wanted.is_none() {
        index.clear(&writer)?;
    }

    for message in messages {
        if wanted
            .as_ref()
            .is_some_and(|set| !set.contains(&message.id))
        {
            continue;
        }

        let thread = threading
            .thread_of(message.id)
            .unwrap_or_else(|| ThreadId::from_root(message.id));
        let tags = store.tags_of(&message.id)?;

        index.add(&writer, message, thread, &tags)?;
        count += 1;
    }

    index.commit(&mut writer)?;

    Ok(Wrote {
        messages: count,
        threads: threading.len(),
    })
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_every_message_of_the_store_reaches_the_index` | invariant | A message that the pass misses is mail that no search can find. |
    //! | `prop_the_index_agrees_with_the_threading` | differential | §10.1 counts `[3/7]` off the index, and `thread` reads it. A thread that disagrees with §5.5 shows the wrong list. |
    //! | `prop_a_pass_leaves_no_message_out_of_the_index` | invariant | A sync that stops before the pass leaves mail that no later sync touches, and no search finds it. |
    //! | `prop_a_second_pass_changes_nothing` | algebraic | Every sync runs the pass again, and must not write a message twice. |

    use hegel::{TestCase, generators as gs};
    use mailbert_core::{
        Clock,
        Vocabulary,
        compile,
        message::Location,
        mime,
        query,
    };
    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::trace::pen::{Pen, capture, open};

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    const DAY: i64 = 86_400;

    fn open_at(dir: &TempDir) -> Store {
        Store::open(dir.path()).expect("a store")
    }

    fn raw_bytes(key: &str, subject: &str, parent: Option<&str>) -> Vec<u8> {
        let reply = match parent {
            Some(parent) => format!("In-Reply-To: <{parent}@x.test>\r\n"),
            None => String::new(),
        };

        format!(
            "From: Alice Smith <alice@example.test>\r\n\
             To: bob@example.test\r\n\
             Subject: {subject}\r\n\
             Date: Fri, 14 Aug 2026 09:30:00 +0000\r\n\
             Message-ID: <{key}@x.test>\r\n\
             {reply}\r\n\
             The deposit is due.\r\n"
        )
        .into_bytes()
    }

    /// Write one message, and give back its identity.
    fn write(
        store: &Store,
        key: &str,
        subject: &str,
        parent: Option<&str>,
    ) -> MessageId {
        let raw = raw_bytes(key, subject, parent);
        let location = Location {
            account: "work".to_string(),
            folder: "INBOX".to_string(),
            uid: 1,
            uid_validity: 1,
            received: 100 * DAY,
            flags: BTreeSet::new(),
        };

        let message = Message::new(
            mime::parse(&raw).expect("a message"),
            location,
            [r"\Seen"],
        );

        store.put(&message, &raw).expect("a write").id
    }

    fn in_ram() -> MailIndex {
        MailIndex::open_in_ram().expect("an index")
    }

    // -----------------------------------------------------------------
    // The index catches up with the store. (§6.1)
    // -----------------------------------------------------------------

    /// A sync that stopped after the download, and before the pass,
    /// leaves mail that no later sync touches. Each pass must find that
    /// mail again, or the search never sees it. (§6.1)
    #[test]
    fn a_pass_writes_the_mail_that_an_earlier_pass_never_wrote() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let index = in_ram();

        write(&store, "a", "The deposit", None);
        write(&store, "b", "The inspection", None);

        // Nothing touched this run, and the index holds nothing.
        let wrote =
            after_sync(&store, &index, &BTreeSet::new()).expect("a pass");

        assert_eq!(wrote.messages, 2);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn a_pass_that_finds_nothing_behind_writes_nothing() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let index = in_ram();

        write(&store, "a", "The deposit", None);
        everything(&store, &index).expect("a first pass");

        let wrote = after_sync(&store, &index, &BTreeSet::new())
            .expect("a second pass");

        assert_eq!(wrote.messages, 0);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn the_log_of_a_pass_says_how_much_the_index_was_behind() {
        open();

        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let index = in_ram();

        write(&store, "a", "The deposit", None);
        write(&store, "b", "The inspection", None);

        let pen = Pen::default();
        tracing::subscriber::with_default(capture(pen.clone()), || {
            after_sync(&store, &index, &BTreeSet::new()).expect("a pass");
        });

        let log = pen.text();
        assert!(log.contains("the index was behind the store"), "{log}");
        assert!(log.contains("messages=2"), "{log}");
    }

    #[test]
    fn the_log_of_a_pass_says_nothing_when_the_index_is_current() {
        open();

        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let index = in_ram();

        write(&store, "a", "The deposit", None);
        everything(&store, &index).expect("a first pass");

        let pen = Pen::default();
        tracing::subscriber::with_default(capture(pen.clone()), || {
            after_sync(&store, &index, &BTreeSet::new()).expect("a pass");
        });

        let log = pen.text();
        assert!(!log.contains("behind the store"), "{log}");
    }

    // -----------------------------------------------------------------
    // Unit tests: the pass that follows a sync.
    // -----------------------------------------------------------------

    #[test]
    fn a_pass_writes_every_message_that_the_store_holds() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = in_ram();
        write(&store, "one", "Deposit", None);
        write(&store, "two", "Invoice", None);

        let wrote = everything(&store, &index).expect("a pass");

        assert_eq!(wrote.messages, 2);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn a_pass_gives_the_messages_of_one_thread_one_identity() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = in_ram();
        let first = write(&store, "one", "Deposit", None);
        let reply = write(&store, "two", "Re: Deposit", Some("one"));

        everything(&store, &index).expect("a pass");

        let one = index.get(&first).expect("a read").expect("a document");
        let two = index.get(&reply).expect("a read").expect("a document");
        assert_eq!(one.thread, two.thread, "a reply left its thread");
        assert_eq!(index.thread(one.thread).expect("a read").len(), 2);
    }

    #[test]
    fn a_pass_writes_the_tags_of_a_message() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = in_ram();
        let id = write(&store, "one", "Deposit", None);
        store.tag(&id, "todo").expect("a tag");

        everything(&store, &index).expect("a pass");

        let hits = found(&store, &index, "tag:todo");
        assert_eq!(hits, vec![id], "the tag never reached the index");
    }

    #[test]
    fn a_message_with_no_copy_left_reaches_the_index_as_gone() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = in_ram();
        let id = write(&store, "one", "Deposit", None);
        store.vanish("work", "INBOX", 1).expect("a write");

        everything(&store, &index).expect("a pass");

        assert_eq!(found(&store, &index, "is:gone"), vec![id]);
    }

    #[test]
    fn a_pass_after_a_sync_writes_the_threads_that_moved() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = in_ram();
        let first = write(&store, "one", "Deposit", None);
        write(&store, "far", "Invoice", None);
        everything(&store, &index).expect("a pass");

        let reply = write(&store, "two", "Re: Deposit", Some("one"));
        let touched = [reply].into_iter().collect();
        let wrote = after_sync(&store, &index, &touched).expect("a pass");

        assert_eq!(wrote.messages, 2, "the pass wrote the wrong messages");
        assert_eq!(index.len(), 3);

        let one = index.get(&first).expect("a read").expect("a document");
        let two = index.get(&reply).expect("a read").expect("a document");
        assert_eq!(one.thread, two.thread);
    }

    #[test]
    fn a_pass_that_touched_nothing_writes_nothing() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = in_ram();
        write(&store, "one", "Deposit", None);
        everything(&store, &index).expect("a pass");

        let wrote =
            after_sync(&store, &index, &BTreeSet::new()).expect("a pass");

        assert_eq!(wrote.messages, 0);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn a_message_that_the_store_dropped_leaves_the_index() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = in_ram();
        let id = write(&store, "one", "Deposit", None);
        everything(&store, &index).expect("a pass");

        store.remove(&id).expect("a delete");
        everything(&store, &index).expect("a pass");

        assert!(index.is_empty(), "a message outlived the store");
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    /// The identities that one query finds, in order.
    fn found(store: &Store, index: &MailIndex, text: &str) -> Vec<MessageId> {
        let clock = Clock::utc(0);
        let query = query::parse(text, clock).expect("a query");
        let vocabulary = Vocabulary::from_store(store).expect("a vocabulary");
        let compiled =
            compile(&query, index, &vocabulary, clock).expect("a query");

        let mut ids: Vec<MessageId> = index
            .top(&*compiled.search, index.len().max(1))
            .expect("a read")
            .into_iter()
            .map(|hit| hit.id)
            .collect();

        ids.sort();
        ids
    }

    /// One or more message keys, and none of them the same.
    #[hegel::composite]
    fn some_keys(tc: TestCase) -> Vec<String> {
        let drawn: Vec<String> = tc.draw(
            gs::vecs(gs::text().alphabet("abcd").min_size(1).max_size(3))
                .min_size(1)
                .max_size(5),
        );

        let mut keys = drawn;
        keys.sort();
        keys.dedup();

        keys
    }

    #[hegel::test(test_cases = 20)]
    fn prop_every_message_of_the_store_reaches_the_index(tc: TestCase) {
        let keys = tc.draw(some_keys());
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = in_ram();

        for key in &keys {
            write(&store, key, "Deposit", None);
        }

        everything(&store, &index).expect("a pass");

        assert_eq!(index.len(), store.len().expect("a read"));
        for id in store.ids().expect("a read") {
            assert!(
                index.get(&id).expect("a read").is_some(),
                "a message never reached the index"
            );
        }
    }

    #[hegel::test(test_cases = 20)]
    fn prop_the_index_agrees_with_the_threading(tc: TestCase) {
        let keys = tc.draw(some_keys());
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = in_ram();

        // Each message answers the one before it, so every message
        // belongs to one thread.
        let mut parent: Option<String> = None;
        for key in &keys {
            write(&store, key, "Deposit", parent.as_deref());
            parent = Some(key.clone());
        }

        everything(&store, &index).expect("a pass");

        let threads: BTreeSet<String> = store
            .ids()
            .expect("a read")
            .iter()
            .map(|id| {
                index
                    .get(id)
                    .expect("a read")
                    .expect("a document")
                    .thread
                    .full_hex()
            })
            .collect();

        assert_eq!(threads.len(), 1, "one thread became {}", threads.len());
    }

    /// Whatever a sync touched, the index must hold the whole store
    /// after the pass. A message that this misses is mail that no
    /// search can find. (§6.1)
    #[hegel::test(test_cases = 20)]
    fn prop_a_pass_leaves_no_message_out_of_the_index(tc: TestCase) {
        let keys = tc.draw(some_keys());
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = in_ram();

        let ids: Vec<MessageId> = keys
            .iter()
            .map(|key| write(&store, key, "Deposit", None))
            .collect();

        // A message that no run touched is mail that a sync left
        // behind when it stopped before the pass.
        let touched: BTreeSet<MessageId> = ids
            .iter()
            .filter(|_| tc.draw(gs::booleans()))
            .copied()
            .collect();

        after_sync(&store, &index, &touched).expect("a pass");

        assert_eq!(index.len(), store.len().expect("a read"));
        for id in store.ids().expect("a read") {
            assert!(
                index.get(&id).expect("a read").is_some(),
                "a message never reached the index"
            );
        }
    }

    #[hegel::test(test_cases = 20)]
    fn prop_a_second_pass_changes_nothing(tc: TestCase) {
        let keys = tc.draw(some_keys());
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let index = in_ram();

        for key in &keys {
            write(&store, key, "Deposit", None);
        }

        everything(&store, &index).expect("a pass");
        let first = index.len();
        everything(&store, &index).expect("a pass");

        assert_eq!(index.len(), first, "the second pass wrote again");
    }
}
