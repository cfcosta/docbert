//! `thread`, which writes every message of one conversation. (§8.4)
//!
//! §10.1 shows one row for each thread, and the position of the
//! message that matched inside it. `thread` opens that thread, and
//! writes each of its messages in the order that they arrived.
//!
//! The rows have the shape of §10.1, so a reader who searched sees the
//! same columns, and a program that reads the JSON of §10.4 parses one
//! shape and not two.

use std::io::Write;

use mailbert_core::{
    index::MailIndex,
    message_id::MessageId,
    rank::Row,
    threading::ThreadId,
};
use serde::Serialize;

use crate::{Tool, cli, error::Result, search, show};

/// What `thread` writes as JSON. (§10.4)
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Answer {
    /// The identity that the reader gave.
    pub id: String,

    /// The thread of §5.5, in full.
    pub thread: String,

    /// One row for each message, earliest first.
    pub rows: Vec<search::Line>,
}

/// The messages of one thread, as rows of §10.1.
///
/// The rows come in the order that [`MailIndex::thread`] gives, which
/// is the order that the messages arrived.
///
/// # Errors
///
/// The function fails if the index refuses.
pub fn rows(index: &MailIndex, thread: ThreadId) -> Result<Vec<Row>> {
    let hits = index.thread(thread)?;
    let total = hits.len();

    Ok(hits
        .into_iter()
        .enumerate()
        .map(|(at, hit)| Row {
            hit,
            score: 0.0,
            position: at + 1,
            total,
        })
        .collect())
}

/// The thread that holds one message, and its rows.
///
/// # Errors
///
/// The function fails if the index does not hold the message, or if
/// the index refuses.
pub fn of(index: &MailIndex, id: &MessageId) -> Result<(ThreadId, Vec<Row>)> {
    let hit = index
        .get(id)?
        .ok_or_else(|| crate::error::Error::NotIndexed(id.short()))?;

    Ok((hit.thread, rows(index, hit.thread)?))
}

/// Write the answer as JSON. (§10.4)
///
/// # Errors
///
/// The function fails if the output does not take the text.
pub fn write_json(answer: &Answer, out: &mut dyn Write) -> Result<()> {
    writeln!(out, "{}", serde_json::to_string_pretty(answer)?)?;

    Ok(())
}

/// Do the work of `thread`. (§8.4)
///
/// # Errors
///
/// The function fails if the message is not there, or if the output
/// does not take the text.
pub fn command(tool: &Tool, args: &cli::One) -> Result<()> {
    let store = tool.store()?;
    let index = tool.index()?;
    let id = show::resolve(&store, &args.id)?;
    let (thread, rows) = of(&index, &id)?;
    let lines = search::lines(&store, &rows, &[], false)?;
    let mut out = std::io::stdout().lock();

    match args.json {
        true => write_json(
            &Answer {
                id: id.short(),
                thread: thread.full_hex(),
                rows: lines,
            },
            &mut out,
        ),
        false => search::write_text(&lines, &mut out),
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_each_message_of_a_thread_has_its_own_place` | invariant | §10.1 writes `[3/7]`. Two rows with the same place, or a place above the total, tells the reader a number that means nothing. |
    //! | `prop_a_thread_holds_its_messages_and_no_other` | round-trip | A thread that loses a message hides a reply. A thread that gains one shows the mail of another conversation. |

    use std::collections::{BTreeSet, HashSet};

    use hegel::{TestCase, generators as gs};
    use mailbert_core::{
        Store,
        message::{Location, Message},
        mime,
    };
    use tempfile::{TempDir, tempdir};

    use super::*;

    /// The smallest budget that a Tantivy writer accepts.
    const BUDGET: usize = 15_000_000;

    fn location(uid: u32) -> Location {
        Location {
            account: "work".to_string(),
            folder: "INBOX".to_string(),
            uid,
            uid_validity: 1,
            received: 1_755_820_800,
            flags: BTreeSet::new(),
        }
    }

    fn raw(key: &str, subject: &str, minute: u32) -> Vec<u8> {
        format!(
            "From: alice@example.test\r\n\
             To: bob@example.test\r\n\
             Subject: {subject}\r\n\
             Date: Fri, 22 Aug 2025 09:{minute:02}:00 +0000\r\n\
             Message-ID: <{key}@x.test>\r\n\
             \r\n\
             the rent is late\r\n"
        )
        .into_bytes()
    }

    struct Shelf {
        _dir: TempDir,
        store: Store,
        index: MailIndex,
    }

    impl Shelf {
        fn new() -> Self {
            let dir = tempdir().expect("a temporary directory");
            let store =
                Store::open(&dir.path().join("store")).expect("a store");
            let index = MailIndex::open_in_ram().expect("an index");

            Self {
                _dir: dir,
                store,
                index,
            }
        }

        /// Write one message, and put it in the thread that is given.
        fn put(
            &self,
            key: &str,
            minute: u32,
            thread: Option<ThreadId>,
        ) -> (MessageId, ThreadId) {
            let bytes = raw(key, "Deposit", minute);
            let message = Message::new(
                mime::parse(&bytes).expect("a message"),
                location(minute + 1),
                Vec::<String>::new(),
            );
            let held = self.store.put(&message, &bytes).expect("a write");
            let thread = thread.unwrap_or_else(|| ThreadId::from_root(held.id));

            let mut writer = self.index.writer(BUDGET).expect("a writer");
            self.index
                .add(&writer, &held, thread, &BTreeSet::new())
                .expect("an index write");
            self.index.commit(&mut writer).expect("a commit");

            (held.id, thread)
        }

        /// A conversation of `count` messages, earliest first.
        fn conversation(&self, count: u32) -> (ThreadId, Vec<MessageId>) {
            let (first, thread) = self.put("m0", 0, None);
            let mut ids = vec![first];

            for number in 1..count {
                let key = format!("m{number}");
                let (id, _) = self.put(&key, number, Some(thread));
                ids.push(id);
            }

            (thread, ids)
        }
    }

    fn text_of(lines: &[search::Line]) -> String {
        let mut out = Vec::new();
        search::write_text(lines, &mut out).expect("a write");

        String::from_utf8(out).expect("the output is text")
    }

    // -----------------------------------------------------------------
    // The rows of §8.4.
    // -----------------------------------------------------------------

    #[test]
    fn a_thread_holds_every_message_of_the_conversation() {
        let shelf = Shelf::new();
        let (thread, ids) = shelf.conversation(4);

        let found = rows(&shelf.index, thread).expect("the rows");

        assert_eq!(found.len(), ids.len());
    }

    #[test]
    fn the_rows_of_a_thread_come_earliest_first() {
        let shelf = Shelf::new();
        let (thread, _) = shelf.conversation(4);

        let found = rows(&shelf.index, thread).expect("the rows");
        let dates: Vec<i64> = found.iter().map(|row| row.hit.date).collect();
        let mut sorted = dates.clone();
        sorted.sort_unstable();

        assert_eq!(dates, sorted);
    }

    #[test]
    fn each_row_knows_its_place_and_the_total() {
        let shelf = Shelf::new();
        let (thread, ids) = shelf.conversation(3);

        let found = rows(&shelf.index, thread).expect("the rows");

        for (at, row) in found.iter().enumerate() {
            assert_eq!(row.position, at + 1);
            assert_eq!(row.total, ids.len());
        }
    }

    #[test]
    fn a_thread_of_one_message_gives_one_row() {
        let shelf = Shelf::new();
        let (id, _) = shelf.put("only", 0, None);

        let (_, found) = of(&shelf.index, &id).expect("the rows");

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].position, 1);
        assert_eq!(found[0].total, 1);
    }

    #[test]
    fn a_message_of_another_conversation_is_not_in_the_rows() {
        let shelf = Shelf::new();
        let (thread, ids) = shelf.conversation(2);
        let (other, _) = shelf.put("apart", 9, None);

        let found = rows(&shelf.index, thread).expect("the rows");
        let held: HashSet<MessageId> =
            found.iter().map(|row| row.hit.id).collect();

        assert_eq!(held.len(), ids.len());
        assert!(!held.contains(&other), "the thread took another message");
    }

    #[test]
    fn the_thread_of_a_message_is_the_thread_that_the_index_holds() {
        let shelf = Shelf::new();
        let (thread, ids) = shelf.conversation(3);

        let (found, _) = of(&shelf.index, &ids[2]).expect("the rows");

        assert_eq!(found, thread);
    }

    /// A message can be in the store and not in the index, because a
    /// sync writes the store first. The command must say so, and never
    /// give an empty thread.
    #[test]
    fn a_message_that_the_index_does_not_hold_is_an_error() {
        let shelf = Shelf::new();
        let id = MessageId::derive(
            Some("<away@x.test>"),
            0,
            "a@x.test",
            "One",
            "one",
        );

        let result = of(&shelf.index, &id);

        assert!(
            matches!(result, Err(crate::error::Error::NotIndexed(_))),
            "{result:?}"
        );
    }

    // -----------------------------------------------------------------
    // The output.
    // -----------------------------------------------------------------

    #[test]
    fn the_text_writes_one_line_for_each_message() {
        let shelf = Shelf::new();
        let (thread, ids) = shelf.conversation(3);
        let found = rows(&shelf.index, thread).expect("the rows");
        let lines =
            search::lines(&shelf.store, &found, &[], false).expect("the lines");

        let held = text_of(&lines);

        assert_eq!(held.lines().count(), ids.len(), "{held}");
        assert!(held.contains("[1/3]"), "{held}");
        assert!(held.contains("[3/3]"), "{held}");
    }

    #[test]
    fn the_json_names_the_thread_and_its_rows() {
        let shelf = Shelf::new();
        let (thread, ids) = shelf.conversation(2);
        let found = rows(&shelf.index, thread).expect("the rows");
        let lines =
            search::lines(&shelf.store, &found, &[], false).expect("the lines");
        let answer = Answer {
            id: ids[0].short(),
            thread: thread.full_hex(),
            rows: lines,
        };

        let mut out = Vec::new();
        write_json(&answer, &mut out).expect("a write");
        let held: serde_json::Value =
            serde_json::from_slice(&out).expect("the JSON parses");

        assert_eq!(held["thread"], thread.full_hex());
        assert_eq!(held["id"], ids[0].short());
        assert_eq!(held["rows"].as_array().expect("an array").len(), 2);
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 25)]
    fn prop_each_message_of_a_thread_has_its_own_place(tc: TestCase) {
        let count = tc.draw(gs::integers::<u32>().min_value(1).max_value(6));
        let shelf = Shelf::new();
        let (thread, _) = shelf.conversation(count);

        let found = rows(&shelf.index, thread).expect("the rows");
        let places: BTreeSet<usize> =
            found.iter().map(|row| row.position).collect();

        assert_eq!(places.len(), found.len(), "two rows share a place");
        assert_eq!(places.iter().copied().min(), Some(1));
        assert_eq!(places.iter().copied().max(), Some(found.len()));
        assert!(found.iter().all(|row| row.total == found.len()));
    }

    #[hegel::test(test_cases = 25)]
    fn prop_a_thread_holds_its_messages_and_no_other(tc: TestCase) {
        let count = tc.draw(gs::integers::<u32>().min_value(1).max_value(6));
        let apart = tc.draw(gs::integers::<u32>().min_value(0).max_value(3));
        let shelf = Shelf::new();
        let (thread, ids) = shelf.conversation(count);

        for number in 0..apart {
            shelf.put(&format!("apart{number}"), 30 + number, None);
        }

        let found = rows(&shelf.index, thread).expect("the rows");
        let held: BTreeSet<MessageId> =
            found.iter().map(|row| row.hit.id).collect();
        let want: BTreeSet<MessageId> = ids.into_iter().collect();

        assert_eq!(held, want);
    }
}
