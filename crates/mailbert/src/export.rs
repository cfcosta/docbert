//! `export`, which writes a maildir of the messages that a query
//! matches. (§4.3)
//!
//! The store is not a maildir, so no MUA can read it. `export` writes
//! the bytes of the server into `cur`, and mutt, aerc, and neomutt
//! then open the result of a query.
//!
//! §4.3 offers a maildir of symlinks. That mode is not possible, and
//! this module writes the bytes each time. §4.2 keeps the bytes in
//! LMDB, so there is no file behind a message to point a link at.
//!
//! The flags of a message are part of the name that a maildir gives
//! it, so an export removes the copy that an earlier export wrote
//! before it writes the new one. The maildir then holds one copy of
//! each message, whatever its flags were.

use std::{collections::BTreeSet, fs, io::Write, path::Path};

use mailbert_core::{
    Store,
    compile::{self, Vocabulary},
    date::Clock,
    index::MailIndex,
    message::{ANSWERED, DELETED, DRAFT, FLAGGED, SEEN},
    message_id::MessageId,
    query,
};

use crate::{Tool, cli, error::Result};

/// The three directories that every maildir holds.
pub const PARTS: [&str; 3] = ["cur", "new", "tmp"];

/// What comes after the identity in the name of a message.
///
/// The mark tells a later export which files are its own, so it can
/// remove them and leave the rest of the maildir alone.
pub const MARK: &str = ".mailbert:2,";

/// The IMAP flags that a maildir names, and the letter of each.
///
/// The letters are in the order of their bytes, because the maildir
/// format asks for that order.
pub const LETTERS: [(&str, char); 5] = [
    (DRAFT, 'D'),
    (FLAGGED, 'F'),
    (ANSWERED, 'R'),
    (SEEN, 'S'),
    (DELETED, 'T'),
];

/// What one export did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    /// How many messages the export wrote.
    pub wrote: usize,

    /// How many messages had no bytes in the store.
    pub missing: usize,
}

/// The name of one message inside `cur`. (§4.3)
///
/// The identity of §4.1 is the unique part, so a second export of the
/// same message gives the same name. The letters come after [`MARK`].
pub fn name(id: &MessageId, flags: &BTreeSet<String>) -> String {
    let mut held = String::with_capacity(id.full_hex().len() + MARK.len() + 5);
    held.push_str(&id.full_hex());
    held.push_str(MARK);

    // The array is in the order of the bytes, so the name is too.
    for (flag, letter) in LETTERS {
        if flags.contains(flag) {
            held.push(letter);
        }
    }

    held
}

/// The letters of a maildir name, or `None` when the name is not ours.
pub fn letters(name: &str) -> Option<&str> {
    let at = name.find(MARK)?;

    Some(&name[at + MARK.len()..])
}

/// Make the three directories of a maildir. (§4.3)
///
/// # Errors
///
/// The function fails if it cannot make a directory.
pub fn make(dir: &Path) -> Result<()> {
    for part in PARTS {
        fs::create_dir_all(dir.join(part))?;
    }

    Ok(())
}

/// Remove the copy that an earlier export left, whatever its flags.
///
/// The name of a message holds its flags, so a message that the reader
/// opened has a new name. The maildir would hold it two times without
/// this step. A file that mailbert did not write has no [`MARK`], so
/// this step never touches it.
///
/// # Errors
///
/// The function fails if it cannot remove a file.
pub fn drop_old(dir: &Path, id: &MessageId) -> Result<()> {
    let start = format!("{}{MARK}", id.full_hex());
    let Ok(entries) = fs::read_dir(dir.join("cur")) else {
        return Ok(());
    };

    for entry in entries.flatten() {
        if entry.file_name().to_string_lossy().starts_with(&start) {
            fs::remove_file(entry.path())?;
        }
    }

    Ok(())
}

/// Write one message into a maildir. (§4.3)
///
/// The bytes go to `tmp` first and then move to `cur`. An MUA that
/// reads the maildir at that moment then sees no half-written file.
///
/// Gives `false` when the store holds no bytes for the identity.
///
/// # Errors
///
/// The function fails if the store or the disk refuses.
pub fn write_one(store: &Store, id: &MessageId, dir: &Path) -> Result<bool> {
    let Some(raw) = store.raw(id)? else {
        return Ok(false);
    };

    let flags = match store.get(id)? {
        Some(message) => message.flags,
        None => BTreeSet::new(),
    };
    let held = name(id, &flags);
    let staged = dir.join("tmp").join(&held);

    let mut file = fs::File::create(&staged)?;
    file.write_all(&raw)?;
    file.sync_all()?;
    drop(file);

    drop_old(dir, id)?;
    fs::rename(&staged, dir.join("cur").join(&held))?;

    Ok(true)
}

/// Every message that a query names, newest first. (§4.3)
///
/// The list is not a page. `export` writes each message that matches,
/// and never the best 100 of them, because a reader cannot see that
/// the other messages are away.
///
/// # Errors
///
/// The function fails if the query is bad, or if the index refuses.
pub fn matching(
    store: &Store,
    index: &MailIndex,
    text: &str,
    clock: Clock,
) -> Result<Vec<MessageId>> {
    let asked = query::parse(text, clock)?;
    let vocabulary = Vocabulary::from_store(store)?;
    let compiled = compile::compile(&asked, index, &vocabulary, clock)?;
    let mut hits = index.top(&*compiled.search, index.len().max(1))?;

    // §10.1 shows the newest message first, and the export writes the
    // messages in the same order. A tie goes to the identity, so that
    // two runs give the same list.
    hits.sort_by(|a, b| b.date.cmp(&a.date).then(a.id.cmp(&b.id)));

    Ok(hits.into_iter().map(|hit| hit.id).collect())
}

/// Write a maildir of the messages that a query names. (§4.3)
///
/// The query runs before the maildir exists, so a query that is bad
/// leaves no directory behind.
///
/// # Errors
///
/// The function fails if the query is bad, or if the disk refuses.
pub fn export(
    store: &Store,
    index: &MailIndex,
    text: &str,
    dir: &Path,
    clock: Clock,
) -> Result<Report> {
    let ids = matching(store, index, text, clock)?;

    make(dir)?;

    let mut report = Report {
        wrote: 0,
        missing: 0,
    };

    for id in &ids {
        match write_one(store, id, dir)? {
            true => report.wrote += 1,
            false => report.missing += 1,
        }
    }

    Ok(report)
}

/// The line that `export` writes when it is done.
pub fn line(report: &Report, dir: &Path) -> String {
    let mut held = format!(
        "wrote {} {} to {}",
        report.wrote,
        match report.wrote {
            1 => "message",
            _ => "messages",
        },
        dir.display()
    );

    if report.missing > 0 {
        held.push_str(&format!(
            ", and {} {} no bytes in the store",
            report.missing,
            match report.missing {
                1 => "message has",
                _ => "messages have",
            }
        ));
    }

    held
}

/// Do the work of `export`. (§4.3)
///
/// # Errors
///
/// The function fails if the query is bad, or if the disk refuses.
pub fn command(tool: &Tool, args: &cli::Export) -> Result<()> {
    let store = tool.store()?;
    let index = tool.index()?;
    let report =
        export(&store, &index, &args.query, &args.dir, crate::clock())?;

    writeln!(std::io::stdout().lock(), "{}", line(&report, &args.dir))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_name_reads_its_flags_back` | round-trip | The name is where a maildir keeps the flags. A name that loses one shows read mail as unread in the MUA. |
    //! | `prop_the_letters_of_a_name_hold_their_order` | invariant | The maildir format asks for the order of the bytes. An MUA that reads them in another order sees no flag at all. |
    //! | `prop_an_export_keeps_every_byte` | round-trip | §4.3 gives the bytes of the server to the MUA. One byte that moves breaks a signature, and can break the parse. |
    //! | `prop_a_second_export_leaves_one_copy` | idempotence | A reader exports the same query again after a sync. Two copies of one message is the failure that a naive name gives. |

    use std::{collections::BTreeSet, path::PathBuf};

    use hegel::{TestCase, generators as gs};
    use mailbert_core::{
        message::{Location, Message},
        mime,
        threading::ThreadId,
    };
    use tempfile::{TempDir, tempdir};

    use super::*;

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    /// The smallest budget that a Tantivy writer accepts.
    const BUDGET: usize = 15_000_000;

    /// A moment inside the day that the test messages carry.
    const NOW: i64 = 1_755_900_000;

    fn clock() -> Clock {
        Clock::utc(NOW)
    }

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

    fn raw(key: &str, subject: &str, body: &str) -> Vec<u8> {
        raw_at(key, subject, body, "Fri, 22 Aug 2025 09:30:00 +0000")
    }

    fn raw_at(key: &str, subject: &str, body: &str, when: &str) -> Vec<u8> {
        format!(
            "From: alice@example.test\r\n\
             To: bob@example.test\r\n\
             Subject: {subject}\r\n\
             Date: {when}\r\n\
             Message-ID: <{key}@x.test>\r\n\
             \r\n\
             {body}\r\n"
        )
        .into_bytes()
    }

    /// A store, an index, and the maildir that the export writes.
    struct Shelf {
        _dir: TempDir,
        store: Store,
        index: MailIndex,
        out: PathBuf,
    }

    impl Shelf {
        fn new() -> Self {
            let dir = tempdir().expect("a temporary directory");
            let store =
                Store::open(&dir.path().join("store")).expect("a store");
            let index = MailIndex::open_in_ram().expect("an index");
            let out = dir.path().join("maildir");

            Self {
                _dir: dir,
                store,
                index,
                out,
            }
        }

        fn put(&self, key: &str, subject: &str, body: &str) -> MessageId {
            self.put_with(key, subject, body, Vec::<String>::new())
        }

        fn put_with(
            &self,
            key: &str,
            subject: &str,
            body: &str,
            flags: Vec<String>,
        ) -> MessageId {
            let bytes = raw(key, subject, body);
            let message = Message::new(
                mime::parse(&bytes).expect("a message"),
                location(1),
                flags,
            );
            let held = self.store.put(&message, &bytes).expect("a write");
            let thread = ThreadId::from_root(held.id);
            let tags = self.store.tags_of(&held.id).expect("the tags");

            let mut writer = self.index.writer(BUDGET).expect("a writer");
            self.index
                .add(&writer, &held, thread, &tags)
                .expect("an index write");
            self.index.commit(&mut writer).expect("a commit");

            held.id
        }

        /// A message that the store knows, whose bytes it does not hold.
        ///
        /// §3.2 writes the bytes before the entry, so a stop between
        /// the two leaves a message in this state.
        fn put_bare(&self, key: &str, subject: &str) -> MessageId {
            let bytes = raw(key, subject, "the rent is late");
            let message = Message::new(
                mime::parse(&bytes).expect("a message"),
                location(1),
                Vec::<String>::new(),
            );
            let held = self.store.put(&message, &[]).expect("a write");
            let thread = ThreadId::from_root(held.id);

            let mut writer = self.index.writer(BUDGET).expect("a writer");
            self.index
                .add(&writer, &held, thread, &BTreeSet::new())
                .expect("an index write");
            self.index.commit(&mut writer).expect("a commit");

            held.id
        }

        /// Write `count` messages with one commit, which is fast.
        fn put_many(&self, count: usize) -> Vec<MessageId> {
            let mut writer = self.index.writer(BUDGET).expect("a writer");
            let mut ids = Vec::with_capacity(count);

            for number in 0..count {
                // Each message takes its own minute, so the order of
                // the list is a thing that a test can see.
                let when = format!(
                    "Fri, 22 Aug 2025 {:02}:{:02}:00 +0000",
                    number / 60,
                    number % 60
                );
                let bytes =
                    raw_at(&format!("m{number}"), "Deposit", "the rent", &when);
                let message = Message::new(
                    mime::parse(&bytes).expect("a message"),
                    location(number as u32 + 1),
                    Vec::<String>::new(),
                );
                let held = self.store.put(&message, &bytes).expect("a write");
                let thread = ThreadId::from_root(held.id);

                self.index
                    .add(&writer, &held, thread, &BTreeSet::new())
                    .expect("an index write");
                ids.push(held.id);
            }

            self.index.commit(&mut writer).expect("a commit");

            ids
        }

        fn run(&self, text: &str) -> Result<Report> {
            export(&self.store, &self.index, text, &self.out, clock())
        }

        /// The names inside `cur`, in order.
        fn kept(&self) -> Vec<String> {
            let mut names: Vec<String> = fs::read_dir(self.out.join("cur"))
                .expect("a read")
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();

            names
        }
    }

    fn flags_of(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|one| (*one).to_string()).collect()
    }

    // -----------------------------------------------------------------
    // The name of §4.3.
    // -----------------------------------------------------------------

    #[test]
    fn a_message_with_no_flag_has_an_empty_info() {
        let id =
            MessageId::derive(Some("<a@x.test>"), 0, "a@x.test", "One", "one");

        assert_eq!(
            name(&id, &BTreeSet::new()),
            format!("{}{MARK}", id.full_hex())
        );
    }

    #[test]
    fn each_imap_flag_puts_its_letter_in_the_name() {
        let id =
            MessageId::derive(Some("<a@x.test>"), 0, "a@x.test", "One", "one");

        for (flag, letter) in LETTERS {
            let held = name(&id, &flags_of(&[flag]));

            assert_eq!(letters(&held), Some(letter.to_string().as_str()));
        }
    }

    #[test]
    fn the_letters_of_a_name_are_in_the_order_of_their_bytes() {
        let id =
            MessageId::derive(Some("<a@x.test>"), 0, "a@x.test", "One", "one");
        let held = name(&id, &flags_of(&[SEEN, DRAFT, ANSWERED, FLAGGED]));

        assert_eq!(letters(&held), Some("DFRS"));
    }

    /// §9 keeps the tags of mailbert away from the server, and a
    /// maildir has no letter for them.
    #[test]
    fn a_tag_never_becomes_a_letter() {
        let id =
            MessageId::derive(Some("<a@x.test>"), 0, "a@x.test", "One", "one");
        let held = name(&id, &flags_of(&["todo", "\\encrypted", SEEN]));

        assert_eq!(letters(&held), Some("S"));
    }

    #[test]
    fn a_name_that_is_not_ours_has_no_letters() {
        assert_eq!(letters("1755900000.M1P2.host"), None);
    }

    // -----------------------------------------------------------------
    // The maildir of §4.3.
    // -----------------------------------------------------------------

    #[test]
    fn an_export_makes_the_three_directories() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");

        shelf.run("deposit").expect("an export");

        for part in PARTS {
            assert!(shelf.out.join(part).is_dir(), "{part} is not there");
        }
    }

    #[test]
    fn an_export_writes_the_bytes_that_the_server_gave() {
        let shelf = Shelf::new();
        let id = shelf.put("a", "Deposit", "the rent is late");

        shelf.run("deposit").expect("an export");

        let names = shelf.kept();
        assert_eq!(names.len(), 1, "{names:?}");
        let held =
            fs::read(shelf.out.join("cur").join(&names[0])).expect("a read");

        assert_eq!(Some(held), shelf.store.raw(&id).expect("a read"));
    }

    #[test]
    fn an_export_leaves_the_temporary_directory_empty() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");

        shelf.run("deposit").expect("an export");

        let left = fs::read_dir(shelf.out.join("tmp")).expect("a read").count();

        assert_eq!(left, 0);
    }

    #[test]
    fn the_name_of_a_written_message_holds_its_flags() {
        let shelf = Shelf::new();
        shelf.put_with(
            "a",
            "Deposit",
            "the rent is late",
            vec![SEEN.to_string(), FLAGGED.to_string()],
        );

        shelf.run("deposit").expect("an export");

        let names = shelf.kept();
        assert_eq!(letters(&names[0]), Some("FS"), "{names:?}");
    }

    #[test]
    fn an_export_writes_only_the_messages_that_the_query_names() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");
        shelf.put("b", "Holiday", "the beach is warm");

        let report = shelf.run("subject:deposit").expect("an export");

        assert_eq!(report.wrote, 1);
        assert_eq!(shelf.kept().len(), 1);
    }

    /// §4.3 gives the MUA every message that matches, and never a page
    /// of them. A reader who exports 300 messages and gets 100 has no
    /// way to see that 200 are not there.
    #[test]
    fn an_export_writes_every_match_and_never_a_page() {
        let shelf = Shelf::new();
        let ids = shelf.put_many(140);

        let report = shelf.run("subject:deposit").expect("an export");

        assert_eq!(report.wrote, ids.len());
        assert_eq!(shelf.kept().len(), ids.len());
    }

    #[test]
    fn a_message_with_no_bytes_counts_as_missing() {
        let shelf = Shelf::new();
        shelf.put_bare("a", "Deposit");

        let report = shelf.run("subject:deposit").expect("an export");

        assert_eq!(report.wrote, 0);
        assert_eq!(report.missing, 1);
    }

    #[test]
    fn a_second_export_leaves_one_copy_of_a_message() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");
        shelf.run("deposit").expect("an export");

        // The reader opened the message, so the flags changed.
        shelf.put_with(
            "a",
            "Deposit",
            "the rent is late",
            vec![SEEN.to_string()],
        );
        shelf.run("deposit").expect("a second export");

        let names = shelf.kept();
        assert_eq!(names.len(), 1, "{names:?}");
        assert_eq!(letters(&names[0]), Some("S"));
    }

    /// A maildir can hold mail that mailbert did not write. An export
    /// must never remove it.
    #[test]
    fn an_export_leaves_a_file_that_it_did_not_write() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");
        make(&shelf.out).expect("a maildir");
        let other = shelf.out.join("cur").join("1755900000.M1P2.host:2,S");
        fs::write(&other, b"From: someone\r\n\r\nhello\r\n").expect("a write");

        shelf.run("deposit").expect("an export");

        assert!(other.is_file(), "the export removed mail of another tool");
    }

    #[test]
    fn a_query_that_names_nothing_writes_an_empty_maildir() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");

        let report = shelf.run("subject:holiday").expect("an export");

        assert_eq!(report.wrote, 0);
        assert_eq!(shelf.kept().len(), 0);
    }

    #[test]
    fn a_bad_query_stops_the_export() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");

        let result = shelf.run("from:");

        assert!(matches!(result, Err(crate::error::Error::Query(_))));
        assert!(!shelf.out.exists(), "a bad query made a maildir");
    }

    #[test]
    fn the_list_of_matches_takes_the_newest_message_first() {
        let shelf = Shelf::new();
        let ids = shelf.put_many(3);

        let found = matching(&shelf.store, &shelf.index, "deposit", clock())
            .expect("a list");

        assert_eq!(found.len(), ids.len());
        let dates: Vec<i64> = found
            .iter()
            .map(|id| {
                shelf
                    .store
                    .get(id)
                    .expect("a read")
                    .expect("a message")
                    .date
            })
            .collect();
        let mut sorted = dates.clone();
        sorted.sort_by(|a, b| b.cmp(a));

        assert_eq!(dates, sorted);
    }

    // -----------------------------------------------------------------
    // The line that the reader sees.
    // -----------------------------------------------------------------

    #[test]
    fn the_line_says_how_many_messages_the_export_wrote() {
        let report = Report {
            wrote: 3,
            missing: 0,
        };
        let held = line(&report, Path::new("/home/me/mail/todo"));

        assert!(held.contains('3'), "{held}");
        assert!(held.contains("/home/me/mail/todo"), "{held}");
        assert!(!held.contains("missing"), "{held}");
    }

    #[test]
    fn the_line_names_the_messages_that_have_no_bytes() {
        let report = Report {
            wrote: 3,
            missing: 2,
        };
        let held = line(&report, Path::new("/home/me/mail/todo"));

        assert!(held.contains('2'), "{held}");
        assert!(held.contains("no bytes"), "{held}");
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::composite]
    fn some_flags(tc: TestCase) -> BTreeSet<String> {
        let mut held = BTreeSet::new();

        for (flag, _) in LETTERS {
            if tc.draw(gs::booleans()) {
                held.insert(flag.to_string());
            }
        }

        let tags = tc.draw(
            gs::vecs(gs::text().alphabet("abc").min_size(1).max_size(3))
                .min_size(0)
                .max_size(3),
        );
        held.extend(tags);

        held
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_name_reads_its_flags_back(tc: TestCase) {
        let flags = tc.draw(some_flags());
        let id =
            MessageId::derive(Some("<a@x.test>"), 0, "a@x.test", "One", "one");
        let held = name(&id, &flags);
        let read: BTreeSet<char> =
            letters(&held).expect("our own name").chars().collect();
        let want: BTreeSet<char> = LETTERS
            .iter()
            .filter(|(flag, _)| flags.contains(*flag))
            .map(|(_, letter)| *letter)
            .collect();

        assert_eq!(read, want);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_letters_of_a_name_hold_their_order(tc: TestCase) {
        let flags = tc.draw(some_flags());
        let id =
            MessageId::derive(Some("<a@x.test>"), 0, "a@x.test", "One", "one");
        let held = name(&id, &flags);
        let read = letters(&held).expect("our own name");
        let mut sorted: Vec<char> = read.chars().collect();
        sorted.sort_unstable();

        assert_eq!(read.chars().collect::<Vec<char>>(), sorted);
    }

    #[hegel::test(test_cases = 30)]
    fn prop_an_export_keeps_every_byte(tc: TestCase) {
        let body = tc
            .draw(gs::text().alphabet("abcdefg \n.").min_size(1).max_size(80));
        let shelf = Shelf::new();
        let id = shelf.put("a", "Deposit", &body);

        shelf.run("subject:deposit").expect("an export");

        let names = shelf.kept();
        assert_eq!(names.len(), 1, "{names:?}");
        let held =
            fs::read(shelf.out.join("cur").join(&names[0])).expect("a read");

        assert_eq!(Some(held), shelf.store.raw(&id).expect("a read"));
    }

    #[hegel::test(test_cases = 20)]
    fn prop_a_second_export_leaves_one_copy(tc: TestCase) {
        let flags = tc.draw(some_flags());
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");
        shelf.run("subject:deposit").expect("an export");

        shelf.put_with(
            "a",
            "Deposit",
            "the rent is late",
            flags.iter().cloned().collect(),
        );
        shelf.run("subject:deposit").expect("a second export");

        assert_eq!(shelf.kept().len(), 1);
    }
}
