//! What `mailbert status` shows. (§10.4)
//!
//! The report gives the counts that tell you if the store, the index,
//! and the vectors agree. A store that holds more messages than the
//! index means that a sync stopped early, and the next sync fixes it.
//!
//! The report never speaks to the IMAP server. §3.3 makes mailbert a
//! download-only mirror, and `status` reads only what is on the disk.

use std::{collections::BTreeMap, io::Write};

use mailbert_core::{Store, config::Config, date::Clock, index::MailIndex};
use serde::Serialize;

use crate::{Tool, cli, error::Result, show};

/// The width of the label column of the text. (§10.1)
const LABEL: usize = 12;

/// The width of the folder column of the text. (§10.1)
const FOLDER: usize = 20;

/// Where one folder of one account stopped. (§3.3)
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Folder {
    /// The name that the server gives the folder.
    pub name: String,

    /// The UIDVALIDITY that the last sync saw.
    pub uid_validity: u32,

    /// The UID that the folder gives to the next message.
    pub uid_next: u32,

    /// True when the last sync left UIDs that it did not fetch.
    pub pending: bool,

    /// When the last sync marked the folder, in seconds of Unix time.
    ///
    /// A zero says that no sync marked it yet.
    pub synced: i64,
}

/// What one account of the configuration holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Mailbox {
    /// The name of the account in the configuration.
    pub account: String,

    /// The folders that a sync marked, by name.
    pub folders: Vec<Folder>,
}

/// What `status` found. (§10.4)
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Report {
    /// The messages that the store holds. (§4.2)
    pub messages: usize,

    /// The documents that the index holds. (§6.1)
    pub indexed: usize,

    /// The messages that carry an embedding. (§6.2)
    pub embedded: usize,

    /// The saved searches of §9.
    pub searches: usize,

    /// Each tag of §9, and the messages that carry it.
    pub tags: BTreeMap<String, usize>,

    /// One entry for each account of the configuration.
    pub accounts: Vec<Mailbox>,
}

impl Report {
    /// The messages that the store holds and the index does not.
    ///
    /// §3.2 writes the store before the index, so a sync that stopped
    /// leaves this number above zero. The next sync makes it zero.
    pub fn behind(&self) -> usize {
        self.messages.saturating_sub(self.indexed)
    }

    /// When the newest sync of any folder ran. (§3.3)
    ///
    /// `None` says that no sync marked a folder yet. A zero in the
    /// record is not a time, and this drops it.
    pub fn synced(&self) -> Option<i64> {
        self.accounts
            .iter()
            .flat_map(|mailbox| &mailbox.folders)
            .map(|folder| folder.synced)
            .filter(|at| *at > 0)
            .max()
    }
}

/// The time of a sync, as §10.1 writes it.
///
/// A folder that no sync marked carries a zero, and a zero is not a
/// time. The reader sees `never` for it.
fn when(at: i64, clock: Clock) -> String {
    match at > 0 {
        true => show::stamp(at, clock.utc_offset()),
        false => "never".to_string(),
    }
}

/// Read the counts of the store, the index, and the configuration.
///
/// # Errors
///
/// The function fails if the store or the index refuses.
pub fn report(
    store: &Store,
    index: &MailIndex,
    config: &Config,
) -> Result<Report> {
    let mut accounts = Vec::with_capacity(config.accounts.len());

    for account in &config.accounts {
        let folders = store
            .states(&account.name)?
            .into_iter()
            .map(|(name, state)| Folder {
                name,
                uid_validity: state.uid_validity,
                uid_next: state.uid_next,
                pending: !state.pending.trim().is_empty(),
                synced: state.synced_at,
            })
            .collect();

        accounts.push(Mailbox {
            account: account.name.clone(),
            folders,
        });
    }

    Ok(Report {
        messages: store.len()?,
        indexed: index.len(),
        embedded: store.embeddings()?.len(),
        searches: store.searches()?.len(),
        tags: store.all_tags()?,
        accounts,
    })
}

/// Write the report as lines of text.
///
/// `clock` gives the time zone of the reader. (§10.1)
///
/// # Errors
///
/// The function fails if the output does not take the text.
pub fn write_text(
    report: &Report,
    clock: Clock,
    out: &mut dyn Write,
) -> Result<()> {
    writeln!(out, "{:LABEL$}{}", "messages", report.messages)?;
    writeln!(out, "{:LABEL$}{}", "indexed", report.indexed)?;
    writeln!(out, "{:LABEL$}{}", "embedded", report.embedded)?;
    writeln!(out, "{:LABEL$}{}", "searches", report.searches)?;

    // §10.4 must say how old the mail is. A reader who does not see
    // this cannot tell an empty search from a store that is behind.
    writeln!(
        out,
        "{:LABEL$}{}",
        "synced",
        when(report.synced().unwrap_or(0), clock)
    )?;

    // §3.2 writes the store before the index, so this number is the
    // work that the next sync owes.
    if report.behind() > 0 {
        writeln!(
            out,
            "{:LABEL$}{}  (run `mailbert sync`)",
            "behind",
            report.behind()
        )?;
    }

    for (tag, count) in &report.tags {
        writeln!(out, "{:LABEL$}{count}", format!("tag {tag}"))?;
    }

    for mailbox in &report.accounts {
        writeln!(out, "{}", mailbox.account)?;

        for folder in &mailbox.folders {
            let owed = match folder.pending {
                true => "  owes UIDs",
                false => "",
            };

            writeln!(
                out,
                "  {:FOLDER$}uidvalidity {}  uidnext {}  {}{owed}",
                folder.name,
                folder.uid_validity,
                folder.uid_next,
                when(folder.synced, clock)
            )?;
        }
    }

    Ok(())
}

/// Write the report as the JSON of §10.4.
///
/// # Errors
///
/// The function fails if the output does not take the text.
pub fn write_json(report: &Report, out: &mut dyn Write) -> Result<()> {
    writeln!(out, "{}", serde_json::to_string_pretty(report)?)?;

    Ok(())
}

/// Do the work of `status`. (§10.4)
///
/// # Errors
///
/// The function fails if the store, the index, or the configuration
/// refuses.
pub fn command(tool: &Tool, args: &cli::Status) -> Result<()> {
    let store = tool.store()?;
    let index = tool.index()?;
    let held = report(&store, &index, &tool.config()?)?;
    let mut out = std::io::stdout().lock();

    match args.json {
        true => write_json(&held, &mut out),
        false => write_text(&held, crate::clock(), &mut out),
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_the_report_counts_what_the_store_holds` | model-based | A count against a tally. A report that lies about the counts hides a sync that stopped early. |
    //! | `prop_behind_is_never_below_zero` | algebraic | The difference of two counts. An index ahead of the store must give zero, and never a number that wraps. |
    //! | `prop_the_report_gives_the_newest_time_of_any_folder` | model-based | A maximum against a tally. A report that gives an old time tells the reader that mail is missing when it is there. |

    use std::collections::BTreeSet;

    use hegel::{TestCase, generators as gs};
    use mailbert_core::{
        config::{Account, ImapConfig},
        date::Clock,
        message::{Location, Message},
        mime,
        store::{Embedded, SyncState},
        threading::ThreadId,
    };
    use tempfile::{TempDir, tempdir};

    use super::*;

    /// The smallest budget that a Tantivy writer accepts.
    const BUDGET: usize = 15_000_000;

    /// The time that each mark of these tests carries.
    ///
    /// 1755820800 is 2025-08-22 00:00 UTC.
    const SYNCED: i64 = 1_755_820_800;

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

    fn account(name: &str) -> Account {
        Account {
            name: name.to_string(),
            imap: ImapConfig {
                host: "mail.example.test".to_string(),
                user: "me@cfcosta.com".to_string(),
                password: Some("secret".to_string()),
                connections: 1,
                ..ImapConfig::default()
            },
            all_folders: true,
            ..Account::default()
        }
    }

    fn config_of(names: &[&str]) -> Config {
        Config {
            accounts: names.iter().map(|name| account(name)).collect(),
            ..Config::default()
        }
    }

    struct Shelf {
        _dir: TempDir,
        store: Store,
        index: MailIndex,
        uid: std::cell::Cell<u32>,
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
                uid: std::cell::Cell::new(1),
            }
        }

        fn put(&self, key: &str, indexed: bool) -> mailbert_core::MessageId {
            let uid = self.uid.get();
            self.uid.set(uid + 1);
            let bytes = format!(
                "From: alice@example.test\r\n\
                 To: bob@example.test\r\n\
                 Subject: Deposit\r\n\
                 Date: Fri, 22 Aug 2025 09:30:00 +0000\r\n\
                 Message-ID: <{key}@x.test>\r\n\
                 \r\n\
                 the rent is late\r\n"
            )
            .into_bytes();
            let message = Message::new(
                mime::parse(&bytes).expect("a message"),
                location(uid),
                Vec::<String>::new(),
            );
            let held = self.store.put(&message, &bytes).expect("a write");

            if indexed {
                let mut writer = self.index.writer(BUDGET).expect("a writer");
                self.index
                    .add(
                        &writer,
                        &held,
                        ThreadId::from_root(held.id),
                        &BTreeSet::new(),
                    )
                    .expect("an index write");
                self.index.commit(&mut writer).expect("a commit");
            }

            held.id
        }

        /// Mark one folder as a sync of `at` left it.
        fn mark(&self, account: &str, folder: &str, at: i64) {
            self.store
                .mark(
                    account,
                    folder,
                    &SyncState {
                        uid_validity: 12,
                        uid_next: 431,
                        highest_mod_seq: 9,
                        pending: String::new(),
                        synced_at: at,
                    },
                )
                .expect("a mark");
        }

        fn report(&self, config: &Config) -> Report {
            super::report(&self.store, &self.index, config).expect("a report")
        }
    }

    fn text_of(report: &Report) -> String {
        let mut out = Vec::new();
        write_text(report, Clock::utc(SYNCED), &mut out).expect("a write");

        String::from_utf8(out).expect("the output is text")
    }

    /// The line of the text that says when the last sync ran.
    fn sync_line(text: &str) -> &str {
        text.lines()
            .find(|line| line.starts_with("synced"))
            .expect("a line for the sync")
    }

    fn json_of(report: &Report) -> serde_json::Value {
        let mut out = Vec::new();
        write_json(report, &mut out).expect("a write");

        serde_json::from_slice(&out).expect("the output is JSON")
    }

    // -----------------------------------------------------------------
    // The counts.
    // -----------------------------------------------------------------

    #[test]
    fn the_report_counts_the_messages_of_the_store() {
        let shelf = Shelf::new();
        shelf.put("a", true);
        shelf.put("b", true);

        assert_eq!(shelf.report(&Config::default()).messages, 2);
    }

    #[test]
    fn the_report_counts_the_documents_of_the_index() {
        let shelf = Shelf::new();
        shelf.put("a", true);
        shelf.put("b", false);

        assert_eq!(shelf.report(&Config::default()).indexed, 1);
    }

    /// §3.2 writes the store before the index. A store that runs ahead
    /// means that a sync stopped, and the reader must see it.
    #[test]
    fn the_report_says_how_far_the_index_is_behind() {
        let shelf = Shelf::new();
        shelf.put("a", true);
        shelf.put("b", false);

        assert_eq!(shelf.report(&Config::default()).behind(), 1);
    }

    #[test]
    fn an_index_that_holds_every_message_is_not_behind() {
        let shelf = Shelf::new();
        shelf.put("a", true);

        assert_eq!(shelf.report(&Config::default()).behind(), 0);
    }

    #[test]
    fn the_report_counts_the_messages_that_carry_an_embedding() {
        let shelf = Shelf::new();
        let first = shelf.put("a", true);
        shelf.put("b", true);
        shelf
            .store
            .mark_embedded(
                &first,
                &Embedded {
                    digest: [7; 32],
                    keys: vec![1],
                },
            )
            .expect("a mark");

        assert_eq!(shelf.report(&Config::default()).embedded, 1);
    }

    #[test]
    fn the_report_counts_the_saved_searches() {
        let shelf = Shelf::new();
        shelf.store.save_search("rent", "tag:todo").expect("a save");
        shelf
            .store
            .save_search("bills", "tag:later")
            .expect("a save");

        assert_eq!(shelf.report(&Config::default()).searches, 2);
    }

    #[test]
    fn the_report_counts_the_messages_of_each_tag() {
        let shelf = Shelf::new();
        let first = shelf.put("a", true);
        let second = shelf.put("b", true);
        shelf.store.tag(&first, "todo").expect("a tag");
        shelf.store.tag(&second, "todo").expect("a tag");
        shelf.store.tag(&second, "later").expect("a tag");

        let report = shelf.report(&Config::default());

        assert_eq!(report.tags.get("todo"), Some(&2));
        assert_eq!(report.tags.get("later"), Some(&1));
    }

    // -----------------------------------------------------------------
    // The accounts. (§3.3)
    // -----------------------------------------------------------------

    #[test]
    fn the_report_names_every_account_of_the_configuration() {
        let shelf = Shelf::new();

        let report = shelf.report(&config_of(&["work", "home"]));

        assert_eq!(report.accounts.len(), 2);
        assert_eq!(report.accounts[0].account, "work");
    }

    #[test]
    fn an_account_that_never_synced_holds_no_folder() {
        let shelf = Shelf::new();

        let report = shelf.report(&config_of(&["work"]));

        assert!(report.accounts[0].folders.is_empty());
    }

    #[test]
    fn a_folder_carries_where_the_last_sync_stopped() {
        let shelf = Shelf::new();
        shelf
            .store
            .mark(
                "work",
                "INBOX",
                &SyncState {
                    uid_validity: 12,
                    uid_next: 431,
                    highest_mod_seq: 9,
                    pending: String::new(),
                    synced_at: SYNCED,
                },
            )
            .expect("a mark");

        let report = shelf.report(&config_of(&["work"]));

        assert_eq!(report.accounts[0].folders.len(), 1);
        assert_eq!(report.accounts[0].folders[0].name, "INBOX");
        assert_eq!(report.accounts[0].folders[0].uid_validity, 12);
        assert_eq!(report.accounts[0].folders[0].uid_next, 431);
        assert!(!report.accounts[0].folders[0].pending);
    }

    /// §3.3 keeps the UIDs that a sync owes, so a stop shows up here.
    #[test]
    fn a_folder_that_owes_uids_is_pending() {
        let shelf = Shelf::new();
        shelf
            .store
            .mark(
                "work",
                "INBOX",
                &SyncState {
                    uid_validity: 12,
                    uid_next: 431,
                    highest_mod_seq: 9,
                    pending: "400:430".to_string(),
                    synced_at: SYNCED,
                },
            )
            .expect("a mark");

        let report = shelf.report(&config_of(&["work"]));

        assert!(report.accounts[0].folders[0].pending);
    }

    #[test]
    fn a_state_of_an_account_that_is_gone_never_shows() {
        let shelf = Shelf::new();
        shelf
            .store
            .mark("old", "INBOX", &SyncState::default())
            .expect("a mark");

        let report = shelf.report(&config_of(&["work"]));

        assert_eq!(report.accounts.len(), 1);
        assert_eq!(report.accounts[0].account, "work");
    }

    /// §10.4 shows when the last sync ran, so the report must carry
    /// the time that the mark of §3.3 left.
    #[test]
    fn a_folder_carries_the_time_of_the_last_sync() {
        let shelf = Shelf::new();
        shelf.mark("work", "INBOX", SYNCED);

        let report = shelf.report(&config_of(&["work"]));

        assert_eq!(report.accounts[0].folders[0].synced, SYNCED);
    }

    #[test]
    fn a_folder_that_no_sync_marked_carries_no_time() {
        let shelf = Shelf::new();
        shelf
            .store
            .mark("work", "INBOX", &SyncState::default())
            .expect("a mark");

        let report = shelf.report(&config_of(&["work"]));

        assert_eq!(report.accounts[0].folders[0].synced, 0);
    }

    /// A sync marks each folder at a different moment. The reader wants
    /// the newest of them, because that says how fresh the store is.
    #[test]
    fn the_report_gives_the_newest_time_of_any_folder() {
        let shelf = Shelf::new();
        shelf.mark("work", "Archive", SYNCED);
        shelf.mark("work", "INBOX", SYNCED + 600);
        shelf.mark("work", "Sent", SYNCED - 600);

        let report = shelf.report(&config_of(&["work"]));

        assert_eq!(report.synced(), Some(SYNCED + 600));
    }

    #[test]
    fn a_report_that_no_sync_touched_has_no_time() {
        let shelf = Shelf::new();

        assert_eq!(shelf.report(&config_of(&["work"])).synced(), None);
    }

    /// A folder of a mailbert that is older carries a zero. A zero is
    /// not a time, so it must never stand for one.
    #[test]
    fn a_time_of_zero_is_no_time_at_all() {
        let shelf = Shelf::new();
        shelf
            .store
            .mark("work", "INBOX", &SyncState::default())
            .expect("a mark");

        assert_eq!(shelf.report(&config_of(&["work"])).synced(), None);
    }

    // -----------------------------------------------------------------
    // The output. (§10.1, §10.4)
    // -----------------------------------------------------------------

    #[test]
    fn the_text_shows_the_counts() {
        let shelf = Shelf::new();
        shelf.put("a", true);

        let held = text_of(&shelf.report(&Config::default()));

        assert!(held.contains("messages"), "{held}");
        assert!(held.contains('1'), "{held}");
    }

    #[test]
    fn the_text_says_when_the_index_is_behind() {
        let shelf = Shelf::new();
        shelf.put("a", false);

        let held = text_of(&shelf.report(&Config::default()));

        assert!(held.contains("behind"), "{held}");
    }

    #[test]
    fn the_text_of_an_index_that_agrees_says_nothing_of_behind() {
        let shelf = Shelf::new();
        shelf.put("a", true);

        let held = text_of(&shelf.report(&Config::default()));

        assert!(!held.contains("behind"), "{held}");
    }

    #[test]
    fn the_text_names_each_folder_of_each_account() {
        let shelf = Shelf::new();
        shelf
            .store
            .mark(
                "work",
                "Archive",
                &SyncState {
                    uid_validity: 17,
                    uid_next: 9,
                    highest_mod_seq: 0,
                    pending: String::new(),
                    synced_at: SYNCED,
                },
            )
            .expect("a mark");

        let held = text_of(&shelf.report(&config_of(&["work"])));

        assert!(held.contains("work"), "{held}");
        assert!(held.contains("Archive"), "{held}");
        assert!(held.contains("uidnext 9"), "{held}");
        // §3.3: a UIDVALIDITY that changed makes every UID worthless,
        // so the reader must see the number.
        assert!(held.contains("uidvalidity 17"), "{held}");
    }

    /// §10.4 keeps the JSON stable, so the field names count.
    #[test]
    fn the_json_names_the_counts() {
        let shelf = Shelf::new();
        shelf.put("a", true);
        let mut out = Vec::new();

        write_json(&shelf.report(&config_of(&["work"])), &mut out)
            .expect("a write");

        let held: serde_json::Value =
            serde_json::from_slice(&out).expect("the output is JSON");
        assert_eq!(held["messages"], 1);
        assert_eq!(held["indexed"], 1);
        assert_eq!(held["embedded"], 0);
        assert_eq!(held["accounts"][0]["account"], "work");
    }

    /// §10.4 must say how old the mail is. A reader who cannot see
    /// the time does not know if a search can find the mail of today.
    #[test]
    fn the_text_says_when_the_last_sync_ran() {
        let shelf = Shelf::new();
        shelf.mark("work", "Archive", SYNCED);
        shelf.mark("work", "INBOX", SYNCED + 86_400);

        let held = text_of(&shelf.report(&config_of(&["work"])));

        // The newest of the folders, and not the first of them.
        assert!(sync_line(&held).contains("2025-08-23 00:00"), "{held}");
    }

    #[test]
    fn the_text_of_a_store_that_no_sync_touched_says_never() {
        let shelf = Shelf::new();

        let held = text_of(&shelf.report(&config_of(&["work"])));

        assert!(sync_line(&held).contains("never"), "{held}");
    }

    /// The reader gives the time zone. A time in UTC misleads a reader
    /// who is not in UTC by as much as 14 hours.
    #[test]
    fn the_text_shows_the_time_in_the_zone_of_the_reader() {
        let shelf = Shelf::new();
        shelf.mark("work", "INBOX", SYNCED);
        let report = shelf.report(&config_of(&["work"]));
        let mut out = Vec::new();

        write_text(&report, Clock::new(SYNCED, 7_200), &mut out)
            .expect("a write");

        let held = String::from_utf8(out).expect("the output is text");
        assert!(held.contains("2025-08-22 02:00 +0200"), "{held}");
    }

    #[test]
    fn the_text_shows_the_time_of_each_folder() {
        let shelf = Shelf::new();
        shelf.mark("work", "Archive", SYNCED);
        shelf.mark("work", "INBOX", SYNCED + 86_400);

        let held = text_of(&shelf.report(&config_of(&["work"])));
        let line = held
            .lines()
            .find(|line| line.contains("Archive"))
            .expect("a line for the folder");

        assert!(line.contains("2025-08-22 00:00"), "{held}");
    }

    #[test]
    fn the_text_of_a_folder_that_no_sync_marked_says_never() {
        let shelf = Shelf::new();
        shelf
            .store
            .mark("work", "INBOX", &SyncState::default())
            .expect("a mark");

        let held = text_of(&shelf.report(&config_of(&["work"])));
        let line = held
            .lines()
            .find(|line| line.contains("INBOX"))
            .expect("a line for the folder");

        assert!(line.contains("never"), "{held}");
    }

    /// §10.4 keeps the JSON stable, so the field name counts.
    #[test]
    fn the_json_carries_the_time_of_each_folder() {
        let shelf = Shelf::new();
        shelf.mark("work", "INBOX", SYNCED);

        let held = json_of(&shelf.report(&config_of(&["work"])));

        assert_eq!(held["accounts"][0]["folders"][0]["synced"], SYNCED);
    }

    // -----------------------------------------------------------------
    // Properties of the report.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 20)]
    fn prop_the_report_counts_what_the_store_holds(tc: TestCase) {
        let count = tc.draw(gs::integers::<usize>().min_value(0).max_value(6));
        let shelf = Shelf::new();
        let mut wanted = 0;

        for at in 0..count {
            let indexed = tc.draw(gs::booleans());
            shelf.put(&format!("m{at}"), indexed);

            if indexed {
                wanted += 1;
            }
        }

        let report = shelf.report(&Config::default());

        assert_eq!(report.messages, count);
        assert_eq!(report.indexed, wanted);
        assert_eq!(report.behind(), count - wanted);
    }

    #[hegel::test(test_cases = 30)]
    fn prop_the_report_gives_the_newest_time_of_any_folder(tc: TestCase) {
        let times: Vec<i64> = tc.draw(
            gs::vecs(
                gs::integers::<i64>().min_value(0).max_value(4_102_444_800),
            )
            .min_size(0)
            .max_size(5),
        );
        let shelf = Shelf::new();

        for (at, time) in times.iter().enumerate() {
            shelf.mark("work", &format!("F{at}"), *time);
        }

        let report = shelf.report(&config_of(&["work"]));
        let wanted = times.iter().copied().filter(|time| *time > 0).max();

        assert_eq!(report.synced(), wanted);
    }

    #[hegel::test(test_cases = 40)]
    fn prop_behind_is_never_below_zero(tc: TestCase) {
        let messages =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(100));
        let indexed =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(100));
        let report = Report {
            messages,
            indexed,
            embedded: 0,
            searches: 0,
            tags: BTreeMap::new(),
            accounts: Vec::new(),
        };

        match messages > indexed {
            true => assert_eq!(report.behind(), messages - indexed),
            false => assert_eq!(report.behind(), 0),
        }
    }
}
