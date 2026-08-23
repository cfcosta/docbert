//! What `mailbert status` shows. (§10.4)
//!
//! The report gives the counts that tell you if the store, the index,
//! and the vectors agree. A store that holds more messages than the
//! index means that a sync stopped early, and the next sync fixes it.
//!
//! The report never speaks to the IMAP server. §3.3 makes mailbert a
//! download-only mirror, and `status` reads only what is on the disk.

use std::{collections::BTreeMap, io::Write};

use mailbert_core::{Store, config::Config, index::MailIndex};
use serde::Serialize;

use crate::{Tool, cli, error::Result};

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
/// # Errors
///
/// The function fails if the output does not take the text.
pub fn write_text(report: &Report, out: &mut dyn Write) -> Result<()> {
    writeln!(out, "{:LABEL$}{}", "messages", report.messages)?;
    writeln!(out, "{:LABEL$}{}", "indexed", report.indexed)?;
    writeln!(out, "{:LABEL$}{}", "embedded", report.embedded)?;
    writeln!(out, "{:LABEL$}{}", "searches", report.searches)?;

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
                "  {:FOLDER$}uidvalidity {}  uidnext {}{owed}",
                folder.name, folder.uid_validity, folder.uid_next
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
        false => write_text(&held, &mut out),
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

    use std::collections::BTreeSet;

    use hegel::{TestCase, generators as gs};
    use mailbert_core::{
        config::Account,
        message::{Location, Message},
        mime,
        store::{Embedded, SyncState},
        threading::ThreadId,
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

    fn account(name: &str) -> Account {
        Account {
            name: name.to_string(),
            host: "mail.example.test".to_string(),
            user: "me@cfcosta.com".to_string(),
            port: 993,
            password_command: None,
            password_file: None,
            password: Some("secret".to_string()),
            folders: Vec::new(),
            exclude: Vec::new(),
            footers: Vec::new(),
            all_folders: true,
            connections: 1,
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

        fn report(&self, config: &Config) -> Report {
            super::report(&self.store, &self.index, config).expect("a report")
        }
    }

    fn text_of(report: &Report) -> String {
        let mut out = Vec::new();
        write_text(report, &mut out).expect("a write");

        String::from_utf8(out).expect("the output is text")
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
