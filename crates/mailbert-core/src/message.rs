//! One message, and every place that a copy of it sits. See §4.2.
//!
//! A Gmail message is in `INBOX`, in `[Gmail]/All Mail`, and in each
//! label folder. The same message sent to a work address and to a
//! personal one arrives on two accounts. mailbert holds one entry for
//! it, with a location for each copy.
//!
//! The entry costs one embedding and gives one result. A search for
//! `folder:INBOX` still finds it, because one of its locations names
//! `INBOX`.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    Address,
    message_id::MessageId,
    mime::{Attachment, Parsed, Source},
    query::Flag,
    threading::ThreadInput,
};

/// The IMAP flag of a message that the reader opened.
pub const SEEN: &str = r"\seen";

/// The IMAP flag of a message that the reader answered.
pub const ANSWERED: &str = r"\answered";

/// The IMAP flag of a message that the reader marked.
pub const FLAGGED: &str = r"\flagged";

/// The IMAP flag of a message that the server will expunge.
pub const DELETED: &str = r"\deleted";

/// The IMAP flag of a message that the reader did not send.
pub const DRAFT: &str = r"\draft";

/// The flag that lasts one session, and that mailbert drops.
const RECENT: &str = r"\recent";

/// One copy of a message on one server.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct Location {
    /// The account name from the configuration file.
    pub account: String,

    /// The folder, as the server names it.
    pub folder: String,

    /// The UID of the copy inside that folder.
    pub uid: u32,

    /// The UIDVALIDITY of the folder when sync read the UID.
    pub uid_validity: u32,

    /// The INTERNALDATE of the copy, in seconds since the epoch.
    pub received: i64,

    /// The IMAP flags that this folder gives the copy, normalized.
    ///
    /// The flags sit on the copy, and not on the message, because the
    /// server sets them per folder. A message that loses a copy loses
    /// the flags of that copy with it.
    pub flags: BTreeSet<String>,
}

/// One message, whatever number of copies the servers hold.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct Message {
    /// The identity from §4.1.
    pub id: MessageId,

    /// The `Message-ID`, normalized.
    pub message_id: Option<String>,

    /// The `In-Reply-To`, normalized.
    pub in_reply_to: Option<String>,

    /// The `References`, normalized.
    pub references: Vec<String>,

    /// The date to sort by, in seconds since the epoch.
    pub date: i64,

    /// The `From` addresses.
    pub from: Vec<Address>,

    /// The `To` addresses.
    pub to: Vec<Address>,

    /// The `Cc` addresses.
    pub cc: Vec<Address>,

    /// The subject, as the reader sees it.
    pub subject: String,

    /// The `List-Id`, reduced to its identifier.
    pub list_id: Option<String>,

    /// Which part gave the text.
    pub source: Source,

    /// The text to index, after §5.2 removes the quotes.
    pub text: String,

    /// True when the message is mail to a list, or is automatic.
    pub is_bulk: bool,

    /// One entry for each attachment.
    pub attachments: Vec<Attachment>,

    /// Every place that a copy sits, ordered and without repeats.
    pub locations: Vec<Location>,

    /// The IMAP flags of every copy, joined.
    ///
    /// This is what the copies say, and nothing else. Use
    /// [`Message::add_flag`] and never write the set itself.
    pub flags: BTreeSet<String>,
}

/// Normalize one IMAP flag.
///
/// Comparison is not case-sensitive, so the set holds one spelling.
/// `\Recent` lasts one session only, so it never enters the set.
pub fn normalize_flag(raw: &str) -> Option<String> {
    let found = raw.trim().to_lowercase();

    if found.is_empty() || found == RECENT {
        return None;
    }

    Some(found)
}

/// Join the copies of each message into one entry. See §4.2.
///
/// Sync reads folders in parallel and finds the same message several
/// times. This is the step that makes those findings one entry.
pub fn collate(messages: Vec<Message>) -> Vec<Message> {
    let mut entries: BTreeMap<MessageId, Message> = BTreeMap::new();

    for found in messages {
        match entries.get_mut(&found.id) {
            Some(entry) => entry.absorb(found),
            None => {
                entries.insert(found.id, found);
            }
        }
    }

    entries.into_values().collect()
}

/// Which of two readings of one folder is the later one.
///
/// A folder that reset its UIDVALIDITY gave every message a new UID,
/// so the validity decides before the UID does. The key is a total
/// order, which is what makes a merge give one answer whatever order
/// the copies arrive in.
fn freshness(location: &Location) -> (u32, u32, i64) {
    (location.uid_validity, location.uid, location.received)
}

impl Message {
    /// Build the entry for one copy of a message.
    pub fn new(
        parsed: Parsed,
        location: Location,
        flags: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Self {
        let id = parsed.identity();

        let carried: BTreeSet<String> = flags
            .into_iter()
            .filter_map(|raw| normalize_flag(raw.as_ref()))
            .collect();
        let mut location = location;
        location.flags.extend(carried.iter().cloned());

        // A message with no `Date` header is dated by the server, which
        // is the only other time that mailbert knows.
        let date = parsed.date.unwrap_or(location.received);

        Self {
            id,
            message_id: parsed.message_id,
            in_reply_to: parsed.in_reply_to,
            references: parsed.references,
            date,
            from: parsed.from,
            to: parsed.to,
            cc: parsed.cc,
            subject: parsed.subject,
            list_id: parsed.list_id,
            source: parsed.source,
            text: parsed.text,
            is_bulk: parsed.is_bulk,
            attachments: parsed.attachments,
            flags: location.flags.clone(),
            locations: vec![location],
        }
    }

    /// Record another place that a copy of this message sits.
    ///
    /// A folder holds one copy, so a second reading of the same folder
    /// replaces the location instead of adding one.
    pub fn add_location(&mut self, location: Location) {
        let same = self.locations.iter().position(|at| {
            at.account == location.account && at.folder == location.folder
        });

        match same {
            // The reading that arrives last speaks for that folder. A
            // FETCH carries every flag of a copy, so a flag that the
            // answer does not hold is a flag that went away. (§3.3)
            Some(at)
                if freshness(&location) >= freshness(&self.locations[at]) =>
            {
                self.locations[at] = location;
            }
            // An older copy of a folder that already has a newer one.
            // It says nothing, because its UID is gone.
            Some(_) => {}
            None => self.locations.push(location),
        }

        self.locations.sort();
        self.rejoin();
    }

    /// Record one IMAP flag on every copy.
    ///
    /// Returns true when the message did not have the flag before.
    pub fn add_flag(&mut self, raw: &str) -> bool {
        let Some(flag) = normalize_flag(raw) else {
            return false;
        };

        let before = self.flags.contains(&flag);
        for at in &mut self.locations {
            at.flags.insert(flag.clone());
        }

        self.rejoin();

        // A message with no copy left gains nothing, because the flags
        // of a message are the flags of its copies.
        !before && self.flags.contains(&flag)
    }

    /// Give one copy the flags that the server now reports (§3.3).
    ///
    /// Returns true when the message has a copy in that folder. This is
    /// the one way that a flag goes away, because a folder that drops
    /// `\Seen` makes the message unread again.
    pub fn set_flags(
        &mut self,
        account: &str,
        folder: &str,
        flags: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> bool {
        let carried: BTreeSet<String> = flags
            .into_iter()
            .filter_map(|raw| normalize_flag(raw.as_ref()))
            .collect();

        let Some(at) = self
            .locations
            .iter_mut()
            .find(|at| at.account == account && at.folder == folder)
        else {
            return false;
        };

        at.flags = carried;
        self.rejoin();

        true
    }

    /// Read the flags of the copies into the set of the message.
    fn rejoin(&mut self) {
        self.flags = self
            .locations
            .iter()
            .flat_map(|at| at.flags.iter().cloned())
            .collect();
    }

    /// Forget one place, because the copy there is gone.
    ///
    /// Returns true when the message had a copy there.
    pub fn remove_location(&mut self, account: &str, folder: &str) -> bool {
        let before = self.locations.len();

        self.locations
            .retain(|at| at.account != account || at.folder != folder);
        self.rejoin();

        self.locations.len() != before
    }

    /// Join another reading of the same message into this one.
    ///
    /// The two must be the same message. `collate` is the way that a
    /// caller gets that guarantee.
    pub(crate) fn absorb(&mut self, other: Message) {
        // The minimum is commutative, so two copies of one message date
        // it the same however the parallel sync finds them.
        self.date = self.date.min(other.date);

        for location in other.locations {
            self.add_location(location);
        }
    }

    /// The accounts that hold a copy, ordered and without repeats.
    pub fn accounts(&self) -> Vec<&str> {
        // The locations are sorted by account, so neighbours repeat.
        let mut found: Vec<&str> = self
            .locations
            .iter()
            .map(|at| at.account.as_str())
            .collect();

        found.dedup();
        found
    }

    /// The folders that hold a copy, ordered and without repeats.
    pub fn folders(&self) -> Vec<&str> {
        let mut found: Vec<&str> =
            self.locations.iter().map(|at| at.folder.as_str()).collect();

        found.sort_unstable();
        found.dedup();
        found
    }

    /// The addresses of the sender and of every recipient.
    pub fn participants(&self) -> Vec<String> {
        let mut found: Vec<String> = self
            .from
            .iter()
            .chain(self.to.iter())
            .chain(self.cc.iter())
            .map(|address| address.address.clone())
            .collect();

        found.sort();
        found.dedup();
        found
    }

    /// Whether the servers no longer hold a copy.
    ///
    /// The entry stays, because the mirror is the archive. `is:gone`
    /// is how the reader finds what the server lost.
    pub fn is_gone(&self) -> bool {
        self.locations.is_empty()
    }

    /// Whether the message is encrypted.
    pub fn is_encrypted(&self) -> bool {
        self.source == Source::Encrypted
    }

    /// Whether the message is in the state that `is:` names.
    pub fn matches(&self, flag: Flag) -> bool {
        match flag {
            Flag::Read => self.flags.contains(SEEN),
            Flag::Unread => !self.flags.contains(SEEN),
            Flag::Flagged => self.flags.contains(FLAGGED),
            Flag::Replied => self.flags.contains(ANSWERED),
            Flag::Draft => self.flags.contains(DRAFT),
            Flag::Encrypted => self.is_encrypted(),
            Flag::Gone => self.is_gone(),
            Flag::Bulk => self.is_bulk,
        }
    }

    /// What the threading of §5.5 reads.
    pub fn thread_input(&self) -> ThreadInput {
        let mut input = ThreadInput::new(self.id, &self.subject, self.date)
            .with_references(self.references.clone())
            .with_participants(self.participants());

        if let Some(message_id) = &self.message_id {
            input = input.with_message_id(message_id.clone());
        }

        if let Some(in_reply_to) = &self.in_reply_to {
            input = input.with_in_reply_to(in_reply_to.clone());
        }

        input
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_collate_ignores_the_order` | differential | Sync reads folders in parallel. Two runs over one mailbox must give the same entries. |
    //! | `prop_collate_is_idempotent` | algebraic | A re-sync runs collate over entries that collate already joined. |
    //! | `prop_collate_keeps_every_location` | invariant | A lost location is a folder that `folder:` can no longer find. |
    //! | `prop_one_location_for_each_folder` | invariant | A folder holds one copy. Two locations for it would count the message twice. |
    //! | `prop_collate_keeps_the_identity` | invariant | The identity is the key of the store, and of every tag. |
    //! | `prop_flags_ignore_case` | invariant | Servers spell `\Seen` and `\SEEN`, and both mean read. |
    //! | `prop_read_and_unread_are_opposites` | model-based | `is:read` and `is:unread` must partition the mailbox, or a search loses messages. |
    //! | `prop_the_date_is_the_earliest_sighting` | algebraic | Merging must not move a message in time, whatever order the copies arrive in. |
    //! | `prop_the_flags_of_a_message_join_the_flags_of_its_copies` | invariant | `is:unread` reads the joined set. A flag that outlives its copy hides mail that the user has not read. |

    use std::collections::HashMap;

    use hegel::{TestCase, generators as gs};

    use super::*;
    use crate::mime;

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    const DAY: i64 = 86_400;

    fn location(account: &str, folder: &str, uid: u32) -> Location {
        Location {
            account: account.to_string(),
            folder: folder.to_string(),
            uid,
            uid_validity: 1,
            received: 100 * DAY,
            flags: BTreeSet::new(),
        }
    }

    /// A message that carries the `Message-ID` `<{key}@x.test>`.
    fn parsed(key: &str, body: &str) -> Parsed {
        let text = format!(
            "From: Alice Smith <alice@example.test>\r\n\
             To: bob@example.test\r\n\
             Subject: Deposit\r\n\
             Date: Fri, 14 Aug 2026 09:30:00 +0000\r\n\
             Message-ID: <{key}@x.test>\r\n\
             \r\n\
             {body}\r\n"
        );

        mime::parse(text.as_bytes()).expect("a message that parses")
    }

    fn message(key: &str, account: &str, folder: &str) -> Message {
        Message::new(
            parsed(key, "The deposit is due."),
            location(account, folder, 1),
            [SEEN],
        )
    }

    fn folder_set(found: &Message) -> BTreeSet<String> {
        found.folders().into_iter().map(str::to_string).collect()
    }

    // -----------------------------------------------------------------
    // Flags.
    // -----------------------------------------------------------------

    #[test]
    fn normalizes_the_case_of_a_system_flag() {
        assert_eq!(normalize_flag(r"\Seen").as_deref(), Some(SEEN));
        assert_eq!(normalize_flag(r"\SEEN").as_deref(), Some(SEEN));
        assert_eq!(normalize_flag(r"  \seen  ").as_deref(), Some(SEEN));
    }

    #[test]
    fn keeps_a_keyword_that_the_server_defines() {
        assert_eq!(normalize_flag("$Label1").as_deref(), Some("$label1"));
        assert_eq!(normalize_flag("NonJunk").as_deref(), Some("nonjunk"));
    }

    #[test]
    fn drops_the_recent_flag_and_the_empty_one() {
        assert_eq!(normalize_flag(r"\Recent"), None);
        assert_eq!(normalize_flag(""), None);
        assert_eq!(normalize_flag("   "), None);
    }

    #[test]
    fn adds_a_flag_one_time() {
        let mut found = message("a", "work", "INBOX");

        assert!(!found.add_flag(r"\SEEN"));
        assert!(found.add_flag(r"\Flagged"));
        assert!(!found.add_flag(FLAGGED));
        assert_eq!(found.flags.len(), 2);
    }

    // -----------------------------------------------------------------
    // Unit tests: the flags of one copy (§4.2).
    // -----------------------------------------------------------------

    /// A message with one copy, and the flags of that copy.
    fn flagged(folder: &str, flags: &[&str]) -> Message {
        Message::new(
            parsed("a", "The deposit is due."),
            location("work", folder, 1),
            flags.iter().copied(),
        )
    }

    #[test]
    fn each_copy_keeps_the_flags_that_its_folder_gave() {
        let mut found = flagged("INBOX", &[SEEN]);
        found.add_location(Location {
            flags: [FLAGGED.to_string()].into_iter().collect(),
            ..location("work", "Archive", 9)
        });

        let inbox = &found.locations[1];
        assert_eq!(inbox.folder, "INBOX");
        assert_eq!(inbox.flags, [SEEN.to_string()].into_iter().collect());
    }

    #[test]
    fn the_flags_of_a_message_join_its_copies() {
        let mut found = flagged("INBOX", &[SEEN]);
        found.add_location(Location {
            flags: [FLAGGED.to_string()].into_iter().collect(),
            ..location("work", "Archive", 9)
        });

        assert!(found.matches(Flag::Read));
        assert!(found.matches(Flag::Flagged));
    }

    #[test]
    fn a_copy_that_goes_away_takes_its_flags_with_it() {
        let mut found = flagged("INBOX", &[SEEN]);
        found.add_location(Location {
            flags: [FLAGGED.to_string()].into_iter().collect(),
            ..location("work", "Archive", 9)
        });

        assert!(found.remove_location("work", "Archive"));

        assert!(found.matches(Flag::Read));
        assert!(!found.matches(Flag::Flagged), "a flag outlived its copy");
    }

    #[test]
    fn a_newer_copy_of_one_folder_gives_the_flags_of_that_folder() {
        let mut found = flagged("INBOX", &[SEEN]);
        found.add_location(Location {
            uid: 42,
            flags: BTreeSet::new(),
            ..location("work", "INBOX", 1)
        });

        assert!(found.matches(Flag::Unread), "the old reading held on");
        assert_eq!(found.locations.len(), 1);
    }

    #[test]
    fn a_second_reading_of_one_uid_gives_the_flags_that_it_carries() {
        let mut found = flagged("INBOX", &[SEEN]);
        found.add_location(Location {
            flags: [FLAGGED.to_string()].into_iter().collect(),
            ..location("work", "INBOX", 1)
        });

        assert!(found.matches(Flag::Flagged));
        assert!(!found.matches(Flag::Read), "the server dropped that flag");
    }

    /// §3.3: an unread message that a sync reads again stays unread.
    #[test]
    fn a_second_reading_that_carries_no_flag_takes_the_flags_away() {
        let mut found = flagged("INBOX", &[SEEN]);
        found.add_location(location("work", "INBOX", 1));

        assert!(found.flags.is_empty(), "{:?}", found.flags);
    }

    #[test]
    fn an_older_copy_of_one_folder_never_speaks_for_it() {
        let mut found = flagged("INBOX", &[SEEN]);
        found.add_location(Location {
            flags: [FLAGGED.to_string()].into_iter().collect(),
            ..location("work", "INBOX", 0)
        });

        assert!(found.matches(Flag::Read));
        assert!(!found.matches(Flag::Flagged));
    }

    #[hegel::test(test_cases = 40)]
    fn prop_the_flags_of_a_message_join_the_flags_of_its_copies(tc: TestCase) {
        let names: Vec<String> = tc.draw(
            gs::vecs(gs::text().alphabet("AB").min_size(1).max_size(2))
                .min_size(1)
                .max_size(4),
        );

        let mut folders: Vec<String> = names;
        folders.sort();
        folders.dedup();

        let mut found = Message::new(
            parsed("a", "The deposit is due."),
            location("work", &folders[0], 1),
            Vec::<String>::new(),
        );
        let mut want: BTreeSet<String> = BTreeSet::new();

        for (step, folder) in folders.iter().enumerate() {
            let carried: Vec<String> = tc.draw(
                gs::vecs(gs::sampled_from(vec![
                    SEEN.to_string(),
                    FLAGGED.to_string(),
                    DRAFT.to_string(),
                ]))
                .min_size(0)
                .max_size(3),
            );

            want.extend(carried.iter().cloned());
            found.add_location(Location {
                flags: carried.into_iter().collect(),
                ..location("work", folder, step as u32 + 1)
            });
        }

        assert_eq!(found.flags, want, "the message and its copies disagree");
    }

    // -----------------------------------------------------------------
    // The `is:` states.
    // -----------------------------------------------------------------

    #[test]
    fn a_message_with_the_seen_flag_is_read() {
        let found = message("a", "work", "INBOX");

        assert!(found.matches(Flag::Read));
        assert!(!found.matches(Flag::Unread));
    }

    #[test]
    fn a_message_without_the_seen_flag_is_unread() {
        let found = Message::new(
            parsed("a", "Body."),
            location("work", "INBOX", 1),
            Vec::<String>::new(),
        );

        assert!(found.matches(Flag::Unread));
        assert!(!found.matches(Flag::Read));
    }

    #[test]
    fn the_other_states_read_their_flags() {
        let mut found = message("a", "work", "INBOX");
        found.add_flag(r"\Answered");
        found.add_flag(r"\Flagged");
        found.add_flag(r"\Draft");

        assert!(found.matches(Flag::Replied));
        assert!(found.matches(Flag::Flagged));
        assert!(found.matches(Flag::Draft));
    }

    #[test]
    fn an_entry_with_no_location_is_gone() {
        let mut found = message("a", "work", "INBOX");

        assert!(!found.matches(Flag::Gone));
        assert!(found.remove_location("work", "INBOX"));
        assert!(found.is_gone());
        assert!(found.matches(Flag::Gone));
        assert!(!found.remove_location("work", "INBOX"));
    }

    #[test]
    fn the_encrypted_state_comes_from_the_body() {
        let raw = "Subject: Private\r\n\
                   Content-Type: application/pkcs7-mime\r\n\
                   \r\n\
                   MIAGCSqGSIb3DQEHA6CA\r\n";
        let found = Message::new(
            mime::parse(raw.as_bytes()).expect("a message"),
            location("work", "INBOX", 1),
            [SEEN],
        );

        assert!(found.matches(Flag::Encrypted));
        assert!(found.is_encrypted());
    }

    #[test]
    fn the_bulk_state_comes_from_the_headers() {
        let raw = "Subject: A thread\r\n\
                   List-Id: Rust Users <users.rust-lang.org>\r\n\
                   \r\n\
                   Body.\r\n";
        let found = Message::new(
            mime::parse(raw.as_bytes()).expect("a message"),
            location("work", "INBOX", 1),
            [SEEN],
        );

        assert!(found.matches(Flag::Bulk));
    }

    // -----------------------------------------------------------------
    // One message, many locations. See §4.2.
    // -----------------------------------------------------------------

    #[test]
    fn one_gmail_message_in_four_folders_is_one_entry() {
        let copies = vec![
            message("a", "gmail", "INBOX"),
            message("a", "gmail", "[Gmail]/All Mail"),
            message("a", "gmail", "work"),
            message("a", "gmail", "receipts"),
        ];

        let entries = collate(copies);

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].locations.len(), 4);
        assert_eq!(entries[0].folders().len(), 4);
        assert_eq!(entries[0].accounts(), vec!["gmail"]);
    }

    #[test]
    fn one_message_on_two_accounts_is_one_entry() {
        let entries = collate(vec![
            message("a", "work", "INBOX"),
            message("a", "personal", "INBOX"),
        ]);

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].accounts(), vec!["personal", "work"]);
        assert_eq!(entries[0].folders(), vec!["INBOX"]);
        assert_eq!(entries[0].locations.len(), 2);
    }

    #[test]
    fn two_messages_stay_two_entries() {
        let entries = collate(vec![
            message("a", "work", "INBOX"),
            message("b", "work", "INBOX"),
        ]);

        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn reading_a_folder_again_replaces_the_location() {
        let mut found = message("a", "work", "INBOX");
        found.add_location(Location {
            uid: 42,
            ..location("work", "INBOX", 1)
        });

        assert_eq!(found.locations.len(), 1);
        assert_eq!(found.locations[0].uid, 42);
    }

    #[test]
    fn collate_joins_the_flags_of_every_copy() {
        let mut one = message("a", "gmail", "INBOX");
        one.flags.clear();
        one.add_flag(r"\Flagged");

        let two = message("a", "gmail", "[Gmail]/All Mail");

        let entries = collate(vec![one, two]);

        assert_eq!(entries.len(), 1);
        assert!(entries[0].matches(Flag::Read));
        assert!(entries[0].matches(Flag::Flagged));
    }

    #[test]
    fn the_locations_come_out_ordered() {
        let entries = collate(vec![
            message("a", "work", "Sent"),
            message("a", "work", "Archive"),
            message("a", "personal", "INBOX"),
        ]);

        let order: Vec<(&str, &str)> = entries[0]
            .locations
            .iter()
            .map(|found| (found.account.as_str(), found.folder.as_str()))
            .collect();

        assert_eq!(
            order,
            vec![("personal", "INBOX"), ("work", "Archive"), ("work", "Sent"),]
        );
    }

    // -----------------------------------------------------------------
    // The bridges to the other modules.
    // -----------------------------------------------------------------

    #[test]
    fn the_participants_hold_the_sender_and_the_recipients() {
        let found = message("a", "work", "INBOX");

        assert_eq!(
            found.participants(),
            vec!["alice@example.test", "bob@example.test"]
        );
    }

    #[test]
    fn the_thread_input_carries_the_identifiers() {
        let raw = "Subject: Re: Deposit\r\n\
                   From: alice@example.test\r\n\
                   Message-ID: <c@x.test>\r\n\
                   In-Reply-To: <b@x.test>\r\n\
                   References: <a@x.test> <b@x.test>\r\n\
                   \r\n\
                   Body.\r\n";
        let found = Message::new(
            mime::parse(raw.as_bytes()).expect("a message"),
            location("work", "INBOX", 1),
            [SEEN],
        );

        let input = found.thread_input();

        assert_eq!(input.id, found.id);
        assert_eq!(input.message_id.as_deref(), Some("c@x.test"));
        assert_eq!(input.in_reply_to.as_deref(), Some("b@x.test"));
        assert_eq!(input.references, vec!["a@x.test", "b@x.test"]);
        assert_eq!(input.subject, "Re: Deposit");
        assert_eq!(input.date, found.date);
    }

    #[test]
    fn a_message_with_no_date_falls_back_to_the_internal_date() {
        let found = Message::new(
            mime::parse(b"Subject: No date\r\n\r\nBody.\r\n")
                .expect("a message"),
            location("work", "INBOX", 1),
            [SEEN],
        );

        assert_eq!(found.date, 100 * DAY);
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    fn a_key() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ])
    }

    fn an_account() -> impl gs::Generator<String> {
        gs::sampled_from(vec!["work".to_string(), "personal".to_string()])
    }

    fn a_folder() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "INBOX".to_string(),
            "Archive".to_string(),
            "[Gmail]/All Mail".to_string(),
            "Sent".to_string(),
        ])
    }

    fn a_flag() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            r"\Seen".to_string(),
            r"\SEEN".to_string(),
            r"\Flagged".to_string(),
            r"\Answered".to_string(),
            r"\Recent".to_string(),
            "$label1".to_string(),
        ])
    }

    /// A message that carries no `Date` header, so that the fallback to
    /// the server date is under test too.
    fn undated(key: &str) -> Parsed {
        let text = format!(
            "From: alice@example.test\r\n\
             Subject: No date\r\n\
             Message-ID: <undated-{key}@x.test>\r\n\
             \r\n\
             Body.\r\n"
        );

        mime::parse(text.as_bytes()).expect("a message that parses")
    }

    #[hegel::composite]
    fn a_copy(tc: TestCase) -> Message {
        let key = tc.draw(a_key());
        let account = tc.draw(an_account());
        let folder = tc.draw(a_folder());
        let uid = tc.draw(gs::integers::<u32>().min_value(1).max_value(50));
        let day = tc.draw(gs::integers::<i64>().min_value(1).max_value(500));
        let flags = tc.draw(gs::vecs(a_flag()).min_size(0).max_size(3));
        let dated = tc.draw(gs::booleans());

        let body = match dated {
            true => parsed(&key, "The deposit is due."),
            false => undated(&key),
        };

        Message::new(
            body,
            Location {
                account,
                folder,
                uid,
                uid_validity: 1,
                received: day * DAY,
                flags: BTreeSet::new(),
            },
            flags,
        )
    }

    fn a_mailbox() -> impl gs::Generator<Vec<Message>> {
        gs::vecs(a_copy()).min_size(0).max_size(12)
    }

    /// The parts of an entry that must not depend on the read order.
    fn shape(entries: &[Message]) -> Vec<(MessageId, BTreeSet<String>, i64)> {
        entries
            .iter()
            .map(|found| {
                let places: BTreeSet<String> = found
                    .locations
                    .iter()
                    .map(|at| format!("{}/{}", at.account, at.folder))
                    .collect();

                (found.id, places, found.date)
            })
            .collect()
    }

    #[hegel::test(test_cases = 200)]
    fn prop_collate_ignores_the_order(tc: TestCase) {
        let copies: Vec<Message> = tc.draw(a_mailbox());
        let reversed: Vec<Message> = copies.iter().rev().cloned().collect();

        let one = shape(&collate(copies));
        let two = shape(&collate(reversed));

        assert_eq!(one, two);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_collate_is_idempotent(tc: TestCase) {
        let copies: Vec<Message> = tc.draw(a_mailbox());

        let once = collate(copies);
        let twice = collate(once.clone());

        assert_eq!(once, twice);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_collate_keeps_every_location(tc: TestCase) {
        let copies: Vec<Message> = tc.draw(a_mailbox());

        let wanted: BTreeSet<(MessageId, String, String)> = copies
            .iter()
            .flat_map(|found| {
                found
                    .locations
                    .iter()
                    .map(|at| (found.id, at.account.clone(), at.folder.clone()))
            })
            .collect();

        let found: BTreeSet<(MessageId, String, String)> = collate(copies)
            .iter()
            .flat_map(|entry| {
                entry
                    .locations
                    .iter()
                    .map(|at| (entry.id, at.account.clone(), at.folder.clone()))
            })
            .collect();

        assert_eq!(found, wanted);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_one_location_for_each_folder(tc: TestCase) {
        let copies: Vec<Message> = tc.draw(a_mailbox());

        for entry in collate(copies) {
            let places: BTreeSet<(&str, &str)> = entry
                .locations
                .iter()
                .map(|at| (at.account.as_str(), at.folder.as_str()))
                .collect();

            assert_eq!(places.len(), entry.locations.len());
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_collate_keeps_the_identity(tc: TestCase) {
        let copies: Vec<Message> = tc.draw(a_mailbox());

        let wanted: BTreeSet<MessageId> =
            copies.iter().map(|found| found.id).collect();
        let found: BTreeSet<MessageId> =
            collate(copies).iter().map(|entry| entry.id).collect();

        assert_eq!(found, wanted);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_flags_ignore_case(tc: TestCase) {
        let raw = tc.draw(a_flag());

        let lower = normalize_flag(&raw.to_lowercase());
        let upper = normalize_flag(&raw.to_uppercase());

        assert_eq!(lower, upper);
        assert_eq!(normalize_flag(&raw), lower);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_read_and_unread_are_opposites(tc: TestCase) {
        let entry: Message = tc.draw(a_copy());

        assert_ne!(entry.matches(Flag::Read), entry.matches(Flag::Unread));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_date_is_the_earliest_sighting(tc: TestCase) {
        let copies: Vec<Message> = tc.draw(a_mailbox());

        let mut earliest: HashMap<MessageId, i64> = HashMap::new();
        for found in &copies {
            let at = earliest.entry(found.id).or_insert(found.date);
            *at = (*at).min(found.date);
        }

        for entry in collate(copies) {
            assert_eq!(Some(&entry.date), earliest.get(&entry.id));
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_folders_never_repeat(tc: TestCase) {
        let copies: Vec<Message> = tc.draw(a_mailbox());

        for entry in collate(copies) {
            let listed = entry.folders();
            let unique = folder_set(&entry);

            assert_eq!(listed.len(), unique.len());
        }
    }
}
