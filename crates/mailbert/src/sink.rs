//! The sink of a sync: what one batch leaves in the store. (§3.4, §4.2)
//!
//! The sink is the one place that writes mail. It parses the bytes that
//! the server sent, joins each copy into the entry that the store
//! holds, and marks the state of the folder after each batch. A sync
//! that stops loses only the batch that was in the air. (§3.4)
//!
//! The sink never decrypts. An encrypted body reaches the store as the
//! ciphertext that arrived, and `view` gives it to gpg on demand.

use std::{collections::BTreeSet, sync::Arc};

use mailbert_core::{
    Message,
    MessageId,
    Store,
    date::internal_date,
    message::Location,
    mime,
    store::SyncState,
};
use mailbert_imap::{
    Batch,
    Result,
    connection::Fetched,
    sequence::UidSet,
    sync::{FolderState, Keep},
};
use regex::Regex;

/// What one sync did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    /// The messages that arrived with a body.
    pub kept: u64,

    /// The copies whose flags moved, and nothing else. (§3.3)
    pub moved: u64,

    /// The copies that the server does not hold any more.
    pub gone: u64,

    /// The bodies that are not a message that mailbert can read.
    pub broken: u64,

    /// The raw bytes that arrived.
    pub bytes: u64,
}

impl Counts {
    /// Join what two sinks of one account did.
    ///
    /// A sync gives one sink to each folder, and the report of the
    /// account holds the sum of them all.
    pub fn and(self, other: Self) -> Self {
        Self {
            kept: self.kept + other.kept,
            moved: self.moved + other.moved,
            gone: self.gone + other.gone,
            broken: self.broken + other.broken,
            bytes: self.bytes + other.bytes,
        }
    }
}

/// The sink of one account.
///
/// The sink holds the store behind an [`Arc`], because a sync gives one
/// sink to each folder and runs the folders at the same time.
pub struct Sink {
    store: Arc<Store>,
    account: String,
    footers: Vec<Regex>,
    touched: BTreeSet<MessageId>,
    counts: Counts,
}

/// The record that the store keeps for one folder. (§3.3)
pub fn keep_state(state: &FolderState) -> SyncState {
    SyncState {
        uid_validity: state.uid_validity,
        uid_next: state.uid_next,
        highest_mod_seq: state.highest_mod_seq,
        pending: state.pending.to_string(),
    }
}

/// The state that a sync starts from, out of the record. (§3.4)
///
/// A `pending` set that does not parse gives an empty set, because a
/// sync that asks for nothing is better than a sync that stops.
pub fn sync_state(kept: &SyncState) -> FolderState {
    FolderState {
        uid_validity: kept.uid_validity,
        uid_next: kept.uid_next,
        highest_mod_seq: kept.highest_mod_seq,
        pending: UidSet::parse(&kept.pending).unwrap_or_default(),
    }
}

impl Sink {
    /// Make the sink of one account.
    pub fn new(store: Arc<Store>, account: &str) -> Self {
        Self {
            store,
            account: account.to_string(),
            footers: Vec::new(),
            touched: BTreeSet::new(),
            counts: Counts::default(),
        }
    }

    /// Remove these footers from every body that arrives. (§5.2)
    pub fn with_footers(mut self, footers: Vec<Regex>) -> Self {
        self.footers = footers;

        self
    }

    /// What this sink did.
    pub fn counts(&self) -> Counts {
        self.counts
    }

    /// The messages that this sink wrote, for the threading pass. (§5.5)
    pub fn touched(&self) -> &BTreeSet<MessageId> {
        &self.touched
    }

    /// Keep one copy of one message.
    fn keep_one(
        &mut self,
        folder: &str,
        uid_validity: u32,
        fetched: Fetched,
    ) -> Result<()> {
        // A fetch that asks only for the flags brings no body. The copy
        // is in the store, and only what the server says about it moved.
        if fetched.body.is_empty() {
            let moved = self.store.reflag(
                &self.account,
                folder,
                fetched.uid,
                &fetched.flags,
            )?;

            if let Some(id) = moved {
                self.touched.insert(id);
                self.counts.moved += 1;
            }

            return Ok(());
        }

        // A body that mailbert cannot read must not stop a sync of
        // 100000 messages. The sync counts it, and goes on.
        let Ok(parsed) = mime::parse_with_footers(&fetched.body, &self.footers)
        else {
            self.counts.broken += 1;

            return Ok(());
        };

        let received = fetched
            .internal_date
            .as_deref()
            .and_then(internal_date)
            .unwrap_or_default();

        let location = Location {
            account: self.account.clone(),
            folder: folder.to_string(),
            uid: fetched.uid,
            uid_validity,
            received,
            flags: BTreeSet::new(),
        };

        let message = Message::new(parsed, location, &fetched.flags);
        let kept = self.store.put(&message, &fetched.body)?;

        self.touched.insert(kept.id);
        self.counts.kept += 1;
        self.counts.bytes += fetched.body.len() as u64;

        Ok(())
    }
}

impl Keep for Sink {
    async fn mark(&mut self, folder: &str, state: &FolderState) -> Result<()> {
        self.store.mark(&self.account, folder, &keep_state(state))?;

        Ok(())
    }

    async fn batch(
        &mut self,
        folder: &str,
        batch: Batch,
        state: &FolderState,
    ) -> Result<()> {
        for fetched in batch.messages {
            self.keep_one(folder, state.uid_validity, fetched)?;
        }

        // mailbert is a mirror, so a copy that went away leaves the
        // message behind. A message with no copy answers `is:gone`.
        for uid in batch.gone {
            if let Some(id) = self.store.vanish(&self.account, folder, uid)? {
                self.touched.insert(id);
                self.counts.gone += 1;
            }
        }

        // The mark goes last, so the state on disk never runs ahead of
        // the mail that the store holds. (§3.4)
        self.mark(folder, state).await
    }

    async fn state(&mut self, folder: &str) -> Result<FolderState> {
        let kept = self.store.state(&self.account, folder)?;

        Ok(kept.as_ref().map(sync_state).unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_the_sink_keeps_every_message_of_a_batch` | invariant | A message that the sink drops is mail that the next sync never asks for again. |
    //! | `prop_a_state_survives_the_sink` | round-trip | §3.4 resumes from the state alone. A number that changes on the way to disk downloads a folder again. |
    //! | `prop_the_sink_never_stops_on_bytes_that_are_not_a_message` | invariant | One message that mailbert cannot read must not stop a sync of 100000. |

    use hegel::{TestCase, generators as gs};
    use mailbert_core::query::Flag;
    use mailbert_imap::UidSet;
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

    fn fetched(uid: u32, key: &str) -> Fetched {
        Fetched {
            uid,
            mod_seq: 10,
            size: 0,
            flags: vec![r"\Seen".to_string()],
            internal_date: Some("14-Aug-2026 09:30:00 +0000".to_string()),
            body: raw_bytes(key),
        }
    }

    fn batch_of(messages: Vec<Fetched>) -> Batch {
        Batch {
            messages,
            gone: Vec::new(),
        }
    }

    fn a_state(uid_next: u32) -> FolderState {
        FolderState {
            uid_validity: 3,
            uid_next,
            highest_mod_seq: 77,
            pending: UidSet::parse("4:9").expect("a set"),
        }
    }

    // -----------------------------------------------------------------
    // Unit tests: what a batch leaves behind.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn a_batch_writes_the_message_and_its_bytes() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        sink.batch("INBOX", batch_of(vec![fetched(7, "one")]), &a_state(8))
            .await
            .expect("a batch");

        let found = store.all().expect("a read");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].subject, "Deposit");
        assert_eq!(found[0].locations[0].uid, 7);
        assert_eq!(found[0].locations[0].uid_validity, 3);
        assert_eq!(
            store.raw(&found[0].id).expect("a read"),
            Some(raw_bytes("one"))
        );
    }

    #[tokio::test]
    async fn a_batch_dates_a_copy_by_the_internal_date() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        sink.batch("INBOX", batch_of(vec![fetched(7, "one")]), &a_state(8))
            .await
            .expect("a batch");

        let found = store.all().expect("a read");
        assert_eq!(found[0].locations[0].received, 1_786_699_800);
    }

    #[tokio::test]
    async fn a_batch_marks_the_state_of_the_folder() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        sink.batch("INBOX", batch_of(vec![fetched(7, "one")]), &a_state(8))
            .await
            .expect("a batch");

        let kept = store.state("work", "INBOX").expect("a read");
        assert_eq!(kept, Some(keep_state(&a_state(8))));
    }

    #[tokio::test]
    async fn a_fetch_with_no_body_moves_only_the_flags() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        sink.batch("INBOX", batch_of(vec![fetched(7, "one")]), &a_state(8))
            .await
            .expect("a batch");

        let later = Fetched {
            flags: vec![r"\Flagged".to_string()],
            body: Vec::new(),
            ..fetched(7, "one")
        };
        sink.batch("INBOX", batch_of(vec![later]), &a_state(8))
            .await
            .expect("a batch");

        let found = store.all().expect("a read");
        assert_eq!(found.len(), 1, "the flags wrote a second message");
        assert!(found[0].matches(Flag::Unread), "the message stayed read");
        assert!(found[0].matches(Flag::Flagged));
    }

    #[tokio::test]
    async fn a_uid_that_went_away_loses_its_copy() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        sink.batch("INBOX", batch_of(vec![fetched(7, "one")]), &a_state(8))
            .await
            .expect("a batch");

        let gone = Batch {
            messages: Vec::new(),
            gone: vec![7],
        };
        sink.batch("INBOX", gone, &a_state(8))
            .await
            .expect("a batch");

        let found = store.all().expect("a read");
        assert_eq!(found.len(), 1, "the mirror dropped the message");
        assert!(found[0].is_gone());
        assert_eq!(sink.counts().gone, 1);
    }

    #[tokio::test]
    async fn a_message_that_cannot_be_read_never_stops_the_sync() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        let broken = Fetched {
            body: b"   \r\n".to_vec(),
            ..fetched(7, "one")
        };
        let both = batch_of(vec![broken, fetched(8, "two")]);

        sink.batch("INBOX", both, &a_state(9))
            .await
            .expect("a batch");

        assert_eq!(sink.counts().broken, 1);
        assert_eq!(sink.counts().kept, 1);
        assert_eq!(store.len().expect("a read"), 1);
    }

    #[tokio::test]
    async fn a_copy_in_a_second_folder_joins_the_message() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        sink.batch("INBOX", batch_of(vec![fetched(7, "one")]), &a_state(8))
            .await
            .expect("a batch");
        sink.batch("Archive", batch_of(vec![fetched(2, "one")]), &a_state(3))
            .await
            .expect("a batch");

        let found = store.all().expect("a read");
        assert_eq!(found.len(), 1, "one message became two entries");
        assert_eq!(found[0].folders(), vec!["Archive", "INBOX"]);
    }

    #[tokio::test]
    async fn an_encrypted_message_keeps_its_bytes_and_gives_no_text() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        let body = b"From: alice@example.test\r\n\
             Subject: Secret\r\n\
             Message-ID: <sealed@x.test>\r\n\
             \r\n\
             -----BEGIN PGP MESSAGE-----\r\n\
             hQIMA0000000\r\n\
             -----END PGP MESSAGE-----\r\n"
            .to_vec();
        let sealed = Fetched {
            body: body.clone(),
            ..fetched(7, "one")
        };

        sink.batch("INBOX", batch_of(vec![sealed]), &a_state(8))
            .await
            .expect("a batch");

        let found = store.all().expect("a read");
        assert!(found[0].is_encrypted());
        assert_eq!(found[0].text, "", "the ciphertext reached the index");
        assert_eq!(store.raw(&found[0].id).expect("a read"), Some(body));
    }

    #[tokio::test]
    async fn a_mark_writes_the_state_and_no_message() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        sink.mark("INBOX", &a_state(8)).await.expect("a mark");

        assert!(store.is_empty().expect("a read"));
        assert_eq!(
            store.state("work", "INBOX").expect("a read"),
            Some(keep_state(&a_state(8)))
        );
    }

    #[tokio::test]
    async fn the_state_reads_back_as_the_sync_left_it() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        sink.mark("INBOX", &a_state(8)).await.expect("a mark");

        assert_eq!(sink.state("INBOX").await.expect("a read"), a_state(8));
    }

    #[tokio::test]
    async fn a_folder_that_no_sync_read_starts_new() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        let found = sink.state("INBOX").await.expect("a read");

        assert!(found.is_new());
    }

    #[tokio::test]
    async fn the_sink_remembers_what_it_touched() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        sink.batch("INBOX", batch_of(vec![fetched(7, "one")]), &a_state(8))
            .await
            .expect("a batch");

        let found = store.all().expect("a read");
        assert_eq!(sink.touched().len(), 1);
        assert!(sink.touched().contains(&found[0].id));
    }

    #[tokio::test]
    async fn a_footer_of_the_account_never_reaches_the_index() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let footers =
            vec![Regex::new("Sent from my phone").expect("a pattern")];
        let mut sink = Sink::new(store.clone(), "work").with_footers(footers);

        let mut body = raw_bytes("one");
        body.extend_from_slice(b"--\r\nSent from my phone\r\n");
        let carried = Fetched {
            body,
            ..fetched(7, "one")
        };

        sink.batch("INBOX", batch_of(vec![carried]), &a_state(8))
            .await
            .expect("a batch");

        let found = store.all().expect("a read");
        assert!(!found[0].text.contains("Sent from my phone"));
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 30)]
    fn prop_the_sink_keeps_every_message_of_a_batch(tc: TestCase) {
        let keys: Vec<String> = tc.draw(
            gs::vecs(gs::text().alphabet("abcd").min_size(1).max_size(3))
                .min_size(1)
                .max_size(5),
        );

        let mut unique = keys;
        unique.sort();
        unique.dedup();

        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        let messages: Vec<Fetched> = unique
            .iter()
            .enumerate()
            .map(|(step, key)| fetched(step as u32 + 1, key))
            .collect();

        crate::block_on(sink.batch(
            "INBOX",
            batch_of(messages),
            &a_state(unique.len() as u32 + 1),
        ))
        .expect("a batch");

        assert_eq!(store.len().expect("a read"), unique.len());
        assert_eq!(sink.counts().kept, unique.len() as u64);
        assert_eq!(sink.touched().len(), unique.len());
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_state_survives_the_sink(tc: TestCase) {
        let uid_validity: u32 =
            tc.draw(gs::integers::<u32>().min_value(1).max_value(9_999));
        let uid_next: u32 =
            tc.draw(gs::integers::<u32>().min_value(1).max_value(9_999));
        let highest_mod_seq: u64 =
            tc.draw(gs::integers::<u64>().min_value(0).max_value(999_999));
        let low: u32 =
            tc.draw(gs::integers::<u32>().min_value(1).max_value(100));
        let high: u32 =
            tc.draw(gs::integers::<u32>().min_value(low).max_value(200));

        let state = FolderState {
            uid_validity,
            uid_next,
            highest_mod_seq,
            pending: UidSet::range(low, high),
        };

        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        crate::block_on(sink.mark("INBOX", &state)).expect("a mark");
        let found = crate::block_on(sink.state("INBOX")).expect("a read");

        assert_eq!(found, state, "the state changed on the way to disk");
    }

    #[hegel::test(test_cases = 30)]
    fn prop_the_sink_never_stops_on_bytes_that_are_not_a_message(tc: TestCase) {
        let body: String =
            tc.draw(gs::text().alphabet("a \r\n:<>@").min_size(0).max_size(40));

        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let mut sink = Sink::new(store.clone(), "work");

        let carried = Fetched {
            body: body.into_bytes(),
            ..fetched(7, "one")
        };

        let answer = crate::block_on(sink.batch(
            "INBOX",
            batch_of(vec![carried]),
            &a_state(8),
        ));

        assert!(answer.is_ok(), "bytes that are not mail stopped the sync");
    }
}
