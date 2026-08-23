//! The passages that the semantic leg reads. (§6.2)
//!
//! mailbert embeds every message, and bulk mail as well. A mailbox of
//! 100000 messages costs an acceptable amount of work, and a message
//! that no embedding covers is a message that the semantic leg cannot
//! find.
//!
//! Each passage carries a preamble of `From`, `Subject`, and `Date` in
//! front of its part of the body. A query such as "the invoice from my
//! landlord" then matches on the sender, and not on the body alone.
//!
//! An encrypted message gives its preamble and nothing more. §5.4
//! keeps the ciphertext out of the index, and this module makes the
//! same promise for the embeddings. It does not trust the pipeline
//! above it, because the embeddings and their backup are files that
//! hold no encryption.

use std::{cmp::Ordering, collections::BTreeMap};

use docbert_core::chunking::chunk_text;

use crate::{address::Address, date::day_text, message::Message};

/// How many characters one passage holds. This is the default of
/// docbert, which the model of docbert also reads.
pub const SIZE: usize = docbert_core::chunking::DEFAULT_CHUNK_SIZE;

/// How much two passages of one message share. Mail is short, so a
/// passage that repeats its neighbour buys little and costs an
/// embedding.
pub const OVERLAP: usize = docbert_core::chunking::DEFAULT_CHUNK_OVERLAP;

/// What stands between two fields of a preamble.
const BAR: &str = " | ";

/// What stands between the preamble and the text of a passage.
const BREAK: &str = "\n\n";

/// One passage of a message, ready for the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Passage {
    /// The key of this passage in the embedding database.
    pub key: u64,

    /// Which passage of the message this is, counted from zero.
    pub index: usize,

    /// The preamble and the body, which is what the model reads.
    pub text: String,
}

/// The header line that every passage of a message carries. (§6.2)
///
/// A field that the message does not have does not appear. The date
/// always appears, because every message has one: a message with no
/// `Date` header takes the day that the server received it.
pub fn preamble(message: &Message) -> String {
    let mut fields = Vec::new();

    let senders: Vec<String> =
        message.from.iter().map(Address::to_string).collect();
    if !senders.is_empty() {
        fields.push(format!("From: {}", senders.join(", ")));
    }

    let subject = message.subject.trim();
    if !subject.is_empty() {
        fields.push(format!("Subject: {subject}"));
    }

    fields.push(format!("Date: {}", day_text(message.date)));

    fields.join(BAR)
}

/// The key of passage `index` of the message that `owner` names.
///
/// The key does not carry the name of the model. A model that changed
/// gives the same passage the same key and a new embedding, so the
/// table of owners stays as it is. [`digest`] is what sees the change.
pub fn key(owner: u64, index: usize) -> u64 {
    let mut hasher = blake3::Hasher::new();

    hasher.update(&owner.to_le_bytes());
    hasher.update(&(index as u64).to_le_bytes());

    let bytes = hasher.finalize();
    let head: [u8; 8] = bytes.as_bytes()[..8].try_into().expect("8 bytes");

    u64::from_le_bytes(head)
}

/// Cut a message into the passages that the model embeds. (§6.2)
///
/// A message always gives one passage or more. A message with no body
/// gives its preamble alone, because a message that gives nothing is a
/// message that the semantic leg cannot find.
pub fn passages(
    message: &Message,
    size: usize,
    overlap: usize,
) -> Vec<Passage> {
    let owner = message.id.numeric();
    let head = preamble(message);

    // §5.4: the ciphertext of an encrypted message never reaches the
    // model. Its headers are the whole passage.
    let body = match message.is_encrypted() {
        true => "",
        false => message.text.trim(),
    };

    let parts = parts(body, size.max(1), overlap);
    let texts = match parts.is_empty() {
        true => vec![head],
        false => parts
            .into_iter()
            .map(|part| format!("{head}{BREAK}{part}"))
            .collect(),
    };

    texts
        .into_iter()
        .enumerate()
        .map(|(index, text)| Passage {
            key: key(owner, index),
            index,
            text,
        })
        .collect()
}

/// Cut `body` into the parts that the passages carry.
///
/// [`chunk_text`] of docbert carries each character of the body, so
/// the parts of a body with no overlap hold that body and nothing
/// else. A body that the store holds is searchable to its end.
fn parts(body: &str, size: usize, overlap: usize) -> Vec<String> {
    chunk_text(body, size, overlap)
        .into_iter()
        .map(|chunk| chunk.text)
        .collect()
}

/// A fingerprint of the passages of one message, under one model.
///
/// A second pass compares this against what the store holds. The two
/// that agree let the pass keep the embedding that it has, and that is
/// what makes a sync of one message cost one message.
pub fn digest(model: &str, passages: &[Passage]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();

    // The name of the model is in the fingerprint, so a model that
    // changed makes every message ask for its embedding again.
    stamp(&mut hasher, model.as_bytes());

    for passage in passages {
        stamp(&mut hasher, &passage.key.to_le_bytes());
        stamp(&mut hasher, passage.text.as_bytes());
    }

    *hasher.finalize().as_bytes()
}

/// Give `bytes` to `hasher`, behind the count of them.
///
/// The count keeps two different lists from making one fingerprint.
fn stamp(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Turn the passages that the semantic leg found into messages. (§8.1)
///
/// `found` holds the passages in the order that the index gave, best
/// first, and `owner` names the message of each passage. A message
/// takes the score of its best passage, and appears one time. This is
/// the MaxSim of ColBERT, one step above the tokens.
pub fn collapse(found: &[(u64, f32)], owner: &BTreeMap<u64, u64>) -> Vec<u64> {
    let mut best: Vec<(u64, f32)> = Vec::new();
    let mut at: BTreeMap<u64, usize> = BTreeMap::new();

    for (passage, score) in found {
        // A passage that no message owns is one that a message let go
        // and the index has not lost yet.
        let Some(message) = owner.get(passage) else {
            continue;
        };

        match at.get(message) {
            Some(seen) => best[*seen].1 = best[*seen].1.max(*score),
            None => {
                at.insert(*message, best.len());
                best.push((*message, *score));
            }
        }
    }

    // The sort keeps the order that it was given, so two messages with
    // one score stay in the order that the index gave them.
    best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    best.into_iter().map(|(message, _)| message).collect()
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_message_always_gives_a_passage` | invariant | §6.2 says that nothing is invisible to the semantic leg. A message with no passage cannot be found by meaning. |
    //! | `prop_every_passage_carries_the_preamble` | invariant | §6.2 puts the sender and the subject in front of each part. A passage that lost them matches on the body alone. |
    //! | `prop_the_passages_hold_the_whole_body` | model-based | A part of the body that no passage holds is a part that no query reaches. |
    //! | `prop_an_encrypted_message_never_gives_its_body` | invariant | §5.4. The embeddings are a file that holds no encryption, so ciphertext in a passage defeats the sender. |
    //! | `prop_the_keys_of_one_message_never_repeat` | invariant | Two passages with one key overwrite each other in the embedding database, and one of them is lost. |
    //! | `prop_the_digest_follows_the_passages` | metamorphic | The digest is what lets a second pass keep an embedding. A digest that misses a change leaves a stale one. |
    //! | `prop_a_collapse_names_each_message_one_time` | model-based | §8.1 fuses ranked lists of messages. A message that appears twice takes two of the 100 places. |
    //! | `prop_a_collapse_gives_a_message_its_best_passage` | model-based | MaxSim, one step above the tokens. A message that takes a worse score ranks below what it matched. |

    use std::collections::BTreeSet;

    use hegel::{TestCase, generators as gs};

    use super::*;
    use crate::{
        message::Location,
        mime::{Parsed, Source},
    };

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    /// The day that the tests date a message.
    const NOW: i64 = 1_755_820_800;

    /// A small chunk, so a short body still cuts into parts.
    const SMALL: usize = 40;

    fn location() -> Location {
        Location {
            account: "work".to_string(),
            folder: "INBOX".to_string(),
            uid: 1,
            uid_validity: 1,
            received: NOW,
            flags: BTreeSet::new(),
        }
    }

    /// One message, built from its parts and not from raw bytes.
    fn message(key: &str, subject: &str, body: &str) -> Message {
        let parsed = Parsed {
            message_id: Some(format!("{key}@x.test")),
            in_reply_to: None,
            references: Vec::new(),
            date: Some(NOW),
            from: vec![
                Address::new(Some("Alice Smith"), "alice@ex.test")
                    .expect("an address"),
            ],
            to: Vec::new(),
            cc: Vec::new(),
            subject: subject.to_string(),
            list_id: None,
            source: Source::Plain,
            full: body.to_string(),
            text: body.to_string(),
            quote_only: false,
            is_bulk: false,
            attachments: Vec::new(),
        };

        Message::new(parsed, location(), Vec::<String>::new())
    }

    /// What a passage holds after its preamble.
    fn body_of(passage: &Passage) -> String {
        match passage.text.split_once(BREAK) {
            Some((_, rest)) => rest.to_string(),
            None => String::new(),
        }
    }

    /// Words that make a body of about `count` characters.
    fn a_body(tc: &TestCase, count: usize) -> String {
        let words: Vec<String> = tc.draw(
            gs::vecs(gs::text().alphabet("abcdefghij").min_size(1).max_size(9))
                .min_size(1)
                .max_size(count),
        );

        words.join(" ")
    }

    // -----------------------------------------------------------------
    // The preamble.
    // -----------------------------------------------------------------

    #[test]
    fn the_preamble_names_the_sender_the_subject_and_the_day() {
        let found = message("a", "Deposit for the apartment", "It is due.");

        assert_eq!(
            preamble(&found),
            "From: Alice Smith <alice@ex.test> | \
             Subject: Deposit for the apartment | Date: 2025-08-22"
        );
    }

    #[test]
    fn a_message_with_no_subject_leaves_that_field_out() {
        let found = message("a", "   ", "It is due.");

        assert_eq!(
            preamble(&found),
            "From: Alice Smith <alice@ex.test> | Date: 2025-08-22"
        );
    }

    #[test]
    fn a_message_with_no_sender_leaves_that_field_out() {
        let mut found = message("a", "Deposit", "It is due.");
        found.from.clear();

        assert_eq!(preamble(&found), "Subject: Deposit | Date: 2025-08-22");
    }

    // -----------------------------------------------------------------
    // The passages.
    // -----------------------------------------------------------------

    #[test]
    fn a_short_message_gives_one_passage() {
        let found = message("a", "Deposit", "The deposit is due on Friday.");
        let cut = passages(&found, SIZE, OVERLAP);

        assert_eq!(cut.len(), 1);
        assert_eq!(cut[0].index, 0);
        assert_eq!(body_of(&cut[0]), "The deposit is due on Friday.");
    }

    #[test]
    fn a_long_message_gives_more_than_one_passage() {
        let body = "invoice ".repeat(60);
        let found = message("a", "Deposit", &body);

        let cut = passages(&found, SMALL, 0);

        assert!(cut.len() > 1, "{} passages", cut.len());
    }

    #[test]
    fn a_message_with_no_body_still_gives_one_passage() {
        let found = message("a", "Deposit", "   ");
        let cut = passages(&found, SIZE, OVERLAP);

        assert_eq!(cut.len(), 1);
        assert_eq!(cut[0].text, preamble(&found));
    }

    #[test]
    fn an_encrypted_message_gives_its_headers_alone() {
        let mut found = message("a", "Deposit", "-----BEGIN PGP MESSAGE-----");
        found.source = Source::Encrypted;

        let cut = passages(&found, SIZE, OVERLAP);

        assert_eq!(cut.len(), 1);
        assert_eq!(cut[0].text, preamble(&found));
        assert!(!cut[0].text.contains("PGP"));
    }

    #[test]
    fn the_passages_of_one_message_take_the_keys_of_its_identity() {
        let found = message("a", "Deposit", &"invoice ".repeat(60));
        let cut = passages(&found, SMALL, 0);
        let owner = found.id.numeric();

        for passage in &cut {
            assert_eq!(passage.key, key(owner, passage.index));
        }
    }

    #[test]
    fn two_messages_never_share_a_key() {
        let first = passages(&message("a", "One", "x"), SIZE, OVERLAP);
        let second = passages(&message("b", "One", "x"), SIZE, OVERLAP);

        assert_ne!(first[0].key, second[0].key);
    }

    // -----------------------------------------------------------------
    // The digest.
    // -----------------------------------------------------------------

    #[test]
    fn the_digest_changes_when_the_model_changes() {
        let cut = passages(&message("a", "Deposit", "It is due."), SIZE, 0);

        assert_ne!(digest("one", &cut), digest("two", &cut));
    }

    #[test]
    fn the_digest_of_the_same_passages_is_the_same() {
        let first = passages(&message("a", "Deposit", "It is due."), SIZE, 0);
        let second = passages(&message("a", "Deposit", "It is due."), SIZE, 0);

        assert_eq!(digest("one", &first), digest("one", &second));
    }

    // -----------------------------------------------------------------
    // The collapse.
    // -----------------------------------------------------------------

    #[test]
    fn a_passage_that_no_message_owns_is_dropped() {
        let owner = BTreeMap::from([(1, 100)]);
        let found = [(1, 0.9), (2, 0.8)];

        assert_eq!(collapse(&found, &owner), vec![100]);
    }

    #[test]
    fn a_message_takes_the_place_of_its_best_passage() {
        let owner = BTreeMap::from([(1, 100), (2, 200), (3, 100)]);
        let found = [(1, 0.4), (2, 0.6), (3, 0.9)];

        assert_eq!(collapse(&found, &owner), vec![100, 200]);
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 200)]
    fn prop_a_message_always_gives_a_passage(tc: TestCase) {
        let subject: String = tc.draw(gs::text().alphabet(" abc").max_size(20));
        let body: String = tc.draw(gs::text().alphabet(" abc\n").max_size(60));

        let found = message("a", &subject, &body);
        let cut = passages(&found, SMALL, 0);

        assert!(!cut.is_empty(), "no passage for {body:?}");
    }

    #[hegel::test(test_cases = 200)]
    fn prop_every_passage_carries_the_preamble(tc: TestCase) {
        let body = a_body(&tc, 40);
        let found = message("a", "Deposit", &body);
        let head = preamble(&found);

        for passage in passages(&found, SMALL, 0) {
            assert!(
                passage.text.starts_with(&head),
                "passage {} lost the preamble",
                passage.index
            );
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_passages_hold_the_whole_body(tc: TestCase) {
        let body = a_body(&tc, 40);
        let found = message("a", "Deposit", &body);

        let joined: String = passages(&found, SMALL, 0)
            .iter()
            .map(body_of)
            .collect::<Vec<_>>()
            .join("");

        let want: String = body
            .trim()
            .chars()
            .filter(|at| !at.is_whitespace())
            .collect();
        let got: String =
            joined.chars().filter(|at| !at.is_whitespace()).collect();

        assert_eq!(got, want);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_an_encrypted_message_never_gives_its_body(tc: TestCase) {
        let secret = a_body(&tc, 40);
        let mut found = message("a", "Deposit", &secret);
        found.source = Source::Encrypted;

        let cut = passages(&found, SMALL, 0);

        // The headers are the whole passage. This is the exact claim,
        // and it holds however the ciphertext reads.
        assert_eq!(cut.len(), 1);
        assert_eq!(cut[0].text, preamble(&found));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_keys_of_one_message_never_repeat(tc: TestCase) {
        let body = a_body(&tc, 60);
        let found = message("a", "Deposit", &body);

        let cut = passages(&found, SMALL, 0);
        let keys: BTreeSet<u64> = cut.iter().map(|one| one.key).collect();

        assert_eq!(keys.len(), cut.len());
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_digest_follows_the_passages(tc: TestCase) {
        let first = a_body(&tc, 40);
        let second = a_body(&tc, 40);

        let one = passages(&message("a", "Deposit", &first), SMALL, 0);
        let two = passages(&message("a", "Deposit", &second), SMALL, 0);

        assert_eq!(digest("m", &one) == digest("m", &two), one == two);
    }

    #[hegel::test(test_cases = 300)]
    fn prop_a_collapse_names_each_message_one_time(tc: TestCase) {
        let (found, owner) = tc.draw(a_result());

        let ranked = collapse(&found, &owner);
        let names: BTreeSet<u64> = ranked.iter().copied().collect();

        assert_eq!(names.len(), ranked.len());
    }

    #[hegel::test(test_cases = 300)]
    fn prop_a_collapse_gives_a_message_its_best_passage(tc: TestCase) {
        let (found, owner) = tc.draw(a_result());

        let mut best: BTreeMap<u64, f32> = BTreeMap::new();
        for (passage, score) in &found {
            if let Some(message) = owner.get(passage) {
                let seen = best.entry(*message).or_insert(f32::MIN);
                *seen = seen.max(*score);
            }
        }

        let ranked = collapse(&found, &owner);

        assert_eq!(ranked.len(), best.len());
        for pair in ranked.windows(2) {
            assert!(
                best[&pair[0]] >= best[&pair[1]],
                "{:?} came before a better message",
                pair[0]
            );
        }
    }

    /// A ranked list of passages, and the message that owns each one.
    #[hegel::composite]
    fn a_result(tc: TestCase) -> (Vec<(u64, f32)>, BTreeMap<u64, u64>) {
        let count: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(12));
        let messages: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(4));

        let mut found = Vec::new();
        let mut owner = BTreeMap::new();

        for passage in 0..count as u64 {
            let known: bool = tc.draw(gs::booleans());
            let score: i64 =
                tc.draw(gs::integers::<i64>().min_value(0).max_value(100));

            if known {
                let at: usize = tc.draw(
                    gs::integers::<usize>()
                        .min_value(0)
                        .max_value(messages - 1),
                );
                owner.insert(passage, 1_000 + at as u64);
            }

            found.push((passage, score as f32 / 100.0));
        }

        // The index gives its best passage first.
        found.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("a score"));

        (found, owner)
    }
}
