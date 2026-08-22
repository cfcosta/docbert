//! Threading: which messages belong to one conversation.
//!
//! mailbert runs the JWZ algorithm on `References` and `In-Reply-To`.
//! When that chain breaks, it merges two threads on a normalized
//! subject **only** if the participants of the two threads overlap and
//! the messages are inside a time window.
//!
//! The constraints prevent the failure mode of subject-only threading,
//! where two unrelated `Re: quick question` threads become one.
//!
//! See `docs/mailbert.md` §5.5.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{
    address::fold,
    message_id::{MessageId, normalize_message_id},
};

/// How far apart two threads may be and still merge on a subject.
pub const DEFAULT_MERGE_WINDOW_DAYS: i64 = 30;

/// Seconds in a day.
const DAY: i64 = 86_400;

/// The prefixes that a client puts in front of a subject it replies to.
const REPLY_PREFIXES: [&str; 6] = ["re", "fwd", "fw", "aw", "sv", "vs"];

/// The brackets that hold the count of a repeated `Re[2]:` prefix.
const COUNT_BRACKETS: [(char, char); 2] = [('[', ']'), ('(', ')')];

/// One message, as threading sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadInput {
    /// The identity of the message. See [`MessageId`].
    pub id: MessageId,

    /// The `Message-ID` header, exactly as it arrived.
    pub message_id: Option<String>,

    /// The `References` header, oldest first.
    pub references: Vec<String>,

    /// The `In-Reply-To` header.
    pub in_reply_to: Option<String>,

    pub subject: String,

    /// Seconds since the Unix epoch.
    pub date: i64,

    /// Every address on the message, lowercased.
    pub participants: Vec<String>,
}

/// A thread, named by the identity of its earliest message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ThreadId(MessageId);

/// Which thread each message belongs to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Threading {
    of_message: HashMap<MessageId, ThreadId>,
    members: BTreeMap<ThreadId, Vec<MessageId>>,
}

/// Reduce a subject to the form that two threads can be compared on.
///
/// Reply prefixes, forward prefixes, and mailing-list tags all leave,
/// case and accents fold, and runs of whitespace collapse.
///
/// ```
/// use mailbert_core::threading::normalize_subject;
///
/// assert_eq!(normalize_subject("Re: [team] Café  Plans"), "cafe plans");
/// ```
pub fn normalize_subject(subject: &str) -> String {
    let folded = fold(subject);
    let mut rest = folded.trim();

    while let Some(shorter) = strip_one_prefix(rest) {
        rest = shorter.trim_start();
    }

    rest.split_whitespace().collect::<Vec<&str>>().join(" ")
}

/// Thread `messages`, with the default merge window.
///
/// ```
/// use mailbert_core::{MessageId, ThreadInput, threading};
///
/// let parent = "<a@x.example>";
/// let root = ThreadInput::new(
///     MessageId::from_message_id(parent).unwrap(),
///     "Budget",
///     0,
/// )
/// .with_message_id(parent);
///
/// let reply = ThreadInput::new(
///     MessageId::from_message_id("<b@x.example>").unwrap(),
///     "Re: Budget",
///     3_600,
/// )
/// .with_in_reply_to(parent);
///
/// let threading = threading::thread(&[root, reply]);
/// assert_eq!(threading.len(), 1);
/// ```
pub fn thread(messages: &[ThreadInput]) -> Threading {
    thread_with_window(messages, DEFAULT_MERGE_WINDOW_DAYS)
}

/// Thread `messages`, merging on a subject inside `window_days`.
///
/// The result does not depend on the order of `messages`, because sync
/// fetches folders in parallel and hands them over in no fixed order.
/// Each merge decision reads the threads as the reference pass left
/// them, so no merge can make a second merge possible.
pub fn thread_with_window(
    messages: &[ThreadInput],
    window_days: i64,
) -> Threading {
    let mut sets = Sets::with_len(messages.len());

    // 1. JWZ. Every `Message-ID` is a node of its own, and a message
    //    joins each identifier it names. An identifier that no message
    //    carries still joins the messages that point at it, which is
    //    how two replies to an absent parent find each other.
    let mut nodes: HashMap<String, usize> = HashMap::new();
    for (index, message) in messages.iter().enumerate() {
        for raw in message.identifiers() {
            let Some(identifier) = normalize_message_id(raw) else {
                continue;
            };

            let node = match nodes.get(&identifier) {
                Some(node) => *node,
                None => {
                    let node = sets.push();
                    nodes.insert(identifier, node);
                    node
                }
            };

            sets.union(index, node);
        }
    }

    // 2. Describe each thread before any merge runs.
    let mut spans: HashMap<usize, Span> = HashMap::new();
    for (index, message) in messages.iter().enumerate() {
        let root = sets.find(index);
        spans
            .entry(root)
            .or_insert_with(|| Span::new(message))
            .absorb(message);
    }

    // 3. Merge on a normalized subject, but only where the participants
    //    overlap and the messages are close in time.
    let window = window_days.saturating_mul(DAY);
    let mut by_subject: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, message) in messages.iter().enumerate() {
        let subject = normalize_subject(&message.subject);
        if subject.is_empty() {
            continue;
        }

        let root = sets.find(index);
        let roots = by_subject.entry(subject).or_default();
        if !roots.contains(&root) {
            roots.push(root);
        }
    }

    for roots in by_subject.values() {
        for (position, left) in roots.iter().enumerate() {
            for right in &roots[position + 1..] {
                let (Some(a), Some(b)) = (spans.get(left), spans.get(right))
                else {
                    continue;
                };

                if a.shares_a_participant(b) && a.gap(b) <= window {
                    sets.union(*left, *right);
                }
            }
        }
    }

    // 4. Collect. Each thread is named by its earliest message.
    let mut grouped: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for index in 0..messages.len() {
        grouped.entry(sets.find(index)).or_default().push(index);
    }

    let mut of_message: HashMap<MessageId, ThreadId> = HashMap::new();
    let mut members: BTreeMap<ThreadId, Vec<MessageId>> = BTreeMap::new();

    for indexes in grouped.values() {
        // The date orders the thread, and the identity breaks a tie, so
        // that two messages of the same second still have one order.
        let mut dated: Vec<(i64, MessageId)> = indexes
            .iter()
            .map(|index| (messages[*index].date, messages[*index].id))
            .collect();
        dated.sort();

        let ordered: Vec<MessageId> =
            dated.into_iter().map(|(_, id)| id).collect();

        let thread = ThreadId(ordered[0]);
        for member in &ordered {
            of_message.insert(*member, thread);
        }
        members.insert(thread, ordered);
    }

    Threading {
        of_message,
        members,
    }
}

impl ThreadInput {
    /// A message with no references and no participants.
    pub fn new(id: MessageId, subject: impl Into<String>, date: i64) -> Self {
        Self {
            id,
            message_id: None,
            references: Vec::new(),
            in_reply_to: None,
            subject: subject.into(),
            date,
            participants: Vec::new(),
        }
    }

    /// Set the `Message-ID` header.
    #[must_use]
    pub fn with_message_id(mut self, raw: impl Into<String>) -> Self {
        self.message_id = Some(raw.into());
        self
    }

    /// Set the `In-Reply-To` header.
    #[must_use]
    pub fn with_in_reply_to(mut self, raw: impl Into<String>) -> Self {
        self.in_reply_to = Some(raw.into());
        self
    }

    /// Set the `References` header.
    #[must_use]
    pub fn with_references(mut self, raw: Vec<String>) -> Self {
        self.references = raw;
        self
    }

    /// Set the addresses on the message.
    #[must_use]
    pub fn with_participants(mut self, who: Vec<String>) -> Self {
        self.participants = who;
        self
    }

    /// Every `Message-ID` the message names, its own included.
    fn identifiers(&self) -> impl Iterator<Item = &str> {
        self.message_id
            .iter()
            .chain(self.in_reply_to.iter())
            .chain(self.references.iter())
            .map(String::as_str)
    }
}

impl ThreadId {
    /// The identity of the earliest message of the thread.
    pub fn root(self) -> MessageId {
        self.0
    }

    /// The full 64-character hex digest. The index stores this.
    pub fn full_hex(self) -> String {
        self.0.full_hex()
    }

    /// The short prefix, for display.
    pub fn short(self) -> String {
        self.0.short()
    }
}

impl Threading {
    /// The thread of one message.
    pub fn thread_of(&self, id: MessageId) -> Option<ThreadId> {
        self.of_message.get(&id).copied()
    }

    /// The messages of one thread, earliest first.
    pub fn members(&self, thread: ThreadId) -> &[MessageId] {
        self.members
            .get(&thread)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// Every thread, in a fixed order.
    pub fn threads(&self) -> impl Iterator<Item = (ThreadId, &[MessageId])> {
        self.members
            .iter()
            .map(|(thread, members)| (*thread, members.as_slice()))
    }

    /// How many threads the messages formed.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether there is no thread at all.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

/// What one thread looked like before any subject merge.
struct Span {
    participants: HashSet<String>,
    first: i64,
    last: i64,
}

impl Span {
    fn new(message: &ThreadInput) -> Self {
        Self {
            participants: HashSet::new(),
            first: message.date,
            last: message.date,
        }
    }

    fn absorb(&mut self, message: &ThreadInput) {
        self.first = self.first.min(message.date);
        self.last = self.last.max(message.date);

        for who in &message.participants {
            let folded = fold(who);
            let trimmed = folded.trim();
            if !trimmed.is_empty() {
                self.participants.insert(trimmed.to_string());
            }
        }
    }

    /// The first condition of a subject merge.
    fn shares_a_participant(&self, other: &Self) -> bool {
        !self.participants.is_disjoint(&other.participants)
    }

    /// The second condition: how far the two date ranges stand apart.
    /// Ranges that touch or overlap have a gap of zero.
    fn gap(&self, other: &Self) -> i64 {
        let start = self.first.max(other.first);
        let end = self.last.min(other.last);

        start.saturating_sub(end).max(0)
    }
}

/// Union-find over message nodes and the `Message-ID` nodes they name.
struct Sets {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl Sets {
    fn with_len(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    /// Add one node and return it.
    fn push(&mut self) -> usize {
        let node = self.parent.len();
        self.parent.push(node);
        self.rank.push(0);
        node
    }

    fn find(&mut self, node: usize) -> usize {
        let mut root = node;
        while self.parent[root] != root {
            root = self.parent[root];
        }

        let mut walk = node;
        while self.parent[walk] != root {
            let next = self.parent[walk];
            self.parent[walk] = root;
            walk = next;
        }

        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a == b {
            return;
        }

        let (low, high) = if self.rank[a] < self.rank[b] {
            (a, b)
        } else {
            (b, a)
        };

        self.parent[low] = high;
        if self.rank[low] == self.rank[high] {
            self.rank[high] += 1;
        }
    }
}

/// Remove one leading list tag or reply prefix, if there is one.
fn strip_one_prefix(text: &str) -> Option<&str> {
    if let Some(rest) = text.strip_prefix('[')
        && let Some(close) = rest.find(']')
    {
        return Some(&rest[close + 1..]);
    }

    for prefix in REPLY_PREFIXES {
        let Some(rest) = text.strip_prefix(prefix) else {
            continue;
        };

        if let Some(rest) = strip_count(rest).strip_prefix(':') {
            return Some(rest);
        }
    }

    None
}

/// Remove the `[2]` or `(2)` that a client adds to a repeated `Re:`.
fn strip_count(text: &str) -> &str {
    for (open, close) in COUNT_BRACKETS {
        if let Some(rest) = text.strip_prefix(open)
            && let Some(end) = rest.find(close)
            && end > 0
            && rest[..end].bytes().all(|byte| byte.is_ascii_digit())
        {
            return &rest[end + 1..];
        }
    }

    text
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_threading_is_a_partition` | invariant | Each message belongs to exactly one thread. A message in two threads is a message the reader sees twice. |
    //! | `prop_threading_ignores_message_order` | differential | Sync fetches folders in parallel. Two runs over the same mailbox must group it the same way. |
    //! | `prop_a_reference_always_joins` | invariant | The `References` chain is the reliable half of threading. Losing a link splits a conversation. |
    //! | `prop_subject_merge_needs_both_conditions` | model-based | This is the rule of §5.5. Either condition alone merges two `Re: quick question` threads that share nothing. |
    //! | `prop_the_thread_id_is_its_earliest_member` | invariant | The thread must be named by a message it holds, or `thread:` cannot find it. |
    //! | `prop_members_are_ordered_by_date` | invariant | Thread grouping prints the members in order. |
    //! | `prop_normalize_subject_is_idempotent` | algebraic | Normalization runs on both sides of a comparison. If it were not stable, the two sides could disagree. |
    //! | `prop_no_reply_prefix_survives` | invariant | `Re: Re: Fwd: X` and `X` name one conversation. |

    use std::collections::BTreeSet;

    use hegel::{TestCase, generators as gs};

    use super::*;

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    fn mid(n: usize) -> String {
        format!("<m{n}@x.example>")
    }

    /// A message identified by its `Message-ID`, dated in whole days.
    fn message(n: usize, subject: &str, day: i64) -> ThreadInput {
        let raw = mid(n);

        ThreadInput::new(
            MessageId::from_message_id(&raw).expect("a valid Message-ID"),
            subject,
            day * DAY,
        )
        .with_message_id(raw)
    }

    fn partition(threading: &Threading) -> BTreeSet<Vec<MessageId>> {
        threading
            .threads()
            .map(|(_, members)| {
                let mut sorted = members.to_vec();
                sorted.sort();
                sorted
            })
            .collect()
    }

    fn same_thread(
        threading: &Threading,
        a: &ThreadInput,
        b: &ThreadInput,
    ) -> bool {
        threading.thread_of(a.id) == threading.thread_of(b.id)
    }

    // -----------------------------------------------------------------
    // Generators.
    // -----------------------------------------------------------------

    fn a_subject() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "Quick question".to_string(),
            "Re: Quick question".to_string(),
            "[team] Quick question".to_string(),
            "Budget 2026".to_string(),
            String::new(),
        ])
    }

    fn an_address() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "alice@x.example".to_string(),
            "bob@x.example".to_string(),
            "carol@y.example".to_string(),
        ])
    }

    /// A mailbox where some messages reply to an earlier one.
    #[hegel::composite]
    fn a_mailbox(tc: TestCase) -> Vec<ThreadInput> {
        let count: usize =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(8));

        let mut mailbox: Vec<ThreadInput> = Vec::new();

        for n in 0..count {
            let subject: String = tc.draw(a_subject());
            let day: i64 =
                tc.draw(gs::integers::<i64>().min_value(0).max_value(120));
            let who: Vec<String> =
                tc.draw(gs::vecs(an_address()).min_size(0).max_size(3));
            let parent: Option<usize> = tc.draw(gs::optional(
                gs::integers::<usize>().min_value(0).max_value(7),
            ));

            let mut input = message(n, &subject, day).with_participants(who);

            if let Some(parent) = parent
                && parent < n
            {
                input = input.with_in_reply_to(mid(parent));
            }

            mailbox.push(input);
        }

        mailbox
    }

    // -----------------------------------------------------------------
    // Unit tests.
    // -----------------------------------------------------------------

    #[test]
    fn an_empty_mailbox_has_no_thread() {
        let threading = thread(&[]);

        assert!(threading.is_empty());
        assert_eq!(threading.len(), 0);
        assert_eq!(threading.threads().count(), 0);
    }

    #[test]
    fn one_message_is_its_own_thread() {
        let a = message(0, "Budget", 0);
        let threading = thread(std::slice::from_ref(&a));

        assert_eq!(threading.len(), 1);

        let id = threading.thread_of(a.id).unwrap();
        assert_eq!(id.root(), a.id);
        assert_eq!(threading.members(id), [a.id]);
    }

    #[test]
    fn a_reply_joins_its_parent() {
        let a = message(0, "Budget", 0);
        let b = message(1, "Re: Budget", 1).with_in_reply_to(mid(0));

        let threading = thread(&[a.clone(), b.clone()]);

        assert_eq!(threading.len(), 1);
        assert!(same_thread(&threading, &a, &b));
    }

    #[test]
    fn a_references_chain_joins_the_whole_thread() {
        let a = message(0, "Budget", 0);
        let b = message(1, "Re: Budget", 1).with_references(vec![mid(0)]);
        let c =
            message(2, "Re: Budget", 2).with_references(vec![mid(0), mid(1)]);

        let threading = thread(&[a.clone(), b.clone(), c.clone()]);

        assert_eq!(threading.len(), 1);
        assert!(same_thread(&threading, &a, &c));
    }

    #[test]
    fn an_absent_ancestor_still_joins_two_replies() {
        // Neither reply holds the message they both answer, which is
        // ordinary when a folder is synced and its parent is not.
        let a = message(1, "Re: Budget", 1).with_references(vec![mid(0)]);
        let b = message(2, "Re: Budget", 2).with_references(vec![mid(0)]);

        let threading = thread(&[a.clone(), b.clone()]);

        assert_eq!(threading.len(), 1);
        assert!(same_thread(&threading, &a, &b));
    }

    #[test]
    fn a_broken_chain_merges_on_subject_when_both_hold() {
        let a = message(0, "Budget 2026", 0)
            .with_participants(vec!["alice@x.example".to_string()]);
        let b = message(1, "Re: Budget 2026", 3)
            .with_participants(vec!["alice@x.example".to_string()]);

        let threading = thread(&[a.clone(), b.clone()]);

        assert_eq!(threading.len(), 1);
        assert!(same_thread(&threading, &a, &b));
    }

    #[test]
    fn two_quick_question_threads_stay_apart() {
        // The named failure mode of subject-only threading.
        let a = message(0, "Quick question", 0)
            .with_participants(vec!["alice@x.example".to_string()]);
        let b = message(1, "Re: quick question", 1)
            .with_participants(vec!["carol@y.example".to_string()]);

        let threading = thread(&[a.clone(), b.clone()]);

        assert_eq!(threading.len(), 2);
        assert!(!same_thread(&threading, &a, &b));
    }

    #[test]
    fn a_distant_message_never_merges_on_subject() {
        let a = message(0, "Budget 2026", 0)
            .with_participants(vec!["alice@x.example".to_string()]);
        let b = message(1, "Re: Budget 2026", 60)
            .with_participants(vec!["alice@x.example".to_string()]);

        let threading = thread(&[a.clone(), b.clone()]);

        assert_eq!(threading.len(), 2);
        assert!(!same_thread(&threading, &a, &b));
    }

    #[test]
    fn a_reference_beats_the_time_window() {
        // An explicit reply is evidence. A year of silence is not.
        let a = message(0, "Budget 2026", 0);
        let b = message(1, "Re: Budget 2026", 400).with_in_reply_to(mid(0));

        let threading = thread(&[a.clone(), b.clone()]);

        assert_eq!(threading.len(), 1);
        assert!(same_thread(&threading, &a, &b));
    }

    #[test]
    fn an_empty_subject_never_merges() {
        let a = message(0, "", 0)
            .with_participants(vec!["alice@x.example".to_string()]);
        let b = message(1, "   ", 1)
            .with_participants(vec!["alice@x.example".to_string()]);

        assert_eq!(thread(&[a, b]).len(), 2);
    }

    #[test]
    fn the_thread_id_is_the_earliest_message() {
        let a = message(0, "Budget", 10);
        let b = message(1, "Re: Budget", 2).with_in_reply_to(mid(0));

        let threading = thread(&[a.clone(), b.clone()]);
        let id = threading.thread_of(a.id).unwrap();

        assert_eq!(id.root(), b.id, "the reply is the earlier message here");
        assert_eq!(threading.members(id), [b.id, a.id]);
    }

    #[test]
    fn the_window_is_configurable() {
        let a = message(0, "Budget 2026", 0)
            .with_participants(vec!["alice@x.example".to_string()]);
        let b = message(1, "Re: Budget 2026", 60)
            .with_participants(vec!["alice@x.example".to_string()]);

        let messages = [a, b];

        assert_eq!(thread_with_window(&messages, 30).len(), 2);
        assert_eq!(thread_with_window(&messages, 90).len(), 1);
    }

    #[test]
    fn normalize_subject_strips_reply_and_forward_prefixes() {
        assert_eq!(normalize_subject("Re: Budget"), "budget");
        assert_eq!(normalize_subject("RE: Re: Fwd: Budget"), "budget");
        assert_eq!(normalize_subject("Fw: Budget"), "budget");
        assert_eq!(normalize_subject("Re[2]: Budget"), "budget");
        assert_eq!(normalize_subject("Re(3): Budget"), "budget");
        assert_eq!(normalize_subject("Aw: Budget"), "budget");
    }

    #[test]
    fn normalize_subject_strips_a_list_tag() {
        assert_eq!(normalize_subject("[team] Budget"), "budget");
        assert_eq!(normalize_subject("Re: [team] Budget"), "budget");
        assert_eq!(normalize_subject("[team] Re: Budget"), "budget");
    }

    #[test]
    fn normalize_subject_folds_case_accents_and_whitespace() {
        assert_eq!(normalize_subject("Café  Plans"), "cafe plans");
        assert_eq!(normalize_subject("  Budget\t2026 "), "budget 2026");
        assert_eq!(normalize_subject(""), "");
    }

    #[test]
    fn normalize_subject_keeps_a_word_that_starts_like_a_prefix() {
        // `read` starts with `re`, and `vscode` with `vs`.
        assert_eq!(normalize_subject("Read this"), "read this");
        assert_eq!(normalize_subject("vscode setup"), "vscode setup");
        assert_eq!(normalize_subject("[unclosed budget"), "[unclosed budget");
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 200)]
    fn prop_threading_is_a_partition(tc: TestCase) {
        let mailbox: Vec<ThreadInput> = tc.draw(a_mailbox());

        let threading = thread(&mailbox);

        let mut seen: Vec<MessageId> = Vec::new();
        for (id, members) in threading.threads() {
            for member in members {
                assert!(!seen.contains(member), "{member:?} is in two threads");
                assert_eq!(threading.thread_of(*member), Some(id));
                seen.push(*member);
            }
        }

        for input in &mailbox {
            assert!(seen.contains(&input.id), "{:?} has no thread", input.id);
        }
        assert_eq!(seen.len(), mailbox.len());
    }

    #[hegel::test(test_cases = 200)]
    fn prop_threading_ignores_message_order(tc: TestCase) {
        let mailbox: Vec<ThreadInput> = tc.draw(a_mailbox());

        let forward = thread(&mailbox);
        let reversed: Vec<ThreadInput> =
            mailbox.iter().rev().cloned().collect();
        let backward = thread(&reversed);

        assert_eq!(partition(&forward), partition(&backward));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_reference_always_joins(tc: TestCase) {
        let mailbox: Vec<ThreadInput> = tc.draw(a_mailbox());

        let threading = thread(&mailbox);

        for input in &mailbox {
            let Some(parent) = &input.in_reply_to else {
                continue;
            };

            let Some(target) = mailbox
                .iter()
                .find(|m| m.message_id.as_deref() == Some(parent.as_str()))
            else {
                continue;
            };

            assert!(
                same_thread(&threading, input, target),
                "{parent} did not join its reply"
            );
        }
    }

    #[hegel::test(test_cases = 300)]
    fn prop_subject_merge_needs_both_conditions(tc: TestCase) {
        let shared: bool = tc.draw(gs::booleans());
        let days: i64 =
            tc.draw(gs::integers::<i64>().min_value(0).max_value(90));

        let a = message(0, "Budget 2026", 0)
            .with_participants(vec!["alice@x.example".to_string()]);
        let b = message(1, "Re: Budget 2026", days).with_participants(vec![
            if shared {
                "alice@x.example".to_string()
            } else {
                "carol@y.example".to_string()
            },
        ]);

        let threading = thread(&[a.clone(), b.clone()]);

        let expected = shared && days <= DEFAULT_MERGE_WINDOW_DAYS;
        assert_eq!(
            same_thread(&threading, &a, &b),
            expected,
            "shared={shared} days={days}"
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_thread_id_is_its_earliest_member(tc: TestCase) {
        let mailbox: Vec<ThreadInput> = tc.draw(a_mailbox());

        let threading = thread(&mailbox);

        for (id, members) in threading.threads() {
            assert!(members.contains(&id.root()));

            let earliest = members
                .iter()
                .filter_map(|m| mailbox.iter().find(|i| i.id == *m))
                .min_by_key(|i| (i.date, i.id))
                .expect("a thread holds a message");

            assert_eq!(id.root(), earliest.id);
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_members_are_ordered_by_date(tc: TestCase) {
        let mailbox: Vec<ThreadInput> = tc.draw(a_mailbox());

        let threading = thread(&mailbox);

        for (_, members) in threading.threads() {
            let dates: Vec<i64> = members
                .iter()
                .filter_map(|m| mailbox.iter().find(|i| i.id == *m))
                .map(|i| i.date)
                .collect();

            for pair in dates.windows(2) {
                assert!(pair[0] <= pair[1], "out of order: {dates:?}");
            }
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_normalize_subject_is_idempotent(tc: TestCase) {
        let subject: String = tc.draw(gs::text().min_size(0).max_size(60));

        let once = normalize_subject(&subject);
        assert_eq!(normalize_subject(&once), once, "{subject:?}");
    }

    #[hegel::test(test_cases = 200)]
    fn prop_no_reply_prefix_survives(tc: TestCase) {
        let subject: String = tc.draw(gs::sampled_from(vec![
            "Budget 2026".to_string(),
            "Quick question".to_string(),
            "Café plans".to_string(),
        ]));
        let prefixes: Vec<String> = tc.draw(
            gs::vecs(gs::sampled_from(vec![
                "Re: ".to_string(),
                "RE: ".to_string(),
                "Fwd: ".to_string(),
                "Fw:".to_string(),
                "Re[2]: ".to_string(),
                "[team] ".to_string(),
            ]))
            .min_size(0)
            .max_size(4),
        );

        let decorated = format!("{}{subject}", prefixes.concat());

        assert_eq!(
            normalize_subject(&decorated),
            normalize_subject(&subject),
            "{decorated:?}"
        );
    }
}
