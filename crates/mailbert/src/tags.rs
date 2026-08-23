//! The words that the `tag` command takes (§2.1).
//!
//! `mailbert tag -todo +done a3f9 b721` holds two groups of words. The
//! changes come first, and the identities of the messages come after.
//! Each change starts with `+` or with `-`, and no identity does,
//! because an identity is hexadecimal (§4.1).

use std::{collections::BTreeSet, io::Write};

use mailbert_core::{
    Store,
    index::MailIndex,
    message_id::MessageId,
    store::normalize_tag,
};
use serde::Serialize;

use crate::{
    Tool,
    cli,
    error::{Error, Result},
    show,
};

/// One change that `tag` makes to one message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Edit {
    /// Put the tag on the message.
    Add(String),

    /// Take the tag off the message.
    Drop(String),
}

impl Edit {
    /// The tag that this change touches.
    pub fn tag(&self) -> &str {
        match self {
            Self::Add(tag) | Self::Drop(tag) => tag,
        }
    }

    /// True if the change puts the tag on the message.
    pub fn adds(&self) -> bool {
        matches!(self, Self::Add(_))
    }

    /// The word that gives this change on the command line.
    pub fn word(&self) -> String {
        let mark = if self.adds() { '+' } else { '-' };

        format!("{mark}{}", self.tag())
    }
}

/// The work of one `tag` command.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Plan {
    /// The changes, in the order that the user gave them.
    pub edits: Vec<Edit>,

    /// The identities, or the prefixes of them, that the changes touch.
    pub ids: Vec<String>,
}

impl Plan {
    /// Apply each change to a set of tags, in order.
    ///
    /// The order counts: `+todo -todo` leaves the tag off, and
    /// `-todo +todo` leaves it on.
    pub fn apply(&self, tags: &mut BTreeSet<String>) {
        for edit in &self.edits {
            match edit {
                Edit::Add(tag) => {
                    tags.insert(tag.clone());
                }
                Edit::Drop(tag) => {
                    tags.remove(tag);
                }
            }
        }
    }

    /// The words that give this plan on the command line.
    pub fn words(&self) -> Vec<String> {
        let mut words: Vec<String> =
            self.edits.iter().map(Edit::word).collect();

        words.extend(self.ids.iter().cloned());

        words
    }
}

/// Read the words of a `tag` command.
///
/// # Errors
///
/// The function refuses a command with no change, a command with no
/// identity, a change that comes after an identity, and a tag that the
/// store does not accept.
pub fn split(words: &[String]) -> Result<Plan> {
    let mut plan = Plan::default();

    for word in words {
        let change = word.starts_with(['+', '-']);

        if change && !plan.ids.is_empty() {
            return Err(Error::LateEdit(word.clone()));
        }

        if !change {
            plan.ids.push(word.clone());
            continue;
        }

        let tag = normalize_tag(&word[1..])
            .ok_or_else(|| Error::bad_tag(&word[1..]))?;

        plan.edits.push(match word.starts_with('+') {
            true => Edit::Add(tag),
            false => Edit::Drop(tag),
        });
    }

    if plan.edits.is_empty() {
        return Err(Error::NoEdits);
    }

    if plan.ids.is_empty() {
        return Err(Error::NoMessages);
    }

    Ok(plan)
}

/// The tags of one message, after a `tag` command. (§10.4)
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Change {
    /// The identity of §4.1, shortened.
    pub id: String,

    /// The tags that the message carries now.
    pub tags: Vec<String>,
}

/// What one `tag` command did. (§10.4)
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Answer {
    /// One entry for each message that the command named.
    pub messages: Vec<Change>,
}

/// The budget of the writer that puts the tags back in the index.
pub const BUDGET: usize = 15_000_000;

/// What the text writes for a message that carries no tag.
pub const NONE: &str = "(no tag)";

/// Put the changes of a plan on the tags that the store keeps. (§9)
///
/// Gives the tags that the message carries after the changes.
///
/// # Errors
///
/// The function fails if the store refuses.
pub fn retag(
    store: &Store,
    id: &MessageId,
    plan: &Plan,
) -> Result<BTreeSet<String>> {
    for edit in &plan.edits {
        match edit.adds() {
            true => store.tag(id, edit.tag())?,
            false => store.untag(id, edit.tag())?,
        };
    }

    Ok(store.tags_of(id)?)
}

/// The identities that a plan names, in the order that it names them.
///
/// Every identity resolves before the first change lands, so a plan
/// that holds one bad identity changes nothing.
///
/// # Errors
///
/// The function fails if an identity names no message, or if it names
/// more than one.
pub fn targets(store: &Store, plan: &Plan) -> Result<Vec<MessageId>> {
    plan.ids
        .iter()
        .map(|prefix| show::resolve(store, prefix))
        .collect()
}

/// Put the changes of a plan on each message that it names. (§9)
///
/// The store keeps the tags, and the index reads them again, because
/// §6.1 keeps a tag in the `flags` field that `tag:` asks. A message
/// that the index does not hold keeps its tags in the store, and the
/// next sync puts them in the index.
///
/// # Errors
///
/// The function fails if an identity names no message, if it names
/// more than one, or if the store or the index refuses.
pub fn apply(store: &Store, index: &MailIndex, plan: &Plan) -> Result<Answer> {
    let ids = targets(store, plan)?;
    let mut messages = Vec::with_capacity(ids.len());
    let mut writer = index.writer(BUDGET)?;

    for id in &ids {
        let tags = retag(store, id, plan)?;

        // §3.2 writes the store before the index, so the store can
        // hold a message that the index does not. The tags stay, and
        // the next sync gives them to the index.
        if let (Some(hit), Some(message)) = (index.get(id)?, store.get(id)?) {
            index.add(&writer, &message, hit.thread, &tags)?;
        }

        messages.push(Change {
            id: id.short(),
            tags: tags.into_iter().collect(),
        });
    }

    index.commit(&mut writer)?;

    Ok(Answer { messages })
}

/// Write what the command did, one line for each message.
///
/// # Errors
///
/// The function fails if the output does not take the text.
pub fn write_text(answer: &Answer, out: &mut dyn Write) -> Result<()> {
    for change in &answer.messages {
        let tags = match change.tags.is_empty() {
            true => NONE.to_string(),
            false => change.tags.join(" "),
        };

        writeln!(out, "{}  {tags}", change.id)?;
    }

    Ok(())
}

/// Do the work of `tag`. (§9)
///
/// # Errors
///
/// The function fails if the words are bad, or if an identity names
/// no message.
pub fn command(tool: &Tool, args: &cli::Tag) -> Result<()> {
    let plan = split(&args.words)?;
    let store = tool.store()?;
    let index = tool.index()?;
    let answer = apply(&store, &index, &plan)?;

    write_text(&answer, &mut std::io::stdout().lock())
}

#[cfg(test)]
mod tests {
    //! # Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_plan_writes_the_words_that_made_it` | round-trip | The command line is the only way in. A plan that cannot write itself back holds a word that it changed silently. |
    //! | `prop_the_edits_of_a_plan_keep_their_order` | model-based | `+todo -todo` and `-todo +todo` end differently. The order is the whole meaning. |
    //! | `prop_a_plan_never_takes_an_identity_for_a_change` | invariant | An identity is hexadecimal, so it never starts with `+`. A plan that tags `a3f9` writes to the wrong place. |
    //! | `prop_a_plan_and_its_reverse_leave_the_tags_alone` | round-trip | A reader who tags by mistake takes the tag off again. The message must come back to the state that it was in. |
    //! | `prop_the_store_and_the_index_hold_the_same_tags` | model-based | `tag:todo` reads the index, and §10.1 reads the store. Two answers that disagree show a message that the reader cannot find. |

    use hegel::{TestCase, generators as gs};

    use super::*;

    fn word(text: &str) -> String {
        text.to_string()
    }

    fn words(list: &[&str]) -> Vec<String> {
        list.iter().map(|text| word(text)).collect()
    }

    #[test]
    fn one_change_and_one_identity_is_a_plan() {
        let plan = split(&words(&["+todo", "a3f9"])).expect("a good plan");

        assert_eq!(plan.edits, vec![Edit::Add("todo".to_string())]);
        assert_eq!(plan.ids, words(&["a3f9"]));
    }

    #[test]
    fn a_plan_takes_many_changes_and_many_identities() {
        let plan = split(&words(&["-todo", "+done", "a3f9", "b721"]))
            .expect("a good plan");

        assert_eq!(
            plan.edits,
            vec![
                Edit::Drop("todo".to_string()),
                Edit::Add("done".to_string()),
            ]
        );
        assert_eq!(plan.ids, words(&["a3f9", "b721"]));
    }

    #[test]
    fn a_tag_becomes_lowercase() {
        let plan = split(&words(&["+TODO", "a3f9"])).expect("a good plan");

        assert_eq!(plan.edits, vec![Edit::Add("todo".to_string())]);
    }

    #[test]
    fn a_command_with_no_change_is_an_error() {
        let result = split(&words(&["a3f9"]));

        assert!(matches!(result, Err(Error::NoEdits)), "{result:?}");
    }

    #[test]
    fn a_command_with_no_identity_is_an_error() {
        let result = split(&words(&["+todo"]));

        assert!(matches!(result, Err(Error::NoMessages)), "{result:?}");
    }

    #[test]
    fn a_command_with_no_word_at_all_is_an_error() {
        let result = split(&[]);

        assert!(matches!(result, Err(Error::NoEdits)), "{result:?}");
    }

    #[test]
    fn a_change_after_an_identity_is_an_error() {
        let result = split(&words(&["+todo", "a3f9", "+done"]));

        assert!(
            matches!(result, Err(Error::LateEdit(ref late)) if late == "+done"),
            "{result:?}"
        );
    }

    #[test]
    fn a_tag_that_the_store_refuses_is_an_error() {
        let result = split(&words(&[r"+\seen", "a3f9"]));

        assert!(
            matches!(
                result,
                Err(Error::Core(mailbert_core::Error::InvalidTag(_)))
            ),
            "{result:?}"
        );
    }

    #[test]
    fn a_change_with_no_tag_is_an_error() {
        let result = split(&words(&["+", "a3f9"]));

        assert!(
            matches!(
                result,
                Err(Error::Core(mailbert_core::Error::InvalidTag(_)))
            ),
            "{result:?}"
        );
    }

    #[test]
    fn the_same_tag_can_go_on_and_off() {
        let plan =
            split(&words(&["+todo", "-todo", "a3f9"])).expect("a good plan");
        let mut tags = BTreeSet::new();
        plan.apply(&mut tags);

        assert!(tags.is_empty(), "{tags:?}");
    }

    #[test]
    fn a_change_drops_a_tag_that_is_there() {
        let plan = split(&words(&["-todo", "a3f9"])).expect("a good plan");
        let mut tags = BTreeSet::from(["todo".to_string()]);
        plan.apply(&mut tags);

        assert!(tags.is_empty(), "{tags:?}");
    }

    #[hegel::composite]
    fn a_plan(tc: TestCase) -> Plan {
        let count: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
        let mut edits = Vec::new();

        for _ in 0..count {
            let tag: String = tc
                .draw(gs::text().alphabet("abcdefgh").min_size(1).max_size(6));
            let adds: bool = tc.draw(gs::booleans());

            edits.push(match adds {
                true => Edit::Add(tag),
                false => Edit::Drop(tag),
            });
        }

        let ids: Vec<String> = tc.draw(
            gs::vecs(
                gs::text()
                    .alphabet("0123456789abcdef")
                    .min_size(4)
                    .max_size(8),
            )
            .min_size(1)
            .max_size(3),
        );

        Plan { edits, ids }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_plan_writes_the_words_that_made_it(tc: TestCase) {
        let plan: Plan = tc.draw(a_plan());
        let again = split(&plan.words()).expect("a plan writes good words");

        assert_eq!(again, plan);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_edits_of_a_plan_keep_their_order(tc: TestCase) {
        let plan: Plan = tc.draw(a_plan());
        let mut tags = BTreeSet::new();
        plan.apply(&mut tags);

        let mut model: BTreeSet<String> = BTreeSet::new();
        for edit in &plan.edits {
            match edit.adds() {
                true => model.insert(edit.tag().to_string()),
                false => model.remove(edit.tag()),
            };
        }

        assert_eq!(tags, model);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_plan_never_takes_an_identity_for_a_change(tc: TestCase) {
        let plan: Plan = tc.draw(a_plan());
        let read = split(&plan.words()).expect("a plan writes good words");

        for id in &read.ids {
            assert!(!id.starts_with(['+', '-']), "`{id}` is not an identity");
        }

        assert_eq!(read.ids, plan.ids);
    }

    // -----------------------------------------------------------------
    // The tags that a plan writes. (§9)
    // -----------------------------------------------------------------

    use std::collections::BTreeMap;

    use mailbert_core::{
        Vocabulary,
        compile,
        date::Clock,
        message::{Location, Message},
        mime,
        query,
        threading::ThreadId,
    };
    use tempfile::{TempDir, tempdir};

    /// A moment inside the day that the test messages carry.
    const NOW: i64 = 1_755_900_000;

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

    fn bytes_of(key: &str) -> Vec<u8> {
        format!(
            "From: alice@example.test\r\n\
             To: bob@example.test\r\n\
             Subject: Deposit\r\n\
             Date: Fri, 22 Aug 2025 09:30:00 +0000\r\n\
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

        fn put(&self, key: &str) -> MessageId {
            self.write(key, true)
        }

        /// A message that the store holds and the index does not.
        fn put_unindexed(&self, key: &str) -> MessageId {
            self.write(key, false)
        }

        fn write(&self, key: &str, indexed: bool) -> MessageId {
            let bytes = bytes_of(key);
            let message = Message::new(
                mime::parse(&bytes).expect("a message"),
                location(1),
                Vec::<String>::new(),
            );
            let held = self.store.put(&message, &bytes).expect("a write");

            if indexed {
                let thread = ThreadId::from_root(held.id);
                let tags = self.store.tags_of(&held.id).expect("the tags");
                let mut writer = self.index.writer(BUDGET).expect("a writer");
                self.index
                    .add(&writer, &held, thread, &tags)
                    .expect("an index write");
                self.index.commit(&mut writer).expect("a commit");
            }

            held.id
        }

        fn run(&self, words: &[&str]) -> Result<Answer> {
            let words: Vec<String> =
                words.iter().map(|one| (*one).to_string()).collect();
            let plan = split(&words)?;

            apply(&self.store, &self.index, &plan)
        }

        /// The identities that a query finds in the index.
        fn found(&self, text: &str) -> BTreeSet<MessageId> {
            let clock = Clock::utc(NOW);
            let asked = query::parse(text, clock).expect("a query");
            let vocabulary =
                Vocabulary::from_store(&self.store).expect("the words");
            let compiled =
                compile::compile(&asked, &self.index, &vocabulary, clock)
                    .expect("a compile");

            self.index
                .top(&*compiled.search, self.index.len().max(1))
                .expect("a search")
                .into_iter()
                .map(|hit| hit.id)
                .collect()
        }

        fn tags(&self, id: &MessageId) -> BTreeSet<String> {
            self.store.tags_of(id).expect("the tags")
        }
    }

    fn answer_text(answer: &Answer) -> String {
        let mut out = Vec::new();
        write_text(answer, &mut out).expect("a write");

        String::from_utf8(out).expect("the output is text")
    }

    #[test]
    fn a_plus_puts_a_tag_on_a_message() {
        let shelf = Shelf::new();
        let id = shelf.put("a");

        shelf.run(&["+todo", &id.short()]).expect("a change");

        assert!(shelf.tags(&id).contains("todo"));
    }

    #[test]
    fn a_minus_takes_a_tag_off_a_message() {
        let shelf = Shelf::new();
        let id = shelf.put("a");
        shelf.run(&["+todo", &id.short()]).expect("a change");

        shelf.run(&["-todo", &id.short()]).expect("a change");

        assert!(!shelf.tags(&id).contains("todo"));
    }

    #[test]
    fn the_order_of_the_changes_counts() {
        let shelf = Shelf::new();
        let id = shelf.put("a");

        shelf
            .run(&["+todo", "-todo", &id.short()])
            .expect("a change");
        assert!(!shelf.tags(&id).contains("todo"));

        shelf
            .run(&["-todo", "+todo", &id.short()])
            .expect("a change");
        assert!(shelf.tags(&id).contains("todo"));
    }

    #[test]
    fn one_command_touches_each_message_that_it_names() {
        let shelf = Shelf::new();
        let first = shelf.put("a");
        let second = shelf.put("b");

        let answer = shelf
            .run(&["+todo", &first.short(), &second.short()])
            .expect("a change");

        assert_eq!(answer.messages.len(), 2);
        assert!(shelf.tags(&first).contains("todo"));
        assert!(shelf.tags(&second).contains("todo"));
    }

    /// §6.1 keeps a tag in the `flags` field, so `tag:` reads the
    /// index. A tag that stops at the store is a tag that no search
    /// finds.
    #[test]
    fn a_new_tag_reaches_the_index() {
        let shelf = Shelf::new();
        let id = shelf.put("a");

        shelf.run(&["+todo", &id.short()]).expect("a change");

        assert_eq!(shelf.found("tag:todo"), BTreeSet::from([id]));
    }

    #[test]
    fn a_tag_that_goes_away_leaves_the_index() {
        let shelf = Shelf::new();
        let id = shelf.put("a");
        shelf.run(&["+todo", &id.short()]).expect("a change");

        shelf.run(&["-todo", &id.short()]).expect("a change");

        assert!(shelf.found("tag:todo").is_empty());
    }

    #[test]
    fn a_command_takes_a_prefix_of_an_identity() {
        let shelf = Shelf::new();
        let id = shelf.put("a");
        let prefix = &id.full_hex()[..4];

        shelf.run(&["+todo", prefix]).expect("a change");

        assert!(shelf.tags(&id).contains("todo"));
    }

    #[test]
    fn a_prefix_that_names_nothing_is_an_error() {
        let shelf = Shelf::new();
        shelf.put("a");

        let result = shelf.run(&["+todo", "ffffffffffffffff"]);

        assert!(
            matches!(
                result,
                Err(Error::Core(mailbert_core::Error::UnknownMessage(_)))
            ),
            "{result:?}"
        );
    }

    /// A plan that holds one bad identity must change nothing, so the
    /// reader can give the whole command again.
    #[test]
    fn a_bad_identity_leaves_every_message_as_it_was() {
        let shelf = Shelf::new();
        let id = shelf.put("a");

        let result = shelf.run(&["+todo", &id.short(), "ffffffffffffffff"]);

        assert!(result.is_err(), "{result:?}");
        assert!(!shelf.tags(&id).contains("todo"));
    }

    /// §3.2 writes the store before the index, so a message can be in
    /// one and not the other. The tag must land, and the next sync
    /// gives it to the index.
    #[test]
    fn a_message_that_the_index_lost_still_takes_a_tag() {
        let shelf = Shelf::new();
        let id = shelf.put_unindexed("a");

        shelf.run(&["+todo", &id.short()]).expect("a change");

        assert!(shelf.tags(&id).contains("todo"));
    }

    #[test]
    fn the_answer_names_each_message_and_the_tags_it_carries() {
        let shelf = Shelf::new();
        let id = shelf.put("a");

        let answer = shelf
            .run(&["+todo", "+later", &id.short()])
            .expect("a change");

        assert_eq!(answer.messages.len(), 1);
        assert_eq!(answer.messages[0].id, id.short());
        assert_eq!(
            answer.messages[0].tags,
            vec!["later".to_string(), "todo".to_string()]
        );
    }

    #[test]
    fn the_text_writes_one_line_for_each_message() {
        let shelf = Shelf::new();
        let first = shelf.put("a");
        let second = shelf.put("b");
        let answer = shelf
            .run(&["+todo", &first.short(), &second.short()])
            .expect("a change");

        let held = answer_text(&answer);

        assert_eq!(held.lines().count(), 2, "{held}");
        assert!(held.contains(&first.short()), "{held}");
        assert!(held.contains("todo"), "{held}");
    }

    #[test]
    fn the_text_of_a_message_with_no_tag_says_so() {
        let shelf = Shelf::new();
        let id = shelf.put("a");
        shelf.run(&["+todo", &id.short()]).expect("a change");
        let answer = shelf.run(&["-todo", &id.short()]).expect("a change");

        let held = answer_text(&answer);

        assert!(held.contains("no tag"), "{held}");
    }

    // -----------------------------------------------------------------
    // Properties of the changes.
    // -----------------------------------------------------------------

    #[hegel::composite]
    fn some_tags(tc: TestCase) -> Vec<String> {
        tc.draw(
            gs::vecs(gs::sampled_from(vec![
                "todo".to_string(),
                "later".to_string(),
                "work".to_string(),
            ]))
            .min_size(1)
            .max_size(3),
        )
    }

    #[hegel::test(test_cases = 25)]
    fn prop_a_plan_and_its_reverse_leave_the_tags_alone(tc: TestCase) {
        let tags = tc.draw(some_tags());
        let shelf = Shelf::new();
        let id = shelf.put("a");
        let before = shelf.tags(&id);

        let mut on: Vec<String> =
            tags.iter().map(|tag| format!("+{tag}")).collect();
        on.push(id.short());
        shelf
            .run(&on.iter().map(String::as_str).collect::<Vec<&str>>())
            .expect("a change");

        let mut off: Vec<String> =
            tags.iter().map(|tag| format!("-{tag}")).collect();
        off.push(id.short());
        shelf
            .run(&off.iter().map(String::as_str).collect::<Vec<&str>>())
            .expect("a change");

        assert_eq!(shelf.tags(&id), before);
    }

    #[hegel::test(test_cases = 25)]
    fn prop_the_store_and_the_index_hold_the_same_tags(tc: TestCase) {
        let tags = tc.draw(some_tags());
        let shelf = Shelf::new();
        let id = shelf.put("a");

        let mut words: Vec<String> = tags
            .iter()
            .map(|tag| match tc.draw(gs::booleans()) {
                true => format!("+{tag}"),
                false => format!("-{tag}"),
            })
            .collect();
        words.push(id.short());
        shelf
            .run(&words.iter().map(String::as_str).collect::<Vec<&str>>())
            .expect("a change");

        let held = shelf.tags(&id);
        let mut model: BTreeMap<String, bool> = BTreeMap::new();
        for tag in ["todo", "later", "work"] {
            model.insert(tag.to_string(), held.contains(tag));
        }

        for (tag, on) in model {
            let found = shelf.found(&format!("tag:{tag}"));

            assert_eq!(found.contains(&id), on, "the index lost `{tag}`");
        }
    }
}
