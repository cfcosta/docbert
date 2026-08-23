//! The words that the `tag` command takes (§2.1).
//!
//! `mailbert tag -todo +done a3f9 b721` holds two groups of words. The
//! changes come first, and the identities of the messages come after.
//! Each change starts with `+` or with `-`, and no identity does,
//! because an identity is hexadecimal (§4.1).

use std::collections::BTreeSet;

use mailbert_core::store::normalize_tag;

use crate::error::{Error, Result};

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

#[cfg(test)]
mod tests {
    //! # Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_plan_writes_the_words_that_made_it` | round-trip | The command line is the only way in. A plan that cannot write itself back holds a word that it changed silently. |
    //! | `prop_the_edits_of_a_plan_keep_their_order` | model-based | `+todo -todo` and `-todo +todo` end differently. The order is the whole meaning. |
    //! | `prop_a_plan_never_takes_an_identity_for_a_change` | invariant | An identity is hexadecimal, so it never starts with `+`. A plan that tags `a3f9` writes to the wrong place. |

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
}
