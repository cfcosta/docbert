//! The contact book that `from:` and `to:` resolve against.
//!
//! During sync, mailbert records each address it sees, each display
//! name for that address, and how frequently you and that address
//! correspond. At query time, `from:caina` becomes a set of addresses,
//! ordered by that frequency, and the filter that reaches the index is
//! an exact match on that set.
//!
//! This is better than fuzzy matching on a text field. The reader can
//! see what the expansion did, and `from:sam` cannot quietly include
//! `samsung`, because a needle matches a whole word and never a prefix
//! of one.
//!
//! See `docs/mailbert.md` §5.6.

use std::collections::HashMap;

use crate::address::{Address, fold, words};

/// How an address appeared on one message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Seen {
    /// The address sent a message that reached you.
    Inbound,

    /// You sent a message that reached the address.
    Outbound,
}

/// What mailbert knows about one address.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Contact {
    address: String,
    names: HashMap<String, u64>,
    inbound: u64,
    outbound: u64,
}

/// Every address that mailbert has seen.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Contacts {
    by_address: HashMap<String, Contact>,
}

impl Contact {
    /// The address, lowercased.
    pub fn address(&self) -> &str {
        &self.address
    }

    /// The part before the `@`.
    fn local_part(&self) -> &str {
        self.address
            .split_once('@')
            .map_or(self.address.as_str(), |(local, _)| local)
    }

    /// The part after the `@`.
    pub fn domain(&self) -> &str {
        self.address
            .split_once('@')
            .map_or("", |(_, domain)| domain)
    }

    /// The display names for this address, most frequent first.
    ///
    /// A person renames themself, so the name they use most is the one
    /// to show. Equal counts fall back to alphabetical order, to keep
    /// the answer the same between two runs.
    pub fn names(&self) -> Vec<&str> {
        let mut ranked: Vec<(&str, u64)> = self
            .names
            .iter()
            .map(|(name, count)| (name.as_str(), *count))
            .collect();

        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        ranked.into_iter().map(|(name, _)| name).collect()
    }

    /// The name to show. `None` when no header ever gave one.
    pub fn primary_name(&self) -> Option<&str> {
        self.names().into_iter().next()
    }

    /// Messages this address sent that reached you.
    pub fn inbound(&self) -> u64 {
        self.inbound
    }

    /// Messages you sent that reached this address.
    pub fn outbound(&self) -> u64 {
        self.outbound
    }

    /// How frequently you and this address correspond.
    pub fn correspondence(&self) -> u64 {
        self.inbound.saturating_add(self.outbound)
    }

    /// Whether `needle` names this contact.
    ///
    /// A needle with an `@` is an exact address. A needle that starts
    /// with `@` is a domain. Any other needle must equal a whole word
    /// of the local part or of a display name, and never a prefix of
    /// one, so `sam` cannot reach `samsung`.
    pub fn matches(&self, needle: &str) -> bool {
        let needle = fold(needle.trim());
        if needle.is_empty() {
            return false;
        }

        if let Some(domain) = needle.strip_prefix('@') {
            return fold(self.domain()) == domain;
        }

        if needle.contains('@') {
            return fold(&self.address) == needle;
        }

        self.terms().any(|term| term == needle)
    }

    /// Every whole word that names this contact.
    ///
    /// The domain is not among them, or `from:example` would match
    /// each contact at `example.com`.
    fn terms(&self) -> impl Iterator<Item = String> + '_ {
        let local = self.local_part();

        std::iter::once(fold(local)).chain(words(local)).chain(
            self.names.keys().flat_map(|name| {
                std::iter::once(fold(name)).chain(words(name))
            }),
        )
    }
}

impl Contacts {
    /// An empty contact book.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many addresses the book holds.
    pub fn len(&self) -> usize {
        self.by_address.len()
    }

    /// Whether the book holds no address.
    pub fn is_empty(&self) -> bool {
        self.by_address.is_empty()
    }

    /// Record one appearance of `address`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mailbert_core::{Contacts, Seen, address};
    ///
    /// let mut contacts = Contacts::new();
    /// let alice = address::parse("Cainã Costa <me@cfcosta.com>").unwrap();
    /// contacts.record(&alice, Seen::Inbound);
    ///
    /// assert_eq!(contacts.resolve_addresses("caina"), ["me@cfcosta.com"]);
    /// ```
    pub fn record(&mut self, address: &Address, seen: Seen) {
        let contact = self
            .by_address
            .entry(address.address.clone())
            .or_insert_with(|| Contact {
                address: address.address.clone(),
                ..Contact::default()
            });

        match seen {
            Seen::Inbound => {
                contact.inbound = contact.inbound.saturating_add(1);
            }
            Seen::Outbound => {
                contact.outbound = contact.outbound.saturating_add(1);
            }
        }

        if let Some(name) = &address.name {
            let count = contact.names.entry(name.clone()).or_default();
            *count = count.saturating_add(1);
        }
    }

    /// The contact for an exact address.
    pub fn get(&self, address: &str) -> Option<&Contact> {
        self.by_address.get(&address.trim().to_lowercase())
    }

    /// Every contact that `needle` names, most frequent first.
    ///
    /// The order is total: correspondence, then the messages you sent,
    /// then the address. Two books built from the same appearances in
    /// a different order therefore resolve identically. Ranking the
    /// people you write to above the ones who only write to you keeps
    /// a mailing list below a colleague of the same name.
    pub fn resolve(&self, needle: &str) -> Vec<&Contact> {
        let mut found: Vec<&Contact> = self
            .by_address
            .values()
            .filter(|contact| contact.matches(needle))
            .collect();

        found.sort_by(|a, b| {
            b.correspondence()
                .cmp(&a.correspondence())
                .then_with(|| b.outbound().cmp(&a.outbound()))
                .then_with(|| a.address().cmp(b.address()))
        });

        found
    }

    /// The addresses that `needle` names, most frequent first.
    ///
    /// This is the set that reaches the index, as an exact match on a
    /// `STRING` field. The reader can print it, so the expansion is
    /// never a surprise.
    pub fn resolve_addresses(&self, needle: &str) -> Vec<String> {
        self.resolve(needle)
            .into_iter()
            .map(|contact| contact.address.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_recording_is_commutative` | algebraic | Sync fetches folders in parallel, so the order of appearances is not fixed. The book must not depend on it. |
    //! | `prop_resolution_ignores_insertion_order` | differential | The same reason, applied to the answer the reader sees. A `HashMap` alone would fail this. |
    //! | `prop_an_exact_address_resolves_to_itself` | invariant | `from:alice@x.example` must always find Alice, whatever her display names were. |
    //! | `prop_a_needle_never_matches_a_longer_word` | invariant | This is the whole claim of §5.6: `from:sam` cannot quietly include `samsung`. |
    //! | `prop_resolution_is_ordered_by_correspondence` | invariant | The reader reads the first few lines. If the order were wrong, the useful address would be below the fold. |
    //! | `prop_correspondence_counts_every_appearance` | model-based | The counters drive the order. A lost count is a wrong order. |
    //! | `prop_names_are_ranked_by_frequency` | invariant | A person renames themself. The name they use most is the one to show. |
    //! | `prop_resolution_is_a_subset_of_the_book` | invariant | Resolution must never invent an address that no message carried. |

    use hegel::{TestCase, generators as gs};

    use super::*;

    // -----------------------------------------------------------------
    // Generators.
    // -----------------------------------------------------------------

    /// A pool small enough that the same address appears repeatedly.
    fn appearance() -> impl gs::Generator<(String, Seen)> {
        gs::sampled_from(vec![
            ("Alice Example <alice@x.example>".to_string(), Seen::Inbound),
            ("Alice <alice@x.example>".to_string(), Seen::Outbound),
            ("Cainã Costa <me@cfcosta.com>".to_string(), Seen::Inbound),
            ("<me@cfcosta.com>".to_string(), Seen::Outbound),
            ("Sam Smith <sam@x.example>".to_string(), Seen::Inbound),
            ("Samsung <samsung@x.example>".to_string(), Seen::Inbound),
            ("bob.smith@work.example".to_string(), Seen::Outbound),
        ])
    }

    #[hegel::composite]
    fn appearances(tc: TestCase) -> Vec<(String, Seen)> {
        tc.draw(gs::vecs(appearance()).min_size(0).max_size(20))
    }

    fn book_of(appearances: &[(String, Seen)]) -> Contacts {
        let mut contacts = Contacts::new();

        for (raw, seen) in appearances {
            let address = crate::address::parse(raw).expect("the pool parses");
            contacts.record(&address, *seen);
        }

        contacts
    }

    fn record(contacts: &mut Contacts, raw: &str, seen: Seen) {
        let address = crate::address::parse(raw).expect("a valid address");
        contacts.record(&address, seen);
    }

    // -----------------------------------------------------------------
    // Unit tests.
    // -----------------------------------------------------------------

    #[test]
    fn an_empty_book_resolves_to_nothing() {
        let contacts = Contacts::new();

        assert!(contacts.is_empty());
        assert_eq!(contacts.len(), 0);
        assert!(contacts.resolve("alice").is_empty());
        assert!(contacts.get("alice@x.example").is_none());
    }

    #[test]
    fn one_address_collects_each_of_its_names() {
        let mut contacts = Contacts::new();
        record(
            &mut contacts,
            "Alice Example <alice@x.example>",
            Seen::Inbound,
        );
        record(&mut contacts, "Alice <alice@x.example>", Seen::Inbound);
        record(&mut contacts, "Alice <alice@x.example>", Seen::Outbound);

        assert_eq!(contacts.len(), 1);

        let contact = contacts.get("alice@x.example").unwrap();
        assert_eq!(contact.inbound(), 2);
        assert_eq!(contact.outbound(), 1);
        assert_eq!(contact.correspondence(), 3);
        // "Alice" was used twice and "Alice Example" once.
        assert_eq!(contact.names(), vec!["Alice", "Alice Example"]);
        assert_eq!(contact.primary_name(), Some("Alice"));
    }

    #[test]
    fn a_nameless_address_has_no_primary_name() {
        let mut contacts = Contacts::new();
        record(&mut contacts, "alice@x.example", Seen::Inbound);

        let contact = contacts.get("alice@x.example").unwrap();
        assert!(contact.names().is_empty());
        assert_eq!(contact.primary_name(), None);
    }

    #[test]
    fn a_needle_matches_a_word_of_the_name_without_its_accents() {
        let mut contacts = Contacts::new();
        record(&mut contacts, "Cainã Costa <me@cfcosta.com>", Seen::Inbound);

        assert_eq!(contacts.resolve_addresses("caina"), vec!["me@cfcosta.com"]);
        assert_eq!(contacts.resolve_addresses("Cainã"), vec!["me@cfcosta.com"]);
        assert_eq!(contacts.resolve_addresses("costa"), vec!["me@cfcosta.com"]);
    }

    #[test]
    fn a_needle_never_reaches_a_longer_word() {
        let mut contacts = Contacts::new();
        record(&mut contacts, "Sam Smith <sam@x.example>", Seen::Inbound);
        record(&mut contacts, "Samsung <samsung@x.example>", Seen::Inbound);

        // This is the claim of §5.6. `sam` must not drag in `samsung`.
        assert_eq!(contacts.resolve_addresses("sam"), vec!["sam@x.example"]);
        assert_eq!(
            contacts.resolve_addresses("samsung"),
            vec!["samsung@x.example"]
        );
    }

    #[test]
    fn a_needle_matches_a_word_of_the_local_part() {
        let mut contacts = Contacts::new();
        record(&mut contacts, "bob.smith@work.example", Seen::Outbound);

        assert_eq!(
            contacts.resolve_addresses("bob"),
            vec!["bob.smith@work.example"]
        );
        assert_eq!(
            contacts.resolve_addresses("smith"),
            vec!["bob.smith@work.example"]
        );
        // The whole local part is a needle of its own.
        assert_eq!(
            contacts.resolve_addresses("bob.smith"),
            vec!["bob.smith@work.example"]
        );
    }

    #[test]
    fn an_at_sign_makes_the_needle_exact() {
        let mut contacts = Contacts::new();
        record(&mut contacts, "Alice <alice@x.example>", Seen::Inbound);
        record(&mut contacts, "Alice <alice@y.example>", Seen::Inbound);

        assert_eq!(
            contacts.resolve_addresses("alice@x.example"),
            vec!["alice@x.example"]
        );
        assert_eq!(contacts.resolve_addresses("alice").len(), 2);
    }

    #[test]
    fn a_leading_at_sign_makes_the_needle_a_domain() {
        let mut contacts = Contacts::new();
        record(&mut contacts, "Alice <alice@x.example>", Seen::Inbound);
        record(&mut contacts, "Bob <bob@x.example>", Seen::Inbound);
        record(&mut contacts, "Carol <carol@y.example>", Seen::Inbound);

        let mut found = contacts.resolve_addresses("@x.example");
        found.sort();

        assert_eq!(found, vec!["alice@x.example", "bob@x.example"]);
    }

    #[test]
    fn the_domain_is_never_a_bare_word() {
        let mut contacts = Contacts::new();
        record(&mut contacts, "Alice <alice@x.example>", Seen::Inbound);

        // Otherwise `from:example` would match everyone.
        assert!(contacts.resolve_addresses("example").is_empty());
        assert!(contacts.resolve_addresses("x").is_empty());
    }

    #[test]
    fn resolution_puts_the_frequent_correspondent_first() {
        let mut contacts = Contacts::new();
        record(
            &mut contacts,
            "Alice Rare <alice@rare.example>",
            Seen::Inbound,
        );
        for _ in 0..5 {
            record(
                &mut contacts,
                "Alice Often <alice@often.example>",
                Seen::Outbound,
            );
        }

        assert_eq!(
            contacts.resolve_addresses("alice"),
            vec!["alice@often.example", "alice@rare.example"]
        );
    }

    #[test]
    fn a_blank_needle_matches_nothing() {
        let mut contacts = Contacts::new();
        record(&mut contacts, "Alice <alice@x.example>", Seen::Inbound);

        assert!(contacts.resolve("").is_empty());
        assert!(contacts.resolve("   ").is_empty());
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 200)]
    fn prop_recording_is_commutative(tc: TestCase) {
        let seen: Vec<(String, Seen)> = tc.draw(appearances());

        let forward = book_of(&seen);
        let backward = book_of(&seen.iter().rev().cloned().collect::<Vec<_>>());

        assert_eq!(forward, backward);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_resolution_ignores_insertion_order(tc: TestCase) {
        let seen: Vec<(String, Seen)> = tc.draw(appearances());
        let needle: String = tc.draw(gs::sampled_from(vec![
            "alice".to_string(),
            "sam".to_string(),
            "caina".to_string(),
            "smith".to_string(),
            "@x.example".to_string(),
        ]));

        let forward = book_of(&seen);
        let backward = book_of(&seen.iter().rev().cloned().collect::<Vec<_>>());

        assert_eq!(
            forward.resolve_addresses(&needle),
            backward.resolve_addresses(&needle),
            "needle {needle:?}"
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_an_exact_address_resolves_to_itself(tc: TestCase) {
        let seen: Vec<(String, Seen)> = tc.draw(appearances());
        tc.assume(!seen.is_empty());

        let contacts = book_of(&seen);

        for address in seen
            .iter()
            .filter_map(|(raw, _)| crate::address::parse(raw))
        {
            assert_eq!(
                contacts.resolve_addresses(&address.address),
                vec![address.address.clone()],
                "{} did not resolve to itself",
                address.address
            );
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_needle_never_matches_a_longer_word(tc: TestCase) {
        let (short, long): (String, String) = tc.draw(gs::sampled_from(vec![
            ("sam".to_string(), "samsung".to_string()),
            ("bob".to_string(), "bobcat".to_string()),
            ("ana".to_string(), "anaconda".to_string()),
        ]));

        let mut contacts = Contacts::new();
        record(&mut contacts, &format!("{long}@x.example"), Seen::Inbound);

        assert!(
            contacts.resolve_addresses(&short).is_empty(),
            "{short} reached {long}"
        );
        assert_eq!(
            contacts.resolve_addresses(&long),
            vec![format!("{long}@x.example")]
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_resolution_is_ordered_by_correspondence(tc: TestCase) {
        let seen: Vec<(String, Seen)> = tc.draw(appearances());
        let needle: String = tc.draw(gs::sampled_from(vec![
            "alice".to_string(),
            "sam".to_string(),
            "caina".to_string(),
            "@x.example".to_string(),
        ]));

        let contacts = book_of(&seen);
        let found = contacts.resolve(&needle);

        for pair in found.windows(2) {
            assert!(
                pair[0].correspondence() >= pair[1].correspondence(),
                "{:?} outranks {:?}",
                pair[1].address(),
                pair[0].address()
            );
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_correspondence_counts_every_appearance(tc: TestCase) {
        let seen: Vec<(String, Seen)> = tc.draw(appearances());

        let contacts = book_of(&seen);

        let total: u64 = contacts
            .resolve("@x.example")
            .iter()
            .map(|c| c.correspondence())
            .sum();
        let expected = seen
            .iter()
            .filter_map(|(raw, _)| crate::address::parse(raw))
            .filter(|a| a.domain() == "x.example")
            .count() as u64;

        assert_eq!(total, expected);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_names_are_ranked_by_frequency(tc: TestCase) {
        let seen: Vec<(String, Seen)> = tc.draw(appearances());

        let contacts = book_of(&seen);

        for address in ["alice@x.example", "me@cfcosta.com"] {
            let Some(contact) = contacts.get(address) else {
                continue;
            };

            let counts: Vec<u64> = contact
                .names()
                .iter()
                .map(|name| {
                    seen.iter()
                        .filter_map(|(raw, _)| crate::address::parse(raw))
                        .filter(|a| {
                            a.address == address
                                && a.name.as_deref() == Some(*name)
                        })
                        .count() as u64
                })
                .collect();

            for pair in counts.windows(2) {
                assert!(pair[0] >= pair[1], "names out of order: {counts:?}");
            }
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_resolution_is_a_subset_of_the_book(tc: TestCase) {
        let seen: Vec<(String, Seen)> = tc.draw(appearances());
        let needle: String = tc.draw(gs::sampled_from(vec![
            "alice".to_string(),
            "sam".to_string(),
            "smith".to_string(),
            "nobody".to_string(),
        ]));

        let contacts = book_of(&seen);

        for address in contacts.resolve_addresses(&needle) {
            assert!(
                contacts.get(&address).is_some(),
                "{address} is not in the book"
            );
        }
    }
}
