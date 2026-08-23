//! The contact book of §5.6.
//!
//! The book holds each address that the store has seen, each display
//! name for that address, and how often you and that address write to
//! each other. `mailbert contacts caina` shows what a name resolves
//! to, so `from:caina` is never a surprise.
//!
//! §5.6 says that sync records the addresses. This build reads them
//! from the store instead, because §4.2 already keeps every message
//! there. The book therefore costs one pass over the store, and it
//! never falls behind the mail that the store holds.

use std::{collections::BTreeSet, io::Write};

use mailbert_core::{Contacts, Seen, Store, config::Config};
use serde::Serialize;

use crate::{Tool, cli, error::Result};

/// What the book knows about one address. (§10.4)
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Person {
    /// The address, lowercased.
    pub address: String,

    /// Each display name that a header carried, most frequent first.
    pub names: Vec<String>,

    /// The messages that this address sent to you.
    pub inbound: u64,

    /// The messages that you sent to this address.
    pub outbound: u64,
}

/// What one `contacts` command found. (§10.4)
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Answer {
    /// The name that the reader gave.
    pub name: String,

    /// The addresses that the name resolves to, most frequent first.
    pub people: Vec<Person>,
}

/// The addresses that belong to you.
///
/// The `user` of an account is its IMAP login. A login that holds an
/// `@` is an address, and a login that does not is a bare name that
/// says nothing about direction.
pub fn mine(config: &Config) -> BTreeSet<String> {
    config
        .accounts
        .iter()
        .map(|account| account.user.trim().to_lowercase())
        .filter(|user| user.contains('@'))
        .collect()
}

/// Build the contact book from the store. (§5.6)
///
/// A message that one of your addresses sent counts as outbound for
/// every address on it. Every other message counts as inbound. One
/// rule keeps the direction the same for `to:` and for `cc:`.
///
/// # Errors
///
/// The function fails if the store refuses.
pub fn book(store: &Store, mine: &BTreeSet<String>) -> Result<Contacts> {
    let mut found = Contacts::new();

    for message in store.all()? {
        let sent = message
            .from
            .iter()
            .any(|address| mine.contains(&address.address));
        let seen = match sent {
            true => Seen::Outbound,
            false => Seen::Inbound,
        };

        for address in message.from.iter().chain(&message.to).chain(&message.cc)
        {
            found.record(address, seen);
        }
    }

    Ok(found)
}

/// The addresses that a name resolves to, most frequent first. (§5.6)
pub fn find(book: &Contacts, name: &str) -> Answer {
    let people = book
        .resolve(name)
        .into_iter()
        .map(|contact| Person {
            address: contact.address().to_string(),
            names: contact.names().into_iter().map(str::to_string).collect(),
            inbound: contact.inbound(),
            outbound: contact.outbound(),
        })
        .collect();

    Answer {
        name: name.to_string(),
        people,
    }
}

/// Write one line for each address.
///
/// # Errors
///
/// The function fails if the output does not take the text.
pub fn write_text(answer: &Answer, out: &mut dyn Write) -> Result<()> {
    let width = answer
        .people
        .iter()
        .map(|person| person.address.chars().count())
        .max()
        .unwrap_or(0);

    for person in &answer.people {
        let name = person.names.first().map_or(NONE, String::as_str);

        writeln!(
            out,
            "{:width$}  {name}  ({} in, {} out)",
            person.address, person.inbound, person.outbound
        )?;
    }

    Ok(())
}

/// Write the addresses as the JSON of §10.4.
///
/// # Errors
///
/// The function fails if the output does not take the text.
pub fn write_json(answer: &Answer, out: &mut dyn Write) -> Result<()> {
    writeln!(out, "{}", serde_json::to_string_pretty(answer)?)?;

    Ok(())
}

/// Do the work of `contacts`. (§5.6)
///
/// # Errors
///
/// The function fails if the store or the configuration refuses.
pub fn command(tool: &Tool, args: &cli::Contacts) -> Result<()> {
    let store = tool.store()?;
    let book = book(&store, &mine(&tool.config()?))?;
    let answer = find(&book, &args.name);
    let mut out = std::io::stdout().lock();

    match args.json {
        true => write_json(&answer, &mut out),
        false => write_text(&answer, &mut out),
    }
}

/// What the text writes for an address that carries no display name.
pub const NONE: &str = "(no name)";

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_the_book_counts_every_message` | model-based | A count against a tally. §5.6 orders `from:` by frequency, so a count that drifts puts the wrong address first. |
    //! | `prop_the_order_of_the_store_never_counts` | algebraic | The same messages in another order. Sync writes folders in parallel, so the book must not depend on the order of the store. |

    use std::collections::BTreeMap;

    use hegel::{TestCase, generators as gs};
    use mailbert_core::{
        config::Account,
        message::{Location, Message},
        mime,
    };
    use tempfile::{TempDir, tempdir};

    use super::*;

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

    struct Shelf {
        _dir: TempDir,
        store: Store,
        uid: std::cell::Cell<u32>,
    }

    impl Shelf {
        fn new() -> Self {
            let dir = tempdir().expect("a temporary directory");
            let store =
                Store::open(&dir.path().join("store")).expect("a store");

            Self {
                _dir: dir,
                store,
                uid: std::cell::Cell::new(1),
            }
        }

        /// Put one message in the store, from `from` to `to`.
        fn put(&self, key: &str, from: &str, to: &str) {
            let uid = self.uid.get();
            self.uid.set(uid + 1);
            let bytes = format!(
                "From: {from}\r\n\
                 To: {to}\r\n\
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

            self.store.put(&message, &bytes).expect("a write");
        }

        fn book(&self, mine: &[&str]) -> Contacts {
            let mine: BTreeSet<String> =
                mine.iter().map(|one| (*one).to_string()).collect();

            super::book(&self.store, &mine).expect("a book")
        }
    }

    fn text_of(answer: &Answer) -> String {
        let mut out = Vec::new();
        write_text(answer, &mut out).expect("a write");

        String::from_utf8(out).expect("the output is text")
    }

    fn config_of(users: &[&str]) -> Config {
        let accounts = users
            .iter()
            .enumerate()
            .map(|(at, user)| Account {
                name: format!("a{at}"),
                host: "mail.example.test".to_string(),
                user: (*user).to_string(),
                port: 993,
                password_command: None,
                password_file: None,
                password: Some("secret".to_string()),
                folders: Vec::new(),
                exclude: Vec::new(),
                footers: Vec::new(),
                all_folders: true,
                connections: 1,
            })
            .collect();

        Config {
            accounts,
            ..Config::default()
        }
    }

    // -----------------------------------------------------------------
    // The addresses that belong to you.
    // -----------------------------------------------------------------

    #[test]
    fn a_login_that_is_an_address_belongs_to_you() {
        let config = config_of(&["me@cfcosta.com"]);

        assert_eq!(
            mine(&config),
            BTreeSet::from(["me@cfcosta.com".to_string()])
        );
    }

    #[test]
    fn a_login_that_is_a_bare_name_is_not_an_address() {
        let config = config_of(&["caina"]);

        assert!(mine(&config).is_empty());
    }

    #[test]
    fn a_login_becomes_lowercase() {
        let config = config_of(&["Me@CFCosta.COM"]);

        assert_eq!(
            mine(&config),
            BTreeSet::from(["me@cfcosta.com".to_string()])
        );
    }

    #[test]
    fn every_account_gives_its_address() {
        let config = config_of(&["me@cfcosta.com", "work@example.test"]);

        assert_eq!(mine(&config).len(), 2);
    }

    // -----------------------------------------------------------------
    // The book. (§5.6)
    // -----------------------------------------------------------------

    #[test]
    fn the_book_holds_every_address_that_a_message_carries() {
        let shelf = Shelf::new();
        shelf.put("a", "alice@example.test", "bob@example.test");

        let book = shelf.book(&[]);

        assert!(book.get("alice@example.test").is_some());
        assert!(book.get("bob@example.test").is_some());
    }

    #[test]
    fn a_message_that_you_did_not_send_counts_as_inbound() {
        let shelf = Shelf::new();
        shelf.put("a", "alice@example.test", "me@cfcosta.com");

        let book = shelf.book(&["me@cfcosta.com"]);
        let alice = book.get("alice@example.test").expect("a contact");

        assert_eq!(alice.inbound(), 1);
        assert_eq!(alice.outbound(), 0);
    }

    #[test]
    fn a_message_that_you_sent_counts_as_outbound() {
        let shelf = Shelf::new();
        shelf.put("a", "me@cfcosta.com", "alice@example.test");

        let book = shelf.book(&["me@cfcosta.com"]);
        let alice = book.get("alice@example.test").expect("a contact");

        assert_eq!(alice.outbound(), 1);
        assert_eq!(alice.inbound(), 0);
    }

    /// §5.6 ranks the people that you write to above the ones who only
    /// write to you. A build that knows no address of yours cannot do
    /// that, and every message must then count as inbound.
    #[test]
    fn a_build_that_knows_no_address_of_yours_counts_inbound() {
        let shelf = Shelf::new();
        shelf.put("a", "me@cfcosta.com", "alice@example.test");

        let book = shelf.book(&[]);
        let alice = book.get("alice@example.test").expect("a contact");

        assert_eq!(alice.inbound(), 1);
    }

    #[test]
    fn the_book_keeps_the_display_name_of_a_header() {
        let shelf = Shelf::new();
        shelf.put(
            "a",
            "Alice Alvarez <alice@example.test>",
            "bob@example.test",
        );

        let book = shelf.book(&[]);
        let alice = book.get("alice@example.test").expect("a contact");

        assert_eq!(alice.primary_name(), Some("Alice Alvarez"));
    }

    #[test]
    fn the_book_of_an_empty_store_holds_nothing() {
        let shelf = Shelf::new();

        assert!(shelf.book(&[]).is_empty());
    }

    // -----------------------------------------------------------------
    // The answer. (§5.6, §10.4)
    // -----------------------------------------------------------------

    #[test]
    fn a_name_resolves_to_the_addresses_that_carry_it() {
        let shelf = Shelf::new();
        shelf.put(
            "a",
            "Alice Alvarez <alice@example.test>",
            "bob@example.test",
        );

        let answer = find(&shelf.book(&[]), "alvarez");

        assert_eq!(answer.name, "alvarez");
        assert_eq!(answer.people.len(), 1);
        assert_eq!(answer.people[0].address, "alice@example.test");
    }

    /// §5.6 orders the set by frequency, so the reader sees the address
    /// that a query reaches first.
    #[test]
    fn the_address_you_write_to_most_comes_first() {
        let shelf = Shelf::new();
        shelf.put("a", "Alice One <one@example.test>", "me@cfcosta.com");
        shelf.put("b", "Alice Two <two@example.test>", "me@cfcosta.com");
        shelf.put("c", "Alice Two <two@example.test>", "me@cfcosta.com");

        let answer = find(&shelf.book(&["me@cfcosta.com"]), "alice");

        assert_eq!(answer.people[0].address, "two@example.test");
        assert_eq!(answer.people[0].inbound, 2);
    }

    #[test]
    fn a_name_that_nobody_carries_gives_no_address() {
        let shelf = Shelf::new();
        shelf.put("a", "alice@example.test", "bob@example.test");

        let answer = find(&shelf.book(&[]), "zebedee");

        assert!(answer.people.is_empty());
    }

    #[test]
    fn the_answer_carries_every_name_of_an_address() {
        let shelf = Shelf::new();
        shelf.put("a", "Alice A <alice@example.test>", "bob@example.test");
        shelf.put("b", "Alice B <alice@example.test>", "bob@example.test");

        let answer = find(&shelf.book(&[]), "alice@example.test");

        assert_eq!(answer.people[0].names.len(), 2);
    }

    #[test]
    fn the_text_writes_one_line_for_each_address() {
        let shelf = Shelf::new();
        shelf.put("a", "Alice One <one@example.test>", "me@cfcosta.com");
        shelf.put("b", "Alice Two <two@example.test>", "me@cfcosta.com");

        let held = text_of(&find(&shelf.book(&[]), "alice"));

        assert_eq!(held.lines().count(), 2, "{held}");
        assert!(held.contains("one@example.test"), "{held}");
        assert!(held.contains("Alice Two"), "{held}");
    }

    /// §5.6 ranks by how often you write to each other, so the text
    /// must show the numbers that put one address above another.
    #[test]
    fn the_text_shows_how_often_you_write_to_each_other() {
        let shelf = Shelf::new();
        shelf.put("a", "Alice A <alice@example.test>", "me@cfcosta.com");
        shelf.put("b", "me@cfcosta.com", "Alice A <alice@example.test>");

        let held = text_of(&find(&shelf.book(&["me@cfcosta.com"]), "alice"));

        assert!(held.contains("1 in"), "{held}");
        assert!(held.contains("1 out"), "{held}");
    }

    #[test]
    fn the_text_of_an_address_with_no_name_says_so() {
        let shelf = Shelf::new();
        shelf.put("a", "alice@example.test", "bob@example.test");

        let held = text_of(&find(&shelf.book(&[]), "alice"));

        assert!(held.contains(NONE), "{held}");
    }

    #[test]
    fn the_text_of_no_address_is_empty() {
        let answer = Answer {
            name: "zebedee".to_string(),
            people: Vec::new(),
        };

        assert!(text_of(&answer).is_empty());
    }

    /// §10.4 keeps the JSON stable, so the field names count.
    #[test]
    fn the_json_names_the_people() {
        let shelf = Shelf::new();
        shelf.put("a", "Alice A <alice@example.test>", "me@cfcosta.com");
        let mut out = Vec::new();

        write_json(&find(&shelf.book(&["me@cfcosta.com"]), "alice"), &mut out)
            .expect("a write");

        let held: serde_json::Value =
            serde_json::from_slice(&out).expect("the output is JSON");
        assert_eq!(held["name"], "alice");
        assert_eq!(held["people"][0]["address"], "alice@example.test");
        assert_eq!(held["people"][0]["inbound"], 1);
        assert_eq!(held["people"][0]["outbound"], 0);
    }

    // -----------------------------------------------------------------
    // Properties of the book.
    // -----------------------------------------------------------------

    /// The addresses that the property draws from.
    const WHO: [&str; 3] =
        ["alice@example.test", "bob@example.test", "me@cfcosta.com"];

    #[hegel::composite]
    fn some_mail(tc: TestCase) -> Vec<(String, String)> {
        let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
        let mut found = Vec::new();

        for _ in 0..count {
            let from =
                tc.draw(gs::integers::<usize>().min_value(0).max_value(2));
            let to = tc.draw(gs::integers::<usize>().min_value(0).max_value(2));

            found.push((WHO[from].to_string(), WHO[to].to_string()));
        }

        found
    }

    #[hegel::test(test_cases = 30)]
    fn prop_the_book_counts_every_message(tc: TestCase) {
        let mail = tc.draw(some_mail());
        let shelf = Shelf::new();
        let mut inbound: BTreeMap<String, u64> = BTreeMap::new();
        let mut outbound: BTreeMap<String, u64> = BTreeMap::new();

        for (at, (from, to)) in mail.iter().enumerate() {
            shelf.put(&format!("m{at}"), from, to);

            let tally = match from.as_str() {
                "me@cfcosta.com" => &mut outbound,
                _ => &mut inbound,
            };

            for address in [from, to] {
                *tally.entry(address.clone()).or_default() += 1;
            }
        }

        let book = shelf.book(&["me@cfcosta.com"]);

        for address in WHO {
            let held = book.get(address);
            let want_in = inbound.get(address).copied().unwrap_or(0);
            let want_out = outbound.get(address).copied().unwrap_or(0);

            assert_eq!(
                held.map_or(0, mailbert_core::contacts::Contact::inbound),
                want_in,
                "the inbound count of `{address}`"
            );
            assert_eq!(
                held.map_or(0, mailbert_core::contacts::Contact::outbound),
                want_out,
                "the outbound count of `{address}`"
            );
        }
    }

    #[hegel::test(test_cases = 25)]
    fn prop_the_order_of_the_store_never_counts(tc: TestCase) {
        let mail = tc.draw(some_mail());
        let first = Shelf::new();
        let second = Shelf::new();

        for (at, (from, to)) in mail.iter().enumerate() {
            first.put(&format!("m{at}"), from, to);
        }
        for (at, (from, to)) in mail.iter().enumerate().rev() {
            second.put(&format!("m{at}"), from, to);
        }

        assert_eq!(
            first.book(&["me@cfcosta.com"]),
            second.book(&["me@cfcosta.com"])
        );
    }
}
