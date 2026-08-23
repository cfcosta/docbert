//! The saved searches of §9.
//!
//! A saved search keeps a query under a name. `saved:rent` puts the
//! query back inside a larger query, and §9 lets a name stand for a
//! query that a reader gives many times.
//!
//! The store keeps the names and the queries. The IMAP server never
//! learns of them, because §3.3 makes mailbert a download-only mirror.

use std::io::Write;

use mailbert_core::{Store, date::Clock, query, store::normalize_tag};
use serde::Serialize;

use crate::{Tool, cli, error::Result};

/// One saved search. (§10.4)
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Entry {
    /// The name, in the lowercase form that the store keeps.
    pub name: String,

    /// The query of §7.1.
    pub query: String,
}

/// Every saved search that the store keeps. (§10.4)
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Answer {
    /// The searches, by name.
    pub searches: Vec<Entry>,
}

/// Keep a query under a name. (§9)
///
/// The query parses before the store keeps it, so a saved search
/// always reads. A name that is already there takes the new query.
///
/// # Errors
///
/// The function fails if the name is bad, if the query does not read,
/// or if the store refuses.
pub fn add(
    store: &Store,
    name: &str,
    words: &[String],
    clock: Clock,
) -> Result<Entry> {
    let name = normalize_tag(name).ok_or_else(|| {
        mailbert_core::Error::InvalidSearchName(name.to_string())
    })?;
    let query = words.join(" ");

    // §9 puts a saved search inside a larger query, so a query that
    // does not read must never reach the store.
    query::parse(&query, clock)?;
    store.save_search(&name, &query)?;

    Ok(Entry { name, query })
}

/// Every saved search, by name.
///
/// # Errors
///
/// The function fails if the store refuses.
pub fn list(store: &Store) -> Result<Answer> {
    let searches = store
        .searches()?
        .into_iter()
        .map(|(name, query)| Entry { name, query })
        .collect();

    Ok(Answer { searches })
}

/// Forget one saved search.
///
/// # Errors
///
/// The function fails if no saved search carries that name.
pub fn remove(store: &Store, name: &str) -> Result<()> {
    match store.forget_search(name)? {
        true => Ok(()),
        false => {
            Err(mailbert_core::Error::UnknownSearch(name.to_string()).into())
        }
    }
}

/// Write one line for each saved search.
///
/// # Errors
///
/// The function fails if the output does not take the text.
pub fn write_text(answer: &Answer, out: &mut dyn Write) -> Result<()> {
    let width = answer
        .searches
        .iter()
        .map(|entry| entry.name.chars().count())
        .max()
        .unwrap_or(0);

    for entry in &answer.searches {
        writeln!(out, "{:width$}  {}", entry.name, entry.query)?;
    }

    Ok(())
}

/// Write the saved searches as the JSON of §10.4.
///
/// # Errors
///
/// The function fails if the output does not take the text.
pub fn write_json(answer: &Answer, out: &mut dyn Write) -> Result<()> {
    writeln!(out, "{}", serde_json::to_string_pretty(answer)?)?;

    Ok(())
}

/// Do the work of `saved`. (§9)
///
/// # Errors
///
/// The function fails if the store refuses, or if a name is bad.
pub fn command(tool: &Tool, action: &cli::Saved) -> Result<()> {
    let store = tool.store()?;
    let mut out = std::io::stdout().lock();

    match action {
        cli::Saved::Add { name, words } => {
            let entry = add(&store, name, words, crate::clock())?;

            writeln!(out, "saved `{}`: {}", entry.name, entry.query)?;
        }
        cli::Saved::List { json } => {
            let answer = list(&store)?;

            match json {
                true => write_json(&answer, &mut out)?,
                false => write_text(&answer, &mut out)?,
            }
        }
        cli::Saved::Remove { name } => {
            remove(&store, name)?;

            writeln!(out, "forgot `{name}`")?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_name_gives_back_the_query_that_it_took` | round-trip | §9 says a name stands for a query. A name that gives back another query sends the reader to the wrong mail. |
    //! | `prop_the_list_holds_every_name_that_stays` | model-based | A list against a set of names. A name that the list drops is a saved search that the reader cannot find again. |

    use std::collections::BTreeMap;

    use hegel::{TestCase, generators as gs};
    use tempfile::{TempDir, tempdir};

    use super::*;

    /// A moment inside the day that the tests use.
    const NOW: i64 = 1_755_900_000;

    fn clock() -> Clock {
        Clock::utc(NOW)
    }

    struct Shelf {
        _dir: TempDir,
        store: Store,
    }

    impl Shelf {
        fn new() -> Self {
            let dir = tempdir().expect("a temporary directory");
            let store =
                Store::open(&dir.path().join("store")).expect("a store");

            Self { _dir: dir, store }
        }

        fn add(&self, name: &str, query: &str) -> Result<Entry> {
            let words: Vec<String> =
                query.split(' ').map(str::to_string).collect();

            super::add(&self.store, name, &words, clock())
        }

        fn names(&self) -> Vec<String> {
            list(&self.store)
                .expect("a list")
                .searches
                .into_iter()
                .map(|entry| entry.name)
                .collect()
        }
    }

    fn text_of(answer: &Answer) -> String {
        let mut out = Vec::new();
        write_text(answer, &mut out).expect("a write");

        String::from_utf8(out).expect("the output is text")
    }

    #[test]
    fn a_name_keeps_the_query_that_it_took() {
        let shelf = Shelf::new();

        shelf.add("rent", "from:alice tag:todo").expect("a save");

        assert_eq!(
            shelf.store.saved("rent").expect("a read").as_deref(),
            Some("from:alice tag:todo")
        );
    }

    #[test]
    fn the_words_of_the_command_join_into_one_query() {
        let shelf = Shelf::new();

        let entry = shelf.add("rent", "from:alice tag:todo").expect("a save");

        assert_eq!(entry.query, "from:alice tag:todo");
    }

    #[test]
    fn a_name_becomes_lowercase() {
        let shelf = Shelf::new();

        let entry = shelf.add("Rent", "tag:todo").expect("a save");

        assert_eq!(entry.name, "rent");
        assert_eq!(shelf.names(), vec!["rent".to_string()]);
    }

    #[test]
    fn a_name_that_is_there_takes_the_new_query() {
        let shelf = Shelf::new();
        shelf.add("rent", "tag:todo").expect("a save");

        shelf.add("rent", "tag:later").expect("a save");

        assert_eq!(shelf.names().len(), 1);
        assert_eq!(
            shelf.store.saved("rent").expect("a read").as_deref(),
            Some("tag:later")
        );
    }

    /// §9 puts a saved search inside a larger query, so a query that
    /// does not read must never reach the store.
    #[test]
    fn a_query_that_does_not_read_is_an_error() {
        let shelf = Shelf::new();

        let result = shelf.add("rent", "from:");

        assert!(
            matches!(result, Err(crate::error::Error::Query(_))),
            "{result:?}"
        );
        assert!(shelf.names().is_empty());
    }

    #[test]
    fn a_name_that_the_store_refuses_is_an_error() {
        let shelf = Shelf::new();

        let result = shelf.add("bad name", "tag:todo");

        assert!(
            matches!(
                result,
                Err(crate::error::Error::Core(
                    mailbert_core::Error::InvalidSearchName(_)
                ))
            ),
            "{result:?}"
        );
    }

    #[test]
    fn the_list_gives_the_searches_by_name() {
        let shelf = Shelf::new();
        shelf.add("rent", "tag:todo").expect("a save");
        shelf.add("bills", "tag:later").expect("a save");

        assert_eq!(
            shelf.names(),
            vec!["bills".to_string(), "rent".to_string()]
        );
    }

    #[test]
    fn the_list_of_a_new_store_is_empty() {
        let shelf = Shelf::new();

        assert!(list(&shelf.store).expect("a list").searches.is_empty());
    }

    #[test]
    fn a_search_that_goes_away_leaves_the_list() {
        let shelf = Shelf::new();
        shelf.add("rent", "tag:todo").expect("a save");
        shelf.add("bills", "tag:later").expect("a save");

        remove(&shelf.store, "rent").expect("a removal");

        assert_eq!(shelf.names(), vec!["bills".to_string()]);
    }

    #[test]
    fn a_name_that_is_not_there_is_an_error() {
        let shelf = Shelf::new();

        let result = remove(&shelf.store, "rent");

        assert!(
            matches!(
                result,
                Err(crate::error::Error::Core(
                    mailbert_core::Error::UnknownSearch(_)
                ))
            ),
            "{result:?}"
        );
    }

    /// §9 expands `saved:` inside a query, and the expansion reads the
    /// same store. A name that `add` writes must be a name that the
    /// compiler finds.
    #[test]
    fn a_saved_search_expands_inside_a_query() {
        let shelf = Shelf::new();
        shelf.add("rent", "tag:todo").expect("a save");
        let asked = query::parse("saved:rent", clock()).expect("a query");
        let vocabulary = mailbert_core::Vocabulary::from_store(&shelf.store)
            .expect("the words");

        let expanded =
            mailbert_core::compile::expand(&asked, &vocabulary, clock())
                .expect("an expansion");

        assert_ne!(expanded, asked);
    }

    #[test]
    fn the_text_writes_one_line_for_each_search() {
        let shelf = Shelf::new();
        shelf.add("rent", "tag:todo").expect("a save");
        shelf.add("bills", "tag:later").expect("a save");

        let held = text_of(&list(&shelf.store).expect("a list"));

        assert_eq!(held.lines().count(), 2, "{held}");
        assert!(held.contains("rent"), "{held}");
        assert!(held.contains("tag:later"), "{held}");
    }

    #[test]
    fn the_text_of_an_empty_list_is_empty() {
        let answer = Answer {
            searches: Vec::new(),
        };

        assert!(text_of(&answer).is_empty());
    }

    /// §10.4 keeps the JSON stable, so the field names count.
    #[test]
    fn the_json_names_the_searches() {
        let shelf = Shelf::new();
        shelf.add("rent", "tag:todo").expect("a save");
        let mut out = Vec::new();

        write_json(&list(&shelf.store).expect("a list"), &mut out)
            .expect("a write");

        let held: serde_json::Value =
            serde_json::from_slice(&out).expect("the output is JSON");
        assert_eq!(held["searches"][0]["name"], "rent");
        assert_eq!(held["searches"][0]["query"], "tag:todo");
    }

    // -----------------------------------------------------------------
    // Properties of the saved searches.
    // -----------------------------------------------------------------

    #[hegel::composite]
    fn a_name(tc: TestCase) -> String {
        tc.draw(gs::text().alphabet("abcdef").min_size(1).max_size(5))
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_name_gives_back_the_query_that_it_took(tc: TestCase) {
        let name = tc.draw(a_name());
        let tag = tc.draw(gs::sampled_from(vec![
            "todo".to_string(),
            "later".to_string(),
        ]));
        let query = format!("tag:{tag}");
        let shelf = Shelf::new();

        let entry = shelf.add(&name, &query).expect("a save");

        assert_eq!(entry.name, name);
        assert_eq!(
            shelf.store.saved(&name).expect("a read").as_deref(),
            Some(query.as_str())
        );
    }

    #[hegel::test(test_cases = 25)]
    fn prop_the_list_holds_every_name_that_stays(tc: TestCase) {
        let names = tc.draw(gs::vecs(a_name()).min_size(1).max_size(6));
        let shelf = Shelf::new();
        let mut model: BTreeMap<String, String> = BTreeMap::new();

        for (at, name) in names.iter().enumerate() {
            let query = format!("tag:t{at}");
            shelf.add(name, &query).expect("a save");
            model.insert(name.clone(), query);
        }

        // The draw can give a name twice, and the second removal of
        // one name is an error, so the model says what is still there.
        for name in &names {
            let drop = tc.draw(gs::booleans());

            if drop && model.contains_key(name) {
                remove(&shelf.store, name).expect("a removal");
                model.remove(name);
            }
        }

        let held: BTreeMap<String, String> = list(&shelf.store)
            .expect("a list")
            .searches
            .into_iter()
            .map(|entry| (entry.name, entry.query))
            .collect();

        assert_eq!(held, model);
    }
}
