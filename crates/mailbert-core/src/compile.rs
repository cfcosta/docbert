//! Compile a query into the two things that §8.2 needs.
//!
//! A filter is not a post-filter. If mailbert ranked first and then
//! removed what does not match, then `from:bob invoice` would give
//! nothing when the invoice of Bob is at rank 300. The failure is
//! silent, and it looks the same as mail that is not there.
//!
//! One query therefore becomes two Tantivy queries:
//!
//! 1. [`Compiled::search`], the filter and the free text together. The
//!    BM25 leg runs this, and it already matches the filter.
//! 2. [`Compiled::filter`], the filter alone. [`MailIndex::allow`] runs
//!    this and gives back the identities that gate the semantic leg,
//!    before that leg ranks.
//!
//! The safety property is that the filter is wider than the search:
//! every message that the search finds, the filter finds as well. A
//! narrower filter would drop mail that the search wanted. Two rules
//! keep this true under `not`:
//!
//! - [`Compiler::widen`] gives a query that finds at least as much.
//!   Free text becomes "everything".
//! - [`Compiler::narrow`] gives a query that finds at most as much.
//!   Free text becomes "nothing".
//!
//! Each rule uses the other under a `not`, because a negation turns a
//! superset into a subset.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Bound,
};

use tantivy::{
    Term,
    query::{
        AllQuery,
        BooleanQuery,
        BoostQuery,
        ConstScoreQuery,
        EmptyQuery,
        Occur,
        PhraseQuery,
        Query as TantivyQuery,
        RangeQuery,
        RegexQuery,
        TermQuery,
    },
    schema::{Field as SchemaField, IndexRecordOption},
};

use crate::{
    date::{Clock, DateRange},
    error::{Error, Result},
    index::{self, Fields, MailIndex},
    message_id,
    query::{self, Field, Flag, Query, Value},
    store::{Store, normalize_tag},
};

/// How many `saved:` expansions one query may do.
///
/// A saved search that names itself would never stop, and a chain that
/// is longer than this is a mistake and not a query.
pub const MAX_DEPTH: usize = 8;

/// What the compiler must know that the index does not hold.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Vocabulary {
    /// The saved searches of §9, by name.
    pub searches: BTreeMap<String, String>,

    /// Every tag that a message carries, for a `tag:` with a `*`.
    pub tags: BTreeSet<String>,
}

/// One query, in the two forms that a search needs.
pub struct Compiled {
    /// The filter and the free text, for the BM25 leg.
    pub search: Box<dyn TantivyQuery>,

    /// The filter alone, which gates the semantic leg.
    pub filter: Box<dyn TantivyQuery>,

    /// The free text of §7.1, for the semantic leg.
    pub text: String,
}

/// The state that the compilation of one query needs.
struct Compiler<'a> {
    index: &'a MailIndex,
    at: Fields,
    vocabulary: &'a Vocabulary,
}

impl Vocabulary {
    /// Read the saved searches and the tags from a store.
    pub fn from_store(store: &Store) -> Result<Self> {
        Ok(Self {
            searches: store.searches()?,
            tags: store.all_tags()?.into_keys().collect(),
        })
    }
}

/// Expand every `saved:` into the query that it names.
///
/// §9 says that a saved search expands inside a larger query, so the
/// expansion happens before anything else reads the query.
pub fn expand(
    query: &Query,
    vocabulary: &Vocabulary,
    clock: Clock,
) -> Result<Query> {
    expand_at(query, vocabulary, clock, 0)
}

/// Expand one query, after `depth` expansions came before it.
fn expand_at(
    query: &Query,
    vocabulary: &Vocabulary,
    clock: Clock,
    depth: usize,
) -> Result<Query> {
    let deeper = |parts: &Vec<Query>| -> Result<Vec<Query>> {
        parts
            .iter()
            .map(|part| expand_at(part, vocabulary, clock, depth))
            .collect()
    };

    Ok(match query {
        Query::Filter {
            field: Field::Saved,
            value,
        } => {
            let name = saved_name(value)?;

            // A saved search that names itself never stops, so the
            // depth is what ends the chain.
            if depth >= MAX_DEPTH {
                return Err(Error::SearchTooDeep(name));
            }

            let Some(text) = vocabulary.searches.get(&name) else {
                return Err(Error::UnknownSearch(name));
            };

            let named = query::parse(text, clock)?;

            expand_at(&named, vocabulary, clock, depth + 1)?
        }
        Query::Not(inner) => {
            Query::Not(Box::new(expand_at(inner, vocabulary, clock, depth)?))
        }
        Query::And(parts) => Query::And(deeper(parts)?),
        Query::Or(parts) => Query::Or(deeper(parts)?),
        other => other.clone(),
    })
}

/// The name that a `saved:` carries.
fn saved_name(value: &Value) -> Result<String> {
    match value {
        Value::Word(name) | Value::Phrase(name) => Ok(name.clone()),
        _ => Err(Error::BadFilterValue { field: "saved" }),
    }
}

/// Compile a query for one index.
pub fn compile(
    query: &Query,
    index: &MailIndex,
    vocabulary: &Vocabulary,
    clock: Clock,
) -> Result<Compiled> {
    let query = expand(query, vocabulary, clock)?;
    let compiler = Compiler::new(index, vocabulary);

    Ok(Compiled {
        search: compiler.search(&query)?,
        filter: compiler.widen(&query)?,
        text: query.text(),
    })
}

/// A glob of §7.1, as a regular expression over a whole term.
///
/// `*` stands for any run of characters, and `?` for one character.
/// Everything else is a literal.
///
/// # Examples
///
/// ```
/// use mailbert_core::compile::glob_to_regex;
///
/// assert_eq!(glob_to_regex("bob*"), "bob.*");
/// assert_eq!(glob_to_regex("a.b?"), r"a\.b.");
/// ```
pub fn glob_to_regex(glob: &str) -> String {
    let mut pattern = String::with_capacity(glob.len() * 2);

    for letter in glob.chars() {
        match letter {
            '*' => pattern.push_str(".*"),
            '?' => pattern.push('.'),
            _ => pattern.push_str(&regex::escape(&letter.to_string())),
        }
    }

    pattern
}

// ---------------------------------------------------------------------
// The pieces that a compiled query is made of.
// ---------------------------------------------------------------------

/// A query that finds every message.
fn everything() -> Box<dyn TantivyQuery> {
    Box::new(AllQuery)
}

/// A query that finds no message.
fn nothing() -> Box<dyn TantivyQuery> {
    Box::new(EmptyQuery)
}

/// One term of one field, which does not rank.
fn term_query(field: SchemaField, text: &str) -> Box<dyn TantivyQuery> {
    let term = Term::from_field_text(field, text);

    Box::new(TermQuery::new(term, IndexRecordOption::Basic))
}

/// A pattern over the whole term of one field.
fn regex_query(
    field: SchemaField,
    pattern: &str,
) -> Result<Box<dyn TantivyQuery>> {
    Ok(Box::new(RegexQuery::from_pattern(pattern, field)?))
}

/// Every part must match.
fn all_of(parts: Vec<Box<dyn TantivyQuery>>) -> Box<dyn TantivyQuery> {
    match parts.len() {
        0 => everything(),
        1 => parts.into_iter().next().expect("one part"),
        _ => Box::new(BooleanQuery::new(
            parts.into_iter().map(|part| (Occur::Must, part)).collect(),
        )),
    }
}

/// At least one part must match.
fn any_of(parts: Vec<Box<dyn TantivyQuery>>) -> Box<dyn TantivyQuery> {
    match parts.len() {
        0 => nothing(),
        1 => parts.into_iter().next().expect("one part"),
        _ => Box::new(BooleanQuery::new(
            parts
                .into_iter()
                .map(|part| (Occur::Should, part))
                .collect(),
        )),
    }
}

/// Everything that `inner` does not find.
///
/// Tantivy needs something to remove the messages from, so the query
/// keeps a `Must` over every message.
///
/// A query that says what is *not* wanted never ranks, so `inner`
/// takes a constant score. The constant also keeps a phrase away from
/// the exclusion, because a phrase cannot step back to an earlier
/// message and Tantivy stops when it must.
fn none_of(inner: Box<dyn TantivyQuery>) -> Box<dyn TantivyQuery> {
    let inner = Box::new(ConstScoreQuery::new(inner, 1.0));

    Box::new(BooleanQuery::new(vec![
        (Occur::Must, everything()),
        (Occur::MustNot, inner),
    ]))
}

/// A range over the fast field that holds the date.
fn date_query(field: SchemaField, range: DateRange) -> Box<dyn TantivyQuery> {
    let at = |instant: i64| Term::from_field_u64(field, instant.max(0) as u64);

    let lower = match range.start {
        Some(start) => Bound::Included(at(start)),
        None => Bound::Unbounded,
    };

    // §7.1 says that the end of a range is not part of it.
    let upper = match range.end {
        Some(end) => Bound::Excluded(at(end)),
        None => Bound::Unbounded,
    };

    Box::new(RangeQuery::new(lower, upper))
}

impl<'a> Compiler<'a> {
    fn new(index: &'a MailIndex, vocabulary: &'a Vocabulary) -> Self {
        Self {
            index,
            at: index.fields(),
            vocabulary,
        }
    }

    /// The query of the BM25 leg: the filter, and the free text.
    fn search(&self, query: &Query) -> Result<Box<dyn TantivyQuery>> {
        Ok(match query {
            Query::All => everything(),
            Query::Text(text) => self.free_text(text, false),
            Query::Phrase(text) => self.free_text(text, true),
            Query::Filter { field, value } => self.filter(*field, value)?,
            Query::Not(inner) => none_of(self.search(inner)?),
            Query::And(parts) => all_of(self.each(parts, Self::search)?),
            Query::Or(parts) => any_of(self.each(parts, Self::search)?),
        })
    }

    /// A query that finds at least as much as `query` does.
    fn widen(&self, query: &Query) -> Result<Box<dyn TantivyQuery>> {
        Ok(match query {
            Query::All | Query::Text(_) | Query::Phrase(_) => everything(),
            Query::Filter { field, value } => self.filter(*field, value)?,
            Query::Not(inner) => none_of(self.narrow(inner)?),
            Query::And(parts) => all_of(self.each(parts, Self::widen)?),
            Query::Or(parts) => any_of(self.each(parts, Self::widen)?),
        })
    }

    /// A query that finds at most as much as `query` does.
    fn narrow(&self, query: &Query) -> Result<Box<dyn TantivyQuery>> {
        Ok(match query {
            Query::All => everything(),
            Query::Text(_) | Query::Phrase(_) => nothing(),
            Query::Filter { field, value } => self.filter(*field, value)?,
            Query::Not(inner) => none_of(self.widen(inner)?),
            Query::And(parts) => all_of(self.each(parts, Self::narrow)?),
            Query::Or(parts) => any_of(self.each(parts, Self::narrow)?),
        })
    }

    /// Apply one of the three rules to each part of a group.
    fn each(
        &self,
        parts: &[Query],
        rule: fn(&Self, &Query) -> Result<Box<dyn TantivyQuery>>,
    ) -> Result<Vec<Box<dyn TantivyQuery>>> {
        parts.iter().map(|part| rule(self, part)).collect()
    }

    /// One filter, as a query on the field of that filter.
    fn filter(
        &self,
        field: Field,
        value: &Value,
    ) -> Result<Box<dyn TantivyQuery>> {
        let at = self.at;

        Ok(match field {
            Field::From => {
                self.address(at.from_addr, Some(at.from_name), value, "from")?
            }

            // §6.1 keeps one field for every recipient, so `cc:` finds
            // the Cc line and the To line together.
            Field::To | Field::Cc => {
                self.address(at.to_addr, None, value, "to")?
            }

            Field::Subject => {
                self.words_filter(at.subject, value, "subject")?
            }
            Field::Body => self.words_filter(at.body, value, "body")?,
            Field::Attachment => {
                self.words_filter(at.attachment, value, "attachment")?
            }

            Field::Folder => self.exact(at.folder, value, "folder")?,
            Field::Account => self.exact(at.account, value, "account")?,
            Field::List => self.exact(at.list_id, value, "list")?,

            Field::Tag => self.tag(value)?,

            Field::Is => match value {
                Value::Flag(flag) => self.state(*flag),
                _ => return Err(Error::BadFilterValue { field: "is" }),
            },

            Field::Has => match value {
                Value::Word(what)
                    if what.eq_ignore_ascii_case("attachment") =>
                {
                    term_query(at.flags, index::ATTACHMENT)
                }
                _ => return Err(Error::BadFilterValue { field: "has" }),
            },

            Field::Date => match value {
                Value::Date(range) => date_query(at.date, *range),
                _ => return Err(Error::BadFilterValue { field: "date" }),
            },

            Field::Mid => self.identity(at.mid_hash, value, "mid")?,
            Field::Thread => self.identity(at.thread_id, value, "thread")?,

            // `expand` removes every `saved:` before the compiler runs,
            // so one that is still here names nothing.
            Field::Saved => {
                return Err(Error::UnknownSearch(saved_name(value)?));
            }
        })
    }

    /// One state of §7.1, as a term of the `flags` field.
    fn state(&self, flag: Flag) -> Box<dyn TantivyQuery> {
        match index::flag_term(flag) {
            index::FlagTerm::Holds(term) => term_query(self.at.flags, term),
            index::FlagTerm::Lacks(term) => {
                none_of(term_query(self.at.flags, term))
            }
        }
    }

    /// `from:`, `to:`, and `cc:`.
    ///
    /// A value that holds an `@` is a whole address. A value that does
    /// not is a part of one, and for `from:` it is also a name.
    fn address(
        &self,
        addr: SchemaField,
        name: Option<SchemaField>,
        value: &Value,
        field: &'static str,
    ) -> Result<Box<dyn TantivyQuery>> {
        Ok(match value {
            Value::Word(word) | Value::Phrase(word) if word.contains('@') => {
                term_query(addr, &word.to_lowercase())
            }

            Value::Word(word) => {
                let inside =
                    format!(".*{}.*", regex::escape(&word.to_lowercase()));
                let mut parts = vec![regex_query(addr, &inside)?];

                if let Some(name) = name {
                    parts.push(self.words(name, word, false));
                }

                any_of(parts)
            }

            Value::Phrase(text) => match name {
                Some(name) => self.words(name, text, true),
                None => nothing(),
            },

            Value::Glob(glob) => {
                regex_query(addr, &glob_to_regex(&glob.to_lowercase()))?
            }

            Value::Date(_) | Value::Flag(_) => {
                return Err(Error::BadFilterValue { field });
            }
        })
    }

    /// A filter over a field that holds words: `subject:` and `body:`.
    fn words_filter(
        &self,
        field: SchemaField,
        value: &Value,
        name: &'static str,
    ) -> Result<Box<dyn TantivyQuery>> {
        Ok(match value {
            Value::Word(word) => self.words(field, word, false),
            Value::Phrase(text) => self.words(field, text, true),
            Value::Glob(glob) => {
                regex_query(field, &glob_to_regex(&glob.to_lowercase()))?
            }
            Value::Date(_) | Value::Flag(_) => {
                return Err(Error::BadFilterValue { field: name });
            }
        })
    }

    /// A filter over a field that holds one whole term.
    fn exact(
        &self,
        field: SchemaField,
        value: &Value,
        name: &'static str,
    ) -> Result<Box<dyn TantivyQuery>> {
        Ok(match value {
            Value::Word(word) | Value::Phrase(word) => term_query(field, word),
            Value::Glob(glob) => regex_query(field, &glob_to_regex(glob))?,
            Value::Date(_) | Value::Flag(_) => {
                return Err(Error::BadFilterValue { field: name });
            }
        })
    }

    /// `tag:`, which never reads a state.
    ///
    /// The `flags` field holds the tags and the states together, so a
    /// pattern runs over the tags that the store knows. A `*` can then
    /// not reach a `\encrypted` or a `\seen`.
    fn tag(&self, value: &Value) -> Result<Box<dyn TantivyQuery>> {
        Ok(match value {
            Value::Word(word) | Value::Phrase(word) => {
                let Some(tag) = normalize_tag(word) else {
                    return Err(Error::InvalidTag(word.clone()));
                };

                term_query(self.at.flags, &tag)
            }

            Value::Glob(glob) => {
                let pattern =
                    format!("^{}$", glob_to_regex(&glob.to_lowercase()));
                let pattern = regex::Regex::new(&pattern)?;

                any_of(
                    self.vocabulary
                        .tags
                        .iter()
                        .filter(|tag| pattern.is_match(tag))
                        .map(|tag| term_query(self.at.flags, tag))
                        .collect(),
                )
            }

            Value::Date(_) | Value::Flag(_) => {
                return Err(Error::BadFilterValue { field: "tag" });
            }
        })
    }

    /// `mid:` and `thread:`, which take a prefix of a hex identity.
    fn identity(
        &self,
        field: SchemaField,
        value: &Value,
        name: &'static str,
    ) -> Result<Box<dyn TantivyQuery>> {
        let prefix = match value {
            Value::Word(word) | Value::Phrase(word) => word.to_lowercase(),
            Value::Glob(glob) => {
                let pattern = glob_to_regex(&glob.to_lowercase());

                return regex_query(field, &pattern);
            }
            Value::Date(_) | Value::Flag(_) => {
                return Err(Error::BadFilterValue { field: name });
            }
        };

        // An identity is hex, so anything else names no message.
        if prefix.is_empty() || !prefix.chars().all(|at| at.is_ascii_hexdigit())
        {
            return Ok(nothing());
        }

        if prefix.len() >= message_id::FULL_LEN {
            return Ok(term_query(field, &prefix));
        }

        regex_query(field, &format!("{prefix}.*"))
    }

    /// The free text of §7.1, over the four fields that hold words.
    ///
    /// §6.1 gives the subject the boost, because a word in a subject
    /// says more than the same word in a long body.
    fn free_text(&self, text: &str, phrase: bool) -> Box<dyn TantivyQuery> {
        let at = self.at;
        let subject = self.words(at.subject, text, phrase);

        any_of(vec![
            Box::new(BoostQuery::new(subject, index::SUBJECT_BOOST)),
            self.words(at.body, text, phrase),
            self.words(at.from_name, text, phrase),
            self.words(at.attachment, text, phrase),
        ])
    }

    /// Words of one field, in order when `phrase` is true.
    fn words(
        &self,
        field: SchemaField,
        text: &str,
        phrase: bool,
    ) -> Box<dyn TantivyQuery> {
        let terms = self.tokens(field, text);
        let one = |term: Term| -> Box<dyn TantivyQuery> {
            Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs))
        };

        match terms.len() {
            0 => nothing(),
            1 => one(terms.into_iter().next().expect("one term")),
            _ if phrase => Box::new(PhraseQuery::new(terms)),
            _ => all_of(terms.into_iter().map(one).collect()),
        }
    }

    /// Cut a text into the terms that the index holds.
    ///
    /// The analyzer of the index does the cut, so a query and a
    /// document give the same terms for the same word.
    fn tokens(&self, field: SchemaField, text: &str) -> Vec<Term> {
        let Some(mut analyzer) =
            self.index.tantivy().tokenizers().get(index::ANALYZER)
        else {
            return Vec::new();
        };

        let mut stream = analyzer.token_stream(text);
        let mut terms = Vec::new();

        while let Some(token) = stream.next() {
            terms.push(Term::from_field_text(field, &token.text));
        }

        terms
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_the_filter_never_drops_what_the_search_finds` | metamorphic | §8.2: the filter gates the semantic leg. A filter that is narrower than the search hides mail that the search wanted, and says nothing. |
    //! | `prop_is_agrees_with_the_message` | differential | `is:` reads the index and `Message::matches` reads the message. The two must give one answer. |
    //! | `prop_a_place_filter_agrees_with_the_message` | differential | `folder:` and `account:` must find every copy of §4.2, and no other message. |
    //! | `prop_not_is_the_complement` | algebraic | A query and its negation must cover everything, and share nothing. |
    //! | `prop_expand_is_a_fixed_point` | algebraic | An expanded query holds no `saved:`, so expanding it again must change nothing. |
    //! | `prop_a_glob_matches_what_the_regex_matches` | differential | A `*` in a filter must mean the same thing in the index and in the tag vocabulary. |

    use hegel::{TestCase, generators as gs};

    use super::*;
    use crate::{
        index::Hit,
        message::{Location, Message, SEEN},
        message_id::MessageId,
        mime::{self, Attachment, Source},
        query,
        threading::ThreadId,
    };

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    /// 2026-08-22, the day that the tests call today.
    const NOW: i64 = 1_787_356_800;

    const BUDGET: usize = 15_000_000;

    /// One message of the corpus, with the tags that it carries.
    struct Item {
        key: &'static str,
        message: Message,
        tags: BTreeSet<String>,
    }

    /// An index, a vocabulary, and a way to name what a query finds.
    struct Mail {
        index: MailIndex,
        vocabulary: Vocabulary,
        clock: Clock,
        items: Vec<Item>,
    }

    fn location(account: &str, folder: &str) -> Location {
        Location {
            account: account.to_string(),
            folder: folder.to_string(),
            uid: 1,
            uid_validity: 1,
            received: NOW,
        }
    }

    /// A message from its headers and its body.
    fn draft(
        key: &'static str,
        headers: &str,
        body: &str,
        account: &str,
        folder: &str,
        flags: &[&str],
    ) -> Message {
        let raw =
            format!("{headers}Message-ID: <{key}@x.test>\r\n\r\n{body}\r\n")
                .into_bytes();

        Message::new(
            mime::parse(&raw).expect("a message"),
            location(account, folder),
            flags.iter().copied(),
        )
    }

    fn attachment(name: &str) -> Attachment {
        Attachment {
            name: Some(name.to_string()),
            content_type: "application/pdf".to_string(),
            size: 2048,
        }
    }

    fn tags(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|it| it.to_string()).collect()
    }

    impl Mail {
        fn new() -> Self {
            let mut invoice = draft(
                "a",
                "From: Alice Smith <alice@example.test>\r\n\
                 To: bob@example.test\r\n\
                 Subject: Invoice for August\r\n\
                 Date: Fri, 14 Aug 2026 09:30:00 +0000\r\n",
                "The invoice is attached, and the deposit is due.",
                "work",
                "INBOX",
                &[SEEN],
            );
            invoice.attachments.push(attachment("invoice.pdf"));

            let lunch = draft(
                "b",
                "From: Bob Jones <bob@work.test>\r\n\
                 To: alice@example.test\r\n\
                 Cc: carol@example.test\r\n\
                 Subject: Lunch on Friday\r\n\
                 Date: Wed, 01 Jul 2026 12:00:00 +0000\r\n",
                "Shall we eat at noon?",
                "work",
                "INBOX",
                &[],
            );

            let notes = draft(
                "c",
                "From: Rust Users <noreply@rust-lang.test>\r\n\
                 To: users@rust-lang.test\r\n\
                 Subject: Release notes\r\n\
                 List-Id: Rust Users <users.rust-lang.test>\r\n\
                 Date: Mon, 15 Jun 2026 08:00:00 +0000\r\n",
                "The compiler is faster than it was.",
                "personal",
                "Lists",
                &[SEEN],
            );

            let mut secret = draft(
                "d",
                "From: Dora Keys <dora@example.test>\r\n\
                 To: alice@example.test\r\n\
                 Subject: Secret plans\r\n\
                 Date: Fri, 01 May 2026 10:00:00 +0000\r\n",
                "Nothing readable here.",
                "personal",
                "Archive",
                &[],
            );
            secret.source = Source::Encrypted;
            secret.text = String::new();

            let mut gone = draft(
                "e",
                "From: Erik Old <erik@example.test>\r\n\
                 To: alice@example.test\r\n\
                 Subject: Old rent receipt\r\n\
                 Date: Wed, 01 Apr 2026 10:00:00 +0000\r\n",
                "The rent is paid.",
                "work",
                "Archive",
                &[SEEN],
            );
            gone.locations.clear();

            let items = vec![
                Item {
                    key: "a",
                    message: invoice,
                    tags: tags(&["todo"]),
                },
                Item {
                    key: "b",
                    message: lunch,
                    tags: BTreeSet::new(),
                },
                Item {
                    key: "c",
                    message: notes,
                    tags: BTreeSet::new(),
                },
                Item {
                    key: "d",
                    message: secret,
                    tags: BTreeSet::new(),
                },
                Item {
                    key: "e",
                    message: gone,
                    tags: tags(&["todo", "rent"]),
                },
            ];

            let index = MailIndex::open_in_ram().expect("an index");
            let mut writer = index.writer(BUDGET).expect("a writer");
            let thread = ThreadId::from_root(items[0].message.id);

            for (at, item) in items.iter().enumerate() {
                // The first two messages share a thread, so `thread:`
                // has more than one message to find.
                let thread = if at < 2 {
                    thread
                } else {
                    ThreadId::from_root(item.message.id)
                };

                index
                    .add(&writer, &item.message, thread, &item.tags)
                    .expect("a write");
            }
            index.commit(&mut writer).expect("a commit");

            let vocabulary = Vocabulary {
                searches: BTreeMap::from([
                    (
                        "work-unread".to_string(),
                        "is:unread folder:INBOX".to_string(),
                    ),
                    (
                        "money".to_string(),
                        "tag:rent or subject:invoice".to_string(),
                    ),
                    (
                        "both".to_string(),
                        "saved:money or saved:work-unread".to_string(),
                    ),
                    ("loop".to_string(), "saved:loop".to_string()),
                ]),
                tags: tags(&["todo", "rent"]),
            };

            Self {
                index,
                vocabulary,
                clock: Clock::utc(NOW),
                items,
            }
        }

        fn parse(&self, text: &str) -> Query {
            query::parse(text, self.clock).expect("a query")
        }

        fn compile(&self, text: &str) -> Compiled {
            compile(
                &self.parse(text),
                &self.index,
                &self.vocabulary,
                self.clock,
            )
            .expect("a compilation")
        }

        /// The key of one identity.
        fn key(&self, id: MessageId) -> &str {
            self.items
                .iter()
                .find(|item| item.message.id == id)
                .map(|item| item.key)
                .expect("a message of the corpus")
        }

        fn keys(&self, hits: &[Hit]) -> Vec<&str> {
            let mut found: Vec<&str> =
                hits.iter().map(|hit| self.key(hit.id)).collect();

            found.sort_unstable();
            found
        }

        /// The keys that the BM25 leg finds.
        fn search(&self, text: &str) -> Vec<&str> {
            let compiled = self.compile(text);
            let hits =
                self.index.top(&*compiled.search, 100).expect("a search");

            self.keys(&hits)
        }

        /// The keys that the filter allows the semantic leg to rank.
        fn allowed(&self, text: &str) -> Vec<&str> {
            let compiled = self.compile(text);
            let hits =
                self.index.top(&*compiled.filter, 100).expect("a search");

            self.keys(&hits)
        }

        fn item(&self, key: &str) -> &Item {
            self.items
                .iter()
                .find(|item| item.key == key)
                .expect("a message of the corpus")
        }
    }

    // -----------------------------------------------------------------
    // Globs.
    // -----------------------------------------------------------------

    #[test]
    fn a_star_stands_for_any_run_of_characters() {
        assert_eq!(glob_to_regex("bob*"), "bob.*");
        assert_eq!(glob_to_regex("*bob*"), ".*bob.*");
    }

    #[test]
    fn a_question_mark_stands_for_one_character() {
        assert_eq!(glob_to_regex("b?b"), "b.b");
    }

    #[test]
    fn everything_else_in_a_glob_is_a_literal() {
        assert_eq!(glob_to_regex("a.b+c"), r"a\.b\+c");
        assert_eq!(glob_to_regex("(x)"), r"\(x\)");
    }

    // -----------------------------------------------------------------
    // Saved searches.
    // -----------------------------------------------------------------

    #[test]
    fn expands_a_saved_search_inside_a_larger_query() {
        let mail = Mail::new();
        let query = mail.parse("saved:work-unread invoice");
        let wide = expand(&query, &mail.vocabulary, mail.clock).expect("it");

        assert_eq!(wide.text(), "invoice");
        assert_eq!(
            mail.search("saved:work-unread invoice"),
            Vec::<&str>::new()
        );
        assert_eq!(mail.search("saved:work-unread"), vec!["b"]);
    }

    #[test]
    fn expands_a_saved_search_that_names_another_one() {
        let mail = Mail::new();

        assert_eq!(mail.search("saved:both"), vec!["a", "b", "e"]);
    }

    #[test]
    fn refuses_a_saved_search_that_is_not_there() {
        let mail = Mail::new();
        let query = mail.parse("saved:nothing");
        let found = expand(&query, &mail.vocabulary, mail.clock);

        assert!(
            matches!(found, Err(Error::UnknownSearch(name)) if name == "nothing")
        );
    }

    #[test]
    fn refuses_a_saved_search_that_names_itself() {
        let mail = Mail::new();
        let query = mail.parse("saved:loop");
        let found = expand(&query, &mail.vocabulary, mail.clock);

        assert!(
            matches!(found, Err(Error::SearchTooDeep(name)) if name == "loop")
        );
    }

    #[test]
    fn reads_the_vocabulary_from_a_store() {
        let dir = tempfile::tempdir().expect("a directory");
        let store = Store::open(dir.path()).expect("a store");
        let mail = Mail::new();

        store
            .save_search("work-unread", "is:unread folder:INBOX")
            .expect("a save");
        store.put(&mail.item("a").message, b"raw").expect("a write");
        store
            .tag(&mail.item("a").message.id, "todo")
            .expect("a tag");

        let vocabulary = Vocabulary::from_store(&store).expect("a vocabulary");

        assert_eq!(
            vocabulary.searches.get("work-unread").map(String::as_str),
            Some("is:unread folder:INBOX")
        );
        assert_eq!(vocabulary.tags, tags(&["todo"]));
    }

    // -----------------------------------------------------------------
    // One filter, one field.
    // -----------------------------------------------------------------

    #[test]
    fn finds_a_sender_by_the_whole_address() {
        let mail = Mail::new();

        assert_eq!(mail.search("from:alice@example.test"), vec!["a"]);
    }

    #[test]
    fn finds_a_sender_by_a_part_of_the_address() {
        let mail = Mail::new();

        assert_eq!(mail.search("from:rust-lang"), vec!["c"]);
    }

    #[test]
    fn finds_a_sender_by_the_display_name() {
        let mail = Mail::new();

        assert_eq!(mail.search("from:jones"), vec!["b"]);
    }

    #[test]
    fn finds_a_sender_by_a_glob() {
        let mail = Mail::new();

        assert_eq!(mail.search("from:*@example.test"), vec!["a", "d", "e"]);
    }

    #[test]
    fn a_recipient_filter_reads_the_cc_line() {
        let mail = Mail::new();

        assert_eq!(mail.search("to:alice@example.test"), vec!["b", "d", "e"]);
        assert_eq!(mail.search("cc:carol@example.test"), vec!["b"]);
    }

    #[test]
    fn finds_a_subject_by_a_word_and_by_a_phrase() {
        let mail = Mail::new();

        assert_eq!(mail.search("subject:invoice"), vec!["a"]);
        assert_eq!(mail.search("subject:\"lunch on friday\""), vec!["b"]);
        assert_eq!(mail.search("subject:\"friday lunch\""), Vec::<&str>::new());
    }

    #[test]
    fn finds_a_body_by_a_word() {
        let mail = Mail::new();

        assert_eq!(mail.search("body:noon"), vec!["b"]);
        assert_eq!(mail.search("body:invoice"), vec!["a"]);
    }

    #[test]
    fn a_body_filter_never_reads_the_subject() {
        let mail = Mail::new();

        assert_eq!(mail.search("body:lunch"), Vec::<&str>::new());
    }

    #[test]
    fn finds_a_message_by_its_folder_and_its_account() {
        let mail = Mail::new();

        assert_eq!(mail.search("folder:INBOX"), vec!["a", "b"]);
        assert_eq!(mail.search("account:personal"), vec!["c", "d"]);
        assert_eq!(mail.search("folder:Arch*"), vec!["d"]);
    }

    #[test]
    fn finds_a_message_by_a_tag() {
        let mail = Mail::new();

        assert_eq!(mail.search("tag:todo"), vec!["a", "e"]);
        assert_eq!(mail.search("tag:rent"), vec!["e"]);
        assert_eq!(mail.search("tag:r*"), vec!["e"]);
    }

    #[test]
    fn a_tag_glob_never_reads_a_state() {
        // The states live in the same field as the tags, and a `*`
        // must not reach them.
        let mail = Mail::new();

        assert_eq!(mail.search("tag:*"), vec!["a", "e"]);
    }

    #[test]
    fn finds_a_message_by_its_state() {
        let mail = Mail::new();

        assert_eq!(mail.search("is:read"), vec!["a", "c", "e"]);
        assert_eq!(mail.search("is:unread"), vec!["b", "d"]);
        assert_eq!(mail.search("is:encrypted"), vec!["d"]);
        assert_eq!(mail.search("is:gone"), vec!["e"]);
        assert_eq!(mail.search("is:bulk"), vec!["c"]);
    }

    #[test]
    fn finds_a_message_that_carries_an_attachment() {
        let mail = Mail::new();

        assert_eq!(mail.search("has:attachment"), vec!["a"]);
        assert_eq!(mail.search("attachment:invoice.pdf"), vec!["a"]);
    }

    #[test]
    fn finds_a_message_inside_a_range_of_dates() {
        let mail = Mail::new();

        assert_eq!(mail.search("date:2026-07-01..2026-12-31"), vec!["a", "b"]);
        assert_eq!(mail.search("date:..2026-05-02"), vec!["d", "e"]);
        assert_eq!(mail.search("date:2026-08-14"), vec!["a"]);
    }

    #[test]
    fn finds_a_message_by_a_prefix_of_its_identity() {
        let mail = Mail::new();
        let id = mail.item("a").message.id;

        assert_eq!(mail.search(&format!("mid:{}", id.short())), vec!["a"]);
        assert_eq!(mail.search(&format!("mid:{}", id.full_hex())), vec!["a"]);
        assert_eq!(mail.search("mid:zzzz"), Vec::<&str>::new());
    }

    #[test]
    fn finds_every_message_of_one_thread() {
        let mail = Mail::new();
        let thread = ThreadId::from_root(mail.item("a").message.id);

        assert_eq!(
            mail.search(&format!("thread:{}", thread.short())),
            vec!["a", "b"]
        );
    }

    #[test]
    fn finds_a_message_by_the_list_that_carried_it() {
        let mail = Mail::new();

        assert_eq!(mail.search("list:users.rust-lang.test"), vec!["c"]);
        assert_eq!(mail.search("list:*rust-lang*"), vec!["c"]);
    }

    // -----------------------------------------------------------------
    // `and`, `or`, and `not`.
    // -----------------------------------------------------------------

    #[test]
    fn adjacency_means_and() {
        let mail = Mail::new();

        assert_eq!(mail.search("folder:INBOX is:read"), vec!["a"]);
        assert_eq!(mail.search("folder:INBOX and is:read"), vec!["a"]);
    }

    #[test]
    fn or_widens_the_answer() {
        let mail = Mail::new();

        assert_eq!(mail.search("tag:rent or subject:invoice"), vec!["a", "e"]);
    }

    #[test]
    fn not_gives_back_everything_else() {
        let mail = Mail::new();

        assert_eq!(mail.search("not folder:INBOX"), vec!["c", "d", "e"]);
        assert_eq!(mail.search("is:read and not tag:todo"), vec!["c"]);
    }

    #[test]
    fn brackets_group_the_terms() {
        let mail = Mail::new();

        assert_eq!(
            mail.search("folder:INBOX and (is:read or tag:todo)"),
            vec!["a"]
        );
    }

    // -----------------------------------------------------------------
    // Free text, and the two legs.
    // -----------------------------------------------------------------

    #[test]
    fn free_text_reads_the_subject_and_the_body() {
        let mail = Mail::new();

        assert_eq!(mail.search("lunch"), vec!["b"]);
        assert_eq!(mail.search("deposit"), vec!["a"]);
    }

    #[test]
    fn free_text_and_a_filter_apply_together() {
        let mail = Mail::new();

        assert_eq!(mail.search("invoice folder:INBOX"), vec!["a"]);
        assert_eq!(mail.search("invoice folder:Lists"), Vec::<&str>::new());
    }

    #[test]
    fn the_text_of_a_compiled_query_holds_no_filter() {
        let mail = Mail::new();
        let compiled = mail.compile("invoice folder:INBOX is:unread");

        assert_eq!(compiled.text, "invoice");
    }

    #[test]
    fn the_filter_of_a_query_with_no_filter_allows_everything() {
        let mail = Mail::new();

        assert_eq!(mail.allowed("invoice"), vec!["a", "b", "c", "d", "e"]);
    }

    #[test]
    fn the_filter_keeps_the_filters_and_drops_the_free_text() {
        let mail = Mail::new();

        assert_eq!(mail.search("invoice folder:INBOX"), vec!["a"]);
        assert_eq!(mail.allowed("invoice folder:INBOX"), vec!["a", "b"]);
    }

    #[test]
    fn the_filter_of_a_negated_free_text_allows_everything() {
        // `narrow` gives nothing for free text, so the negation of it
        // gives everything. A narrower filter would drop mail that the
        // search wanted.
        let mail = Mail::new();

        assert_eq!(mail.allowed("not invoice"), vec!["a", "b", "c", "d", "e"]);
    }

    #[test]
    fn the_filter_of_an_or_with_free_text_allows_everything() {
        let mail = Mail::new();

        assert_eq!(
            mail.allowed("folder:INBOX or invoice"),
            vec!["a", "b", "c", "d", "e"]
        );
    }

    #[test]
    fn the_allowlist_holds_the_key_of_the_embedding_database() {
        let mail = Mail::new();
        let compiled = mail.compile("folder:INBOX");
        let allowed = mail.index.allow(&*compiled.filter).expect("a list");

        let wanted: BTreeSet<u64> = ["a", "b"]
            .iter()
            .map(|key| mail.item(key).message.id.numeric())
            .collect();

        assert_eq!(allowed, wanted);
    }

    #[test]
    fn an_empty_query_allows_every_message() {
        let mail = Mail::new();
        let compiled = mail.compile("");

        assert_eq!(
            mail.index.allow(&*compiled.filter).expect("a list").len(),
            5
        );
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    fn a_term() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "invoice".to_string(),
            "lunch".to_string(),
            "\"the rent\"".to_string(),
            "from:alice@example.test".to_string(),
            "from:jones".to_string(),
            "to:alice@example.test".to_string(),
            "subject:invoice".to_string(),
            "body:noon".to_string(),
            "folder:INBOX".to_string(),
            "folder:Arch*".to_string(),
            "account:work".to_string(),
            "tag:todo".to_string(),
            "tag:r*".to_string(),
            "is:unread".to_string(),
            "is:read".to_string(),
            "is:encrypted".to_string(),
            "is:gone".to_string(),
            "has:attachment".to_string(),
            "list:*rust-lang*".to_string(),
            "date:2026-07-01..2026-12-31".to_string(),
            "saved:money".to_string(),
        ])
    }

    #[hegel::composite]
    fn a_query_text(tc: TestCase) -> String {
        let terms: Vec<String> =
            tc.draw(gs::vecs(a_term()).min_size(1).max_size(3));
        let joiner = if tc.draw(gs::booleans()) { " " } else { " or " };
        let text = terms.join(joiner);

        if tc.draw(gs::booleans()) {
            format!("not ({text})")
        } else {
            text
        }
    }

    #[hegel::test(test_cases = 150)]
    fn prop_the_filter_never_drops_what_the_search_finds(tc: TestCase) {
        let text = tc.draw(a_query_text());
        let mail = Mail::new();

        let found = mail.search(&text);
        let allowed = mail.allowed(&text);

        for key in &found {
            assert!(
                allowed.contains(key),
                "`{text}` finds `{key}`, and the filter drops it"
            );
        }
    }

    #[hegel::test(test_cases = 40)]
    fn prop_is_agrees_with_the_message(tc: TestCase) {
        let flag = tc.draw(gs::sampled_from(Flag::ALL.to_vec()));
        let mail = Mail::new();

        let found = mail.search(&format!("is:{}", flag.name()));
        let mut wanted: Vec<&str> = mail
            .items
            .iter()
            .filter(|item| item.message.matches(flag))
            .map(|item| item.key)
            .collect();
        wanted.sort_unstable();

        assert_eq!(found, wanted, "`is:{}` disagrees", flag.name());
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_place_filter_agrees_with_the_message(tc: TestCase) {
        let folder = tc.draw(gs::sampled_from(vec![
            "INBOX".to_string(),
            "Archive".to_string(),
            "Lists".to_string(),
        ]));
        let mail = Mail::new();

        let found = mail.search(&format!("folder:{folder}"));
        let mut wanted: Vec<&str> = mail
            .items
            .iter()
            .filter(|item| item.message.folders().contains(&folder.as_str()))
            .map(|item| item.key)
            .collect();
        wanted.sort_unstable();

        assert_eq!(found, wanted, "`folder:{folder}` disagrees");
    }

    #[hegel::test(test_cases = 60)]
    fn prop_not_is_the_complement(tc: TestCase) {
        let term = tc.draw(a_term());
        let mail = Mail::new();

        // Free text ranks, and does not gate. Only a filter has a
        // complement that the index can name.
        if !mail.parse(&term).has_filter() {
            return;
        }

        let found = mail.search(&term);
        let other = mail.search(&format!("not {term}"));
        let every: Vec<&str> = vec!["a", "b", "c", "d", "e"];

        for key in &every {
            assert_eq!(
                found.contains(key),
                !other.contains(key),
                "`{term}` and its negation disagree about `{key}`"
            );
        }
    }

    #[hegel::test(test_cases = 60)]
    fn prop_expand_is_a_fixed_point(tc: TestCase) {
        let text = tc.draw(a_query_text());
        let mail = Mail::new();
        let query = mail.parse(&text);

        let once = expand(&query, &mail.vocabulary, mail.clock).expect("it");
        let twice = expand(&once, &mail.vocabulary, mail.clock).expect("it");

        assert_eq!(once, twice);
    }

    #[hegel::test(test_cases = 60)]
    fn prop_a_glob_matches_what_the_regex_matches(tc: TestCase) {
        let glob = tc.draw(gs::sampled_from(vec![
            "r*".to_string(),
            "*o*".to_string(),
            "to?o".to_string(),
            "*".to_string(),
            "rent".to_string(),
        ]));
        let mail = Mail::new();

        let pattern = regex::Regex::new(&format!("^{}$", glob_to_regex(&glob)))
            .expect("a pattern");
        let mut wanted: Vec<&str> = mail
            .items
            .iter()
            .filter(|item| item.tags.iter().any(|tag| pattern.is_match(tag)))
            .map(|item| item.key)
            .collect();
        wanted.sort_unstable();

        assert_eq!(mail.search(&format!("tag:{glob}")), wanted);
    }
}
