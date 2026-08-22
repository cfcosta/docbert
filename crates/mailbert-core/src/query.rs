//! The query language.
//!
//! The grammar is a pragmatic subset of notmuch:
//!
//! ```text
//! query    := or_expr
//! or_expr  := and_expr ("or" and_expr)*
//! and_expr := unary (("and")? unary)*        # adjacency means "and"
//! unary    := "not" unary | primary
//! primary  := "(" query ")" | term
//! term     := field ":" value | phrase | word
//! ```
//!
//! Each term that has no field prefix is free text, and free text goes
//! to the hybrid leg. Each term that has a field prefix is a filter,
//! and filters never go to the hybrid leg.
//!
//! A query is short and the user retypes it at once, so every character
//! of an error message has a high value. The parser therefore collects
//! **every** problem it finds and reports them together, through
//! `miette`, against the text of the query.
//!
//! See `docs/mailbert.md` §7.

use std::{fmt, ops::Range};

use chumsky::{
    IterParser,
    Parser,
    error::Rich,
    extra,
    prelude::{any, choice, end, just, recursive},
    span::SimpleSpan,
};
use miette::{Diagnostic, LabeledSpan, NamedSource, SourceCode};
use thiserror::Error;

use crate::date::{self, Clock, DateError, DateRange};

/// The name that a query carries in an error report.
const SOURCE_NAME: &str = "query";

/// The words that join and negate terms.
const KEYWORDS: [&str; 3] = ["and", "or", "not"];

/// The only value that `has:` accepts.
const ATTACHMENT: &str = "attachment";

/// What to tell a user whose `date:` value did not parse.
const DATE_HELP: &str = "a date is 2026-08-14, a range is \
     2026-01-01..2026-06-30, an open range is ..2026-01-01, and the \
     keywords are today, yesterday, and now. An offset such as 7d, 3w, \
     6m, or 2y means \"since then\"";

/// What to tell a user whose query did not parse at all.
const SYNTAX_HELP: &str = "terms join with a space or with `and`, `or` \
     separates them, `not` negates one, and brackets group them";

/// Names that the user may reasonably type for a field.
const ALIASES: [(&str, Field); 17] = [
    ("author", Field::From),
    ("sender", Field::From),
    ("recipient", Field::To),
    ("rcpt", Field::To),
    ("subj", Field::Subject),
    ("title", Field::Subject),
    ("content", Field::Body),
    ("text", Field::Body),
    ("box", Field::Folder),
    ("dir", Field::Folder),
    ("mailbox", Field::Folder),
    ("label", Field::Tag),
    ("flag", Field::Is),
    ("when", Field::Date),
    ("id", Field::Mid),
    ("msgid", Field::Mid),
    ("file", Field::Attachment),
];

/// A parsed query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Query {
    /// An empty query, which matches every message.
    All,

    /// Free text, for the hybrid leg.
    Text(String),

    /// Free text that must appear in this order.
    Phrase(String),

    /// A filter, which gates both legs and never ranks.
    Filter {
        field: Field,
        value: Value,
    },

    Not(Box<Query>),
    And(Vec<Query>),
    Or(Vec<Query>),
}

/// The field of a filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Field {
    From,
    To,
    Cc,
    Subject,
    Body,
    Folder,
    Account,
    Tag,
    Is,
    Has,
    Date,
    Mid,
    Thread,
    List,
    Attachment,
    Saved,
}

/// The value of a filter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    /// One word.
    Word(String),

    /// Words that must appear in this order.
    Phrase(String),

    /// A word that holds `*` or `?`.
    Glob(String),

    /// A range of instants, from `date:`.
    Date(DateRange),

    /// A state of a message, from `is:`.
    Flag(Flag),
}

/// What `is:` accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Flag {
    Read,
    Unread,
    Flagged,
    Replied,
    Draft,
    Encrypted,
    Gone,
    Bulk,
}

/// One thing that is wrong with a query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Problem {
    /// Where in the query the problem is.
    pub span: Range<usize>,

    /// The headline, if this problem is the first one.
    pub message: String,

    /// The short text that goes under the caret.
    pub label: String,

    /// What the user can do about it.
    pub help: Option<String>,
}

/// Every problem that one query has.
#[derive(Debug, Error)]
#[error("{headline}")]
pub struct QueryError {
    headline: String,
    query: NamedSource<String>,
    problems: Vec<Problem>,
}

/// Parse `text` into a [`Query`].
///
/// Every problem is reported together, because the user retypes the
/// whole query anyway and one round of corrections beats three.
///
/// ```
/// use mailbert_core::{Field, Query, Value, date::Clock};
/// use mailbert_core::query::parse;
///
/// let clock = Clock::utc(1_787_400_000);
/// let query = parse("from:bob invoice", clock).unwrap();
///
/// assert!(query.has_filter());
/// assert_eq!(query.text(), "invoice");
///
/// let error = parse("sender:bob", clock).unwrap_err();
/// assert_eq!(error.headline(), "unknown filter `sender`");
/// ```
pub fn parse(text: &str, clock: Clock) -> Result<Query, QueryError> {
    if text.trim().is_empty() {
        return Ok(Query::All);
    }

    // A phrase that never closes swallows the rest of the query, and
    // the error that the grammar then gives points at the wrong place.
    if let Some(problem) = unclosed_quote(text) {
        return Err(QueryError::new(text, vec![problem]));
    }

    let raw = match grammar().parse(text).into_result() {
        Ok(raw) => raw,
        Err(errors) => {
            let problems = errors
                .iter()
                .map(|error| syntax_problem(error, text))
                .collect();

            return Err(QueryError::new(text, problems));
        }
    };

    let mut problems = Vec::new();
    let query = resolve(raw, clock, &mut problems);

    if let Some(query) = query
        && problems.is_empty()
    {
        return Ok(query);
    }

    Err(QueryError::new(text, problems))
}

// ---------------------------------------------------------------------
// The grammar.
// ---------------------------------------------------------------------

/// What the grammar produces, before any name is checked.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Raw {
    Term {
        field: Option<Spanned<String>>,
        value: Spanned<Text>,
    },
    Not(Box<Raw>),
    And(Vec<Raw>),
    Or(Vec<Raw>),
}

/// The text of a term, and how the user wrote it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Text {
    Word(String),
    Phrase(String),
}

/// A value and where it sits in the query.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Spanned<T> {
    value: T,
    span: Range<usize>,
}

/// The error type that the grammar reports with.
type Extra<'src> = extra::Err<Rich<'src, char>>;

/// Whether `found` may appear in a word that carries no quotes.
fn is_word_char(found: char) -> bool {
    !found.is_whitespace() && !matches!(found, '(' | ')' | '"')
}

/// One bare word, whatever it holds.
fn bare_word<'src>()
-> impl Parser<'src, &'src str, &'src str, Extra<'src>> + Clone {
    any()
        .filter(|found: &char| is_word_char(*found))
        .repeated()
        .at_least(1)
        .to_slice()
}

/// One bare word equal to `word`. The comparison ignores case, and a
/// longer word never matches, so `android` is not `and`.
fn keyword<'src>(
    word: &'static str,
) -> impl Parser<'src, &'src str, (), Extra<'src>> + Clone {
    bare_word().try_map(move |found: &str, span| {
        match found.eq_ignore_ascii_case(word) {
            true => Ok(()),
            false => Err(Rich::custom(span, format!("expected `{word}`"))),
        }
    })
}

fn grammar<'src>() -> impl Parser<'src, &'src str, Raw, Extra<'src>> {
    recursive(|query| {
        let phrase = any()
            .filter(|found: &char| *found != '"')
            .repeated()
            .to_slice()
            .delimited_by(just('"'), just('"'));

        let value = choice((
            phrase.map(|found: &str| Text::Phrase(found.to_string())),
            bare_word().map(|found: &str| Text::Word(found.to_string())),
        ))
        .map_with(|value, extra| Spanned::new(value, extra.span()));

        // A field name commits the term: `from:` with nothing after it
        // is a mistake, not the free text `from:`.
        let field = any()
            .filter(|found: &char| found.is_ascii_alphabetic())
            .repeated()
            .at_least(1)
            .to_slice()
            .map_with(|name: &str, extra| {
                Spanned::new(name.to_string(), extra.span())
            })
            .then_ignore(just(':'));

        let term =
            field.or_not().then(value).try_map(|(field, value), span| {
                // `and` alone joins terms, but `from:and` is a person.
                if field.is_none() && value.value.is_keyword() {
                    return Err(Rich::custom(span, "a keyword joins terms"));
                }

                Ok(Raw::Term { field, value })
            });

        let atom = choice((
            query.delimited_by(just('(').padded(), just(')').padded()),
            term,
        ))
        .padded();

        let unary = recursive(|unary| {
            choice((
                keyword("not")
                    .padded()
                    .ignore_then(unary)
                    .map(|inner| Raw::Not(Box::new(inner))),
                atom,
            ))
        });

        let conjunction = unary
            .clone()
            .then(
                keyword("and")
                    .padded()
                    .or_not()
                    .ignore_then(unary)
                    .repeated()
                    .collect::<Vec<Raw>>(),
            )
            .map(|(head, tail)| gather(Raw::And, head, tail));

        conjunction
            .clone()
            .then(
                keyword("or")
                    .padded()
                    .ignore_then(conjunction)
                    .repeated()
                    .collect::<Vec<Raw>>(),
            )
            .map(|(head, tail)| gather(Raw::Or, head, tail))
    })
    .padded()
    .then_ignore(end())
}

/// Join `head` and `tail`, unless there is only `head`.
fn gather(build: fn(Vec<Raw>) -> Raw, head: Raw, tail: Vec<Raw>) -> Raw {
    if tail.is_empty() {
        return head;
    }

    let mut parts = vec![head];
    parts.extend(tail);

    build(parts)
}

// ---------------------------------------------------------------------
// From names to fields.
// ---------------------------------------------------------------------

/// Check every name of the raw query and build the real one.
///
/// A node that fails gives `None`, and its problem joins `problems`.
/// The walk goes on either way, so that one report holds every problem.
fn resolve(
    raw: Raw,
    clock: Clock,
    problems: &mut Vec<Problem>,
) -> Option<Query> {
    match raw {
        Raw::Term { field: None, value } => Some(match value.value {
            Text::Word(word) => Query::Text(word),
            Text::Phrase(phrase) => Query::Phrase(phrase),
        }),
        Raw::Term {
            field: Some(name),
            value,
        } => resolve_filter(name, value, clock, problems),
        Raw::Not(inner) => {
            let inner = resolve(*inner, clock, problems)?;

            Some(Query::Not(Box::new(inner)))
        }
        Raw::And(parts) => resolve_each(parts, clock, problems).map(Query::And),
        Raw::Or(parts) => resolve_each(parts, clock, problems).map(Query::Or),
    }
}

/// Resolve every part before giving up on any of them.
fn resolve_each(
    parts: Vec<Raw>,
    clock: Clock,
    problems: &mut Vec<Problem>,
) -> Option<Vec<Query>> {
    let resolved: Vec<Option<Query>> = parts
        .into_iter()
        .map(|part| resolve(part, clock, problems))
        .collect();

    resolved.into_iter().collect()
}

fn resolve_filter(
    name: Spanned<String>,
    value: Spanned<Text>,
    clock: Clock,
    problems: &mut Vec<Problem>,
) -> Option<Query> {
    let Some(field) = Field::parse(&name.value) else {
        problems.push(unknown_field(&name));
        return None;
    };

    let value = match field {
        Field::Is => {
            let Some(flag) = Flag::parse(value.value.as_str()) else {
                problems.push(unknown_flag(&value));
                return None;
            };

            Value::Flag(flag)
        }
        Field::Has => {
            if !value.value.as_str().eq_ignore_ascii_case(ATTACHMENT) {
                problems.push(unknown_has(&value));
                return None;
            }

            Value::Word(ATTACHMENT.to_string())
        }
        Field::Date => match date::parse(value.value.as_str(), clock) {
            Ok(range) => Value::Date(range),
            Err(error) => {
                problems.push(bad_date(&value, &error));
                return None;
            }
        },
        _ => match value.value {
            Text::Phrase(phrase) => Value::Phrase(phrase),
            Text::Word(word) if word.contains(['*', '?']) => Value::Glob(word),
            Text::Word(word) => Value::Word(word),
        },
    };

    Some(Query::Filter { field, value })
}

// ---------------------------------------------------------------------
// Problems.
// ---------------------------------------------------------------------

fn unknown_field(name: &Spanned<String>) -> Problem {
    let label = match Field::suggest(&name.value) {
        Some(field) => format!("did you mean `{field}`?"),
        None => "this is not a filter".to_string(),
    };

    Problem {
        span: name.span.clone(),
        message: format!("unknown filter `{}`", name.value),
        label,
        help: Some(format!("the filters are: {}", names(&Field::ALL))),
    }
}

fn unknown_flag(value: &Spanned<Text>) -> Problem {
    Problem {
        span: value.span.clone(),
        message: format!(
            "`{}` is not a state of a message",
            value.value.as_str()
        ),
        label: "this is not a state".to_string(),
        help: Some(format!("`is:` accepts: {}", names(&Flag::ALL))),
    }
}

fn unknown_has(value: &Spanned<Text>) -> Problem {
    Problem {
        span: value.span.clone(),
        message: format!(
            "`has:` accepts `attachment`, and not `{}`",
            value.value.as_str()
        ),
        label: "the only value is `attachment`".to_string(),
        help: Some(
            "`has:attachment` finds every message that carries a file"
                .to_string(),
        ),
    }
}

fn bad_date(value: &Spanned<Text>, error: &DateError) -> Problem {
    Problem {
        span: value.span.clone(),
        message: error.to_string(),
        label: "this date is not a format that I know".to_string(),
        help: Some(DATE_HELP.to_string()),
    }
}

/// Find a phrase that never closes.
///
/// There is no escape inside a phrase, so an odd number of quotes can
/// only mean that the last one opens a phrase and nothing closes it.
fn unclosed_quote(text: &str) -> Option<Problem> {
    let quotes: Vec<usize> =
        text.match_indices('"').map(|(at, _)| at).collect();

    if quotes.len().is_multiple_of(2) {
        return None;
    }

    let at = *quotes.last()?;

    Some(Problem {
        span: at..at + 1,
        message: "this quote is never closed".to_string(),
        label: "the phrase starts here".to_string(),
        help: Some("close the phrase with another `\"`".to_string()),
    })
}

fn syntax_problem(error: &Rich<'_, char>, text: &str) -> Problem {
    let found = error.span();
    let mut start = floor_boundary(text, found.start);
    let mut end = ceil_boundary(text, found.end.max(found.start));

    // A span of no width has nothing to point at, which happens at the
    // end of the input. Widen it onto the character next to it.
    if start == end {
        end = ceil_boundary(text, end + 1);
    }
    if start == end {
        start = floor_boundary(text, start.saturating_sub(1));
    }

    Problem {
        span: start..end,
        message: format!("{error}"),
        label: "the query stops making sense here".to_string(),
        help: Some(SYNTAX_HELP.to_string()),
    }
}

/// The nearest character boundary at or before `at`.
fn floor_boundary(text: &str, at: usize) -> usize {
    let mut at = at.min(text.len());
    while at > 0 && !text.is_char_boundary(at) {
        at -= 1;
    }

    at
}

/// The nearest character boundary at or after `at`.
fn ceil_boundary(text: &str, at: usize) -> usize {
    let mut at = at.min(text.len());
    while at < text.len() && !text.is_char_boundary(at) {
        at += 1;
    }

    at
}

/// The names of every field or flag, for a help line.
fn names<T: fmt::Display>(all: &[T]) -> String {
    all.iter()
        .map(T::to_string)
        .collect::<Vec<String>>()
        .join(", ")
}

impl Query {
    /// The free text of the query, for the hybrid leg.
    ///
    /// Filters do not appear, because they gate the legs instead of
    /// ranking inside them. Text under a `not` does not appear either,
    /// because a semantic leg cannot rank an absence.
    pub fn text(&self) -> String {
        let mut words: Vec<&str> = Vec::new();
        self.gather_text(&mut words);

        words.join(" ")
    }

    /// Whether any filter appears in the query.
    pub fn has_filter(&self) -> bool {
        match self {
            Query::Filter { .. } => true,
            Query::Not(inner) => inner.has_filter(),
            Query::And(parts) | Query::Or(parts) => {
                parts.iter().any(Query::has_filter)
            }
            Query::All | Query::Text(_) | Query::Phrase(_) => false,
        }
    }

    fn gather_text<'a>(&'a self, into: &mut Vec<&'a str>) {
        match self {
            Query::Text(words) | Query::Phrase(words) => into.push(words),
            Query::And(parts) | Query::Or(parts) => {
                for part in parts {
                    part.gather_text(into);
                }
            }
            // A semantic leg cannot rank an absence, and a filter gates
            // the legs instead of ranking inside them.
            Query::All | Query::Filter { .. } | Query::Not(_) => {}
        }
    }
}

impl Field {
    /// Every field, in the order that the help text lists them.
    pub const ALL: [Field; 16] = [
        Field::From,
        Field::To,
        Field::Cc,
        Field::Subject,
        Field::Body,
        Field::Folder,
        Field::Account,
        Field::Tag,
        Field::Is,
        Field::Has,
        Field::Date,
        Field::Mid,
        Field::Thread,
        Field::List,
        Field::Attachment,
        Field::Saved,
    ];

    /// The name that the query language uses.
    pub fn name(self) -> &'static str {
        match self {
            Field::From => "from",
            Field::To => "to",
            Field::Cc => "cc",
            Field::Subject => "subject",
            Field::Body => "body",
            Field::Folder => "folder",
            Field::Account => "account",
            Field::Tag => "tag",
            Field::Is => "is",
            Field::Has => "has",
            Field::Date => "date",
            Field::Mid => "mid",
            Field::Thread => "thread",
            Field::List => "list",
            Field::Attachment => "attachment",
            Field::Saved => "saved",
        }
    }

    /// Read a field name. The name is not case-sensitive.
    pub fn parse(text: &str) -> Option<Self> {
        Field::ALL
            .into_iter()
            .find(|field| field.name().eq_ignore_ascii_case(text))
    }

    /// The field that the user most probably meant.
    ///
    /// A name that people reasonably reach for wins first, because the
    /// distance from `sender` to `from` is too large to guess. A near
    /// miss on a real name comes next, for an ordinary typo.
    pub fn suggest(text: &str) -> Option<Self> {
        if let Some(field) = Self::parse(text) {
            return Some(field);
        }

        let text = text.to_ascii_lowercase();
        if let Some((_, field)) = ALIASES.iter().find(|(name, _)| *name == text)
        {
            return Some(*field);
        }

        Field::ALL
            .into_iter()
            .map(|field| (distance(&text, field.name()), field))
            .filter(|(found, field)| *found <= near_enough(field.name()))
            .min_by_key(|(found, field)| (*found, field.name()))
            .map(|(_, field)| field)
    }
}

impl Flag {
    /// Every flag, in the order that the help text lists them.
    pub const ALL: [Flag; 8] = [
        Flag::Read,
        Flag::Unread,
        Flag::Flagged,
        Flag::Replied,
        Flag::Draft,
        Flag::Encrypted,
        Flag::Gone,
        Flag::Bulk,
    ];

    /// The name that the query language uses.
    pub fn name(self) -> &'static str {
        match self {
            Flag::Read => "read",
            Flag::Unread => "unread",
            Flag::Flagged => "flagged",
            Flag::Replied => "replied",
            Flag::Draft => "draft",
            Flag::Encrypted => "encrypted",
            Flag::Gone => "gone",
            Flag::Bulk => "bulk",
        }
    }

    /// Read a flag name. The name is not case-sensitive.
    pub fn parse(text: &str) -> Option<Self> {
        Flag::ALL
            .into_iter()
            .find(|flag| flag.name().eq_ignore_ascii_case(text))
    }
}

impl QueryError {
    fn new(text: &str, mut problems: Vec<Problem>) -> Self {
        problems.sort_by_key(|problem| problem.span.start);

        let headline = match problems.first() {
            Some(problem) => problem.message.clone(),
            None => "the query did not parse".to_string(),
        };

        Self {
            headline,
            query: NamedSource::new(SOURCE_NAME, text.to_string()),
            problems,
        }
    }

    /// The first problem, which is also the headline of the report.
    pub fn headline(&self) -> &str {
        &self.headline
    }

    /// Every problem, in the order that they appear in the query.
    pub fn problems(&self) -> &[Problem] {
        &self.problems
    }
}

impl Diagnostic for QueryError {
    fn code(&self) -> Option<Box<dyn fmt::Display + '_>> {
        Some(Box::new("mailbert::query"))
    }

    fn source_code(&self) -> Option<&dyn SourceCode> {
        Some(&self.query)
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = LabeledSpan> + '_>> {
        Some(Box::new(self.problems.iter().map(|problem| {
            LabeledSpan::at(problem.span.clone(), problem.label.clone())
        })))
    }

    fn help(&self) -> Option<Box<dyn fmt::Display + '_>> {
        let help = self.problems.iter().find_map(|p| p.help.as_ref())?;

        Some(Box::new(help))
    }
}

/// How wrong a name may be and still suggest `name`.
///
/// A short name gets less room, because two edits turn almost any short
/// word into almost any other.
fn near_enough(name: &str) -> usize {
    match name.len() <= 3 {
        true => 1,
        false => 2,
    }
}

/// The number of edits that turn `a` into `b`.
///
/// This is the Levenshtein distance, over two rows of the usual table.
fn distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();

    let mut previous: Vec<usize> = (0..=a.len()).collect();
    let mut current: Vec<usize> = vec![0; a.len() + 1];

    for (row, right) in b.iter().enumerate() {
        current[0] = row + 1;

        for (column, left) in a.iter().enumerate() {
            let cost = usize::from(left != right);

            current[column + 1] = (current[column] + 1)
                .min(previous[column + 1] + 1)
                .min(previous[column] + cost);
        }

        std::mem::swap(&mut previous, &mut current);
    }

    previous[a.len()]
}

impl<T> Spanned<T> {
    fn new(value: T, span: SimpleSpan) -> Self {
        Self {
            value,
            span: span.into_range(),
        }
    }
}

impl Text {
    fn as_str(&self) -> &str {
        match self {
            Text::Word(text) | Text::Phrase(text) => text,
        }
    }

    /// Whether a word with no field prefix joins terms instead of
    /// being one.
    fn is_keyword(&self) -> bool {
        let Text::Word(word) = self else {
            return false;
        };

        KEYWORDS
            .iter()
            .any(|keyword| word.eq_ignore_ascii_case(keyword))
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl fmt::Display for Flag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_parse_never_panics` | invariant | The text comes from a command line, so any bytes can arrive. |
    //! | `prop_a_generated_query_always_parses` | invariant | Everything the grammar allows must parse, or a user finds the hole before the tests do. |
    //! | `prop_a_word_is_free_text` | invariant | A word with no field prefix is the common case, and it must reach the hybrid leg unchanged. |
    //! | `prop_adjacency_is_the_and_keyword` | differential | `a b` and `a and b` are two spellings of one query. |
    //! | `prop_brackets_around_a_term_change_nothing` | metamorphic | Brackets group, and grouping one term groups nothing. |
    //! | `prop_a_filter_never_reaches_the_text` | invariant | A filter that leaked into the hybrid leg would rank on its own name. |
    //! | `prop_free_text_reaches_the_text` | invariant | The other half: text that the user typed must not disappear. |
    //! | `prop_a_field_name_reads_back` | round-trip | The name in the help text must be the name that the parser accepts. |
    //! | `prop_a_flag_name_reads_back` | round-trip | The same, for `is:`. |
    //! | `prop_a_field_suggests_itself` | invariant | A suggestion that does not find an exact name would send the user in circles. |
    //! | `prop_distance_is_a_metric` | algebraic | The suggestions stand on it, so it must be reflexive, symmetric, and obey the triangle rule. |
    //! | `prop_every_problem_points_inside_the_query` | invariant | miette slices the query with the span. A span past the end panics the renderer. |

    use hegel::{TestCase, generators as gs};

    use super::*;

    /// 2026-08-22 12:00:00 UTC.
    const NOON: i64 = 1_787_400_000;

    fn clock() -> Clock {
        Clock::utc(NOON)
    }

    fn parsed(text: &str) -> Query {
        parse(text, clock())
            .unwrap_or_else(|error| panic!("{text:?} failed: {error}"))
    }

    fn failed(text: &str) -> QueryError {
        parse(text, clock())
            .expect_err(&format!("{text:?} parsed but should not have"))
    }

    fn text(word: &str) -> Query {
        Query::Text(word.to_string())
    }

    fn filter(field: Field, value: Value) -> Query {
        Query::Filter { field, value }
    }

    fn word(value: &str) -> Value {
        Value::Word(value.to_string())
    }

    // -----------------------------------------------------------------
    // Generators.
    // -----------------------------------------------------------------

    fn a_word() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "invoice".to_string(),
            "quarterly".to_string(),
            "bob".to_string(),
            "2026".to_string(),
            "re-send".to_string(),
        ])
    }

    fn a_field() -> impl gs::Generator<Field> {
        gs::sampled_from(Field::ALL.to_vec())
    }

    /// A term that the grammar accepts, and the words it contributes.
    #[hegel::composite]
    fn a_term(tc: TestCase) -> (String, Vec<String>) {
        let word: String = tc.draw(a_word());
        let shape: usize =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(4));

        match shape {
            0 => (word.clone(), vec![word]),
            1 => (format!("\"{word} report\""), vec![format!("{word} report")]),
            2 => (format!("from:{word}"), Vec::new()),
            3 => (format!("subject:\"{word}\""), Vec::new()),
            _ => ("date:7d".to_string(), Vec::new()),
        }
    }

    /// A whole query, and the words that its free text must hold.
    #[hegel::composite]
    fn a_query(tc: TestCase) -> (String, Vec<String>) {
        let terms: Vec<(String, Vec<String>)> =
            tc.draw(gs::vecs(a_term()).min_size(1).max_size(4));
        let joiner: String = tc.draw(gs::sampled_from(vec![
            " ".to_string(),
            " and ".to_string(),
            " AND ".to_string(),
            " or ".to_string(),
        ]));

        let query = terms
            .iter()
            .map(|(source, _)| source.as_str())
            .collect::<Vec<&str>>()
            .join(&joiner);

        let words = terms.into_iter().flat_map(|(_, words)| words).collect();

        (query, words)
    }

    // -----------------------------------------------------------------
    // Unit tests: shape.
    // -----------------------------------------------------------------

    #[test]
    fn an_empty_query_matches_everything() {
        assert_eq!(parsed(""), Query::All);
        assert_eq!(parsed("   "), Query::All);
    }

    #[test]
    fn a_word_is_free_text() {
        assert_eq!(parsed("invoice"), text("invoice"));
    }

    #[test]
    fn a_quoted_run_is_a_phrase() {
        assert_eq!(
            parsed("\"quarterly report\""),
            Query::Phrase("quarterly report".to_string())
        );
    }

    #[test]
    fn adjacency_means_and() {
        assert_eq!(
            parsed("invoice bob"),
            Query::And(vec![text("invoice"), text("bob")])
        );
    }

    #[test]
    fn the_and_keyword_is_the_same_as_adjacency() {
        assert_eq!(parsed("invoice and bob"), parsed("invoice bob"));
        assert_eq!(parsed("invoice AND bob"), parsed("invoice bob"));
    }

    #[test]
    fn or_binds_looser_than_and() {
        assert_eq!(
            parsed("a or b c"),
            Query::Or(vec![text("a"), Query::And(vec![text("b"), text("c")]),])
        );
    }

    #[test]
    fn not_negates_one_term() {
        assert_eq!(parsed("not spam"), Query::Not(Box::new(text("spam"))));
        assert_eq!(
            parsed("invoice not spam"),
            Query::And(vec![
                text("invoice"),
                Query::Not(Box::new(text("spam"))),
            ])
        );
    }

    #[test]
    fn not_stacks() {
        assert_eq!(
            parsed("not not spam"),
            Query::Not(Box::new(Query::Not(Box::new(text("spam")))))
        );
    }

    #[test]
    fn brackets_group() {
        assert_eq!(
            parsed("(a or b) c"),
            Query::And(vec![Query::Or(vec![text("a"), text("b")]), text("c"),])
        );
    }

    #[test]
    fn brackets_around_one_term_change_nothing() {
        assert_eq!(parsed("(invoice)"), parsed("invoice"));
        assert_eq!(parsed("((invoice))"), parsed("invoice"));
    }

    // -----------------------------------------------------------------
    // Unit tests: filters.
    // -----------------------------------------------------------------

    #[test]
    fn a_field_prefix_makes_a_filter() {
        assert_eq!(parsed("from:bob"), filter(Field::From, word("bob")));
    }

    #[test]
    fn a_field_name_is_not_case_sensitive() {
        assert_eq!(parsed("FROM:bob"), parsed("from:bob"));
    }

    #[test]
    fn a_filter_value_may_be_a_phrase() {
        assert_eq!(
            parsed("subject:\"quarterly report\""),
            filter(
                Field::Subject,
                Value::Phrase("quarterly report".to_string())
            )
        );
    }

    #[test]
    fn a_star_makes_the_value_a_glob() {
        assert_eq!(
            parsed("attachment:*.pdf"),
            filter(Field::Attachment, Value::Glob("*.pdf".to_string()))
        );
        assert_eq!(
            parsed("folder:INBOX/2026-??"),
            filter(Field::Folder, Value::Glob("INBOX/2026-??".to_string()))
        );
    }

    #[test]
    fn a_keyword_is_an_ordinary_filter_value() {
        // `from:and` is a person called And, not a conjunction.
        assert_eq!(parsed("from:and"), filter(Field::From, word("and")));
        assert_eq!(
            parsed("from:and and from:or"),
            Query::And(vec![
                filter(Field::From, word("and")),
                filter(Field::From, word("or")),
            ])
        );
    }

    #[test]
    fn a_quoted_keyword_is_free_text() {
        assert_eq!(parsed("\"and\""), Query::Phrase("and".to_string()));
    }

    #[test]
    fn a_value_may_hold_a_colon() {
        assert_eq!(
            parsed("mid:<abc@example.test>"),
            filter(Field::Mid, word("<abc@example.test>"))
        );
    }

    #[test]
    fn is_takes_a_flag() {
        assert_eq!(
            parsed("is:unread"),
            filter(Field::Is, Value::Flag(Flag::Unread))
        );
        assert_eq!(parsed("is:UNREAD"), parsed("is:unread"));
    }

    #[test]
    fn has_takes_attachment() {
        assert_eq!(
            parsed("has:attachment"),
            filter(Field::Has, word("attachment"))
        );
    }

    #[test]
    fn date_takes_a_range() {
        let expected = date::parse("2026-08-22", clock()).unwrap();

        assert_eq!(
            parsed("date:2026-08-22"),
            filter(Field::Date, Value::Date(expected))
        );
    }

    #[test]
    fn date_takes_a_relative_offset() {
        let expected = date::parse("7d", clock()).unwrap();

        assert_eq!(
            parsed("date:7d"),
            filter(Field::Date, Value::Date(expected))
        );
    }

    // -----------------------------------------------------------------
    // Unit tests: errors.
    // -----------------------------------------------------------------

    #[test]
    fn an_unknown_field_names_itself_and_suggests_one() {
        let error = failed("sender:bob");

        assert_eq!(error.headline(), "unknown filter `sender`");
        assert_eq!(error.problems().len(), 1);

        let problem = &error.problems()[0];
        assert_eq!(problem.span, 0..6);
        assert_eq!(problem.label, "did you mean `from`?");
        assert!(problem.help.as_ref().unwrap().contains("from, to, cc"));
    }

    #[test]
    fn an_unknown_field_with_no_near_name_still_lists_the_filters() {
        let error = failed("zzzzzz:bob");

        assert_eq!(error.headline(), "unknown filter `zzzzzz`");
        assert_eq!(error.problems()[0].label, "this is not a filter");
        assert!(error.problems()[0].help.is_some());
    }

    #[test]
    fn a_typo_in_a_field_suggests_the_field() {
        for (typo, field) in [
            ("fom", Field::From),
            ("subjct", Field::Subject),
            ("attachement", Field::Attachment),
        ] {
            let error = failed(&format!("{typo}:x"));

            assert_eq!(
                error.problems()[0].label,
                format!("did you mean `{}`?", field.name()),
                "{typo}"
            );
        }
    }

    #[test]
    fn an_unknown_flag_lists_the_flags() {
        let error = failed("is:starred");

        assert_eq!(error.headline(), "`starred` is not a state of a message");
        assert!(
            error.problems()[0]
                .help
                .as_ref()
                .unwrap()
                .contains("unread")
        );
        assert_eq!(error.problems()[0].span, 3..10);
    }

    #[test]
    fn has_refuses_anything_but_attachment() {
        let error = failed("has:banana");

        assert_eq!(error.problems()[0].span, 4..10);
        assert!(error.headline().contains("attachment"));
    }

    #[test]
    fn a_bad_date_says_so() {
        let error = failed("date:last-tuesday");

        assert!(error.headline().contains("last-tuesday"));
        assert_eq!(error.problems()[0].span, 5..17);
        assert!(error.problems()[0].help.as_ref().unwrap().contains("7d"));
    }

    #[test]
    fn every_problem_of_a_query_is_reported_together() {
        // The example of the design document.
        let error = failed("sender:bob and date:last-tuesday");

        assert_eq!(error.problems().len(), 2);
        assert_eq!(error.headline(), "unknown filter `sender`");
        assert_eq!(error.problems()[0].span, 0..6);
        assert_eq!(error.problems()[1].span, 20..32);
    }

    #[test]
    fn an_unclosed_quote_says_so() {
        let error = failed("\"quarterly report");

        assert!(error.headline().contains("quote"));
        assert_eq!(error.problems()[0].span, 0..1);
    }

    #[test]
    fn an_unclosed_bracket_is_an_error() {
        assert!(!failed("(a or b").problems().is_empty());
        assert!(!failed("a or b)").problems().is_empty());
    }

    #[test]
    fn a_dangling_keyword_is_an_error() {
        for query in ["and", "a or", "not", "a and"] {
            assert!(!failed(query).problems().is_empty(), "{query}");
        }
    }

    #[test]
    fn a_field_with_no_value_is_an_error() {
        assert!(!failed("from:").problems().is_empty());
    }

    // -----------------------------------------------------------------
    // Unit tests: what the legs receive.
    // -----------------------------------------------------------------

    #[test]
    fn the_text_holds_the_free_words_only() {
        assert_eq!(
            parsed("from:bob quarterly invoice is:unread").text(),
            "quarterly invoice"
        );
    }

    #[test]
    fn the_text_keeps_a_phrase_whole() {
        assert_eq!(
            parsed("\"quarterly report\" invoice").text(),
            "quarterly report invoice"
        );
    }

    #[test]
    fn the_text_drops_what_is_negated() {
        assert_eq!(parsed("invoice not draft").text(), "invoice");
    }

    #[test]
    fn a_query_of_filters_has_no_text() {
        assert_eq!(parsed("from:bob is:unread date:7d").text(), "");
    }

    #[test]
    fn has_filter_sees_a_filter_anywhere() {
        assert!(parsed("from:bob").has_filter());
        assert!(parsed("invoice or (bob and is:unread)").has_filter());
        assert!(!parsed("invoice bob").has_filter());
        assert!(!parsed("").has_filter());
    }

    // -----------------------------------------------------------------
    // Unit tests: names.
    // -----------------------------------------------------------------

    #[test]
    fn the_field_names_are_the_names_of_the_document() {
        let names: Vec<&str> =
            Field::ALL.iter().map(|field| field.name()).collect();

        assert_eq!(
            names,
            [
                "from",
                "to",
                "cc",
                "subject",
                "body",
                "folder",
                "account",
                "tag",
                "is",
                "has",
                "date",
                "mid",
                "thread",
                "list",
                "attachment",
                "saved",
            ]
        );
    }

    #[test]
    fn the_flag_names_are_the_names_of_the_document() {
        let names: Vec<&str> =
            Flag::ALL.iter().map(|flag| flag.name()).collect();

        assert_eq!(
            names,
            [
                "read",
                "unread",
                "flagged",
                "replied",
                "draft",
                "encrypted",
                "gone",
                "bulk",
            ]
        );
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 500)]
    fn prop_parse_never_panics(tc: TestCase) {
        let source: String = tc.draw(gs::text().min_size(0).max_size(40));

        let _ = parse(&source, clock());
    }

    #[hegel::test(test_cases = 300)]
    fn prop_a_generated_query_always_parses(tc: TestCase) {
        let (source, _): (String, Vec<String>) = tc.draw(a_query());

        parse(&source, clock())
            .unwrap_or_else(|error| panic!("{source:?} failed: {error}"));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_word_is_free_text(tc: TestCase) {
        let word: String = tc.draw(a_word());

        assert_eq!(parse(&word, clock()).unwrap(), Query::Text(word));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_adjacency_is_the_and_keyword(tc: TestCase) {
        let left: String = tc.draw(a_word());
        let right: String = tc.draw(a_word());

        assert_eq!(
            parse(&format!("{left} {right}"), clock()).unwrap(),
            parse(&format!("{left} and {right}"), clock()).unwrap()
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_brackets_around_a_term_change_nothing(tc: TestCase) {
        let (source, _): (String, Vec<String>) = tc.draw(a_term());

        assert_eq!(
            parse(&format!("({source})"), clock()).unwrap(),
            parse(&source, clock()).unwrap()
        );
    }

    #[hegel::test(test_cases = 300)]
    fn prop_a_filter_never_reaches_the_text(tc: TestCase) {
        let field: Field = tc.draw(a_field());
        let value: String = match field {
            Field::Is => "unread".to_string(),
            Field::Has => "attachment".to_string(),
            Field::Date => "7d".to_string(),
            _ => tc.draw(a_word()),
        };

        let query = parse(&format!("{field}:{value}"), clock()).unwrap();

        assert_eq!(query.text(), "");
        assert!(query.has_filter());
    }

    #[hegel::test(test_cases = 300)]
    fn prop_free_text_reaches_the_text(tc: TestCase) {
        let (source, words): (String, Vec<String>) = tc.draw(a_query());

        let text = parse(&source, clock()).unwrap().text();

        for word in &words {
            assert!(text.contains(word), "{source:?} lost {word:?}");
        }
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_field_name_reads_back(tc: TestCase) {
        let field: Field = tc.draw(a_field());

        assert_eq!(Field::parse(field.name()), Some(field));
        assert_eq!(Field::parse(&field.name().to_uppercase()), Some(field));
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_flag_name_reads_back(tc: TestCase) {
        let flag: Flag = tc.draw(gs::sampled_from(Flag::ALL.to_vec()));

        assert_eq!(Flag::parse(flag.name()), Some(flag));
        assert_eq!(Flag::parse(&flag.name().to_uppercase()), Some(flag));
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_field_suggests_itself(tc: TestCase) {
        let field: Field = tc.draw(a_field());

        assert_eq!(Field::suggest(field.name()), Some(field));
    }

    #[hegel::test(test_cases = 400)]
    fn prop_distance_is_a_metric(tc: TestCase) {
        let words: Vec<String> = tc.draw(
            gs::vecs(gs::text().min_size(0).max_size(8))
                .min_size(3)
                .max_size(3),
        );
        let (a, b, c) = (&words[0], &words[1], &words[2]);

        assert_eq!(distance(a, a), 0);
        assert_eq!(distance(a, b), distance(b, a));
        assert!(distance(a, c) <= distance(a, b) + distance(b, c));

        if a != b {
            assert!(distance(a, b) > 0);
        }
    }

    #[hegel::test(test_cases = 500)]
    fn prop_every_problem_points_inside_the_query(tc: TestCase) {
        let source: String = tc.draw(gs::text().min_size(1).max_size(40));

        let Err(error) = parse(&source, clock()) else {
            return;
        };

        for problem in error.problems() {
            assert!(problem.span.start <= problem.span.end, "{problem:?}");
            assert!(problem.span.end <= source.len(), "{problem:?}");
            assert!(source.is_char_boundary(problem.span.start), "{problem:?}");
            assert!(source.is_char_boundary(problem.span.end), "{problem:?}");
        }
    }
}
