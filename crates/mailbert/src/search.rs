//! `search` and `ksearch`, and the rows that they write. (§10.1)
//!
//! §8.2 gives the shape of the work. One query becomes a filter and a
//! free text. The filter goes to both legs before either one ranks, so
//! neither leg gives back a message that the filter refuses.
//!
//! `ksearch` runs the lexical leg alone and loads no model. `search`
//! runs both legs and fuses them. A `search` over mail that no pass
//! embedded gives the lexical leg alone, and says so, because an empty
//! answer must never look like mail that is not there.

use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
};

use candle_core::Tensor;
use docbert_plaid::index::Index as PlaidIndex;
use mailbert_core::{
    Store,
    address::Address,
    compile::{self, Vocabulary},
    config::Config,
    date::{Clock, day_text},
    index::{Hit, MailIndex},
    message::Message,
    message_id::MessageId,
    query,
    rank::{self, CANDIDATES, Options, Row, Sort},
};
use serde::Serialize;

use crate::{
    Tool,
    cli,
    error::Result,
    semantic::{self, Brain},
};

/// How wide the sender column is, at the most.
pub const WHO: usize = 20;

/// How wide the subject column is, at the most.
pub const SUBJECT: usize = 44;

/// How many characters of a body one snippet shows. (§10.1)
pub const SNIPPET: usize = 160;

/// Which legs a command runs. (§8.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Legs {
    /// Both legs, and the fusion of §8.3. This is `search`.
    Both,

    /// The lexical leg alone. This is `ksearch`, which loads no model.
    Words,
}

/// One row of §10.1, ready to write.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Line {
    /// The git-style prefix of §4.1.
    pub id: String,

    /// The day of the message, as `YYYY-MM-DD`.
    pub date: String,

    /// Where the message sits in its thread, counted from 1.
    pub position: usize,

    /// How many messages the thread holds.
    pub total: usize,

    /// The name of the sender, or the address when there is no name.
    pub who: String,

    /// The subject, as the reader sees it.
    pub subject: String,

    /// The folders that hold a copy, and then the tags. (§9)
    pub tags: Vec<String>,

    /// The passage that matched, when `--snippet` asks for it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,

    /// The score that put the row here. (§8.3)
    pub score: f32,
}

/// The whole answer of one search, in the shape that `--json` writes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Answer {
    /// The query, as the reader wrote it.
    pub query: String,

    /// True when the semantic leg ran. (§8.1)
    pub semantic: bool,

    /// One row for each thread. (§8.4)
    pub rows: Vec<Line>,
}

/// The candidates of both legs, fused and grouped. (§8.1, §8.4)
///
/// `brain` runs the semantic leg. `None` runs the lexical leg alone,
/// and so does a query that carries no free text, because a filter
/// with no words gives the model nothing to read. The second value of
/// the answer says whether the semantic leg ran.
///
/// # Errors
///
/// The function fails if the query is bad, if the index cannot read,
/// or if the model refuses the query.
pub fn find(
    store: &Store,
    index: &MailIndex,
    brain: Option<&mut Brain>,
    text: &str,
    options: Options,
    clock: Clock,
) -> Result<(Vec<Row>, bool)> {
    let asked = query::parse(text, clock)?;
    let vocabulary = Vocabulary::from_store(store)?;
    let compiled = compile::compile(&asked, index, &vocabulary, clock)?;

    let lexical = index.top(&*compiled.search, CANDIDATES)?;
    let words: Vec<u64> = lexical.iter().map(|hit| hit.num_id).collect();
    let mut hits: BTreeMap<u64, Hit> =
        lexical.into_iter().map(|hit| (hit.num_id, hit)).collect();

    let mut meaning: Vec<u64> = Vec::new();
    let mut ran = false;

    if let Some(brain) = brain
        && !compiled.text.is_empty()
        && let Some(plaid) = semantic::load(&brain.at)?
    {
        // §8.2: the filter gates the leg before it ranks. A query with
        // no filter lets every message through, and reads no keys.
        let allow = match asked.has_filter() {
            true => Some(index.allow(&*compiled.filter)?),
            false => None,
        };

        let point = brain.model.encode_query(&compiled.text)?;
        meaning = leg(store, index, &plaid, &point, allow.as_ref(), &mut hits)?;
        ran = true;
    }

    let all: Vec<Hit> = hits.values().cloned().collect();
    let threads = rank::threads_of(index, &all)?;
    let legs: [&[u64]; 2] = [&words, &meaning];

    Ok((rank::rank(&legs, &hits, &threads, options), ran))
}

/// The candidates of the semantic leg, and the rows that they name.
///
/// `hits` takes the row of each message that the leg found and the
/// lexical leg did not. The fusion of §8.3 reads them from there.
///
/// The model gives `point`, and nothing else here needs the model. A
/// test therefore drives the whole leg with a hand-made vector.
///
/// # Errors
///
/// The function fails if the index or the store cannot read.
pub fn leg(
    store: &Store,
    index: &MailIndex,
    plaid: &PlaidIndex,
    point: &Tensor,
    allow: Option<&BTreeSet<u64>>,
    hits: &mut BTreeMap<u64, Hit>,
) -> Result<Vec<u64>> {
    let mut keys = Vec::new();

    for id in semantic::leg(plaid, store, point, allow, CANDIDATES)? {
        // The index can lose a message that the store still holds,
        // and a row without a hit has nothing to show.
        if let Some(hit) = index.get(&id)? {
            keys.push(hit.num_id);
            hits.entry(hit.num_id).or_insert(hit);
        }
    }

    Ok(keys)
}

/// Turn the rows into the lines of §10.1.
///
/// The store gives the sender and the tags, which the index does not
/// hold. A row whose message the store lost keeps its subject, so a
/// search never goes quiet about a message that it found.
///
/// # Errors
///
/// The function fails if the store cannot read.
pub fn lines(
    store: &Store,
    rows: &[Row],
    terms: &[String],
    snippet: bool,
) -> Result<Vec<Line>> {
    let mut lines = Vec::with_capacity(rows.len());

    for row in rows {
        let message = store.get(&row.hit.id)?;

        lines.push(Line {
            id: row.hit.id.short(),
            date: day_text(row.hit.date),
            position: row.position,
            total: row.total,
            who: message
                .as_ref()
                .map(|held| sender(&held.from))
                .unwrap_or_default(),
            subject: row.hit.subject.clone(),
            tags: labels(store, &row.hit.id, message.as_ref())?,
            snippet: match snippet {
                true => message.as_ref().map(|held| best(&held.text, terms)),
                false => None,
            },
            score: row.score,
        });
    }

    Ok(lines)
}

/// The folders that hold a copy, and then the tags. (§9)
fn labels(
    store: &Store,
    id: &MessageId,
    message: Option<&Message>,
) -> Result<Vec<String>> {
    let mut seen: Vec<String> = Vec::new();

    for at in message.map(|held| held.locations.as_slice()).unwrap_or(&[]) {
        let folder = at.folder.to_lowercase();

        if !seen.contains(&folder) {
            seen.push(folder);
        }
    }

    seen.extend(store.tags_of(id)?);

    Ok(seen)
}

/// The name of the sender, or the address when there is no name.
pub fn sender(from: &[Address]) -> String {
    let Some(first) = from.first() else {
        return String::new();
    };

    first.name.clone().unwrap_or_else(|| first.address.clone())
}

/// The part of `text` that holds a word of the query. (§10.1)
///
/// A body with none of those words gives its start, because a snippet
/// that shows nothing is worse than a snippet that shows the first
/// line of the message.
pub fn best(text: &str, terms: &[String]) -> String {
    let flat = text.split_whitespace().collect::<Vec<&str>>().join(" ");

    if flat.is_empty() {
        return String::new();
    }

    let lower = flat.to_lowercase();
    let at = terms
        .iter()
        .filter(|term| !term.is_empty())
        .filter_map(|term| lower.find(&term.to_lowercase()))
        .min()
        .unwrap_or(0);

    // The window opens a little in front of the word, so the reader
    // sees what the sentence said before it.
    let start = boundary(&flat, at.saturating_sub(SNIPPET / 4), false);
    let stop = boundary(&flat, start.saturating_add(SNIPPET), true);
    let mut cut = flat[start..stop].to_string();

    if start > 0 {
        cut.insert(0, '…');
    }
    if stop < flat.len() {
        cut.push('…');
    }

    cut
}

/// The character boundary at `at`, or the nearest one.
///
/// A cut inside a character makes a panic, and a body carries whatever
/// the sender wrote. `forward` gives the direction of the search.
fn boundary(text: &str, at: usize, forward: bool) -> usize {
    let mut at = at.min(text.len());

    while !text.is_char_boundary(at) {
        match forward {
            true => at += 1,
            false => at -= 1,
        }
    }

    at
}

/// Write the rows of §10.1 as text.
///
/// # Errors
///
/// The function fails if the output does not take the text.
pub fn write_text(lines: &[Line], out: &mut dyn Write) -> Result<()> {
    let who = width(lines.iter().map(|line| line.who.as_str()), WHO);
    let subject =
        width(lines.iter().map(|line| line.subject.as_str()), SUBJECT);

    for line in lines {
        let place = format!("[{}/{}]", line.position, line.total);
        let tags = match line.tags.is_empty() {
            true => String::new(),
            false => format!("  ({})", line.tags.join(" ")),
        };

        writeln!(
            out,
            "{}  {}  {:>7}  {}  {}{}",
            line.id,
            line.date,
            place,
            pad(&line.who, who),
            pad(&line.subject, subject),
            tags
        )?;

        if let Some(snippet) = &line.snippet {
            writeln!(out, "    {snippet}")?;
        }
    }

    Ok(())
}

/// How wide a column is: the widest value, and no wider than `most`.
fn width<'a>(values: impl Iterator<Item = &'a str>, most: usize) -> usize {
    values
        .map(|one| one.chars().count())
        .max()
        .unwrap_or(0)
        .min(most)
}

/// Write the whole answer as JSON. (§10.4)
///
/// # Errors
///
/// The function fails if the output does not take the text.
pub fn write_json(answer: &Answer, out: &mut dyn Write) -> Result<()> {
    writeln!(out, "{}", serde_json::to_string_pretty(answer)?)?;

    Ok(())
}

/// The words that a snippet looks for: the free text of the query.
pub fn words_of(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter(|word| !word.contains(':'))
        .map(|word| word.trim_matches('"').to_string())
        .filter(|word| !word.is_empty())
        .collect()
}

/// `text` in a column of `wide` characters.
pub fn pad(text: &str, wide: usize) -> String {
    if wide == 0 {
        return String::new();
    }

    let held = text.chars().count();

    // A value that runs over loses its end, because a column that
    // grows breaks every row under it.
    match held > wide {
        true => {
            let mut cut: String =
                text.chars().take(wide.saturating_sub(1)).collect();
            cut.push('…');

            cut
        }
        false => format!("{text}{}", " ".repeat(wide - held)),
    }
}

/// The sort of §8.3 that the flag names.
pub fn sort_of(order: cli::Order) -> Sort {
    match order {
        cli::Order::Best => Sort::Best,
        cli::Order::Score => Sort::Score,
        cli::Order::Date => Sort::Date,
    }
}

/// The options of §8.3 that the flags and the configuration give.
pub fn options(args: &cli::Find, config: &Config, now: i64) -> Options {
    Options::new(now)
        .with_sort(sort_of(args.sort))
        .with_half_life(config.search.recency_half_life_days)
        .with_limit(args.count.unwrap_or(config.search.count))
}

/// The brain that a command needs.
///
/// §2.1 says that `ksearch` loads no model. The command therefore
/// opens no brain for it, and never touches the embedding database.
///
/// # Errors
///
/// The function fails if the brain cannot open its files.
pub fn brain_for(tool: &Tool, legs: Legs) -> Result<Option<Brain>> {
    match legs {
        Legs::Both => Ok(Some(tool.brain()?)),
        Legs::Words => Ok(None),
    }
}

/// Do the work of `search` and of `ksearch`. (§2.1)
///
/// # Errors
///
/// The function fails if the query is bad, if the store or the index
/// cannot read, or if the output does not take the text.
pub fn command(tool: &Tool, args: &cli::Find, legs: Legs) -> Result<()> {
    let config = tool.config()?;
    let store = tool.store()?;
    let index = tool.index()?;
    let clock = crate::clock();
    let text = args.text();

    let mut brain = brain_for(tool, legs)?;

    let options = options(args, &config, clock.now());
    let (rows, ran) =
        find(&store, &index, brain.as_mut(), &text, options, clock)?;
    let lines = lines(&store, &rows, &words_of(&text), args.snippet)?;

    let mut out = std::io::stdout().lock();

    if args.json {
        return write_json(
            &Answer {
                query: text,
                semantic: ran,
                rows: lines,
            },
            &mut out,
        );
    }

    // §8.1: a `search` that could not run the semantic leg gives the
    // lexical leg alone. The reader must know which answer this is.
    if legs == Legs::Both && !ran {
        eprintln!("mailbert: no message is embedded yet, so this is `ksearch`");
    }

    write_text(&lines, &mut out)
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_column_always_holds_its_width` | invariant | Every row shares one column width. A value that runs over breaks every row under it. |
    //! | `prop_a_snippet_never_cuts_a_character` | invariant | A body carries whatever the sender wrote. A cut inside a character makes a panic, so `-v` would kill the command. |
    //! | `prop_a_search_names_each_thread_once` | invariant | §8.4 gives one row for each thread. A thread that comes twice pushes another thread off the page. |
    //! | `prop_a_search_never_writes_more_rows_than_the_limit` | invariant | `-n` is the whole promise of the flag. A page that is longer scrolls the first row away. |

    use std::collections::BTreeSet;

    use candle_core::Device;
    use clap::Parser;
    use hegel::{TestCase, generators as gs};
    use mailbert_core::{
        message::{Location, Message},
        mime,
        threading::ThreadId,
    };
    use tempfile::{TempDir, tempdir};

    use super::*;

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    /// The smallest budget that a Tantivy writer accepts.
    const BUDGET: usize = 15_000_000;

    /// A moment inside the day that the test messages carry.
    const NOW: i64 = 1_755_900_000;

    /// The clock of every test, at UTC.
    fn clock() -> Clock {
        Clock::utc(NOW)
    }

    /// The options that a test uses when it does not say otherwise.
    fn plain() -> Options {
        Options::new(NOW).with_limit(10).with_half_life(180.0)
    }

    fn location(folder: &str, uid: u32) -> Location {
        Location {
            account: "work".to_string(),
            folder: folder.to_string(),
            uid,
            uid_validity: 1,
            received: 1_755_820_800,
            flags: BTreeSet::new(),
        }
    }

    fn raw(key: &str, who: &str, subject: &str, body: &str) -> Vec<u8> {
        format!(
            "From: {who}\r\n\
             To: bob@example.test\r\n\
             Subject: {subject}\r\n\
             Date: Fri, 22 Aug 2025 09:30:00 +0000\r\n\
             Message-ID: <{key}@x.test>\r\n\
             \r\n\
             {body}\r\n"
        )
        .into_bytes()
    }

    /// The mail that every search test reads.
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

        /// Write one message into the store and into the index.
        fn put(
            &self,
            key: &str,
            who: &str,
            subject: &str,
            body: &str,
        ) -> MessageId {
            self.put_at(key, who, subject, body, "INBOX", None)
        }

        fn put_at(
            &self,
            key: &str,
            who: &str,
            subject: &str,
            body: &str,
            folder: &str,
            thread: Option<ThreadId>,
        ) -> MessageId {
            let bytes = raw(key, who, subject, body);
            let message = Message::new(
                mime::parse(&bytes).expect("a message"),
                location(folder, 1),
                Vec::<String>::new(),
            );
            let held = self.store.put(&message, &bytes).expect("a write");
            let thread = thread.unwrap_or_else(|| ThreadId::from_root(held.id));
            let tags = self.store.tags_of(&held.id).expect("the tags");

            let mut writer = self.index.writer(BUDGET).expect("a writer");
            self.index
                .add(&writer, &held, thread, &tags)
                .expect("an index write");
            self.index.commit(&mut writer).expect("a commit");

            held.id
        }

        /// Run one search with the lexical leg alone.
        fn find(&self, text: &str) -> Vec<Row> {
            self.find_with(text, plain())
        }

        fn find_with(&self, text: &str, options: Options) -> Vec<Row> {
            find(&self.store, &self.index, None, text, options, clock())
                .expect("a search")
                .0
        }

        fn lines(
            &self,
            rows: &[Row],
            terms: &[String],
            snip: bool,
        ) -> Vec<Line> {
            lines(&self.store, rows, terms, snip).expect("the lines")
        }
    }

    fn terms(list: &[&str]) -> Vec<String> {
        list.iter().map(|one| one.to_string()).collect()
    }

    fn text_of(lines: &[Line]) -> String {
        let mut out = Vec::new();
        write_text(lines, &mut out).expect("a write");

        String::from_utf8(out).expect("the output is text")
    }

    fn line(id: &str, who: &str, subject: &str) -> Line {
        Line {
            id: id.to_string(),
            date: "2025-08-22".to_string(),
            position: 1,
            total: 1,
            who: who.to_string(),
            subject: subject.to_string(),
            tags: Vec::new(),
            snippet: None,
            score: 1.0,
        }
    }

    // -----------------------------------------------------------------
    // The legs of §8.1.
    // -----------------------------------------------------------------

    #[test]
    fn a_search_finds_the_message_that_holds_the_word() {
        let shelf = Shelf::new();
        let wanted =
            shelf.put("a", "Alice <alice@x.test>", "Deposit", "the deposit");
        shelf.put("b", "Bob <bob@x.test>", "Lunch", "the sandwich");

        let rows = shelf.find("deposit");

        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].hit.id, wanted);
    }

    #[test]
    fn a_search_reads_the_filters_of_the_query_language() {
        let shelf = Shelf::new();
        shelf.put("a", "Alice <alice@x.test>", "Deposit", "the deposit");
        let wanted =
            shelf.put("b", "Bob <bob@x.test>", "Deposit", "the deposit");

        let rows = shelf.find("from:bob@x.test deposit");

        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].hit.id, wanted);
    }

    #[test]
    fn a_search_with_no_word_at_all_gives_nothing() {
        let shelf = Shelf::new();
        shelf.put("a", "Alice <alice@x.test>", "Deposit", "the deposit");

        assert!(shelf.find("sandwich").is_empty());
    }

    #[test]
    fn a_search_with_no_brain_never_runs_the_semantic_leg() {
        let shelf = Shelf::new();
        shelf.put("a", "Alice <alice@x.test>", "Deposit", "the deposit");

        let (rows, ran) = find(
            &shelf.store,
            &shelf.index,
            None,
            "deposit",
            plain(),
            clock(),
        )
        .expect("a search");

        assert!(!ran, "the leg ran without a model");
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn a_bad_query_gives_the_error_of_the_parser() {
        let shelf = Shelf::new();

        let result =
            find(&shelf.store, &shelf.index, None, "from:", plain(), clock());

        assert!(matches!(result, Err(crate::error::Error::Query(_))));
    }

    #[test]
    fn a_search_gives_one_row_for_each_thread() {
        let shelf = Shelf::new();
        let root = shelf.put("a", "Alice <a@x.test>", "Deposit", "the deposit");
        let thread = ThreadId::from_root(root);
        shelf.put_at(
            "b",
            "Bob <b@x.test>",
            "Re: Deposit",
            "the deposit again",
            "INBOX",
            Some(thread),
        );

        let rows = shelf.find("deposit");

        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].total, 2, "the row lost the size of the thread");
    }

    #[test]
    fn the_limit_of_the_options_cuts_the_rows() {
        let shelf = Shelf::new();
        shelf.put("a", "Alice <a@x.test>", "Deposit", "the deposit");
        shelf.put("b", "Bob <b@x.test>", "Deposit", "the deposit");

        let rows = shelf.find_with("deposit", plain().with_limit(1));

        assert_eq!(rows.len(), 1);
    }

    // -----------------------------------------------------------------
    // The semantic leg of §8.2.
    // -----------------------------------------------------------------

    /// How wide a hand-made embedding is.
    const DIM: u32 = 2;

    /// The direction that the passages of a near message point in.
    const NEAR: [f32; 2] = [1.0, 0.0];

    /// The direction that the passages of a far message point in.
    ///
    /// It leans away from [`NEAR`] and does not stand square to it,
    /// because PLAID prunes what a query points nowhere near, and a
    /// test of the filter must see both messages. (§8.2)
    const FAR: [f32; 2] = [0.6, 0.8];

    /// A model that no test loads, because no test needs one.
    const ABSENT: &str = "nobody/no-such-model";

    /// A query that points in one direction.
    fn point(at: [f32; 2]) -> Tensor {
        Tensor::from_vec(at.to_vec(), (1, 2), &Device::Cpu)
            .expect("a query tensor")
    }

    /// Give every passage of the shelf a hand-made embedding, and
    /// build the PLAID index over them.
    ///
    /// `place` gives the direction that the passages of one message
    /// point in. The model gives a unit vector, and these do the same,
    /// so a test puts two messages apart without a model.
    fn seed(
        shelf: &Shelf,
        dir: &TempDir,
        place: impl Fn(&MessageId) -> [f32; 2],
    ) -> (Brain, PlaidIndex) {
        let db = docbert_core::embedding_db::EmbeddingDb::open(
            &dir.path().join("embeddings.db"),
        )
        .expect("a database");
        let plan = semantic::plan(&shelf.store, ABSENT).expect("a plan");

        for work in &plan.work {
            semantic::record(&shelf.store, work).expect("a record");

            let at = place(&work.id);
            for passage in &work.passages {
                db.store(passage.key, 1, DIM, &at).expect("an embedding");
            }
        }

        let params = docbert_core::plaid::PlaidBuildParams {
            k_centroids: 2,
            nbits: 2,
            max_kmeans_iters: 50,
        };
        let at = dir.path().join("plaid");
        assert!(
            semantic::rebuild(&db, &at, &plan, params).expect("a build"),
            "the index holds no passage"
        );

        let brain =
            Brain::open(&dir.path().join("embeddings.db"), &at, Some(ABSENT))
                .expect("a brain");
        let plaid = semantic::load(&at).expect("a read").expect("an index");

        (brain, plaid)
    }

    fn ran(shelf: &Shelf, brain: &mut Brain, text: &str) -> bool {
        find(
            &shelf.store,
            &shelf.index,
            Some(brain),
            text,
            plain(),
            clock(),
        )
        .expect("a search")
        .1
    }

    fn search(
        shelf: &Shelf,
        brain: &mut Brain,
        text: &str,
    ) -> Result<(Vec<Row>, bool)> {
        find(
            &shelf.store,
            &shelf.index,
            Some(brain),
            text,
            plain(),
            clock(),
        )
    }

    #[test]
    fn the_semantic_leg_names_the_message_that_the_query_points_at() {
        let dir = tempdir().expect("a temporary directory");
        let shelf = Shelf::new();
        let near = shelf.put("a", "Alice <a@x.test>", "Deposit", "the deposit");
        let far = shelf.put("b", "Bob <b@x.test>", "Lunch", "the sandwich");

        let (_brain, plaid) = seed(&shelf, &dir, |id| match *id == near {
            true => NEAR,
            false => FAR,
        });
        let mut hits = BTreeMap::new();
        let keys = leg(
            &shelf.store,
            &shelf.index,
            &plaid,
            &point(NEAR),
            None,
            &mut hits,
        )
        .expect("a leg");

        assert_eq!(keys.first(), Some(&near.numeric()), "{keys:?}");
        assert!(hits.contains_key(&near.numeric()));
        let _ = far;
    }

    #[test]
    fn the_semantic_leg_refuses_what_the_filter_does_not_allow() {
        let dir = tempdir().expect("a temporary directory");
        let shelf = Shelf::new();
        let near = shelf.put("a", "Alice <a@x.test>", "Deposit", "the deposit");
        let far = shelf.put("b", "Bob <b@x.test>", "Lunch", "the sandwich");

        let (_brain, plaid) = seed(&shelf, &dir, |id| match *id == near {
            true => NEAR,
            false => FAR,
        });
        let allow: BTreeSet<u64> = [far.numeric()].into_iter().collect();
        let mut hits = BTreeMap::new();
        let keys = leg(
            &shelf.store,
            &shelf.index,
            &plaid,
            &point(NEAR),
            Some(&allow),
            &mut hits,
        )
        .expect("a leg");

        assert_eq!(keys, vec![far.numeric()], "{keys:?}");
    }

    #[test]
    fn the_semantic_leg_drops_a_message_that_the_index_lost() {
        let dir = tempdir().expect("a temporary directory");
        let shelf = Shelf::new();
        let lost = shelf.put("a", "Alice <a@x.test>", "Deposit", "a deposit");
        let kept = shelf.put("b", "Bob <b@x.test>", "Deposit", "a deposit");

        let (_brain, plaid) = seed(&shelf, &dir, |_| NEAR);
        let mut writer = shelf.index.writer(BUDGET).expect("a writer");
        shelf.index.remove(&writer, &lost);
        shelf.index.commit(&mut writer).expect("a commit");

        let mut hits = BTreeMap::new();
        let keys = leg(
            &shelf.store,
            &shelf.index,
            &plaid,
            &point(NEAR),
            None,
            &mut hits,
        )
        .expect("a leg");

        assert_eq!(keys, vec![kept.numeric()], "{keys:?}");
        assert!(!hits.contains_key(&lost.numeric()));
    }

    /// A `search` over an index that a pass built must reach the model.
    ///
    /// The model of the test is not there, so the search stops at it.
    /// That error is the proof that the leg ran, because nothing else
    /// in the command asks the model for anything.
    #[test]
    fn a_search_with_free_text_and_an_index_asks_the_model() {
        let dir = tempdir().expect("a temporary directory");
        let shelf = Shelf::new();
        shelf.put("a", "Alice <a@x.test>", "Deposit", "the deposit");
        shelf.put("b", "Bob <b@x.test>", "Lunch", "the sandwich");
        let (mut brain, _plaid) = seed(&shelf, &dir, |_| NEAR);

        let result = search(&shelf, &mut brain, "deposit");

        assert!(
            matches!(result, Err(crate::error::Error::Model(_))),
            "the leg never asked the model"
        );
    }

    #[test]
    fn a_search_over_mail_that_no_pass_embedded_runs_one_leg() {
        let dir = tempdir().expect("a temporary directory");
        let shelf = Shelf::new();
        shelf.put("a", "Alice <a@x.test>", "Deposit", "the deposit");
        let mut brain = Brain::open(
            &dir.path().join("embeddings.db"),
            &dir.path().join("plaid"),
            Some(ABSENT),
        )
        .expect("a brain");

        let (rows, ran) = find(
            &shelf.store,
            &shelf.index,
            Some(&mut brain),
            "deposit",
            plain(),
            clock(),
        )
        .expect("a search");

        assert!(!ran, "the leg ran over an index that is not there");
        assert_eq!(rows.len(), 1, "the lexical leg went quiet");
    }

    #[test]
    fn a_query_of_filters_alone_never_asks_the_model() {
        let dir = tempdir().expect("a temporary directory");
        let shelf = Shelf::new();
        shelf.put("a", "Alice <a@x.test>", "Deposit", "the deposit");
        shelf.put("b", "Bob <b@x.test>", "Lunch", "the sandwich");
        let (mut brain, _plaid) = seed(&shelf, &dir, |_| NEAR);

        assert!(!ran(&shelf, &mut brain, "from:a@x.test"), "the leg ran");
        assert!(!brain.model.is_loaded(), "the leg loaded a model");
    }

    // -----------------------------------------------------------------
    // The lines of §10.1.
    // -----------------------------------------------------------------

    #[test]
    fn a_line_carries_the_short_identity_and_the_day() {
        let shelf = Shelf::new();
        let id = shelf.put("a", "Alice <a@x.test>", "Deposit", "the deposit");
        let rows = shelf.find("deposit");

        let lines = shelf.lines(&rows, &terms(&["deposit"]), false);

        assert_eq!(lines[0].id, id.short());
        assert_eq!(lines[0].date, "2025-08-22");
        assert_eq!(lines[0].subject, "Deposit");
    }

    #[test]
    fn a_line_names_the_sender_by_name() {
        let shelf = Shelf::new();
        shelf.put("a", "Alice Smith <alice@x.test>", "Deposit", "the deposit");
        let rows = shelf.find("deposit");

        let lines = shelf.lines(&rows, &[], false);

        assert_eq!(lines[0].who, "Alice Smith");
    }

    #[test]
    fn a_sender_with_no_name_gives_its_address() {
        let shelf = Shelf::new();
        shelf.put("a", "alice@x.test", "Deposit", "the deposit");
        let rows = shelf.find("deposit");

        let lines = shelf.lines(&rows, &[], false);

        assert_eq!(lines[0].who, "alice@x.test");
    }

    #[test]
    fn a_line_holds_the_folder_and_then_the_tags() {
        let shelf = Shelf::new();
        let id = shelf.put_at(
            "a",
            "Alice <a@x.test>",
            "Deposit",
            "the deposit",
            "Archive",
            None,
        );
        shelf.store.tag(&id, "todo").expect("the store takes a tag");
        let rows = shelf.find("deposit");

        let lines = shelf.lines(&rows, &[], false);

        assert_eq!(lines[0].tags, terms(&["archive", "todo"]));
    }

    #[test]
    fn a_line_takes_no_snippet_when_the_flag_is_off() {
        let shelf = Shelf::new();
        shelf.put("a", "Alice <a@x.test>", "Deposit", "the deposit");
        let rows = shelf.find("deposit");

        let lines = shelf.lines(&rows, &terms(&["deposit"]), false);

        assert_eq!(lines[0].snippet, None);
    }

    #[test]
    fn a_line_takes_a_snippet_when_the_flag_is_on() {
        let shelf = Shelf::new();
        shelf.put("a", "Alice <a@x.test>", "Deposit", "the deposit is late");
        let rows = shelf.find("deposit");

        let lines = shelf.lines(&rows, &terms(&["deposit"]), true);

        let snippet = lines[0].snippet.as_deref().expect("a snippet");
        assert!(snippet.contains("deposit"), "{snippet}");
    }

    // -----------------------------------------------------------------
    // The sender.
    // -----------------------------------------------------------------

    #[test]
    fn a_message_with_no_sender_gives_an_empty_name() {
        assert_eq!(sender(&[]), "");
    }

    #[test]
    fn the_sender_is_the_first_address_of_the_header() {
        let from = vec![
            Address {
                name: Some("Alice".to_string()),
                address: "alice@x.test".to_string(),
            },
            Address {
                name: Some("Bob".to_string()),
                address: "bob@x.test".to_string(),
            },
        ];

        assert_eq!(sender(&from), "Alice");
    }

    // -----------------------------------------------------------------
    // The snippet of §10.1.
    // -----------------------------------------------------------------

    #[test]
    fn a_snippet_holds_the_word_that_the_query_asked_for() {
        let text = format!("{} deposit at the end", "filler word ".repeat(40));

        let cut = best(&text, &terms(&["deposit"]));

        assert!(cut.contains("deposit"), "{cut}");
    }

    #[test]
    fn a_snippet_of_a_body_with_no_word_gives_the_start() {
        let cut = best("the sandwich was good", &terms(&["deposit"]));

        assert!(cut.starts_with("the sandwich"), "{cut}");
    }

    #[test]
    fn a_snippet_marks_the_text_that_it_left_out() {
        let text = format!("{} deposit", "filler word ".repeat(40));

        let cut = best(&text, &terms(&["deposit"]));

        assert!(cut.starts_with('…'), "{cut}");
    }

    #[test]
    fn a_snippet_shows_the_text_in_front_of_the_word() {
        let text = format!("{}deposit", "filler word ".repeat(40));

        let cut = best(&text, &terms(&["deposit"]));

        assert!(cut.contains("filler"), "{cut}");
        assert!(cut.ends_with("deposit"), "{cut}");
    }

    #[test]
    fn a_column_of_no_width_holds_nothing() {
        assert_eq!(pad("abc", 0), "");
    }

    #[test]
    fn a_snippet_of_a_short_body_marks_nothing() {
        let cut = best("the deposit is late", &terms(&["deposit"]));

        assert_eq!(cut, "the deposit is late");
    }

    #[test]
    fn a_snippet_puts_every_line_of_the_body_on_one_line() {
        let cut = best("first line\nsecond line", &terms(&["second"]));

        assert_eq!(cut, "first line second line");
    }

    #[test]
    fn a_snippet_of_an_empty_body_is_empty() {
        assert_eq!(best("   \n  ", &terms(&["deposit"])), "");
    }

    #[test]
    fn a_snippet_reads_the_word_whatever_its_case() {
        let text = format!("{} Deposit at the end", "filler word ".repeat(40));

        let cut = best(&text, &terms(&["DEPOSIT"]));

        assert!(cut.contains("Deposit"), "{cut}");
    }

    // -----------------------------------------------------------------
    // The words of a query.
    // -----------------------------------------------------------------

    #[test]
    fn the_words_of_a_query_drop_the_filters() {
        assert_eq!(words_of("from:bob deposit"), terms(&["deposit"]));
    }

    #[test]
    fn the_words_of_a_query_lose_the_quotes_of_a_phrase() {
        assert_eq!(words_of("\"late deposit\""), terms(&["late", "deposit"]));
    }

    #[test]
    fn a_query_of_filters_alone_gives_no_word() {
        assert!(words_of("from:bob is:unread").is_empty());
    }

    // -----------------------------------------------------------------
    // The columns of §10.1.
    // -----------------------------------------------------------------

    #[test]
    fn a_short_value_takes_the_whole_column() {
        assert_eq!(pad("ab", 5), "ab   ");
    }

    #[test]
    fn a_long_value_loses_its_end_and_says_so() {
        assert_eq!(pad("abcdef", 4), "abc…");
    }

    #[test]
    fn a_column_counts_characters_and_not_bytes() {
        assert_eq!(pad("héllo", 5), "héllo");
    }

    #[test]
    fn the_text_holds_every_field_of_the_row() {
        let mut one = line("a3f9c1d2", "Alice", "Deposit");
        one.tags = terms(&["inbox", "todo"]);

        let text = text_of(&[one]);

        assert!(text.contains("a3f9c1d2"), "{text}");
        assert!(text.contains("2025-08-22"), "{text}");
        assert!(text.contains("[1/1]"), "{text}");
        assert!(text.contains("Alice"), "{text}");
        assert!(text.contains("Deposit"), "{text}");
        assert!(text.contains("(inbox todo)"), "{text}");
    }

    #[test]
    fn the_text_of_a_row_with_no_tag_writes_no_parentheses() {
        let text = text_of(&[line("a3f9c1d2", "Alice", "Deposit")]);

        assert!(!text.contains('('), "{text}");
    }

    #[test]
    fn the_text_gives_the_snippet_its_own_line() {
        let mut one = line("a3f9c1d2", "Alice", "Deposit");
        one.snippet = Some("the deposit is late".to_string());

        let text = text_of(&[one]);
        let written: Vec<&str> = text.lines().collect();

        assert_eq!(written.len(), 2, "{text}");
        assert!(written[1].trim() == "the deposit is late", "{text}");
    }

    #[test]
    fn every_row_of_the_text_shares_one_column_width() {
        let text = text_of(&[
            line("a3f9c1d2", "Alice", "Deposit"),
            line("b721e4f0", "Bartholomew", "Lunch"),
        ]);
        let written: Vec<&str> = text.lines().collect();

        let first = written[0].find("Deposit").expect("the subject");
        let second = written[1].find("Lunch").expect("the subject");

        assert_eq!(first, second, "{text}");
    }

    #[test]
    fn the_text_of_no_row_at_all_is_empty() {
        assert_eq!(text_of(&[]), "");
    }

    // -----------------------------------------------------------------
    // The JSON of §10.4.
    // -----------------------------------------------------------------

    #[test]
    fn the_json_holds_the_query_and_the_rows() {
        let answer = Answer {
            query: "deposit".to_string(),
            semantic: false,
            rows: vec![line("a3f9c1d2", "Alice", "Deposit")],
        };
        let mut out = Vec::new();
        write_json(&answer, &mut out).expect("a write");

        let text = String::from_utf8(out).expect("the output is text");
        let read: serde_json::Value =
            serde_json::from_str(&text).expect("good JSON");

        assert_eq!(read["query"], "deposit");
        assert_eq!(read["semantic"], false);
        assert_eq!(read["rows"][0]["id"], "a3f9c1d2");
        assert_eq!(read["rows"][0]["who"], "Alice");
        assert_eq!(read["rows"][0]["position"], 1);
    }

    #[test]
    fn the_json_of_a_row_with_no_snippet_holds_no_field_for_it() {
        let answer = Answer {
            query: "deposit".to_string(),
            semantic: true,
            rows: vec![line("a3f9c1d2", "Alice", "Deposit")],
        };
        let mut out = Vec::new();
        write_json(&answer, &mut out).expect("a write");

        let text = String::from_utf8(out).expect("the output is text");

        assert!(!text.contains("snippet"), "{text}");
    }

    // -----------------------------------------------------------------
    // The flags.
    // -----------------------------------------------------------------

    #[test]
    fn each_order_of_the_flag_names_one_sort() {
        assert_eq!(sort_of(cli::Order::Best), Sort::Best);
        assert_eq!(sort_of(cli::Order::Score), Sort::Score);
        assert_eq!(sort_of(cli::Order::Date), Sort::Date);
    }

    #[test]
    fn the_options_take_the_count_of_the_flag_before_the_file() {
        let config = Config::default();
        let args = cli::Find {
            words: terms(&["deposit"]),
            count: Some(3),
            sort: cli::Order::Date,
            snippet: false,
            json: false,
        };

        let options = options(&args, &config, NOW);

        assert_eq!(options.limit, 3);
        assert_eq!(options.sort, Sort::Date);
        assert_eq!(options.now, NOW);
    }

    #[test]
    fn the_options_fall_back_to_the_count_of_the_file() {
        let mut config = Config::default();
        config.search.count = 7;
        config.search.recency_half_life_days = 42.0;
        let args = cli::Find {
            words: terms(&["deposit"]),
            count: None,
            sort: cli::Order::Best,
            snippet: false,
            json: false,
        };

        let options = options(&args, &config, NOW);

        assert_eq!(options.limit, 7);
        assert!((options.half_life_days - 42.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------
    // The brain of a command. (§2.1)
    // -----------------------------------------------------------------

    /// A tool over a directory that no other test writes to.
    fn tool(dir: &TempDir) -> Tool {
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[[account]]\nname = \"work\"\nhost = \"a\"\nuser = \"b\"\n\
             password_command = \"true\"\n",
        )
        .expect("the file is writable");

        let cli = crate::cli::Cli::try_parse_from([
            "mailbert",
            "--data-dir",
            dir.path().to_str().expect("a name of text"),
            "--config",
            path.to_str().expect("a name of text"),
            "status",
        ])
        .expect("a good command line");

        Tool::open(&cli).expect("the flags give both paths")
    }

    #[test]
    fn ksearch_opens_no_brain() {
        let dir = tempdir().expect("a temporary directory");
        let tool = tool(&dir);

        let brain =
            brain_for(&tool, Legs::Words).expect("a command with one leg");

        assert!(brain.is_none(), "`ksearch` opened a model");
        assert!(
            !dir.path().join("embeddings.db").exists(),
            "`ksearch` made the embedding database"
        );
    }

    #[test]
    fn search_opens_a_brain() {
        let dir = tempdir().expect("a temporary directory");
        let tool = tool(&dir);

        let brain =
            brain_for(&tool, Legs::Both).expect("a command with two legs");

        assert!(brain.is_some(), "`search` opened no model");
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    /// Any text that a column can hold.
    #[hegel::composite]
    fn a_value(tc: TestCase) -> String {
        tc.draw(gs::text().alphabet("aé漢 ").min_size(0).max_size(30))
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_column_always_holds_its_width(tc: TestCase) {
        let text: String = tc.draw(a_value());
        let wide: usize =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(30));

        let held = pad(&text, wide);

        assert_eq!(
            held.chars().count(),
            wide,
            "`{text}` in {wide} gave `{held}`"
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_snippet_never_cuts_a_character(tc: TestCase) {
        let text: String =
            tc.draw(gs::text().alphabet("aé漢 \n").min_size(0).max_size(400));
        let words: Vec<String> =
            tc.draw(gs::vecs(a_value()).min_size(0).max_size(3));

        let cut = best(&text, &words);

        // A cut inside a character panics, so arriving here is most of
        // the property. The rest keeps the window near its size.
        assert!(cut.chars().count() <= SNIPPET + 2, "`{cut}` is too long");
    }

    #[hegel::test(test_cases = 30)]
    fn prop_a_search_names_each_thread_once(tc: TestCase) {
        let count: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
        let together: bool = tc.draw(gs::booleans());

        let shelf = Shelf::new();
        let mut thread = None;

        for one in 0..count {
            let key = format!("k{one}");
            let id = shelf.put_at(
                &key,
                "Alice <a@x.test>",
                "Deposit",
                "the deposit",
                "INBOX",
                thread,
            );

            if together && thread.is_none() {
                thread = Some(ThreadId::from_root(id));
            }
        }

        let rows = shelf.find("deposit");
        let mut seen = BTreeSet::new();

        for row in &rows {
            assert!(
                seen.insert(row.hit.thread),
                "the thread of `{}` came twice",
                row.hit.id.short()
            );
        }

        let wanted = match together {
            true => 1,
            false => count,
        };
        assert_eq!(rows.len(), wanted, "{rows:?}");
    }

    #[hegel::test(test_cases = 30)]
    fn prop_a_search_never_writes_more_rows_than_the_limit(tc: TestCase) {
        let count: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
        let limit: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(8));

        let shelf = Shelf::new();
        for one in 0..count {
            let key = format!("k{one}");
            shelf.put(&key, "Alice <a@x.test>", "Deposit", "the deposit");
        }

        let rows = shelf.find_with("deposit", plain().with_limit(limit));

        assert!(rows.len() <= limit, "{} rows for {limit}", rows.len());
        assert_eq!(rows.len(), count.min(limit));
    }
}
