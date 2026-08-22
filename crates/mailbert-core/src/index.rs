//! The lexical index: the schema of §6.1, and the writer that fills it.
//!
//! mailbert owns its Tantivy schema. This is the difference from
//! rustbert, which uses `docbert_core::SearchIndex` and its fixed
//! schema of a path, a title, and a body. Mail has no path, and it has
//! an account, a folder, a sender, a thread, and a set of flags that a
//! search must filter on.
//!
//! Every field that a filter reads is `FAST`. §8.2 says why, and the
//! reason is worth repeating. A filter must gate the two legs before
//! they rank, and must never remove results after the ranking. If
//! mailbert ranked first and then removed what does not match, then
//! `from:bob invoice` would give nothing when the invoice of Bob is at
//! rank 300. That failure is silent, and it looks the same as mail
//! that is not there.
//!
//! This module writes the index and reads a document back. It does not
//! build a query. §8.2 compiles a query into a filter, and gives that
//! filter to [`MailIndex::top`].

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use tantivy::{
    DocId,
    Index,
    IndexReader,
    IndexSettings,
    IndexWriter,
    ReloadPolicy,
    TantivyDocument,
    Term,
    collector::{DocSetCollector, TopDocs},
    directory::MmapDirectory,
    query::{Query, QueryParser, TermQuery},
    schema::{
        FAST,
        Field,
        IndexRecordOption,
        STORED,
        STRING,
        Schema,
        TextFieldIndexing,
        TextOptions,
        Value,
    },
    tokenizer::{
        LowerCaser,
        RemoveLongFilter,
        SimpleTokenizer,
        Stemmer,
        TextAnalyzer,
    },
};

use crate::{
    error::{Error, Result},
    message::{self, Message},
    message_id::MessageId,
    query::Flag,
    threading::ThreadId,
};

/// The names of the fields of §6.1.
pub mod fields {
    /// The message identity of §4.1, in hex. (STRING, STORED)
    pub const MID_HASH: &str = "mid_hash";
    /// The key into the embedding database. (u64, STORED, FAST)
    pub const NUM_ID: &str = "num_id";
    /// The accounts that hold a copy. (STRING, FAST, multi)
    pub const ACCOUNT: &str = "account";
    /// The folders that hold a copy. (STRING, FAST, multi)
    pub const FOLDER: &str = "folder";
    /// The sender addresses. (STRING, FAST, multi)
    pub const FROM_ADDR: &str = "from_addr";
    /// The sender display names. (TEXT)
    pub const FROM_NAME: &str = "from_name";
    /// The recipient addresses, To and Cc. (STRING, FAST, multi)
    pub const TO_ADDR: &str = "to_addr";
    /// The subject, with a 2x boost. (TEXT, STORED)
    pub const SUBJECT: &str = "subject";
    /// The text after §5.2 removes the quotes. (TEXT)
    pub const BODY: &str = "body";
    /// The `List-Id` header. (STRING, FAST)
    pub const LIST_ID: &str = "list_id";
    /// Seconds since the Unix epoch. (u64, FAST, STORED)
    pub const DATE: &str = "date";
    /// The thread of §5.5. (STRING, FAST, STORED)
    pub const THREAD_ID: &str = "thread_id";
    /// The IMAP flags, the tags, and the states. (STRING, FAST, multi)
    pub const FLAGS: &str = "flags";
    /// The attachment filenames. (TEXT, STORED)
    pub const ATTACHMENT: &str = "attachment";

    /// Every name, for a caller that walks the schema.
    pub const ALL: [&str; 14] = [
        MID_HASH, NUM_ID, ACCOUNT, FOLDER, FROM_ADDR, FROM_NAME, TO_ADDR,
        SUBJECT, BODY, LIST_ID, DATE, THREAD_ID, FLAGS, ATTACHMENT,
    ];

    /// The names that a filter of §8.2 reads. Each must be `FAST`.
    pub const FILTERED: [&str; 9] = [
        NUM_ID, ACCOUNT, FOLDER, FROM_ADDR, TO_ADDR, LIST_ID, DATE, THREAD_ID,
        FLAGS,
    ];
}

/// The `flags` term for a message that no header carries.
///
/// A tag can never start with `\`, so these never collide with a tag.
/// See [`crate::store::normalize_tag`].
pub const ENCRYPTED: &str = r"\encrypted";

/// The `flags` term for a message that no folder holds any more.
pub const GONE: &str = r"\gone";

/// The `flags` term for mail to a list, or for automatic mail.
pub const BULK: &str = r"\bulk";

/// The `flags` term for a message that carries an attachment.
pub const ATTACHMENT: &str = r"\attachment";

/// The name of the analyzer that the `TEXT` fields use.
pub const ANALYZER: &str = "en_stem";

/// The boost that §6.1 gives the subject.
pub const SUBJECT_BOOST: f32 = 2.0;

/// The longest token that the analyzer keeps.
const TOKEN_LIMIT: usize = 40;

/// What the `flags` field must hold, or must not hold, for one `is:`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagTerm {
    /// The field holds this term.
    Holds(&'static str),

    /// The field does not hold this term.
    Lacks(&'static str),
}

/// The resolved field handles of the schema.
#[derive(Debug, Clone, Copy)]
pub struct Fields {
    pub mid_hash: Field,
    pub num_id: Field,
    pub account: Field,
    pub folder: Field,
    pub from_addr: Field,
    pub from_name: Field,
    pub to_addr: Field,
    pub subject: Field,
    pub body: Field,
    pub list_id: Field,
    pub date: Field,
    pub thread_id: Field,
    pub flags: Field,
    pub attachment: Field,
}

impl Fields {
    /// Find each field of §6.1 in a schema.
    ///
    /// An index that opens from disk carries the schema that made it,
    /// so read the handles from that schema and not from a new one.
    fn resolve(schema: &Schema) -> Result<Self> {
        Ok(Self {
            mid_hash: schema.get_field(fields::MID_HASH)?,
            num_id: schema.get_field(fields::NUM_ID)?,
            account: schema.get_field(fields::ACCOUNT)?,
            folder: schema.get_field(fields::FOLDER)?,
            from_addr: schema.get_field(fields::FROM_ADDR)?,
            from_name: schema.get_field(fields::FROM_NAME)?,
            to_addr: schema.get_field(fields::TO_ADDR)?,
            subject: schema.get_field(fields::SUBJECT)?,
            body: schema.get_field(fields::BODY)?,
            list_id: schema.get_field(fields::LIST_ID)?,
            date: schema.get_field(fields::DATE)?,
            thread_id: schema.get_field(fields::THREAD_ID)?,
            flags: schema.get_field(fields::FLAGS)?,
            attachment: schema.get_field(fields::ATTACHMENT)?,
        })
    }
}

/// One row that the index gives back.
///
/// It holds what §6.1 stores, and nothing more. The name of the sender
/// and the tags come from the store, because the index does not keep
/// them.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    /// The BM25 score.
    pub score: f32,

    /// The identity of §4.1.
    pub id: MessageId,

    /// The key into the embedding database.
    pub num_id: u64,

    /// The subject, as the reader sees it.
    pub subject: String,

    /// Seconds since the Unix epoch.
    pub date: i64,

    /// The thread of §5.5.
    pub thread: ThreadId,
}

/// The lexical index of §6.1.
pub struct MailIndex {
    index: Index,
    reader: IndexReader,
    fields: Fields,
}

/// The schema of §6.1.
fn build_schema() -> Schema {
    let mut builder = Schema::builder();

    builder.add_text_field(fields::MID_HASH, STRING | STORED);
    builder.add_u64_field(fields::NUM_ID, STORED | FAST);
    builder.add_text_field(fields::ACCOUNT, STRING | FAST);
    builder.add_text_field(fields::FOLDER, STRING | FAST);
    builder.add_text_field(fields::FROM_ADDR, STRING | FAST);
    builder.add_text_field(fields::FROM_NAME, words());
    builder.add_text_field(fields::TO_ADDR, STRING | FAST);
    builder.add_text_field(fields::SUBJECT, words().set_stored());
    builder.add_text_field(fields::BODY, words());
    builder.add_text_field(fields::LIST_ID, STRING | FAST);
    builder.add_u64_field(fields::DATE, STORED | FAST);
    builder.add_text_field(fields::THREAD_ID, STRING | STORED | FAST);
    builder.add_text_field(fields::FLAGS, STRING | FAST);
    builder.add_text_field(fields::ATTACHMENT, words().set_stored());

    builder.build()
}

/// The options of a field that holds words, and not one token.
///
/// Positions are necessary, because §7.1 gives the reader a phrase.
fn words() -> TextOptions {
    TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer(ANALYZER)
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    )
}

/// The analyzer of the fields that hold words.
///
/// It is the analyzer of docbert, so the two tools rank alike.
fn analyzer() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(TOKEN_LIMIT))
        .filter(LowerCaser)
        .filter(Stemmer::new(tantivy::tokenizer::Language::English))
        .build()
}

/// The text of the first value of a field.
fn text_of(doc: &TantivyDocument, field: Field) -> Option<&str> {
    doc.get_first(field).and_then(|value| value.as_str())
}

/// The number of the first value of a field.
fn u64_of(doc: &TantivyDocument, field: Field) -> Option<u64> {
    doc.get_first(field).and_then(|value| value.as_u64())
}

/// The term of the `flags` field that answers one `is:` question.
///
/// `is:unread` is the one question that no term answers, because a
/// server marks what a reader has read and not what is unread.
///
/// # Examples
///
/// ```
/// use mailbert_core::index::{FlagTerm, flag_term};
/// use mailbert_core::query::Flag;
///
/// assert_eq!(flag_term(Flag::Read), FlagTerm::Holds(r"\seen"));
/// assert_eq!(flag_term(Flag::Unread), FlagTerm::Lacks(r"\seen"));
/// ```
pub fn flag_term(flag: Flag) -> FlagTerm {
    match flag {
        Flag::Read => FlagTerm::Holds(message::SEEN),
        Flag::Unread => FlagTerm::Lacks(message::SEEN),
        Flag::Flagged => FlagTerm::Holds(message::FLAGGED),
        Flag::Replied => FlagTerm::Holds(message::ANSWERED),
        Flag::Draft => FlagTerm::Holds(message::DRAFT),
        Flag::Encrypted => FlagTerm::Holds(ENCRYPTED),
        Flag::Gone => FlagTerm::Holds(GONE),
        Flag::Bulk => FlagTerm::Holds(BULK),
    }
}

/// Everything that the `flags` field holds for one message.
///
/// The field holds three kinds of term. The IMAP flags come from the
/// server. The tags come from the user (§9). The states come from the
/// message itself, because `is:encrypted`, `is:gone`, `is:bulk`, and
/// `has:attachment` must be fast-field predicates, and no header
/// carries them.
pub fn flag_terms(
    message: &Message,
    tags: &BTreeSet<String>,
) -> BTreeSet<String> {
    let mut terms: BTreeSet<String> = message.flags.iter().cloned().collect();
    terms.extend(tags.iter().cloned());

    if message.is_encrypted() {
        terms.insert(ENCRYPTED.to_string());
    }
    if message.is_gone() {
        terms.insert(GONE.to_string());
    }
    if message.is_bulk {
        terms.insert(BULK.to_string());
    }
    if !message.attachments.is_empty() {
        terms.insert(ATTACHMENT.to_string());
    }

    terms
}

impl MailIndex {
    /// Open the index in `dir`, and make it when it is not there.
    ///
    /// # Examples
    ///
    /// ```
    /// # let dir = tempfile::tempdir().unwrap();
    /// use mailbert_core::MailIndex;
    ///
    /// let index = MailIndex::open(&dir.path().join("tantivy")).unwrap();
    ///
    /// assert!(index.is_empty());
    /// ```
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let directory = MmapDirectory::open(dir)?;

        let index = if Index::exists(&directory)? {
            Index::open(directory)?
        } else {
            Index::create(directory, build_schema(), IndexSettings::default())?
        };

        Self::wrap(index)
    }

    /// Make an index in memory. A test uses this.
    ///
    /// # Examples
    ///
    /// ```
    /// use mailbert_core::MailIndex;
    ///
    /// let index = MailIndex::open_in_ram().unwrap();
    ///
    /// assert_eq!(index.len(), 0);
    /// ```
    pub fn open_in_ram() -> Result<Self> {
        Self::wrap(Index::create_in_ram(build_schema()))
    }

    /// Give an index its analyzer, its handles, and its reader.
    fn wrap(index: Index) -> Result<Self> {
        index.tokenizers().register(ANALYZER, analyzer());
        let fields = Fields::resolve(&index.schema())?;

        // The reader reloads when [`MailIndex::commit`] tells it to.
        // The policy that reloads on a timer makes a test flaky, and
        // gives a reader of the tool no gain, because the tool commits
        // and searches in one process.
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        Ok(Self {
            index,
            reader,
            fields,
        })
    }

    /// The field handles of the schema.
    pub fn fields(&self) -> Fields {
        self.fields
    }

    /// The Tantivy index behind this one.
    ///
    /// §8.2 compiles a filter into a query against this index, and
    /// gives the query to [`MailIndex::top`].
    pub fn tantivy(&self) -> &Index {
        &self.index
    }

    /// Make a writer with a memory budget in bytes.
    pub fn writer(&self, budget: usize) -> Result<IndexWriter> {
        Ok(self.index.writer(budget)?)
    }

    /// A parser for the free text of a query.
    ///
    /// §7.1 says that a term with no field prefix is free text. This
    /// parser reads that text over the four fields that hold words a
    /// reader remembers, and gives the subject the boost of §6.1.
    ///
    /// §8.2 builds the filter, and joins it to what this parser gives.
    pub fn parser(&self) -> QueryParser {
        let mut parser = QueryParser::for_index(
            &self.index,
            vec![
                self.fields.subject,
                self.fields.body,
                self.fields.from_name,
                self.fields.attachment,
            ],
        );

        parser.set_field_boost(self.fields.subject, SUBJECT_BOOST);
        parser
    }

    /// Commit the writer, and make the new documents visible.
    pub fn commit(&self, writer: &mut IndexWriter) -> Result<()> {
        writer.commit()?;
        self.reader.reload()?;

        Ok(())
    }

    /// Add a message, and replace the document that it had before.
    ///
    /// Every sync writes each message again, so the write must replace
    /// and not repeat. The identity of §4.1 is the key of the replace.
    pub fn add(
        &self,
        writer: &IndexWriter,
        message: &Message,
        thread: ThreadId,
        tags: &BTreeSet<String>,
    ) -> Result<()> {
        let at = self.fields;
        let hex = message.id.full_hex();
        self.remove(writer, &message.id);

        let mut doc = TantivyDocument::default();
        doc.add_text(at.mid_hash, &hex);
        doc.add_u64(at.num_id, message.id.numeric());

        // §4.2: one document holds every place that has a copy.
        for account in message.accounts() {
            doc.add_text(at.account, account);
        }
        for folder in message.folders() {
            doc.add_text(at.folder, folder);
        }

        for address in &message.from {
            doc.add_text(at.from_addr, &address.address);

            if let Some(name) = &address.name {
                doc.add_text(at.from_name, name);
            }
        }

        // §7.1: `to:` reads the Cc line as well as the To line.
        for address in message.to.iter().chain(message.cc.iter()) {
            doc.add_text(at.to_addr, &address.address);
        }

        doc.add_text(at.subject, &message.subject);

        // §5.4: the index is a plaintext file, and so is a backup of
        // it. Ciphertext must never become a term. The writer refuses
        // the body itself, and does not trust the pipeline above it.
        if !message.is_encrypted() {
            doc.add_text(at.body, &message.text);
        }

        if let Some(list_id) = &message.list_id {
            doc.add_text(at.list_id, list_id);
        }

        // A `Date` header before the epoch has no place on a timeline
        // of mail, so it becomes the epoch itself.
        doc.add_u64(at.date, message.date.max(0) as u64);
        doc.add_text(at.thread_id, thread.full_hex());

        for term in flag_terms(message, tags) {
            doc.add_text(at.flags, &term);
        }

        // §5.3: the index holds the filename, and not the bytes.
        for name in message
            .attachments
            .iter()
            .filter_map(|it| it.name.as_deref())
        {
            doc.add_text(at.attachment, name);
        }

        writer.add_document(doc)?;

        Ok(())
    }

    /// Remove the document of one message.
    ///
    /// The removal becomes visible at the next [`MailIndex::commit`].
    pub fn remove(&self, writer: &IndexWriter, id: &MessageId) {
        let term = Term::from_field_text(self.fields.mid_hash, &id.full_hex());
        writer.delete_term(term);
    }

    /// How many documents the index holds.
    pub fn len(&self) -> usize {
        self.reader.searcher().num_docs() as usize
    }

    /// Whether the index holds no document.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The document of one identity.
    pub fn get(&self, id: &MessageId) -> Result<Option<Hit>> {
        let query = TermQuery::new(
            Term::from_field_text(self.fields.mid_hash, &id.full_hex()),
            IndexRecordOption::Basic,
        );

        Ok(self.top(&query, 1)?.into_iter().next())
    }

    /// Run a query that a caller built, and give back the best rows.
    ///
    /// §8.2 builds that query, and it already holds the filter. This
    /// is why the search removes nothing after the ranking.
    pub fn top(&self, query: &dyn Query, limit: usize) -> Result<Vec<Hit>> {
        let searcher = self.reader.searcher();
        let ranked = TopDocs::with_limit(limit).order_by_score();
        let found = searcher.search(query, &ranked)?;
        let mut hits = Vec::with_capacity(found.len());

        for (score, address) in found {
            let doc: TantivyDocument = searcher.doc(address)?;
            hits.push(self.read(score, &doc)?);
        }

        Ok(hits)
    }

    /// Every embedding key that a filter allows.
    ///
    /// §8.2 gates the semantic leg with this list, before that leg
    /// ranks. The keys are the `num_id` of §6.1, because that is what
    /// the embedding database is keyed by.
    pub fn allow(&self, filter: &dyn Query) -> Result<BTreeSet<u64>> {
        let searcher = self.reader.searcher();
        let found = searcher.search(filter, &DocSetCollector)?;

        // Group by segment, so the fast field opens one time for each
        // segment and not one time for each document.
        let mut by_segment: BTreeMap<u32, Vec<DocId>> = BTreeMap::new();
        for address in found {
            by_segment
                .entry(address.segment_ord)
                .or_default()
                .push(address.doc_id);
        }

        let mut allowed = BTreeSet::new();
        for (ord, docs) in by_segment {
            let segment = searcher.segment_reader(ord);
            let column = segment.fast_fields().u64(fields::NUM_ID)?;

            for doc in docs {
                let key = column.first(doc).ok_or(Error::BrokenDocument)?;
                allowed.insert(key);
            }
        }

        Ok(allowed)
    }

    /// Make a row from the stored fields of a document.
    fn read(&self, score: f32, doc: &TantivyDocument) -> Result<Hit> {
        let at = self.fields;
        let broken = || Error::BrokenDocument;

        let hex = text_of(doc, at.mid_hash).ok_or_else(broken)?;
        let id = MessageId::from_hex(hex).ok_or_else(broken)?;

        let hex = text_of(doc, at.thread_id).ok_or_else(broken)?;
        let root = MessageId::from_hex(hex).ok_or_else(broken)?;

        Ok(Hit {
            score,
            id,
            num_id: u64_of(doc, at.num_id).ok_or_else(broken)?,
            subject: text_of(doc, at.subject).unwrap_or_default().to_string(),
            date: u64_of(doc, at.date).ok_or_else(broken)? as i64,
            thread: ThreadId::from_root(root),
        })
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_the_index_agrees_with_matches` | differential | `is:` reads the index and `Message::matches` reads the message. A disagreement makes a filter drop mail without a word. |
    //! | `prop_a_message_reads_back_from_the_index` | round-trip | Every row of the output comes from a stored field. A wrong field shows the wrong mail. |
    //! | `prop_every_place_finds_the_message` | model-based | `folder:` and `account:` must find every copy, because §4.2 keeps one document for many places. |
    //! | `prop_a_second_write_keeps_one_document` | algebraic | Every re-sync writes each message again, and must not make a duplicate. |
    //! | `prop_a_tag_is_never_a_state` | algebraic | A tag that reads as `\encrypted` would answer `is:encrypted` for mail that is not encrypted. |

    use hegel::{TestCase, generators as gs};
    use tantivy::query::{BooleanQuery, Occur};
    use tempfile::tempdir;

    use super::*;
    use crate::{
        message::{ANSWERED, DRAFT, FLAGGED, Location, SEEN},
        mime::{self, Attachment, Source},
        store::normalize_tag,
    };

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    const DAY: i64 = 86_400;

    /// The smallest budget that a Tantivy writer accepts.
    const BUDGET: usize = 15_000_000;

    fn location(account: &str, folder: &str, uid: u32) -> Location {
        Location {
            account: account.to_string(),
            folder: folder.to_string(),
            uid,
            uid_validity: 1,
            received: 100 * DAY,
        }
    }

    fn raw_bytes(key: &str, subject: &str, body: &str) -> Vec<u8> {
        format!(
            "From: Alice Smith <alice@example.test>\r\n\
             To: bob@example.test\r\n\
             Cc: carol@example.test\r\n\
             Subject: {subject}\r\n\
             Date: Fri, 14 Aug 2026 09:30:00 +0000\r\n\
             Message-ID: <{key}@x.test>\r\n\
             \r\n\
             {body}\r\n"
        )
        .into_bytes()
    }

    fn message(key: &str, account: &str, folder: &str) -> Message {
        let raw = raw_bytes(key, "Deposit", "The deposit is due.");

        Message::new(
            mime::parse(&raw).expect("a message"),
            location(account, folder, 1),
            [SEEN],
        )
    }

    fn attachment() -> Attachment {
        Attachment {
            name: Some("report.pdf".to_string()),
            content_type: "application/pdf".to_string(),
            size: 1024,
        }
    }

    fn no_tags() -> BTreeSet<String> {
        BTreeSet::new()
    }

    /// Write one message, and make it visible.
    fn write(index: &MailIndex, found: &Message) {
        let mut writer = index.writer(BUDGET).expect("a writer");
        let thread = ThreadId::from_root(found.id);

        index
            .add(&writer, found, thread, &no_tags())
            .expect("a write");
        index.commit(&mut writer).expect("a commit");
    }

    /// A query for one term of a `STRING` field.
    fn term_query(index: &MailIndex, field: Field, text: &str) -> TermQuery {
        let _ = index;

        TermQuery::new(
            Term::from_field_text(field, text),
            IndexRecordOption::Basic,
        )
    }

    /// How many documents one query finds.
    fn count(index: &MailIndex, query: &dyn Query) -> usize {
        index.top(query, 100).expect("a search").len()
    }

    // -----------------------------------------------------------------
    // The schema of §6.1.
    // -----------------------------------------------------------------

    #[test]
    fn the_schema_has_every_field_of_the_design() {
        let index = MailIndex::open_in_ram().expect("an index");
        let schema = index.tantivy().schema();

        for name in fields::ALL {
            assert!(schema.get_field(name).is_ok(), "`{name}` is not there");
        }

        assert_eq!(schema.fields().count(), fields::ALL.len());
    }

    #[test]
    fn every_field_that_a_filter_reads_is_fast() {
        // §8.2: a filter must be a fast-field predicate. A field that
        // is not FAST makes the filter a post-filter, and a post-filter
        // drops mail silently.
        let index = MailIndex::open_in_ram().expect("an index");
        let schema = index.tantivy().schema();

        for name in fields::FILTERED {
            let field = schema.get_field(name).expect("a field");
            let entry = schema.get_field_entry(field);

            assert!(entry.is_fast(), "`{name}` must be FAST — see §8.2");
        }
    }

    #[test]
    fn stores_only_what_a_row_of_the_output_needs() {
        const STORED_FIELDS: [&str; 6] = [
            fields::MID_HASH,
            fields::NUM_ID,
            fields::SUBJECT,
            fields::DATE,
            fields::THREAD_ID,
            fields::ATTACHMENT,
        ];

        let index = MailIndex::open_in_ram().expect("an index");
        let schema = index.tantivy().schema();

        for name in fields::ALL {
            let field = schema.get_field(name).expect("a field");
            let entry = schema.get_field_entry(field);

            assert_eq!(
                entry.is_stored(),
                STORED_FIELDS.contains(&name),
                "`{name}` has the wrong STORED setting"
            );
        }
    }

    #[test]
    fn indexes_every_field_that_holds_a_term() {
        // The two numbers are fast-field predicates, and not terms.
        const NOT_INDEXED: [&str; 2] = [fields::NUM_ID, fields::DATE];

        let index = MailIndex::open_in_ram().expect("an index");
        let schema = index.tantivy().schema();

        for name in fields::ALL {
            let field = schema.get_field(name).expect("a field");
            let entry = schema.get_field_entry(field);

            assert_eq!(
                entry.is_indexed(),
                !NOT_INDEXED.contains(&name),
                "`{name}` has the wrong INDEXED setting"
            );
        }
    }

    // -----------------------------------------------------------------
    // The terms of the `flags` field.
    // -----------------------------------------------------------------

    #[test]
    fn every_flag_of_the_query_language_has_a_term() {
        assert_eq!(flag_term(Flag::Read), FlagTerm::Holds(SEEN));
        assert_eq!(flag_term(Flag::Unread), FlagTerm::Lacks(SEEN));
        assert_eq!(flag_term(Flag::Flagged), FlagTerm::Holds(FLAGGED));
        assert_eq!(flag_term(Flag::Replied), FlagTerm::Holds(ANSWERED));
        assert_eq!(flag_term(Flag::Draft), FlagTerm::Holds(DRAFT));
        assert_eq!(flag_term(Flag::Encrypted), FlagTerm::Holds(ENCRYPTED));
        assert_eq!(flag_term(Flag::Gone), FlagTerm::Holds(GONE));
        assert_eq!(flag_term(Flag::Bulk), FlagTerm::Holds(BULK));
    }

    #[test]
    fn the_flags_field_holds_the_imap_flags_the_tags_and_the_states() {
        let mut found = message("a", "work", "INBOX");
        found.add_flag(r"\Flagged");
        found.is_bulk = true;
        found.attachments.push(attachment());

        let tags = BTreeSet::from(["todo".to_string()]);
        let terms = flag_terms(&found, &tags);

        // The server gives these two.
        assert!(terms.contains(SEEN));
        assert!(terms.contains(FLAGGED));

        // The user gives this one.
        assert!(terms.contains("todo"));

        // The message itself gives these two.
        assert!(terms.contains(BULK));
        assert!(terms.contains(ATTACHMENT));

        // Nothing made these two true.
        assert!(!terms.contains(ENCRYPTED));
        assert!(!terms.contains(GONE));
    }

    #[test]
    fn the_flags_field_marks_a_message_that_no_folder_holds() {
        let mut found = message("a", "work", "INBOX");
        found.locations.clear();

        let terms = flag_terms(&found, &no_tags());

        assert!(terms.contains(GONE));
    }

    #[test]
    fn the_flags_field_marks_a_message_that_is_encrypted() {
        let mut found = message("a", "work", "INBOX");
        found.source = Source::Encrypted;

        let terms = flag_terms(&found, &no_tags());

        assert!(terms.contains(ENCRYPTED));
    }

    #[test]
    fn a_tag_can_never_look_like_a_state() {
        // A tag goes into the same field as a state, so the two
        // vocabularies must not touch. `normalize_tag` refuses `\`.
        for state in [SEEN, FLAGGED, ENCRYPTED, GONE, BULK, ATTACHMENT] {
            assert_eq!(normalize_tag(state), None, "`{state}` reads as a tag");
        }
    }

    // -----------------------------------------------------------------
    // Write, replace, and remove.
    // -----------------------------------------------------------------

    #[test]
    fn a_new_index_is_empty() {
        let index = MailIndex::open_in_ram().expect("an index");

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn reads_back_the_row_of_a_message_that_it_wrote() {
        let index = MailIndex::open_in_ram().expect("an index");
        let found = message("a", "work", "INBOX");
        write(&index, &found);

        let hit = index.get(&found.id).expect("a read").expect("a row");

        assert_eq!(hit.id, found.id);
        assert_eq!(hit.num_id, found.id.numeric());
        assert_eq!(hit.subject, "Deposit");
        assert_eq!(hit.date, found.date);
        assert_eq!(hit.thread, ThreadId::from_root(found.id));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn reads_nothing_for_a_message_that_is_not_there() {
        let index = MailIndex::open_in_ram().expect("an index");
        let found = message("a", "work", "INBOX");

        assert_eq!(index.get(&found.id).expect("a read"), None);
    }

    #[test]
    fn a_second_write_replaces_the_document() {
        let index = MailIndex::open_in_ram().expect("an index");
        let found = message("a", "work", "INBOX");
        write(&index, &found);

        let mut again = found.clone();
        again.add_location(location("work", "Archive", 2));
        write(&index, &again);

        let fields = index.fields();

        assert_eq!(index.len(), 1);
        assert_eq!(
            count(&index, &term_query(&index, fields.folder, "Archive")),
            1
        );
    }

    #[test]
    fn removes_the_document_of_a_message() {
        let index = MailIndex::open_in_ram().expect("an index");
        let found = message("a", "work", "INBOX");
        write(&index, &found);

        let mut writer = index.writer(BUDGET).expect("a writer");
        index.remove(&writer, &found.id);
        index.commit(&mut writer).expect("a commit");

        assert!(index.is_empty());
        assert_eq!(index.get(&found.id).expect("a read"), None);
    }

    #[test]
    fn one_message_in_two_folders_is_one_document() {
        // §4.2: one message, many locations. Both folders must find it,
        // and the count must stay at one.
        let index = MailIndex::open_in_ram().expect("an index");
        let mut found = message("a", "work", "INBOX");
        found.add_location(location("personal", "Archive", 7));
        write(&index, &found);

        let fields = index.fields();

        assert_eq!(index.len(), 1);
        for folder in ["INBOX", "Archive"] {
            let query = term_query(&index, fields.folder, folder);
            assert_eq!(count(&index, &query), 1, "`folder:{folder}` missed");
        }
        for account in ["work", "personal"] {
            let query = term_query(&index, fields.account, account);
            assert_eq!(count(&index, &query), 1, "`account:{account}` missed");
        }
    }

    #[test]
    fn opens_an_index_that_is_already_on_disk() {
        let dir = tempdir().expect("a directory");
        let found = message("a", "work", "INBOX");

        {
            let index = MailIndex::open(dir.path()).expect("an index");
            write(&index, &found);
        }

        let index = MailIndex::open(dir.path()).expect("an index");

        assert_eq!(index.len(), 1);
        assert!(index.get(&found.id).expect("a read").is_some());
    }

    // -----------------------------------------------------------------
    // Ciphertext never enters the index.
    // -----------------------------------------------------------------

    #[test]
    fn the_index_never_writes_the_body_of_an_encrypted_message() {
        // §5.4: the index and its backups are plaintext files, so the
        // ciphertext of a message must never become a term. The writer
        // refuses the body itself, and does not trust the pipeline.
        let index = MailIndex::open_in_ram().expect("an index");
        let mut found = message("a", "work", "INBOX");
        found.source = Source::Encrypted;
        found.text = "hedgehog ciphertext".to_string();
        write(&index, &found);

        let fields = index.fields();
        let leak = term_query(&index, fields.flags, ENCRYPTED);
        let query = index.parser().parse_query("hedgehog").expect("a query");

        assert_eq!(count(&index, &*query), 0, "the index holds ciphertext");
        assert_eq!(count(&index, &leak), 1);
    }

    // -----------------------------------------------------------------
    // Reading a row, and a query that a caller built.
    // -----------------------------------------------------------------

    #[test]
    fn the_subject_outranks_the_body() {
        let index = MailIndex::open_in_ram().expect("an index");
        let mut writer = index.writer(BUDGET).expect("a writer");

        let subject_raw = raw_bytes("a", "Quarterly hedgehog", "Nothing here.");
        let body_raw = raw_bytes("b", "Nothing here", "Quarterly hedgehog.");

        let mut wrote = Vec::new();
        for raw in [&subject_raw, &body_raw] {
            let found = Message::new(
                mime::parse(raw).expect("a message"),
                location("work", "INBOX", 1),
                [SEEN],
            );
            let thread = ThreadId::from_root(found.id);

            index
                .add(&writer, &found, thread, &no_tags())
                .expect("a write");
            wrote.push(found);
        }
        index.commit(&mut writer).expect("a commit");

        let query = index.parser().parse_query("hedgehog").expect("a query");
        let hits = index.top(&*query, 10).expect("a search");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, wrote[0].id, "the boost of §6.1 is not there");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn free_text_reads_the_name_of_the_sender() {
        let index = MailIndex::open_in_ram().expect("an index");
        write(&index, &message("a", "work", "INBOX"));

        let query = index.parser().parse_query("Alice").expect("a query");

        assert_eq!(count(&index, &*query), 1);
    }

    #[test]
    fn free_text_reads_the_name_of_an_attachment() {
        let index = MailIndex::open_in_ram().expect("an index");
        let mut found = message("a", "work", "INBOX");
        found.attachments.push(attachment());
        write(&index, &found);

        let query = index.parser().parse_query("report.pdf").expect("a query");

        assert_eq!(count(&index, &*query), 1);
    }

    #[test]
    fn runs_a_filter_that_a_caller_built() {
        // This is the shape that §8.2 makes: a BooleanQuery of the
        // fast fields, which gates the BM25 leg before it ranks.
        let index = MailIndex::open_in_ram().expect("an index");
        let mut writer = index.writer(BUDGET).expect("a writer");

        let read = message("a", "work", "INBOX");
        let mut unread = message("b", "work", "INBOX");
        unread.flags.clear();

        for found in [&read, &unread] {
            let thread = ThreadId::from_root(found.id);
            index
                .add(&writer, found, thread, &no_tags())
                .expect("a write");
        }
        index.commit(&mut writer).expect("a commit");

        let fields = index.fields();
        let query = BooleanQuery::new(vec![
            (
                Occur::Must,
                Box::new(term_query(&index, fields.folder, "INBOX"))
                    as Box<dyn Query>,
            ),
            (
                Occur::MustNot,
                Box::new(term_query(&index, fields.flags, SEEN))
                    as Box<dyn Query>,
            ),
        ]);

        let hits = index.top(&query, 10).expect("a search");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, unread.id);
    }

    #[test]
    fn finds_a_message_by_the_address_of_its_sender() {
        let index = MailIndex::open_in_ram().expect("an index");
        write(&index, &message("a", "work", "INBOX"));

        let fields = index.fields();
        let from = term_query(&index, fields.from_addr, "alice@example.test");
        let to = term_query(&index, fields.to_addr, "carol@example.test");

        assert_eq!(count(&index, &from), 1);
        assert_eq!(count(&index, &to), 1, "`to:` must read Cc as well");
    }

    #[test]
    fn finds_every_message_of_one_thread() {
        let index = MailIndex::open_in_ram().expect("an index");
        let mut writer = index.writer(BUDGET).expect("a writer");

        let root = message("a", "work", "INBOX");
        let reply = message("b", "work", "INBOX");
        let thread = ThreadId::from_root(root.id);

        for found in [&root, &reply] {
            index
                .add(&writer, found, thread, &no_tags())
                .expect("a write");
        }
        index.commit(&mut writer).expect("a commit");

        let fields = index.fields();
        let query = term_query(&index, fields.thread_id, &thread.full_hex());

        assert_eq!(count(&index, &query), 2);
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    fn a_key() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ])
    }

    fn a_folder() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "INBOX".to_string(),
            "Archive".to_string(),
            "Sent".to_string(),
        ])
    }

    fn an_account() -> impl gs::Generator<String> {
        gs::sampled_from(vec!["work".to_string(), "personal".to_string()])
    }

    fn a_flag() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            SEEN.to_string(),
            FLAGGED.to_string(),
            ANSWERED.to_string(),
            DRAFT.to_string(),
        ])
    }

    fn a_tag() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "todo".to_string(),
            "rent".to_string(),
            "bills".to_string(),
        ])
    }

    /// A message in any of the states that `is:` names.
    #[hegel::composite]
    fn a_message(tc: TestCase) -> (Message, BTreeSet<String>) {
        let key = tc.draw(a_key());
        let account = tc.draw(an_account());
        let folder = tc.draw(a_folder());
        let mut found = message(&key, &account, &folder);

        found.flags.clear();
        for flag in tc.draw(gs::vecs(a_flag()).min_size(0).max_size(4)) {
            found.add_flag(&flag);
        }

        if tc.draw(gs::booleans()) {
            found.source = Source::Encrypted;
            found.text = String::new();
        }

        if tc.draw(gs::booleans()) {
            found.add_location(location("personal", "Archive", 9));
        }

        if tc.draw(gs::booleans()) {
            found.locations.clear();
        }

        if tc.draw(gs::booleans()) {
            found.attachments.push(attachment());
        }

        found.is_bulk = tc.draw(gs::booleans());

        let tags = tc.draw(gs::vecs(a_tag()).min_size(0).max_size(3));

        (found, tags.into_iter().collect())
    }

    #[hegel::test(test_cases = 100)]
    fn prop_the_index_agrees_with_matches(tc: TestCase) {
        let (found, tags) = tc.draw(a_message());
        let terms = flag_terms(&found, &tags);

        for flag in Flag::ALL {
            let indexed = match flag_term(flag) {
                FlagTerm::Holds(term) => terms.contains(term),
                FlagTerm::Lacks(term) => !terms.contains(term),
            };

            assert_eq!(
                indexed,
                found.matches(flag),
                "`is:{}` reads one way in the index and another in the \
                 message",
                flag.name()
            );
        }
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_tag_is_never_a_state(tc: TestCase) {
        let (found, tags) = tc.draw(a_message());
        let terms = flag_terms(&found, &tags);

        for tag in &tags {
            assert!(terms.contains(tag));
            assert!(!tag.starts_with('\\'), "`{tag}` reads as a state");
        }
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_message_reads_back_from_the_index(tc: TestCase) {
        let (found, tags) = tc.draw(a_message());
        let index = MailIndex::open_in_ram().expect("an index");
        let mut writer = index.writer(BUDGET).expect("a writer");
        let thread = ThreadId::from_root(found.id);

        index.add(&writer, &found, thread, &tags).expect("a write");
        index.commit(&mut writer).expect("a commit");

        let hit = index.get(&found.id).expect("a read").expect("a row");

        assert_eq!(hit.id, found.id);
        assert_eq!(hit.num_id, found.id.numeric());
        assert_eq!(hit.subject, found.subject);
        assert_eq!(hit.date, found.date);
        assert_eq!(hit.thread, thread);
    }

    #[hegel::test(test_cases = 40)]
    fn prop_every_place_finds_the_message(tc: TestCase) {
        let (found, tags) = tc.draw(a_message());
        let index = MailIndex::open_in_ram().expect("an index");
        let mut writer = index.writer(BUDGET).expect("a writer");
        let thread = ThreadId::from_root(found.id);

        index.add(&writer, &found, thread, &tags).expect("a write");
        index.commit(&mut writer).expect("a commit");

        let fields = index.fields();
        for folder in found.folders() {
            let query = term_query(&index, fields.folder, folder);
            assert_eq!(count(&index, &query), 1, "`folder:{folder}` missed");
        }
        for account in found.accounts() {
            let query = term_query(&index, fields.account, account);
            assert_eq!(count(&index, &query), 1, "`account:{account}` missed");
        }
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_second_write_keeps_one_document(tc: TestCase) {
        let (found, tags) = tc.draw(a_message());
        let index = MailIndex::open_in_ram().expect("an index");
        let thread = ThreadId::from_root(found.id);

        for _ in 0..2 {
            let mut writer = index.writer(BUDGET).expect("a writer");
            index.add(&writer, &found, thread, &tags).expect("a write");
            index.commit(&mut writer).expect("a commit");
        }

        assert_eq!(index.len(), 1);
    }
}
