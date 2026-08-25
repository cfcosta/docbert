//! The MCP server of §2.2.
//!
//! The server holds the model in memory, because the process lives a
//! long time. This removes the cold start that the CLI pays for each
//! hybrid search.
//!
//! Two tools write. `tag` writes to mailbert's own tag table, and
//! `send` writes one message out through a submission server and files
//! the copy locally. Neither writes to the IMAP server, because §3.3
//! makes mailbert a download-only mirror. No tool decrypts, because
//! §5.4 opens a body only for `view`.

use std::{
    io::Write,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use mailbert_core::{Store, config::Config, index::MailIndex, rank};
use rmcp::{
    ServerHandler,
    ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolResult,
        ContentBlock,
        Implementation,
        ServerCapabilities,
        ServerInfo,
    },
    tool,
    tool_handler,
    tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::{
    Tool,
    contacts,
    error,
    paths::Paths,
    search,
    semantic::Brain,
    send,
    show,
    status,
    tags,
    thread,
};

/// How many rows a tool gives when the caller asks for no count.
pub const COUNT: usize = 10;

/// The most rows that one tool call gives.
///
/// A caller that asks for more takes this instead. A reply of 100 mail
/// messages is already more than a model reads well.
pub const MOST: usize = 100;

/// How a search orders its rows. (§8.3)
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum By {
    /// The fused score, with the recency prior. The default.
    #[default]
    Best,

    /// The fused score alone, with no decay.
    Score,

    /// The date, newest first.
    Date,
}

impl By {
    /// The order of §8.3 that this name gives.
    pub fn sort(self) -> rank::Sort {
        match self {
            Self::Best => rank::Sort::Best,
            Self::Score => rank::Sort::Score,
            Self::Date => rank::Sort::Date,
        }
    }
}

/// What a search tool takes. (§2.2)
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct Ask {
    /// The query, in the language of §7.1.
    pub query: String,

    /// How many rows to give. The default is 10, and the most is 100.
    #[serde(default)]
    pub count: Option<usize>,

    /// Give the passage of the body that matched.
    #[serde(default)]
    pub snippet: Option<bool>,

    /// The order of the rows: `best`, `score`, or `date`.
    #[serde(default)]
    pub sort: Option<By>,
}

impl Ask {
    /// A search for one query, with the defaults of §2.2.
    pub fn new(query: &str) -> Self {
        Self {
            query: query.to_string(),
            ..Self::default()
        }
    }

    /// How many rows to give, inside the bound of `MOST`.
    pub fn count(&self) -> usize {
        self.count.unwrap_or(COUNT).clamp(1, MOST)
    }

    /// The order that the caller asked for.
    pub fn by(&self) -> rank::Sort {
        self.sort.unwrap_or_default().sort()
    }
}

/// What the server holds while it runs. (§2.2)
pub struct Desk {
    /// The messages of §4.2.
    pub store: Arc<Store>,

    /// The lexical index of §6.1.
    pub index: MailIndex,

    /// The accounts and the defaults of §1.2.
    pub config: Config,

    /// Where the data of the tool sits.
    pub paths: Paths,

    /// The model, once something asked for it. (§2.2)
    brain: Mutex<Option<Brain>>,
}

impl Desk {
    /// A desk over parts that a caller already opened.
    pub fn new(
        store: Arc<Store>,
        index: MailIndex,
        config: Config,
        paths: Paths,
    ) -> Self {
        Self {
            store,
            index,
            config,
            paths,
            brain: Mutex::new(None),
        }
    }

    /// A desk over the store, the index, and the configuration.
    ///
    /// # Errors
    ///
    /// The function fails if the store, the index, or the
    /// configuration cannot open.
    pub fn open(tool: &Tool) -> error::Result<Self> {
        Ok(Self::new(
            tool.store()?,
            tool.index()?,
            tool.config()?,
            tool.paths.clone(),
        ))
    }

    /// The model, loaded once and then held. (§2.2)
    ///
    /// # Errors
    ///
    /// The function fails if the model cannot load.
    fn mind(&self) -> error::Result<MutexGuard<'_, Option<Brain>>> {
        let mut held =
            self.brain.lock().unwrap_or_else(PoisonError::into_inner);

        if held.is_none() {
            *held = Some(Brain::open(
                &self.paths.embeddings(),
                &self.paths.plaid(),
                None,
            )?);
        }

        Ok(held)
    }

    /// True when the model sits in memory. (§2.2)
    ///
    /// The model loads on the first hybrid search, and then stays. A
    /// desk that only served `bm25_search` never loads it.
    pub fn loaded(&self) -> bool {
        self.brain
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }

    /// Run one search. (§8.1)
    ///
    /// `Legs::Words` never touches the model, so `bm25_search` costs
    /// nothing that `search` costs.
    ///
    /// # Errors
    ///
    /// The function fails if the query is bad, or if the store, the
    /// index, or the model refuses.
    pub fn find(
        &self,
        ask: &Ask,
        legs: search::Legs,
    ) -> error::Result<search::Answer> {
        let clock = crate::clock();
        let options = rank::Options::new(clock.now())
            .with_sort(ask.by())
            .with_half_life(self.config.search.recency_half_life_days)
            .with_limit(ask.count());

        // §2.2: `bm25_search` takes the lexical leg alone, so it never
        // asks for the model, and never waits for it to load.
        let mut held = match legs {
            search::Legs::Both => Some(self.mind()?),
            search::Legs::Words => None,
        };
        let brain = held.as_mut().and_then(|guard| guard.as_mut());

        let (rows, ran) = search::find(
            &self.store,
            &self.index,
            brain,
            &ask.query,
            options,
            clock,
        )?;

        Ok(search::Answer {
            rows: search::lines(
                &self.store,
                &rows,
                &search::words_of(&ask.query),
                ask.snippet.unwrap_or(false),
            )?,
            query: ask.query.clone(),
            semantic: ran,
        })
    }

    /// The headers and the body of one message. (§10.2)
    ///
    /// §5.4 never decrypts here, so an encrypted body reaches the
    /// caller as ciphertext.
    ///
    /// # Errors
    ///
    /// The function fails if the prefix names no message, or names
    /// more than one.
    pub fn get(&self, id: &str) -> error::Result<show::Whole> {
        let id = show::resolve(&self.store, id)?;
        let (head, text) =
            show::read(&self.store, &id, crate::clock().utc_offset())?;

        Ok(show::Whole { head, text })
    }

    /// Each message of the thread that one message is in. (§8.4)
    ///
    /// # Errors
    ///
    /// The function fails if the prefix names no message, or if the
    /// index does not hold it.
    pub fn thread(&self, id: &str) -> error::Result<thread::Answer> {
        let id = show::resolve(&self.store, id)?;
        let (thread, rows) = thread::of(&self.index, &id)?;

        Ok(thread::Answer {
            id: id.short(),
            thread: thread.full_hex(),
            rows: search::lines(&self.store, &rows, &[], false)?,
        })
    }

    /// The addresses that a name resolves to. (§5.6)
    ///
    /// # Errors
    ///
    /// The function fails if the store refuses.
    pub fn contacts(&self, name: &str) -> error::Result<contacts::Answer> {
        let book = contacts::book(&self.store, &contacts::mine(&self.config))?;

        Ok(contacts::find(&book, name))
    }

    /// Put tags on messages, and take tags off them. (§9)
    ///
    /// This is the only tool that writes, and it writes only to the
    /// tag table of mailbert. §3.3 never lets it reach the server.
    ///
    /// # Errors
    ///
    /// The function fails if the words are bad, or if an identity
    /// names no message.
    pub fn tag(&self, words: &[String]) -> error::Result<tags::Answer> {
        tags::apply(&self.store, &self.index, &tags::split(words)?)
    }

    /// Write one message out, and file the copy. (§11)
    ///
    /// This is the tool that reaches past the machine. It hands the
    /// message to the submission server of the account, and files the
    /// copy in mailbert's own store; §3.3 still holds, so nothing of
    /// it is written to the IMAP server.
    ///
    /// # Errors
    ///
    /// The function fails if no account can send, if the letter names
    /// no recipient or a bad address, if the server refuses the
    /// message, or if the store refuses the copy.
    pub async fn send(
        &self,
        letter: &send::Letter,
    ) -> error::Result<send::Answer> {
        send::run(&self.store, &self.index, &self.config, letter).await
    }

    /// The counts of the store, the index, and the vectors. (§10.4)
    ///
    /// # Errors
    ///
    /// The function fails if the store or the index refuses.
    pub fn status(&self) -> error::Result<status::Report> {
        status::report(&self.store, &self.index, &self.config)
    }
}

/// One message that a tool names. (§4.1)
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct Id {
    /// The identity of the message, or a git-style prefix of it.
    pub id: String,
}

/// The name that `contacts` resolves. (§5.6)
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct Name {
    /// A part of a display name, of an address, or of a domain.
    pub name: String,
}

/// The words that `tag` takes. (§9)
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct Words {
    /// Each change first, such as `+todo` or `-done`, then each
    /// identity. Example: `["+todo", "-inbox", "a1b2c3d4"]`.
    pub words: Vec<String>,
}

/// True when the caller can fix the problem.
///
/// A bad query, or an identity that names nothing, comes from the
/// caller. The model reads the message and asks again, so the report
/// must not tell it that the server broke.
fn asked(problem: &error::Error) -> bool {
    use error::Error as Bad;
    use mailbert_core::Error as Core;

    matches!(
        problem,
        Bad::Query(_)
            | Bad::AmbiguousMessage { .. }
            | Bad::NotIndexed(_)
            | Bad::NoEdits
            | Bad::NoMessages
            | Bad::LateEdit(_)
            | Bad::UnknownAccount(_)
            | Bad::BadAddress(_)
            | Bad::NoRecipient
            | Bad::ManySenders(_)
            | Bad::Core(
                Core::Query(_)
                    | Core::BadGlob(_)
                    | Core::BadFilterValue { .. }
                    | Core::InvalidTag(_)
                    | Core::InvalidSearchName(_)
                    | Core::UnknownSearch(_)
                    | Core::EmptySearch(_)
                    | Core::UnknownMessage(_)
                    | Core::NoSmtp(_)
            )
    )
}

/// The report that one tool gives when the work fails.
fn fault(problem: error::Error) -> rmcp::ErrorData {
    let text = problem.to_string();

    match asked(&problem) {
        true => rmcp::ErrorData::invalid_params(text, None),
        false => rmcp::ErrorData::internal_error(text, None),
    }
}

/// One answer, as the text that a reader sees and as its JSON.
///
/// The text costs the model less to read than the JSON. The JSON
/// carries the fields that the text leaves out, such as the score.
fn answer<T: serde::Serialize>(
    value: &T,
    write: impl FnOnce(&mut dyn Write) -> error::Result<()>,
) -> error::Result<CallToolResult> {
    let mut text = Vec::new();
    write(&mut text)?;

    let mut result = CallToolResult::success(vec![ContentBlock::text(
        String::from_utf8_lossy(&text).into_owned(),
    )]);
    result.structured_content = Some(serde_json::to_value(value)?);

    Ok(result)
}

/// The MCP server of §2.2.
#[derive(Clone)]
pub struct Server {
    desk: Arc<Desk>,
    tool_router: ToolRouter<Self>,
}

impl Server {
    /// A server over one desk.
    pub fn new(desk: Desk) -> Self {
        Self::over(Arc::new(desk))
    }

    /// A server over a desk that a caller already holds.
    pub fn over(desk: Arc<Desk>) -> Self {
        Self {
            desk,
            tool_router: sanitize(Self::tool_router()),
        }
    }
}

/// Drop the `$schema` keyword from each tool schema.
///
/// Some clients refuse a schema that holds it.
fn sanitize(mut router: ToolRouter<Server>) -> ToolRouter<Server> {
    for route in router.map.values_mut() {
        route.attr.input_schema = without_schema(&route.attr.input_schema);
        route.attr.output_schema =
            route.attr.output_schema.as_ref().map(without_schema);
    }

    router
}

/// The same schema, with no `$schema` keyword.
fn without_schema(
    schema: &Arc<serde_json::Map<String, serde_json::Value>>,
) -> Arc<serde_json::Map<String, serde_json::Value>> {
    if !schema.contains_key("$schema") {
        return schema.clone();
    }

    let mut clean = schema.as_ref().clone();
    clean.remove("$schema");

    Arc::new(clean)
}

#[tool_router(router = tool_router)]
impl Server {
    /// The hybrid search of §8.1.
    #[tool(
        name = "search",
        description = "Search the mail by meaning and by word together. Takes \
                       the full query language: from:, to:, subject:, tag:, \
                       date:, is:, has:, and AND/OR/NOT. Use this when the \
                       words of the question may not be the words of the mail."
    )]
    pub async fn search(
        &self,
        params: Parameters<Ask>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let found = self
            .desk
            .find(&params.0, search::Legs::Both)
            .map_err(fault)?;

        answer(&found, |out| search::write_text(&found.rows, out))
            .map_err(fault)
    }

    /// The lexical search of §8.1.
    #[tool(
        name = "bm25_search",
        description = "Search the mail by word alone. Takes the same query \
                       language as `search`, and never loads the model, so it \
                       answers faster. Use this for an address, an identifier, \
                       or a phrase that the mail holds word for word."
    )]
    pub async fn bm25_search(
        &self,
        params: Parameters<Ask>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let found = self
            .desk
            .find(&params.0, search::Legs::Words)
            .map_err(fault)?;

        answer(&found, |out| search::write_text(&found.rows, out))
            .map_err(fault)
    }

    /// The text of one message. (§10.2)
    #[tool(
        name = "get",
        description = "Give the headers and the text of one message. Takes the \
                       identity that a search gave, or a prefix of it. An \
                       encrypted body comes back as its ciphertext, because \
                       mailbert decrypts for `view` alone."
    )]
    pub async fn get(
        &self,
        params: Parameters<Id>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let whole = self.desk.get(&params.0.id).map_err(fault)?;

        answer(&whole, |out| {
            show::write_plain(&whole.head, &whole.text, out)
        })
        .map_err(fault)
    }

    /// Each message of one thread. (§8.4)
    #[tool(
        name = "thread",
        description = "Give one row for each message of the thread that holds \
                       a message, oldest first. Use this after a search, to \
                       read a conversation in order."
    )]
    pub async fn thread(
        &self,
        params: Parameters<Id>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let whole = self.desk.thread(&params.0.id).map_err(fault)?;

        answer(&whole, |out| search::write_text(&whole.rows, out))
            .map_err(fault)
    }

    /// The addresses of one name. (§5.6)
    #[tool(
        name = "contacts",
        description = "Give each address that a name resolves to, with how \
                       often the mail goes each way. Use this to turn a name \
                       into the address that `from:` and `to:` take."
    )]
    pub async fn contacts(
        &self,
        params: Parameters<Name>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let found = self.desk.contacts(&params.0.name).map_err(fault)?;

        answer(&found, |out| contacts::write_text(&found, out)).map_err(fault)
    }

    /// Put tags on messages, and take tags off them. (§9)
    #[tool(
        name = "tag",
        description = "Add tags to messages, and remove tags from them. Put \
                       each change first, such as `+todo` or `-done`, and then \
                       each identity. The tags live in mailbert alone, and \
                       never reach the mail server."
    )]
    pub async fn tag(
        &self,
        params: Parameters<Words>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let done = self.desk.tag(&params.0.words).map_err(fault)?;

        answer(&done, |out| tags::write_text(&done, out)).map_err(fault)
    }

    /// One message out through the submission server. (§11)
    #[tool(
        name = "send",
        description = "Send one message, and file the copy in mailbert. Give \
                       `to`, `subject`, and `body`, or give `reply_to` with \
                       the identity of a message to answer and take its \
                       sender, its subject, and its thread. `reply_all` \
                       answers everyone it named. This is the one tool that \
                       reaches past the machine, and what it sends cannot be \
                       taken back, so read the message to the person first \
                       unless they asked you to send it outright."
    )]
    pub async fn send(
        &self,
        params: Parameters<send::Letter>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let sent = self.desk.send(&params.0).await.map_err(fault)?;

        answer(&sent, |out| send::write_text(&sent, out)).map_err(fault)
    }

    /// The counts of the store and the index. (§10.4)
    #[tool(
        name = "status",
        description = "Give the counts of the store, the index, and the \
                       vectors, with the tags and the state of each folder. \
                       Use this to see whether a search can find recent mail."
    )]
    pub async fn status(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        let report = self.desk.status().map_err(fault)?;

        answer(&report, |out| {
            status::write_text(&report, crate::clock(), out)
        })
        .map_err(fault)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for Server {
    fn get_info(&self) -> ServerInfo {
        let about = Implementation::new("mailbert", env!("CARGO_PKG_VERSION"))
            .with_title("mailbert MCP")
            .with_description("Search and read your mail through MCP.");

        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(about)
            .with_instructions(
                "Pick a search tool by signal: bm25_search for an address, an \
                 identifier, or a phrase that the mail holds word for word, \
                 and search when the words of the question may differ from \
                 the words of the mail. Both take from:, to:, subject:, tag:, \
                 date:, is:, has:, and AND/OR/NOT. Turn a name into an \
                 address with contacts first. Read one message with get, and \
                 the whole conversation with thread. tag writes to mailbert \
                 alone. send is the one tool that leaves the machine: it \
                 hands a message to the submission server and files the copy \
                 locally, and neither it nor tag ever writes to the mail \
                 server.",
            )
    }
}

/// Serve the tools of §2.2 over stdio, until the client goes away.
///
/// # Errors
///
/// The function fails if the store, the index, or the transport
/// refuses.
pub fn command(tool: &Tool) -> error::Result<()> {
    let server = Server::new(Desk::open(tool)?);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async move {
        let running = server.serve(rmcp::transport::stdio()).await?;
        running.waiting().await?;

        Ok(())
    })
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_search_never_gives_more_rows_than_it_asks` | metamorphic | The count against the rows. A tool that gives 300 rows to a caller who asked for 10 fills the context of the model. |
    //! | `prop_bm25_search_never_reads_the_model` | algebraic | The loaded flag of the desk, over each shape of query. §2.2 says `bm25_search` is the cheap tool, and a model that loads in it makes the two tools cost the same. |

    use std::collections::BTreeSet;

    use hegel::{TestCase, generators as gs};
    use mailbert_core::{
        message::{Location, Message},
        message_id::MessageId,
        mime,
        threading::ThreadId,
    };
    use tempfile::{TempDir, tempdir};

    use super::*;

    /// The smallest budget that a Tantivy writer accepts.
    const BUDGET: usize = 15_000_000;

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
        desk: Arc<Desk>,
        uid: std::cell::Cell<u32>,
    }

    impl Shelf {
        fn new() -> Self {
            let dir = tempdir().expect("a temporary directory");
            let store =
                Store::open(&dir.path().join("store")).expect("a store");
            let index = MailIndex::open_in_ram().expect("an index");
            let paths = Paths {
                data: dir.path().to_path_buf(),
                config: dir.path().join("mailbert.toml"),
            };

            Self {
                desk: Arc::new(Desk::new(
                    Arc::new(store),
                    index,
                    Config::default(),
                    paths,
                )),
                _dir: dir,
                uid: std::cell::Cell::new(1),
            }
        }

        fn put(&self, key: &str, subject: &str, body: &str) -> MessageId {
            self.write(key, subject, body, None)
        }

        /// A message that reaches the store, but not the index.
        ///
        /// §3.2 writes the store first, so a sync that stops between
        /// the two leaves the index behind. `status` must show that.
        fn stash(&self, key: &str) -> MessageId {
            let uid = self.uid.get();
            self.uid.set(uid + 1);
            let bytes = self.bytes(key, "Pending", "not indexed yet", uid);
            let message = Message::new(
                mime::parse(&bytes).expect("a message"),
                location(uid),
                Vec::<String>::new(),
            );

            self.desk.store.put(&message, &bytes).expect("a write").id
        }

        fn reply(
            &self,
            key: &str,
            subject: &str,
            thread: ThreadId,
        ) -> MessageId {
            self.write(key, subject, "a reply", Some(thread))
        }

        fn write(
            &self,
            key: &str,
            subject: &str,
            body: &str,
            thread: Option<ThreadId>,
        ) -> MessageId {
            let uid = self.uid.get();
            self.uid.set(uid + 1);
            let bytes = self.bytes(key, subject, body, uid);
            let message = Message::new(
                mime::parse(&bytes).expect("a message"),
                location(uid),
                Vec::<String>::new(),
            );
            let held = self.desk.store.put(&message, &bytes).expect("a write");
            let thread = thread.unwrap_or_else(|| ThreadId::from_root(held.id));
            let tags = self
                .desk
                .store
                .tags_of(&held.id)
                .expect("the tags of the message");

            let mut writer =
                self.desk.index.writer(BUDGET).expect("an index writer");
            self.desk
                .index
                .add(&writer, &held, thread, &tags)
                .expect("an index write");
            self.desk.index.commit(&mut writer).expect("a commit");

            held.id
        }

        /// The bytes of one message. A later UID gets a later date.
        fn bytes(
            &self,
            key: &str,
            subject: &str,
            body: &str,
            uid: u32,
        ) -> Vec<u8> {
            format!(
                "From: Alice Alvarez <alice@example.test>\r\n\
                 To: bob@example.test\r\n\
                 Subject: {subject}\r\n\
                 Date: Fri, 22 Aug 2025 09:{:02}:00 +0000\r\n\
                 Message-ID: <{key}@x.test>\r\n\
                 \r\n\
                 {body}\r\n",
                uid % 60
            )
            .into_bytes()
        }

        fn server(&self) -> Server {
            Server::over(self.desk.clone())
        }

        fn ksearch(&self, query: &str) -> search::Answer {
            self.desk
                .find(&Ask::new(query), search::Legs::Words)
                .expect("a search")
        }
    }

    // -----------------------------------------------------------------
    // What a search takes. (§2.2)
    // -----------------------------------------------------------------

    #[test]
    fn a_search_with_no_count_gives_the_default() {
        assert_eq!(Ask::new("rent").count(), COUNT);
    }

    #[test]
    fn a_count_above_the_bound_takes_the_bound() {
        let ask = Ask {
            count: Some(5000),
            ..Ask::new("rent")
        };

        assert_eq!(ask.count(), MOST);
    }

    #[test]
    fn a_count_of_zero_gives_one_row() {
        let ask = Ask {
            count: Some(0),
            ..Ask::new("rent")
        };

        assert_eq!(ask.count(), 1);
    }

    #[test]
    fn a_search_with_no_order_takes_the_best() {
        assert_eq!(Ask::new("rent").by(), rank::Sort::Best);
    }

    #[test]
    fn each_order_gives_the_sort_of_the_same_name() {
        assert_eq!(By::Best.sort(), rank::Sort::Best);
        assert_eq!(By::Score.sort(), rank::Sort::Score);
        assert_eq!(By::Date.sort(), rank::Sort::Date);
    }

    /// §2.2 gives the schema to the model, so the names of the orders
    /// must read as the reader writes them.
    #[test]
    fn an_order_reads_from_its_lowercase_name() {
        let held: By =
            serde_json::from_str("\"date\"").expect("the name reads");

        assert_eq!(held, By::Date);
    }

    // -----------------------------------------------------------------
    // The search tools. (§8.1)
    // -----------------------------------------------------------------

    #[test]
    fn a_search_finds_the_message_that_holds_the_word() {
        let shelf = Shelf::new();
        let id = shelf.put("a", "Deposit", "the rent is late");
        shelf.put("b", "Lunch", "a sandwich");

        let answer = shelf.ksearch("rent");

        assert_eq!(answer.rows.len(), 1);
        assert_eq!(answer.rows[0].id, id.short());
    }

    #[test]
    fn a_search_keeps_the_query_that_the_caller_gave() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");

        assert_eq!(shelf.ksearch("rent").query, "rent");
    }

    /// §2.2 says that `bm25_search` is lexical only. A leg that runs
    /// makes it the same tool as `search`, and just as slow.
    #[test]
    fn bm25_search_never_runs_the_semantic_leg() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");

        assert!(!shelf.ksearch("rent").semantic);
    }

    /// §2.2 loads the model once and holds it. A `bm25_search` that
    /// loads it pays the cost that the cheap tool exists to avoid.
    #[test]
    fn bm25_search_never_loads_the_model() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");

        shelf.ksearch("rent");

        assert!(!shelf.desk.loaded());
    }

    #[test]
    fn a_search_gives_no_more_rows_than_the_count() {
        let shelf = Shelf::new();
        for at in 0..5 {
            shelf.put(&format!("m{at}"), "Deposit", "the rent is late");
        }
        let ask = Ask {
            count: Some(2),
            ..Ask::new("rent")
        };

        let answer = shelf
            .desk
            .find(&ask, search::Legs::Words)
            .expect("a search");

        assert_eq!(answer.rows.len(), 2);
    }

    #[test]
    fn a_search_with_no_snippet_gives_no_snippet() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");

        assert!(shelf.ksearch("rent").rows[0].snippet.is_none());
    }

    #[test]
    fn a_search_gives_the_snippet_when_it_is_asked() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");
        let ask = Ask {
            snippet: Some(true),
            ..Ask::new("rent")
        };

        let answer = shelf
            .desk
            .find(&ask, search::Legs::Words)
            .expect("a search");

        assert!(answer.rows[0].snippet.is_some());
    }

    /// §8.3 lets the caller pick the order. A tool that drops the
    /// choice gives the model an answer that it did not ask for.
    #[test]
    fn each_order_puts_a_different_message_first() {
        let shelf = Shelf::new();
        let strong = shelf.put("a", "Rent", "rent rent rent rent");
        let late = shelf.put("b", "Note", "the rent");

        let by = |sort| {
            let ask = Ask {
                sort: Some(sort),
                ..Ask::new("rent")
            };

            shelf
                .desk
                .find(&ask, search::Legs::Words)
                .expect("a search")
                .rows[0]
                .id
                .clone()
        };

        assert_eq!(by(By::Score), strong.short(), "score wants the match");
        assert_eq!(by(By::Date), late.short(), "date wants the newest");
    }

    #[test]
    fn a_query_that_does_not_read_is_an_error() {
        let shelf = Shelf::new();

        let result = shelf.desk.find(&Ask::new("from:"), search::Legs::Words);

        assert!(
            matches!(result, Err(crate::error::Error::Query(_))),
            "{result:?}"
        );
    }

    // -----------------------------------------------------------------
    // The message tools. (§10.2, §8.4)
    // -----------------------------------------------------------------

    #[test]
    fn get_gives_the_headers_and_the_body() {
        let shelf = Shelf::new();
        let id = shelf.put("a", "Deposit", "the rent is late");

        let whole = shelf.desk.get(&id.short()).expect("a message");

        assert_eq!(whole.head.subject, "Deposit");
        assert!(whole.text.contains("the rent is late"), "{}", whole.text);
    }

    #[test]
    fn get_takes_a_prefix_of_an_identity() {
        let shelf = Shelf::new();
        let id = shelf.put("a", "Deposit", "the rent is late");

        let whole = shelf.desk.get(&id.full_hex()[..4]).expect("a message");

        assert_eq!(whole.head.id, id.full_hex());
    }

    #[test]
    fn get_of_a_prefix_that_names_nothing_is_an_error() {
        let shelf = Shelf::new();

        let result = shelf.desk.get("ffffffffffffffff");

        assert!(result.is_err(), "{result:?}");
    }

    /// §5.4 decrypts for `view` alone. The index and its backups are
    /// plaintext files, and an MCP tool that decrypts would put the
    /// plaintext into the reply of a model that logs it.
    #[test]
    fn get_never_decrypts_an_encrypted_body() {
        let shelf = Shelf::new();
        let id = shelf.put(
            "a",
            "Secret",
            "-----BEGIN PGP MESSAGE-----\n\nhQIMA0abcdef\n             -----END PGP MESSAGE-----",
        );

        let whole = shelf.desk.get(&id.short()).expect("a message");

        assert!(whole.head.encrypted, "the message reads as plaintext");
        assert!(whole.text.contains("BEGIN PGP MESSAGE"), "{}", whole.text);
        assert!(!whole.text.contains("hello"), "{}", whole.text);
    }

    #[test]
    fn thread_gives_each_message_of_the_conversation() {
        let shelf = Shelf::new();
        let root = shelf.put("a", "Deposit", "the rent is late");
        shelf.reply("b", "Re: Deposit", ThreadId::from_root(root));
        shelf.reply("c", "Re: Deposit", ThreadId::from_root(root));

        let answer = shelf.desk.thread(&root.short()).expect("a thread");

        assert_eq!(answer.rows.len(), 3);
        assert_eq!(answer.id, root.short());
    }

    #[test]
    fn thread_of_a_prefix_that_names_nothing_is_an_error() {
        let shelf = Shelf::new();

        let result = shelf.desk.thread("ffffffffffffffff");

        assert!(result.is_err(), "{result:?}");
    }

    // -----------------------------------------------------------------
    // The other tools. (§5.6, §9, §10.4)
    // -----------------------------------------------------------------

    #[test]
    fn contacts_resolves_a_name_to_an_address() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");

        let answer = shelf.desk.contacts("alvarez").expect("the contacts");

        assert_eq!(answer.people.len(), 1);
        assert_eq!(answer.people[0].address, "alice@example.test");
    }

    #[test]
    fn tag_puts_a_tag_on_a_message() {
        let shelf = Shelf::new();
        let id = shelf.put("a", "Deposit", "the rent is late");

        let answer = shelf
            .desk
            .tag(&["+todo".to_string(), id.short()])
            .expect("a change");

        assert_eq!(answer.messages.len(), 1);
        assert_eq!(answer.messages[0].tags, vec!["todo".to_string()]);
    }

    /// §6.1 keeps a tag in the `flags` field, so the search tools must
    /// see a tag that the tag tool wrote.
    #[test]
    fn a_tag_reaches_the_search_tools() {
        let shelf = Shelf::new();
        let id = shelf.put("a", "Deposit", "the rent is late");
        shelf
            .desk
            .tag(&["+todo".to_string(), id.short()])
            .expect("a change");

        assert_eq!(shelf.ksearch("tag:todo").rows.len(), 1);
    }

    #[test]
    fn a_tag_command_with_no_change_is_an_error() {
        let shelf = Shelf::new();
        let id = shelf.put("a", "Deposit", "the rent is late");

        let result = shelf.desk.tag(&[id.short()]);

        assert!(
            matches!(result, Err(crate::error::Error::NoEdits)),
            "{result:?}"
        );
    }

    #[test]
    fn status_counts_the_store_and_the_index() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");
        shelf.put("b", "Lunch", "a sandwich");

        let report = shelf.desk.status().expect("a report");

        assert_eq!(report.messages, 2);
        assert_eq!(report.indexed, 2);
        assert_eq!(report.behind(), 0);
    }

    /// §10.4 tells the reader when the index is behind the store. The
    /// two counts must come from the two places, and not from one.
    #[test]
    fn status_shows_the_index_behind_the_store() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");
        shelf.stash("b");

        let report = shelf.desk.status().expect("a report");

        assert_eq!(report.messages, 2);
        assert_eq!(report.indexed, 1);
        assert_eq!(report.behind(), 1);
    }

    // -----------------------------------------------------------------
    // The server. (§2.2)
    // -----------------------------------------------------------------

    fn names(server: &Server) -> Vec<String> {
        let mut held: Vec<String> = server
            .tool_router
            .map
            .keys()
            .map(|name| name.to_string())
            .collect();
        held.sort();

        held
    }

    #[test]
    fn the_server_gives_the_eight_tools_of_the_design() {
        let shelf = Shelf::new();

        assert_eq!(
            names(&shelf.server()),
            vec![
                "bm25_search",
                "contacts",
                "get",
                "search",
                "send",
                "status",
                "tag",
                "thread",
            ]
        );
    }

    /// Some clients refuse a schema that holds the `$schema` keyword.
    #[test]
    fn no_tool_schema_holds_the_schema_keyword() {
        let shelf = Shelf::new();
        let server = shelf.server();

        for (name, route) in &server.tool_router.map {
            assert!(
                !route.attr.input_schema.contains_key("$schema"),
                "{name} keeps the keyword"
            );
        }
    }

    /// §2.2 tells the model where each write goes. A model that thinks
    /// a tag reaches the server writes to the mail of the reader,
    /// which §3.3 never allows, and a model that does not know `send`
    /// leaves the machine sends mail it should have shown first.
    #[test]
    fn the_instructions_say_where_each_write_goes() {
        let shelf = Shelf::new();
        let told = shelf
            .server()
            .get_info()
            .instructions
            .expect("the server tells the model how to work");

        assert!(told.contains("tag writes to mailbert alone"), "{told}");
        assert!(
            told.contains("send is the one tool that leaves the machine"),
            "{told}"
        );
        assert!(
            told.contains("neither it nor tag ever writes to the mail server"),
            "{told}"
        );
    }

    #[tokio::test]
    async fn a_tool_gives_the_text_and_the_fields() {
        let shelf = Shelf::new();
        let id = shelf.put("a", "Deposit", "the rent is late");

        let result = shelf
            .server()
            .bm25_search(Parameters(Ask::new("rent")))
            .await
            .expect("a search");

        let text = format!("{:?}", result.content);
        assert!(text.contains(&id.short()), "{text}");

        let held = result.structured_content.expect("the fields of the rows");
        assert_eq!(held["rows"][0]["id"], id.short());
        assert_eq!(held["semantic"], false);
    }

    /// §2.2 gives the model two search tools so that the caller can
    /// pick the cheap one. A `bm25_search` that loads the model costs
    /// what `search` costs, and the choice means nothing.
    #[tokio::test]
    async fn the_cheap_search_tool_never_loads_the_model() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");

        shelf
            .server()
            .bm25_search(Parameters(Ask::new("rent")))
            .await
            .expect("a search");

        assert!(!shelf.desk.loaded());
    }

    /// §2.2 holds the model because the process is long-lived. A
    /// `search` that does not hold it pays the load on each call.
    #[tokio::test]
    async fn the_hybrid_search_tool_holds_the_model() {
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");

        shelf
            .server()
            .search(Parameters(Ask::new("rent")))
            .await
            .expect("a search");

        assert!(shelf.desk.loaded());
    }

    #[tokio::test]
    async fn a_tool_that_writes_gives_the_new_tags() {
        let shelf = Shelf::new();
        let id = shelf.put("a", "Deposit", "the rent is late");

        let result = shelf
            .server()
            .tag(Parameters(Words {
                words: vec!["+todo".to_string(), id.short()],
            }))
            .await
            .expect("a change");

        let held = result.structured_content.expect("the changed messages");
        assert_eq!(held["messages"][0]["tags"][0], "todo");
    }

    /// A model reads the code of the report. `internal_error` tells it
    /// that the server broke, so it gives up instead of asking again.
    #[tokio::test]
    async fn a_bad_query_reads_as_the_fault_of_the_caller() {
        let shelf = Shelf::new();

        let problem = shelf
            .server()
            .search(Parameters(Ask::new("from:")))
            .await
            .expect_err("`from:` has no value");

        assert_eq!(problem.code, rmcp::model::ErrorCode::INVALID_PARAMS);
    }

    #[test]
    fn a_missing_directory_does_not_read_as_the_fault_of_the_caller() {
        assert!(!asked(&crate::error::Error::NoDataDir));
        assert_ne!(
            fault(crate::error::Error::NoDataDir).code,
            rmcp::model::ErrorCode::INVALID_PARAMS
        );
    }

    // -----------------------------------------------------------------
    // Properties of the tools.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 20)]
    fn prop_a_search_never_gives_more_rows_than_it_asks(tc: TestCase) {
        let held = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
        let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
        let shelf = Shelf::new();

        for at in 0..held {
            shelf.put(&format!("m{at}"), "Deposit", "the rent is late");
        }

        let ask = Ask {
            count: Some(count),
            ..Ask::new("rent")
        };
        let answer = shelf
            .desk
            .find(&ask, search::Legs::Words)
            .expect("a search");

        assert!(
            answer.rows.len() <= count,
            "{} rows for a count of {count}",
            answer.rows.len()
        );
    }

    #[hegel::test(test_cases = 20)]
    fn prop_bm25_search_never_reads_the_model(tc: TestCase) {
        let words = tc.draw(gs::sampled_from(vec![
            "rent".to_string(),
            "sandwich".to_string(),
            "tag:todo".to_string(),
            "from:alice".to_string(),
        ]));
        let shelf = Shelf::new();
        shelf.put("a", "Deposit", "the rent is late");

        assert!(!shelf.ksearch(&words).semantic, "a leg ran");
        assert!(!shelf.desk.loaded(), "the model loaded");
    }
}
