//! One IMAP connection, and the commands that a sync needs.
//!
//! The connection reads and never writes to the server. It has no
//! `SELECT`, so a folder always opens read-only, and it fetches with
//! `BODY.PEEK[]`, which sets no `\Seen` flag. See `docs/mailbert.md` §3.

use std::{collections::BTreeSet, fmt, time::Duration};

use tokio::{
    io::{AsyncWriteExt, BufReader, ReadHalf, WriteHalf, split},
    time::{Instant, timeout},
};

use crate::{
    error::{Error, Result},
    sequence::{LAST, UidSet},
    stream::Stream,
    token::{Token, encode},
    wire::{Tags, read_answer},
};

/// How many UIDs one fetch asks for. (§3.1 "ranges of a few hundred")
pub const BATCH: u32 = 300;

/// The most UIDs that one `VANISHED` line may name.
///
/// A server that names more than this is broken, or hostile. The
/// connection drops the line instead of filling the memory of the
/// machine with it.
const MOST_GONE: u64 = 5_000_000;

/// How a command ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// The server did the command.
    Ok,
    /// The server understood the command, and refused it.
    No,
    /// The server did not understand the command.
    Bad,
}

/// The answer to one command.
#[derive(Debug, Clone)]
pub struct Answer {
    pub state: State,
    pub text: String,
    pub lines: Vec<Vec<Token>>,
}

impl Answer {
    /// This answer, when the server said OK.
    pub fn ok(self) -> Result<Self> {
        match self.state {
            State::Ok => Ok(self),
            State::No => Err(Error::No(self.text)),
            State::Bad => Err(Error::Bad(self.text)),
        }
    }

    /// Each untagged line whose first word is this name.
    pub fn each(&self, name: &str) -> Vec<&[Token]> {
        self.lines
            .iter()
            .filter(|line| word(line, 1) == name.to_ascii_uppercase())
            .map(Vec::as_slice)
            .collect()
    }
}

/// One folder, as `LIST` names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Folder {
    pub name: String,
    pub separator: Option<char>,
    pub attributes: Vec<String>,
}

impl Folder {
    /// True when the folder can hold mail.
    ///
    /// A folder with `\Noselect` is only a name above other folders.
    pub fn holds_mail(&self) -> bool {
        !self
            .attributes
            .iter()
            .any(|name| name.eq_ignore_ascii_case("\\Noselect"))
    }
}

/// What `EXAMINE` says about a folder. (§3.3)
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct View {
    pub name: String,
    pub exists: u32,
    pub uid_validity: u32,
    pub uid_next: u32,
    pub highest_mod_seq: u64,
}

impl View {
    /// The set of UIDs that the folder can hold.
    pub fn all(&self) -> UidSet {
        UidSet::range(1, self.uid_next.saturating_sub(1))
    }
}

/// One message, as `UID FETCH` gives it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Fetched {
    pub uid: u32,
    pub mod_seq: u64,
    pub size: u64,
    pub flags: Vec<String>,
    pub internal_date: Option<String>,
    pub body: Vec<u8>,
}

/// What one `UID FETCH` gave.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Batch {
    pub messages: Vec<Fetched>,
    pub gone: Vec<u32>,
}

/// One connection to one server.
pub struct Connection {
    reader: BufReader<ReadHalf<Stream>>,
    writer: WriteHalf<Stream>,
    tags: Tags,
    capabilities: BTreeSet<String>,
    selected: Option<String>,
    tls: bool,
}

impl fmt::Debug for Connection {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.debug_struct("Connection")
            .field("tls", &self.tls)
            .field("selected", &self.selected)
            .field("capabilities", &self.capabilities.len())
            .finish()
    }
}

impl Connection {
    /// Open a connection, and read the greeting.
    pub async fn open(host: &str, port: u16, tls: bool) -> Result<Self> {
        let stream = Stream::open(host, port, tls).await?;
        let tls = stream.is_tls();
        let (reader, writer) = split(stream);

        let mut connection = Self {
            reader: BufReader::new(reader),
            writer,
            tags: Tags::new(),
            capabilities: BTreeSet::new(),
            selected: None,
            tls,
        };
        connection.greet().await?;

        Ok(connection)
    }

    /// True when the connection has TLS on it.
    pub fn is_tls(&self) -> bool {
        self.tls
    }

    /// True when the server announced this capability.
    pub fn can(&self, name: &str) -> bool {
        self.capabilities.contains(&name.to_ascii_uppercase())
    }

    pub fn capabilities(&self) -> &BTreeSet<String> {
        &self.capabilities
    }

    /// The folder that is open, when one is open.
    pub fn selected(&self) -> Option<&str> {
        self.selected.as_deref()
    }

    /// Send one command, and read the answer to it.
    pub async fn run(&mut self, words: &[Token]) -> Result<Answer> {
        let tag = self.tags.next_tag();
        let raw = pipelined(
            std::slice::from_ref(&tag),
            std::slice::from_ref(&words.to_vec()),
        );
        self.writer.write_all(&raw).await?;
        self.writer.flush().await?;

        Ok(self.collect(&[tag]).await?.1)
    }

    /// Send each command at once, and read every answer. (§3.1)
    ///
    /// The commands go out in one write. The connection does not wait
    /// for one answer before it sends the next command, which is where
    /// a sync wins most of its time.
    pub async fn pipeline(
        &mut self,
        commands: &[Vec<Token>],
    ) -> Result<Vec<Answer>> {
        if commands.is_empty() {
            return Ok(Vec::new());
        }

        let tags: Vec<String> =
            commands.iter().map(|_| self.tags.next_tag()).collect();
        self.writer.write_all(&pipelined(&tags, commands)).await?;
        self.writer.flush().await?;

        let mut answers: Vec<Option<Answer>> =
            (0..commands.len()).map(|_| None).collect();
        for _ in 0..commands.len() {
            let (tag, answer) = self.collect(&tags).await?;
            let Some(at) = tags.iter().position(|held| *held == tag) else {
                return Err(Error::Malformed(format!("the tag `{tag}`")));
            };
            answers[at] = Some(answer);
        }

        answers
            .into_iter()
            .map(|answer| {
                answer.ok_or_else(|| {
                    Error::Malformed("an answer that never came".to_string())
                })
            })
            .collect()
    }

    /// Ask the server which capabilities it has.
    pub async fn capability(&mut self) -> Result<()> {
        self.run(&[Token::Atom("CAPABILITY".to_string())])
            .await?
            .ok()?;

        Ok(())
    }

    /// Give the name and the password of the account.
    pub async fn login(&mut self, user: &str, password: &str) -> Result<()> {
        // A control character in a credential could add a second
        // command to the line that goes to the server.
        for text in [user, password] {
            if text.chars().any(|letter| letter.is_control()) {
                return Err(Error::Malformed(
                    "a name or a password with a control character in it"
                        .to_string(),
                ));
            }
        }

        let before = self.capabilities.clone();
        self.run(&[
            Token::Atom("LOGIN".to_string()),
            Token::Quoted(user.to_string()),
            Token::Quoted(password.to_string()),
        ])
        .await?
        .ok()?;

        // RFC 3501 §6.2.3: the capabilities can change after a login.
        if self.capabilities == before {
            self.capability().await?;
        }

        Ok(())
    }

    /// Turn on each extension, when the server has `ENABLE`.
    pub async fn enable(&mut self, names: &[&str]) -> Result<()> {
        let names: Vec<&&str> =
            names.iter().filter(|name| self.can(name)).collect();
        if names.is_empty() || !self.can("ENABLE") {
            return Ok(());
        }

        let mut words = vec![Token::Atom("ENABLE".to_string())];
        words.extend(
            names
                .into_iter()
                .map(|name| Token::Atom((*name).to_string())),
        );
        self.run(&words).await?.ok()?;

        Ok(())
    }

    /// Every folder of the account.
    pub async fn folders(&mut self) -> Result<Vec<Folder>> {
        let answer = self
            .run(&[
                Token::Atom("LIST".to_string()),
                Token::Quoted(String::new()),
                Token::Quoted("*".to_string()),
            ])
            .await?
            .ok()?;

        Ok(answer.each("LIST").into_iter().filter_map(folder).collect())
    }

    /// Open a folder read-only, and read the state of it. (§3.3)
    ///
    /// The command is `EXAMINE`, never `SELECT`. A folder that opens
    /// read-only cannot take a flag by accident.
    pub async fn examine(&mut self, name: &str) -> Result<View> {
        self.selected = None;
        let answer = self
            .run(&[
                Token::Atom("EXAMINE".to_string()),
                Token::Quoted(name.to_string()),
            ])
            .await?
            .ok()?;

        let mut view = View {
            name: name.to_string(),
            ..View::default()
        };
        for line in &answer.lines {
            if word(line, 2) == "EXISTS"
                && let Some(count) = line.get(1).and_then(Token::number)
            {
                view.exists = count.min(u64::from(u32::MAX)) as u32;
            }
            if let Some(number) = code(line, "UIDVALIDITY") {
                view.uid_validity = number.min(u64::from(u32::MAX)) as u32;
            }
            if let Some(number) = code(line, "UIDNEXT") {
                view.uid_next = number.min(u64::from(u32::MAX)) as u32;
            }
            if let Some(number) = code(line, "HIGHESTMODSEQ") {
                view.highest_mod_seq = number;
            }
        }

        self.selected = Some(name.to_string());

        Ok(view)
    }

    /// Fetch each message of a set, with the body of it. (§3.2)
    ///
    /// `since` names a `MODSEQ`. With it, the server sends only the
    /// messages that changed after that point, and names the messages
    /// that went away. (§3.3)
    pub async fn fetch(
        &mut self,
        set: &UidSet,
        since: Option<u64>,
    ) -> Result<Batch> {
        if set.is_empty() {
            return Ok(Batch::default());
        }

        let answer = self.run(&self.command(set, since)).await?.ok()?;
        let mut batch = Batch::default();

        for line in &answer.lines {
            if word(line, 1) == "VANISHED" {
                batch.gone.extend(vanished(line));
            }
            if word(line, 2) == "FETCH"
                && let Some(Token::List(items)) = line.get(3)
            {
                batch.messages.push(message(items));
            }
        }
        batch.messages.sort_by_key(|message| message.uid);

        Ok(batch)
    }

    /// Wait until the open folder changes, or until the wait ends.
    ///
    /// This is what makes `--watch` cheap: the server speaks first, and
    /// mailbert asks for nothing until it does. (§3.1)
    ///
    /// The answer is true when the server reported a change. A server
    /// with no `IDLE`, and a connection with no open folder, report
    /// nothing and give false at once, so the caller falls back to a
    /// timed pass.
    ///
    /// `IDLE` reads. It sets no flag, and it moves no message. (§3)
    ///
    /// # Errors
    ///
    /// The function fails if the connection breaks, or if the server
    /// refuses the command.
    pub async fn idle(&mut self, wait: Duration) -> Result<bool> {
        if !self.can("IDLE") || self.selected.is_none() {
            return Ok(false);
        }

        let tag = self.tags.next_tag();
        let words = [Token::Atom(tag.clone()), Token::Atom("IDLE".to_string())];
        self.writer.write_all(&encode(&words)).await?;
        self.writer.flush().await?;

        let news = self.listen(wait).await?;

        // `DONE` ends the wait. The tagged line must leave the
        // connection here, or the next command reads it as its own.
        self.writer
            .write_all(
                b"DONE
",
            )
            .await?;
        self.writer.flush().await?;
        self.collect(&[tag]).await?.1.ok()?;

        Ok(news)
    }

    /// Read the lines of one `IDLE`, until one of them is news.
    async fn listen(&mut self, wait: Duration) -> Result<bool> {
        let end = Instant::now() + wait;

        loop {
            let left = end.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Ok(false);
            }

            // The wait that ends first is the answer `false`, and not
            // an error. Nothing arrived, which is what a quiet mailbox
            // looks like.
            let Ok(line) = timeout(left, read_answer(&mut self.reader)).await
            else {
                return Ok(false);
            };

            let line = line?;
            self.learn(&line);

            if news(&line) {
                return Ok(true);
            }
        }
    }

    /// Say goodbye, and close the connection.
    pub async fn logout(&mut self) -> Result<()> {
        self.run(&[Token::Atom("LOGOUT".to_string())]).await?;
        self.writer.shutdown().await?;

        Ok(())
    }

    /// The words of one `UID FETCH`.
    fn command(&self, set: &UidSet, since: Option<u64>) -> Vec<Token> {
        let mut items = vec![
            Token::Atom("UID".to_string()),
            Token::Atom("FLAGS".to_string()),
            Token::Atom("RFC822.SIZE".to_string()),
            Token::Atom("INTERNALDATE".to_string()),
        ];
        if self.can("CONDSTORE") || self.can("QRESYNC") {
            items.push(Token::Atom("MODSEQ".to_string()));
        }
        // `BODY.PEEK[]` reads the whole message and sets no `\Seen`
        // flag. `BODY[]` sets one, so mailbert never sends it. (§3)
        items.push(Token::Atom("BODY.PEEK[]".to_string()));

        let mut words = vec![
            Token::Atom("UID".to_string()),
            Token::Atom("FETCH".to_string()),
            Token::Atom(set.to_string()),
            Token::List(items),
        ];

        if let Some(since) = since {
            let mut extra = vec![
                Token::Atom("CHANGEDSINCE".to_string()),
                Token::Atom(since.to_string()),
            ];
            if self.can("QRESYNC") {
                extra.push(Token::Atom("VANISHED".to_string()));
            }
            words.push(Token::List(extra));
        }

        words
    }

    /// Read the greeting, and stop when the server refuses.
    async fn greet(&mut self) -> Result<()> {
        let line = read_answer(&mut self.reader).await?;

        match word(&line, 1).as_str() {
            "OK" | "PREAUTH" => {
                self.learn(&line);

                Ok(())
            }
            _ => Err(Error::Refused(rest(&line, 2))),
        }
    }

    /// Read until a line with one of these tags, and give the answer.
    async fn collect(&mut self, tags: &[String]) -> Result<(String, Answer)> {
        let mut lines = Vec::new();

        loop {
            let line = read_answer(&mut self.reader).await?;
            self.learn(&line);

            let Some(tag) = line
                .first()
                .and_then(Token::text)
                .filter(|text| tags.iter().any(|held| held == text))
                .map(str::to_string)
            else {
                lines.push(line);
                continue;
            };

            let state = match word(&line, 1).as_str() {
                "OK" => State::Ok,
                "NO" => State::No,
                _ => State::Bad,
            };

            return Ok((
                tag,
                Answer {
                    state,
                    text: rest(&line, 2),
                    lines,
                },
            ));
        }
    }

    /// Take the capabilities out of a line that names them.
    fn learn(&mut self, line: &[Token]) {
        if word(line, 1) == "CAPABILITY" {
            self.capabilities = names(&line[2.min(line.len())..]);

            return;
        }

        for token in line {
            if let Token::Section(items) = token
                && word(items, 0) == "CAPABILITY"
            {
                self.capabilities = names(&items[1.min(items.len())..]);
            }
        }
    }
}

/// True when this untagged line says that the folder changed.
///
/// `EXISTS` names a message that arrived, `EXPUNGE` and `VANISHED` name
/// one that went away, and `FETCH` names a flag that moved. (§3.1)
fn news(line: &[Token]) -> bool {
    matches!(word(line, 2).as_str(), "EXISTS" | "EXPUNGE" | "FETCH")
        || word(line, 1) == "VANISHED"
}

/// The bytes of a group of commands, ready for one write. (§3.1)
pub fn pipelined(tags: &[String], commands: &[Vec<Token>]) -> Vec<u8> {
    let mut out = Vec::new();

    for (tag, words) in tags.iter().zip(commands) {
        let mut line = vec![Token::Atom(tag.clone())];
        line.extend(words.iter().cloned());
        out.extend_from_slice(&encode(&line));
    }

    out
}

/// The word at this place, in capitals.
fn word(tokens: &[Token], at: usize) -> String {
    tokens
        .get(at)
        .and_then(Token::text)
        .unwrap_or_default()
        .to_ascii_uppercase()
}

/// The text of the tokens from this place, as the server wrote it.
fn rest(tokens: &[Token], from: usize) -> String {
    if from >= tokens.len() {
        return String::new();
    }

    let raw = encode(&tokens[from..]);

    String::from_utf8_lossy(&raw).trim_end().to_string()
}

/// The text of a token, whatever kind it is.
fn any_text(token: &Token) -> Option<String> {
    match token {
        Token::Atom(text) | Token::Quoted(text) => Some(text.clone()),
        Token::Literal(raw) => Some(String::from_utf8_lossy(raw).to_string()),
        _ => None,
    }
}

/// Each name of a list, in capitals.
fn names(tokens: &[Token]) -> BTreeSet<String> {
    tokens
        .iter()
        .filter_map(Token::text)
        .map(str::to_ascii_uppercase)
        .collect()
}

/// The number of a response code, such as `[UIDNEXT 42]`.
fn code(line: &[Token], name: &str) -> Option<u64> {
    for token in line {
        if let Token::Section(items) = token
            && word(items, 0) == name.to_ascii_uppercase()
        {
            return items.get(1).and_then(Token::number);
        }
    }

    None
}

/// One folder, out of a `LIST` line.
fn folder(line: &[Token]) -> Option<Folder> {
    let attributes = match line.get(2) {
        Some(Token::List(items)) => items.iter().filter_map(any_text).collect(),
        _ => Vec::new(),
    };
    let separator = line
        .get(3)
        .and_then(any_text)
        .and_then(|text| text.chars().next());
    let name = line.get(4).and_then(any_text)?;

    Some(Folder {
        name,
        separator,
        attributes,
    })
}

/// One message, out of the items of a `FETCH` line.
fn message(items: &[Token]) -> Fetched {
    let mut out = Fetched::default();
    let mut at = 0;

    while at < items.len() {
        let name = word(items, at);
        let value = items.get(at + 1);
        at += 2;

        match name.as_str() {
            "UID" => {
                out.uid = number(value).min(u64::from(u32::MAX)) as u32;
            }
            "RFC822.SIZE" => out.size = number(value),
            "MODSEQ" => {
                out.mod_seq = match value {
                    Some(Token::List(inner)) => {
                        inner.first().and_then(Token::number).unwrap_or(0)
                    }
                    other => number(other),
                };
            }
            "FLAGS" => {
                if let Some(Token::List(inner)) = value {
                    out.flags = inner.iter().filter_map(any_text).collect();
                }
            }
            "INTERNALDATE" => out.internal_date = value.and_then(any_text),
            "RFC822" | "RFC822.HEADER" | "RFC822.TEXT" => {
                out.body = bytes(value);
            }
            _ if name.starts_with("BODY[") => out.body = bytes(value),
            _ => {}
        }
    }

    out
}

fn number(value: Option<&Token>) -> u64 {
    value.and_then(Token::number).unwrap_or(0)
}

/// The bytes of a value, whatever kind it is.
fn bytes(value: Option<&Token>) -> Vec<u8> {
    match value {
        Some(Token::Literal(raw)) => raw.clone(),
        Some(Token::Atom(text) | Token::Quoted(text)) => {
            text.as_bytes().to_vec()
        }
        _ => Vec::new(),
    }
}

/// Each UID of a `VANISHED` line. (§3.3)
fn vanished(line: &[Token]) -> Vec<u32> {
    let Some(text) = line.iter().rev().find_map(Token::text) else {
        return Vec::new();
    };
    let Ok(set) = UidSet::parse(text) else {
        return Vec::new();
    };
    if set.count() > MOST_GONE
        || set.ranges().iter().any(|(_, high)| *high == LAST)
    {
        return Vec::new();
    }

    set.ranges()
        .iter()
        .flat_map(|(low, high)| *low..=*high)
        .collect()
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_message_survives_the_fetch` | round-trip | The fetch is the whole point of the crate. A body that changes on the way in is mail that the user cannot read. |
    //! | `prop_a_pipeline_answers_every_command` | model-based | §3.1 sends commands together. An answer that goes to the wrong command corrupts the sync. |
    //! | `prop_a_fetch_never_asks_the_server_to_write` | metamorphic | §3 makes mailbert a download-only mirror. `BODY[]` sets `\Seen` on the server, `BODY.PEEK[]` does not. |

    use hegel::{TestCase, generators as gs};

    use super::*;
    use crate::{
        fake::{FakeFolder, FakeMessage, FakeServer, Plan},
        sequence::batches,
    };

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    fn a_plan() -> Plan {
        Plan::new()
            .with(
                FakeFolder::new("INBOX")
                    .with_uid_validity(77)
                    .with(
                        FakeMessage::new(
                            1,
                            "Subject: one\r\n\r\nthe first body\r\n",
                        )
                        .with_flags(&["\\Seen"])
                        .with_mod_seq(4),
                    )
                    .with(
                        FakeMessage::new(
                            2,
                            "Subject: two\r\n\r\nthe second body\r\n",
                        )
                        .with_mod_seq(9),
                    ),
            )
            .with(FakeFolder::new("INBOX/Work"))
    }

    /// Open a connection to a fake server, and log in.
    async fn dial(server: &FakeServer) -> Connection {
        let mut connection =
            Connection::open("127.0.0.1", server.port(), false)
                .await
                .unwrap();
        connection.login("me", "secret").await.unwrap();

        connection
    }

    async fn a_server() -> FakeServer {
        FakeServer::start(a_plan()).await.unwrap()
    }

    fn run<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    fn noop() -> Vec<Token> {
        vec![Token::Atom("NOOP".into())]
    }

    // -----------------------------------------------------------------
    // IDLE, and the watch loop. (§3.1)
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn an_idle_that_sees_nothing_ends_when_the_wait_runs_out() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        connection.examine("INBOX").await.unwrap();

        let news = connection
            .idle(Duration::from_millis(60))
            .await
            .expect("the wait ends well");

        assert!(!news);
        assert!(server.writes().is_empty(), "{:?}", server.writes());
    }

    #[tokio::test]
    async fn an_idle_says_that_a_message_arrived() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        connection.examine("INBOX").await.unwrap();

        let news =
            tokio::join!(connection.idle(Duration::from_secs(5)), async {
                tokio::time::sleep(Duration::from_millis(40)).await;
                server.change(|plan| {
                    plan.folder_mut("INBOX").unwrap().messages.push(
                        FakeMessage::new(3, "Subject: three\r\n\r\nnew\r\n"),
                    );
                });
            })
            .0
            .expect("the server reports the message");

        assert!(news);
    }

    /// A connection must take a command after `DONE`, or the answer of
    /// the IDLE lands on the next command.
    #[tokio::test]
    async fn a_connection_reads_a_fetch_after_an_idle() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        connection.examine("INBOX").await.unwrap();

        connection.idle(Duration::from_millis(40)).await.unwrap();
        let batch = connection.fetch(&UidSet::range(1, 2), None).await.unwrap();

        assert_eq!(batch.messages.len(), 2);
        assert_eq!(batch.messages[0].uid, 1);
    }

    #[tokio::test]
    async fn a_server_that_has_no_idle_never_waits() {
        let plan = a_plan().with_capabilities(&["IMAP4rev1"]);
        let server = FakeServer::start(plan).await.unwrap();
        let mut connection = dial(&server).await;
        connection.examine("INBOX").await.unwrap();

        let started = tokio::time::Instant::now();
        let news = connection.idle(Duration::from_secs(30)).await.unwrap();

        assert!(!news);
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(
            !server
                .seen()
                .commands
                .iter()
                .any(|line| line.contains("IDLE"))
        );
    }

    #[tokio::test]
    async fn an_idle_with_no_folder_open_never_waits() {
        let server = a_server().await;
        let mut connection = dial(&server).await;

        let started = tokio::time::Instant::now();
        let news = connection.idle(Duration::from_secs(30)).await.unwrap();

        assert!(!news);
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    // -----------------------------------------------------------------
    // The greeting, and the capabilities.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn a_connection_reads_the_greeting() {
        let server = a_server().await;
        let connection = Connection::open("127.0.0.1", server.port(), false)
            .await
            .unwrap();

        assert!(connection.can("IMAP4rev1"));
        assert!(connection.can("IDLE"));
        assert!(!connection.can("XSOMETHING"));
        assert_eq!(connection.selected(), None);
    }

    #[tokio::test]
    async fn a_capability_name_is_not_case_sensitive() {
        let server = a_server().await;
        let connection = Connection::open("127.0.0.1", server.port(), false)
            .await
            .unwrap();

        assert!(connection.can("condstore"));
        assert!(connection.can("CondStore"));
    }

    #[tokio::test]
    async fn a_server_that_says_bye_refuses_the_connection() {
        let server = FakeServer::start(a_plan().max_connections(0))
            .await
            .unwrap();
        let error = Connection::open("127.0.0.1", server.port(), false).await;

        assert!(matches!(error, Err(Error::Refused(_))), "{error:?}");
    }

    #[tokio::test]
    async fn a_connection_asks_again_for_the_capabilities() {
        let server = a_server().await;
        let mut connection =
            Connection::open("127.0.0.1", server.port(), false)
                .await
                .unwrap();
        connection.capability().await.unwrap();

        assert!(connection.can("QRESYNC"));
    }

    // -----------------------------------------------------------------
    // Login.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn a_good_password_logs_in() {
        let server = a_server().await;
        let mut connection =
            Connection::open("127.0.0.1", server.port(), false)
                .await
                .unwrap();

        assert!(connection.login("me", "secret").await.is_ok());
    }

    #[tokio::test]
    async fn a_bad_password_is_an_error() {
        let server = a_server().await;
        let mut connection =
            Connection::open("127.0.0.1", server.port(), false)
                .await
                .unwrap();
        let error = connection.login("me", "wrong").await;

        assert!(matches!(error, Err(Error::No(_))), "{error:?}");
    }

    #[tokio::test]
    async fn a_password_with_a_quote_in_it_logs_in() {
        let plan = a_plan().with_login("me", "a\"b\\c");
        let server = FakeServer::start(plan).await.unwrap();
        let mut connection =
            Connection::open("127.0.0.1", server.port(), false)
                .await
                .unwrap();

        assert!(connection.login("me", "a\"b\\c").await.is_ok());
    }

    #[tokio::test]
    async fn a_password_with_a_new_line_in_it_is_an_error() {
        let server = a_server().await;
        let mut connection =
            Connection::open("127.0.0.1", server.port(), false)
                .await
                .unwrap();
        let error = connection.login("me", "secret\r\nA001 NOOP").await;

        assert!(matches!(error, Err(Error::Malformed(_))), "{error:?}");
    }

    // -----------------------------------------------------------------
    // Folders.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn a_connection_lists_every_folder() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        let folders = connection.folders().await.unwrap();
        let names: Vec<&str> =
            folders.iter().map(|folder| folder.name.as_str()).collect();

        assert_eq!(names, vec!["INBOX", "INBOX/Work"]);
        assert_eq!(folders[0].separator, Some('/'));
        assert!(folders[0].holds_mail());
    }

    #[tokio::test]
    async fn a_folder_that_holds_no_mail_says_so() {
        let folder = Folder {
            name: "Archive".into(),
            separator: Some('/'),
            attributes: vec!["\\Noselect".into()],
        };

        assert!(!folder.holds_mail());
    }

    // -----------------------------------------------------------------
    // Examine.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn examine_reads_the_state_of_a_folder() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        let view = connection.examine("INBOX").await.unwrap();

        assert_eq!(view.name, "INBOX");
        assert_eq!(view.exists, 2);
        assert_eq!(view.uid_validity, 77);
        assert_eq!(view.uid_next, 3);
        assert_eq!(view.highest_mod_seq, 9);
        assert_eq!(connection.selected(), Some("INBOX"));
    }

    #[tokio::test]
    async fn examine_never_sends_select() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        connection.examine("INBOX").await.unwrap();

        let commands = server.seen().commands;
        assert!(commands.iter().any(|line| line.contains("EXAMINE")));
        assert!(!commands.iter().any(|line| line.contains("SELECT")));
    }

    #[tokio::test]
    async fn examine_of_a_folder_that_is_not_there_is_an_error() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        let error = connection.examine("Nowhere").await;

        assert!(matches!(error, Err(Error::No(_))), "{error:?}");
        assert_eq!(connection.selected(), None);
    }

    // -----------------------------------------------------------------
    // Fetch.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn a_fetch_reads_each_message_of_the_set() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        connection.examine("INBOX").await.unwrap();
        let batch = connection
            .fetch(&UidSet::parse("1:2").unwrap(), None)
            .await
            .unwrap();

        assert_eq!(batch.messages.len(), 2);
        assert_eq!(batch.messages[0].uid, 1);
        assert_eq!(batch.messages[0].flags, vec!["\\Seen".to_string()]);
        assert_eq!(
            batch.messages[0].body,
            b"Subject: one\r\n\r\nthe first body\r\n".to_vec()
        );
        assert_eq!(batch.messages[1].uid, 2);
        assert!(batch.messages[1].flags.is_empty());
        assert_eq!(batch.messages[1].mod_seq, 9);
        assert!(batch.messages[1].size > 0);
        assert!(batch.messages[1].internal_date.is_some());
    }

    #[tokio::test]
    async fn a_fetch_asks_for_a_body_that_it_may_not_touch() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        connection.examine("INBOX").await.unwrap();
        connection
            .fetch(&UidSet::parse("1:2").unwrap(), None)
            .await
            .unwrap();

        let command = server
            .seen()
            .commands
            .into_iter()
            .find(|line| line.contains("FETCH"))
            .unwrap();

        assert!(command.contains("BODY.PEEK[]"), "{command}");
        assert!(!command.contains("BODY[]"), "{command}");
    }

    #[tokio::test]
    async fn a_fetch_of_nothing_gives_nothing() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        connection.examine("INBOX").await.unwrap();
        let batch = connection
            .fetch(&UidSet::parse("90:99").unwrap(), None)
            .await
            .unwrap();

        assert!(batch.messages.is_empty());
        assert!(batch.gone.is_empty());
    }

    #[tokio::test]
    async fn a_fetch_since_a_sequence_reads_only_what_changed() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        connection.examine("INBOX").await.unwrap();
        let batch = connection
            .fetch(&UidSet::parse("1:*").unwrap(), Some(5))
            .await
            .unwrap();

        assert_eq!(batch.messages.len(), 1);
        assert_eq!(batch.messages[0].uid, 2);
    }

    #[tokio::test]
    async fn a_fetch_since_a_sequence_names_the_messages_that_went() {
        let mut plan = a_plan();
        plan.folder_mut("INBOX").unwrap().gone = vec![7, 8];
        let server = FakeServer::start(plan).await.unwrap();
        let mut connection = dial(&server).await;
        connection.examine("INBOX").await.unwrap();
        let batch = connection
            .fetch(&UidSet::parse("1:20").unwrap(), Some(0))
            .await
            .unwrap();

        assert_eq!(batch.gone, vec![7, 8]);
    }

    #[tokio::test]
    async fn a_fetch_without_a_folder_is_an_error() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        let error = connection.fetch(&UidSet::parse("1").unwrap(), None).await;

        assert!(matches!(error, Err(Error::No(_))), "{error:?}");
    }

    // -----------------------------------------------------------------
    // A whole session.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn a_whole_session_writes_nothing_to_the_server() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        connection.enable(&["CONDSTORE", "QRESYNC"]).await.unwrap();

        for folder in connection.folders().await.unwrap() {
            if !folder.holds_mail() {
                continue;
            }

            let view = connection.examine(&folder.name).await.unwrap();
            for set in batches(1, view.uid_next.saturating_sub(1), BATCH) {
                connection.fetch(&set, None).await.unwrap();
            }
        }
        connection.logout().await.unwrap();

        assert!(server.writes().is_empty(), "{:?}", server.writes());
    }

    // -----------------------------------------------------------------
    // Pipelining. (§3.1)
    // -----------------------------------------------------------------

    #[test]
    fn a_pipeline_puts_every_command_in_one_write() {
        let tags = vec!["a0001".to_string(), "a0002".to_string()];
        let commands = vec![noop(), vec![Token::Atom("CAPABILITY".into())]];
        let raw = String::from_utf8(pipelined(&tags, &commands)).unwrap();

        assert_eq!(raw, "a0001 NOOP\r\na0002 CAPABILITY\r\n");
    }

    #[tokio::test]
    async fn a_pipeline_reads_the_answer_of_each_command() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        let answers = connection
            .pipeline(&[noop(), vec![Token::Atom("CAPABILITY".into())], noop()])
            .await
            .unwrap();

        assert_eq!(answers.len(), 3);
        assert!(answers.iter().all(|answer| answer.state == State::Ok));
        assert_eq!(answers[1].each("CAPABILITY").len(), 1);
        assert!(answers[0].each("CAPABILITY").is_empty());
    }

    #[tokio::test]
    async fn a_pipeline_of_nothing_gives_nothing() {
        let server = a_server().await;
        let mut connection = dial(&server).await;

        assert!(connection.pipeline(&[]).await.unwrap().is_empty());
    }

    // -----------------------------------------------------------------
    // Errors.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn a_command_that_the_server_does_not_know_is_an_error() {
        let server = a_server().await;
        let mut connection = dial(&server).await;
        let answer = connection
            .run(&[Token::Atom("XNOPE".into())])
            .await
            .unwrap();

        assert_eq!(answer.state, State::Bad);
        assert!(matches!(answer.ok(), Err(Error::Bad(_))));
    }

    #[tokio::test]
    async fn a_server_that_goes_away_is_an_error() {
        let server = FakeServer::start(a_plan().cut_after(2)).await.unwrap();
        let mut connection =
            Connection::open("127.0.0.1", server.port(), false)
                .await
                .unwrap();
        connection.run(&noop()).await.unwrap();
        let error = connection.run(&noop()).await;

        assert!(matches!(error, Err(Error::Closed) | Err(Error::Io(_))));
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::composite]
    fn a_mailbox(tc: TestCase) -> Vec<(u32, String, Vec<String>)> {
        let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
        let mut out = Vec::new();

        for at in 0..count {
            let body = tc.draw(
                gs::text()
                    .alphabet("abcdefghijklmnopqrstuvwxyz .,\r\n")
                    .min_size(0)
                    .max_size(120),
            );
            let flags = tc.draw(
                gs::vecs(gs::sampled_from(vec![
                    "\\Seen".to_string(),
                    "\\Flagged".to_string(),
                    "\\Draft".to_string(),
                ]))
                .min_size(0)
                .max_size(3),
            );

            out.push((at as u32 + 1, body, flags));
        }

        out
    }

    #[hegel::test(test_cases = 60)]
    fn prop_a_message_survives_the_fetch(tc: TestCase) {
        let wanted = tc.draw(a_mailbox());
        let mut folder = FakeFolder::new("INBOX");

        for (uid, body, flags) in &wanted {
            let names: Vec<&str> = flags.iter().map(String::as_str).collect();
            folder =
                folder.with(FakeMessage::new(*uid, body).with_flags(&names));
        }

        let got = run(async {
            let server =
                FakeServer::start(Plan::new().with(folder)).await.unwrap();
            let mut connection = dial(&server).await;
            connection.examine("INBOX").await.unwrap();

            connection
                .fetch(&UidSet::parse("1:*").unwrap(), None)
                .await
                .unwrap()
        });

        assert_eq!(got.messages.len(), wanted.len());
        for (message, (uid, body, flags)) in got.messages.iter().zip(&wanted) {
            assert_eq!(message.uid, *uid);
            assert_eq!(message.body, body.as_bytes());
            assert_eq!(message.size, body.len() as u64);
            assert_eq!(&message.flags, flags);
        }
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_pipeline_answers_every_command(tc: TestCase) {
        let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
        let commands: Vec<Vec<Token>> = (0..count)
            .map(|at| {
                if at % 2 == 0 {
                    noop()
                } else {
                    vec![Token::Atom("CAPABILITY".into())]
                }
            })
            .collect();

        let answers = run(async {
            let server = a_server().await;
            let mut connection = dial(&server).await;

            connection.pipeline(&commands).await.unwrap()
        });

        assert_eq!(answers.len(), count);
        for (at, answer) in answers.iter().enumerate() {
            assert_eq!(answer.state, State::Ok);
            let named = !answer.each("CAPABILITY").is_empty();
            assert_eq!(named, at % 2 == 1, "answer {at} went to the wrong tag");
        }
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_fetch_never_asks_the_server_to_write(tc: TestCase) {
        let low = tc.draw(gs::integers::<u32>().min_value(1).max_value(5));
        let span = tc.draw(gs::integers::<u32>().min_value(0).max_value(20));
        let since = tc.draw(gs::optional(
            gs::integers::<u64>().min_value(0).max_value(20),
        ));

        let writes = run(async {
            let server = a_server().await;
            let mut connection = dial(&server).await;
            connection.examine("INBOX").await.unwrap();
            let _ = connection
                .fetch(&UidSet::range(low, low + span), since)
                .await
                .unwrap();

            server.writes()
        });

        assert!(writes.is_empty(), "{writes:?}");
    }
}
