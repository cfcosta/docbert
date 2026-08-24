//! A fake IMAP server, so a test needs no network.
//!
//! The server listens on a free local port and answers from a [`Plan`].
//! It holds folders and messages, it counts its connections, and it
//! keeps every command that it read.
//!
//! The server refuses each command that would change what it holds.
//! mailbert is a mirror that only reads, and a test asserts that with
//! [`FakeServer::writes`].

use std::{
    net::SocketAddr,
    sync::{Arc, Mutex, MutexGuard},
};

use tokio::{
    io::{AsyncWriteExt, BufReader},
    net::{
        TcpListener,
        TcpStream,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
    task::JoinHandle,
};

use crate::{
    error::Result,
    sequence::UidSet,
    token::{Token, lex},
    wire::read_line,
};

/// The commands that change a server. mailbert sends none of them.
pub const WRITES: [&str; 12] = [
    "APPEND",
    "COPY",
    "CREATE",
    "DELETE",
    "EXPUNGE",
    "MOVE",
    "RENAME",
    "SETACL",
    "SETQUOTA",
    "STORE",
    "SUBSCRIBE",
    "UNSUBSCRIBE",
];

/// What a server announces when a [`Plan`] names nothing.
pub const CAPABILITIES: [&str; 7] = [
    "IMAP4rev1",
    "LITERAL+",
    "ENABLE",
    "IDLE",
    "CONDSTORE",
    "QRESYNC",
    "UIDPLUS",
];

// ---------------------------------------------------------------------
// What the server holds.
// ---------------------------------------------------------------------

/// One message on the fake server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FakeMessage {
    pub uid: u32,
    pub mod_seq: u64,
    pub flags: Vec<String>,
    pub internal_date: String,
    pub raw: Vec<u8>,
}

impl FakeMessage {
    pub fn new(uid: u32, raw: &str) -> Self {
        Self {
            uid,
            mod_seq: u64::from(uid),
            flags: Vec::new(),
            internal_date: "01-Jan-2020 00:00:00 +0000".to_string(),
            raw: raw.as_bytes().to_vec(),
        }
    }

    pub fn with_flags(mut self, flags: &[&str]) -> Self {
        self.flags = flags.iter().map(|flag| (*flag).to_string()).collect();
        self
    }

    pub fn with_mod_seq(mut self, mod_seq: u64) -> Self {
        self.mod_seq = mod_seq;
        self
    }

    /// The header of the message, up to the empty line after it.
    pub fn header(&self) -> &[u8] {
        let end = self
            .raw
            .windows(4)
            .position(|four| four == b"\r\n\r\n")
            .map(|at| at + 4)
            .or_else(|| {
                self.raw
                    .windows(2)
                    .position(|two| two == b"\n\n")
                    .map(|at| at + 2)
            })
            .unwrap_or(self.raw.len());

        &self.raw[..end]
    }
}

/// The words that the body of a message in bulk holds.
///
/// Real prose makes the passages of §6.2 look like the passages of a
/// mailbox, so a benchmark of the model reads a true number.
const WORDS: [&str; 16] = [
    "invoice", "deposit", "report", "meeting", "review", "release", "account",
    "payment", "schedule", "summary", "question", "answer", "project",
    "October", "attached", "thanks",
];

/// Mail in bulk, for a benchmark. (§10.5)
///
/// The messages take the UIDs from 1 to `count`. Each one holds its own
/// `Message-ID`, so the store keeps one entry for each of them, and a
/// benchmark of 10000 messages measures 10000 writes.
///
/// `size` is about how many bytes one whole message takes. A message
/// never comes out shorter than that, and never more than 512 bytes
/// longer. A `size` below the header gives the header and one line.
///
/// # Examples
///
/// ```
/// use mailbert_imap::fake::bulk;
///
/// let mail = bulk(3, 1024);
///
/// assert_eq!(mail.len(), 3);
/// assert_eq!(mail[0].uid, 1);
/// assert!(mail[0].raw.len() >= 1024);
/// ```
pub fn bulk(count: u32, size: usize) -> Vec<FakeMessage> {
    bulk_from(1, count, size)
}

/// Mail in bulk, from the UID `first`. (§10.5)
///
/// The UID also names the message, so two folders that start apart
/// hold no message together. A benchmark that spreads mail across
/// folders needs that, because the store keeps one entry for one set
/// of bytes.
///
/// # Examples
///
/// ```
/// use mailbert_imap::fake::bulk_from;
///
/// let mail = bulk_from(500, 2, 128);
///
/// assert_eq!(mail[0].uid, 500);
/// assert_eq!(mail[1].uid, 501);
/// ```
pub fn bulk_from(first: u32, count: u32, size: usize) -> Vec<FakeMessage> {
    (first..first + count)
        .map(|uid| FakeMessage::new(uid, &one_mail(uid, size)))
        .collect()
}

/// The next number of the sequence that gives the words of a body.
///
/// Two messages that start from a different number keep a different
/// body, because this mixes every bit of the number that it takes.
fn mix(seed: u64) -> u64 {
    let mut next = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    next ^= next >> 33;
    next = next.wrapping_mul(0xff51afd7ed558ccd);
    next ^ (next >> 29)
}

/// One message of [`bulk`], about `size` bytes long.
fn one_mail(uid: u32, size: usize) -> String {
    let mut mail = format!(
        "From: Alice Smith <alice{uid}@example.test>\r\n\
         To: bob@example.test\r\n\
         Subject: The report of {uid}\r\n\
         Date: Fri, 14 Aug 2026 09:30:00 +0000\r\n\
         Message-ID: <{uid}@bulk.example.test>\r\n\
         \r\n\
         Here is the report of {uid}.\r\n"
    );

    let mut seed = uid as u64;
    let mut written = 0usize;
    while mail.len() < size {
        seed = mix(seed);
        mail.push_str(WORDS[(seed % WORDS.len() as u64) as usize]);
        written += 1;

        match written % 12 {
            0 => mail.push_str("\r\n"),
            _ => mail.push(' '),
        }
    }
    mail.push_str("\r\n");

    mail
}

/// One folder on the fake server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FakeFolder {
    pub name: String,
    pub uid_validity: u32,
    pub messages: Vec<FakeMessage>,
    /// The UIDs that the server held once, and holds no more.
    pub gone: Vec<u32>,
    /// The attributes of RFC 6154, such as `\All` or `\Trash`.
    pub attributes: Vec<String>,
}

impl FakeFolder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            uid_validity: 1,
            messages: Vec::new(),
            gone: Vec::new(),
            attributes: Vec::new(),
        }
    }

    /// A folder that holds `count` messages of about `size` bytes.
    ///
    /// This is what a benchmark opens. See [`bulk`].
    pub fn filled(name: &str, count: u32, size: usize) -> Self {
        Self::filled_from(name, 1, count, size)
    }

    /// A folder of `count` messages, from the UID `first`.
    ///
    /// Two folders that start apart hold no message together. See
    /// [`bulk_from`].
    pub fn filled_from(
        name: &str,
        first: u32,
        count: u32,
        size: usize,
    ) -> Self {
        Self {
            messages: bulk_from(first, count, size),
            ..Self::new(name)
        }
    }

    /// Give the folder one attribute of RFC 6154.
    pub fn with_attribute(mut self, attribute: &str) -> Self {
        self.attributes.push(attribute.to_string());
        self
    }

    pub fn with(mut self, message: FakeMessage) -> Self {
        self.messages.push(message);
        self
    }

    pub fn with_uid_validity(mut self, uid_validity: u32) -> Self {
        self.uid_validity = uid_validity;
        self
    }

    pub fn last_uid(&self) -> u32 {
        self.messages.iter().map(|held| held.uid).max().unwrap_or(0)
    }

    pub fn uid_next(&self) -> u32 {
        self.last_uid()
            .max(self.gone.iter().copied().max().unwrap_or(0))
            + 1
    }

    pub fn highest_mod_seq(&self) -> u64 {
        self.messages
            .iter()
            .map(|held| held.mod_seq)
            .max()
            .unwrap_or(1)
    }
}

/// What the fake server holds, and how it answers.
#[derive(Debug, Clone)]
pub struct Plan {
    pub user: String,
    pub password: String,
    pub capabilities: Vec<String>,
    pub folders: Vec<FakeFolder>,
    pub separator: char,
    /// Close the connection after this count of commands. (§3.4)
    pub cut_after: Option<usize>,
    /// Refuse a connection above this count. (§3.1)
    pub max_connections: Option<usize>,
}

impl Default for Plan {
    fn default() -> Self {
        Self {
            user: "me".to_string(),
            password: "secret".to_string(),
            capabilities: CAPABILITIES
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            folders: Vec::new(),
            separator: '/',
            cut_after: None,
            max_connections: None,
        }
    }
}

impl Plan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, folder: FakeFolder) -> Self {
        self.folders.push(folder);
        self
    }

    pub fn with_capabilities(mut self, names: &[&str]) -> Self {
        self.capabilities =
            names.iter().map(|name| (*name).to_string()).collect();
        self
    }

    pub fn with_login(mut self, user: &str, password: &str) -> Self {
        self.user = user.to_string();
        self.password = password.to_string();
        self
    }

    pub fn cut_after(mut self, count: usize) -> Self {
        self.cut_after = Some(count);
        self
    }

    /// Refuse a connection above this count. (§3.1)
    pub fn max_connections(mut self, count: usize) -> Self {
        self.max_connections = Some(count);
        self
    }

    pub fn folder(&self, name: &str) -> Option<&FakeFolder> {
        self.folders.iter().find(|held| held.name == name)
    }

    pub fn folder_mut(&mut self, name: &str) -> Option<&mut FakeFolder> {
        self.folders.iter_mut().find(|held| held.name == name)
    }
}

/// What the server saw while it ran.
#[derive(Debug, Clone, Default)]
pub struct Seen {
    /// Each command line, in the order that they arrived.
    pub commands: Vec<String>,
    /// How many connections opened.
    pub connections: usize,
    /// How many are open now.
    pub open: usize,
    /// The largest count of connections that were open together.
    pub most_open: usize,
}

// ---------------------------------------------------------------------
// The server.
// ---------------------------------------------------------------------

/// A fake IMAP server on a free local port.
pub struct FakeServer {
    address: SocketAddr,
    plan: Arc<Mutex<Plan>>,
    seen: Arc<Mutex<Seen>>,
    handle: JoinHandle<()>,
}

impl FakeServer {
    /// Start a server, and give the address that it listens on.
    pub async fn start(plan: Plan) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let plan = Arc::new(Mutex::new(plan));
        let seen = Arc::new(Mutex::new(Seen::default()));

        let handle = tokio::spawn({
            let plan = Arc::clone(&plan);
            let seen = Arc::clone(&seen);

            async move {
                while let Ok((stream, _)) = listener.accept().await {
                    tokio::spawn(serve(
                        stream,
                        Arc::clone(&plan),
                        Arc::clone(&seen),
                    ));
                }
            }
        });

        Ok(Self {
            address,
            plan,
            seen,
            handle,
        })
    }

    pub fn address(&self) -> SocketAddr {
        self.address
    }

    pub fn port(&self) -> u16 {
        self.address.port()
    }

    pub fn seen(&self) -> Seen {
        hold(&self.seen).clone()
    }

    /// Each command that would change the server. It must stay empty.
    pub fn writes(&self) -> Vec<String> {
        hold(&self.seen)
            .commands
            .iter()
            .filter(|command| is_write(command))
            .cloned()
            .collect()
    }

    /// Change what the server holds, while it runs.
    pub fn change<T>(&self, edit: impl FnOnce(&mut Plan) -> T) -> T {
        edit(&mut hold(&self.plan))
    }
}

impl Drop for FakeServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// Does this command line change what a server holds?
pub fn is_write(command: &str) -> bool {
    command
        .split_whitespace()
        .skip(1)
        .take(2)
        .any(|word| WRITES.contains(&word.to_ascii_uppercase().as_str()))
}

/// Does a folder name match a `LIST` pattern?
///
/// A `*` stands for any text. A `%` stands for any text that holds no
/// separator. (RFC 3501 §6.3.8)
pub fn matches(pattern: &str, name: &str, separator: char) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let name: Vec<char> = name.chars().collect();

    matches_at(&pattern, &name, separator)
}

fn matches_at(pattern: &[char], name: &[char], separator: char) -> bool {
    match pattern.first() {
        None => name.is_empty(),
        Some('*') => (0..=name.len())
            .any(|take| matches_at(&pattern[1..], &name[take..], separator)),
        Some('%') => (0..=name.len())
            .take_while(|take| {
                name[..*take].iter().all(|letter| *letter != separator)
            })
            .any(|take| matches_at(&pattern[1..], &name[take..], separator)),
        Some(letter) => {
            name.first() == Some(letter)
                && matches_at(&pattern[1..], &name[1..], separator)
        }
    }
}

fn hold<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(|held| held.into_inner())
}

// ---------------------------------------------------------------------
// One connection.
// ---------------------------------------------------------------------

/// What one connection knows about itself.
#[derive(Debug, Default)]
struct Session {
    selected: Option<String>,
}

/// What the server does after it answers.
enum After {
    Go,
    Stop,
}

async fn serve(
    stream: TcpStream,
    plan: Arc<Mutex<Plan>>,
    seen: Arc<Mutex<Seen>>,
) {
    {
        let mut seen = hold(&seen);
        seen.connections += 1;
        seen.open += 1;
        seen.most_open = seen.most_open.max(seen.open);
    }

    let _ = talk(stream, &plan, &seen).await;

    hold(&seen).open -= 1;
}

async fn talk(
    stream: TcpStream,
    plan: &Mutex<Plan>,
    seen: &Mutex<Seen>,
) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut session = Session::default();
    let mut count = 0;

    // §3.1: a server can refuse a connection. A client that sees this
    // must use fewer connections.
    let most = hold(plan).max_connections;
    let open = hold(seen).open;
    if most.is_some_and(|most| open > most) {
        writer
            .write_all(b"* BYE too many connections are open\r\n")
            .await?;

        return Ok(());
    }

    let greeting = {
        let plan = hold(plan);
        format!(
            "* OK [CAPABILITY {}] mailbert fake ready\r\n",
            plan.capabilities.join(" ")
        )
    };
    writer.write_all(greeting.as_bytes()).await?;

    loop {
        let Ok(raw) = read_line(&mut reader).await else {
            return Ok(());
        };

        let text = String::from_utf8_lossy(&raw).trim_end().to_string();
        hold(seen).commands.push(text);
        count += 1;

        if hold(plan).cut_after == Some(count) {
            return Ok(());
        }

        let Ok(tokens) = lex(&raw) else {
            writer.write_all(b"* BAD I cannot read that\r\n").await?;
            continue;
        };

        if word(&tokens, 1) == "IDLE" {
            idle(&mut reader, &mut writer, plan, seen, &tokens, &mut session)
                .await?;
            continue;
        }

        let (reply, after) = {
            let plan = hold(plan);
            answer(&plan, &mut session, &tokens)
        };
        writer.write_all(&reply).await?;

        if matches!(after, After::Stop) {
            return Ok(());
        }
    }
}

/// Hold the connection open, and report a message that arrives.
async fn idle(
    reader: &mut BufReader<OwnedReadHalf>,
    writer: &mut OwnedWriteHalf,
    plan: &Mutex<Plan>,
    seen: &Mutex<Seen>,
    tokens: &[Token],
    session: &mut Session,
) -> Result<()> {
    let tag = word_of(tokens, 0);
    writer.write_all(b"+ idling\r\n").await?;

    let count = |plan: &Plan, session: &Session| {
        session
            .selected
            .as_deref()
            .and_then(|name| plan.folder(name))
            .map_or(0, |folder| folder.messages.len())
    };
    let mut held = count(&hold(plan), session);

    let done = loop {
        let waiting = tokio::time::timeout(
            std::time::Duration::from_millis(20),
            read_line(reader),
        )
        .await;

        match waiting {
            Ok(line) => break line?,
            Err(_) => {
                let now = count(&hold(plan), session);
                if now != held {
                    held = now;
                    writer
                        .write_all(format!("* {now} EXISTS\r\n").as_bytes())
                        .await?;
                }
            }
        }
    };

    hold(seen)
        .commands
        .push(String::from_utf8_lossy(&done).trim_end().to_string());
    writer
        .write_all(format!("{tag} OK IDLE terminated\r\n").as_bytes())
        .await?;

    Ok(())
}

// ---------------------------------------------------------------------
// The answer to one command.
// ---------------------------------------------------------------------

fn answer(
    plan: &Plan,
    session: &mut Session,
    tokens: &[Token],
) -> (Vec<u8>, After) {
    let tag = word_of(tokens, 0);
    let verb = word(tokens, 1);
    let mut reply = Vec::new();

    match verb.as_str() {
        "CAPABILITY" => {
            line(
                &mut reply,
                &format!("* CAPABILITY {}", plan.capabilities.join(" ")),
            );
            done(&mut reply, &tag, "OK", "CAPABILITY completed");
        }
        "NOOP" | "CHECK" | "ENABLE" => {
            if verb == "ENABLE" {
                let names: Vec<String> =
                    tokens[2..].iter().filter_map(text_of).collect();
                line(&mut reply, &format!("* ENABLED {}", names.join(" ")));
            }
            done(&mut reply, &tag, "OK", &format!("{verb} completed"));
        }
        "LOGOUT" => {
            line(&mut reply, "* BYE mailbert fake is closing");
            done(&mut reply, &tag, "OK", "LOGOUT completed");
            return (reply, After::Stop);
        }
        "LOGIN" => login(plan, &tag, tokens, &mut reply),
        "LIST" | "LSUB" => list(plan, &verb, &tag, tokens, &mut reply),
        "SELECT" | "EXAMINE" => {
            select(plan, session, &verb, &tag, tokens, &mut reply);
        }
        "STATUS" => status(plan, &tag, tokens, &mut reply),
        "UID" => uid(plan, session, &tag, tokens, &mut reply),
        _ if WRITES.contains(&verb.as_str()) => {
            done(
                &mut reply,
                &tag,
                "NO",
                "this server is a mirror, and mailbert must not write to it",
            );
        }
        _ => {
            done(
                &mut reply,
                &tag,
                "BAD",
                &format!("`{verb}` is not a command that this server knows"),
            );
        }
    }

    (reply, After::Go)
}

fn login(plan: &Plan, tag: &str, tokens: &[Token], reply: &mut Vec<u8>) {
    let user = word_of(tokens, 2);
    let password = word_of(tokens, 3);

    if user == plan.user && password == plan.password {
        done(
            reply,
            tag,
            "OK",
            &format!(
                "[CAPABILITY {}] LOGIN completed",
                plan.capabilities.join(" ")
            ),
        );
    } else {
        done(
            reply,
            tag,
            "NO",
            "[AUTHENTICATIONFAILED] that is not the password",
        );
    }
}

fn list(
    plan: &Plan,
    verb: &str,
    tag: &str,
    tokens: &[Token],
    reply: &mut Vec<u8>,
) {
    let reference = word_of(tokens, 2);
    let pattern = format!("{reference}{}", word_of(tokens, 3));

    for folder in &plan.folders {
        if !matches(&pattern, &folder.name, plan.separator) {
            continue;
        }

        let prefix = format!("{}{}", folder.name, plan.separator);
        let children = plan
            .folders
            .iter()
            .any(|other| other.name.starts_with(&prefix));
        let shape = if children {
            "\\HasChildren"
        } else {
            "\\HasNoChildren"
        };
        let attribute = std::iter::once(shape.to_string())
            .chain(folder.attributes.iter().cloned())
            .collect::<Vec<String>>()
            .join(" ");

        line(
            reply,
            &format!(
                "* {verb} ({attribute}) \"{}\" \"{}\"",
                plan.separator, folder.name
            ),
        );
    }

    done(reply, tag, "OK", &format!("{verb} completed"));
}

fn select(
    plan: &Plan,
    session: &mut Session,
    verb: &str,
    tag: &str,
    tokens: &[Token],
    reply: &mut Vec<u8>,
) {
    let name = word_of(tokens, 2);
    let Some(folder) = plan.folder(&name) else {
        session.selected = None;
        done(reply, tag, "NO", "that folder is not on this server");
        return;
    };

    session.selected = Some(name);
    line(
        reply,
        "* FLAGS (\\Answered \\Flagged \\Deleted \\Seen \\Draft)",
    );
    line(reply, &format!("* {} EXISTS", folder.messages.len()));
    line(reply, "* 0 RECENT");
    line(
        reply,
        &format!("* OK [UIDVALIDITY {}] UIDs valid", folder.uid_validity),
    );
    line(
        reply,
        &format!("* OK [UIDNEXT {}] the next UID", folder.uid_next()),
    );
    line(
        reply,
        &format!(
            "* OK [HIGHESTMODSEQ {}] the highest sequence",
            folder.highest_mod_seq()
        ),
    );

    let mode = if verb == "EXAMINE" {
        "READ-ONLY"
    } else {
        "READ-WRITE"
    };
    done(reply, tag, "OK", &format!("[{mode}] {verb} completed"));
}

fn status(plan: &Plan, tag: &str, tokens: &[Token], reply: &mut Vec<u8>) {
    let name = word_of(tokens, 2);
    let Some(folder) = plan.folder(&name) else {
        done(reply, tag, "NO", "that folder is not on this server");
        return;
    };

    let counts: Vec<String> = items(tokens.get(3))
        .iter()
        .filter_map(|item| match item.as_str() {
            "MESSAGES" => Some(format!("MESSAGES {}", folder.messages.len())),
            "UIDNEXT" => Some(format!("UIDNEXT {}", folder.uid_next())),
            "UIDVALIDITY" => {
                Some(format!("UIDVALIDITY {}", folder.uid_validity))
            }
            "HIGHESTMODSEQ" => {
                Some(format!("HIGHESTMODSEQ {}", folder.highest_mod_seq()))
            }
            "UNSEEN" => Some("UNSEEN 0".to_string()),
            "RECENT" => Some("RECENT 0".to_string()),
            _ => None,
        })
        .collect();

    line(
        reply,
        &format!("* STATUS \"{name}\" ({})", counts.join(" ")),
    );
    done(reply, tag, "OK", "STATUS completed");
}

fn uid(
    plan: &Plan,
    session: &Session,
    tag: &str,
    tokens: &[Token],
    reply: &mut Vec<u8>,
) {
    let verb = word(tokens, 2);
    let Some(folder) = session
        .selected
        .as_deref()
        .and_then(|name| plan.folder(name))
    else {
        done(reply, tag, "NO", "no folder is selected");
        return;
    };

    let Ok(set) = UidSet::parse(&word_of(tokens, 3)) else {
        done(reply, tag, "BAD", "that is not a set of UIDs");
        return;
    };
    let set = set.resolve(folder.last_uid());

    match verb.as_str() {
        "FETCH" => fetch(folder, &set, tag, tokens, reply),
        "SEARCH" => {
            let found: Vec<String> = folder
                .messages
                .iter()
                .map(|message| message.uid.to_string())
                .collect();
            line(reply, &format!("* SEARCH {}", found.join(" ")));
            done(reply, tag, "OK", "UID SEARCH completed");
        }
        _ => done(
            reply,
            tag,
            "NO",
            "this server is a mirror, and mailbert must not write to it",
        ),
    }
}

fn fetch(
    folder: &FakeFolder,
    set: &UidSet,
    tag: &str,
    tokens: &[Token],
    reply: &mut Vec<u8>,
) {
    let names = items(tokens.get(4));
    let extra = items(tokens.get(5));
    let since = extra
        .iter()
        .position(|name| name == "CHANGEDSINCE")
        .and_then(|at| extra.get(at + 1))
        .and_then(|value| value.parse::<u64>().ok());

    for (at, message) in folder.messages.iter().enumerate() {
        if !set.holds(message.uid) {
            continue;
        }
        if since.is_some_and(|since| message.mod_seq <= since) {
            continue;
        }

        fetch_one(at + 1, message, &names, reply);
    }

    if extra.iter().any(|name| name == "VANISHED") {
        let went: Vec<u32> = folder
            .gone
            .iter()
            .copied()
            .filter(|uid| set.holds(*uid))
            .collect();

        if !went.is_empty() {
            line(
                reply,
                &format!("* VANISHED (EARLIER) {}", UidSet::of(&went)),
            );
        }
    }

    done(reply, tag, "OK", "UID FETCH completed");
}

fn fetch_one(
    number: usize,
    message: &FakeMessage,
    names: &[String],
    reply: &mut Vec<u8>,
) {
    let mut parts: Vec<Vec<u8>> = Vec::new();
    let uid = format!("UID {}", message.uid).into_bytes();

    for name in names {
        match name.as_str() {
            "UID" => parts.push(uid.clone()),
            "FLAGS" => parts.push(
                format!("FLAGS ({})", message.flags.join(" ")).into_bytes(),
            ),
            "RFC822.SIZE" => parts.push(
                format!("RFC822.SIZE {}", message.raw.len()).into_bytes(),
            ),
            "INTERNALDATE" => parts.push(
                format!("INTERNALDATE \"{}\"", message.internal_date)
                    .into_bytes(),
            ),
            "MODSEQ" => {
                parts
                    .push(format!("MODSEQ ({})", message.mod_seq).into_bytes());
            }
            "BODY[]" | "BODY.PEEK[]" | "RFC822" => {
                parts.push(part("BODY[]", &message.raw));
            }
            "BODY[HEADER]" | "BODY.PEEK[HEADER]" | "RFC822.HEADER" => {
                parts.push(part("BODY[HEADER]", message.header()));
            }
            _ => {}
        }
    }

    // A `UID FETCH` always names the UID, even when the client did not
    // ask for it. (RFC 3501 §6.4.8)
    if !names.iter().any(|name| name == "UID") {
        parts.insert(0, uid);
    }

    reply.extend_from_slice(format!("* {number} FETCH (").as_bytes());
    for (at, item) in parts.iter().enumerate() {
        if at > 0 {
            reply.push(b' ');
        }
        reply.extend_from_slice(item);
    }
    reply.extend_from_slice(b")\r\n");
}

fn part(name: &str, bytes: &[u8]) -> Vec<u8> {
    let mut out = format!("{name} {{{}}}\r\n", bytes.len()).into_bytes();
    out.extend_from_slice(bytes);

    out
}

// ---------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------

fn line(reply: &mut Vec<u8>, text: &str) {
    reply.extend_from_slice(text.as_bytes());
    reply.extend_from_slice(b"\r\n");
}

fn done(reply: &mut Vec<u8>, tag: &str, state: &str, text: &str) {
    line(reply, &format!("{tag} {state} {text}"));
}

/// The token at this place, in capitals.
fn word(tokens: &[Token], at: usize) -> String {
    tokens
        .get(at)
        .and_then(Token::text)
        .unwrap_or_default()
        .to_ascii_uppercase()
}

/// The token at this place, as it arrived.
fn word_of(tokens: &[Token], at: usize) -> String {
    tokens
        .get(at)
        .and_then(Token::text)
        .unwrap_or_default()
        .to_string()
}

fn text_of(token: &Token) -> Option<String> {
    token.text().map(str::to_string)
}

/// The names in a list, in capitals. One name alone counts as a list.
fn items(token: Option<&Token>) -> Vec<String> {
    match token {
        Some(Token::List(held)) => held
            .iter()
            .map(|item| match item {
                Token::List(inner) => inner
                    .iter()
                    .filter_map(Token::text)
                    .map(str::to_ascii_uppercase)
                    .collect::<Vec<String>>()
                    .join(" "),
                other => other.text().unwrap_or_default().to_ascii_uppercase(),
            })
            .collect(),
        Some(other) => other
            .text()
            .map(|name| vec![name.to_ascii_uppercase()])
            .unwrap_or_default(),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_star_pattern_names_every_folder` | algebraic | `LIST "" "*"` is how a sync finds the folders. A pattern that drops one folder hides all of its mail. |
    //! | `prop_a_percent_never_crosses_the_separator` | model-based | RFC 3501 §6.3.8 gives `%` one level. A `%` that goes deeper lists a folder twice. |

    use std::collections::BTreeSet;

    use hegel::{TestCase, generators as gs};
    use mailbert_core::{
        message_id::MessageId,
        threading::{ThreadInput, thread},
    };
    use tokio::io::AsyncWriteExt;

    use super::*;
    use crate::wire::Tags;

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    /// A client that speaks to the fake server over a local socket.
    struct Client {
        reader: BufReader<OwnedReadHalf>,
        writer: OwnedWriteHalf,
        tags: Tags,
    }

    impl Client {
        /// Connect, and read the greeting.
        async fn connect(server: &FakeServer) -> (Self, String) {
            let stream = TcpStream::connect(server.address()).await.unwrap();
            let (reader, writer) = stream.into_split();
            let mut client = Self {
                reader: BufReader::new(reader),
                writer,
                tags: Tags::new(),
            };
            let greeting = client.take().await;

            (client, greeting)
        }

        async fn take(&mut self) -> String {
            let raw = read_line(&mut self.reader).await.unwrap();

            String::from_utf8_lossy(&raw).trim_end().to_string()
        }

        /// Send a command, and read until the answer with its tag.
        async fn send(&mut self, command: &str) -> Vec<Vec<u8>> {
            let tag = self.tags.next_tag();
            self.writer
                .write_all(format!("{tag} {command}\r\n").as_bytes())
                .await
                .unwrap();
            self.writer.flush().await.unwrap();

            let mut lines = Vec::new();
            loop {
                let raw = read_line(&mut self.reader).await.unwrap();
                let last = raw.starts_with(tag.as_bytes());
                lines.push(raw);

                if last {
                    return lines;
                }
            }
        }

        /// Send a command, and read each answer as text.
        async fn ask(&mut self, command: &str) -> Vec<String> {
            self.send(command)
                .await
                .iter()
                .map(|raw| String::from_utf8_lossy(raw).trim_end().to_string())
                .collect()
        }
    }

    fn last(lines: &[String]) -> String {
        lines.last().cloned().unwrap_or_default()
    }

    fn a_plan() -> Plan {
        Plan::new()
            .with(
                FakeFolder::new("INBOX")
                    .with_uid_validity(77)
                    .with(FakeMessage::new(
                        1,
                        "Subject: one\r\n\r\nthe first body\r\n",
                    ))
                    .with(
                        FakeMessage::new(
                            4,
                            "Subject: two\r\n\r\nthe second body\r\n",
                        )
                        .with_flags(&["\\Seen"])
                        .with_mod_seq(9),
                    ),
            )
            .with(FakeFolder::new("INBOX/Archive"))
            .with(FakeFolder::new("Sent"))
    }

    async fn a_server() -> FakeServer {
        FakeServer::start(a_plan()).await.unwrap()
    }

    // -----------------------------------------------------------------
    // Unit tests: the greeting, and who may speak.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn the_server_greets_a_client_with_its_capabilities() {
        let server = a_server().await;
        let (_client, greeting) = Client::connect(&server).await;

        assert!(greeting.starts_with("* OK [CAPABILITY IMAP4rev1"));
        assert!(greeting.contains("CONDSTORE"));
    }

    #[tokio::test]
    async fn the_server_answers_capability() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;

        let lines = client.ask("CAPABILITY").await;

        assert!(lines[0].starts_with("* CAPABILITY IMAP4rev1"));
        assert!(last(&lines).ends_with("OK CAPABILITY completed"));
    }

    #[tokio::test]
    async fn the_server_accepts_the_password_of_its_plan() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;

        let lines = client.ask("LOGIN \"me\" \"secret\"").await;

        assert!(last(&lines).contains("OK [CAPABILITY"));
    }

    #[tokio::test]
    async fn the_server_refuses_another_password() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;

        let lines = client.ask("LOGIN \"me\" \"guess\"").await;

        assert!(last(&lines).contains("NO [AUTHENTICATIONFAILED]"));
    }

    // -----------------------------------------------------------------
    // Unit tests: folders.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn the_server_lists_its_folders() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;

        let lines = client.ask("LIST \"\" \"*\"").await;

        assert_eq!(
            lines[..3],
            [
                "* LIST (\\HasChildren) \"/\" \"INBOX\"",
                "* LIST (\\HasNoChildren) \"/\" \"INBOX/Archive\"",
                "* LIST (\\HasNoChildren) \"/\" \"Sent\"",
            ]
        );
    }

    #[tokio::test]
    async fn a_pattern_of_one_level_lists_no_child() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;

        let lines = client.ask("LIST \"\" \"%\"").await;

        assert_eq!(
            lines[..2],
            [
                "* LIST (\\HasChildren) \"/\" \"INBOX\"",
                "* LIST (\\HasNoChildren) \"/\" \"Sent\"",
            ]
        );
    }

    #[tokio::test]
    async fn selecting_a_folder_tells_what_a_sync_needs() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;

        let lines = client.ask("EXAMINE \"INBOX\"").await;
        let all = lines.join("\n");

        assert!(all.contains("* 2 EXISTS"));
        assert!(all.contains("[UIDVALIDITY 77]"));
        assert!(all.contains("[UIDNEXT 5]"));
        assert!(all.contains("[HIGHESTMODSEQ 9]"));
        assert!(last(&lines).contains("OK [READ-ONLY]"));
    }

    #[tokio::test]
    async fn selecting_a_folder_for_writing_says_so() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;

        let lines = client.ask("SELECT \"INBOX\"").await;

        assert!(last(&lines).contains("OK [READ-WRITE]"));
    }

    #[tokio::test]
    async fn selecting_a_folder_that_is_not_there_is_a_no() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;

        let lines = client.ask("EXAMINE \"Nowhere\"").await;

        assert!(last(&lines).contains("NO "));
    }

    #[tokio::test]
    async fn a_status_counts_the_messages_of_a_folder() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;

        let lines = client
            .ask(
                "STATUS \"INBOX\" (MESSAGES UIDNEXT UIDVALIDITY HIGHESTMODSEQ)",
            )
            .await;

        assert_eq!(
            lines[0],
            "* STATUS \"INBOX\" (MESSAGES 2 UIDNEXT 5 UIDVALIDITY 77 \
             HIGHESTMODSEQ 9)"
        );
    }

    // -----------------------------------------------------------------
    // Unit tests: fetching.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn a_uid_fetch_gives_the_whole_message() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;
        client.ask("EXAMINE \"INBOX\"").await;

        let lines = client.send("UID FETCH 1:* (UID BODY.PEEK[])").await;
        let first = lex(&lines[0]).unwrap();

        assert_eq!(
            first,
            vec![
                Token::Atom("*".to_string()),
                Token::Atom("1".to_string()),
                Token::Atom("FETCH".to_string()),
                Token::List(vec![
                    Token::Atom("UID".to_string()),
                    Token::Atom("1".to_string()),
                    Token::Atom("BODY[]".to_string()),
                    Token::Literal(
                        b"Subject: one\r\n\r\nthe first body\r\n".to_vec()
                    ),
                ]),
            ]
        );
    }

    #[tokio::test]
    async fn a_uid_fetch_names_the_uid_even_when_no_one_asked() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;
        client.ask("EXAMINE \"INBOX\"").await;

        let lines = client.ask("UID FETCH 4 (FLAGS)").await;

        assert_eq!(lines[0], "* 2 FETCH (UID 4 FLAGS (\\Seen))");
    }

    #[tokio::test]
    async fn a_uid_fetch_gives_only_the_uids_of_the_set() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;
        client.ask("EXAMINE \"INBOX\"").await;

        let lines = client.ask("UID FETCH 2:3 (UID)").await;

        assert_eq!(lines.len(), 1, "only the tagged answer");
        assert!(last(&lines).contains("OK UID FETCH completed"));
    }

    #[tokio::test]
    async fn a_uid_fetch_gives_the_header_alone() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;
        client.ask("EXAMINE \"INBOX\"").await;

        let lines = client.send("UID FETCH 1 (BODY.PEEK[HEADER])").await;

        assert_eq!(
            lines[0],
            b"* 1 FETCH (UID 1 BODY[HEADER] {16}\r\nSubject: one\r\n\r\n)\r\n"
                .to_vec()
        );
    }

    #[tokio::test]
    async fn changedsince_gives_only_what_moved() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;
        client.ask("EXAMINE \"INBOX\"").await;

        let lines = client
            .ask("UID FETCH 1:* (UID MODSEQ) (CHANGEDSINCE 5)")
            .await;

        assert_eq!(lines[0], "* 2 FETCH (UID 4 MODSEQ (9))");
        assert_eq!(lines.len(), 2);
    }

    #[tokio::test]
    async fn vanished_names_the_messages_that_went() {
        let mut plan = a_plan();
        plan.folder_mut("INBOX").unwrap().gone = vec![2, 3, 8];
        let server = FakeServer::start(plan).await.unwrap();
        let (mut client, _) = Client::connect(&server).await;
        client.ask("EXAMINE \"INBOX\"").await;

        let lines = client
            .ask("UID FETCH 1:6 (UID) (CHANGEDSINCE 100 VANISHED)")
            .await;

        assert_eq!(lines[0], "* VANISHED (EARLIER) 2:3");
    }

    #[tokio::test]
    async fn a_uid_search_names_every_uid() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;
        client.ask("EXAMINE \"INBOX\"").await;

        let lines = client.ask("UID SEARCH 1:* ALL").await;

        assert_eq!(lines[0], "* SEARCH 1 4");
    }

    #[tokio::test]
    async fn a_fetch_without_a_folder_is_a_no() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;

        let lines = client.ask("UID FETCH 1:* (UID)").await;

        assert!(last(&lines).contains("NO no folder is selected"));
    }

    // -----------------------------------------------------------------
    // Unit tests: the server never lets a client write.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn a_write_command_is_a_no() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;
        client.ask("EXAMINE \"INBOX\"").await;

        for command in [
            "STORE 1 +FLAGS (\\Seen)",
            "UID STORE 1 +FLAGS (\\Deleted)",
            "EXPUNGE",
            "APPEND \"INBOX\" (\\Seen) \"a body\"",
            "UID COPY 1 \"Sent\"",
            "DELETE \"Sent\"",
        ] {
            let lines = client.ask(command).await;

            assert!(
                last(&lines).contains("NO "),
                "the server took `{command}`"
            );
        }

        assert_eq!(server.writes().len(), 6);
    }

    #[tokio::test]
    async fn a_server_that_only_read_records_no_write() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;
        client.ask("LOGIN \"me\" \"secret\"").await;
        client.ask("LIST \"\" \"*\"").await;
        client.ask("EXAMINE \"INBOX\"").await;
        client.ask("UID FETCH 1:* (UID BODY.PEEK[])").await;

        assert!(server.writes().is_empty());
    }

    #[test]
    fn a_command_that_changes_a_server_reads_as_a_write() {
        assert!(is_write("a1 STORE 1 +FLAGS (\\Seen)"));
        assert!(is_write("a1 UID STORE 1 +FLAGS (\\Seen)"));
        assert!(is_write("a1 uid move 1 \"Sent\""));
        assert!(!is_write("a1 UID FETCH 1:* (UID)"));
        assert!(!is_write("a1 SELECT \"STORE\""));
    }

    // -----------------------------------------------------------------
    // Unit tests: what the server saw.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn the_server_keeps_every_command() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;
        client.ask("NOOP").await;
        client.ask("CAPABILITY").await;

        assert_eq!(server.seen().commands, ["a0001 NOOP", "a0002 CAPABILITY"]);
    }

    #[tokio::test]
    async fn the_server_counts_its_connections() {
        let server = a_server().await;
        let (mut one, _) = Client::connect(&server).await;
        let (mut two, _) = Client::connect(&server).await;
        one.ask("NOOP").await;
        two.ask("NOOP").await;

        let seen = server.seen();

        assert_eq!(seen.connections, 2);
        assert_eq!(seen.most_open, 2);
    }

    #[tokio::test]
    async fn the_server_cuts_the_connection_after_the_count_of_the_plan() {
        let server = FakeServer::start(a_plan().cut_after(2)).await.unwrap();
        let (mut client, _) = Client::connect(&server).await;

        client.ask("NOOP").await;
        client.writer.write_all(b"a0002 NOOP\r\n").await.unwrap();

        assert!(matches!(
            read_line(&mut client.reader).await,
            Err(crate::Error::Closed)
        ));
    }

    #[tokio::test]
    async fn the_answer_holds_the_tag_exactly_as_it_arrived() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;

        client.writer.write_all(b"xYz9 NOOP\r\n").await.unwrap();
        client.writer.flush().await.unwrap();

        assert_eq!(client.take().await, "xYz9 OK NOOP completed");
    }

    #[tokio::test]
    async fn logout_says_bye() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;

        let lines = client.ask("LOGOUT").await;

        assert_eq!(lines[0], "* BYE mailbert fake is closing");
        assert!(last(&lines).contains("OK LOGOUT completed"));
    }

    // -----------------------------------------------------------------
    // Unit tests: idle.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn idle_ends_when_the_client_says_done() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;
        client.ask("EXAMINE \"INBOX\"").await;

        let tag = client.tags.next_tag();
        client
            .writer
            .write_all(format!("{tag} IDLE\r\n").as_bytes())
            .await
            .unwrap();
        assert_eq!(client.take().await, "+ idling");

        client.writer.write_all(b"DONE\r\n").await.unwrap();

        assert_eq!(client.take().await, format!("{tag} OK IDLE terminated"));
    }

    #[tokio::test]
    async fn a_message_that_arrives_during_idle_shows_as_exists() {
        let server = a_server().await;
        let (mut client, _) = Client::connect(&server).await;
        client.ask("EXAMINE \"INBOX\"").await;

        let tag = client.tags.next_tag();
        client
            .writer
            .write_all(format!("{tag} IDLE\r\n").as_bytes())
            .await
            .unwrap();
        assert_eq!(client.take().await, "+ idling");

        server.change(|plan| {
            plan.folder_mut("INBOX")
                .unwrap()
                .messages
                .push(FakeMessage::new(5, "Subject: three\r\n\r\nnew\r\n"));
        });

        assert_eq!(client.take().await, "* 3 EXISTS");

        client.writer.write_all(b"DONE\r\n").await.unwrap();
        assert_eq!(client.take().await, format!("{tag} OK IDLE terminated"));
    }

    // -----------------------------------------------------------------
    // Mail in bulk, for a benchmark. (§10.5)
    // -----------------------------------------------------------------

    /// A benchmark of 10000 messages must not write 10000 literals.
    #[test]
    fn bulk_gives_as_many_messages_as_it_is_asked_for() {
        assert_eq!(bulk(2500, 1024).len(), 2500);
    }

    /// A plan reads UIDs from 1, so the first message must hold UID 1.
    #[test]
    fn bulk_numbers_the_uids_from_one() {
        let messages = bulk(4, 64);
        let uids: Vec<u32> = messages.iter().map(|one| one.uid).collect();

        assert_eq!(uids, vec![1, 2, 3, 4]);
    }

    /// The store keys a message by its bytes. Messages that share their
    /// bytes become one entry, and a benchmark of 10000 would measure 1.
    #[test]
    fn every_message_of_bulk_has_its_own_identity() {
        let messages = bulk(500, 512);
        let seen: BTreeSet<Vec<u8>> =
            messages.iter().map(|one| one.raw.clone()).collect();
        let bodies: BTreeSet<String> = messages
            .iter()
            .map(|one| {
                mailbert_core::mime::parse(&one.raw)
                    .expect("bulk gives a message that parses")
                    .text
                    .lines()
                    .skip(1)
                    .collect()
            })
            .collect();

        assert_eq!(seen.len(), 500, "two messages share their bytes");
        let senders: BTreeSet<String> = messages
            .iter()
            .map(|one| {
                mailbert_core::mime::parse(&one.raw)
                    .expect("bulk gives a message that parses")
                    .from
                    .first()
                    .expect("bulk gives a message that has a sender")
                    .address
                    .clone()
            })
            .collect();

        assert!(
            bodies.len() > 400,
            "the prose of the mail says the same thing"
        );
        assert_eq!(senders.len(), 500, "one person wrote the whole mailbox");
    }

    /// The size of a message decides how much the socket carries. A
    /// benchmark that asks for 4 KiB must not get 40 bytes.
    /// A thread comes from the `Message-ID` of a message. If every
    /// message of a benchmark holds the same one, the whole mailbox
    /// becomes one thread, and the index pass measures the wrong shape.
    #[test]
    fn every_message_of_bulk_has_its_own_message_id() {
        let ids: BTreeSet<String> = bulk(500, 512)
            .iter()
            .map(|one| {
                mailbert_core::mime::parse(&one.raw)
                    .expect("bulk gives a message that parses")
                    .message_id
                    .expect("bulk gives a message that has an identity")
            })
            .collect();

        assert_eq!(ids.len(), 500);
    }

    #[test]
    fn bulk_from_numbers_the_uids_from_where_it_was_told() {
        let uids: Vec<u32> =
            bulk_from(500, 3, 128).iter().map(|one| one.uid).collect();

        assert_eq!(uids, vec![500, 501, 502]);
    }

    /// A benchmark that spreads mail across folders needs each folder
    /// to hold its own mail. The store keeps one entry for one set of
    /// bytes, so two folders of one mail would measure half the work.
    #[test]
    fn two_folders_that_start_apart_share_no_message() {
        let mine = FakeFolder::filled_from("A", 1, 10, 256);
        let yours = FakeFolder::filled_from("B", 11, 10, 256);

        let seen: BTreeSet<Vec<u8>> =
            mine.messages.iter().map(|one| one.raw.clone()).collect();

        assert!(
            yours.messages.iter().all(|one| !seen.contains(&one.raw)),
            "the two folders hold the same mail"
        );
    }

    /// The threading of §4.1 merges two messages that share a subject,
    /// a participant, and a day. Mail in bulk shares its reader and its
    /// date. A shared subject would make the whole mailbox one thread,
    /// and the index pass of a benchmark would read the wrong number.
    #[test]
    fn the_mail_of_bulk_does_not_fall_into_one_thread() {
        let inputs: Vec<ThreadInput> = bulk(200, 512)
            .iter()
            .map(|one| {
                let mail = mailbert_core::mime::parse(&one.raw)
                    .expect("bulk gives a message that parses");
                let header = mail
                    .message_id
                    .clone()
                    .expect("bulk gives a message that has an identity");
                let id = MessageId::from_message_id(&header)
                    .expect("the header holds a whole identity");
                let who = mail
                    .from
                    .iter()
                    .chain(mail.to.iter())
                    .map(|one| one.address.clone())
                    .collect();

                ThreadInput::new(id, &mail.subject, mail.date.unwrap_or(0))
                    .with_message_id(header)
                    .with_participants(who)
            })
            .collect();

        assert_eq!(thread(&inputs).len(), 200, "the mail became few threads");
    }

    #[test]
    fn a_message_of_bulk_is_about_as_long_as_it_asked_for() {
        let one = &bulk(1, 4096)[0];

        assert!(
            (4096..4096 + 512).contains(&one.raw.len()),
            "a 4096 byte message came out {} bytes long",
            one.raw.len()
        );
    }

    /// A body shorter than the header is still a message that parses.
    #[test]
    fn bulk_gives_a_message_that_mailbert_can_read() {
        let one = &bulk(1, 0)[0];

        mailbert_core::mime::parse(&one.raw).expect("a message");
    }

    /// A folder that holds mail in bulk is what a benchmark opens.
    #[test]
    fn a_filled_folder_holds_the_mail_that_it_was_given() {
        let folder = FakeFolder::filled("INBOX", 12, 2048);

        assert_eq!(folder.messages.len(), 12);
        assert_eq!(folder.uid_next(), 13);
        assert!(
            folder.messages.iter().all(|one| one.raw.len() >= 2048),
            "a folder gave shorter mail than it was asked for"
        );
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 100)]
    fn prop_a_star_pattern_names_every_folder(tc: TestCase) {
        let name: String =
            tc.draw(gs::text().alphabet("ab/.").min_size(0).max_size(10));

        assert!(matches("*", &name, '/'));
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_percent_never_crosses_the_separator(tc: TestCase) {
        let name: String =
            tc.draw(gs::text().alphabet("ab/.").min_size(0).max_size(10));

        assert_eq!(matches("%", &name, '/'), !name.contains('/'));
    }

    /// A benchmark measures one message for each message that it asked
    /// for. Two messages that share their bytes become one entry in the
    /// store, so the count that the bench reports would be a lie.
    #[hegel::test(test_cases = 60)]
    fn prop_bulk_gives_one_identity_for_each_message(tc: TestCase) {
        let count: u32 =
            tc.draw(gs::integers::<u32>().min_value(1).max_value(60));
        let size: usize =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(2048));

        let messages = bulk(count, size);
        let seen: BTreeSet<Vec<u8>> =
            messages.iter().map(|one| one.raw.clone()).collect();

        assert_eq!(messages.len(), count as usize);
        assert_eq!(seen.len(), count as usize);
    }
}
