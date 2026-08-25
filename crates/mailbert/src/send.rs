//! `send`: one message out through a submission server. (§11)
//!
//! Every other command of mailbert reads. This one writes, and it
//! writes in two places: the submission server takes the message, and
//! the local store takes a copy. The store copy is what makes a `send`
//! visible to `search` at once, without waiting for the server to hand
//! the message back down its own `Sent` folder. §11.3 says why that
//! copy never goes up over IMAP: mailbert is still a mirror that only
//! reads, and a message written into `Sent` from here would be a write.
//!
//! A later sync that finds the server's own copy does not double it.
//! §4.1 gives both copies the same identity, because both carry the
//! same `Message-ID`, so the second one lands as another location of
//! the entry that is already there.

use std::{
    collections::BTreeSet,
    io::{Read, Write},
};

use lettre::{
    AsyncSmtpTransport,
    AsyncTransport,
    Tokio1Executor,
    message::{Mailbox, MessageBuilder},
    transport::smtp::authentication::Credentials,
};
use mailbert_core::{
    Store,
    address,
    config::{Account, Config, Tls},
    index::MailIndex,
    message::{Location, Message},
    mime,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    Tool,
    cli,
    error::{Error, Result},
    pass,
    settings,
    show,
};

/// The `User-Agent` that a sent message carries.
const AGENT: &str = concat!("mailbert ", env!("CARGO_PKG_VERSION"));

/// What `send` writes as JSON. (§10.4)
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Answer {
    /// The identity of §4.1, in full.
    pub id: String,

    /// The `Message-ID` that the message carries.
    pub message_id: String,

    /// The account that sent it.
    pub account: String,

    /// The `From` header.
    pub from: String,

    /// The `To` addresses.
    pub to: Vec<String>,

    /// The `Cc` addresses.
    pub cc: Vec<String>,

    /// The `Bcc` addresses, which the message itself does not name.
    pub bcc: Vec<String>,

    /// The subject.
    pub subject: String,

    /// The folder that the local copy went into. (§11.3)
    pub folder: String,
}

/// What a message says before lettre turns it into bytes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Draft {
    /// The sender, as the `From` header will read.
    pub from: String,

    /// The recipients.
    pub to: Vec<String>,

    /// The carbon copies.
    pub cc: Vec<String>,

    /// The blind carbon copies. The envelope carries these, and no
    /// header names them.
    pub bcc: Vec<String>,

    /// The subject.
    pub subject: String,

    /// The body, as plain text.
    pub body: String,

    /// The `Message-ID` that this answers, with its angle brackets.
    pub in_reply_to: Option<String>,

    /// The `References` chain, each with its angle brackets.
    pub references: Vec<String>,
}

/// One message that a caller asks `send` to write. (§11)
///
/// The CLI of §2.1 fills this from its flags and the standard input,
/// and the MCP tool of §2.2 takes it as the arguments of one call, so
/// the two doors into `send` open on one room.
#[derive(
    Debug, Clone, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema,
)]
pub struct Letter {
    /// The recipients. Required, unless `reply_to` names a message to
    /// take them from.
    #[serde(default)]
    pub to: Vec<String>,

    /// The carbon copies.
    #[serde(default)]
    pub cc: Vec<String>,

    /// The blind carbon copies. No header names these, so a recipient
    /// never learns that they took a copy.
    #[serde(default)]
    pub bcc: Vec<String>,

    /// The subject. An answer that leaves it empty takes the answered
    /// subject, with one `Re: ` on it.
    #[serde(default)]
    pub subject: String,

    /// The body, as plain text.
    #[serde(default)]
    pub body: String,

    /// The identity of a message to answer, or a prefix of it. (§4.1)
    #[serde(default)]
    pub reply_to: Option<String>,

    /// Answer everyone the message named, and not its sender alone.
    #[serde(default)]
    pub reply_all: bool,

    /// The account to send through. The one account that can send
    /// speaks when this is absent.
    #[serde(default)]
    pub account: Option<String>,
}

impl Letter {
    /// The letter that the flags and the standard input describe.
    ///
    /// The body is `--body`, or the whole standard input when that
    /// flag is absent, because §11 leaves the writing to your editor.
    ///
    /// # Errors
    ///
    /// The function fails if the input cannot be read.
    pub fn read(args: &cli::Send, input: &mut dyn Read) -> Result<Self> {
        let body = match &args.body {
            Some(text) => text.clone(),
            None => {
                let mut text = String::new();
                input.read_to_string(&mut text)?;
                text
            }
        };

        Ok(Self {
            to: args.to.clone(),
            cc: args.cc.clone(),
            bcc: args.bcc.clone(),
            subject: args.subject.clone().unwrap_or_default(),
            body,
            reply_to: args.reply_to.clone(),
            reply_all: args.reply_all,
            account: args.account.clone(),
        })
    }
}

/// Write one message, and file a copy. (§11)
///
/// # Errors
///
/// The function fails if no account can send, if an address is not an
/// address, if the standard input is not readable, if the server
/// refuses the message, or if the store refuses the copy.
pub fn command(tool: &Tool, args: &cli::Send) -> Result<()> {
    let config = tool.config()?;
    let store = tool.store()?;
    let letter = Letter::read(args, &mut std::io::stdin())?;
    let mut out = std::io::stdout().lock();

    // A dry run writes the message itself, and neither `--json` nor the
    // one-line report says anything a reader could not read there. It
    // is the only output of the tool that is bytes and not a report.
    // It opens no index and reads no password, either.
    if args.dry_run {
        let account = pick(&config, letter.account.as_deref())?;
        let message = compose(&draft(&store, account, &letter)?)?;

        out.write_all(&message.formatted())?;

        return Ok(());
    }

    let index = tool.index()?;
    let answer = crate::block_on(run(&store, &index, &config, &letter))?;

    match args.json {
        true => write_json(&answer, &mut out),
        false => write_text(&answer, &mut out),
    }
}

/// Submit one letter, and file the copy. (§11)
///
/// This is what both doors into `send` walk through: the CLI of §2.1
/// after it reads the standard input, and the MCP tool of §2.2 with
/// the arguments of the call.
///
/// # Errors
///
/// The function fails if no account can send, if the letter names no
/// recipient, if an address is not an address, if the server refuses
/// the message, or if the store refuses the copy.
pub async fn run(
    store: &Store,
    index: &MailIndex,
    config: &Config,
    letter: &Letter,
) -> Result<Answer> {
    let account = pick(config, letter.account.as_deref())?;
    let draft = draft(store, account, letter)?;
    let message = compose(&draft)?;
    let raw = message.formatted();
    let password = settings::smtp_secret(account)?;

    submit(account, &password, message).await?;

    let filed = file(store, index, account, &raw, crate::clock().now())?;

    Ok(Answer {
        id: filed.id.full_hex(),
        message_id: filed.message_id.unwrap_or_default(),
        account: account.name.clone(),
        from: draft.from,
        to: draft.to,
        cc: draft.cc,
        bcc: draft.bcc,
        subject: draft.subject,
        folder: account.sent.clone(),
    })
}

/// The account that a `send` goes out through.
///
/// A named account must be able to send, and says so by name when it
/// cannot. With no name, the one account that has an `[account.smtp]`
/// speaks, because an unnamed `send` on a machine with two sending
/// accounts would pick a `From` at random.
///
/// # Errors
///
/// The function fails if the name is unknown, if the named account has
/// no submission server, or if no exactly-one account has one.
pub fn pick<'a>(config: &'a Config, name: Option<&str>) -> Result<&'a Account> {
    if let Some(name) = name {
        let found = config
            .account(name)
            .ok_or_else(|| Error::UnknownAccount(name.to_string()))?;

        found.smtp()?;

        return Ok(found);
    }

    let mut senders = config.accounts.iter().filter(|one| one.smtp.is_some());

    let first = senders.next().ok_or(Error::NoSender)?;

    match senders.next() {
        None => Ok(first),
        Some(_) => Err(Error::ManySenders(
            config
                .accounts
                .iter()
                .filter(|one| one.smtp.is_some())
                .map(|one| one.name.clone())
                .collect(),
        )),
    }
}

/// Build the draft that one letter describes.
///
/// # Errors
///
/// The function fails if `reply_to` names no message in the store, or
/// if nothing is left holding a recipient.
pub fn draft(
    store: &Store,
    account: &Account,
    letter: &Letter,
) -> Result<Draft> {
    let answered = letter
        .reply_to
        .as_deref()
        .map(|prefix| answered(store, prefix))
        .transpose()?;

    let mut draft = Draft {
        from: account.sender(),
        to: letter.to.clone(),
        cc: letter.cc.clone(),
        bcc: letter.bcc.clone(),
        subject: letter.subject.clone(),
        body: letter.body.clone(),
        ..Draft::default()
    };

    if let Some(answered) = answered {
        inherit(&mut draft, &answered, letter, account);
    }

    // The CLI of §2.1 has clap ask for a recipient, and a caller of
    // §2.2 has nobody asking, so the check lives here for both.
    if draft.to.is_empty() && draft.cc.is_empty() && draft.bcc.is_empty() {
        return Err(Error::NoRecipient);
    }

    Ok(draft)
}

/// The message that `--reply-to` names.
fn answered(store: &Store, prefix: &str) -> Result<Message> {
    let id = show::resolve(store, prefix)?;

    store.get(&id)?.ok_or_else(|| {
        Error::Core(mailbert_core::Error::UnknownMessage(id.short()))
    })
}

/// Take from the answered message what an answer inherits. (§11.2)
///
/// The recipients, the subject and the thread all come from it, and
/// each of them steps aside for a flag that says otherwise. The
/// `References` chain grows by the answered message, which is what
/// keeps §5.5 threading the two together in every other mail program
/// as well as in mailbert.
fn inherit(
    draft: &mut Draft,
    answered: &Message,
    letter: &Letter,
    account: &Account,
) {
    if draft.to.is_empty() {
        draft.to = answered.from.iter().map(ToString::to_string).collect();
    }

    // `--reply-all` writes to everyone the message named, less the
    // addresses of this account, so that an answer does not go to you.
    if letter.reply_all {
        let mine = own(account);
        let named = answered
            .to
            .iter()
            .chain(answered.cc.iter())
            .filter(|one| !mine.contains(&one.address))
            .map(ToString::to_string);

        draft.cc.extend(named);
        draft.cc.sort();
        draft.cc.dedup();
    }

    if draft.subject.is_empty() {
        draft.subject = answering(&answered.subject);
    }

    let Some(message_id) = &answered.message_id else {
        return;
    };

    draft.in_reply_to = Some(format!("<{message_id}>"));
    draft.references = answered
        .references
        .iter()
        .chain(std::iter::once(message_id))
        .map(|one| format!("<{one}>"))
        .collect();
}

/// The addresses that one account speaks under.
fn own(account: &Account) -> BTreeSet<String> {
    [
        Some(account.imap.user.clone()),
        account.smtp.as_ref().and_then(|smtp| smtp.user.clone()),
        account.from.clone(),
    ]
    .into_iter()
    .flatten()
    .filter_map(|name| address::parse(&name).map(|one| one.address))
    .collect()
}

/// The subject of an answer: one `Re: `, however many the original had.
fn answering(subject: &str) -> String {
    let mut rest = subject.trim();

    while let Some(shorter) = strip_re(rest) {
        rest = shorter.trim_start();
    }

    format!("Re: {rest}")
}

/// The text after a leading `Re:`, in whatever case it was written.
fn strip_re(subject: &str) -> Option<&str> {
    let (head, rest) = subject.split_once(':')?;

    head.trim().eq_ignore_ascii_case("re").then_some(rest)
}

/// Turn a draft into the message that goes on the wire.
///
/// # Errors
///
/// The function fails if an address is not an address, or if lettre
/// refuses the body.
pub fn compose(draft: &Draft) -> Result<lettre::Message> {
    let from = mailbox(&draft.from)?;
    let mut builder = MessageBuilder::new()
        .message_id(Some(identity(&from)))
        .from(from)
        .subject(&draft.subject)
        .user_agent(AGENT.to_string())
        .date_now();

    for (addresses, field) in [
        (&draft.to, Field::To),
        (&draft.cc, Field::Cc),
        (&draft.bcc, Field::Bcc),
    ] {
        for one in addresses {
            let one = mailbox(one)?;

            builder = match field {
                Field::To => builder.to(one),
                Field::Cc => builder.cc(one),
                Field::Bcc => builder.bcc(one),
            };
        }
    }

    if let Some(id) = &draft.in_reply_to {
        builder = builder.in_reply_to(id.clone());
    }

    if !draft.references.is_empty() {
        builder = builder.references(draft.references.join(" "));
    }

    Ok(builder.body(draft.body.clone())?)
}

/// The `Message-ID` of a new message, with its angle brackets.
///
/// RFC 5322 §3.6.4 asks for a right side that is unique in the world,
/// and lettre writes the name of the machine there, which is unique to
/// the machine and to nothing else: two people running mailbert on a
/// laptop each named `battlecruiser` mint the same domain. The domain
/// the message is sent from is the one name it already carries that a
/// reader anywhere can resolve, so that is the one this uses.
fn identity(from: &Mailbox) -> String {
    let token: String = std::iter::repeat_with(fastrand::alphanumeric)
        .take(24)
        .collect();

    format!("<{token}@{}>", from.email.domain())
}

/// Which of the three recipient headers one address goes into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Field {
    To,
    Cc,
    Bcc,
}

/// Read one address, with or without a display name.
///
/// mailbert already knows how to read a `From` line, and lettre wants
/// its own type, so this goes through [`address::parse`] and hands the
/// pieces over. That keeps `send` reading an address the same way that
/// `search` does.
fn mailbox(text: &str) -> Result<Mailbox> {
    let Some(one) = address::parse(text) else {
        return Err(Error::BadAddress(text.to_string()));
    };

    Ok(Mailbox::new(
        one.name.clone(),
        one.address
            .parse()
            .map_err(|_| Error::BadAddress(text.to_string()))?,
    ))
}

/// Hand the message to the submission server. (§11.1)
///
/// # Errors
///
/// The function fails if the connection, the login, or the message is
/// refused.
pub async fn submit(
    account: &Account,
    password: &str,
    message: lettre::Message,
) -> Result<()> {
    let smtp = account.smtp()?;

    let builder = match smtp.tls {
        Tls::Implicit => {
            AsyncSmtpTransport::<Tokio1Executor>::relay(&smtp.host)?
        }
        Tls::Start => {
            AsyncSmtpTransport::<Tokio1Executor>::starttls_relay(&smtp.host)?
        }
        Tls::None => {
            AsyncSmtpTransport::<Tokio1Executor>::builder_dangerous(&smtp.host)
        }
    };

    let user = account.smtp_user()?.to_string();
    let transport = builder
        .port(smtp.port)
        .credentials(Credentials::new(user, password.to_string()))
        .build();

    transport.send(message).await?;

    Ok(())
}

/// Put the sent message in the store and the index. (§11.3)
///
/// The location is a local one: the account is the sending account, the
/// folder is its `sent`, and the UID is zero because no server gave one.
/// [`Message::add_location`] keys a location by account and folder, so
/// the copy that a later sync brings down from the server's own `Sent`
/// sits beside this one instead of replacing it.
///
/// # Errors
///
/// The function fails if the bytes do not parse, or if the store or the
/// index refuses the write.
pub fn file(
    store: &Store,
    index: &MailIndex,
    account: &Account,
    raw: &[u8],
    now: i64,
) -> Result<Message> {
    let location = Location {
        account: account.name.clone(),
        folder: account.sent.clone(),
        uid: 0,
        uid_validity: 0,
        received: now,
        flags: BTreeSet::new(),
    };

    // A message you wrote is one you have read.
    let message = Message::new(mime::parse(raw)?, location, ["\\Seen"]);
    let stored = store.put(&message, raw)?;

    pass::after_sync(store, index, &BTreeSet::from([stored.id]))?;

    Ok(stored)
}

/// Write the JSON of §10.4.
///
/// # Errors
///
/// The function fails if the output refuses a write.
pub fn write_json(answer: &Answer, out: &mut dyn Write) -> Result<()> {
    writeln!(out, "{}", serde_json::to_string_pretty(answer)?)?;

    Ok(())
}

/// Write the one line that a `send` says. (§11.4)
///
/// # Errors
///
/// The function fails if the output refuses a write.
pub fn write_text(answer: &Answer, out: &mut dyn Write) -> Result<()> {
    let count = answer.to.len() + answer.cc.len() + answer.bcc.len();
    let plural = match count {
        1 => "",
        _ => "s",
    };

    writeln!(
        out,
        "sent to {count} recipient{plural}, filed as {}",
        short(&answer.id)
    )?;

    Ok(())
}

/// The short identity of §4.1, from a full one.
fn short(full: &str) -> &str {
    &full[..full.len().min(7)]
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_an_answer_carries_one_re` | invariant | §11.2 says an answer reads `Re: ` once. A chain of them is what makes a long thread unreadable in every other mail program. |
    //! | `prop_an_answer_keeps_the_whole_chain` | invariant | §5.5 threads on `References`. An answer that drops one link splits the thread for everyone who reads the mail elsewhere. |
    //! | `prop_a_composed_message_parses_back` | round-trip | §11.3 files the bytes that go on the wire. Bytes that mailbert cannot read back are bytes it cannot file or find. |
    //! | `prop_a_reply_never_writes_to_itself` | invariant | `--reply-all` answers everyone the message named, and answering yourself puts your own copy back in your inbox. |

    use hegel::{TestCase, generators as gs};
    use mailbert_core::{
        config::{ImapConfig, SmtpConfig},
        message::Location,
    };
    use tempfile::TempDir;

    use super::*;

    /// An account that can send, under one name.
    fn sender() -> Account {
        Account {
            name: "work".to_string(),
            imap: ImapConfig {
                host: "imap.example.test".to_string(),
                user: "me@example.test".to_string(),
                password: Some("secret".to_string()),
                ..ImapConfig::default()
            },
            smtp: Some(SmtpConfig {
                host: "smtp.example.test".to_string(),
                ..SmtpConfig::default()
            }),
            ..Account::default()
        }
    }

    /// The flags of a `send` with nothing on them.
    fn flags() -> cli::Send {
        cli::Send {
            to: Vec::new(),
            cc: Vec::new(),
            bcc: Vec::new(),
            subject: None,
            body: None,
            reply_to: None,
            reply_all: false,
            account: None,
            dry_run: false,
            json: false,
        }
    }

    /// A store with one message in it, which an answer can name.
    struct Mailbox {
        _dir: TempDir,
        store: Store,
    }

    impl Mailbox {
        fn open() -> Self {
            let dir = tempfile::tempdir().expect("a temporary directory");
            let store =
                Store::open(&dir.path().join("store")).expect("a store");

            Self { _dir: dir, store }
        }

        /// Put one message in the store, and give back its identity.
        fn put(&self, raw: &str) -> String {
            let bytes = raw.replace('\n', "\r\n").into_bytes();
            let location = Location {
                account: "work".to_string(),
                folder: "INBOX".to_string(),
                uid: 1,
                uid_validity: 1,
                received: 1_700_000_000,
                flags: BTreeSet::new(),
            };
            let message = Message::new(
                mime::parse(&bytes).expect("a message"),
                location,
                Vec::<String>::new(),
            );

            self.store
                .put(&message, &bytes)
                .expect("a write")
                .id
                .full_hex()
        }
    }

    /// A message from Ada, to this account and to Grace.
    const FROM_ADA: &str = "\
From: Ada Lovelace <ada@example.test>
To: me@example.test, Grace Hopper <grace@example.test>
Cc: babbage@example.test
Subject: The Analytical Engine
Date: Fri, 22 Aug 2025 09:30:00 +0000
Message-ID: <engine-1@example.test>
References: <engine-0@example.test>

it weaves algebraic patterns
";

    // -----------------------------------------------------------------
    // Unit tests.
    // -----------------------------------------------------------------

    #[test]
    fn a_lone_sending_account_needs_no_name() {
        let config = Config {
            accounts: vec![sender()],
            ..Config::default()
        };

        assert_eq!(pick(&config, None).unwrap().name, "work");
    }

    #[test]
    fn an_account_that_cannot_send_is_not_picked() {
        let mut quiet = sender();
        quiet.name = "quiet".to_string();
        quiet.smtp = None;

        let config = Config {
            accounts: vec![quiet],
            ..Config::default()
        };

        assert!(matches!(pick(&config, None), Err(Error::NoSender)));
    }

    #[test]
    fn a_named_account_that_cannot_send_says_which() {
        let mut quiet = sender();
        quiet.name = "quiet".to_string();
        quiet.smtp = None;

        let config = Config {
            accounts: vec![quiet],
            ..Config::default()
        };

        assert!(matches!(
            pick(&config, Some("quiet")),
            Err(Error::Core(mailbert_core::Error::NoSmtp(name)))
                if name == "quiet"
        ));
    }

    #[test]
    fn two_sending_accounts_ask_for_a_name() {
        let mut other = sender();
        other.name = "home".to_string();

        let config = Config {
            accounts: vec![sender(), other],
            ..Config::default()
        };

        let picked = pick(&config, None);

        assert!(
            matches!(&picked, Err(Error::ManySenders(names))
                if names == &["work".to_string(), "home".to_string()]),
            "{picked:?}"
        );

        // A name resolves it.
        assert_eq!(pick(&config, Some("home")).unwrap().name, "home");
    }

    #[test]
    fn the_body_comes_from_the_standard_input() {
        let box_ = Mailbox::open();
        let args = cli::Send {
            to: vec!["alice@example.test".to_string()],
            subject: Some("Hi".to_string()),
            ..flags()
        };

        let letter = Letter::read(&args, &mut "typed in\n".as_bytes())
            .expect("a letter");
        let draft = draft(&box_.store, &sender(), &letter).expect("a draft");

        assert_eq!(draft.body, "typed in\n");
        assert_eq!(draft.from, "me@example.test");
    }

    #[test]
    fn a_flag_beats_the_standard_input() {
        let box_ = Mailbox::open();
        let args = cli::Send {
            to: vec!["alice@example.test".to_string()],
            subject: Some("Hi".to_string()),
            body: Some("on the flag".to_string()),
            ..flags()
        };

        let letter =
            Letter::read(&args, &mut "ignored".as_bytes()).expect("a letter");
        let draft = draft(&box_.store, &sender(), &letter).expect("a draft");

        assert_eq!(draft.body, "on the flag");
    }

    #[test]
    fn an_answer_goes_back_to_the_sender_and_the_thread() {
        let box_ = Mailbox::open();
        let id = box_.put(FROM_ADA);
        let letter = Letter {
            reply_to: Some(id[..7].to_string()),
            ..Letter::default()
        };

        let draft = draft(&box_.store, &sender(), &letter).expect("a draft");

        assert_eq!(draft.to, ["Ada Lovelace <ada@example.test>"]);
        assert!(draft.cc.is_empty(), "{:?}", draft.cc);
        assert_eq!(draft.subject, "Re: The Analytical Engine");
        assert_eq!(
            draft.in_reply_to.as_deref(),
            Some("<engine-1@example.test>")
        );
        assert_eq!(
            draft.references,
            ["<engine-0@example.test>", "<engine-1@example.test>"]
        );
    }

    #[test]
    fn an_answer_to_everyone_leaves_you_out() {
        let box_ = Mailbox::open();
        let id = box_.put(FROM_ADA);
        let letter = Letter {
            reply_to: Some(id[..7].to_string()),
            reply_all: true,
            ..Letter::default()
        };

        let draft = draft(&box_.store, &sender(), &letter).expect("a draft");

        assert_eq!(draft.to, ["Ada Lovelace <ada@example.test>"]);
        assert_eq!(
            draft.cc,
            ["Grace Hopper <grace@example.test>", "babbage@example.test"]
        );
    }

    #[test]
    fn a_flag_beats_what_the_answered_message_says() {
        let box_ = Mailbox::open();
        let id = box_.put(FROM_ADA);
        let letter = Letter {
            to: vec!["someone@example.test".to_string()],
            subject: "A new subject".to_string(),
            reply_to: Some(id[..7].to_string()),
            ..Letter::default()
        };

        let draft = draft(&box_.store, &sender(), &letter).expect("a draft");

        assert_eq!(draft.to, ["someone@example.test"]);
        assert_eq!(draft.subject, "A new subject");

        // The thread still holds, because the answer is still an answer.
        assert_eq!(
            draft.in_reply_to.as_deref(),
            Some("<engine-1@example.test>")
        );
    }

    #[test]
    fn a_subject_takes_one_re_however_many_it_had() {
        assert_eq!(answering("Lunch"), "Re: Lunch");
        assert_eq!(answering("Re: Lunch"), "Re: Lunch");
        assert_eq!(answering("RE: re: Re:  Lunch"), "Re: Lunch");
        assert_eq!(answering("Report: the numbers"), "Re: Report: the numbers");
    }

    #[test]
    fn a_composed_message_carries_what_the_draft_said() {
        let draft = Draft {
            from: "Ada <ada@example.test>".to_string(),
            to: vec!["grace@example.test".to_string()],
            cc: vec!["Babbage <babbage@example.test>".to_string()],
            bcc: vec!["quiet@example.test".to_string()],
            subject: "Punch cards".to_string(),
            body: "they weave patterns\n".to_string(),
            in_reply_to: Some("<a@example.test>".to_string()),
            references: vec!["<a@example.test>".to_string()],
        };

        let message = compose(&draft).expect("a message");
        let text = String::from_utf8(message.formatted()).expect("text");

        assert!(text.contains("From: Ada <ada@example.test>"), "{text}");
        assert!(text.contains("To: grace@example.test"), "{text}");
        assert!(
            text.contains("Cc: Babbage <babbage@example.test>"),
            "{text}"
        );
        assert!(text.contains("In-Reply-To: <a@example.test>"), "{text}");
        assert!(text.contains("they weave patterns"), "{text}");

        // A blind copy is on the envelope, and in no header.
        assert!(!text.contains("quiet@example.test"), "{text}");
        assert!(
            message
                .envelope()
                .to()
                .iter()
                .any(|one| one.to_string() == "quiet@example.test"),
            "the envelope carries the blind copy"
        );
    }

    #[test]
    fn a_subject_that_is_not_ascii_goes_out_encoded_and_comes_back_whole() {
        let draft = Draft {
            from: "ada@example.test".to_string(),
            to: vec!["grace@example.test".to_string()],
            subject: "Höhere Analysis — für Ada".to_string(),
            ..Draft::default()
        };

        let raw = compose(&draft).expect("a message").formatted();
        let text = String::from_utf8(raw.clone()).expect("text");
        let line = text
            .lines()
            .find(|line| line.starts_with("Subject:"))
            .expect("a subject");

        // A header is seven bits wide, so the word goes out spelled
        // out in the alphabet of RFC 2047.
        assert!(line.is_ascii(), "{line}");
        assert!(line.contains("=?utf-8?"), "{line}");

        let parsed = mime::parse(&raw).expect("the bytes parse back");
        assert_eq!(parsed.subject, "Höhere Analysis — für Ada");
    }

    #[test]
    fn the_message_id_names_the_domain_that_sent_it() {
        let draft = Draft {
            from: "Ada <ada@example.test>".to_string(),
            to: vec!["grace@example.test".to_string()],
            ..Draft::default()
        };

        let identity = |draft: &Draft| {
            mime::parse(&compose(draft).expect("a message").formatted())
                .expect("the bytes parse back")
                .message_id
                .expect("an identity")
        };

        let first = identity(&draft);
        let second = identity(&draft);

        // The name of this machine resolves on this machine alone, and
        // RFC 5322 §3.6.4 wants a right side that resolves anywhere.
        assert!(first.ends_with("@example.test"), "{first}");
        assert_ne!(first, second, "two messages, two identities");
    }

    #[test]
    fn a_letter_with_nobody_to_read_it_is_refused() {
        let box_ = Mailbox::open();
        let letter = Letter {
            subject: "Hi".to_string(),
            body: "there".to_string(),
            ..Letter::default()
        };

        assert!(matches!(
            draft(&box_.store, &sender(), &letter),
            Err(Error::NoRecipient)
        ));

        // A blind copy is a recipient, even though no header says so.
        let quiet = Letter {
            bcc: vec!["quiet@example.test".to_string()],
            ..letter
        };

        assert!(draft(&box_.store, &sender(), &quiet).is_ok());
    }

    #[test]
    fn a_name_that_is_not_an_address_is_an_error() {
        let draft = Draft {
            from: "not an address".to_string(),
            to: vec!["grace@example.test".to_string()],
            ..Draft::default()
        };

        assert!(matches!(compose(&draft), Err(Error::BadAddress(_))));
    }

    #[test]
    fn a_sent_message_lands_in_the_sent_folder_as_read() {
        let box_ = Mailbox::open();
        let dir = tempfile::tempdir().expect("a temporary directory");
        let index = MailIndex::open(&dir.path().join("index")).expect("index");
        let draft = Draft {
            from: "me@example.test".to_string(),
            to: vec!["grace@example.test".to_string()],
            subject: "Punch cards".to_string(),
            body: "they weave patterns\n".to_string(),
            ..Draft::default()
        };
        let raw = compose(&draft).expect("a message").formatted();

        let filed = file(&box_.store, &index, &sender(), &raw, 1_700_000_000)
            .expect("a write");

        assert_eq!(filed.subject, "Punch cards");
        assert_eq!(filed.locations.len(), 1);
        assert_eq!(filed.locations[0].folder, "Sent");
        assert_eq!(filed.locations[0].account, "work");
        assert!(filed.flags.contains("\\seen"), "{:?}", filed.flags);

        // The index found it, so a `search` would too.
        assert!(index.get(&filed.id).expect("a lookup").is_some());
    }

    #[test]
    fn the_copy_that_comes_back_down_joins_the_one_that_is_there() {
        // §11.3: a later sync of the server's own Sent folder must not
        // make a second message out of the one that `send` filed.
        let box_ = Mailbox::open();
        let dir = tempfile::tempdir().expect("a temporary directory");
        let index = MailIndex::open(&dir.path().join("index")).expect("index");
        let draft = Draft {
            from: "me@example.test".to_string(),
            to: vec!["grace@example.test".to_string()],
            subject: "Punch cards".to_string(),
            body: "they weave patterns\n".to_string(),
            ..Draft::default()
        };
        let raw = compose(&draft).expect("a message").formatted();
        let filed = file(&box_.store, &index, &sender(), &raw, 1_700_000_000)
            .expect("a write");

        // The same bytes, now arriving from the server with a UID.
        let back = Message::new(
            mime::parse(&raw).expect("a message"),
            Location {
                account: "work".to_string(),
                folder: "[Gmail]/Sent Mail".to_string(),
                uid: 42,
                uid_validity: 7,
                received: 1_700_000_100,
                flags: BTreeSet::new(),
            },
            ["\\Seen"],
        );
        let merged = box_.store.put(&back, &raw).expect("a write");

        assert_eq!(merged.id, filed.id);
        assert_eq!(box_.store.len().expect("a count"), 1);
        assert_eq!(merged.locations.len(), 2);
    }

    #[test]
    fn the_report_says_where_the_copy_went() {
        let answer = Answer {
            id: "4f2a1c9dead".to_string(),
            message_id: "x@example.test".to_string(),
            account: "work".to_string(),
            from: "me@example.test".to_string(),
            to: vec!["alice@example.test".to_string()],
            cc: Vec::new(),
            bcc: Vec::new(),
            subject: "Hi".to_string(),
            folder: "Sent".to_string(),
        };

        let mut out = Vec::new();
        write_text(&answer, &mut out).expect("a write");

        assert_eq!(
            String::from_utf8(out).expect("text"),
            "sent to 1 recipient, filed as 4f2a1c9\n"
        );
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    /// A subject, sometimes already answered a few times over.
    #[hegel::composite]
    fn subject(tc: TestCase) -> String {
        let stem: String =
            tc.draw(gs::text().alphabet("abc ").min_size(1).max_size(8));
        let depth: usize =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(4));
        let prefix: String = tc.draw(gs::sampled_from(vec![
            "Re: ".to_string(),
            "RE: ".to_string(),
            "re:".to_string(),
            "Re:  ".to_string(),
        ]));

        format!("{}{}", prefix.repeat(depth), stem.trim())
    }

    #[hegel::test(test_cases = 200)]
    fn prop_an_answer_carries_one_re(tc: TestCase) {
        let subject: String = tc.draw(subject());
        let answered = answering(&subject);

        assert!(answered.starts_with("Re: "), "{answered:?}");
        assert!(strip_re(&answered[4..]).is_none(), "{answered:?}");

        // Answering twice is answering once.
        assert_eq!(answering(&answered), answered);
    }

    #[hegel::test(test_cases = 100)]
    fn prop_an_answer_keeps_the_whole_chain(tc: TestCase) {
        let depth: usize =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(6));

        let chain: Vec<String> = (0..depth)
            .map(|at| format!("<link{at}@example.test>"))
            .collect();
        let references = match chain.is_empty() {
            true => String::new(),
            false => format!("References: {}\n", chain.join(" ")),
        };
        let raw = format!(
            "From: ada@example.test\n\
             To: me@example.test\n\
             Subject: chain\n\
             Date: Fri, 22 Aug 2025 09:30:00 +0000\n\
             Message-ID: <last@example.test>\n\
             {references}\n\
             body\n"
        );

        let box_ = Mailbox::open();
        let id = box_.put(&raw);
        let letter = Letter {
            reply_to: Some(id[..7].to_string()),
            ..Letter::default()
        };
        let draft = draft(&box_.store, &sender(), &letter).expect("a draft");

        let mut want = chain;
        want.push("<last@example.test>".to_string());

        assert_eq!(draft.references, want);
        assert_eq!(
            draft.in_reply_to.as_deref(),
            draft.references.last().map(String::as_str)
        );
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_composed_message_parses_back(tc: TestCase) {
        let drawn: String =
            tc.draw(gs::text().alphabet("abcé ").min_size(0).max_size(20));
        let body: String =
            tc.draw(gs::text().alphabet("abc\n é").min_size(0).max_size(40));
        let names: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(4));

        // A run of spaces around a word that has to be encoded is not
        // a thing a header can hold: the encoder spends one encoded
        // word per run, and RFC 2047 has a reader drop the space
        // between two of them. One space between words survives, so
        // one space between words is what the subject is written with.
        let subject = drawn.split_whitespace().collect::<Vec<_>>().join(" ");

        let draft = Draft {
            from: "me@example.test".to_string(),
            to: (0..names)
                .map(|at| format!("one{at}@example.test"))
                .collect(),
            subject: subject.clone(),
            body,
            ..Draft::default()
        };

        let raw = compose(&draft).expect("a message").formatted();
        let parsed = mime::parse(&raw).expect("the bytes parse back");

        assert_eq!(parsed.subject, subject);
        assert_eq!(parsed.to.len(), names);
        assert!(parsed.message_id.is_some(), "every message has an identity");
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_reply_never_writes_to_itself(tc: TestCase) {
        let at: usize =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(3));
        let from: bool = tc.draw(gs::booleans());

        // The account speaks under one of its three names, and the
        // answered message names it among the others.
        let mut account = sender();
        account.from = from.then(|| "Ada <ada@example.test>".to_string());

        let mine = ["me@example.test", "ada@example.test"];
        let named: Vec<String> = (0..4)
            .map(|one| match one == at {
                true => mine[usize::from(from)].to_string(),
                false => format!("other{one}@example.test"),
            })
            .collect();

        let raw = format!(
            "From: sender@example.test\n\
             To: {}\n\
             Cc: {}\n\
             Subject: chain\n\
             Date: Fri, 22 Aug 2025 09:30:00 +0000\n\
             Message-ID: <one@example.test>\n\n\
             body\n",
            named[..2].join(", "),
            named[2..].join(", ")
        );

        let box_ = Mailbox::open();
        let id = box_.put(&raw);
        let letter = Letter {
            reply_to: Some(id[..7].to_string()),
            reply_all: true,
            ..Letter::default()
        };
        let draft = draft(&box_.store, &account, &letter).expect("a draft");

        for one in draft.to.iter().chain(draft.cc.iter()) {
            for name in mine {
                assert!(!one.contains(name), "answered {name} in {one:?}");
            }
        }
    }
}
