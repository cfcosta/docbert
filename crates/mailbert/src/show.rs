//! `get` and `view`: one message, plain or with color. (§10.2, §10.3)
//!
//! `get` is the machine-readable form. It writes minimal headers and
//! the decoded text, and it writes no color at all. An agent reads it,
//! and a shell pipes it into another program.
//!
//! `view` writes ANSI. It colors the headers, colors each quote by its
//! depth, highlights the code blocks of the body, and obeys `NO_COLOR`.
//! It does not page, so a reader pipes it to `less -R`.
//!
//! Only `view` decrypts. §5.4 keeps the ciphertext of an encrypted
//! message out of the index, and out of `get`, because the index and
//! its backup are plaintext files. A reader who wants the plaintext
//! asks for it, and [`crate::pgp`] asks gpg-agent at that moment.

use std::io::Write;

use mailbert_core::{
    Store,
    address::Address,
    message_id::{MessageId, PrefixMatch},
    mime,
};
use serde::Serialize;
use syntect::{
    easy::HighlightLines,
    highlighting::{Theme, ThemeSet},
    parsing::SyntaxSet,
    util::{LinesWithEndings, as_24_bit_terminal_escaped},
};

use crate::{
    Tool,
    cli,
    error::{Error, Result},
};

/// How wide the text is when nothing says otherwise.
pub const WIDTH: usize = 100;

/// The escape that ends every color of §10.3.
pub const RESET: &str = "\x1b[0m";

/// The color of the name of a header.
pub const FIELD: &str = "\x1b[1;36m";

/// The color of the subject.
pub const SUBJECT: &str = "\x1b[1m";

/// One color for each depth of a quote, cycled. (§10.3)
pub const QUOTES: [&str; 3] = ["\x1b[32m", "\x1b[33m", "\x1b[35m"];

/// How `view` writes. (§10.3)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Style {
    /// True when the output takes ANSI.
    pub color: bool,

    /// How wide the text is.
    pub width: usize,

    /// The theme that `syntect` highlights code with.
    pub theme: String,
}

/// The headers that both commands write. (§10.2)
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Head {
    /// The identity of §4.1, in full.
    pub id: String,

    /// The date, in the time zone of the reader.
    pub date: String,

    /// The `From` addresses.
    pub from: Vec<String>,

    /// The `To` addresses.
    pub to: Vec<String>,

    /// The `Cc` addresses.
    pub cc: Vec<String>,

    /// The subject, as the reader sees it.
    pub subject: String,

    /// Every folder that holds a copy. (§4.2)
    pub folders: Vec<String>,

    /// The tags of §9.
    pub tags: Vec<String>,

    /// The name of each attachment.
    pub attachments: Vec<String>,

    /// True when §5.4 kept the body as ciphertext.
    pub encrypted: bool,
}

/// One message, in the shape that `--json` writes. (§10.4)
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Whole {
    /// The headers.
    #[serde(flatten)]
    pub head: Head,

    /// The text of the body, with its quotes.
    pub text: String,
}

/// The message that a git-style prefix names. (§4.1)
///
/// # Errors
///
/// The function fails if nothing matches the prefix, or if more than
/// one message does.
pub fn resolve(store: &Store, prefix: &str) -> Result<MessageId> {
    match store.resolve(prefix)? {
        PrefixMatch::Unique(id) => Ok(id),
        PrefixMatch::Ambiguous(ids) => Err(Error::AmbiguousMessage {
            prefix: prefix.to_string(),
            ids: ids.iter().map(MessageId::short).collect(),
        }),
        PrefixMatch::NotFound => Err(Error::Core(
            mailbert_core::Error::UnknownMessage(prefix.to_string()),
        )),
    }
}

/// The headers and the body of one message.
///
/// The store keeps the raw bytes, so this reads the body again, and
/// gives back the whole of it. The text of the record has no quotes,
/// because §5.2 removed them for the index, and a reader wants to read
/// the conversation. No caller therefore chooses between the two.
///
/// # Errors
///
/// The function fails if the store lost the message, or if the bytes
/// are not a message.
pub fn read(
    store: &Store,
    id: &MessageId,
    offset: i32,
) -> Result<(Head, String)> {
    let missing =
        || Error::Core(mailbert_core::Error::UnknownMessage(id.short()));
    let message = store.get(id)?.ok_or_else(missing)?;
    let raw = store.raw(id)?.ok_or_else(missing)?;
    let parsed = mime::parse(&raw)?;

    let head = Head {
        id: id.full_hex(),
        date: stamp(message.date, offset),
        from: message.from.iter().map(one_address).collect(),
        to: message.to.iter().map(one_address).collect(),
        cc: message.cc.iter().map(one_address).collect(),
        subject: message.subject.clone(),
        folders: message.folders().into_iter().map(str::to_string).collect(),
        tags: store.tags_of(id)?.into_iter().collect(),
        attachments: parsed
            .attachments
            .iter()
            .filter_map(|one| one.name.clone())
            .collect(),
        encrypted: message.is_encrypted(),
    };

    // §5.4: the parse keeps the ciphertext out of the text, so the read
    // takes the armor from the bytes. A ciphertext is not a plaintext,
    // so `get` can show it, and a reader with no agent still sees it.
    let text = match head.encrypted {
        true => body_of(&raw),
        false => parsed.full,
    };

    Ok((head, text))
}

/// The body of a message, as the bytes of the server hold it. (§5.4)
///
/// The headers stop at the first empty line. The function gives an
/// empty text when there is no such line.
pub fn body_of(raw: &[u8]) -> String {
    let text = String::from_utf8_lossy(raw);

    if let Some((_, body)) = text.split_once("\r\n\r\n") {
        return body.replace("\r\n", "\n");
    }

    match text.split_once("\n\n") {
        Some((_, body)) => body.to_string(),
        None => String::new(),
    }
}

/// One address, as §10.3 writes it.
pub fn one_address(address: &Address) -> String {
    // §10.3 asks for the display name of §5.6. The contact book is not
    // in the store yet, so the name of the header stands in for it.
    match &address.name {
        Some(name) => format!("{name} <{}>", address.address),
        None => address.address.clone(),
    }
}

/// The date of a message, in the time zone of the reader.
pub fn stamp(at: i64, offset: i32) -> String {
    let Ok(moment) = jiff::Timestamp::from_second(at) else {
        return mailbert_core::date::day_text(at);
    };
    let Ok(offset) = jiff::tz::Offset::from_seconds(offset) else {
        return mailbert_core::date::day_text(at);
    };

    moment
        .to_zoned(jiff::tz::TimeZone::fixed(offset))
        .strftime("%Y-%m-%d %H:%M %z")
        .to_string()
}

/// Write the headers and the text with no color. (§10.2)
///
/// # Errors
///
/// The function fails if the output does not take the text.
pub fn write_plain(head: &Head, text: &str, out: &mut dyn Write) -> Result<()> {
    for (name, value) in fields(head) {
        writeln!(out, "{name}: {value}")?;
    }

    writeln!(out)?;
    writeln!(out, "{}", text.trim_end())?;

    Ok(())
}

/// The headers that carry something, in the order that both forms
/// write them.
fn fields(head: &Head) -> Vec<(&'static str, String)> {
    let mut held: Vec<(&'static str, String)> = vec![
        ("Date", head.date.clone()),
        ("From", head.from.join(", ")),
        ("To", head.to.join(", ")),
        ("Cc", head.cc.join(", ")),
        ("Subject", head.subject.clone()),
        ("Folders", head.folders.join(", ")),
        ("Tags", head.tags.join(", ")),
        ("Attachments", head.attachments.join(", ")),
        ("Id", head.id.clone()),
    ];

    held.retain(|(_, value)| !value.is_empty());

    held
}

/// Write one message as JSON. (§10.4)
///
/// # Errors
///
/// The function fails if the output does not take the text.
pub fn write_json(whole: &Whole, out: &mut dyn Write) -> Result<()> {
    writeln!(out, "{}", serde_json::to_string_pretty(whole)?)?;

    Ok(())
}

/// How deep a quote the line sits in. (§10.3)
///
/// `> > text` is depth 2, and a line that is not a quote is depth 0.
/// The marks can carry spaces between them, because clients differ.
pub fn quote_depth(line: &str) -> usize {
    let mut depth = 0;

    for one in line.chars() {
        match one {
            '>' => depth += 1,
            ' ' | '\t' => {}
            _ => break,
        }
    }

    depth
}

/// Write the headers and the body with color. (§10.3)
///
/// # Errors
///
/// The function fails if the output does not take the text.
pub fn render(
    head: &Head,
    text: &str,
    style: &Style,
    out: &mut dyn Write,
) -> Result<()> {
    for (name, value) in fields(head) {
        match style.color {
            true => writeln!(
                out,
                "{FIELD}{name}{RESET}: {}",
                match name {
                    "Subject" => format!("{SUBJECT}{value}{RESET}"),
                    _ => value,
                }
            )?,
            false => writeln!(out, "{name}: {value}")?,
        }
    }

    // §5.4: an encrypted body reaches here as its ciphertext unless
    // the caller asked the agent for the plaintext. Say which one it is.
    if head.encrypted {
        writeln!(out, "\n[encrypted — mailbert did not index this body]")?;
    }

    writeln!(out)?;

    let body = match style.color {
        true => highlight(text, &style.theme),
        false => text.to_string(),
    };

    for line in body.trim_end().lines() {
        let depth = quote_depth(line);

        match (style.color, depth) {
            (true, 1..) => {
                let color = QUOTES[(depth - 1) % QUOTES.len()];
                writeln!(out, "{color}{line}{RESET}")?;
            }
            _ => writeln!(out, "{line}")?,
        }
    }

    Ok(())
}

/// The body with each code block highlighted. (§10.3)
///
/// A fence names its language, and `syntect` highlights what it knows.
/// A fence that names nothing, or names a language that `syntect` does
/// not hold, keeps its text unchanged.
pub fn highlight(text: &str, theme: &str) -> String {
    let themes = ThemeSet::load_defaults();
    let Some(theme) = themes.themes.get(theme) else {
        return text.to_string();
    };

    let syntaxes = SyntaxSet::load_defaults_newlines();
    let mut out = String::with_capacity(text.len());
    let mut fence: Option<(String, String)> = None;

    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_end();

        match &mut fence {
            // Inside a fence: gather until the closing mark.
            Some((_, held)) if trimmed == "```" => {
                let (name, held) = fence.take().expect("the fence is open");
                out.push_str(&color(&held, &name, &syntaxes, theme));
                out.push_str(line);
            }
            Some((_, held)) => held.push_str(line),

            // Outside a fence: a mark opens one, and names its
            // language. A mark that names nothing opens a fence whose
            // language `color` does not know, and that keeps its text.
            None => {
                if let Some(name) = trimmed.strip_prefix("```") {
                    fence = Some((name.trim().to_string(), String::new()));
                }

                out.push_str(line);
            }
        }
    }

    // A fence that never closes is not a fence. Give back the text.
    match fence {
        Some(_) => text.to_string(),
        None => out,
    }
}

/// One block of code, with the colors that `syntect` gives it.
///
/// A language that `syntect` does not hold keeps its text unchanged.
fn color(
    code: &str,
    name: &str,
    syntaxes: &SyntaxSet,
    theme: &Theme,
) -> String {
    let Some(syntax) = syntaxes
        .find_syntax_by_token(name)
        .or_else(|| syntaxes.find_syntax_by_extension(name))
    else {
        return code.to_string();
    };

    let mut lines = HighlightLines::new(syntax, theme);
    let mut out = String::with_capacity(code.len());

    for line in LinesWithEndings::from(code) {
        let Ok(parts) = lines.highlight_line(line, syntaxes) else {
            return code.to_string();
        };

        out.push_str(&as_24_bit_terminal_escaped(&parts, false));
    }

    out.push_str(RESET);

    out
}

/// Do the work of `get`. (§10.2)
///
/// # Errors
///
/// The function fails if the message is not there, or if the output
/// does not take the text.
pub fn get(tool: &Tool, args: &cli::One) -> Result<()> {
    let store = tool.store()?;
    let id = resolve(&store, &args.id)?;
    let (head, text) = read(&store, &id, crate::clock().utc_offset())?;
    let mut out = std::io::stdout().lock();

    // §5.4: `get` never decrypts. A reader who wants the plaintext of
    // an encrypted body asks `view` for it.
    match args.json {
        true => write_json(&Whole { head, text }, &mut out),
        false => write_plain(&head, &text, &mut out),
    }
}

/// Do the work of `view`. (§10.3)
///
/// # Errors
///
/// The function fails if the message is not there, if gpg-agent
/// refuses, or if the output does not take the text.
pub fn view(tool: &Tool, args: &cli::Show) -> Result<()> {
    let config = tool.config()?;
    let store = tool.store()?;
    let id = resolve(&store, &args.id)?;
    let (mut head, body) = read(&store, &id, crate::clock().utc_offset())?;

    // §5.4: only `view` asks the agent, and it asks at this moment.
    // Nothing of the plaintext reaches the store or the index.
    let text = match head.encrypted {
        true => {
            let raw = store.raw(&id)?.unwrap_or_default();
            let plain = crate::pgp::decrypt(&config.pgp, &raw)?;
            head.encrypted = false;

            plain
        }
        false => body,
    };

    let style = Style {
        color: crate::wants_color(args.color),
        width: args.width.unwrap_or(config.view.width),
        theme: args.theme.clone().unwrap_or(config.view.theme),
    };

    render(&head, &text, &style, &mut std::io::stdout().lock())
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_quote_of_any_depth_reads_back_its_depth` | round-trip | The color of a quote is its depth. A depth that is wrong colors the words of one person as the words of another. |
    //! | `prop_a_plain_body_never_reads_as_a_quote` | invariant | Depth 0 is the text of the sender. A line that reads as a quote by mistake hides what the sender wrote. |
    //! | `prop_the_plain_form_never_writes_an_escape` | invariant | §10.2 is the machine-readable form. One escape in it breaks every program that reads the pipe. |
    //! | `prop_a_body_with_no_fence_keeps_every_word` | metamorphic | Highlighting must not lose text. A word that the fence reader eats is a word that the reader never sees. |

    use std::collections::{BTreeMap, BTreeSet};

    use hegel::{TestCase, generators as gs};
    use mailbert_core::message::{Location, Message};
    use tempfile::{TempDir, tempdir};

    use super::*;

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    /// The offset of the reader in every test: UTC.
    const UTC: i32 = 0;

    /// A theme that `syntect` always holds.
    const THEME: &str = "base16-ocean.dark";

    fn location(folder: &str) -> Location {
        Location {
            account: "work".to_string(),
            folder: folder.to_string(),
            uid: 1,
            uid_validity: 1,
            received: 1_755_820_800,
            flags: BTreeSet::new(),
        }
    }

    fn raw(headers: &str, body: &str) -> Vec<u8> {
        // A test that writes many messages gives its own `Message-ID`,
        // because §4.1 makes one identity out of one `Message-ID`.
        let id = match headers.contains("Message-ID:") {
            true => String::new(),
            false => "Message-ID: <a@x.test>\n".to_string(),
        };

        format!(
            "{headers}{id}\
             Date: Fri, 22 Aug 2025 09:30:00 +0000\n\
             \n\
             {body}"
        )
        .replace('\n', "\r\n")
        .into_bytes()
    }

    /// A store that holds one message, and the temporary directory
    /// that it lives in.
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

        fn put(&self, headers: &str, body: &str, folder: &str) -> MessageId {
            let bytes = raw(headers, body);
            let message = Message::new(
                mime::parse(&bytes).expect("a message"),
                location(folder),
                Vec::<String>::new(),
            );

            self.store.put(&message, &bytes).expect("a write").id
        }

        fn head(&self, id: &MessageId) -> Head {
            read(&self.store, id, UTC).expect("a read").0
        }
    }

    fn plain() -> Style {
        Style {
            color: false,
            width: WIDTH,
            theme: THEME.to_string(),
        }
    }

    fn colored() -> Style {
        Style {
            color: true,
            width: WIDTH,
            theme: THEME.to_string(),
        }
    }

    /// The text with every color removed.
    fn bare(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut held = text;

        while let Some(at) = held.find('\x1b') {
            out.push_str(&held[..at]);
            let rest = &held[at..];
            let end = rest.find('m').map_or(rest.len(), |one| one + 1);
            held = &rest[end..];
        }

        out.push_str(held);

        out
    }

    fn text_of(head: &Head, text: &str) -> String {
        let mut out = Vec::new();
        write_plain(head, text, &mut out).expect("a write");

        String::from_utf8(out).expect("the output is text")
    }

    fn shown(head: &Head, text: &str, style: &Style) -> String {
        let mut out = Vec::new();
        render(head, text, style, &mut out).expect("a write");

        String::from_utf8(out).expect("the output is text")
    }

    fn head_of(shelf: &Shelf) -> Head {
        let id = shelf.put(
            "From: Alice Smith <alice@x.test>\n\
             To: bob@x.test\n\
             Subject: Deposit\n",
            "the deposit is late\n",
            "INBOX",
        );

        shelf.head(&id)
    }

    // -----------------------------------------------------------------
    // The identity of §4.1.
    // -----------------------------------------------------------------

    #[test]
    fn a_prefix_names_the_message_that_it_starts() {
        let shelf = Shelf::new();
        let id = shelf.put("From: a@x.test\nSubject: One\n", "one\n", "INBOX");

        assert_eq!(resolve(&shelf.store, &id.short()).expect("a lookup"), id);
    }

    #[test]
    fn a_prefix_that_names_nothing_is_an_error() {
        let shelf = Shelf::new();

        let result = resolve(&shelf.store, "deadbeef");

        assert!(
            matches!(
                result,
                Err(Error::Core(mailbert_core::Error::UnknownMessage(_)))
            ),
            "{result:?}"
        );
    }

    #[test]
    fn a_prefix_that_two_messages_share_names_them_both() {
        let shelf = Shelf::new();
        let mut seen: BTreeMap<char, Vec<String>> = BTreeMap::new();

        // An identity starts with one of 16 characters. More than 16
        // messages therefore give a prefix that two of them share.
        for one in 0..24 {
            let head = format!(
                "From: a@x.test\nSubject: {one}\nMessage-ID: <{one}@x.test>\n"
            );
            let id = shelf.put(&head, "one\n", "INBOX");
            let first = id.short().chars().next().expect("a hex digit");

            seen.entry(first).or_default().push(id.short());
        }

        let (shared, held) = seen
            .iter()
            .find(|(_, ids)| ids.len() > 1)
            .expect("two identities share a first character");
        let result = resolve(&shelf.store, &shared.to_string());

        let Err(Error::AmbiguousMessage { ids, prefix }) = result else {
            panic!("`{shared}` names {} messages: {result:?}", held.len());
        };
        assert_eq!(prefix, shared.to_string());
        assert_eq!(ids.len(), held.len(), "{ids:?}");
    }

    // -----------------------------------------------------------------
    // The headers of §10.2.
    // -----------------------------------------------------------------

    #[test]
    fn the_headers_hold_every_field_that_the_reader_needs() {
        let shelf = Shelf::new();
        let id = shelf.put(
            "From: Alice Smith <alice@x.test>\n\
             To: bob@x.test\n\
             Cc: carol@x.test\n\
             Subject: Deposit\n",
            "the deposit is late\n",
            "Archive",
        );
        shelf.store.tag(&id, "todo").expect("a tag");

        let head = shelf.head(&id);

        assert_eq!(head.id, id.full_hex());
        assert_eq!(head.from, vec!["Alice Smith <alice@x.test>"]);
        assert_eq!(head.to, vec!["bob@x.test"]);
        assert_eq!(head.cc, vec!["carol@x.test"]);
        assert_eq!(head.subject, "Deposit");
        assert_eq!(head.folders, vec!["Archive"]);
        assert_eq!(head.tags, vec!["todo"]);
        assert!(!head.encrypted);
    }

    #[test]
    fn the_date_of_a_message_carries_its_hour() {
        let shelf = Shelf::new();
        let head = head_of(&shelf);

        assert!(head.date.starts_with("2025-08-22"), "{}", head.date);
        assert!(head.date.contains("09:30"), "{}", head.date);
    }

    #[test]
    fn the_date_moves_with_the_zone_of_the_reader() {
        let ahead = stamp(1_755_855_000, 3_600);
        let behind = stamp(1_755_855_000, -3_600);

        assert!(ahead.contains("10:30"), "{ahead}");
        assert!(behind.contains("08:30"), "{behind}");
    }

    #[test]
    fn the_body_of_the_read_keeps_the_quotes_that_the_index_lost() {
        let shelf = Shelf::new();
        let id = shelf.put(
            "From: a@x.test\nSubject: Deposit\n",
            "yes, Tuesday works\n> Can we meet Tuesday?\n",
            "INBOX",
        );

        let (_, body) = read(&shelf.store, &id, UTC).expect("a read");
        let record = shelf.store.get(&id).expect("a read").expect("a message");

        assert!(body.contains("Can we meet Tuesday?"), "{body}");
        assert!(
            !record.text.contains("Can we meet Tuesday?"),
            "the index kept the quote: {}",
            record.text
        );
    }

    /// §5.4 is the reason this module is careful. The index holds the
    /// ciphertext, so the read gives the ciphertext back. `get` calls
    /// only this function, so `get` cannot show the plaintext.
    #[test]
    fn the_read_of_an_encrypted_message_gives_the_ciphertext() {
        let shelf = Shelf::new();
        let id = shelf.put(
            "From: a@x.test\nSubject: Secret\n",
            "-----BEGIN PGP MESSAGE-----\n\nhQIMA1234\n\
             -----END PGP MESSAGE-----\n",
            "INBOX",
        );

        let (head, text) = read(&shelf.store, &id, UTC).expect("a read");

        assert!(head.encrypted, "the read lost the mark of §5.4");
        assert!(text.contains("BEGIN PGP MESSAGE"), "{text}");
    }

    /// A truncated download has no empty line, so it has no body. The
    /// read must give an empty text, and never a header.
    #[test]
    fn bytes_with_no_empty_line_hold_no_body() {
        assert_eq!(body_of(b"From: a@x.test\r\nSubject: One\r\n"), "");
    }

    #[test]
    fn the_body_of_the_bytes_ends_every_line_the_same_way() {
        let body = body_of(b"From: a@x.test\r\n\r\none\r\ntwo\r\n");

        assert_eq!(body, "one\ntwo\n");
    }

    /// A body that carries no ciphertext must report that, and never a
    /// plaintext. The test needs no agent, because the refusal comes
    /// before mailbert opens the socket.
    #[test]
    fn a_body_that_carries_no_ciphertext_is_an_error() {
        let config = mailbert_core::config::PgpConfig::default();
        let result =
            crate::pgp::decrypt(&config, b"these bytes are not a message\n");

        assert!(matches!(result, Err(Error::NoCiphertext)), "{result:?}");
    }

    #[test]
    fn a_message_that_the_store_lost_is_an_error() {
        let shelf = Shelf::new();
        let id = shelf.put("From: a@x.test\nSubject: One\n", "one\n", "INBOX");
        shelf.store.remove(&id).expect("a removal");

        let result = read(&shelf.store, &id, UTC);

        assert!(
            matches!(
                result,
                Err(Error::Core(mailbert_core::Error::UnknownMessage(_)))
            ),
            "{result:?}"
        );
    }

    #[test]
    fn an_address_with_a_name_keeps_both_parts() {
        let address = Address {
            name: Some("Alice Smith".to_string()),
            address: "alice@x.test".to_string(),
        };

        assert_eq!(one_address(&address), "Alice Smith <alice@x.test>");
    }

    #[test]
    fn an_address_with_no_name_is_the_address_alone() {
        let address = Address {
            name: None,
            address: "alice@x.test".to_string(),
        };

        assert_eq!(one_address(&address), "alice@x.test");
    }

    // -----------------------------------------------------------------
    // The plain form of §10.2.
    // -----------------------------------------------------------------

    #[test]
    fn the_plain_form_names_each_header_and_then_the_body() {
        let shelf = Shelf::new();
        let head = head_of(&shelf);

        let text = text_of(&head, "the deposit is late");
        let lines: Vec<&str> = text.lines().collect();

        assert!(lines[0].starts_with("Date: "), "{text}");
        assert!(text.contains("From: Alice Smith <alice@x.test>"), "{text}");
        assert!(text.contains("Subject: Deposit"), "{text}");
        assert!(text.contains("\n\nthe deposit is late"), "{text}");
    }

    #[test]
    fn the_plain_form_leaves_out_a_header_that_is_empty() {
        let shelf = Shelf::new();
        let head = head_of(&shelf);

        let text = text_of(&head, "the deposit is late");

        assert!(!text.contains("Cc:"), "{text}");
    }

    #[test]
    fn the_plain_form_writes_no_escape_at_all() {
        let shelf = Shelf::new();
        let head = head_of(&shelf);

        let text = text_of(&head, "the deposit is late");

        assert!(!text.contains('\x1b'), "{text:?}");
    }

    #[test]
    fn the_plain_form_names_the_attachments() {
        let mut head = head_of(&Shelf::new());
        head.attachments = vec!["report.pdf".to_string()];

        let text = text_of(&head, "see the file");

        assert!(text.contains("Attachments: report.pdf"), "{text}");
    }

    // -----------------------------------------------------------------
    // The JSON of §10.4.
    // -----------------------------------------------------------------

    #[test]
    fn the_json_holds_the_headers_and_the_body_together() {
        let shelf = Shelf::new();
        let head = head_of(&shelf);
        let whole = Whole {
            head,
            text: "the deposit is late".to_string(),
        };
        let mut out = Vec::new();
        write_json(&whole, &mut out).expect("a write");

        let read: serde_json::Value =
            serde_json::from_str(&String::from_utf8(out).expect("text"))
                .expect("good JSON");

        assert_eq!(read["subject"], "Deposit");
        assert_eq!(read["text"], "the deposit is late");
        assert_eq!(read["from"][0], "Alice Smith <alice@x.test>");
    }

    // -----------------------------------------------------------------
    // The quotes of §10.3.
    // -----------------------------------------------------------------

    #[test]
    fn a_line_that_is_not_a_quote_is_at_depth_zero() {
        assert_eq!(quote_depth("yes, Tuesday works"), 0);
    }

    #[test]
    fn one_mark_is_one_level_of_quote() {
        assert_eq!(quote_depth("> Can we meet Tuesday?"), 1);
    }

    #[test]
    fn two_marks_are_two_levels_of_quote() {
        assert_eq!(quote_depth("> > Alice wrote this"), 2);
    }

    #[test]
    fn the_marks_of_a_quote_need_no_space_between_them() {
        assert_eq!(quote_depth(">>> deep"), 3);
    }

    #[test]
    fn a_quote_can_start_after_a_space() {
        assert_eq!(quote_depth("  > indented"), 1);
    }

    // -----------------------------------------------------------------
    // The rendering of §10.3.
    // -----------------------------------------------------------------

    #[test]
    fn the_rendering_colors_the_name_of_each_header() {
        let shelf = Shelf::new();
        let head = head_of(&shelf);

        let text = shown(&head, "the deposit is late", &colored());

        assert!(text.contains(&format!("{FIELD}From{RESET}")), "{text:?}");
    }

    #[test]
    fn the_rendering_gives_each_depth_of_quote_its_own_color() {
        let shelf = Shelf::new();
        let head = head_of(&shelf);

        let text = shown(&head, "plain\n> one\n> > two", &colored());

        assert!(text.contains(QUOTES[0]), "{text:?}");
        assert!(text.contains(QUOTES[1]), "{text:?}");
    }

    #[test]
    fn the_rendering_with_no_color_writes_no_escape() {
        let shelf = Shelf::new();
        let head = head_of(&shelf);

        let text = shown(&head, "plain\n> one\n> > two", &plain());

        assert!(!text.contains('\x1b'), "{text:?}");
    }

    #[test]
    fn the_rendering_with_no_color_keeps_every_line_of_the_body() {
        let shelf = Shelf::new();
        let head = head_of(&shelf);

        let text = shown(&head, "plain\n> one\n> > two", &plain());

        assert!(text.contains("plain"), "{text}");
        assert!(text.contains("> one"), "{text}");
        assert!(text.contains("> > two"), "{text}");
    }

    #[test]
    fn the_rendering_marks_a_message_that_it_did_not_decrypt() {
        let shelf = Shelf::new();
        let mut head = head_of(&shelf);
        head.encrypted = true;

        let text = shown(&head, "-----BEGIN PGP MESSAGE-----", &plain());

        assert!(text.to_lowercase().contains("encrypted"), "{text}");
    }

    // -----------------------------------------------------------------
    // The code blocks of §10.3.
    // -----------------------------------------------------------------

    #[test]
    fn a_body_with_no_fence_comes_back_unchanged() {
        let body = "the deposit is late\nand the rent is due";

        assert_eq!(highlight(body, THEME), body);
    }

    #[test]
    fn a_fence_that_names_a_language_gets_color() {
        let body = "look:\n```rust\nfn main() {}\n```\ndone";

        let held = highlight(body, THEME);

        assert!(held.contains('\x1b'), "{held:?}");
        assert_eq!(bare(&held), body, "the color ate a character");
    }

    #[test]
    fn a_fence_keeps_the_text_around_it() {
        let body = "look:\n```rust\nfn main() {}\n```\ndone";

        let held = highlight(body, THEME);

        assert!(held.contains("look:"), "{held}");
        assert!(held.contains("done"), "{held}");
    }

    #[test]
    fn a_fence_that_names_nothing_stays_as_it_is() {
        let body = "look:\n```\nplain text\n```\ndone";

        assert_eq!(highlight(body, THEME), body);
    }

    #[test]
    fn a_fence_that_never_closes_stays_as_it_is() {
        let body = "look:\n```rust\nfn main() {}";

        assert_eq!(highlight(body, THEME), body);
    }

    #[test]
    fn a_theme_that_is_not_there_leaves_the_code_alone() {
        let body = "look:\n```rust\nfn main() {}\n```\ndone";

        assert_eq!(highlight(body, "no-such-theme"), body);
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    /// A line that carries no quote mark of its own.
    #[hegel::composite]
    fn a_plain_line(tc: TestCase) -> String {
        tc.draw(gs::text().alphabet("abc ,.").min_size(0).max_size(20))
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_quote_of_any_depth_reads_back_its_depth(tc: TestCase) {
        let depth: usize =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(6));
        let spaced: bool = tc.draw(gs::booleans());
        let text: String = tc.draw(a_plain_line());

        let mark = match spaced {
            true => "> ",
            false => ">",
        };
        let line = format!("{}{text}", mark.repeat(depth));

        assert_eq!(quote_depth(&line), depth, "`{line}`");
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_plain_body_never_reads_as_a_quote(tc: TestCase) {
        let text: String = tc.draw(a_plain_line());

        assert_eq!(quote_depth(&text), 0, "`{text}`");
    }

    #[hegel::test(test_cases = 100)]
    fn prop_the_plain_form_never_writes_an_escape(tc: TestCase) {
        let subject: String = tc.draw(a_plain_line());
        let body: String = tc
            .draw(gs::text().alphabet("abc \n\x1b[").min_size(0).max_size(60));

        let mut head = head_of(&Shelf::new());
        head.subject = subject;
        let written = text_of(&head, &body);

        // The body is the only place that an escape can come from, and
        // §10.2 gives the body as it is. The headers must add none.
        let (headers, _) = written.split_once("\n\n").expect("a blank line");
        assert!(!headers.contains('\x1b'), "{headers:?}");
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_body_with_no_fence_keeps_every_word(tc: TestCase) {
        let body: String =
            tc.draw(gs::text().alphabet("abc \n").min_size(0).max_size(80));

        assert_eq!(highlight(&body, THEME), body);
    }
}
