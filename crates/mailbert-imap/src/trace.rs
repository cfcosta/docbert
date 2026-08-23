//! What the log says about the IMAP conversation. (§10.5)
//!
//! A reader who gives `--verbose` sees each command that goes to the
//! server. A password never goes with it. The log is a plain file that
//! many eyes read, and §3 keeps the credential out of every file that
//! is not the one the reader chose.
//!
//! A command line is also cut, because one `UID FETCH` can name ten
//! thousand numbers.

use crate::token::{Token, encode};

/// What the log shows in place of a credential.
pub const HIDDEN: &str = "***";

/// How many characters of one command reach the log.
pub const MOST: usize = 160;

/// The commands whose words are credentials. (RFC 3501 §6.2)
pub const SECRET: [&str; 2] = ["LOGIN", "AUTHENTICATE"];

/// The text of one command, for the log.
///
/// The name of the command always shows. The words after a `LOGIN` or
/// an `AUTHENTICATE` never do.
pub fn said(words: &[Token]) -> String {
    let Some(first) = words.first() else {
        return String::new();
    };
    let name = first.text().unwrap_or_default().to_ascii_uppercase();

    // The name always shows, so a reader sees that the login went out.
    // Nothing after it does.
    if SECRET.contains(&name.as_str()) {
        return format!("{name} {HIDDEN}");
    }

    let shown: Vec<Token> = words.iter().map(counted).collect();

    cut(&text(&shown))
}

/// The same token, with the bytes of each literal counted.
fn counted(token: &Token) -> Token {
    match token {
        Token::Literal(raw) => Token::Atom(format!("{{{}}}", raw.len())),
        Token::List(items) => Token::List(items.iter().map(counted).collect()),
        Token::Section(items) => {
            Token::Section(items.iter().map(counted).collect())
        }
        other => other.clone(),
    }
}

/// The line that these tokens make, without the line break.
fn text(tokens: &[Token]) -> String {
    String::from_utf8_lossy(&encode(tokens))
        .trim_end()
        .to_string()
}

/// The first `MOST` characters, and the true length after them.
fn cut(line: &str) -> String {
    if line.len() <= MOST {
        return line.to_string();
    }

    let mut end = MOST;
    while !line.is_char_boundary(end) {
        end -= 1;
    }

    format!("{}... ({} bytes)", &line[..end], line.len())
}

/// The text of one answer, for the log.
///
/// A `FETCH` answer holds the mail itself, so this counts the lines
/// and gives no line.
pub fn heard(state: &str, lines: usize) -> String {
    match lines {
        1 => format!("{state}, 1 line"),
        _ => format!("{state}, {lines} lines"),
    }
}

/// A log that the tests read. (§10.5)
#[cfg(test)]
pub(crate) mod pen {
    use std::{
        io,
        sync::{Arc, Mutex, Once},
    };

    use tracing::{
        Event,
        Metadata,
        Subscriber,
        level_filters::LevelFilter,
        span,
        subscriber::Interest,
    };
    use tracing_subscriber::{EnvFilter, fmt::MakeWriter};

    /// A writer that keeps every line in memory.
    #[derive(Clone, Default)]
    pub struct Pen(Arc<Mutex<Vec<u8>>>);

    impl Pen {
        /// Everything that the log holds now.
        pub fn text(&self) -> String {
            let held = self.0.lock().expect("no writer panicked");

            String::from_utf8_lossy(&held).to_string()
        }
    }

    impl io::Write for Pen {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().expect("no writer panicked").extend(bytes);

            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for Pen {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// A subscriber that answers "maybe" for every callsite.
    ///
    /// `tracing` keeps one answer for each callsite, for the whole
    /// process. A test that touches a callsite first, while no
    /// subscriber is there, makes that answer "never". Every later
    /// event of that callsite goes away, and the pen of another test
    /// stays empty. This subscriber is always there, and it always
    /// answers "maybe", so the pen of each test gets its events.
    struct Always;

    impl Subscriber for Always {
        fn register_callsite(&self, _: &Metadata<'_>) -> Interest {
            Interest::sometimes()
        }

        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }

        fn max_level_hint(&self) -> Option<LevelFilter> {
            Some(LevelFilter::TRACE)
        }

        fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
            span::Id::from_u64(1)
        }

        fn record(&self, _: &span::Id, _: &span::Record<'_>) {}

        fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}

        fn event(&self, _: &Event<'_>) {}

        fn enter(&self, _: &span::Id) {}

        fn exit(&self, _: &span::Id) {}
    }

    /// Open the answer of every callsite, one time for each run.
    ///
    /// A test that reads the log calls this before it makes a pen.
    pub fn open() {
        static ONCE: Once = Once::new();

        ONCE.call_once(|| {
            let _ = tracing::subscriber::set_global_default(Always);
        });
    }

    /// A subscriber that keeps every event, at every level.
    pub fn capture(pen: Pen) -> impl Subscriber + Send + Sync + 'static {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::new("trace"))
            .with_writer(pen)
            .with_ansi(false)
            .without_time()
            .finish()
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | A password never reaches the log | the text of the password | §3 keeps a credential out of every file that the reader did not choose. |
    //! | A command line never grows without a limit | `MOST` | One `UID FETCH` can name ten thousand numbers. |

    use hegel::{TestCase, generators as gs};

    use super::*;

    fn atom(text: &str) -> Token {
        Token::Atom(text.to_string())
    }

    fn quoted(text: &str) -> Token {
        Token::Quoted(text.to_string())
    }

    #[test]
    fn a_command_reads_as_the_server_gets_it() {
        let said = said(&[atom("LIST"), quoted(""), quoted("*")]);

        assert_eq!(said, "LIST \"\" \"*\"");
    }

    #[test]
    fn a_command_with_no_word_reads_as_nothing() {
        assert_eq!(said(&[]), "");
    }

    #[test]
    fn a_login_never_shows_the_password() {
        let said = said(&[atom("LOGIN"), quoted("me@here"), quoted("hunter2")]);

        assert!(!said.contains("hunter2"), "{said}");
    }

    /// The name of the account is in the span of §10.5 already, and a
    /// login name is an address that the reader may not want in a file.
    #[test]
    fn a_login_never_shows_the_name_either() {
        let said = said(&[atom("LOGIN"), quoted("me@here"), quoted("hunter2")]);

        assert!(!said.contains("me@here"), "{said}");
    }

    #[test]
    fn a_login_keeps_its_own_name() {
        let said = said(&[atom("LOGIN"), quoted("me@here"), quoted("hunter2")]);

        assert_eq!(said, format!("LOGIN {HIDDEN}"));
    }

    /// A server that offers `AUTHENTICATE` takes the credential in the
    /// same line, so the same rule holds.
    #[test]
    fn an_authenticate_hides_what_comes_after_it() {
        let said = said(&[atom("AUTHENTICATE"), atom("PLAIN"), atom("aGk=")]);

        assert_eq!(said, format!("AUTHENTICATE {HIDDEN}"));
    }

    /// A client may write the name of a command in small letters, and
    /// the rule must not turn on the letters.
    #[test]
    fn a_login_in_small_letters_hides_its_password_too() {
        let said = said(&[atom("login"), quoted("me"), quoted("hunter2")]);

        assert!(!said.contains("hunter2"), "{said}");
    }

    #[test]
    fn a_command_that_names_no_credential_keeps_every_word() {
        let said = said(&[atom("EXAMINE"), quoted("INBOX")]);

        assert_eq!(said, "EXAMINE \"INBOX\"");
    }

    #[test]
    fn a_long_command_is_cut() {
        let set = (1..4000).map(|n| n.to_string()).collect::<Vec<_>>();
        let said = said(&[
            atom("UID"),
            atom("FETCH"),
            atom(&set.join(",")),
            atom("(UID)"),
        ]);

        assert!(said.len() < MOST * 2, "the log holds {} bytes", said.len());
        assert!(said.starts_with("UID FETCH 1,2,3,"), "{said}");
    }

    /// The cut says the true length, so a reader knows what it lost.
    #[test]
    fn a_command_that_is_cut_says_how_long_it_was() {
        let set = (1..4000).map(|n| n.to_string()).collect::<Vec<_>>();
        let whole = set.join(",");
        let said =
            said(&[atom("UID"), atom("FETCH"), atom(&whole), atom("(UID)")]);
        let full = format!("UID FETCH {whole} (UID)");

        assert!(said.contains(&format!("{} bytes", full.len())), "{said}");
    }

    /// A literal in a command holds bytes of mail. The log counts them.
    #[test]
    fn a_literal_shows_its_length_and_not_its_bytes() {
        let said =
            said(&[atom("APPEND"), Token::Literal(b"secret mail".to_vec())]);

        assert_eq!(said, "APPEND {11}");
        assert!(!said.contains("secret"), "{said}");
    }

    #[test]
    fn a_literal_inside_a_list_shows_its_length_too() {
        let said = said(&[
            atom("X"),
            Token::List(vec![Token::Literal(b"secret".to_vec())]),
        ]);

        assert_eq!(said, "X ({6})");
    }

    /// Each state has the word that the server itself writes, so the
    /// log and the answer of the server read the same.
    #[test]
    fn every_state_carries_the_word_of_the_server() {
        use crate::connection::State;

        assert_eq!(State::Ok.name(), "OK");
        assert_eq!(State::No.name(), "NO");
        assert_eq!(State::Bad.name(), "BAD");
    }

    #[test]
    fn an_answer_counts_the_lines_and_gives_none() {
        assert_eq!(heard("OK", 3), "OK, 3 lines");
    }

    #[test]
    fn an_answer_of_one_line_counts_that_line() {
        assert_eq!(heard("NO", 1), "NO, 1 line");
    }

    #[hegel::test(test_cases = 300)]
    fn prop_a_login_never_shows_its_password(tc: TestCase) {
        let password = tc.draw(
            gs::text()
                .alphabet("abcdefghijklmnopqrstuvwxyz0123456789!@#$%")
                .min_size(6)
                .max_size(40),
        );
        let said = said(&[atom("LOGIN"), quoted("me"), quoted(&password)]);

        assert!(!said.contains(&password), "`{said}` holds the password");
    }

    #[hegel::test(test_cases = 300)]
    fn prop_a_command_never_grows_without_a_limit(tc: TestCase) {
        let count = tc.draw(gs::integers::<u32>().min_value(0).max_value(9000));
        let set = (0..count).map(|n| n.to_string()).collect::<Vec<_>>();
        let said = said(&[atom("UID"), atom("FETCH"), atom(&set.join(","))]);

        assert!(said.len() < MOST * 2, "the log holds {} bytes", said.len());
    }
}
