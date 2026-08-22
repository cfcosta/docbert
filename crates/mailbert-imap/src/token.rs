//! The token tree of an IMAP response. (RFC 3501 §4)
//!
//! A server answer is a line of items. An item is a word, a quoted
//! string, a literal, `NIL`, a list in parentheses, or a response code
//! in brackets. [`lex`] turns the bytes of one answer into that tree,
//! and [`encode`] turns the tree back into bytes.
//!
//! The lexer works on a complete answer. The reader of `wire` collects
//! that answer first, because a literal holds a length and can hold a
//! line break.

use crate::error::{Error, Result};

/// One item of an IMAP answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    /// A word, such as `OK` or `BODY[]`.
    Atom(String),
    /// A string in quotes. The quotes are not part of the value.
    Quoted(String),
    /// A block of bytes that a length announces.
    Literal(Vec<u8>),
    /// `NIL`, which is the empty value.
    Nil,
    /// A list in parentheses.
    List(Vec<Token>),
    /// A response code in brackets, such as `[UIDVALIDITY 1]`.
    Section(Vec<Token>),
}

/// Read one complete answer into its tokens.
///
/// The lexer is lenient. It reads a line break as a space, because a
/// server can pad its answer.
pub fn lex(raw: &[u8]) -> Result<Vec<Token>> {
    let mut at = 0;

    read_items(raw, &mut at, None)
}

/// Write the tokens of one line, and the line break after them.
pub fn encode(tokens: &[Token]) -> Vec<u8> {
    let mut out = Vec::new();

    for (at, token) in tokens.iter().enumerate() {
        if at > 0 {
            out.push(b' ');
        }
        token.write(&mut out);
    }
    out.extend_from_slice(b"\r\n");

    out
}

impl Token {
    /// Write one token, without a line break after it.
    pub fn write(&self, out: &mut Vec<u8>) {
        match self {
            Self::Atom(text) => out.extend_from_slice(text.as_bytes()),
            Self::Nil => out.extend_from_slice(b"NIL"),
            Self::Quoted(text) => {
                out.push(b'"');
                for byte in text.bytes() {
                    if byte == b'"' || byte == b'\\' {
                        out.push(b'\\');
                    }
                    out.push(byte);
                }
                out.push(b'"');
            }
            Self::Literal(bytes) => {
                out.extend_from_slice(
                    format!("{{{}}}\r\n", bytes.len()).as_bytes(),
                );
                out.extend_from_slice(bytes);
            }
            Self::List(items) => write_group(items, b'(', b')', out),
            Self::Section(items) => write_group(items, b'[', b']', out),
        }
    }

    /// The text of a word or of a quoted string.
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Atom(text) | Self::Quoted(text) => Some(text),
            _ => None,
        }
    }

    /// The value of a word that holds only digits.
    pub fn number(&self) -> Option<u64> {
        self.text()?.parse().ok()
    }
}

fn write_group(items: &[Token], open: u8, close: u8, out: &mut Vec<u8>) {
    out.push(open);
    for (at, item) in items.iter().enumerate() {
        if at > 0 {
            out.push(b' ');
        }
        item.write(out);
    }
    out.push(close);
}

/// Read items until `close`, or until the end when `close` is `None`.
fn read_items(
    raw: &[u8],
    at: &mut usize,
    close: Option<u8>,
) -> Result<Vec<Token>> {
    let mut items = Vec::new();

    loop {
        skip_space(raw, at);

        let Some(&byte) = raw.get(*at) else {
            return match close {
                None => Ok(items),
                Some(want) => Err(Error::Malformed(format!(
                    "`{}` closes nothing in this answer",
                    want as char
                ))),
            };
        };

        if Some(byte) == close {
            *at += 1;
            return Ok(items);
        }

        items.push(match byte {
            b')' | b']' => {
                return Err(Error::Malformed(format!(
                    "`{}` has no `{}` before it",
                    byte as char,
                    if byte == b')' { '(' } else { '[' }
                )));
            }
            b'(' => {
                *at += 1;
                Token::List(read_items(raw, at, Some(b')'))?)
            }
            b'[' => {
                *at += 1;
                Token::Section(read_items(raw, at, Some(b']'))?)
            }
            b'"' => read_quoted(raw, at)?,
            b'{' => read_literal(raw, at)?,
            _ => read_atom(raw, at)?,
        });
    }
}

fn skip_space(raw: &[u8], at: &mut usize) {
    while raw
        .get(*at)
        .is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
    {
        *at += 1;
    }
}

fn read_quoted(raw: &[u8], at: &mut usize) -> Result<Token> {
    *at += 1;
    let mut text = Vec::new();

    loop {
        let Some(&byte) = raw.get(*at) else {
            return Err(Error::Malformed(
                "a quoted string has no second quote".to_string(),
            ));
        };
        *at += 1;

        match byte {
            b'"' => break,
            b'\\' => {
                let Some(&next) = raw.get(*at) else {
                    return Err(Error::Malformed(
                        "a quoted string has no second quote".to_string(),
                    ));
                };
                *at += 1;
                text.push(next);
            }
            _ => text.push(byte),
        }
    }

    Ok(Token::Quoted(String::from_utf8(text)?))
}

fn read_literal(raw: &[u8], at: &mut usize) -> Result<Token> {
    *at += 1;
    let start = *at;

    while raw.get(*at).is_some_and(|byte| *byte != b'}') {
        *at += 1;
    }
    if raw.get(*at) != Some(&b'}') {
        return Err(Error::Malformed("a literal has no `}`".to_string()));
    }

    let digits = &raw[start..*at];
    *at += 1;

    // A `+` after the length says that the literal needs no go-ahead.
    // (RFC 7888)
    let digits = digits.strip_suffix(b"+").unwrap_or(digits);
    let count = str::from_utf8(digits)
        .ok()
        .and_then(|text| text.parse::<usize>().ok())
        .ok_or_else(|| {
            Error::Malformed(format!(
                "`{}` is not the length of a literal",
                String::from_utf8_lossy(digits)
            ))
        })?;

    // The line break after the length belongs to the marker.
    if raw[*at..].starts_with(b"\r\n") {
        *at += 2;
    } else if raw[*at..].starts_with(b"\n") {
        *at += 1;
    } else {
        return Err(Error::Malformed(
            "a literal has no line break after its length".to_string(),
        ));
    }

    let end = at.checked_add(count).ok_or_else(|| {
        Error::Malformed("a literal is longer than this machine".to_string())
    })?;
    let bytes = raw.get(*at..end).ok_or_else(|| {
        Error::Malformed(format!(
            "a literal wants {count} bytes, and the answer holds fewer"
        ))
    })?;
    *at = end;

    Ok(Token::Literal(bytes.to_vec()))
}

fn read_atom(raw: &[u8], at: &mut usize) -> Result<Token> {
    let start = *at;

    while raw.get(*at).is_some_and(|byte| is_atom_byte(*byte)) {
        *at += 1;
    }

    // `BODY[HEADER.FIELDS (DATE)]` is one name. The brackets and the
    // count after them belong to the word before them.
    if *at > start && raw.get(*at) == Some(&b'[') {
        let mut depth = 0;

        while let Some(&byte) = raw.get(*at) {
            *at += 1;
            match byte {
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth != 0 {
            return Err(Error::Malformed(
                "a fetch name has no `]`".to_string(),
            ));
        }

        while raw.get(*at).is_some_and(|byte| is_atom_byte(*byte)) {
            *at += 1;
        }
    }

    if *at == start {
        return Err(Error::Malformed(format!(
            "byte {} starts no token",
            raw[start]
        )));
    }

    let text = String::from_utf8(raw[start..*at].to_vec())?;
    if text.eq_ignore_ascii_case("NIL") {
        return Ok(Token::Nil);
    }

    Ok(Token::Atom(text))
}

fn is_atom_byte(byte: u8) -> bool {
    !matches!(
        byte,
        b' ' | b'\t' | b'\r' | b'\n' | b'(' | b')' | b'[' | b']' | b'{' | b'"'
    ) && byte >= 0x20
        && byte != 0x7f
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_token_tree_survives_the_round_trip` | round-trip | The client writes commands and reads answers with one tree. A tree that changes shape makes the client read the wrong field. |
    //! | `prop_a_literal_keeps_every_byte` | round-trip | §3.2 fetches complete messages as literals. One lost byte breaks the MIME parse. |
    //! | `prop_a_quoted_string_keeps_every_character` | round-trip | Mailbox names arrive in quotes. A lost escape selects the wrong folder. |
    //! | `prop_extra_space_between_items_changes_nothing` | metamorphic | Servers pad their answers. The tree must not depend on the padding. |
    //! | `prop_the_lexer_answers_every_input` | model-based | §3.4 wants a sync that resumes. A panic on a broken answer stops the whole run. |

    use hegel::{TestCase, generators as gs};

    use super::*;

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    fn atom(text: &str) -> Token {
        Token::Atom(text.to_string())
    }

    fn quoted(text: &str) -> Token {
        Token::Quoted(text.to_string())
    }

    fn literal(bytes: &[u8]) -> Token {
        Token::Literal(bytes.to_vec())
    }

    /// A word that reads back as one atom.
    fn draw_atom(tc: &TestCase) -> String {
        tc.draw(
            gs::text()
                .alphabet("abcXYZ019\\*.-_+/=$")
                .min_size(1)
                .max_size(6),
        )
    }

    /// The text of a quoted string, with the characters that escape.
    fn draw_quoted(tc: &TestCase) -> String {
        tc.draw(gs::text().alphabet("ab \"\\@.<>").min_size(0).max_size(6))
    }

    fn draw_token(tc: &TestCase, depth: usize) -> Token {
        let last = if depth >= 2 { 3 } else { 5 };
        let shape: usize =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(last));

        match shape {
            0 => Token::Atom(draw_atom(tc)),
            1 => Token::Quoted(draw_quoted(tc)),
            2 => Token::Literal(tc.draw(gs::binary().min_size(0).max_size(8))),
            3 => Token::Nil,
            4 => Token::List(draw_items(tc, depth + 1)),
            _ => Token::Section(draw_items(tc, depth + 1)),
        }
    }

    fn draw_items(tc: &TestCase, depth: usize) -> Vec<Token> {
        let count: usize =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(3));

        (0..count).map(|_| draw_token(tc, depth)).collect()
    }

    /// A whole answer.
    #[hegel::composite]
    fn a_token_line(tc: TestCase) -> Vec<Token> {
        draw_items(&tc, 0)
    }

    // -----------------------------------------------------------------
    // Unit tests: the lexer.
    // -----------------------------------------------------------------

    #[test]
    fn a_tagged_answer_reads_as_words() {
        assert_eq!(
            lex(b"a001 OK LOGIN completed\r\n").unwrap(),
            vec![atom("a001"), atom("OK"), atom("LOGIN"), atom("completed")]
        );
    }

    #[test]
    fn an_empty_line_reads_as_no_token() {
        assert_eq!(lex(b"\r\n").unwrap(), Vec::new());
    }

    #[test]
    fn a_line_without_a_carriage_return_still_reads() {
        assert_eq!(lex(b"* OK\n").unwrap(), vec![atom("*"), atom("OK")]);
        assert_eq!(lex(b"* OK").unwrap(), vec![atom("*"), atom("OK")]);
    }

    #[test]
    fn a_quoted_string_keeps_its_spaces() {
        assert_eq!(
            lex(b"* LIST (\\HasNoChildren) \".\" \"INBOX/Sent Mail\"\r\n")
                .unwrap(),
            vec![
                atom("*"),
                atom("LIST"),
                Token::List(vec![atom("\\HasNoChildren")]),
                quoted("."),
                quoted("INBOX/Sent Mail"),
            ]
        );
    }

    #[test]
    fn a_quoted_string_reads_its_escapes() {
        assert_eq!(lex(br#""a\"b\\c""#).unwrap(), vec![quoted("a\"b\\c")]);
    }

    #[test]
    fn an_empty_quoted_string_reads_as_empty_text() {
        assert_eq!(lex(b"\"\"\r\n").unwrap(), vec![quoted("")]);
    }

    #[test]
    fn a_literal_reads_its_bytes() {
        assert_eq!(
            lex(b"* 1 FETCH (UID 5 BODY[] {5}\r\nhello)\r\n").unwrap(),
            vec![
                atom("*"),
                atom("1"),
                atom("FETCH"),
                Token::List(vec![
                    atom("UID"),
                    atom("5"),
                    atom("BODY[]"),
                    literal(b"hello"),
                ]),
            ]
        );
    }

    #[test]
    fn a_literal_holds_a_line_break() {
        assert_eq!(
            lex(b"{6}\r\na\r\nb\r\n\r\n").unwrap(),
            vec![literal(b"a\r\nb\r\n")]
        );
    }

    #[test]
    fn a_literal_that_needs_no_continuation_reads_as_a_literal() {
        assert_eq!(lex(b"{5+}\r\nhello\r\n").unwrap(), vec![literal(b"hello")]);
    }

    #[test]
    fn an_empty_literal_reads_as_no_byte() {
        assert_eq!(
            lex(b"A {0}\r\n\r\n").unwrap(),
            vec![atom("A"), literal(b"")]
        );
    }

    #[test]
    fn a_response_code_reads_as_a_section() {
        assert_eq!(
            lex(b"* OK [UIDVALIDITY 3857529045] UIDs valid\r\n").unwrap(),
            vec![
                atom("*"),
                atom("OK"),
                Token::Section(vec![atom("UIDVALIDITY"), atom("3857529045")]),
                atom("UIDs"),
                atom("valid"),
            ]
        );
    }

    #[test]
    fn a_section_holds_a_list() {
        assert_eq!(
            lex(b"* OK [PERMANENTFLAGS (\\Seen \\*)] limited\r\n").unwrap(),
            vec![
                atom("*"),
                atom("OK"),
                Token::Section(vec![
                    atom("PERMANENTFLAGS"),
                    Token::List(vec![atom("\\Seen"), atom("\\*")]),
                ]),
                atom("limited"),
            ]
        );
    }

    #[test]
    fn a_bracket_joins_the_word_before_it() {
        assert_eq!(
            lex(b"BODY[HEADER.FIELDS (DATE FROM)]\r\n").unwrap(),
            vec![atom("BODY[HEADER.FIELDS (DATE FROM)]")]
        );
    }

    #[test]
    fn a_partial_fetch_keeps_its_octet_count() {
        assert_eq!(
            lex(b"BODY[]<0.512>\r\n").unwrap(),
            vec![atom("BODY[]<0.512>")]
        );
    }

    #[test]
    fn nil_reads_as_nil_in_any_case() {
        assert_eq!(lex(b"NIL nil\r\n").unwrap(), vec![Token::Nil, Token::Nil]);
    }

    #[test]
    fn an_empty_list_reads_as_an_empty_list() {
        assert_eq!(lex(b"()\r\n").unwrap(), vec![Token::List(Vec::new())]);
    }

    #[test]
    fn a_list_holds_a_list() {
        assert_eq!(
            lex(b"(a (b c) d)\r\n").unwrap(),
            vec![Token::List(vec![
                atom("a"),
                Token::List(vec![atom("b"), atom("c")]),
                atom("d"),
            ])]
        );
    }

    // -----------------------------------------------------------------
    // Unit tests: answers that the lexer refuses.
    // -----------------------------------------------------------------

    #[test]
    fn a_list_that_does_not_close_is_an_error() {
        assert!(lex(b"(a b\r\n").is_err());
    }

    #[test]
    fn a_section_that_does_not_close_is_an_error() {
        assert!(lex(b"[a b\r\n").is_err());
    }

    #[test]
    fn a_quote_that_does_not_close_is_an_error() {
        assert!(lex(b"\"abc\r\n").is_err());
    }

    #[test]
    fn a_close_parenthesis_without_an_open_one_is_an_error() {
        assert!(lex(b"a)\r\n").is_err());
    }

    #[test]
    fn a_literal_that_is_too_short_is_an_error() {
        assert!(lex(b"{10}\r\nabc\r\n").is_err());
    }

    #[test]
    fn a_literal_length_that_is_not_a_number_is_an_error() {
        assert!(lex(b"{abc}\r\nxx\r\n").is_err());
    }

    #[test]
    fn a_literal_without_a_line_break_is_an_error() {
        assert!(lex(b"{3}abc\r\n").is_err());
    }

    // -----------------------------------------------------------------
    // Unit tests: the encoder.
    // -----------------------------------------------------------------

    #[test]
    fn the_encoder_writes_a_tagged_command() {
        assert_eq!(
            encode(&[atom("a001"), atom("LOGIN"), quoted("me"), quoted("pw")]),
            b"a001 LOGIN \"me\" \"pw\"\r\n".to_vec()
        );
    }

    #[test]
    fn the_encoder_escapes_a_quote_and_a_backslash() {
        assert_eq!(
            encode(&[quoted("a\"b\\c")]),
            br#""a\"b\\c""#.to_vec().into_iter().chain(*b"\r\n").collect::<Vec<u8>>()
        );
    }

    #[test]
    fn the_encoder_writes_the_length_of_a_literal() {
        assert_eq!(
            encode(&[atom("A"), literal(b"hi")]),
            b"A {2}\r\nhi\r\n".to_vec()
        );
    }

    #[test]
    fn the_encoder_writes_an_empty_list() {
        assert_eq!(encode(&[Token::List(Vec::new())]), b"()\r\n".to_vec());
    }

    #[test]
    fn the_encoder_writes_nil() {
        assert_eq!(encode(&[Token::Nil]), b"NIL\r\n".to_vec());
    }

    #[test]
    fn the_encoder_writes_a_section() {
        assert_eq!(
            encode(&[Token::Section(vec![atom("UIDVALIDITY"), atom("1")])]),
            b"[UIDVALIDITY 1]\r\n".to_vec()
        );
    }

    #[test]
    fn the_encoder_writes_no_token_as_one_empty_line() {
        assert_eq!(encode(&[]), b"\r\n".to_vec());
    }

    // -----------------------------------------------------------------
    // Unit tests: reading a value out of a token.
    // -----------------------------------------------------------------

    #[test]
    fn a_word_and_a_quoted_string_both_give_text() {
        assert_eq!(atom("INBOX").text(), Some("INBOX"));
        assert_eq!(quoted("Sent Mail").text(), Some("Sent Mail"));
        assert_eq!(Token::Nil.text(), None);
        assert_eq!(Token::List(Vec::new()).text(), None);
    }

    #[test]
    fn a_word_of_digits_gives_a_number() {
        assert_eq!(atom("4294967295").number(), Some(4_294_967_295));
        assert_eq!(atom("0").number(), Some(0));
        assert_eq!(atom("12x").number(), None);
        assert_eq!(Token::Nil.number(), None);
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 200)]
    fn prop_a_token_tree_survives_the_round_trip(tc: TestCase) {
        let tokens: Vec<Token> = tc.draw(a_token_line());
        let raw = encode(&tokens);

        assert_eq!(lex(&raw).unwrap(), tokens);
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_literal_keeps_every_byte(tc: TestCase) {
        let bytes: Vec<u8> = tc.draw(gs::binary().min_size(0).max_size(64));
        let raw = encode(&[atom("A"), Token::Literal(bytes.clone())]);

        assert_eq!(lex(&raw).unwrap(), vec![atom("A"), Token::Literal(bytes)]);
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_quoted_string_keeps_every_character(tc: TestCase) {
        let text = draw_quoted(&tc);
        let raw = encode(&[Token::Quoted(text.clone())]);

        assert_eq!(lex(&raw).unwrap(), vec![Token::Quoted(text)]);
    }

    #[hegel::test(test_cases = 100)]
    fn prop_extra_space_between_items_changes_nothing(tc: TestCase) {
        let tokens: Vec<Token> = tc.draw(a_token_line());
        let gap: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(3));

        let mut raw = Vec::new();
        for (at, token) in tokens.iter().enumerate() {
            if at > 0 {
                raw.extend(std::iter::repeat_n(b' ', gap));
            }
            token.write(&mut raw);
        }
        raw.extend_from_slice(b"\r\n");

        assert_eq!(lex(&raw).unwrap(), tokens);
    }

    #[hegel::test(test_cases = 300)]
    fn prop_the_lexer_answers_every_input(tc: TestCase) {
        let raw: Vec<u8> = tc.draw(gs::binary().min_size(0).max_size(48));

        // The answer is an error or a tree. It is never a panic.
        let _ = lex(&raw);
    }
}
