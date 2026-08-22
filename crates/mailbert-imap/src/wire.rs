//! Read one answer off a stream, and write one command to it.
//!
//! A literal makes an IMAP answer longer than one line. The reader
//! therefore reads a line, looks for a length at the end of it, and then
//! reads exactly that many bytes. It repeats until a line ends without a
//! length. [`token::lex`](crate::token::lex) reads the bytes that come
//! out.

use tokio::io::{
    AsyncBufRead,
    AsyncBufReadExt,
    AsyncReadExt,
    AsyncWrite,
    AsyncWriteExt,
};

use crate::{
    error::{Error, Result},
    token::{Token, encode, lex},
};

/// The tag of each command. A tag never repeats on one connection.
#[derive(Debug, Clone, Default)]
pub struct Tags {
    count: u64,
}

impl Tags {
    pub fn new() -> Self {
        Self::default()
    }

    /// The tag of the next command.
    pub fn next_tag(&mut self) -> String {
        self.count += 1;

        format!("a{:04}", self.count)
    }
}

/// The length of the literal that a line announces at its end.
///
/// A line that ends with `{5}` says that 5 bytes come after it. A `+`
/// after the length says that the client needs no go-ahead. (RFC 7888)
pub fn literal_length(line: &[u8]) -> Option<usize> {
    let line = line.strip_suffix(b"\n").unwrap_or(line);
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    let line = line.strip_suffix(b"}")?;

    let open = line.iter().rposition(|byte| *byte == b'{')?;
    let digits = &line[open + 1..];
    let digits = digits.strip_suffix(b"+").unwrap_or(digits);

    str::from_utf8(digits).ok()?.parse().ok()
}

/// Read the bytes of one complete answer, with the literals in it.
///
/// The answer holds its line break at the end. The reader gives
/// [`Error::Closed`] when the server sends nothing more.
pub async fn read_line<R>(reader: &mut R) -> Result<Vec<u8>>
where
    R: AsyncBufRead + Unpin,
{
    let mut raw = Vec::new();

    loop {
        let start = raw.len();
        if reader.read_until(b'\n', &mut raw).await? == 0 {
            return Err(Error::Closed);
        }

        let Some(count) = literal_length(&raw[start..]) else {
            return Ok(raw);
        };

        let at = raw.len();
        raw.resize(at + count, 0);
        reader.read_exact(&mut raw[at..]).await?;
    }
}

/// Read one answer, and read the tokens of it.
pub async fn read_answer<R>(reader: &mut R) -> Result<Vec<Token>>
where
    R: AsyncBufRead + Unpin,
{
    lex(&read_line(reader).await?)
}

/// Write one command, and flush it.
///
/// A literal in a command goes out with no wait for a go-ahead. Only a
/// server that announces `LITERAL+` accepts that.
pub async fn write_line<W>(writer: &mut W, tokens: &[Token]) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    writer.write_all(&encode(tokens)).await?;
    writer.flush().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_written_line_reads_back` | round-trip | The reader frames each answer. A frame that is one byte wrong loses the rest of the connection. |
    //! | `prop_the_reader_splits_a_stream_of_answers` | model-based | §3.1 pipelines commands, so many answers arrive together. A reader that takes too much drops an answer. |
    //! | `prop_a_tag_never_repeats` | algebraic | A pipelined client pairs each answer with a tag. Two equal tags pair the wrong ones. |

    use std::future::Future;

    use hegel::{TestCase, generators as gs};
    use tokio::io::BufReader;

    use super::*;

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    fn run<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    fn reader(bytes: &[u8]) -> BufReader<&[u8]> {
        BufReader::new(bytes)
    }

    fn atom(text: &str) -> Token {
        Token::Atom(text.to_string())
    }

    /// A short answer, and the bytes of it.
    #[hegel::composite]
    fn an_answer(tc: TestCase) -> Vec<Token> {
        let count: usize =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(4));

        (0..count)
            .map(|_| {
                let shape: usize =
                    tc.draw(gs::integers::<usize>().min_value(0).max_value(2));
                let word: String = tc.draw(
                    gs::text().alphabet("abXY01*.").min_size(1).max_size(5),
                );

                match shape {
                    0 => Token::Atom(word),
                    1 => Token::Quoted(word),
                    _ => Token::Literal(
                        tc.draw(gs::binary().min_size(0).max_size(12)),
                    ),
                }
            })
            .collect()
    }

    // -----------------------------------------------------------------
    // Unit tests: the length at the end of a line.
    // -----------------------------------------------------------------

    #[test]
    fn a_line_that_ends_with_a_length_announces_a_literal() {
        assert_eq!(literal_length(b"* 1 FETCH (BODY[] {5}\r\n"), Some(5));
    }

    #[test]
    fn a_line_without_a_length_announces_no_literal() {
        assert_eq!(literal_length(b"a001 OK done\r\n"), None);
    }

    #[test]
    fn a_length_that_needs_no_continuation_still_counts() {
        assert_eq!(literal_length(b"A {12+}\r\n"), Some(12));
    }

    #[test]
    fn a_bracket_at_the_end_announces_no_literal() {
        assert_eq!(literal_length(b"* OK [UIDVALIDITY 1]\r\n"), None);
    }

    #[test]
    fn a_length_that_is_not_a_number_announces_no_literal() {
        assert_eq!(literal_length(b"A {x}\r\n"), None);
    }

    #[test]
    fn a_brace_without_an_open_one_announces_no_literal() {
        assert_eq!(literal_length(b"A }\r\n"), None);
    }

    #[test]
    fn an_empty_line_announces_no_literal() {
        assert_eq!(literal_length(b"\r\n"), None);
        assert_eq!(literal_length(b""), None);
    }

    // -----------------------------------------------------------------
    // Unit tests: tags.
    // -----------------------------------------------------------------

    #[test]
    fn a_tag_starts_at_one_and_grows() {
        let mut tags = Tags::new();

        assert_eq!(tags.next_tag(), "a0001");
        assert_eq!(tags.next_tag(), "a0002");
        assert_eq!(tags.next_tag(), "a0003");
    }

    // -----------------------------------------------------------------
    // Unit tests: the reader.
    // -----------------------------------------------------------------

    #[test]
    fn the_reader_reads_one_line() {
        let bytes = run(read_line(&mut reader(b"* OK ready\r\n"))).unwrap();

        assert_eq!(bytes, b"* OK ready\r\n".to_vec());
    }

    #[test]
    fn the_reader_reads_a_literal_that_holds_a_line_break() {
        let raw = b"* 1 FETCH (BODY[] {8}\r\na\r\nb\r\nc)\r\n";
        let bytes = run(read_line(&mut reader(raw))).unwrap();

        assert_eq!(bytes, raw.to_vec());
    }

    #[test]
    fn the_reader_reads_two_literals_in_one_answer() {
        let raw = b"* 1 FETCH (A {2}\r\nhi B {3}\r\nbye)\r\n";
        let bytes = run(read_line(&mut reader(raw))).unwrap();

        assert_eq!(bytes, raw.to_vec());
    }

    #[test]
    fn the_reader_stops_at_the_end_of_one_answer() {
        run(async {
            let mut source = reader(b"* 1 EXISTS\r\na001 OK done\r\n");

            assert_eq!(
                read_line(&mut source).await.unwrap(),
                b"* 1 EXISTS\r\n".to_vec()
            );
            assert_eq!(
                read_line(&mut source).await.unwrap(),
                b"a001 OK done\r\n".to_vec()
            );
        });
    }

    #[test]
    fn the_reader_reports_a_closed_connection() {
        let answer = run(read_line(&mut reader(b"")));

        assert!(matches!(answer, Err(Error::Closed)));
    }

    #[test]
    fn the_reader_gives_tokens() {
        let tokens =
            run(read_answer(&mut reader(b"a001 OK done\r\n"))).unwrap();

        assert_eq!(tokens, vec![atom("a001"), atom("OK"), atom("done")]);
    }

    // -----------------------------------------------------------------
    // Unit tests: the writer.
    // -----------------------------------------------------------------

    #[test]
    fn the_writer_writes_a_tagged_command() {
        let mut out: Vec<u8> = Vec::new();
        run(write_line(
            &mut out,
            &[atom("a001"), atom("LOGIN"), Token::Quoted("me".to_string())],
        ))
        .unwrap();

        assert_eq!(out, b"a001 LOGIN \"me\"\r\n".to_vec());
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 100)]
    fn prop_a_written_line_reads_back(tc: TestCase) {
        let tokens: Vec<Token> = tc.draw(an_answer());
        let raw = encode(&tokens);

        let read = run(read_line(&mut reader(&raw))).unwrap();

        assert_eq!(read, raw);
        assert_eq!(lex(&read).unwrap(), tokens);
    }

    #[hegel::test(test_cases = 100)]
    fn prop_the_reader_splits_a_stream_of_answers(tc: TestCase) {
        let answers: Vec<Vec<Token>> =
            tc.draw(gs::vecs(an_answer()).min_size(1).max_size(4));

        let mut raw = Vec::new();
        for answer in &answers {
            raw.extend_from_slice(&encode(answer));
        }

        run(async {
            let mut source = reader(&raw);

            for answer in &answers {
                let read = read_line(&mut source).await.unwrap();
                assert_eq!(lex(&read).unwrap(), *answer);
            }

            assert!(matches!(read_line(&mut source).await, Err(Error::Closed)));
        });
    }

    #[hegel::test(test_cases = 50)]
    fn prop_a_tag_never_repeats(tc: TestCase) {
        let count: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(64));

        let mut tags = Tags::new();
        let made: Vec<String> = (0..count).map(|_| tags.next_tag()).collect();

        let mut sorted = made.clone();
        sorted.sort();
        sorted.dedup();

        assert_eq!(sorted.len(), made.len());
    }
}
