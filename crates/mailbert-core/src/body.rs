//! Removal of quoted text and signatures from a message body.
//!
//! This is the most important step for search quality. Message 40 of a
//! thread holds 39 copies of the text before it. If mailbert indexed
//! that text, the IDF of the BM25 index would collapse, each message of
//! the thread would match each query about the thread, and the
//! embeddings of the thread would converge.
//!
//! mailbert removes the quoted blocks from the **indexed** text only.
//! The raw bytes stay in `blobs.db`, so `view` shows the whole message.
//!
//! See `docs/mailbert.md` §5.2.

use regex::Regex;

/// The Outlook separator, in lower case for a case-insensitive compare.
const ORIGINAL_MESSAGE: &str = "-----original message-----";

/// The RFC 3676 signature separator, after the trailing space is cut.
const SIGNATURE: &str = "--";

/// The text of a body after mailbert removes the quotes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stripped {
    /// The text to index. Line endings are normalized to `\n`.
    pub text: String,

    /// True when the body held no original text. `text` is then the
    /// whole body, because an empty document is worse than a noisy one.
    pub quote_only: bool,
}

/// Whether a line is quoted text.
///
/// # Examples
///
/// ```
/// use mailbert_core::body::is_quote_line;
///
/// assert!(is_quote_line("> Can we meet Tuesday?"));
/// assert!(!is_quote_line("Yes, Tuesday works."));
/// ```
pub fn is_quote_line(line: &str) -> bool {
    line.trim_start().starts_with('>')
}

/// Whether a line attributes the quote that follows it.
///
/// mailbert accepts the `On <date>, <person> wrote:` form that Gmail,
/// Apple Mail, and Thunderbird all write. Both ends are necessary: `On
/// Tuesday I will send the report` is prose, and so is `Alice wrote:`.
pub fn is_attribution_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with("On ") && trimmed.ends_with("wrote:")
}

/// Whether a line separates a signature from the body.
///
/// RFC 3676 makes this exactly `--` and a space. Many clients cut the
/// space, so mailbert accepts the line with or without it. Leading
/// space is not accepted, because an indented `--` is usually prose.
pub fn is_signature_separator(line: &str) -> bool {
    line.trim_end() == SIGNATURE
}

/// Whether a line starts an Outlook quote.
///
/// Outlook does not prefix the quote that follows, so this separator is
/// the only mark of where the text of the sender ends.
pub fn is_original_message_separator(line: &str) -> bool {
    line.trim()
        .get(..ORIGINAL_MESSAGE.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(ORIGINAL_MESSAGE))
}

/// Remove quoted text and signatures from `body`.
///
/// # Examples
///
/// ```
/// use mailbert_core::strip;
///
/// let stripped = strip(
///     "Yes, Tuesday works.\n\
///      \n\
///      On 2026-01-05, Alice wrote:\n\
///      > Can we meet Tuesday?\n",
/// );
///
/// assert_eq!(stripped.text, "Yes, Tuesday works.");
/// assert!(!stripped.quote_only);
/// ```
pub fn strip(body: &str) -> Stripped {
    strip_with_footers(body, &[])
}

/// Remove quoted text, signatures, and corporate footers from `body`.
///
/// A line that matches one of `footers`, and each line after it, is
/// removed. The patterns come from the account configuration, because
/// only the reader knows what their employer appends.
///
/// # Examples
///
/// ```
/// use mailbert_core::strip_with_footers;
/// use regex::Regex;
///
/// let footers = vec![Regex::new(r"^This email is confidential").unwrap()];
/// let stripped = strip_with_footers(
///     "The invoice is attached.\n\nThis email is confidential.\n",
///     &footers,
/// );
///
/// assert_eq!(stripped.text, "The invoice is attached.");
/// ```
pub fn strip_with_footers(body: &str, footers: &[Regex]) -> Stripped {
    let lines: Vec<&str> = body
        .lines()
        .map(|line| line.trim_end_matches('\r'))
        .collect();

    let mut kept: Vec<&str> = Vec::new();
    let mut index = 0;

    while index < lines.len() {
        let line = lines[index];

        // A quote comes first, so that a footer or a separator inside a
        // quote cannot cut the answers that follow it.
        if is_quote_line(line) {
            index += 1;
            continue;
        }

        if is_signature_separator(line)
            || is_original_message_separator(line)
            || footers.iter().any(|footer| footer.is_match(line))
        {
            break;
        }

        let attribution = attribution_lines(&lines, index);
        if attribution > 0 {
            index += attribution;
            continue;
        }

        kept.push(line);
        index += 1;
    }

    let text = render(&kept);
    if !text.is_empty() {
        return Stripped {
            text,
            quote_only: false,
        };
    }

    // A forward with no comment. An empty document is unreachable by
    // search, so keep the quote and say so.
    let whole = render(&lines);
    let quote_only = !whole.is_empty();

    Stripped {
        text: whole,
        quote_only,
    }
}

/// How many lines the attribution at `index` occupies, or 0 for none.
///
/// Gmail wraps a long attribution, and its tail alone (`example.com>
/// wrote:`) would otherwise look like text the sender typed.
fn attribution_lines(lines: &[&str], index: usize) -> usize {
    let first = lines[index].trim();

    if !first.starts_with("On ") {
        return 0;
    }

    if first.ends_with("wrote:") {
        return 1;
    }

    match lines.get(index + 1) {
        Some(next)
            if !next.trim().is_empty()
                && !is_quote_line(next)
                && next.trim_end().ends_with("wrote:") =>
        {
            2
        }
        _ => 0,
    }
}

/// Join the kept lines, without a leading, trailing, or repeated blank.
///
/// Each line stays byte-identical, so the caller can always find the
/// output in the input.
fn render(lines: &[&str]) -> String {
    let mut out: Vec<&str> = Vec::new();

    for line in lines {
        let blank = line.trim().is_empty();
        let after_blank = out.last().is_some_and(|last| last.trim().is_empty());

        if blank && (out.is_empty() || after_blank) {
            continue;
        }

        out.push(line);
    }

    while out.last().is_some_and(|last| last.trim().is_empty()) {
        out.pop();
    }

    out.join("\n")
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_stripping_is_idempotent` | algebraic | The pipeline may strip a body more than once. A second pass must not eat more text. |
    //! | `prop_output_lines_come_from_the_input` | invariant | Stripping removes text. It must never invent a line. |
    //! | `prop_original_text_always_survives` | model-based | The whole point. If a reply loses the words the sender typed, search cannot find it. |
    //! | `prop_no_quote_survives_original_text` | invariant | A surviving quote is what collapses the IDF. |
    //! | `prop_never_blank_when_the_body_is_not` | invariant | An empty document is unreachable by search, so a forward with no comment must keep its quote. |
    //! | `prop_quote_only_agrees_with_the_text` | invariant | The flag drives that fallback. If it disagrees with the text, the caller cannot trust it. |
    //! | `prop_a_cut_removes_the_whole_tail` | invariant | A signature or an Outlook quote ends the original text. Nothing after it is the sender's. |
    //! | `prop_output_is_never_longer` | invariant | A cheap guard against a rule that duplicates instead of removes. |

    use hegel::{TestCase, generators as gs};

    use super::*;

    // -----------------------------------------------------------------
    // Generators.
    // -----------------------------------------------------------------

    /// A line of the sender's own text. Never quoted, never a separator.
    fn original_line() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "Yes, Tuesday works.".to_string(),
            "I pushed the fix.".to_string(),
            "Can you look at the invoice?".to_string(),
            "Thanks!".to_string(),
            "See the attached report.".to_string(),
        ])
    }

    fn quoted_line() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "> Can we meet Tuesday?".to_string(),
            ">> An older message.".to_string(),
            ">".to_string(),
            "  > An indented quote.".to_string(),
            "> -- ".to_string(),
            "> Alice".to_string(),
        ])
    }

    fn attribution() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "On Mon, Jan 5, 2026 at 3:04 PM Alice <a@x.example> wrote:"
                .to_string(),
            "On 2026-01-05, Alice Example wrote:".to_string(),
        ])
    }

    /// A body of original text, then an attribution, then a quote.
    /// Returns the body and the original lines it must keep.
    #[hegel::composite]
    fn reply_body(tc: TestCase) -> (String, Vec<String>) {
        let original: Vec<String> =
            tc.draw(gs::vecs(original_line()).min_size(1).max_size(4));
        let quoted: Vec<String> =
            tc.draw(gs::vecs(quoted_line()).min_size(1).max_size(6));
        let with_attribution: bool = tc.draw(gs::booleans());

        let mut lines = original.clone();
        lines.push(String::new());

        if with_attribution {
            lines.push(tc.draw(attribution()));
        }

        lines.extend(quoted);

        (lines.join("\n"), original)
    }

    /// An arbitrary body built from every kind of line, in any order.
    #[hegel::composite]
    fn any_body(tc: TestCase) -> String {
        let line = hegel::one_of!(
            original_line(),
            quoted_line(),
            attribution(),
            gs::sampled_from(vec![
                String::new(),
                "-- ".to_string(),
                "--".to_string(),
                "-----Original Message-----".to_string(),
                "Sent from my phone".to_string(),
            ]),
        );

        let lines: Vec<String> =
            tc.draw(gs::vecs(line).min_size(0).max_size(12));

        lines.join("\n")
    }

    // -----------------------------------------------------------------
    // Unit tests.
    // -----------------------------------------------------------------

    #[test]
    fn quote_lines_are_recognized_at_any_indent() {
        assert!(is_quote_line("> hello"));
        assert!(is_quote_line(">"));
        assert!(is_quote_line(">> nested"));
        assert!(is_quote_line("   > indented"));
        assert!(!is_quote_line("hello > world"));
        assert!(!is_quote_line(""));
    }

    #[test]
    fn attribution_lines_need_both_ends() {
        assert!(is_attribution_line(
            "On Mon, Jan 5, 2026 at 3:04 PM Alice <a@x.example> wrote:"
        ));
        assert!(is_attribution_line("On 2026-01-05, Alice wrote:"));
        // "On" without "wrote:" is ordinary prose.
        assert!(!is_attribution_line("On Tuesday I will send the report"));
        // "wrote:" without "On" is also ordinary prose.
        assert!(!is_attribution_line("Alice wrote:"));
    }

    #[test]
    fn the_signature_separator_is_exactly_two_dashes() {
        assert!(is_signature_separator("-- "));
        assert!(is_signature_separator("--"));
        assert!(is_signature_separator("--\r"));
        // An em-dash rule and a diff marker are not signatures.
        assert!(!is_signature_separator("---"));
        assert!(!is_signature_separator(" -- "));
        assert!(!is_signature_separator("-- Alice"));
    }

    #[test]
    fn the_outlook_separator_ignores_case() {
        assert!(is_original_message_separator("-----Original Message-----"));
        assert!(is_original_message_separator("-----ORIGINAL MESSAGE-----"));
        assert!(!is_original_message_separator("-----Original-----"));
    }

    #[test]
    fn a_top_posted_reply_keeps_only_the_new_text() {
        let stripped = strip(
            "Yes, Tuesday works.\n\
             \n\
             On Mon, Jan 5, 2026 at 3:04 PM Alice <a@x.example> wrote:\n\
             > Can we meet Tuesday?\n\
             > -- \n\
             > Alice\n",
        );

        assert_eq!(stripped.text, "Yes, Tuesday works.");
        assert!(!stripped.quote_only);
    }

    #[test]
    fn an_interleaved_reply_keeps_each_answer() {
        // Removing everything after the attribution would lose "Yes."
        // and "No." — the only text the sender actually typed.
        let stripped = strip(
            "On 2026-01-05, Alice wrote:\n\
             > Can we meet Tuesday?\n\
             Yes.\n\
             > And Wednesday?\n\
             No.\n",
        );

        assert_eq!(stripped.text, "Yes.\nNo.");
        assert!(!stripped.quote_only);
    }

    #[test]
    fn the_signature_cuts_the_rest_of_the_body() {
        let stripped = strip(
            "I pushed the fix.\n\
             \n\
             -- \n\
             Alice Example\n\
             Staff Engineer\n",
        );

        assert_eq!(stripped.text, "I pushed the fix.");
    }

    #[test]
    fn the_outlook_separator_cuts_an_unquoted_reply() {
        // Outlook does not prefix its quote, so only the separator can
        // tell mailbert where the sender's text ends.
        let stripped = strip(
            "Approved.\n\
             \n\
             -----Original Message-----\n\
             From: Alice <a@x.example>\n\
             Subject: Budget\n\
             \n\
             Please approve the budget.\n",
        );

        assert_eq!(stripped.text, "Approved.");
    }

    #[test]
    fn a_forward_with_no_comment_keeps_its_quote() {
        let body = "On 2026-01-05, Alice wrote:\n\
                    > Please approve the budget.\n";

        let stripped = strip(body);

        assert!(stripped.quote_only);
        assert!(stripped.text.contains("Please approve the budget."));
    }

    #[test]
    fn a_blank_body_is_blank_and_not_quote_only() {
        let stripped = strip("   \n\n  \n");

        assert!(stripped.text.is_empty());
        assert!(!stripped.quote_only);
    }

    #[test]
    fn crlf_line_endings_are_normalized() {
        let stripped = strip("Thanks!\r\n\r\n> quoted\r\n");

        assert_eq!(stripped.text, "Thanks!");
    }

    #[test]
    fn a_wrapped_attribution_is_removed_whole() {
        // Gmail wraps a long attribution, and the tail alone would
        // otherwise look like original text.
        let stripped = strip(
            "Sure.\n\
             \n\
             On Mon, Jan 5, 2026 at 3:04 PM Alice Example <alice@\n\
             example.com> wrote:\n\
             > Can we meet Tuesday?\n",
        );

        assert_eq!(stripped.text, "Sure.");
    }

    #[test]
    fn a_footer_pattern_cuts_the_rest_of_the_body() {
        let footers =
            vec![Regex::new(r"^This email and any attachments").unwrap()];

        let stripped = strip_with_footers(
            "The invoice is attached.\n\
             \n\
             This email and any attachments are confidential.\n\
             If you are not the intended recipient, delete it.\n",
            &footers,
        );

        assert_eq!(stripped.text, "The invoice is attached.");
    }

    #[test]
    fn blank_runs_collapse_to_one_line() {
        let stripped = strip("Thanks!\n\n> quoted\n\n> more\n\nSee you.\n");

        assert_eq!(stripped.text, "Thanks!\n\nSee you.");
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 200)]
    fn prop_stripping_is_idempotent(tc: TestCase) {
        let body: String = tc.draw(any_body());

        let once = strip(&body);
        let twice = strip(&once.text);

        assert_eq!(once.text, twice.text, "second pass changed {body:?}");
    }

    #[hegel::test(test_cases = 200)]
    fn prop_output_lines_come_from_the_input(tc: TestCase) {
        let body: String = tc.draw(any_body());

        let input: Vec<&str> =
            body.lines().map(|l| l.trim_end_matches('\r')).collect();
        let stripped = strip(&body);

        for line in stripped.text.lines() {
            assert!(
                input.contains(&line),
                "invented line {line:?} for body {body:?}"
            );
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_original_text_always_survives(tc: TestCase) {
        let (body, original) = tc.draw(reply_body());

        let stripped = strip(&body);

        for line in &original {
            assert!(
                stripped.text.contains(line.as_str()),
                "lost {line:?} from {body:?}"
            );
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_no_quote_survives_original_text(tc: TestCase) {
        let body: String = tc.draw(any_body());

        let stripped = strip(&body);
        tc.assume(!stripped.quote_only);

        for line in stripped.text.lines() {
            assert!(!is_quote_line(line), "quote {line:?} survived {body:?}");
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_never_blank_when_the_body_is_not(tc: TestCase) {
        let body: String = tc.draw(any_body());
        tc.assume(body.chars().any(|c| !c.is_whitespace()));

        let stripped = strip(&body);

        assert!(
            stripped.text.chars().any(|c| !c.is_whitespace()),
            "{body:?} stripped to nothing"
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_quote_only_agrees_with_the_text(tc: TestCase) {
        let body: String = tc.draw(any_body());

        let stripped = strip(&body);

        if stripped.quote_only {
            // The fallback keeps the whole body, so a quote is there.
            assert!(!stripped.text.is_empty());
        }

        // Blank in, blank out, and the flag stays down.
        if !body.chars().any(|c| !c.is_whitespace()) {
            assert!(stripped.text.is_empty());
            assert!(!stripped.quote_only);
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_cut_removes_the_whole_tail(tc: TestCase) {
        let head: Vec<String> =
            tc.draw(gs::vecs(original_line()).min_size(1).max_size(3));
        let cut: String = tc.draw(gs::sampled_from(vec![
            "-- ".to_string(),
            "-----Original Message-----".to_string(),
        ]));
        let tail: String = tc.draw(gs::sampled_from(vec![
            "Alice Example, Staff Engineer".to_string(),
            "Sent: Monday 5 January 2026".to_string(),
        ]));

        let mut lines = head.clone();
        lines.push(cut);
        lines.push(tail.clone());

        let stripped = strip(&lines.join("\n"));

        assert!(
            !stripped.text.contains(tail.as_str()),
            "tail {tail:?} survived the cut in {stripped:?}"
        );
        for line in &head {
            assert!(stripped.text.contains(line.as_str()));
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_output_is_never_longer(tc: TestCase) {
        let body: String = tc.draw(any_body());

        let stripped = strip(&body);

        assert!(
            stripped.text.len() <= body.len(),
            "{body:?} grew to {:?}",
            stripped.text
        );
    }
}
