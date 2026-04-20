/// Default number of lines in a snippet when no match is found.
pub const DEFAULT_SNIPPET_LINES: usize = 6;

/// Maximum number of characters in a snippet before truncation.
pub const DEFAULT_SNIPPET_MAX_CHARS: usize = 400;

const MATCH_CONTEXT_LINES_BEFORE: usize = 2;
const MATCH_CONTEXT_LINES_AFTER: usize = 2;

/// A text excerpt with inclusive line range metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextExcerpt {
    pub text: String,
    pub start_line: usize,
    pub end_line: usize,
}

/// Strip a leading YAML frontmatter block when one is present.
///
/// Frontmatter must start at the beginning of the document with `---` and end
/// with a closing `---` or `...` line. If the block is unterminated, the
/// original text is returned unchanged.
pub fn strip_yaml_frontmatter(text: &str) -> &str {
    let rest = if let Some(rest) = text.strip_prefix("---\n") {
        rest
    } else if let Some(rest) = text.strip_prefix("---\r\n") {
        rest
    } else {
        return text;
    };

    let mut offset = text.len() - rest.len();
    for line in rest.split_inclusive('\n') {
        offset += line.len();
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed == "---" || trimmed == "..." {
            return &text[offset..];
        }
    }

    text
}

/// Add line numbers to each line of text.
///
/// `start_line` is the number assigned to the first line. It is 1-indexed.
///
/// # Examples
///
/// ```
/// use docbert_core::text::add_line_numbers;
///
/// let numbered = add_line_numbers("foo\nbar\nbaz", 1);
/// assert_eq!(numbered, "1: foo\n2: bar\n3: baz");
///
/// let numbered = add_line_numbers("first\nsecond", 10);
/// assert_eq!(numbered, "10: first\n11: second");
/// ```
pub fn add_line_numbers(text: &str, start_line: usize) -> String {
    text.lines()
        .enumerate()
        .map(|(i, line)| format!("{}: {}", start_line + i, line))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extract up to `max_excerpts` distinct excerpts around literal query matches.
///
/// Excerpts are line-based. Each match includes two lines of context before and
/// after the matching line when available. Overlapping windows are merged into a
/// single excerpt. When the query is not found literally, the first few lines of
/// the document are returned instead. Empty input returns an empty vector.
///
/// Returned line numbers are 1-indexed and inclusive.
pub fn extract_excerpts(
    text: &str,
    query: &str,
    max_excerpts: usize,
) -> Vec<TextExcerpt> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() || max_excerpts == 0 {
        return Vec::new();
    }

    let query_lower = query.to_lowercase();
    let mut ranges: Vec<(usize, usize)> = lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| {
            line.to_lowercase().contains(&query_lower).then_some((
                idx.saturating_sub(MATCH_CONTEXT_LINES_BEFORE),
                (idx + MATCH_CONTEXT_LINES_AFTER + 1).min(lines.len()),
            ))
        })
        .collect();

    if ranges.is_empty() {
        ranges.push((0, DEFAULT_SNIPPET_LINES.min(lines.len())));
    }

    let mut merged_ranges: Vec<(usize, usize)> = Vec::new();
    for (start, end) in ranges {
        if let Some((last_start, last_end)) = merged_ranges.last_mut()
            && start <= *last_end
        {
            *last_start = (*last_start).min(start);
            *last_end = (*last_end).max(end);
            continue;
        }
        merged_ranges.push((start, end));
    }

    merged_ranges
        .into_iter()
        .take(max_excerpts)
        .map(|(start, end)| TextExcerpt {
            text: truncate_snippet(lines[start..end].join("\n")),
            start_line: start + 1,
            end_line: end,
        })
        .collect()
}

/// Extract a snippet around the first occurrence of `query` in `text`.
///
/// Returns `(snippet_text, start_line_number)`, where `start_line_number` is
/// 1-indexed. If the query is not found, the first few lines are returned instead.
/// Empty input returns `None`.
///
/// # Examples
///
/// ```
/// use docbert_core::text::extract_snippet;
///
/// let text = "line1\nline2\nline3\nfind me here\nline5";
/// let (snippet, start) = extract_snippet(text, "find me").unwrap();
/// assert!(snippet.contains("find me here"));
/// assert!(start >= 1);
///
/// assert!(extract_snippet("", "query").is_none());
/// ```
pub fn extract_snippet(text: &str, query: &str) -> Option<(String, usize)> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return None;
    }

    let query_lower = query.to_lowercase();
    let mut match_idx = None;

    for (idx, line) in lines.iter().enumerate() {
        if line.to_lowercase().contains(&query_lower) {
            match_idx = Some(idx);
            break;
        }
    }

    let (start, end) = if let Some(idx) = match_idx {
        let start = idx.saturating_sub(MATCH_CONTEXT_LINES_BEFORE);
        let end = (idx + MATCH_CONTEXT_LINES_AFTER + 1).min(lines.len());
        (start, end)
    } else {
        (0, DEFAULT_SNIPPET_LINES.min(lines.len()))
    };

    let snippet = truncate_snippet(lines[start..end].join("\n"));

    Some((snippet, start + 1))
}

/// Slice text to an inclusive 1-indexed line range.
///
/// `start_line` and `end_line` are both inclusive and 1-indexed. `None` on
/// either bound means "open-ended" — `None` for `start_line` is treated as line
/// 1, and `None` for `end_line` means "until the last line".
///
/// When the range omits trailing content, a `[... N more lines remaining]`
/// footer is appended so callers know more content exists beyond the window.
///
/// # Examples
///
/// ```
/// use docbert_core::text::apply_line_range;
///
/// let text = "line1\nline2\nline3\nline4\nline5";
///
/// assert_eq!(apply_line_range(text, Some(2), None), "line2\nline3\nline4\nline5");
///
/// let result = apply_line_range(text, Some(1), Some(2));
/// assert!(result.starts_with("line1\nline2"));
/// assert!(result.contains("more lines remaining"));
/// ```
pub fn apply_line_range(
    text: &str,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    let start_idx = start_line.unwrap_or(1).saturating_sub(1).min(lines.len());
    if start_idx >= lines.len() {
        return String::new();
    }

    let end_idx = match end_line {
        Some(end) => end.min(lines.len()),
        None => lines.len(),
    };

    if end_idx <= start_idx {
        return String::new();
    }

    let mut slice = lines[start_idx..end_idx].join("\n");
    let remaining = lines.len() - end_idx;
    if end_line.is_some() && remaining > 0 {
        let suffix = if remaining == 1 { "line" } else { "lines" };
        slice.push_str(&format!(
            "\n\n[... {remaining} more {suffix} remaining]"
        ));
    }

    slice
}

/// Slice text to an inclusive byte range.
///
/// `start_byte` and `end_byte` are both inclusive and 0-indexed. `None` on
/// either bound means "open-ended". Byte offsets that land inside a multi-byte
/// UTF-8 character are rounded down to the previous character boundary so the
/// returned string is always valid UTF-8.
///
/// When the range omits trailing content, a `[... N more bytes remaining]`
/// footer is appended.
///
/// # Examples
///
/// ```
/// use docbert_core::text::apply_byte_range;
///
/// assert_eq!(apply_byte_range("hello world", Some(6), None), "world");
///
/// let result = apply_byte_range("hello world", Some(0), Some(4));
/// assert!(result.starts_with("hello"));
/// assert!(result.contains("more bytes remaining"));
/// ```
pub fn apply_byte_range(
    text: &str,
    start_byte: Option<u64>,
    end_byte: Option<u64>,
) -> String {
    let total = text.len();
    if total == 0 {
        return String::new();
    }

    let start = start_byte.unwrap_or(0) as usize;
    let start = floor_char_boundary(text, start.min(total));
    if start >= total {
        return String::new();
    }

    let end_exclusive = match end_byte {
        Some(end) => {
            let end = end as usize;
            if end < start {
                return String::new();
            }
            floor_char_boundary(text, end.saturating_add(1).min(total))
        }
        None => total,
    };

    if end_exclusive <= start {
        return String::new();
    }

    let mut slice = text[start..end_exclusive].to_string();
    let remaining = total - end_exclusive;
    if end_byte.is_some() && remaining > 0 {
        let suffix = if remaining == 1 { "byte" } else { "bytes" };
        slice.push_str(&format!(
            "\n\n[... {remaining} more {suffix} remaining]"
        ));
    }

    slice
}

fn floor_char_boundary(text: &str, mut byte: usize) -> usize {
    while byte > 0 && !text.is_char_boundary(byte) {
        byte -= 1;
    }
    byte
}

fn truncate_snippet(mut snippet: String) -> String {
    if snippet.len() > DEFAULT_SNIPPET_MAX_CHARS {
        snippet.truncate(DEFAULT_SNIPPET_MAX_CHARS);
        snippet.push_str("...");
    }

    snippet
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_yaml_frontmatter_removes_leading_block() {
        let text = "---\ntitle: Hello\ntags: [a, b]\n---\n# Heading\n\nBody";
        assert_eq!(strip_yaml_frontmatter(text), "# Heading\n\nBody");
    }

    #[test]
    fn strip_yaml_frontmatter_keeps_plain_text() {
        let text = "# Heading\n\nBody";
        assert_eq!(strip_yaml_frontmatter(text), text);
    }

    #[test]
    fn strip_yaml_frontmatter_ignores_unterminated_block() {
        let text = "---\ntitle: Hello\n# Heading";
        assert_eq!(strip_yaml_frontmatter(text), text);
    }

    #[test]
    fn add_line_numbers_basic() {
        assert_eq!(add_line_numbers("foo\nbar", 1), "1: foo\n2: bar");
    }

    #[test]
    fn add_line_numbers_custom_start() {
        assert_eq!(add_line_numbers("foo\nbar", 10), "10: foo\n11: bar");
    }

    #[test]
    fn add_line_numbers_single_line() {
        assert_eq!(add_line_numbers("hello", 5), "5: hello");
    }

    #[test]
    fn extract_excerpts_returns_multiple_non_overlapping_matches() {
        let text = [
            "line1",
            "rust line one",
            "line3",
            "line4",
            "line5",
            "line6",
            "line7",
            "line8",
            "rust line two",
            "line10",
            "line11",
        ]
        .join("\n");

        let excerpts = extract_excerpts(&text, "rust", 3);

        assert_eq!(excerpts.len(), 2);
        assert_eq!(
            excerpts[0],
            TextExcerpt {
                text: "line1\nrust line one\nline3\nline4".to_string(),
                start_line: 1,
                end_line: 4,
            }
        );
        assert_eq!(
            excerpts[1],
            TextExcerpt {
                text: "line7\nline8\nrust line two\nline10\nline11".to_string(),
                start_line: 7,
                end_line: 11,
            }
        );
    }

    #[test]
    fn extract_excerpts_no_match_returns_head_excerpt() {
        let text = "line1\nline2\nline3\nline4\nline5\nline6\nline7";

        let excerpts = extract_excerpts(text, "zzz_nomatch", 3);

        assert_eq!(
            excerpts,
            vec![TextExcerpt {
                text: "line1\nline2\nline3\nline4\nline5\nline6".to_string(),
                start_line: 1,
                end_line: 6,
            }]
        );
    }

    #[test]
    fn extract_excerpts_caps_result_count() {
        let text = [
            "rust a", "line2", "line3", "line4", "line5", "line6", "line7",
            "rust b", "line9", "line10", "line11", "line12", "line13",
            "line14", "rust c", "line16", "line17", "line18", "line19",
            "line20", "line21", "rust d",
        ]
        .join("\n");

        let excerpts = extract_excerpts(&text, "rust", 3);

        assert_eq!(excerpts.len(), 3);
        assert!(excerpts.iter().all(|excerpt| excerpt.text.contains("rust")));
        assert!(
            !excerpts
                .iter()
                .any(|excerpt| excerpt.text.contains("rust d"))
        );
    }

    #[test]
    fn extract_excerpts_merges_overlapping_match_windows() {
        let text = [
            "line1",
            "line2",
            "rust first",
            "line4",
            "rust second",
            "line6",
            "line7",
            "line8",
        ]
        .join("\n");

        let excerpts = extract_excerpts(&text, "rust", 3);

        assert_eq!(excerpts.len(), 1);
        assert_eq!(excerpts[0].start_line, 1);
        assert_eq!(excerpts[0].end_line, 7);
        assert!(excerpts[0].text.contains("rust first"));
        assert!(excerpts[0].text.contains("rust second"));
    }

    #[test]
    fn extract_excerpts_reports_line_ranges() {
        let text = "line1\nline2\nline3\nline4\nrust line\nline6\nline7\nline8";

        let excerpts = extract_excerpts(text, "rust", 3);

        assert_eq!(excerpts.len(), 1);
        assert_eq!(excerpts[0].start_line, 3);
        assert_eq!(excerpts[0].end_line, 7);
    }

    #[test]
    fn extract_snippet_match_found() {
        let text = "line1\nline2\nline3\nrust is great\nline5\nline6\nline7";
        let (snippet, start) = extract_snippet(text, "rust").unwrap();
        assert!(snippet.contains("rust is great"));
        assert!(start >= 1);
    }

    #[test]
    fn extract_snippet_no_match_returns_head() {
        let text = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8";
        let (snippet, start) = extract_snippet(text, "zzz_nomatch").unwrap();
        assert_eq!(start, 1);
        assert!(snippet.starts_with("line1"));
    }

    #[test]
    fn extract_snippet_empty_text() {
        assert!(extract_snippet("", "query").is_none());
    }

    #[test]
    fn extract_snippet_truncates_long() {
        let long_line = "a".repeat(500);
        let text = format!("{long_line}\n{long_line}");
        let (snippet, _) = extract_snippet(&text, "a").unwrap();
        assert!(snippet.len() <= DEFAULT_SNIPPET_MAX_CHARS + 3); // +3 for "..."
        assert!(snippet.ends_with("..."));
    }

    #[test]
    fn extract_snippet_keeps_existing_first_match_behavior() {
        let text = "line1\nline2\nline3\nrust is great\nline5\nline6\nrust later\nline8";
        let (snippet, start) = extract_snippet(text, "rust").unwrap();

        assert_eq!(snippet, "line2\nline3\nrust is great\nline5\nline6");
        assert_eq!(start, 2);
    }

    #[test]
    fn apply_line_range_full_text() {
        let text = "line1\nline2\nline3";
        assert_eq!(apply_line_range(text, None, None), text);
    }

    #[test]
    fn apply_line_range_start_only() {
        let text = "line1\nline2\nline3\nline4\nline5";
        assert_eq!(
            apply_line_range(text, Some(3), None),
            "line3\nline4\nline5"
        );
    }

    #[test]
    fn apply_line_range_end_only_truncates() {
        let text = "line1\nline2\nline3\nline4\nline5";
        assert_eq!(
            apply_line_range(text, None, Some(2)),
            "line1\nline2\n\n[... 3 more lines remaining]"
        );
    }

    #[test]
    fn apply_line_range_inclusive_both_bounds() {
        let text = "line1\nline2\nline3\nline4\nline5";
        assert_eq!(
            apply_line_range(text, Some(2), Some(4)),
            "line2\nline3\nline4\n\n[... 1 more line remaining]"
        );
    }

    #[test]
    fn apply_line_range_end_at_last_line_has_no_footer() {
        let text = "line1\nline2\nline3";
        assert_eq!(
            apply_line_range(text, Some(1), Some(3)),
            "line1\nline2\nline3"
        );
    }

    #[test]
    fn apply_line_range_past_end_returns_empty() {
        let text = "line1\nline2";
        assert!(apply_line_range(text, Some(100), None).is_empty());
    }

    #[test]
    fn apply_line_range_inverted_returns_empty() {
        let text = "line1\nline2\nline3";
        assert!(apply_line_range(text, Some(3), Some(1)).is_empty());
    }

    #[test]
    fn apply_byte_range_full_text() {
        let text = "hello world";
        assert_eq!(apply_byte_range(text, None, None), text);
    }

    #[test]
    fn apply_byte_range_slices_prefix_with_footer() {
        let text = "hello world";
        assert_eq!(
            apply_byte_range(text, Some(0), Some(4)),
            "hello\n\n[... 6 more bytes remaining]"
        );
    }

    #[test]
    fn apply_byte_range_slices_suffix_without_footer() {
        let text = "hello world";
        assert_eq!(apply_byte_range(text, Some(6), Some(10)), "world");
    }

    #[test]
    fn apply_byte_range_start_only_reads_to_end() {
        let text = "hello world";
        assert_eq!(apply_byte_range(text, Some(6), None), "world");
    }

    #[test]
    fn apply_byte_range_past_end_returns_empty() {
        let text = "hello";
        assert!(apply_byte_range(text, Some(100), None).is_empty());
    }

    #[test]
    fn apply_byte_range_inverted_returns_empty() {
        let text = "hello world";
        assert!(apply_byte_range(text, Some(5), Some(2)).is_empty());
    }

    #[test]
    fn apply_byte_range_clamps_to_utf8_char_boundary() {
        // Greek αβγ: each char is 2 bytes, total 6 bytes.
        // end_byte = 2 (inclusive) => exclusive end = 3, which lands mid-β;
        // we clamp to the previous char boundary (byte 2), yielding "α".
        let text = "αβγ";
        assert_eq!(text.len(), 6);
        assert_eq!(
            apply_byte_range(text, Some(0), Some(2)),
            "α\n\n[... 4 more bytes remaining]"
        );
    }

    #[test]
    fn apply_byte_range_start_on_mid_char_clamps_back() {
        // start=1 is mid-α → clamped to byte 0; end=3 exclusive-ends at byte 4
        // which is a char boundary → slice covers "αβ".
        let text = "αβγ";
        assert_eq!(
            apply_byte_range(text, Some(1), Some(3)),
            "αβ\n\n[... 2 more bytes remaining]"
        );
    }

    #[test]
    fn apply_byte_range_single_long_line_slices_cleanly() {
        // The case that broke in the real world: one giant line that
        // line-based slicing can't cut.
        let text = "this is one big line with no newlines at all";
        assert_eq!(
            apply_byte_range(text, Some(0), Some(10)),
            "this is one\n\n[... 33 more bytes remaining]"
        );
    }
}
