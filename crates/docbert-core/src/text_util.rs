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
/// use docbert_core::text_util::add_line_numbers;
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
/// use docbert_core::text_util::extract_snippet;
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

/// Apply a line offset and optional line limit to a block of text.
///
/// `start_line` is 1-indexed. If `max_lines` is `Some(n)`, at most `n` lines
/// are returned and a truncation notice is appended.
///
/// # Examples
///
/// ```
/// use docbert_core::text_util::apply_line_limits;
///
/// let text = "line1\nline2\nline3\nline4\nline5";
///
/// // Start from line 2, no limit
/// assert_eq!(apply_line_limits(text, 2, None), "line2\nline3\nline4\nline5");
///
/// // Start from line 1, take 2 lines
/// let result = apply_line_limits(text, 1, Some(2));
/// assert!(result.starts_with("line1\nline2"));
/// assert!(result.contains("truncated"));
/// ```
pub fn apply_line_limits(
    text: &str,
    start_line: usize,
    max_lines: Option<usize>,
) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    let start_idx = start_line.saturating_sub(1).min(lines.len());
    if start_idx >= lines.len() {
        return String::new();
    }

    let end_idx = match max_lines {
        Some(max) => (start_idx + max).min(lines.len()),
        None => lines.len(),
    };

    let mut slice = lines[start_idx..end_idx].join("\n");
    if max_lines.is_some() && end_idx < lines.len() {
        slice.push_str(&format!(
            "\n\n[... truncated {} more lines]",
            lines.len() - end_idx
        ));
    }

    slice
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
    fn apply_line_limits_full_text() {
        let text = "line1\nline2\nline3";
        assert_eq!(apply_line_limits(text, 1, None), text);
    }

    #[test]
    fn apply_line_limits_with_offset() {
        let text = "line1\nline2\nline3\nline4\nline5";
        let result = apply_line_limits(text, 3, None);
        assert_eq!(result, "line3\nline4\nline5");
    }

    #[test]
    fn apply_line_limits_with_max() {
        let text = "line1\nline2\nline3\nline4\nline5";
        let result = apply_line_limits(text, 1, Some(2));
        assert!(result.starts_with("line1\nline2"));
        assert!(result.contains("truncated"));
    }

    #[test]
    fn apply_line_limits_past_end() {
        let text = "line1\nline2";
        let result = apply_line_limits(text, 100, None);
        assert!(result.is_empty());
    }

    #[test]
    fn apply_line_limits_offset_and_max() {
        let text = "line1\nline2\nline3\nline4\nline5";
        let result = apply_line_limits(text, 2, Some(2));
        assert!(result.starts_with("line2\nline3"));
        assert!(result.contains("truncated"));
    }
}
