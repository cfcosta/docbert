/// Default number of lines in a snippet when no match is found.
pub const DEFAULT_SNIPPET_LINES: usize = 6;

/// Maximum number of characters in a snippet before truncation.
pub const DEFAULT_SNIPPET_MAX_CHARS: usize = 400;

/// Prepend line numbers to each line of text.
///
/// `start_line` is the number to assign to the first line (1-indexed).
pub fn add_line_numbers(text: &str, start_line: usize) -> String {
    text.lines()
        .enumerate()
        .map(|(i, line)| format!("{}: {}", start_line + i, line))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extract a snippet around the first occurrence of `query` in `text`.
///
/// Returns `(snippet_text, start_line_number)` where start_line_number is
/// 1-indexed. If `query` is not found, returns the first few lines.
/// Returns `None` if the text is empty.
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
        let start = idx.saturating_sub(2);
        let end = (idx + 3).min(lines.len());
        (start, end)
    } else {
        (0, DEFAULT_SNIPPET_LINES.min(lines.len()))
    };

    let mut snippet = lines[start..end].join("\n");
    if snippet.len() > DEFAULT_SNIPPET_MAX_CHARS {
        snippet.truncate(DEFAULT_SNIPPET_MAX_CHARS);
        snippet.push_str("...");
    }

    Some((snippet, start + 1))
}

/// Apply line offset and optional line limit to a block of text.
///
/// `start_line` is 1-indexed. If `max_lines` is `Some(n)`, at most `n` lines
/// are returned with a truncation notice appended.
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

#[cfg(test)]
mod tests {
    use super::*;

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
