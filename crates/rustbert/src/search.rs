//! In-memory search over a cached crate's items.
//!
//! Every public surface in rustbert (`rustbert search`,
//! `rustbert_search` MCP tool) takes a single `(crate, version)` and a
//! query, so search is always crate-scoped — no cross-crate ranking,
//! no global index.
//!
//! Scoring is a simple weighted-overlap metric tuned for code search:
//!
//! - +5 per query term found in the item's qualified path,
//! - +3 per term found in the signature,
//! - +1 per term found in the doc markdown.
//!
//! All matches are case-insensitive substring. This isn't BM25, but
//! for a per-crate corpus of a few hundred items it ranks the obvious
//! signal-vs-noise tradeoffs correctly and stays predictable.

use crate::item::{RustItem, RustItemKind};

#[derive(Debug, Clone)]
pub struct SearchHit<'a> {
    pub item: &'a RustItem,
    pub score: u32,
}

#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub kind: Option<RustItemKind>,
    pub module_prefix: Option<String>,
    pub limit: Option<usize>,
}

pub fn search<'a>(
    items: &'a [RustItem],
    query: &str,
    options: &SearchOptions,
) -> Vec<SearchHit<'a>> {
    let terms = tokenize(query);
    let limit = options.limit.unwrap_or(10);

    let mut hits: Vec<SearchHit<'a>> = items
        .iter()
        .filter(|item| matches_filters(item, options))
        .filter_map(|item| {
            let score = score_item(item, &terms);
            (score > 0).then_some(SearchHit { item, score })
        })
        .collect();

    hits.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.item.qualified_path.cmp(&b.item.qualified_path))
    });
    hits.truncate(limit);
    hits
}

/// Look up one item by exact qualified path. Used by `rustbert get`
/// and `rustbert_get`.
pub fn get<'a>(
    items: &'a [RustItem],
    qualified_path: &str,
) -> Option<&'a RustItem> {
    items.iter().find(|i| i.qualified_path == qualified_path)
}

/// Filtered listing — used by `rustbert list` and `rustbert_list`.
pub fn list<'a>(
    items: &'a [RustItem],
    options: &SearchOptions,
) -> Vec<&'a RustItem> {
    let mut out: Vec<&RustItem> = items
        .iter()
        .filter(|i| matches_filters(i, options))
        .collect();
    out.sort_by(|a, b| a.qualified_path.cmp(&b.qualified_path));
    if let Some(limit) = options.limit {
        out.truncate(limit);
    }
    out
}

fn matches_filters(item: &RustItem, options: &SearchOptions) -> bool {
    if let Some(k) = options.kind
        && item.kind != k
    {
        return false;
    }
    if let Some(prefix) = &options.module_prefix
        && !item.qualified_path.starts_with(prefix)
    {
        return false;
    }
    true
}

fn tokenize(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn score_item(item: &RustItem, terms: &[String]) -> u32 {
    if terms.is_empty() {
        return 0;
    }
    let path_lower = item.qualified_path.to_lowercase();
    let sig_lower = item.signature.to_lowercase();
    let doc_lower = item.doc_markdown.to_lowercase();

    let mut score = 0u32;
    for term in terms {
        if path_lower.contains(term) {
            score += 5;
        }
        if sig_lower.contains(term) {
            score += 3;
        }
        if doc_lower.contains(term) {
            score += 1;
        }
    }
    score
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::item::Visibility;

    fn item(kind: RustItemKind, qpath: &str, sig: &str, doc: &str) -> RustItem {
        RustItem {
            kind,
            crate_name: "x".to_string(),
            crate_version: semver::Version::new(0, 1, 0),
            module_path: vec![],
            name: Some("f".to_string()),
            qualified_path: qpath.to_string(),
            signature: sig.to_string(),
            doc_markdown: doc.to_string(),
            body: String::new(),
            source_file: PathBuf::from("src/lib.rs"),
            byte_start: 0,
            byte_len: 0,
            line_start: 1,
            line_end: 1,
            visibility: Visibility::Public,
            attrs: vec![],
        }
    }

    #[test]
    fn empty_query_returns_no_hits() {
        let items = vec![item(RustItemKind::Fn, "x::a", "fn a()", "doc")];
        let hits = search(&items, "", &SearchOptions::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn matches_qualified_path_with_high_score() {
        let items = vec![
            item(RustItemKind::Fn, "x::serializer", "fn s()", ""),
            item(RustItemKind::Fn, "x::other", "fn o()", "serializer"),
        ];
        let hits = search(&items, "serializer", &SearchOptions::default());
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].item.qualified_path, "x::serializer");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn case_insensitive_match() {
        let items =
            vec![item(RustItemKind::Struct, "x::Foo", "struct Foo", "")];
        let hits = search(&items, "FOO", &SearchOptions::default());
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn kind_filter_excludes_others() {
        let items = vec![
            item(RustItemKind::Fn, "x::foo", "fn foo()", ""),
            item(RustItemKind::Struct, "x::foo", "struct foo", ""),
        ];
        let opts = SearchOptions {
            kind: Some(RustItemKind::Struct),
            ..Default::default()
        };
        let hits = search(&items, "foo", &opts);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].item.kind, RustItemKind::Struct);
    }

    #[test]
    fn module_prefix_filter_scopes_results() {
        let items = vec![
            item(RustItemKind::Fn, "x::ser::a", "fn a()", "thing"),
            item(RustItemKind::Fn, "x::de::a", "fn a()", "thing"),
        ];
        let opts = SearchOptions {
            module_prefix: Some("x::ser".to_string()),
            ..Default::default()
        };
        let hits = search(&items, "thing", &opts);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].item.qualified_path.starts_with("x::ser"));
    }

    #[test]
    fn limit_caps_results() {
        let items: Vec<RustItem> = (0..20)
            .map(|i| {
                item(RustItemKind::Fn, &format!("x::matchme_{i}"), "fn _()", "")
            })
            .collect();
        let opts = SearchOptions {
            limit: Some(5),
            ..Default::default()
        };
        let hits = search(&items, "matchme", &opts);
        assert_eq!(hits.len(), 5);
    }

    #[test]
    fn ties_break_alphabetically() {
        let items = vec![
            item(RustItemKind::Fn, "x::z_thing", "fn _()", ""),
            item(RustItemKind::Fn, "x::a_thing", "fn _()", ""),
        ];
        let hits = search(&items, "thing", &SearchOptions::default());
        assert_eq!(hits[0].item.qualified_path, "x::a_thing");
    }

    #[test]
    fn get_finds_exact_qualified_path() {
        let items = vec![
            item(RustItemKind::Fn, "x::a", "", ""),
            item(RustItemKind::Fn, "x::b", "", ""),
        ];
        let got = get(&items, "x::b").unwrap();
        assert_eq!(got.qualified_path, "x::b");
        assert!(get(&items, "x::missing").is_none());
    }

    #[test]
    fn list_filters_and_sorts() {
        let items = vec![
            item(RustItemKind::Fn, "x::z", "", ""),
            item(RustItemKind::Struct, "x::a", "", ""),
            item(RustItemKind::Fn, "x::b", "", ""),
        ];
        let opts = SearchOptions {
            kind: Some(RustItemKind::Fn),
            ..Default::default()
        };
        let listed = list(&items, &opts);
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].qualified_path, "x::b");
        assert_eq!(listed[1].qualified_path, "x::z");
    }

    #[test]
    fn tokenize_splits_on_punctuation() {
        assert_eq!(tokenize("Foo::Bar::baz"), vec!["foo", "bar", "baz"]);
        assert_eq!(tokenize("hello, world!"), vec!["hello", "world"]);
    }

    #[test]
    fn multiple_terms_score_additively() {
        let items = vec![
            item(RustItemKind::Fn, "x::serialize", "fn s()", "struct"),
            item(RustItemKind::Fn, "x::other", "fn o()", ""),
        ];
        let hits =
            search(&items, "serialize struct", &SearchOptions::default());
        // "serialize" hits the path (5), "struct" hits doc (1) => 6.
        assert_eq!(hits[0].item.qualified_path, "x::serialize");
        assert_eq!(hits[0].score, 5 + 1);
    }
}
