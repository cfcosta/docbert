//! Direct cache lookups: `get` (exact qualified path) and `list`
//! (filtered enumeration). These are not search — search lives in
//! [`crate::indexer::Indexer::search`] and routes through the full
//! docbert-core hybrid stack (BM25 + ColBERT/PLAID).
//!
//! Both operations pull from the JSON cache because that's the only
//! place that holds the full `RustItem` (signature, doc markdown,
//! body, attrs, source span). Tantivy stores a subset shaped for
//! retrieval; reading items back from there would lose fields the
//! CLI / MCP need to display.

use crate::item::{RustItem, RustItemKind, normalize_qualified_path};

#[derive(Debug, Clone, Default)]
pub struct ListOptions {
    pub kind: Option<RustItemKind>,
    pub module_prefix: Option<String>,
    pub limit: Option<usize>,
}

/// Find one item by exact qualified path.
///
/// The query's first segment is normalised so callers can type the
/// crates.io name with dashes (`candle-core::tensor`) and still hit
/// the stored path, which always uses the underscored module name.
pub fn get<'a>(
    items: &'a [RustItem],
    qualified_path: &str,
) -> Option<&'a RustItem> {
    let needle = normalize_qualified_path(qualified_path);
    items.iter().find(|i| i.qualified_path == needle)
}

/// Filtered alphabetical listing of items in a cached crate.
pub fn list<'a>(
    items: &'a [RustItem],
    options: &ListOptions,
) -> Vec<&'a RustItem> {
    let mut out: Vec<&RustItem> =
        items.iter().filter(|i| matches(i, options)).collect();
    out.sort_by(|a, b| a.qualified_path.cmp(&b.qualified_path));
    if let Some(limit) = options.limit {
        out.truncate(limit);
    }
    out
}

/// True when `item` passes every filter in `options`.
///
/// `module_prefix` is normalised the same way as [`get`]'s query so
/// callers can write either `candle-core::tensor` or `candle_core::tensor`.
pub fn matches(item: &RustItem, options: &ListOptions) -> bool {
    if let Some(k) = options.kind
        && item.kind != k
    {
        return false;
    }
    if let Some(prefix) = &options.module_prefix {
        let prefix = normalize_qualified_path(prefix);
        if !item.qualified_path.starts_with(&prefix) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::item::Visibility;

    fn item(kind: RustItemKind, qpath: &str) -> RustItem {
        RustItem {
            kind,
            crate_name: "x".to_string(),
            crate_version: semver::Version::new(0, 1, 0),
            module_path: vec![],
            name: Some("f".to_string()),
            qualified_path: qpath.to_string(),
            signature: "pub fn f()".to_string(),
            doc_markdown: String::new(),
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
    fn get_finds_exact_qualified_path() {
        let items = vec![
            item(RustItemKind::Fn, "x::a"),
            item(RustItemKind::Fn, "x::b"),
        ];
        let got = get(&items, "x::b").unwrap();
        assert_eq!(got.qualified_path, "x::b");
        assert!(get(&items, "x::missing").is_none());
    }

    #[test]
    fn get_accepts_dashed_crate_name_in_query() {
        // Stored path uses the canonical Rust name (`candle_core::…`);
        // a query that types the crates.io spelling (`candle-core::…`)
        // must still resolve.
        let items =
            vec![item(RustItemKind::Struct, "candle_core::tensor::Tensor")];
        let got = get(&items, "candle-core::tensor::Tensor").unwrap();
        assert_eq!(got.qualified_path, "candle_core::tensor::Tensor");
    }

    #[test]
    fn list_module_prefix_accepts_dashed_crate_name() {
        let items = vec![
            item(RustItemKind::Fn, "candle_core::tensor::a"),
            item(RustItemKind::Fn, "candle_core::tensor::b"),
            item(RustItemKind::Fn, "other::c"),
        ];
        let opts = ListOptions {
            module_prefix: Some("candle-core::tensor".to_string()),
            ..Default::default()
        };
        let listed = list(&items, &opts);
        assert_eq!(listed.len(), 2);
        assert!(
            listed
                .iter()
                .all(|i| i.qualified_path.starts_with("candle_core::tensor"))
        );
    }

    #[test]
    fn list_sorts_alphabetically() {
        let items = vec![
            item(RustItemKind::Fn, "x::z"),
            item(RustItemKind::Fn, "x::a"),
            item(RustItemKind::Fn, "x::m"),
        ];
        let listed = list(&items, &ListOptions::default());
        assert_eq!(listed[0].qualified_path, "x::a");
        assert_eq!(listed[2].qualified_path, "x::z");
    }

    #[test]
    fn list_kind_filter() {
        let items = vec![
            item(RustItemKind::Fn, "x::a"),
            item(RustItemKind::Struct, "x::b"),
            item(RustItemKind::Fn, "x::c"),
        ];
        let opts = ListOptions {
            kind: Some(RustItemKind::Fn),
            ..Default::default()
        };
        let listed = list(&items, &opts);
        assert_eq!(listed.len(), 2);
        assert!(listed.iter().all(|i| i.kind == RustItemKind::Fn));
    }

    #[test]
    fn list_module_prefix_filter() {
        let items = vec![
            item(RustItemKind::Fn, "x::ser::a"),
            item(RustItemKind::Fn, "x::de::a"),
            item(RustItemKind::Fn, "x::ser::b"),
        ];
        let opts = ListOptions {
            module_prefix: Some("x::ser".to_string()),
            ..Default::default()
        };
        let listed = list(&items, &opts);
        assert_eq!(listed.len(), 2);
        assert!(
            listed
                .iter()
                .all(|i| i.qualified_path.starts_with("x::ser"))
        );
    }

    #[test]
    fn list_limit() {
        let items: Vec<RustItem> = (0..20)
            .map(|i| item(RustItemKind::Fn, &format!("x::a_{i:02}")))
            .collect();
        let opts = ListOptions {
            limit: Some(5),
            ..Default::default()
        };
        let listed = list(&items, &opts);
        assert_eq!(listed.len(), 5);
    }
}
