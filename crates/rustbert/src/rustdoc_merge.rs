//! Merge rustdoc JSON enrichment into the syn-extracted [`RustItem`]
//! list.
//!
//! docs.rs publishes the JSON output of `cargo +nightly rustdoc
//! --output-format json` for many crate builds. The format is the
//! `rustdoc-types` schema, version-tagged. Rather than depending on
//! the moving-target `rustdoc-types` crate, this module walks the
//! JSON as `serde_json::Value` and pulls out exactly the two pieces
//! we need:
//!
//! - **resolved doc strings** — rustdoc resolves intra-doc links
//!   (`` [`Foo`] `` → the actual full path) and de-Markdownifies a
//!   couple of conventions, so its `docs` field is strictly higher
//!   quality than the raw `///` comments syn extracts.
//! - **canonical paths** — rustdoc's `paths` table maps each item
//!   ID to the slice of segments that form its canonical qualified
//!   path. We use that to match items against our syn output.
//!
//! When a match is found, the syn item's `doc_markdown` is replaced
//! with the rustdoc version. Items that aren't in the JSON (because
//! they're macros, hidden, or syn over-extracted) keep their syn
//! docstring.

use std::collections::HashMap;

use serde_json::Value;

use crate::item::RustItem;

/// Merge resolved doc strings from a rustdoc JSON payload into
/// `items`. Returns the number of items whose `doc_markdown` was
/// updated.
///
/// Best-effort: malformed or schema-version-incompatible JSON
/// returns 0 without touching anything. The syn-extracted docs are
/// always present as a fallback.
pub fn merge_rustdoc_docs(items: &mut [RustItem], json_bytes: &[u8]) -> usize {
    let Ok(value) = serde_json::from_slice::<Value>(json_bytes) else {
        return 0;
    };

    let path_to_docs = build_path_to_docs(&value);
    if path_to_docs.is_empty() {
        return 0;
    }

    let mut merged = 0;
    for item in items {
        if let Some(docs) = path_to_docs.get(&item.qualified_path)
            && !docs.is_empty()
            && docs != &item.doc_markdown
        {
            item.doc_markdown = docs.clone();
            merged += 1;
        }
    }
    merged
}

/// Walk the rustdoc JSON `paths` + `index` tables to build a
/// `qualified_path → docs` map. Path segments are joined with `::`
/// to match `RustItem::qualified_path` exactly.
fn build_path_to_docs(value: &Value) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Some(paths) = value.get("paths").and_then(Value::as_object) else {
        return out;
    };
    let Some(index) = value.get("index").and_then(Value::as_object) else {
        return out;
    };

    for (id, path_info) in paths {
        let Some(path_array) = path_info.get("path").and_then(Value::as_array)
        else {
            continue;
        };
        let qualified: String = path_array
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join("::");
        if qualified.is_empty() {
            continue;
        }

        if let Some(item) = index.get(id)
            && let Some(docs) = item.get("docs").and_then(Value::as_str)
            && !docs.is_empty()
        {
            out.insert(qualified, docs.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::item::{RustItemKind, Visibility};

    fn item(qpath: &str, doc: &str) -> RustItem {
        RustItem {
            kind: RustItemKind::Fn,
            crate_name: "demo".to_string(),
            crate_version: semver::Version::new(0, 1, 0),
            module_path: vec![],
            name: Some("f".to_string()),
            qualified_path: qpath.to_string(),
            signature: "pub fn f()".to_string(),
            doc_markdown: doc.to_string(),
            body: "pub fn f() {}".to_string(),
            source_file: PathBuf::from("src/lib.rs"),
            byte_start: 0,
            byte_len: 0,
            line_start: 1,
            line_end: 1,
            visibility: Visibility::Public,
            attrs: vec![],
        }
    }

    fn rustdoc_json() -> Vec<u8> {
        serde_json::json!({
            "format_version": 30,
            "paths": {
                "0:1": { "path": ["demo", "greet"], "kind": "function" },
                "0:2": { "path": ["demo", "Holder"], "kind": "struct" },
                "0:3": { "path": ["demo", "ignored"], "kind": "function" }
            },
            "index": {
                "0:1": {
                    "name": "greet",
                    "docs": "Resolved docs with [demo::Holder] linked properly",
                    "inner": { "function": {} }
                },
                "0:2": {
                    "name": "Holder",
                    "docs": "A typed holder",
                    "inner": { "struct": {} }
                },
                "0:3": {
                    "name": "ignored",
                    "docs": "",
                    "inner": { "function": {} }
                }
            }
        })
        .to_string()
        .into_bytes()
    }

    #[test]
    fn merges_resolved_docs_into_matching_items() {
        let mut items = vec![
            item("demo::greet", "raw /// docs"),
            item("demo::Holder", "raw /// docs"),
            item("demo::nope", "raw /// docs"),
        ];
        let merged = merge_rustdoc_docs(&mut items, &rustdoc_json());
        assert_eq!(merged, 2);
        assert!(items[0].doc_markdown.contains("Resolved docs"));
        assert_eq!(items[1].doc_markdown, "A typed holder");
        // Item without a JSON match keeps its syn doc.
        assert_eq!(items[2].doc_markdown, "raw /// docs");
    }

    #[test]
    fn empty_rustdoc_docs_dont_clobber_syn_docs() {
        let mut items = vec![item("demo::ignored", "syn doc preserved")];
        merge_rustdoc_docs(&mut items, &rustdoc_json());
        assert_eq!(items[0].doc_markdown, "syn doc preserved");
    }

    #[test]
    fn malformed_json_is_a_no_op() {
        let mut items = vec![item("demo::greet", "syn doc")];
        let merged = merge_rustdoc_docs(&mut items, b"not json");
        assert_eq!(merged, 0);
        assert_eq!(items[0].doc_markdown, "syn doc");
    }

    #[test]
    fn json_with_no_paths_table_is_a_no_op() {
        let mut items = vec![item("demo::greet", "syn doc")];
        let merged =
            merge_rustdoc_docs(&mut items, br#"{"format_version":30}"#);
        assert_eq!(merged, 0);
        assert_eq!(items[0].doc_markdown, "syn doc");
    }
}
