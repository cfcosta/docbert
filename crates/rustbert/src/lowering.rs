//! Lower a [`RustItem`] into a [`docbert_core::preparation::SearchDocument`].
//!
//! Lowering rules (from `docs/rustbert.md` §5.2):
//!
//! - `did` ← `DocumentId::new(synthetic_collection_name, qualified_path)`
//!   — deterministic 48-bit blake3 hash, fits docbert-core's chunk-family
//!   bit budget.
//! - `relative_path` ← `<source_file>#L<start>-L<end>` so editor "open at
//!   line" workflows hit the right spot.
//! - `title` ← qualified path.
//! - `searchable_body` ← `<kind> <qualified_path>\n\n<signature>\n\n<doc>`.
//! - `metadata` ← JSON blob with kind / crate / version / module path /
//!   visibility / attrs / source_file / line_span.

use docbert_core::{DocumentId, preparation::SearchDocument};
use serde_json::json;

use crate::{collection::SyntheticCollection, item::RustItem};

/// Lower a [`RustItem`] under the given synthetic collection.
pub fn lower(
    collection: &SyntheticCollection,
    item: &RustItem,
) -> SearchDocument {
    let collection_name = collection.to_string();
    let did = DocumentId::new(&collection_name, &item.qualified_path);
    let relative_path = format!(
        "{}#L{}-L{}",
        item.source_file.display(),
        item.line_start,
        item.line_end,
    );
    SearchDocument {
        did,
        relative_path,
        title: item.qualified_path.clone(),
        searchable_body: build_searchable_body(item),
        raw_content: None,
        metadata: Some(build_metadata(item)),
        mtime: 0,
    }
}

fn build_searchable_body(item: &RustItem) -> String {
    let mut body = String::new();
    body.push_str(item.kind.as_str());
    body.push(' ');
    body.push_str(&item.qualified_path);
    body.push_str("\n\n");
    body.push_str(&item.signature);
    if !item.doc_markdown.is_empty() {
        body.push_str("\n\n");
        body.push_str(&item.doc_markdown);
    }
    body
}

fn build_metadata(item: &RustItem) -> serde_json::Value {
    json!({
        "kind": item.kind.as_str(),
        "crate": item.crate_name,
        "version": item.crate_version.to_string(),
        "module_path": item.module_path,
        "visibility": item.visibility.as_str(),
        "attrs": item.attrs,
        "source_file": item.source_file.display().to_string(),
        "line_span": [item.line_start, item.line_end],
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::item::{RustItemKind, Visibility};

    fn collection(name: &str, version: (u64, u64, u64)) -> SyntheticCollection {
        SyntheticCollection {
            crate_name: name.to_string(),
            version: semver::Version::new(version.0, version.1, version.2),
        }
    }

    fn sample_item() -> RustItem {
        RustItem {
            kind: RustItemKind::Fn,
            crate_name: "serde".to_string(),
            crate_version: semver::Version::new(1, 0, 219),
            module_path: vec!["ser".to_string()],
            name: Some("serialize_struct".to_string()),
            qualified_path: "serde::ser::serialize_struct".to_string(),
            signature: "pub fn serialize_struct(&self) -> Result<()>"
                .to_string(),
            doc_markdown: "Serialize a struct.".to_string(),
            source_file: PathBuf::from("src/ser/mod.rs"),
            byte_start: 0,
            byte_len: 0,
            line_start: 42,
            line_end: 78,
            visibility: Visibility::Public,
            attrs: vec!["#[deprecated]".to_string()],
        }
    }

    #[test]
    fn relative_path_is_file_with_line_anchor() {
        let coll = collection("serde", (1, 0, 219));
        let doc = lower(&coll, &sample_item());
        assert_eq!(doc.relative_path, "src/ser/mod.rs#L42-L78");
    }

    #[test]
    fn title_is_qualified_path() {
        let coll = collection("serde", (1, 0, 219));
        let doc = lower(&coll, &sample_item());
        assert_eq!(doc.title, "serde::ser::serialize_struct");
    }

    #[test]
    fn searchable_body_starts_with_kind_and_path() {
        let coll = collection("serde", (1, 0, 219));
        let doc = lower(&coll, &sample_item());
        assert!(
            doc.searchable_body
                .starts_with("fn serde::ser::serialize_struct")
        );
    }

    #[test]
    fn searchable_body_includes_signature_and_doc() {
        let coll = collection("serde", (1, 0, 219));
        let doc = lower(&coll, &sample_item());
        assert!(doc.searchable_body.contains("pub fn serialize_struct"));
        assert!(doc.searchable_body.contains("Serialize a struct."));
    }

    #[test]
    fn metadata_carries_kind_crate_version() {
        let coll = collection("serde", (1, 0, 219));
        let doc = lower(&coll, &sample_item());
        let meta = doc.metadata.unwrap();
        assert_eq!(meta["kind"], "fn");
        assert_eq!(meta["crate"], "serde");
        assert_eq!(meta["version"], "1.0.219");
    }

    #[test]
    fn document_id_is_deterministic() {
        let coll = collection("serde", (1, 0, 219));
        let item = sample_item();
        let a = lower(&coll, &item);
        let b = lower(&coll, &item);
        assert_eq!(a.did.numeric, b.did.numeric);
    }

    #[test]
    fn document_id_fits_in_48_bits() {
        let coll = collection("serde", (1, 0, 219));
        let doc = lower(&coll, &sample_item());
        assert_eq!(doc.did.numeric >> 48, 0);
    }
}
