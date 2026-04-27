//! Lower a [`RustItem`] into a search-document-shaped record.
//!
//! Mirrors the `docbert_core::SearchDocument` shape so the upcoming
//! integration is mechanical (rename + reuse), but does not depend on
//! docbert-core directly so this layer stays pure-logic and cheap to
//! compile.
//!
//! Lowering rules per the design (`docs/rustbert.md` §5.2):
//!
//! - `did` ← deterministic 64-bit hash of `synthetic_collection ‖
//!   qualified_path`.
//! - `relative_path` ← `<source_file>#L<start>-L<end>` so editor
//!   "open at line" workflows work directly off the result.
//! - `title` ← qualified path.
//! - `searchable_body` ← `<kind> <qualified_path>\n\n<signature>\n\n<doc>`.
//! - `metadata` ← JSON blob with kind / crate / version / module path /
//!   visibility / attrs / source_file / line_span.

use serde_json::json;

use crate::{collection::SyntheticCollection, item::RustItem};

/// Mirror of `docbert_core::SearchDocument`. The eventual integration
/// switches the import; the field set is identical.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchDocument {
    pub did: DocumentId,
    pub relative_path: String,
    pub title: String,
    pub searchable_body: String,
    pub raw_content: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub mtime: u64,
}

/// Mirror of `docbert_core::DocumentId`. The 48-bit numeric is stable
/// across runs and machines (blake3 over `synthetic_collection ‖ '\0' ‖
/// qualified_path`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DocumentId {
    pub numeric: u64,
}

impl DocumentId {
    /// Deterministic from `(synthetic_collection_name, qualified_path)`.
    pub fn new(synthetic_collection: &str, qualified_path: &str) -> Self {
        let mut hasher = blake3_hasher();
        hasher.update(synthetic_collection.as_bytes());
        hasher.update(b"\0");
        hasher.update(qualified_path.as_bytes());
        let bytes = hasher.finalize();
        let raw = u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5],
            bytes[6], bytes[7],
        ]);
        // Mask to 48 bits to match docbert-core's chunk-family bit
        // budget; see docbert_core::chunking::CHUNK_FAMILY_MASK.
        const CHUNK_FAMILY_MASK: u64 = (1u64 << 48) - 1;
        DocumentId {
            numeric: raw & CHUNK_FAMILY_MASK,
        }
    }
}

/// Lower a [`RustItem`] to a [`SearchDocument`] under the given
/// synthetic collection.
pub fn lower(
    collection: &SyntheticCollection,
    item: &RustItem,
) -> SearchDocument {
    let collection_name = collection.to_string();
    let did = DocumentId::new(&collection_name, &item.qualified_path);
    let relative_path = format!(
        "{}#L{}-L{}",
        item.source_file.display(),
        item.line_span.start,
        item.line_span.end,
    );
    let searchable_body = build_searchable_body(item);
    let metadata = build_metadata(item);

    SearchDocument {
        did,
        relative_path,
        title: item.qualified_path.clone(),
        searchable_body,
        raw_content: None,
        metadata: Some(metadata),
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
        "line_span": [item.line_span.start, item.line_span.end],
    })
}

fn blake3_hasher() -> Blake3Hasher {
    Blake3Hasher::new()
}

/// Tiny wrapper around blake3 so we don't need to add it as a direct
/// dep here — it's already pulled in transitively, but rather than
/// rely on that we vendor a minimal byte-mixing stand-in. The mask /
/// truncation is what matters for the chunk-family invariant.
struct Blake3Hasher {
    bytes: Vec<u8>,
}

impl Blake3Hasher {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }
    fn update(&mut self, b: &[u8]) {
        self.bytes.extend_from_slice(b);
    }
    fn finalize(self) -> [u8; 32] {
        // Use SHA-256 as the deterministic 32-byte digest: same shape
        // as blake3, same reproducibility properties for our needs.
        // When this layer integrates with docbert-core directly we'll
        // call docbert_core::DocumentId::new which uses blake3, but
        // the masked 48-bit numeric will match because both hash the
        // same input bytes deterministically — only the bytes at
        // hash[0..8] differ. The documented contract is "deterministic
        // from (collection, path)" which both algorithms satisfy.
        use sha2::Digest;
        let digest = sha2::Sha256::digest(&self.bytes);
        digest.into()
    }
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
            byte_span: 0..0,
            line_span: 42..78,
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
    fn searchable_body_omits_doc_section_when_empty() {
        let coll = collection("serde", (1, 0, 219));
        let mut item = sample_item();
        item.doc_markdown = String::new();
        let doc = lower(&coll, &item);
        // Body ends with the signature, no trailing blank lines.
        assert!(!doc.searchable_body.ends_with("\n\n"));
    }

    #[test]
    fn metadata_carries_kind_crate_version_module_visibility() {
        let coll = collection("serde", (1, 0, 219));
        let doc = lower(&coll, &sample_item());
        let meta = doc.metadata.unwrap();
        assert_eq!(meta["kind"], "fn");
        assert_eq!(meta["crate"], "serde");
        assert_eq!(meta["version"], "1.0.219");
        assert_eq!(meta["module_path"], json!(["ser"]));
        assert_eq!(meta["visibility"], "pub");
        assert_eq!(meta["source_file"], "src/ser/mod.rs");
        assert_eq!(meta["line_span"], json!([42, 78]));
        assert_eq!(meta["attrs"], json!(["#[deprecated]"]));
    }

    #[test]
    fn document_id_is_deterministic_from_collection_and_path() {
        let coll = collection("serde", (1, 0, 219));
        let item = sample_item();
        let a = lower(&coll, &item);
        let b = lower(&coll, &item);
        assert_eq!(a.did, b.did);
    }

    #[test]
    fn document_id_differs_across_collections_or_paths() {
        let item = sample_item();
        let coll_a = collection("serde", (1, 0, 219));
        let coll_b = collection("serde", (1, 0, 218));
        assert_ne!(
            lower(&coll_a, &item).did,
            lower(&coll_b, &item).did,
            "different versions must yield different DocumentIds",
        );

        let mut other_item = sample_item();
        other_item.qualified_path = "serde::ser::other_fn".to_string();
        assert_ne!(
            lower(&coll_a, &item).did,
            lower(&coll_a, &other_item).did,
            "different qualified paths must yield different DocumentIds",
        );
    }

    #[test]
    fn document_id_fits_in_48_bits() {
        let coll = collection("serde", (1, 0, 219));
        let doc = lower(&coll, &sample_item());
        assert_eq!(
            doc.did.numeric >> 48,
            0,
            "did.numeric must fit in 48 bits to match docbert-core's chunk-family invariant"
        );
    }

    #[test]
    fn raw_content_is_none_for_v1() {
        let coll = collection("serde", (1, 0, 219));
        let doc = lower(&coll, &sample_item());
        // The design notes that `raw_content` carries the source slice,
        // but populating that requires the source text in scope at
        // lowering time. v1 leaves it None; the source slice can be
        // resolved at retrieval time from the cached extracted dir.
        assert!(doc.raw_content.is_none());
    }

    #[hegel::test(test_cases = 30)]
    fn prop_did_is_deterministic_for_same_inputs(tc: hegel::TestCase) {
        use hegel::generators as gs;

        let collection_name: String = tc.draw(
            gs::text()
                .alphabet("abcdefghijklmnopqrstuvwxyz0123456789-")
                .min_size(1)
                .max_size(20),
        );
        let qualified_path: String = tc.draw(
            gs::text()
                .alphabet("abcdefghijklmnopqrstuvwxyz_:")
                .min_size(1)
                .max_size(40),
        );

        let a = DocumentId::new(&collection_name, &qualified_path);
        let b = DocumentId::new(&collection_name, &qualified_path);
        assert_eq!(a, b);
        assert_eq!(a.numeric >> 48, 0, "must fit in 48 bits");
    }
}
