/// Stable document ID built from a collection name and relative path.
///
/// Uses a blake3 hash for collision resistance. The full hex digest is
/// stored in Tantivy and used for upsert/delete operations. A short
/// prefix (minimum 6 hex chars) is used for human display, extended
/// when needed to disambiguate — just like git commit SHAs.
///
/// # Examples
///
/// ```
/// use docbert_core::DocumentId;
///
/// let id = DocumentId::new("notes", "hello.md");
/// assert_eq!(id.short.len(), 6);
/// assert!(id.numeric > 0);
/// assert!(id.to_string().starts_with('#'));
///
/// // Full hex is 64 chars (blake3 = 32 bytes)
/// assert_eq!(id.full_hex().len(), 64);
/// assert!(id.full_hex().starts_with(&id.short));
///
/// // Same inputs always produce the same ID
/// let id2 = DocumentId::new("notes", "hello.md");
/// assert_eq!(id, id2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentId {
    /// The numeric ID used as the key in redb tables and embedding DB.
    /// Derived from the first 8 bytes of the blake3 hash.
    pub numeric: u64,
    /// The short hex string for human display (first 6 hex chars).
    pub short: String,
    /// The full blake3 hash (32 bytes).
    hash: [u8; 32],
}

/// Strip a leading `#` from a user-facing document reference.
pub fn strip_document_ref_prefix(reference: &str) -> &str {
    reference.strip_prefix('#').unwrap_or(reference)
}

/// Format a short document id as a user-facing `#`-prefixed reference.
pub fn format_document_ref(short_id: &str) -> String {
    format!("#{}", strip_document_ref_prefix(short_id))
}

/// Minimum display prefix length (hex chars).
const MIN_SHORT_LEN: usize = 6;

impl DocumentId {
    /// Build a stable document ID from a collection name and relative path.
    ///
    /// # Examples
    ///
    /// ```
    /// use docbert_core::DocumentId;
    ///
    /// let id = DocumentId::new("notes", "hello.md");
    /// assert_eq!(id.short.len(), 6);
    /// assert!(id.numeric > 0);
    ///
    /// // Deterministic
    /// assert_eq!(id, DocumentId::new("notes", "hello.md"));
    ///
    /// // Different inputs -> different ID
    /// assert_ne!(id, DocumentId::new("notes", "other.md"));
    /// ```
    pub fn new(collection: &str, relative_path: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(collection.as_bytes());
        hasher.update(b"\0");
        hasher.update(relative_path.as_bytes());
        let hash: [u8; 32] = *hasher.finalize().as_bytes();

        // Mask to 48 bits so the numeric id fits in the chunk-family
        // bit space. The chunking scheme (see `crate::chunking`)
        // packs a per-chunk index into the top 16 bits via XOR, and
        // the search path collapses chunks back to bases by masking
        // those bits off with `document_family_key`. For that
        // round-trip to succeed, a base numeric id must itself have
        // zero top 16 bits — otherwise every search result would be
        // silently dropped by `metadata.contains_key(&family_key)`.
        //
        // 48 bits still gives 2^48 ≈ 2.8×10^14 distinct ids, which is
        // comfortably collision-resistant for any realistic corpus.
        // The `short` / `full_hex` displays continue to carry the
        // full blake3 digest, so user-facing disambiguation is
        // unaffected.
        let numeric_raw = u64::from_be_bytes([
            hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6],
            hash[7],
        ]);
        let numeric = numeric_raw & crate::chunking::CHUNK_FAMILY_MASK;
        let full = hex_encode(&hash);
        let short = full[..MIN_SHORT_LEN].to_string();

        Self {
            numeric,
            short,
            hash,
        }
    }

    /// The full hex digest (64 chars).
    ///
    /// Used as the Tantivy `doc_id` field for collision-safe upsert and
    /// delete operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use docbert_core::DocumentId;
    ///
    /// let id = DocumentId::new("notes", "hello.md");
    /// assert_eq!(id.full_hex().len(), 64);
    /// assert!(id.full_hex().starts_with(&id.short));
    /// ```
    pub fn full_hex(&self) -> String {
        hex_encode(&self.hash)
    }

    /// Extend the short display ID to `len` hex chars.
    ///
    /// The length is clamped to `[6, 64]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use docbert_core::DocumentId;
    ///
    /// let id = DocumentId::new("notes", "hello.md");
    /// let extended = id.extend_short(10);
    /// assert_eq!(extended.short.len(), 10);
    /// assert!(extended.short.starts_with(&id.short));
    /// assert_eq!(extended.numeric, id.numeric);
    /// ```
    pub fn extend_short(&self, len: usize) -> Self {
        let len = len.clamp(MIN_SHORT_LEN, 64);
        let full = hex_encode(&self.hash);
        Self {
            numeric: self.numeric,
            short: full[..len].to_string(),
            hash: self.hash,
        }
    }
}

impl std::fmt::Display for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format_document_ref(&self.short))
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            use std::fmt::Write;
            let _ = write!(s, "{b:02x}");
            s
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let a = DocumentId::new("notes", "hello.md");
        let b = DocumentId::new("notes", "hello.md");
        assert_eq!(a, b);
        assert_eq!(a.full_hex(), b.full_hex());
    }

    #[test]
    fn different_inputs_differ() {
        let a = DocumentId::new("notes", "hello.md");
        let b = DocumentId::new("notes", "world.md");
        assert_ne!(a.numeric, b.numeric);
        assert_ne!(a.full_hex(), b.full_hex());
    }

    #[test]
    fn short_id_is_six_chars() {
        let id = DocumentId::new("notes", "hello.md");
        assert_eq!(id.short.len(), 6);
    }

    #[test]
    fn full_hex_is_64_chars() {
        let id = DocumentId::new("notes", "hello.md");
        assert_eq!(id.full_hex().len(), 64);
        assert!(id.full_hex().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn full_hex_starts_with_short() {
        let id = DocumentId::new("notes", "hello.md");
        assert!(id.full_hex().starts_with(&id.short));
    }

    #[test]
    fn display_has_hash_prefix() {
        let id = DocumentId::new("notes", "hello.md");
        let s = id.to_string();
        assert!(s.starts_with('#'));
        assert_eq!(s.len(), 7); // # + 6 hex chars
    }

    #[test]
    fn extend_short_grows() {
        let id = DocumentId::new("notes", "hello.md");
        let extended = id.extend_short(10);
        assert_eq!(extended.short.len(), 10);
        assert!(extended.short.starts_with(&id.short));
        assert_eq!(extended.numeric, id.numeric);
    }

    #[test]
    fn extend_short_clamps() {
        let id = DocumentId::new("notes", "hello.md");
        let too_small = id.extend_short(2);
        assert_eq!(too_small.short.len(), 6);
        let too_big = id.extend_short(100);
        assert_eq!(too_big.short.len(), 64);
    }

    #[test]
    fn doc_id_helpers_normalize_prefixed_and_unprefixed_refs_equally() {
        assert_eq!(strip_document_ref_prefix("abc123"), "abc123");
        assert_eq!(strip_document_ref_prefix("#abc123"), "abc123");
        assert_eq!(format_document_ref("abc123"), "#abc123");
        assert_eq!(format_document_ref("#abc123"), "#abc123");
    }

    #[hegel::test(test_cases = 200)]
    fn prop_full_hex_always_starts_with_short(tc: hegel::TestCase) {
        use hegel::generators as gs;

        let collection: String = tc.draw(gs::text().min_size(1).max_size(20));
        let path: String = tc.draw(gs::text().min_size(1).max_size(40));

        let id = DocumentId::new(&collection, &path);
        assert!(
            id.full_hex().starts_with(&id.short),
            "full_hex {} does not start with short {}",
            id.full_hex(),
            id.short
        );
        assert_eq!(id.full_hex().len(), 64);
        assert_eq!(id.short.len(), 6);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_different_inputs_produce_different_full_hex(tc: hegel::TestCase) {
        use hegel::generators as gs;

        let c1: String = tc.draw(gs::text().min_size(1).max_size(20));
        let p1: String = tc.draw(gs::text().min_size(1).max_size(40));
        let c2: String = tc.draw(gs::text().min_size(1).max_size(20));
        let p2: String = tc.draw(gs::text().min_size(1).max_size(40));

        tc.assume(c1 != c2 || p1 != p2);

        let a = DocumentId::new(&c1, &p1);
        let b = DocumentId::new(&c2, &p2);
        // With blake3, full hex collision is astronomically unlikely
        assert_ne!(
            a.full_hex(),
            b.full_hex(),
            "collision between ({c1}, {p1}) and ({c2}, {p2})"
        );
    }

    #[test]
    fn numeric_fits_in_chunk_family_bit_space() {
        // The chunking scheme reserves the top 16 bits of a u64 for a
        // chunk index: chunk_doc_id(base, k) == base ^ (k << 48), and
        // document_family_key(chunk_id) == chunk_id & 0x0000_FFFF_FFFF_FFFF.
        // For the metadata lookup at search time — which collapses
        // chunk ids back to base ids via document_family_key — to
        // round-trip, `DocumentId::new().numeric` must itself fit in
        // 48 bits. Any path whose blake3 prefix has a set bit in the
        // top 16 would otherwise cause the search to silently drop
        // every hit ("No results found") because
        // `metadata.contains_key(&family_key(chunk_id))` would compare
        // a masked key against an unmasked one.
        //
        // Exercise a mix of realistic collection/path pairs; real
        // blake3 prefixes almost always have top-16 bits set, so this
        // test relies on the masking contract rather than specific
        // values.
        for (coll, path) in [
            ("blog", "plaid.md"),
            ("diary", "2024-01-01.md"),
            ("mugraph", "intro.md"),
            ("openclaw", "README.md"),
            ("resources", "paper.pdf"),
            ("lobster", "engine.txt"),
            ("docbert", "architecture.md"),
        ] {
            let id = DocumentId::new(coll, path);
            assert_eq!(
                id.numeric >> 48,
                0,
                "DocumentId::new({coll:?}, {path:?}).numeric = {:#018x} \
                 does not fit in 48 bits; top 16 bits = {:#06x} \
                 would collide with chunk-index encoding",
                id.numeric,
                id.numeric >> 48,
            );
        }
    }

    #[test]
    fn numeric_survives_document_family_key_unchanged() {
        // Round-trip test: family_key(did.numeric) must equal
        // did.numeric, otherwise the search path's
        // `metadata.contains_key(&family_key(chunk_id))` lookup drops
        // every result.
        use crate::chunking::document_family_key;
        for (coll, path) in [
            ("blog", "plaid.md"),
            ("diary", "2024-01-01.md"),
            ("mugraph", "intro.md"),
        ] {
            let id = DocumentId::new(coll, path);
            assert_eq!(
                document_family_key(id.numeric),
                id.numeric,
                "family_key round-trip failed for ({coll:?}, {path:?})",
            );
        }
    }
}
