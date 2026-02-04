use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

/// A stable document identifier derived from (collection_name, relative_path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentId {
    /// The numeric ID used as the key in redb tables.
    pub numeric: u64,
    /// The short hex string for human display (e.g. "a1b2c3").
    pub short: String,
}

impl DocumentId {
    /// Generate a stable document ID from collection name and relative path.
    pub fn new(collection: &str, relative_path: &str) -> Self {
        let numeric = Self::hash_pair(collection, relative_path);
        let short = Self::short_hex(numeric, 6);
        Self { numeric, short }
    }

    fn hash_pair(collection: &str, relative_path: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        collection.hash(&mut hasher);
        relative_path.hash(&mut hasher);
        hasher.finish()
    }

    fn short_hex(value: u64, len: usize) -> String {
        let full = format!("{value:016x}");
        full[..len].to_string()
    }

    /// Extend the short ID to avoid collisions.
    /// Returns a new DocumentId with a longer short hex string.
    pub fn extend_short(&self, len: usize) -> Self {
        let len = len.clamp(6, 16);
        Self {
            numeric: self.numeric,
            short: Self::short_hex(self.numeric, len),
        }
    }
}

impl std::fmt::Display for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.short)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let a = DocumentId::new("notes", "hello.md");
        let b = DocumentId::new("notes", "hello.md");
        assert_eq!(a, b);
    }

    #[test]
    fn different_inputs_differ() {
        let a = DocumentId::new("notes", "hello.md");
        let b = DocumentId::new("notes", "world.md");
        assert_ne!(a.numeric, b.numeric);
    }

    #[test]
    fn short_id_is_six_chars() {
        let id = DocumentId::new("notes", "hello.md");
        assert_eq!(id.short.len(), 6);
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
        assert_eq!(too_big.short.len(), 16);
    }
}
