use crate::storage_codec::{decode_bytes, encode_bytes};

/// Length in bytes of every persisted Merkle hash.
///
/// docbert uses BLAKE3 for all collection snapshot hashes.
pub const MERKLE_HASH_LEN: usize = blake3::OUT_LEN;

/// Fixed-size BLAKE3 hash used by Merkle snapshots.
pub type MerkleHash = [u8; MERKLE_HASH_LEN];

/// Whether a Merkle child entry points at a file leaf or a directory node.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub enum MerkleNodeKind {
    File,
    Directory,
}

/// A child stored under a directory node.
///
/// Children are persisted in a deterministic sorted order so later hashing and
/// diffing code can rely on stable structure, independent of filesystem walk
/// order.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct MerkleChildEntry {
    pub name: String,
    pub kind: MerkleNodeKind,
    pub hash: MerkleHash,
}

impl MerkleChildEntry {
    pub fn file(name: impl Into<String>, hash: MerkleHash) -> Self {
        Self {
            name: name.into(),
            kind: MerkleNodeKind::File,
            hash,
        }
    }

    pub fn directory(name: impl Into<String>, hash: MerkleHash) -> Self {
        Self {
            name: name.into(),
            kind: MerkleNodeKind::Directory,
            hash,
        }
    }
}

/// A persisted file leaf in a collection snapshot.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct MerkleFileLeaf {
    /// Path relative to the collection root, for example `notes/todo.md`.
    pub relative_path: String,
    /// BLAKE3 of the file contents.
    pub content_hash: MerkleHash,
    /// BLAKE3 of the leaf node that will participate in the Merkle tree.
    pub leaf_hash: MerkleHash,
}

impl MerkleFileLeaf {
    pub fn new(
        relative_path: impl Into<String>,
        content_hash: MerkleHash,
        leaf_hash: MerkleHash,
    ) -> Self {
        Self {
            relative_path: relative_path.into(),
            content_hash,
            leaf_hash,
        }
    }
}

/// A persisted directory node in a collection snapshot.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct MerkleDirectoryNode {
    /// Path relative to the collection root. The root directory uses `""`.
    pub relative_path: String,
    /// BLAKE3 of this directory node.
    pub node_hash: MerkleHash,
    /// Sorted child entries stored deterministically.
    pub children: Vec<MerkleChildEntry>,
}

impl MerkleDirectoryNode {
    pub fn new(
        relative_path: impl Into<String>,
        node_hash: MerkleHash,
        mut children: Vec<MerkleChildEntry>,
    ) -> Self {
        children.sort_by(|a, b| {
            a.name
                .cmp(&b.name)
                .then_with(|| a.kind.cmp(&b.kind))
                .then_with(|| a.hash.cmp(&b.hash))
        });

        Self {
            relative_path: relative_path.into(),
            node_hash,
            children,
        }
    }
}

/// One persisted Merkle snapshot for a collection.
///
/// The full snapshot is serialized as one blob per collection in `config.db`.
/// Files and directories are stored in deterministic path order so byte-level
/// persistence stays stable once hashing logic is added.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct CollectionMerkleSnapshot {
    pub collection: String,
    pub root_hash: MerkleHash,
    pub directories: Vec<MerkleDirectoryNode>,
    pub files: Vec<MerkleFileLeaf>,
}

impl CollectionMerkleSnapshot {
    pub fn new(
        collection: impl Into<String>,
        root_hash: MerkleHash,
        mut directories: Vec<MerkleDirectoryNode>,
        mut files: Vec<MerkleFileLeaf>,
    ) -> Self {
        directories.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
        files.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

        Self {
            collection: collection.into(),
            root_hash,
            directories,
            files,
        }
    }

    /// Serialize to a byte vector for persistence in the config database.
    pub fn serialize(&self) -> Vec<u8> {
        encode_bytes(self)
            .expect("CollectionMerkleSnapshot serialization should succeed")
    }

    /// Deserialize from bytes. Returns `None` for invalid payloads.
    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        decode_bytes(bytes).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: u8) -> MerkleHash {
        [byte; MERKLE_HASH_LEN]
    }

    #[test]
    fn collection_snapshot_roundtrips_through_rkyv() {
        let snapshot = CollectionMerkleSnapshot::new(
            "notes",
            hash(9),
            vec![
                MerkleDirectoryNode::new(
                    "nested",
                    hash(4),
                    vec![
                        MerkleChildEntry::file("b.md", hash(2)),
                        MerkleChildEntry::directory("deeper", hash(3)),
                    ],
                ),
                MerkleDirectoryNode::new(
                    "",
                    hash(8),
                    vec![
                        MerkleChildEntry::file("z.md", hash(7)),
                        MerkleChildEntry::directory("nested", hash(4)),
                    ],
                ),
            ],
            vec![
                MerkleFileLeaf::new("z.md", hash(5), hash(7)),
                MerkleFileLeaf::new("nested/b.md", hash(1), hash(2)),
            ],
        );

        let bytes = snapshot.serialize();
        let restored = CollectionMerkleSnapshot::deserialize(&bytes).unwrap();

        assert_eq!(restored, snapshot);
    }

    #[test]
    fn collection_snapshot_new_sorts_children_directories_and_files() {
        let snapshot = CollectionMerkleSnapshot::new(
            "notes",
            hash(9),
            vec![
                MerkleDirectoryNode::new(
                    "",
                    hash(8),
                    vec![
                        MerkleChildEntry::file("z.md", hash(7)),
                        MerkleChildEntry::directory("alpha", hash(4)),
                        MerkleChildEntry::file("a.md", hash(6)),
                    ],
                ),
                MerkleDirectoryNode::new(
                    "alpha",
                    hash(4),
                    vec![
                        MerkleChildEntry::file("c.md", hash(3)),
                        MerkleChildEntry::file("b.md", hash(2)),
                    ],
                ),
            ],
            vec![
                MerkleFileLeaf::new("z.md", hash(5), hash(7)),
                MerkleFileLeaf::new("alpha/b.md", hash(1), hash(2)),
                MerkleFileLeaf::new("a.md", hash(6), hash(6)),
            ],
        );

        assert_eq!(
            snapshot
                .directories
                .iter()
                .map(|dir| dir.relative_path.as_str())
                .collect::<Vec<_>>(),
            vec!["", "alpha"]
        );
        assert_eq!(
            snapshot
                .files
                .iter()
                .map(|file| file.relative_path.as_str())
                .collect::<Vec<_>>(),
            vec!["a.md", "alpha/b.md", "z.md"]
        );
        assert_eq!(
            snapshot.directories[0]
                .children
                .iter()
                .map(|child| (child.name.as_str(), child.kind))
                .collect::<Vec<_>>(),
            vec![
                ("a.md", MerkleNodeKind::File),
                ("alpha", MerkleNodeKind::Directory),
                ("z.md", MerkleNodeKind::File),
            ]
        );
        assert_eq!(
            snapshot.directories[1]
                .children
                .iter()
                .map(|child| child.name.as_str())
                .collect::<Vec<_>>(),
            vec!["b.md", "c.md"]
        );
    }

    #[test]
    fn collection_snapshot_invalid_bytes_return_none() {
        assert!(
            CollectionMerkleSnapshot::deserialize(b"not a snapshot").is_none()
        );
    }
}
