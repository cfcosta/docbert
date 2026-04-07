use std::{collections::BTreeMap, path::Path};

use crate::{
    Result,
    storage_codec::{decode_bytes, encode_bytes},
    walker::DiscoveredFile,
};

/// Length in bytes of every persisted Merkle hash.
///
/// docbert uses BLAKE3 for all collection snapshot hashes.
pub const MERKLE_HASH_LEN: usize = blake3::OUT_LEN;

/// Fixed-size BLAKE3 hash used by Merkle snapshots.
pub type MerkleHash = [u8; MERKLE_HASH_LEN];

const DOMAIN_FILE_CONTENT: &[u8] = b"docbert.merkle.file-content.v1";
const DOMAIN_FILE_LEAF: &[u8] = b"docbert.merkle.file-leaf.v1";
const DOMAIN_DIRECTORY_NODE: &[u8] = b"docbert.merkle.directory-node.v1";
const DOMAIN_COLLECTION_ROOT: &[u8] = b"docbert.merkle.collection-root.v1";

fn update_len(hasher: &mut blake3::Hasher, len: usize) {
    hasher.update(&(len as u64).to_le_bytes());
}

fn update_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    update_len(hasher, bytes.len());
    hasher.update(bytes);
}

fn update_str(hasher: &mut blake3::Hasher, value: &str) {
    update_bytes(hasher, value.as_bytes());
}

fn finish_hash(hasher: blake3::Hasher) -> MerkleHash {
    *hasher.finalize().as_bytes()
}

fn sort_child_entries(children: &mut [MerkleChildEntry]) {
    children.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| a.kind.cmp(&b.kind))
            .then_with(|| a.hash.cmp(&b.hash))
    });
}

/// Hash raw file contents with the BLAKE3 file-content domain.
pub fn hash_file_content(content: &[u8]) -> MerkleHash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DOMAIN_FILE_CONTENT);
    update_bytes(&mut hasher, content);
    finish_hash(hasher)
}

/// Hash a file leaf from its relative path and content hash.
pub fn hash_file_leaf(
    relative_path: &str,
    content_hash: &MerkleHash,
) -> MerkleHash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DOMAIN_FILE_LEAF);
    update_str(&mut hasher, relative_path);
    hasher.update(content_hash);
    finish_hash(hasher)
}

/// Hash a directory node from its relative path and child entries.
pub fn hash_directory_node(
    relative_path: &str,
    children: &[MerkleChildEntry],
) -> MerkleHash {
    let mut sorted_children = children.to_vec();
    sort_child_entries(&mut sorted_children);

    let mut hasher = blake3::Hasher::new();
    hasher.update(DOMAIN_DIRECTORY_NODE);
    update_str(&mut hasher, relative_path);
    update_len(&mut hasher, sorted_children.len());
    for child in &sorted_children {
        update_str(&mut hasher, &child.name);
        hasher.update(&[child.kind.discriminator()]);
        hasher.update(&child.hash);
    }
    finish_hash(hasher)
}

/// Hash a collection root from the root directory child entries.
pub fn hash_collection_root(children: &[MerkleChildEntry]) -> MerkleHash {
    let mut sorted_children = children.to_vec();
    sort_child_entries(&mut sorted_children);

    let mut hasher = blake3::Hasher::new();
    hasher.update(DOMAIN_COLLECTION_ROOT);
    update_len(&mut hasher, sorted_children.len());
    for child in &sorted_children {
        update_str(&mut hasher, &child.name);
        hasher.update(&[child.kind.discriminator()]);
        hasher.update(&child.hash);
    }
    finish_hash(hasher)
}

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

impl MerkleNodeKind {
    fn discriminator(self) -> u8 {
        match self {
            Self::File => 0,
            Self::Directory => 1,
        }
    }
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
        sort_child_entries(&mut children);

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

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn parent_directory_path(relative_path: &Path) -> String {
    relative_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(path_to_string)
        .unwrap_or_default()
}

fn child_name(relative_path: &Path) -> String {
    relative_path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| path_to_string(relative_path))
}

fn directory_depth(relative_path: &str) -> usize {
    if relative_path.is_empty() {
        0
    } else {
        Path::new(relative_path).components().count()
    }
}

fn ensure_directory_ancestors(
    directory_children: &mut BTreeMap<String, Vec<MerkleChildEntry>>,
    relative_path: &Path,
) {
    directory_children.entry(String::new()).or_default();

    let mut current = relative_path.parent();
    while let Some(directory) = current {
        if directory.as_os_str().is_empty() {
            break;
        }
        directory_children
            .entry(path_to_string(directory))
            .or_default();
        current = directory.parent();
    }
}

/// Build a full persisted Merkle snapshot for one collection from discovered files.
///
/// This reads each discovered file, hashes its contents, creates file leaves,
/// builds directory nodes bottom-up, and computes the collection root hash.
pub fn build_collection_snapshot(
    collection: &str,
    discovered: &[DiscoveredFile],
) -> Result<CollectionMerkleSnapshot> {
    let mut directory_children: BTreeMap<String, Vec<MerkleChildEntry>> =
        BTreeMap::new();
    directory_children.entry(String::new()).or_default();

    let mut files = Vec::with_capacity(discovered.len());

    for file in discovered {
        let relative_path = path_to_string(&file.relative_path);
        let content = std::fs::read(&file.absolute_path)?;
        let content_hash = hash_file_content(&content);
        let leaf_hash = hash_file_leaf(&relative_path, &content_hash);
        files.push(MerkleFileLeaf::new(
            relative_path.clone(),
            content_hash,
            leaf_hash,
        ));

        ensure_directory_ancestors(
            &mut directory_children,
            &file.relative_path,
        );
        let parent = parent_directory_path(&file.relative_path);
        directory_children.entry(parent).or_default().push(
            MerkleChildEntry::file(child_name(&file.relative_path), leaf_hash),
        );
    }

    let mut directory_paths: Vec<String> =
        directory_children.keys().cloned().collect();
    directory_paths.sort_by(|a, b| {
        directory_depth(b)
            .cmp(&directory_depth(a))
            .then_with(|| a.cmp(b))
    });

    let mut directories = Vec::with_capacity(directory_paths.len());

    for relative_path in directory_paths {
        let children = directory_children
            .get(&relative_path)
            .cloned()
            .unwrap_or_default();
        let node_hash = hash_directory_node(&relative_path, &children);
        directories.push(MerkleDirectoryNode::new(
            relative_path.clone(),
            node_hash,
            children,
        ));

        if !relative_path.is_empty() {
            let parent = parent_directory_path(Path::new(&relative_path));
            directory_children.entry(parent).or_default().push(
                MerkleChildEntry::directory(
                    child_name(Path::new(&relative_path)),
                    node_hash,
                ),
            );
        }
    }

    let root_hash = hash_collection_root(
        directory_children.get("").map(Vec::as_slice).unwrap_or(&[]),
    );

    Ok(CollectionMerkleSnapshot::new(
        collection,
        root_hash,
        directories,
        files,
    ))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn hash(byte: u8) -> MerkleHash {
        [byte; MERKLE_HASH_LEN]
    }

    fn discovered_file(
        root: &tempfile::TempDir,
        relative_path: &str,
        mtime: u64,
    ) -> DiscoveredFile {
        DiscoveredFile {
            relative_path: PathBuf::from(relative_path),
            absolute_path: root.path().join(relative_path),
            mtime,
        }
    }

    #[test]
    fn file_hashes_are_stable_for_same_content() {
        let content = b"# hello\nworld\n";
        let first_content_hash = hash_file_content(content);
        let second_content_hash = hash_file_content(content);
        let first_leaf_hash =
            hash_file_leaf("notes/hello.md", &first_content_hash);
        let second_leaf_hash =
            hash_file_leaf("notes/hello.md", &second_content_hash);

        assert_eq!(first_content_hash, second_content_hash);
        assert_eq!(first_leaf_hash, second_leaf_hash);
    }

    #[test]
    fn file_hashes_change_when_content_changes() {
        let original_content_hash = hash_file_content(b"hello");
        let updated_content_hash = hash_file_content(b"hello!");
        let original_leaf_hash =
            hash_file_leaf("notes/hello.md", &original_content_hash);
        let updated_leaf_hash =
            hash_file_leaf("notes/hello.md", &updated_content_hash);

        assert_ne!(original_content_hash, updated_content_hash);
        assert_ne!(original_leaf_hash, updated_leaf_hash);
    }

    #[test]
    fn directory_and_root_hashes_ignore_child_input_order() {
        let children_in_order = vec![
            MerkleChildEntry::file("a.md", hash(1)),
            MerkleChildEntry::directory("nested", hash(2)),
            MerkleChildEntry::file("z.md", hash(3)),
        ];
        let children_out_of_order = vec![
            MerkleChildEntry::file("z.md", hash(3)),
            MerkleChildEntry::file("a.md", hash(1)),
            MerkleChildEntry::directory("nested", hash(2)),
        ];

        assert_eq!(
            hash_directory_node("notes", &children_in_order),
            hash_directory_node("notes", &children_out_of_order)
        );
        assert_eq!(
            hash_collection_root(&children_in_order),
            hash_collection_root(&children_out_of_order)
        );
    }

    #[test]
    fn file_directory_and_root_domains_remain_distinct() {
        let content_hash = hash_file_content(b"same bytes");
        let child = MerkleChildEntry::file("same", content_hash);

        assert_ne!(
            hash_file_leaf("same", &content_hash),
            hash_directory_node("same", std::slice::from_ref(&child))
        );
        assert_ne!(
            hash_directory_node("", std::slice::from_ref(&child)),
            hash_collection_root(&[child])
        );
    }

    #[test]
    fn build_collection_snapshot_handles_empty_collection() {
        let snapshot = build_collection_snapshot("notes", &[]).unwrap();

        assert_eq!(snapshot.collection, "notes");
        assert!(snapshot.files.is_empty());
        assert_eq!(snapshot.directories.len(), 1);
        assert_eq!(snapshot.directories[0].relative_path, "");
        assert!(snapshot.directories[0].children.is_empty());
        assert_eq!(
            snapshot.directories[0].node_hash,
            hash_directory_node("", &[])
        );
        assert_eq!(snapshot.root_hash, hash_collection_root(&[]));
    }

    #[test]
    fn build_collection_snapshot_includes_nested_directories_and_files() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("nested/deeper")).unwrap();
        std::fs::write(tmp.path().join("top.md"), "top-level").unwrap();
        std::fs::write(tmp.path().join("nested/child.md"), "child").unwrap();
        std::fs::write(tmp.path().join("nested/deeper/leaf.txt"), "leaf")
            .unwrap();

        let discovered = vec![
            discovered_file(&tmp, "top.md", 1),
            discovered_file(&tmp, "nested/child.md", 2),
            discovered_file(&tmp, "nested/deeper/leaf.txt", 3),
        ];
        let snapshot = build_collection_snapshot("notes", &discovered).unwrap();

        assert_eq!(
            snapshot
                .directories
                .iter()
                .map(|dir| dir.relative_path.as_str())
                .collect::<Vec<_>>(),
            vec!["", "nested", "nested/deeper"]
        );
        assert_eq!(
            snapshot
                .files
                .iter()
                .map(|file| file.relative_path.as_str())
                .collect::<Vec<_>>(),
            vec!["nested/child.md", "nested/deeper/leaf.txt", "top.md"]
        );

        let root = &snapshot.directories[0];
        let nested = &snapshot.directories[1];
        let deeper = &snapshot.directories[2];
        assert_eq!(
            root.children
                .iter()
                .map(|child| (child.name.as_str(), child.kind))
                .collect::<Vec<_>>(),
            vec![
                ("nested", MerkleNodeKind::Directory),
                ("top.md", MerkleNodeKind::File),
            ]
        );
        assert_eq!(
            nested
                .children
                .iter()
                .map(|child| (child.name.as_str(), child.kind))
                .collect::<Vec<_>>(),
            vec![
                ("child.md", MerkleNodeKind::File),
                ("deeper", MerkleNodeKind::Directory),
            ]
        );
        assert_eq!(
            deeper
                .children
                .iter()
                .map(|child| (child.name.as_str(), child.kind))
                .collect::<Vec<_>>(),
            vec![("leaf.txt", MerkleNodeKind::File)]
        );
        assert_eq!(snapshot.root_hash, hash_collection_root(&root.children));
    }

    #[test]
    fn build_collection_snapshot_ignores_mtime_when_content_matches() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("note.md"), "same content").unwrap();

        let original = [discovered_file(&tmp, "note.md", 1)];
        let updated = [discovered_file(&tmp, "note.md", 999_999)];

        let original_snapshot =
            build_collection_snapshot("notes", &original).unwrap();
        let updated_snapshot =
            build_collection_snapshot("notes", &updated).unwrap();

        assert_eq!(original_snapshot, updated_snapshot);
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
