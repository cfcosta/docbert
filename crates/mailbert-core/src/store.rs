//! The local store: the messages, their raw bytes, the tags, and the
//! saved searches.
//!
//! The store holds two LMDB files, as in §1.1. `meta.db` holds one
//! entry for each message, the tags of each message, and the saved
//! searches. `blobs.db` holds the raw RFC822 bytes.
//!
//! Two files, and not one, because the two have very different sizes.
//! The metadata of 100000 messages is some tens of megabytes, and the
//! raw bytes of the same mailbox are some gigabytes. A search reads
//! only `meta.db`, so the pages that it touches stay warm.
//!
//! The key of every entry is the identity from §4.1, in its full hex
//! form. The identity does not come from a folder, a UID, or a path,
//! and this is what lets a tag survive a re-sync (§9).
//!
//! The store keeps the raw bytes exactly as the server gave them. It
//! never decrypts, and it never rewrites. `mailbert view` gives the
//! bytes of an encrypted message to gpg on demand (§5.4), and
//! `mailbert export` writes them to a maildir (§4.3). Both need the
//! bytes unchanged.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use heed::{
    Database,
    Env,
    EnvFlags,
    EnvOpenOptions,
    RoTxn,
    types::{Bytes, Str},
};
use rkyv::{
    Archive,
    Deserialize,
    Serialize,
    api::high::{HighDeserializer, HighSerializer, HighValidator, to_bytes},
    bytecheck::CheckBytes,
    rancor::Error as RecordError,
    ser::allocator::ArenaHandle,
    util::AlignedVec,
};
use unicode_normalization::UnicodeNormalization;

use crate::{
    error::{Error, Result},
    message::Message,
    message_id::{MessageId, PrefixMatch, resolve_prefix},
};

/// The file that holds the messages, the tags, and the searches.
pub const META_FILE: &str = "meta.db";

/// The file that holds the raw bytes of each message.
pub const BLOB_FILE: &str = "blobs.db";

/// The named databases inside `meta.db`.
const MESSAGES_DB: &str = "messages";
const TAGS_DB: &str = "tags";
const SAVED_DB: &str = "saved";

/// The named database inside `blobs.db`.
const RAW_DB: &str = "raw";

/// One gibibyte.
const GIB: usize = 1024 * 1024 * 1024;

/// The address space that `meta.db` may grow into. LMDB reserves the
/// space, and does not use it, so a generous number costs nothing.
const META_MAP_SIZE: usize = 8 * GIB;

/// The address space that `blobs.db` may grow into. The raw bytes are
/// the large part of a mailbox, so this is the larger number.
const BLOB_MAP_SIZE: usize = 64 * GIB;

/// How many named databases each file holds. LMDB needs the count when
/// the environment opens, so keep these in step with the names above.
const META_MAX_DBS: u32 = 3;
const BLOB_MAX_DBS: u32 = 1;

/// The characters that a tag must not hold, because the query language
/// gives each of them another meaning.
const TAG_BREAKS: [char; 3] = ['(', ')', '"'];

/// The named databases of the store.
struct Tables {
    /// The identity of a message, to the message.
    messages: Database<Str, Bytes>,

    /// The identity of a message, to the tags of that message.
    tags: Database<Str, Bytes>,

    /// The name of a saved search, to its query.
    saved: Database<Str, Str>,

    /// The identity of a message, to its raw bytes.
    raw: Database<Str, Bytes>,
}

/// The local store.
pub struct Store {
    meta: Env,
    blobs: Env,
    db: Tables,
}

/// Make a tag out of what the user typed.
///
/// A tag is lowercase, and it is composed (NFC), so two spellings of
/// one word give one tag. A tag must be writable in `tag:name` without
/// quotes, so it holds no space, no quote, and no parenthesis. A tag
/// must not start with `\`, because that is the namespace of the IMAP
/// system flags, and a tag must never look like `\seen`.
///
/// # Examples
///
/// ```
/// use mailbert_core::store::normalize_tag;
///
/// assert_eq!(normalize_tag("  TODO "), Some("todo".to_string()));
/// assert_eq!(normalize_tag(r"\seen"), None);
/// assert_eq!(normalize_tag("two words"), None);
/// ```
pub fn normalize_tag(raw: &str) -> Option<String> {
    let found: String = raw.trim().nfc().flat_map(char::to_lowercase).collect();

    if found.is_empty() || found.starts_with('\\') {
        return None;
    }

    let broken = found
        .chars()
        .any(|c| c.is_whitespace() || TAG_BREAKS.contains(&c));

    match broken {
        true => None,
        false => Some(found),
    }
}

/// Open one LMDB file, and make the directory that holds it.
///
/// The file is the environment, because `NO_SUB_DIR` is set. LMDB adds
/// a `<path>-lock` file beside it.
fn open_env(path: &Path, map_size: usize, max_dbs: u32) -> Result<Env> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    // LMDB copies the mode bits of the data file onto the lock file, so
    // the data file must be there before the environment opens.
    if !path.exists() {
        std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(path)?;
    }

    // SAFETY: LMDB maps the file into memory, so another writer that
    // does not go through LMDB makes the map unsound. mailbert owns
    // this file, and only this environment writes it.
    let env = unsafe {
        let mut options = EnvOpenOptions::new();
        options.map_size(map_size);
        options.max_dbs(max_dbs);
        options.flags(EnvFlags::NO_SUB_DIR);
        options.open(path)?
    };

    Ok(env)
}

/// Make the bytes of a record.
fn encode(
    value: &impl for<'a> Serialize<
        HighSerializer<AlignedVec, ArenaHandle<'a>, RecordError>,
    >,
) -> Result<Vec<u8>> {
    Ok(to_bytes::<RecordError>(value)?.into_vec())
}

/// Read a record back, and check its bytes first.
fn decode<T>(bytes: &[u8]) -> Result<T>
where
    T: Archive,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, RecordError>>
        + Deserialize<T, HighDeserializer<RecordError>>,
{
    Ok(rkyv::from_bytes::<T, RecordError>(bytes)?)
}

impl Store {
    /// Open the store in `dir`, and make the files that are not there.
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;

        let meta = open_env(&dir.join(META_FILE), META_MAP_SIZE, META_MAX_DBS)?;
        let blobs =
            open_env(&dir.join(BLOB_FILE), BLOB_MAP_SIZE, BLOB_MAX_DBS)?;

        let mut wtxn = meta.write_txn()?;
        let messages = meta.create_database(&mut wtxn, Some(MESSAGES_DB))?;
        let tags = meta.create_database(&mut wtxn, Some(TAGS_DB))?;
        let saved = meta.create_database(&mut wtxn, Some(SAVED_DB))?;
        wtxn.commit()?;

        let mut wtxn = blobs.write_txn()?;
        let raw = blobs.create_database(&mut wtxn, Some(RAW_DB))?;
        wtxn.commit()?;

        Ok(Self {
            meta,
            blobs,
            db: Tables {
                messages,
                tags,
                saved,
                raw,
            },
        })
    }

    /// Write a message and its raw bytes.
    ///
    /// A message that is already there absorbs the new reading, so a
    /// copy in another folder adds a location and does not replace the
    /// entry (§4.2). Returns the entry as it now stands.
    ///
    /// The raw bytes of an identity never change, because the identity
    /// comes from the bytes. The second write of one message keeps the
    /// first bytes, and does not copy them again.
    pub fn put(&self, message: &Message, raw: &[u8]) -> Result<Message> {
        let key = message.id.full_hex();

        // The bytes go first. A stop between the two commits then leaves
        // a blob that the next sync writes over, and never an entry that
        // has no bytes behind it.
        if !raw.is_empty() {
            let mut wtxn = self.blobs.write_txn()?;
            if self.db.raw.get(&wtxn, &key)?.is_none() {
                self.db.raw.put(&mut wtxn, &key, raw)?;
            }
            wtxn.commit()?;
        }

        let mut wtxn = self.meta.write_txn()?;
        let kept: Option<Message> =
            self.db.messages.get(&wtxn, &key)?.map(decode).transpose()?;

        let merged = match kept {
            Some(mut kept) => {
                kept.absorb(message.clone());
                kept
            }
            None => message.clone(),
        };

        self.db.messages.put(&mut wtxn, &key, &encode(&merged)?)?;
        wtxn.commit()?;

        Ok(merged)
    }

    /// Read one message.
    pub fn get(&self, id: &MessageId) -> Result<Option<Message>> {
        let rtxn = self.meta.read_txn()?;

        self.db
            .messages
            .get(&rtxn, &id.full_hex())?
            .map(decode)
            .transpose()
    }

    /// Read the raw bytes of one message, as the server gave them.
    ///
    /// `view` gives these bytes to gpg (§5.4), and `export` writes them
    /// to a maildir (§4.3). Neither works if one byte moves.
    pub fn raw(&self, id: &MessageId) -> Result<Option<Vec<u8>>> {
        let rtxn = self.blobs.read_txn()?;

        Ok(self.db.raw.get(&rtxn, &id.full_hex())?.map(<[u8]>::to_vec))
    }

    /// Remove a message, its raw bytes, and its tags.
    ///
    /// Returns `true` when the message was there.
    pub fn remove(&self, id: &MessageId) -> Result<bool> {
        let key = id.full_hex();

        let mut wtxn = self.meta.write_txn()?;
        let existed = self.db.messages.delete(&mut wtxn, &key)?;
        self.db.tags.delete(&mut wtxn, &key)?;
        wtxn.commit()?;

        let mut wtxn = self.blobs.write_txn()?;
        self.db.raw.delete(&mut wtxn, &key)?;
        wtxn.commit()?;

        Ok(existed)
    }

    /// How many messages the store holds.
    pub fn len(&self) -> Result<usize> {
        let rtxn = self.meta.read_txn()?;

        Ok(self.db.messages.len(&rtxn)? as usize)
    }

    /// Whether the store holds no message.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// The identity of every message, in hex order.
    pub fn ids(&self) -> Result<Vec<MessageId>> {
        let rtxn = self.meta.read_txn()?;
        let mut found = Vec::new();

        for entry in self.db.messages.iter(&rtxn)? {
            let (key, _) = entry?;
            if let Some(id) = MessageId::from_hex(key) {
                found.push(id);
            }
        }

        Ok(found)
    }

    /// Every message, in the hex order of its identity.
    pub fn all(&self) -> Result<Vec<Message>> {
        let rtxn = self.meta.read_txn()?;
        let mut found = Vec::new();

        for entry in self.db.messages.iter(&rtxn)? {
            let (_, bytes) = entry?;
            found.push(decode(bytes)?);
        }

        Ok(found)
    }

    /// Resolve a git-style prefix against the store (§4.1).
    ///
    /// The key of an entry is the hex identity, and hex keeps the order
    /// of the bytes, so this reads only the entries that can match.
    pub fn resolve(&self, prefix: &str) -> Result<PrefixMatch> {
        let prefix = prefix.trim().to_ascii_lowercase();

        if prefix.is_empty() || !prefix.chars().all(|c| c.is_ascii_hexdigit()) {
            return Ok(PrefixMatch::NotFound);
        }

        let rtxn = self.meta.read_txn()?;
        let mut hits = Vec::new();

        for entry in self.db.messages.prefix_iter(&rtxn, prefix.as_str())? {
            let (key, _) = entry?;
            if let Some(id) = MessageId::from_hex(key) {
                hits.push(id);
            }
        }

        Ok(resolve_prefix(&prefix, hits))
    }

    /// The tags of one message.
    pub fn tags_of(&self, id: &MessageId) -> Result<BTreeSet<String>> {
        let rtxn = self.meta.read_txn()?;

        Ok(self
            .db
            .tags
            .get(&rtxn, &id.full_hex())?
            .map(decode)
            .transpose()?
            .unwrap_or_default())
    }

    /// Add a tag to a message. Returns `true` when the tag is new.
    ///
    /// The tag goes on the identity of §4.1, and not on a folder or a
    /// UID, so a re-sync and a re-index both keep it (§9).
    pub fn tag(&self, id: &MessageId, raw: &str) -> Result<bool> {
        let tag = normalize_tag(raw)
            .ok_or_else(|| Error::InvalidTag(raw.to_string()))?;
        let key = id.full_hex();

        let mut wtxn = self.meta.write_txn()?;
        if self.db.messages.get(&wtxn, &key)?.is_none() {
            return Err(Error::UnknownMessage(id.short()));
        }

        let mut tags = self.read_tags(&wtxn, &key)?;
        let added = tags.insert(tag);

        if added {
            self.db.tags.put(&mut wtxn, &key, &encode(&tags)?)?;
            wtxn.commit()?;
        }

        Ok(added)
    }

    /// Remove a tag from a message. Returns `true` when it was there.
    pub fn untag(&self, id: &MessageId, raw: &str) -> Result<bool> {
        let tag = normalize_tag(raw)
            .ok_or_else(|| Error::InvalidTag(raw.to_string()))?;
        let key = id.full_hex();

        let mut wtxn = self.meta.write_txn()?;
        if self.db.messages.get(&wtxn, &key)?.is_none() {
            return Err(Error::UnknownMessage(id.short()));
        }

        let mut tags = self.read_tags(&wtxn, &key)?;
        let removed = tags.remove(&tag);

        if removed {
            // An empty entry would still count in `all_tags`, so it goes.
            match tags.is_empty() {
                true => {
                    self.db.tags.delete(&mut wtxn, &key)?;
                }
                false => {
                    self.db.tags.put(&mut wtxn, &key, &encode(&tags)?)?;
                }
            }
            wtxn.commit()?;
        }

        Ok(removed)
    }

    /// The messages that carry a tag, in hex order.
    ///
    /// This reads every tagged message, and that is the right cost: a
    /// `tag:` filter in a search is a fast-field predicate in Tantivy
    /// (§6.1), and this call serves `mailbert tags` and the indexer,
    /// which both walk the store anyway.
    pub fn tagged(&self, raw: &str) -> Result<Vec<MessageId>> {
        let Some(tag) = normalize_tag(raw) else {
            return Ok(Vec::new());
        };

        let rtxn = self.meta.read_txn()?;
        let mut found = Vec::new();

        for entry in self.db.tags.iter(&rtxn)? {
            let (key, bytes) = entry?;
            let tags: BTreeSet<String> = decode(bytes)?;

            if tags.contains(&tag)
                && let Some(id) = MessageId::from_hex(key)
            {
                found.push(id);
            }
        }

        Ok(found)
    }

    /// Each tag in the store, and how many messages carry it.
    pub fn all_tags(&self) -> Result<BTreeMap<String, usize>> {
        let rtxn = self.meta.read_txn()?;
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();

        for entry in self.db.tags.iter(&rtxn)? {
            let (_, bytes) = entry?;
            let tags: BTreeSet<String> = decode(bytes)?;

            for tag in tags {
                *counts.entry(tag).or_default() += 1;
            }
        }

        Ok(counts)
    }

    /// Give a name to a query (§9).
    ///
    /// The store does not read the query. A name that already exists
    /// takes the new query.
    pub fn save_search(&self, name: &str, query: &str) -> Result<()> {
        let name = normalize_tag(name)
            .ok_or_else(|| Error::InvalidSearchName(name.to_string()))?;
        let query = query.trim();

        if query.is_empty() {
            return Err(Error::EmptySearch(name));
        }

        let mut wtxn = self.meta.write_txn()?;
        self.db.saved.put(&mut wtxn, &name, query)?;
        wtxn.commit()?;

        Ok(())
    }

    /// The query behind a name.
    pub fn saved(&self, name: &str) -> Result<Option<String>> {
        let Some(name) = normalize_tag(name) else {
            return Ok(None);
        };

        let rtxn = self.meta.read_txn()?;

        Ok(self.db.saved.get(&rtxn, &name)?.map(str::to_string))
    }

    /// Forget a saved search. Returns `true` when it was there.
    pub fn forget_search(&self, name: &str) -> Result<bool> {
        let Some(name) = normalize_tag(name) else {
            return Ok(false);
        };

        let mut wtxn = self.meta.write_txn()?;
        let existed = self.db.saved.delete(&mut wtxn, &name)?;
        wtxn.commit()?;

        Ok(existed)
    }

    /// Every saved search, by name.
    pub fn searches(&self) -> Result<BTreeMap<String, String>> {
        let rtxn = self.meta.read_txn()?;
        let mut found = BTreeMap::new();

        for entry in self.db.saved.iter(&rtxn)? {
            let (name, query) = entry?;
            found.insert(name.to_string(), query.to_string());
        }

        Ok(found)
    }

    /// The tags of one key, inside a transaction that a caller holds.
    fn read_tags(
        &self,
        txn: &RoTxn<'_>,
        key: &str,
    ) -> Result<BTreeSet<String>> {
        Ok(self
            .db
            .tags
            .get(txn, key)?
            .map(decode)
            .transpose()?
            .unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_message_survives_the_store` | round-trip | The store is the only copy of a message. An entry that reads back wrong is mail that is gone. |
    //! | `prop_the_raw_bytes_never_change` | round-trip | `view` gives these bytes to gpg, and `export` writes them to a maildir. One changed byte breaks both. |
    //! | `prop_the_store_agrees_with_collate` | differential | The store and the in-memory join must give one answer, whatever order the sync writes in. |
    //! | `prop_a_second_write_changes_nothing` | algebraic | Every re-sync writes each message again, and must not move it. |
    //! | `prop_the_identity_resolves_to_the_message` | model-based | `mailbert get <id>` must find the message that the identity names. |
    //! | `prop_a_tag_goes_on_one_time` | algebraic | Two tags of one name would count one message twice. |
    //! | `prop_tag_and_untag_cancel` | algebraic | An undo must give back the state before it. |
    //! | `prop_tagged_agrees_with_the_tags_of` | differential | The tags of a message, and the messages behind a tag, are two views of one fact. |
    //! | `prop_normalize_tag_is_idempotent` | algebraic | The stored form must be a fixed point, or a tag can never be found again. |
    //! | `prop_a_saved_search_survives` | round-trip | A saved search that reads back wrong runs another query. |

    use hegel::{TestCase, generators as gs};
    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::{
        message::{Location, SEEN, collate},
        mime,
    };

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    const DAY: i64 = 86_400;

    fn open_at(dir: &TempDir) -> Store {
        Store::open(dir.path()).expect("a store")
    }

    fn location(account: &str, folder: &str, uid: u32) -> Location {
        Location {
            account: account.to_string(),
            folder: folder.to_string(),
            uid,
            uid_validity: 1,
            received: 100 * DAY,
        }
    }

    /// The raw bytes of a message that carries `<{key}@x.test>`.
    fn raw_bytes(key: &str, body: &str) -> Vec<u8> {
        format!(
            "From: Alice Smith <alice@example.test>\r\n\
             To: bob@example.test\r\n\
             Subject: Deposit\r\n\
             Date: Fri, 14 Aug 2026 09:30:00 +0000\r\n\
             Message-ID: <{key}@x.test>\r\n\
             \r\n\
             {body}\r\n"
        )
        .into_bytes()
    }

    fn message(key: &str, account: &str, folder: &str) -> Message {
        let raw = raw_bytes(key, "The deposit is due.");

        Message::new(
            mime::parse(&raw).expect("a message"),
            location(account, folder, 1),
            [SEEN],
        )
    }

    /// Write a message, and give back its identity.
    fn write(store: &Store, key: &str, folder: &str) -> MessageId {
        let found = message(key, "work", folder);
        let raw = raw_bytes(key, "The deposit is due.");
        store.put(&found, &raw).expect("a write");

        found.id
    }

    // -----------------------------------------------------------------
    // Tag names.
    // -----------------------------------------------------------------

    #[test]
    fn normalizes_the_case_and_the_space_of_a_tag() {
        assert_eq!(normalize_tag("  TODO  ").as_deref(), Some("todo"));
        assert_eq!(normalize_tag("Work").as_deref(), Some("work"));
    }

    #[test]
    fn refuses_a_tag_in_the_system_flag_namespace() {
        assert_eq!(normalize_tag(r"\seen"), None);
        assert_eq!(normalize_tag(r"\Flagged"), None);
    }

    #[test]
    fn refuses_a_tag_that_the_query_language_cannot_write() {
        assert_eq!(normalize_tag("two words"), None);
        assert_eq!(normalize_tag("a\"quote"), None);
        assert_eq!(normalize_tag("(group)"), None);
        assert_eq!(normalize_tag("   "), None);
        assert_eq!(normalize_tag(""), None);
    }

    #[test]
    fn keeps_a_tag_that_the_query_language_can_write() {
        assert_eq!(normalize_tag("to-do").as_deref(), Some("to-do"));
        assert_eq!(normalize_tag("home/rent").as_deref(), Some("home/rent"));
        assert_eq!(normalize_tag("$label1").as_deref(), Some("$label1"));
    }

    #[test]
    fn composes_the_accents_of_a_tag() {
        // The same word, decomposed and composed, must give one tag.
        let decomposed = "fami\u{0301}lia";
        let composed = "fam\u{00ed}lia";

        assert_eq!(normalize_tag(decomposed), normalize_tag(composed));
    }

    // -----------------------------------------------------------------
    // Messages, and their raw bytes.
    // -----------------------------------------------------------------

    #[test]
    fn a_new_store_is_empty() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        assert!(store.is_empty().expect("a count"));
        assert_eq!(store.len().expect("a count"), 0);
        assert!(store.ids().expect("the ids").is_empty());
    }

    #[test]
    fn makes_the_two_files_of_the_store() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        drop(store);

        assert!(dir.path().join(META_FILE).exists());
        assert!(dir.path().join(BLOB_FILE).exists());
    }

    #[test]
    fn reads_back_a_message_that_it_wrote() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let wrote = message("a", "work", "INBOX");

        store.put(&wrote, &raw_bytes("a", "x")).expect("a write");
        let read = store.get(&wrote.id).expect("a read").expect("a message");

        assert_eq!(read, wrote);
        assert_eq!(store.len().expect("a count"), 1);
    }

    #[test]
    fn reads_nothing_for_a_message_that_is_not_there() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let missing = message("gone", "work", "INBOX");

        assert_eq!(store.get(&missing.id).expect("a read"), None);
        assert_eq!(store.raw(&missing.id).expect("a read"), None);
    }

    #[test]
    fn keeps_the_raw_bytes_byte_for_byte() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let wrote = message("a", "work", "INBOX");
        let raw = raw_bytes("a", "The deposit is due.");

        store.put(&wrote, &raw).expect("a write");

        assert_eq!(store.raw(&wrote.id).expect("a read"), Some(raw));
    }

    #[test]
    fn writes_no_blob_when_the_caller_has_no_bytes() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let wrote = message("a", "work", "INBOX");

        store.put(&wrote, &[]).expect("a write");

        // No entry, and not an entry of no bytes: `view` and `export`
        // must be able to tell the two apart.
        assert!(store.get(&wrote.id).expect("a read").is_some());
        assert_eq!(store.raw(&wrote.id).expect("a read"), None);
    }

    #[test]
    fn keeps_no_plaintext_for_an_encrypted_message() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let raw = b"From: alice@example.test\r\n\
             Subject: Numbers\r\n\
             Message-ID: <secret@x.test>\r\n\
             \r\n\
             -----BEGIN PGP MESSAGE-----\r\n\
             \r\n\
             hQIMA0abcdef\r\n\
             -----END PGP MESSAGE-----\r\n"
            .to_vec();
        let wrote = Message::new(
            mime::parse(&raw).expect("a message"),
            location("work", "INBOX", 1),
            [SEEN],
        );

        store.put(&wrote, &raw).expect("a write");
        let read = store.get(&wrote.id).expect("a read").expect("a message");

        assert!(read.is_encrypted());
        assert_eq!(read.text, "");
        // The ciphertext is still there for gpg, and only there.
        assert_eq!(store.raw(&wrote.id).expect("a read"), Some(raw));
    }

    #[test]
    fn a_copy_in_another_folder_adds_a_location() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        let id = write(&store, "a", "INBOX");
        write(&store, "a", "Archive");

        let read = store.get(&id).expect("a read").expect("a message");

        assert_eq!(store.len().expect("a count"), 1);
        assert_eq!(read.folders(), vec!["Archive", "INBOX"]);
    }

    #[test]
    fn a_re_sync_does_not_change_the_entry() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        let id = write(&store, "a", "INBOX");
        let once = store.get(&id).expect("a read");
        write(&store, "a", "INBOX");
        let twice = store.get(&id).expect("a read");

        assert_eq!(once, twice);
        assert_eq!(store.len().expect("a count"), 1);
    }

    #[test]
    fn removes_the_message_the_bytes_and_the_tags() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = write(&store, "a", "INBOX");
        store.tag(&id, "todo").expect("a tag");

        assert!(store.remove(&id).expect("a removal"));

        assert_eq!(store.get(&id).expect("a read"), None);
        assert_eq!(store.raw(&id).expect("a read"), None);
        assert!(store.all_tags().expect("the tags").is_empty());
        assert!(store.is_empty().expect("a count"));
    }

    #[test]
    fn reports_a_removal_of_a_message_that_was_not_there() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let missing = message("gone", "work", "INBOX");

        assert!(!store.remove(&missing.id).expect("a removal"));
    }

    #[test]
    fn sees_everything_after_a_reopen() {
        let dir = tempdir().expect("a directory");

        let id = {
            let store = open_at(&dir);
            let id = write(&store, "a", "INBOX");
            store.tag(&id, "todo").expect("a tag");
            store.save_search("rent", "tag:todo").expect("a search");
            id
        };

        let store = open_at(&dir);

        assert!(store.get(&id).expect("a read").is_some());
        assert!(store.raw(&id).expect("a read").is_some());
        assert_eq!(store.tagged("todo").expect("a lookup"), vec![id]);
        assert_eq!(
            store.saved("rent").expect("a search").as_deref(),
            Some("tag:todo")
        );
    }

    // -----------------------------------------------------------------
    // The git-style prefixes of §4.1.
    // -----------------------------------------------------------------

    #[test]
    fn resolves_a_unique_prefix() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = write(&store, "a", "INBOX");
        write(&store, "b", "INBOX");

        assert_eq!(
            store.resolve(&id.short()).expect("a lookup"),
            PrefixMatch::Unique(id)
        );
    }

    #[test]
    fn ignores_the_case_of_a_prefix() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = write(&store, "a", "INBOX");

        assert_eq!(
            store.resolve(&id.short().to_uppercase()).expect("a lookup"),
            PrefixMatch::Unique(id)
        );
    }

    #[test]
    fn reports_a_prefix_that_two_messages_share() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        // 16 first characters and 100 keys: two must collide.
        let mut seen: BTreeMap<char, MessageId> = BTreeMap::new();
        let mut pair: Option<(char, MessageId, MessageId)> = None;
        for n in 0..100 {
            let key = format!("k{n}");
            let id = message(&key, "work", "INBOX").id;
            let first = id.full_hex().chars().next().expect("a character");
            if let Some(other) = seen.get(&first) {
                pair = Some((first, *other, id));
                break;
            }
            seen.insert(first, id);
        }

        let (first, one, two) = pair.expect("two keys that share a character");
        for n in 0..100 {
            let key = format!("k{n}");
            let id = message(&key, "work", "INBOX").id;
            if id == one || id == two {
                write(&store, &key, "INBOX");
            }
        }

        let mut wanted = vec![one, two];
        wanted.sort();

        assert_eq!(
            store.resolve(&first.to_string()).expect("a lookup"),
            PrefixMatch::Ambiguous(wanted)
        );
    }

    #[test]
    fn resolves_nothing_for_an_empty_or_unknown_prefix() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        write(&store, "a", "INBOX");

        assert_eq!(store.resolve("").expect("a lookup"), PrefixMatch::NotFound);
        assert_eq!(
            store.resolve("zznothex").expect("a lookup"),
            PrefixMatch::NotFound
        );
        assert_eq!(
            store.resolve("deadbeefdeadbeef").expect("a lookup"),
            PrefixMatch::NotFound
        );
    }

    // -----------------------------------------------------------------
    // Tags (§9).
    // -----------------------------------------------------------------

    #[test]
    fn tags_a_message_one_time() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = write(&store, "a", "INBOX");

        assert!(store.tag(&id, "TODO").expect("a tag"));
        assert!(!store.tag(&id, "todo").expect("a tag"));

        let tags = store.tags_of(&id).expect("the tags");

        assert_eq!(tags.len(), 1);
        assert!(tags.contains("todo"));
    }

    #[test]
    fn untags_a_message() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = write(&store, "a", "INBOX");
        store.tag(&id, "todo").expect("a tag");

        assert!(store.untag(&id, "TODO").expect("an untag"));
        assert!(!store.untag(&id, "todo").expect("an untag"));
        assert!(store.tags_of(&id).expect("the tags").is_empty());
    }

    #[test]
    fn refuses_to_tag_a_message_that_is_not_there() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let missing = message("gone", "work", "INBOX");

        let error = store.tag(&missing.id, "todo").expect_err("a refusal");

        assert!(matches!(error, Error::UnknownMessage(_)));
    }

    #[test]
    fn refuses_a_tag_that_is_not_writable() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = write(&store, "a", "INBOX");

        let error = store.tag(&id, r"\seen").expect_err("a refusal");

        assert!(matches!(error, Error::InvalidTag(_)));
    }

    #[test]
    fn refuses_to_untag_a_message_that_is_not_there() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let missing = message("gone", "work", "INBOX");

        let error = store.untag(&missing.id, "todo").expect_err("a refusal");

        assert!(matches!(error, Error::UnknownMessage(_)));
    }

    #[test]
    fn lists_the_messages_that_carry_a_tag() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let one = write(&store, "a", "INBOX");
        let two = write(&store, "b", "INBOX");
        write(&store, "c", "INBOX");

        store.tag(&one, "todo").expect("a tag");
        store.tag(&two, "todo").expect("a tag");

        let mut wanted = vec![one, two];
        wanted.sort();

        assert_eq!(store.tagged("todo").expect("a lookup"), wanted);
        assert!(store.tagged("rent").expect("a lookup").is_empty());
    }

    #[test]
    fn counts_each_tag() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let one = write(&store, "a", "INBOX");
        let two = write(&store, "b", "INBOX");

        store.tag(&one, "todo").expect("a tag");
        store.tag(&two, "todo").expect("a tag");
        store.tag(&two, "rent").expect("a tag");

        let counts = store.all_tags().expect("the tags");

        assert_eq!(counts.get("todo"), Some(&2));
        assert_eq!(counts.get("rent"), Some(&1));
        assert_eq!(counts.len(), 2);
    }

    // -----------------------------------------------------------------
    // Saved searches (§9).
    // -----------------------------------------------------------------

    #[test]
    fn saves_and_reads_a_search() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        store
            .save_search("work-todo", "account:work and tag:todo")
            .expect("a search");

        assert_eq!(
            store.saved("work-todo").expect("a read").as_deref(),
            Some("account:work and tag:todo")
        );
        assert_eq!(store.saved("nothing").expect("a read"), None);
    }

    #[test]
    fn replaces_a_search_that_has_the_same_name() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        store.save_search("rent", "tag:todo").expect("a search");
        store.save_search("RENT", "tag:rent").expect("a search");

        assert_eq!(store.searches().expect("the searches").len(), 1);
        assert_eq!(
            store.saved("rent").expect("a read").as_deref(),
            Some("tag:rent")
        );
    }

    #[test]
    fn forgets_a_search() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        store.save_search("rent", "tag:rent").expect("a search");

        assert!(store.forget_search("rent").expect("a removal"));
        assert!(!store.forget_search("rent").expect("a removal"));
        assert!(store.searches().expect("the searches").is_empty());
    }

    #[test]
    fn refuses_a_search_with_no_query() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        let error = store.save_search("rent", "   ").expect_err("a refusal");

        assert!(matches!(error, Error::EmptySearch(_)));
    }

    #[test]
    fn refuses_a_search_name_that_is_not_writable() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        let error = store
            .save_search("two words", "tag:x")
            .expect_err("a refusal");

        assert!(matches!(error, Error::InvalidSearchName(_)));
    }

    #[test]
    fn lists_the_searches_by_name() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        store.save_search("rent", "tag:rent").expect("a search");
        store.save_search("bills", "tag:bills").expect("a search");

        let names: Vec<String> = store
            .searches()
            .expect("the searches")
            .into_keys()
            .collect();

        assert_eq!(names, vec!["bills".to_string(), "rent".to_string()]);
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    fn a_key() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ])
    }

    fn a_folder() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "INBOX".to_string(),
            "Archive".to_string(),
            "Sent".to_string(),
        ])
    }

    fn an_account() -> impl gs::Generator<String> {
        gs::sampled_from(vec!["work".to_string(), "personal".to_string()])
    }

    fn a_tag() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "todo".to_string(),
            "rent".to_string(),
            "bills".to_string(),
        ])
    }

    #[hegel::composite]
    fn a_copy(tc: TestCase) -> (Message, Vec<u8>) {
        let key = tc.draw(a_key());
        let account = tc.draw(an_account());
        let folder = tc.draw(a_folder());
        let uid = tc.draw(gs::integers::<u32>().min_value(1).max_value(50));

        let raw = raw_bytes(&key, "The deposit is due.");
        let found = Message::new(
            mime::parse(&raw).expect("a message"),
            Location {
                account,
                folder,
                uid,
                uid_validity: 1,
                received: 100 * DAY,
            },
            [SEEN],
        );

        (found, raw)
    }

    fn a_mailbox() -> impl gs::Generator<Vec<(Message, Vec<u8>)>> {
        gs::vecs(a_copy()).min_size(0).max_size(8)
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_message_survives_the_store(tc: TestCase) {
        let (wrote, raw) = tc.draw(a_copy());
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        store.put(&wrote, &raw).expect("a write");

        assert_eq!(store.get(&wrote.id).expect("a read"), Some(wrote));
    }

    #[hegel::test(test_cases = 40)]
    fn prop_the_raw_bytes_never_change(tc: TestCase) {
        let (wrote, raw) = tc.draw(a_copy());
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        store.put(&wrote, &raw).expect("a write");

        assert_eq!(store.raw(&wrote.id).expect("a read"), Some(raw));
    }

    #[hegel::test(test_cases = 40)]
    fn prop_the_store_agrees_with_collate(tc: TestCase) {
        let copies: Vec<(Message, Vec<u8>)> = tc.draw(a_mailbox());
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        for (found, raw) in &copies {
            store.put(found, raw).expect("a write");
        }

        let wanted =
            collate(copies.into_iter().map(|(found, _)| found).collect());

        assert_eq!(store.all().expect("the messages"), wanted);
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_second_write_changes_nothing(tc: TestCase) {
        let copies: Vec<(Message, Vec<u8>)> = tc.draw(a_mailbox());
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        for (found, raw) in &copies {
            store.put(found, raw).expect("a write");
        }
        let once = store.all().expect("the messages");

        for (found, raw) in &copies {
            store.put(found, raw).expect("a write");
        }

        assert_eq!(store.all().expect("the messages"), once);
    }

    #[hegel::test(test_cases = 40)]
    fn prop_the_identity_resolves_to_the_message(tc: TestCase) {
        let copies: Vec<(Message, Vec<u8>)> = tc.draw(a_mailbox());
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        for (found, raw) in &copies {
            store.put(found, raw).expect("a write");
        }

        for (found, _) in &copies {
            assert_eq!(
                store.resolve(&found.id.full_hex()).expect("a lookup"),
                PrefixMatch::Unique(found.id)
            );
        }
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_tag_goes_on_one_time(tc: TestCase) {
        let (wrote, raw) = tc.draw(a_copy());
        let tags = tc.draw(gs::vecs(a_tag()).min_size(0).max_size(6));
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        store.put(&wrote, &raw).expect("a write");

        for tag in &tags {
            store.tag(&wrote.id, tag).expect("a tag");
        }

        let wanted: BTreeSet<String> = tags.into_iter().collect();

        assert_eq!(store.tags_of(&wrote.id).expect("the tags"), wanted);
    }

    #[hegel::test(test_cases = 40)]
    fn prop_tag_and_untag_cancel(tc: TestCase) {
        let (wrote, raw) = tc.draw(a_copy());
        let tags = tc.draw(gs::vecs(a_tag()).min_size(0).max_size(4));
        let extra = tc.draw(a_tag());
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        store.put(&wrote, &raw).expect("a write");
        for tag in &tags {
            store.tag(&wrote.id, tag).expect("a tag");
        }
        let before = store.tags_of(&wrote.id).expect("the tags");

        let added = store.tag(&wrote.id, &extra).expect("a tag");
        if added {
            store.untag(&wrote.id, &extra).expect("an untag");
        }

        assert_eq!(store.tags_of(&wrote.id).expect("the tags"), before);
    }

    #[hegel::test(test_cases = 40)]
    fn prop_tagged_agrees_with_the_tags_of(tc: TestCase) {
        let copies: Vec<(Message, Vec<u8>)> = tc.draw(a_mailbox());
        let tag = tc.draw(a_tag());
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        let mut wanted: Vec<MessageId> = Vec::new();
        for (found, raw) in &copies {
            store.put(found, raw).expect("a write");
            if found.id.numeric().is_multiple_of(2)
                && store.tag(&found.id, &tag).expect("a tag")
            {
                wanted.push(found.id);
            }
        }
        wanted.sort();

        assert_eq!(store.tagged(&tag).expect("a lookup"), wanted);
        for id in &wanted {
            assert!(store.tags_of(id).expect("the tags").contains(&tag));
        }
    }

    #[hegel::test(test_cases = 300)]
    fn prop_normalize_tag_is_idempotent(tc: TestCase) {
        let raw = tc.draw(gs::text().min_size(0).max_size(30));

        if let Some(once) = normalize_tag(&raw) {
            assert_eq!(normalize_tag(&once).as_deref(), Some(once.as_str()));
        }
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_saved_search_survives(tc: TestCase) {
        let name = tc.draw(a_tag());
        let query = tc.draw(gs::sampled_from(vec![
            "tag:todo".to_string(),
            "account:work and is:unread".to_string(),
            "\"the deposit\"".to_string(),
        ]));
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        store.save_search(&name, &query).expect("a search");

        assert_eq!(store.saved(&name).expect("a read"), Some(query.clone()));
        assert_eq!(
            store.searches().expect("the searches").get(&name),
            Some(&query)
        );
    }
}
