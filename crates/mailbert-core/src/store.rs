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
    RwTxn,
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
const STATE_DB: &str = "state";
const PLACES_DB: &str = "places";
const EMBEDS_DB: &str = "embeds";
const OWNERS_DB: &str = "owners";

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
const META_MAX_DBS: u32 = 7;
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

    /// An account and a folder, to the state of the last sync.
    state: Database<Str, Bytes>,

    /// An account, a folder, and a UID, to the message that sits there.
    places: Database<Str, Str>,

    /// The identity of a message, to what the last embedding pass gave.
    embeds: Database<Str, Bytes>,

    /// The key of a passage, to the message that owns it.
    owners: Database<Str, Str>,

    /// The identity of a message, to its raw bytes.
    raw: Database<Str, Bytes>,
}

/// The local store.
pub struct Store {
    meta: Env,
    blobs: Env,
    db: Tables,
}

/// The mark that a sync leaves on one folder of one account (§3.3).
///
/// The record holds what the next sync needs to ask the server for the
/// messages that it does not have. `pending` holds the UIDs that a
/// sync asked for and never received, in the text of a UID set, such
/// as `"1:20,44"`.
#[derive(
    Debug, Clone, PartialEq, Eq, Default, Archive, Serialize, Deserialize,
)]
pub struct SyncState {
    /// The UIDVALIDITY of the folder. A number that changed makes every
    /// UID of that folder worthless.
    pub uid_validity: u32,

    /// The UID that the folder gives to the next message.
    pub uid_next: u32,

    /// The HIGHESTMODSEQ that CONDSTORE gave (§3.3).
    pub highest_mod_seq: u64,

    /// The UIDs that the sync owes, in the text of a UID set.
    pub pending: String,

    /// When the last sync marked this folder, in seconds of Unix time.
    ///
    /// A zero says that no sync marked it yet. §10.4 shows this, so a
    /// reader knows whether a search can find recent mail.
    pub synced_at: i64,
}

/// What one embedding pass gave one message. (§6.2)
///
/// The digest is the fingerprint of the passages under the model that
/// made them. A second pass that reads the same fingerprint keeps the
/// embedding that it has, and a mailbox of 100000 messages then costs
/// one message for one new message.
#[derive(
    Debug, Clone, PartialEq, Eq, Default, Archive, Serialize, Deserialize,
)]
pub struct Embedded {
    /// The fingerprint from `embed::digest`.
    pub digest: [u8; 32],

    /// The key of each passage, in the order that the message cut.
    pub keys: Vec<u64>,
}

/// The character that holds an account name apart from a folder name.
///
/// An IMAP folder name never holds a control character, so this makes
/// a key that no name can copy.
const STATE_BREAK: char = '\u{1}';

/// The key of one folder of one account.
fn state_key(account: &str, folder: &str) -> String {
    format!("{account}{STATE_BREAK}{folder}")
}

/// The key of one copy of a message (§4.2).
///
/// The UID is written wide, so the keys of one folder keep the order of
/// the numbers behind them.
fn place_key(account: &str, folder: &str, uid: u32) -> String {
    format!("{account}{STATE_BREAK}{folder}{STATE_BREAK}{uid:010}")
}

/// The key of one passage in the table of owners.
///
/// The number is written wide and in hexadecimal, so every key has one
/// spelling and one length.
fn owner_key(key: u64) -> String {
    format!("{key:016x}")
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
///
/// LMDB gives the bytes at the offset that the page holds them at, and
/// rkyv reads a record in place, so the bytes go into an aligned buffer
/// first. Without the copy, a key of the wrong length moves the value
/// off an 8-byte boundary and the read fails.
fn decode<T>(bytes: &[u8]) -> Result<T>
where
    T: Archive,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, RecordError>>
        + Deserialize<T, HighDeserializer<RecordError>>,
{
    let mut aligned = AlignedVec::<16>::with_capacity(bytes.len());
    aligned.extend_from_slice(bytes);

    Ok(rkyv::from_bytes::<T, RecordError>(&aligned)?)
}

/// Drop the embedding record of one message, and the owner of each of
/// its passages. Returns the keys that the record held.
///
/// The caller holds the transaction, because a message that goes away
/// must lose its passages in the same write.
fn clear_embedding(
    wtxn: &mut RwTxn<'_>,
    db: &Tables,
    key: &str,
) -> Result<Vec<u64>> {
    let held: Option<Embedded> =
        db.embeds.get(wtxn, key)?.map(decode).transpose()?;
    let keys = held.map(|one| one.keys).unwrap_or_default();

    for one in &keys {
        db.owners.delete(wtxn, &owner_key(*one))?;
    }
    db.embeds.delete(wtxn, key)?;

    Ok(keys)
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
        let state = meta.create_database(&mut wtxn, Some(STATE_DB))?;
        let places = meta.create_database(&mut wtxn, Some(PLACES_DB))?;
        let embeds = meta.create_database(&mut wtxn, Some(EMBEDS_DB))?;
        let owners = meta.create_database(&mut wtxn, Some(OWNERS_DB))?;
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
                state,
                places,
                embeds,
                owners,
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

        // A folder that gave the message a new UID leaves the old key
        // behind, so the places of the entry go before the merge does.
        if let Some(kept) = &kept {
            for at in &kept.locations {
                let old = place_key(&at.account, &at.folder, at.uid);
                self.db.places.delete(&mut wtxn, &old)?;
            }
        }

        let merged = match kept {
            Some(mut kept) => {
                kept.absorb(message.clone());
                kept
            }
            None => message.clone(),
        };

        for at in &merged.locations {
            let place = place_key(&at.account, &at.folder, at.uid);
            self.db.places.put(&mut wtxn, &place, &key)?;
        }

        self.db.messages.put(&mut wtxn, &key, &encode(&merged)?)?;
        wtxn.commit()?;

        Ok(merged)
    }

    /// Write a whole batch of messages. (§4.2)
    ///
    /// This is [`put`] for many messages, and it makes the same store.
    /// The difference is the cost. `put` commits two transactions for
    /// each message, and LMDB lets one writer into a database at a
    /// time, so a fetch of 500 messages waits for 1000 commits. This
    /// takes one transaction of each database for the whole batch.
    ///
    /// The answers come back in the order of `batch`. A batch that
    /// holds one message twice gives two answers, and the second one
    /// absorbed the first, exactly as two calls of `put` would.
    ///
    /// [`put`]: Self::put
    ///
    /// # Errors
    ///
    /// The function fails if either database refuses the write. A
    /// failure leaves the batch out, because neither transaction
    /// commits.
    pub fn put_all(
        &self,
        batch: &[(Message, Vec<u8>)],
    ) -> Result<Vec<Message>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        // The bytes go first, as in `put`. A stop between the two
        // commits leaves blobs that the next sync writes over, and
        // never an entry that has no bytes behind it.
        let mut wtxn = self.blobs.write_txn()?;
        for (message, raw) in batch {
            let key = message.id.full_hex();

            if !raw.is_empty() && self.db.raw.get(&wtxn, &key)?.is_none() {
                self.db.raw.put(&mut wtxn, &key, raw)?;
            }
        }
        wtxn.commit()?;

        let mut wtxn = self.meta.write_txn()?;
        let mut kept = Vec::with_capacity(batch.len());

        for (message, _) in batch {
            let key = message.id.full_hex();

            // The read sees what this transaction already wrote, so a
            // message that comes twice in one batch absorbs itself.
            let held: Option<Message> =
                self.db.messages.get(&wtxn, &key)?.map(decode).transpose()?;

            // A folder that gave the message a new UID leaves the old
            // key behind, so the places go before the merge does.
            if let Some(held) = &held {
                for at in &held.locations {
                    let old = place_key(&at.account, &at.folder, at.uid);
                    self.db.places.delete(&mut wtxn, &old)?;
                }
            }

            let merged = match held {
                Some(mut held) => {
                    held.absorb(message.clone());
                    held
                }
                None => message.clone(),
            };

            for at in &merged.locations {
                let place = place_key(&at.account, &at.folder, at.uid);
                self.db.places.put(&mut wtxn, &place, &key)?;
            }

            self.db.messages.put(&mut wtxn, &key, &encode(&merged)?)?;
            kept.push(merged);
        }

        wtxn.commit()?;

        Ok(kept)
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
        let kept: Option<Message> =
            self.db.messages.get(&wtxn, &key)?.map(decode).transpose()?;

        if let Some(kept) = &kept {
            for at in &kept.locations {
                let place = place_key(&at.account, &at.folder, at.uid);
                self.db.places.delete(&mut wtxn, &place)?;
            }
        }

        let existed = self.db.messages.delete(&mut wtxn, &key)?;
        self.db.tags.delete(&mut wtxn, &key)?;

        // The embedding record stays. Only the pass of §6.2 knows
        // the embedding database and the PLAID index, so only the pass
        // can drop the passages of this message from them. It sees a
        // record that names a message that is gone, and it cleans both.
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

impl Store {
    /// Write the state of one folder of one account (§3.3).
    pub fn mark(
        &self,
        account: &str,
        folder: &str,
        state: &SyncState,
    ) -> Result<()> {
        let mut wtxn = self.meta.write_txn()?;
        self.db.state.put(
            &mut wtxn,
            &state_key(account, folder),
            &encode(state)?,
        )?;
        wtxn.commit()?;

        Ok(())
    }

    /// Read the state of one folder of one account.
    ///
    /// A folder that no sync ever read gives `None`.
    pub fn state(
        &self,
        account: &str,
        folder: &str,
    ) -> Result<Option<SyncState>> {
        let rtxn = self.meta.read_txn()?;

        self.db
            .state
            .get(&rtxn, &state_key(account, folder))?
            .map(decode)
            .transpose()
    }

    /// The state of each folder of one account, by folder name.
    pub fn states(&self, account: &str) -> Result<BTreeMap<String, SyncState>> {
        let rtxn = self.meta.read_txn()?;
        let head = format!("{account}{STATE_BREAK}");
        let mut found = BTreeMap::new();

        for entry in self.db.state.prefix_iter(&rtxn, head.as_str())? {
            let (key, bytes) = entry?;
            let Some(folder) = key.strip_prefix(head.as_str()) else {
                continue;
            };

            found.insert(folder.to_string(), decode(bytes)?);
        }

        Ok(found)
    }

    /// Forget the state of one folder, which makes the next sync read
    /// the whole folder again.
    ///
    /// Returns `true` when a state was there.
    pub fn forget_state(&self, account: &str, folder: &str) -> Result<bool> {
        let mut wtxn = self.meta.write_txn()?;
        let existed = self
            .db
            .state
            .delete(&mut wtxn, &state_key(account, folder))?;
        wtxn.commit()?;

        Ok(existed)
    }

    /// The message that sits at one UID of one folder (§4.2).
    ///
    /// A sync reads a UID and not an identity, so this is the way from
    /// what the server says to what the store holds.
    pub fn placed(
        &self,
        account: &str,
        folder: &str,
        uid: u32,
    ) -> Result<Option<MessageId>> {
        let rtxn = self.meta.read_txn()?;
        let key = place_key(account, folder, uid);

        Ok(self
            .db
            .places
            .get(&rtxn, &key)?
            .and_then(MessageId::from_hex))
    }

    /// Take away the copy that sits at one UID of one folder.
    ///
    /// The message stays, because mailbert is a mirror and keeps mail
    /// that the server dropped. A message that loses its last copy
    /// answers `is:gone`. Returns the identity when a copy went away.
    pub fn vanish(
        &self,
        account: &str,
        folder: &str,
        uid: u32,
    ) -> Result<Option<MessageId>> {
        let place = place_key(account, folder, uid);

        let mut wtxn = self.meta.write_txn()?;
        let Some(hex) = self.db.places.get(&wtxn, &place)?.map(str::to_string)
        else {
            return Ok(None);
        };

        let Some(mut message) = self
            .db
            .messages
            .get(&wtxn, &hex)?
            .map(decode::<Message>)
            .transpose()?
        else {
            return Ok(None);
        };

        message.remove_location(account, folder);
        self.db.places.delete(&mut wtxn, &place)?;
        self.db.messages.put(&mut wtxn, &hex, &encode(&message)?)?;
        wtxn.commit()?;

        Ok(Some(message.id))
    }

    /// Give one copy the flags that the server now reports (§3.3).
    ///
    /// The flags replace what that folder said before, because a folder
    /// that drops `\Seen` makes the message unread again. Returns the
    /// identity when a copy sits there.
    pub fn reflag(
        &self,
        account: &str,
        folder: &str,
        uid: u32,
        flags: &[String],
    ) -> Result<Option<MessageId>> {
        let place = place_key(account, folder, uid);

        let mut wtxn = self.meta.write_txn()?;
        let Some(hex) = self.db.places.get(&wtxn, &place)?.map(str::to_string)
        else {
            return Ok(None);
        };

        let Some(mut message) = self
            .db
            .messages
            .get(&wtxn, &hex)?
            .map(decode::<Message>)
            .transpose()?
        else {
            return Ok(None);
        };

        if !message.set_flags(account, folder, flags) {
            return Ok(None);
        }

        self.db.messages.put(&mut wtxn, &hex, &encode(&message)?)?;
        wtxn.commit()?;

        Ok(Some(message.id))
    }
}

impl Store {
    /// Write what an embedding pass gave one message. (§6.2)
    ///
    /// Returns the keys that the message held before and holds no
    /// more. The caller must drop those keys from the embedding
    /// database and from the PLAID index, or they answer a search with
    /// text that no message carries.
    pub fn mark_embedded(
        &self,
        id: &MessageId,
        embedded: &Embedded,
    ) -> Result<Vec<u64>> {
        let key = id.full_hex();

        let mut wtxn = self.meta.write_txn()?;
        let before: Option<Embedded> =
            self.db.embeds.get(&wtxn, &key)?.map(decode).transpose()?;

        let kept: BTreeSet<u64> = embedded.keys.iter().copied().collect();
        let dropped: Vec<u64> = before
            .map(|old| old.keys)
            .unwrap_or_default()
            .into_iter()
            .filter(|one| !kept.contains(one))
            .collect();

        for one in &dropped {
            self.db.owners.delete(&mut wtxn, &owner_key(*one))?;
        }
        for one in &kept {
            self.db.owners.put(&mut wtxn, &owner_key(*one), &key)?;
        }

        self.db.embeds.put(&mut wtxn, &key, &encode(embedded)?)?;
        wtxn.commit()?;

        Ok(dropped)
    }

    /// What the last embedding pass gave one message.
    ///
    /// A message that no pass ever read gives `None`.
    pub fn embedded(&self, id: &MessageId) -> Result<Option<Embedded>> {
        let rtxn = self.meta.read_txn()?;

        self.db
            .embeds
            .get(&rtxn, &id.full_hex())?
            .map(decode)
            .transpose()
    }

    /// The message that owns one passage. (§8.1)
    ///
    /// The semantic leg gives back the keys of passages, and this is
    /// the way from a passage to the message that it belongs to.
    pub fn owner(&self, key: u64) -> Result<Option<MessageId>> {
        let rtxn = self.meta.read_txn()?;

        Ok(self
            .db
            .owners
            .get(&rtxn, &owner_key(key))?
            .and_then(MessageId::from_hex))
    }

    /// The message that owns each passage of `keys`.
    ///
    /// A key that no message owns is not in the answer. One read
    /// transaction serves the whole list, because the semantic leg
    /// asks for a hundred keys at a time.
    pub fn owners(&self, keys: &[u64]) -> Result<BTreeMap<u64, MessageId>> {
        let rtxn = self.meta.read_txn()?;
        let mut found = BTreeMap::new();

        for key in keys {
            let Some(hex) = self.db.owners.get(&rtxn, &owner_key(*key))? else {
                continue;
            };
            let Some(id) = MessageId::from_hex(hex) else {
                continue;
            };

            found.insert(*key, id);
        }

        Ok(found)
    }

    /// Forget the embedding of one message.
    ///
    /// Returns the keys that the message held, which the caller must
    /// drop from the embedding database and from the PLAID index.
    pub fn forget_embedding(&self, id: &MessageId) -> Result<Vec<u64>> {
        let mut wtxn = self.meta.write_txn()?;
        let dropped = clear_embedding(&mut wtxn, &self.db, &id.full_hex())?;
        wtxn.commit()?;

        Ok(dropped)
    }

    /// Every message that the store holds, and the fingerprint of the
    /// last embedding pass over it.
    ///
    /// A message that no pass ever read is not in the answer, so a
    /// pass reads this one time and knows what it has.
    pub fn embeddings(&self) -> Result<BTreeMap<MessageId, [u8; 32]>> {
        let rtxn = self.meta.read_txn()?;
        let mut found = BTreeMap::new();

        for entry in self.db.embeds.iter(&rtxn)? {
            let (key, bytes) = entry?;
            let Some(id) = MessageId::from_hex(key) else {
                continue;
            };
            let embedded: Embedded = decode(bytes)?;

            found.insert(id, embedded.digest);
        }

        Ok(found)
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
    //! | `prop_a_sync_state_survives_the_store` | round-trip | §3.4 resumes from this record alone. A number that reads back wrong downloads a folder again, or loses mail. |
    //! | `prop_the_state_of_one_account_stays_there` | invariant | Two accounts hold folders of one name, such as INBOX. A key that mixes them syncs one account with the state of another. |
    //! | `prop_every_copy_of_a_message_answers_to_its_uid` | invariant | A copy that answers to nobody can never go away, and `is:gone` would never be true. |
    //! | `prop_a_vanish_never_moves_another_folder` | invariant | A UID that took the wrong copy loses mail that the server still holds. |
    //! | `prop_a_pass_leaves_one_owner_for_each_key_that_it_kept` | invariant | §6.2 keys every passage. An owner that stays behind answers a search with a message that no longer holds that text. |
    //! | `prop_a_reflag_says_what_the_copies_say` | invariant | The flags of a message are the flags of its copies. A set that drifts answers `is:unread` for mail that the user read. |
    //! | `prop_a_batch_writes_what_single_writes_write` | differential | §4.2. A batch takes one transaction, and single writes take two for each message. The two paths must leave one store. |
    //! | `prop_a_batch_gives_back_what_it_wrote` | round-trip | The sink counts what the store kept. An answer that does not match the store counts the wrong mail. |

    use hegel::{TestCase, generators as gs};
    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::{
        message::{Location, SEEN, collate},
        mime,
        query::Flag,
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
            flags: BTreeSet::new(),
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

    fn message_at(key: &str, account: &str, folder: &str, uid: u32) -> Message {
        let raw = raw_bytes(key, "The deposit is due.");

        Message::new(
            mime::parse(&raw).expect("a message"),
            location(account, folder, uid),
            [SEEN],
        )
    }

    /// Write one copy of a message, and give back its identity.
    fn write_at(
        store: &Store,
        key: &str,
        account: &str,
        folder: &str,
        uid: u32,
    ) -> MessageId {
        let found = message_at(key, account, folder, uid);
        let raw = raw_bytes(key, "The deposit is due.");
        store.put(&found, &raw).expect("a write");

        found.id
    }

    /// Folder names that are not the same, and are one or more.
    #[hegel::composite]
    fn some_folders(tc: TestCase) -> Vec<String> {
        let drawn: Vec<String> = tc.draw(
            gs::vecs(gs::text().alphabet("AB/").min_size(1).max_size(3))
                .min_size(1)
                .max_size(4),
        );

        let mut folders = drawn;
        folders.sort();
        folders.dedup();

        folders
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
    // The embeddings. (§6.2)
    // -----------------------------------------------------------------

    /// One embedding record, with the keys that it names.
    fn embedded(digest: u8, keys: &[u64]) -> Embedded {
        Embedded {
            digest: [digest; 32],
            keys: keys.to_vec(),
        }
    }

    #[test]
    fn a_message_that_no_pass_read_has_no_embedding() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = store.put(&message("a", "work", "INBOX"), b"raw").unwrap();

        assert_eq!(store.embedded(&id.id).expect("a read"), None);
    }

    #[test]
    fn an_embedding_reads_back_what_the_pass_wrote() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = store
            .put(&message("a", "work", "INBOX"), b"raw")
            .unwrap()
            .id;
        let record = embedded(7, &[10, 11, 12]);

        let dropped = store.mark_embedded(&id, &record).expect("a write");

        assert!(dropped.is_empty());
        assert_eq!(store.embedded(&id).expect("a read"), Some(record));
    }

    #[test]
    fn every_passage_names_the_message_that_owns_it() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = store
            .put(&message("a", "work", "INBOX"), b"raw")
            .unwrap()
            .id;

        store.mark_embedded(&id, &embedded(7, &[10, 11])).unwrap();

        assert_eq!(store.owner(10).expect("a read"), Some(id));
        assert_eq!(store.owner(11).expect("a read"), Some(id));
        assert_eq!(store.owner(12).expect("a read"), None);
    }

    #[test]
    fn a_shorter_message_gives_back_the_keys_that_it_dropped() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = store
            .put(&message("a", "work", "INBOX"), b"raw")
            .unwrap()
            .id;

        store
            .mark_embedded(&id, &embedded(7, &[10, 11, 12]))
            .unwrap();
        let dropped = store
            .mark_embedded(&id, &embedded(8, &[10]))
            .expect("a write");

        assert_eq!(dropped, vec![11, 12]);
        assert_eq!(store.owner(10).expect("a read"), Some(id));
        assert_eq!(store.owner(11).expect("a read"), None);
    }

    /// The next pass drops these passages from the embedding database
    /// and from the PLAID index. It finds them because the record
    /// stays behind and names a message that the store does not hold.
    #[test]
    fn a_message_that_goes_away_leaves_its_record_for_the_next_pass() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = store
            .put(&message("a", "work", "INBOX"), b"raw")
            .unwrap()
            .id;
        store.mark_embedded(&id, &embedded(7, &[10, 11])).unwrap();

        assert!(store.remove(&id).expect("a delete"));

        assert_eq!(store.get(&id).expect("a read"), None);
        assert_eq!(
            store.embedded(&id).expect("a read").map(|one| one.keys),
            Some(vec![10, 11])
        );
        assert_eq!(store.owner(10).expect("a read"), Some(id));
    }

    #[test]
    fn forgetting_an_embedding_gives_back_every_key() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = store
            .put(&message("a", "work", "INBOX"), b"raw")
            .unwrap()
            .id;
        store.mark_embedded(&id, &embedded(7, &[10, 11])).unwrap();

        let dropped = store.forget_embedding(&id).expect("a delete");

        assert_eq!(dropped, vec![10, 11]);
        assert_eq!(store.owner(10).expect("a read"), None);
        assert!(store.embeddings().expect("a read").is_empty());
    }

    #[test]
    fn the_fingerprints_name_every_message_that_a_pass_read() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let first = store
            .put(&message("a", "work", "INBOX"), b"raw")
            .unwrap()
            .id;
        let second = store
            .put(&message("b", "work", "INBOX"), b"raw")
            .unwrap()
            .id;

        store.mark_embedded(&first, &embedded(7, &[10])).unwrap();
        store.mark_embedded(&second, &embedded(8, &[20])).unwrap();

        assert_eq!(
            store.embeddings().expect("a read"),
            BTreeMap::from([(first, [7; 32]), (second, [8; 32])])
        );
    }

    #[test]
    fn the_owners_of_a_batch_leave_out_a_key_that_nobody_owns() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = store
            .put(&message("a", "work", "INBOX"), b"raw")
            .unwrap()
            .id;
        store.mark_embedded(&id, &embedded(7, &[10, 11])).unwrap();

        assert_eq!(
            store.owners(&[10, 99, 11]).expect("a read"),
            BTreeMap::from([(10, id), (11, id)])
        );
    }

    #[hegel::test(test_cases = 60)]
    fn prop_a_pass_leaves_one_owner_for_each_key_that_it_kept(tc: TestCase) {
        let first: Vec<u64> = tc.draw(a_key_set());
        let second: Vec<u64> = tc.draw(a_key_set());

        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = store
            .put(&message("a", "work", "INBOX"), b"raw")
            .unwrap()
            .id;

        store.mark_embedded(&id, &embedded(1, &first)).unwrap();
        let dropped = store
            .mark_embedded(&id, &embedded(2, &second))
            .expect("a write");

        let kept: BTreeSet<u64> = second.iter().copied().collect();
        let gone: BTreeSet<u64> = dropped.iter().copied().collect();

        // What the pass kept, and what it gave back, cover what was
        // there and never meet.
        assert!(gone.is_disjoint(&kept));
        assert_eq!(
            first.iter().copied().collect::<BTreeSet<u64>>(),
            gone.union(&kept)
                .copied()
                .filter(|one| first.contains(one))
                .collect()
        );

        for key in &kept {
            assert_eq!(store.owner(*key).expect("a read"), Some(id));
        }
        for key in &gone {
            assert_eq!(store.owner(*key).expect("a read"), None);
        }
    }

    /// A short list of passage keys, without repeats.
    #[hegel::composite]
    fn a_key_set(tc: TestCase) -> Vec<u64> {
        let keys: Vec<u64> = tc.draw(
            gs::vecs(gs::integers::<u64>().min_value(1).max_value(8))
                .min_size(0)
                .max_size(6),
        );

        keys.into_iter()
            .collect::<BTreeSet<u64>>()
            .into_iter()
            .collect()
    }

    // -----------------------------------------------------------------
    // A batch write. (§4.2)
    // -----------------------------------------------------------------

    /// Mail for a batch: a key, a folder, and a UID for each message.
    ///
    /// The keys and the folders are few, so a batch holds one message
    /// twice, and holds one message in two folders.
    #[hegel::composite]
    fn some_mail(tc: TestCase) -> Vec<(String, String, u32)> {
        let count: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
        let mut mail = Vec::new();

        for _ in 0..count {
            mail.push((
                tc.draw(gs::sampled_from(vec![
                    "a".to_string(),
                    "b".to_string(),
                    "c".to_string(),
                ])),
                tc.draw(gs::sampled_from(vec![
                    "INBOX".to_string(),
                    "Sent".to_string(),
                ])),
                tc.draw(gs::integers::<u32>().min_value(1).max_value(3)),
            ));
        }

        mail
    }

    /// The batch that `mail` names.
    fn a_batch(mail: &[(String, String, u32)]) -> Vec<(Message, Vec<u8>)> {
        mail.iter()
            .map(|(key, folder, uid)| {
                (
                    message_at(key, "work", folder, *uid),
                    raw_bytes(key, "The deposit is due."),
                )
            })
            .collect()
    }

    #[test]
    fn a_batch_keeps_every_message_that_it_was_given() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let batch = a_batch(&[
            ("a".to_string(), "INBOX".to_string(), 1),
            ("b".to_string(), "INBOX".to_string(), 2),
            ("c".to_string(), "Sent".to_string(), 3),
        ]);

        let kept = store.put_all(&batch).expect("a batch write");

        assert_eq!(kept.len(), 3);
        assert_eq!(store.all().expect("a read").len(), 3);
    }

    /// The store keys a message by its bytes. A batch that holds one
    /// message twice, from two folders, keeps one entry of two places.
    #[test]
    fn a_batch_that_repeats_a_message_keeps_one_entry() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let batch = a_batch(&[
            ("a".to_string(), "INBOX".to_string(), 1),
            ("a".to_string(), "Sent".to_string(), 2),
        ]);

        store.put_all(&batch).expect("a batch write");
        let all = store.all().expect("a read");

        assert_eq!(all.len(), 1, "the batch made two entries of one message");
        assert_eq!(all[0].locations.len(), 2, "the batch lost a place");
    }

    /// The bytes of a batch reach the blob database, and read back.
    #[test]
    fn a_batch_writes_the_raw_bytes() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let batch = a_batch(&[("a".to_string(), "INBOX".to_string(), 1)]);

        let kept = store.put_all(&batch).expect("a batch write");

        assert_eq!(
            store.raw(&kept[0].id).expect("a read"),
            Some(batch[0].1.clone())
        );
    }

    #[test]
    fn a_batch_of_nothing_writes_nothing() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        assert!(store.put_all(&[]).expect("a batch write").is_empty());
        assert!(store.all().expect("a read").is_empty());
    }

    /// §4.2. A batch takes one transaction of each database, and the
    /// single writes take two for each message. A sync of 100000
    /// messages must not read one store and write another.
    #[hegel::test(test_cases = 60)]
    fn prop_a_batch_writes_what_single_writes_write(tc: TestCase) {
        let mail = tc.draw(some_mail());
        let batch = a_batch(&mail);

        let one = tempdir().expect("a directory");
        let single = open_at(&one);
        for (found, raw) in &batch {
            single.put(found, raw).expect("a write");
        }

        let both = tempdir().expect("a directory");
        let together = open_at(&both);
        together.put_all(&batch).expect("a batch write");

        assert_eq!(
            single.all().expect("a read"),
            together.all().expect("a read"),
            "the batch and the single writes left two stores"
        );

        // The entries alone do not say everything. A UID that moved
        // leaves an old place behind, and only `placed` shows it.
        let folders: BTreeSet<&String> =
            mail.iter().map(|(_, folder, _)| folder).collect();

        for folder in folders {
            for uid in 1..=4 {
                assert_eq!(
                    single.placed("work", folder, uid).expect("a read"),
                    together.placed("work", folder, uid).expect("a read"),
                    "the stores put a different message at {folder}/{uid}"
                );
            }
        }
    }

    /// The sink counts the mail that the store kept, and names it for
    /// the index. An answer that the store does not hold counts mail
    /// that no search can find.
    #[hegel::test(test_cases = 60)]
    fn prop_a_batch_gives_back_what_it_wrote(tc: TestCase) {
        let mail = tc.draw(some_mail());
        let batch = a_batch(&mail);
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        let kept = store.put_all(&batch).expect("a batch write");

        assert_eq!(kept.len(), batch.len(), "the batch lost an answer");

        // Every answer names mail that the store holds.
        for one in &kept {
            assert!(
                store.get(&one.id).expect("a read").is_some(),
                "the batch named a message that the store lacks"
            );
        }

        // A batch that holds one message twice answers twice, and the
        // second answer absorbed the first. The last answer of each
        // identity is therefore what the store now holds.
        let mut last: BTreeMap<MessageId, &Message> = BTreeMap::new();
        for one in &kept {
            last.insert(one.id, one);
        }

        for (id, one) in last {
            assert_eq!(
                store.get(&id).expect("a read").as_ref(),
                Some(one),
                "the last answer does not say what the store holds"
            );
        }
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
                flags: BTreeSet::new(),
            },
            [SEEN],
        );

        (found, raw)
    }

    fn a_mailbox() -> impl gs::Generator<Vec<(Message, Vec<u8>)>> {
        gs::vecs(a_copy()).min_size(0).max_size(8)
    }

    // -----------------------------------------------------------------
    // The sync state (§3.3).
    // -----------------------------------------------------------------

    fn a_state(uid_next: u32, pending: &str) -> SyncState {
        SyncState {
            uid_validity: 77,
            uid_next,
            highest_mod_seq: 900,
            pending: pending.to_string(),
            synced_at: 1_755_820_800,
        }
    }

    /// §10.4 shows when the last sync ran, and the mark is the only
    /// record that knows. A time that the store loses is a time that
    /// `status` cannot show.
    #[test]
    fn the_state_keeps_the_time_of_the_sync() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        store
            .mark("work", "INBOX", &a_state(400, ""))
            .expect("a write");

        let found = store.state("work", "INBOX").expect("a read");

        assert_eq!(found.expect("a state").synced_at, 1_755_820_800);
    }

    #[test]
    fn a_folder_that_no_sync_marked_has_no_time() {
        assert_eq!(SyncState::default().synced_at, 0);
    }

    #[test]
    fn a_folder_that_no_sync_read_has_no_state() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        assert_eq!(store.state("work", "INBOX").expect("a read"), None);
    }

    #[test]
    fn a_state_comes_back_as_it_went_in() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let state = a_state(400, "12:20,44");

        store.mark("work", "INBOX", &state).expect("a write");

        assert_eq!(store.state("work", "INBOX").expect("a read"), Some(state));
    }

    #[test]
    fn a_second_mark_replaces_the_first() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        store
            .mark("work", "INBOX", &a_state(400, ""))
            .expect("a write");
        store
            .mark("work", "INBOX", &a_state(500, ""))
            .expect("a write");

        let found = store.state("work", "INBOX").expect("a read");

        assert_eq!(found.expect("a state").uid_next, 500);
    }

    #[test]
    fn two_accounts_keep_the_state_of_one_folder_name_apart() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        store
            .mark("work", "INBOX", &a_state(400, ""))
            .expect("a write");
        store
            .mark("home", "INBOX", &a_state(9, ""))
            .expect("a write");

        let work = store.state("work", "INBOX").expect("a read");
        let home = store.state("home", "INBOX").expect("a read");

        assert_eq!(work.expect("a state").uid_next, 400);
        assert_eq!(home.expect("a state").uid_next, 9);
    }

    #[test]
    fn the_states_of_an_account_hold_each_folder_of_it() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        store
            .mark("work", "INBOX", &a_state(400, ""))
            .expect("a write");
        store
            .mark("work", "Archive", &a_state(20, ""))
            .expect("a write");
        store
            .mark("home", "INBOX", &a_state(9, ""))
            .expect("a write");

        let found = store.states("work").expect("a read");

        assert_eq!(found.len(), 2);
        assert!(found.contains_key("INBOX"));
        assert!(found.contains_key("Archive"));
    }

    #[test]
    fn an_account_with_no_sync_has_no_states() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        assert!(store.states("work").expect("a read").is_empty());
    }

    #[test]
    fn forget_state_removes_one_folder_only() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        store
            .mark("work", "INBOX", &a_state(400, ""))
            .expect("a write");
        store
            .mark("work", "Archive", &a_state(20, ""))
            .expect("a write");

        assert!(store.forget_state("work", "INBOX").expect("a delete"));
        assert!(!store.forget_state("work", "INBOX").expect("a delete"));
        assert_eq!(store.state("work", "INBOX").expect("a read"), None);
        assert!(store.state("work", "Archive").expect("a read").is_some());
    }

    #[test]
    fn a_folder_name_with_a_stroke_in_it_keeps_its_own_state() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        store
            .mark("work", "Lists/Rust", &a_state(3, ""))
            .expect("a write");

        let found = store.states("work").expect("a read");

        assert!(found.contains_key("Lists/Rust"), "{found:?}");
    }

    #[hegel::test(test_cases = 60)]
    fn prop_a_sync_state_survives_the_store(tc: TestCase) {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let state = SyncState {
            uid_validity: tc
                .draw(gs::integers::<u32>().min_value(0).max_value(u32::MAX)),
            uid_next: tc
                .draw(gs::integers::<u32>().min_value(0).max_value(u32::MAX)),
            highest_mod_seq: tc
                .draw(gs::integers::<u64>().min_value(0).max_value(u64::MAX)),
            pending: tc.draw(
                gs::text().alphabet("0123456789:,").min_size(0).max_size(20),
            ),
            synced_at: tc.draw(
                gs::integers::<i64>().min_value(0).max_value(4_102_444_800),
            ),
        };
        let folder: String =
            tc.draw(gs::text().alphabet("ABCdef/. ").min_size(1).max_size(10));

        store.mark("work", &folder, &state).expect("a write");

        assert_eq!(store.state("work", &folder).expect("a read"), Some(state));
    }

    #[hegel::test(test_cases = 40)]
    fn prop_the_state_of_one_account_stays_there(tc: TestCase) {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);

        let names: Vec<String> = tc.draw(
            gs::vecs(gs::text().alphabet("ab").min_size(1).max_size(3))
                .min_size(1)
                .max_size(4),
        );
        let folder: String =
            tc.draw(gs::text().alphabet("XY/").min_size(1).max_size(4));

        let mut unique: Vec<String> = names.clone();
        unique.sort();
        unique.dedup();

        for (step, account) in unique.iter().enumerate() {
            let mut state = a_state(step as u32 + 1, "");
            state.uid_validity = step as u32 + 1;
            store.mark(account, &folder, &state).expect("a write");
        }

        for (step, account) in unique.iter().enumerate() {
            let found = store.states(account).expect("a read");

            assert_eq!(found.len(), 1, "`{account}` sees another account");
            assert_eq!(
                found[&folder].uid_next,
                step as u32 + 1,
                "`{account}` reads the state of another account"
            );
        }
    }

    // -----------------------------------------------------------------
    // Unit tests: the place of each copy (§4.2).
    // -----------------------------------------------------------------

    #[test]
    fn a_copy_answers_to_its_uid() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let id = write_at(&store, "one", "work", "INBOX", 7);

        assert_eq!(store.placed("work", "INBOX", 7).expect("a read"), Some(id));
    }

    #[test]
    fn a_uid_that_no_copy_holds_answers_to_nobody() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        write_at(&store, "one", "work", "INBOX", 7);

        assert_eq!(store.placed("work", "INBOX", 8).expect("a read"), None);
        assert_eq!(store.placed("home", "INBOX", 7).expect("a read"), None);
        assert_eq!(store.placed("work", "Sent", 7).expect("a read"), None);
    }

    #[test]
    fn each_folder_of_one_message_answers_to_its_own_uid() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let id = write_at(&store, "one", "work", "INBOX", 7);
        write_at(&store, "one", "work", "Archive", 31);

        assert_eq!(store.placed("work", "INBOX", 7).expect("a read"), Some(id));
        assert_eq!(
            store.placed("work", "Archive", 31).expect("a read"),
            Some(id)
        );
    }

    #[test]
    fn a_second_reading_of_a_folder_moves_the_place() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let id = write_at(&store, "one", "work", "INBOX", 7);
        write_at(&store, "one", "work", "INBOX", 9);

        assert_eq!(store.placed("work", "INBOX", 7).expect("a read"), None);
        assert_eq!(store.placed("work", "INBOX", 9).expect("a read"), Some(id));
    }

    #[test]
    fn a_message_that_goes_away_loses_that_folder() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let id = write_at(&store, "one", "work", "INBOX", 7);

        assert_eq!(
            store.vanish("work", "INBOX", 7).expect("a write"),
            Some(id)
        );

        let found = store.get(&id).expect("a read").expect("the message");
        assert!(found.is_gone(), "the message keeps a place that is gone");
        assert_eq!(store.placed("work", "INBOX", 7).expect("a read"), None);
    }

    #[test]
    fn a_message_that_sits_elsewhere_is_not_gone() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let id = write_at(&store, "one", "work", "INBOX", 7);
        write_at(&store, "one", "work", "Archive", 31);

        store.vanish("work", "INBOX", 7).expect("a write");

        let found = store.get(&id).expect("a read").expect("the message");
        assert!(!found.is_gone(), "one copy went, and the message went too");
        assert_eq!(found.folders(), vec!["Archive"]);
    }

    #[test]
    fn a_vanish_of_a_uid_that_is_not_there_changes_nothing() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let id = write_at(&store, "one", "work", "INBOX", 7);

        assert_eq!(store.vanish("work", "INBOX", 8).expect("a write"), None);

        let found = store.get(&id).expect("a read").expect("the message");
        assert!(!found.is_gone(), "a UID that is not there took a copy");
    }

    #[test]
    fn a_removed_message_takes_its_places_with_it() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let id = write_at(&store, "one", "work", "INBOX", 7);

        assert!(store.remove(&id).expect("a delete"));

        assert_eq!(store.placed("work", "INBOX", 7).expect("a read"), None);
    }

    #[hegel::test(test_cases = 40)]
    fn prop_every_copy_of_a_message_answers_to_its_uid(tc: TestCase) {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let folders = tc.draw(some_folders());

        let mut id = None;
        for (step, folder) in folders.iter().enumerate() {
            id = Some(write_at(&store, "one", "work", folder, step as u32 + 1));
        }

        for (step, folder) in folders.iter().enumerate() {
            assert_eq!(
                store
                    .placed("work", folder, step as u32 + 1)
                    .expect("a read"),
                id,
                "the copy in `{folder}` answers to nobody"
            );
        }
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_vanish_never_moves_another_folder(tc: TestCase) {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let folders = tc.draw(some_folders());
        let which: usize = tc.draw(
            gs::integers::<usize>()
                .min_value(0)
                .max_value(folders.len() - 1),
        );

        let mut id = None;
        for (step, folder) in folders.iter().enumerate() {
            id = Some(write_at(&store, "one", "work", folder, step as u32 + 1));
        }

        let uid = which as u32 + 1;
        store.vanish("work", &folders[which], uid).expect("a write");

        for (step, folder) in folders.iter().enumerate() {
            let found = store
                .placed("work", folder, step as u32 + 1)
                .expect("a read");

            match step == which {
                true => assert_eq!(found, None, "`{folder}` stayed"),
                false => assert_eq!(found, id, "`{folder}` went with it"),
            }
        }
    }

    // -----------------------------------------------------------------
    // Unit tests: the flags that a folder reports again (§3.3).
    // -----------------------------------------------------------------

    #[test]
    fn a_reflag_replaces_the_flags_of_that_copy() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let id = write_at(&store, "one", "work", "INBOX", 7);
        let flagged = vec![r"\Flagged".to_string()];

        assert_eq!(
            store.reflag("work", "INBOX", 7, &flagged).expect("a write"),
            Some(id)
        );

        let found = store.get(&id).expect("a read").expect("the message");
        assert!(found.matches(Flag::Unread), "the message stayed read");
        assert!(found.matches(Flag::Flagged));
    }

    #[test]
    fn a_reflag_leaves_the_flags_of_another_copy() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let id = write_at(&store, "one", "work", "INBOX", 7);
        write_at(&store, "one", "work", "Archive", 31);

        store.reflag("work", "INBOX", 7, &[]).expect("a write");

        let found = store.get(&id).expect("a read").expect("the message");
        assert!(found.matches(Flag::Read), "the other copy lost its flags");
    }

    #[test]
    fn a_reflag_of_a_uid_that_is_not_there_changes_nothing() {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let id = write_at(&store, "one", "work", "INBOX", 7);

        assert_eq!(
            store.reflag("work", "INBOX", 8, &[]).expect("a write"),
            None
        );

        let found = store.get(&id).expect("a read").expect("the message");
        assert!(found.matches(Flag::Read), "a UID that is not there wrote");
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_reflag_says_what_the_copies_say(tc: TestCase) {
        let dir = tempdir().expect("a temporary directory");
        let store = open_at(&dir);
        let folders = tc.draw(some_folders());

        let mut id = None;
        for (step, folder) in folders.iter().enumerate() {
            id = Some(write_at(&store, "one", "work", folder, step as u32 + 1));
        }

        let which: usize = tc.draw(
            gs::integers::<usize>()
                .min_value(0)
                .max_value(folders.len() - 1),
        );
        let carried: Vec<String> = tc.draw(
            gs::vecs(gs::sampled_from(vec![
                r"\Seen".to_string(),
                r"\Flagged".to_string(),
            ]))
            .min_size(0)
            .max_size(2),
        );

        let uid = which as u32 + 1;
        store
            .reflag("work", &folders[which], uid, &carried)
            .expect("a write");

        let id = id.expect("an identity");
        let found = store.get(&id).expect("a read").expect("the message");
        let want: BTreeSet<String> = found
            .locations
            .iter()
            .flat_map(|at| at.flags.iter().cloned())
            .collect();

        assert_eq!(found.flags, want, "the message left its copies behind");
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
