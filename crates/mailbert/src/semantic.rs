//! The semantic leg: the embeddings, and the search over them. (§6.2)
//!
//! A pass reads the store, cuts each message into passages, and gives
//! the passages that changed to the model. The embeddings go into the
//! embedding database of docbert, and the PLAID index of docbert ranks
//! them. mailbert holds the way from a passage back to its message, so
//! a hit becomes a message and not a chunk.
//!
//! A pass costs one message for one new message. The fingerprint of
//! §6.2 says what changed, so a mailbox of 100000 messages does not go
//! through the model again when one letter arrives.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use candle_core::Tensor;
use docbert_core::{
    EmbeddingDb,
    embedding::embed_and_store_in_batches,
    model_manager::ModelManager,
    plaid::{
        PlaidBuildParams,
        build_index_from_embedding_db,
        search as plaid_search,
        update_index_with_chunks,
    },
};
use docbert_plaid::{index::Index as PlaidIndex, persistence};
use mailbert_core::{
    MessageId,
    Store,
    embed::{self, Passage},
    message::Message,
};

use crate::error::Result;

/// How many passages go to the model in one submission.
pub const BATCH: usize = 256;

/// How much deeper than the wanted count the leg reads.
///
/// §8.2 asks a filter to gate the leg before it ranks, and PLAID takes
/// no allowlist. The leg therefore reads deep and then keeps what the
/// filter allows. A filter that names few messages in a large mailbox
/// can still come back short, and that is the limit of this approach.
pub const OVERSAMPLE: usize = 8;

/// The least that the leg reads, whatever the caller asks for.
pub const FLOOR: usize = 64;

/// The model, the embeddings, and the index of the semantic leg.
///
/// A sync makes one of these and sweeps with it. `ksearch` never makes
/// one, because §2.1 asks that command to load no model.
pub struct Brain {
    /// The model that turns a passage into token embeddings.
    pub model: ModelManager,

    /// The token embeddings of each passage.
    pub db: EmbeddingDb,

    /// The file that holds the PLAID index.
    pub at: PathBuf,

    /// How the PLAID index is built, when a pass builds it whole.
    pub params: PlaidBuildParams,
}

impl std::fmt::Debug for Brain {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("Brain")
            .field("model", &self.model.model_id())
            .field("at", &self.at)
            .finish_non_exhaustive()
    }
}

impl Brain {
    /// Open the embedding database, and name the model and the index.
    ///
    /// The model does not load here. It loads when a sweep first has
    /// something to embed, so a sync of a mailbox that did not change
    /// costs no model.
    ///
    /// # Errors
    ///
    /// The function fails if the embedding database cannot open.
    pub fn open(db: &Path, at: &Path, model: Option<&str>) -> Result<Self> {
        let model = match model {
            Some(name) => ModelManager::with_model_id(name.to_string()),
            None => ModelManager::new(),
        };

        Ok(Self {
            model,
            db: EmbeddingDb::open(db)?,
            at: at.to_path_buf(),
            params: PlaidBuildParams::default(),
        })
    }

    /// Embed what changed, and write the PLAID index. (§6.2)
    ///
    /// `report` sees how many messages the pass has finished, once for
    /// each batch, so a long sweep can show its progress.
    ///
    /// # Errors
    ///
    /// The function fails if the model, the store, or either index
    /// refuses the work.
    pub fn sweep(
        &mut self,
        store: &Store,
        report: impl FnMut(usize),
    ) -> Result<Embedded> {
        let plan = plan(store, self.model.model_id())?;

        if plan.is_empty() {
            return Ok(Embedded::default());
        }

        let done = apply(store, &self.db, &mut self.model, &plan, report)?;
        rebuild(&self.db, &self.at, &plan, self.params)?;

        Ok(done)
    }
}

/// What one message needs from the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Work {
    /// The message that the passages belong to.
    pub id: MessageId,

    /// The passages, in the order that the message cut.
    pub passages: Vec<Passage>,

    /// The fingerprint that the store keeps when the work is done.
    pub digest: [u8; 32],
}

/// What a pass must do to make the embeddings agree with the store.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Plan {
    /// The messages whose passages the model has not read.
    pub work: Vec<Work>,

    /// The passages of messages that the store no longer holds.
    pub stale: Vec<u64>,
}

impl Plan {
    /// Whether the embeddings already agree with the store.
    pub fn is_empty(&self) -> bool {
        self.work.is_empty() && self.stale.is_empty()
    }

    /// How many passages the model must read.
    pub fn passages(&self) -> usize {
        self.work.iter().map(|one| one.passages.len()).sum()
    }
}

/// What a pass did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Embedded {
    /// The messages that went to the model.
    pub messages: usize,

    /// The passages that went to the model.
    pub passages: usize,

    /// The passages that left the index.
    pub dropped: usize,
}

/// What a pass must do, for the messages that `store` holds. (§6.2)
///
/// A message whose fingerprint agrees with the store is not in the
/// plan, and that is what makes a second pass cheap.
pub fn plan(store: &Store, model: &str) -> Result<Plan> {
    let messages = store.all()?;
    let mut known = store.embeddings()?;
    let mut work = Vec::new();

    for message in &messages {
        let passages = embed::passages(message, embed::SIZE, embed::OVERLAP);
        let digest = embed::digest(model, &passages);

        // The fingerprint holds the name of the model, so a model that
        // changed sends every message to the new one.
        match known.remove(&message.id) {
            Some(seen) if seen == digest => {}
            _ => work.push(Work {
                id: message.id,
                passages,
                digest,
            }),
        }
    }

    // What is left of `known` is a message that the store let go
    // between two passes. Its passages must leave the index with it.
    let mut stale = Vec::new();
    for id in known.keys() {
        stale.extend(store.forget_embedding(id)?);
    }

    Ok(Plan { work, stale })
}

/// The passages that one message no longer needs.
///
/// A message that grew shorter keeps the keys of the passages that it
/// still has, and gives back the rest.
pub fn record(store: &Store, work: &Work) -> Result<Vec<u64>> {
    let embedded = mailbert_core::store::Embedded {
        digest: work.digest,
        keys: work.passages.iter().map(|one| one.key).collect(),
    };

    Ok(store.mark_embedded(&work.id, &embedded)?)
}

/// Run the plan: embed what changed, and forget what went away.
///
/// The embeddings go into `db`, and the store learns what each message
/// now holds. The caller then writes the PLAID index with [`rebuild`].
pub fn apply(
    store: &Store,
    db: &EmbeddingDb,
    model: &mut ModelManager,
    plan: &Plan,
    report: impl FnMut(usize),
) -> Result<Embedded> {
    let mut give = |texts: Vec<(u64, String)>| -> Result<usize> {
        Ok(embed_and_store_in_batches(model, db, texts, BATCH, |_| {})?)
    };

    run(store, db, plan, report, &mut give)
}

/// The bookkeeping of a pass, with the model behind `give`.
///
/// [`apply`] gives the passages to a real model. This holds everything
/// else: what the store learns, what leaves the embedding database,
/// and in which order. A test gives its own vectors and reads the
/// same book.
///
/// `give` receives the key and the text of each passage of one batch,
/// and answers with how many embeddings it wrote.
pub fn run(
    store: &Store,
    db: &EmbeddingDb,
    plan: &Plan,
    mut report: impl FnMut(usize),
    give: &mut impl FnMut(Vec<(u64, String)>) -> Result<usize>,
) -> Result<Embedded> {
    let mut done = Embedded {
        dropped: plan.stale.len(),
        ..Embedded::default()
    };

    db.batch_remove(&plan.stale)?;

    for group in plan.work.chunks(BATCH) {
        let mut texts = Vec::new();

        for work in group {
            // The store learns first. A stop after the model wrote and
            // before the store did would leave a passage that no
            // message owns, and the next pass would never find it.
            let dropped = record(store, work)?;
            db.batch_remove(&dropped)?;
            done.dropped += dropped.len();

            texts.extend(
                work.passages.iter().map(|one| (one.key, one.text.clone())),
            );
        }

        done.passages += give(texts)?;
        done.messages += group.len();

        report(done.messages);
    }

    Ok(done)
}

/// Write the PLAID index over every embedding that `db` holds.
///
/// A first pass builds the index, and a later pass gives it only the
/// passages that moved. An empty database writes no index, because a
/// mailbox with no mail has nothing to rank.
pub fn rebuild(
    db: &EmbeddingDb,
    at: &Path,
    plan: &Plan,
    params: PlaidBuildParams,
) -> Result<bool> {
    if db.list_ids()?.is_empty() {
        let _ = std::fs::remove_file(at);
        return Ok(false);
    }

    let upserts: Vec<u64> = plan
        .work
        .iter()
        .flat_map(|one| one.passages.iter().map(|two| two.key))
        .collect();

    let index = match load(at)? {
        Some(held) => {
            update_index_with_chunks(db, held, &upserts, &plan.stale)?
        }
        None => build_index_from_embedding_db(db, params)?,
    };

    persistence::save(&index, at)?;

    Ok(true)
}

/// Read the PLAID index, if a pass has written one.
pub fn load(at: &Path) -> Result<Option<PlaidIndex>> {
    match at.exists() {
        true => Ok(Some(persistence::load(at)?)),
        false => Ok(None),
    }
}

/// The candidates of the semantic leg, best first. (§8.1)
///
/// `allow` gates the leg before it ranks (§8.2). `None` lets every
/// message through, and a set keeps the messages that it names. The
/// answer holds the numeric key of each message, which is what the
/// index of §6.1 stores and what the fusion of §8.1 reads.
pub fn leg(
    index: &PlaidIndex,
    store: &Store,
    query: &Tensor,
    allow: Option<&BTreeSet<u64>>,
    count: usize,
) -> Result<Vec<u64>> {
    let deep = count.saturating_mul(OVERSAMPLE).max(FLOOR);
    let found = plaid_search(index, query, deep)?;

    let keys: Vec<u64> = found.iter().map(|one| one.doc_id).collect();
    let owners: BTreeMap<u64, u64> = store
        .owners(&keys)?
        .into_iter()
        .map(|(key, id)| (key, id.numeric()))
        .filter(|(_, owner)| allow.is_none_or(|set| set.contains(owner)))
        .collect();

    let scored: Vec<(u64, f32)> = found
        .into_iter()
        .map(|one| (one.doc_id, one.score))
        .collect();

    let mut ranked = embed::collapse(&scored, &owners);
    ranked.truncate(count);

    Ok(ranked)
}

/// The messages that a pass has embedded, by their numeric key.
///
/// `search` reads this to see whether the semantic leg can run at all.
pub fn covered(store: &Store) -> Result<BTreeSet<u64>> {
    Ok(store.embeddings()?.keys().map(MessageId::numeric).collect())
}

/// Whether the store holds a message that no pass has embedded.
pub fn behind(store: &Store, model: &str) -> Result<bool> {
    Ok(!plan(store, model)?.work.is_empty())
}

/// The passages of one message, for a caller that has no store.
pub fn passages_of(message: &Message) -> Vec<Passage> {
    embed::passages(message, embed::SIZE, embed::OVERLAP)
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_second_plan_asks_for_nothing` | metamorphic | §6.2 embeds every message. A pass that asks again sends a whole mailbox to the model for one new letter. |
    //! | `prop_a_plan_covers_every_message_of_the_store` | model-based | A message that no plan names is a message that the semantic leg can never find. |
    //! | `prop_a_message_that_went_away_takes_its_passages` | invariant | A passage of a message that is gone answers a search with text that the store no longer holds. |
    //! | `prop_the_leg_keeps_what_the_filter_allows` | invariant | §8.2 gates the leg before it ranks. A message that the filter refuses must never reach the fusion. |

    use std::collections::BTreeSet;

    use candle_core::Device;
    use hegel::{TestCase, generators as gs};
    use mailbert_core::{message::Location, mime};
    use tempfile::{TempDir, tempdir};

    use super::*;

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    /// The name that the tests give the model.
    const MODEL: &str = "a-model";

    /// How wide a hand-made embedding is.
    const DIM: u32 = 2;

    /// Two centroids are enough for a handful of passages.
    fn small() -> PlaidBuildParams {
        PlaidBuildParams {
            k_centroids: 2,
            nbits: 2,
            max_kmeans_iters: 50,
        }
    }

    fn open_at(dir: &TempDir) -> Store {
        Store::open(&dir.path().join("store")).expect("a store")
    }

    fn open_db(dir: &TempDir) -> EmbeddingDb {
        EmbeddingDb::open(&dir.path().join("embeddings.db")).expect("a db")
    }

    fn location(uid: u32) -> Location {
        Location {
            account: "work".to_string(),
            folder: "INBOX".to_string(),
            uid,
            uid_validity: 1,
            received: 1_755_820_800,
            flags: BTreeSet::new(),
        }
    }

    fn raw(key: &str, body: &str) -> Vec<u8> {
        format!(
            "From: Alice Smith <alice@example.test>\r\n\
             Subject: Deposit {key}\r\n\
             Date: Fri, 22 Aug 2025 09:30:00 +0000\r\n\
             Message-ID: <{key}@x.test>\r\n\
             \r\n\
             {body}\r\n"
        )
        .into_bytes()
    }

    /// Write one message into the store, and give back its identity.
    fn write(store: &Store, key: &str, body: &str) -> MessageId {
        let bytes = raw(key, body);
        let message = Message::new(
            mime::parse(&bytes).expect("a message"),
            location(1),
            Vec::<String>::new(),
        );

        store.put(&message, &bytes).expect("a write").id
    }

    /// The direction that the passages of a near message point in.
    const NEAR: [f32; 2] = [1.0, 0.0];

    /// The direction that the passages of a far message point in.
    ///
    /// It leans away from [`NEAR`] and does not stand square to it,
    /// because PLAID prunes what a query points nowhere near, and a
    /// test of the filter must see both messages. (§8.2)
    const FAR: [f32; 2] = [0.6, 0.8];

    /// Put a hand-made embedding on every passage of a plan.
    ///
    /// `place` gives the direction that the passages of one message
    /// point in. The model gives a unit vector, and these do the same,
    /// so a test puts two messages apart without a model.
    fn seed(
        store: &Store,
        db: &EmbeddingDb,
        plan: &Plan,
        place: impl Fn(&MessageId) -> [f32; 2],
    ) {
        for work in &plan.work {
            record(store, work).expect("a record");

            let at = place(&work.id);
            for passage in &work.passages {
                db.store(passage.key, 1, DIM, &at).expect("an embedding");
            }
        }
    }

    /// A query of one token that points at `at`.
    fn query(at: [f32; 2]) -> Tensor {
        Tensor::from_vec(at.to_vec(), (1, DIM as usize), &Device::Cpu)
            .expect("a tensor")
    }

    // -----------------------------------------------------------------
    // The plan.
    // -----------------------------------------------------------------

    #[test]
    fn a_store_with_no_mail_gives_an_empty_plan() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);

        assert!(plan(&store, MODEL).expect("a plan").is_empty());
    }

    #[test]
    fn a_first_plan_names_every_message() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let first = write(&store, "a", "The deposit is due.");
        let second = write(&store, "b", "The inspection is done.");

        let made = plan(&store, MODEL).expect("a plan");

        let named: BTreeSet<MessageId> =
            made.work.iter().map(|one| one.id).collect();
        assert_eq!(named, BTreeSet::from([first, second]));
        assert_eq!(made.passages(), 2);
    }

    #[test]
    fn a_second_plan_asks_for_nothing() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        write(&store, "a", "The deposit is due.");

        let first = plan(&store, MODEL).expect("a plan");
        for work in &first.work {
            record(&store, work).expect("a record");
        }

        assert!(plan(&store, MODEL).expect("a plan").is_empty());
    }

    #[test]
    fn a_model_that_changed_asks_for_every_message_again() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        write(&store, "a", "The deposit is due.");

        let first = plan(&store, MODEL).expect("a plan");
        for work in &first.work {
            record(&store, work).expect("a record");
        }

        let second = plan(&store, "another-model").expect("a plan");

        assert_eq!(second.work.len(), 1);
    }

    #[test]
    fn a_message_that_went_away_leaves_a_stale_passage() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let id = write(&store, "a", "The deposit is due.");

        let first = plan(&store, MODEL).expect("a plan");
        let keys: Vec<u64> =
            first.work[0].passages.iter().map(|one| one.key).collect();
        for work in &first.work {
            record(&store, work).expect("a record");
        }

        // The message goes without the store, so the record stays.
        // This is what a rebuild of the store looks like from here.
        store.remove(&id).expect("a delete");

        let second = plan(&store, MODEL).expect("a plan");

        assert_eq!(second.stale, keys);
        assert!(second.work.is_empty());
    }

    #[test]
    fn a_message_that_is_behind_says_so() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        write(&store, "a", "The deposit is due.");

        assert!(behind(&store, MODEL).expect("a plan"));

        for work in &plan(&store, MODEL).expect("a plan").work {
            record(&store, work).expect("a record");
        }

        assert!(!behind(&store, MODEL).expect("a plan"));
        assert_eq!(covered(&store).expect("a read").len(), 1);
    }

    // -----------------------------------------------------------------
    // The index, and the leg.
    // -----------------------------------------------------------------

    #[test]
    fn an_empty_database_writes_no_index() {
        let dir = tempdir().expect("a directory");
        let db = open_db(&dir);
        let at = dir.path().join("plaid.idx");

        assert!(
            !rebuild(&db, &at, &Plan::default(), small()).expect("a build")
        );
        assert!(!at.exists());
    }

    #[test]
    fn the_leg_gives_the_message_that_sits_nearest() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let db = open_db(&dir);
        let at = dir.path().join("plaid.idx");

        let near = write(&store, "a", "The deposit is due.");
        let far = write(&store, "b", "The inspection is done.");

        let made = plan(&store, MODEL).expect("a plan");
        seed(&store, &db, &made, |id| match *id == near {
            true => NEAR,
            false => FAR,
        });

        assert!(rebuild(&db, &at, &made, small()).expect("a build"));
        let index = load(&at).expect("a read").expect("an index");

        let ranked = leg(&index, &store, &query(NEAR), None, 2).expect("a leg");

        assert_eq!(ranked.first(), Some(&near.numeric()));
        assert_ne!(near.numeric(), far.numeric());
    }

    #[test]
    fn the_leg_keeps_only_what_the_filter_allows() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let db = open_db(&dir);
        let at = dir.path().join("plaid.idx");

        let near = write(&store, "a", "The deposit is due.");
        let far = write(&store, "b", "The inspection is done.");

        let made = plan(&store, MODEL).expect("a plan");
        seed(&store, &db, &made, |id| match *id == near {
            true => NEAR,
            false => FAR,
        });
        rebuild(&db, &at, &made, small()).expect("a build");
        let index = load(&at).expect("a read").expect("an index");

        let allow = BTreeSet::from([far.numeric()]);
        let ranked =
            leg(&index, &store, &query(NEAR), Some(&allow), 2).expect("a leg");

        assert_eq!(ranked, vec![far.numeric()]);
    }

    #[test]
    fn a_message_gives_one_row_however_many_passages_it_has() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let db = open_db(&dir);
        let at = dir.path().join("plaid.idx");

        let long = write(&store, "a", &"invoice ".repeat(600));
        write(&store, "b", "The inspection is done.");

        let made = plan(&store, MODEL).expect("a plan");
        assert!(
            made.work.iter().any(|one| one.passages.len() > 1),
            "the long message gave one passage"
        );

        seed(&store, &db, &made, |id| match *id == long {
            true => NEAR,
            false => FAR,
        });
        rebuild(&db, &at, &made, small()).expect("a build");
        let index = load(&at).expect("a read").expect("an index");

        let ranked = leg(&index, &store, &query(NEAR), None, 8).expect("a leg");

        let names: BTreeSet<u64> = ranked.iter().copied().collect();
        assert_eq!(names.len(), ranked.len());
        assert_eq!(ranked.first(), Some(&long.numeric()));
    }

    // -----------------------------------------------------------------
    // The pass.
    // -----------------------------------------------------------------

    /// A model that writes one hand-made vector for each passage.
    ///
    /// It keeps every text that it saw, so a test reads what the pass
    /// gave it and in which order.
    fn fake(
        db: &EmbeddingDb,
        seen: &mut Vec<(u64, String)>,
    ) -> impl FnMut(Vec<(u64, String)>) -> Result<usize> {
        move |texts| {
            for (key, text) in &texts {
                db.store(*key, 1, DIM, &NEAR).expect("an embedding");
                seen.push((*key, text.clone()));
            }

            Ok(texts.len())
        }
    }

    #[test]
    fn a_pass_gives_every_passage_of_every_message_to_the_model() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let db = open_db(&dir);
        write(&store, "a", "The deposit is due.");
        write(&store, "b", &"invoice ".repeat(600));

        let made = plan(&store, MODEL).expect("a plan");
        let mut seen = Vec::new();
        let done = run(&store, &db, &made, |_| {}, &mut fake(&db, &mut seen))
            .expect("a pass");

        assert_eq!(done.messages, 2);
        assert_eq!(done.passages, made.passages());
        assert_eq!(done.dropped, 0);
        assert_eq!(seen.len(), made.passages());
        assert!(
            seen.iter().all(|(_, text)| text.contains("Alice Smith")),
            "a passage lost its preamble"
        );
    }

    #[test]
    fn a_pass_that_ran_leaves_nothing_for_the_next_one() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let db = open_db(&dir);
        write(&store, "a", "The deposit is due.");

        let made = plan(&store, MODEL).expect("a plan");
        run(&store, &db, &made, |_| {}, &mut fake(&db, &mut Vec::new()))
            .expect("a pass");

        assert!(plan(&store, MODEL).expect("a plan").is_empty());
    }

    #[test]
    fn a_pass_drops_the_passages_of_a_message_that_went_away() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let db = open_db(&dir);
        let id = write(&store, "a", "The deposit is due.");

        let first = plan(&store, MODEL).expect("a plan");
        let keys: Vec<u64> =
            first.work[0].passages.iter().map(|one| one.key).collect();
        run(&store, &db, &first, |_| {}, &mut fake(&db, &mut Vec::new()))
            .expect("a pass");
        store.remove(&id).expect("a delete");

        let second = plan(&store, MODEL).expect("a plan");
        let done =
            run(&store, &db, &second, |_| {}, &mut fake(&db, &mut vec![]))
                .expect("a pass");

        assert_eq!(done.dropped, keys.len());
        for key in &keys {
            assert!(
                db.load(*key).expect("a read").is_none(),
                "the embedding of a message that is gone stayed behind"
            );
        }
    }

    /// A stop between the model and the store leaves a passage that no
    /// message owns, and no pass ever finds it again. The store must
    /// learn first, so a stop only costs the work of one batch.
    #[test]
    fn the_store_learns_before_the_model_writes() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let db = open_db(&dir);
        write(&store, "a", "The deposit is due.");

        let made = plan(&store, MODEL).expect("a plan");
        let mut owned = Vec::new();
        let mut watch = |texts: Vec<(u64, String)>| -> Result<usize> {
            for (key, _) in &texts {
                owned.push(store.owner(*key).expect("a read"));
            }

            Ok(texts.len())
        };
        run(&store, &db, &made, |_| {}, &mut watch).expect("a pass");

        assert!(
            owned.iter().all(Option::is_some),
            "the model saw a passage that the store did not own yet"
        );
    }

    // -----------------------------------------------------------------
    // The brain.
    // -----------------------------------------------------------------

    /// The name of a model that no machine can load. A sweep that
    /// touches it fails, and that is how these tests see the model
    /// stay away.
    const ABSENT: &str = "nobody/no-such-model";

    fn brain_at(dir: &TempDir) -> Brain {
        Brain::open(
            &dir.path().join("embeddings.db"),
            &dir.path().join("plaid.idx"),
            Some(ABSENT),
        )
        .expect("a brain")
    }

    #[test]
    fn a_sweep_of_a_store_with_no_mail_asks_for_no_model() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let mut brain = brain_at(&dir);

        let done = brain.sweep(&store, |_| {}).expect("a sweep");

        assert_eq!(done, Embedded::default());
        assert!(!brain.model.is_loaded());
    }

    /// §6.2: a mailbox of 100000 messages does not go through the
    /// model again when one letter arrives. A sync of a mailbox that
    /// did not change must therefore load nothing at all.
    #[test]
    fn a_sweep_that_has_nothing_to_do_asks_for_no_model() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let mut brain = brain_at(&dir);
        write(&store, "a", "The deposit is due.");

        let made = plan(&store, ABSENT).expect("a plan");
        run(
            &store,
            &brain.db,
            &made,
            |_| {},
            &mut fake(&brain.db, &mut vec![]),
        )
        .expect("a pass");

        let done = brain.sweep(&store, |_| {}).expect("a sweep");

        assert_eq!(done, Embedded::default());
        assert!(!brain.model.is_loaded());
    }

    /// A sweep that only has passages to drop asks for no model
    /// either, because the model has nothing left to read.
    #[test]
    fn a_sweep_that_only_drops_passages_asks_for_no_model() {
        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let mut brain = brain_at(&dir);
        let id = write(&store, "a", "The deposit is due.");

        let made = plan(&store, ABSENT).expect("a plan");
        let keys: Vec<u64> =
            made.work[0].passages.iter().map(|one| one.key).collect();
        run(
            &store,
            &brain.db,
            &made,
            |_| {},
            &mut fake(&brain.db, &mut vec![]),
        )
        .expect("a pass");
        store.remove(&id).expect("a delete");

        let done = brain.sweep(&store, |_| {}).expect("a sweep");

        assert_eq!(done.dropped, keys.len());
        assert_eq!(done.messages, 0);
        assert!(!brain.model.is_loaded());
        assert!(!brain.at.exists(), "an empty index stayed on the disk");
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 40)]
    fn prop_a_second_plan_asks_for_nothing(tc: TestCase) {
        let bodies: Vec<String> = tc.draw(a_mailbox());

        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        for (at, body) in bodies.iter().enumerate() {
            write(&store, &format!("m{at}"), body);
        }

        let first = plan(&store, MODEL).expect("a plan");
        for work in &first.work {
            record(&store, work).expect("a record");
        }

        assert!(plan(&store, MODEL).expect("a plan").is_empty());
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_plan_covers_every_message_of_the_store(tc: TestCase) {
        let bodies: Vec<String> = tc.draw(a_mailbox());

        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let mut wanted = BTreeSet::new();
        for (at, body) in bodies.iter().enumerate() {
            wanted.insert(write(&store, &format!("m{at}"), body));
        }

        let made = plan(&store, MODEL).expect("a plan");
        let named: BTreeSet<MessageId> =
            made.work.iter().map(|one| one.id).collect();

        assert_eq!(named, wanted);
        assert!(made.passages() >= wanted.len());
    }

    #[hegel::test(test_cases = 40)]
    fn prop_a_message_that_went_away_takes_its_passages(tc: TestCase) {
        let bodies: Vec<String> = tc.draw(a_mailbox());

        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let mut ids = Vec::new();
        for (at, body) in bodies.iter().enumerate() {
            ids.push(write(&store, &format!("m{at}"), body));
        }

        let first = plan(&store, MODEL).expect("a plan");
        for work in &first.work {
            record(&store, work).expect("a record");
        }

        let gone = ids[tc.draw(
            gs::integers::<usize>()
                .min_value(0)
                .max_value(ids.len() - 1),
        )];
        let keys: Vec<u64> = first
            .work
            .iter()
            .find(|one| one.id == gone)
            .expect("the work")
            .passages
            .iter()
            .map(|one| one.key)
            .collect();

        store.remove(&gone).expect("a delete");
        let second = plan(&store, MODEL).expect("a plan");

        assert_eq!(second.stale, keys);
        for key in &keys {
            assert_eq!(store.owner(*key).expect("a read"), None);
        }
    }

    #[hegel::test(test_cases = 20)]
    fn prop_the_leg_keeps_what_the_filter_allows(tc: TestCase) {
        let count: usize =
            tc.draw(gs::integers::<usize>().min_value(2).max_value(5));

        let dir = tempdir().expect("a directory");
        let store = open_at(&dir);
        let db = open_db(&dir);
        let at = dir.path().join("plaid.idx");

        let mut ids = Vec::new();
        for one in 0..count {
            ids.push(write(&store, &format!("m{one}"), "The deposit is due."));
        }

        // Each message leans a little further from the query, so the
        // ranking is not a tie and the filter has something to cut.
        let made = plan(&store, MODEL).expect("a plan");
        let places: Vec<MessageId> = ids.clone();
        seed(&store, &db, &made, |id| {
            let step = places.iter().position(|one| one == id).unwrap_or(0);

            [1.0, step as f32 * 0.1]
        });
        rebuild(&db, &at, &made, small()).expect("a build");
        let index = load(&at).expect("a read").expect("an index");

        let mut allow = BTreeSet::new();
        for id in &ids {
            if tc.draw(gs::booleans()) {
                allow.insert(id.numeric());
            }
        }

        let ranked = leg(&index, &store, &query(NEAR), Some(&allow), count)
            .expect("a leg");

        for one in &ranked {
            assert!(allow.contains(one), "the filter let {one} through");
        }
    }

    /// A few short bodies, one for each message of a mailbox.
    #[hegel::composite]
    fn a_mailbox(tc: TestCase) -> Vec<String> {
        tc.draw(
            gs::vecs(gs::text().alphabet("abc ").min_size(1).max_size(30))
                .min_size(1)
                .max_size(5),
        )
    }
}
