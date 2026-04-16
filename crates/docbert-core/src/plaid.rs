//! Bridge between docbert's embedding database and the `docbert-plaid`
//! crate.
//!
//! This module is thin by design: it translates between docbert's native
//! types (`EmbeddingDb`, `DataDir`, candle tensors) and `docbert-plaid`'s
//! algorithm-facing types (`DocumentTokens`, `Index`, flat `&[f32]`).
//! The actual PLAID machinery (k-means, residual codec, IVF, search) all
//! lives in `docbert-plaid`; nothing in this file is clever.
//!
//! Lifecycle:
//!
//! - [`build_index_from_embedding_db`] scans every stored embedding,
//!   flattens it into a `DocumentTokens` row, and hands the whole corpus
//!   to `docbert_plaid::index::build_index`.
//! - [`save_index`] / [`load_index`] persist the result under
//!   [`crate::DataDir::plaid_index`].
//! - [`search`] runs a query tensor through the stored index and returns
//!   ranked `(doc_id, score)` pairs.
//!
//! Wiring the semantic leg in `search.rs` to prefer the PLAID index when
//! one is present is a separate, follow-up change; today this module
//! only exposes the pieces that change needs.

use candle_core::Tensor;
use docbert_plaid::{
    index::{
        self as plaid_index,
        DocumentTokens,
        Index as PlaidIndex,
        IndexParams,
    },
    persistence,
    search::{self as plaid_search, SearchParams},
    update::{self as plaid_update, IndexUpdate},
};

use crate::{
    data_dir::DataDir,
    embedding_db::EmbeddingDb,
    error::{Error, Result},
};

/// Parameters controlling how the PLAID index is trained over docbert's
/// embeddings.
///
/// The defaults (via [`PlaidBuildParams::default`]) are tuned for a
/// small personal-notes corpus: 256 coarse centroids with 2-bit residual
/// quantization and a generous k-means iteration cap. Larger corpora
/// typically want proportionally more centroids.
#[derive(Debug, Clone, Copy)]
pub struct PlaidBuildParams {
    /// Number of coarse centroids trained by k-means.
    pub k_centroids: usize,
    /// Residual quantization bit-width per dimension (typically 2 or 4).
    pub nbits: u32,
    /// Upper bound on k-means iterations during training.
    pub max_kmeans_iters: usize,
}

impl Default for PlaidBuildParams {
    fn default() -> Self {
        Self {
            k_centroids: 256,
            nbits: 2,
            max_kmeans_iters: 20,
        }
    }
}

/// A scored document returned by a PLAID search over docbert's corpus.
///
/// This mirrors `docbert_plaid::search::SearchResult` but is re-exported
/// so callers don't have to reach into the inner crate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaidResult {
    pub doc_id: u64,
    pub score: f32,
}

/// Build a PLAID index over every embedding currently stored in
/// `embedding_db`.
///
/// All embeddings must share the same dimensionality (enforced here with
/// a clear error; normally guaranteed by the encoder). Empty databases
/// produce a [`Error::Config`] rather than a panic so callers can treat
/// "nothing indexed yet" as a recoverable state.
pub fn build_index_from_embedding_db(
    embedding_db: &EmbeddingDb,
    params: PlaidBuildParams,
) -> Result<PlaidIndex> {
    let ids = embedding_db.list_ids()?;
    if ids.is_empty() {
        return Err(Error::Config(
            "cannot build PLAID index: embedding_db is empty".to_string(),
        ));
    }

    let mut documents: Vec<DocumentTokens> = Vec::with_capacity(ids.len());
    let mut dim: Option<usize> = None;
    let mut total_tokens: usize = 0;

    for id in ids {
        let Some(matrix) = embedding_db.load(id)? else {
            continue;
        };
        let this_dim = matrix.dimension as usize;
        match dim {
            None => dim = Some(this_dim),
            Some(d) if d == this_dim => {}
            Some(d) => {
                return Err(Error::Config(format!(
                    "cannot build PLAID index: embedding {id} has dim \
                     {this_dim}, expected {d}",
                )));
            }
        }
        let n_tokens = matrix.num_tokens as usize;
        total_tokens += n_tokens;
        documents.push(DocumentTokens {
            doc_id: id,
            tokens: matrix.data,
            n_tokens,
        });
    }

    let dim = dim.ok_or_else(|| {
        Error::Config(
            "cannot build PLAID index: no loadable embeddings".to_string(),
        )
    })?;

    if total_tokens < params.k_centroids {
        return Err(Error::Config(format!(
            "cannot build PLAID index: need at least {} tokens for \
             {} centroids, got {total_tokens}",
            params.k_centroids, params.k_centroids,
        )));
    }

    let index_params = IndexParams {
        dim,
        nbits: params.nbits,
        k_centroids: params.k_centroids,
        max_kmeans_iters: params.max_kmeans_iters,
    };

    Ok(plaid_index::build_index(&documents, index_params))
}

/// Apply the given sync deltas to `existing`, reusing its codec.
///
/// `changed_ids` are doc_ids whose embeddings have changed since the
/// last build (treated as upserts: the old entry is removed, new
/// tokens are read from `embedding_db` and encoded against the
/// existing codec). `deleted_ids` are removed outright.
///
/// Returns [`Error::Config`] if any `changed_ids` entry is missing
/// from `embedding_db`, or if the stored dimensionality disagrees
/// with the existing index — the caller is responsible for falling
/// back to a full rebuild in those cases.
///
/// Does not re-train centroids or the residual codec. For a corpus
/// whose distribution hasn't drifted much, this is strictly cheaper
/// than [`build_index_from_embedding_db`] since k-means and quantizer
/// training are skipped; the heavy work reduces to encoding just the
/// `changed_ids` tokens.
pub fn update_index_from_embedding_db(
    embedding_db: &EmbeddingDb,
    existing: PlaidIndex,
    changed_ids: &[u64],
    deleted_ids: &[u64],
) -> Result<PlaidIndex> {
    let dim = existing.params.dim;
    let mut upserts: Vec<DocumentTokens> =
        Vec::with_capacity(changed_ids.len());
    for &id in changed_ids {
        let Some(matrix) = embedding_db.load(id)? else {
            return Err(Error::Config(format!(
                "cannot update PLAID index: changed doc_id {id} not found \
                 in embedding_db",
            )));
        };
        let this_dim = matrix.dimension as usize;
        if this_dim != dim {
            return Err(Error::Config(format!(
                "cannot update PLAID index: doc {id} has dim {this_dim} \
                 but index expects {dim}",
            )));
        }
        upserts.push(DocumentTokens {
            doc_id: id,
            tokens: matrix.data,
            n_tokens: matrix.num_tokens as usize,
        });
    }

    Ok(plaid_update::apply_update(
        existing,
        IndexUpdate {
            deletions: deleted_ids,
            upserts: &upserts,
        },
    ))
}

/// Incrementally sync `existing` against `embedding_db` given the set
/// of *base* doc_ids that were re-embedded during this sync pass.
///
/// The bridge handles the two pieces of bookkeeping the CLI would
/// otherwise have to duplicate per call site:
///
/// 1. Each touched base is expanded into its current chunk doc_ids
///    via [`EmbeddingDb::batch_load_document_families`]. Every such
///    chunk is re-encoded against the existing codec — this covers
///    both "file had the same chunking" (chunk ids match, tokens may
///    have shifted) and "file re-chunked" (some new ids, some gone).
/// 2. Deletions are computed as the set of doc_ids present in
///    `existing` but absent from `embedding_db.list_ids()`. This
///    catches both explicitly-deleted base families and any orphan
///    chunks left behind when a changed file ended up with fewer
///    chunks than last time.
///
/// Returns [`Error::Config`] on dim mismatch, mirroring
/// [`update_index_from_embedding_db`]. Returns a fresh `PlaidIndex`
/// with the same codec as `existing`.
pub fn update_index_for_touched_bases(
    embedding_db: &EmbeddingDb,
    existing: PlaidIndex,
    touched_bases: &[u64],
) -> Result<PlaidIndex> {
    use std::collections::HashSet;

    let dim = existing.params.dim;

    // Expand touched bases into (id, matrix) pairs. Uses one read
    // transaction over the whole set.
    let family_chunks =
        embedding_db.batch_load_document_families(touched_bases)?;

    let mut upsert_docs: Vec<DocumentTokens> =
        Vec::with_capacity(family_chunks.len());
    let mut upsert_ids: HashSet<u64> =
        HashSet::with_capacity(family_chunks.len());
    for (id, matrix) in family_chunks {
        let this_dim = matrix.dimension as usize;
        if this_dim != dim {
            return Err(Error::Config(format!(
                "cannot update PLAID index: doc {id} has dim {this_dim} \
                 but index expects {dim}",
            )));
        }
        if !upsert_ids.insert(id) {
            // Duplicate within one family would violate apply_update's
            // "unique upsert ids" contract; surface it explicitly.
            return Err(Error::Config(format!(
                "embedding_db returned doc_id {id} twice in a single \
                 family enumeration",
            )));
        }
        upsert_docs.push(DocumentTokens {
            doc_id: id,
            tokens: matrix.data,
            n_tokens: matrix.num_tokens as usize,
        });
    }

    // Deletions: anything the old index remembers that the db no
    // longer has. Upserts win if a chunk is both re-encoded and
    // somehow also missing from the db — but that can't happen
    // since upsert_ids came from the db in this same read.
    let current_db_ids: HashSet<u64> =
        embedding_db.list_ids()?.into_iter().collect();
    let deletions: Vec<u64> = existing
        .doc_ids
        .iter()
        .copied()
        .filter(|id| !current_db_ids.contains(id))
        .filter(|id| !upsert_ids.contains(id))
        .collect();

    Ok(plaid_update::apply_update(
        existing,
        IndexUpdate {
            upserts: &upsert_docs,
            deletions: &deletions,
        },
    ))
}

/// Write `index` to the canonical PLAID index path under `data_dir`.
pub fn save_index(index: &PlaidIndex, data_dir: &DataDir) -> Result<()> {
    persistence::save(index, &data_dir.plaid_index())?;
    Ok(())
}

/// Load the PLAID index from `data_dir` if one has been built.
///
/// Returns `Ok(None)` when the file does not exist — callers typically
/// treat that as "fall back to the linear-scan semantic leg".
pub fn load_index(data_dir: &DataDir) -> Result<Option<PlaidIndex>> {
    let path = data_dir.plaid_index();
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(persistence::load(&path)?))
}

/// Search `index` with a ColBERT-shaped `query_embedding` and return the
/// top-`top_k` `(doc_id, score)` pairs.
///
/// `query_embedding` must be a 2-D tensor of shape `[n_query_tokens, dim]`
/// — i.e. the direct output of `ModelManager::encode_query`. The tensor
/// is flattened to a contiguous f32 buffer and handed to
/// `docbert_plaid::search::search` unchanged.
pub fn search(
    index: &PlaidIndex,
    query_embedding: &Tensor,
    top_k: usize,
    n_probe: usize,
) -> Result<Vec<PlaidResult>> {
    let (query_tokens, query_dim) = query_embedding.dims2()?;
    if query_dim != index.params.dim {
        return Err(Error::Config(format!(
            "PLAID query has dim {query_dim} but index expects {}",
            index.params.dim,
        )));
    }
    let query_flat: Vec<f32> = query_embedding
        .contiguous()?
        .to_device(&candle_core::Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    debug_assert_eq!(query_flat.len(), query_tokens * query_dim);

    let params = SearchParams { top_k, n_probe };
    let out = plaid_search::search(index, &query_flat, params);
    Ok(out
        .into_iter()
        .map(|r| PlaidResult {
            doc_id: r.doc_id,
            score: r.score,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seed an embedding db with three tiny, well-separated documents
    /// (two in one cluster, one in another).
    fn seed_small_db(db: &EmbeddingDb) {
        // doc 1: two tokens in "origin" cluster
        db.store(1, 2, 2, &[0.0, 0.0, 0.1, 0.1]).unwrap();
        // doc 2: two tokens in "far" cluster
        db.store(2, 2, 2, &[10.0, 10.0, 10.1, 9.9]).unwrap();
        // doc 3: one token near origin
        db.store(3, 1, 2, &[0.05, -0.05]).unwrap();
    }

    fn small_build_params() -> PlaidBuildParams {
        PlaidBuildParams {
            k_centroids: 2,
            nbits: 2,
            max_kmeans_iters: 50,
        }
    }

    #[test]
    fn build_from_empty_db_returns_config_error() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();

        let err = build_index_from_embedding_db(&db, small_build_params())
            .unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn build_index_reads_every_stored_document() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        seed_small_db(&db);

        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();
        assert_eq!(index.num_documents(), 3);
        let mut ids = index.doc_ids.clone();
        ids.sort();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn build_rejects_mixed_dimensionalities() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        db.store(1, 1, 2, &[0.0, 0.0]).unwrap();
        db.store(2, 1, 3, &[1.0, 2.0, 3.0]).unwrap();

        let err = build_index_from_embedding_db(&db, small_build_params())
            .unwrap_err();
        match err {
            Error::Config(msg) => {
                assert!(msg.contains("dim"), "unexpected message: {msg}");
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn build_rejects_when_total_tokens_below_k_centroids() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        db.store(1, 1, 2, &[0.0, 0.0]).unwrap();

        let mut params = small_build_params();
        params.k_centroids = 16;
        let err = build_index_from_embedding_db(&db, params).unwrap_err();
        match err {
            Error::Config(msg) => assert!(msg.contains("centroids")),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn save_and_load_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let db = EmbeddingDb::open(&data_dir.embeddings_db()).unwrap();
        seed_small_db(&db);

        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();
        save_index(&index, &data_dir).unwrap();

        let loaded = load_index(&data_dir).unwrap().expect("present");
        assert_eq!(loaded.doc_ids, index.doc_ids);
        assert_eq!(loaded.codec.centroids, index.codec.centroids);
    }

    #[test]
    fn load_returns_none_when_no_index_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        assert!(load_index(&data_dir).unwrap().is_none());
    }

    #[test]
    fn search_returns_nearest_cluster_document() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        seed_small_db(&db);
        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();

        // A 1×2 query near the origin cluster.
        let query = Tensor::from_slice(
            &[0.0f32, 0.0],
            (1, 2),
            &candle_core::Device::Cpu,
        )
        .unwrap();

        let results = search(&index, &query, 2, 2).unwrap();
        assert!(!results.is_empty());
        // Doc 1 or doc 3 lives in this cluster; doc 2 is far.
        assert!(results[0].doc_id == 1 || results[0].doc_id == 3);
    }

    #[test]
    fn update_with_no_changes_returns_equivalent_index() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        seed_small_db(&db);
        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();
        let before_ids = index.doc_ids.clone();
        let before_codec = index.codec.clone();

        let updated =
            update_index_from_embedding_db(&db, index, &[], &[]).unwrap();

        assert_eq!(updated.doc_ids, before_ids);
        assert_eq!(updated.codec.centroids, before_codec.centroids);
        assert_eq!(updated.codec.bucket_cutoffs, before_codec.bucket_cutoffs);
        assert_eq!(updated.codec.bucket_weights, before_codec.bucket_weights);
    }

    #[test]
    fn update_preserves_existing_codec_when_adding_a_new_doc() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        seed_small_db(&db);
        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();
        let before_codec = index.codec.clone();

        // Introduce a new document after the codec has been frozen.
        db.store(4, 1, 2, &[0.2, -0.3]).unwrap();

        let updated =
            update_index_from_embedding_db(&db, index, &[4], &[]).unwrap();

        assert!(updated.doc_ids.contains(&4));
        assert_eq!(updated.codec.centroids, before_codec.centroids);
        assert_eq!(updated.codec.bucket_cutoffs, before_codec.bucket_cutoffs);
        assert_eq!(updated.codec.bucket_weights, before_codec.bucket_weights);
    }

    #[test]
    fn update_removes_deleted_docs() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        seed_small_db(&db);
        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();

        let updated =
            update_index_from_embedding_db(&db, index, &[], &[2]).unwrap();

        assert!(!updated.doc_ids.contains(&2));
        assert!(updated.doc_ids.contains(&1));
        assert!(updated.doc_ids.contains(&3));
    }

    #[test]
    fn update_reads_replacement_tokens_for_changed_id_from_db() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        seed_small_db(&db);
        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();
        let old_encoded =
            index.doc_tokens[index.position_of(1).unwrap()].clone();

        // Overwrite doc 1's embedding with tokens from the far cluster.
        // After update, the encoded centroid_id should flip — proving
        // the bridge re-read tokens from the db rather than reusing
        // the cached encoding.
        db.store(1, 2, 2, &[10.0, 10.0, 10.1, 9.9]).unwrap();

        let updated =
            update_index_from_embedding_db(&db, index, &[1], &[]).unwrap();

        let pos = updated.position_of(1).expect("doc 1 must survive update");
        let new_encoded = &updated.doc_tokens[pos];
        assert_eq!(new_encoded.len(), 2);
        assert_ne!(
            *new_encoded, old_encoded,
            "update must re-encode from the db's current tokens",
        );
    }

    #[test]
    fn update_errors_when_changed_id_is_missing_from_db() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        seed_small_db(&db);
        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();

        let err = update_index_from_embedding_db(&db, index, &[999], &[])
            .unwrap_err();
        match err {
            Error::Config(msg) => {
                assert!(
                    msg.contains("999"),
                    "message should cite the id: {msg}"
                );
                assert!(msg.contains("not found"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn update_errors_on_dim_mismatch_between_index_and_db() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        seed_small_db(&db); // dim=2 entries
        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();

        // Insert a 3-D embedding that the 2-D index can't possibly
        // encode against its trained centroids.
        db.store(42, 1, 3, &[1.0, 2.0, 3.0]).unwrap();

        let err =
            update_index_from_embedding_db(&db, index, &[42], &[]).unwrap_err();
        match err {
            Error::Config(msg) => {
                assert!(
                    msg.contains("dim"),
                    "message should mention dim: {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn touched_bases_upsert_rewrites_a_single_chunk_family() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        seed_small_db(&db);
        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();
        let old_encoded =
            index.doc_tokens[index.position_of(1).unwrap()].clone();

        // Simulate a re-embedding: the file for base doc_id 1 got
        // different tokens (single-chunk family).
        db.store(1, 2, 2, &[10.0, 10.0, 10.1, 9.9]).unwrap();

        let updated = update_index_for_touched_bases(&db, index, &[1]).unwrap();

        let pos = updated.position_of(1).expect("doc 1 survives upsert");
        assert_ne!(
            updated.doc_tokens[pos], old_encoded,
            "the touched base must be re-encoded with the new tokens",
        );
    }

    #[test]
    fn touched_bases_with_no_touches_only_prunes_orphans() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        seed_small_db(&db);
        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();
        let before_codec = index.codec.clone();

        // Remove doc 2 directly from the db to simulate a sync that
        // ran before we got here. No bases were "touched" by this
        // hypothetical sync — the doc was just deleted.
        db.remove(2).unwrap();

        let updated = update_index_for_touched_bases(&db, index, &[]).unwrap();

        assert!(!updated.doc_ids.contains(&2));
        assert_eq!(updated.codec.centroids, before_codec.centroids);
    }

    #[test]
    fn touched_bases_drops_chunks_that_disappeared_during_re_embedding() {
        use crate::chunking::chunk_doc_id;

        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        // Seed with a few docs so k-means has enough training data.
        seed_small_db(&db);

        // Add a two-chunk family for base doc_id 100.
        let base = 100u64;
        let chunk_a = base; // chunk_doc_id(base, 0) == base
        let chunk_b = chunk_doc_id(base, 1);
        db.store(chunk_a, 1, 2, &[0.2, 0.2]).unwrap();
        db.store(chunk_b, 1, 2, &[-0.1, 0.3]).unwrap();

        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();
        assert!(index.doc_ids.contains(&chunk_a));
        assert!(index.doc_ids.contains(&chunk_b));

        // Simulate a re-embedding: the file for base 100 now produces
        // only one chunk (chunk_a), with different tokens. chunk_b
        // is orphaned in the index and must be deleted.
        db.remove(chunk_b).unwrap();
        db.store(chunk_a, 1, 2, &[9.9, 10.0]).unwrap();

        let updated =
            update_index_for_touched_bases(&db, index, &[base]).unwrap();

        assert!(
            updated.doc_ids.contains(&chunk_a),
            "surviving chunk must still be present",
        );
        assert!(
            !updated.doc_ids.contains(&chunk_b),
            "chunk that disappeared from the db must be removed from the index",
        );
    }

    #[test]
    fn touched_bases_errors_on_dim_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        seed_small_db(&db); // dim=2 entries
        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();

        // Replace doc 1's family with a 3-D embedding — incompatible
        // with the 2-D codec on the existing index.
        db.remove(1).unwrap();
        db.store(1, 1, 3, &[1.0, 2.0, 3.0]).unwrap();

        let err = update_index_for_touched_bases(&db, index, &[1]).unwrap_err();
        match err {
            Error::Config(msg) => assert!(
                msg.contains("dim"),
                "expected dim-related message, got: {msg}"
            ),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn touched_bases_preserves_untouched_chunks_verbatim() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        seed_small_db(&db);
        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();
        // Snapshot the encoded tokens for docs we DON'T touch.
        let untouched_2 =
            index.doc_tokens[index.position_of(2).unwrap()].clone();
        let untouched_3 =
            index.doc_tokens[index.position_of(3).unwrap()].clone();

        // Touch only doc 1 — 2 and 3 must carry through unchanged.
        db.store(1, 1, 2, &[0.15, -0.15]).unwrap();

        let updated = update_index_for_touched_bases(&db, index, &[1]).unwrap();

        assert_eq!(
            updated.doc_tokens[updated.position_of(2).unwrap()],
            untouched_2,
        );
        assert_eq!(
            updated.doc_tokens[updated.position_of(3).unwrap()],
            untouched_3,
        );
    }

    #[test]
    fn search_rejects_dim_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        seed_small_db(&db);
        let index =
            build_index_from_embedding_db(&db, small_build_params()).unwrap();

        // Index was trained with dim=2; query has dim=3.
        let bad_query = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0],
            (1, 3),
            &candle_core::Device::Cpu,
        )
        .unwrap();

        let err = search(&index, &bad_query, 1, 1).unwrap_err();
        match err {
            Error::Config(msg) => assert!(msg.contains("dim")),
            other => panic!("expected Config error, got {other:?}"),
        }
    }
}
