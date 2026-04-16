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
