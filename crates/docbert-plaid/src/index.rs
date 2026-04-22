//! Index construction: turn a corpus of token embeddings into a searchable
//! PLAID index.
//!
//! `build_index` ties together the three lower layers:
//!
//! 1. Flatten every document's token matrix into one big cloud of points.
//! 2. Run k-means to pick coarse centroids ([`crate::kmeans::fit`]).
//! 3. Assign every token to a centroid, compute residuals, train cutoffs
//!    and weights on the residuals ([`crate::codec::train_quantizer`]),
//!    and encode each token against the fresh codec.
//!
//! The resulting [`Index`] keeps one [`EncodedVector`] per original token
//! plus the per-document `doc_id` bookkeeping. Inverted-file construction
//! and the query-time search path will be added on top in later
//! TDD cycles.

use crate::{
    Result,
    codec::{EncodedVector, ResidualCodec, train_quantizer},
    kmeans::{assign_points, fit},
};

/// Cap on tokens used to train the residual quantizer.
///
/// Quantile estimation for `2^nbits` bucket cutoffs converges well
/// below 100k samples; training on the full corpus (potentially
/// millions of tokens × the embedding dim, so gigabytes of residuals)
/// adds no statistical benefit while pushing peak RSS past what a
/// host can absorb on large collections. 65,536 tokens × 128 dims is
/// ~8M residual values, which still over-samples every cutoff by
/// several orders of magnitude and sorts in under a second.
const MAX_QUANTIZER_TRAINING_TOKENS: usize = 65_536;

/// Inverted file: for each centroid, the sorted list of unique document
/// indices that have at least one token clustered in that centroid.
///
/// This mirrors PLAID's "centroid → unique passage ids" layout from
/// §3 of the paper: candidate generation only needs to know which
/// documents are reachable via a probed centroid, and deduplicating
/// per-doc keeps the posting lists small even when a single document
/// has many tokens mapped to the same cluster.
///
/// The search path uses this to expand a query token to a shortlist of
/// document candidates: find the centroids with the highest dot-product
/// against the query token, then gather every document listed under
/// those centroids.
#[derive(Debug, Clone, Default)]
pub struct InvertedFile {
    /// `lists[c]` holds the sorted, deduplicated `doc_idx`s of every
    /// document with at least one token assigned to centroid `c`.
    pub lists: Vec<Vec<u32>>,
}

impl InvertedFile {
    /// Total number of centroids the IVF spans.
    pub fn num_centroids(&self) -> usize {
        self.lists.len()
    }

    /// Document indices currently associated with `centroid_id`, or an
    /// empty slice if the centroid is out of range. Entries are sorted
    /// ascending and contain no duplicates.
    pub fn docs_for_centroid(&self, centroid_id: usize) -> &[u32] {
        self.lists
            .get(centroid_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Total number of (centroid, doc) postings across every list.
    ///
    /// This is the sum of `lists[c].len()` over all centroids `c`. It
    /// is at most `num_centroids * num_documents` and at least equal
    /// to the number of documents that contain any tokens at all.
    pub fn total_doc_postings(&self) -> usize {
        self.lists.iter().map(Vec::len).sum()
    }
}

/// A single document's worth of token embeddings, ready to index.
///
/// `tokens` is a flat row-major `n_tokens × dim` buffer. Keeping the
/// tokens flat mirrors the way docbert already stores ColBERT outputs in
/// `embeddings.db` and avoids an intermediate `Vec<Vec<f32>>` allocation.
#[derive(Debug, Clone)]
pub struct DocumentTokens {
    pub doc_id: u64,
    pub tokens: Vec<f32>,
    pub n_tokens: usize,
}

impl DocumentTokens {
    /// Total number of f32 values this document contributes.
    pub fn flat_len(&self) -> usize {
        self.tokens.len()
    }
}

/// Parameters that control how an [`Index`] is built.
#[derive(Debug, Clone, Copy)]
pub struct IndexParams {
    /// Dimensionality of each token embedding.
    pub dim: usize,
    /// Number of bits per residual dimension (typically 2 or 4).
    pub nbits: u32,
    /// Number of coarse centroids (k in k-means).
    pub k_centroids: usize,
    /// Maximum iterations for k-means clustering.
    pub max_kmeans_iters: usize,
}

/// A fully-built PLAID index over a corpus of multi-vector embeddings.
///
/// Stored in a **flat, StridedTensor-style layout** — every document's
/// encoded tokens live in one of two big contiguous buffers,
/// `doc_centroid_ids` (one u32 per token) and `doc_residual_bytes`
/// (`packed_bytes_per_token` u8s per token). Per-document slicing
/// happens through the precomputed `doc_offsets` cumulative lengths.
///
/// This layout matches fast-plaid's `StridedTensor` and buys three
/// things over the older `Vec<Vec<EncodedVector>>` we used to keep:
///
/// 1. Search doesn't have to walk per-document `Vec`s on every query
///    to gather codes and residuals — it slices into the flat buffer.
/// 2. Peak RAM drops: 6.8M tokens at nbits=2 was ~500 MiB of heap
///    for the old nested-Vec headers; the flat layout is ~240 MiB.
/// 3. GPU decode can `Tensor::from_slice` the whole slice for a batch
///    of candidates in one kernel launch instead of one-per-token.
///
/// Callers that want the per-token view use [`Index::doc_centroid_ids`]
/// / [`Index::doc_residual_bytes`] — slices into the flat buffers with
/// no allocation.
#[derive(Debug, Clone)]
pub struct Index {
    pub params: IndexParams,
    pub codec: ResidualCodec,
    pub doc_ids: Vec<u64>,
    /// Flat `[total_tokens]` vector of per-token centroid indices.
    pub doc_centroid_ids: Vec<u32>,
    /// Flat `[total_tokens * packed_bytes_per_token]` residual bytes,
    /// row-major — each `packed_bytes_per_token`-long slice is one
    /// token's packed residual.
    pub doc_residual_bytes: Vec<u8>,
    /// Cumulative per-document token counts; length `num_docs + 1`,
    /// `doc_offsets[i + 1] - doc_offsets[i]` = `n_tokens` for doc `i`.
    pub doc_offsets: Vec<usize>,
    /// Centroid → tokens inverted file used for candidate generation.
    pub ivf: InvertedFile,
}

impl Index {
    /// Construct an `Index` from per-document `EncodedVector`s.
    ///
    /// Convenience for tests, [`crate::update::apply_update`], and
    /// [`crate::persistence`]'s legacy-format loader: they still think
    /// in terms of `Vec<Vec<EncodedVector>>`, and this flattens that
    /// into the canonical [`Index`] layout in one pass.
    ///
    /// `ivf` should already reflect the `doc_tokens` contents — this
    /// helper does not recompute the inverted file, only the flat
    /// per-token buffers.
    pub fn from_encoded_docs(
        params: IndexParams,
        codec: ResidualCodec,
        doc_ids: Vec<u64>,
        doc_tokens: Vec<Vec<EncodedVector>>,
        ivf: InvertedFile,
    ) -> Self {
        let packed_bytes = codec.packed_bytes();
        let total_tokens: usize = doc_tokens.iter().map(Vec::len).sum();
        let mut doc_centroid_ids: Vec<u32> = Vec::with_capacity(total_tokens);
        let mut doc_residual_bytes: Vec<u8> =
            Vec::with_capacity(total_tokens * packed_bytes);
        let mut doc_offsets: Vec<usize> =
            Vec::with_capacity(doc_tokens.len() + 1);
        doc_offsets.push(0);
        for tokens in &doc_tokens {
            for ev in tokens {
                doc_centroid_ids.push(ev.centroid_id);
                debug_assert_eq!(ev.codes.len(), packed_bytes);
                doc_residual_bytes.extend_from_slice(&ev.codes);
            }
            doc_offsets.push(doc_centroid_ids.len());
        }

        Self {
            params,
            codec,
            doc_ids,
            doc_centroid_ids,
            doc_residual_bytes,
            doc_offsets,
            ivf,
        }
    }

    /// Number of documents currently stored in the index.
    pub fn num_documents(&self) -> usize {
        self.doc_ids.len()
    }

    /// Total number of encoded tokens across all documents.
    pub fn num_tokens(&self) -> usize {
        self.doc_centroid_ids.len()
    }

    /// Find the position of a document inside [`Index::doc_ids`].
    pub fn position_of(&self, doc_id: u64) -> Option<usize> {
        self.doc_ids.iter().position(|id| *id == doc_id)
    }

    /// Number of encoded tokens for the `idx`-th document.
    pub fn doc_token_count(&self, idx: usize) -> usize {
        self.doc_offsets[idx + 1] - self.doc_offsets[idx]
    }

    /// Slice of per-token centroid indices for the `idx`-th document.
    pub fn doc_centroid_ids(&self, idx: usize) -> &[u32] {
        &self.doc_centroid_ids[self.doc_offsets[idx]..self.doc_offsets[idx + 1]]
    }

    /// Slice of packed residual bytes for the `idx`-th document,
    /// row-major `[n_tokens, packed_bytes_per_token]`.
    pub fn doc_residual_bytes(&self, idx: usize) -> &[u8] {
        let pb = self.codec.packed_bytes();
        let start = self.doc_offsets[idx] * pb;
        let end = self.doc_offsets[idx + 1] * pb;
        &self.doc_residual_bytes[start..end]
    }

    /// Reconstruct the `idx`-th document's tokens as an owned
    /// `Vec<EncodedVector>`.
    ///
    /// Allocation-heavy — prefer [`Index::doc_centroid_ids`] /
    /// [`Index::doc_residual_bytes`] on the hot path. Provided as a
    /// compatibility shim for [`crate::update::apply_update`] and
    /// legacy callers that still work in `EncodedVector` terms.
    pub fn doc_tokens_vec(&self, idx: usize) -> Vec<EncodedVector> {
        let pb = self.codec.packed_bytes();
        let cids = self.doc_centroid_ids(idx);
        let res = self.doc_residual_bytes(idx);
        (0..cids.len())
            .map(|i| EncodedVector {
                centroid_id: cids[i],
                codes: res[i * pb..(i + 1) * pb].to_vec(),
            })
            .collect()
    }
}

/// Build a [`Index`] from a corpus of documents.
///
/// Every document must share the same embedding dimensionality as
/// `params.dim`. Documents with zero tokens are preserved in the index —
/// they contribute nothing to centroid/codec training but still occupy a
/// slot in `doc_ids` so callers can resolve their position by `doc_id`
/// later.
///
/// # Errors
///
/// Returns [`PlaidError::Tensor`] if the matmul-driven k-means
/// training or nearest-centroid assignment fails.
///
/// # Panics
///
/// Panics if any document's flat length is not a multiple of `dim`, if
/// the total number of tokens is smaller than `params.k_centroids`, or
/// if `params.k_centroids == 0`.
///
/// [`PlaidError::Tensor`]: crate::PlaidError::Tensor
pub fn build_index(
    documents: &[DocumentTokens],
    params: IndexParams,
) -> Result<Index> {
    assert!(params.dim > 0, "build_index: dim must be positive");
    assert!(
        params.k_centroids > 0,
        "build_index: k_centroids must be positive"
    );
    assert!(
        params.nbits > 0 && params.nbits <= 8,
        "build_index: nbits must be in 1..=8, got {}",
        params.nbits,
    );

    for doc in documents {
        assert!(
            doc.tokens.len() == doc.n_tokens * params.dim,
            "build_index: doc {} declared {} tokens but carries {} f32s (dim={})",
            doc.doc_id,
            doc.n_tokens,
            doc.tokens.len(),
            params.dim,
        );
    }

    // Flatten all token embeddings into one training cloud and forward
    // to the pool-based core path. This mirrors the memory-lean path
    // used by `build_index_from_pool` callers that already own a
    // contiguous buffer.
    let total_tokens: usize = documents.iter().map(|d| d.n_tokens).sum();
    let mut pool: Vec<f32> = Vec::with_capacity(total_tokens * params.dim);
    let mut doc_meta: Vec<(u64, usize)> = Vec::with_capacity(documents.len());
    for doc in documents {
        pool.extend_from_slice(&doc.tokens);
        doc_meta.push((doc.doc_id, doc.n_tokens));
    }

    build_index_from_pool(pool, doc_meta, params)
}

/// Build an [`Index`] from a pre-assembled token pool and per-document
/// `(doc_id, n_tokens)` metadata.
///
/// This is the memory-lean entry point: callers that can stream tokens
/// straight into a single contiguous `Vec<f32>` (e.g. the `EmbeddingDb`
/// bridge) avoid holding both a `Vec<DocumentTokens>` *and* a flat pool
/// at the same time — on a real corpus that doubling is worth several
/// GB of peak RSS.
///
/// `pool` is laid out row-major with `n_tokens × dim` entries, where
/// `n_tokens = doc_meta.iter().map(|(_, n)| n).sum()`. Documents keep
/// the order of `doc_meta` in the resulting index.
///
/// # Errors
///
/// Returns [`PlaidError::Tensor`] if the matmul-driven k-means
/// training or nearest-centroid assignment fails.
///
/// # Panics
///
/// Panics if `pool.len()` is not a multiple of `dim`, if the sum of
/// `doc_meta`'s token counts disagrees with `pool.len() / dim`, if the
/// total number of tokens is smaller than `params.k_centroids`, or if
/// `params.k_centroids == 0`.
///
/// [`PlaidError::Tensor`]: crate::PlaidError::Tensor
pub fn build_index_from_pool(
    pool: Vec<f32>,
    doc_meta: Vec<(u64, usize)>,
    params: IndexParams,
) -> Result<Index> {
    assert!(
        params.dim > 0,
        "build_index_from_pool: dim must be positive"
    );
    assert!(
        params.k_centroids > 0,
        "build_index_from_pool: k_centroids must be positive"
    );
    assert!(
        params.nbits > 0 && params.nbits <= 8,
        "build_index_from_pool: nbits must be in 1..=8, got {}",
        params.nbits,
    );
    assert!(
        pool.len().is_multiple_of(params.dim),
        "build_index_from_pool: pool length {} is not a multiple of dim {}",
        pool.len(),
        params.dim,
    );
    let total_tokens = pool.len() / params.dim;
    let meta_tokens: usize = doc_meta.iter().map(|(_, n)| n).sum();
    assert_eq!(
        total_tokens, meta_tokens,
        "build_index_from_pool: pool carries {total_tokens} tokens but doc_meta sums to {meta_tokens}",
    );
    assert!(
        total_tokens >= params.k_centroids,
        "build_index_from_pool: need at least {} tokens for {} centroids, got {}",
        params.k_centroids,
        params.k_centroids,
        total_tokens,
    );

    // Upload the [n, dim] pool tensor once and share it across the
    // k-means phase and the final batch-encode pass. Without this, each
    // phase uploaded its own ~3.47 GB copy for a real docbert corpus,
    // cudarc's caching allocator held the stale block, and the PLAID
    // build tipped a 12 GB card into CUDA OOM as soon as the encoder
    // model was also resident.
    let pool_bytes = pool.len() * std::mem::size_of::<f32>();
    let device_info = crate::device::device_memory_info();
    eprintln!(
        "  Pool: {total_tokens} tokens × {} dim = {} MiB{}",
        params.dim,
        pool_bytes / (1 << 20),
        match device_info {
            Some((free, total)) => format!(
                " (device: {} MiB free / {} MiB total)",
                free / (1 << 20),
                total / (1 << 20),
            ),
            None => String::new(),
        },
    );

    // 1. Train coarse centroids with k-means. `fit` subsamples to
    //    `k * MAX_POINTS_PER_CENTROID` rows and uploads only that
    //    subsample, so peak VRAM here is bounded regardless of pool
    //    size or `dim`.
    let centroids = fit(
        &pool,
        params.k_centroids,
        params.dim,
        params.max_kmeans_iters.max(1),
    )?;

    // 2. Residuals for quantizer training. We only need enough samples
    //    to place `2^nbits` quantile cutoffs — materialising a
    //    residual-per-token for a large corpus would allocate multiple
    //    gigabytes for no statistical benefit. Stride-sample the pool,
    //    compute residuals only for the sampled tokens, and move on.
    let sample_stride =
        total_tokens.div_ceil(MAX_QUANTIZER_TRAINING_TOKENS).max(1);
    let sample_count = total_tokens.div_ceil(sample_stride);

    let mut sampled_tokens: Vec<f32> =
        Vec::with_capacity(sample_count * params.dim);
    for (i, token) in pool.chunks_exact(params.dim).enumerate() {
        if i.is_multiple_of(sample_stride) {
            sampled_tokens.extend_from_slice(token);
        }
    }
    let sample_assignments =
        assign_points(&sampled_tokens, &centroids, params.dim)?;
    let mut residual_sample: Vec<f32> =
        Vec::with_capacity(sampled_tokens.len());
    for (token, &cluster) in sampled_tokens
        .chunks_exact(params.dim)
        .zip(&sample_assignments)
    {
        let centroid =
            &centroids[cluster * params.dim..(cluster + 1) * params.dim];
        for (t, c) in token.iter().zip(centroid) {
            residual_sample.push(*t - *c);
        }
    }
    drop(sampled_tokens);

    // 3. Learn cutoffs + weights on the sampled residuals.
    let (bucket_cutoffs, bucket_weights) =
        train_quantizer(residual_sample, params.nbits);

    let codec = ResidualCodec {
        nbits: params.nbits,
        dim: params.dim,
        centroids,
        bucket_cutoffs,
        bucket_weights,
    };
    codec.validate()?;

    // 4. Encode every token across the whole corpus. The encoder
    //    walks the host `pool` in `[chunk_rows, dim]` tiles, uploads
    //    each tile, runs assign + bucketize + pack on it, drains the
    //    results back to host, and drops the tile before the next
    //    one uploads. Peak VRAM is bounded by
    //    `codec state + one chunk` (≈128 MiB at default settings) —
    //    unchanged by corpus size or embedding dimension, so a
    //    1536-dim model on a 6.8M-token corpus builds without the
    //    40 GB contiguous allocation the old single-shot path
    //    required.
    let (doc_centroid_ids, doc_residual_bytes) =
        codec.batch_encode_tokens(&pool)?;
    drop(pool);
    debug_assert_eq!(doc_centroid_ids.len(), total_tokens);
    debug_assert_eq!(
        doc_residual_bytes.len(),
        total_tokens * codec.packed_bytes()
    );

    let mut doc_ids = Vec::with_capacity(doc_meta.len());
    let mut doc_offsets = Vec::with_capacity(doc_meta.len() + 1);
    doc_offsets.push(0usize);
    let mut running = 0usize;
    for (doc_id, n_tok) in doc_meta {
        doc_ids.push(doc_id);
        running += n_tok;
        doc_offsets.push(running);
    }

    let ivf = build_inverted_file_from_flat(
        &doc_centroid_ids,
        &doc_offsets,
        params.k_centroids,
    );

    Ok(Index {
        params,
        codec,
        doc_ids,
        doc_centroid_ids,
        doc_residual_bytes,
        doc_offsets,
        ivf,
    })
}

/// Build the centroid → unique-doc-ids inverted file from a flat-layout
/// index's centroid assignments.
///
/// Each doc contributes at most one entry per centroid it touches;
/// entries within a list are sorted ascending. `doc_offsets` carries
/// the cumulative token counts (length `num_docs + 1`) that delimit
/// each document's slice of `doc_centroid_ids`.
pub(crate) fn build_inverted_file_from_flat(
    doc_centroid_ids: &[u32],
    doc_offsets: &[usize],
    k_centroids: usize,
) -> InvertedFile {
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); k_centroids];
    let n_docs = doc_offsets.len().saturating_sub(1);
    for doc_idx in 0..n_docs {
        let doc_codes =
            &doc_centroid_ids[doc_offsets[doc_idx]..doc_offsets[doc_idx + 1]];
        let mut touched: Vec<u32> = doc_codes.to_vec();
        touched.sort_unstable();
        touched.dedup();
        for cid in touched {
            lists[cid as usize].push(doc_idx as u32);
        }
    }
    InvertedFile { lists }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::squared_l2;

    /// Build a tiny 2-D corpus with two clear clusters of tokens.
    fn small_corpus() -> Vec<DocumentTokens> {
        vec![
            DocumentTokens {
                doc_id: 1,
                tokens: vec![0.0, 0.0, 0.1, 0.2, -0.1, 0.1],
                n_tokens: 3,
            },
            DocumentTokens {
                doc_id: 2,
                tokens: vec![10.0, 10.0, 10.2, 9.9, 9.8, 10.1],
                n_tokens: 3,
            },
            DocumentTokens {
                doc_id: 3,
                tokens: vec![0.3, -0.2, 9.7, 10.2],
                n_tokens: 2,
            },
        ]
    }

    fn default_params() -> IndexParams {
        IndexParams {
            dim: 2,
            nbits: 2,
            k_centroids: 2,
            max_kmeans_iters: 50,
        }
    }

    #[test]
    fn build_index_encodes_every_token() {
        let docs = small_corpus();
        let params = default_params();
        let expected_total: usize = docs.iter().map(|d| d.n_tokens).sum();

        let index = build_index(&docs, params).unwrap();

        assert_eq!(index.num_documents(), docs.len());
        assert_eq!(index.num_tokens(), expected_total);
        for (i, doc) in docs.iter().enumerate() {
            assert_eq!(index.doc_token_count(i), doc.n_tokens);
        }
    }

    #[test]
    fn build_index_assigns_tokens_in_each_cluster_to_the_closest_centroid() {
        let docs = small_corpus();
        let params = default_params();
        let index = build_index(&docs, params).unwrap();

        // The two tight clusters around (0,0) and (10,10) should produce
        // centroids close to those means.
        let c0 = &index.codec.centroids[0..2];
        let c1 = &index.codec.centroids[2..4];

        let (near_origin, near_ten) =
            if squared_l2(c0, &[0.0, 0.0]) < squared_l2(c0, &[10.0, 10.0]) {
                (c0, c1)
            } else {
                (c1, c0)
            };

        assert!(squared_l2(near_origin, &[0.0, 0.0]) < 1.0);
        assert!(squared_l2(near_ten, &[10.0, 10.0]) < 1.0);
    }

    #[test]
    fn build_index_round_trip_reconstruction_error_is_bounded() {
        let docs = small_corpus();
        let params = default_params();
        let index = build_index(&docs, params).unwrap();

        for (i, doc) in docs.iter().enumerate() {
            let encoded_doc = index.doc_tokens_vec(i);
            for (token, encoded) in
                doc.tokens.chunks_exact(params.dim).zip(encoded_doc.iter())
            {
                let decoded = index.codec.decode_vector(encoded).unwrap();
                let err = squared_l2(token, &decoded).sqrt();
                // Residuals on a 2-D toy corpus with tight clusters stay
                // small; each bucket should cover well under 0.5 per dim.
                assert!(
                    err < 0.6,
                    "reconstruction error {err} too large for token {token:?}"
                );
            }
        }
    }

    #[test]
    fn build_index_preserves_document_id_order() {
        let docs = small_corpus();
        let index = build_index(&docs, default_params()).unwrap();
        let expected_ids: Vec<u64> = docs.iter().map(|d| d.doc_id).collect();
        assert_eq!(index.doc_ids, expected_ids);
        assert_eq!(index.position_of(2), Some(1));
        assert_eq!(index.position_of(999), None);
    }

    #[test]
    fn build_index_handles_document_with_no_tokens() {
        // Empty documents are still indexable: they contribute no tokens
        // to training but keep their slot so callers can look them up.
        let mut docs = small_corpus();
        docs.push(DocumentTokens {
            doc_id: 42,
            tokens: vec![],
            n_tokens: 0,
        });
        let index = build_index(&docs, default_params()).unwrap();
        assert_eq!(index.num_documents(), 4);
        assert_eq!(index.doc_token_count(3), 0);
    }

    #[test]
    #[should_panic(expected = "declared")]
    fn build_index_panics_on_mismatched_token_count() {
        let docs = vec![DocumentTokens {
            doc_id: 1,
            tokens: vec![0.0, 0.0, 1.0],
            n_tokens: 2, // says 2 but only 3 f32s and dim=2
        }];
        let _ = build_index(&docs, default_params()).unwrap();
    }

    #[test]
    fn build_index_ivf_has_one_list_per_centroid() {
        let docs = small_corpus();
        let index = build_index(&docs, default_params()).unwrap();

        assert_eq!(
            index.ivf.num_centroids(),
            default_params().k_centroids,
            "IVF has one list per centroid",
        );
    }

    #[test]
    fn build_index_ivf_postings_cover_every_doc_that_touches_each_centroid() {
        // Every (doc, token) pair implies the doc_idx must appear in
        // that centroid's posting list. This is the PLAID "centroid →
        // unique passage ids" contract: postings are indexed by doc.
        let docs = small_corpus();
        let index = build_index(&docs, default_params()).unwrap();

        for doc_idx in 0..index.num_documents() {
            for &cid in index.doc_centroid_ids(doc_idx) {
                let postings = index.ivf.docs_for_centroid(cid as usize);
                assert!(
                    postings.contains(&(doc_idx as u32)),
                    "doc_idx={doc_idx} missing from centroid {cid} postings",
                );
            }
        }
    }

    #[test]
    fn build_index_ivf_postings_are_unique_per_centroid() {
        // PLAID stores centroid → unique doc ids, not token refs. A doc
        // with multiple tokens in the same centroid must appear at most
        // once in that centroid's list.
        let docs = small_corpus();
        let index = build_index(&docs, default_params()).unwrap();

        for c in 0..index.ivf.num_centroids() {
            let postings = index.ivf.docs_for_centroid(c);
            let mut unique: Vec<u32> = postings.to_vec();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(
                unique.len(),
                postings.len(),
                "centroid {c} has duplicate doc entries: {postings:?}",
            );
        }
    }

    #[test]
    fn build_index_dedupes_repeated_tokens_in_same_centroid() {
        // Doc 1 has three tokens that all cluster to the same coarse
        // centroid. The doc_idx should show up once in that centroid's
        // posting, not three times.
        let docs = vec![
            DocumentTokens {
                doc_id: 1,
                tokens: vec![0.0, 0.0, 0.05, -0.02, -0.03, 0.01],
                n_tokens: 3,
            },
            DocumentTokens {
                doc_id: 2,
                tokens: vec![10.0, 10.0, 10.1, 9.9],
                n_tokens: 2,
            },
        ];
        let index = build_index(&docs, default_params()).unwrap();

        for c in 0..index.ivf.num_centroids() {
            let postings = index.ivf.docs_for_centroid(c);
            let count_of_doc_0 = postings.iter().filter(|&&d| d == 0).count();
            assert!(
                count_of_doc_0 <= 1,
                "doc 0 appears {count_of_doc_0} times in centroid {c}",
            );
        }
    }

    #[test]
    fn inverted_file_out_of_range_returns_empty_slice() {
        let ivf = InvertedFile {
            lists: vec![vec![0u32]],
        };
        assert_eq!(ivf.docs_for_centroid(0).len(), 1);
        assert!(ivf.docs_for_centroid(999).is_empty());
    }

    #[test]
    #[should_panic(expected = "at least")]
    fn build_index_panics_when_too_few_tokens_for_k() {
        let docs = vec![DocumentTokens {
            doc_id: 1,
            tokens: vec![0.0, 1.0],
            n_tokens: 1,
        }];
        let params = IndexParams {
            dim: 2,
            nbits: 2,
            k_centroids: 4,
            max_kmeans_iters: 10,
        };
        let _ = build_index(&docs, params).unwrap();
    }
}
