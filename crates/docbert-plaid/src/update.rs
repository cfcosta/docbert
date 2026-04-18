//! Incremental updates: mutate an existing [`Index`] without retraining.
//!
//! A freshly built index "freezes" its codec — the centroids and
//! residual cutoffs/weights were learned from the token distribution
//! at that point in time. In docbert's sync loop the vast majority of
//! documents are unchanged from one sync to the next; only a handful
//! are added, updated, or deleted. Retraining the codec every sync is
//! wasteful: k-means and quantizer training dominate [`build_index`].
//!
//! [`apply_update`] takes a small mutation plan and produces a new
//! index with deleted documents removed, upserted documents re-encoded
//! against the *existing* codec, and the centroid → tokens inverted
//! file rebuilt from the final encoded token list. No k-means, no
//! quantizer retraining.
//!
//! Callers are expected to trigger a full rebuild (via
//! [`build_index`]) periodically to combat codec drift as the corpus
//! evolves — the existing codec only stays well-calibrated while the
//! underlying token distribution doesn't change dramatically.
//!
//! [`build_index`]: crate::index::build_index

use std::collections::HashSet;

use crate::{
    codec::EncodedVector,
    index::{DocumentTokens, Index, build_inverted_file},
};

/// Mutation plan consumed by [`apply_update`].
///
/// Deletions are applied before upserts, so upserting a doc_id that
/// is simultaneously listed in `deletions` is equivalent to upserting
/// it alone. Duplicate doc_ids *within* `upserts` are rejected —
/// the caller should pre-merge them since "last one wins" would be
/// surprising and "first one wins" would be ambiguous.
#[derive(Debug, Clone, Copy)]
pub struct IndexUpdate<'a> {
    /// Doc IDs to drop from the index.
    pub deletions: &'a [u64],
    /// Documents to add or replace. A document with a doc_id already
    /// in the index has its old tokens removed before the new ones
    /// are encoded.
    pub upserts: &'a [DocumentTokens],
}

/// Produce a new [`Index`] reflecting `update` applied to `index`,
/// reusing the existing codec and centroids.
///
/// # Panics
///
/// Panics if any upserted document's `tokens.len()` disagrees with
/// `n_tokens * index.params.dim`, or if `upserts` contains two
/// entries with the same `doc_id`.
pub fn apply_update(index: Index, update: IndexUpdate<'_>) -> Index {
    let params = index.params;

    for doc in update.upserts {
        assert!(
            doc.tokens.len() == doc.n_tokens * params.dim,
            "apply_update: doc {} declared {} tokens but carries {} f32s (dim={})",
            doc.doc_id,
            doc.n_tokens,
            doc.tokens.len(),
            params.dim,
        );
    }

    // Guard against ambiguous duplicate upserts early so callers get a
    // loud failure rather than silently losing one of their writes.
    let mut seen: HashSet<u64> = HashSet::with_capacity(update.upserts.len());
    for doc in update.upserts {
        assert!(
            seen.insert(doc.doc_id),
            "apply_update: duplicate doc_id {} in upserts",
            doc.doc_id,
        );
    }

    // Everything in `deletions` leaves the index. Anything in
    // `upserts` also leaves first (so the upsert replaces the old
    // copy when both live in the old index).
    let mut to_remove: HashSet<u64> =
        update.deletions.iter().copied().collect();
    for doc in update.upserts {
        to_remove.insert(doc.doc_id);
    }

    let Index {
        codec,
        doc_ids,
        doc_tokens,
        ..
    } = index;

    // Retain existing docs that survive the mutation.
    let mut new_doc_ids: Vec<u64> = Vec::with_capacity(doc_ids.len());
    let mut new_doc_tokens: Vec<Vec<EncodedVector>> =
        Vec::with_capacity(doc_tokens.len());
    for (id, tokens) in doc_ids.into_iter().zip(doc_tokens) {
        if to_remove.contains(&id) {
            continue;
        }
        new_doc_ids.push(id);
        new_doc_tokens.push(tokens);
    }

    // Encode every upsert's tokens against the existing codec. Doing
    // this in one batched call lets the matmul-driven
    // nearest-centroid lookup amortise across the whole upsert set
    // rather than paying per-document kernel overhead.
    let total_upsert_tokens: usize =
        update.upserts.iter().map(|d| d.n_tokens).sum();
    if total_upsert_tokens > 0 {
        let mut pool: Vec<f32> =
            Vec::with_capacity(total_upsert_tokens * params.dim);
        for doc in update.upserts {
            pool.extend_from_slice(&doc.tokens);
        }
        let (all_centroid_ids, all_codes) = codec.batch_encode_tokens(&pool);
        let mut offset = 0usize;
        for doc in update.upserts {
            let n = doc.n_tokens;
            let cids = &all_centroid_ids[offset..offset + n];
            let codes_slice =
                &all_codes[offset * params.dim..(offset + n) * params.dim];
            let encoded: Vec<EncodedVector> = (0..n)
                .map(|i| EncodedVector {
                    centroid_id: cids[i],
                    codes: codes_slice[i * params.dim..(i + 1) * params.dim]
                        .to_vec(),
                })
                .collect();
            new_doc_ids.push(doc.doc_id);
            new_doc_tokens.push(encoded);
            offset += n;
        }
    } else {
        // No tokens to encode, but preserve empty upserted documents'
        // slots so callers can look them up by doc_id.
        for doc in update.upserts {
            new_doc_ids.push(doc.doc_id);
            new_doc_tokens.push(Vec::new());
        }
    }

    // Rebuild the IVF from the final token list. A full rebuild is
    // O(n_docs · avg_tokens_per_doc) and avoids tracking per-token
    // positions inside each centroid's list.
    let ivf = build_inverted_file(&new_doc_tokens, params.k_centroids);

    Index {
        params,
        codec,
        doc_ids: new_doc_ids,
        doc_tokens: new_doc_tokens,
        ivf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{IndexParams, build_index};

    fn seed_corpus() -> Vec<DocumentTokens> {
        // Two well-separated clusters so k-means produces stable
        // centroids we can reason about, plus a mixed doc so the IVF
        // postings span more than one centroid per document.
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

    fn seed_params() -> IndexParams {
        IndexParams {
            dim: 2,
            nbits: 2,
            k_centroids: 2,
            max_kmeans_iters: 50,
        }
    }

    fn seed_index() -> Index {
        build_index(&seed_corpus(), seed_params())
    }

    fn assert_ivf_covers_every_doc_centroid_pair(index: &Index) {
        for (doc_idx, encoded) in index.doc_tokens.iter().enumerate() {
            for ev in encoded {
                let list = index.ivf.docs_for_centroid(ev.centroid_id as usize);
                assert!(
                    list.contains(&(doc_idx as u32)),
                    "missing posting for doc_idx={doc_idx} centroid={}",
                    ev.centroid_id,
                );
            }
        }
        // Postings are unique per centroid.
        for c in 0..index.ivf.num_centroids() {
            let postings = index.ivf.docs_for_centroid(c);
            let mut sorted = postings.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                postings.len(),
                "duplicate docs in centroid {c} postings",
            );
        }
    }

    #[test]
    fn apply_update_with_empty_mutations_preserves_every_document() {
        let index = seed_index();
        let before_ids = index.doc_ids.clone();
        let before_tokens = index.doc_tokens.clone();

        let updated = apply_update(
            index,
            IndexUpdate {
                deletions: &[],
                upserts: &[],
            },
        );

        assert_eq!(updated.doc_ids, before_ids);
        assert_eq!(updated.doc_tokens, before_tokens);
        assert_ivf_covers_every_doc_centroid_pair(&updated);
    }

    #[test]
    fn apply_update_removes_the_listed_deletions() {
        let index = seed_index();
        let original_tokens = index.num_tokens();
        let removed_tokens = index
            .doc_tokens
            .iter()
            .zip(&index.doc_ids)
            .find(|(_, id)| **id == 2)
            .map(|(t, _)| t.len())
            .unwrap();

        let updated = apply_update(
            index,
            IndexUpdate {
                deletions: &[2],
                upserts: &[],
            },
        );

        assert_eq!(updated.doc_ids, vec![1, 3]);
        assert_eq!(updated.doc_tokens.len(), 2);
        assert_eq!(updated.num_tokens(), original_tokens - removed_tokens);
        assert_ivf_covers_every_doc_centroid_pair(&updated);
    }

    #[test]
    fn apply_update_appends_upsert_of_a_new_doc_id() {
        let index = seed_index();
        let new_doc = DocumentTokens {
            doc_id: 99,
            tokens: vec![0.05, -0.05, 0.1, 0.0],
            n_tokens: 2,
        };

        let updated = apply_update(
            index,
            IndexUpdate {
                deletions: &[],
                upserts: std::slice::from_ref(&new_doc),
            },
        );

        assert_eq!(updated.doc_ids, vec![1, 2, 3, 99]);
        let encoded = updated.doc_tokens.last().unwrap();
        assert_eq!(encoded.len(), 2);
        assert_ivf_covers_every_doc_centroid_pair(&updated);
    }

    #[test]
    fn apply_update_replaces_an_existing_doc_when_upserted() {
        let index = seed_index();
        let replacement = DocumentTokens {
            doc_id: 1,
            // Different tokens in the far cluster — on replacement
            // the encoded centroid_ids should shift accordingly.
            tokens: vec![10.0, 10.1, 9.9, 10.0, 10.1, 9.8, 10.2, 10.0],
            n_tokens: 4,
        };
        let old_encoded = index
            .doc_tokens
            .iter()
            .zip(&index.doc_ids)
            .find(|(_, id)| **id == 1)
            .map(|(t, _)| t.clone())
            .unwrap();

        let updated = apply_update(
            index,
            IndexUpdate {
                deletions: &[],
                upserts: std::slice::from_ref(&replacement),
            },
        );

        // doc_id 1 moves to the end because deletions happen first,
        // then upserts are appended. Count stays the same.
        assert_eq!(updated.doc_ids.len(), 3);
        assert_eq!(updated.position_of(1), Some(2));

        let new_encoded = &updated.doc_tokens[2];
        assert_eq!(new_encoded.len(), 4);
        assert_ne!(
            *new_encoded, old_encoded,
            "upsert must replace the old encoded tokens",
        );
        assert_ivf_covers_every_doc_centroid_pair(&updated);
    }

    #[test]
    fn apply_update_keeps_the_codec_bit_for_bit() {
        let index = seed_index();
        let before = index.codec.clone();
        let upsert = DocumentTokens {
            doc_id: 4,
            tokens: vec![5.0, 5.0],
            n_tokens: 1,
        };

        let updated = apply_update(
            index,
            IndexUpdate {
                deletions: &[2],
                upserts: std::slice::from_ref(&upsert),
            },
        );

        // Incremental updates must never touch the trained codec.
        assert_eq!(updated.codec.centroids, before.centroids);
        assert_eq!(updated.codec.bucket_cutoffs, before.bucket_cutoffs);
        assert_eq!(updated.codec.bucket_weights, before.bucket_weights);
        assert_eq!(updated.codec.nbits, before.nbits);
        assert_eq!(updated.codec.dim, before.dim);
    }

    #[test]
    fn apply_update_preserves_surviving_documents_verbatim() {
        let index = seed_index();
        // Snapshot every surviving doc's encoded tokens BEFORE the
        // update so we can assert they're carried through unchanged.
        let keep_ids: Vec<u64> = index
            .doc_ids
            .iter()
            .copied()
            .filter(|id| *id != 2)
            .collect();
        let keep_tokens: Vec<Vec<EncodedVector>> = index
            .doc_ids
            .iter()
            .zip(&index.doc_tokens)
            .filter(|(id, _)| **id != 2)
            .map(|(_, t)| t.clone())
            .collect();

        let updated = apply_update(
            index,
            IndexUpdate {
                deletions: &[2],
                upserts: &[],
            },
        );

        assert_eq!(updated.doc_ids, keep_ids);
        assert_eq!(updated.doc_tokens, keep_tokens);
    }

    #[test]
    fn apply_update_handles_upsert_of_an_empty_document() {
        let index = seed_index();
        let empty = DocumentTokens {
            doc_id: 77,
            tokens: vec![],
            n_tokens: 0,
        };

        let updated = apply_update(
            index,
            IndexUpdate {
                deletions: &[],
                upserts: std::slice::from_ref(&empty),
            },
        );

        assert!(updated.doc_ids.contains(&77));
        assert_eq!(
            updated.doc_tokens[updated.position_of(77).unwrap()].len(),
            0
        );
        assert_ivf_covers_every_doc_centroid_pair(&updated);
    }

    #[test]
    #[should_panic(expected = "carries")]
    fn apply_update_panics_on_dim_mismatch_in_upsert() {
        let index = seed_index();
        let bad = DocumentTokens {
            doc_id: 5,
            tokens: vec![1.0, 2.0, 3.0], // dim=2 so 3 f32s is invalid
            n_tokens: 2,
        };
        let _ = apply_update(
            index,
            IndexUpdate {
                deletions: &[],
                upserts: std::slice::from_ref(&bad),
            },
        );
    }

    #[test]
    #[should_panic(expected = "duplicate doc_id")]
    fn apply_update_panics_on_duplicate_upsert_doc_ids() {
        let index = seed_index();
        let doc_a = DocumentTokens {
            doc_id: 1,
            tokens: vec![0.0, 0.0],
            n_tokens: 1,
        };
        let doc_b = DocumentTokens {
            doc_id: 1,
            tokens: vec![1.0, 1.0],
            n_tokens: 1,
        };
        let _ = apply_update(
            index,
            IndexUpdate {
                deletions: &[],
                upserts: &[doc_a, doc_b],
            },
        );
    }
}
