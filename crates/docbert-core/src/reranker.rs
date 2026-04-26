use candle_core::Tensor;

use crate::{
    chunking::document_family_key,
    embedding::batch_load_document_family_tensors,
    embedding_db::EmbeddingDb,
    error::{Error, Result},
    model_manager::ModelManager,
};

/// Candidate document after ColBERT reranking.
///
/// Returned by [`rerank`]. Results are sorted by score, highest first.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedDocument {
    /// Numeric document identifier.
    pub doc_num_id: u64,
    /// ColBERT MaxSim similarity score (higher = more relevant).
    pub score: f32,
    /// Numeric ID of the best-scoring chunk for this document, when
    /// known. Equal to `doc_num_id` when chunk 0 (the base chunk) was
    /// the winner; differs by the chunk-index encoding in the high bits
    /// for later chunks. `None` for ranking flows that don't expose
    /// per-chunk scores (e.g. legacy reranking over already-collapsed
    /// document embeddings).
    pub best_chunk_doc_id: Option<u64>,
}

/// Score a batch of already loaded document embeddings with ColBERT MaxSim.
///
/// `query_3d` must have shape `[1, tokens, dim]`. Document embeddings are moved
/// to the same device as the query before similarity is computed. Missing
/// embeddings are skipped. Results keep input order.
pub(crate) fn score_loaded_embeddings(
    query_3d: &Tensor,
    embeddings: Vec<(u64, Option<Tensor>)>,
    model: &ModelManager,
) -> Result<Vec<RankedDocument>> {
    let mut ranked = Vec::new();
    for (doc_id, doc_embedding_opt) in embeddings {
        let Some(doc_embedding) = doc_embedding_opt else {
            continue;
        };
        let doc_embedding = doc_embedding.to_device(query_3d.device())?;
        let doc_3d = doc_embedding.unsqueeze(0)?;

        let similarities = model.similarity(query_3d, &doc_3d)?;
        let score = similarities
            .data
            .first()
            .and_then(|row| row.first())
            .copied()
            .ok_or_else(|| {
                Error::Config(format!(
                    "missing similarity score for doc {doc_id}"
                ))
            })?;

        ranked.push(RankedDocument {
            doc_num_id: doc_id,
            score,
            best_chunk_doc_id: Some(doc_id),
        });
    }

    Ok(ranked)
}

/// Collapse scored chunk embeddings to one best score per requested document.
///
/// Scores are grouped by document family. The highest chunk score in each
/// family wins. Output preserves the order of `requested_doc_ids`, skipping
/// duplicate family requests and families with no scored chunks.
pub(crate) fn collapse_best_chunk_scores(
    requested_doc_ids: &[u64],
    scored_chunks: Vec<RankedDocument>,
) -> Vec<RankedDocument> {
    // Track each family's best score *and* which chunk produced it so
    // the search consumer can surface the chunk's byte range alongside
    // the result. The chunk-doc-id either comes from the input
    // RankedDocument (when score_loaded_embeddings populated it) or
    // falls back to the doc_num_id of that chunk.
    let mut best_by_family: std::collections::HashMap<u64, (f32, u64)> =
        std::collections::HashMap::new();
    for chunk in scored_chunks {
        let family_key = document_family_key(chunk.doc_num_id);
        let chunk_id = chunk.best_chunk_doc_id.unwrap_or(chunk.doc_num_id);
        best_by_family
            .entry(family_key)
            .and_modify(|(best_score, best_chunk_id)| {
                if chunk.score > *best_score {
                    *best_score = chunk.score;
                    *best_chunk_id = chunk_id;
                }
            })
            .or_insert((chunk.score, chunk_id));
    }

    let mut emitted_families = std::collections::HashSet::new();
    let mut collapsed = Vec::new();
    for &doc_num_id in requested_doc_ids {
        let family_key = document_family_key(doc_num_id);
        if !emitted_families.insert(family_key) {
            continue;
        }
        let Some(&(score, best_chunk_id)) = best_by_family.get(&family_key)
        else {
            continue;
        };
        collapsed.push(RankedDocument {
            doc_num_id,
            score,
            best_chunk_doc_id: Some(best_chunk_id),
        });
    }

    collapsed
}

fn load_candidate_family_tensors_for_rerank(
    embedding_db: &EmbeddingDb,
    candidate_ids: &[u64],
) -> Result<Vec<(u64, Tensor)>> {
    batch_load_document_family_tensors(embedding_db, candidate_ids)
}

fn rerank_with_scorer<F>(
    query_embedding: &Tensor,
    candidate_ids: &[u64],
    embedding_db: &EmbeddingDb,
    model: &ModelManager,
    score_loaded_embeddings_fn: F,
) -> Result<Vec<RankedDocument>>
where
    F: Fn(
        &Tensor,
        Vec<(u64, Option<Tensor>)>,
        &ModelManager,
    ) -> Result<Vec<RankedDocument>>,
{
    // pylate similarity expects 3D tensors: [batch, tokens, dim]
    // Query is [Q, D], unsqueeze to [1, Q, D]
    let query_3d = query_embedding.unsqueeze(0)?;

    let embeddings =
        load_candidate_family_tensors_for_rerank(embedding_db, candidate_ids)?;
    let embeddings_with_presence: Vec<(u64, Option<Tensor>)> = embeddings
        .into_iter()
        .map(|(doc_id, tensor)| (doc_id, Some(tensor)))
        .collect();

    let scored_chunks =
        score_loaded_embeddings_fn(&query_3d, embeddings_with_presence, model)?;
    let mut ranked = collapse_best_chunk_scores(candidate_ids, scored_chunks);

    // Sort by score descending.
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(ranked)
}

/// Rerank candidate documents with ColBERT MaxSim scoring.
///
/// For each candidate, this loads the stored embedding, scores it against the
/// query embedding, and keeps the result.
///
/// Documents without stored embeddings are skipped. The returned list is sorted
/// by score, highest first.
pub fn rerank(
    query_embedding: &Tensor,
    candidate_ids: &[u64],
    embedding_db: &EmbeddingDb,
    model: &ModelManager,
) -> Result<Vec<RankedDocument>> {
    rerank_with_scorer(
        query_embedding,
        candidate_ids,
        embedding_db,
        model,
        score_loaded_embeddings,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DocumentId, chunking::chunk_doc_id};

    #[test]
    fn collapse_best_chunk_scores_returns_one_row_for_multi_chunk_family() {
        let base_doc_id = DocumentId::new("notes", "hello.md").numeric;
        let scored_chunks = vec![
            RankedDocument {
                doc_num_id: base_doc_id,
                score: 0.7,
                best_chunk_doc_id: None,
            },
            RankedDocument {
                doc_num_id: chunk_doc_id(base_doc_id, 1),
                score: 0.9,
                best_chunk_doc_id: None,
            },
            RankedDocument {
                doc_num_id: chunk_doc_id(base_doc_id, 2),
                score: 0.8,
                best_chunk_doc_id: None,
            },
        ];

        let collapsed =
            collapse_best_chunk_scores(&[base_doc_id], scored_chunks);

        assert_eq!(collapsed.len(), 1);
        assert_eq!(collapsed[0].doc_num_id, base_doc_id);
        assert_eq!(collapsed[0].score, 0.9);
    }

    #[test]
    fn collapse_best_chunk_scores_uses_highest_chunk_score() {
        let base_doc_id = DocumentId::new("notes", "hello.md").numeric;
        let winning_chunk = chunk_doc_id(base_doc_id, 1);

        let collapsed = collapse_best_chunk_scores(
            &[base_doc_id],
            vec![
                RankedDocument {
                    doc_num_id: chunk_doc_id(base_doc_id, 2),
                    score: 0.4,
                    best_chunk_doc_id: Some(chunk_doc_id(base_doc_id, 2)),
                },
                RankedDocument {
                    doc_num_id: winning_chunk,
                    score: 1.1,
                    best_chunk_doc_id: Some(winning_chunk),
                },
                RankedDocument {
                    doc_num_id: base_doc_id,
                    score: 0.6,
                    best_chunk_doc_id: Some(base_doc_id),
                },
            ],
        );

        // The collapse keeps the family's base id but threads the
        // winning chunk's id forward — the search consumer needs both
        // (the doc to fetch, and the chunk whose byte range to surface).
        assert_eq!(
            collapsed,
            vec![RankedDocument {
                doc_num_id: base_doc_id,
                score: 1.1,
                best_chunk_doc_id: Some(winning_chunk),
            }]
        );
    }

    #[test]
    fn collapse_best_chunk_scores_keeps_unrelated_families_separate() {
        let first_doc_id = DocumentId::new("notes", "hello.md").numeric;
        let second_doc_id = DocumentId::new("notes", "guide.md").numeric;
        let first_chunk = chunk_doc_id(first_doc_id, 1);
        let second_chunk = chunk_doc_id(second_doc_id, 1);

        let collapsed = collapse_best_chunk_scores(
            &[first_doc_id, second_doc_id],
            vec![
                RankedDocument {
                    doc_num_id: first_chunk,
                    score: 0.8,
                    best_chunk_doc_id: Some(first_chunk),
                },
                RankedDocument {
                    doc_num_id: second_chunk,
                    score: 0.5,
                    best_chunk_doc_id: Some(second_chunk),
                },
            ],
        );

        assert_eq!(
            collapsed,
            vec![
                RankedDocument {
                    doc_num_id: first_doc_id,
                    score: 0.8,
                    best_chunk_doc_id: Some(first_chunk),
                },
                RankedDocument {
                    doc_num_id: second_doc_id,
                    score: 0.5,
                    best_chunk_doc_id: Some(second_chunk),
                },
            ]
        );
    }

    #[test]
    fn collapse_best_chunk_scores_does_not_emit_duplicates() {
        let base_doc_id = DocumentId::new("notes", "hello.md").numeric;

        let collapsed = collapse_best_chunk_scores(
            &[base_doc_id, base_doc_id],
            vec![
                RankedDocument {
                    doc_num_id: chunk_doc_id(base_doc_id, 1),
                    score: 0.8,
                    best_chunk_doc_id: None,
                },
                RankedDocument {
                    doc_num_id: chunk_doc_id(base_doc_id, 2),
                    score: 0.9,
                    best_chunk_doc_id: None,
                },
            ],
        );

        assert_eq!(collapsed.len(), 1);
        assert_eq!(collapsed[0].doc_num_id, base_doc_id);
        assert_eq!(collapsed[0].score, 0.9);
    }

    #[test]
    fn collapse_best_chunk_scores_preserves_base_only_scores() {
        let base_doc_id = DocumentId::new("notes", "hello.md").numeric;

        let collapsed = collapse_best_chunk_scores(
            &[base_doc_id],
            vec![RankedDocument {
                doc_num_id: base_doc_id,
                score: 0.75,
                best_chunk_doc_id: Some(base_doc_id),
            }],
        );

        // For base-only families, the winning "chunk" *is* the base
        // document — chunk 0 reuses the base id by construction.
        assert_eq!(
            collapsed,
            vec![RankedDocument {
                doc_num_id: base_doc_id,
                score: 0.75,
                best_chunk_doc_id: Some(base_doc_id),
            }]
        );
    }

    #[test]
    fn collapse_best_chunk_scores_follows_requested_doc_order() {
        let first_doc_id = DocumentId::new("notes", "hello.md").numeric;
        let second_doc_id = DocumentId::new("notes", "guide.md").numeric;
        let first_chunk = chunk_doc_id(first_doc_id, 1);
        let second_chunk = chunk_doc_id(second_doc_id, 1);

        let collapsed = collapse_best_chunk_scores(
            &[second_doc_id, first_doc_id],
            vec![
                RankedDocument {
                    doc_num_id: first_chunk,
                    score: 0.8,
                    best_chunk_doc_id: Some(first_chunk),
                },
                RankedDocument {
                    doc_num_id: second_chunk,
                    score: 0.5,
                    best_chunk_doc_id: Some(second_chunk),
                },
            ],
        );

        assert_eq!(
            collapsed,
            vec![
                RankedDocument {
                    doc_num_id: second_doc_id,
                    score: 0.5,
                    best_chunk_doc_id: Some(second_chunk),
                },
                RankedDocument {
                    doc_num_id: first_doc_id,
                    score: 0.8,
                    best_chunk_doc_id: Some(first_chunk),
                },
            ]
        );
    }

    #[test]
    fn load_candidate_family_tensors_for_rerank_includes_chunk_only_families() {
        let tmp = tempfile::tempdir().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let base_doc_id = DocumentId::new("notes", "hello.md").numeric;
        let chunk_only_id = chunk_doc_id(base_doc_id, 1);

        embedding_db
            .store(chunk_only_id, 2, 2, &[1.0, 2.0, 3.0, 4.0])
            .unwrap();

        let loaded = load_candidate_family_tensors_for_rerank(
            &embedding_db,
            &[base_doc_id],
        )
        .unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, chunk_only_id);
        assert_eq!(loaded[0].1.dims2().unwrap(), (2, 2));
    }

    #[test]
    fn load_candidate_family_tensors_for_rerank_preserves_candidate_family_order()
     {
        let tmp = tempfile::tempdir().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let first_doc_id = DocumentId::new("notes", "hello.md").numeric;
        let second_doc_id = DocumentId::new("notes", "guide.md").numeric;
        let first_chunk_id = chunk_doc_id(first_doc_id, 1);
        let second_chunk_id = chunk_doc_id(second_doc_id, 1);

        embedding_db.store(first_chunk_id, 1, 1, &[1.0]).unwrap();
        embedding_db.store(second_chunk_id, 1, 1, &[2.0]).unwrap();
        embedding_db.store(first_doc_id, 1, 1, &[3.0]).unwrap();
        embedding_db.store(second_doc_id, 1, 1, &[4.0]).unwrap();

        let loaded = load_candidate_family_tensors_for_rerank(
            &embedding_db,
            &[second_doc_id, first_doc_id],
        )
        .unwrap();
        let loaded_ids: Vec<u64> =
            loaded.iter().map(|(doc_id, _)| *doc_id).collect();

        let mut second_family_expected = vec![second_doc_id, second_chunk_id];
        second_family_expected.sort_unstable();
        let mut first_family_expected = vec![first_doc_id, first_chunk_id];
        first_family_expected.sort_unstable();
        let expected_ids: Vec<u64> = second_family_expected
            .into_iter()
            .chain(first_family_expected)
            .collect();

        assert_eq!(loaded_ids, expected_ids);
    }

    #[test]
    fn family_aware_rerank_pipeline_loads_and_collapses_chunk_only_families() {
        let tmp = tempfile::tempdir().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let first_doc_id = DocumentId::new("notes", "hello.md").numeric;
        let second_doc_id = DocumentId::new("notes", "guide.md").numeric;
        let first_chunk_id = chunk_doc_id(first_doc_id, 1);
        let second_chunk_id = chunk_doc_id(second_doc_id, 1);
        let second_chunk_id_2 = chunk_doc_id(second_doc_id, 2);

        embedding_db.store(first_chunk_id, 1, 1, &[1.0]).unwrap();
        embedding_db.store(second_chunk_id, 1, 1, &[2.0]).unwrap();
        embedding_db.store(second_chunk_id_2, 1, 1, &[3.0]).unwrap();

        let loaded = load_candidate_family_tensors_for_rerank(
            &embedding_db,
            &[second_doc_id, first_doc_id],
        )
        .unwrap();
        let synthetic_scores = loaded
            .into_iter()
            .map(|(doc_id, _)| RankedDocument {
                doc_num_id: doc_id,
                score: match doc_id {
                    id if id == first_chunk_id => 0.4,
                    id if id == second_chunk_id => 0.6,
                    id if id == second_chunk_id_2 => 0.9,
                    _ => 0.0,
                },
                best_chunk_doc_id: Some(doc_id),
            })
            .collect();

        let collapsed = collapse_best_chunk_scores(
            &[second_doc_id, first_doc_id],
            synthetic_scores,
        );

        assert_eq!(
            collapsed,
            vec![
                RankedDocument {
                    doc_num_id: second_doc_id,
                    score: 0.9,
                    best_chunk_doc_id: Some(second_chunk_id_2),
                },
                RankedDocument {
                    doc_num_id: first_doc_id,
                    score: 0.4,
                    best_chunk_doc_id: Some(first_chunk_id),
                },
            ]
        );
    }

    #[test]
    fn rerank_with_scorer_returns_base_document_ids_for_chunk_families() {
        let tmp = tempfile::tempdir().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let model = ModelManager::new();
        let first_doc_id = DocumentId::new("notes", "hello.md").numeric;
        let second_doc_id = DocumentId::new("notes", "guide.md").numeric;
        let first_chunk_id = chunk_doc_id(first_doc_id, 1);
        let second_chunk_id = chunk_doc_id(second_doc_id, 1);
        let second_chunk_id_2 = chunk_doc_id(second_doc_id, 2);
        let query = Tensor::zeros(
            &[2, 128],
            candle_core::DType::F32,
            &crate::test_util::test_device(),
        )
        .unwrap();

        embedding_db.store(first_chunk_id, 1, 1, &[1.0]).unwrap();
        embedding_db.store(second_chunk_id, 1, 1, &[2.0]).unwrap();
        embedding_db.store(second_chunk_id_2, 1, 1, &[3.0]).unwrap();

        let ranked = rerank_with_scorer(
            &query,
            &[second_doc_id, first_doc_id],
            &embedding_db,
            &model,
            |_, embeddings, _| {
                Ok(embeddings
                    .into_iter()
                    .filter_map(|(doc_id, tensor)| {
                        tensor.map(|_| RankedDocument {
                            doc_num_id: doc_id,
                            score: match doc_id {
                                id if id == first_chunk_id => 0.4,
                                id if id == second_chunk_id => 0.6,
                                id if id == second_chunk_id_2 => 0.9,
                                _ => 0.0,
                            },
                            best_chunk_doc_id: Some(doc_id),
                        })
                    })
                    .collect())
            },
        )
        .unwrap();

        assert_eq!(
            ranked,
            vec![
                RankedDocument {
                    doc_num_id: second_doc_id,
                    score: 0.9,
                    best_chunk_doc_id: Some(second_chunk_id_2),
                },
                RankedDocument {
                    doc_num_id: first_doc_id,
                    score: 0.4,
                    best_chunk_doc_id: Some(first_chunk_id),
                },
            ]
        );
    }

    #[test]
    fn rerank_empty_candidates() {
        let tmp = tempfile::tempdir().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let model = ModelManager::new();

        // Create a dummy 2D query tensor [2, 128]
        let query = Tensor::zeros(
            &[2, 128],
            candle_core::DType::F32,
            &crate::test_util::test_device(),
        )
        .unwrap();

        let results = rerank(&query, &[], &embedding_db, &model).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn rerank_missing_embeddings_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let model = ModelManager::new();

        // Create a dummy query tensor
        let query = Tensor::zeros(
            &[2, 128],
            candle_core::DType::F32,
            &crate::test_util::test_device(),
        )
        .unwrap();

        // These IDs have no stored embeddings
        let ids = vec![999, 1000, 1001];
        let results = rerank(&query, &ids, &embedding_db, &model).unwrap();
        assert!(results.is_empty(), "missing embeddings should be skipped");
    }

    #[test]
    fn rerank_propagates_similarity_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let model = ModelManager::new();

        let query = Tensor::zeros(
            &[2, 128],
            candle_core::DType::F32,
            &crate::test_util::test_device(),
        )
        .unwrap();
        embedding_db.store(42, 2, 128, &vec![0.0; 256]).unwrap();

        let err = rerank(&query, &[42], &embedding_db, &model).unwrap_err();
        assert!(err.to_string().contains("model not loaded"));
    }
}
