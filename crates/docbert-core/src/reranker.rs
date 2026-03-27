use candle_core::Tensor;

use crate::{
    embedding::batch_load_embedding_tensors,
    embedding_db::EmbeddingDb,
    error::{Error, Result},
    model_manager::ModelManager,
};

/// Candidate document after ColBERT reranking.
///
/// Returned by [`rerank`]. Results are sorted by score, highest first.
#[derive(Debug, Clone)]
pub struct RankedDocument {
    /// Numeric document identifier.
    pub doc_num_id: u64,
    /// ColBERT MaxSim similarity score (higher = more relevant).
    pub score: f32,
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
            .ok_or_else(|| Error::Config(format!("missing similarity score for doc {doc_id}")))?;

        ranked.push(RankedDocument {
            doc_num_id: doc_id,
            score,
        });
    }

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
    // pylate similarity expects 3D tensors: [batch, tokens, dim]
    // Query is [Q, D], unsqueeze to [1, Q, D]
    let query_3d = query_embedding.unsqueeze(0)?;

    // Batch load all embeddings in a single transaction
    let embeddings = batch_load_embedding_tensors(embedding_db, candidate_ids)?;

    let mut ranked = score_loaded_embeddings(&query_3d, embeddings, model)?;

    // Sort by score descending.
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(ranked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rerank_empty_candidates() {
        let tmp = tempfile::tempdir().unwrap();
        let embedding_db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let model = ModelManager::new();

        // Create a dummy 2D query tensor [2, 128]
        let query = Tensor::zeros(
            &[2, 128],
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();

        let results = rerank(&query, &[], &embedding_db, &model).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn rerank_missing_embeddings_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let embedding_db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let model = ModelManager::new();

        // Create a dummy query tensor
        let query = Tensor::zeros(
            &[2, 128],
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
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
        let embedding_db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let model = ModelManager::new();

        let query = Tensor::zeros(
            &[2, 128],
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        embedding_db.store(42, 2, 128, &vec![0.0; 256]).unwrap();

        let err = rerank(&query, &[42], &embedding_db, &model).unwrap_err();
        assert!(err.to_string().contains("model not loaded"));
    }
}
