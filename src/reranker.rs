use candle_core::Tensor;

use crate::{
    embedding::batch_load_embedding_tensors,
    embedding_db::EmbeddingDb,
    error::Result,
    model_manager::ModelManager,
};

/// A reranked document with its MaxSim score.
#[derive(Debug, Clone)]
pub struct RankedDocument {
    pub doc_num_id: u64,
    pub score: f32,
}

/// Rerank candidate documents using ColBERT MaxSim scoring via pylate-rs.
///
/// For each candidate document:
/// 1. Load its embedding matrix from the database (batch load for efficiency)
/// 2. Use pylate's similarity function to compute MaxSim score
///
/// Returns candidates sorted by score descending.
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

    let mut ranked: Vec<RankedDocument> = embeddings
        .into_iter()
        .filter_map(|(doc_id, doc_embedding_opt)| {
            // Skip documents without embeddings
            let doc_embedding = doc_embedding_opt?;
            let doc_3d = doc_embedding.unsqueeze(0).ok()?;

            // Use pylate's similarity: returns Similarities { data: [[score]] }
            let similarities = model.similarity(&query_3d, &doc_3d).ok()?;
            let score = *similarities.data.first()?.first()?;

            Some(RankedDocument {
                doc_num_id: doc_id,
                score,
            })
        })
        .collect();

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
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
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
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
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
}
