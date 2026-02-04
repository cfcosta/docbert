use candle_core::Tensor;

use crate::{
    embedding::load_embedding_tensor,
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
/// 1. Load its embedding matrix from the database
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

    let mut ranked: Vec<RankedDocument> = candidate_ids
        .iter()
        .filter_map(|&doc_id| {
            // Load doc embedding as [T, D], unsqueeze to [1, T, D]
            let doc_embedding =
                load_embedding_tensor(embedding_db, doc_id).ok().flatten()?;
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
