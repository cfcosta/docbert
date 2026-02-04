use candle_core::Tensor;
use rayon::prelude::*;

use crate::{
    embedding::load_embedding_tensor,
    embedding_db::EmbeddingDb,
    error::Result,
};

/// A reranked document with its MaxSim score.
#[derive(Debug, Clone)]
pub struct RankedDocument {
    pub doc_num_id: u64,
    pub score: f32,
}

/// Rerank candidate documents using ColBERT MaxSim scoring.
///
/// For each candidate document:
/// 1. Load its embedding matrix from the database
/// 2. Compute similarity matrix: query_emb @ doc_emb^T
/// 3. Take row-wise max (best matching document token per query token)
/// 4. Sum the maxes to get the MaxSim score
///
/// Returns candidates sorted by score descending.
pub fn rerank(
    query_embedding: &Tensor,
    candidate_ids: &[u64],
    embedding_db: &EmbeddingDb,
) -> Result<Vec<RankedDocument>> {
    // Load embeddings and compute MaxSim in parallel across candidates.
    let mut ranked: Vec<RankedDocument> = candidate_ids
        .par_iter()
        .filter_map(|&doc_id| {
            let doc_embedding =
                load_embedding_tensor(embedding_db, doc_id).ok().flatten()?;
            let score = maxsim(query_embedding, &doc_embedding).ok()?;
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

/// Compute the MaxSim score between a query embedding and a document embedding.
///
/// query_embedding: [Q, D] where Q = query tokens, D = embedding dimension
/// doc_embedding: [T, D] where T = document tokens, D = embedding dimension
///
/// MaxSim = sum over query tokens of max(query_token . doc_token for all doc tokens)
fn maxsim(query_embedding: &Tensor, doc_embedding: &Tensor) -> Result<f32> {
    // Compute similarity matrix [Q, T] = query_emb @ doc_emb^T
    let sim_matrix = query_embedding
        .matmul(&doc_embedding.t().map_err(map_candle_err)?)
        .map_err(map_candle_err)?;

    // Take max along dimension 1 (best document token per query token)
    let row_maxes = sim_matrix.max(1).map_err(map_candle_err)?;

    // Sum the maxes to get the final score
    let score = row_maxes
        .sum_all()
        .map_err(map_candle_err)?
        .to_scalar::<f32>()
        .map_err(map_candle_err)?;

    Ok(score)
}

fn map_candle_err(e: candle_core::Error) -> crate::error::Error {
    crate::error::Error::Config(format!("tensor computation error: {e}"))
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    fn make_tensor(data: &[f32], shape: (usize, usize)) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn maxsim_identical_vectors() {
        // Query: 1 token, dim=3: [1, 0, 0]
        // Doc: 1 token, dim=3: [1, 0, 0]
        // MaxSim = max([1.0]) = 1.0 (dot product of identical unit vectors)
        let q = make_tensor(&[1.0, 0.0, 0.0], (1, 3));
        let d = make_tensor(&[1.0, 0.0, 0.0], (1, 3));
        let score = maxsim(&q, &d).unwrap();
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn maxsim_orthogonal_vectors() {
        let q = make_tensor(&[1.0, 0.0, 0.0], (1, 3));
        let d = make_tensor(&[0.0, 1.0, 0.0], (1, 3));
        let score = maxsim(&q, &d).unwrap();
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn maxsim_multiple_query_tokens() {
        // 2 query tokens, 3 doc tokens, dim=2
        let q = make_tensor(&[1.0, 0.0, 0.0, 1.0], (2, 2));
        let d = make_tensor(&[1.0, 0.0, 0.0, 1.0, 0.5, 0.5], (3, 2));
        // sim_matrix [2, 3]:
        // q[0]=[1,0] . d[0]=[1,0]=1.0, d[1]=[0,1]=0.0, d[2]=[0.5,0.5]=0.5
        // q[1]=[0,1] . d[0]=[1,0]=0.0, d[1]=[0,1]=1.0, d[2]=[0.5,0.5]=0.5
        // row maxes: [1.0, 1.0], sum = 2.0
        let score = maxsim(&q, &d).unwrap();
        assert!((score - 2.0).abs() < 1e-6);
    }

    #[test]
    fn rerank_with_stored_embeddings() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.redb")).unwrap();

        // Store two document embeddings (2 tokens, dim=3)
        db.store(1, 2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        db.store(2, 2, 3, &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

        // Query embedding: 1 token, dim=3, pointing in [1, 0, 0] direction
        let query = make_tensor(&[1.0, 0.0, 0.0], (1, 3));

        let results = rerank(&query, &[1, 2], &db).unwrap();
        assert_eq!(results.len(), 2);
        // Doc 1 should rank higher (has a [1,0,0] token matching query)
        assert_eq!(results[0].doc_num_id, 1);
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn rerank_skips_missing_embeddings() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.redb")).unwrap();

        db.store(1, 1, 2, &[1.0, 0.0]).unwrap();
        // Doc 2 has no embedding

        let query = make_tensor(&[1.0, 0.0], (1, 2));
        let results = rerank(&query, &[1, 2], &db).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_num_id, 1);
    }
}
