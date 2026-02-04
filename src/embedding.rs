use candle_core::Tensor;

use crate::{
    embedding_db::EmbeddingDb,
    error::Result,
    model_manager::ModelManager,
};

/// Encode a batch of documents and store their embeddings in the database.
///
/// Takes (doc_numeric_id, document_text) pairs, encodes them via the
/// ColBERT model, and stores the resulting per-token embeddings.
///
/// Returns the number of documents successfully embedded.
pub fn embed_and_store(
    model: &mut ModelManager,
    db: &EmbeddingDb,
    documents: &[(u64, String)],
) -> Result<usize> {
    if documents.is_empty() {
        return Ok(0);
    }

    let texts: Vec<String> = documents.iter().map(|(_, t)| t.clone()).collect();
    let embeddings = model.encode_documents(&texts)?;

    // embeddings shape: [batch_size, num_tokens, dimension]
    let dims = embeddings.dims3().map_err(|e| {
        crate::error::Error::Config(format!(
            "unexpected embedding tensor shape: {e}"
        ))
    })?;
    let (batch_size, _num_tokens, dimension) = dims;

    for (i, (doc_id, _)) in documents.iter().enumerate().take(batch_size) {
        let doc_embedding = embeddings.get(i).map_err(|e| {
            crate::error::Error::Config(format!(
                "failed to extract embedding for doc index {i}: {e}"
            ))
        })?;

        let flat = tensor_to_flat_f32(&doc_embedding)?;
        let num_tokens = flat.len() / dimension;

        db.store(*doc_id, num_tokens as u32, dimension as u32, &flat)?;
    }

    Ok(batch_size)
}

/// Convert a 2D Tensor [tokens, dimension] into a flat Vec<f32>.
fn tensor_to_flat_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    let flat = tensor
        .flatten_all()
        .map_err(|e| {
            crate::error::Error::Config(format!(
                "failed to flatten tensor: {e}"
            ))
        })?
        .to_vec1::<f32>()
        .map_err(|e| {
            crate::error::Error::Config(format!(
                "failed to convert tensor to f32: {e}"
            ))
        })?;
    Ok(flat)
}

/// Load a document's embedding from the database and convert to a Tensor.
///
/// Returns None if the document has no stored embedding.
pub fn load_embedding_tensor(
    db: &EmbeddingDb,
    doc_id: u64,
) -> Result<Option<Tensor>> {
    let matrix = match db.load(doc_id)? {
        Some(m) => m,
        None => return Ok(None),
    };

    let num_tokens = matrix.num_tokens as usize;
    let dimension = matrix.dimension as usize;

    // Collect all token embeddings into a flat vector.
    let mut data = Vec::with_capacity(num_tokens * dimension);
    for i in 0..num_tokens {
        data.extend_from_slice(matrix.token_embedding(i as u32));
    }

    let tensor = Tensor::from_vec(
        data,
        (num_tokens, dimension),
        &candle_core::Device::Cpu,
    )
    .map_err(|e| {
        crate::error::Error::Config(format!(
            "failed to create tensor from embedding data: {e}"
        ))
    })?;

    Ok(Some(tensor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_nonexistent_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.redb")).unwrap();
        assert!(load_embedding_tensor(&db, 999).unwrap().is_none());
    }

    #[test]
    fn roundtrip_tensor_through_db() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.redb")).unwrap();

        // Manually store an embedding (3 tokens, 4 dimensions)
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        db.store(42, 3, 4, &data).unwrap();

        // Load it back as a Tensor
        let tensor = load_embedding_tensor(&db, 42).unwrap().unwrap();
        let dims = tensor.dims2().unwrap();
        assert_eq!(dims, (3, 4));

        let flat: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(flat, data);
    }
}
