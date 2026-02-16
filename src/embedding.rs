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
/// Consumes the input to avoid cloning document content.
///
/// Returns the number of documents successfully embedded.
pub fn embed_and_store(
    model: &mut ModelManager,
    db: &EmbeddingDb,
    documents: Vec<(u64, String)>,
) -> Result<usize> {
    if documents.is_empty() {
        return Ok(0);
    }

    // Unzip to avoid cloning - we take ownership of the strings
    let (doc_ids, texts): (Vec<u64>, Vec<String>) =
        documents.into_iter().unzip();
    let embeddings = model.encode_documents(&texts)?;

    // embeddings shape: [batch_size, num_tokens, dimension]
    let dims = embeddings.dims3().map_err(|e| {
        crate::error::Error::Config(format!(
            "unexpected embedding tensor shape: {e}"
        ))
    })?;
    let (batch_size, _num_tokens, dimension) = dims;

    let mut entries = Vec::with_capacity(batch_size);
    for (i, doc_id) in doc_ids.into_iter().enumerate().take(batch_size) {
        let doc_embedding = embeddings.get(i).map_err(|e| {
            crate::error::Error::Config(format!(
                "failed to extract embedding for doc index {i}: {e}"
            ))
        })?;

        let flat = tensor_to_flat_f32(&doc_embedding)?;
        let num_tokens = flat.len() / dimension;

        entries.push((doc_id, num_tokens as u32, dimension as u32, flat));
    }
    db.batch_store(&entries)?;

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

/// Load multiple document embeddings from the database and convert to Tensors.
///
/// Returns a vector of `(doc_id, Option<Tensor>)` preserving input order.
/// Uses a single database transaction for efficiency.
pub fn batch_load_embedding_tensors(
    db: &EmbeddingDb,
    doc_ids: &[u64],
) -> Result<Vec<(u64, Option<Tensor>)>> {
    let matrices = db.batch_load(doc_ids)?;

    matrices
        .into_iter()
        .map(|(doc_id, matrix_opt)| {
            let tensor_opt = match matrix_opt {
                Some(matrix) => Some(matrix_to_tensor(&matrix)?),
                None => None,
            };
            Ok((doc_id, tensor_opt))
        })
        .collect()
}

/// Convert an EmbeddingMatrix to a Tensor.
fn matrix_to_tensor(
    matrix: &crate::embedding_db::EmbeddingMatrix,
) -> Result<Tensor> {
    let num_tokens = matrix.num_tokens as usize;
    let dimension = matrix.dimension as usize;

    Tensor::from_vec(
        matrix.data.clone(),
        (num_tokens, dimension),
        &candle_core::Device::Cpu,
    )
    .map_err(|e| {
        crate::error::Error::Config(format!(
            "failed to create tensor from embedding data: {e}"
        ))
    })
}

/// Load a document's embedding from the database and convert to a Tensor.
///
/// Returns `None` if the document has no stored embedding.
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// use docbert::{EmbeddingDb, embedding};
///
/// let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
/// db.store(42, 3, 4, &vec![0.0f32; 12]).unwrap();
///
/// let tensor = embedding::load_embedding_tensor(&db, 42).unwrap().unwrap();
/// assert_eq!(tensor.dims2().unwrap(), (3, 4));
///
/// assert!(embedding::load_embedding_tensor(&db, 999).unwrap().is_none());
/// ```
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
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        assert!(load_embedding_tensor(&db, 999).unwrap().is_none());
    }

    #[test]
    fn roundtrip_tensor_through_db() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();

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
