use candle_core::Tensor;

use crate::{
    embedding_db::EmbeddingDb,
    error::{Error, Result},
    model_manager::ModelManager,
};

/// Number of documents to submit per `encode_documents` call.
///
/// Keep this larger than pylate-rs' internal 32-document batch size so CPU
/// encoding can fan out across multiple internal batches via Rayon, while still
/// bounding the size of the returned embedding tensor before it is serialized
/// into `EmbeddingDb`.
pub const EMBEDDING_SUBMISSION_BATCH_SIZE: usize = 128;

trait DocumentEncoder {
    fn encode_documents(&mut self, texts: &[String]) -> Result<Tensor>;
}

impl DocumentEncoder for ModelManager {
    fn encode_documents(&mut self, texts: &[String]) -> Result<Tensor> {
        ModelManager::encode_documents(self, texts)
    }
}

/// Encode a batch of documents and store their embeddings in the database.
///
/// Takes `(doc_numeric_id, document_text)` pairs, encodes them via the
/// ColBERT model, and stores the resulting per-token embeddings in
/// `EmbeddingDb` using a single batch transaction.
///
/// Consumes the input vector to avoid cloning document content.
/// Downloads the model on first call if not already cached.
///
/// Returns the number of documents successfully embedded.
pub fn embed_and_store(
    model: &mut ModelManager,
    db: &EmbeddingDb,
    documents: Vec<(u64, String)>,
) -> Result<usize> {
    embed_and_store_with(model, db, documents)
}

/// Encode and store many documents using coarser submission batches.
///
/// docbert submits larger groups of documents to `pylate-rs` than its internal
/// 32-document batch size so CPU execution can parallelize across multiple
/// internal batches. `on_progress` receives the cumulative number of embedded
/// documents after each submission batch is stored.
pub fn embed_and_store_in_batches<F>(
    model: &mut ModelManager,
    db: &EmbeddingDb,
    documents: Vec<(u64, String)>,
    submission_batch_size: usize,
    on_progress: F,
) -> Result<usize>
where
    F: FnMut(usize),
{
    embed_and_store_in_batches_with(
        model,
        db,
        documents,
        submission_batch_size,
        on_progress,
    )
}

fn embed_and_store_with<E: DocumentEncoder>(
    model: &mut E,
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
        Error::Config(format!("unexpected embedding tensor shape: {e}"))
    })?;
    let (batch_size, padded_tokens, dimension) = dims;

    let flat_embeddings = tensor_to_flat_f32(&embeddings)?;
    let doc_stride = padded_tokens * dimension;

    let mut entries = Vec::with_capacity(batch_size);
    for (i, doc_id) in doc_ids.into_iter().enumerate().take(batch_size) {
        let start = i * doc_stride;
        let end = start + doc_stride;
        let doc_embedding =
            flat_embeddings.get(start..end).ok_or_else(|| {
                Error::Config(format!(
                    "failed to slice embedding batch for doc index {i}"
                ))
            })?;
        let trimmed = trim_trailing_padding_rows(doc_embedding, dimension);
        let num_tokens = trimmed.len() / dimension;

        entries.push((
            doc_id,
            num_tokens as u32,
            dimension as u32,
            trimmed.to_vec(),
        ));
    }
    db.batch_store(&entries)?;

    Ok(batch_size)
}

fn embed_and_store_in_batches_with<E, F>(
    model: &mut E,
    db: &EmbeddingDb,
    documents: Vec<(u64, String)>,
    submission_batch_size: usize,
    mut on_progress: F,
) -> Result<usize>
where
    E: DocumentEncoder,
    F: FnMut(usize),
{
    if submission_batch_size == 0 {
        return Err(Error::Config(
            "embedding submission batch size must be greater than zero"
                .to_string(),
        ));
    }

    let mut embedded_total = 0;
    let mut documents = documents.into_iter();

    loop {
        let batch: Vec<(u64, String)> =
            documents.by_ref().take(submission_batch_size).collect();
        if batch.is_empty() {
            break;
        }

        embedded_total += embed_and_store_with(model, db, batch)?;
        on_progress(embedded_total);
    }

    Ok(embedded_total)
}

/// Convert a 2D Tensor [tokens, dimension] into a flat Vec<f32>.
fn tensor_to_flat_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    let flat = tensor
        .flatten_all()
        .map_err(|e| Error::Config(format!("failed to flatten tensor: {e}")))?
        .to_vec1::<f32>()
        .map_err(|e| {
            Error::Config(format!("failed to convert tensor to f32: {e}"))
        })?;
    Ok(flat)
}

/// Trim trailing all-zero token rows introduced by pylate-rs batch padding.
fn trim_trailing_padding_rows(data: &[f32], dimension: usize) -> &[f32] {
    if dimension == 0 || data.len() <= dimension {
        return data;
    }

    let mut end = data.len();
    while end > dimension {
        let row = &data[end - dimension..end];
        if row.iter().all(|&value| value == 0.0) {
            end -= dimension;
        } else {
            break;
        }
    }

    &data[..end]
}

/// Load multiple document embeddings from the database and convert to Tensors.
///
/// Returns a vector of `(doc_id, Option<Tensor>)` preserving input order.
/// Missing embeddings return `None`. Uses a single database transaction
/// for efficiency.
///
/// Each returned tensor has shape `[num_tokens, dimension]`.
pub fn batch_load_embedding_tensors(
    db: &EmbeddingDb,
    doc_ids: &[u64],
) -> Result<Vec<(u64, Option<Tensor>)>> {
    let matrices = db.batch_load(doc_ids)?;

    matrices
        .into_iter()
        .map(|(doc_id, matrix_opt)| {
            let tensor_opt = match matrix_opt {
                Some(matrix) => Some(matrix_to_tensor(matrix)?),
                None => None,
            };
            Ok((doc_id, tensor_opt))
        })
        .collect()
}

/// Convert an EmbeddingMatrix to a Tensor.
fn matrix_to_tensor(
    matrix: crate::embedding_db::EmbeddingMatrix,
) -> Result<Tensor> {
    let num_tokens = matrix.num_tokens as usize;
    let dimension = matrix.dimension as usize;

    Tensor::from_vec(
        matrix.data,
        (num_tokens, dimension),
        &candle_core::Device::Cpu,
    )
    .map_err(|e| {
        Error::Config(format!(
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
    let tensor = Tensor::from_vec(
        matrix.data,
        (num_tokens, dimension),
        &candle_core::Device::Cpu,
    )
    .map_err(|e| {
        Error::Config(format!(
            "failed to create tensor from embedding data: {e}"
        ))
    })?;

    Ok(Some(tensor))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RecordingEncoder {
        call_sizes: Vec<usize>,
    }

    impl RecordingEncoder {
        fn new() -> Self {
            Self {
                call_sizes: Vec::new(),
            }
        }
    }

    impl DocumentEncoder for RecordingEncoder {
        fn encode_documents(&mut self, texts: &[String]) -> Result<Tensor> {
            self.call_sizes.push(texts.len());
            let data = vec![0.0f32; texts.len() * 2];
            Ok(Tensor::from_vec(
                data,
                (texts.len(), 1, 2),
                &candle_core::Device::Cpu,
            )?)
        }
    }

    struct PaddedEncoder;

    impl DocumentEncoder for PaddedEncoder {
        fn encode_documents(&mut self, _texts: &[String]) -> Result<Tensor> {
            // Two documents, padded to 3 token rows. The first document only
            // has 2 real token embeddings; the last row is padding.
            let data = vec![
                1.0f32, 2.0, 3.0, 4.0, 0.0,
                0.0, // doc 1: 2 real rows + padding
                5.0, 6.0, 7.0, 8.0, 9.0, 10.0, // doc 2: 3 real rows
            ];
            Ok(Tensor::from_vec(
                data,
                (2, 3, 2),
                &candle_core::Device::Cpu,
            )?)
        }
    }

    #[test]
    fn trims_trailing_zero_padding_before_storing() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let mut encoder = PaddedEncoder;

        embed_and_store_with(
            &mut encoder,
            &db,
            vec![(1, "short".to_string()), (2, "long".to_string())],
        )
        .unwrap();

        let first = db.load(1).unwrap().unwrap();
        assert_eq!(first.num_tokens, 2);
        assert_eq!(first.dimension, 2);
        assert_eq!(first.data, vec![1.0, 2.0, 3.0, 4.0]);

        let second = db.load(2).unwrap().unwrap();
        assert_eq!(second.num_tokens, 3);
        assert_eq!(second.dimension, 2);
        assert_eq!(second.data, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

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

    #[test]
    fn submission_batches_span_multiple_pylate_internal_batches() {
        const PYLATE_INTERNAL_BATCH_SIZE: usize = 32;

        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let mut encoder = RecordingEncoder::new();
        let doc_count = PYLATE_INTERNAL_BATCH_SIZE * 3;
        let docs: Vec<(u64, String)> = (0..doc_count)
            .map(|i| (i as u64, format!("doc {i}")))
            .collect();

        let embedded = embed_and_store_in_batches_with(
            &mut encoder,
            &db,
            docs,
            EMBEDDING_SUBMISSION_BATCH_SIZE,
            |_| {},
        )
        .unwrap();

        assert_eq!(embedded, doc_count);
        assert_eq!(db.list_ids().unwrap().len(), doc_count);
        assert!(
            encoder.call_sizes.len()
                < doc_count.div_ceil(PYLATE_INTERNAL_BATCH_SIZE),
            "expected outer submission batching to coalesce work, got {:?}",
            encoder.call_sizes
        );
        assert!(
            encoder
                .call_sizes
                .iter()
                .any(|&size| size > PYLATE_INTERNAL_BATCH_SIZE),
            "expected at least one encode call to span multiple internal pylate batches, got {:?}",
            encoder.call_sizes
        );
    }

    #[test]
    fn submission_batches_report_cumulative_progress() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let mut encoder = RecordingEncoder::new();
        let docs: Vec<(u64, String)> =
            (0..85).map(|i| (i as u64, format!("doc {i}"))).collect();
        let mut progress_updates = Vec::new();

        let embedded =
            embed_and_store_in_batches_with(&mut encoder, &db, docs, 40, |n| {
                progress_updates.push(n)
            })
            .unwrap();

        assert_eq!(embedded, 85);
        assert_eq!(encoder.call_sizes, vec![40, 40, 5]);
        assert_eq!(progress_updates, vec![40, 80, 85]);
    }

    #[test]
    fn zero_submission_batch_size_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let mut encoder = RecordingEncoder::new();

        let err = embed_and_store_in_batches_with(
            &mut encoder,
            &db,
            vec![(1, "doc".to_string())],
            0,
            |_| {},
        )
        .unwrap_err();

        assert!(err.to_string().contains(
            "embedding submission batch size must be greater than zero"
        ));
    }
}
