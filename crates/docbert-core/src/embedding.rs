use candle_core::Tensor;

use crate::{
    embedding_db::EmbeddingDb,
    error::{Error, Result},
    model_manager::ModelManager,
};

/// How many documents to hand to `encode_documents` in one outer batch.
///
/// This stays above docbert-pylate' internal 32-document batch so CPU work can fan
/// out across several inner batches, but not so large that the returned tensor
/// becomes awkward to serialize into `EmbeddingDb`.
pub const EMBEDDING_SUBMISSION_BATCH_SIZE: usize = 128;

trait DocumentEncoder {
    /// Encode documents and return `(embeddings_tensor, per_doc_valid_token_counts)`.
    ///
    /// The tensor has shape `[batch, padded_tokens, dim]`. Each row `i` has
    /// its first `lengths[i]` token rows populated; everything after is
    /// zero-padding from the tokenizer's batch-longest strategy.
    fn encode_documents(
        &mut self,
        texts: &[String],
    ) -> Result<(Tensor, Vec<u32>)>;
}

impl DocumentEncoder for ModelManager {
    fn encode_documents(
        &mut self,
        texts: &[String],
    ) -> Result<(Tensor, Vec<u32>)> {
        ModelManager::encode_documents_with_lengths(self, texts)
    }
}

/// Encoded embedding entry ready to write into [`EmbeddingDb`].
///
/// The tuple fields are `(doc_id, num_tokens, dimension, data)`.
pub type EncodedEmbeddingEntry = (u64, u32, u32, Vec<f32>);

/// Encode a batch of documents without writing them to the database.
///
/// Accepts `(doc_numeric_id, document_text)` pairs, runs them through ColBERT,
/// and returns the per-token embeddings in the same format used by
/// [`EmbeddingDb::batch_store`].
///
/// This is useful when callers need embedding generation to succeed before they
/// mutate other storage layers.
pub fn embed_documents(
    model: &mut ModelManager,
    documents: Vec<(u64, String)>,
) -> Result<Vec<EncodedEmbeddingEntry>> {
    embed_documents_with(model, documents)
}

/// Encode a batch of documents and write the embeddings to the database.
///
/// Accepts `(doc_numeric_id, document_text)` pairs, runs them through ColBERT,
/// and stores the per-token embeddings in one batch transaction.
///
/// The input vector is consumed so callers do not need to clone document text.
/// The model is downloaded on first use if it is not cached yet.
///
/// Returns the number of documents written.
pub fn embed_and_store(
    model: &mut ModelManager,
    db: &EmbeddingDb,
    documents: Vec<(u64, String)>,
) -> Result<usize> {
    embed_and_store_with(model, db, documents)
}

/// Encode and store many documents using larger submission batches.
///
/// docbert hands `docbert-pylate` groups that are bigger than its internal
/// 32-document batch size so CPU work can spread across multiple inner batches.
/// `on_progress` receives the cumulative document count after each batch is stored.
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

fn embed_documents_with<E: DocumentEncoder>(
    model: &mut E,
    documents: Vec<(u64, String)>,
) -> Result<Vec<EncodedEmbeddingEntry>> {
    if documents.is_empty() {
        return Ok(vec![]);
    }

    // Unzip to avoid cloning - we take ownership of the strings
    let (doc_ids, texts): (Vec<u64>, Vec<String>) =
        documents.into_iter().unzip();
    let (embeddings, lengths) = model.encode_documents(&texts)?;

    // embeddings shape: [batch_size, num_tokens, dimension]
    let dims = embeddings.dims3().map_err(|e| {
        Error::Config(format!("unexpected embedding tensor shape: {e}"))
    })?;
    let (batch_size, padded_tokens, dimension) = dims;

    if lengths.len() != batch_size {
        return Err(Error::Config(format!(
            "encoder returned {} lengths for a batch of {}",
            lengths.len(),
            batch_size
        )));
    }

    let flat_embeddings = tensor_to_flat_f32(&embeddings)?;
    let doc_stride = padded_tokens * dimension;

    let mut entries = Vec::with_capacity(batch_size);
    for (i, doc_id) in doc_ids.into_iter().enumerate().take(batch_size) {
        let num_tokens = usize::min(lengths[i] as usize, padded_tokens);
        let start = i * doc_stride;
        // Take only the first `num_tokens` rows; the tail of the doc's
        // slice in the padded tensor is guaranteed-zero padding that
        // `finalize_embeddings` already masked out in pylate. We used
        // to scan for trailing all-zero rows here; with explicit
        // lengths threaded through from the tokenizer, we slice
        // directly in O(num_tokens · dim) instead of the worst-case
        // O(padded_tokens · dim) per doc.
        let trimmed_end = start + num_tokens * dimension;
        let trimmed =
            flat_embeddings.get(start..trimmed_end).ok_or_else(|| {
                Error::Config(format!(
                    "failed to slice embedding batch for doc index {i} (len={num_tokens}, padded={padded_tokens})"
                ))
            })?;

        entries.push((
            doc_id,
            num_tokens as u32,
            dimension as u32,
            trimmed.to_vec(),
        ));
    }

    Ok(entries)
}

fn embed_and_store_with<E: DocumentEncoder>(
    model: &mut E,
    db: &EmbeddingDb,
    documents: Vec<(u64, String)>,
) -> Result<usize> {
    let entries = embed_documents_with(model, documents)?;
    let batch_size = entries.len();
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

/// Load several document embeddings and convert them to tensors.
///
/// The returned `Vec<(doc_id, Option<Tensor>)>` keeps the same order as the
/// input IDs. Missing embeddings come back as `None`.
///
/// Everything is loaded inside one database transaction. Each tensor has shape
/// `[num_tokens, dimension]`.
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

/// Load all embeddings for the requested document families as tensors.
///
/// The returned `(doc_id, Tensor)` rows preserve the ordering contract of
/// [`EmbeddingDb::batch_load_document_families`]: families appear in request
/// order and embeddings within each family are sorted by stored `doc_id`.
pub fn batch_load_document_family_tensors(
    db: &EmbeddingDb,
    base_doc_ids: &[u64],
) -> Result<Vec<(u64, Tensor)>> {
    db.batch_load_document_families(base_doc_ids)?
        .into_iter()
        .map(|(doc_id, matrix)| Ok((doc_id, matrix_to_tensor(matrix)?)))
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

/// Load one document embedding from the database and convert it to a tensor.
///
/// Returns `None` when the document has no stored embedding.
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// use docbert_core::{EmbeddingDb, embedding};
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
    use crate::{DocumentId, chunking::chunk_doc_id};

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
        fn encode_documents(
            &mut self,
            texts: &[String],
        ) -> Result<(Tensor, Vec<u32>)> {
            self.call_sizes.push(texts.len());
            let data = vec![0.0f32; texts.len() * 2];
            let tensor = Tensor::from_vec(
                data,
                (texts.len(), 1, 2),
                &crate::test_util::test_device(),
            )?;
            Ok((tensor, vec![1; texts.len()]))
        }
    }

    struct PaddedEncoder;

    impl DocumentEncoder for PaddedEncoder {
        fn encode_documents(
            &mut self,
            _texts: &[String],
        ) -> Result<(Tensor, Vec<u32>)> {
            // Two documents, padded to 3 token rows. The first document only
            // has 2 real token embeddings; the last row is padding.
            let data = vec![
                1.0f32, 2.0, 3.0, 4.0, 0.0,
                0.0, // doc 1: 2 real rows + padding
                5.0, 6.0, 7.0, 8.0, 9.0, 10.0, // doc 2: 3 real rows
            ];
            let tensor = Tensor::from_vec(
                data,
                (2, 3, 2),
                &crate::test_util::test_device(),
            )?;
            Ok((tensor, vec![2, 3]))
        }
    }

    #[test]
    fn embed_documents_slices_by_reported_lengths() {
        let mut encoder = PaddedEncoder;

        let entries = embed_documents_with(
            &mut encoder,
            vec![(1, "short".to_string()), (2, "long".to_string())],
        )
        .unwrap();

        assert_eq!(entries.len(), 2);
        // Doc 0 reports length 2, so only the first two token rows
        // (two dims each) survive — the tail zero row of the doc's
        // padded slice is dropped by the length slice, not by a zero
        // scan on the row values.
        assert_eq!(entries[0], (1, 2, 2, vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(entries[1], (2, 3, 2, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]));
    }

    #[test]
    fn embed_documents_slices_match_lengths_before_storing() {
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

    /// If the encoder over-reports a length (e.g. a length larger than
    /// the tensor's actual padded dimension), we clamp to the padded
    /// dimension rather than panic. This covers the pathological case
    /// where a tokenizer config and a model forward pass disagree.
    #[test]
    fn embed_documents_clamps_length_at_padded_dim() {
        struct OverReportingEncoder;
        impl DocumentEncoder for OverReportingEncoder {
            fn encode_documents(
                &mut self,
                _texts: &[String],
            ) -> Result<(Tensor, Vec<u32>)> {
                let data = vec![1.0f32, 2.0];
                let tensor = Tensor::from_vec(
                    data,
                    (1, 1, 2),
                    &crate::test_util::test_device(),
                )?;
                // Claims 9 valid tokens against a tensor that only has 1.
                Ok((tensor, vec![9]))
            }
        }

        let mut encoder = OverReportingEncoder;
        let entries =
            embed_documents_with(&mut encoder, vec![(1, "x".into())]).unwrap();
        assert_eq!(entries, vec![(1, 1, 2, vec![1.0, 2.0])]);
    }

    /// Property test: for any batch of docs where each doc reports a
    /// length `L` in [0, padded], slicing by length matches the naive
    /// trailing-all-zero scan on a tensor whose tail is actual zeros.
    /// This is the oracle version of the optimisation we just shipped.
    #[hegel::test(test_cases = 200)]
    fn prop_length_slice_matches_zero_scan(tc: hegel::TestCase) {
        use hegel::generators as gs;

        let dim: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(16));
        let padded: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(32));
        let batch_size: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(8));

        // Build a synthetic [batch, padded, dim] tensor and a
        // `lengths` vec where the first L rows per doc are non-zero
        // and the rest are zero.
        let mut data = vec![0.0f32; batch_size * padded * dim];
        let mut lengths = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let l: usize =
                tc.draw(gs::integers::<usize>().min_value(0).max_value(padded));
            for t in 0..l {
                for d in 0..dim {
                    // Use a non-zero value for valid rows; shape
                    // doesn't matter, only non-zero-ness.
                    data[b * padded * dim + t * dim + d] =
                        (b + t + d + 1) as f32;
                }
            }
            lengths.push(l as u32);
        }

        // Reference: scan trailing all-zero rows per doc, same logic
        // the previous implementation used.
        fn reference_trim(
            data: &[f32],
            batch_size: usize,
            padded: usize,
            dim: usize,
        ) -> Vec<Vec<f32>> {
            (0..batch_size)
                .map(|b| {
                    let start = b * padded * dim;
                    let end = start + padded * dim;
                    let doc = &data[start..end];
                    let mut cut = doc.len();
                    while cut >= dim
                        && doc[cut - dim..cut].iter().all(|&v| v == 0.0)
                    {
                        cut -= dim;
                    }
                    doc[..cut].to_vec()
                })
                .collect()
        }

        // SUT: slice by reported lengths directly.
        fn sut_slice(
            data: &[f32],
            batch_size: usize,
            padded: usize,
            dim: usize,
            lengths: &[u32],
        ) -> Vec<Vec<f32>> {
            (0..batch_size)
                .map(|b| {
                    let start = b * padded * dim;
                    let take = usize::min(lengths[b] as usize, padded) * dim;
                    data[start..start + take].to_vec()
                })
                .collect()
        }

        let reference = reference_trim(&data, batch_size, padded, dim);
        let got = sut_slice(&data, batch_size, padded, dim, &lengths);

        assert_eq!(got, reference);
    }

    #[test]
    fn load_nonexistent_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        assert!(load_embedding_tensor(&db, 999).unwrap().is_none());
    }

    #[test]
    fn batch_load_document_family_tensors_converts_family_embeddings_to_expected_shapes()
     {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let base_doc_id = DocumentId::new("notes", "hello.md").numeric;
        let first_chunk_id = chunk_doc_id(base_doc_id, 1);
        let second_chunk_id = chunk_doc_id(base_doc_id, 2);

        db.store(base_doc_id, 2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        db.store(first_chunk_id, 1, 2, &[7.0, 8.0]).unwrap();
        db.store(second_chunk_id, 3, 1, &[9.0, 10.0, 11.0]).unwrap();

        let loaded =
            batch_load_document_family_tensors(&db, &[base_doc_id]).unwrap();
        let loaded_dims: Vec<(u64, (usize, usize))> = loaded
            .iter()
            .map(|(doc_id, tensor)| (*doc_id, tensor.dims2().unwrap()))
            .collect();

        let mut expected = vec![
            (base_doc_id, (2, 3)),
            (first_chunk_id, (1, 2)),
            (second_chunk_id, (3, 1)),
        ];
        expected.sort_by_key(|(doc_id, _)| *doc_id);
        assert_eq!(loaded_dims, expected);
    }

    #[test]
    fn batch_load_document_family_tensors_supports_base_only_documents() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let base_doc_id = DocumentId::new("notes", "hello.md").numeric;

        db.store(base_doc_id, 2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();

        let loaded =
            batch_load_document_family_tensors(&db, &[base_doc_id]).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, base_doc_id);
        assert_eq!(loaded[0].1.dims2().unwrap(), (2, 2));
        let flat: Vec<f32> =
            loaded[0].1.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn batch_load_document_family_tensors_preserves_family_loader_order() {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let first_base_doc_id = DocumentId::new("notes", "a.md").numeric;
        let second_base_doc_id = DocumentId::new("notes", "b.md").numeric;
        let first_chunk_id = chunk_doc_id(first_base_doc_id, 1);
        let second_chunk_id = chunk_doc_id(second_base_doc_id, 1);

        db.store(first_chunk_id, 1, 1, &[1.0]).unwrap();
        db.store(second_chunk_id, 1, 1, &[2.0]).unwrap();
        db.store(first_base_doc_id, 1, 1, &[3.0]).unwrap();
        db.store(second_base_doc_id, 1, 1, &[4.0]).unwrap();

        let loaded = batch_load_document_family_tensors(
            &db,
            &[second_base_doc_id, first_base_doc_id],
        )
        .unwrap();
        let loaded_ids: Vec<u64> =
            loaded.iter().map(|(doc_id, _)| *doc_id).collect();

        let mut second_family_expected =
            vec![second_base_doc_id, second_chunk_id];
        second_family_expected.sort_unstable();
        let mut first_family_expected = vec![first_base_doc_id, first_chunk_id];
        first_family_expected.sort_unstable();
        let expected_ids: Vec<u64> = second_family_expected
            .into_iter()
            .chain(first_family_expected)
            .collect();

        assert_eq!(loaded_ids, expected_ids);
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
