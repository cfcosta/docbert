use std::cmp::Reverse;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use rayon::prelude::*;
use tokenizers::{
    Encoding,
    PaddingParams,
    PaddingStrategy,
    Tokenizer,
    pad_encodings,
};

use crate::{
    builder::ColbertBuilder,
    error::ColbertError,
    modernbert::{Config as ModernBertConfig, ModernBert},
    types::Similarities,
    utils::normalize_l2,
};

/// An enum to abstract over different underlying BERT-based models.
///
/// This allows `ColBERT` to use different architectures like
/// `BertModel` or `ModernBert` without changing the core logic.
pub enum BaseModel {
    /// A variant holding a `ModernBert` model.
    ModernBert(ModernBert),
    /// A variant holding a standard `BertModel`.
    Bert(BertModel),
}

impl BaseModel {
    /// Performs a forward pass through the appropriate underlying model.
    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        match self {
            BaseModel::ModernBert(model) => {
                model.forward(input_ids, attention_mask)
            }
            BaseModel::Bert(model) => {
                model.forward(input_ids, token_type_ids, Some(attention_mask))
            }
        }
    }
}

/// Normalizes embeddings and zeros out masked rows without changing the sequence length.
///
/// This is the fast path used for document encoding. Documents are tokenized with
/// right-padding to the batch longest sequence, so filtering rows out and then padding them
/// back in produces the same layout as simply zeroing the masked rows after normalization.
/// Keeping the whole operation on-device avoids per-row CPU roundtrips and tiny tensor ops.
pub(crate) fn normalize_and_mask_padded(
    embeddings: &Tensor,
    attention_mask: &Tensor,
) -> Result<Tensor, candle_core::Error> {
    let normalized = normalize_l2(embeddings)?;
    let mask = attention_mask.to_dtype(normalized.dtype())?.unsqueeze(2)?;
    normalized.broadcast_mul(&mask)
}

/// Filters rows with a mask, normalizes the kept rows, and pads back to the batch max length.
#[allow(dead_code)]
pub(crate) fn filter_normalize_and_pad_compact(
    embeddings: &Tensor,
    attention_mask: &Tensor,
    device: &Device,
) -> Result<Tensor, candle_core::Error> {
    let (batch_size, _, dim) = embeddings.dims3()?;
    let dtype = embeddings.dtype();
    let mut processed_embeddings: Vec<Tensor> = Vec::with_capacity(batch_size);
    let mut max_len = 0;

    for i in 0..batch_size {
        let single_embedding = embeddings.i(i)?;
        let single_mask = attention_mask.i(i)?.to_vec1::<u32>()?;

        let mut kept_rows = Vec::new();
        for (j, &mask_val) in single_mask.iter().enumerate() {
            if mask_val == 1 {
                kept_rows.push(single_embedding.i(j)?);
            }
        }

        let (normalized, current_len) = if kept_rows.is_empty() {
            let zeros = Tensor::zeros((1, dim), dtype, device)?;
            (zeros, 1)
        } else {
            let filtered = Tensor::stack(&kept_rows, 0)?;
            let len = filtered.dim(0)?;
            (normalize_l2(&filtered)?, len)
        };

        if current_len > max_len {
            max_len = current_len;
        }
        processed_embeddings.push(normalized);
    }

    let mut padded_tensors = Vec::with_capacity(batch_size);
    for tensor in &processed_embeddings {
        let current_len = tensor.dim(0)?;
        let dim = tensor.dim(1)?;
        let pad_len = max_len - current_len;

        if pad_len > 0 {
            let padding = Tensor::zeros((pad_len, dim), dtype, device)?;
            let padded = Tensor::cat(&[tensor, &padding], 0)?;
            padded_tensors.push(padded);
        } else {
            padded_tensors.push(tensor.clone());
        }
    }

    Tensor::stack(&padded_tensors, 0)
}

/// Fast path for right-padded masks: normalize on-device, zero masked rows, then trim the
/// shared padded suffix down to the batch's maximum valid length.
pub(crate) fn normalize_mask_and_truncate_right_padded(
    embeddings: &Tensor,
    attention_mask: &Tensor,
    max_len: usize,
) -> Result<Tensor, candle_core::Error> {
    let masked = normalize_and_mask_padded(embeddings, attention_mask)?;
    masked.narrow(1, 0, max_len.max(1))
}

pub(crate) fn concatenate_embedding_batches(
    embeddings: Vec<Tensor>,
) -> Result<Tensor, candle_core::Error> {
    if embeddings.is_empty() {
        return Err(candle_core::Error::Msg(
            "embedding batches cannot be empty".into(),
        ));
    }
    if embeddings.len() == 1 {
        return Ok(embeddings.into_iter().next().unwrap());
    }

    let mut max_tokens = 0;
    let mut needs_padding = false;
    for batch in &embeddings {
        let (_, tokens, _) = batch.dims3()?;
        if max_tokens == 0 {
            max_tokens = tokens;
        } else if tokens != max_tokens {
            needs_padding = true;
            max_tokens = max_tokens.max(tokens);
        }
    }

    if !needs_padding {
        return Tensor::cat(&embeddings, 0);
    }

    let mut padded_batches = Vec::with_capacity(embeddings.len());
    for batch in embeddings {
        let (batch_size, tokens, dim) = batch.dims3()?;
        if tokens == max_tokens {
            padded_batches.push(batch);
            continue;
        }

        let padding = Tensor::zeros(
            (batch_size, max_tokens - tokens, dim),
            batch.dtype(),
            batch.device(),
        )?;
        padded_batches.push(Tensor::cat(&[&batch, &padding], 1)?);
    }

    Tensor::cat(&padded_batches, 0)
}

/// Computes MaxSim similarity scores between query and document embeddings.
///
/// `queries_embeddings` has shape `(n_queries, q_tokens, dim)` and
/// `documents_embeddings` has shape `(n_documents, d_tokens, dim)`. The
/// returned matrix is `(n_queries, n_documents)` where each entry is
/// `Σ_t max_k dot(Q[i,t], D[j,k])`.
pub(crate) fn compute_similarities(
    queries_embeddings: &Tensor,
    documents_embeddings: &Tensor,
) -> Result<Similarities, ColbertError> {
    let scores =
        compute_raw_similarity(queries_embeddings, documents_embeddings)?;
    let max_scores = scores.max(3)?;
    let similarities = max_scores.sum(2)?;
    let similarities_vec = similarities.to_vec2::<f32>()?;
    Ok(Similarities {
        data: similarities_vec,
    })
}

/// Computes the raw, un-reduced similarity matrix between query and document embeddings.
///
/// Output shape is `(n_queries, n_documents, q_tokens, d_tokens)` where each
/// entry is `dot(Q[i,t], D[j,k])`.
pub(crate) fn compute_raw_similarity(
    queries_embeddings: &Tensor,
    documents_embeddings: &Tensor,
) -> Result<Tensor, ColbertError> {
    queries_embeddings
        .unsqueeze(1)?
        .broadcast_matmul(&documents_embeddings.transpose(1, 2)?.unsqueeze(0)?)
        .map_err(ColbertError::from)
}

/// The main ColBERT model structure.
///
/// This struct encapsulates the language model, a linear projection layer,
/// the tokenizer, and all necessary configuration for performing encoding
/// and similarity calculations based on the ColBERT architecture.
pub struct ColBERT {
    pub(crate) model: BaseModel,
    pub(crate) linear: Linear,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) mask_token_id: u32,
    pub(crate) mask_token: String,
    pub(crate) query_prefix: String,
    pub(crate) document_prefix: String,
    pub(crate) do_query_expansion: bool,
    pub(crate) attend_to_expansion_tokens: bool,
    pub(crate) query_length: usize,
    pub(crate) document_length: usize,
    pub(crate) batch_size: usize,
    /// The device (CPU or GPU) on which the model is loaded.
    pub device: Device,
}

impl ColBERT {
    /// Creates a new instance of the `ColBERT` model from byte buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weights: Vec<u8>,
        dense_weights: Vec<u8>,
        tokenizer_bytes: Vec<u8>,
        config_bytes: Vec<u8>,
        dense_config_bytes: Vec<u8>,
        query_prefix: String,
        document_prefix: String,
        mask_token: String,
        do_query_expansion: bool,
        attend_to_expansion_tokens: bool,
        query_length: Option<usize>,
        document_length: Option<usize>,
        batch_size: Option<usize>,
        device: &Device,
    ) -> Result<Self, ColbertError> {
        let vb =
            VarBuilder::from_buffered_safetensors(weights, DType::F32, device)?;

        let config_value: serde_json::Value =
            serde_json::from_slice(&config_bytes)?;
        let architectures = config_value["architectures"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ColbertError::Operation(
                    "Missing or invalid 'architectures' in config.json".into(),
                )
            })?;

        let model = match architectures {
            "ModernBertModel" => {
                let config: ModernBertConfig =
                    serde_json::from_slice(&config_bytes)?;
                let model = ModernBert::load(vb.clone(), &config)?;
                BaseModel::ModernBert(model)
            }
            "BertForMaskedLM" | "BertModel" => {
                let config: BertConfig = serde_json::from_slice(&config_bytes)?;
                let model = BertModel::load(vb.clone(), &config)?;
                BaseModel::Bert(model)
            }
            arch => {
                return Err(ColbertError::Operation(format!(
                    "Unsupported architecture: {}",
                    arch
                )));
            }
        };

        let dense_config: serde_json::Value =
            serde_json::from_slice(&dense_config_bytes)?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)?;

        let mask_token_id =
            tokenizer.token_to_id(mask_token.as_str()).ok_or_else(|| {
                ColbertError::Operation(format!(
                    "Token '{}' not found in the tokenizer's vocabulary.",
                    mask_token
                ))
            })?;

        let dense_vb = VarBuilder::from_buffered_safetensors(
            dense_weights,
            DType::F32,
            device,
        )?;
        let in_features = dense_config["in_features"]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| {
                ColbertError::Operation(
                    "Missing 'in_features' in dense config".into(),
                )
            })?;
        let out_features = dense_config["out_features"]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| {
                ColbertError::Operation(
                    "Missing 'out_features' in dense config".into(),
                )
            })?;

        let linear = candle_nn::linear_no_bias(
            in_features,
            out_features,
            dense_vb.pp("linear"),
        )?;

        // If do_query_expansion is false, attend_to_expansion_tokens should also be false
        let final_attend_to_expansion_tokens = if !do_query_expansion {
            false
        } else {
            attend_to_expansion_tokens
        };

        Ok(Self {
            model,
            linear,
            tokenizer,
            mask_token_id,
            mask_token,
            query_prefix,
            document_prefix,
            do_query_expansion,
            attend_to_expansion_tokens: final_attend_to_expansion_tokens,
            query_length: query_length.unwrap_or(32),
            document_length: document_length.unwrap_or(180),
            batch_size: batch_size.unwrap_or(32),
            device: device.clone(),
        })
    }

    /// Creates a `ColbertBuilder` to construct a `ColBERT` model from a Hugging Face repository.
    pub fn from(repo_id: &str) -> ColbertBuilder {
        ColbertBuilder::new(repo_id)
    }

    /// Finalizes projected embeddings after the linear layer.
    ///
    /// Queries without query expansion and documents both use an on-device right-padding fast
    /// path that preserves the same batch max token count while avoiding row-by-row CPU work.
    fn finalize_embeddings(
        &self,
        projected_embeddings: &Tensor,
        attention_mask: &Tensor,
        max_valid_len: usize,
        is_query: bool,
    ) -> Result<Tensor, candle_core::Error> {
        if is_query && self.do_query_expansion {
            normalize_l2(projected_embeddings).map_err(candle_core::Error::from)
        } else {
            normalize_mask_and_truncate_right_padded(
                projected_embeddings,
                attention_mask,
                max_valid_len,
            )
        }
    }

    /// Encodes a batch of sentences (queries or documents) into embeddings.
    ///
    /// On CPU, this method leverages Rayon for parallel batch processing
    /// to accelerate encoding. On accelerators (GPU), it processes batches sequentially.
    pub fn encode(
        &mut self,
        sentences: &[String],
        is_query: bool,
    ) -> Result<Tensor, ColbertError> {
        if sentences.is_empty() {
            return Err(ColbertError::Operation(
                "Input sentences cannot be empty.".into(),
            ));
        }

        if self.device.is_cpu() {
            let mut tokenized_batches = Vec::new();
            for batch_sentences in sentences.chunks(self.batch_size) {
                tokenized_batches
                    .push(self.tokenize(batch_sentences, is_query)?);
            }

            let all_embeddings = tokenized_batches
                .into_par_iter()
                .map(
                    |(
                        token_ids,
                        attention_mask,
                        token_type_ids,
                        max_valid_len,
                    )|
                     -> Result<Tensor, ColbertError> {
                        let token_embeddings = self.model.forward(
                            &token_ids,
                            &attention_mask,
                            &token_type_ids,
                        )?;
                        let token_embeddings =
                            if token_embeddings.is_contiguous() {
                                token_embeddings
                            } else {
                                token_embeddings.contiguous()?
                            };
                        let projected_embeddings =
                            self.linear.forward(&token_embeddings)?;

                        self.finalize_embeddings(
                            &projected_embeddings,
                            &attention_mask,
                            max_valid_len,
                            is_query,
                        )
                        .map_err(ColbertError::from)
                    },
                )
                .collect::<Result<Vec<_>, _>>()?;

            return concatenate_embedding_batches(all_embeddings)
                .map_err(ColbertError::from);
        }

        // Fallback to sequential processing for GPU, WASM, or other devices.
        if !is_query && sentences.len() > self.batch_size {
            let texts_with_prefix: Vec<_> = sentences
                .iter()
                .map(|text| format!("{}{}", self.document_prefix, text))
                .collect();
            let _ = self.tokenizer.with_truncation(Some(
                tokenizers::TruncationParams {
                    max_length: self.document_length,
                    ..Default::default()
                },
            ));
            self.tokenizer.with_padding(None);

            let encodings =
                self.tokenizer.encode_batch_fast(texts_with_prefix, true)?;
            let mut indexed_encodings: Vec<(usize, Encoding)> =
                encodings.into_iter().enumerate().collect();
            indexed_encodings.sort_unstable_by_key(|(_, encoding)| {
                Reverse(encoding.get_ids().len())
            });

            let mut inverse = vec![0u32; indexed_encodings.len()];
            for (sorted_idx, (original_idx, _)) in
                indexed_encodings.iter().enumerate()
            {
                inverse[*original_idx] = sorted_idx as u32;
            }
            let inverse_len = inverse.len();
            let mut sorted_encodings: Vec<Encoding> = indexed_encodings
                .into_iter()
                .map(|(_, encoding)| encoding)
                .collect();

            let mut all_embeddings = Vec::with_capacity(
                (sorted_encodings.len() + self.batch_size - 1)
                    / self.batch_size,
            );
            let padding = PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            let max_tokens_per_batch =
                self.batch_size * self.document_length.max(1);
            let mut batch_start = 0usize;
            while batch_start < sorted_encodings.len() {
                let first_len =
                    sorted_encodings[batch_start].get_ids().len().max(1);
                let batch_cap = (max_tokens_per_batch / first_len).max(1);
                let batch_end =
                    (batch_start + batch_cap).min(sorted_encodings.len());
                let batch_encodings =
                    &mut sorted_encodings[batch_start..batch_end];
                let first_len = batch_encodings
                    .first()
                    .map_or(0, |encoding| encoding.get_ids().len());
                let last_len = batch_encodings
                    .last()
                    .map_or(0, |encoding| encoding.get_ids().len());
                let has_padding = first_len != last_len;
                if has_padding {
                    pad_encodings(batch_encodings, &padding)?;
                }
                let (token_ids, attention_mask, token_type_ids, max_valid_len) =
                    self.tensorize_encodings(batch_encodings, false)?;

                let token_embeddings = {
                    #[cfg(feature = "cuda")]
                    {
                        let valid_lens = if has_padding {
                            Some(
                                batch_encodings
                                    .iter()
                                    .map(|encoding| encoding.get_ids().len())
                                    .collect::<Vec<_>>(),
                            )
                        } else {
                            None
                        };

                        if !has_padding {
                            if let BaseModel::ModernBert(model) = &self.model {
                                model.forward_unmasked(&token_ids)?
                            } else {
                                self.model.forward(
                                    &token_ids,
                                    &attention_mask,
                                    &token_type_ids,
                                )?
                            }
                        } else if let (
                            BaseModel::ModernBert(model),
                            Some(valid_lens),
                        ) = (&self.model, valid_lens.as_ref())
                        {
                            model
                                .forward_varlen_padded(&token_ids, valid_lens)?
                        } else {
                            self.model.forward(
                                &token_ids,
                                &attention_mask,
                                &token_type_ids,
                            )?
                        }
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        self.model.forward(
                            &token_ids,
                            &attention_mask,
                            &token_type_ids,
                        )?
                    }
                };
                let token_embeddings = if token_embeddings.is_contiguous() {
                    token_embeddings
                } else {
                    token_embeddings.contiguous()?
                };
                let projected_embeddings =
                    self.linear.forward(&token_embeddings)?;
                let final_embeddings = self.finalize_embeddings(
                    &projected_embeddings,
                    &attention_mask,
                    max_valid_len,
                    false,
                )?;
                all_embeddings.push(final_embeddings);
                batch_start = batch_end;
            }

            let embeddings = concatenate_embedding_batches(all_embeddings)
                .map_err(ColbertError::from)?;
            let restore_indices =
                Tensor::from_vec(inverse, inverse_len, &self.device)?;
            return embeddings
                .index_select(&restore_indices, 0)
                .map_err(ColbertError::from);
        }

        let mut all_embeddings = Vec::with_capacity(
            (sentences.len() + self.batch_size - 1) / self.batch_size,
        );
        for batch_sentences in sentences.chunks(self.batch_size) {
            let (token_ids, attention_mask, token_type_ids, max_valid_len) =
                self.tokenize(batch_sentences, is_query)?;

            let token_embeddings = self.model.forward(
                &token_ids,
                &attention_mask,
                &token_type_ids,
            )?;
            let token_embeddings = if token_embeddings.is_contiguous() {
                token_embeddings
            } else {
                token_embeddings.contiguous()?
            };

            let projected_embeddings =
                self.linear.forward(&token_embeddings)?;

            let final_embeddings = self.finalize_embeddings(
                &projected_embeddings,
                &attention_mask,
                max_valid_len,
                is_query,
            )?;

            all_embeddings.push(final_embeddings);
        }

        concatenate_embedding_batches(all_embeddings)
            .map_err(ColbertError::from)
    }

    /// Calculates the similarity scores between query and document embeddings.
    pub fn similarity(
        &self,
        queries_embeddings: &Tensor,
        documents_embeddings: &Tensor,
    ) -> Result<Similarities, ColbertError> {
        compute_similarities(queries_embeddings, documents_embeddings)
    }

    /// Computes the raw, un-reduced similarity matrix between query and document embeddings.
    pub fn raw_similarity(
        &self,
        queries_embeddings: &Tensor,
        documents_embeddings: &Tensor,
    ) -> Result<Tensor, ColbertError> {
        compute_raw_similarity(queries_embeddings, documents_embeddings)
    }

    fn tensorize_encodings(
        &self,
        encodings: &[Encoding],
        is_query: bool,
    ) -> Result<(Tensor, Tensor, Tensor, usize), ColbertError> {
        let device = &self.device;
        let batch_size = encodings.len();
        if batch_size == 0 {
            return Err(ColbertError::Operation(
                "Input sentences cannot be empty.".into(),
            ));
        }

        // Collect tokenization outputs into flat vectors. For documents, the padded sequence
        // length already equals the batch max valid length. For non-expansion queries, compute the
        // max valid length while we are already walking the CPU-side attention masks so the CUDA
        // path can skip its own mask-length readback.
        let seq_len = encodings.first().map_or(0, |e| e.get_ids().len());
        let needs_query_valid_len = is_query
            && !self.do_query_expansion
            && !self.attend_to_expansion_tokens;
        let needs_token_type_ids = matches!(&self.model, BaseModel::Bert(_));
        let mut max_valid_len = if needs_query_valid_len {
            1
        } else {
            seq_len.max(1)
        };
        let flat_len = batch_size * seq_len;
        let mut ids_vec = Vec::<u32>::with_capacity(flat_len);
        let mut mask_vec = Vec::<u32>::with_capacity(flat_len);
        let mut type_ids_vec =
            needs_token_type_ids.then(|| Vec::<u32>::with_capacity(flat_len));
        for enc in encodings {
            ids_vec.extend(enc.get_ids());
            let attention = enc.get_attention_mask();
            if needs_query_valid_len {
                let mut valid_len = 0usize;
                for &mask in attention {
                    valid_len += mask as usize;
                    mask_vec.push(mask);
                }
                max_valid_len = max_valid_len.max(valid_len.max(1));
            } else {
                mask_vec.extend(attention);
            }
            if let Some(type_ids_vec) = type_ids_vec.as_mut() {
                type_ids_vec.extend(enc.get_type_ids());
            }
        }

        let token_ids =
            Tensor::from_vec(ids_vec, (batch_size, seq_len), device)?;
        let mut attention_mask =
            Tensor::from_vec(mask_vec, (batch_size, seq_len), device)?;
        let token_type_ids = match type_ids_vec {
            Some(type_ids_vec) => {
                Tensor::from_vec(type_ids_vec, (batch_size, seq_len), device)?
            }
            None => Tensor::zeros((1, 1), DType::U32, device)?,
        };

        if is_query && self.attend_to_expansion_tokens {
            attention_mask = attention_mask.ones_like()?;
        }

        Ok((token_ids, attention_mask, token_type_ids, max_valid_len))
    }

    /// Tokenizes a batch of texts, applying specific logic for queries and documents.
    pub(crate) fn tokenize(
        &mut self,
        texts: &[String],
        is_query: bool,
    ) -> Result<(Tensor, Tensor, Tensor, usize), ColbertError> {
        let (prefix, max_length) = if is_query {
            (self.query_prefix.as_str(), self.query_length)
        } else {
            (self.document_prefix.as_str(), self.document_length)
        };

        let texts_with_prefix: Vec<_> = texts
            .iter()
            .map(|text| format!("{}{}", prefix, text))
            .collect();

        let _ = self.tokenizer.with_truncation(Some(
            tokenizers::TruncationParams {
                max_length,
                ..Default::default()
            },
        ));

        let padding_params = if is_query {
            PaddingParams {
                strategy: PaddingStrategy::Fixed(max_length),
                pad_id: self.mask_token_id,
                pad_token: self.mask_token.clone(),
                ..Default::default()
            }
        } else {
            PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }
        };
        self.tokenizer.with_padding(Some(padding_params));

        let encodings =
            self.tokenizer.encode_batch_fast(texts_with_prefix, true)?;
        self.tensorize_encodings(&encodings, is_query)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::{
        concatenate_embedding_batches,
        filter_normalize_and_pad_compact,
        normalize_and_mask_padded,
        normalize_mask_and_truncate_right_padded,
    };

    #[test]
    fn fast_document_path_matches_compact_path_for_right_padded_masks() {
        let device = Device::Cpu;
        let embeddings = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // doc 1
                9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, // doc 2
            ],
            (2, 4, 2),
            &device,
        )
        .unwrap();
        let attention_mask =
            Tensor::from_vec(vec![1u32, 1, 1, 1, 1, 1, 0, 0], (2, 4), &device)
                .unwrap();

        let compact = filter_normalize_and_pad_compact(
            &embeddings,
            &attention_mask,
            &device,
        )
        .unwrap();
        let fast =
            normalize_and_mask_padded(&embeddings, &attention_mask).unwrap();

        let compact = compact.to_vec3::<f32>().unwrap();
        let fast = fast.to_vec3::<f32>().unwrap();

        assert_eq!(compact.len(), fast.len());
        for (compact_doc, fast_doc) in compact.iter().zip(fast.iter()) {
            assert_eq!(compact_doc.len(), fast_doc.len());
            for (compact_row, fast_row) in
                compact_doc.iter().zip(fast_doc.iter())
            {
                assert_eq!(compact_row.len(), fast_row.len());
                for (compact_value, fast_value) in
                    compact_row.iter().zip(fast_row.iter())
                {
                    assert!((compact_value - fast_value).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn fast_query_path_matches_compact_path_for_right_padded_masks() {
        let device = Device::Cpu;
        let embeddings = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // q1
                9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, // q2
            ],
            (2, 4, 2),
            &device,
        )
        .unwrap();
        let attention_mask =
            Tensor::from_vec(vec![1u32, 1, 1, 0, 1, 1, 0, 0], (2, 4), &device)
                .unwrap();

        let compact = filter_normalize_and_pad_compact(
            &embeddings,
            &attention_mask,
            &device,
        )
        .unwrap();
        let fast = normalize_mask_and_truncate_right_padded(
            &embeddings,
            &attention_mask,
            3,
        )
        .unwrap();

        assert_eq!(
            compact.to_vec3::<f32>().unwrap(),
            fast.to_vec3::<f32>().unwrap()
        );
    }

    #[test]
    fn fast_document_path_zeroes_masked_rows() {
        let device = Device::Cpu;
        let embeddings = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
            (1, 4, 2),
            &device,
        )
        .unwrap();
        let attention_mask =
            Tensor::from_vec(vec![1u32, 1, 0, 0], (1, 4), &device).unwrap();

        let fast = normalize_and_mask_padded(&embeddings, &attention_mask)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();

        assert!((fast[0][0][0] - 1.0).abs() < 1e-6);
        assert!((fast[0][0][1] - 0.0).abs() < 1e-6);
        assert!((fast[0][1][0] - 0.0).abs() < 1e-6);
        assert!((fast[0][1][1] - 1.0).abs() < 1e-6);
        assert_eq!(fast[0][2], vec![0.0, 0.0]);
        assert_eq!(fast[0][3], vec![0.0, 0.0]);
    }

    #[test]
    fn concatenate_embedding_batches_pads_variable_sequence_lengths() {
        let device = Device::Cpu;
        let first = Tensor::zeros((64, 514, 128), DType::F32, &device).unwrap();
        let second =
            Tensor::zeros((64, 519, 128), DType::F32, &device).unwrap();

        assert!(Tensor::cat(&[&first, &second], 0).is_err());

        let combined =
            concatenate_embedding_batches(vec![first, second]).unwrap();
        assert_eq!(combined.dims3().unwrap(), (128, 519, 128));
    }
}

#[cfg(test)]
mod hegel_tests {
    //! Hegel property tests for `model.rs` internals.
    //!
    //! Covers two tiers that need `pub(crate)` access and therefore can't live
    //! in `tests/properties.rs`:
    //!
    //! - **C** — `compute_similarities` / `compute_raw_similarity`: MaxSim
    //!   differential against a hand-rolled reference, shape contract,
    //!   zero-doc-token monotonicity, and query-scaling linearity.
    //! - **E** — masking and concatenation helpers: the right-padded fast
    //!   path must match the compact path, `normalize_and_mask_padded` must
    //!   zero masked rows and bound unmasked rows, and
    //!   `concatenate_embedding_batches` must preserve the batch/dim invariants
    //!   while zero-padding short batches up to the longest token length.
    use candle_core::{Device, Tensor};
    use hegel::{TestCase, generators as gs};

    use super::{
        compute_raw_similarity,
        compute_similarities,
        concatenate_embedding_batches,
        filter_normalize_and_pad_compact,
        normalize_and_mask_padded,
        normalize_mask_and_truncate_right_padded,
    };

    /// Selects the test device based on compile-time features, falling back
    /// to CPU if the preferred accelerator can't be initialised at runtime
    /// (e.g., the `cuda` feature is on but no GPU is present in this
    /// environment). Picks CUDA, then Metal, then CPU — matches how
    /// `ColBERT::with_device` prioritises them in production.
    fn test_device() -> Device {
        #[cfg(feature = "cuda")]
        {
            if let Ok(d) = Device::new_cuda(0) {
                return d;
            }
        }
        #[cfg(feature = "metal")]
        {
            if let Ok(d) = Device::new_metal(0) {
                return d;
            }
        }
        Device::Cpu
    }

    // -----------------------------------------------------------------------
    // Shared generators
    // -----------------------------------------------------------------------

    /// Draws an `(b, s, d)` embedding tensor plus an `(b, s)` attention mask.
    /// The mask is arbitrary 0/1 — used for properties that don't require
    /// right-padding (E1, E2).
    #[hegel::composite]
    fn embeddings_with_free_mask(
        tc: TestCase,
        dev: Device,
    ) -> (Tensor, Tensor) {
        let b: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
        let s: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
        let d: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
        let emb_data: Vec<f32> = tc.draw(
            gs::vecs(
                gs::floats::<f32>()
                    .min_value(-5.0)
                    .max_value(5.0)
                    .allow_nan(false)
                    .allow_infinity(false),
            )
            .min_size(b * s * d)
            .max_size(b * s * d),
        );
        let mask_data: Vec<u32> = tc.draw(
            gs::vecs(gs::integers::<u32>().min_value(0).max_value(1))
                .min_size(b * s)
                .max_size(b * s),
        );
        let embeddings = Tensor::from_vec(emb_data, (b, s, d), &dev).unwrap();
        let mask = Tensor::from_vec(mask_data, (b, s), &dev).unwrap();
        (embeddings, mask)
    }

    /// Draws an `(b, s, d)` embedding tensor plus a right-padded attention
    /// mask (each row is a prefix of 1s then 0s). Returns `(emb, mask,
    /// max_valid_len)`. This is exactly the precondition under which the
    /// fast and compact paths must agree (E3, E4).
    #[hegel::composite]
    fn embeddings_with_right_padded_mask(
        tc: TestCase,
        dev: Device,
    ) -> (Tensor, Tensor, usize) {
        let b: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
        let s: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
        let d: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
        let emb_data: Vec<f32> = tc.draw(
            gs::vecs(
                gs::floats::<f32>()
                    .min_value(-5.0)
                    .max_value(5.0)
                    .allow_nan(false)
                    .allow_infinity(false),
            )
            .min_size(b * s * d)
            .max_size(b * s * d),
        );
        let mut mask_flat = Vec::<u32>::with_capacity(b * s);
        let mut max_valid = 0usize;
        for _ in 0..b {
            let valid: usize =
                tc.draw(gs::integers::<usize>().min_value(0).max_value(s));
            max_valid = max_valid.max(valid);
            for j in 0..s {
                mask_flat.push(u32::from(j < valid));
            }
        }
        let embeddings = Tensor::from_vec(emb_data, (b, s, d), &dev).unwrap();
        let mask = Tensor::from_vec(mask_flat, (b, s), &dev).unwrap();
        (embeddings, mask, max_valid)
    }

    /// Draws a non-empty `Vec<Tensor>` with matching `(batch, dim)` and
    /// varying per-batch sequence length. Used to exercise
    /// `concatenate_embedding_batches`.
    #[hegel::composite]
    fn embedding_batch_list(tc: TestCase, dev: Device) -> Vec<Tensor> {
        let n_batches: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
        let batch: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
        let dim: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
        let finite = || {
            gs::floats::<f32>()
                .min_value(-3.0)
                .max_value(3.0)
                .allow_nan(false)
                .allow_infinity(false)
        };
        let mut out = Vec::with_capacity(n_batches);
        for _ in 0..n_batches {
            let tokens: usize =
                tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
            let data: Vec<f32> = tc.draw(
                gs::vecs(finite())
                    .min_size(batch * tokens * dim)
                    .max_size(batch * tokens * dim),
            );
            out.push(
                Tensor::from_vec(data, (batch, tokens, dim), &dev).unwrap(),
            );
        }
        out
    }

    /// Draws query and document embeddings that share the last dim but can
    /// differ in batch size and token count. Used for every C-tier property.
    #[hegel::composite]
    fn query_doc_pair(tc: TestCase, dev: Device) -> (Tensor, Tensor) {
        let dim: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
        let q_batch: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
        let q_tokens: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
        let d_batch: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
        let d_tokens: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
        let finite = || {
            gs::floats::<f32>()
                .min_value(-1.0)
                .max_value(1.0)
                .allow_nan(false)
                .allow_infinity(false)
        };
        let q_data: Vec<f32> = tc.draw(
            gs::vecs(finite())
                .min_size(q_batch * q_tokens * dim)
                .max_size(q_batch * q_tokens * dim),
        );
        let d_data: Vec<f32> = tc.draw(
            gs::vecs(finite())
                .min_size(d_batch * d_tokens * dim)
                .max_size(d_batch * d_tokens * dim),
        );
        let q =
            Tensor::from_vec(q_data, (q_batch, q_tokens, dim), &dev).unwrap();
        let d =
            Tensor::from_vec(d_data, (d_batch, d_tokens, dim), &dev).unwrap();
        (q, d)
    }

    // -----------------------------------------------------------------------
    // E — masking / concat helpers
    // -----------------------------------------------------------------------

    /// E1: `normalize_and_mask_padded` zeros every row whose mask is 0.
    /// E2: rows whose mask is 1 have squared L2 norm ≤ 1 + ε.
    /// These are one property in two assertions because the generator is
    /// shared.
    #[hegel::test(test_cases = 200)]
    fn normalize_and_mask_padded_respects_mask(tc: TestCase) {
        let dev = test_device();
        let (emb, mask) = tc.draw(embeddings_with_free_mask(dev));
        let out = normalize_and_mask_padded(&emb, &mask).unwrap();
        assert_eq!(out.dims(), emb.dims(), "shape must be preserved");

        let out_v: Vec<Vec<Vec<f32>>> = out.to_vec3::<f32>().unwrap();
        let mask_v: Vec<Vec<u32>> = mask.to_vec2::<u32>().unwrap();
        for (b_idx, row_block) in out_v.iter().enumerate() {
            for (s_idx, row) in row_block.iter().enumerate() {
                let bit = mask_v[b_idx][s_idx];
                if bit == 0 {
                    for v in row {
                        assert_eq!(
                            *v, 0.0,
                            "masked row at ({b_idx},{s_idx}) not zeroed",
                        );
                    }
                } else {
                    let n2: f32 = row.iter().map(|v| v * v).sum();
                    assert!(
                        n2 <= 1.0 + 1e-4,
                        "unmasked row at ({b_idx},{s_idx}) has n²={n2}",
                    );
                }
            }
        }
    }

    /// E3: `normalize_mask_and_truncate_right_padded` output shape is
    /// `(batch, max(max_len, 1), dim)`.
    #[hegel::test(test_cases = 200)]
    fn truncate_right_padded_has_expected_shape(tc: TestCase) {
        let dev = test_device();
        let (emb, mask, max_valid) =
            tc.draw(embeddings_with_right_padded_mask(dev));
        let (b, _, d) = emb.dims3().unwrap();
        let out =
            normalize_mask_and_truncate_right_padded(&emb, &mask, max_valid)
                .unwrap();
        assert_eq!(out.dim(0).unwrap(), b);
        assert_eq!(out.dim(1).unwrap(), max_valid.max(1));
        assert_eq!(out.dim(2).unwrap(), d);
    }

    /// E4: under the right-padded-mask precondition the fast path and the
    /// compact path produce the same tensor. This is the single highest-value
    /// property in the suite — a bug that diverges the two would silently
    /// corrupt document embeddings.
    #[hegel::test(test_cases = 200)]
    fn truncate_right_padded_matches_compact(tc: TestCase) {
        let dev = test_device();
        let (emb, mask, max_valid) =
            tc.draw(embeddings_with_right_padded_mask(dev.clone()));
        let fast =
            normalize_mask_and_truncate_right_padded(&emb, &mask, max_valid)
                .unwrap();
        let compact =
            filter_normalize_and_pad_compact(&emb, &mask, &dev).unwrap();

        // When every row in a given batch is masked out, the compact path
        // emits one zero row while the fast path emits `max(max_valid, 1)`
        // zero rows. Both are legitimate zero-padding layouts; only compare
        // the rows that the compact path actually produced.
        let (fast_b, fast_s, fast_d) = fast.dims3().unwrap();
        let (comp_b, comp_s, comp_d) = compact.dims3().unwrap();
        assert_eq!(fast_b, comp_b);
        assert_eq!(fast_d, comp_d);
        let common = fast_s.min(comp_s);
        let fast_cmp = fast.narrow(1, 0, common).unwrap();
        let comp_cmp = compact.narrow(1, 0, common).unwrap();

        let fv: Vec<Vec<Vec<f32>>> = fast_cmp.to_vec3::<f32>().unwrap();
        let cv: Vec<Vec<Vec<f32>>> = comp_cmp.to_vec3::<f32>().unwrap();
        for (fb, cb) in fv.iter().zip(cv.iter()) {
            for (fr, cr) in fb.iter().zip(cb.iter()) {
                for (fv, cv) in fr.iter().zip(cr.iter()) {
                    assert!(
                        (fv - cv).abs() < 1e-5,
                        "fast vs compact divergence: {fv} vs {cv}",
                    );
                }
            }
        }
    }

    /// E5: `concatenate_embedding_batches` is identity on a single-element
    /// input — the fast-path clone returns the tensor unchanged.
    #[hegel::test(test_cases = 100)]
    fn concatenate_single_is_identity(tc: TestCase) {
        let dev = test_device();
        let list = tc.draw(embedding_batch_list(dev));
        let only = list.into_iter().next().unwrap();
        let clone = only.to_vec3::<f32>().unwrap();
        let out = concatenate_embedding_batches(vec![only.clone()]).unwrap();
        let out_v: Vec<Vec<Vec<f32>>> = out.to_vec3::<f32>().unwrap();
        assert_eq!(clone, out_v);
    }

    /// E6: concatenation preserves `dim`, sums `batch` across inputs, and
    /// takes the max `tokens` across inputs. E7: every row beyond a batch's
    /// original token count is zero.
    #[hegel::test(test_cases = 150)]
    fn concatenate_shape_and_zero_padding(tc: TestCase) {
        let dev = test_device();
        let list = tc.draw(embedding_batch_list(dev));
        let expected_batch: usize =
            list.iter().map(|t| t.dim(0).unwrap()).sum();
        let expected_tokens: usize =
            list.iter().map(|t| t.dim(1).unwrap()).max().unwrap();
        let expected_dim = list[0].dim(2).unwrap();

        let originals: Vec<Vec<Vec<Vec<f32>>>> =
            list.iter().map(|t| t.to_vec3::<f32>().unwrap()).collect();

        let out = concatenate_embedding_batches(list).unwrap();
        assert_eq!(out.dim(0).unwrap(), expected_batch);
        assert_eq!(out.dim(1).unwrap(), expected_tokens);
        assert_eq!(out.dim(2).unwrap(), expected_dim);

        let out_v: Vec<Vec<Vec<f32>>> = out.to_vec3::<f32>().unwrap();
        let mut row = 0usize;
        for orig_batch in originals {
            let tokens_here = orig_batch[0].len();
            for orig_row in orig_batch {
                let out_row = &out_v[row];
                // Unpadded region must match the input verbatim.
                for (t, ot) in orig_row.iter().enumerate() {
                    assert_eq!(&out_row[t], ot);
                }
                // Padded region beyond the batch's own tokens is zero.
                for (t, pad_row) in out_row.iter().enumerate().skip(tokens_here)
                {
                    for v in pad_row {
                        assert_eq!(
                            *v, 0.0,
                            "pad region at (row={row}, t={t}) not zero",
                        );
                    }
                }
                row += 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // C — similarity / raw_similarity
    // -----------------------------------------------------------------------

    fn naive_raw_similarity(q: &Tensor, d: &Tensor) -> Vec<Vec<Vec<Vec<f32>>>> {
        let qv: Vec<Vec<Vec<f32>>> = q.to_vec3::<f32>().unwrap();
        let dv: Vec<Vec<Vec<f32>>> = d.to_vec3::<f32>().unwrap();
        qv.iter()
            .map(|query| {
                dv.iter()
                    .map(|doc| {
                        query
                            .iter()
                            .map(|qt| {
                                doc.iter()
                                    .map(|dt| {
                                        qt.iter()
                                            .zip(dt.iter())
                                            .map(|(a, b)| a * b)
                                            .sum::<f32>()
                                    })
                                    .collect::<Vec<f32>>()
                            })
                            .collect::<Vec<Vec<f32>>>()
                    })
                    .collect::<Vec<Vec<Vec<f32>>>>()
            })
            .collect()
    }

    fn naive_max_sim(q: &Tensor, d: &Tensor) -> Vec<Vec<f32>> {
        naive_raw_similarity(q, d)
            .iter()
            .map(|query| {
                query
                    .iter()
                    .map(|doc| {
                        doc.iter()
                            .map(|per_qtok| {
                                per_qtok
                                    .iter()
                                    .copied()
                                    .fold(f32::NEG_INFINITY, f32::max)
                            })
                            .sum::<f32>()
                    })
                    .collect::<Vec<f32>>()
            })
            .collect()
    }

    fn approx_eq_matrix(a: &[Vec<f32>], b: &[Vec<f32>], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (ra, rb) in a.iter().zip(b.iter()) {
            assert_eq!(ra.len(), rb.len());
            for (x, y) in ra.iter().zip(rb.iter()) {
                assert!(
                    (x - y).abs() < tol,
                    "matrix drift: {x} vs {y} (tol={tol})",
                );
            }
        }
    }

    /// C1: `compute_similarities` agrees with a hand-rolled MaxSim reference.
    #[hegel::test(test_cases = 200)]
    fn similarity_matches_naive_maxsim(tc: TestCase) {
        let dev = test_device();
        let (q, d) = tc.draw(query_doc_pair(dev));
        let got = compute_similarities(&q, &d).unwrap();
        let want = naive_max_sim(&q, &d);
        approx_eq_matrix(&got.data, &want, 1e-4);
    }

    /// C2: `compute_raw_similarity` equals the pointwise `Q · Dᵀ` reference.
    /// Candle only exposes `to_vec0`…`to_vec3`, so we reshape the 4-D output
    /// `(nq, nd, qt, dt)` down to 3-D `(nq*nd, qt, dt)` and walk the flat
    /// reference in the same order.
    #[hegel::test(test_cases = 150)]
    fn raw_similarity_matches_naive(tc: TestCase) {
        let dev = test_device();
        let (q, d) = tc.draw(query_doc_pair(dev));
        let raw = compute_raw_similarity(&q, &d).unwrap();
        let (nq, nd, qt, dt) = raw.dims4().unwrap();
        let flat = raw.reshape((nq * nd, qt, dt)).unwrap();
        let got: Vec<Vec<Vec<f32>>> = flat.to_vec3::<f32>().unwrap();
        let want = naive_raw_similarity(&q, &d);

        let mut idx = 0usize;
        for query_block in &want {
            for doc_block in query_block {
                let got_slab = &got[idx];
                idx += 1;
                assert_eq!(got_slab.len(), doc_block.len());
                for (g_row, w_row) in got_slab.iter().zip(doc_block.iter()) {
                    assert_eq!(g_row.len(), w_row.len());
                    for (x, y) in g_row.iter().zip(w_row.iter()) {
                        assert!(
                            (x - y).abs() < 1e-4,
                            "raw sim drift: {x} vs {y}",
                        );
                    }
                }
            }
        }
        assert_eq!(idx, nq * nd);
    }

    /// C3: output shape is `(n_queries, n_documents)` — the plumbing must not
    /// drop or duplicate rows.
    #[hegel::test(test_cases = 100)]
    fn similarity_shape_contract(tc: TestCase) {
        let dev = test_device();
        let (q, d) = tc.draw(query_doc_pair(dev));
        let nq = q.dim(0).unwrap();
        let nd = d.dim(0).unwrap();
        let out = compute_similarities(&q, &d).unwrap();
        assert_eq!(out.data.len(), nq);
        for row in &out.data {
            assert_eq!(row.len(), nd);
        }
    }

    /// C4: appending a zero-valued token row to every document cannot reduce
    /// the similarity — `max_k` now includes `0.0` as an option, so the per-
    /// query-token max is non-decreasing and the sum follows.
    #[hegel::test(test_cases = 150)]
    fn zero_doc_token_is_non_decreasing(tc: TestCase) {
        let dev = test_device();
        let (q, d) = tc.draw(query_doc_pair(dev.clone()));
        let (db, dt, dd) = d.dims3().unwrap();
        let zeros = Tensor::zeros((db, 1, dd), d.dtype(), &dev).unwrap();
        let d_padded = Tensor::cat(&[&d, &zeros], 1).unwrap();
        assert_eq!(d_padded.dim(1).unwrap(), dt + 1);

        let before = compute_similarities(&q, &d).unwrap();
        let after = compute_similarities(&q, &d_padded).unwrap();
        for (rb, ra) in before.data.iter().zip(after.data.iter()) {
            for (vb, va) in rb.iter().zip(ra.iter()) {
                assert!(
                    *va + 1e-4 >= *vb,
                    "zero-doc-token decreased similarity: {vb} → {va}",
                );
            }
        }
    }

    /// C5: scaling queries uniformly by `k > 0` scales the similarity matrix
    /// by `k`. MaxSim is `Σ_t max_k dot(k·Q[i,t], D[j,k]) = k · Σ_t max_k
    /// dot(Q[i,t], D[j,k])` because `k > 0` preserves the argmax of each
    /// per-q-token inner dot-product row.
    #[hegel::test(test_cases = 150)]
    fn similarity_linear_in_positive_query_scale(tc: TestCase) {
        let dev = test_device();
        let (q, d) = tc.draw(query_doc_pair(dev));
        let k: f32 = tc.draw(
            gs::floats::<f32>()
                .min_value(0.25)
                .max_value(4.0)
                .allow_nan(false)
                .allow_infinity(false),
        );
        let q_scaled = q.affine(f64::from(k), 0.0).unwrap();

        let base = compute_similarities(&q, &d).unwrap();
        let scaled = compute_similarities(&q_scaled, &d).unwrap();
        for (rb, rs) in base.data.iter().zip(scaled.data.iter()) {
            for (vb, vs) in rb.iter().zip(rs.iter()) {
                assert!(
                    (*vs - vb * k).abs() < 1e-3,
                    "scale-linearity drift: k·{vb}={} vs {vs} (k={k})",
                    vb * k,
                );
            }
        }
    }
}
