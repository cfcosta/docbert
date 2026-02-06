use candle_core::{Device, Tensor};
use pylate_rs::{ColBERT, Similarities};

use crate::error::Result;

pub const DEFAULT_MODEL_ID: &str = "lightonai/GTE-ModernColBERT-v1";
pub const MODEL_ENV_VAR: &str = "DOCBERT_MODEL";

/// Select the best available compute device.
///
/// Uses CUDA when compiled with the `cuda` feature, Metal when compiled with
/// the `metal` feature, and falls back to CPU otherwise.
fn default_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            return device;
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            return device;
        }
    }

    Device::Cpu
}

/// Manages the ColBERT model lifecycle, supporting lazy loading on first use.
pub struct ModelManager {
    model: Option<ColBERT>,
    model_id: String,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelManager {
    /// Creates a new `ModelManager`. The model ID is resolved from:
    /// 1. The `DOCBERT_MODEL` environment variable, if set
    /// 2. Otherwise, the default model (`lightonai/GTE-ModernColBERT-v1`)
    ///
    /// The model is not loaded until the first call to `encode_documents`,
    /// `encode_query`, or `similarity`.
    pub fn new() -> Self {
        let model_id = std::env::var(MODEL_ENV_VAR)
            .unwrap_or_else(|_| DEFAULT_MODEL_ID.to_string());

        Self {
            model: None,
            model_id,
        }
    }

    /// Creates a `ModelManager` with an explicit model ID, bypassing
    /// environment variable resolution.
    pub fn with_model_id(model_id: String) -> Self {
        Self {
            model: None,
            model_id,
        }
    }

    /// Returns the model ID that will be (or has been) loaded.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns `true` if the model has already been loaded into memory.
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Ensures the model is loaded, downloading from HuggingFace Hub if needed.
    fn ensure_loaded(&mut self) -> Result<&mut ColBERT> {
        if self.model.is_none() {
            let device = default_device();
            let colbert: ColBERT = ColBERT::from(&self.model_id)
                .with_device(device)
                .try_into()?;
            self.model = Some(colbert);
        }

        Ok(self.model.as_mut().unwrap())
    }

    /// Encodes document texts into ColBERT token-level embeddings.
    pub fn encode_documents(&mut self, texts: &[String]) -> Result<Tensor> {
        let model = self.ensure_loaded()?;
        Ok(model.encode(texts, false)?)
    }

    /// Encodes a query string into ColBERT token-level embeddings.
    ///
    /// Returns a 2D tensor of shape `[Q, D]` where Q is the number of query
    /// tokens and D is the embedding dimension.
    pub fn encode_query(&mut self, query: &str) -> Result<Tensor> {
        let model = self.ensure_loaded()?;
        let embeddings = model.encode(&[query.to_string()], true)?;
        // Squeeze the batch dimension: [1, Q, D] -> [Q, D]
        Ok(embeddings.squeeze(0)?)
    }

    /// Computes MaxSim similarity scores between query and document embeddings.
    pub fn similarity(
        &self,
        query_embeddings: &Tensor,
        document_embeddings: &Tensor,
    ) -> Result<Similarities> {
        let model = self.model.as_ref().ok_or_else(|| {
            crate::error::Error::Config("model not loaded".to_string())
        })?;
        Ok(model.similarity(query_embeddings, document_embeddings)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_model_id() {
        let manager = ModelManager::with_model_id("custom/model".to_string());
        assert_eq!(manager.model_id(), "custom/model");
        assert!(!manager.is_loaded());
    }

    #[test]
    fn with_model_id_not_loaded_by_default() {
        let manager = ModelManager::with_model_id(DEFAULT_MODEL_ID.to_string());
        assert!(!manager.is_loaded());
        assert_eq!(manager.model_id(), DEFAULT_MODEL_ID);
    }
}
