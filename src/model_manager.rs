use candle_core::{Device, Tensor};
use pylate_rs::{ColBERT, Similarities};

use crate::error::Result;

pub const DEFAULT_MODEL_ID: &str = "lightonai/GTE-ModernColBERT-v1";
pub const MODEL_ENV_VAR: &str = "DOCBERT_MODEL";

/// Default document length in tokens for encoding.
///
/// GTE-ModernColBERT was trained on 300 tokens but generalizes well to longer
/// contexts (tested up to 32K). We use 1024 as a balance between chunk count
/// and encoding speed.
pub const DEFAULT_DOCUMENT_LENGTH: usize = 1024;

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
    document_length: usize,
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
            document_length: DEFAULT_DOCUMENT_LENGTH,
        }
    }

    /// Creates a `ModelManager` with an explicit model ID, bypassing
    /// environment variable resolution.
    pub fn with_model_id(model_id: String) -> Self {
        Self {
            model: None,
            model_id,
            document_length: DEFAULT_DOCUMENT_LENGTH,
        }
    }

    /// Sets the document length for encoding.
    ///
    /// This overrides the model's default document length from its config file.
    /// Must be called before the model is loaded (before first encode call).
    pub fn with_document_length(mut self, length: usize) -> Self {
        self.document_length = length;
        self
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
                .with_document_length(self.document_length)
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

/// How the model ID was resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSource {
    /// Set via `--model` CLI flag.
    Cli,
    /// Set via `DOCBERT_MODEL` environment variable.
    Env,
    /// Stored in config.db `model_name` setting.
    Config,
    /// Hardcoded default.
    Default,
}

impl ModelSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelSource::Cli => "cli",
            ModelSource::Env => "env",
            ModelSource::Config => "config",
            ModelSource::Default => "default",
        }
    }
}

/// The result of resolving which model to use.
#[derive(Debug, Clone)]
pub struct ModelResolution {
    pub model_id: String,
    pub source: ModelSource,
    pub env_model: Option<String>,
    pub config_model: Option<String>,
    pub cli_model: Option<String>,
}

/// Resolve the model ID from (in priority order): CLI flag, environment
/// variable, config.db setting, or the compiled-in default.
pub fn resolve_model(
    config_db: &crate::config_db::ConfigDb,
    cli_model: Option<&str>,
) -> crate::error::Result<ModelResolution> {
    let env_model = std::env::var(MODEL_ENV_VAR).ok();
    let config_model = config_db.get_setting("model_name")?;
    let cli_model = cli_model.map(|s| s.to_string());

    let (model_id, source) = if let Some(cli) = cli_model.clone() {
        (cli, ModelSource::Cli)
    } else if let Some(env) = env_model.clone() {
        (env, ModelSource::Env)
    } else if let Some(cfg) = config_model.clone() {
        (cfg, ModelSource::Config)
    } else {
        (DEFAULT_MODEL_ID.to_string(), ModelSource::Default)
    };

    Ok(ModelResolution {
        model_id,
        source,
        env_model,
        config_model,
        cli_model,
    })
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

    #[test]
    fn with_document_length_is_stored() {
        let manager = ModelManager::new().with_document_length(512);
        assert_eq!(manager.document_length, 512);
    }

    #[test]
    fn default_impl_matches_new() {
        let from_default = ModelManager::default();
        let from_new = ModelManager::new();
        assert_eq!(from_default.model_id(), from_new.model_id());
        assert_eq!(from_default.is_loaded(), from_new.is_loaded());
    }

    #[test]
    fn default_document_length() {
        let manager = ModelManager::new();
        assert_eq!(manager.document_length, DEFAULT_DOCUMENT_LENGTH);
    }

    #[test]
    fn resolve_model_cli_overrides_config() {
        let tmp = tempfile::tempdir().unwrap();
        let config_db =
            crate::config_db::ConfigDb::open(&tmp.path().join("config.db"))
                .unwrap();
        config_db.set_setting("model_name", "config/model").unwrap();
        let resolution = resolve_model(&config_db, Some("cli/model")).unwrap();
        assert_eq!(resolution.model_id, "cli/model");
        assert_eq!(resolution.source, ModelSource::Cli);
        assert_eq!(resolution.cli_model.as_deref(), Some("cli/model"));
        assert_eq!(resolution.config_model.as_deref(), Some("config/model"));
    }

    #[test]
    fn resolve_model_config_used_when_no_cli() {
        let tmp = tempfile::tempdir().unwrap();
        let config_db =
            crate::config_db::ConfigDb::open(&tmp.path().join("config.db"))
                .unwrap();
        config_db.set_setting("model_name", "config/model").unwrap();
        let resolution = resolve_model(&config_db, None).unwrap();
        // Config should be used (unless env var is also set, which we
        // can't control in tests without unsafe). Just verify the config
        // model is populated.
        assert_eq!(resolution.config_model.as_deref(), Some("config/model"));
        assert!(resolution.cli_model.is_none());
        // If DOCBERT_MODEL env is not set, source should be Config
        if resolution.env_model.is_none() {
            assert_eq!(resolution.model_id, "config/model");
            assert_eq!(resolution.source, ModelSource::Config);
        }
    }

    #[test]
    fn resolve_model_no_config_no_cli() {
        let tmp = tempfile::tempdir().unwrap();
        let config_db =
            crate::config_db::ConfigDb::open(&tmp.path().join("config.db"))
                .unwrap();
        let resolution = resolve_model(&config_db, None).unwrap();
        assert!(resolution.config_model.is_none());
        assert!(resolution.cli_model.is_none());
        // Without env var, should be default
        if resolution.env_model.is_none() {
            assert_eq!(resolution.model_id, DEFAULT_MODEL_ID);
            assert_eq!(resolution.source, ModelSource::Default);
        }
    }

    #[test]
    fn model_source_as_str() {
        assert_eq!(ModelSource::Cli.as_str(), "cli");
        assert_eq!(ModelSource::Env.as_str(), "env");
        assert_eq!(ModelSource::Config.as_str(), "config");
        assert_eq!(ModelSource::Default.as_str(), "default");
    }
}
