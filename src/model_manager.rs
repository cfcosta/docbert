use std::path::Path;

use candle_core::{Device, Tensor};
use pylate_rs::{ColBERT, Similarities};

use crate::error::{Error, Result};

/// The default ColBERT model loaded when no override is provided.
pub const DEFAULT_MODEL_ID: &str = "lightonai/ColBERT-Zero";

/// Environment variable checked for a model ID override (`DOCBERT_MODEL`).
pub const MODEL_ENV_VAR: &str = "DOCBERT_MODEL";

/// Default document length in tokens for encoding.
///
/// ColBERT-Zero was trained on 519-token documents, but the underlying
/// ModernBERT backbone generalizes well to longer contexts (up to 8192
/// tokens). We use 2048 as a balance between chunk count and encoding speed.
pub const DEFAULT_DOCUMENT_LENGTH: usize = 2048;

/// Environment variable checked for a pylate-rs internal batch size override.
pub const EMBEDDING_BATCH_SIZE_ENV_VAR: &str = "DOCBERT_EMBEDDING_BATCH_SIZE";

/// Default pylate-rs internal batch size for CPU execution.
pub const DEFAULT_CPU_EMBEDDING_BATCH_SIZE: usize = 32;

/// Default pylate-rs internal batch size for accelerated execution.
pub const DEFAULT_ACCELERATED_EMBEDDING_BATCH_SIZE: usize = 64;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComputeDeviceKind {
    Cpu,
    Cuda,
    Metal,
}

/// Select the best available compute device.
///
/// Uses CUDA when compiled with the `cuda` feature, Metal when compiled with
/// the `metal` feature, and falls back to CPU otherwise.
fn default_device() -> (Device, ComputeDeviceKind) {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            return (device, ComputeDeviceKind::Cuda);
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            return (device, ComputeDeviceKind::Metal);
        }
    }

    (Device::Cpu, ComputeDeviceKind::Cpu)
}

fn default_embedding_batch_size(device_kind: ComputeDeviceKind) -> usize {
    match device_kind {
        ComputeDeviceKind::Cpu => DEFAULT_CPU_EMBEDDING_BATCH_SIZE,
        ComputeDeviceKind::Cuda | ComputeDeviceKind::Metal => {
            DEFAULT_ACCELERATED_EMBEDDING_BATCH_SIZE
        }
    }
}

fn resolve_embedding_batch_size(
    configured_batch_size: Option<usize>,
    env_batch_size: Option<&str>,
    device_kind: ComputeDeviceKind,
) -> Result<usize> {
    if let Some(batch_size) = configured_batch_size {
        if batch_size == 0 {
            return Err(Error::Config(
                "embedding batch size must be greater than zero".to_string(),
            ));
        }
        return Ok(batch_size);
    }

    if let Some(raw_batch_size) = env_batch_size {
        let batch_size = raw_batch_size.parse::<usize>().map_err(|_| {
            Error::Config(format!(
                "{EMBEDDING_BATCH_SIZE_ENV_VAR} must be a positive integer"
            ))
        })?;
        if batch_size == 0 {
            return Err(Error::Config(format!(
                "{EMBEDDING_BATCH_SIZE_ENV_VAR} must be greater than zero"
            )));
        }
        return Ok(batch_size);
    }

    Ok(default_embedding_batch_size(device_kind))
}

/// Manages the ColBERT model lifecycle, supporting lazy loading on first use.
///
/// The model is downloaded from HuggingFace Hub on first use and cached locally.
/// Subsequent uses load from the cache.
///
/// # Examples
///
/// ```
/// use docbert::ModelManager;
///
/// // Create with default model (lightonai/ColBERT-Zero)
/// let manager = ModelManager::new();
/// assert!(!manager.is_loaded());
///
/// // Create with a specific model
/// let manager = ModelManager::with_model_id("custom/model".to_string());
/// assert_eq!(manager.model_id(), "custom/model");
///
/// // Override the document encoding length
/// let manager = ModelManager::new().with_document_length(512);
/// ```
pub struct ModelManager {
    model: Option<ColBERT>,
    model_id: String,
    document_length: usize,
    embedding_batch_size: Option<usize>,
    query_prompt: String,
    document_prompt: String,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelManager {
    /// Creates a new `ModelManager`. The model ID is resolved from:
    /// 1. The `DOCBERT_MODEL` environment variable, if set
    /// 2. Otherwise, the default model (`lightonai/ColBERT-Zero`)
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
            embedding_batch_size: None,
            query_prompt: String::new(),
            document_prompt: String::new(),
        }
    }

    /// Creates a `ModelManager` with an explicit model ID, bypassing
    /// environment variable resolution.
    pub fn with_model_id(model_id: String) -> Self {
        Self {
            model: None,
            model_id,
            document_length: DEFAULT_DOCUMENT_LENGTH,
            embedding_batch_size: None,
            query_prompt: String::new(),
            document_prompt: String::new(),
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

    /// Sets pylate-rs' internal batch size for encoding.
    ///
    /// Useful for increasing GPU utilization when the default batch size is
    /// too conservative for the available memory.
    pub fn with_embedding_batch_size(mut self, batch_size: usize) -> Self {
        self.embedding_batch_size = Some(batch_size);
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
    ///
    /// Also resolves the `prompts` field from `config_sentence_transformers.json`
    /// for prepending to queries and documents (e.g. `"search_query: "` for
    /// ColBERT-Zero). Falls back to empty strings for models without prompts.
    fn ensure_loaded(&mut self) -> Result<&mut ColBERT> {
        if self.model.is_none() {
            // Resolve prompts from model config before loading
            let (query_prompt, document_prompt) =
                resolve_prompts(&self.model_id);
            self.query_prompt = query_prompt;
            self.document_prompt = document_prompt;

            let (device, device_kind) = default_device();
            let env_batch_size =
                std::env::var(EMBEDDING_BATCH_SIZE_ENV_VAR).ok();
            let embedding_batch_size = resolve_embedding_batch_size(
                self.embedding_batch_size,
                env_batch_size.as_deref(),
                device_kind,
            )?;
            let colbert: ColBERT = ColBERT::from(&self.model_id)
                .with_device(device)
                .with_document_length(self.document_length)
                .with_batch_size(embedding_batch_size)
                .try_into()?;
            self.embedding_batch_size = Some(embedding_batch_size);
            self.model = Some(colbert);
        }

        Ok(self.model.as_mut().unwrap())
    }

    /// Encodes document texts into ColBERT token-level embeddings.
    ///
    /// Prepends the model's document prompt (e.g. `"search_document: "`) if
    /// configured in `config_sentence_transformers.json`.
    ///
    /// Returns a 3D tensor of shape `[batch_size, num_tokens, dimension]`.
    /// Downloads the model on first call if not already cached.
    pub fn encode_documents(&mut self, texts: &[String]) -> Result<Tensor> {
        let prompt = self.document_prompt.clone();
        let model = self.ensure_loaded()?;
        if prompt.is_empty() {
            Ok(model.encode(texts, false)?)
        } else {
            let prompted: Vec<String> =
                texts.iter().map(|t| format!("{prompt}{t}")).collect();
            Ok(model.encode(&prompted, false)?)
        }
    }

    /// Encodes a query string into ColBERT token-level embeddings.
    ///
    /// Prepends the model's query prompt (e.g. `"search_query: "`) if
    /// configured in `config_sentence_transformers.json`.
    ///
    /// Returns a 2D tensor of shape `[Q, D]` where Q is the number of query
    /// tokens and D is the embedding dimension.
    pub fn encode_query(&mut self, query: &str) -> Result<Tensor> {
        let prompt = self.query_prompt.clone();
        let model = self.ensure_loaded()?;
        let text = if prompt.is_empty() {
            query.to_string()
        } else {
            format!("{prompt}{query}")
        };
        let embeddings = model.encode(&[text], true)?;
        // Squeeze the batch dimension: [1, Q, D] -> [Q, D]
        Ok(embeddings.squeeze(0)?)
    }

    /// Computes MaxSim similarity scores between query and document embeddings.
    ///
    /// Both tensors must be 3D: `[batch, tokens, dimension]`. Returns a
    /// `Similarities` struct containing a `data: Vec<Vec<f32>>` matrix of scores.
    ///
    /// The model must already be loaded (via a prior `encode_*` call),
    /// otherwise returns a `Config` error.
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

/// Resolve the `prompts` field from a model's `config_sentence_transformers.json`.
///
/// Returns `(query_prompt, document_prompt)`. Falls back to empty strings if the
/// config is missing or doesn't contain a `prompts` field (backwards compatible
/// with models like GTE-ModernColBERT that don't use prompts).
fn resolve_prompts(model_id: &str) -> (String, String) {
    let st_config_path = if Path::new(model_id).is_dir() {
        // Local model directory
        Path::new(model_id).join("config_sentence_transformers.json")
    } else {
        // HuggingFace Hub model -- resolve via hf-hub (uses cache)
        let api = match hf_hub::api::sync::Api::new() {
            Ok(api) => api,
            Err(_) => return (String::new(), String::new()),
        };
        let repo = api.repo(hf_hub::Repo::with_revision(
            model_id.to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));
        match repo.get("config_sentence_transformers.json") {
            Ok(path) => path,
            Err(_) => return (String::new(), String::new()),
        }
    };

    let bytes = match std::fs::read(&st_config_path) {
        Ok(b) => b,
        Err(_) => return (String::new(), String::new()),
    };

    let config: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(c) => c,
        Err(_) => return (String::new(), String::new()),
    };

    let query_prompt = config["prompts"]["query"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let document_prompt = config["prompts"]["document"]
        .as_str()
        .unwrap_or("")
        .to_string();

    (query_prompt, document_prompt)
}

/// How the model ID was resolved, in priority order.
///
/// [`resolve_model`] tries each source in order: CLI > Env > Config > Default.
///
/// # Examples
///
/// ```
/// use docbert::model_manager::ModelSource;
///
/// assert_eq!(ModelSource::Cli.as_str(), "cli");
/// assert_eq!(ModelSource::Default.as_str(), "default");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSource {
    /// Set via `--model` CLI flag (highest priority).
    Cli,
    /// Set via `DOCBERT_MODEL` environment variable.
    Env,
    /// Stored in `config.db` as the `model_name` setting.
    Config,
    /// Hardcoded default (`lightonai/ColBERT-Zero`).
    Default,
}

impl ModelSource {
    /// Returns the source as a short string label (`"cli"`, `"env"`, `"config"`, `"default"`).
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
///
/// Contains the final model ID, which source it came from, and the
/// raw values from each source for diagnostic display.
#[derive(Debug, Clone)]
pub struct ModelResolution {
    /// The resolved model ID that will be used.
    pub model_id: String,
    /// Which source provided the model ID.
    pub source: ModelSource,
    /// Value from the `DOCBERT_MODEL` environment variable, if set.
    pub env_model: Option<String>,
    /// Value from the `model_name` setting in `config.db`, if set.
    pub config_model: Option<String>,
    /// Value from the `--model` CLI flag, if provided.
    pub cli_model: Option<String>,
}

/// Resolve the model ID from (in priority order): CLI flag, environment
/// variable, config.db setting, or the compiled-in default.
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// use docbert::ConfigDb;
/// use docbert::model_manager::{resolve_model, ModelSource};
///
/// let db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
///
/// // With a CLI override
/// let res = resolve_model(&db, Some("my/model")).unwrap();
/// assert_eq!(res.model_id, "my/model");
/// assert_eq!(res.source, ModelSource::Cli);
///
/// // Without CLI, falls back to env/config/default
/// let res = resolve_model(&db, None).unwrap();
/// assert!(res.cli_model.is_none());
/// ```
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
    fn with_embedding_batch_size_is_stored() {
        let manager = ModelManager::new().with_embedding_batch_size(96);
        assert_eq!(manager.embedding_batch_size, Some(96));
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
    fn default_embedding_batch_size_uses_device_defaults() {
        assert_eq!(
            default_embedding_batch_size(ComputeDeviceKind::Cpu),
            DEFAULT_CPU_EMBEDDING_BATCH_SIZE
        );
        assert_eq!(
            default_embedding_batch_size(ComputeDeviceKind::Cuda),
            DEFAULT_ACCELERATED_EMBEDDING_BATCH_SIZE
        );
        assert_eq!(
            default_embedding_batch_size(ComputeDeviceKind::Metal),
            DEFAULT_ACCELERATED_EMBEDDING_BATCH_SIZE
        );
    }

    #[test]
    fn resolve_embedding_batch_size_prefers_explicit_value() {
        let resolved = resolve_embedding_batch_size(
            Some(96),
            Some("128"),
            ComputeDeviceKind::Cuda,
        )
        .unwrap();
        assert_eq!(resolved, 96);
    }

    #[test]
    fn resolve_embedding_batch_size_uses_env_override() {
        let resolved = resolve_embedding_batch_size(
            None,
            Some("128"),
            ComputeDeviceKind::Cpu,
        )
        .unwrap();
        assert_eq!(resolved, 128);
    }

    #[test]
    fn resolve_embedding_batch_size_rejects_invalid_env() {
        let err = resolve_embedding_batch_size(
            None,
            Some("nope"),
            ComputeDeviceKind::Cpu,
        )
        .unwrap_err();
        assert!(err.to_string().contains(
            "DOCBERT_EMBEDDING_BATCH_SIZE must be a positive integer"
        ));
    }

    #[test]
    fn resolve_embedding_batch_size_rejects_zero() {
        let err =
            resolve_embedding_batch_size(Some(0), None, ComputeDeviceKind::Cpu)
                .unwrap_err();
        assert!(
            err.to_string()
                .contains("embedding batch size must be greater than zero")
        );
    }

    #[test]
    fn resolve_embedding_batch_size_falls_back_to_device_default() {
        let resolved =
            resolve_embedding_batch_size(None, None, ComputeDeviceKind::Cuda)
                .unwrap();
        assert_eq!(resolved, DEFAULT_ACCELERATED_EMBEDDING_BATCH_SIZE);
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

    #[test]
    fn prompts_default_to_empty() {
        let manager = ModelManager::new();
        assert!(manager.query_prompt.is_empty());
        assert!(manager.document_prompt.is_empty());
    }

    #[test]
    fn resolve_prompts_from_local_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "prompts": {
                "query": "search_query: ",
                "document": "search_document: "
            },
            "query_prefix": "[Q] ",
            "document_prefix": "[D] "
        });
        std::fs::write(
            tmp.path().join("config_sentence_transformers.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let (qp, dp) = resolve_prompts(tmp.path().to_str().unwrap());
        assert_eq!(qp, "search_query: ");
        assert_eq!(dp, "search_document: ");
    }

    #[test]
    fn resolve_prompts_missing_prompts_field() {
        let tmp = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "query_prefix": "[Q] ",
            "document_prefix": "[D] "
        });
        std::fs::write(
            tmp.path().join("config_sentence_transformers.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let (qp, dp) = resolve_prompts(tmp.path().to_str().unwrap());
        assert!(qp.is_empty());
        assert!(dp.is_empty());
    }

    #[test]
    fn resolve_prompts_missing_config_file() {
        let tmp = tempfile::tempdir().unwrap();
        let (qp, dp) = resolve_prompts(tmp.path().to_str().unwrap());
        assert!(qp.is_empty());
        assert!(dp.is_empty());
    }
}
