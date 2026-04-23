use candle_core::{Device, Tensor};
use docbert_pylate::{ColBERT, Similarities};

use crate::error::{Error, Result};

/// The default ColBERT model loaded when no override is provided.
pub const DEFAULT_MODEL_ID: &str = "lightonai/ColBERT-Zero";

/// Environment variable checked for a model ID override (`DOCBERT_MODEL`).
pub const MODEL_ENV_VAR: &str = "DOCBERT_MODEL";

/// Default document length, in tokens, when encoding documents.
///
/// ColBERT-Zero was trained at 519 tokens, so docbert uses that unless you
/// override it explicitly.
pub const DEFAULT_DOCUMENT_LENGTH: usize = 519;

/// Environment variable checked for a docbert-pylate internal batch size override.
pub const EMBEDDING_BATCH_SIZE_ENV_VAR: &str = "DOCBERT_EMBEDDING_BATCH_SIZE";

/// Default docbert-pylate internal batch size for CPU execution.
pub const DEFAULT_CPU_EMBEDDING_BATCH_SIZE: usize = 32;

/// Default docbert-pylate internal batch size for accelerated execution.
///
/// This number is **not** a document count; pylate's GPU document path
/// treats it as a token budget ceiling via
/// `batch_size * document_length` — at `document_length = 519` and
/// `batch_size = 64` the ceiling is ~33 k tokens per forward pass.
///
/// Tuning target is a 3060-class GPU with ~8 GB of _usable_ VRAM (12 GB
/// card minus the desktop compositor, CUDA runtime, and model weights).
/// The `encode_batch_size` criterion bench in `docbert-pylate` sweeps
/// 32 / 64 / 96 / 128 on `lightonai/LateOn` and finds end-to-end
/// throughput flat at ~187 docs/s across that range — the GPU is
/// compute-bound before we hit 64, so larger batches don't buy us
/// anything but eat more VRAM. Users on fatter cards can raise the
/// ceiling through `DOCBERT_EMBEDDING_BATCH_SIZE`.
pub const DEFAULT_ACCELERATED_EMBEDDING_BATCH_SIZE: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComputeDeviceKind {
    Cpu,
    // `Cuda` is only constructed inside `#[cfg(feature = "cuda")]`
    // branches in `default_device` / `probe_cuda_backend`, so without
    // the feature the variant is read-only (matched in `as_str`) and
    // the dead-code lint flags it.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Cuda,
    // Same story for `Metal` under the `metal` feature.
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    Metal,
}

impl ComputeDeviceKind {
    fn as_str(self) -> &'static str {
        match self {
            ComputeDeviceKind::Cpu => "cpu",
            ComputeDeviceKind::Cuda => "cuda",
            ComputeDeviceKind::Metal => "metal",
        }
    }
}

#[derive(Debug, Clone)]
struct SelectedDevice {
    device: Device,
    kind: ComputeDeviceKind,
    fallback_note: Option<String>,
}

/// Availability information for a single accelerator backend.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct BackendDoctorReport {
    pub compiled: bool,
    pub usable: bool,
    pub error: Option<String>,
}

/// Runtime accelerator diagnostics for the current build and host.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct DoctorReport {
    pub selected_device: String,
    pub fallback_note: Option<String>,
    pub cuda: BackendDoctorReport,
    pub metal: BackendDoctorReport,
}

/// Runtime information for the loaded embedding model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRuntimeConfig {
    pub device: String,
    pub embedding_batch_size: usize,
    pub document_length: usize,
    pub fallback_note: Option<String>,
}

fn cpu_fallback_note(failed_backends: &[String]) -> Option<String> {
    if failed_backends.is_empty() {
        None
    } else {
        Some(format!(
            "falling back to cpu after backend probe failures: {}",
            failed_backends.join("; ")
        ))
    }
}

fn probe_cuda_backend() -> BackendDoctorReport {
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(_) => BackendDoctorReport {
                compiled: true,
                usable: true,
                error: None,
            },
            Err(err) => BackendDoctorReport {
                compiled: true,
                usable: false,
                error: Some(format!("cuda unavailable: {err}")),
            },
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        BackendDoctorReport {
            compiled: false,
            usable: false,
            error: None,
        }
    }
}

fn probe_metal_backend() -> BackendDoctorReport {
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(_) => BackendDoctorReport {
                compiled: true,
                usable: true,
                error: None,
            },
            Err(err) => BackendDoctorReport {
                compiled: true,
                usable: false,
                error: Some(format!("metal unavailable: {err}")),
            },
        }
    }

    #[cfg(not(feature = "metal"))]
    {
        BackendDoctorReport {
            compiled: false,
            usable: false,
            error: None,
        }
    }
}

fn summarize_doctor_report(
    cuda: BackendDoctorReport,
    metal: BackendDoctorReport,
) -> DoctorReport {
    let selected_device = if cuda.usable {
        ComputeDeviceKind::Cuda
    } else if metal.usable {
        ComputeDeviceKind::Metal
    } else {
        ComputeDeviceKind::Cpu
    };

    let mut failed_backends = Vec::new();
    if cuda.compiled
        && !cuda.usable
        && let Some(err) = cuda.error.clone()
    {
        failed_backends.push(err);
    }
    if metal.compiled
        && !metal.usable
        && let Some(err) = metal.error.clone()
    {
        failed_backends.push(err);
    }

    let fallback_note = if selected_device == ComputeDeviceKind::Cpu {
        cpu_fallback_note(&failed_backends)
    } else {
        None
    };

    DoctorReport {
        selected_device: selected_device.as_str().to_string(),
        fallback_note,
        cuda,
        metal,
    }
}

pub fn doctor_report() -> DoctorReport {
    summarize_doctor_report(probe_cuda_backend(), probe_metal_backend())
}

/// Choose the best compute device available at runtime.
///
/// CUDA wins if it is compiled in and usable, then Metal, then CPU.
fn default_device() -> SelectedDevice {
    #[allow(unused_mut)]
    let mut failed_backends = Vec::new();

    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(device) => {
                return SelectedDevice {
                    device,
                    kind: ComputeDeviceKind::Cuda,
                    fallback_note: None,
                };
            }
            Err(err) => {
                failed_backends.push(format!("cuda unavailable: {err}"))
            }
        }
    }

    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                return SelectedDevice {
                    device,
                    kind: ComputeDeviceKind::Metal,
                    fallback_note: None,
                };
            }
            Err(err) => {
                failed_backends.push(format!("metal unavailable: {err}"))
            }
        }
    }

    SelectedDevice {
        device: Device::Cpu,
        kind: ComputeDeviceKind::Cpu,
        fallback_note: cpu_fallback_note(&failed_backends),
    }
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

/// Lazy loader and cache for the ColBERT model.
///
/// The model is downloaded from HuggingFace Hub on first use, then loaded from
/// the local cache after that.
///
/// # Examples
///
/// ```
/// use docbert_core::ModelManager;
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
    runtime_config: Option<ModelRuntimeConfig>,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelManager {
    fn unloaded(model_id: String) -> Self {
        Self {
            model: None,
            model_id,
            document_length: DEFAULT_DOCUMENT_LENGTH,
            embedding_batch_size: None,
            runtime_config: None,
        }
    }

    /// Create a new `ModelManager`.
    ///
    /// The model ID comes from `DOCBERT_MODEL` if that variable is set.
    /// Otherwise docbert uses `lightonai/ColBERT-Zero`.
    ///
    /// The model itself is not loaded until you call `encode_documents`,
    /// `encode_query`, or `similarity`.
    ///
    /// Document encoding defaults to 519 tokens unless you override it with
    /// [`with_document_length`](Self::with_document_length).
    pub fn new() -> Self {
        let model_id = std::env::var(MODEL_ENV_VAR)
            .unwrap_or_else(|_| DEFAULT_MODEL_ID.to_string());
        Self::unloaded(model_id)
    }

    /// Creates a `ModelManager` with an explicit model ID, bypassing
    /// environment variable resolution.
    pub fn with_model_id(model_id: String) -> Self {
        Self::unloaded(model_id)
    }

    /// Sets the document length for encoding.
    ///
    /// This overrides the model's default document length from its config file.
    /// Must be called before the model is loaded (before first encode call).
    pub fn with_document_length(mut self, length: usize) -> Self {
        self.document_length = length;
        self
    }

    /// Sets docbert-pylate' internal batch size for encoding.
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

    /// Load the model if needed and return runtime details for diagnostics.
    pub fn runtime_config(&mut self) -> Result<ModelRuntimeConfig> {
        self.ensure_loaded_and_config().map(|(_, cfg)| cfg.clone())
    }

    /// Ensures the model is loaded, downloading from HuggingFace Hub if needed.
    ///
    /// Prompts from `config_sentence_transformers.json` (e.g. `"search_query: "`
    /// for ColBERT-Zero) are resolved and applied inside pylate itself, so
    /// callers of `encode_*` don't need to prepend anything.
    fn ensure_loaded(&mut self) -> Result<&mut ColBERT> {
        Ok(self.ensure_loaded_and_config()?.0)
    }

    /// Core load-and-return helper shared by `ensure_loaded` and
    /// `runtime_config`. On first call it constructs the `ColBERT`
    /// model plus its `ModelRuntimeConfig` and installs them via
    /// `Option::insert`, which hands back a `&mut T` directly — no
    /// trailing `unwrap` on the Option.
    fn ensure_loaded_and_config(
        &mut self,
    ) -> Result<(&mut ColBERT, &ModelRuntimeConfig)> {
        if self.model.is_none() {
            let selected_device = default_device();
            let env_batch_size =
                std::env::var(EMBEDDING_BATCH_SIZE_ENV_VAR).ok();
            let embedding_batch_size = resolve_embedding_batch_size(
                self.embedding_batch_size,
                env_batch_size.as_deref(),
                selected_device.kind,
            )?;
            let colbert: ColBERT = ColBERT::from(&self.model_id)
                .with_device(selected_device.device)
                .with_document_length(self.document_length)
                .with_batch_size(embedding_batch_size)
                .try_into()?;
            self.embedding_batch_size = Some(embedding_batch_size);
            self.runtime_config = Some(ModelRuntimeConfig {
                device: selected_device.kind.as_str().to_string(),
                embedding_batch_size,
                document_length: self.document_length,
                fallback_note: selected_device.fallback_note,
            });
            self.model = Some(colbert);
        }

        // Both Options are Some at this point. `as_mut()` / `as_ref()`
        // combined with the match pattern turn the "can't happen"
        // branch into dead code the compiler optimises out, without a
        // user-visible panic site.
        match (self.model.as_mut(), self.runtime_config.as_ref()) {
            (Some(model), Some(cfg)) => Ok((model, cfg)),
            _ => unreachable!(
                "model and runtime_config are set immediately above",
            ),
        }
    }

    /// Encodes document texts into ColBERT token-level embeddings.
    ///
    /// Pylate prepends the model's document prompt (e.g. `"search_document: "`)
    /// internally when one is configured in `config_sentence_transformers.json`.
    ///
    /// Returns a 3D tensor of shape `[batch_size, num_tokens, dimension]`.
    /// Downloads the model on first call if not already cached.
    pub fn encode_documents(&mut self, texts: &[String]) -> Result<Tensor> {
        let model = self.ensure_loaded()?;
        Ok(model.encode(texts, false)?)
    }

    /// Encode documents and return per-doc valid-token counts alongside.
    ///
    /// The returned tensor is identical to what [`encode_documents`] would
    /// produce; the extra `Vec<u32>` contains the real token count for each
    /// input in input order, matching the first N rows of axis 1 of the
    /// tensor for each doc. Callers slicing per-doc embeddings use these
    /// counts to avoid an all-zero row scan on the padded tail.
    pub fn encode_documents_with_lengths(
        &mut self,
        texts: &[String],
    ) -> Result<(Tensor, Vec<u32>)> {
        let model = self.ensure_loaded()?;
        Ok(model.encode_documents_with_lengths(texts)?)
    }

    /// Encodes a query string into ColBERT token-level embeddings.
    ///
    /// Pylate prepends the model's query prompt (e.g. `"search_query: "`)
    /// internally when one is configured in `config_sentence_transformers.json`.
    ///
    /// Returns a 2D tensor of shape `[Q, D]` where Q is the number of query
    /// tokens and D is the embedding dimension.
    ///
    /// When the `DOCBERT_WEB_TEST_FAKE_EMBEDDINGS` environment variable is
    /// set, returns a deterministic 1×2 tensor and skips the model load
    /// entirely. This matches the fake document embeddings produced by the
    /// web upload path under the same env var, so integration tests can
    /// exercise the full search pipeline without a real ColBERT model.
    pub fn encode_query(&mut self, query: &str) -> Result<Tensor> {
        if std::env::var_os("DOCBERT_WEB_TEST_FAKE_EMBEDDINGS").is_some() {
            return Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &Device::Cpu)
                .map_err(|e| crate::Error::Config(e.to_string()));
        }
        let model = self.ensure_loaded()?;
        let embeddings = model.encode(&[query.to_string()], true)?;
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
    ///
    /// When the `DOCBERT_WEB_TEST_FAKE_EMBEDDINGS` environment variable is
    /// set, returns a deterministic score of `1.0` per (query, doc) pair so
    /// integration tests can run through the reranker without loading the
    /// real model.
    pub fn similarity(
        &self,
        query_embeddings: &Tensor,
        document_embeddings: &Tensor,
    ) -> Result<Similarities> {
        if std::env::var_os("DOCBERT_WEB_TEST_FAKE_EMBEDDINGS").is_some() {
            let query_batch =
                query_embeddings.dims().first().copied().unwrap_or(1);
            let doc_batch =
                document_embeddings.dims().first().copied().unwrap_or(1);
            return Ok(Similarities {
                data: vec![vec![1.0; doc_batch]; query_batch],
            });
        }
        let model = self.model.as_ref().ok_or_else(|| {
            crate::error::Error::Config("model not loaded".to_string())
        })?;
        Ok(model.similarity(query_embeddings, document_embeddings)?)
    }
}

/// Where the chosen model ID came from.
///
/// [`resolve_model`] checks sources in this order: CLI, env, config, then default.
///
/// # Examples
///
/// ```
/// use docbert_core::model_manager::ModelSource;
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

/// Final result of model resolution.
///
/// This keeps the chosen model ID, the source that won, and the raw values from
/// each source for debugging and status output.
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

/// Choose the model ID using this priority order: CLI flag, environment
/// variable, `config.db`, then the built-in default.
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// use docbert_core::ConfigDb;
/// use docbert_core::model_manager::{resolve_model, ModelSource};
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
        assert_eq!(DEFAULT_DOCUMENT_LENGTH, 519);
        assert_eq!(manager.document_length, DEFAULT_DOCUMENT_LENGTH);
    }

    #[test]
    fn compute_device_kind_labels_are_stable() {
        assert_eq!(ComputeDeviceKind::Cpu.as_str(), "cpu");
        assert_eq!(ComputeDeviceKind::Cuda.as_str(), "cuda");
        assert_eq!(ComputeDeviceKind::Metal.as_str(), "metal");
    }

    #[test]
    fn cpu_fallback_note_reports_failed_backends() {
        assert_eq!(cpu_fallback_note(&[]), None);
        assert_eq!(
            cpu_fallback_note(&[
                "cuda unavailable: boom".to_string(),
                "metal unavailable: nope".to_string(),
            ]),
            Some(
                "falling back to cpu after backend probe failures: cuda unavailable: boom; metal unavailable: nope"
                    .to_string()
            )
        );
    }

    #[test]
    fn summarize_doctor_report_prefers_usable_accelerators() {
        let report = summarize_doctor_report(
            BackendDoctorReport {
                compiled: true,
                usable: true,
                error: None,
            },
            BackendDoctorReport {
                compiled: true,
                usable: false,
                error: Some("metal unavailable: nope".to_string()),
            },
        );

        assert_eq!(report.selected_device, "cuda");
        assert!(report.fallback_note.is_none());
    }

    #[test]
    fn summarize_doctor_report_collects_cpu_fallback_errors() {
        let report = summarize_doctor_report(
            BackendDoctorReport {
                compiled: true,
                usable: false,
                error: Some("cuda unavailable: boom".to_string()),
            },
            BackendDoctorReport {
                compiled: true,
                usable: false,
                error: Some("metal unavailable: nope".to_string()),
            },
        );

        assert_eq!(report.selected_device, "cpu");
        assert_eq!(
            report.fallback_note,
            Some(
                "falling back to cpu after backend probe failures: cuda unavailable: boom; metal unavailable: nope"
                    .to_string()
            )
        );
    }

    #[test]
    fn doctor_report_reflects_compiled_features() {
        let report = doctor_report();

        #[cfg(feature = "cuda")]
        assert!(report.cuda.compiled);
        #[cfg(not(feature = "cuda"))]
        assert!(!report.cuda.compiled);

        #[cfg(feature = "metal")]
        assert!(report.metal.compiled);
        #[cfg(not(feature = "metal"))]
        assert!(!report.metal.compiled);
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
}
