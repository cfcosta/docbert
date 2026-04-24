use std::{
    convert::TryFrom,
    fs,
    path::{Path, PathBuf},
};

use candle_core::Device;
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde::Deserialize;

use crate::{error::ColbertError, model::ColBERT};

/// SentenceTransformers `pylate.models.Dense.Dense` module type marker.
const PYLATE_DENSE_TYPE: &str = "pylate.models.Dense.Dense";

/// Raw bytes for one Dense projection layer in the SentenceTransformers
/// pipeline, in the order they appear in `modules.json`.
pub struct DenseModuleData {
    /// Contents of `<path>/config.json` (in_features, out_features,
    /// activation_function, bias, optional use_residual).
    pub config_bytes: Vec<u8>,
    /// Contents of `<path>/model.safetensors` (always contains
    /// `linear.weight`; also contains `residual.weight` when the module's
    /// `use_residual` is true).
    pub weights_bytes: Vec<u8>,
}

#[derive(Deserialize)]
struct ModuleEntry {
    path: String,
    #[serde(rename = "type")]
    module_type: String,
}

/// A builder for configuring and creating a `ColBERT` model from the Hugging Face Hub.
///
/// This struct provides an interface to set various configuration options
/// before downloading the model files and initializing the `ColBERT` instance.
pub struct ColbertBuilder {
    repo_id: String,
    query_prefix: Option<String>,
    document_prefix: Option<String>,
    query_prompt: Option<String>,
    document_prompt: Option<String>,
    mask_token: Option<String>,
    do_query_expansion: Option<bool>,
    attend_to_expansion_tokens: Option<bool>,
    query_length: Option<usize>,
    document_length: Option<usize>,
    batch_size: Option<usize>,
    device: Option<Device>,
}

impl ColbertBuilder {
    /// Creates a new `ColbertBuilder`.
    pub(crate) fn new(repo_id: &str) -> Self {
        Self {
            repo_id: repo_id.to_string(),
            query_prefix: None,
            document_prefix: None,
            query_prompt: None,
            document_prompt: None,
            mask_token: None,
            do_query_expansion: None,
            attend_to_expansion_tokens: None,
            query_length: None,
            document_length: None,
            batch_size: None,
            device: None,
        }
    }

    /// Sets the query prefix token. Overrides the value from the config file.
    pub fn with_query_prefix(mut self, query_prefix: String) -> Self {
        self.query_prefix = Some(query_prefix);
        self
    }

    /// Sets the document prefix token. Overrides the value from the config file.
    pub fn with_document_prefix(mut self, document_prefix: String) -> Self {
        self.document_prefix = Some(document_prefix);
        self
    }

    /// Sets the SentenceTransformers-style query prompt (e.g. `"search_query: "`).
    /// Overrides the `prompts.query` field from `config_sentence_transformers.json`.
    pub fn with_query_prompt(mut self, query_prompt: String) -> Self {
        self.query_prompt = Some(query_prompt);
        self
    }

    /// Sets the SentenceTransformers-style document prompt (e.g. `"search_document: "`).
    /// Overrides the `prompts.document` field from `config_sentence_transformers.json`.
    pub fn with_document_prompt(mut self, document_prompt: String) -> Self {
        self.document_prompt = Some(document_prompt);
        self
    }

    /// Sets the mask token. Overrides the value from the `special_tokens_map.json` file.
    pub fn with_mask_token(mut self, mask_token: String) -> Self {
        self.mask_token = Some(mask_token);
        self
    }

    /// Sets whether to perform query expansion. Overrides the value from the config file.
    pub fn with_do_query_expansion(mut self, do_expansion: bool) -> Self {
        self.do_query_expansion = Some(do_expansion);
        self
    }

    /// Sets whether to attend to expansion tokens. Overrides the value from the config file.
    pub fn with_attend_to_expansion_tokens(mut self, attend: bool) -> Self {
        self.attend_to_expansion_tokens = Some(attend);
        self
    }

    /// Sets the maximum query length. Overrides the value from the config file.
    pub fn with_query_length(mut self, query_length: usize) -> Self {
        self.query_length = Some(query_length);
        self
    }

    /// Sets the maximum document length. Overrides the value from the config file.
    pub fn with_document_length(mut self, document_length: usize) -> Self {
        self.document_length = Some(document_length);
        self
    }

    /// Sets the batch size for encoding. Defaults to 32.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Sets the device to run the model on.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }
}

/// Parses a `modules.json` payload and returns the relative paths of every
/// `pylate.models.Dense.Dense` module, in declaration order.
///
/// Errors when the file isn't a JSON array of objects with `path` and `type`
/// fields, or when no Dense modules are listed.
pub(crate) fn discover_dense_module_paths(
    modules_json: &[u8],
) -> Result<Vec<String>, ColbertError> {
    let entries: Vec<ModuleEntry> = serde_json::from_slice(modules_json)?;
    let dense_paths: Vec<String> = entries
        .into_iter()
        .filter(|e| e.module_type == PYLATE_DENSE_TYPE)
        .map(|e| e.path)
        .collect();
    if dense_paths.is_empty() {
        return Err(ColbertError::Operation(
            "modules.json declares no pylate.models.Dense.Dense modules".into(),
        ));
    }
    Ok(dense_paths)
}

/// Bag of bytes the builder hands to [`ColBERT::new`].
struct LoadedAssets {
    tokenizer: Vec<u8>,
    weights: Vec<u8>,
    config: Vec<u8>,
    st_config: Vec<u8>,
    special_tokens_map: Vec<u8>,
    dense_modules: Vec<DenseModuleData>,
}

impl TryFrom<ColbertBuilder> for ColBERT {
    type Error = ColbertError;

    /// Builds the `ColBERT` model by downloading files from the hub and initializing the model.
    fn try_from(builder: ColbertBuilder) -> Result<Self, Self::Error> {
        let device = builder.device.unwrap_or(Device::Cpu);

        let local_path = PathBuf::from(&builder.repo_id);
        let assets = if local_path.is_dir() {
            load_local_assets(&local_path)?
        } else {
            load_hub_assets(&builder.repo_id)?
        };

        let st_config: serde_json::Value =
            serde_json::from_slice(&assets.st_config)?;
        let special_tokens_map: serde_json::Value =
            serde_json::from_slice(&assets.special_tokens_map)?;

        let final_query_prefix = builder.query_prefix.unwrap_or_else(|| {
            st_config["query_prefix"]
                .as_str()
                .unwrap_or("[Q]")
                .to_string()
        });
        let final_document_prefix =
            builder.document_prefix.unwrap_or_else(|| {
                st_config["document_prefix"]
                    .as_str()
                    .unwrap_or("[D]")
                    .to_string()
            });

        let final_query_prompt = builder.query_prompt.unwrap_or_else(|| {
            st_config["prompts"]["query"]
                .as_str()
                .unwrap_or("")
                .to_string()
        });
        let final_document_prompt =
            builder.document_prompt.unwrap_or_else(|| {
                st_config["prompts"]["document"]
                    .as_str()
                    .unwrap_or("")
                    .to_string()
            });

        let mask_token = builder.mask_token.unwrap_or_else(|| {
            special_tokens_map["mask_token"]
                .as_str()
                .unwrap_or("[MASK]")
                .to_string()
        });

        let final_do_query_expansion =
            builder.do_query_expansion.unwrap_or_else(|| {
                st_config["do_query_expansion"].as_bool().unwrap_or(true)
            });

        let final_attend_to_expansion_tokens =
            builder.attend_to_expansion_tokens.unwrap_or_else(|| {
                st_config["attend_to_expansion_tokens"]
                    .as_bool()
                    .unwrap_or(false)
            });
        let final_query_length = builder
            .query_length
            .or_else(|| st_config["query_length"].as_u64().map(|v| v as usize));
        let final_document_length = builder.document_length.or_else(|| {
            st_config["document_length"].as_u64().map(|v| v as usize)
        });

        ColBERT::new(
            assets.weights,
            assets.dense_modules,
            assets.tokenizer,
            assets.config,
            final_query_prefix,
            final_document_prefix,
            final_query_prompt,
            final_document_prompt,
            mask_token,
            final_do_query_expansion,
            final_attend_to_expansion_tokens,
            final_query_length,
            final_document_length,
            builder.batch_size,
            &device,
        )
    }
}

/// Reads every required asset from a local model directory.
fn load_local_assets(local_path: &Path) -> Result<LoadedAssets, ColbertError> {
    let modules_path = local_path.join("modules.json");
    if !modules_path.exists() {
        return Err(ColbertError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "modules.json not found in local model directory: {}",
                modules_path.display()
            ),
        )));
    }
    let modules_bytes = fs::read(&modules_path)?;
    let dense_paths = discover_dense_module_paths(&modules_bytes)?;

    let tokenizer_path = local_path.join("tokenizer.json");
    let weights_path = local_path.join("model.safetensors");
    let config_path = local_path.join("config.json");
    let st_config_path = local_path.join("config_sentence_transformers.json");
    let special_tokens_map_path = local_path.join("special_tokens_map.json");
    for path in [
        &tokenizer_path,
        &weights_path,
        &config_path,
        &st_config_path,
        &special_tokens_map_path,
    ] {
        if !path.exists() {
            return Err(ColbertError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "File not found in local directory: {}",
                    path.display()
                ),
            )));
        }
    }

    let mut dense_modules = Vec::with_capacity(dense_paths.len());
    for rel_path in dense_paths {
        let dense_dir = local_path.join(&rel_path);
        let cfg_path = dense_dir.join("config.json");
        let dense_weights_path = dense_dir.join("model.safetensors");
        for path in [&cfg_path, &dense_weights_path] {
            if !path.exists() {
                return Err(ColbertError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Dense module file not found: {}", path.display()),
                )));
            }
        }
        dense_modules.push(DenseModuleData {
            config_bytes: fs::read(cfg_path)?,
            weights_bytes: fs::read(dense_weights_path)?,
        });
    }

    Ok(LoadedAssets {
        tokenizer: fs::read(tokenizer_path)?,
        weights: fs::read(weights_path)?,
        config: fs::read(config_path)?,
        st_config: fs::read(st_config_path)?,
        special_tokens_map: fs::read(special_tokens_map_path)?,
        dense_modules,
    })
}

/// Downloads every required asset from the Hugging Face Hub.
fn load_hub_assets(repo_id: &str) -> Result<LoadedAssets, ColbertError> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        repo_id.to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    let modules_path = repo.get("modules.json")?;
    let modules_bytes = fs::read(&modules_path)?;
    let dense_paths = discover_dense_module_paths(&modules_bytes)?;

    let mut dense_modules = Vec::with_capacity(dense_paths.len());
    for rel_path in dense_paths {
        let cfg_remote = format!("{rel_path}/config.json");
        let weights_remote = format!("{rel_path}/model.safetensors");
        let cfg_path = repo.get(&cfg_remote)?;
        let weights_path = repo.get(&weights_remote)?;
        dense_modules.push(DenseModuleData {
            config_bytes: fs::read(cfg_path)?,
            weights_bytes: fs::read(weights_path)?,
        });
    }

    Ok(LoadedAssets {
        tokenizer: fs::read(repo.get("tokenizer.json")?)?,
        weights: fs::read(repo.get("model.safetensors")?)?,
        config: fs::read(repo.get("config.json")?)?,
        st_config: fs::read(repo.get("config_sentence_transformers.json")?)?,
        special_tokens_map: fs::read(repo.get("special_tokens_map.json")?)?,
        dense_modules,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_dense_modules_in_declaration_order() {
        let modules_json = br#"[
            {"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},
            {"idx":1,"name":"1","path":"1_Dense","type":"pylate.models.Dense.Dense"},
            {"idx":2,"name":"2","path":"2_Dense","type":"pylate.models.Dense.Dense"},
            {"idx":3,"name":"3","path":"3_Dense","type":"pylate.models.Dense.Dense"}
        ]"#;
        let paths = discover_dense_module_paths(modules_json).unwrap();
        assert_eq!(paths, vec!["1_Dense", "2_Dense", "3_Dense"]);
    }

    #[test]
    fn discovers_single_dense_module_when_only_one_listed() {
        let modules_json = br#"[
            {"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},
            {"idx":1,"name":"1","path":"1_Dense","type":"pylate.models.Dense.Dense"}
        ]"#;
        let paths = discover_dense_module_paths(modules_json).unwrap();
        assert_eq!(paths, vec!["1_Dense"]);
    }

    #[test]
    fn errors_when_modules_json_has_no_dense_modules() {
        let modules_json = br#"[
            {"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"}
        ]"#;
        let err = discover_dense_module_paths(modules_json).unwrap_err();
        assert!(matches!(err, ColbertError::Operation(_)));
    }

    #[test]
    fn ignores_non_dense_module_entries() {
        let modules_json = br#"[
            {"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},
            {"idx":1,"name":"1","path":"1_Dense","type":"pylate.models.Dense.Dense"},
            {"idx":2,"name":"pool","path":"2_Pooling","type":"sentence_transformers.models.Pooling"},
            {"idx":3,"name":"3","path":"3_Dense","type":"pylate.models.Dense.Dense"}
        ]"#;
        let paths = discover_dense_module_paths(modules_json).unwrap();
        assert_eq!(paths, vec!["1_Dense", "3_Dense"]);
    }
}
