use std::path::Path;

use docbert_core::{DataDir, error, model_manager::ModelResolution};
use serde::Serialize;

fn serialize_json<T: Serialize + ?Sized>(
    value: &T,
    error_context: &str,
) -> error::Result<String> {
    serde_json::to_string(value)
        .map_err(|e| error::Error::Config(format!("{error_context}: {e}")))
}

#[derive(Serialize)]
struct CollectionListItem<'a> {
    name: &'a str,
    path: &'a str,
}

pub(super) fn collection_list_json_string(
    collections: &[(String, String)],
) -> error::Result<String> {
    let items: Vec<_> = collections
        .iter()
        .map(|(name, path)| CollectionListItem { name, path })
        .collect();
    serialize_json(&items, "failed to serialize collection list")
}

#[derive(Serialize)]
struct ContextListItem<'a> {
    uri: &'a str,
    description: &'a str,
}

pub(super) fn context_list_json_string(
    contexts: &[(String, String)],
) -> error::Result<String> {
    let items: Vec<_> = contexts
        .iter()
        .map(|(uri, description)| ContextListItem { uri, description })
        .collect();
    serialize_json(&items, "failed to serialize context list")
}

#[derive(Serialize)]
struct GetJsonOutput<'a> {
    collection: &'a str,
    path: &'a str,
    file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
}

pub(super) fn get_json_string(
    collection: &str,
    path: &str,
    full_path: &Path,
    content: Option<&str>,
) -> error::Result<String> {
    serialize_json(
        &GetJsonOutput {
            collection,
            path,
            file: full_path.display().to_string(),
            content,
        },
        "failed to serialize get response",
    )
}

#[derive(Serialize)]
pub(super) struct MultiGetJsonItem {
    pub(super) collection: String,
    pub(super) path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) content: Option<String>,
}

pub(super) fn multi_get_json_string(
    items: &[MultiGetJsonItem],
) -> error::Result<String> {
    serialize_json(items, "failed to serialize multi-get response")
}

#[derive(Serialize)]
struct StatusJsonOutput<'a> {
    data_dir: String,
    model: &'a str,
    model_source: &'a str,
    embedding_model: Option<&'a str>,
    collections: usize,
    documents: usize,
}

pub(super) fn status_json_string(
    data_dir: &DataDir,
    model_resolution: &ModelResolution,
    embedding_model: Option<&str>,
    collection_count: usize,
    doc_count: usize,
) -> error::Result<String> {
    serialize_json(
        &StatusJsonOutput {
            data_dir: data_dir.root().display().to_string(),
            model: &model_resolution.model_id,
            model_source: model_resolution.source.as_str(),
            embedding_model,
            collections: collection_count,
            documents: doc_count,
        },
        "failed to serialize status response",
    )
}

#[derive(Serialize)]
struct ModelShowJsonOutput<'a> {
    resolved: &'a str,
    source: &'a str,
    cli: Option<&'a str>,
    env: Option<&'a str>,
    config: Option<&'a str>,
}

pub(super) fn model_show_json_string(
    model_resolution: &ModelResolution,
) -> error::Result<String> {
    serialize_json(
        &ModelShowJsonOutput {
            resolved: &model_resolution.model_id,
            source: model_resolution.source.as_str(),
            cli: model_resolution.cli_model.as_deref(),
            env: model_resolution.env_model.as_deref(),
            config: model_resolution.config_model.as_deref(),
        },
        "failed to serialize model resolution",
    )
}
