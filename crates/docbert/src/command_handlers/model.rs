use std::path::Path;

use docbert_core::{
    ConfigDb,
    DataDir,
    ModelManager,
    error,
    model_manager::{MODEL_ENV_VAR, ModelResolution},
};

use super::json_output::{model_show_json_string, status_json_string};

pub(super) const EMBEDDING_MODEL_KEY: &str = "embedding_model";

pub(super) fn log_model_runtime(model: &mut ModelManager) -> error::Result<()> {
    let runtime = model.runtime_config()?;
    eprintln!(
        "Embedding runtime: device={}, document_length={}, pylate_batch_size={}",
        runtime.device, runtime.document_length, runtime.embedding_batch_size
    );
    if let Some(note) = runtime.fallback_note {
        eprintln!("Warning: {note}");
    }
    Ok(())
}

pub(crate) fn cmd_doctor(json: bool) -> error::Result<()> {
    let report = docbert_core::model_manager::doctor_report();

    if json {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|e| {
                error::Error::Config(format!(
                    "failed to serialize doctor report: {e}"
                ))
            })?
        );
        return Ok(());
    }

    println!("Selected device: {}", report.selected_device);
    println!(
        "CUDA support: {}",
        if report.cuda.compiled {
            "compiled in"
        } else {
            "not compiled in"
        }
    );
    if report.cuda.compiled {
        println!(
            "CUDA usable: {}",
            if report.cuda.usable { "yes" } else { "no" }
        );
        if let Some(err) = report.cuda.error {
            println!("CUDA error: {err}");
        }
    }
    println!(
        "Metal support: {}",
        if report.metal.compiled {
            "compiled in"
        } else {
            "not compiled in"
        }
    );
    if report.metal.compiled {
        println!(
            "Metal usable: {}",
            if report.metal.usable { "yes" } else { "no" }
        );
        if let Some(err) = report.metal.error {
            println!("Metal error: {err}");
        }
    }
    if let Some(note) = report.fallback_note {
        println!("Note: {note}");
    }

    Ok(())
}

pub(crate) fn cmd_status(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    model_resolution: &ModelResolution,
    json: bool,
) -> error::Result<()> {
    let collections = config_db.list_collections()?;
    let doc_count = config_db.list_document_ids()?.len();
    let model_name = &model_resolution.model_id;
    let embedding_model = config_db.get_setting(EMBEDDING_MODEL_KEY)?;

    if json {
        println!(
            "{}",
            status_json_string(
                data_dir,
                model_resolution,
                embedding_model.as_deref(),
                collections.len(),
                doc_count,
            )?
        );
    } else {
        println!("Data directory: {}", data_dir.root().display());
        println!("Model: {model_name}");
        println!("Model source: {}", model_resolution.source.as_str());
        if let Some(ref emb) = embedding_model {
            if emb != model_name {
                println!(
                    "Embedding model: {emb} (MISMATCH -- run `docbert rebuild`)"
                );
            } else {
                println!("Embedding model: {emb}");
            }
        } else {
            println!("Embedding model: (not set)");
        }
        println!("Collections: {}", collections.len());
        for (name, path) in &collections {
            println!("  {name}: {path}");
        }
        println!("Documents: {doc_count}");
    }
    Ok(())
}

pub(crate) fn cmd_model_show(
    model_resolution: &ModelResolution,
    json: bool,
) -> error::Result<()> {
    if json {
        println!("{}", model_show_json_string(model_resolution)?);
    } else {
        println!("Resolved model: {}", model_resolution.model_id);
        println!("Source: {}", model_resolution.source.as_str());
        if let Some(cli) = model_resolution.cli_model.as_deref() {
            println!("CLI override: {cli}");
        }
        if let Some(env) = model_resolution.env_model.as_deref() {
            println!("{MODEL_ENV_VAR}: {env}");
        }
        if let Some(cfg) = model_resolution.config_model.as_deref() {
            println!("Config setting: {cfg}");
        } else {
            println!("Config setting: (unset)");
        }
    }
    Ok(())
}

pub(crate) fn cmd_model_set(
    config_db: &ConfigDb,
    model: &str,
) -> error::Result<()> {
    config_db.set_setting("model_name", model)?;

    let model_path = Path::new(model);
    if model_path.is_dir() {
        let st_config = model_path.join("config_sentence_transformers.json");
        if !st_config.exists() {
            eprintln!(
                "Warning: {} is missing config_sentence_transformers.json; docbert-pylate may not load this model.",
                model_path.display()
            );
        }
    }

    println!("Stored model_name: {model}");
    Ok(())
}

pub(crate) fn cmd_model_clear(config_db: &ConfigDb) -> error::Result<()> {
    if config_db.remove_setting("model_name")? {
        println!("Cleared model_name setting.");
    } else {
        println!("model_name setting was already unset.");
    }
    Ok(())
}
