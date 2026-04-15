use docbert_core::{ConfigDb, DataDir, EmbeddingDb, SearchIndex, error};

use super::{
    indexing::{
        remove_document_artifacts_for_ids,
        remove_document_embeddings_for_ids,
    },
    json_output::collection_list_json_string,
};

pub(crate) fn collection_add(
    config_db: &ConfigDb,
    path: &std::path::Path,
    name: &str,
) -> error::Result<()> {
    if !path.exists() {
        return Err(error::Error::Config(format!(
            "directory does not exist: {}",
            path.display()
        )));
    }
    if !path.is_dir() {
        return Err(error::Error::Config(format!(
            "path is not a directory: {}",
            path.display()
        )));
    }

    let abs_path = path.canonicalize().map_err(|e| {
        error::Error::Config(format!(
            "cannot resolve path {}: {e}",
            path.display()
        ))
    })?;

    if config_db.get_collection(name)?.is_some() {
        return Err(error::Error::Config(format!(
            "collection '{name}' already exists"
        )));
    }

    config_db.set_collection(name, &abs_path.to_string_lossy())?;

    println!("Added collection '{name}' -> {}", abs_path.display());
    Ok(())
}

pub(crate) fn collection_remove(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    name: &str,
) -> error::Result<()> {
    if config_db.get_collection(name)?.is_none() {
        return Err(error::Error::NotFound {
            kind: "collection",
            name: name.to_string(),
        });
    }

    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let mut writer = search_index.writer(15_000_000)?;
    search_index.delete_collection(&writer, name);
    writer.commit()?;

    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
    let doc_ids: Vec<u64> = config_db
        .list_all_document_metadata_typed()?
        .into_iter()
        .filter_map(|(doc_id, meta)| {
            (meta.collection == name).then_some(doc_id)
        })
        .collect();
    remove_document_embeddings_for_ids(&embedding_db, &doc_ids)?;
    remove_document_artifacts_for_ids(config_db, &doc_ids)?;

    config_db.remove_collection_merkle_snapshot(name)?;
    config_db.remove_collection(name)?;

    println!("Removed collection '{name}'");
    Ok(())
}

pub(crate) fn collection_list(
    config_db: &ConfigDb,
    json: bool,
) -> error::Result<()> {
    let collections = config_db.list_collections()?;

    if json {
        println!("{}", collection_list_json_string(&collections)?);
    } else if collections.is_empty() {
        println!("No collections registered.");
    } else {
        for (name, path) in &collections {
            println!("{name}\t{path}");
        }
    }
    Ok(())
}
