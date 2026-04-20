use docbert_core::{
    ConfigDb,
    DataDir,
    ModelManager,
    SearchIndex,
    error,
    model_manager::ModelResolution,
    search,
};

use super::{
    json_output::{MultiGetJsonItem, get_json_string, multi_get_json_string},
    model::log_model_runtime,
};
use crate::cli;

pub(crate) fn run(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    model_resolution: &ModelResolution,
    args: &cli::SearchArgs,
) -> error::Result<()> {
    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let mut model =
        ModelManager::with_model_id(model_resolution.model_id.clone());
    if !args.bm25_only {
        log_model_runtime(&mut model)?;
    }

    let mut results = if args.bm25_only || args.no_fuzzy || args.all {
        let params = search::SearchParams {
            query: args.query.clone(),
            count: args.count,
            collection: args.collection.clone(),
            min_score: args.min_score,
            bm25_only: args.bm25_only,
            no_fuzzy: args.no_fuzzy,
            all: args.all,
        };

        search::execute_search(
            &params,
            &search_index,
            config_db,
            data_dir,
            &mut model,
        )?
    } else {
        let request = search::SearchQuery {
            query: args.query.clone(),
            collection: args.collection.clone(),
            count: args.count,
            min_score: args.min_score,
        };
        search::execute_search_mode(
            search::SearchMode::Hybrid,
            &request,
            &search_index,
            config_db,
            data_dir,
            &mut model,
        )?
    };

    search::disambiguate_doc_ids(&mut results, config_db);

    if args.json {
        search::format_json(&results, &args.query);
    } else if args.files {
        search::format_files(&results, config_db);
    } else {
        search::format_human(&results);
    }
    Ok(())
}

pub(crate) fn semantic(
    config_db: &ConfigDb,
    data_dir: &DataDir,
    model_resolution: &ModelResolution,
    args: &cli::SemanticSearchArgs,
) -> error::Result<()> {
    let mut model =
        ModelManager::with_model_id(model_resolution.model_id.clone());
    log_model_runtime(&mut model)?;

    let params = search::SemanticSearchParams {
        query: args.query.clone(),
        collection: None,
        count: args.count,
        min_score: args.min_score,
        all: args.all,
    };

    let mut results = search::execute_semantic_search(
        &params, config_db, data_dir, &mut model,
    )?;

    search::disambiguate_doc_ids(&mut results, config_db);

    if args.json {
        search::format_json(&results, &args.query);
    } else if args.files {
        search::format_files(&results, config_db);
    } else {
        search::format_human(&results);
    }
    Ok(())
}

pub(crate) fn get(
    config_db: &ConfigDb,
    args: &cli::GetArgs,
) -> error::Result<()> {
    let reference = &args.reference;

    let (collection, path) = search::resolve_reference(config_db, reference)
        .ok_or_else(|| error::Error::NotFound {
            kind: "document",
            name: reference.to_string(),
        })?;

    let collection_path =
        config_db.get_collection(&collection)?.ok_or_else(|| {
            error::Error::NotFound {
                kind: "collection",
                name: collection.clone(),
            }
        })?;

    let full_path = docbert_core::path_safety::resolve_document_path(
        std::path::Path::new(&collection_path),
        &path,
    )?;

    if args.meta {
        println!("collection: {collection}");
        println!("path: {path}");
        println!("file: {}", full_path.display());
    } else if args.json {
        let content = docbert_core::preparation::load_preview_content(
            std::path::Path::new(&path),
            &full_path,
        )?;
        println!(
            "{}",
            get_json_string(&collection, &path, &full_path, Some(&content))?
        );
    } else {
        let content = docbert_core::preparation::load_preview_content(
            std::path::Path::new(&path),
            &full_path,
        )?;
        print!("{content}");
    }

    Ok(())
}

pub(crate) fn multi_get(
    config_db: &ConfigDb,
    args: &cli::MultiGetArgs,
) -> error::Result<()> {
    let glob = globset::Glob::new(&args.pattern)
        .map_err(|e| {
            error::Error::Config(format!("invalid glob pattern: {e}"))
        })?
        .compile_matcher();

    let mut matches: Vec<(String, String)> = Vec::new();

    for (_doc_id, meta) in config_db.list_all_document_metadata_typed()? {
        if let Some(ref coll) = args.collection
            && meta.collection != *coll
        {
            continue;
        }

        if glob.is_match(&meta.relative_path) {
            matches.push((meta.collection, meta.relative_path));
        }
    }

    matches.sort();

    if args.json {
        let mut items = Vec::with_capacity(matches.len());
        for (collection, path) in &matches {
            let collection_path = config_db.get_collection(collection)?;
            let (file, content) = if let Some(ref cp) = collection_path {
                let full_path = std::path::Path::new(cp).join(path);
                let content = if args.full {
                    docbert_core::preparation::load_preview_content(
                        std::path::Path::new(path),
                        &full_path,
                    )
                    .ok()
                } else {
                    None
                };
                (Some(full_path.to_string_lossy().to_string()), content)
            } else {
                (None, None)
            };

            items.push(MultiGetJsonItem {
                collection: collection.clone(),
                path: path.clone(),
                file,
                content,
            });
        }

        println!("{}", multi_get_json_string(&items)?);
    } else if args.files {
        for (collection, path) in &matches {
            if let Ok(Some(collection_path)) =
                config_db.get_collection(collection)
            {
                let full_path =
                    std::path::Path::new(&collection_path).join(path);
                println!("{}", full_path.display());
            }
        }
    } else if args.full {
        for (collection, path) in &matches {
            if let Ok(Some(collection_path)) =
                config_db.get_collection(collection)
            {
                let full_path =
                    std::path::Path::new(&collection_path).join(path);
                println!("--- {collection}:{path} ---");
                if let Ok(content) =
                    docbert_core::preparation::load_preview_content(
                        std::path::Path::new(path),
                        &full_path,
                    )
                {
                    print!("{content}");
                    if !content.ends_with('\n') {
                        println!();
                    }
                }
            }
        }
    } else if matches.is_empty() {
        println!("No documents match '{}'", args.pattern);
    } else {
        for (collection, path) in &matches {
            println!("{collection}:{path}");
        }
        println!("\n{} match(es)", matches.len());
    }

    Ok(())
}
