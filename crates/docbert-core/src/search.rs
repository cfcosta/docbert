use std::{collections::HashMap, path::Path};

use crate::{
    config_db::{CollectionLocation, ConfigDb},
    doc_id::{format_document_ref, strip_document_ref_prefix},
    embedding_db::EmbeddingDb,
    error::Result,
    incremental::DocumentMetadata,
    ingestion,
    model_manager::ModelManager,
    reranker::{self, RankedDocument},
    tantivy_index::{SearchIndex, SearchResult},
    text_util,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Semantic,
    Hybrid,
}

impl SearchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            SearchMode::Semantic => "semantic",
            SearchMode::Hybrid => "hybrid",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "semantic" => Some(SearchMode::Semantic),
            "hybrid" => Some(SearchMode::Hybrid),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub query: String,
    pub collection: Option<String>,
    pub count: usize,
    pub min_score: f32,
}

/// Options for hybrid search: BM25 first, ColBERT reranking second.
///
/// # Examples
///
/// ```
/// use docbert_core::search::SearchParams;
///
/// let params = SearchParams {
///     query: "rust programming".to_string(),
///     count: 10,
///     collection: Some("docs".to_string()),
///     min_score: 0.5,
///     bm25_only: false,
///     no_fuzzy: false,
///     all: false,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// The search query.
    pub query: String,
    /// Number of results to return.
    pub count: usize,
    /// Search only within this collection.
    pub collection: Option<String>,
    /// Minimum score threshold.
    pub min_score: f32,
    /// Skip ColBERT reranking, return BM25 results directly.
    pub bm25_only: bool,
    /// Disable fuzzy matching in the first stage.
    pub no_fuzzy: bool,
    /// Return all results above the score threshold.
    pub all: bool,
}

/// Options for semantic-only search.
///
/// # Examples
///
/// ```
/// use docbert_core::search::SemanticSearchParams;
///
/// let params = SemanticSearchParams {
///     query: "machine learning concepts".to_string(),
///     collection: None,
///     count: 5,
///     min_score: 0.0,
///     all: false,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct SemanticSearchParams {
    /// The search query.
    pub query: String,
    /// Optional collection filter. When `None`, searches all collections.
    pub collection: Option<String>,
    /// Number of results to return.
    pub count: usize,
    /// Minimum score threshold.
    pub min_score: f32,
    /// Return all results above the score threshold.
    pub all: bool,
}

/// Search result returned by [`execute_search`] or [`execute_semantic_search`].
///
/// Results are sorted by score, highest first, and ranks are assigned after
/// filtering and limiting.
#[derive(Debug, Clone)]
pub struct FinalResult {
    /// 1-indexed position in the result list.
    pub rank: usize,
    /// Relevance score (BM25 or ColBERT MaxSim, depending on search mode).
    pub score: f32,
    /// Short hex document identifier (e.g., `"a1b2c3"`).
    pub doc_id: String,
    /// Numeric document identifier (key in embedding/metadata databases).
    pub doc_num_id: u64,
    /// Collection this document belongs to.
    pub collection: String,
    /// Relative file path within the collection.
    pub path: String,
    /// Document title extracted from content or filename.
    pub title: String,
}

/// Run the normal search pipeline.
///
/// The steps are:
/// 1. **BM25 retrieval** - Tantivy finds the top 1000 candidates, with optional fuzzy matching.
/// 2. **ColBERT reranking** - candidates are rescored with MaxSim unless `bm25_only` is set.
/// 3. **Score filtering** - results below `min_score` are dropped.
/// 4. **Limit** - at most `count` results are returned, unless `all` is set.
///
/// # Examples
///
/// ```no_run
/// use docbert_core::{SearchIndex, EmbeddingDb, ModelManager};
/// use docbert_core::search::{execute_search, SearchParams};
///
/// # let tmp = tempfile::tempdir().unwrap();
/// let index = SearchIndex::open_in_ram().unwrap();
/// let emb_db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
/// let mut model = ModelManager::new();
///
/// let params = SearchParams {
///     query: "rust programming".to_string(),
///     count: 10,
///     collection: None,
///     min_score: 0.0,
///     bm25_only: true,
///     no_fuzzy: false,
///     all: false,
/// };
///
/// let results = execute_search(&params, &index, &emb_db, &mut model).unwrap();
/// for r in &results {
///     println!("{}: {} (score {:.3})", r.rank, r.title, r.score);
/// }
/// ```
pub fn execute_search(
    args: &SearchParams,
    search_index: &SearchIndex,
    embedding_db: &EmbeddingDb,
    model: &mut ModelManager,
) -> Result<Vec<FinalResult>> {
    let bm25_limit = 1000;

    // Stage 1: BM25 retrieval (with optional fuzzy matching)
    let bm25_results = if args.no_fuzzy {
        // Pure BM25 without fuzzy
        if let Some(ref collection) = args.collection {
            search_index.search_in_collection(
                &args.query,
                collection,
                bm25_limit,
            )?
        } else {
            search_index.search(&args.query, bm25_limit)?
        }
    } else {
        // BM25 + fuzzy matching
        search_index.search_fuzzy(
            &args.query,
            args.collection.as_deref(),
            bm25_limit,
        )?
    };

    if bm25_results.is_empty() {
        return Ok(vec![]);
    }

    // Stage 2: ColBERT reranking (unless --bm25-only)
    let results = if args.bm25_only {
        bm25_to_final(&bm25_results)
    } else {
        rerank_results(&bm25_results, embedding_db, model, &args.query)?
    };

    // Stage 3: Filter by min_score
    let filtered: Vec<FinalResult> = results
        .into_iter()
        .filter(|r| r.score >= args.min_score)
        .collect();

    // Stage 4: Limit results
    let limit = if args.all { filtered.len() } else { args.count };
    let limited: Vec<FinalResult> = filtered
        .into_iter()
        .take(limit)
        .enumerate()
        .map(|(i, mut r)| {
            r.rank = i + 1;
            r
        })
        .collect();

    Ok(limited)
}

fn semantic_ranked_from_query_embedding(
    query_embedding: &candle_core::Tensor,
    doc_ids: &[u64],
    embedding_db: &EmbeddingDb,
    model: &ModelManager,
) -> Result<Vec<RankedDocument>> {
    reranker::rerank(query_embedding, doc_ids, embedding_db, model)
}

fn semantic_final_results_from_ranked(
    metadata: &HashMap<u64, DocumentMetadata>,
    ranked: Vec<RankedDocument>,
    min_score: f32,
    count: usize,
    all: bool,
) -> Vec<FinalResult> {
    let mut results: Vec<FinalResult> = ranked
        .into_iter()
        .filter(|ranked| ranked.score >= min_score)
        .filter_map(|RankedDocument { doc_num_id, score }| {
            let meta = metadata.get(&doc_num_id)?;
            Some(FinalResult {
                rank: 0,
                score,
                doc_id: short_doc_id(doc_num_id),
                doc_num_id,
                collection: meta.collection.clone(),
                path: meta.relative_path.clone(),
                title: String::new(),
            })
        })
        .collect();

    let limit = if all { results.len() } else { count };
    results.truncate(limit);

    for (i, result) in results.iter_mut().enumerate() {
        result.rank = i + 1;
    }

    results
}

/// Run semantic-only search across all indexed documents.
///
/// Unlike [`execute_search`], this skips BM25 and scores every stored embedding
/// with ColBERT MaxSim. That can surface related documents even when they share
/// little wording with the query.
///
/// The ColBERT model is loaded on first use.
pub fn execute_semantic_search(
    args: &SemanticSearchParams,
    config_db: &ConfigDb,
    embedding_db: &EmbeddingDb,
    model: &mut ModelManager,
) -> Result<Vec<FinalResult>> {
    let metadata_entries = config_db.list_all_document_metadata_typed()?;
    if metadata_entries.is_empty() {
        return Ok(vec![]);
    }

    let mut metadata = HashMap::with_capacity(metadata_entries.len());
    let mut doc_ids = Vec::with_capacity(metadata_entries.len());
    let mut collection_paths = HashMap::new();

    for (doc_id, meta) in metadata_entries {
        if args
            .collection
            .as_ref()
            .is_none_or(|c| c == &meta.collection)
            && document_has_semantic_body(
                config_db,
                &mut collection_paths,
                &meta,
            )
        {
            doc_ids.push(doc_id);
            metadata.insert(doc_id, meta);
        }
    }

    if doc_ids.is_empty() {
        return Ok(vec![]);
    }

    let query_embedding = model.encode_query(&args.query)?;
    let ranked = semantic_ranked_from_query_embedding(
        &query_embedding,
        &doc_ids,
        embedding_db,
        model,
    )?;

    let mut results = semantic_final_results_from_ranked(
        &metadata,
        ranked,
        args.min_score,
        args.count,
        args.all,
    );

    populate_titles(&mut results, config_db);

    Ok(results)
}

pub fn execute_search_mode(
    mode: SearchMode,
    request: &SearchQuery,
    search_index: &SearchIndex,
    config_db: &ConfigDb,
    embedding_db: &EmbeddingDb,
    model: &mut ModelManager,
) -> Result<Vec<FinalResult>> {
    match mode {
        SearchMode::Semantic => execute_semantic_search(
            &SemanticSearchParams {
                query: request.query.clone(),
                collection: request.collection.clone(),
                count: request.count,
                min_score: request.min_score,
                all: false,
            },
            config_db,
            embedding_db,
            model,
        ),
        SearchMode::Hybrid => execute_search(
            &SearchParams {
                query: request.query.clone(),
                count: request.count,
                collection: request.collection.clone(),
                min_score: request.min_score,
                bm25_only: false,
                no_fuzzy: false,
                all: false,
            },
            search_index,
            embedding_db,
            model,
        ),
    }
}

/// Look up a document by its short ID, such as `"a1b2c3"` or `"#a1b2c3"`.
///
/// This walks the known metadata and matches on the short ID or a containing
/// display form. On success it returns `(collection, relative_path)`.
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// use docbert_core::{ConfigDb, DocumentId};
/// use docbert_core::incremental::DocumentMetadata;
/// use docbert_core::search::resolve_by_doc_id;
///
/// let db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
/// let id = DocumentId::new("notes", "hello.md");
/// let meta = DocumentMetadata {
///     collection: "notes".to_string(),
///     relative_path: "hello.md".to_string(),
///     mtime: 1000,
/// };
/// db.set_document_metadata_typed(id.numeric, &meta).unwrap();
///
/// let (coll, path) = resolve_by_doc_id(&db, &id.short).unwrap();
/// assert_eq!(coll, "notes");
/// assert_eq!(path, "hello.md");
/// ```
pub fn resolve_by_doc_id(
    config_db: &ConfigDb,
    short_id: &str,
) -> Option<(String, String)> {
    config_db
        .find_document_by_short_id(strip_document_ref_prefix(short_id))
        .ok()
        .flatten()
        .map(|(_doc_id, meta)| (meta.collection, meta.relative_path))
}

/// Look up a document by its relative path across all collections.
///
/// Returns `(collection, relative_path)` or `None` when nothing matches.
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// use docbert_core::{ConfigDb, DocumentId};
/// use docbert_core::incremental::DocumentMetadata;
/// use docbert_core::search::resolve_by_path;
///
/// let db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
/// let id = DocumentId::new("notes", "hello.md");
/// let meta = DocumentMetadata {
///     collection: "notes".to_string(),
///     relative_path: "hello.md".to_string(),
///     mtime: 1000,
/// };
/// db.set_document_metadata_typed(id.numeric, &meta).unwrap();
///
/// let (coll, path) = resolve_by_path(&db, "hello.md").unwrap();
/// assert_eq!(coll, "notes");
/// assert!(resolve_by_path(&db, "nonexistent.md").is_none());
/// ```
pub fn resolve_by_path(
    config_db: &ConfigDb,
    path: &str,
) -> Option<(String, String)> {
    config_db
        .find_document_by_path(path)
        .ok()
        .flatten()
        .map(|(_doc_id, meta)| (meta.collection, meta.relative_path))
}

/// Resolve a user-supplied document reference.
///
/// Supports `#short_id`, `collection:path`, and bare relative paths.
pub fn resolve_reference(
    config_db: &ConfigDb,
    reference: &str,
) -> Option<(String, String)> {
    if reference.starts_with('#') {
        return resolve_by_doc_id(config_db, reference);
    }

    if let Some((collection, path)) = reference.split_once(':') {
        return Some((collection.to_string(), path.to_string()));
    }

    if reference.len() == 6
        && reference.chars().all(|ch| ch.is_ascii_hexdigit())
        && let Some(resolved) = resolve_by_doc_id(config_db, reference)
    {
        return Some(resolved);
    }

    resolve_by_path(config_db, reference)
}

fn bm25_to_final(results: &[SearchResult]) -> Vec<FinalResult> {
    results
        .iter()
        .enumerate()
        .map(|(i, r)| FinalResult {
            rank: i + 1,
            score: r.score,
            doc_id: r.doc_id.clone(),
            doc_num_id: r.doc_num_id,
            collection: r.collection.clone(),
            path: r.path.clone(),
            title: r.title.clone(),
        })
        .collect()
}

/// Turn a numeric document ID into the short display form, like `"#a1b2c3"`.
///
/// # Examples
///
/// ```
/// use docbert_core::search::short_doc_id;
///
/// let id = short_doc_id(0xabcdef1234567890);
/// assert_eq!(id, "#abcdef");
/// assert!(id.starts_with('#'));
/// assert_eq!(id.len(), 7);
/// ```
pub fn short_doc_id(numeric: u64) -> String {
    let full = format!("{numeric:016x}");
    format_document_ref(&full[..6])
}

fn document_has_semantic_body(
    config_db: &ConfigDb,
    collection_locations: &mut std::collections::HashMap<
        String,
        Option<CollectionLocation>,
    >,
    meta: &DocumentMetadata,
) -> bool {
    let collection_location = collection_locations
        .entry(meta.collection.clone())
        .or_insert_with(|| {
            config_db
                .get_collection_location(&meta.collection)
                .ok()
                .flatten()
        });
    let Some(collection_location) = collection_location.as_ref() else {
        return false;
    };

    match collection_location {
        CollectionLocation::Managed => true,
        CollectionLocation::Filesystem(path) => {
            let full_path = Path::new(path).join(&meta.relative_path);
            let Ok(content) = std::fs::read_to_string(full_path) else {
                return false;
            };

            !text_util::strip_yaml_frontmatter(&content)
                .trim()
                .is_empty()
        }
    }
}

fn populate_titles(results: &mut [FinalResult], config_db: &ConfigDb) {
    let mut collection_locations: std::collections::HashMap<
        String,
        Option<CollectionLocation>,
    > = std::collections::HashMap::new();

    for r in results {
        let fallback = ingestion::extract_title("", Path::new(&r.path));
        let collection_location = collection_locations
            .entry(r.collection.clone())
            .or_insert_with(|| {
                config_db
                    .get_collection_location(&r.collection)
                    .ok()
                    .flatten()
            });
        let Some(collection_location) = collection_location.as_ref() else {
            r.title = fallback;
            continue;
        };

        let CollectionLocation::Filesystem(path) = collection_location else {
            r.title = fallback;
            continue;
        };

        let full_path = Path::new(path).join(&r.path);
        let content = std::fs::read_to_string(&full_path).unwrap_or_default();
        let title = if content.is_empty() {
            fallback
        } else {
            ingestion::extract_title(&content, Path::new(&r.path))
        };
        r.title = title;
    }
}

fn rerank_results(
    bm25_results: &[SearchResult],
    embedding_db: &EmbeddingDb,
    model: &mut ModelManager,
    query: &str,
) -> Result<Vec<FinalResult>> {
    let query_embedding = model.encode_query(query)?;

    let candidate_ids: Vec<u64> =
        bm25_results.iter().map(|r| r.doc_num_id).collect();
    let ranked = reranker::rerank(
        &query_embedding,
        &candidate_ids,
        embedding_db,
        model,
    )?;

    // Build a lookup from doc_num_id to BM25 result for metadata.
    let bm25_lookup: std::collections::HashMap<u64, &SearchResult> =
        bm25_results.iter().map(|r| (r.doc_num_id, r)).collect();

    Ok(ranked
        .into_iter()
        .filter_map(|RankedDocument { doc_num_id, score }| {
            bm25_lookup.get(&doc_num_id).map(|bm25| FinalResult {
                rank: 0, // Set later
                score,
                doc_id: bm25.doc_id.clone(),
                doc_num_id,
                collection: bm25.collection.clone(),
                path: bm25.path.clone(),
                title: bm25.title.clone(),
            })
        })
        .collect())
}

/// Print results in the default terminal format.
///
/// Each result is shown as `rank. [score] collection:path #doc_id`, with the
/// title on the next line when one is available. A total count is printed last.
pub fn format_human(results: &[FinalResult]) {
    if results.is_empty() {
        println!("No results found.");
        return;
    }

    for r in results {
        println!(
            "{:>3}. [{:.3}] {}:{} {}",
            r.rank, r.score, r.collection, r.path, r.doc_id
        );
        if !r.title.is_empty() {
            println!("     {}", r.title);
        }
    }
    println!("\n{} result(s)", results.len());
}

fn search_result_json_string(result: &FinalResult) -> String {
    format!(
        "{{\"rank\":{},\"score\":{:.6},\"doc_id\":{},\"collection\":{},\"path\":{},\"title\":{}}}",
        result.rank,
        result.score,
        json_escape(&result.doc_id),
        json_escape(&result.collection),
        json_escape(&result.path),
        json_escape(&result.title),
    )
}

fn format_json_string(results: &[FinalResult], query: &str) -> String {
    let mut output = format!(
        "{{\"query\":{},\"result_count\":{},\"results\":[",
        json_escape(query),
        results.len()
    );

    for (i, result) in results.iter().enumerate() {
        if i > 0 {
            output.push(',');
        }
        output.push_str(&search_result_json_string(result));
    }

    output.push_str("]}");
    output
}

/// Print results as JSON.
///
/// The output object contains `query`, `result_count`, and a `results` array
/// with `rank`, `score`, `doc_id`, `collection`, `path`, and `title`.
pub fn format_json(results: &[FinalResult], query: &str) {
    println!("{}", format_json_string(results, query));
}

/// Print matching files as absolute paths, one per line.
///
/// Relative paths are resolved against each collection root, which makes the
/// output easy to pipe into other tools.
pub fn format_files(
    results: &[FinalResult],
    config_db: &crate::config_db::ConfigDb,
) {
    let mut collection_paths: std::collections::HashMap<
        String,
        Option<String>,
    > = std::collections::HashMap::new();

    for r in results {
        let collection_path = collection_paths
            .entry(r.collection.clone())
            .or_insert_with(|| {
                config_db.get_collection(&r.collection).ok().flatten()
            });

        if let Some(collection_path) = collection_path {
            let full_path =
                std::path::Path::new(&collection_path).join(&r.path);
            println!("{}", full_path.display());
        }
    }
}

/// Escape a string as a JSON string literal, including the surrounding quotes.
///
/// # Examples
///
/// ```
/// use docbert_core::search::json_escape;
///
/// assert_eq!(json_escape("hello"), "\"hello\"");
/// assert_eq!(json_escape("line\nnewline"), "\"line\\nnewline\"");
/// assert_eq!(json_escape("tab\there"), "\"tab\\there\"");
/// assert_eq!(json_escape(r#"say "hi""#), r#""say \"hi\"""#);
/// ```
pub fn json_escape(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 2);
    result.push('"');
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c < '\x20' => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result.push('"');
    result
}

#[cfg(test)]
mod tests {
    use candle_core::Tensor;

    use super::*;
    use crate::{
        config_db::ConfigDb,
        doc_id::DocumentId,
        incremental::DocumentMetadata,
    };

    fn make_semantic_args(query: &str) -> SemanticSearchParams {
        SemanticSearchParams {
            query: query.to_string(),
            collection: None,
            count: 10,
            all: false,
            min_score: 0.0,
        }
    }

    fn make_search_args(query: &str) -> SearchParams {
        SearchParams {
            query: query.to_string(),
            count: 10,
            collection: None,
            all: false,
            min_score: 0.0,
            bm25_only: true,
            no_fuzzy: false,
        }
    }

    #[test]
    fn semantic_body_skips_empty_and_frontmatter_only_docs() {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let collection_dir = tmp.path().join("notes");
        std::fs::create_dir_all(&collection_dir).unwrap();
        config_db
            .set_collection("notes", collection_dir.to_str().unwrap())
            .unwrap();

        std::fs::write(collection_dir.join("empty.md"), "").unwrap();
        std::fs::write(
            collection_dir.join("frontmatter.md"),
            "---\nid: 1\ntags:\n  - diary\n---\n",
        )
        .unwrap();
        std::fs::write(
            collection_dir.join("body.md"),
            "---\ntitle: Hello\n---\nActual body text",
        )
        .unwrap();

        let mut collection_paths = std::collections::HashMap::new();

        let empty = DocumentMetadata {
            collection: "notes".to_string(),
            relative_path: "empty.md".to_string(),
            mtime: 1,
        };
        assert!(!document_has_semantic_body(
            &config_db,
            &mut collection_paths,
            &empty,
        ));

        let frontmatter_only = DocumentMetadata {
            collection: "notes".to_string(),
            relative_path: "frontmatter.md".to_string(),
            mtime: 1,
        };
        assert!(!document_has_semantic_body(
            &config_db,
            &mut collection_paths,
            &frontmatter_only,
        ));

        let with_body = DocumentMetadata {
            collection: "notes".to_string(),
            relative_path: "body.md".to_string(),
            mtime: 1,
        };
        assert!(document_has_semantic_body(
            &config_db,
            &mut collection_paths,
            &with_body,
        ));
    }

    #[test]
    fn document_has_semantic_body_returns_true_for_managed_collection() {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        config_db.set_managed_collection("notes").unwrap();

        let managed = DocumentMetadata {
            collection: "notes".to_string(),
            relative_path: "api-doc.md".to_string(),
            mtime: 1,
        };

        let mut collection_locations = std::collections::HashMap::new();
        assert!(document_has_semantic_body(
            &config_db,
            &mut collection_locations,
            &managed,
        ));
    }

    #[test]
    fn resolve_helpers_work_with_typed_document_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let id = DocumentId::new("notes", "hello.md");
        let meta = DocumentMetadata {
            collection: "notes".to_string(),
            relative_path: "hello.md".to_string(),
            mtime: 1000,
        };

        db.set_document_metadata_typed(id.numeric, &meta).unwrap();

        assert_eq!(
            resolve_by_doc_id(&db, &id.short),
            Some(("notes".to_string(), "hello.md".to_string()))
        );
        assert_eq!(
            resolve_by_path(&db, "hello.md"),
            Some(("notes".to_string(), "hello.md".to_string()))
        );
    }

    #[test]
    fn semantic_ranked_from_query_embedding_uses_shared_reranker_for_chunk_only_families()
     {
        let tmp = tempfile::tempdir().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let model = ModelManager::new();
        let base_doc_id = DocumentId::new("notes", "hello.md").numeric;
        let chunk_only_id = crate::chunking::chunk_doc_id(base_doc_id, 1);

        let query_embedding = Tensor::zeros(
            &[2, 128],
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        embedding_db
            .store(chunk_only_id, 2, 128, &vec![0.0; 256])
            .unwrap();

        let err = semantic_ranked_from_query_embedding(
            &query_embedding,
            &[base_doc_id],
            &embedding_db,
            &model,
        )
        .unwrap_err();
        assert!(err.to_string().contains("model not loaded"));
    }

    #[test]
    fn semantic_final_results_from_ranked_drops_rows_without_base_document_metadata()
     {
        let base_doc_id = DocumentId::new("notes", "hello.md").numeric;
        let chunk_doc_id = crate::chunking::chunk_doc_id(base_doc_id, 1);
        let mut metadata = HashMap::new();
        metadata.insert(
            base_doc_id,
            DocumentMetadata {
                collection: "notes".to_string(),
                relative_path: "hello.md".to_string(),
                mtime: 1,
            },
        );

        let results = semantic_final_results_from_ranked(
            &metadata,
            vec![RankedDocument {
                doc_num_id: chunk_doc_id,
                score: 0.9,
            }],
            0.0,
            10,
            false,
        );

        assert!(results.is_empty());
    }

    #[test]
    fn semantic_final_results_from_ranked_attaches_base_document_metadata() {
        let base_doc_id = DocumentId::new("notes", "hello.md").numeric;
        let mut metadata = HashMap::new();
        metadata.insert(
            base_doc_id,
            DocumentMetadata {
                collection: "notes".to_string(),
                relative_path: "nested/hello.md".to_string(),
                mtime: 1,
            },
        );

        let results = semantic_final_results_from_ranked(
            &metadata,
            vec![RankedDocument {
                doc_num_id: base_doc_id,
                score: 0.8,
            }],
            0.0,
            10,
            false,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].collection, "notes");
        assert_eq!(results[0].path, "nested/hello.md");
        assert_eq!(results[0].title, "");
    }

    #[test]
    fn semantic_final_results_from_ranked_applies_min_score_and_count() {
        let first_doc_id = DocumentId::new("notes", "a.md").numeric;
        let second_doc_id = DocumentId::new("notes", "b.md").numeric;
        let third_doc_id = DocumentId::new("notes", "c.md").numeric;
        let metadata = HashMap::from([
            (
                first_doc_id,
                DocumentMetadata {
                    collection: "notes".to_string(),
                    relative_path: "a.md".to_string(),
                    mtime: 1,
                },
            ),
            (
                second_doc_id,
                DocumentMetadata {
                    collection: "notes".to_string(),
                    relative_path: "b.md".to_string(),
                    mtime: 1,
                },
            ),
            (
                third_doc_id,
                DocumentMetadata {
                    collection: "notes".to_string(),
                    relative_path: "c.md".to_string(),
                    mtime: 1,
                },
            ),
        ]);

        let results = semantic_final_results_from_ranked(
            &metadata,
            vec![
                RankedDocument {
                    doc_num_id: first_doc_id,
                    score: 0.9,
                },
                RankedDocument {
                    doc_num_id: second_doc_id,
                    score: 0.7,
                },
                RankedDocument {
                    doc_num_id: third_doc_id,
                    score: 0.4,
                },
            ],
            0.5,
            1,
            false,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_num_id, first_doc_id);
        assert_eq!(results[0].rank, 1);
        assert_eq!(results[0].score, 0.9);
    }

    /// Set up a search index with sample documents and commit them.
    fn setup_index_with_docs() -> (SearchIndex, EmbeddingDb, tempfile::TempDir)
    {
        let tmp = tempfile::tempdir().unwrap();
        let idx = SearchIndex::open_in_ram().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();

        let mut writer = idx.writer(15_000_000).unwrap();

        let docs = vec![
            (
                "notes",
                "rust-guide.md",
                "The Rust Programming Language",
                "Rust is a systems programming language focused on safety, \
                 concurrency, and performance. It achieves memory safety \
                 without garbage collection.",
            ),
            (
                "notes",
                "python-intro.md",
                "Introduction to Python",
                "Python is a high-level interpreted programming language \
                 known for its readability and simplicity. It supports \
                 multiple programming paradigms.",
            ),
            (
                "docs",
                "cooking-pasta.md",
                "How to Cook Pasta",
                "Boil water in a large pot. Add salt. Cook the pasta \
                 according to package directions. Drain and serve with \
                 your favorite sauce.",
            ),
            (
                "docs",
                "gardening.md",
                "Gardening Tips",
                "Water your plants regularly. Ensure proper sunlight \
                 exposure. Use compost for healthy soil. Prune dead \
                 leaves periodically.",
            ),
            (
                "notes",
                "machine-learning.md",
                "Machine Learning Basics",
                "Machine learning is a subset of artificial intelligence \
                 that enables systems to learn from data. Neural networks \
                 and deep learning are popular approaches.",
            ),
        ];

        for (collection, path, title, body) in &docs {
            let doc_id = DocumentId::new(collection, path);
            idx.add_document(
                &writer,
                &doc_id.short,
                doc_id.numeric,
                collection,
                path,
                title,
                body,
                1000,
            )
            .unwrap();
        }

        writer.commit().unwrap();
        (idx, embedding_db, tmp)
    }

    #[test]
    fn bm25_only_returns_relevant_results() {
        let (idx, emb_db, _tmp) = setup_index_with_docs();
        let mut model = ModelManager::new();
        let args = make_search_args("rust programming");

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(!results.is_empty(), "search should return results");
        // The Rust guide should be the top result
        assert_eq!(
            results[0].path, "rust-guide.md",
            "Rust guide should rank first for 'rust programming'"
        );
    }

    #[test]
    fn bm25_only_results_have_correct_ranks() {
        let (idx, emb_db, _tmp) = setup_index_with_docs();
        let mut model = ModelManager::new();
        let args = make_search_args("programming");

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(results.len() >= 2, "should find multiple programming docs");
        for (i, r) in results.iter().enumerate() {
            assert_eq!(
                r.rank,
                i + 1,
                "ranks should be 1-indexed and sequential"
            );
        }
    }

    #[test]
    fn bm25_only_respects_count_limit() {
        let (idx, emb_db, _tmp) = setup_index_with_docs();
        let mut model = ModelManager::new();
        let mut args = make_search_args("programming");
        args.count = 1;

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert_eq!(results.len(), 1, "should respect count limit");
    }

    #[test]
    fn bm25_only_respects_min_score() {
        let (idx, emb_db, _tmp) = setup_index_with_docs();
        let mut model = ModelManager::new();
        let mut args = make_search_args("programming");
        args.min_score = 999.0; // impossibly high threshold

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(
            results.is_empty(),
            "no results should pass a very high min_score"
        );
    }

    #[test]
    fn bm25_only_collection_filter() {
        let (idx, emb_db, _tmp) = setup_index_with_docs();
        let mut model = ModelManager::new();
        let mut args = make_search_args("programming");
        args.collection = Some("notes".to_string());
        args.no_fuzzy = true;

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(
                r.collection, "notes",
                "all results should be from the 'notes' collection"
            );
        }
    }

    #[test]
    fn bm25_only_no_results_for_unrelated_query() {
        let (idx, emb_db, _tmp) = setup_index_with_docs();
        let mut model = ModelManager::new();
        let args = make_search_args("xyzzy_nonexistent_term_12345");

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(results.is_empty(), "should return no results for gibberish");
    }

    #[test]
    fn bm25_only_all_flag_returns_everything() {
        let (idx, emb_db, _tmp) = setup_index_with_docs();
        let mut model = ModelManager::new();
        let mut args = make_search_args("programming");
        args.all = true;
        args.count = 1; // should be ignored when all=true

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(
            results.len() > 1,
            "all=true should return more than count=1"
        );
    }

    #[test]
    fn bm25_only_scores_are_descending() {
        let (idx, emb_db, _tmp) = setup_index_with_docs();
        let mut model = ModelManager::new();
        let args = make_search_args("programming language");

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        for window in results.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "scores should be in descending order"
            );
        }
    }

    #[test]
    fn bm25_only_fuzzy_matching_finds_typos() {
        let (idx, emb_db, _tmp) = setup_index_with_docs();
        let mut model = ModelManager::new();
        // "programing" (one 'm') should fuzzy-match "programming"
        let args = make_search_args("programing");

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(
            !results.is_empty(),
            "fuzzy search should find results despite typo"
        );
    }

    #[test]
    fn bm25_only_no_fuzzy_flag() {
        let (idx, emb_db, _tmp) = setup_index_with_docs();
        let mut model = ModelManager::new();
        let mut args = make_search_args("rust");
        args.no_fuzzy = true;

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(!results.is_empty(), "exact search should find 'rust'");
        assert_eq!(results[0].path, "rust-guide.md");
    }

    #[test]
    fn bm25_only_result_fields_populated() {
        let (idx, emb_db, _tmp) = setup_index_with_docs();
        let mut model = ModelManager::new();
        let mut args = make_search_args("pasta");
        args.no_fuzzy = true;

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.collection, "docs");
        assert_eq!(r.path, "cooking-pasta.md");
        assert_eq!(r.title, "How to Cook Pasta");
        assert!(r.score > 0.0);
        assert!(!r.doc_id.is_empty());
        assert!(r.doc_num_id > 0);
    }

    /// Helper to set up index + embeddings for end-to-end tests.
    ///
    /// Creates documents, indexes them in tantivy, computes ColBERT
    /// embeddings, and returns everything needed for search.
    fn setup_e2e() -> (SearchIndex, EmbeddingDb, ModelManager, tempfile::TempDir)
    {
        use crate::embedding::embed_and_store;

        let tmp = tempfile::tempdir().unwrap();
        let idx = SearchIndex::open_in_ram().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let mut model = ModelManager::new();

        let mut writer = idx.writer(15_000_000).unwrap();

        let docs: Vec<(&str, &str, &str, &str)> = vec![
            (
                "notes",
                "rust-guide.md",
                "The Rust Programming Language",
                "Rust is a systems programming language focused on safety, \
                 concurrency, and performance. It achieves memory safety \
                 without garbage collection through its ownership system \
                 and borrow checker.",
            ),
            (
                "notes",
                "python-intro.md",
                "Introduction to Python",
                "Python is a high-level interpreted programming language \
                 known for its readability and simplicity. It supports \
                 multiple programming paradigms including object-oriented \
                 and functional programming.",
            ),
            (
                "docs",
                "cooking-pasta.md",
                "How to Cook Pasta",
                "Boil water in a large pot. Add salt generously. Cook the \
                 pasta according to package directions until al dente. \
                 Drain and serve with your favorite sauce.",
            ),
            (
                "docs",
                "gardening.md",
                "Gardening Tips for Beginners",
                "Water your plants regularly in the morning. Ensure proper \
                 sunlight exposure. Use compost for healthy soil. Prune \
                 dead leaves periodically to promote growth.",
            ),
            (
                "notes",
                "machine-learning.md",
                "Machine Learning Basics",
                "Machine learning is a subset of artificial intelligence \
                 that enables systems to learn from data. Neural networks \
                 and deep learning are popular approaches for tasks like \
                 image recognition and natural language processing.",
            ),
        ];

        let mut embed_docs: Vec<(u64, String)> = Vec::new();

        for (collection, path, title, body) in &docs {
            let doc_id = DocumentId::new(collection, path);
            idx.add_document(
                &writer,
                &doc_id.short,
                doc_id.numeric,
                collection,
                path,
                title,
                body,
                1000,
            )
            .unwrap();
            // Include both title and body in the embedding text
            embed_docs.push((doc_id.numeric, format!("{title}\n{body}")));
        }

        writer.commit().unwrap();

        // Compute and store ColBERT embeddings
        let count =
            embed_and_store(&mut model, &embedding_db, embed_docs).unwrap();
        assert_eq!(count, docs.len(), "all docs should be embedded");

        (idx, embedding_db, model, tmp)
    }

    fn setup_semantic_e2e()
    -> (ConfigDb, EmbeddingDb, ModelManager, tempfile::TempDir) {
        use crate::embedding::embed_and_store;

        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let mut model = ModelManager::new();

        let docs: Vec<(&str, &str, &str, &str)> = vec![
            (
                "notes",
                "rust-guide.md",
                "The Rust Programming Language",
                "Rust is a systems programming language focused on safety, \
                 concurrency, and performance. It achieves memory safety \
                 without garbage collection through its ownership system \
                 and borrow checker.",
            ),
            (
                "notes",
                "python-intro.md",
                "Introduction to Python",
                "Python is a high-level interpreted programming language \
                 known for its readability and simplicity. It supports \
                 multiple programming paradigms including object-oriented \
                 and functional programming.",
            ),
            (
                "docs",
                "cooking-pasta.md",
                "How to Cook Pasta",
                "Boil water in a large pot. Add salt generously. Cook the \
                 pasta according to package directions until al dente. \
                 Drain and serve with your favorite sauce.",
            ),
        ];

        let mut embed_docs: Vec<(u64, String)> = Vec::new();

        for (collection, path, title, body) in &docs {
            let doc_id = DocumentId::new(collection, path);
            let meta = DocumentMetadata {
                collection: (*collection).to_string(),
                relative_path: (*path).to_string(),
                mtime: 1,
            };
            config_db
                .set_document_metadata_typed(doc_id.numeric, &meta)
                .unwrap();
            embed_docs.push((doc_id.numeric, format!("{title}\n{body}")));
        }

        let count =
            embed_and_store(&mut model, &embedding_db, embed_docs).unwrap();
        assert_eq!(count, docs.len(), "all docs should be embedded");

        (config_db, embedding_db, model, tmp)
    }

    #[test]
    fn semantic_search_empty_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let mut model = ModelManager::new();
        let args = make_semantic_args("anything");

        let results = execute_semantic_search(
            &args,
            &config_db,
            &embedding_db,
            &mut model,
        )
        .unwrap();

        assert!(results.is_empty());
    }

    #[test]
    #[ignore = "requires ColBERT model download"]
    fn e2e_search_with_reranking_returns_results() {
        let (idx, emb_db, mut model, _tmp) = setup_e2e();

        let mut args = make_search_args("rust programming language");
        args.bm25_only = false;

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(!results.is_empty(), "reranked search should return results");
        // Rust guide should be the top result for "rust programming language"
        assert_eq!(
            results[0].path, "rust-guide.md",
            "Rust guide should rank first after ColBERT reranking"
        );
    }

    #[test]
    #[ignore = "requires ColBERT model download"]
    fn e2e_semantic_only_search_returns_results() {
        let (config_db, embedding_db, mut model, _tmp) = setup_semantic_e2e();
        let args = make_semantic_args("rust programming language");

        let results = execute_semantic_search(
            &args,
            &config_db,
            &embedding_db,
            &mut model,
        )
        .unwrap();

        assert!(!results.is_empty(), "semantic search should return results");
        assert_eq!(
            results[0].path, "rust-guide.md",
            "Rust guide should rank first for rust query"
        );
    }

    #[test]
    #[ignore = "requires ColBERT model download"]
    fn e2e_reranking_improves_relevance() {
        let (idx, emb_db, mut model, _tmp) = setup_e2e();

        // Query about cooking - should rank pasta doc highest
        let mut args = make_search_args("how to cook food");
        args.bm25_only = false;

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(!results.is_empty());
        assert_eq!(
            results[0].path, "cooking-pasta.md",
            "cooking doc should rank first for cooking query after reranking"
        );
    }

    #[test]
    #[ignore = "requires ColBERT model download"]
    fn e2e_reranked_scores_are_descending() {
        let (idx, emb_db, mut model, _tmp) = setup_e2e();

        let mut args = make_search_args("programming");
        args.bm25_only = false;

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(results.len() >= 2);
        for window in results.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "reranked scores should be in descending order: {} >= {}",
                window[0].score,
                window[1].score,
            );
        }
    }

    #[test]
    #[ignore = "requires ColBERT model download"]
    fn e2e_reranking_with_min_score() {
        let (idx, emb_db, mut model, _tmp) = setup_e2e();

        // First get all results to find a reasonable threshold
        let mut args = make_search_args("rust");
        args.bm25_only = false;
        args.all = true;

        let all_results =
            execute_search(&args, &idx, &emb_db, &mut model).unwrap();
        assert!(!all_results.is_empty());

        // Use the median score as threshold - should filter out ~half
        let mid = all_results.len() / 2;
        let threshold = all_results[mid].score;

        args.min_score = threshold;
        let filtered =
            execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(
            filtered.len() <= all_results.len(),
            "min_score filter should reduce result count"
        );
        for r in &filtered {
            assert!(
                r.score >= threshold,
                "all results should meet min_score threshold"
            );
        }
    }

    #[test]
    #[ignore = "requires ColBERT model download"]
    fn e2e_semantic_search_understands_meaning() {
        let (idx, emb_db, mut model, _tmp) = setup_e2e();

        // "memory management" doesn't appear verbatim, but is
        // semantically related to Rust's ownership/borrow checker
        let mut args = make_search_args("memory management");
        args.bm25_only = false;

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(
            !results.is_empty(),
            "semantic search should find related docs"
        );
        // The Rust doc discusses memory safety - should rank high
        let rust_pos = results.iter().position(|r| r.path == "rust-guide.md");
        assert!(
            rust_pos.is_some(),
            "Rust guide should appear in semantic search for 'memory management'"
        );
    }

    #[test]
    #[ignore = "requires ColBERT model download"]
    fn e2e_all_result_fields_populated() {
        let (idx, emb_db, mut model, _tmp) = setup_e2e();

        let mut args = make_search_args("gardening plants");
        args.bm25_only = false;

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(!results.is_empty());
        let r = &results[0];
        assert!(!r.doc_id.is_empty(), "doc_id should be populated");
        assert!(r.doc_num_id > 0, "doc_num_id should be non-zero");
        assert!(!r.collection.is_empty(), "collection should be populated");
        assert!(!r.path.is_empty(), "path should be populated");
        assert!(!r.title.is_empty(), "title should be populated");
        assert!(r.score > 0.0, "score should be positive");
        assert_eq!(r.rank, 1, "first result should have rank 1");
    }

    #[test]
    #[ignore = "requires ColBERT model download"]
    fn e2e_reranking_with_collection_filter() {
        let (idx, emb_db, mut model, _tmp) = setup_e2e();

        let mut args = make_search_args("programming");
        args.bm25_only = false;
        args.collection = Some("notes".to_string());
        args.no_fuzzy = true;

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(
                r.collection, "notes",
                "all reranked results should be from 'notes' collection"
            );
        }
        // "docs" collection results should be excluded
        assert!(
            results.iter().all(|r| r.collection == "notes"),
            "no results from other collections"
        );
    }

    #[test]
    #[ignore = "requires ColBERT model download"]
    fn e2e_reranking_count_and_score_combined() {
        let (idx, emb_db, mut model, _tmp) = setup_e2e();

        // First get all results
        let mut args = make_search_args("programming language");
        args.bm25_only = false;
        args.all = true;

        let all_results =
            execute_search(&args, &idx, &emb_db, &mut model).unwrap();
        assert!(
            all_results.len() >= 2,
            "should have multiple results to test filtering"
        );

        // Apply count limit
        args.all = false;
        args.count = 1;
        let limited = execute_search(&args, &idx, &emb_db, &mut model).unwrap();
        assert_eq!(limited.len(), 1, "count=1 should limit to 1 result");
        assert_eq!(
            limited[0].path, all_results[0].path,
            "limited result should be the top-ranked one"
        );
    }

    #[test]
    #[ignore = "requires ColBERT model download"]
    fn e2e_bm25_and_reranked_return_same_docs() {
        let (idx, emb_db, mut model, _tmp) = setup_e2e();

        // BM25-only
        let mut bm25_args = make_search_args("rust safety");
        bm25_args.bm25_only = true;
        bm25_args.all = true;
        let bm25_results =
            execute_search(&bm25_args, &idx, &emb_db, &mut model).unwrap();

        // With reranking
        let mut rerank_args = make_search_args("rust safety");
        rerank_args.bm25_only = false;
        rerank_args.all = true;
        let reranked_results =
            execute_search(&rerank_args, &idx, &emb_db, &mut model).unwrap();

        // Both should find results
        assert!(!bm25_results.is_empty());
        assert!(!reranked_results.is_empty());

        // Reranked should contain a subset of BM25 results (only those
        // with embeddings, but we embedded everything)
        let bm25_paths: std::collections::HashSet<&str> =
            bm25_results.iter().map(|r| r.path.as_str()).collect();
        let reranked_paths: std::collections::HashSet<&str> =
            reranked_results.iter().map(|r| r.path.as_str()).collect();

        assert!(
            reranked_paths.is_subset(&bm25_paths),
            "reranked results should be a subset of BM25 results"
        );
    }

    #[test]
    #[ignore = "requires ColBERT model download"]
    fn e2e_reranking_ranks_are_sequential() {
        let (idx, emb_db, mut model, _tmp) = setup_e2e();

        let mut args = make_search_args("programming");
        args.bm25_only = false;

        let results = execute_search(&args, &idx, &emb_db, &mut model).unwrap();

        assert!(results.len() >= 2);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(
                r.rank,
                i + 1,
                "reranked ranks should be 1-indexed and sequential"
            );
        }
    }

    // -- Unit tests for helper functions --

    #[test]
    fn search_mode_parse_roundtrips_semantic() {
        assert_eq!(SearchMode::parse("semantic"), Some(SearchMode::Semantic));
        assert_eq!(SearchMode::Semantic.as_str(), "semantic");
    }

    #[test]
    fn search_mode_parse_roundtrips_hybrid() {
        assert_eq!(SearchMode::parse("hybrid"), Some(SearchMode::Hybrid));
        assert_eq!(SearchMode::Hybrid.as_str(), "hybrid");
    }

    #[test]
    fn search_mode_parse_rejects_unknown_value() {
        assert_eq!(SearchMode::parse("bm25"), None);
    }

    #[test]
    fn short_doc_id_format() {
        let id = short_doc_id(0x123456789abcdef0);
        assert!(id.starts_with('#'));
        assert_eq!(id.len(), 7); // # + 6 hex chars
    }

    #[test]
    fn short_doc_id_uses_first_six_hex_digits() {
        assert_eq!(short_doc_id(0x123456789abcdef0), "#123456");
        assert_eq!(short_doc_id(0xabcdef1234567890), "#abcdef");
    }

    #[test]
    fn short_doc_id_zero() {
        let id = short_doc_id(0);
        assert_eq!(id, "#000000");
    }

    #[test]
    fn bm25_to_final_empty() {
        let results = bm25_to_final(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn bm25_to_final_preserves_fields() {
        let input = vec![SearchResult {
            score: 1.5,
            doc_id: "abc".to_string(),
            doc_num_id: 42,
            collection: "notes".to_string(),
            path: "hello.md".to_string(),
            title: "Hello".to_string(),
            mtime: 1000,
        }];
        let output = bm25_to_final(&input);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].rank, 1);
        assert_eq!(output[0].score, 1.5);
        assert_eq!(output[0].doc_id, "abc");
        assert_eq!(output[0].doc_num_id, 42);
        assert_eq!(output[0].collection, "notes");
        assert_eq!(output[0].path, "hello.md");
        assert_eq!(output[0].title, "Hello");
    }

    #[test]
    fn bm25_to_final_sets_sequential_ranks() {
        let input = vec![
            SearchResult {
                score: 3.0,
                doc_id: "a".to_string(),
                doc_num_id: 1,
                collection: "c".to_string(),
                path: "a.md".to_string(),
                title: "A".to_string(),
                mtime: 1,
            },
            SearchResult {
                score: 2.0,
                doc_id: "b".to_string(),
                doc_num_id: 2,
                collection: "c".to_string(),
                path: "b.md".to_string(),
                title: "B".to_string(),
                mtime: 2,
            },
            SearchResult {
                score: 1.0,
                doc_id: "c".to_string(),
                doc_num_id: 3,
                collection: "c".to_string(),
                path: "c.md".to_string(),
                title: "C".to_string(),
                mtime: 3,
            },
        ];
        let output = bm25_to_final(&input);
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].rank, 1);
        assert_eq!(output[1].rank, 2);
        assert_eq!(output[2].rank, 3);
    }

    #[test]
    fn search_json_snapshot() {
        let results = vec![FinalResult {
            rank: 1,
            score: 1.2345678,
            doc_id: "#abc123".to_string(),
            doc_num_id: 42,
            collection: "notes".to_string(),
            path: "hello.md".to_string(),
            title: "Hello \"Rust\"".to_string(),
        }];

        let json = format_json_string(&results, "rust\nquery");

        assert_eq!(
            json,
            "{\"query\":\"rust\\nquery\",\"result_count\":1,\"results\":[{\"rank\":1,\"score\":1.234568,\"doc_id\":\"#abc123\",\"collection\":\"notes\",\"path\":\"hello.md\",\"title\":\"Hello \\\"Rust\\\"\"}]}"
        );
    }

    #[test]
    fn search_json_score_precision_is_six_decimals() {
        let results = vec![FinalResult {
            rank: 1,
            score: 1.2,
            doc_id: "#abc123".to_string(),
            doc_num_id: 42,
            collection: "notes".to_string(),
            path: "hello.md".to_string(),
            title: "Hello".to_string(),
        }];

        let json = format_json_string(&results, "rust");
        assert!(json.contains("\"score\":1.200000"));
    }

    #[test]
    fn json_escape_basic() {
        assert_eq!(json_escape("hello"), "\"hello\"");
        assert_eq!(json_escape("he\"llo"), "\"he\\\"llo\"");
        assert_eq!(json_escape("he\\llo"), "\"he\\\\llo\"");
        assert_eq!(json_escape("he\nllo"), "\"he\\nllo\"");
        assert_eq!(json_escape("he\tllo"), "\"he\\tllo\"");
        assert_eq!(json_escape("he\rllo"), "\"he\\rllo\"");
        assert_eq!(json_escape("he\x01llo"), "\"he\\u0001llo\"");
        // Unicode passes through
        assert_eq!(json_escape("caf\u{00e9}"), "\"caf\u{00e9}\"");
    }
}
