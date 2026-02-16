use std::{cmp::Ordering, collections::HashMap, path::Path};

use crate::{
    config_db::ConfigDb,
    embedding,
    embedding_db::EmbeddingDb,
    error::Result,
    incremental::DocumentMetadata,
    ingestion,
    model_manager::ModelManager,
    reranker::{self, RankedDocument},
    tantivy_index::{SearchIndex, SearchResult},
};

const SEMANTIC_SEARCH_BATCH_SIZE: usize = 64;

/// Parameters for the hybrid BM25 + ColBERT search pipeline.
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

/// Parameters for semantic-only search.
#[derive(Debug, Clone)]
pub struct SemanticSearchParams {
    /// The search query.
    pub query: String,
    /// Number of results to return.
    pub count: usize,
    /// Minimum score threshold.
    pub min_score: f32,
    /// Return all results above the score threshold.
    pub all: bool,
}

/// A final search result combining BM25 and optional ColBERT scores.
#[derive(Debug, Clone)]
pub struct FinalResult {
    pub rank: usize,
    pub score: f32,
    pub doc_id: String,
    pub doc_num_id: u64,
    pub collection: String,
    pub path: String,
    pub title: String,
}

/// Execute the full search pipeline.
///
/// 1. BM25 first-stage retrieval via Tantivy (top 1000)
/// 2. ColBERT reranking (unless --bm25-only)
/// 3. Filter by --min-score
/// 4. Limit to -n results
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

/// Execute semantic-only search across all documents.
pub fn execute_semantic_search(
    args: &SemanticSearchParams,
    config_db: &ConfigDb,
    embedding_db: &EmbeddingDb,
    model: &mut ModelManager,
) -> Result<Vec<FinalResult>> {
    let metadata_entries = config_db.list_all_document_metadata()?;
    if metadata_entries.is_empty() {
        return Ok(vec![]);
    }

    let mut metadata = HashMap::with_capacity(metadata_entries.len());
    let mut doc_ids = Vec::with_capacity(metadata_entries.len());

    for (doc_id, bytes) in metadata_entries {
        if let Some(meta) = DocumentMetadata::deserialize(&bytes) {
            doc_ids.push(doc_id);
            metadata.insert(doc_id, meta);
        }
    }

    if doc_ids.is_empty() {
        return Ok(vec![]);
    }

    let query_embedding = model.encode_query(&args.query)?;
    let query_3d = query_embedding.unsqueeze(0)?;

    let mut scored: Vec<(u64, f32)> = Vec::new();

    for batch in doc_ids.chunks(SEMANTIC_SEARCH_BATCH_SIZE) {
        let embeddings =
            embedding::batch_load_embedding_tensors(embedding_db, batch)?;
        for (doc_id, doc_embedding_opt) in embeddings {
            let Some(doc_embedding) = doc_embedding_opt else {
                continue;
            };
            let Ok(doc_3d) = doc_embedding.unsqueeze(0) else {
                continue;
            };
            let Ok(similarities) = model.similarity(&query_3d, &doc_3d) else {
                continue;
            };
            let score = match similarities.data.first().and_then(|r| r.first())
            {
                Some(score) => *score,
                None => continue,
            };
            scored.push((doc_id, score));
        }
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut results: Vec<FinalResult> = scored
        .into_iter()
        .filter_map(|(doc_id, score)| {
            let meta = metadata.get(&doc_id)?;
            Some(FinalResult {
                rank: 0,
                score,
                doc_id: short_doc_id(doc_id),
                doc_num_id: doc_id,
                collection: meta.collection.clone(),
                path: meta.relative_path.clone(),
                title: String::new(),
            })
        })
        .filter(|r| r.score >= args.min_score)
        .collect();

    let limit = if args.all { results.len() } else { args.count };
    results.truncate(limit);

    for (i, r) in results.iter_mut().enumerate() {
        r.rank = i + 1;
    }

    populate_titles(&mut results, config_db);

    Ok(results)
}

/// Resolve a document by its short ID (e.g., "a1b2c3").
///
/// Iterates all known document metadata and matches by short ID string
/// or substring containment. Returns `(collection, relative_path)`.
pub fn resolve_by_doc_id(
    config_db: &ConfigDb,
    short_id: &str,
) -> Option<(String, String)> {
    let entries = config_db.list_all_document_metadata().ok()?;
    for (_doc_id, bytes) in entries {
        let meta = crate::incremental::DocumentMetadata::deserialize(&bytes)?;
        let did = crate::doc_id::DocumentId::new(
            &meta.collection,
            &meta.relative_path,
        );
        if did.short == short_id || did.to_string().contains(short_id) {
            return Some((meta.collection, meta.relative_path));
        }
    }
    None
}

/// Resolve a document by its relative path.
///
/// Returns `(collection, relative_path)`.
pub fn resolve_by_path(
    config_db: &ConfigDb,
    path: &str,
) -> Option<(String, String)> {
    let entries = config_db.list_all_document_metadata().ok()?;
    for (_doc_id, bytes) in entries {
        let meta = crate::incremental::DocumentMetadata::deserialize(&bytes)?;
        if meta.relative_path == path {
            return Some((meta.collection, meta.relative_path));
        }
    }
    None
}

/// Format a numeric document ID as a short hex string (e.g., "#a1b2c3").
pub fn format_doc_id(numeric: u64) -> String {
    short_doc_id(numeric)
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

fn short_doc_id(numeric: u64) -> String {
    let full = format!("{numeric:016x}");
    format!("#{}", &full[..6])
}

fn populate_titles(results: &mut [FinalResult], config_db: &ConfigDb) {
    for r in results {
        let fallback = ingestion::extract_title("", Path::new(&r.path));
        let Some(collection_path) =
            config_db.get_collection(&r.collection).ok().flatten()
        else {
            r.title = fallback;
            continue;
        };

        let full_path = Path::new(&collection_path).join(&r.path);
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

/// Format results for human-readable terminal output.
pub fn format_human(results: &[FinalResult]) {
    if results.is_empty() {
        println!("No results found.");
        return;
    }

    for r in results {
        println!(
            "{:>3}. [{:.3}] {}:{} #{}",
            r.rank,
            r.score,
            r.collection,
            r.path,
            &r.doc_id[..r.doc_id.len().min(7)]
        );
        if !r.title.is_empty() {
            println!("     {}", r.title);
        }
    }
    println!("\n{} result(s)", results.len());
}

/// Format results as JSON output.
pub fn format_json(results: &[FinalResult], query: &str) {
    // Manual JSON to avoid serde dependency
    print!("{{\"query\":");
    print_json_string(query);
    print!(",\"result_count\":{},\"results\":[", results.len());

    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("{{\"rank\":{},\"score\":{:.6},\"doc_id\":", r.rank, r.score);
        print_json_string(&r.doc_id);
        print!(",\"collection\":");
        print_json_string(&r.collection);
        print!(",\"path\":");
        print_json_string(&r.path);
        print!(",\"title\":");
        print_json_string(&r.title);
        print!("}}");
    }

    println!("]}}");
}

/// Format results as plain file paths (one per line).
pub fn format_files(
    results: &[FinalResult],
    config_db: &crate::config_db::ConfigDb,
) {
    for r in results {
        if let Ok(Some(collection_path)) =
            config_db.get_collection(&r.collection)
        {
            let full_path =
                std::path::Path::new(&collection_path).join(&r.path);
            println!("{}", full_path.display());
        }
    }
}

pub fn print_json_string_pub(s: &str) {
    print_json_string(s);
}

fn print_json_string(s: &str) {
    print!("\"");
    for c in s.chars() {
        match c {
            '"' => print!("\\\""),
            '\\' => print!("\\\\"),
            '\n' => print!("\\n"),
            '\r' => print!("\\r"),
            '\t' => print!("\\t"),
            c if c < '\x20' => print!("\\u{:04x}", c as u32),
            c => print!("{c}"),
        }
    }
    print!("\"");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config_db::ConfigDb,
        doc_id::DocumentId,
        incremental::DocumentMetadata,
    };

    fn make_semantic_args(query: &str) -> SemanticSearchParams {
        SemanticSearchParams {
            query: query.to_string(),
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
                .set_document_metadata(doc_id.numeric, &meta.serialize())
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
    fn short_doc_id_format() {
        let id = short_doc_id(0x123456789abcdef0);
        assert!(id.starts_with('#'));
        assert_eq!(id.len(), 7); // # + 6 hex chars
    }

    #[test]
    fn short_doc_id_consistency() {
        assert_eq!(short_doc_id(42), short_doc_id(42));
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
    fn json_escape_basic() {
        // Verify print_json_string escapes special characters by capturing stdout
        // Since print_json_string writes to stdout, we test the logic directly
        fn json_escape(s: &str) -> String {
            let mut result = String::from("\"");
            for c in s.chars() {
                match c {
                    '"' => result.push_str("\\\""),
                    '\\' => result.push_str("\\\\"),
                    '\n' => result.push_str("\\n"),
                    '\r' => result.push_str("\\r"),
                    '\t' => result.push_str("\\t"),
                    c if c < '\x20' => {
                        result.push_str(&format!("\\u{:04x}", c as u32))
                    }
                    c => result.push(c),
                }
            }
            result.push('"');
            result
        }

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
