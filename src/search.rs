use crate::{
    cli::SearchArgs,
    embedding_db::EmbeddingDb,
    error::Result,
    model_manager::ModelManager,
    reranker::{self, RankedDocument},
    tantivy_index::{SearchIndex, SearchResult},
};

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
    args: &SearchArgs,
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
    use crate::doc_id::DocumentId;

    fn make_search_args(query: &str) -> SearchArgs {
        SearchArgs {
            query: query.to_string(),
            count: 10,
            collection: None,
            json: false,
            all: false,
            files: false,
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
            EmbeddingDb::open(&tmp.path().join("emb.redb")).unwrap();

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
            EmbeddingDb::open(&tmp.path().join("emb.redb")).unwrap();
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
            embed_and_store(&mut model, &embedding_db, &embed_docs).unwrap();
        assert_eq!(count, docs.len(), "all docs should be embedded");

        (idx, embedding_db, model, tmp)
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
}
