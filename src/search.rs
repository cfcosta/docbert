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
    let ranked =
        reranker::rerank(&query_embedding, &candidate_ids, embedding_db)?;

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
}
