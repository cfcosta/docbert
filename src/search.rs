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

    // Stage 1: BM25 retrieval
    let bm25_results = if let Some(ref collection) = args.collection {
        search_index.search_in_collection(
            &args.query,
            collection,
            bm25_limit,
        )?
    } else {
        search_index.search(&args.query, bm25_limit)?
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
