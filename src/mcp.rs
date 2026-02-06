use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use rmcp::{
    ServerHandler,
    ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolResult,
        Content,
        Implementation,
        ServerCapabilities,
        ServerInfo,
    },
    tool,
    tool_handler,
    tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    cli::SearchArgs,
    config_db::ConfigDb,
    data_dir::DataDir,
    embedding_db::EmbeddingDb,
    error,
    model_manager::ModelManager,
    search,
    tantivy_index::SearchIndex,
};

const DEFAULT_SEARCH_LIMIT: usize = 10;
const DEFAULT_SNIPPET_LINES: usize = 6;
const DEFAULT_SNIPPET_MAX_CHARS: usize = 400;

struct DocbertState {
    config_db: ConfigDb,
    search_index: SearchIndex,
    embedding_db: EmbeddingDb,
    model: Mutex<ModelManager>,
}

#[derive(Clone)]
pub struct DocbertMcpServer {
    state: Arc<DocbertState>,
    tool_router: ToolRouter<Self>,
}

impl DocbertMcpServer {
    fn new(state: DocbertState) -> Self {
        Self {
            state: Arc::new(state),
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router(router = tool_router)]
impl DocbertMcpServer {
    /// Search indexed documents with BM25 + optional ColBERT reranking.
    #[tool(
        name = "docbert_search",
        description = "Search indexed documents. Supports collection filtering, score thresholds, and BM25-only mode."
    )]
    pub async fn docbert_search(
        &self,
        params: Parameters<SearchParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let params = params.0;
        let query = params.query.clone();

        let args = SearchArgs {
            query: params.query,
            count: params.limit.unwrap_or(DEFAULT_SEARCH_LIMIT),
            collection: params.collection.clone(),
            json: false,
            all: params.all.unwrap_or(false),
            files: false,
            min_score: params.min_score.unwrap_or(0.0),
            bm25_only: params.bm25_only.unwrap_or(false),
            no_fuzzy: params.no_fuzzy.unwrap_or(false),
        };

        let mut model = self.state.model.lock().map_err(|_| {
            rmcp::ErrorData::internal_error("model lock poisoned", None)
        })?;

        let results = search::execute_search(
            &args,
            &self.state.search_index,
            &self.state.embedding_db,
            &mut model,
        )
        .map_err(|e| mcp_error("search failed", e))?;

        let include_snippet = params.include_snippet.unwrap_or(true);
        let mut items = Vec::with_capacity(results.len());

        for r in results {
            let file = format!("{}/{}", r.collection, r.path);
            let context =
                context_for_doc(&self.state.config_db, &r.collection, &r.path);
            let snippet = if include_snippet {
                resolve_full_path(&self.state.config_db, &r.collection, &r.path)
                    .and_then(|path| std::fs::read_to_string(path).ok())
                    .and_then(|content| extract_snippet(&content, &query))
                    .map(|(snippet, start_line)| {
                        add_line_numbers(&snippet, start_line)
                    })
            } else {
                None
            };

            items.push(SearchResultItem {
                doc_id: format!("#{}", r.doc_id),
                collection: r.collection,
                path: r.path,
                file,
                title: r.title,
                score: r.score,
                context,
                snippet,
            });
        }

        let summary = format_search_summary(&items, &query);
        let structured = serde_json::to_value(SearchResponse {
            query,
            result_count: items.len(),
            results: items,
        })
        .map_err(|e| mcp_error("failed to serialize search results", e))?;

        Ok(CallToolResult {
            content: vec![Content::text(summary)],
            structured_content: Some(structured),
            is_error: Some(false),
            meta: None,
        })
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for DocbertMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "docbert".to_string(),
                title: Some("docbert MCP".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                icons: None,
                website_url: Some("https://github.com/cfcosta/docbert".to_string()),
            },
            instructions: Some(
                "Use docbert_search to find documents by keyword or concept. Use collection filters when possible."
                    .to_string(),
            ),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SearchParams {
    /// Search query string.
    pub query: String,
    /// Maximum number of results (default: 10).
    pub limit: Option<usize>,
    /// Minimum score threshold.
    pub min_score: Option<f32>,
    /// Restrict to a specific collection name.
    pub collection: Option<String>,
    /// Skip ColBERT reranking, return BM25 results directly.
    pub bm25_only: Option<bool>,
    /// Disable fuzzy matching in the first stage.
    pub no_fuzzy: Option<bool>,
    /// Return all results above the score threshold.
    pub all: Option<bool>,
    /// Include a snippet preview (default: true).
    pub include_snippet: Option<bool>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchResponse {
    query: String,
    result_count: usize,
    results: Vec<SearchResultItem>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultItem {
    doc_id: String,
    collection: String,
    path: String,
    file: String,
    title: String,
    score: f32,
    context: Option<String>,
    snippet: Option<String>,
}

fn format_search_summary(results: &[SearchResultItem], query: &str) -> String {
    if results.is_empty() {
        return format!("No results found for \"{query}\"");
    }

    let mut lines = Vec::with_capacity(results.len() + 1);
    let suffix = if results.len() == 1 { "" } else { "s" };
    lines.push(format!(
        "Found {} result{} for \"{query}\":",
        results.len(),
        suffix
    ));

    for item in results {
        lines.push(format!("{} {:.3} {}", item.doc_id, item.score, item.file));
    }

    lines.join("\n")
}

fn add_line_numbers(text: &str, start_line: usize) -> String {
    text.lines()
        .enumerate()
        .map(|(i, line)| format!("{}: {}", start_line + i, line))
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_snippet(text: &str, query: &str) -> Option<(String, usize)> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return None;
    }

    let query_lower = query.to_lowercase();
    let mut match_idx = None;

    for (idx, line) in lines.iter().enumerate() {
        if line.to_lowercase().contains(&query_lower) {
            match_idx = Some(idx);
            break;
        }
    }

    let (start, end) = if let Some(idx) = match_idx {
        let start = idx.saturating_sub(2);
        let end = (idx + 3).min(lines.len());
        (start, end)
    } else {
        (0, DEFAULT_SNIPPET_LINES.min(lines.len()))
    };

    let mut snippet = lines[start..end].join("\n");
    if snippet.len() > DEFAULT_SNIPPET_MAX_CHARS {
        snippet.truncate(DEFAULT_SNIPPET_MAX_CHARS);
        snippet.push_str("...");
    }

    Some((snippet, start + 1))
}

fn resolve_full_path(
    config_db: &ConfigDb,
    collection: &str,
    path: &str,
) -> Option<PathBuf> {
    let base = config_db.get_collection(collection).ok()??;
    Some(PathBuf::from(base).join(path))
}

fn context_for_doc(
    config_db: &ConfigDb,
    collection: &str,
    path: &str,
) -> Option<String> {
    let doc_uri = format!("bert://{collection}/{path}");
    if let Ok(Some(ctx)) = config_db.get_context(&doc_uri) {
        return Some(ctx);
    }

    let collection_uri = format!("bert://{collection}");
    config_db.get_context(&collection_uri).ok().flatten()
}

fn mcp_error(message: &str, error: impl std::fmt::Display) -> rmcp::ErrorData {
    rmcp::ErrorData::internal_error(
        message.to_string(),
        Some(json!({ "error": error.to_string() })),
    )
}

pub fn run_mcp(data_dir: DataDir, config_db: ConfigDb) -> error::Result<()> {
    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;

    let state = DocbertState {
        config_db,
        search_index,
        embedding_db,
        model: Mutex::new(ModelManager::default()),
    };

    let server = DocbertMcpServer::new(state);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| {
            error::Error::Config(format!("failed to start tokio runtime: {e}"))
        })?;

    runtime.block_on(async move {
        let transport = rmcp::transport::stdio();
        let running = server.serve(transport).await.map_err(|e| {
            error::Error::Config(format!(
                "MCP server initialization failed: {e}"
            ))
        })?;
        running.waiting().await.map_err(|e| {
            error::Error::Config(format!("MCP server error: {e}"))
        })?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc_id::DocumentId;

    #[tokio::test]
    async fn search_tool_returns_structured_results() {
        let tmp = tempfile::tempdir().unwrap();
        let collection_dir = tmp.path().join("notes");
        std::fs::create_dir_all(&collection_dir).unwrap();
        let file_path = collection_dir.join("rust.md");
        std::fs::write(
            &file_path,
            "Rust is fast.\nOwnership keeps memory safe.\n",
        )
        .unwrap();

        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        config_db
            .set_collection("notes", collection_dir.to_str().unwrap())
            .unwrap();
        config_db
            .set_context("bert://notes", "Personal notes")
            .unwrap();

        let search_index = SearchIndex::open_in_ram().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("embeddings.db")).unwrap();

        let mut writer = search_index.writer(15_000_000).unwrap();
        let doc_id = DocumentId::new("notes", "rust.md");
        search_index
            .add_document(
                &writer,
                &doc_id.short,
                doc_id.numeric,
                "notes",
                "rust.md",
                "Rust Intro",
                "Rust is fast. Ownership keeps memory safe.",
                1,
            )
            .unwrap();
        writer.commit().unwrap();

        let server = DocbertMcpServer::new(DocbertState {
            config_db,
            search_index,
            embedding_db,
            model: Mutex::new(ModelManager::new()),
        });

        let params = SearchParams {
            query: "Rust".to_string(),
            limit: Some(5),
            min_score: Some(0.0),
            collection: Some("notes".to_string()),
            bm25_only: Some(true),
            no_fuzzy: Some(true),
            all: Some(false),
            include_snippet: Some(true),
        };

        let result = server.docbert_search(Parameters(params)).await.unwrap();

        let structured = result.structured_content.expect("structured");
        let results = structured
            .get("results")
            .and_then(|v| v.as_array())
            .expect("results array");

        assert_eq!(results.len(), 1);
        let first = &results[0];

        assert_eq!(
            first.get("collection").and_then(|v| v.as_str()),
            Some("notes")
        );
        assert_eq!(first.get("path").and_then(|v| v.as_str()), Some("rust.md"));
        assert_eq!(
            first.get("context").and_then(|v| v.as_str()),
            Some("Personal notes")
        );
        let snippet =
            first.get("snippet").and_then(|v| v.as_str()).unwrap_or("");
        assert!(snippet.contains("1: Rust is fast."));

        let summary = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map(|t| t.text.clone())
            .unwrap_or_default();
        assert!(summary.contains("Found 1 result"));
    }
}
