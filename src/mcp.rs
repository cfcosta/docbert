use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use globset::Glob;
use percent_encoding::{
    NON_ALPHANUMERIC,
    percent_decode_str,
    utf8_percent_encode,
};
use rmcp::{
    RoleServer,
    ServerHandler,
    ServiceExt,
    handler::server::{
        router::{prompt::PromptRouter, tool::ToolRouter},
        wrapper::Parameters,
    },
    model::{
        AnnotateAble,
        CallToolResult,
        Content,
        GetPromptRequestParams,
        GetPromptResult,
        Implementation,
        ListPromptsResult,
        ListResourceTemplatesResult,
        PaginatedRequestParams,
        PromptMessage,
        PromptMessageRole,
        RawResourceTemplate,
        ReadResourceRequestParams,
        ReadResourceResult,
        ResourceContents,
        ServerCapabilities,
        ServerInfo,
    },
    prompt,
    prompt_handler,
    prompt_router,
    service::RequestContext,
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
    doc_id::DocumentId,
    embedding_db::EmbeddingDb,
    error,
    incremental::DocumentMetadata,
    model_manager::{DEFAULT_MODEL_ID, ModelManager},
    search,
    tantivy_index::SearchIndex,
};

const DEFAULT_SEARCH_LIMIT: usize = 10;
const DEFAULT_SNIPPET_LINES: usize = 6;
const DEFAULT_SNIPPET_MAX_CHARS: usize = 400;
const DEFAULT_MULTI_GET_MAX_BYTES: u64 = 10_240;

struct DocbertState {
    data_dir: DataDir,
    config_db: ConfigDb,
    search_index: SearchIndex,
    embedding_db: EmbeddingDb,
    model: Mutex<ModelManager>,
}

#[derive(Clone)]
pub struct DocbertMcpServer {
    state: Arc<DocbertState>,
    tool_router: ToolRouter<Self>,
    prompt_router: PromptRouter<Self>,
}

impl DocbertMcpServer {
    fn new(state: DocbertState) -> Self {
        Self {
            state: Arc::new(state),
            tool_router: Self::tool_router(),
            prompt_router: Self::prompt_router(),
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

    /// Retrieve a document by reference (collection:path, #doc_id, or path).
    #[tool(
        name = "docbert_get",
        description = "Retrieve a document by reference (collection:path, #doc_id, or path). Supports optional line ranges."
    )]
    pub async fn docbert_get(
        &self,
        params: Parameters<GetParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let params = params.0;

        let mut reference = params.reference.clone();
        let mut from_line = params.from_line;

        if from_line.is_none()
            && let Some((base, line_str)) = reference.rsplit_once(':')
            && !line_str.is_empty()
            && line_str.chars().all(|c| c.is_ascii_digit())
        {
            from_line = line_str.parse::<usize>().ok();
            reference = base.to_string();
        }

        let (collection, path) =
            resolve_reference(&self.state.config_db, &reference)?;

        let full_path =
            resolve_full_path(&self.state.config_db, &collection, &path)
                .ok_or_else(|| {
                    rmcp::ErrorData::resource_not_found(
                        format!("collection not found: {collection}"),
                        None,
                    )
                })?;

        if let Some(max_bytes) = params.max_bytes {
            let size = std::fs::metadata(&full_path)
                .map(|m| m.len())
                .unwrap_or_default();
            if size > max_bytes {
                return Ok(CallToolResult::error(vec![Content::text(
                    format!(
                        "File too large ({} bytes > {}): {}",
                        size,
                        max_bytes,
                        full_path.display()
                    ),
                )]));
            }
        }

        let content = std::fs::read_to_string(&full_path)
            .map_err(|e| mcp_error("failed to read document", e))?;

        let start_line = from_line.unwrap_or(1);
        let mut body =
            apply_line_limits(&content, start_line, params.max_lines);

        if params.line_numbers.unwrap_or(false) {
            body = add_line_numbers(&body, start_line);
        }

        if let Some(context) =
            context_for_doc(&self.state.config_db, &collection, &path)
        {
            body = format!("<!-- Context: {context} -->\n\n{body}");
        }

        let uri = format!("bert://{}/{}", collection, encode_bert_path(&path));
        let resource = ResourceContents::TextResourceContents {
            uri,
            mime_type: Some("text/markdown".to_string()),
            text: body,
            meta: None,
        };

        Ok(CallToolResult::success(vec![Content::resource(resource)]))
    }

    /// Retrieve multiple documents by glob pattern.
    #[tool(
        name = "docbert_multi_get",
        description = "Retrieve multiple documents by glob pattern. Supports collection filters and size limits."
    )]
    pub async fn docbert_multi_get(
        &self,
        params: Parameters<MultiGetParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let params = params.0;
        let matcher = Glob::new(&params.pattern)
            .map_err(|e| {
                rmcp::ErrorData::invalid_params(
                    format!("invalid glob pattern: {e}"),
                    None,
                )
            })?
            .compile_matcher();

        let mut matches: Vec<(String, String)> = Vec::new();
        let metadata = self
            .state
            .config_db
            .list_all_document_metadata()
            .map_err(|e| mcp_error("failed to load document metadata", e))?;

        for (_doc_id, bytes) in metadata {
            if let Some(meta) = DocumentMetadata::deserialize(&bytes) {
                if let Some(ref collection) = params.collection
                    && meta.collection != *collection
                {
                    continue;
                }

                if matcher.is_match(&meta.relative_path) {
                    matches.push((meta.collection, meta.relative_path));
                }
            }
        }

        matches.sort();

        if matches.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "No documents match '{}'",
                params.pattern
            ))]));
        }

        let max_bytes = params.max_bytes.unwrap_or(DEFAULT_MULTI_GET_MAX_BYTES);
        let mut content: Vec<Content> = Vec::new();

        for (collection, path) in matches {
            let full_path =
                resolve_full_path(&self.state.config_db, &collection, &path);
            let Some(full_path) = full_path else {
                content.push(Content::text(format!(
                    "[SKIPPED: {collection}:{path} - collection not found]"
                )));
                continue;
            };

            let size = std::fs::metadata(&full_path)
                .map(|m| m.len())
                .unwrap_or_default();
            if size > max_bytes {
                content.push(Content::text(format!(
                    "[SKIPPED: {collection}:{path} - {size} bytes exceeds limit {max_bytes}]"
                )));
                continue;
            }

            let Ok(mut body) = std::fs::read_to_string(&full_path) else {
                content.push(Content::text(format!(
                    "[SKIPPED: {collection}:{path} - failed to read]"
                )));
                continue;
            };

            body = apply_line_limits(&body, 1, params.max_lines);
            if params.line_numbers.unwrap_or(false) {
                body = add_line_numbers(&body, 1);
            }

            if let Some(context) =
                context_for_doc(&self.state.config_db, &collection, &path)
            {
                body = format!("<!-- Context: {context} -->\n\n{body}");
            }

            let uri =
                format!("bert://{}/{}", collection, encode_bert_path(&path));
            let resource = ResourceContents::TextResourceContents {
                uri,
                mime_type: Some("text/markdown".to_string()),
                text: body,
                meta: None,
            };
            content.push(Content::resource(resource));
        }

        Ok(CallToolResult::success(content))
    }

    /// Show index status and collection summary.
    #[tool(
        name = "docbert_status",
        description = "Show index status, collections, and document counts."
    )]
    pub async fn docbert_status(
        &self,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let collections = self
            .state
            .config_db
            .list_collections()
            .map_err(|e| mcp_error("failed to list collections", e))?;
        let doc_ids = self
            .state
            .config_db
            .list_document_ids()
            .map_err(|e| mcp_error("failed to list document ids", e))?;
        let model_name = self
            .state
            .config_db
            .get_setting_or("model_name", DEFAULT_MODEL_ID)
            .map_err(|e| mcp_error("failed to read model setting", e))?;

        let mut counts = std::collections::HashMap::new();
        for (_doc_id, bytes) in self
            .state
            .config_db
            .list_all_document_metadata()
            .map_err(|e| mcp_error("failed to load document metadata", e))?
        {
            if let Some(meta) = DocumentMetadata::deserialize(&bytes) {
                *counts.entry(meta.collection).or_insert(0usize) += 1;
            }
        }

        let mut collection_status = Vec::with_capacity(collections.len());
        for (name, path) in collections {
            let documents = counts.get(&name).copied().unwrap_or(0);
            collection_status.push(StatusCollection {
                name,
                path,
                documents,
            });
        }

        let data_dir = self.state.data_dir.root().display().to_string();
        let summary = format_status_summary(
            &data_dir,
            &model_name,
            doc_ids.len(),
            &collection_status,
        );

        let structured = serde_json::to_value(StatusResponse {
            data_dir,
            model: model_name,
            documents: doc_ids.len(),
            collections: collection_status,
        })
        .map_err(|e| mcp_error("failed to serialize status", e))?;

        Ok(CallToolResult {
            content: vec![Content::text(summary)],
            structured_content: Some(structured),
            is_error: Some(false),
            meta: None,
        })
    }
}

#[prompt_router]
impl DocbertMcpServer {
    /// Docbert MCP query guide.
    #[prompt(
        name = "docbert_query",
        title = "Docbert Query Guide",
        description = "How to search and retrieve documents with docbert MCP"
    )]
    pub async fn query_guide(&self) -> Vec<PromptMessage> {
        vec![PromptMessage::new_text(
            PromptMessageRole::User,
            r#"# Docbert MCP Quick Guide

docbert indexes local document collections and provides MCP tools for search and retrieval.

## Tools

- docbert_search: keyword + semantic search (use collection filters when possible)
- docbert_get: fetch a single document by path or #doc_id
- docbert_multi_get: fetch multiple documents by glob pattern
- docbert_status: index health and collection summary

## Tips

- Use min_score to filter low-confidence results
- Use bm25_only for fast keyword-only search
- docbert_get supports from_line/max_lines and optional line numbers
"#,
        )]
    }
}

#[tool_handler(router = self.tool_router)]
#[prompt_handler(router = self.prompt_router)]
impl ServerHandler for DocbertMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
            server_info: Implementation {
                name: "docbert".to_string(),
                title: Some("docbert MCP".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                icons: None,
                website_url: Some("https://github.com/cfcosta/docbert".to_string()),
            },
            instructions: Some(
                "Use docbert_search to find documents, then docbert_get or docbert_multi_get to retrieve content. Use docbert_status for index health."
                    .to_string(),
            ),
            ..Default::default()
        }
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<ListResourceTemplatesResult, rmcp::ErrorData> {
        let template = RawResourceTemplate {
            uri_template: "bert://{+path}".to_string(),
            name: "docbert-document".to_string(),
            title: Some("docbert document".to_string()),
            description: Some(
                "A document from your docbert index. Use search tools to discover documents."
                    .to_string(),
            ),
            mime_type: Some("text/markdown".to_string()),
            icons: None,
        }
        .no_annotation();

        Ok(ListResourceTemplatesResult {
            meta: None,
            next_cursor: None,
            resource_templates: vec![template],
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<ReadResourceResult, rmcp::ErrorData> {
        let contents =
            read_resource_contents(&self.state.config_db, &request.uri)?;
        Ok(ReadResourceResult {
            contents: vec![contents],
        })
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

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GetParams {
    /// Document reference: collection:path, #doc_id, or path.
    pub reference: String,
    /// Start from this line number (1-indexed).
    pub from_line: Option<usize>,
    /// Maximum number of lines to return.
    pub max_lines: Option<usize>,
    /// Maximum bytes to read before skipping.
    pub max_bytes: Option<u64>,
    /// Include line numbers in the output.
    pub line_numbers: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MultiGetParams {
    /// Glob pattern to match relative paths.
    pub pattern: String,
    /// Restrict to a specific collection.
    pub collection: Option<String>,
    /// Maximum lines per file.
    pub max_lines: Option<usize>,
    /// Maximum bytes per file (default: 10240).
    pub max_bytes: Option<u64>,
    /// Include line numbers in output.
    pub line_numbers: Option<bool>,
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

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StatusResponse {
    data_dir: String,
    model: String,
    documents: usize,
    collections: Vec<StatusCollection>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StatusCollection {
    name: String,
    path: String,
    documents: usize,
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

fn format_status_summary(
    data_dir: &str,
    model: &str,
    documents: usize,
    collections: &[StatusCollection],
) -> String {
    let mut lines = Vec::new();
    lines.push("Docbert index status:".to_string());
    lines.push(format!("  Data dir: {data_dir}"));
    lines.push(format!("  Model: {model}"));
    lines.push(format!("  Documents: {documents}"));
    lines.push(format!("  Collections: {}", collections.len()));
    for collection in collections {
        lines.push(format!(
            "    - {} ({} docs) {}",
            collection.name, collection.documents, collection.path
        ));
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

fn apply_line_limits(
    text: &str,
    start_line: usize,
    max_lines: Option<usize>,
) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    let start_idx = start_line.saturating_sub(1).min(lines.len());
    if start_idx >= lines.len() {
        return String::new();
    }

    let end_idx = match max_lines {
        Some(max) => (start_idx + max).min(lines.len()),
        None => lines.len(),
    };

    let mut slice = lines[start_idx..end_idx].join("\n");
    if max_lines.is_some() && end_idx < lines.len() {
        slice.push_str(&format!(
            "\n\n[... truncated {} more lines]",
            lines.len() - end_idx
        ));
    }

    slice
}

fn resolve_reference(
    config_db: &ConfigDb,
    reference: &str,
) -> Result<(String, String), rmcp::ErrorData> {
    if let Some(short_id) = reference.strip_prefix('#') {
        return resolve_by_doc_id(config_db, short_id).ok_or_else(|| {
            rmcp::ErrorData::resource_not_found(
                format!("document not found: #{short_id}"),
                None,
            )
        });
    }

    if let Some((collection, path)) = reference.split_once(':') {
        return Ok((collection.to_string(), path.to_string()));
    }

    resolve_by_path(config_db, reference).ok_or_else(|| {
        rmcp::ErrorData::resource_not_found(
            format!("document not found: {reference}"),
            None,
        )
    })
}

fn resolve_by_doc_id(
    config_db: &ConfigDb,
    short_id: &str,
) -> Option<(String, String)> {
    let entries = config_db.list_all_document_metadata().ok()?;
    for (_doc_id, bytes) in entries {
        let meta = DocumentMetadata::deserialize(&bytes)?;
        let did = DocumentId::new(&meta.collection, &meta.relative_path);
        if did.short == short_id || did.to_string().contains(short_id) {
            return Some((meta.collection, meta.relative_path));
        }
    }
    None
}

fn resolve_by_path(
    config_db: &ConfigDb,
    path: &str,
) -> Option<(String, String)> {
    let entries = config_db.list_all_document_metadata().ok()?;
    for (_doc_id, bytes) in entries {
        let meta = DocumentMetadata::deserialize(&bytes)?;
        if meta.relative_path == path {
            return Some((meta.collection, meta.relative_path));
        }
    }
    None
}

fn encode_bert_path(path: &str) -> String {
    path.split('/')
        .map(|segment| {
            utf8_percent_encode(segment, NON_ALPHANUMERIC).to_string()
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn decode_bert_path(path: &str) -> String {
    path.split('/')
        .map(|segment| {
            percent_decode_str(segment).decode_utf8_lossy().to_string()
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn parse_bert_uri(uri: &str) -> Result<(String, String), rmcp::ErrorData> {
    let stripped = uri.strip_prefix("bert://").ok_or_else(|| {
        rmcp::ErrorData::resource_not_found(
            format!("unsupported uri: {uri}"),
            None,
        )
    })?;

    let decoded = decode_bert_path(stripped);
    let mut parts = decoded.split('/');
    let collection = parts.next().unwrap_or("");
    let path = parts.collect::<Vec<_>>().join("/");

    if collection.is_empty() || path.is_empty() {
        return Err(rmcp::ErrorData::resource_not_found(
            format!("invalid bert uri: {uri}"),
            None,
        ));
    }

    Ok((collection.to_string(), path))
}

fn read_resource_contents(
    config_db: &ConfigDb,
    uri: &str,
) -> Result<ResourceContents, rmcp::ErrorData> {
    let (collection, path) = parse_bert_uri(uri)?;
    let full_path = resolve_full_path(config_db, &collection, &path)
        .ok_or_else(|| {
            rmcp::ErrorData::resource_not_found(
                format!("collection not found: {collection}"),
                None,
            )
        })?;

    let mut text = std::fs::read_to_string(&full_path)
        .map_err(|e| mcp_error("failed to read resource", e))?;
    text = add_line_numbers(&text, 1);

    if let Some(context) = context_for_doc(config_db, &collection, &path) {
        text = format!("<!-- Context: {context} -->\n\n{text}");
    }

    Ok(ResourceContents::TextResourceContents {
        uri: uri.to_string(),
        mime_type: Some("text/markdown".to_string()),
        text,
        meta: None,
    })
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

pub fn run_mcp(
    data_dir: DataDir,
    config_db: ConfigDb,
    model_id: String,
) -> error::Result<()> {
    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;

    let state = DocbertState {
        data_dir,
        config_db,
        search_index,
        embedding_db,
        model: Mutex::new(ModelManager::with_model_id(model_id)),
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

    fn build_server(
        files: &[(&str, &str)],
    ) -> (DocbertMcpServer, tempfile::TempDir, Vec<DocumentId>) {
        let tmp = tempfile::tempdir().unwrap();
        let collection_dir = tmp.path().join("notes");
        std::fs::create_dir_all(&collection_dir).unwrap();

        let data_dir = DataDir::resolve(Some(tmp.path())).unwrap();
        let config_db = ConfigDb::open(&data_dir.config_db()).unwrap();
        config_db
            .set_collection("notes", collection_dir.to_str().unwrap())
            .unwrap();
        config_db
            .set_context("bert://notes", "Personal notes")
            .unwrap();

        let search_index = SearchIndex::open_in_ram().unwrap();
        let embedding_db =
            EmbeddingDb::open(&data_dir.embeddings_db()).unwrap();

        let mut writer = search_index.writer(15_000_000).unwrap();
        let mut doc_ids = Vec::new();

        for (path, body) in files {
            let file_path = collection_dir.join(path);
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&file_path, body).unwrap();

            let doc_id = DocumentId::new("notes", path);
            search_index
                .add_document(
                    &writer,
                    &doc_id.short,
                    doc_id.numeric,
                    "notes",
                    path,
                    path,
                    body,
                    1,
                )
                .unwrap();
            config_db
                .set_document_metadata(
                    doc_id.numeric,
                    &DocumentMetadata {
                        collection: "notes".to_string(),
                        relative_path: path.to_string(),
                        mtime: 1,
                    }
                    .serialize(),
                )
                .unwrap();
            doc_ids.push(doc_id);
        }

        writer.commit().unwrap();

        let server = DocbertMcpServer::new(DocbertState {
            data_dir,
            config_db,
            search_index,
            embedding_db,
            model: Mutex::new(ModelManager::new()),
        });

        (server, tmp, doc_ids)
    }

    #[tokio::test]
    async fn search_tool_returns_structured_results() {
        let (server, _tmp, _doc_ids) = build_server(&[(
            "rust.md",
            "Rust is fast.\nOwnership keeps memory safe.\n",
        )]);

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

    #[tokio::test]
    async fn get_tool_returns_resource_with_line_numbers() {
        let (server, _tmp, doc_ids) =
            build_server(&[("rust.md", "Rust is fast.\nLine two.\n")]);
        let doc_id = doc_ids.first().unwrap();

        let params = GetParams {
            reference: format!("#{}", doc_id.short),
            from_line: Some(1),
            max_lines: Some(1),
            max_bytes: None,
            line_numbers: Some(true),
        };

        let result = server.docbert_get(Parameters(params)).await.unwrap();
        let resource = result
            .content
            .first()
            .and_then(|c| c.as_resource())
            .expect("resource content");

        match &resource.resource {
            ResourceContents::TextResourceContents { text, .. } => {
                assert!(text.contains("1: Rust is fast."));
                assert!(text.contains("truncated"));
            }
            _ => panic!("expected text resource"),
        }
    }

    #[tokio::test]
    async fn multi_get_skips_large_files() {
        let (server, _tmp, _doc_ids) = build_server(&[
            ("small.md", "tiny\n"),
            ("large.md", "this is a larger file\nwith more content\n"),
        ]);

        let params = MultiGetParams {
            pattern: "*.md".to_string(),
            collection: Some("notes".to_string()),
            max_lines: None,
            max_bytes: Some(10),
            line_numbers: None,
        };

        let result =
            server.docbert_multi_get(Parameters(params)).await.unwrap();

        let mut saw_resource = false;
        let mut saw_skip = false;

        for item in &result.content {
            if item.as_resource().is_some() {
                saw_resource = true;
            }
            if let Some(text) = item.as_text()
                && text.text.contains("SKIPPED")
            {
                saw_skip = true;
            }
        }

        assert!(saw_resource);
        assert!(saw_skip);
    }

    #[test]
    fn read_resource_decodes_uri() {
        let (server, _tmp, _doc_ids) =
            build_server(&[("space name.md", "Hello world\n")]);
        let uri = "bert://notes/space%20name.md";
        let contents =
            read_resource_contents(&server.state.config_db, uri).unwrap();
        match contents {
            ResourceContents::TextResourceContents { text, .. } => {
                assert!(text.contains("1: Hello world"));
            }
            _ => panic!("expected text resource"),
        }
    }

    #[tokio::test]
    async fn status_tool_returns_structured_content() {
        let (server, _tmp, _doc_ids) = build_server(&[("rust.md", "Rust\n")]);

        let result = server.docbert_status().await.unwrap();
        let structured = result.structured_content.expect("structured");

        assert_eq!(
            structured.get("documents").and_then(|v| v.as_u64()),
            Some(1)
        );
        let collections = structured
            .get("collections")
            .and_then(|v| v.as_array())
            .expect("collections array");
        assert_eq!(collections.len(), 1);
    }

    #[test]
    fn prompt_router_includes_query_guide() {
        let (server, _tmp, _doc_ids) = build_server(&[("rust.md", "Rust\n")]);
        let prompts = server.prompt_router.list_all();
        assert!(prompts.iter().any(|p| p.name == "docbert_query"));
    }
}
