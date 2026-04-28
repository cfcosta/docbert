//! Minimal MCP server over JSON-RPC stdio.
//!
//! Implements the four tools described in the design (`rustbert_search`,
//! `rustbert_get`, `rustbert_list`, `rustbert_status`) plus the standard
//! MCP `initialize` / `tools/list` / `tools/call` lifecycle.
//!
//! Hand-rolled rather than going through rmcp because the surface is
//! small and the standalone JSON shape matches what every MCP client
//! expects on stdio. If a richer feature set is needed later (resource
//! templates, prompts, sampling), swapping for rmcp is mechanical.
//!
//! Request handling is per-line JSON-RPC 2.0:
//!
//! ```jsonc
//! { "jsonrpc": "2.0", "id": 1, "method": "tools/list" }
//! { "jsonrpc": "2.0", "id": 2, "method": "tools/call",
//!   "params": { "name": "rustbert_search", "arguments": {"crate":"serde", "query":"Serializer"}}}
//! ```

use std::io::{BufRead, Write};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    cache::CrateCache,
    crate_ref::CrateRef,
    crates_io::CratesIoApi,
    error::Result,
    fetcher::Fetcher,
    ingestion::{self, IngestionOptions},
    item::{RustItem, RustItemKind},
    lookup::{self, ListOptions},
    reqwest_fetcher::ReqwestFetcher,
};

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "rustbert";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Deserialize)]
struct Request {
    #[serde(default)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct Response {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Debug, Serialize)]
struct RpcError {
    code: i32,
    message: String,
}

/// Run the MCP server on stdio until EOF on stdin.
pub async fn serve(cache: CrateCache) -> Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Some(response) = handle_line(&line, &cache).await else {
            continue;
        };
        let bytes = serde_json::to_vec(&response).map_err(|e| {
            crate::error::Error::Cache(format!("mcp encode: {e}"))
        })?;
        stdout.write_all(&bytes)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}

async fn handle_line(line: &str, cache: &CrateCache) -> Option<Response> {
    let request: Request = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(e) => {
            return Some(Response {
                jsonrpc: "2.0",
                id: Value::Null,
                result: None,
                error: Some(RpcError {
                    code: -32700,
                    message: format!("parse error: {e}"),
                }),
            });
        }
    };

    // JSON-RPC 2.0: a request without `id` is a notification — never reply.
    let id = request.id.clone()?;
    if request.jsonrpc != "2.0" {
        return Some(error_response(id, -32600, "invalid jsonrpc version"));
    }

    Some(match dispatch(&request, cache).await {
        Ok(value) => Response {
            jsonrpc: "2.0",
            id,
            result: Some(value),
            error: None,
        },
        Err((code, msg)) => error_response(id, code, &msg),
    })
}

fn error_response(id: Value, code: i32, msg: &str) -> Response {
    Response {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(RpcError {
            code,
            message: msg.to_string(),
        }),
    }
}

async fn dispatch(
    request: &Request,
    cache: &CrateCache,
) -> std::result::Result<Value, (i32, String)> {
    match request.method.as_str() {
        "initialize" => Ok(initialize_result()),
        "tools/list" => Ok(tools_list_result()),
        "tools/call" => tools_call(request, cache).await,
        _ => Err((-32601, format!("method not found: {}", request.method))),
    }
}

fn initialize_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": { "tools": {} },
        "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION },
    })
}

fn tools_list_result() -> Value {
    json!({ "tools": tool_definitions() })
}

fn tool_definitions() -> Vec<Value> {
    vec![
        tool_def(
            "rustbert_search",
            "Search a Rust crate's items by query.",
            json!({
                "type": "object",
                "properties": {
                    "crate": { "type": "string" },
                    "version": { "type": "string", "default": "latest" },
                    "query": { "type": "string" },
                    "kind": { "type": "string" },
                    "module_prefix": { "type": "string" },
                    "limit": { "type": "integer", "default": 10 }
                },
                "required": ["crate", "query"]
            }),
        ),
        tool_def(
            "rustbert_get",
            "Fetch one item from a Rust crate by qualified path.",
            json!({
                "type": "object",
                "properties": {
                    "crate": { "type": "string" },
                    "version": { "type": "string", "default": "latest" },
                    "path": { "type": "string" }
                },
                "required": ["crate", "path"]
            }),
        ),
        tool_def(
            "rustbert_list",
            "List items in a Rust crate.",
            json!({
                "type": "object",
                "properties": {
                    "crate": { "type": "string" },
                    "version": { "type": "string", "default": "latest" },
                    "kind": { "type": "string" },
                    "module_prefix": { "type": "string" },
                    "limit": { "type": "integer", "default": 50 }
                },
                "required": ["crate"]
            }),
        ),
        tool_def(
            "rustbert_status",
            "Report cache state for a crate.",
            json!({
                "type": "object",
                "properties": { "crate": { "type": "string" } }
            }),
        ),
    ]
}

fn tool_def(name: &str, description: &str, schema: Value) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": schema,
    })
}

async fn tools_call(
    request: &Request,
    cache: &CrateCache,
) -> std::result::Result<Value, (i32, String)> {
    let name = request
        .params
        .get("name")
        .and_then(Value::as_str)
        .ok_or((-32602, "missing tool name".to_string()))?;
    let args = request
        .params
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Null);

    let text = match name {
        "rustbert_search" => tool_search(&args, cache).await,
        "rustbert_get" => tool_get(&args, cache).await,
        "rustbert_list" => tool_list(&args, cache).await,
        "rustbert_status" => tool_status(&args, cache),
        other => Err((-32601, format!("unknown tool: {other}"))),
    }?;

    Ok(json!({
        "content": [ { "type": "text", "text": text } ]
    }))
}

fn parse_crate_ref(
    args: &Value,
) -> std::result::Result<CrateRef, (i32, String)> {
    let name = args
        .get("crate")
        .and_then(Value::as_str)
        .ok_or((-32602, "missing `crate` argument".to_string()))?;
    let version = args
        .get("version")
        .and_then(Value::as_str)
        .unwrap_or("latest");
    let spec = if version == "latest" {
        name.to_string()
    } else {
        format!("{name}@{version}")
    };
    CrateRef::parse(&spec).map_err(|e| (-32602, e.to_string()))
}

async fn ensure_cached<F: Fetcher + Clone>(
    cache: &CrateCache,
    indexer: &mut crate::indexer::Indexer,
    fetcher: &F,
    api: &CratesIoApi<F>,
    crate_ref: &CrateRef,
) -> std::result::Result<crate::collection::SyntheticCollection, (i32, String)>
{
    let report = ingestion::ingest(
        fetcher,
        api,
        cache,
        indexer,
        crate_ref,
        IngestionOptions::default(),
    )
    .await
    .map_err(|e| (-32000, e.to_string()))?;
    Ok(report.collection().clone())
}

async fn tool_search(
    args: &Value,
    cache: &CrateCache,
) -> std::result::Result<String, (i32, String)> {
    let crate_ref = parse_crate_ref(args)?;
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .ok_or((-32602, "missing `query`".to_string()))?;
    let kind_filter = args
        .get("kind")
        .and_then(Value::as_str)
        .and_then(RustItemKind::parse);
    let module_filter: Option<String> = args
        .get("module_prefix")
        .and_then(Value::as_str)
        .map(String::from);
    let limit: usize = args
        .get("limit")
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .unwrap_or(10);

    let fetcher = ReqwestFetcher::new().map_err(|e| (-32000, e.to_string()))?;
    let api = CratesIoApi::new(fetcher.clone());
    let mut indexer = crate::indexer::Indexer::open(cache.data_dir())
        .map_err(|e| (-32000, e.to_string()))?;

    // ensure_cached calls `ingest()`, which fetches + parses +
    // indexes + embeds + rebuilds PLAID on the first hit for any
    // (crate, version) not already in the indexer. By the time we
    // get past this line, the crate is fully searchable.
    let coll =
        ensure_cached(cache, &mut indexer, &fetcher, &api, &crate_ref).await?;

    // Hybrid (BM25 + ColBERT/PLAID) search via the docbert-core
    // stack — the same path the CLI takes. Kind / module filters
    // apply post-rank against the cached items.
    let params = docbert_core::search::SearchParams {
        query: query.to_string(),
        count: limit * 4, // overfetch for post-filter headroom
        collection: Some(coll.to_string()),
        min_score: 0.0,
        bm25_only: false,
        no_fuzzy: false,
        all: false,
    };
    let results = indexer
        .search(params)
        .map_err(|e| (-32000, e.to_string()))?;
    let items = cache.load(&coll).map_err(|e| (-32000, e.to_string()))?;
    Ok(format_hybrid_hits(
        &coll,
        &items,
        &results,
        kind_filter,
        module_filter.as_deref(),
        limit,
    ))
}

fn format_hybrid_hits(
    coll: &crate::collection::SyntheticCollection,
    items: &[RustItem],
    results: &[docbert_core::search::FinalResult],
    kind_filter: Option<RustItemKind>,
    module_filter: Option<&str>,
    limit: usize,
) -> String {
    let mut out = String::new();
    let mut shown = 0usize;
    for r in results {
        let Some(item) = items.iter().find(|i| i.qualified_path == r.title)
        else {
            continue;
        };
        if let Some(k) = kind_filter
            && item.kind != k
        {
            continue;
        }
        if let Some(prefix) = module_filter
            && !item.qualified_path.starts_with(prefix)
        {
            continue;
        }
        out.push_str(&format!(
            "[{score:.3}] {}\n",
            format_item_one_line(item),
            score = r.score,
        ));
        shown += 1;
        if shown >= limit {
            break;
        }
    }
    if shown == 0 {
        return format!("(no matches in {}@{})", coll.crate_name, coll.version);
    }
    format!(
        "{shown} matches in {}@{}\n\n{out}",
        coll.crate_name, coll.version
    )
}

async fn tool_get(
    args: &Value,
    cache: &CrateCache,
) -> std::result::Result<String, (i32, String)> {
    let crate_ref = parse_crate_ref(args)?;
    let path = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or((-32602, "missing `path`".to_string()))?;
    let fetcher = ReqwestFetcher::new().map_err(|e| (-32000, e.to_string()))?;
    let api = CratesIoApi::new(fetcher.clone());
    let mut indexer = crate::indexer::Indexer::open(cache.data_dir())
        .map_err(|e| (-32000, e.to_string()))?;
    let coll =
        ensure_cached(cache, &mut indexer, &fetcher, &api, &crate_ref).await?;
    let items = cache.load(&coll).map_err(|e| (-32000, e.to_string()))?;
    match lookup::get(&items, path) {
        Some(item) => Ok(format_item_full(item)),
        None => Err((
            -32000,
            format!(
                "item `{path}` not found in {}@{}",
                coll.crate_name, coll.version
            ),
        )),
    }
}

async fn tool_list(
    args: &Value,
    cache: &CrateCache,
) -> std::result::Result<String, (i32, String)> {
    let crate_ref = parse_crate_ref(args)?;
    let opts = ListOptions {
        kind: args
            .get("kind")
            .and_then(Value::as_str)
            .and_then(RustItemKind::parse),
        module_prefix: args
            .get("module_prefix")
            .and_then(Value::as_str)
            .map(String::from),
        limit: args
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n as usize),
    };
    let fetcher = ReqwestFetcher::new().map_err(|e| (-32000, e.to_string()))?;
    let api = CratesIoApi::new(fetcher.clone());
    let mut indexer = crate::indexer::Indexer::open(cache.data_dir())
        .map_err(|e| (-32000, e.to_string()))?;
    let coll =
        ensure_cached(cache, &mut indexer, &fetcher, &api, &crate_ref).await?;
    let items = cache.load(&coll).map_err(|e| (-32000, e.to_string()))?;
    let listed = lookup::list(&items, &opts);
    let mut out = format!(
        "{count} items in {name}@{version}\n",
        count = listed.len(),
        name = coll.crate_name,
        version = coll.version,
    );
    for item in listed {
        out.push_str(&format_item_one_line(item));
        out.push('\n');
    }
    Ok(out)
}

fn tool_status(
    args: &Value,
    cache: &CrateCache,
) -> std::result::Result<String, (i32, String)> {
    let filter = args.get("crate").and_then(Value::as_str);
    let entries = cache.entries().map_err(|e| (-32000, e.to_string()))?;
    let filtered: Vec<_> = match filter {
        Some(name) => entries
            .into_iter()
            .filter(|e| e.crate_name == name)
            .collect(),
        None => entries,
    };
    if filtered.is_empty() {
        return Ok("(no cached crates)".to_string());
    }
    let mut out = String::new();
    for e in filtered {
        out.push_str(&format!(
            "{name}@{version}\t{count} items\tfetched_at={ts}\n",
            name = e.crate_name,
            version = e.version,
            count = e.item_count,
            ts = e.fetched_at,
        ));
    }
    Ok(out)
}

fn format_item_one_line(item: &RustItem) -> String {
    format!(
        "{kind} {path}  ({file}:{start}-{end})",
        kind = item.kind.as_str(),
        path = item.qualified_path,
        file = item.source_file.display(),
        start = item.line_start,
        end = item.line_end,
    )
}

fn format_item_full(item: &RustItem) -> String {
    let mut out = format!(
        "{kind} {path}\n  {sig}\n  source: {file}:{start}-{end}\n  visibility: {vis}\n",
        kind = item.kind.as_str(),
        path = item.qualified_path,
        sig = item.signature,
        file = item.source_file.display(),
        start = item.line_start,
        end = item.line_end,
        vis = item.visibility.as_str(),
    );
    if !item.attrs.is_empty() {
        out.push_str(&format!("  attrs: {}\n", item.attrs.join(" ")));
    }
    if !item.doc_markdown.is_empty() {
        out.push('\n');
        for line in item.doc_markdown.lines() {
            out.push_str(&format!("    {line}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[tokio::test]
    async fn initialize_returns_protocol_version() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let line = r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#;
        let response = handle_line(line, &cache).await.unwrap();
        let result = response.result.unwrap();
        assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(result["serverInfo"]["name"], "rustbert");
    }

    #[tokio::test]
    async fn tools_list_returns_four_tools() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let line = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#;
        let response = handle_line(line, &cache).await.unwrap();
        let tools = &response.result.unwrap()["tools"];
        let names: Vec<_> = tools
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(names.len(), 4);
        assert!(names.contains(&"rustbert_search".to_string()));
        assert!(names.contains(&"rustbert_get".to_string()));
        assert!(names.contains(&"rustbert_list".to_string()));
        assert!(names.contains(&"rustbert_status".to_string()));
    }

    #[tokio::test]
    async fn unknown_method_returns_error() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let line = r#"{"jsonrpc":"2.0","id":1,"method":"nonsense"}"#;
        let response = handle_line(line, &cache).await.unwrap();
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn malformed_json_returns_parse_error() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let response = handle_line("not json", &cache).await.unwrap();
        assert_eq!(response.error.unwrap().code, -32700);
    }

    #[tokio::test]
    async fn invalid_jsonrpc_version_is_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let line = r#"{"jsonrpc":"1.0","id":1,"method":"tools/list"}"#;
        let response = handle_line(line, &cache).await.unwrap();
        assert_eq!(response.error.unwrap().code, -32600);
    }

    #[tokio::test]
    async fn status_tool_reports_empty_cache() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let line = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rustbert_status","arguments":{}}}"#;
        let response = handle_line(line, &cache).await.unwrap();
        let result = response.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("no cached crates"));
    }

    #[tokio::test]
    async fn unknown_tool_returns_error() {
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let line = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"unknown","arguments":{}}}"#;
        let response = handle_line(line, &cache).await.unwrap();
        assert!(response.error.is_some());
    }

    #[tokio::test]
    async fn notifications_are_silent() {
        // JSON-RPC 2.0: messages without an `id` are notifications and
        // MUST NOT be answered. Replying — even with an error — breaks
        // strict clients (Claude Code's MCP loop drops the connection).
        let tmp = TempDir::new().unwrap();
        let cache = CrateCache::new(tmp.path()).unwrap();
        let line = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        assert!(handle_line(line, &cache).await.is_none());
    }
}
