use std::path::{Path, PathBuf};

use docbert::{ConfigDb, SearchIndex};
use rmcp::{
    ServiceExt,
    model::{
        CallToolRequestParams,
        ReadResourceRequestParams,
        ResourceContents,
    },
    transport::{ConfigureCommandExt, TokioChildProcess},
};
use serde_json::json;

fn setup_config(
    config_db: &ConfigDb,
    collection_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    config_db.set_collection("notes", collection_path.to_str().unwrap())?;
    config_db.set_context("bert://notes", "Test notes")?;
    Ok(())
}

fn setup_fixture(data_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let collection_dir = data_dir.join("notes");
    std::fs::create_dir_all(&collection_dir)?;
    std::fs::write(
        collection_dir.join("hello.md"),
        "Hello world\nSecond line\n",
    )?;

    let config_db = ConfigDb::open(&data_dir.join("config.db"))?;
    setup_config(&config_db, &collection_dir)?;

    let tantivy_dir = data_dir.join("tantivy");
    std::fs::create_dir_all(&tantivy_dir)?;
    let index = SearchIndex::open(&tantivy_dir)?;
    let mut writer = index.writer(15_000_000)?;
    index.add_document(
        &writer,
        "testid",
        1,
        "notes",
        "hello.md",
        "Hello",
        "Hello world",
        1,
    )?;
    writer.commit()?;

    Ok(())
}

#[tokio::test]
async fn mcp_stdio_search_roundtrip() -> Result<(), Box<dyn std::error::Error>>
{
    let tempdir = tempfile::tempdir()?;
    setup_fixture(tempdir.path())?;

    let bin = docbert_bin()?;
    let transport = TokioChildProcess::new(
        tokio::process::Command::new(bin).configure(|cmd| {
            cmd.arg("mcp").env("DOCBERT_DATA_DIR", tempdir.path());
        }),
    )?;

    let client = ().serve(transport).await?;

    let args = json!({
        "query": "Hello",
        "limit": 5,
        "bm25Only": true,
        "noFuzzy": true,
        "includeSnippet": false
    });

    let result = client
        .peer()
        .call_tool(CallToolRequestParams {
            meta: None,
            name: "docbert_search".into(),
            arguments: Some(args.as_object().unwrap().clone()),
            task: None,
        })
        .await?;

    let structured = result.structured_content.expect("structured content");
    let results = structured
        .get("results")
        .and_then(|v| v.as_array())
        .expect("results array");

    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].get("collection").and_then(|v| v.as_str()),
        Some("notes")
    );

    let get_args = json!({
        "reference": "notes:hello.md",
        "lineNumbers": true,
        "maxLines": 1
    });
    let get_result = client
        .peer()
        .call_tool(CallToolRequestParams {
            meta: None,
            name: "docbert_get".into(),
            arguments: Some(get_args.as_object().unwrap().clone()),
            task: None,
        })
        .await?;
    let get_resource = get_result
        .content
        .iter()
        .find_map(|c| c.as_resource())
        .expect("docbert_get resource");
    match &get_resource.resource {
        ResourceContents::TextResourceContents { text, .. } => {
            assert!(text.contains("<!-- Context: Test notes -->"));
            assert!(text.contains("1: Hello world"));
        }
        _ => panic!("expected text resource"),
    }

    let resource_result = client
        .peer()
        .read_resource(ReadResourceRequestParams {
            meta: None,
            uri: "bert://notes/hello.md".to_string(),
        })
        .await?;
    let resource = resource_result.contents.first().expect("resource content");
    match resource {
        ResourceContents::TextResourceContents { text, .. } => {
            assert!(text.contains("<!-- Context: Test notes -->"));
            assert!(text.contains("1: Hello world"));
        }
        _ => panic!("expected text resource"),
    }

    client.cancel().await?;
    Ok(())
}

fn docbert_bin() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(bin) = std::env::var("CARGO_BIN_EXE_docbert") {
        return Ok(PathBuf::from(bin));
    }

    let mut path = std::env::current_exe()?;
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("docbert");

    if cfg!(windows) {
        path.set_extension("exe");
    }

    Ok(path)
}
