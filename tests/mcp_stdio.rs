use std::path::{Path, PathBuf};

use redb::{Database, TableDefinition};
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
use tantivy::{
    Index,
    IndexSettings,
    doc,
    schema::{
        FAST,
        IndexRecordOption,
        STORED,
        STRING,
        Schema,
        TextFieldIndexing,
        TextOptions,
    },
    tokenizer::{
        LowerCaser,
        RemoveLongFilter,
        SimpleTokenizer,
        Stemmer,
        TextAnalyzer,
    },
};

const COLLECTIONS: TableDefinition<&str, &str> =
    TableDefinition::new("collections");
const CONTEXTS: TableDefinition<&str, &str> = TableDefinition::new("contexts");
const DOCUMENT_METADATA: TableDefinition<u64, &[u8]> =
    TableDefinition::new("document_metadata");
const SETTINGS: TableDefinition<&str, &str> = TableDefinition::new("settings");

fn setup_config(
    data_dir: &Path,
    collection_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::create(data_dir.join("config.db"))?;
    let txn = db.begin_write()?;
    {
        let mut table = txn.open_table(COLLECTIONS)?;
        table.insert("notes", collection_path.to_str().unwrap())?;
    }
    {
        let mut table = txn.open_table(CONTEXTS)?;
        table.insert("bert://notes", "Test notes")?;
    }
    txn.open_table(DOCUMENT_METADATA)?;
    txn.open_table(SETTINGS)?;
    txn.commit()?;
    Ok(())
}

fn setup_fixture(data_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let collection_dir = data_dir.join("notes");
    std::fs::create_dir_all(&collection_dir)?;
    let file_path = collection_dir.join("hello.md");
    std::fs::write(&file_path, "Hello world\nSecond line\n")?;
    setup_config(data_dir, &collection_dir)?;
    build_index(data_dir)?;
    Ok(())
}

fn build_index(data_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let tantivy_dir = data_dir.join("tantivy");
    std::fs::create_dir_all(&tantivy_dir)?;

    let mut builder = Schema::builder();
    let doc_id = builder.add_text_field("doc_id", STRING | STORED);
    let doc_num_id = builder.add_u64_field("doc_num_id", STORED | FAST);
    let collection =
        builder.add_text_field("collection", STRING | STORED | FAST);
    let path = builder.add_text_field("path", STRING | STORED);

    let title_opts = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("en_stem")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();
    let title = builder.add_text_field("title", title_opts);

    let body_opts = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("en_stem")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );
    let body = builder.add_text_field("body", body_opts);

    let mtime = builder.add_u64_field("mtime", STORED | FAST);

    let schema = builder.build();
    let mmap_dir = tantivy::directory::MmapDirectory::open(&tantivy_dir)?;
    let index = Index::create(mmap_dir, schema, IndexSettings::default())?;

    let en_stem = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(40))
        .filter(LowerCaser)
        .filter(Stemmer::new(tantivy::tokenizer::Language::English))
        .build();
    index.tokenizers().register("en_stem", en_stem);

    let mut writer = index.writer(15_000_000)?;
    writer.add_document(doc!(
        doc_id => "testid",
        doc_num_id => 1u64,
        collection => "notes",
        path => "hello.md",
        title => "Hello",
        body => "Hello world",
        mtime => 1u64,
    ))?;
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
