use std::path::{Path, PathBuf};

use rmcp::{
    ServiceExt,
    model::CallToolRequestParams,
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
    build_index(tempdir.path())?;

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
