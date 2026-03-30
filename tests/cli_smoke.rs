use std::path::{Path, PathBuf};

use docbert_core::{ConfigDb, DocumentId, SearchIndex, incremental};
use serde_json::Value;

fn setup_fixture(data_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let collection_dir = data_dir.join("notes");
    std::fs::create_dir_all(&collection_dir)?;
    std::fs::write(
        collection_dir.join("hello.md"),
        "Hello world\nSecond line\n",
    )?;

    let config_db = ConfigDb::open(&data_dir.join("config.db"))?;
    config_db.set_collection("notes", collection_dir.to_str().unwrap())?;
    let did = DocumentId::new("notes", "hello.md");
    config_db.set_document_metadata_typed(
        did.numeric,
        &incremental::DocumentMetadata {
            collection: "notes".to_string(),
            relative_path: "hello.md".to_string(),
            mtime: 1,
        },
    )?;

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

#[test]
fn doctor_short_circuits_without_data_dir()
-> Result<(), Box<dyn std::error::Error>> {
    let tempdir = tempfile::tempdir()?;
    let invalid_data_dir = tempdir.path().join("not-a-directory");
    std::fs::write(&invalid_data_dir, "occupied by file")?;

    let output = std::process::Command::new(docbert_bin()?)
        .arg("--data-dir")
        .arg(&invalid_data_dir)
        .arg("doctor")
        .arg("--json")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let json: Value = serde_json::from_str(&stdout)?;
    assert!(json.get("selected_device").is_some());
    assert!(String::from_utf8(output.stderr)?.is_empty());

    Ok(())
}

#[test]
fn search_json_writes_results_to_stdout_and_keeps_stderr_clean_in_bm25_mode()
-> Result<(), Box<dyn std::error::Error>> {
    let tempdir = tempfile::tempdir()?;
    setup_fixture(tempdir.path())?;

    let output = std::process::Command::new(docbert_bin()?)
        .arg("--data-dir")
        .arg(tempdir.path())
        .arg("search")
        .arg("Hello")
        .arg("--json")
        .arg("--bm25-only")
        .arg("--no-fuzzy")
        .output()?;

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout)?;
    let json: Value = serde_json::from_str(&stdout)?;
    assert_eq!(json.get("query").and_then(Value::as_str), Some("Hello"));
    let results = json
        .get("results")
        .and_then(Value::as_array)
        .expect("results array");
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].get("collection").and_then(Value::as_str),
        Some("notes")
    );

    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");

    Ok(())
}

#[test]
fn get_json_accepts_hash_prefixed_and_bare_short_document_refs()
-> Result<(), Box<dyn std::error::Error>> {
    let tempdir = tempfile::tempdir()?;
    setup_fixture(tempdir.path())?;
    let did = DocumentId::new("notes", "hello.md");

    for reference in [did.to_string(), did.short.clone()] {
        let output = std::process::Command::new(docbert_bin()?)
            .arg("--data-dir")
            .arg(tempdir.path())
            .arg("get")
            .arg(&reference)
            .arg("--json")
            .output()?;

        assert!(output.status.success(), "reference {reference} failed");
        let stdout = String::from_utf8(output.stdout)?;
        let json: Value = serde_json::from_str(&stdout)?;
        assert_eq!(
            json.get("collection").and_then(Value::as_str),
            Some("notes")
        );
        assert_eq!(json.get("path").and_then(Value::as_str), Some("hello.md"));
    }

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
