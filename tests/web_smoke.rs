use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::Stdio,
    thread,
    time::{Duration, Instant},
};

use docbert_core::ConfigDb;

#[test]
fn web_help_lists_flags() -> Result<(), Box<dyn std::error::Error>> {
    let output = std::process::Command::new(docbert_bin()?)
        .arg("web")
        .arg("--help")
        .output()?;

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("Start the web UI server"));
    assert!(stdout.contains("--host"));
    assert!(stdout.contains("--port"));

    Ok(())
}

#[test]
fn web_boot_starts_without_docserver_env()
-> Result<(), Box<dyn std::error::Error>> {
    let tempdir = tempfile::tempdir()?;
    let port = free_tcp_port()?;
    let mut child = spawn_web_server(tempdir.path(), port)?;

    wait_for_server(&mut child, port)?;

    child.kill()?;
    child.wait()?;

    Ok(())
}

#[test]
fn web_server_allows_sync_while_running()
-> Result<(), Box<dyn std::error::Error>> {
    let tempdir = tempfile::tempdir()?;
    let port = free_tcp_port()?;
    let mut child = spawn_web_server(tempdir.path(), port)?;

    wait_for_server(&mut child, port)?;

    let sync = run_sync(tempdir.path())?;
    assert!(
        sync.status.success(),
        "docbert sync failed while web was running\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&sync.stdout),
        String::from_utf8_lossy(&sync.stderr)
    );
    assert!(
        String::from_utf8_lossy(&sync.stderr)
            .contains("No collections to sync.")
    );

    let response = http_get(port, "/v1/settings/llm")?;
    assert!(
        response.starts_with("HTTP/1.1 200 OK\r\n"),
        "unexpected response after sync: {response}"
    );

    child.kill()?;
    child.wait()?;

    Ok(())
}

#[test]
fn web_settings_route_responds() -> Result<(), Box<dyn std::error::Error>> {
    let tempdir = tempfile::tempdir()?;
    let port = free_tcp_port()?;
    let mut child = spawn_web_server(tempdir.path(), port)?;

    wait_for_server(&mut child, port)?;

    let response = http_get(port, "/v1/settings/llm")?;
    assert!(
        response.starts_with("HTTP/1.1 200 OK\r\n"),
        "unexpected response: {response}"
    );
    assert!(response.contains(
        "{\"provider\":null,\"model\":null,\"api_key\":null,\"oauth_connected\":false}"
    ));

    child.kill()?;
    child.wait()?;

    Ok(())
}

#[test]
fn web_root_serves_index_html() -> Result<(), Box<dyn std::error::Error>> {
    let tempdir = tempfile::tempdir()?;
    let port = free_tcp_port()?;
    let mut child = spawn_web_server(tempdir.path(), port)?;

    wait_for_server(&mut child, port)?;

    let response = http_get(port, "/")?;
    assert!(
        response.starts_with("HTTP/1.1 200 OK\r\n"),
        "unexpected response: {response}"
    );
    assert!(response.contains("<!doctype html>"));
    assert!(response.contains("<div id=\"root\"></div>"));

    child.kill()?;
    child.wait()?;

    Ok(())
}

#[test]
fn web_spa_fallback_serves_index_html() -> Result<(), Box<dyn std::error::Error>>
{
    let tempdir = tempfile::tempdir()?;
    let port = free_tcp_port()?;
    let mut child = spawn_web_server(tempdir.path(), port)?;

    wait_for_server(&mut child, port)?;

    let response = http_get(port, "/documents/notes/hello.md")?;
    assert!(
        response.starts_with("HTTP/1.1 200 OK\r\n"),
        "unexpected response: {response}"
    );
    assert!(response.contains("<!doctype html>"));
    assert!(response.contains("<div id=\"root\"></div>"));

    child.kill()?;
    child.wait()?;

    Ok(())
}

#[test]
fn web_search_reads_excerpts_from_disk()
-> Result<(), Box<dyn std::error::Error>> {
    let tempdir = tempfile::tempdir()?;
    let port = free_tcp_port()?;
    let mut child = spawn_web_server(tempdir.path(), port)?;

    wait_for_server(&mut child, port)?;

    let response = http_post_json(
        port,
        "/v1/search",
        r#"{"query":"rust","mode":"hybrid","count":10,"min_score":0.0}"#,
    )?;
    assert!(
        response.starts_with("HTTP/1.1 200 OK\r\n"),
        "unexpected response: {response}"
    );
    assert!(response.contains("\"query\":\"rust\""));
    assert!(response.contains("\"result_count\":0"));
    assert!(response.contains("\"results\":[]"));

    child.kill()?;
    child.wait()?;

    Ok(())
}

#[test]
fn web_upload_then_get_then_search() -> Result<(), Box<dyn std::error::Error>> {
    let tempdir = tempfile::tempdir()?;
    let collection_root = setup_collection(tempdir.path(), "notes")?;
    let port = free_tcp_port()?;
    let mut child =
        spawn_web_server_with_test_embeddings(tempdir.path(), port)?;

    wait_for_server(&mut child, port)?;

    let upload = http_post_json(
        port,
        "/v1/documents",
        r##"{"collection":"notes","documents":[{"path":"nested/uploaded.md","content":"# Uploaded\n\nBody on disk","content_type":"text/markdown"}]}"##,
    )?;
    assert!(
        upload.starts_with("HTTP/1.1 200 OK\r\n"),
        "unexpected response: {upload}"
    );
    assert!(upload.contains("\"ingested\":1"));
    assert!(
        std::fs::read_to_string(collection_root.join("nested/uploaded.md"))?
            .contains("Body on disk")
    );

    let get = http_get(port, "/v1/documents/notes/nested/uploaded.md")?;
    assert!(
        get.starts_with("HTTP/1.1 200 OK\r\n"),
        "unexpected response: {get}"
    );
    assert!(get.contains("\"path\":\"nested/uploaded.md\""));
    assert!(get.contains("Body on disk"));

    let search = http_post_json(
        port,
        "/v1/search",
        r#"{"query":"uploaded","mode":"hybrid","count":10,"min_score":0.0}"#,
    )?;
    assert!(
        search.starts_with("HTTP/1.1 200 OK\r\n"),
        "unexpected response: {search}"
    );
    assert!(search.contains("\"query\":\"uploaded\""));
    // Under RRF the uploaded doc surfaces in either the BM25 leg (exact
    // match on "uploaded") or the semantic leg (fake embeddings make every
    // indexed doc reachable), so result_count should be 1.
    assert!(search.contains("\"result_count\":1"));
    assert!(search.contains("nested/uploaded.md"));

    child.kill()?;
    child.wait()?;

    Ok(())
}

#[test]
fn web_delete_then_get_and_search_fail()
-> Result<(), Box<dyn std::error::Error>> {
    let tempdir = tempfile::tempdir()?;
    let collection_root = setup_collection(tempdir.path(), "notes")?;
    let port = free_tcp_port()?;
    let mut child =
        spawn_web_server_with_test_embeddings(tempdir.path(), port)?;

    wait_for_server(&mut child, port)?;

    let upload = http_post_json(
        port,
        "/v1/documents",
        r##"{"collection":"notes","documents":[{"path":"nested/uploaded.md","content":"# Uploaded\n\nBody on disk","content_type":"text/markdown"}]}"##,
    )?;
    assert!(
        upload.starts_with("HTTP/1.1 200 OK\r\n"),
        "unexpected response: {upload}"
    );
    assert!(collection_root.join("nested/uploaded.md").exists());

    let delete = http_delete(port, "/v1/documents/notes/nested/uploaded.md")?;
    assert!(
        delete.starts_with("HTTP/1.1 204 No Content\r\n"),
        "unexpected response: {delete}"
    );
    assert!(!collection_root.join("nested/uploaded.md").exists());

    let get = http_get(port, "/v1/documents/notes/nested/uploaded.md")?;
    assert!(
        get.starts_with("HTTP/1.1 404 Not Found\r\n"),
        "unexpected response: {get}"
    );

    let search = http_post_json(
        port,
        "/v1/search",
        r#"{"query":"uploaded","mode":"hybrid","count":10,"min_score":0.0}"#,
    )?;
    assert!(
        search.starts_with("HTTP/1.1 200 OK\r\n"),
        "unexpected response: {search}"
    );
    assert!(search.contains("\"result_count\":0"));

    child.kill()?;
    child.wait()?;

    Ok(())
}

fn spawn_web_server(
    data_dir: &Path,
    port: u16,
) -> Result<std::process::Child, Box<dyn std::error::Error>> {
    spawn_web_server_with(data_dir, port, false)
}

fn spawn_web_server_with_test_embeddings(
    data_dir: &Path,
    port: u16,
) -> Result<std::process::Child, Box<dyn std::error::Error>> {
    spawn_web_server_with(data_dir, port, true)
}

fn spawn_web_server_with(
    data_dir: &Path,
    port: u16,
    fake_embeddings: bool,
) -> Result<std::process::Child, Box<dyn std::error::Error>> {
    let mut command = std::process::Command::new(docbert_bin()?);
    command
        .arg("--data-dir")
        .arg(data_dir)
        .arg("web")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .env_remove("DOCSERVER_DATA_DIR")
        .env_remove("DOCSERVER_HOST")
        .env_remove("DOCSERVER_PORT")
        .env_remove("DOCSERVER_MODEL")
        .env_remove("DOCSERVER_LOG")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if fake_embeddings {
        command.env("DOCBERT_WEB_TEST_FAKE_EMBEDDINGS", "1");
    }

    Ok(command.spawn()?)
}

fn run_sync(
    data_dir: &Path,
) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    Ok(std::process::Command::new(docbert_bin()?)
        .arg("--data-dir")
        .arg(data_dir)
        .arg("sync")
        .output()?)
}

fn setup_collection(
    data_dir: &Path,
    name: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let root = data_dir.join(name);
    std::fs::create_dir_all(&root)?;
    let config_db = ConfigDb::open(&data_dir.join("config.db"))?;
    config_db.set_collection(name, root.to_str().unwrap())?;
    drop(config_db);
    Ok(root)
}

fn free_tcp_port() -> Result<u16, Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

fn http_get(
    port: u16,
    path: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    write!(
        stream,
        "GET {} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n\r\n",
        path, port
    )?;
    stream.flush()?;

    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

fn http_post_json(
    port: u16,
    path: &str,
    body: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    write!(
        stream,
        "POST {} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        path,
        port,
        body.len(),
        body,
    )?;
    stream.flush()?;

    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

fn http_delete(
    port: u16,
    path: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    write!(
        stream,
        "DELETE {} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n\r\n",
        path, port
    )?;
    stream.flush()?;

    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

fn wait_for_server(
    child: &mut std::process::Child,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let deadline = Instant::now() + Duration::from_secs(10);
    let addr = format!("127.0.0.1:{port}");

    loop {
        if TcpStream::connect(&addr).is_ok() {
            return Ok(());
        }

        if let Some(_status) = child.try_wait()? {
            let mut stdout = String::new();
            if let Some(mut handle) = child.stdout.take() {
                let _ = handle.read_to_string(&mut stdout);
            }
            let mut stderr = String::new();
            if let Some(mut handle) = child.stderr.take() {
                let _ = handle.read_to_string(&mut stderr);
            }
            return Err(format!(
                "docbert web exited before accepting connections\nstdout:\n{}\nstderr:\n{}",
                stdout, stderr
            )
            .into());
        }

        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "docbert web did not accept connections on {addr} before timeout"
            )
            .into());
        }

        thread::sleep(Duration::from_millis(50));
    }
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
