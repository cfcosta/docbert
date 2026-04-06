use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    process::Stdio,
    thread,
    time::{Duration, Instant},
};

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
fn web_settings_route_responds() -> Result<(), Box<dyn std::error::Error>> {
    let tempdir = tempfile::tempdir()?;
    let port = free_tcp_port()?;
    let mut child = spawn_web_server(tempdir.path(), port)?;

    wait_for_server(&mut child, port)?;

    let response = http_get(port, "/v1/settings/llm")?;
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "unexpected response: {response}");
    assert!(response.contains("{\"provider\":null,\"model\":null,\"api_key\":null}"));

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
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "unexpected response: {response}");
    assert!(response.contains("<!doctype html>"));
    assert!(response.contains("<div id=\"root\"></div>"));

    child.kill()?;
    child.wait()?;

    Ok(())
}

#[test]
fn web_spa_fallback_serves_index_html() -> Result<(), Box<dyn std::error::Error>> {
    let tempdir = tempfile::tempdir()?;
    let port = free_tcp_port()?;
    let mut child = spawn_web_server(tempdir.path(), port)?;

    wait_for_server(&mut child, port)?;

    let response = http_get(port, "/documents/notes/hello.md")?;
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "unexpected response: {response}");
    assert!(response.contains("<!doctype html>"));
    assert!(response.contains("<div id=\"root\"></div>"));

    child.kill()?;
    child.wait()?;

    Ok(())
}

#[test]
fn web_search_reads_excerpts_from_disk() -> Result<(), Box<dyn std::error::Error>> {
    let tempdir = tempfile::tempdir()?;
    let port = free_tcp_port()?;
    let mut child = spawn_web_server(tempdir.path(), port)?;

    wait_for_server(&mut child, port)?;

    let response = http_post_json(
        port,
        "/v1/search",
        r#"{"query":"rust","mode":"hybrid","count":10,"min_score":0.0}"#,
    )?;
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "unexpected response: {response}");
    assert!(response.contains("\"query\":\"rust\""));
    assert!(response.contains("\"result_count\":0"));
    assert!(response.contains("\"results\":[]"));

    child.kill()?;
    child.wait()?;

    Ok(())
}

fn spawn_web_server(
    data_dir: &std::path::Path,
    port: u16,
) -> Result<std::process::Child, Box<dyn std::error::Error>> {
    Ok(std::process::Command::new(docbert_bin()?)
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
        .stderr(Stdio::piped())
        .spawn()?)
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
