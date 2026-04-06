use std::{
    io::Read,
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
    let mut child = std::process::Command::new(docbert_bin()?)
        .arg("--data-dir")
        .arg(tempdir.path())
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
        .spawn()?;

    wait_for_server(&mut child, port)?;

    child.kill()?;
    child.wait()?;

    Ok(())
}

fn free_tcp_port() -> Result<u16, Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
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
