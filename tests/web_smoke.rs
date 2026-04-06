use std::path::PathBuf;

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
