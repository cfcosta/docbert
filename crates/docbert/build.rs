use std::{
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let ui_dir = manifest_dir.join("ui");
    let dist_dir = ui_dir.join("dist");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=ui/src");
    println!("cargo:rerun-if-changed=ui/public");
    println!("cargo:rerun-if-changed=ui/index.html");
    println!("cargo:rerun-if-changed=ui/package.json");
    println!("cargo:rerun-if-changed=ui/bun.lock");
    println!("cargo:rerun-if-changed=ui/package-lock.json");
    println!("cargo:rerun-if-changed=ui/vite.config.ts");
    println!("cargo:rerun-if-changed=ui/tsconfig.json");
    println!("cargo:rerun-if-changed=ui/tsconfig.app.json");
    println!("cargo:rerun-if-changed=ui/tsconfig.node.json");
    println!("cargo:rerun-if-env-changed=DOCBERT_SKIP_UI_BUILD");

    if std::env::var_os("DOCBERT_SKIP_UI_BUILD").is_some() {
        ensure_dist_dir(&dist_dir);
        return;
    }

    // Once built, skip rebuilding. Developers iterating on the UI should use
    // `bun run dev` or delete `ui/dist/` to force a rebuild via cargo.
    if dist_dir.join("index.html").exists() {
        return;
    }

    if !ui_dir.join("package.json").exists() {
        ensure_dist_dir(&dist_dir);
        return;
    }

    if try_build_ui(
        "bun",
        &["install", "--frozen-lockfile"],
        &["run", "build"],
        &ui_dir,
    ) {
        return;
    }

    if try_build_ui("npm", &["ci"], &["run", "build"], &ui_dir) {
        return;
    }

    println!(
        "cargo:warning=Could not build the web UI. \
         Ensure bun or npm is installed and the UI compiles without errors. \
         The binary will work but `docbert web` will serve no UI."
    );
    ensure_dist_dir(&dist_dir);
}

fn ensure_dist_dir(dist_dir: &Path) {
    if let Err(e) = std::fs::create_dir_all(dist_dir) {
        println!(
            "cargo:warning=Failed to create ui/dist fallback directory: {e}"
        );
    }
}

fn try_build_ui(
    cmd: &str,
    install_args: &[&str],
    build_args: &[&str],
    ui_dir: &Path,
) -> bool {
    let Ok(status) = Command::new(cmd)
        .args(install_args)
        .current_dir(ui_dir)
        .status()
    else {
        return false;
    };

    if !status.success() {
        return false;
    }

    Command::new(cmd)
        .args(build_args)
        .current_dir(ui_dir)
        .status()
        .is_ok_and(|s| s.success())
}
