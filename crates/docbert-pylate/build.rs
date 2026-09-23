use std::{env, path::PathBuf, process::Command};

fn main() {
    let kernels = ["geglu", "residual_norm", "rope_pair"];
    println!("cargo::rerun-if-changed=build.rs");
    for kernel in kernels {
        println!("cargo::rerun-if-changed=src/kernels/{kernel}.cu");
    }
    println!("cargo::rerun-if-env-changed=CUDA_PATH");
    println!("cargo::rerun-if-env-changed=NVCC");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let nvcc = env::var_os("NVCC")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("CUDA_PATH")
                .map(|path| PathBuf::from(path).join("bin/nvcc"))
        })
        .unwrap_or_else(|| PathBuf::from("nvcc"));
    let output_dir =
        PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    // Candle's BF16 CUDA kernels require Ampere or newer.
    for kernel in kernels {
        let status = Command::new(&nvcc)
            .args([
                "--ptx",
                "--gpu-architecture=compute_80",
                "-O3",
                "--std=c++17",
            ])
            .arg(format!("src/kernels/{kernel}.cu"))
            .arg("-o")
            .arg(output_dir.join(format!("{kernel}.ptx")))
            .status()
            .expect("CUDA builds require nvcc (set NVCC or CUDA_PATH)");
        assert!(status.success(), "compiling {kernel} failed");
    }
}
