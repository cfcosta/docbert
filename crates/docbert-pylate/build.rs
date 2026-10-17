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
    #[cfg(feature = "cuda")]
    {
        println!("cargo::rerun-if-changed=src/kernels/cutlass_geglu.cu");
        println!("cargo::rerun-if-changed=src/kernels/cutlass_dual");
        println!("cargo::rerun-if-env-changed=CUDAFORGE_HOME");
        // Same pinned headers/cache as candle-flash-attn; Nix prepopulates it.
        let headers = cudaforge::ExternalDependency::cutlass(Some(
            "7d49e6c7e2f8896c47f586706e67e1fb215529dc",
        ))
        .fetch(&output_dir)
        .expect("resolve pinned CUTLASS headers");
        let is_msvc = env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
        let mut compiler = Command::new(&nvcc);
        compiler.args([
            "--lib",
            "-arch=sm_80",
            "-O3",
            "--std=c++17",
            "--expt-relaxed-constexpr",
        ]);
        if !is_msvc {
            compiler.args(["-Xcompiler", "-fPIC"]);
            // The pinned CUTLASS calls a misspelled `set_slice3x3` in
            // uninstantiated templates of `matrix.h`; GCC 14+ rejects that
            // by default.
            compiler.args(["-Xcompiler", "-Wno-template-body"]);
        }
        let status = compiler
            .arg("-I")
            .arg(headers.join("include"))
            .arg("src/kernels/cutlass_geglu.cu")
            .arg("-o")
            .arg(output_dir.join("libdocbert_cutlass.a"))
            .status()
            .expect("compile CUTLASS epilogue");
        assert!(status.success(), "compiling CUTLASS epilogue failed");
        println!("cargo::rustc-link-search=native={}", output_dir.display());
        println!("cargo::rustc-link-lib=static=docbert_cutlass");
        println!("cargo::rustc-link-lib=dylib=cudart");
        if !is_msvc {
            println!("cargo::rustc-link-lib=dylib=stdc++");
        }
    }
}
