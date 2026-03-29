use std::path::PathBuf;

fn main() {
    let libtorch_lib = match std::env::var("DEP_TCH_LIBTORCH_LIB") {
        Ok(v) => PathBuf::from(v),
        Err(_) => {
            eprintln!("cargo:warning=DEP_TCH_LIBTORCH_LIB not set, skipping hydra_gpu build");
            return;
        }
    };

    let libtorch_root = libtorch_lib
        .parent()
        .expect("libtorch lib dir should have a parent");
    let include_dir = libtorch_root.join("include");
    let torch_csrc_include = libtorch_root.join("include/torch/csrc/api/include");

    if !include_dir.exists() {
        eprintln!(
            "cargo:warning=libtorch include dir not found at {}",
            include_dir.display()
        );
        return;
    }

    println!("cargo:rerun-if-changed=csrc/hydra_gpu.cpp");

    let has_cuda = include_dir.join("ATen/cuda/CUDAGraph.h").exists()
        && std::env::var("CUDA_HOME").is_ok()
        || std::env::var("CUDA_PATH").is_ok()
        || std::path::Path::new("/usr/local/cuda/include/cuda_runtime_api.h").exists();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .pic(true)
        .warnings(false)
        .include(&include_dir)
        .include(&torch_csrc_include)
        .flag("-std=c++17")
        .file("csrc/hydra_gpu.cpp");

    if has_cuda {
        if let Ok(cuda_home) = std::env::var("CUDA_HOME").or_else(|_| std::env::var("CUDA_PATH")) {
            build.include(format!("{cuda_home}/include"));
        } else if std::path::Path::new("/usr/local/cuda/include").exists() {
            build.include("/usr/local/cuda/include");
        }
        build.define("USE_CUDA", None);
    }

    build.compile("hydra_gpu");
}
