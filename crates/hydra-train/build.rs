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
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA_GRAPH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rustc-link-arg=-Wl,-rpath={}", libtorch_lib.display());

    let cuda_graph_enabled = std::env::var_os("CARGO_FEATURE_CUDA_GRAPH").is_some();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .pic(true)
        .warnings(false)
        .include(&include_dir)
        .include(&torch_csrc_include)
        .flag("-std=c++17")
        .file("csrc/hydra_gpu.cpp");

    if cuda_graph_enabled {
        build.define("HYDRA_ENABLE_CUDA_GRAPH_FFI", None);

        let cuda_include = ["CUDA_HOME", "CUDA_PATH"]
            .into_iter()
            .filter_map(std::env::var_os)
            .map(PathBuf::from)
            .map(|path| path.join("include"))
            .find(|path| path.join("cuda_runtime_api.h").exists())
            .or_else(|| {
                let default = PathBuf::from("/usr/local/cuda/include");
                default
                    .join("cuda_runtime_api.h")
                    .exists()
                    .then_some(default)
            });

        if let Some(cuda_include) = cuda_include {
            build.include(cuda_include);
            build.define("HYDRA_USE_CUDA_GRAPH", None);
        } else {
            println!(
                "cargo:warning=cuda-graph feature enabled without CUDA headers; building shim stubs"
            );
        }
    }

    build.compile("hydra_gpu");
}
