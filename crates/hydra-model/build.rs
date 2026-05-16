use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_LIBTORCH");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_LIBTORCH_TESTS");

    if env::var_os("CARGO_FEATURE_LIBTORCH").is_none()
        && env::var_os("CARGO_FEATURE_LIBTORCH_TESTS").is_none()
    {
        return;
    }

    println!("cargo:rerun-if-changed=csrc/cuda_autocast.cpp");
    println!("cargo:rerun-if-env-changed=DEP_TCH_LIBTORCH_LIB");
    println!("cargo:rerun-if-env-changed=LIBTORCH_INCLUDE");
    println!("cargo:rerun-if-env-changed=LIBTORCH_USE_PYTORCH");

    let Ok(libtorch_lib) = env::var("DEP_TCH_LIBTORCH_LIB") else {
        return;
    };
    let libtorch_lib = PathBuf::from(libtorch_lib);

    let Some(torch_root_include) = find_libtorch_root_include_dir(&libtorch_lib) else {
        return;
    };
    let torch_csrc_include = torch_root_include.join("torch/csrc/api/include");
    println!("cargo:rustc-link-arg=-Wl,-rpath={}", libtorch_lib.display());

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .pic(true)
        .warnings(false)
        .include(&torch_root_include)
        .include(&torch_csrc_include)
        .flag("-std=c++17")
        .file("csrc/cuda_autocast.cpp");
    build.compile("hydra_model_native");
}

fn find_libtorch_root_include_dir(libtorch_lib: &Path) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("LIBTORCH_INCLUDE") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(root) = libtorch_lib.parent() {
        candidates.push(root.join("include"));
    }
    candidates.push(libtorch_lib.join("include"));
    if let Some(torch_package) = libtorch_lib.parent() {
        candidates.push(torch_package.join("include"));
    }

    candidates.into_iter().find(|path| {
        path.join("ATen/Context.h").exists()
            && path.join("torch/csrc/api/include/torch/all.h").exists()
    })
}
