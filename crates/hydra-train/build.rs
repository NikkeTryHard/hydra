use std::path::PathBuf;

fn main() {
    let libtorch_lib = match std::env::var("DEP_TCH_LIBTORCH_LIB") {
        Ok(v) => PathBuf::from(v),
        Err(_) => return,
    };

    let libtorch_root = libtorch_lib
        .parent()
        .expect("libtorch lib dir should have a parent");
    let include_dir = libtorch_root.join("include");
    let torch_csrc_include = libtorch_root.join("include/torch/csrc/api/include");

    if !include_dir.exists() {
        return;
    }

    println!("cargo:rerun-if-changed=csrc/hydra_gpu.cpp");

    cc::Build::new()
        .cpp(true)
        .pic(true)
        .warnings(false)
        .include(&include_dir)
        .include(&torch_csrc_include)
        .flag("-std=c++17")
        .file("csrc/hydra_gpu.cpp")
        .compile("hydra_gpu");
}
