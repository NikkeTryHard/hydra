use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_TCH_LIBTORCH_LIB");
    println!("cargo:rerun-if-env-changed=LIBTORCH_USE_PYTORCH");
    if env::var_os("CARGO_FEATURE_TRAINING").is_none()
        && env::var_os("CARGO_FEATURE_CUDA_GRAPH").is_none()
    {
        return;
    }

    let Ok(libtorch_lib) = env::var("DEP_TCH_LIBTORCH_LIB") else {
        return;
    };
    let libtorch_lib = PathBuf::from(libtorch_lib);
    if let Some(libtorch_cuda_lib) = find_libtorch_cuda_lib_dir(&libtorch_lib) {
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath={}:{}",
            libtorch_cuda_lib.display(),
            libtorch_lib.display()
        );
    } else {
        println!("cargo:rustc-link-arg=-Wl,-rpath={}", libtorch_lib.display());
    }
}

fn find_libtorch_cuda_lib_dir(libtorch_lib: &Path) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    candidates.push(libtorch_lib.to_path_buf());
    if let Some(python_site_packages) = python_site_packages_from_torch_sys_out(libtorch_lib) {
        candidates.push(python_site_packages.join("torch/lib"));
    }
    candidates
        .into_iter()
        .find(|path| path.join("libc10_cuda.so").exists())
}

fn python_site_packages_from_torch_sys_out(libtorch_lib: &Path) -> Option<PathBuf> {
    let marker = Path::new("/target/");
    let path = libtorch_lib.to_string_lossy();
    let repo_root = path.split(marker.to_str()?).next()?;
    let home = Path::new(repo_root).parent()?.parent()?;
    let python_lib = home.join(".pyenv/versions/3.12.13/lib/python3.12/site-packages");
    python_lib.exists().then_some(python_lib)
}
