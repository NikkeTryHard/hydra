use std::env;
use std::path::{Path, PathBuf};

const CUDA_ENV_VARS: [&str; 2] = ["CUDA_HOME", "CUDA_PATH"];

#[derive(Debug, Clone)]
struct CudaPaths {
    include_dir: PathBuf,
    lib_dir: PathBuf,
}

fn main() {
    println!("cargo:rerun-if-changed=csrc/hydra_gpu.cpp");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA_GRAPH");
    println!("cargo:rerun-if-env-changed=DEP_TCH_LIBTORCH_LIB");
    for name in CUDA_ENV_VARS {
        println!("cargo:rerun-if-env-changed={name}");
    }

    let cuda_graph_enabled = env::var_os("CARGO_FEATURE_CUDA_GRAPH").is_some();
    let libtorch_lib = match env::var("DEP_TCH_LIBTORCH_LIB") {
        Ok(v) => PathBuf::from(v),
        Err(_) if cuda_graph_enabled => {
            panic!(
                "cuda-graph feature requires torch-sys libtorch discovery, but \
                 DEP_TCH_LIBTORCH_LIB is not set; build with tch/torch-sys available \
                 (for PyTorch wheels, set LIBTORCH_USE_PYTORCH=1)"
            );
        }
        Err(_) => {
            println!(
                "cargo:warning=hydra_gpu: DEP_TCH_LIBTORCH_LIB not set; building without hydra_gpu FFI"
            );
            return;
        }
    };

    let torch_root_include = match find_libtorch_root_include_dir(&libtorch_lib) {
        Some(path) => path,
        None if cuda_graph_enabled => {
            panic!(
                "cuda-graph feature requires libtorch headers, but none were found near {}; \
                 expected include/ATen/Context.h plus include/torch/csrc/api/include/torch/all.h. \
                 For PyTorch wheels, set LIBTORCH_USE_PYTORCH=1 and ensure torch headers are installed.",
                libtorch_lib.display()
            );
        }
        None => {
            println!(
                "cargo:warning=hydra_gpu: libtorch include dir not found near {}; skipping hydra_gpu build",
                libtorch_lib.display()
            );
            return;
        }
    };
    let torch_csrc_include = torch_root_include.join("torch/csrc/api/include");
    if cuda_graph_enabled {
        assert!(
            torch_csrc_include.exists(),
            "cuda-graph feature requires libtorch C++ API headers at {}; \
             set LIBTORCH_USE_PYTORCH=1 or LIBTORCH to a complete libtorch install",
            torch_csrc_include.display()
        );
    }
    if !torch_csrc_include.exists() {
        println!(
            "cargo:warning=hydra_gpu: libtorch C++ API include dir not found at {}; skipping hydra_gpu build",
            torch_csrc_include.display()
        );
        return;
    }

    println!("cargo:rustc-link-arg=-Wl,-rpath={}", libtorch_lib.display());

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .pic(true)
        .warnings(false)
        .include(&torch_root_include)
        .include(&torch_csrc_include)
        .flag("-std=c++17")
        .file("csrc/hydra_gpu.cpp");

    if cuda_graph_enabled {
        let cuda_paths = discover_cuda_paths().unwrap_or_else(|| {
            panic!(
                "cuda-graph feature requires CUDA headers and cudart, but discovery failed. \
                 Probed include dirs: {}. Probed lib dirs: {}. \
                 On this workstation use CUDA_HOME=/opt/cuda or install CUDA toolkit development files.",
                format_paths(&cuda_include_candidates()),
                format_paths(&cuda_lib_candidates())
            )
        });
        let libtorch_cuda_lib = find_libtorch_cuda_lib_dir(&libtorch_lib).unwrap_or_else(|| {
            panic!(
                "cuda-graph feature requires PyTorch CUDA library libc10_cuda.so near {}; \
                 install/use a CUDA-enabled PyTorch/libtorch build and set LIBTORCH_USE_PYTORCH=1 if using a wheel",
                libtorch_lib.display()
            )
        });
        require_libtorch_cuda_component(&libtorch_cuda_lib, "c10_cuda");

        build.define("HYDRA_ENABLE_CUDA_GRAPH_FFI", None);
        build.define("HYDRA_USE_CUDA_GRAPH", None);
        build.include(&cuda_paths.include_dir);

        println!(
            "cargo:rustc-link-search=native={}",
            libtorch_cuda_lib.display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_paths.lib_dir.display()
        );
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=c10_cuda");
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath={}",
            cuda_paths.lib_dir.display()
        );
        println!(
            "cargo:warning=hydra_gpu: cuda-graph enabled; building real CUDA graph/pinned FFI with CUDA include={} lib={} libtorch_lib={}",
            cuda_paths.include_dir.display(),
            cuda_paths.lib_dir.display(),
            libtorch_cuda_lib.display()
        );
    } else {
        println!(
            "cargo:warning=hydra_gpu: cuda-graph disabled; building TF32 shim only (no pinned async H2D FFI)"
        );
    }

    build.compile("hydra_gpu");
}

fn find_libtorch_root_include_dir(libtorch_lib: &Path) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(root) = libtorch_lib.parent() {
        candidates.push(root.join("include"));
    }
    candidates.push(libtorch_lib.join("include"));

    candidates.into_iter().find(|path| {
        path.join("ATen/Context.h").exists()
            && path.join("torch/csrc/api/include/torch/all.h").exists()
    })
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

fn discover_cuda_paths() -> Option<CudaPaths> {
    let include_dir = cuda_include_candidates()
        .into_iter()
        .find(|path| path.join("cuda_runtime_api.h").exists())?;
    let lib_dir = cuda_lib_candidates()
        .into_iter()
        .find(|path| has_cudart(path))?;
    Some(CudaPaths {
        include_dir,
        lib_dir,
    })
}

fn cuda_roots() -> Vec<PathBuf> {
    let mut roots = CUDA_ENV_VARS
        .into_iter()
        .filter_map(env::var_os)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    roots.push(PathBuf::from("/opt/cuda"));
    roots.push(PathBuf::from("/usr/local/cuda"));
    roots
}

fn cuda_include_candidates() -> Vec<PathBuf> {
    cuda_roots()
        .into_iter()
        .map(|root| root.join("include"))
        .collect()
}

fn cuda_lib_candidates() -> Vec<PathBuf> {
    cuda_roots()
        .into_iter()
        .flat_map(|root| {
            [
                root.join("targets/x86_64-linux/lib"),
                root.join("lib64"),
                root.join("lib"),
            ]
        })
        .collect()
}

fn has_cudart(path: &Path) -> bool {
    path.join("libcudart.so").exists() || path.join("libcudart_static.a").exists()
}

fn require_libtorch_cuda_component(libtorch_lib: &Path, name: &str) {
    let file_name = format!("lib{name}.so");
    assert!(
        libtorch_lib.join(&file_name).exists(),
        "cuda-graph feature requires PyTorch CUDA library {} in {}; \
         install/use a CUDA-enabled PyTorch/libtorch build and set LIBTORCH_USE_PYTORCH=1 if using a wheel",
        file_name,
        libtorch_lib.display()
    );
}

fn format_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}
