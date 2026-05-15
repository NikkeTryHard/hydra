use std::env;
use std::path::PathBuf;

fn main() {
    if env::var_os("CARGO_FEATURE_LIBTORCH_TESTS").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=DEP_TCH_LIBTORCH_LIB");
    println!("cargo:rerun-if-env-changed=LIBTORCH_USE_PYTORCH");

    let Ok(libtorch_lib) = env::var("DEP_TCH_LIBTORCH_LIB") else {
        return;
    };
    let libtorch_lib = PathBuf::from(libtorch_lib);
    println!("cargo:rustc-link-arg=-Wl,-rpath={}", libtorch_lib.display());
}
