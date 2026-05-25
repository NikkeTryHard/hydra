use std::{env, path::PathBuf};

pub fn init_ort_from_env() -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(path) = env::var("HYDRA_ONNXRUNTIME_DYLIB") {
        ort::init_from(path)?.commit();
        return Ok(());
    }
    if let Ok(root) = env::var("PIXI_PROJECT_ROOT") {
        let gpu_path = PathBuf::from(&root).join(
            ".pixi/envs/default/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2",
        );
        if gpu_path.exists() {
            ort::init_from(gpu_path)?.commit();
            return Ok(());
        }
        let path = PathBuf::from(root).join(
            ".pixi/envs/default/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.26.0",
        );
        if path.exists() {
            ort::init_from(path)?.commit();
            return Ok(());
        }
    }
    ort::init().commit();
    Ok(())
}
