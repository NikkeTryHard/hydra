use std::sync::Once;

static CPU_POOL_ENV_CONFIG: Once = Once::new();
static CUDA_PERFORMANCE_FLAGS: Once = Once::new();

fn device_requests_cuda(device: &str) -> bool {
    device
        .split(':')
        .next()
        .is_some_and(|kind| kind.eq_ignore_ascii_case("cuda"))
}

/// Process-wide CPU-pool environment defaults for Python launcher children.
pub fn configure_python_cpu_threads(num_threads: usize) {
    CPU_POOL_ENV_CONFIG.call_once(|| {
        let threads = num_threads.max(1).to_string();
        // SAFETY: this crate calls the configurator during launcher setup, before
        // spawning worker processes that inherit the environment.
        unsafe {
            std::env::set_var("OMP_NUM_THREADS", &threads);
            std::env::set_var("MKL_NUM_THREADS", &threads);
        }
    });
}

/// Backward-compatible name for callers that have not yet been moved to the
/// Python-specific helper. This no longer calls LibTorch/tch APIs.
pub fn configure_libtorch_cpu_threads(num_threads: usize) {
    configure_python_cpu_threads(num_threads);
}

/// Global CUDA performance environment flags for Python launcher children.
pub fn apply_gpu_performance_flags(device: &str) {
    if !device_requests_cuda(device) {
        return;
    }

    CUDA_PERFORMANCE_FLAGS.call_once(|| {
        // SAFETY: this crate calls the configurator during launcher setup, before
        // spawning worker processes that inherit the environment.
        unsafe {
            std::env::set_var("OMP_NUM_THREADS", "1");
            std::env::set_var("MKL_NUM_THREADS", "1");
            // Cap the NVIDIA driver's internal PTX JIT compiler to 2 threads.
            // Without this, it may saturate cores when Python/PyTorch compiles
            // PTX for GPUs that lack precompiled SASS in the wheel.
            if std::env::var("CUDA_BINARY_LOADER_THREAD_COUNT").is_err() {
                std::env::set_var("CUDA_BINARY_LOADER_THREAD_COUNT", "2");
            }
            if std::env::var("CUDA_CACHE_MAXSIZE").is_err() {
                std::env::set_var("CUDA_CACHE_MAXSIZE", "4294967296");
            }
        }
    });
}

#[cfg(test)]
mod tests;
