use std::sync::Once;

#[cfg(feature = "cuda-graph")]
unsafe extern "C" {
    fn hydra_set_tf32_precision(b: std::ffi::c_int);
}

static LIBTORCH_CPU_POOL_CONFIG: Once = Once::new();
static CUDA_PERFORMANCE_FLAGS: Once = Once::new();

fn device_requests_cuda(device: &str) -> bool {
    device
        .split(':')
        .next()
        .is_some_and(|kind| kind.eq_ignore_ascii_case("cuda"))
}

/// Global libtorch CPU-pool flags. Must be called before any tensor ops.
pub fn configure_libtorch_cpu_threads(num_threads: usize) {
    LIBTORCH_CPU_POOL_CONFIG.call_once(|| {
        let threads = num_threads.max(1) as i32;
        tch::set_num_interop_threads(1);
        tch::set_num_threads(threads);
    });
}

/// Global CUDA performance flags. Must be called before any tensor ops.
pub fn apply_gpu_performance_flags(device: &str) {
    if !device_requests_cuda(device) {
        return;
    }

    CUDA_PERFORMANCE_FLAGS.call_once(|| {
        // SAFETY: called inside Once, before any tensor ops or thread spawning.
        unsafe {
            std::env::set_var("OMP_NUM_THREADS", "1");
            std::env::set_var("MKL_NUM_THREADS", "1");
            // Cap the NVIDIA driver's internal PTX JIT compiler to 2 threads.
            // Without this, it saturates ALL cores when compiling PTX->SASS
            // for GPUs that lack precompiled SASS in the libtorch binary
            // (e.g., Blackwell sm_120). Requires driver R570+ (CUDA 13.1).
            if std::env::var("CUDA_BINARY_LOADER_THREAD_COUNT").is_err() {
                std::env::set_var("CUDA_BINARY_LOADER_THREAD_COUNT", "2");
            }
            if std::env::var("CUDA_CACHE_MAXSIZE").is_err() {
                std::env::set_var("CUDA_CACHE_MAXSIZE", "4294967296");
            }
        }
    });

    if tch::Cuda::is_available() {
        // Auto-tunes conv algorithms per input shape on first call, caches
        // the fastest. Safe with fixed tensor shapes (Hydra's case).
        tch::Cuda::cudnn_set_benchmark(true);

        #[cfg(feature = "cuda-graph")]
        {
            // TF32: on Ampere+ GPUs, uses Tensor Cores for FP32 matmul/conv
            // with 10-bit mantissa. Same exponent range, no overflow risk.
            // tch-rs doesn't expose globalContext TF32 setters.
            unsafe {
                hydra_set_tf32_precision(1);
            }
        }
    }
}

#[cfg(test)]
mod tests;
