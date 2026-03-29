unsafe extern "C" {
    fn hydra_set_allow_tf32_cublas(b: std::ffi::c_int);
    fn hydra_set_allow_tf32_cudnn(b: std::ffi::c_int);
}

/// Global libtorch performance flags.  Must be called before any tensor ops.
pub(crate) fn apply_gpu_performance_flags() {
    if tch::Cuda::is_available() {
        // Auto-tunes conv algorithms per input shape on first call, caches
        // the fastest.  Safe with fixed tensor shapes (Hydra's case).
        tch::Cuda::cudnn_set_benchmark(true);

        // TF32: on Ampere+ GPUs, uses Tensor Cores for FP32 matmul/conv
        // with 10-bit mantissa.  Same exponent range, no overflow risk.
        // tch-rs doesn't expose globalContext TF32 setters.
        unsafe {
            hydra_set_allow_tf32_cublas(1);
            hydra_set_allow_tf32_cudnn(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_gpu_performance_flags_is_safe_on_cpu() {
        apply_gpu_performance_flags();
    }
}
