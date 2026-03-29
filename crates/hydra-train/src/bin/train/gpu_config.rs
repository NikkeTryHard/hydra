/// GPU-level performance flags applied at process startup.
///
/// Global libtorch settings affecting kernel selection without changing
/// training logic.  Must be called before any tensor operations.
pub(crate) fn apply_gpu_performance_flags() {
    if tch::Cuda::is_available() {
        // Auto-tunes conv algorithms per input shape on first call, caches
        // the fastest.  Safe with fixed tensor shapes (Hydra's case).
        tch::Cuda::cudnn_set_benchmark(true);
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
