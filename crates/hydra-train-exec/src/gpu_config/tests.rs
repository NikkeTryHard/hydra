use super::*;

#[test]
fn apply_gpu_performance_flags_is_safe_on_cpu() {
    apply_gpu_performance_flags("cpu");
}

#[test]
fn configure_libtorch_cpu_threads_is_safe_before_tensor_ops() {
    configure_libtorch_cpu_threads(2);
}

#[test]
fn device_requests_cuda_only_for_cuda_prefix() {
    assert!(device_requests_cuda("cuda"));
    assert!(device_requests_cuda("cuda:0"));
    assert!(device_requests_cuda("CUDA:1"));
    assert!(!device_requests_cuda("cpu"));
    assert!(!device_requests_cuda("metal:0"));
}
