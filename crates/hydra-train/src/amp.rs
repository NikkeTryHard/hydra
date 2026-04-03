// tch::autocast and at::autocast::set_autocast_enabled(at::kCUDA) both fail
// with Burn's LibTorch backend: Burn calls conv/matmul through torch_sys FFI
// which enters at::native:: directly, causing CUDAHalfType input vs
// CUDAFloatType weight mismatches.  Autocast is disabled until Burn gains
// native AMP support.  The use_amp plumbing remains so callers don't need
// changes when AMP becomes viable.
pub fn maybe_autocast<T, F>(_enabled: bool, f: F) -> T
where
    F: FnOnce() -> T,
{
    f()
}
