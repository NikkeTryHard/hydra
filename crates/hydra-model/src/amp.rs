#[cfg(feature = "libtorch")]
#[repr(C)]
#[derive(Clone, Copy)]
struct CudaAutocastState {
    enabled: i32,
    dtype: i32,
    cache_enabled: i32,
}

#[cfg(feature = "libtorch")]
unsafe extern "C" {
    fn hydra_model_cuda_autocast_enter_bf16(previous_out: *mut CudaAutocastState) -> i32;
    fn hydra_model_cuda_autocast_exit(previous: *const CudaAutocastState) -> i32;
}

#[cfg(feature = "libtorch")]
struct CudaAutocastGuard {
    previous: CudaAutocastState,
    active: bool,
}

#[cfg(feature = "libtorch")]
impl Drop for CudaAutocastGuard {
    fn drop(&mut self) {
        if self.active {
            // SAFETY: `previous` is a valid saved autocast state for this guard scope.
            let _ = unsafe { hydra_model_cuda_autocast_exit(&self.previous) };
        }
    }
}

/// Runs `f` under CUDA BF16 autocast when `enabled` and the libtorch feature is active.
#[cfg(feature = "libtorch")]
pub fn maybe_autocast<T, F>(enabled: bool, f: F) -> T
where
    F: FnOnce() -> T,
{
    if !enabled {
        return f();
    }

    let mut previous = CudaAutocastState {
        enabled: 0,
        dtype: 0,
        cache_enabled: 0,
    };
    // SAFETY: `previous` is a valid out pointer for the native shim.
    let active = unsafe { hydra_model_cuda_autocast_enter_bf16(&mut previous) } == 0;
    assert!(active, "failed to enter CUDA BF16 autocast scope");
    let _guard = CudaAutocastGuard { previous, active };
    f()
}

/// Executes `f` without autocast when libtorch support is not compiled.
#[cfg(not(feature = "libtorch"))]
pub fn maybe_autocast<T, F>(_enabled: bool, f: F) -> T
where
    F: FnOnce() -> T,
{
    f()
}
