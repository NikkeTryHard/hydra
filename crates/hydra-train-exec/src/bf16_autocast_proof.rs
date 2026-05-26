#![allow(
    dead_code,
    reason = "proof-only BF16 autocast shim is exercised by temporary Wave 2 probes"
)]

use std::ffi::c_int;
use std::panic::{UnwindSafe, resume_unwind};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HydraCudaAutocastState {
    pub(crate) enabled: c_int,
    pub(crate) dtype: c_int,
    pub(crate) cache_enabled: c_int,
}

unsafe extern "C" {
    fn hydra_cuda_autocast_get_state(state_out: *mut HydraCudaAutocastState) -> c_int;
    pub(crate) fn hydra_cuda_autocast_restore_state(state: *const HydraCudaAutocastState) -> c_int;
    fn hydra_cuda_autocast_enter_bf16(previous_out: *mut HydraCudaAutocastState) -> c_int;
    fn hydra_cuda_autocast_exit(previous: *const HydraCudaAutocastState) -> c_int;
}

struct CudaAutocastStateRestore {
    old_state: HydraCudaAutocastState,
    active: bool,
}

impl Drop for CudaAutocastStateRestore {
    fn drop(&mut self) {
        if self.active {
            // SAFETY: Restores the exact state value returned by libtorch before the proof closure.
            unsafe {
                let _ = hydra_cuda_autocast_exit(&self.old_state);
            }
        }
    }
}

pub(crate) fn current_cuda_autocast_state() -> Result<HydraCudaAutocastState, &'static str> {
    let mut state = HydraCudaAutocastState {
        enabled: 0,
        dtype: 0,
        cache_enabled: 0,
    };
    // SAFETY: `state` is a valid out pointer for the C shim.
    let status = unsafe { hydra_cuda_autocast_get_state(&mut state) };
    if status == 0 {
        Ok(state)
    } else {
        Err("hydra_cuda_autocast_get_state failed")
    }
}

pub(crate) fn restore_cuda_autocast_state(
    state: &HydraCudaAutocastState,
) -> Result<(), &'static str> {
    // SAFETY: `state` is a valid pointer for the C shim.
    let status = unsafe { hydra_cuda_autocast_restore_state(state) };
    if status == 0 {
        Ok(())
    } else {
        Err("hydra_cuda_autocast_restore_state failed")
    }
}

/// Runs `f` with CUDA autocast enabled and dtype forced to BF16.
///
/// This is crate-private BF16 proof plumbing only. It must not be used by production
/// training until external proof gates explicitly enable AMP.
pub(crate) fn with_cuda_bf16_autocast_dtype_proof_only<F, R>(f: F) -> Result<R, &'static str>
where
    F: FnOnce() -> R + UnwindSafe,
{
    let mut old_state = HydraCudaAutocastState {
        enabled: 0,
        dtype: 0,
        cache_enabled: 0,
    };
    // SAFETY: `old_state` is a valid out pointer for the C shim.
    let status = unsafe { hydra_cuda_autocast_enter_bf16(&mut old_state) };
    if status != 0 {
        return Err("hydra_cuda_autocast_enter_bf16 failed");
    }

    let restore = CudaAutocastStateRestore {
        old_state,
        active: true,
    };
    let result = std::panic::catch_unwind(f);
    drop(restore);

    match result {
        Ok(value) => Ok(value),
        Err(payload) => resume_unwind(payload),
    }
}

#[cfg(test)]
mod tests {
    use super::{current_cuda_autocast_state, with_cuda_bf16_autocast_dtype_proof_only};

    #[test]
    fn restores_autocast_state_after_closure() {
        let before = current_cuda_autocast_state()
            .expect("BF16 autocast proof shim should read CUDA autocast state");
        let value = with_cuda_bf16_autocast_dtype_proof_only(|| 7)
            .expect("BF16 autocast proof shim should enter CUDA autocast state");
        let after = current_cuda_autocast_state()
            .expect("BF16 autocast proof shim should read restored CUDA autocast state");
        assert_eq!(value, 7);
        assert_eq!(after, before);
    }
}
