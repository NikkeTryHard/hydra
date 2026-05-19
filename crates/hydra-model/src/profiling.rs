//! Lightweight model-local profiling scopes.

use std::env;
#[cfg(all(not(test), feature = "libtorch"))]
use std::ffi::CString;
#[cfg(all(not(test), feature = "libtorch"))]
use std::sync::{Once, OnceLock};

#[cfg(all(not(test), feature = "libtorch"))]
use libloading::{Library, Symbol};

const NVTX_ENV: &str = "HYDRA_NVTX";

/// Guard returned by [`scope`].
pub struct ModelScope(GuardState);

enum GuardState {
    Noop,
    #[cfg(all(not(test), feature = "libtorch"))]
    Active(u64),
}

impl Drop for ModelScope {
    fn drop(&mut self) {
        match self.0 {
            GuardState::Noop => {}
            #[cfg(all(not(test), feature = "libtorch"))]
            GuardState::Active(range_id) => backend_end(range_id),
        }
    }
}

/// Opens a model profiling scope when `HYDRA_NVTX` is enabled.
#[inline]
#[cfg(not(all(not(test), feature = "libtorch")))]
pub fn scope(_stage: &'static str) -> ModelScope {
    let _enabled = env::var(NVTX_ENV)
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    ModelScope(GuardState::Noop)
}

/// Opens a model profiling scope when `HYDRA_NVTX` is enabled.
#[inline]
#[cfg(all(not(test), feature = "libtorch"))]
pub fn scope(stage: &'static str) -> ModelScope {
    if !nvtx_enabled() {
        return ModelScope(GuardState::Noop);
    }
    let Some(backend) = backend() else {
        return ModelScope(GuardState::Noop);
    };
    let Ok(name) = CString::new(stage) else {
        return ModelScope(GuardState::Noop);
    };
    let range_id = unsafe { (backend.start)(name.as_ptr()) };
    if range_id == 0 {
        ModelScope(GuardState::Noop)
    } else {
        ModelScope(GuardState::Active(range_id))
    }
}

#[cfg(all(not(test), feature = "libtorch"))]
type NvtxRangeStart = unsafe extern "C" fn(*const std::ffi::c_char) -> u64;
#[cfg(all(not(test), feature = "libtorch"))]
type NvtxRangeEnd = unsafe extern "C" fn(u64) -> i32;

#[cfg(all(not(test), feature = "libtorch"))]
struct NvtxBackend {
    _library: Library,
    start: NvtxRangeStart,
    end: NvtxRangeEnd,
}

#[cfg(all(not(test), feature = "libtorch"))]
fn nvtx_enabled() -> bool {
    env::var(NVTX_ENV)
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

#[cfg(all(not(test), feature = "libtorch"))]
fn backend_end(range_id: u64) {
    if let Some(backend) = backend() {
        let _ = unsafe { (backend.end)(range_id) };
    }
}

#[cfg(all(not(test), feature = "libtorch"))]
fn backend() -> Option<&'static NvtxBackend> {
    static INIT: Once = Once::new();
    static BACKEND: OnceLock<Option<NvtxBackend>> = OnceLock::new();
    INIT.call_once(|| {
        let _ = BACKEND.set(load_backend());
    });
    BACKEND.get().and_then(Option::as_ref)
}

#[cfg(all(not(test), feature = "libtorch"))]
fn load_backend() -> Option<NvtxBackend> {
    const LIB_NAMES: &[&str] = &[
        "libnvToolsExt.so.1",
        "libnvToolsExt.so",
        "nvToolsExt64_1.dll",
        "nvToolsExt64.dll",
    ];

    LIB_NAMES.iter().find_map(|name| {
        let library = unsafe { Library::new(name).ok()? };
        let (start, end) = unsafe {
            let start: Symbol<'_, NvtxRangeStart> = library.get(b"nvtxRangeStartA\0").ok()?;
            let end: Symbol<'_, NvtxRangeEnd> = library.get(b"nvtxRangeEnd\0").ok()?;
            (*start, *end)
        };
        Some(NvtxBackend {
            _library: library,
            start,
            end,
        })
    })
}
