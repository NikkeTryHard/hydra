//! NVTX execution scope helpers.

use std::env;
#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
use std::ffi::CString;
#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
use std::sync::{Once, OnceLock};

#[cfg(any(test, feature = "test-nvtx-recorder"))]
use std::cell::RefCell;

#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
use libloading::{Library, Symbol};

const NVTX_ENV: &str = "HYDRA_NVTX";

/// Guard that closes an NVTX range when dropped.
pub struct NvtxRangeGuard(GuardState);

enum GuardState {
    Noop,
    #[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
    Active,
    #[cfg(any(test, feature = "test-nvtx-recorder"))]
    Recorded(&'static str),
}

impl Drop for NvtxRangeGuard {
    fn drop(&mut self) {
        match &self.0 {
            GuardState::Noop => {}
            #[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
            GuardState::Active => backend_pop(),
            #[cfg(any(test, feature = "test-nvtx-recorder"))]
            GuardState::Recorded(stage) => record_test_event("pop", stage),
        }
    }
}

/// Opens an NVTX scope when `HYDRA_NVTX` is enabled and NVTX can be loaded.
#[cfg(any(test, feature = "test-nvtx-recorder"))]
pub fn scope(stage: &'static str) -> NvtxRangeGuard {
    if test_recorder_active() {
        record_test_event("push", stage);
        return NvtxRangeGuard(GuardState::Recorded(stage));
    }

    let _enabled = env::var(NVTX_ENV)
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(false);

    NvtxRangeGuard(GuardState::Noop)
}

/// Opens an NVTX scope when `HYDRA_NVTX` is enabled and NVTX can be loaded.
#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
pub fn scope(stage: &'static str) -> NvtxRangeGuard {
    if !nvtx_enabled() {
        return NvtxRangeGuard(GuardState::Noop);
    }

    let Some(backend) = backend() else {
        return NvtxRangeGuard(GuardState::Noop);
    };

    let Ok(name) = CString::new(stage) else {
        return NvtxRangeGuard(GuardState::Noop);
    };

    if unsafe { (backend.push)(name.as_ptr()) } >= 0 {
        NvtxRangeGuard(GuardState::Active)
    } else {
        NvtxRangeGuard(GuardState::Noop)
    }
}

#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
type NvtxPush = unsafe extern "C" fn(*const std::ffi::c_char) -> i32;
#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
type NvtxPop = unsafe extern "C" fn() -> i32;

#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
struct NvtxBackend {
    _library: Library,
    push: NvtxPush,
    pop: NvtxPop,
}

#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
fn nvtx_enabled() -> bool {
    env::var(NVTX_ENV)
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
fn backend_pop() {
    if let Some(backend) = backend() {
        let _ = unsafe { (backend.pop)() };
    }
}

#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
fn backend() -> Option<&'static NvtxBackend> {
    static INIT: Once = Once::new();
    static BACKEND: OnceLock<Option<NvtxBackend>> = OnceLock::new();
    INIT.call_once(|| {
        let _ = BACKEND.set(load_backend());
    });
    BACKEND.get().and_then(Option::as_ref)
}

#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
fn load_backend() -> Option<NvtxBackend> {
    const LIB_NAMES: &[&str] = &[
        "libnvToolsExt.so.1",
        "libnvToolsExt.so",
        "nvToolsExt64_1.dll",
        "nvToolsExt64.dll",
    ];

    LIB_NAMES.iter().find_map(|name| {
        let library = unsafe { Library::new(name).ok()? };
        let (push, pop) = unsafe {
            let push: Symbol<'_, NvtxPush> = library.get(b"nvtxRangePushA\0").ok()?;
            let pop: Symbol<'_, NvtxPop> = library.get(b"nvtxRangePop\0").ok()?;
            (*push, *pop)
        };
        Some(NvtxBackend {
            _library: library,
            push,
            pop,
        })
    })
}

#[cfg(any(test, feature = "test-nvtx-recorder"))]
thread_local! {
    static TEST_EVENTS: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
}

#[cfg(any(test, feature = "test-nvtx-recorder"))]
fn test_recorder_active() -> bool {
    TEST_EVENTS.with(|events| events.borrow().is_some())
}

#[cfg(any(test, feature = "test-nvtx-recorder"))]
fn record_test_event(kind: &str, stage: &str) {
    TEST_EVENTS.with(|events| {
        if let Some(events) = events.borrow_mut().as_mut() {
            events.push(format!("{kind}:{stage}"));
        }
    });
}

/// Runs a closure while recording test NVTX events.
#[cfg(any(test, feature = "test-nvtx-recorder"))]
pub fn with_test_recorder<T>(f: impl FnOnce() -> T) -> (T, Vec<String>) {
    TEST_EVENTS.with(|events| {
        *events.borrow_mut() = Some(Vec::new());
    });
    let result = f();
    let events = TEST_EVENTS.with(|events| events.borrow_mut().take().unwrap_or_default());
    (result, events)
}

#[cfg(test)]
mod tests;
