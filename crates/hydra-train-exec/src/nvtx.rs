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
    Active(u64),
    #[cfg(any(test, feature = "test-nvtx-recorder"))]
    Recorded(&'static str),
}

impl Drop for NvtxRangeGuard {
    fn drop(&mut self) {
        match &self.0 {
            GuardState::Noop => {}
            #[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
            GuardState::Active(range_id) => backend_pop(*range_id),
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

    let range_id = unsafe { (backend.start)(name.as_ptr()) };
    if range_id != 0 {
        NvtxRangeGuard(GuardState::Active(range_id))
    } else {
        NvtxRangeGuard(GuardState::Noop)
    }
}

#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
type NvtxRangeStart = unsafe extern "C" fn(*const std::ffi::c_char) -> u64;
#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
type NvtxRangeEnd = unsafe extern "C" fn(u64) -> i32;

#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
struct NvtxBackend {
    _library: Library,
    start: NvtxRangeStart,
    end: NvtxRangeEnd,
}

#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
fn nvtx_enabled() -> bool {
    env::var(NVTX_ENV)
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

#[cfg(all(not(test), not(feature = "test-nvtx-recorder")))]
fn backend_pop(range_id: u64) {
    if let Some(backend) = backend() {
        let _ = unsafe { (backend.end)(range_id) };
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
