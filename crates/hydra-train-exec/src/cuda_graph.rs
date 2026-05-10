#![allow(
    dead_code,
    reason = "CUDA graph bindings are exercised through FFI and runtime feature gates"
)]

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr::NonNull;

unsafe extern "C" {
    fn hydra_cuda_graph_backend_kind() -> c_int;
    fn hydra_cuda_graph_new(keep_graph: c_int) -> *mut c_void;
    fn hydra_cuda_graph_capture_begin(g: *mut c_void, pool_first: u64, pool_second: u64) -> c_int;
    fn hydra_cuda_graph_capture_end(g: *mut c_void) -> c_int;
    fn hydra_cuda_graph_replay(g: *mut c_void) -> c_int;
    fn hydra_cuda_graph_reset(g: *mut c_void) -> c_int;
    fn hydra_cuda_graph_free(g: *mut c_void);
    fn hydra_cuda_last_error_code() -> c_int;
    fn hydra_cuda_error_name(code: c_int) -> *const c_char;
    fn hydra_cuda_error_string(code: c_int) -> *const c_char;
    fn hydra_cuda_device_synchronize() -> c_int;
    fn hydra_cuda_last_exception_message() -> *const c_char;

    fn hydra_cuda_stream_from_pool(
        device_index: i64,
        stream_id: *mut i64,
        device_idx_out: *mut i64,
        device_type: *mut i64,
    );
    fn hydra_cuda_stream_get_current(
        device_index: i64,
        stream_id: *mut i64,
        device_idx_out: *mut i64,
        device_type: *mut i64,
    );
    fn hydra_cuda_stream_set_current(stream_id: i64, device_idx: i64, device_type: i64);
    fn hydra_cuda_stream_synchronize(stream_id: i64, device_idx: i64, device_type: i64);

    fn hydra_cuda_event_create(enable_timing: c_int) -> *mut c_void;
    fn hydra_cuda_event_destroy(event: *mut c_void);
    fn hydra_cuda_event_record(
        event: *mut c_void,
        stream_id: i64,
        device_idx: i64,
        device_type: i64,
    ) -> c_int;
    fn hydra_cuda_event_synchronize(event: *mut c_void) -> c_int;
    fn hydra_cuda_event_query(event: *mut c_void) -> c_int;
    fn hydra_cuda_event_elapsed_ms(
        start: *mut c_void,
        end: *mut c_void,
        elapsed_ms: *mut f32,
    ) -> c_int;
    fn hydra_cuda_stream_wait_event(
        stream_id: i64,
        device_idx: i64,
        device_type: i64,
        event: *mut c_void,
    ) -> c_int;

    fn hydra_pinned_malloc(size_bytes: u64) -> *mut c_void;
    fn hydra_pinned_free(ptr: *mut c_void);
    fn hydra_memcpy_async_h2d(
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: u64,
        stream_id: i64,
        device_idx: i64,
        device_type: i64,
    ) -> c_int;
}

const CUDA_GRAPH_BACKEND_REAL: c_int = 1;

fn cuda_graph_backend_kind() -> c_int {
    unsafe { hydra_cuda_graph_backend_kind() }
}

fn cuda_graph_backend_label() -> &'static str {
    match cuda_graph_backend_kind() {
        CUDA_GRAPH_BACKEND_REAL => "real CUDA graph/pinned FFI",
        _ => "stub/unavailable CUDA graph FFI",
    }
}

fn cuda_graph_backend_guidance() -> &'static str {
    match cuda_graph_backend_kind() {
        CUDA_GRAPH_BACKEND_REAL => {
            "real CUDA FFI is active; for allocation failures check pinned-memory pressure, batch size, driver/runtime health, and concurrent GPU processes"
        }
        _ => {
            "rebuild hydra-train with --features cuda-graph and a complete CUDA/libtorch setup, e.g. CUDA_HOME=/opt/cuda LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1; build.rs must find CUDA headers/libs and link cudart plus c10_cuda"
        }
    }
}

fn assert_cuda_graph_backend_available(op: &str) {
    assert_eq!(
        cuda_graph_backend_kind(),
        CUDA_GRAPH_BACKEND_REAL,
        "CUDA {op} unavailable: backend={} ({})",
        cuda_graph_backend_label(),
        cuda_graph_backend_guidance()
    );
}

/// Handle to a CUDA stream identified by the FFI bridge.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaStream {
    stream_id: i64,
    device_index: i64,
    device_type: i64,
}

impl CudaStream {
    /// Returns a pooled CUDA stream for `device_index`.
    pub fn from_pool(device_index: i64) -> Self {
        assert_cuda_graph_backend_available("stream_from_pool");
        let mut s = Self::default();
        unsafe {
            hydra_cuda_stream_from_pool(
                device_index,
                &mut s.stream_id,
                &mut s.device_index,
                &mut s.device_type,
            );
        }
        s
    }

    /// Returns the current CUDA stream for `device_index`.
    pub fn current(device_index: i64) -> Self {
        assert_cuda_graph_backend_available("stream_get_current");
        let mut s = Self::default();
        unsafe {
            hydra_cuda_stream_get_current(
                device_index,
                &mut s.stream_id,
                &mut s.device_index,
                &mut s.device_type,
            );
        }
        s
    }

    /// Makes this stream current on its device.
    pub fn set_current(&self) {
        unsafe {
            hydra_cuda_stream_set_current(self.stream_id, self.device_index, self.device_type);
        }
    }

    /// Blocks until this stream completes queued work.
    pub fn synchronize(&self) {
        unsafe {
            hydra_cuda_stream_synchronize(self.stream_id, self.device_index, self.device_type);
        }
    }
}

/// Synchronizes the active CUDA device through the FFI bridge.
pub fn synchronize_device() -> Result<(), String> {
    cuda_result(
        unsafe { hydra_cuda_device_synchronize() },
        "device_synchronize",
    )
}

/// RAII wrapper for a CUDA graph capture/replay handle.
pub struct CudaGraph {
    ptr: NonNull<c_void>,
}

impl CudaGraph {
    /// Allocates a CUDA graph wrapper.
    pub fn new(keep_graph: bool) -> Self {
        assert_cuda_graph_backend_available("graph_new");
        let ptr = NonNull::new(unsafe { hydra_cuda_graph_new(c_int::from(keep_graph)) })
            .unwrap_or_else(|| {
                panic!(
                    "CUDA graph_new failed: backend={} ({})",
                    cuda_graph_backend_label(),
                    cuda_graph_backend_guidance()
                )
            });
        Self { ptr }
    }

    /// Begins graph capture and panics on failure.
    pub fn capture_begin(&self, pool: (u64, u64)) {
        check_cuda_graph_status(
            unsafe { hydra_cuda_graph_capture_begin(self.ptr.as_ptr(), pool.0, pool.1) },
            "capture_begin",
        );
    }

    /// Begins graph capture and returns an error on failure.
    pub fn try_capture_begin(&self, pool: (u64, u64)) -> Result<(), String> {
        cuda_graph_result(
            unsafe { hydra_cuda_graph_capture_begin(self.ptr.as_ptr(), pool.0, pool.1) },
            "capture_begin",
        )
    }

    /// Ends graph capture and panics on failure.
    pub fn capture_end(&self) {
        check_cuda_graph_status(
            unsafe { hydra_cuda_graph_capture_end(self.ptr.as_ptr()) },
            "capture_end",
        );
    }

    /// Ends graph capture and returns an error on failure.
    pub fn try_capture_end(&self) -> Result<(), String> {
        cuda_graph_result(
            unsafe { hydra_cuda_graph_capture_end(self.ptr.as_ptr()) },
            "capture_end",
        )
    }

    /// Replays the captured graph and panics on failure.
    pub fn replay(&self) {
        check_cuda_graph_status(
            unsafe { hydra_cuda_graph_replay(self.ptr.as_ptr()) },
            "replay",
        );
    }

    /// Replays the captured graph and returns an error on failure.
    pub fn try_replay(&self) -> Result<(), String> {
        cuda_graph_result(
            unsafe { hydra_cuda_graph_replay(self.ptr.as_ptr()) },
            "replay",
        )
    }

    /// Resets captured graph state.
    pub fn reset(&self) {
        check_cuda_graph_status(
            unsafe { hydra_cuda_graph_reset(self.ptr.as_ptr()) },
            "reset",
        );
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe { hydra_cuda_graph_free(self.ptr.as_ptr()) }
    }
}

fn check_cuda_graph_status(status: c_int, op: &str) {
    assert_eq!(
        status,
        0,
        "CUDA graph {op} failed: backend={} ({})",
        cuda_graph_backend_label(),
        cuda_graph_backend_guidance()
    );
}

fn cuda_graph_result(status: c_int, op: &str) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        let cuda_code = unsafe { hydra_cuda_last_error_code() };
        Err(format!(
            "CUDA graph {op} failed: backend={} ({}) status={status} cuda_last_error={} {}: {} exception={}",
            cuda_graph_backend_label(),
            cuda_graph_backend_guidance(),
            cuda_code,
            cuda_error_name(cuda_code),
            cuda_error_string(cuda_code),
            cuda_last_exception_message(),
        ))
    }
}

fn cuda_result(status: c_int, op: &str) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        let cuda_code = unsafe { hydra_cuda_last_error_code() };
        Err(format!(
            "CUDA {op} failed: backend={} ({}) status={status} cuda_last_error={} {}: {} exception={}",
            cuda_graph_backend_label(),
            cuda_graph_backend_guidance(),
            cuda_code,
            cuda_error_name(cuda_code),
            cuda_error_string(cuda_code),
            cuda_last_exception_message(),
        ))
    }
}

fn cuda_error_name(code: c_int) -> String {
    unsafe { c_string_lossy(hydra_cuda_error_name(code)) }
}

fn cuda_error_string(code: c_int) -> String {
    unsafe { c_string_lossy(hydra_cuda_error_string(code)) }
}

fn cuda_last_exception_message() -> String {
    unsafe { c_string_lossy(hydra_cuda_last_exception_message()) }
}

unsafe fn c_string_lossy(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return "<null>".to_string();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}
fn check_cuda_status(status: c_int, op: &str) {
    assert_eq!(
        status,
        0,
        "CUDA {op} failed: backend={} ({})",
        cuda_graph_backend_label(),
        cuda_graph_backend_guidance()
    );
}

/// RAII wrapper for a CUDA event.
pub struct CudaEvent {
    ptr: NonNull<c_void>,
}

impl CudaEvent {
    /// Creates a CUDA event, optionally with timing enabled.
    pub fn new(enable_timing: bool) -> Self {
        assert_cuda_graph_backend_available("event_create");
        let ptr = NonNull::new(unsafe { hydra_cuda_event_create(c_int::from(enable_timing)) })
            .unwrap_or_else(|| {
                panic!(
                    "CUDA event creation failed: backend={} ({})",
                    cuda_graph_backend_label(),
                    cuda_graph_backend_guidance()
                )
            });
        Self { ptr }
    }

    /// Records this event on `stream`.
    pub fn record(&self, stream: &CudaStream) {
        check_cuda_status(
            unsafe {
                hydra_cuda_event_record(
                    self.ptr.as_ptr(),
                    stream.stream_id,
                    stream.device_index,
                    stream.device_type,
                )
            },
            "event_record",
        );
    }

    /// Blocks until this event completes.
    pub fn synchronize(&self) {
        check_cuda_status(
            unsafe { hydra_cuda_event_synchronize(self.ptr.as_ptr()) },
            "event_synchronize",
        );
    }

    /// Returns `true` if the event has completed, `false` if still pending.
    pub fn query(&self) -> bool {
        let status = unsafe { hydra_cuda_event_query(self.ptr.as_ptr()) };
        match status {
            0 => true,
            1 => false,
            _ => panic!("CUDA event_query failed"),
        }
    }

    /// Elapsed milliseconds between two timing-enabled events.
    pub fn elapsed_ms(start: &CudaEvent, end: &CudaEvent) -> f32 {
        let mut ms: f32 = 0.0;
        check_cuda_status(
            unsafe { hydra_cuda_event_elapsed_ms(start.ptr.as_ptr(), end.ptr.as_ptr(), &mut ms) },
            "event_elapsed_ms",
        );
        ms
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe { hydra_cuda_event_destroy(self.ptr.as_ptr()) }
    }
}

impl CudaStream {
    /// Makes this stream wait until `event` completes.
    pub fn wait_event(&self, event: &CudaEvent) {
        check_cuda_status(
            unsafe {
                hydra_cuda_stream_wait_event(
                    self.stream_id,
                    self.device_index,
                    self.device_type,
                    event.ptr.as_ptr(),
                )
            },
            "stream_wait_event",
        );
    }
}

/// Page-locked host allocation owned by the CUDA FFI bridge.
pub struct PinnedBuffer {
    ptr: NonNull<c_void>,
    len: usize,
}

impl PinnedBuffer {
    /// Allocates an unnamed pinned buffer with `size_bytes` capacity.
    pub fn new(size_bytes: usize) -> Self {
        Self::new_labeled("unnamed", size_bytes)
    }

    /// Allocates a pinned buffer with a diagnostic label.
    pub fn new_labeled(label: &'static str, size_bytes: usize) -> Self {
        assert!(size_bytes > 0, "PinnedBuffer size must be > 0");
        let ptr =
            NonNull::new(unsafe { hydra_pinned_malloc(size_bytes as u64) }).unwrap_or_else(|| {
                panic!(
                    "CUDA pinned malloc failed for {label}: requested {} bytes; backend={} ({})",
                    size_bytes,
                    cuda_graph_backend_label(),
                    cuda_graph_backend_guidance()
                )
            });
        Self {
            ptr,
            len: size_bytes,
        }
    }

    /// Returns the buffer size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true when the pinned byte buffer contains no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the immutable raw byte pointer.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr().cast()
    }

    /// Returns the mutable raw byte pointer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr().cast()
    }

    /// Copy `count` bytes from this pinned host buffer into `dst` (device ptr)
    /// on `stream`. The caller owns the device pointer and must ensure it is
    /// valid and large enough.
    #[expect(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "FFI copies bytes to a CUDA device pointer without Rust dereference"
    )]
    pub fn copy_to_device_async(&self, dst: *mut c_void, count: usize, stream: &CudaStream) {
        assert!(
            count <= self.len,
            "copy_to_device_async: count exceeds buffer length"
        );
        let status = unsafe {
            hydra_memcpy_async_h2d(
                dst,
                self.ptr.as_ptr(),
                count as u64,
                stream.stream_id,
                stream.device_index,
                stream.device_type,
            )
        };
        assert_eq!(
            status,
            0,
            "CUDA memcpy_async_h2d failed for {} bytes: backend={} ({})",
            count,
            cuda_graph_backend_label(),
            cuda_graph_backend_guidance()
        );
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe { hydra_pinned_free(self.ptr.as_ptr()) }
    }
}
