#![allow(dead_code)]

use std::ffi::{c_int, c_void};
use std::ptr::NonNull;

unsafe extern "C" {
    fn hydra_cuda_graph_new(keep_graph: c_int) -> *mut c_void;
    fn hydra_cuda_graph_capture_begin(g: *mut c_void, pool_first: u64, pool_second: u64) -> c_int;
    fn hydra_cuda_graph_capture_end(g: *mut c_void) -> c_int;
    fn hydra_cuda_graph_replay(g: *mut c_void) -> c_int;
    fn hydra_cuda_graph_reset(g: *mut c_void) -> c_int;
    fn hydra_cuda_graph_free(g: *mut c_void);

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CudaStream {
    stream_id: i64,
    device_index: i64,
    device_type: i64,
}

impl CudaStream {
    pub(crate) fn from_pool(device_index: i64) -> Self {
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

    pub(crate) fn current(device_index: i64) -> Self {
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

    pub(crate) fn set_current(&self) {
        unsafe {
            hydra_cuda_stream_set_current(self.stream_id, self.device_index, self.device_type);
        }
    }

    pub(crate) fn synchronize(&self) {
        unsafe {
            hydra_cuda_stream_synchronize(self.stream_id, self.device_index, self.device_type);
        }
    }
}

pub(crate) struct CudaGraph {
    ptr: NonNull<c_void>,
}

impl CudaGraph {
    pub(crate) fn new(keep_graph: bool) -> Self {
        let ptr = NonNull::new(unsafe { hydra_cuda_graph_new(c_int::from(keep_graph)) })
            .expect("CUDA graph backend unavailable");
        Self { ptr }
    }

    pub(crate) fn capture_begin(&self, pool: (u64, u64)) {
        check_cuda_graph_status(
            unsafe { hydra_cuda_graph_capture_begin(self.ptr.as_ptr(), pool.0, pool.1) },
            "capture_begin",
        );
    }

    pub(crate) fn capture_end(&self) {
        check_cuda_graph_status(
            unsafe { hydra_cuda_graph_capture_end(self.ptr.as_ptr()) },
            "capture_end",
        );
    }

    pub(crate) fn replay(&self) {
        check_cuda_graph_status(
            unsafe { hydra_cuda_graph_replay(self.ptr.as_ptr()) },
            "replay",
        );
    }

    pub(crate) fn reset(&self) {
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
    assert_eq!(status, 0, "CUDA graph {op} failed");
}

fn check_cuda_status(status: c_int, op: &str) {
    assert_eq!(status, 0, "CUDA {op} failed");
}

pub(crate) struct CudaEvent {
    ptr: NonNull<c_void>,
}

impl CudaEvent {
    pub(crate) fn new(enable_timing: bool) -> Self {
        let ptr = NonNull::new(unsafe { hydra_cuda_event_create(c_int::from(enable_timing)) })
            .expect("CUDA event creation failed");
        Self { ptr }
    }

    pub(crate) fn record(&self, stream: &CudaStream) {
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

    pub(crate) fn synchronize(&self) {
        check_cuda_status(
            unsafe { hydra_cuda_event_synchronize(self.ptr.as_ptr()) },
            "event_synchronize",
        );
    }

    /// Returns `true` if the event has completed, `false` if still pending.
    pub(crate) fn query(&self) -> bool {
        let status = unsafe { hydra_cuda_event_query(self.ptr.as_ptr()) };
        match status {
            0 => true,
            1 => false,
            _ => panic!("CUDA event_query failed"),
        }
    }

    /// Elapsed milliseconds between two timing-enabled events.
    pub(crate) fn elapsed_ms(start: &CudaEvent, end: &CudaEvent) -> f32 {
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
    pub(crate) fn wait_event(&self, event: &CudaEvent) {
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

pub(crate) struct PinnedBuffer {
    ptr: NonNull<c_void>,
    len: usize,
}

impl PinnedBuffer {
    pub(crate) fn new(size_bytes: usize) -> Self {
        assert!(size_bytes > 0, "PinnedBuffer size must be > 0");
        let ptr = NonNull::new(unsafe { hydra_pinned_malloc(size_bytes as u64) })
            .expect("CUDA pinned malloc failed");
        Self {
            ptr,
            len: size_bytes,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr().cast()
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr().cast()
    }

    /// Copy `count` bytes from this pinned host buffer into `dst` (device ptr)
    /// on `stream`. The caller owns the device pointer and must ensure it is
    /// valid and large enough.
    pub(crate) fn copy_to_device_async(&self, dst: *mut c_void, count: usize, stream: &CudaStream) {
        assert!(
            count <= self.len,
            "copy_to_device_async: count exceeds buffer length"
        );
        check_cuda_status(
            unsafe {
                hydra_memcpy_async_h2d(
                    dst,
                    self.ptr.as_ptr(),
                    count as u64,
                    stream.stream_id,
                    stream.device_index,
                    stream.device_type,
                )
            },
            "memcpy_async_h2d",
        );
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe { hydra_pinned_free(self.ptr.as_ptr()) }
    }
}
