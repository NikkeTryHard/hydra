#![cfg(feature = "cuda-graph")]
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
