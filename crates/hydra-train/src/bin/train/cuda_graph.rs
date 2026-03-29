use std::ffi::c_void;

unsafe extern "C" {
    fn hydra_cuda_graph_new(keep_graph: i32) -> *mut c_void;
    fn hydra_cuda_graph_capture_begin(g: *mut c_void, pool_first: u64, pool_second: u64) -> i32;
    fn hydra_cuda_graph_capture_end(g: *mut c_void) -> i32;
    fn hydra_cuda_graph_replay(g: *mut c_void);
    fn hydra_cuda_graph_reset(g: *mut c_void);
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct CudaStream {
    stream_id: i64,
    device_index: i64,
    device_type: i64,
}

impl CudaStream {
    pub(crate) fn from_pool(device_index: i64) -> Self {
        let mut s = Self {
            stream_id: 0,
            device_index: 0,
            device_type: 0,
        };
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
        let mut s = Self {
            stream_id: 0,
            device_index: 0,
            device_type: 0,
        };
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
            hydra_cuda_stream_set_current(self.stream_id, self.device_index, self.device_type)
        }
    }

    pub(crate) fn synchronize(&self) {
        unsafe {
            hydra_cuda_stream_synchronize(self.stream_id, self.device_index, self.device_type)
        }
    }
}

pub(crate) struct CudaGraph {
    ptr: *mut c_void,
}

impl CudaGraph {
    pub(crate) fn new(keep_graph: bool) -> Self {
        let ptr = unsafe { hydra_cuda_graph_new(i32::from(keep_graph)) };
        assert!(!ptr.is_null(), "CUDAGraph allocation failed");
        Self { ptr }
    }

    pub(crate) fn capture_begin(&self, pool: (u64, u64)) -> Result<(), String> {
        let rc = unsafe { hydra_cuda_graph_capture_begin(self.ptr, pool.0, pool.1) };
        if rc == 0 {
            Ok(())
        } else {
            Err("CUDA graph capture_begin failed".into())
        }
    }

    pub(crate) fn capture_end(&self) -> Result<(), String> {
        let rc = unsafe { hydra_cuda_graph_capture_end(self.ptr) };
        if rc == 0 {
            Ok(())
        } else {
            Err("CUDA graph capture_end failed".into())
        }
    }

    pub(crate) fn replay(&self) {
        unsafe { hydra_cuda_graph_replay(self.ptr) }
    }

    #[allow(dead_code)]
    pub(crate) fn reset(&self) {
        unsafe { hydra_cuda_graph_reset(self.ptr) }
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe { hydra_cuda_graph_free(self.ptr) }
    }
}
