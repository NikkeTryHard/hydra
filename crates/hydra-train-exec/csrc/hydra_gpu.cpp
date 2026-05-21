#include <cstdint>
#include <exception>
#include <string>

#ifdef __has_include
#if __has_include(<ATen/Context.h>)
#include <ATen/Context.h>
#define HYDRA_HAS_TORCH 1
#else
#define HYDRA_HAS_TORCH 0
#endif
#else
#define HYDRA_HAS_TORCH 0
#endif

#if HYDRA_HAS_TORCH && defined(HYDRA_USE_CUDA_GRAPH)
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#define HYDRA_HAS_CUDA_GRAPH 1
#else
#define HYDRA_HAS_CUDA_GRAPH 0
#endif

#define HYDRA_PROTECT(x) \
  try { x } catch (const std::exception&) { }

#define HYDRA_PROTECT_ERR(x) \
  try { x } catch (const std::exception& e) { hydra_last_error_message = e.what(); return -1; } catch (...) { hydra_last_error_message = "unknown C++ exception"; return -1; }

extern "C" {

#if HYDRA_HAS_TORCH

void hydra_set_tf32_precision(int b) {
  HYDRA_PROTECT(
    if (b != 0) {
      at::globalContext().setFloat32Precision("cuda", "matmul", "tf32");
      at::globalContext().setFloat32Precision("cudnn", "conv", "tf32");
    } else {
      at::globalContext().setFloat32Precision("cuda", "matmul", "ieee");
      at::globalContext().setFloat32Precision("cudnn", "conv", "ieee");
    }
  )
}

#else

void hydra_set_tf32_precision(int) {}

#endif

#ifdef HYDRA_ENABLE_CUDA_GRAPH_FFI

#if HYDRA_HAS_CUDA_GRAPH
int hydra_cuda_graph_backend_kind() {
  return 1;
}


namespace {
static thread_local std::string hydra_last_error_message;


void hydra_pack_stream(
    const c10::cuda::CUDAStream& stream,
    int64_t* stream_id,
    int64_t* device_idx_out,
    int64_t* device_type) {
  const auto packed = stream.pack3();
  if (stream_id != nullptr) {
    *stream_id = packed.stream_id;
  }
  if (device_idx_out != nullptr) {
    *device_idx_out = packed.device_index;
  }
  if (device_type != nullptr) {
    *device_type = static_cast<int64_t>(packed.device_type);
  }
}

}
const char* hydra_cuda_last_exception_message() {
  return hydra_last_error_message.c_str();
}

void* hydra_cuda_graph_new(int keep_graph) {
  try {
    return new at::cuda::CUDAGraph(keep_graph != 0);
  } catch (const std::exception&) {
    return nullptr;
  }
}

int hydra_cuda_graph_capture_begin(void* graph, uint64_t pool_first, uint64_t pool_second) {
  HYDRA_PROTECT_ERR(
      static_cast<at::cuda::CUDAGraph*>(graph)->capture_begin(
          at::cuda::MempoolId_t{pool_first, pool_second});
      return 0;)
}

int hydra_cuda_graph_capture_end(void* graph) {
  HYDRA_PROTECT_ERR(
      static_cast<at::cuda::CUDAGraph*>(graph)->capture_end();
      return 0;)
}

int hydra_cuda_graph_replay(void* graph) {
  HYDRA_PROTECT_ERR(
      static_cast<at::cuda::CUDAGraph*>(graph)->replay();
      return 0;)
}

int hydra_cuda_graph_reset(void* graph) {
  HYDRA_PROTECT_ERR(
      static_cast<at::cuda::CUDAGraph*>(graph)->reset();
      return 0;)
}

int hydra_cuda_last_error_code() {
  return static_cast<int>(cudaGetLastError());
}

const char* hydra_cuda_error_name(int code) {
  return cudaGetErrorName(static_cast<cudaError_t>(code));
}

const char* hydra_cuda_error_string(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}


int hydra_cuda_device_synchronize() {
  HYDRA_PROTECT_ERR(
      cudaDeviceSynchronize();
      return 0;)
}
void hydra_cuda_graph_free(void* graph) {
  delete static_cast<at::cuda::CUDAGraph*>(graph);
}

void hydra_cuda_stream_from_pool(
    int64_t device_index,
    int64_t* stream_id,
    int64_t* device_idx_out,
    int64_t* device_type) {
  HYDRA_PROTECT(
      hydra_pack_stream(
          c10::cuda::getStreamFromPool(false, static_cast<c10::DeviceIndex>(device_index)),
          stream_id,
          device_idx_out,
          device_type);)
}

void hydra_cuda_stream_get_current(
    int64_t device_index,
    int64_t* stream_id,
    int64_t* device_idx_out,
    int64_t* device_type) {
  HYDRA_PROTECT(
      hydra_pack_stream(
          c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_index)),
          stream_id,
          device_idx_out,
          device_type);)
}

void hydra_cuda_stream_set_current(int64_t stream_id, int64_t device_idx, int64_t device_type) {
  HYDRA_PROTECT(
      const auto stream = c10::cuda::CUDAStream::unpack3(
          static_cast<c10::StreamId>(stream_id),
          static_cast<c10::DeviceIndex>(device_idx),
          static_cast<c10::DeviceType>(device_type));
      c10::cuda::setCurrentCUDAStream(stream);)
}




void hydra_cuda_stream_synchronize(int64_t stream_id, int64_t device_idx, int64_t device_type) {
  HYDRA_PROTECT(
      c10::cuda::CUDAStream::unpack3(
          static_cast<c10::StreamId>(stream_id),
          static_cast<c10::DeviceIndex>(device_idx),
          static_cast<c10::DeviceType>(device_type))
          .synchronize();)
}

// ---------------------------------------------------------------------------
// CUDA event primitives
// ---------------------------------------------------------------------------

void* hydra_cuda_event_create(int enable_timing) {
  cudaEvent_t event = nullptr;
  unsigned flags = cudaEventDisableTiming;
  if (enable_timing) {
    flags = cudaEventDefault;
  }
  if (cudaEventCreateWithFlags(&event, flags) != cudaSuccess) {
    return nullptr;
  }
  return static_cast<void*>(event);
}

void hydra_cuda_event_destroy(void* event) {
  if (event != nullptr) {
    cudaEventDestroy(static_cast<cudaEvent_t>(event));
  }
}

int hydra_cuda_event_record(void* event, int64_t stream_id, int64_t device_idx, int64_t device_type) {
  HYDRA_PROTECT_ERR(
      auto stream = c10::cuda::CUDAStream::unpack3(
          static_cast<c10::StreamId>(stream_id),
          static_cast<c10::DeviceIndex>(device_idx),
          static_cast<c10::DeviceType>(device_type));
      if (cudaEventRecord(static_cast<cudaEvent_t>(event), stream.stream()) != cudaSuccess) {
        return -1;
      }
      return 0;)
}

int hydra_cuda_event_synchronize(void* event) {
  if (cudaEventSynchronize(static_cast<cudaEvent_t>(event)) != cudaSuccess) {
    return -1;
  }
  return 0;
}

int hydra_cuda_event_query(void* event) {
  const auto status = cudaEventQuery(static_cast<cudaEvent_t>(event));
  if (status == cudaSuccess) {
    return 1;
  }
  if (status == cudaErrorNotReady) {
    return 0;
  }
  return -1;
}

int hydra_cuda_event_elapsed_ms(void* start, void* end, float* elapsed_ms) {
  if (cudaEventElapsedTime(elapsed_ms, static_cast<cudaEvent_t>(start), static_cast<cudaEvent_t>(end)) != cudaSuccess) {
    return -1;
  }
  return 0;
}

int hydra_cuda_stream_wait_event(int64_t stream_id, int64_t device_idx, int64_t device_type, void* event) {
  HYDRA_PROTECT_ERR(
      auto stream = c10::cuda::CUDAStream::unpack3(
          static_cast<c10::StreamId>(stream_id),
          static_cast<c10::DeviceIndex>(device_idx),
          static_cast<c10::DeviceType>(device_type));
      if (cudaStreamWaitEvent(stream.stream(), static_cast<cudaEvent_t>(event), 0) != cudaSuccess) {
        return -1;
      }
      return 0;)
}

// ---------------------------------------------------------------------------
// Pinned host memory + async H2D
// ---------------------------------------------------------------------------

void* hydra_pinned_malloc(uint64_t size_bytes) {
  void* ptr = nullptr;
  if (cudaHostAlloc(&ptr, static_cast<size_t>(size_bytes), cudaHostAllocDefault) != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

void hydra_pinned_free(void* ptr) {
  if (ptr != nullptr) {
    cudaFreeHost(ptr);
  }
}

int hydra_memcpy_async_h2d(void* dst, const void* src, uint64_t size_bytes, int64_t stream_id, int64_t device_idx, int64_t device_type) {
  HYDRA_PROTECT_ERR(
      auto stream = c10::cuda::CUDAStream::unpack3(
          static_cast<c10::StreamId>(stream_id),
          static_cast<c10::DeviceIndex>(device_idx),
          static_cast<c10::DeviceType>(device_type));
      if (cudaMemcpyAsync(dst, src, static_cast<size_t>(size_bytes), cudaMemcpyHostToDevice, stream.stream()) != cudaSuccess) {
        return -1;
      }
      return 0;)
}

#else

int hydra_cuda_graph_backend_kind() {
  return 0;
}

const char* hydra_cuda_last_exception_message() {
  return "CUDA graph FFI unavailable";
}

void* hydra_cuda_graph_new(int) { return nullptr; }
int hydra_cuda_graph_capture_begin(void*, uint64_t, uint64_t) { return -1; }
int hydra_cuda_graph_capture_end(void*) { return -1; }
int hydra_cuda_graph_replay(void*) { return -1; }
int hydra_cuda_graph_reset(void*) { return -1; }
void hydra_cuda_graph_free(void*) {}
int hydra_cuda_last_error_code() { return -1; }
const char* hydra_cuda_error_name(int) { return "cuda_graph_unavailable"; }
const char* hydra_cuda_error_string(int) { return "CUDA graph FFI unavailable"; }
int hydra_cuda_device_synchronize() { return -1; }
void hydra_cuda_stream_from_pool(int64_t, int64_t*, int64_t*, int64_t*) {}
void hydra_cuda_stream_get_current(int64_t, int64_t*, int64_t*, int64_t*) {}
void hydra_cuda_stream_set_current(int64_t, int64_t, int64_t) {}
void hydra_cuda_stream_synchronize(int64_t, int64_t, int64_t) {}
void* hydra_cuda_event_create(int) { return nullptr; }
void hydra_cuda_event_destroy(void*) {}
int hydra_cuda_event_record(void*, int64_t, int64_t, int64_t) { return -1; }
int hydra_cuda_event_synchronize(void*) { return -1; }
int hydra_cuda_event_query(void*) { return -1; }
int hydra_cuda_event_elapsed_ms(void*, void*, float*) { return -1; }
int hydra_cuda_stream_wait_event(int64_t, int64_t, int64_t, void*) { return -1; }
void* hydra_pinned_malloc(uint64_t) { return nullptr; }
void hydra_pinned_free(void*) {}
int hydra_memcpy_async_h2d(void*, const void*, uint64_t, int64_t, int64_t, int64_t) { return -1; }

#endif

#endif

}
