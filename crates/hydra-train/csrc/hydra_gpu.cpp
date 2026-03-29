#include <torch/torch.h>
#include <ATen/Context.h>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#endif

#define HYDRA_PROTECT(x) \
  try { x } catch (const std::exception&) { }

extern "C" {

void hydra_set_allow_tf32_cublas(int b) {
  HYDRA_PROTECT(at::globalContext().setAllowTF32CuBLAS(b);)
}

void hydra_set_allow_tf32_cudnn(int b) {
  HYDRA_PROTECT(at::globalContext().setAllowTF32CuDNN(b);)
}

#ifdef USE_CUDA

void* hydra_cuda_graph_new(int keep_graph) {
  HYDRA_PROTECT(return new at::cuda::CUDAGraph(keep_graph != 0);)
  return nullptr;
}

int hydra_cuda_graph_capture_begin(void* g, uint64_t pool_first, uint64_t pool_second) {
  HYDRA_PROTECT(
    static_cast<at::cuda::CUDAGraph*>(g)->capture_begin(
      MempoolId_t{pool_first, pool_second});
    return 0;
  )
  return -1;
}

int hydra_cuda_graph_capture_end(void* g) {
  HYDRA_PROTECT(
    static_cast<at::cuda::CUDAGraph*>(g)->capture_end();
    return 0;
  )
  return -1;
}

void hydra_cuda_graph_replay(void* g) {
  HYDRA_PROTECT(static_cast<at::cuda::CUDAGraph*>(g)->replay();)
}

void hydra_cuda_graph_reset(void* g) {
  HYDRA_PROTECT(static_cast<at::cuda::CUDAGraph*>(g)->reset();)
}

void hydra_cuda_graph_free(void* g) {
  delete static_cast<at::cuda::CUDAGraph*>(g);
}

void hydra_cuda_stream_from_pool(int64_t device_index,
    int64_t* stream_id, int64_t* device_idx_out, int64_t* device_type) {
  HYDRA_PROTECT(
    auto stream = c10::cuda::getStreamFromPool(
      false, static_cast<c10::DeviceIndex>(device_index));
    auto packed = stream.pack3();
    *stream_id = packed.stream_id;
    *device_idx_out = packed.device_index;
    *device_type = static_cast<int64_t>(packed.device_type);
  )
}

void hydra_cuda_stream_get_current(int64_t device_index,
    int64_t* stream_id, int64_t* device_idx_out, int64_t* device_type) {
  HYDRA_PROTECT(
    auto stream = c10::cuda::getCurrentCUDAStream(
      static_cast<c10::DeviceIndex>(device_index));
    auto packed = stream.pack3();
    *stream_id = packed.stream_id;
    *device_idx_out = packed.device_index;
    *device_type = static_cast<int64_t>(packed.device_type);
  )
}

void hydra_cuda_stream_set_current(int64_t stream_id,
    int64_t device_idx, int64_t device_type) {
  HYDRA_PROTECT(
    auto stream = c10::cuda::CUDAStream::unpack3(
      stream_id, static_cast<c10::DeviceIndex>(device_idx),
      static_cast<c10::DeviceType>(device_type));
    c10::cuda::setCurrentCUDAStream(stream);
  )
}

void hydra_cuda_stream_synchronize(int64_t stream_id,
    int64_t device_idx, int64_t device_type) {
  HYDRA_PROTECT(
    auto stream = c10::cuda::CUDAStream::unpack3(
      stream_id, static_cast<c10::DeviceIndex>(device_idx),
      static_cast<c10::DeviceType>(device_type));
    stream.synchronize();
  )
}

#endif // USE_CUDA

}
