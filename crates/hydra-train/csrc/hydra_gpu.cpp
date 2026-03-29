#include <torch/torch.h>
#include <ATen/Context.h>

#define HYDRA_PROTECT(x) \
  try { x } catch (const std::exception&) { }

extern "C" {

void hydra_set_allow_tf32_cublas(int b) {
  HYDRA_PROTECT(at::globalContext().setAllowTF32CuBLAS(b);)
}

void hydra_set_allow_tf32_cudnn(int b) {
  HYDRA_PROTECT(at::globalContext().setAllowTF32CuDNN(b);)
}

}
