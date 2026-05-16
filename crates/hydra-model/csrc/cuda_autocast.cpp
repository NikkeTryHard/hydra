#include <cstdint>
#include <exception>

#include <ATen/autocast_mode.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>

extern "C" {

struct HydraModelCudaAutocastState {
  int32_t enabled;
  int32_t dtype;
  int32_t cache_enabled;
};

int hydra_model_cuda_autocast_get_state(HydraModelCudaAutocastState* state_out) {
  if (state_out == nullptr) {
    return -1;
  }
  try {
    state_out->enabled = at::autocast::is_autocast_enabled(at::kCUDA) ? 1 : 0;
    state_out->dtype = static_cast<int32_t>(at::autocast::get_autocast_dtype(at::kCUDA));
    state_out->cache_enabled = at::autocast::is_autocast_cache_enabled() ? 1 : 0;
    return 0;
  } catch (const std::exception&) {
    return -1;
  } catch (...) {
    return -1;
  }
}

int hydra_model_cuda_autocast_restore_state(const HydraModelCudaAutocastState* state) {
  if (state == nullptr) {
    return -1;
  }
  try {
    at::autocast::set_autocast_enabled(at::kCUDA, state->enabled != 0);
    at::autocast::set_autocast_dtype(at::kCUDA, static_cast<at::ScalarType>(state->dtype));
    at::autocast::set_autocast_cache_enabled(state->cache_enabled != 0);
    return 0;
  } catch (const std::exception&) {
    return -1;
  } catch (...) {
    return -1;
  }
}

int hydra_model_cuda_autocast_enter_bf16(HydraModelCudaAutocastState* previous_out) {
  if (previous_out == nullptr) {
    return -1;
  }
  if (hydra_model_cuda_autocast_get_state(previous_out) != 0) {
    return -1;
  }
  try {
    at::autocast::increment_nesting();
    at::autocast::set_autocast_enabled(at::kCUDA, true);
    at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
    at::autocast::set_autocast_cache_enabled(true);
    return 0;
  } catch (const std::exception&) {
    return -1;
  } catch (...) {
    return -1;
  }
}

int hydra_model_cuda_autocast_exit(const HydraModelCudaAutocastState* previous) {
  if (previous == nullptr) {
    return -1;
  }
  try {
    if (at::autocast::decrement_nesting() == 0) {
      at::autocast::clear_cache();
    }
  } catch (const std::exception&) {
    return -1;
  } catch (...) {
    return -1;
  }
  return hydra_model_cuda_autocast_restore_state(previous);
}

}  // extern "C"
