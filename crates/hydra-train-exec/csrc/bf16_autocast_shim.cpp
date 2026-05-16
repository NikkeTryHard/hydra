#include <cstdint>
#include <exception>

#include <ATen/autocast_mode.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>

namespace {

int hydra_scalar_type_to_int(at::ScalarType dtype) {
  return static_cast<int>(dtype);
}

at::ScalarType hydra_int_to_scalar_type(int dtype) {
  return static_cast<at::ScalarType>(dtype);
}

}  // namespace

extern "C" {

struct HydraCudaAutocastState {
  int32_t enabled;
  int32_t dtype;
  int32_t cache_enabled;
};

int hydra_cuda_autocast_get_state(HydraCudaAutocastState* state_out) {
  if (state_out == nullptr) {
    return -1;
  }
  try {
    state_out->enabled = at::autocast::is_autocast_enabled(at::kCUDA) ? 1 : 0;
    state_out->dtype = hydra_scalar_type_to_int(
        at::autocast::get_autocast_dtype(at::kCUDA));
    state_out->cache_enabled = at::autocast::is_autocast_cache_enabled() ? 1 : 0;
    return 0;
  } catch (const std::exception&) {
    return -1;
  } catch (...) {
    return -1;
  }
}

int hydra_cuda_autocast_restore_state(const HydraCudaAutocastState* state) {
  if (state == nullptr) {
    return -1;
  }
  try {
    at::autocast::set_autocast_enabled(at::kCUDA, state->enabled != 0);
    at::autocast::set_autocast_dtype(
        at::kCUDA, hydra_int_to_scalar_type(state->dtype));
    at::autocast::set_autocast_cache_enabled(state->cache_enabled != 0);
    return 0;
  } catch (const std::exception&) {
    return -1;
  } catch (...) {
    return -1;
  }
}

int hydra_cuda_autocast_enter_bf16(HydraCudaAutocastState* previous_out) {
  if (previous_out == nullptr) {
    return -1;
  }
  if (hydra_cuda_autocast_get_state(previous_out) != 0) {
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

int hydra_cuda_autocast_exit(const HydraCudaAutocastState* previous) {
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
  return hydra_cuda_autocast_restore_state(previous);
}

int hydra_cuda_autocast_get_dtype(int* dtype_out) {
  if (dtype_out == nullptr) {
    return -1;
  }
  try {
    *dtype_out = hydra_scalar_type_to_int(
        at::autocast::get_autocast_dtype(at::kCUDA));
    return 0;
  } catch (const std::exception&) {
    return -1;
  } catch (...) {
    return -1;
  }
}

int hydra_cuda_autocast_set_dtype(int dtype) {
  try {
    at::autocast::set_autocast_dtype(at::kCUDA, hydra_int_to_scalar_type(dtype));
    return 0;
  } catch (const std::exception&) {
    return -1;
  } catch (...) {
    return -1;
  }
}

int hydra_cuda_autocast_set_bf16() {
  try {
    at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
    return 0;
  } catch (const std::exception&) {
    return -1;
  } catch (...) {
    return -1;
  }
}

}
