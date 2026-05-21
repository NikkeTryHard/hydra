#include <exception>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>

#ifdef __has_include
#if __has_include(<torch/csrc/autograd/profiler_kineto.h>) && __has_include(<torch/csrc/profiler/orchestration/observer.h>)
#include <ATen/record_function.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/orchestration/observer.h>
#define HYDRA_HAS_TORCH_PROFILER 1
#else
#define HYDRA_HAS_TORCH_PROFILER 0
#endif
#else
#define HYDRA_HAS_TORCH_PROFILER 0
#endif

namespace {
static thread_local std::string hydra_torch_profiler_last_error;

void hydra_torch_profiler_set_error(const char* message) {
  hydra_torch_profiler_last_error = message;
}

void hydra_torch_profiler_set_error(const std::exception& error) {
  hydra_torch_profiler_last_error = error.what();
}
} // namespace

extern "C" {

const char* hydra_torch_profiler_last_exception_message() {
  return hydra_torch_profiler_last_error.c_str();
}

#if HYDRA_HAS_TORCH_PROFILER

int hydra_torch_profiler_start(int record_shapes) {
  try {
    using torch::profiler::impl::ActivityType;
    using torch::profiler::impl::ExperimentalConfig;
    using torch::profiler::impl::ProfilerConfig;
    using torch::profiler::impl::ProfilerState;

    if (torch::profiler::impl::profilerEnabled()) {
      hydra_torch_profiler_set_error("PyTorch profiler is already active");
      return -1;
    }

    ExperimentalConfig experimental_config(
        {},
        false,
        false,
        {},
        false,
        false,
        false,
        true,
        false,
        false,
        "",
        false);
    ProfilerConfig config(
        ProfilerState::KINETO,
        record_shapes != 0,
        false,
        false,
        false,
        false,
        experimental_config,
        "hydra_bc_backward");
    std::set<ActivityType> activities = {ActivityType::CPU, ActivityType::CUDA};
    torch::autograd::profiler::enableProfiler(config, activities, {});
    return 0;
  } catch (const std::exception& error) {
    hydra_torch_profiler_set_error(error);
    return -1;
  } catch (...) {
    hydra_torch_profiler_set_error("unknown C++ exception starting PyTorch profiler");
    return -1;
  }
}

int hydra_torch_profiler_stop_and_save(const char* path) {
  try {
    if (path == nullptr || path[0] == '\0') {
      hydra_torch_profiler_set_error("profile output path is empty");
      return -1;
    }
    auto result = torch::autograd::profiler::disableProfiler();
    if (!result) {
      hydra_torch_profiler_set_error("PyTorch profiler returned no result");
      return -1;
    }
    result->save(std::string(path));
    return 0;
  } catch (const std::exception& error) {
    hydra_torch_profiler_set_error(error);
    return -1;
  } catch (...) {
    hydra_torch_profiler_set_error("unknown C++ exception stopping PyTorch profiler");
    return -1;
  }
}

int hydra_torch_profiler_start_nvtx(int record_shapes) {
  try {
    using torch::profiler::impl::ExperimentalConfig;
    using torch::profiler::impl::ProfilerConfig;
    using torch::profiler::impl::ProfilerState;

    if (torch::profiler::impl::profilerEnabled()) {
      hydra_torch_profiler_set_error("PyTorch profiler is already active");
      return -1;
    }

    ExperimentalConfig experimental_config(
        {}, false, false, {}, false, false, false, true, false, false, "", false);
    ProfilerConfig config(
        ProfilerState::NVTX,
        record_shapes != 0,
        false,
        false,
        false,
        false,
        experimental_config,
        "hydra_bc_backward_nvtx");
    torch::autograd::profiler::enableProfiler(config, {}, {});
    return 0;
  } catch (const std::exception& error) {
    hydra_torch_profiler_set_error(error);
    return -1;
  } catch (...) {
    hydra_torch_profiler_set_error("unknown C++ exception starting PyTorch NVTX profiler");
    return -1;
  }
}

int hydra_torch_profiler_stop_nvtx() {
  try {
    auto result = torch::autograd::profiler::disableProfiler();
    if (!result) {
      hydra_torch_profiler_set_error("PyTorch NVTX profiler returned no result");
      return -1;
    }
    return 0;
  } catch (const std::exception& error) {
    hydra_torch_profiler_set_error(error);
    return -1;
  } catch (...) {
    hydra_torch_profiler_set_error("unknown C++ exception stopping PyTorch NVTX profiler");
    return -1;
  }
}

#else

int hydra_torch_profiler_start(int) {
  hydra_torch_profiler_set_error(
      "LibTorch profiler headers were unavailable at compile time");
  return -1;
}

int hydra_torch_profiler_stop_and_save(const char*) {
  hydra_torch_profiler_set_error(
      "LibTorch profiler headers were unavailable at compile time");
  return -1;
}

int hydra_torch_profiler_start_nvtx(int) {
  hydra_torch_profiler_set_error(
      "LibTorch profiler headers were unavailable at compile time");
  return -1;
}

int hydra_torch_profiler_stop_nvtx() {
  hydra_torch_profiler_set_error(
      "LibTorch profiler headers were unavailable at compile time");
  return -1;
}

#endif

} // extern "C"
