# RUST_STACK.md -- 100% Rust Training Stack Decision

> Decision Record: Hydra will use a 100% Rust stack for training, inference,
> and self-play. No Python dependency at any point in the pipeline.

## 1. Executive Summary

Hydra adopts **Burn** (tracel-ai/burn) as its deep learning framework with
the **`burn-tch` backend** (libtorch/cuDNN) for production training, and
**`burn-cuda`** (CubeCL JIT) as a future upgrade path.

This eliminates Python entirely. The game engine (riichienv-core), observation
encoder (hydra-core), training loop, self-play arena, and inference all run
in a single Rust binary with zero IPC, zero GIL, and zero interpreter overhead.

### Why 100% Rust

- **Same GPU performance**: `burn-tch` wraps libtorch, which calls the same
  cuDNN/cuBLAS CUDA kernels as Python PyTorch. Identical GPU compute.
- **3.5-4x less CPU overhead**: Benchmarked: C++ LibTorch trains ResNet18
  3.56x faster than Python PyTorch (same model, same GPU).
  Python's `torch.compile` exists to claw back this overhead. Rust never has it.
- **Self-play integration**: PPO alternates between game simulation and training.
  In Python, this requires subprocess/IPC. In Rust, same process, same memory.
- **Single binary**: No pip, no conda, no virtualenv, no dependency hell.
  `cargo build --release` produces one artifact.

### Why Burn over raw tch-rs

- Built-in training infrastructure (Learner, DataLoader, metrics, checkpointing)
- Built-in DDP with NCCL for multi-GPU
- Built-in LR schedulers (cosine annealing, linear warmup, exponential, noam, step, composed)
- Built-in gradient clipping (by value and by norm)
- Backend-generic code: swap `burn-tch` to `burn-cuda` by changing one type parameter
- CubeCL JIT fusion as future upgrade (Burn's answer to torch.compile)

## 2. Framework Comparison

### Rust ML Frameworks Evaluated

| Framework | Stars | Training | GPU | Autograd | cuDNN | JIT Fusion | Verdict |
|---|---|---|---|---|---|---|---|
| **Burn** | 9.5k | Full | CUDA/WGPU/Metal | Native autodiff | Via burn-tch | CubeCL | **Selected** |
| tch-rs | 4.4k | Full | CUDA | libtorch autograd | Yes (libtorch) | No | Backend only |
| Candle | 16k | Basic | CUDA/Metal | Basic | No | No | Inference-focused |
| Linfa | -- | No | No | No | No | No | Classical ML only |

### Why Burn Wins

1. **Backend abstraction**: Same model code runs on `burn-tch` (cuDNN) or `burn-cuda` (CubeCL).
   Swap one generic parameter, zero code changes.
2. **Training infrastructure**: `burn-train` crate provides Learner, DataLoader (multi-threaded),
   metric tracking, and checkpointing out of the box.
3. **DDP built-in**: `burn-collective` with NCCL for CUDA, AllReduce/AllGather/Broadcast.
   Multi-node via WebSocket. Feature flag: `collective`.
4. **CubeCL JIT fusion**: Burn's answer to torch.compile. Serializes tensor ops into symbolic
   graph, fuses elementwise ops, auto-tunes kernels for hardware. Works for training + inference.
5. **All required layers exist**: GroupNorm, Mish, AdaptiveAvgPool2d, Conv2d, Linear, residual
   connections, SE blocks (compose from primitives). Verified in source.

### burn-tch Backend (Production Config)

The `burn-tch` backend wraps tch-rs which wraps libtorch. This gives us:
- cuDNN conv2d (same algorithms as Python PyTorch)
- cuBLAS matmul (identical performance)
- libtorch autograd (battle-tested by millions of users)
- libtorch CUDA caching allocator (proven memory management)
- `tch::autocast` for bf16 mixed precision
- `cudnn_benchmark` for convolution algorithm autotuning

### burn-cuda Backend (Future Upgrade)

The `burn-cuda` backend uses CubeCL to JIT-generate CUDA kernels:
- Implicit GEMM conv2d with tensor cores (CMMA + MMA) and autotuning
- Operator fusion (fuse-on-read, fuse-on-write) for elementwise chains
- 3-tier memory pool (SlicedPool, ExclusivePool, PersistentPool)
- Published benchmarks: matches cuBLAS on matmul, 3-33x faster than libtorch on CPU ops
- Upgrade path: swap `Burn<LibTorch>` to `Burn<CudaRuntime>`, zero code changes

## 3. hydra-core vs Mortal's libriichi

### Architecture: Delegation vs Monolith

hydra-core delegates game logic to riichienv-core. libriichi is fully self-contained.

| Area | hydra-core | libriichi |
|---|---|---|
| Game state | Delegates to riichienv-core | Own PlayerState (10 files) |
| Shanten | Delegates to riichienv-core | Own solver with lookup tables |
| Scoring | Delegates to riichienv-core | Own agari + point calculation |
| Encoder | **85x34** tensor (own) | **1012x34** tensor (own) |
| Safety | **Dedicated module** (23 channels) | Embedded in state |
| Action space | 46 actions (Mortal-compatible) | 46 actions |
| Tile encoding | TileType(0-33) + suit permutation | Tile(0-37) incl aka+unknown |
| Augmentation | **6-way suit permutation** | m/p swap only |
| Encoding | **Incremental with DirtyFlags** | Full re-encode per turn |
| Seeding | **SHA-256 KDF + vendored Fisher-Yates** | Standard RNG |
| Arena | Not yet | Built-in self-play |
| Dataset | Not yet | mjai log reader + batch pipeline |
| Inference | Planned (ort/Burn) | tch (libtorch) |
| LoC | ~3,500 | ~15,000+ |
| License | MIT | AGPL-3.0 |

### What hydra-core Owns (unique to Hydra)

- 85x34 observation encoding with 23 dedicated safety channels
- Incremental encoding with DirtyFlags (skip unchanged channels)
- 6-way suit permutation for data augmentation
- SHA-256 KDF wall generation for cross-version determinism
- ActionSelector trait for pluggable policies
- Bridge layer between riichienv-core and encoder

## 4. Performance Analysis: Max Rust vs Max Python

### Python 3.2x Speedup Breakdown (ResNet50, CIFAR-10, Nsight-profiled)

| Optimization | img/s | Gain | Share of Total |
|---|---:|---:|---:|
| Baseline (eager Python) | 994 | -- | -- |
| Fix .item() sync | 1,049 | +5.5% | 2.5% |
| pin_memory + non_blocking | 1,063 | +1.3% | 0.6% |
| cudnn.benchmark = True | 1,093 | +2.9% | 1.4% |
| torch.compile() default | 1,290 | +18.1% | 9.0% |
| torch.compile(max-autotune) | 1,393 | +8.0% | 4.7% |
| Inductor exhaustive search | 1,427 | +2.4% | 1.6% |
| **AMP (mixed precision)** | **3,026** | **+112%** | **73.2%** |
| Channels-last memory | 3,178 | +5.0% | 7.0% |
| **Total** | **3,178** | | **3.2x** |

Key insight: **73% of all gains come from bf16 tensor cores.** This is
hardware-level, not Python-specific. Rust gets it identically.

### C++ LibTorch vs Python PyTorch (no torch.compile)

| Model | Python (s/epoch) | C++ LibTorch (s/epoch) | C++ Speedup |
|---|---:|---:|---:|
| ResNet18 | 25.78 | 7.24 | **3.56x** |
| ResNet34 | 45.24 | 11.06 | **4.09x** |

C++/Rust is 3.5-4x faster than Python for training **before any
optimizations**. torch.compile exists to close this gap for Python.
Rust starts where torch.compile tries to get to.

### Head-to-Head: Max Python vs Max Rust

| Factor | Python (max perf) | Rust (burn-tch) |
|---|---|---|
| Conv2d kernels | cuDNN | **cuDNN (identical)** |
| Matmul kernels | cuBLAS | **cuBLAS (identical)** |
| bf16 tensor cores | autocast | **autocast (identical)** |
| Operator fusion | torch.compile Inductor | None needed (no Python dispatch) |
| CUDA graphs | Built into Inductor | Manual (or async execution) |
| Interpreter overhead | ~3.5x penalty, clawed back by compile | **Zero** |
| Self-play integration | Subprocess + IPC | **Same process, zero-copy** |
| GIL contention | Yes (data loading, logging) | **None** |
| Build reproducibility | pip/conda dependency tree | **cargo build** |

**Verdict: Max Rust >= Max Python.** Same CUDA kernels, zero Python tax,
zero IPC for self-play. For RL workloads where self-play dominates wall
clock time, Rust wins by a significant margin.

## 5. All Concerns Raised and Resolutions

### Resolved: "Just Write Rust" (17 concerns)

These were removed because they require only Rust code, not framework features.

| # | Concern | Resolution |
|---|---|---|
| 1 | No ONNX export | Use Burn directly for inference. Or save via burn-tch as TorchScript. |
| 2 | No param groups | Multiple optimizers on different param subsets. |
| 3 | No PPO impl | Write PPO training loop from scratch. Core Hydra work. |
| 4 | Compile times | Pin to stable Burn version. Incremental builds mitigate. |
| 5 | DDP is new | burn-tch backend's DDP tested. Manual NCCL via cudarc as fallback. |
| 6 | No W&B | Hit W&B REST API via reqwest. Or use tensorboard-rs. |
| 7 | Debugging grads | Write our own gradcheck (finite difference vs analytical). |
| 8 | CUDA profiling | Nsight Systems/Compute work on any CUDA calls. |
| 9 | ONNX import limited | Training from scratch. Not importing. |
| 10 | bf16 not autocast | Whole model bf16 via device policy. bf16 has fp32 range. |
| 11 | Checkpoint format | Burn's Record system. Converter if needed. |
| 12 | Binary size | Acceptable for server-side training. |
| 13 | API stability | Pin Burn version. Upgrade deliberately. |
| 14 | Ecosystem lock-in | Accepted. Committed to Rust. |
| 15 | Data pipeline | Burn's MultiThreadDataLoader. Custom Dataset for mjai logs. |
| 16 | Self-play loop | Bypass Learner, write manual PPO loop. |
| 17 | No no_grad scope | Use non-Autodiff backend for inference during rollout. |

### Bridged: Framework-Level Concerns (6 concerns)

These required research to confirm Rust solutions exist.

| # | Concern | Bridge | Effort |
|---|---|---|---|
| 1 | conv2d perf vs cuDNN | burn-tch uses cuDNN. Identical perf. burn-cuda has implicit GEMM + tensor cores + autotuning. | Benchmark to verify |
| 2 | Autodiff correctness | Write gradcheck. Cross-backend gradient comparison (burn-tch vs burn-cuda). Burn has per-op gradient tests. | ~100 lines Rust |
| 3 | Kernel fusion bugs | Cross-backend tensor comparison for our architecture. Burn's test suite runs across all backends. | ~200 lines Rust |
| 4 | GPU memory allocator | CubeCL has 3-tier memory pool (Sliced + Exclusive ring buffer + Persistent). burn-tch uses libtorch caching allocator. | Profile at first run |
| 5 | Numerical divergence | Accepted. Tolerance-based assertions (1e-5 fp32, 1e-3 bf16). Different kernels = different FP rounding. | None needed |
| 6 | Small community | burn-tch fallback. Contribute fixes upstream (Rust is readable). Pin versions. Dual-backend CI. | Ongoing |

### New Concerns from Research (7 concerns)

| # | Concern | Resolution |
|---|---|---|
| 1 | Small tensor overhead (85x34 input) | Profile CubeCL JIT latency. Use burn-tch if JIT overhead dominates for small tensors. |
| 2 | Backend swap mid-training | Validate numerical stability of burn-tch to burn-cuda switch. May need retrain. |
| 3 | Thread safety of autodiff | Clone model for inference threads. Single model for training. Standard RL pattern. |
| 4 | Generic compile errors | Rust limitation. Mitigate with type aliases and wrapper types. |
| 5 | Tensor creation from raw slices | Verify Burn's Tensor::from_data is zero-copy on CPU. Profile if bottleneck. |
| 6 | NaN gradient detection | Write custom detect_anomaly. Hook into backward pass. |
| 7 | Checkpoint compat across versions | Pin Burn version for training duration. Test checkpoint load before upgrading. |

### Hard Blockers Found: Zero

No concern was identified that cannot be solved in Rust. Every PyTorch
capability needed for Hydra has a Rust equivalent or can be built.

## 6. Migration Strategy and Remaining Risks

### Phase 1: burn-tch (Production)

Start with `burn-tch` backend for all training:
- cuDNN conv2d (proven, identical to Python PyTorch)
- libtorch autograd (battle-tested)
- libtorch CUDA caching allocator (proven memory management)
- `tch::autocast` for bf16
- Full Burn training infrastructure (Learner, DDP, schedulers)

### Phase 2: Benchmark burn-cuda

Run comparative benchmarks with our exact model (SE-ResNet, 40 blocks,
256ch, batch=256, bf16):
- Forward pass throughput: burn-tch vs burn-cuda
- Backward pass throughput: burn-tch vs burn-cuda
- Gradient correctness: cross-backend tensor comparison
- Memory usage: peak GPU memory under load
- Conv2d specifically: implicit GEMM + autotuning vs cuDNN

### Phase 3: Upgrade to burn-cuda (If Benchmarks Pass)

If burn-cuda matches or beats burn-tch:
- Swap generic parameter: `type B = Autodiff<CudaRuntime>` instead of `Autodiff<LibTorch>`
- Pick up CubeCL JIT fusion benefits (operator fusion, reduced memory traffic)
- Drop libtorch dependency entirely (pure Rust binary, no C++ build)

If burn-cuda does not match:
- Stay on burn-tch. No penalty. Same cuDNN kernels as Python PyTorch.
- Revisit on future Burn releases as CubeCL matures.

### Remaining Risks (Quality, not Capability)

| Risk | Severity | Mitigation |
|---|---|---|
| burn-cuda conv2d slower than cuDNN | Medium | Stay on burn-tch. Zero impact. |
| Burn autodiff bug on deep network | Low | Gradcheck + cross-backend comparison. |
| CubeCL fusion produces wrong values | Low | Dual-backend CI catches this. |
| GPU memory fragmentation | Low | burn-tch uses libtorch allocator. Profile early. |
| Burn upstream development slows | Low | burn-tch uses libtorch (maintained by Meta). |

## 7. Revised Architecture

### Previous Plan (INFRASTRUCTURE.md)

```
Rust: game engine, encoder, simulator, ONNX inference
Python: model definition, training loop, data pipeline
```

### New Plan (100% Rust)

```
hydra-core/     Rust: game engine, encoder, safety, simulator, seeding
hydra-train/    Rust: Burn model definition, PPO training loop, self-play arena
hydra-py/       REMOVED (no Python bindings needed)
scripts/        Rust: evaluation, export utilities
```

### Dependencies (Production)

```toml
[dependencies]
burn = { version = "0.21", features = ["tch", "train", "collective"] }
riichienv-core = { path = "../riichienv-core" }
rayon = "1.10"   # parallel self-play
serde = "1.0"    # checkpoint serialization
```

### Key Crates

| Crate | Purpose |
|---|---|
| burn | Core framework + tensor ops |
| burn-tch | libtorch/cuDNN backend (production) |
| burn-cuda | CubeCL backend (future upgrade) |
| burn-train | Learner, metrics, checkpointing |
| burn-optim | Adam/AdamW, LR schedulers, gradient clipping |
| burn-collective | DDP, NCCL, AllReduce |
| burn-autodiff | Automatic differentiation |
| burn-dataset | Dataset trait, transforms |

## 8. Layer Verification (All Confirmed in Burn Source)

| Layer | Burn Module | Source Path |
|---|---|---|
| GroupNorm(32) | `burn::nn::GroupNorm` | `crates/burn-nn/src/modules/norm/group.rs` |
| Mish | `burn::tensor::activation::mish()` | `crates/burn-tensor/src/tensor/activation/base.rs` |
| AdaptiveAvgPool2d | `burn::nn::AdaptiveAvgPool2d` | `crates/burn-nn/src/modules/pool/adaptive_avg_pool2d.rs` |
| Conv2d | `burn::nn::Conv2d` | Native |
| Linear | `burn::nn::Linear` | Native |
| Residual add | `tensor + tensor` | Native operator overload |
| SE block | Compose: adaptive_avg_pool2d + linear + relu + sigmoid + mul | efficientnet.rs reference |
| Multi-head | Custom struct implementing ModuleT | Return tuple of tensors |
| Skip connections | Tensor addition | resnet.rs reference |
| Gradient clipping | `burn::optim::GradientClipping` | By value and by norm |
| Cosine annealing LR | `burn::lr_scheduler::CosineAnnealingLrSchedulerConfig` | Built-in |
| Linear warmup LR | `burn::lr_scheduler::LinearLrSchedulerConfig` | Built-in |
| Composed scheduler | `burn::lr_scheduler::ComposedLrSchedulerConfig` | Chains multiple schedulers |
| DataLoader | `burn::data::DataLoaderBuilder` | Multi-threaded with shuffle |
| bf16 | `FloatDType::BF16` device policy | Tensor type system |
| DDP | `burn-collective` + `burn-train` ddp feature | NCCL for CUDA |

## 9. Conflicts with Existing Docs

Every existing spec assumes Python+PyTorch. The following sections conflict
with the 100% Rust decision and must be updated.

### INFRASTRUCTURE.md (heaviest -- almost every section)

| Section / Line | Conflict | Rust Replacement |
|---|---|---|
| L5: "hybrid Rust + Python architecture" | Architecture overview | "100% Rust architecture" |
| L16-63: System Architecture diagram | Shows hydra-py, hydra-train Python subgraphs | Remove Python subgraphs, single Rust binary |
| L72: pyo3 dependency | PyO3 crate | Remove (no Python bindings needed) |
| L116: python.rs module | PyO3 binding definitions | Remove from module table |
| L276-293: PyTorch Pipeline diagram | IterableDataset, DataLoader, PyTorch | Burn DataLoaderBuilder + Rust GameplayLoader |
| L313: Data loading decision | "PyTorch IterableDataset backed by Rust via PyO3" | burn::data::DataLoaderBuilder with custom Dataset |
| L317-328: DataLoader config table | num_workers, pin_memory, persistent_workers | Burn DataLoaderBuilder::num_workers(), shuffle() |
| L435-479: hydra-py section | Full PyO3 interface, Gym-like Environment | Remove entirely (no Python bindings) |
| L481-494: hydra-train dependencies | torch, numpy, wandb, einops, rich | burn, burn-train, reqwest (W&B API), indicatif |
| L523: torch.compile | "Enabled, ~10-30% throughput gain" | CubeCL JIT fusion (burn-cuda upgrade path) |
| L548: "W&B" monitoring | wandb Python SDK | reqwest + W&B REST API, or tensorboard-rs |
| L629: Self-play architecture | "Python threading", "PyO3 + rayon" | Pure Rust rayon threads (no GIL concern) |
| L779-783: Model Definition | torch.compile, PyTorch model | Burn Module derive, burn-tch backend |
| L795-817: ONNX Export / Deployment | torch.onnx.export pipeline | Burn direct inference or CModule via burn-tch |
| L850-886: CI Pipeline | pytest, pyright, ruff for Python | cargo test, cargo clippy only |

### HYDRA_SPEC.md

| Section / Line | Conflict | Rust Replacement |
|---|---|---|
| L34: Design principle 4 | "Rust + Python Hybrid" | "100% Rust" |
| L526: Inference optimization | "torch.compile in reduce-overhead mode" | Burn inference or CModule via burn-tch |
| L587-593: Dependency licenses | Lists PyTorch (BSD), PyO3 | Replace with burn (Apache-2.0/MIT) |
| L609-628: System Overview diagram | "Python Training" subgraph | Single Rust binary diagram |

### TRAINING.md

| Section / Line | Conflict | Rust Replacement |
|---|---|---|
| L326: GRP label indexing | "itertools.permutations(range(4))" (Python) | Const array of 24 permutations in Rust |
| Phase 1/2/3 infra references | Points to INFRASTRUCTURE.md Python sections | Update cross-references to Rust equivalents |
| Loss function code | Assumes PyTorch API (torch.nn.functional) | Burn tensor ops (equivalent math) |

### TESTING.md

| Section / Line | Conflict | Rust Replacement |
|---|---|---|
| L226-246: Python Training Stack tests | PyTorch model smoke tests, ONNX export | Burn model forward pass tests, Burn inference |
| L35: Scoring cross-validation | "mahjong Python library" | Port validation to Rust or use build.rs script |
| L232: ONNX export test | "export model, run through ONNX Runtime" | Burn native inference test |

### SEEDING.md

| Section / Line | Conflict | Rust Replacement |
|---|---|---|
| L30-32: Root RNG | "NumPy SeedSequence" | Rust rand::SeedableRng with ChaCha20 |
| L38-39: Component seeds | "torch.manual_seed()", "torch.Generator" | Burn backend seeding + rand crate |

### CHECKPOINTING.md

| Section / Line | Conflict | Rust Replacement |
|---|---|---|
| L19: Serialization | "torch.save" | Burn Record system (NamedMpkFileRecorder) |
| L25-34: Checkpoint keys | model_state_dict, optimizer_state_dict | Burn Module::record(), Optimizer::record() |
| L48: Dtype discipline | "torch.save preserves dtype" | Burn Record preserves dtype natively |

### AGENTS.md (repo root)

| Section | Conflict | Rust Replacement |
|---|---|---|
| Python build/lint/test commands | ruff, pyright, pytest for hydra-train/ | Remove Python sections entirely |
| PyO3 bindings section | maturin develop | Remove (no Python bindings) |

## 10. Implementation References

Concrete crate versions, APIs, and code patterns for starting immediately.

### Cargo.toml (hydra-train crate)

```toml
[package]
name = "hydra-train"
version = "0.1.0"
edition = "2024"

[dependencies]
burn = { version = "0.21", features = ["tch", "train", "dataset"] }
burn-tch = "0.21"
burn-autodiff = "0.21"
riichienv-core = { path = "../riichienv-core" }
hydra-core = { path = "../hydra-core" }
rayon = "1.10"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.12", features = ["json"] }  # W&B REST API
indicatif = "0.17"  # progress bars
flate2 = "1.0"  # gzip decompression for mjai logs
```

### Model Definition Pattern (Burn)

```rust
use burn::prelude::*;
use burn::nn::*;

#[derive(Module, Debug)]
pub struct SEResBlock<B: Backend> {
    gn1: GroupNorm<B>,
    conv1: Conv1d<B>,
    gn2: GroupNorm<B>,
    conv2: Conv1d<B>,
    se_fc1: Linear<B>,
    se_fc2: Linear<B>,
    pool: AdaptiveAvgPool1d,
}
```

### Training Loop Pattern (Manual PPO)

```rust
// Bypass burn-train Learner for custom PPO loop
let device = burn_tch::TchDevice::Cuda(0);
let model = HydraModelConfig::new().init::<Autodiff<TchBackend>>(&device);
let optim = burn::optim::AdamWConfig::new()
    .with_weight_decay(0.01)
    .with_grad_clipping(GradientClipping::Norm(0.5))
    .init();
```

### Key Burn Documentation Links

| Resource | URL |
|---|---|
| Burn Book (official guide) | https://burn.dev/books/burn/ |
| API Docs | https://burn.dev/docs/burn/ |
| burn-tch backend | https://docs.rs/burn-tch/latest/ |
| burn-train Learner | https://burn.dev/books/burn/building-blocks/learner.html |
| burn-optim (optimizers) | https://docs.rs/burn-optim/latest/ |
| burn-dataset | https://burn.dev/books/burn/building-blocks/dataset.html |
| GroupNorm API | https://burn.dev/docs/burn/nn/modules/struct.GroupNorm.html |
| PyTorch weight import | https://burn.dev/books/burn/saving-and-loading.html |
| Kernel fusion guide | https://burn.dev/books/burn/performance/good-practices/kernel-fusion.html |
| CubeCL (JIT compiler) | https://github.com/tracel-ai/cubecl |
| Burn GitHub | https://github.com/tracel-ai/burn |
| Burn examples | https://github.com/tracel-ai/burn/tree/main/examples |

### Key tch-rs Documentation Links

| Resource | URL |
|---|---|
| tch-rs GitHub | https://github.com/LaurentMazare/tch-rs |
| tch-rs API docs | https://docs.rs/tch/latest/ |
| tch-rs PPO example | https://github.com/LaurentMazare/tch-rs/blob/main/examples/reinforcement-learning/ppo.rs |
| tch::autocast (bf16) | https://docs.rs/tch/latest/tch/fn.autocast.html |

### Python-to-Rust Migration Cheat Sheet

| Python (PyTorch) | Rust (Burn) |
|---|---|
| `torch.nn.Module` | `#[derive(Module)]` struct |
| `model.parameters()` | Module trait (automatic) |
| `torch.optim.AdamW` | `burn::optim::AdamWConfig::new().init()` |
| `torch.compile()` | CubeCL JIT fusion (burn-cuda backend) |
| `torch.autocast("cuda", dtype=torch.bfloat16)` | `set_default_dtypes(device, FloatDType::BF16, ...)` |
| `torch.no_grad()` | Use non-Autodiff backend for inference |
| `DataLoader(dataset, batch_size=2048, num_workers=8)` | `DataLoaderBuilder::new(batcher).batch_size(2048).num_workers(8).build(dataset)` |
| `torch.save(checkpoint, path)` | `model.save_file(path, &recorder)` |
| `model.load_state_dict(...)` | `model.load_file(path, &recorder, &device)` |
| `wandb.log({"loss": loss})` | `reqwest::Client::post("https://api.wandb.ai/...")` |
| `torch.nn.functional.cross_entropy` | `burn::tensor::loss::cross_entropy_with_logits` |
| `CosineAnnealingLR` | `CosineAnnealingLrSchedulerConfig::new(...)` |
