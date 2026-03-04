# Hydra Ecosystem Survey

Curated integration priority guide for every repo, tool, dataset, and framework that Hydra can benefit from. Organized by category with license status and concrete use cases.

---

## 1. Rust Mahjong Libraries

### Direct Dependencies (MIT/Apache — safe to use)

> Full descriptions for xiangting, mahc, and agari in [REFERENCES.md § Components](REFERENCES.md#components).

| Repo | License | Hydra Use |
|------|---------|-----------| 
| [Apricot-S/xiangting](https://github.com/Apricot-S/xiangting) | MIT | **Primary shanten library** — already selected |
| [DrCheeseFace/mahc](https://github.com/DrCheeseFace/mahc) | BSD-3 | **Scoring reference** — explicit `Fu` enum pattern worth adopting |
| [harphield/riichi-tools-rs](https://github.com/harphield/riichi-tools-rs) | MIT | **WASM deployment reference** — fast hand classifier architecture |
| [m4tx/riichi-hand-rs](https://github.com/m4tx/riichi-hand-rs) | MIT | **Tile rendering** for visualization/debugging tools |
| [penta2019/mahjong_server](https://github.com/penta2019/mahjong_server) | MIT | **Evaluation server** — MJAI endpoint for bot testing |
| [rysb-dev/agari](https://github.com/rysb-dev/agari) | MIT (no LICENSE file) | **Primary scoring reference** — most architecturally clean Rust implementation. `Cargo.toml` declares MIT. |

### Reference Only (Copyleft — study architecture, don't copy code)

> Full descriptions for Mortal, Mortal-Policy, and mjai.app in [REFERENCES.md § Open Source Projects](REFERENCES.md#open-source-projects).

| Repo | License | Hydra Use |
|------|---------|-----------|
| [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal) | AGPL-3.0 | **Primary competitor/benchmark** — architecture study only |
| [Equim-chan/mjai-reviewer](https://github.com/Equim-chan/mjai-reviewer) | Apache-2.0 | **Evaluation tool** — run Hydra through mjai-reviewer for move analysis |
| [summivox/riichi-rs](https://github.com/summivox/riichi-rs) | LGPL-2.1 | **Reference only** — LGPL. `tenhou-shuffle` subcrate is interesting |
| [Nitasurin/Mortal-Policy](https://github.com/Nitasurin/Mortal-Policy) | AGPL-3.0 | **PPO transition reference** — exactly what Hydra does |
| [smly/mjai.app](https://github.com/smly/mjai.app) | AGPL-3.0 | **Competition target** — Hydra must be MJAI-compatible |
| [Apricot-S/lizhisim](https://github.com/Apricot-S/lizhisim) | MIT | **Watch** — same author as our shanten lib |

---

## 2. Python ML/RL Repos

### High Value

> Full descriptions for mahjax, RiichiEnv, Meowjong, and mahjong (Python) in [REFERENCES.md § Components](REFERENCES.md#components).

| Repo | License | Hydra Use |
|------|---------|-----------| 
| [nissymori/mahjax](https://github.com/nissymori/mahjax) | Apache-2.0 | **Fast RL training environment** — JAX vectorization for self-play |
| [Agony5757/mahjong](https://github.com/Agony5757/mahjong) | Unlicensed | **Observation encoding reference** — 93×34/111×34 design is well-researched (C++ sim with Python bindings, ICLR 2022) |
| [smly/RiichiEnv](https://github.com/smly/RiichiEnv) | Apache-2.0 | **Gym environment** for training loop development |
| [VictorZXY/Meowjong](https://github.com/VictorZXY/Meowjong) | MIT | **Critical for Sanma support** — only open-source 3-player mahjong AI |
| [MahjongRepository/mahjong](https://github.com/MahjongRepository/mahjong) | MIT | **Scoring oracle** — already added as dev dependency |
| [CharlesC63/mahjong_ev](https://github.com/CharlesC63/mahjong_ev) | Unlicensed | **Port defense + EV logic to Rust** — the only repo with all three in one package |

### Training Infrastructure

> Full descriptions for CleanRL and OpenSpiel in [REFERENCES.md § Components](REFERENCES.md#components).

| Repo | License | Hydra Use |
|------|---------|-----------| 
| [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | MIT | **PPO reference** — gold standard for clean RL implementations |
| [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | MIT | **Backup PPO** — if custom PPO has issues |
| [pytorch/rl](https://github.com/pytorch/rl) | MIT | **Self-play infrastructure** — built-in distributed game collection |
| [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel) | Apache-2.0 | **Self-play architecture blueprint** — ELO tracking, opponent sampling |
| [yoshitomo-matsubara/torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) | MIT | **Teacher → student distillation** — oracle model → blind model |

### Scoring/Rules References (Other Languages)

| Repo | License | Lang | Hydra Use |
|------|---------|------|-----------|
| [dnovikoff/tempai-core](https://github.com/dnovikoff/tempai-core) | MIT | Go | **Rules engine architecture** — best cross-language reference |
| [Cryolite/tsumonya](https://github.com/Cryolite/tsumonya) | MIT | C++ | **O(1) win detection** via lookup tables |
| [pwmarcz/minefield](https://github.com/pwmarcz/minefield) | Custom | Rust | **Fu verification** — cross-check against agari and mahc |

---

## 3. Data Sources & Datasets

> **Training data is ready**: ~6.6M high-rank 4p hanchan across three sources (2M Tenhou Houou + 1M Majsoul Throne + 3M Majsoul Jade). Tenhou logs pre-converted to MJAI. See [archive/DATA_SOURCES.md](archive/DATA_SOURCES.md) for full details on sources, converters, and alternative datasets.

### Synthetic Data (for self-play)

| Tool | License | Speed | Notes |
|------|---------|-------|-------|
| [mjx-project/mjx](https://github.com/mjx-project/mjx) | — | 100x faster than Mjai | Gym API, Tenhou-compatible rules, gRPC distributed, IEEE CoG 2022 |
| [nissymori/mahjax](https://github.com/nissymori/mahjax) | Apache-2.0 | ~1.6M steps/sec (8×A100) | JAX-vectorized, JIT-compilable |
| [smly/mjai.app](https://github.com/smly/mjai.app) | AGPL-3.0 | — | MJAI-compatible game simulator |

---

## 4. Inference & Deployment

### ONNX Optimization Pipeline

> Full descriptions for ort, tract, candle, burn, and Olive in [REFERENCES.md](REFERENCES.md#components)

| Tool | License | Hydra Use |
|------|---------|-----------|
| [microsoft/Olive](https://github.com/microsoft/Olive) | MIT | **Primary optimization tool** — YAML config → optimized model |
| [onnx/neural-compressor](https://github.com/onnx/neural-compressor) | Apache-2.0 | **Post-export optimization** — model compression directly on `.onnx` files |
| [NVIDIA/TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | Apache-2.0 | **NVIDIA-specific optimization** — INT8/FP16 QAT |
| ONNX Runtime quantization (built-in) | MIT | **Zero-dependency quantization** — `quantize_dynamic()` / `quantize_static()` |

### Rust Inference Engines

> Full descriptions in [REFERENCES.md](REFERENCES.md#components)

| Engine | License | GPU Support | Hydra Role |
|--------|---------|-------------|------------|
| [pykeio/ort](https://github.com/pykeio/ort) | Apache-2.0 | CUDA, TensorRT, CoreML, DirectML, WebGPU | **Primary inference engine** |
| [sonos/tract](https://github.com/sonos/tract) | MIT OR Apache-2.0 | CPU only | **CPU fallback** — pure Rust, no C++ deps |
| [huggingface/candle](https://github.com/huggingface/candle) | Apache-2.0 | CUDA, Metal | **Native Rust models** — skip ONNX entirely |
| [tracel-ai/burn](https://github.com/tracel-ai/burn) | Apache-2.0 | WGPU, CUDA, LibTorch | **Long-term option** — growing ONNX import |

### Inference Optimization Checklist

1. **CUDA Graphs** via `ort` — eliminates kernel dispatch overhead for batch-1 (~5-10ms saved)
2. **I/O Binding** — preallocate GPU buffers, zero host↔device copies (~2-3ms saved)
3. **Fixed input shapes** — enables static graph optimization (no dynamic axes)
4. **INT8 quantization** — 2-4x throughput improvement
5. **Graph optimization Level 3** — operator fusion, constant folding
6. **Target: <5ms** per inference on modern GPU (down from 15ms)

---

## 5. Recommended Integration Priority

### P0 — Core Dependencies (Use Directly)

| Tool | Category | License | Action |
|------|----------|---------|--------|
| xiangting | Shanten | MIT | `cargo add xiangting` — already selected |
| MahjongRepository/mahjong | Scoring oracle | MIT | `pip install mahjong==1.4.0` — already added |
| ort | Rust inference | Apache-2.0 | `cargo add ort` when inference pipeline is built |
| CleanRL | PPO reference | MIT | Study + adapt PPO implementation |

### P1 — High Value References

| Tool | Category | License | Action |
|------|----------|---------|--------|
| rysb-dev/agari | Rust scoring | MIT (no LICENSE file) | Primary reference for hand evaluation implementation — `Cargo.toml` declares MIT but repo has no LICENSE file; safe to use as reference |
| mahc | Rust scoring | BSD-3 | Secondary reference, especially fu enum pattern |
| mahjax | RL environment | Apache-2.0 | Evaluate for JAX-based self-play training |
| RiichiEnv | Gym environment | Apache-2.0 | Evaluate for Python training loop |
| OpenSpiel | Self-play arch | Apache-2.0 | Study AlphaZero self-play loop design |
| Meowjong | Sanma AI | MIT | Reference for 3-player implementation |
| mjai-reviewer | Evaluation | Apache-2.0 | Evaluate Hydra's play quality |
| Microsoft Olive | ONNX optimization | MIT | Use when deploying optimized inference |

### P2 — Useful Tools

| Tool | Category | License | Action |
|------|----------|---------|--------|
| mahjong_ev | Defense/EV | Unlicensed | Port defense analyzer + EV engine concepts to Rust |
| torchdistill | Distillation | MIT | Oracle → blind model distillation |
| tempai-core | Rules engine | MIT | Cross-language reference for configurable rules |

### P3 — Watch / Future Use

| Tool | Category | License | Action |
|------|----------|---------|--------|
| Mortal-Policy | PPO fork | AGPL-3.0 | Study AWR→PPO transition approach |
| lizhisim | Simulator | MIT | Watch — same author as xiangting |
| Burn | Rust ML | Apache-2.0 | Long-term: native Rust training + inference |
| candle | Rust ML | Apache-2.0 | Alternative: skip ONNX, write inference in Rust |

