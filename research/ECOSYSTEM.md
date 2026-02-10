# Hydra Ecosystem Survey

Comprehensive survey of every repo, tool, dataset, and framework that Hydra can benefit from. Organized by category with license status and concrete use cases.

---

## 1. Rust Mahjong Libraries

### Direct Dependencies (MIT/Apache — safe to use)

| Repo | Stars | License | What It Provides | Hydra Use |
|------|-------|---------|-----------------|-----------|
| [Apricot-S/xiangting](https://github.com/Apricot-S/xiangting) | 9.7K dl | MIT | Shanten calculation, necessary/unnecessary tiles, 3p support, `no_std`, compile-time tables | **Primary shanten library** — already selected |
| [DrCheeseFace/mahc](https://github.com/DrCheeseFace/mahc) | 4 | BSD-3 | Scoring, fu enum, yaku detection (38 yaku), payment, CLI. 30K crates.io downloads | **Scoring reference** — explicit `Fu` enum pattern worth adopting |
| [harphield/riichi-tools-rs](https://github.com/harphield/riichi-tools-rs) | 13 | MIT | Shanten (fast LUT + no-LUT modes), shape finding, yaku, scoring, Tenhou replay parsing, WASM | **WASM deployment reference** — fast hand classifier architecture |
| [m4tx/riichi-hand-rs](https://github.com/m4tx/riichi-hand-rs) | 12 | MIT | Hand representation, string parser, tile image renderer, scoring table calculator | **Tile rendering** for visualization/debugging tools |
| [penta2019/mahjong_server](https://github.com/penta2019/mahjong_server) | 17 | MIT | Full game server with Bevy GUI, MJAI protocol, bot framework, Tenhou replay export | **Evaluation server** — MJAI endpoint for bot testing |
| [rysb-dev/agari](https://github.com/rysb-dev/agari) | 3 | MIT | Complete scoring engine: yaku (35), fu, payment, hand decomposition, WASM. ~100 unit tests | **Primary scoring reference** — most architecturally clean Rust implementation |

### Reference Only (Copyleft — study architecture, don't copy code)

| Repo | Stars | License | What It Provides | Hydra Use |
|------|-------|---------|-----------------|-----------|
| [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal) | 1,334 | AGPL-3.0 | Complete mahjong AI: ResNet+SE backbone, DQN+CQL, GRP, MJAI protocol, self-play | **Primary competitor/benchmark** — architecture study only |
| [Equim-chan/mjai-reviewer](https://github.com/Equim-chan/mjai-reviewer) | 1,170 | Apache-2.0 | Game log review with MJAI-compatible AI engines, HTML reports, Tenhou/Majsoul log support | **Evaluation tool** — run Hydra through mjai-reviewer for move analysis |
| [summivox/riichi-rs](https://github.com/summivox/riichi-rs) | 4 | LGPL-2.1 | Full engine monorepo: `riichi`, `riichi-elements`, `riichi-decomp`, `tenhou-shuffle`, `tenhou-db` | **Reference only** — LGPL. `tenhou-shuffle` subcrate is interesting |
| [Nitasurin/Mortal-Policy](https://github.com/Nitasurin/Mortal-Policy) | 37 | AGPL-3.0 | Policy-based Mortal fork (V4): AWR → PPO transition, offline-to-online pipeline | **PPO transition reference** — exactly what Hydra does |
| [smly/mjai.app](https://github.com/smly/mjai.app) | 109 | AGPL-3.0 | RiichiLab mahjong AI competition platform, Docker-based evaluation, MJAI protocol | **Competition target** — Hydra must be MJAI-compatible |
| [Apricot-S/lizhisim](https://github.com/Apricot-S/lizhisim) | 2 | MIT | Game simulator (WIP) by xiangting author, inspired by Kanachan | **Watch** — same author as our shanten lib |

---

## 2. Python ML/RL Repos

### High Value

| Repo | Stars | License | What It Provides | Hydra Use |
|------|-------|---------|-----------------|-----------|
| [nissymori/mahjax](https://github.com/nissymori/mahjax) | 22 | Apache-2.0 | JAX-vectorized riichi environment, ~1.6M steps/sec on 8×A100, behavior cloning + RL examples | **Fast RL training environment** — JAX vectorization for self-play |
| [Agony5757/mahjong](https://github.com/Agony5757/mahjong) | 123 | Unlicensed | C++ simulator with Python bindings, 93×34 obs encoding, offline RL dataset (~40K games/batch), ICLR 2022 paper | **Observation encoding reference** — 93×34/111×34 design is well-researched |
| [smly/RiichiEnv](https://github.com/smly/RiichiEnv) | 9 | Apache-2.0 | Gym-style RL environment, Rust core + Python bindings, Mortal-compatible MJAI, verified 1M+ games | **Gym environment** for training loop development |
| [VictorZXY/Meowjong](https://github.com/VictorZXY/Meowjong) | 42 | MIT | First open-source Sanma (3-player) AI, IEEE CoG 2022, 5 CNN models, Tenhou Sanma log downloader | **Critical for Sanma support** — only open-source 3-player mahjong AI |
| [MahjongRepository/mahjong](https://github.com/MahjongRepository/mahjong) | 449 | MIT | Hand scoring oracle: yaku, han, fu, score. Validated against 11M+ Tenhou hands. Pin to v1.4.0 | **Scoring oracle** — already added as dev dependency |
| [CharlesC63/mahjong_ev](https://github.com/CharlesC63/mahjong_ev) | 0 | Unlicensed | EV engine, ukeire analysis, defense (suji/kabe/genbutsu), push/fold advisor | **Port defense + EV logic to Rust** — the only repo with all three in one package |

### Training Infrastructure

| Repo | Stars | License | What It Provides | Hydra Use |
|------|-------|---------|-----------------|-----------|
| [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | 9,079 | MIT | Single-file PPO implementation (~250 lines), wandb integration, 37 implementation details blog | **PPO reference** — gold standard for clean RL implementations |
| [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | 12,715 | MIT | Battle-tested PPO/A2C/SAC/TD3/DQN, vectorized environments | **Backup PPO** — if custom PPO has issues |
| [pytorch/rl](https://github.com/pytorch/rl) | 3,294 | MIT | Official PyTorch RL library, TensorDict, distributed collectors | **Self-play infrastructure** — built-in distributed game collection |
| [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel) | 5,019 | Apache-2.0 | 70+ game implementations, AlphaZero, MCTS, CFR, self-play training loops | **Self-play architecture blueprint** — ELO tracking, opponent sampling |
| [yoshitomo-matsubara/torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) | 1,591 | MIT | Config-driven knowledge distillation, 26+ methods from CVPR/NeurIPS/ICLR | **Teacher → student distillation** — oracle model → blind model |

### Scoring/Rules References (Other Languages)

| Repo | Stars | License | Lang | What It Provides | Hydra Use |
|------|-------|---------|------|-----------------|-----------|
| [dnovikoff/tempai-core](https://github.com/dnovikoff/tempai-core) | 24 | MIT | Go | Full yaku, fu, rules engine with tag-based meld classification, configurable rule variants | **Rules engine architecture** — best cross-language reference |
| [Cryolite/tsumonya](https://github.com/Cryolite/tsumonya) | 20 | MIT | C++ | Pre-computed lookup table for all winning hands | **O(1) win detection** via lookup tables |
| [pwmarcz/minefield](https://github.com/pwmarcz/minefield) | 35 | Custom | Rust | Most compact fu implementation (43 lines), game server, bot AI | **Fu verification** — cross-check against agari and mahc |

---

## 3. Data Sources & Datasets

> **Training data is ready**: 3M+ Tenhou Phoenix MJAI games (pre-converted). See [archive/DATA_SOURCES.md](archive/DATA_SOURCES.md) for full details on sources, converters, and alternative datasets.

### Synthetic Data (for self-play)

| Tool | Stars | License | Speed | Notes |
|------|-------|---------|-------|-------|
| [mjx-project/mjx](https://github.com/mjx-project/mjx) | 202 | — | 100x faster than Mjai | Gym API, Tenhou-compatible rules, gRPC distributed, IEEE CoG 2022 |
| [nissymori/mahjax](https://github.com/nissymori/mahjax) | 22 | Apache-2.0 | ~1.6M steps/sec (8×A100) | JAX-vectorized, JIT-compilable |
| [smly/mjai.app](https://github.com/smly/mjai.app) | 109 | AGPL-3.0 | — | MJAI-compatible game simulator |

---

## 4. Inference & Deployment

### ONNX Optimization Pipeline

| Tool | Stars | License | What It Does | Hydra Use |
|------|-------|---------|-------------|-----------|
| [microsoft/Olive](https://github.com/microsoft/Olive) | 2,249 | MIT | End-to-end PyTorch → optimized ONNX: quantization, pruning, fusion, shape inference | **Primary optimization tool** — YAML config → optimized model |
| [onnx/neural-compressor](https://github.com/onnx/neural-compressor) | 98 | Apache-2.0 | Model compression directly on `.onnx` files | **Post-export optimization** |
| [NVIDIA/TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | 1,963 | Apache-2.0 | Quantization, pruning, distillation for TensorRT/ONNX Runtime | **NVIDIA-specific optimization** — INT8/FP16 QAT |
| ONNX Runtime quantization (built-in) | — | MIT | `quantize_dynamic()` / `quantize_static()` — INT8/UINT8/INT4 | **Zero-dependency quantization** |

### Rust Inference Engines

| Engine | Stars | License | GPU Support | Hydra Role |
|--------|-------|---------|-------------|------------|
| [pykeio/ort](https://github.com/pykeio/ort) | 1,969 | Apache-2.0 | CUDA, TensorRT, CoreML, DirectML, WebGPU | **Primary inference engine** — CUDA graphs, I/O binding, Level3 optimization |
| [sonos/tract](https://github.com/sonos/tract) | 2,767 | MIT OR Apache-2.0 | CPU only | **CPU fallback** — pure Rust, no C++ deps |
| [huggingface/candle](https://github.com/huggingface/candle) | 19,333 | Apache-2.0 | CUDA, Metal | **Native Rust models** — skip ONNX entirely, write inference in Rust |
| [tracel-ai/burn](https://github.com/tracel-ai/burn) | 14,280 | Apache-2.0 | WGPU, CUDA, LibTorch | **Long-term option** — growing ONNX import, Rust-native training |

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
| rysb-dev/agari | Rust scoring | MIT | Primary reference for hand evaluation implementation |
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

---

## 6. Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                          │
│                                                          │
│  3M+ Phoenix MJAI games (pre-converted, ready)           │
│                                                          │
│  Self-play (mahjax/RiichiEnv) ──→ MJAI                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│                    TRAINING                               │
│                                                           │
│  MJAI data ──→ PPO (CleanRL-style) + AMP (bf16)           │
│            ──→ Oracle distillation (torchdistill)          │
│            ──→ Self-play loop (OpenSpiel architecture)     │
│            ──→ wandb tracking                              │
│                                                           │
│  Export: torch.onnx.export(fixed shapes, opset 17)        │
│  Optimize: Microsoft Olive → INT8/FP16 quantization       │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│                    INFERENCE                              │
│                                                           │
│  ort (Rust) + CUDA EP + CUDA graphs + I/O binding         │
│  Target: <5ms per inference                               │
│                                                           │
│  Evaluation: mjai-reviewer, mjai.app competition          │
└──────────────────────────────────────────────────────────┘
```
