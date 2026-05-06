# Hydra Ecosystem Survey

Curated integration priority guide for repos, tools, datasets, frameworks Hydra can use. Grouped by category, license, concrete use.

---

## 1. Rust Mahjong Libraries

### Direct Dependencies (MIT/Apache — safe to use)

> Full descriptions for xiangting, mahc, and agari in [REFERENCES.md § Components](REFERENCES.md#components).

| Repo | License | Hydra Use |
|------|---------|-----------| 
| [Apricot-S/xiangting](https://github.com/Apricot-S/xiangting) | MIT | **Primary shanten lib** — already selected |
| [DrCheeseFace/mahc](https://github.com/DrCheeseFace/mahc) | BSD-3 | **Scoring ref** — `Fu` enum pattern worth copying |
| [harphield/riichi-tools-rs](https://github.com/harphield/riichi-tools-rs) | MIT | **WASM deploy ref** — fast hand-classifier arch |
| [m4tx/riichi-hand-rs](https://github.com/m4tx/riichi-hand-rs) | MIT | **Tile rendering** for viz/debug tools |
| [penta2019/mahjong_server](https://github.com/penta2019/mahjong_server) | MIT | **Eval server** — MJAI endpoint for bot testing |
| [rysb-dev/agari](https://github.com/rysb-dev/agari) | MIT (no LICENSE file) | **Primary scoring ref** — cleanest Rust impl. `Cargo.toml` says MIT. |

### Reference Only (Copyleft — study architecture, don't copy code)

> Full descriptions for Mortal, Mortal-Policy, and mjai.app in [REFERENCES.md § Open Source Projects](REFERENCES.md#open-source-projects).

| Repo | License | Hydra Use |
|------|---------|-----------|
| [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal) | AGPL-3.0 | **Primary competitor/benchmark** — arch study only |
| [Equim-chan/mjai-reviewer](https://github.com/Equim-chan/mjai-reviewer) | Apache-2.0 | **Eval tool** — run Hydra through mjai-reviewer for move analysis |
| [summivox/riichi-rs](https://github.com/summivox/riichi-rs) | LGPL-2.1 | **Ref only** — LGPL. `tenhou-shuffle` subcrate interesting |
| [Nitasurin/Mortal-Policy](https://github.com/Nitasurin/Mortal-Policy) | AGPL-3.0 | **PPO transition ref** — closest match to Hydra |
| [smly/mjai.app](https://github.com/smly/mjai.app) | AGPL-3.0 | **Competition target** — Hydra must stay MJAI-compatible |
| [Apricot-S/lizhisim](https://github.com/Apricot-S/lizhisim) | MIT | **Watch** — same author as shanten lib |

---

## 2. Python ML/RL Repos

### High Value

> Full descriptions for mahjax, RiichiEnv, Meowjong, and mahjong (Python) in [REFERENCES.md § Components](REFERENCES.md#components).

| Repo | License | Hydra Use |
|------|---------|-----------| 
| [nissymori/mahjax](https://github.com/nissymori/mahjax) | Apache-2.0 | **Fast RL env** — JAX vectorization for self-play |
| [Agony5757/mahjong](https://github.com/Agony5757/mahjong) | Unlicensed | **Obs-encoding ref** — 93×34/111×34 design well-studied (C++ sim + Python bindings, ICLR 2022) |
| [smly/RiichiEnv](https://github.com/smly/RiichiEnv) | Apache-2.0 | **Gym env** for training-loop dev |
| [VictorZXY/Meowjong](https://github.com/VictorZXY/Meowjong) | MIT | **Critical for Sanma** — only open-source 3-player mahjong AI |
| [MahjongRepository/mahjong](https://github.com/MahjongRepository/mahjong) | MIT | **Scoring oracle** — already added as dev dep |
| [CharlesC63/mahjong_ev](https://github.com/CharlesC63/mahjong_ev) | Unlicensed | **Port defense + EV logic to Rust** — only repo with all 3 in one package |

### Training Infrastructure

> Full descriptions for CleanRL and OpenSpiel in [REFERENCES.md § Components](REFERENCES.md#components).

| Repo | License | Hydra Use |
|------|---------|-----------| 
| [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | MIT | **PPO ref** — clean RL gold standard |
| [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | MIT | **Backup PPO** — if custom PPO breaks |
| [pytorch/rl](https://github.com/pytorch/rl) | MIT | **Self-play infra** — built-in distributed game collection |
| [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel) | Apache-2.0 | **Self-play arch blueprint** — ELO tracking, opponent sampling |
| [yoshitomo-matsubara/torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) | MIT | **Teacher → student distill** — oracle model → blind model |

### Scoring/Rules References (Other Languages)

| Repo | License | Lang | Hydra Use |
|------|---------|------|-----------|
| [dnovikoff/tempai-core](https://github.com/dnovikoff/tempai-core) | MIT | Go | **Rules-engine arch** — best cross-lang ref |
| [Cryolite/tsumonya](https://github.com/Cryolite/tsumonya) | MIT | C++ | **O(1) win detection** via lookup tables |
| [pwmarcz/minefield](https://github.com/pwmarcz/minefield) | Custom | Rust | **Fu verification** — cross-check vs agari + mahc |

---

## 3. Data Sources & Datasets

> **Training data ready**: ~6.6M high-rank 4p hanchan across 3 sources (2M Tenhou Houou + 1M Majsoul Throne + 3M Majsoul Jade). Tenhou logs already MJAI-converted. Separate `archive/DATA_SOURCES.md` missing in current repo, so treat section as surviving high-level summary.

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
| [onnx/neural-compressor](https://github.com/onnx/neural-compressor) | Apache-2.0 | **Post-export optimization** — model compression on `.onnx` files |
| [NVIDIA/TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | Apache-2.0 | **NVIDIA-specific optimization** — INT8/FP16 QAT |
| ONNX Runtime quantization (built-in) | MIT | **Zero-dep quantization** — `quantize_dynamic()` / `quantize_static()` |

### Rust Inference Engines

> Full descriptions in [REFERENCES.md](REFERENCES.md#components)

| Engine | License | GPU Support | Hydra Role |
|--------|---------|-------------|------------|
| [pykeio/ort](https://github.com/pykeio/ort) | Apache-2.0 | CUDA, TensorRT, CoreML, DirectML, WebGPU | **Primary inference engine** |
| [sonos/tract](https://github.com/sonos/tract) | MIT OR Apache-2.0 | CPU only | **CPU fallback** — pure Rust, no C++ deps |
| [huggingface/candle](https://github.com/huggingface/candle) | Apache-2.0 | CUDA, Metal | **Native Rust models** — skip ONNX |
| [tracel-ai/burn](https://github.com/tracel-ai/burn) | Apache-2.0 | WGPU, CUDA, LibTorch | **Long-term option** — ONNX import growing |

### Inference Optimization Checklist

1. **CUDA Graphs** via `ort` — kill kernel dispatch overhead for batch-1 (~5-10ms saved)
2. **I/O Binding** — prealloc GPU buffers, zero host↔device copies (~2-3ms saved)
3. **Fixed input shapes** — enable static-graph optimization (no dynamic axes)
4. **INT8 quantization** — 2-4x throughput gain
5. **Graph optimization Level 3** — op fusion, constant folding
6. **Target: <5ms** per inference on modern GPU (from 15ms)

---

## 5. Recommended Integration Priority

### P0 — Core Dependencies (Use Directly)

| Tool | Category | License | Action |
|------|----------|---------|--------|
| xiangting | Shanten | MIT | `cargo add xiangting` — already selected |
| MahjongRepository/mahjong | Scoring oracle | MIT | `pip install mahjong==1.4.0` — already added |
| ort | Rust inference | Apache-2.0 | `cargo add ort` when inference pipeline exists |
| CleanRL | PPO reference | MIT | Study + adapt PPO impl |

### P1 — High Value References

| Tool | Category | License | Action |
|------|----------|---------|--------|
| rysb-dev/agari | Rust scoring | MIT (no LICENSE file) | Primary ref for hand-eval impl — `Cargo.toml` says MIT but repo lacks LICENSE; safe as ref |
| mahc | Rust scoring | BSD-3 | Secondary ref, esp `Fu` enum pattern |
| mahjax | RL environment | Apache-2.0 | Evaluate for JAX self-play training |
| RiichiEnv | Gym environment | Apache-2.0 | Evaluate for Python training loop |
| OpenSpiel | Self-play arch | Apache-2.0 | Study AlphaZero self-play loop design |
| Meowjong | Sanma AI | MIT | Ref for 3-player impl |
| mjai-reviewer | Evaluation | Apache-2.0 | Evaluate Hydra play quality |
| Microsoft Olive | ONNX optimization | MIT | Use for optimized inference deploy |

### P2 — Useful Tools

| Tool | Category | License | Action |
|------|----------|---------|--------|
| mahjong_ev | Defense/EV | Unlicensed | Port defense analyzer + EV engine concepts to Rust |
| torchdistill | Distillation | MIT | Oracle → blind model distill |
| tempai-core | Rules engine | MIT | Cross-lang ref for configurable rules |

### P3 — Watch / Future Use

| Tool | Category | License | Action |
|------|----------|---------|--------|
| Mortal-Policy | PPO fork | AGPL-3.0 | Study AWR→PPO transition approach |
| lizhisim | Simulator | MIT | Watch — same author as xiangting |
| Burn | Rust ML | Apache-2.0 | Long-term: native Rust training + inference |
| candle | Rust ML | Apache-2.0 | Alt: skip ONNX, write inference in Rust |