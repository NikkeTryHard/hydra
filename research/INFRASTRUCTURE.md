# Hydra Infrastructure Specification

## Overview

Hydra uses a hybrid Rust + Python architecture. Rust handles the game engine, observation encoding, and simulation — everything that benefits from low-level performance. Python handles neural network training and experiment tracking — everything that benefits from the ML ecosystem. This mirrors Mortal's design but with all original code (no AGPL-derived components).

## System Architecture

The system is composed of four major subsystems: the Rust core game engine, Python bindings via PyO3, the Python training stack, and the deployment pipeline. Data flows from the game engine through PyO3 into the training loop, and trained models are exported back through ONNX for pure-Rust inference.

```mermaid
graph TB
    subgraph "Rust Core (hydra-core)"
        ENGINE[Game Engine]
        SHANTEN[xiangting crate]
        ENCODER[Observation Encoder]
        MJAI[MJAI Protocol]
        SIM[Batch Simulator]
    end

    subgraph "Python Bindings (hydra-py)"
        PYO3[PyO3 Bridge]
        ENV[Gym-like Environment]
    end

    subgraph "Python Training (hydra-train)"
        MODEL[PyTorch Model]
        TRAINER[PPO Trainer]
        WANDB[Weights & Biases]
    end

    subgraph "Deployment"
        ONNX[ONNX Export]
        INFER[Rust Inference]
    end

    ENGINE --> SHANTEN
    ENGINE --> ENCODER
    ENGINE --> MJAI
    ENGINE --> SIM
    SIM --> PYO3
    PYO3 --> ENV
    ENV --> TRAINER
    MODEL --> TRAINER
    TRAINER --> WANDB
    MODEL --> ONNX
    ONNX --> INFER
```

## Rust Core (hydra-core)

### Crate Dependencies

| Crate | Version | Purpose | License |
|-------|---------|---------|---------|
| xiangting | 0.5+ | Shanten calculation | MIT |
| pyo3 | 0.22+ | Python bindings | Apache-2.0 |
| rayon | 1.10+ | Parallel simulation | Apache-2.0 |
| serde | 1.0+ | JSON serialization | Apache-2.0 |
| serde_json | 1.0+ | MJAI parsing | Apache-2.0 |
| ndarray | 0.16+ | Tensor operations | Apache-2.0 |
| rand | 0.8+ | RNG for shuffle | Apache-2.0 |

### Module Structure

The `hydra-core` crate is organized as a flat module layout under `src/`:

| File | Responsibility |
|------|----------------|
| `lib.rs` | Crate root and public API surface |
| `tile.rs` | Tile representation using the 0–33 index scheme |
| `hand.rs` | Hand management and meld tracking |
| `wall.rs` | Wall, dead wall, and dora indicator logic |
| `player.rs` | Player state and discard tracking |
| `game.rs` | Game state machine (see state diagram below) |
| `rules.rs` | Riichi rules validation and scoring |
| `shanten.rs` | Wrapper around the `xiangting` crate for shanten calculation |
| `encoder.rs` | Observation tensor encoder (87×34 output) |
| `safety.rs` | Suji, kabe, and genbutsu safety calculations |
| `mjai.rs` | MJAI protocol parser for log compatibility |
| `simulator.rs` | Batch game simulation with rayon parallelism |
| `python.rs` | PyO3 binding definitions exposed to Python |

### Tile Representation

The standard 34-tile index used by the `xiangting` crate:

| Index | Tiles |
|-------|-------|
| 0–8 | 1–9m (Manzu) |
| 9–17 | 1–9p (Pinzu) |
| 18–26 | 1–9s (Souzu) |
| 27–33 | E S W N 白 發 中 |

### Game State Machine

The game engine drives a finite state machine that governs the flow of each round. States transition through dealing, drawing, discarding, call checks, riichi declarations, and win checks until the round ends by tsumo, ron, or exhaustive draw.

```mermaid
stateDiagram-v2
    [*] --> Dealing
    Dealing --> Drawing : Deal complete
    Drawing --> Discarding : Draw tile
    Discarding --> CallCheck : Discard
    CallCheck --> Drawing : No call (next player)
    CallCheck --> Calling : Call made
    Calling --> Discarding : After call
    Discarding --> RiichiCheck : Riichi declared
    RiichiCheck --> Drawing : Continue
    Drawing --> WinCheck : Can tsumo?
    WinCheck --> GameEnd : Tsumo
    CallCheck --> WinCheck : Can ron?
    WinCheck --> GameEnd : Ron
    WinCheck --> Drawing : No win
    Drawing --> GameEnd : Exhaustive draw
```

### Observation Encoder

The observation encoder produces the 87×34 tensor defined in the input encoding specification. It translates the current game state — hand tiles, discards, melds, dora indicators, and safety information — into a fixed-size numerical representation suitable for neural network input.

Key performance considerations:

- **Incremental updates** — the encoder does not recompute the full tensor each turn; it applies deltas from the previous state
- **SIMD-friendly array operations** — data layouts are chosen to enable vectorized computation
- **Pre-allocated buffers** — tensor memory is allocated once and reused across turns to avoid allocation overhead

### Batch Simulator

For self-play training, the batch simulator runs many games in parallel using rayon's work-stealing thread pool. Each game runs independently, and rayon distributes games across available CPU threads automatically.

```mermaid
graph LR
    subgraph "Parallel Simulation"
        G1[Game 1]
        G2[Game 2]
        G3[Game ...]
        GN[Game N]
    end

    subgraph "Rayon Thread Pool"
        T1[Thread 1]
        T2[Thread 2]
        TM[Thread M]
    end

    G1 --> T1
    G2 --> T1
    G3 --> T2
    GN --> TM
```

Target throughput: 10,000+ games/hour on a modern CPU.

## Python Bindings (hydra-py)

### PyO3 Interface

The following Rust functions are exposed to Python via PyO3:

| Python Function | Rust Implementation |
|-----------------|---------------------|
| `Game.new()` | `Game::new()` |
| `Game.step(action)` | `Game::step()` |
| `Game.get_observation()` | `Encoder::encode()` |
| `Game.legal_actions()` | `Game::legal_actions()` |
| `Game.is_done()` | `Game::is_terminal()` |
| `simulate_batch(n)` | `Simulator::run_batch()` |

### Gym-like Environment

The Python environment wraps the Rust game engine in a Gymnasium-compatible interface. Two classes are provided:

**MahjongEnv** — single-environment interface for one game at a time:

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset()` | → `observation` | Resets the environment and returns the initial observation tensor |
| `step(action)` | → `(obs, reward, done, info)` | Advances the game by one action, returning the new observation, reward signal, terminal flag, and info dictionary |
| `legal_actions()` | → `list` | Returns the list of currently legal action indices |
| `render()` | → `str` | Returns a human-readable string representation of the current game state |

**VectorEnv** — batched interface that manages multiple `MahjongEnv` instances in parallel:

| Method / Attribute | Signature | Description |
|--------------------|-----------|-------------|
| `num_envs` | `int` (attribute) | The number of parallel environments in the batch |
| `reset()` | → `observations` | Resets all environments and returns a batch of initial observations |
| `step(actions)` | → `(obs, rewards, dones, infos)` | Steps all environments simultaneously with a batch of actions |

`VectorEnv` extends `MahjongEnv`, inheriting its interface while adding batched operation.

### Installation

The `hydra-py` package is built and installed via maturin:

```
maturin develop --release
```

## Python Training (hydra-train)

### Dependencies

| Package | Purpose |
|---------|---------|
| torch | Neural network definition and training |
| numpy | Array operations and data manipulation |
| wandb | Experiment tracking and visualization |
| hydra-py | Game environment (Rust-backed) |
| einops | Tensor reshaping utilities |
| rich | Progress bars and terminal output |

### Training Loop Architecture

The training loop follows the standard PPO (Proximal Policy Optimization) cycle: collect rollout data from the vectorized environment, sample mini-batches, compute forward and backward passes, update the policy, and repeat. All metrics flow to Weights & Biases for monitoring.

```mermaid
graph TB
    subgraph "Data Collection"
        ENV[VectorEnv] --> ROLLOUT[Rollout Buffer]
    end

    subgraph "Policy Update"
        ROLLOUT --> BATCH[Batch Sampler]
        BATCH --> FORWARD[Forward Pass]
        FORWARD --> LOSS[Loss Calculation]
        LOSS --> BACKWARD[Backward Pass]
        BACKWARD --> OPTIM[Optimizer Step]
    end

    subgraph "Logging"
        LOSS --> WANDB[W&B Logger]
        METRICS[Metrics] --> WANDB
    end

    OPTIM --> ENV
```

### Model Definition

The PyTorch model implements the neural network architecture defined in the architecture specification. Key considerations for the implementation:

- **`torch.compile()`** is used for optimized inference and training throughput
- **Mixed precision (fp16) training** reduces memory usage and increases GPU throughput
- **Gradient checkpointing** trades compute for memory, enabling larger batch sizes

## Deployment

### ONNX Export

For production inference, trained PyTorch models are exported to the ONNX format. This enables the model to run through ONNX Runtime, which can then be called from Rust — eliminating the Python dependency entirely at serving time.

```mermaid
graph LR
    PT[PyTorch Model] --> EXPORT[torch.onnx.export]
    EXPORT --> ONNX[ONNX Model]
    ONNX --> ORT[ONNX Runtime]
    ORT --> RUST[Rust Inference]
```

### Rust Inference Options

For maximum performance, inference can run entirely in Rust using one of three crate options:

| Crate | Description |
|-------|-------------|
| `ort` | Rust bindings to ONNX Runtime — mature, GPU-accelerated, broadest operator coverage |
| `tract` | Pure Rust inference engine — no C/C++ dependencies, easier to cross-compile |
| `candle` | HuggingFace's Rust ML framework — native tensor operations, growing ecosystem |

The core advantage of Rust-side inference is the complete elimination of Python at runtime. This removes the GIL, startup overhead, and Python dependency management from the inference path, enabling sub-15ms decision latency suitable for real-time play.

## Hardware Requirements

### Training

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 (10GB) | RTX 6000 (98GB) |
| CPU | 8 cores | 32+ cores |
| RAM | 32GB | 128GB+ |
| Storage | 100GB SSD | 1TB NVMe |

### Inference

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 (6GB) | RTX 4070 (12GB) |
| CPU | 4 cores | 8 cores |
| RAM | 8GB | 16GB |

## Performance Targets

| Metric | Target |
|--------|--------|
| Training throughput | 10k+ games/hour |
| Inference latency | <15ms |
| Inference VRAM | <1.5GB |
| Model size (fp16) | ~80MB |

## Development Workflow

The end-to-end workflow flows through three phases: development (write code, run tests, benchmark), training (launch training runs, monitor via W&B, evaluate results), and deployment (export to ONNX, deploy for inference).

```mermaid
graph LR
    subgraph "Development"
        CODE[Write Code] --> TEST[Run Tests]
        TEST --> BENCH[Benchmark]
    end

    subgraph "Training"
        BENCH --> TRAIN[Launch Training]
        TRAIN --> MONITOR[Monitor W&B]
        MONITOR --> EVAL[Evaluate]
    end

    subgraph "Deployment"
        EVAL --> EXPORT[Export ONNX]
        EXPORT --> DEPLOY[Deploy]
    end
```
