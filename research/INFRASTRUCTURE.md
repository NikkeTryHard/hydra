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
| xiangting | 5.0+ | Shanten calculation | MIT |
| pyo3 | 0.22+ | Python bindings | MIT OR Apache-2.0 |
| rayon | 1.10+ | Parallel simulation | MIT OR Apache-2.0 |
| serde | 1.0+ | JSON serialization | MIT OR Apache-2.0 |
| serde_json | 1.0+ | MJAI parsing | MIT OR Apache-2.0 |
| ndarray | 0.16+ | Tensor operations | MIT OR Apache-2.0 |
| rand | 0.9+ | RNG for shuffle | MIT OR Apache-2.0 |

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
| `encoder.rs` | Observation tensor encoder (84×34 output) |
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

The game engine drives a finite state machine that governs the flow of each round. States transition through dealing, drawing, discarding, call checks, kan processing, riichi declarations, and win checks until the round ends by tsumo, ron, or draw.

```mermaid
stateDiagram-v2
    [*] --> Dealing
    Dealing --> Drawing : Deal complete
    Drawing --> Discarding : Draw tile
    Drawing --> WinCheck : Can tsumo?
    Drawing --> GameEnd : Exhaustive draw
    Discarding --> CallCheck : Discard
    Discarding --> RiichiCheck : Riichi declared
    RiichiCheck --> CallCheck : Continue
    CallCheck --> Drawing : No call (next player)
    CallCheck --> Calling : Chi/Pon
    CallCheck --> KanProcess : Daiminkan (open kan)
    CallCheck --> WinCheck : Can ron?
    Calling --> Discarding : After call
    KanProcess --> Drawing : Rinshan draw (from dead wall)
    WinCheck --> GameEnd : Tsumo / Ron
    WinCheck --> Drawing : No win

    state WinCheck {
        [*] --> SingleWin : One winner
        [*] --> MultiRon : Multiple ron
        SingleWin --> [*]
        MultiRon --> [*]
    }

    state KanProcess {
        [*] --> FlipDora : New dora indicator
        FlipDora --> ChankanCheck : Can chankan?
        ChankanCheck --> DeadWallDraw : No chankan
        ChankanCheck --> GameEnd : Chankan ron
        DeadWallDraw --> RinshanCheck : Rinshan tsumo?
        RinshanCheck --> [*]
    }

    state GameEnd {
        [*] --> Tsumo
        [*] --> Ron
        [*] --> ExhaustiveDraw
        [*] --> AbortiveDraw
    }
```

**Abortive draws handled:**

| Condition | Japanese | Description |
|-----------|----------|-------------|
| Kyuushu Kyuuhai | 九種九牌 | 9+ unique terminals/honors in opening hand (player choice, action 44) |
| Suufon Renda | 四風連打 | All 4 players discard the same wind on their first turn |
| Suucha Riichi | 四家立直 | All 4 players declare riichi |
| Suukaikan | 四開槓 | 4 kans declared by different players (not all by one player) |
| Sanchahou | 三家和 | Triple ron (3 players win on same discard — abortive in most rulesets) |

> **Nagashi Mangan** is checked at exhaustive draw: if a player's entire discard pile consists of terminals and honors, and none were called by opponents, they receive mangan payment.

### Observation Encoder

The observation encoder produces the 87×34 tensor defined in the input encoding specification. It translates the current game state — hand tiles, discards, melds, dora indicators, and safety information — into a fixed-size numerical representation suitable for neural network input.

Key performance considerations:

- **Pre-allocated buffers** — tensor memory is allocated once per environment instance and reused across turns to avoid allocation overhead
- **Contiguous memory layout** — the 87×34 tensor is stored as a flat contiguous array for cache efficiency and compatibility with downstream BLAS/NN operations
- **Incremental updates (planned optimization)** — most planes change minimally between turns (hand ±1 tile, discards +1). Delta-based encoding could reduce per-turn work, pending benchmarking to confirm the bookkeeping overhead is worthwhile. Mortal's encoder recomputes the full tensor from scratch each turn.

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

Target throughput for pure Rust simulation (no NN inference): 100,000+ games/hour per core. End-to-end training throughput (with GPU inference in the loop) targets 10,000+ games/hour, bottlenecked by neural network forward passes rather than game simulation.

## Data Pipeline

This section specifies the complete data pipeline for Phase 1 behavioral cloning (supervised learning from expert game logs). It resolves five design gaps: storage format, data loading architecture, filtering strategy, suit permutation augmentation, and volume/throughput estimates. The same pipeline foundations extend to Phases 2–3 (oracle distillation and self-play RL), where the data source shifts from static logs to live self-play trajectories.

### Pipeline Architecture

```mermaid
graph LR
    subgraph "Storage"
        LOGS["MJAI .json.gz\n(raw game logs)"]
        MANIFEST["Manifest\n(filtered game list)"]
    end

    subgraph "Rust Data Workers (rayon)"
        LOADER["GameplayLoader\n(parse + encode)"]
        AUG["Suit Permutation\n(random 1-of-6)"]
    end

    subgraph "PyTorch Pipeline"
        IDS["IterableDataset"]
        DL["DataLoader\n(8 workers, pin_memory)"]
        SHUFFLE["3-Level Shuffle\n(file + buffer + reserve)"]
    end

    subgraph "Training"
        GPU["GPU\n(forward/backward)"]
    end

    LOGS --> LOADER
    MANIFEST --> LOADER
    LOADER --> AUG
    AUG --> IDS
    IDS --> SHUFFLE
    SHUFFLE --> DL
    DL --> GPU
```

### Gap 1: Storage Format

**Decision: On-the-fly Rust parsing (default) with optional pre-encoded shards for production.**

The default path stores raw MJAI logs as `.json.gz` files (~70 GB for 7M games). A Rust `GameplayLoader` parses and encodes observations on-the-fly using rayon parallelism. This is Mortal's proven architecture — Mortal processes even larger tensors (1012×34 vs Hydra's 84×34) with the same approach, demonstrating that on-the-fly parsing is not a bottleneck at this scale.

For production runs where maximum GPU utilization is critical, an optional pre-encoding step writes sharded binary files with Blosc+LZ4 compression (~500–800 GB for 7M games compressed at ~7:1 ratio). This path eliminates all CPU parsing overhead but requires re-encoding whenever features change. Pre-encoding is only justified if GPU utilization drops below 80% with on-the-fly parsing.

**Rejected alternatives:**
- **HDF5:** Thread-safety issues with h5py; single-writer bottleneck prevents parallel encoding.
- **Parquet:** Columnar format optimized for heterogeneous tabular data, wrong data model for dense homogeneous tensors.
- **FFCV:** Optimizes for JPEG decode-to-GPU pipeline — our bottleneck is not image decoding.
- **WebDataset:** Overkill for single-node local training.

**Evidence from game AI systems:** AlphaZero, AlphaStar, and OpenAI Five all use in-memory buffers or on-the-fly encoding. lc0 uses fixed-size binary structs in `.gz` files. KataGo uses `.npz` files. None use HDF5 or Parquet.

### Gap 2: Data Loading Pipeline

**Decision: PyTorch IterableDataset backed by Rust batch decoding via PyO3.**

Each DataLoader worker owns a disjoint slice of the file list (Mortal's partitioning pattern). Within each worker, the Rust `GameplayLoader` uses rayon parallel iterators and `GzDecoder` for concurrent file parsing, producing pre-encoded observation tensors that are yielded directly to the PyTorch training loop.

**DataLoader configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `batch_size` | 2048 | 4× Mortal's 512; use linear LR scaling rule |
| `num_workers` | 8 | Each worker spawns rayon threads internally |
| `RAYON_NUM_THREADS` | 4 per worker | 8×4 = 32 logical threads matches recommended CPU |
| `pin_memory` | `True` | Saves 0.5–2ms per batch on host-to-device transfer |
| `persistent_workers` | `True` | Avoids worker re-spawn overhead between epochs (Mortal misses this) |
| `prefetch_factor` | 2 | Default is sufficient for NVMe-backed storage |
| `drop_last` | `True` | Clean batch boundaries for consistent gradient scale |

**Shuffling strategy (3-level):**

1. **File-level:** Shuffle the shard/file order at the start of each epoch. Every worker receives a freshly permuted file list.
2. **Buffer-level:** Each worker loads `file_batch_size=100` shards (~100K decisions) into a memory buffer and performs a full Fisher-Yates shuffle within the buffer before yielding batches.
3. **Reserve mixing:** Retain 20% of the buffer when loading the next file batch, mixing old and new data to prevent sharp shard boundaries.

This 3-level strategy achieves temporal decorrelation comparable to a full-dataset shuffle without requiring the entire dataset to fit in memory. It mirrors KataGo's sliding-window shuffler design.

**Sharding:** Raw MJAI logs (7M individual game files) are pre-packed into ~700 mega-shards of 10K games each. Each shard is a concatenated gzip archive with a small JSON index mapping game offsets. This avoids filesystem metadata overhead from millions of small files and enables efficient sequential reads.

### Gap 3: Filtering Strategy

**Decision: Pre-filter once, cache filtered game list in a manifest file.**

Rather than filtering at training time (wasting CPU cycles re-evaluating criteria every epoch), a one-time scan builds a manifest of all games with metadata. Training loads only from the filtered manifest.

**Three-step process:**

1. **Scan:** Parse all 7M game files and extract metadata (player names, room/lobby, source, basic stats). Store as a JSON manifest or SQLite database.
2. **Filter:** Apply quality criteria per data tier (see below). Output a filtered file list.
3. **Train:** The DataLoader reads only files in the filtered list. Zero runtime filtering overhead.

**Data quality tiers:**

| Tier | Source | Filter Criteria | Training Weight |
|------|--------|-----------------|-----------------|
| Tier 1 | Tenhou Houou (Phoenix) | No additional filter (already R>=2000, 7-dan+) | 1.0 |
| Tier 1 | Majsoul Throne Room | No additional filter (Saint+ room) | 1.0 |
| Tier 2 | Majsoul Jade Room | Player-level stats filter (see below) | 0.5–0.7 |

**Player-level filtering for Tier 2 data** (inspired by Mortal Discussion #91, where Nitasurin demonstrated +1.8–2.0 PT improvement from player-level cleaning alone):

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| `games` | >= 30 | Minimum sample size for reliable stats |
| `avg_rank` | <= 2.60 | Better than random (2.50 = uniform) |
| `dealin_rate` | <= 0.16 | Not recklessly aggressive |
| `win_rate` | >= 0.17 | Actually wins hands |

Only qualifying players' decision perspectives are used for training, even in games that include weaker players. This trains the model on what good players do, not what their opponents do.

**Game-level exclusions:**

- Games with disconnected players (incomplete decision sequences)
- Games with >3 timeouts per player (AFK/bot behavior)
- Games with <4 actions per player (aborted or bugged games)

**Estimated dataset after filtering:** ~5–6M high-quality game perspectives from the 7M raw corpus.

### Gap 4: Suit Permutation Augmentation

**Decision: On-the-fly random permutation, 1 of 6 per game per epoch.**

Mahjong strategy is invariant to suit identity — manzu, pinzu, and souzu are functionally interchangeable. Exploiting this symmetry via suit permutation augmentation effectively multiplies the dataset by up to 6× without any disk overhead.

**Specification:**

- **Full permutation group:** All 3! = 6 permutations of the three suit indices. Mortal only implements 2× (man-pin swap); Hydra uses the complete group.
- **Application point:** At the MJAI Event level, before observation encoding. Each tile ID in the event stream is remapped through the chosen permutation. This is the same architectural point as Mortal but extended to all 6 permutations.
- **Granularity:** One random permutation is sampled per game, not per decision. This maintains game-internal consistency (the same physical tiles map to the same permuted tiles throughout the game).
- **Implementation:** `permute_suits(tile_id: u8, perm: [u8; 3]) -> u8` where `perm` maps `[manzu_target, pinzu_target, souzu_target]` and is one of the 6 permutations of `[0, 1, 2]`.
- **Aka-dora handling:** `deaka()` to get the base tile, permute the base, then `re_akaize()` if the original was aka. This is Mortal's proven pattern for handling the red five special tiles.
- **CPU cost:** Negligible — a `const fn` that divides by 9, remaps the suit, and reconstructs the tile ID. Nanoseconds per tile.
- **Coverage:** Over 6 epochs, each game is statistically seen under all 6 permutations. No need to pre-encode augmented copies.

### Gap 5: Volume and Throughput Estimates

All estimates scaled to 7M games × ~60 decisions per game = ~420M total decisions.

**Storage volumes:**

| Component | Size | Notes |
|-----------|------|-------|
| Raw MJAI logs (`.json.gz`) | ~70 GB | Source of truth; always retained |
| Student obs (84×34×f32, uncompressed) | ~4.5 TB | Never stored — generated on-the-fly |
| Oracle obs (205×34×f32, uncompressed) | ~10.9 TB | Never stored — generated on-the-fly |
| With 6× suit augmentation (uncompressed) | ~27 TB | Never pre-computed — applied on-the-fly |
| Pre-encoded shards (Blosc+LZ4, ~7:1) | ~640 GB | Optional production path only |
| Actions + masks + metadata | ~22 GB | Compact ancillary data |

**Memory budget (RTX PRO 6000 Blackwell, 96 GB VRAM):**

| Component | Memory | Phase |
|-----------|--------|-------|
| Model (16.7M params, bf16) | ~34 MB | All phases |
| Batch (2048 × 84×34 × 4B) | ~22 MB | Phase 1 (BC) |
| DataLoader workers (8 × prefetch 2) | ~180 MB | Phase 1 (BC) |
| PPO rollout buffer (4096 steps × 512 envs, +oracle) | ~77 GB | Phase 2–3 (RL) |
| **Total training footprint** | **< 1 GB** (BC) / **~78 GB** (PPO+oracle) | — |

The PPO rollout buffer in Phases 2–3 is the dominant memory consumer, using ~77 GB of the 96 GB available VRAM. Phase 1 behavioral cloning is memory-trivial.

**Throughput estimates:**

| Metric | Value | Notes |
|--------|-------|-------|
| Target training speed | 10K steps/hour @ batch 2048 | ~20M samples/hour |
| Required sustained I/O | ~32 MB/s | From MJAI source files |
| NVMe sequential read capacity | ~7,000 MB/s | 218× headroom over requirement |
| Rust parser capacity (8 workers) | ~160K samples/sec | 28× headroom over requirement |
| GPU compute (forward + backward) | Bottleneck | Only true constraint |

**Bottom line:** On-the-fly Rust parsing is the clear winner. Both NVMe I/O and CPU parsing capacity are massively overprovisioned relative to the training throughput target. The GPU is the sole bottleneck — specifically the forward/backward pass during training and the PPO rollout buffer size during Phases 2–3.

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

`VectorEnv` wraps multiple `MahjongEnv` instances and manages them in parallel, providing a batched API for efficient PPO rollout collection.

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
| mahjong (dev) | Hand scoring oracle for Rust engine verification — yaku/han/fu/score validation against 11M+ Tenhou hands (pin to v1.4.0) |

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

- **`torch.compile()`** is used for optimized inference throughput and is available for training (Mortal uses it via `config['control']['enable_compile']`). For PPO training, dynamic action masks may trigger occasional recompilation, so this should be benchmarked per-workload.
- **Mixed precision (fp16) training** reduces memory usage and increases GPU throughput
- **Gradient checkpointing** is available as an option to trade compute for memory if larger batch sizes are needed (not used in Mortal's training loop but supported by the architecture)

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
| GPU | RTX 3080 (10GB) | RTX PRO 6000 Blackwell (96GB) |
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
| Simulation throughput (CPU only) | 100k+ games/hour/core |
| Training throughput (with GPU inference) | 10k+ games/hour |
| Inference latency | <15ms |
| Inference VRAM | <1.5GB |
| Model size (fp16) | ~33MB |

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
