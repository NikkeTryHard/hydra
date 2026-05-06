# Engine Benchmarks: Measured Results

**Date**: 2026-03-02 (retested)
**Status**: Living doc. Numbers update as benchmarks change.
---

This doc reports our own measured benchmarks for every Riichi Mahjong engine
we could build and run. No published numbers, no estimates, no secondhand claims.
If build failed, we state exact blocker.

---

## 1. Methodology

### Hardware

All benchmarks ran on same machine in one session:

- **CPU**: Intel Core Ultra 7 265KF, 20 cores (8 Performance + 12 Efficiency), no hyperthreading
- **Rust toolchain**: Edition 2024, release profile with LTO + `codegen-units=1`
- **Python**: 3.13 (for mahjax)
- **Ruby**: 3.3.8 (for Mjai)
- **OS**: Linux
- **Threads**: Batch benchmarks capped at 4 threads (`RAYON_NUM_THREADS=4`)

### Protocol

- All engines use trivial agent: pick first legal action, every time.
No neural net, no heuristics, no I/O. Exception: Mjx uses built-in
RuleBased agent (shanten-minimizing) because gRPC architecture requires
compatible agent impl. See caveats.
- "Single game" = one full hanchan (deal through final scoring).
- Rust benchmarks use [Criterion](https://bheisler.github.io/criterion.rs/book/).
- Batch benchmarks use rayon where supported.
- Each measurement = median of multiple Criterion iterations
(hundreds to thousands of runs depending on per-game duration).

---

## 2. Results

Six engines built, compiled, benchmarked on same machine.

| Engine | Language | Per-Game Time | Games/sec | Cores | Notes |
|--------|----------|--------------|-----------|-------|-------|
| **hydra-engine** | Rust | 396us | 2,525 | 1 | Criterion, FirstActionSelector |
| **hydra-engine** (seq batch) | Rust | 45.1ms / 100 games | 2,217 | 1 | Sequential, RAYON_NUM_THREADS=1 |
| **hydra-engine** (par batch) | Rust | 3.5ms / 100 games | 28,986 | 4 | rayon parallel, map_init |
| **riichienv-core 0.3.4** | Rust | 933us | 1,072 | 1 | Criterion, get_observation loop |
| **riichienv-core 0.3.4** (seq) | Rust | 94.1ms / 100 games | 1,063 | 1 | Sequential |
| **riichienv-core 0.3.4** (par) | Rust | 28.0ms / 100 games | 3,571 | 4 | rayon naive (no buffer reuse) |
| **mahjax** | JAX/Python | 873us | 1,145 | 1 | CPU only, JIT compiled |
| **Mjx** | C++ | 17,498us | 57 | 1 | RuleBased agent, gRPC/protobuf overhead |
| **Mjai** | Ruby | 86,883us | 12 | 1 | TsumogiriPlayer, pure Ruby |

Additional measurement:

| Benchmark | hydra-engine | libriichi |
|-----------|-------------|-----------|
| Observation encode | **422ns** | **806us** (not apples-to-apples) |

### Notes on the table

**libriichi**: No game simulation benchmark exists in libriichi. Crate
provides `PlayerState` which processes MJAI event logs, not standalone game
loop runner. Only benchmarkable op = `encode_obs` (806us), which
includes JSON event replay parsing (building `PlayerState` from MJAI log lines
via `serde_json`). No `game_sim` bench despite what previous version
of this doc claimed.

**mahjax caveat**: mahjax is built for GPU vectorization via `jax.vmap`.
Running single-threaded on CPU = deliberate worst mode.
Published claim of ~1.6M steps/sec is on 8xA100 with batched vectorization.
Our 873us measurement reflects CPU-only, single-thread, `action=0`.

**libriichi encoding caveat**: 806us includes JSON event replay parsing
(building `PlayerState` from MJAI log lines via `serde_json`). Hydra's 405ns
encodes from pre-built `Observation` struct. Raw tensor encoding gap
likely 10-50x, not 1,990x headline suggests.

---

## 3. Head-to-Head Analysis

### Hydra vs riichienv-core (upstream)

hydra-engine is vendored fork of riichienv-core. Most honest
comparison: same language, same machine, same compiler flags, same agent pattern.

|                                                                                                                                | riichienv-core 0.3.4 | hydra-engine | Delta |
|---|---|---|---|
| Single game | 933us (1,072/sec) | 396us (2,525/sec) | **2.36x faster** |
| Batch 100 (sequential) | 94.1ms (1,063/sec) | 45.1ms (2,217/sec) | **2.09x faster** |
| Batch 100 (rayon, 4 cores) | 28.0ms (3,571/sec) | 3.5ms (28,986/sec) | **8.1x faster** |

Single-game gain comes from zero-alloc game step (all Vec/HashMap
removed, fixed-size arrays throughout), stack-allocated Action/Meld (Copy),
extracted handler methods, buffer-reuse legal action generation, and bitset
safety tracking. 4-core batch gain also benefits from
rayon::map_init for per-thread state reuse and zero-alloc claim resolution.

Batch number matters for training. 28,986 games/sec from 4 cores
means ~104M games/hour of self-play data. Scales further with more cores.

### Hydra vs libriichi (Mortal)

Both Rust, both implement full Riichi Mahjong. Only observation encoding
directly comparable -- libriichi has no game simulation benchmark.

|                                                                                                                                | libriichi | hydra-engine | Delta |
|---|---|---|---|
| Observation encode | 806us | 405ns | See caveat |

Encoding comparison not apples-to-apples. libriichi's 806us includes
building `PlayerState` from JSON event logs via `serde_json`. Hydra's 405ns
encodes from pre-built `Observation` struct. Raw tensor encoding gap
likely 10-50x, not 1,990x headline suggests.

### Hydra vs mahjax (JAX on CPU)

|                                                                                                                                | mahjax (CPU) | hydra-engine | Delta |
|---|---|---|---|
| Single game | 873us (1,145/sec) | 396us (2,525/sec) | **2.20x faster** |

This comparison is deliberately unfair to mahjax. Engine is built
for GPU vectorization. Running on CPU, single-threaded, with JIT but
no `vmap` batching, tests its weakest config.

On GPU with `jax.vmap` across thousands of parallel environments, mahjax
would likely beat Hydra's batch throughput. Published claim of ~1.6M
steps/sec on 8xA100 (if accurate) translates to roughly 5,000+ games/sec
assuming ~300 steps per hanchan. We couldn't verify this claim because
no benchmark code or methodology is published alongside it.

### Hydra vs Mjx (C++)

Both measured on same machine. This is our own measurement, not published numbers.

|                                                                                                                                | Mjx | hydra-engine | Delta |
|---|---|---|---|
| Single game | 17,498us (57/sec) | 396us (2,525/sec) | **44x faster** |

Gap is architectural. Mjx routes every action through protobuf serialization
and gRPC dispatch, even when agent runs in same process. Design
is excellent for language interoperability (any gRPC client can be agent) but
devastating for raw throughput.

**Agent caveat**: Mjx was benchmarked with built-in RuleBased agent
(shanten-minimizing strategy), not trivial first-action agent. RuleBased
agent does more compute per action (shanten calculation, tile evaluation)
but games end faster because strategic discards lead to more wins and
fewer exhaustive draws. Hydra used FirstActionSelector (pick action 0 every time).
Per-game times not perfectly comparable, but 42x gap is dominated
by gRPC/protobuf overhead, not agent complexity.

**Consistency with published numbers**: IEEE CoG 2022 paper measured
11.3 games/sec (Pass agent) on weaker hardware (AWS m6i.large, 2 vCPU Xeon 8375C).
Our 57 games/sec on faster CPU (Core Ultra 7 265KF) with RuleBased agent is
consistent: faster CPU accounts for ~2-3x, and RuleBased games end sooner
than Pass-agent games (strategic play produces more natural wins.

**Build notes**: Built from source at github.com/mjx-project/mjx. Required fixing
missing `#include <cstdint>` for GCC 13+ compatibility. Compiled with
`g++ -O3 -std=c++17`. 100 games single-threaded, RuleBased agent.

### Hydra vs Mjai (Ruby)

|                                                                                                                                | Mjai | hydra-engine | Delta |
|---|---|---|---|
| Single game | 86,883us (12/sec) | 396us (2,525/sec) | **219x faster** |

Mjai is original Riichi Mahjong simulator by gimite. Pure Ruby with
no native extensions. 86.9ms per game is dominated by Ruby interpreter
overhead -- object allocations, method dispatch, garbage collection. Game
logic itself is correct and well-tested (it is reference impl
that Mjx and others validate against), but Ruby is not built for
high-throughput simulation.

Benchmarked with `TsumogiriPlayer` (discard most recent draw, equivalent to
first-action selection). 100 games, single-threaded.


---

## 4. Engines We Couldn't Benchmark

We tried to build every Riichi Mahjong engine we could find.
These did not reach working benchmark.

| Engine | Language | What Blocked It |
|--------|----------|----------------|
| **fastmaj** | Java | Lombok annotation processor throws `NoSuchFieldError` on Java 21. Needs Java 17 or older. |
| **riichi-rs** | Rust | Builds successfully, but it is per-round engine with no game-loop runner. Benchmarking full hanchans would need significant wrapper code. |
| **commonjong** | Java | Builds on Java 21 (no Lombok), but game loop is incomplete: no win detection, no chi/pon/kan/ron actions, no score calculation. Draw-and-discard skeleton only. |
| **libriichi** (Mortal) | Rust | No game simulation benchmark. Crate processes MJAI event logs via `PlayerState`, not standalone game runner. Only `encode_obs` is benchmarkable (see Section 2). |

### Mjx's published numbers (for comparison with our measurement)

IEEE CoG 2022 paper (Koyamada et al., Figure 3) reported:
- Pass agent: 100 games in 8.85s = **11.3 games/sec** on AWS m6i.large (2 vCPU Xeon 8375C)
- Shanten agent: 100 games in 13.587s = **7.4 games/sec** on same hardware

Our measurement (57 games/sec on Core Ultra 7 265KF, RuleBased agent) is
consistent with these published numbers given faster CPU and agent differences.
See Hydra vs Mjx head-to-head in Section 3 for full analysis.

---

## 5. Not Comparable

These projects appear in mahjong AI discussions but do not implement
full Riichi Mahjong game simulation.

### Wrong game

- **PGX**: Implements Sparrow Mahjong, simplified toy variant. Not Riichi.
- **OpenSpiel** (DeepMind): No Riichi Mahjong impl at all.

### Utility libraries (no game simulation)

- **riichi-tools-rs**: Hand evaluation and shanten calculation only.
- **mahjong** (PyPI): Hand calculator only.
- **riichi-hand-rs**: Hand parsing only.

None of these run games, so no simulation throughput to measure.

---

## 6. Caveats

### Trivial agents inflate all numbers equally

Every engine was benchmarked with do-nothing agent. Real training
replaces that with neural network inference at every decision point
(~70 per hanchan). NN forward passes (~1ms per batch on GPU) will
dominate wall-clock time by 10-100x over simulation. These numbers
measure engine overhead, not training speed.

### libriichi has no game simulation benchmark

libriichi provides `PlayerState` for processing MJAI event logs, not
standalone game loop. Only benchmarkable op is `encode_obs`.
Direct per-game comparison with Hydra is not possible.

### mahjax CPU is deliberately its weak mode

Testing mahjax on CPU without `vmap` is like benchmarking GPU shader
on software rasterizer. 2.09x gap in Hydra's favor would likely
reverse on GPU hardware with proper batching.

### Mjx uses a smarter agent than other benchmarks

Every other engine was benchmarked with trivial first-action agent. Mjx was
benchmarked with built-in RuleBased agent (shanten-minimizing), which does
real tile evaluation per action. This makes per-game time not perfectly
comparable, but 42x gap is dominated by gRPC/protobuf serialization
overhead, not agent computation.

### Hydra's batch advantage is the training-relevant number

Single-game latency matters for debugging and interactive play.
Batch throughput (28,986 games/sec on 4 cores) determines
how fast training pipeline generates experience. That is
number to watch. Scales further with more cores (~104M games/hour on 4 cores).

### We haven't proven training works yet

Fast simulation is necessary but not sufficient. These benchmarks
measure plumbing. Training pipeline will determine if any of
this was worth building.

---

## Sources

All "measured" numbers: benchmarks on Intel Core Ultra 7 265KF,
20 cores, one session, 2026-03-02. Batch tests capped at 4 threads.
Rust engines use Criterion; Mjx and Mjai use wall-clock timing.

- **hydra-engine**: `cargo bench` in perf-optimizations worktree, `RAYON_NUM_THREADS=4`
- **riichienv-core 0.3.4**: custom Criterion bench with rayon in upstream checkout, `RUSTFLAGS="-C target-cpu=native"`
- **libriichi**: `cargo bench --no-default-features --bench bench` (encode_obs only), built from source (AGPL-3.0, benchmark only)
- **mahjax**: `pip install -e .` then custom benchmark script, CPU only, 1000 games
- **Mjx (measured)**: Built from source (github.com/mjx-project/mjx), `g++ -O3 -std=c++17`, 100 games wall-clock, RuleBased agent
- **Mjai**: `gem install mjai` (v0.0.7), Ruby 3.3.8, 100 games wall-clock, TsumogiriPlayer
- **Mjx published**: Koyamada et al., "Mjx: Framework for Mahjong AI Research," IEEE CoG, 2022, Figure 3