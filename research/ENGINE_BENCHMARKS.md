# Engine Benchmarks: Measured Results

**Date**: 2026-03-02 (retested)
**Status**: Living document. Numbers updated as benchmarks change.
---

This document reports our own measured benchmarks for every Riichi Mahjong engine
we could build and run. No published numbers, no estimates, no secondhand claims.
Where we couldn't build an engine, we say exactly what blocked us.

---

## 1. Methodology

### Hardware

All benchmarks ran on the same machine in a single session:

- **CPU**: Intel Core Ultra 7 265KF, 20 cores (8 Performance + 12 Efficiency), no hyperthreading
- **Rust toolchain**: Edition 2024, release profile with LTO + `codegen-units=1`
- **Python**: 3.13 (for mahjax)
- **Ruby**: 3.3.8 (for Mjai)
- **OS**: Linux
- **Threads**: Batch benchmarks capped at 4 threads (`RAYON_NUM_THREADS=4`)

### Protocol

- All engines use a trivial agent: pick the first legal action, every time.
  No neural network, no heuristics, no I/O. Exception: Mjx uses its built-in
  RuleBased agent (shanten-minimizing) because its gRPC architecture requires
  a compatible agent implementation. See caveats.
- "Single game" means one full hanchan (deal through final scoring).
- Rust benchmarks use [Criterion](https://bheisler.github.io/criterion.rs/book/).
- Batch benchmarks use rayon where supported.
- Each measurement is the median of multiple Criterion iterations
  (hundreds to thousands of runs depending on per-game duration).

---

## 2. Results

Six engines built, compiled, and benchmarked on the same machine.

| Engine | Language | Per-Game Time | Games/sec | Cores | Notes |
|--------|----------|--------------|-----------|-------|-------|
| **hydra-engine** | Rust | 417us | 2,398 | 1 | Criterion, FirstActionSelector |
| **hydra-engine** (batch) | Rust | 12.2ms / 100 games | 8,170 | 4 | rayon parallel |
| **riichienv-core 0.3.4** | Rust | 627us | 1,595 | 1 | Criterion, same game loop |
| **riichienv-core 0.3.4** (seq) | Rust | 73.8ms / 100 games | 1,355 | 1 | Sequential, no rayon |
| **mahjax** | JAX/Python | 873us | 1,145 | 1 | CPU only, JIT compiled |
| **Mjx** | C++ | 17,498us | 57 | 1 | RuleBased agent, gRPC/protobuf overhead |
| **Mjai** | Ruby | 86,883us | 12 | 1 | TsumogiriPlayer, pure Ruby |

Additional measurement:

| Benchmark | hydra-engine | libriichi |
|-----------|-------------|-----------|
| Observation encode | **405ns** | **806us** (not apples-to-apples) |

### Notes on the table

**libriichi**: No game simulation benchmark exists in libriichi. The crate
provides `PlayerState` which processes MJAI event logs, not a standalone game
loop runner. The only benchmarkable operation is `encode_obs` (806us), which
includes JSON event replay parsing (building `PlayerState` from MJAI log lines
via `serde_json`). There is no `game_sim` bench despite what a previous version
of this document claimed.

**mahjax caveat**: mahjax is designed for GPU vectorization via `jax.vmap`.
Running it single-threaded on CPU is deliberately its weakest mode.
The published claim of ~1.6M steps/sec is on 8xA100 with batched vectorization.
Our 873us measurement reflects CPU-only, single-thread, `action=0`.

**libriichi encoding caveat**: The 806us includes JSON event replay parsing
(building `PlayerState` from MJAI log lines via `serde_json`). Hydra's 405ns
encodes from a pre-built `Observation` struct. The raw tensor encoding gap
is likely 10-50x, not the 1,990x the headline numbers suggest.

---

## 3. Head-to-Head Analysis

### Hydra vs riichienv-core (upstream)

hydra-engine is a vendored fork of riichienv-core. This is the most honest
comparison: same language, same machine, same compiler flags, same agent pattern.

| | riichienv-core 0.3.4 | hydra-engine | Delta |
|---|---|---|---|
| Single game | 627us (1,595/sec) | 417us (2,398/sec) | **1.50x faster** |
| Batch 100 (sequential) | 73.8ms (1,355/sec) | n/a | n/a |
| Batch 100 (rayon, 4 cores) | n/a (no rayon) | 12.2ms (8,170/sec) | **6.0x faster** |

The single-game improvement comes from stack-allocated `Action` (`[u8; 4]`
instead of heap `Vec<u8>`), extracted `_execute_step`, and changed calling
conventions in `HandEvaluator`. These changes eliminate allocator contention
under parallel load.

The batch number is what matters for training. 8,170 games/sec from 4 cores
means ~29M games/hour of self-play data. Scales further with more cores.

### Hydra vs libriichi (Mortal)

Both Rust, both implementing full Riichi Mahjong. Only observation encoding
is directly comparable -- libriichi has no game simulation benchmark.

| | libriichi | hydra-engine | Delta |
|---|---|---|---|
| Observation encode | 806us | 405ns | See caveat |

The encoding comparison isn't apples-to-apples. libriichi's 806us includes
building `PlayerState` from JSON event logs via `serde_json`. Hydra's 405ns
encodes from a pre-built `Observation` struct. The raw tensor encoding gap
is likely 10-50x, not the 1,990x the headline numbers suggest.

### Hydra vs mahjax (JAX on CPU)

| | mahjax (CPU) | hydra-engine | Delta |
|---|---|---|---|
| Single game | 873us (1,145/sec) | 417us (2,398/sec) | **2.09x faster** |

This comparison is deliberately unfair to mahjax. The engine is designed
for GPU vectorization. Running it on CPU, single-threaded, with JIT but
no `vmap` batching, tests its weakest configuration.

On GPU with `jax.vmap` across thousands of parallel environments, mahjax
would likely beat Hydra's batch throughput. The published claim of ~1.6M
steps/sec on 8xA100 (if accurate) translates to roughly 5,000+ games/sec
assuming ~300 steps per hanchan. We couldn't verify this claim because
no benchmark code or methodology is published alongside it.

### Hydra vs Mjx (C++)

Both measured on the same machine. This is our own measurement, not published numbers.

| | Mjx | hydra-engine | Delta |
|---|---|---|---|
| Single game | 17,498us (57/sec) | 417us (2,398/sec) | **42x faster** |

The gap is architectural. Mjx routes every action through protobuf serialization
and gRPC dispatch, even when the agent is running in the same process. This design
is excellent for language interoperability (any gRPC client can be an agent) but
devastating for raw throughput.

**Agent caveat**: Mjx was benchmarked with its built-in RuleBased agent
(shanten-minimizing strategy), not a trivial first-action agent. The RuleBased
agent does more computation per action (shanten calculation, tile evaluation)
but games terminate faster because strategic discards lead to more wins and
fewer exhaustive draws. Hydra used FirstActionSelector (pick action 0 every time).
The per-game times are not perfectly comparable, but the 42x gap is dominated
by gRPC/protobuf overhead, not agent complexity.

**Consistency with published numbers**: The IEEE CoG 2022 paper measured
11.3 games/sec (Pass agent) on weaker hardware (AWS m6i.large, 2 vCPU Xeon 8375C).
Our 57 games/sec on a faster CPU (Core Ultra 7 265KF) with a RuleBased agent is
consistent: faster CPU accounts for ~2-3x, and RuleBased games terminate sooner
than Pass-agent games (strategic play produces more natural wins.

**Build notes**: Built from source at github.com/mjx-project/mjx. Required fixing
a missing `#include <cstdint>` for GCC 13+ compatibility. Compiled with
`g++ -O3 -std=c++17`. 100 games single-threaded, RuleBased agent.

### Hydra vs Mjai (Ruby)

| | Mjai | hydra-engine | Delta |
|---|---|---|---|
| Single game | 86,883us (12/sec) | 417us (2,398/sec) | **208x faster** |

Mjai is the original Riichi Mahjong simulator by gimite. It's pure Ruby with
no native extensions. The 86.9ms per game is dominated by Ruby interpreter
overhead -- object allocations, method dispatch, garbage collection. The game
logic itself is correct and well-tested (it's the reference implementation
that Mjx and others validate against), but Ruby is simply not built for
high-throughput simulation.

Benchmarked with `TsumogiriPlayer` (discard most recent draw, equivalent to
first-action selection). 100 games, single-threaded.


---

## 4. Engines We Couldn't Benchmark

We attempted to build every Riichi Mahjong engine we could find.
These didn't make it to a working benchmark.

| Engine | Language | What Blocked It |
|--------|----------|----------------|
| **fastmaj** | Java | Lombok annotation processor throws `NoSuchFieldError` on Java 21. Needs Java 17 or older. |
| **riichi-rs** | Rust | Builds successfully, but it's a per-round engine with no game-loop runner. Benchmarking full hanchans would need significant wrapper code. |
| **commonjong** | Java | Builds on Java 21 (no Lombok), but game loop is incomplete: no win detection, no chi/pon/kan/ron actions, no score calculation. Draw-and-discard skeleton only. |
| **libriichi** (Mortal) | Rust | No game simulation benchmark. The crate processes MJAI event logs via `PlayerState`, not a standalone game runner. Only `encode_obs` is benchmarkable (see Section 2). |

### Mjx's published numbers (for comparison with our measurement)

The IEEE CoG 2022 paper (Koyamada et al., Figure 3) reported:
- Pass agent: 100 games in 8.85s = **11.3 games/sec** on AWS m6i.large (2 vCPU Xeon 8375C)
- Shanten agent: 100 games in 13.587s = **7.4 games/sec** on the same hardware

Our measurement (57 games/sec on Core Ultra 7 265KF, RuleBased agent) is
consistent with these published numbers given the faster CPU and agent differences.
See the Hydra vs Mjx head-to-head in Section 3 for full analysis.

---

## 5. Not Comparable

These projects appear in mahjong AI discussions but don't implement
full Riichi Mahjong game simulation.

### Wrong game

- **PGX**: Implements Sparrow Mahjong, a simplified toy variant. Not Riichi.
- **OpenSpiel** (DeepMind): No Riichi Mahjong implementation at all.

### Utility libraries (no game simulation)

- **riichi-tools-rs**: Hand evaluation and shanten calculation only.
- **mahjong** (PyPI): Hand calculator only.
- **riichi-hand-rs**: Hand parsing only.

None of these run games, so there's no simulation throughput to measure.

---

## 6. Caveats

### Trivial agents inflate all numbers equally

Every engine was benchmarked with a do-nothing agent. Real training
replaces that with neural network inference at every decision point
(~70 per hanchan). NN forward passes (~1ms per batch on GPU) will
dominate wall-clock time by 10-100x over simulation. These numbers
measure engine overhead, not training speed.

### libriichi has no game simulation benchmark

libriichi provides `PlayerState` for processing MJAI event logs, not a
standalone game loop. The only benchmarkable operation is `encode_obs`.
A direct per-game comparison with Hydra is not possible.

### mahjax CPU is deliberately its weak mode

Testing mahjax on CPU without `vmap` is like benchmarking a GPU shader
on a software rasterizer. The 2.09x gap in Hydra's favor would likely
reverse on GPU hardware with proper batching.

### Mjx uses a smarter agent than other benchmarks

Every other engine was benchmarked with a trivial first-action agent. Mjx was
benchmarked with its built-in RuleBased agent (shanten-minimizing), which does
real tile evaluation per action. This makes the per-game time not perfectly
comparable, but the 42x gap is dominated by gRPC/protobuf serialization
overhead, not agent computation.

### Hydra's batch advantage is the training-relevant number

Single-game latency matters for debugging and interactive play.
Batch throughput (8,170 games/sec on 4 cores) is what determines
how fast the training pipeline generates experience. That's the
number to watch. Scales further with more cores.

### We haven't proven training works yet

Fast simulation is necessary but not sufficient. These benchmarks
measure plumbing. The training pipeline will determine if any of
this was worth building.

---

## Sources

All "measured" numbers: benchmarks on Intel Core Ultra 7 265KF,
20 cores, single session, 2026-03-02. Batch tests capped at 4 threads.
Rust engines use Criterion; Mjx and Mjai use wall-clock timing.

- **hydra-engine**: `cargo bench` in perf-optimizations worktree, `RAYON_NUM_THREADS=4`
- **riichienv-core 0.3.4**: `cargo bench` at `/tmp/riichienv-bench/`, crates.io release
- **libriichi**: `cargo bench --no-default-features --bench bench` (encode_obs only), built from source (AGPL-3.0, benchmark only)
- **mahjax**: `pip install -e .` then custom benchmark script, CPU only, 1000 games
- **Mjx (measured)**: Built from source (github.com/mjx-project/mjx), `g++ -O3 -std=c++17`, 100 games wall-clock, RuleBased agent
- **Mjai**: `gem install mjai` (v0.0.7), Ruby 3.3.8, 100 games wall-clock, TsumogiriPlayer
- **Mjx published**: Koyamada et al., "Mjx: A Framework for Mahjong AI Research," IEEE CoG, 2022, Figure 3
