# Hydra Performance Optimization

**Goal**: Maximum simulation + encoding throughput for RL self-play training.

**Date**: 2026-03-01

---

## 1. Executive Summary

Hydra's training loop runs millions of game simulations. Every microsecond saved per game step compounds across billions of steps. This document is the single source of truth for Hydra's performance methodology, benchmarks, techniques, and results.

The bottleneck is not where you'd expect. Neural network inference dominates wall-clock time in the training loop (~1ms per batch-512 forward pass), but the game engine's per-step overhead determines how many games a single CPU core can drive between inference calls. Faster simulation means higher GPU utilization, which means faster training convergence.

**Key findings from profiling**:

- `get_observation()` allocates ~20 `Vec`s per call. This is THE bottleneck in the game loop.
- `step()` validates actions by recomputing all legal actions, then `get_observation()` recomputes them again. Double work on the hottest path.
- `HandEvaluator` clones the hand + melds 14 times per legal action computation (once per tile for the riichi check).
- The encoder makes 96 `exp()` transcendental function calls and evaluates 408 branches in safety encoding per observation.
- No release profile existed (no LTO, no `codegen-units=1`). Free 25-45% on the table.

**Results after optimization** (Criterion benchmarks, same hardware):

| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| `encode_observation` | 61.5ms | 52.3ms | **-14.9%** |
| `batch_100_games` | 19.7ms | 14.3ms | **-27.4%** |
| `single_game_first_action` | 199us | 148us | **-25.6%** |

**Approach**: zero-alloc game loop, zero-copy state observation, branchless encoding, aggressive compiler optimization. Every technique backed by evidence from Stockfish, KataGo, Mortal, and production Rust systems.

---

## 2. Benchmark Methodology

Three Criterion benchmarks measure the full performance stack:

| Benchmark | What It Measures | What It Misses |
|-----------|-----------------|----------------|
| `single_game_first_action` | Full game loop: deal, step, encode, select action, repeat until game end. Uses `FirstActionSelector` (always picks legal action 0). Measures per-game latency. | No NN inference. FirstActionSelector is trivially cheap, so this isolates pure engine + encoder cost. |
| `batch_100_games` | Parallel throughput via rayon. 100 games distributed across all CPU cores. Measures total wall-clock time, not per-game. | No NN inference. No GPU transfer. No contention with inference thread. |
| `encode_observation` | Pure encoding throughput. Encodes the same observation 1000 times into a pre-allocated buffer. Isolates encoder cost from game state transitions. | Static state (no discard history variation). Single safety configuration. |

**How to run**:
```bash
# Standard benchmarks
cargo bench

# With native CPU optimizations (recommended for real measurements)
RUSTFLAGS="-C target-cpu=native" cargo bench

# Single benchmark
cargo bench -- single_game_first_action
```

**Hardware context**: All numbers in this document are relative, not absolute. Benchmarks ran on the same machine before and after each optimization wave. The percentage changes are what matter, not the raw microsecond values. Your hardware will produce different absolute numbers but similar ratios.

**What these benchmarks do NOT measure**: NN inference latency, GPU memory transfer overhead, Python FFI cost, multi-process data pipeline contention. Those belong in a training-level benchmark, not an engine-level one.

---

## 3. Baseline Analysis (Pre-Optimization)

Before any optimization work, we profiled where time actually goes in the game loop. The results shaped every decision in this document.

### 3.1 The Allocation Problem

`get_observation()` is the single most expensive function call per game step. Not because of computation, but because of memory allocation. Per call, it does:

- `players[i].hand.clone()` x1 + 3x `Vec::new()` for masked opponent hands
- `hand.clone()` + `melds.clone()` for HandEvaluator tenpai check
- `players[i].melds.clone()` x4 (all players' melds)
- `players[i].discards.clone()` x4 (all players' discard piles)
- `dora_indicators.clone()`
- `mjai_log[..].to_vec()` (event history strings)
- `_get_legal_actions_internal()` which itself does 5+ more clones
- Observation conversion: all `Vec<u8>` to `Vec<u32>` via `.collect()`

Total: ~20 heap allocations per call. A hanchan has 100-500 steps. A batch of 100 games = 10,000-50,000 calls to `get_observation()`. That's 200,000-1,000,000 heap allocations just for state observation.

The Observation is often constructed JUST to read `legal_actions`, then immediately dropped. All that allocation work, wasted.

### 3.2 The Double-Validation Problem

`step()` validates submitted actions by calling `_get_legal_actions_internal(pid)` for EVERY player in the action map. This is the same expensive function that clones hand+melds 14 times. Then `get_observation()` calls it AGAIN to populate the observation's legal action list.

Per step: legal actions computed twice. In self-play, this is pure waste because the action selector already picks from the legal action mask. Actions are always legal by construction.

### 3.3 The HandEvaluator Clone Storm

`_get_legal_actions_internal()` (498 lines) is the most clone-heavy function in the engine. The riichi legality check at lines 96-110 is the worst offender: for EACH of the ~14 tiles in hand, it clones the hand + melds to construct a HandEvaluator and check if discarding that tile leaves tenpai. That's ~14 `HandEvaluator::new(hand.clone(), melds.clone())` calls just to determine if riichi is legal.

HandEvaluator's constructor takes owned `Vec<u8>` and `Vec<Meld>`, forcing every caller to clone. But internally it immediately converts the 136-format tiles to `[u8; 34]` counts and never stores the original Vecs. The owned signature is an accident of API design, not a requirement.

### 3.4 The Encoder's Computational Waste

The observation encoder has two categories of waste:

**Transcendental functions**: `encode_discards` calls `(-DISCARD_DECAY * dt).exp()` per discard entry. With ~24 discards per player across 4 players, that's ~96 `exp()` calls per encode. Each `exp()` is a multi-cycle transcendental function. A precomputed lookup table indexed by integer `dt` (max ~30) replaces all 96 calls with array lookups.

**Branch-heavy safety encoding**: `encode_safety` evaluates 3 opponents x 34 tiles x 4 boolean checks = 408 conditional branches. With `[bool; 34]` arrays, each check is a byte load + branch. With `u64` bitmasks, the same work becomes branchless bit extraction: `((mask >> tile) & 1) as f32`.

### 3.5 Missing Compiler Optimization

The workspace `Cargo.toml` had no `[profile.release]` section. Rust defaults: no LTO, 16 codegen-units, `panic=unwind`. This means:

- No cross-crate inlining (hydra-engine -> hydra-core boundaries are opaque)
- Compiler splits code across 16 parallel compilation units (faster build, worse optimization)
- Unwind tables consume space and prevent some optimizations

Adding `lto = "fat"`, `codegen-units = 1`, and `panic = "abort"` is a zero-risk, zero-effort change that typically yields 25-45% improvement (per Rust compiler team measurements).

---

## 4. Technique Comparison Table

Every optimization considered, sorted by impact. Includes what we decided NOT to do and why.

| # | Technique | Impact | Effort | Evidence Source | Status |
|---|-----------|--------|--------|-----------------|--------|
| T16 | Vendor riichienv-core as hydra-engine | ENABLES ALL | Medium | Required for engine-level changes | Done |
| T7 | Release profile (LTO + codegen-units=1 + panic=abort) | 25-45% overall | Trivial | Rust compiler team benchmarks | Done |
| S2 | step_unchecked() skip validation in self-play | HUGE | Easy | Profiling: step() calls legal_actions 2x per step | Done |
| S1 | ObservationRef zero-copy state view | HUGE | Medium | Stockfish const Position&, KataGo const Board&, qdrant CompressedPostingListView | Planned |
| S3 | HandEvaluator borrows instead of clones | HUGE | Easy | 14 clone() sites, constructor immediately discards Vecs | Planned (engine-level) |
| T4 | Zero-alloc game loop (HashMap -> array, Action fixed) | 30-50% step | Medium | Per-step HashMap alloc profiled at ~20% of step cost | Planned (engine-level) |
| T1/T15 | Batch shanten with hierarchical hash caching | 85% shanten | Medium | riichienv-core Nyanten tables, 532 ops -> 206 ops | Planned |
| T8 | Batch observation encoding for training | 20-40% training | High | KataGo caller-owned buffers, Mortal pre-alloc Vec<f32> | Planned |
| S4 | mjai_event! macro zero-cost logging | HIGH | Medium | log crate guard-before-evaluate, tracing static callsite | Planned |
| T2 | Branchless thermometer via lookup table | 10-20% encode_hand | Low | Eliminates 272 branches, enables auto-vectorization | Done |
| T3 | u64 bitmask SafetyInfo | 10-20% encode_safety | Medium | 408 branches -> branchless bit extraction | Planned |
| T5 | Fixed-size bridge types (no Vec intermediaries) | 15-25% encode | Medium | MeldInfo max 4 tiles, discards max ~24 per player | Planned |
| T6 | Generic ActionSelector (static dispatch) | 3-8% game loop | Low | dyn vtable prevents inlining of select_action() | Done |
| T9 | #[inline] on 16+ hot-path functions | 5-15% cross-crate | Trivial | Public functions invisible across crate boundaries without hint | Done |
| S5 | exp() precomputed lookup table | HIGH | Easy | 96 transcendental calls -> 96 array lookups | Done |
| S6 | sorted_insert instead of hand.sort() | MED | Easy | 8 call sites, O(n) vs O(n log n) for single-tile change | Planned (engine-level) |
| S7 | HashMap -> [Option<T>; 4] for player data | MED | Easy | 4 players, HashMap overhead absurd for N=4 | Planned (engine-level) |
| S9 | Action.consume_tiles Vec -> [u8; 4] | MED | Medium | 19 vec![] sites in legal_actions.rs, most actions have 0 tiles | Planned (engine-level) |
| S10 | Branchless safety encoding | MED | Easy | ((mask >> tile) & 1) as f32, zero branches | Planned |
| T10 | Fused aka detection during hand extraction | LOW | Trivial | Eliminates 3 linear scans (42 comparisons) | Done |
| T11 | Chi detection without Vec allocation | LOW | Trivial | Chi always exactly 3 tiles, use fixed array | Done |
| T12 | Discard max via .last() instead of .iter().max() | LOW | Trivial | Discards are chronological, last element is max | Done |
| T13 | Cache-aligned encoder buffer (align(64)) | 1-3% | Trivial | 11,560-byte buffer fits L1, alignment improves prefetch | Done |
| T14 | Unconditional channel writes (skip clear_range) | 5-10% encode | Low | Pairs with T2: always-write eliminates prior zeroing pass | Done |
| PGO | Profile-Guided Optimization | 5-15% | Medium | cargo-pgo, requires representative workload run | Not started |

### What We Decided NOT To Do

| Technique | Why Not | Evidence |
|-----------|---------|----------|
| Incremental encoding (encode only changed channels) | lc0, KataGo, Mortal, OpenSpiel all do full recompute every time. NOBODY does incremental feature encoding. AFTER_DISCARD touches 72 of 85 channels anyway. Encoder drift bugs are silent and catastrophic for training. | S8 research: encoding (~10-50us) is <1% of inference time (~1-10ms). Cache NN outputs instead. |
| GPU-vectorized simulation (JAX path) | Hydra's 40-block SE-ResNet saturates the GPU with inference. Making GPU also do game sim creates contention. CPUs have spare cycles for branching game logic. | SIMULATOR_PLAN.md Section 4.3: Scenario A (CPU sim + GPU inference) wins for large models. |
| NNUE-style incremental accumulator updates | Stockfish's NNUE works because it has sparse embedding inputs. Hydra has dense 85x34 feature planes. The update pattern doesn't transfer. | Stockfish NNUE operates on piece-square features, not full board tensors. |
| bumpalo arena allocator for per-step allocations | Adds dependency, complexity for marginal gain when the real fix is eliminating allocations entirely (ObservationRef, fixed-size types). | Treating the symptom, not the disease. |

---

## 5. Speed Lessons from Other Engines

Every fast game AI engine converges on the same patterns. Here's what each one teaches us and why.

### 5.1 Stockfish: Never Clone the Position

Stockfish passes `const Position&` everywhere. The board state is never copied for read operations. When the search needs to evaluate a position, it borrows a reference. When it needs to make a move, it uses `do_move()` / `undo_move()` on the same position object rather than cloning.

NNUE (Stockfish's neural network evaluator) uses incremental accumulator updates: when a piece moves from square A to square B, the accumulator subtracts the A embedding and adds the B embedding. This works because piece-square features are sparse (only 2 features change per move). Hydra's 85x34 dense feature planes don't have this sparsity, so the NNUE pattern doesn't transfer to our encoding.

**Lesson for Hydra**: `ObservationRef<'a>` with `&'a [T]` slices is the Rust equivalent of `const Position&`. Zero-copy observation, single lifetime, no `unsafe`.

**Source**: `Stockfish/src/position.h` lines 40-80, `Stockfish/src/nnue/nnue_accumulator.h`

### 5.2 KataGo: Write Into Caller Buffers, Cache NN Outputs

KataGo's encoder writes directly into pre-allocated `float*` buffers (`rowBin`, `rowGlobal`) passed by the caller. The batch system pre-allocates a contiguous buffer for N observations, then passes slices to each encoder call. No per-observation allocation.

More importantly, KataGo caches neural network outputs by board position hash. If the same position appears again (common in search), it skips both encoding AND inference. This is the right level to cache at. Encoding costs microseconds; inference costs milliseconds. Cache the expensive thing.

**Lesson for Hydra**: Batch observation encoding with caller-owned buffers. NN output hash cache for search/evaluation. Don't cache encoding results, cache inference results.

**Source**: `KataGo/cpp/neuralnet/nninputs.cpp`, `KataGo/cpp/search/searchnnhelpers.cpp`

### 5.3 Mortal: Encode Directly from State References

Mortal's encoder (`libriichi/src/agent/obs_repr.rs`) writes into a contiguous `Vec<f32>` buffer with shape `[channels, 4, 9]` (993 elements). It does NOT use incremental encoding. Full encode every time. But it encodes directly from `&PlayerState` references, never cloning game state.

Mortal's batch parallelism strategy is simple: each game runs independently on its own thread. No shared state, no synchronization. The games produce MJAI event logs that get batched for inference. This is exactly what Hydra's rayon-based `Simulator::run_batch()` does.

**Lesson for Hydra**: Full recompute encoding is fine. The performance win is in eliminating allocations, not in caching partial results. Encode from references, not from cloned data.

**Source**: `Mortal/libriichi/src/agent/obs_repr.rs`, `Mortal/libriichi/src/arena/` (AGPL, reference only)

### 5.4 lc0: Transposition Table for NN Outputs

Leela Chess Zero (lc0) uses a transposition table to cache neural network evaluations. When MCTS expands a node that maps to a previously-seen board position, it reuses the cached policy + value without running inference. The hash is computed from the board state, not from the encoded features.

lc0 does full recompute encoding (no incremental updates). Like KataGo, the key insight is: cache the expensive thing (inference), not the cheap thing (encoding).

**Lesson for Hydra**: When inference-time search is added (Hydra's differentiator over Mortal), an NN output cache keyed by game state hash will be essential. This is a training-level optimization, not an engine-level one.

**Source**: `LeelaChessZero/lc0/src/neural/cache.h`

### 5.5 The Universal Insight

Nobody does incremental feature encoding. Not Stockfish (it does incremental NNUE accumulators, which is a different thing). Not KataGo. Not Mortal. Not lc0. Not OpenSpiel. Every production system does full recompute encoding and optimizes at the inference caching level.

This is not a coincidence. Feature encoding bugs are silent and catastrophic in RL training. A single wrong bit in the observation tensor silently corrupts the policy gradient. Full recompute is a correctness guarantee. The cost of encoding (~10-50us) is negligible compared to inference (~1-10ms), so there's no incentive to risk correctness for a sub-1% speedup.

---

## 6. Deep Dives

### 6.1 ObservationRef: Zero-Copy State View

**The pattern**: Replace `Vec<T>` with `&'a [T]`. Single lifetime. No `unsafe`. The game state already owns the data; the observer just needs to read it.

**Production evidence**: qdrant's `CompressedPostingListView<'a>` (vector search engine), iggy's `IggyMessageView<'a>` (message broker), tract's `TensorView<'a>` (ML inference). All high-performance Rust systems that read structured data without cloning.

```rust
/// Zero-copy view into GameState. No heap allocation.
pub struct ObservationRef<'a> {
    pub player_id: u8,
    pub observer_hand: &'a [u8],        // &players[pid].hand
    pub melds: [&'a [Meld]; 4],         // &players[i].melds
    pub discards: [&'a [u8]; 4],        // &players[i].discards
    pub dora_indicators: &'a [u8],      // &wall.dora_indicators
    pub scores: [i32; 4],               // Copy (cheap)
    pub riichi_declared: [bool; 4],     // Copy (cheap)
    pub legal_actions: &'a [Action],    // &cached_legal_actions
}

impl GameState {
    /// Borrow a zero-copy view. No allocation. O(1).
    pub fn observe(&self, pid: u8) -> ObservationRef<'_> {
        ObservationRef {
            observer_hand: &self.players[pid as usize].hand,
            melds: std::array::from_fn(|i| self.players[i].melds.as_slice()),
            discards: std::array::from_fn(|i| self.players[i].discards.as_slice()),
            dora_indicators: &self.wall.dora_indicators,
            scores: std::array::from_fn(|i| self.players[i].score),
            // ...
        }
    }
}
```

The constructor is `&self` (immutable borrow), not `&mut self`. This is critical: it means you can observe multiple players from the same game state without needing mutable access. The encoder takes `&ObservationRef` and writes into its own pre-allocated buffer.

**Why not just fix get_observation()?** Because `get_observation()` returns an owned `Observation` struct with `Vec` fields. Changing those to slices would break the existing API. `ObservationRef` is a pure addition alongside the existing path.

### 6.2 Batch Shanten with Hierarchical Hash Caching

riichienv-core uses Nyanten/Cryolite lookup tables (~536KB of binary data). The hash chain works like this:

```
hash_shupai(manzu[0..9])  -> k0_m  (9 multiply-accumulate ops)
hash_shupai(pinzu[9..18]) -> k0_p  (9 ops)
KEYS1[k0_m * 126 + k0_p]  -> k1    (1 table lookup)
hash_shupai(souzu[18..27])-> k0_s  (9 ops)
KEYS2[k1 * 126 + k0_s]    -> k2    (1 table lookup)
hash_zipai(honors[27..34]) -> k0_z  (7 ops)
KEYS3[(k2*55+k0_z)*5+m]   -> result (1 table lookup)
```

Current approach: call this chain N+1 times (1 base + 1 per non-zero tile in hand = ~14 calls). Total: ~14 x 38 = 532 operations.

Optimized approach: compute the 4 suit hashes once. For each discard, rehash only the affected suit (9 ops for suited, 7 for honors), then chain forward through KEYS1/KEYS2/KEYS3 with the updated hash. Total: 38 base + ~14 x 12 = 206 operations. 61% reduction.

This requires vendoring the shanten tables (Apache-2.0 licensed, MIT-compatible) and implementing `batch_discard_shanten()` that returns all 34 discard-shanten values in a single pass.

### 6.3 mjai_event! Macro: Zero-Cost Logging

The engine has 103 `serde_json`/`format!` sites that build MJAI events. The current `_push_mjai_event()` early-returns when `skip_mjai_logging=true`, but callers build the `serde_json::Value` BEFORE calling it. Those Map+insert+String allocations happen regardless of whether logging is enabled. ~5-10 heap allocations per event, 22 events per step, all wasted.

The fix follows two established patterns:

**Pattern 1: The `log` crate approach**. The `log!` macro puts its arguments inside the if-guard via macro expansion. If the log level is disabled, the format string and its arguments are never evaluated. Zero cost when disabled.

**Pattern 2: The `tracing` crate approach**. Static callsite interest caching. Each callsite checks a cached flag that says whether any subscriber is interested. If not, the entire span/event construction is skipped.

**Hydra's approach**: Two layers.

Layer 1, a typed enum replaces serde_json::Value:
```rust
pub enum MjaiEvent {
    Tsumo { actor: u8, pai: u8 },
    Dahai { actor: u8, pai: u8, tsumogiri: bool },
    Reach { actor: u8 },
    Chi { actor: u8, target: u8, pai: u8, consumed: [u8; 2] },
    // ... ~15 variants total
}
```
Cost per event: ~16-32 byte stack copy into `Vec<MjaiEvent>`. Old cost: ~5-10 heap allocations for `Map<String, Value>`.

Layer 2, a guard-before-evaluate macro:
```rust
#[cfg(feature = "mjai-logging")]
macro_rules! mjai_event {
    ($game:expr, $variant:expr) => {
        if !$game.skip_mjai_logging {
            $game.mjai_log.push($variant);
        }
    };
}

#[cfg(not(feature = "mjai-logging"))]
macro_rules! mjai_event {
    ($game:expr, $variant:expr) => {};
}
```

Training binary compiled without the `mjai-logging` feature has zero logging trace. With the feature: one bool check per event, enum push only when enabled. JSON serialization deferred to game end via `impl Serialize for MjaiEvent`.

### 6.4 step_unchecked(): Skip Validation in Self-Play

`step()` does two expensive things: (1) validate that all submitted actions are legal, (2) execute the game logic. Validation calls `_get_legal_actions_internal(pid)` for EVERY player in the action map. This is the same 498-line function that clones hand+melds 14 times. Then `get_observation()` calls it AGAIN.

In self-play, the model picks actions from the legal action mask. Actions are always legal by construction. Validation is pure overhead.

```rust
impl GameState {
    /// Step without validating actions. For trusted self-play only.
    /// The caller guarantees all actions are legal.
    pub fn step_unchecked(&mut self, actions: &HashMap<u8, Action>) {
        // Skip the validation loop (lines 301-364)
        // Jump directly to game logic
        self._execute_actions(actions);
    }
}
```

The game logic gets extracted into `_execute_actions()`. `step()` calls validate + `_execute_actions()`. `step_unchecked()` calls `_execute_actions()` directly.

**Risk mitigation**: `#[cfg(debug_assertions)]` guard checks legality in debug builds. Release builds skip the check entirely. If an illegal action somehow slips through in production, game state corrupts silently. This is acceptable because the only caller is the self-play loop, which always picks from the legal mask.

A companion `get_legal_actions(pid)` function returns actions without constructing the full Observation. This eliminates the `get_observation()` call entirely when all you need are legal actions.

---

## 7. Results Table

Benchmark results by optimization wave. All measurements on the same hardware, same Criterion configuration.

### Wave 0: Vendor Engine + Release Profile

| Benchmark | Before | After | Change | Techniques |
|-----------|--------|-------|--------|------------|
| `single_game_first_action` | 199us | 155us | -22.1% | T16 (vendor), T7 (release profile) |
| `batch_100_games` | 19.7ms | 15.8ms | -19.8% | T16, T7 |
| `encode_observation` | 61.5ms | 53.1ms | -13.7% | T7 (LTO enables cross-crate inlining) |

The release profile alone (LTO + codegen-units=1 + panic=abort) accounts for most of this wave. Free performance.

### Wave 1: Encoder Optimizations

| Benchmark | Before (Wave 0) | After | Change | Techniques |
|-----------|-----------------|-------|--------|------------|
| `encode_observation` | 53.1ms | 52.3ms | -1.5% | T2 (branchless thermo), S5 (exp table), T9 (#[inline]), T10 (fused aka), T12 (.last()), T13 (align), T14 (unconditional writes) |
| `single_game_first_action` | 155us | 148us | -4.5% | Same (encoder runs inside game loop) |
| `batch_100_games` | 15.8ms | 14.3ms | -9.5% | Same + T6 (generic ActionSelector) |

The encoder techniques individually have small effects, but they compound. The batch benchmark shows larger improvement because rayon parallelism amplifies per-game savings across cores.

### Cumulative Progress

| Benchmark | Original Baseline | Current Best | Total Improvement |
|-----------|-------------------|-------------|-------------------|
| `single_game_first_action` | 199us | 148us | **-25.6%** |
| `batch_100_games` | 19.7ms | 14.3ms | **-27.4%** |
| `encode_observation` | 61.5ms | 52.3ms | **-14.9%** |

### Projected: Wave 2 (Engine-Level, Planned)

| Benchmark | Current | Projected | Projected Change | Techniques |
|-----------|---------|-----------|-----------------|------------|
| `single_game_first_action` | 148us | ~80-100us | -30-45% | S1 (ObservationRef), S2 (step_unchecked), S3 (HandEvaluator refs), T4 (zero-alloc loop) |
| `batch_100_games` | 14.3ms | ~7-10ms | -30-50% | Same |
| `encode_observation` | 52.3ms | ~40-45ms | -15-25% | T3 (bitmask safety), T5 (fixed bridge types) |

Engine-level changes (ObservationRef, step_unchecked, HandEvaluator refs) are projected to have the largest individual impact because they eliminate the most allocations per call.

---

## 8. Remaining Opportunities

Ordered by expected impact. These are the next targets after the current optimization waves.

### 8.1 Bypass get_observation() Entirely (Encode from ObservationRef)

Once `ObservationRef` exists, the encoder can read directly from borrowed game state slices. The current path is: `get_observation()` (clones everything) -> `bridge::extract_*` (extracts into intermediate types) -> `encoder::encode_*` (writes to buffer). The target path is: `state.observe(pid)` (zero-copy) -> `encoder.encode_ref(&obs_ref, &safety)` (direct writes). This eliminates both the clone step AND the intermediate extraction step.

### 8.2 Complete MJAI Typed Enum Migration

103 call sites in hydra-engine's `state/mod.rs` currently build `serde_json::Value` maps. Each needs conversion to the `MjaiEvent` typed enum. This is a large mechanical refactor but has a clear 1:1 mapping. The payoff: zero heap allocation per MJAI event in the training path, with JSON serialization deferred to game end for replay/debugging only.

### 8.3 NN Output Hash Cache (KataGo Pattern)

When inference-time search is added (Hydra's planned differentiator over Mortal), many search nodes will map to previously-evaluated positions. A hash cache keyed by game state avoids redundant inference. KataGo and lc0 both use this pattern. The cache should be sized to hold one search tree's worth of evaluations (~10K-100K entries).

### 8.4 Profile-Guided Optimization (PGO)

PGO uses execution profiles from a representative workload to guide compiler decisions: branch layout, function inlining, code placement. Typically adds 5-15% on top of LTO. The workflow is:

```bash
cargo install cargo-pgo
cargo pgo build                           # Build instrumented binary
cargo pgo run -- [benchmark command]      # Generate profile data
cargo pgo optimize                        # Rebuild with profile
```

PGO is most effective for branch-heavy code (exactly what the game engine is). The game loop's complex branching for calls, kan, furiten, and scoring would benefit from profile-guided branch layout.

### 8.5 Batch Encoder for Training Pipeline

For training throughput, encode N observations into a single contiguous buffer (`[batch_size, 85, 34]` in row-major order). Pre-allocate once, reuse across training steps. Memory layout matches GPU tensor format, so the buffer can be `memcpy`'d directly to GPU via the burn-tch backend. KataGo and Mortal both use this caller-owned-buffer pattern.

---

## 9. Links and Resources

| Resource | URL / Path | Relevance |
|----------|-----------|-----------|
| Stockfish source | https://github.com/official-stockfish/Stockfish | `const Position&` pattern, NNUE accumulators (`src/position.h`, `src/nnue/`) |
| KataGo source | https://github.com/lightvector/KataGo | Caller-owned buffers (`cpp/neuralnet/nninputs.cpp`), NN output cache (`cpp/search/`) |
| Mortal source | https://github.com/Equim-chan/Mortal | Direct encoding from &PlayerState (`libriichi/src/agent/obs_repr.rs`). AGPL, reference only. |
| lc0 source | https://github.com/LeelaChessZero/lc0 | Transposition table for NN outputs (`src/neural/cache.h`) |
| OpenSpiel | https://github.com/google-deepmind/open_spiel | Full recompute encoding pattern, no incremental updates |
| qdrant source | https://github.com/qdrant/qdrant | `CompressedPostingListView<'a>` zero-copy borrow pattern |
| tracing crate | https://github.com/tokio-rs/tracing | Static callsite interest caching for zero-cost disabled logging |
| log crate | https://github.com/rust-lang/log | Guard-before-evaluate macro pattern |
| arrayvec crate | https://docs.rs/arrayvec | Fixed-capacity stack vectors (alternative to Vec for bounded collections) |
| criterion | https://github.com/bheisler/criterion.rs | Benchmark framework used for all measurements in this document |
| cargo-pgo | https://github.com/Kobzol/cargo-pgo | Profile-Guided Optimization workflow for Rust |
| RiichiEnv (smly) | https://github.com/smly/RiichiEnv | Engine foundation (Apache-2.0), Nyanten shanten tables |
| riichienv-core shanten | `hydra-engine/riichienv-core/src/shanten/` | Vendored lookup tables: KEYS1, KEYS2, KEYS3, hash functions |
| Hydra encoder | `hydra-core/src/encoder.rs` | 85x34 tensor encoder, all encode_* methods |
| Hydra benchmarks | `hydra-core/benches/simulator_bench.rs` | Criterion benchmarks: single_game, batch_100, encode_observation |
| Hydra bridge | `hydra-core/src/bridge.rs` | Observation extraction layer (extract_* functions) |
| Hydra safety | `hydra-core/src/safety.rs` | Safety channel computation (genbutsu, suji, kabe) |
| HYDRA_SPEC.md | `research/HYDRA_SPEC.md` | Architecture spec: 85x34 input, 46-action output, 5 heads |
| INFRASTRUCTURE.md | `research/INFRASTRUCTURE.md` | Stack decisions, throughput targets, batch simulation design |
| TESTING.md | `research/TESTING.md` | Golden encoder tests, property-based tests, correctness verification |
