# Hydra Performance Optimization

**Goal**: Max sim + encode throughput for RL self-play training.

**Date**: 2026-03-01

---

## 1. Executive Summary

Hydra training loop runs millions of game sims. Tiny per-step wins compound across billions of steps. This doc = main ref for Hydra perf method, benches, techniques, results; not repo-wide authority router.

Bottleneck not obvious. NN inference dominates wall time in loop (~1ms per batch-512 forward pass), but engine per-step overhead decides how many games one CPU core can drive between inference calls. Faster sim -> higher GPU util -> faster training convergence.

**Key findings from profiling**:

- `get_observation()` allocates ~20 `Vec`s per call. Main loop bottleneck.
- `step()` validates actions by recomputing all legal actions, then `get_observation()` recomputes them again. Double work on hottest path.
- `HandEvaluator` clones hand + melds 14 times per legal-action computation (once per tile for riichi check).
- Encoder does 96 `exp()` transcendental calls and 408 branches in safety encoding per observation.
- No release profile existed (no LTO, no `codegen-units=1`). Free 25-45% left unused.

**Results after optimization** (Criterion benchmarks, same hardware):

| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| `encode_observation` | 61.5ms | 52.3ms | **-14.9%** |
| `batch_100_games` | 19.7ms | 14.3ms | **-27.4%** |
| `single_game_first_action` | 199us | 148us | **-25.6%** |

**Approach**: zero-alloc game loop, zero-copy observation, branchless encoding, aggressive compiler opts. Every technique backed by evidence from Stockfish, KataGo, Mortal, production Rust systems.

---

## 2. Benchmark Methodology

Three Criterion benches cover full perf stack:

| Benchmark | What It Measures | What It Misses |
|-----------|-----------------|----------------|
| `single_game_first_action` | Full loop: deal, step, encode, select action, repeat until game end. Uses `FirstActionSelector` picks legal action 0). Measures per-game latency. | No NN inference. `FirstActionSelector` near-free, so this isolates engine + encoder cost. |
| `batch_100_games` | Parallel throughput via rayon. 100 games spread across CPU cores. Measures total wall time, not per-game. | No NN inference. No GPU transfer. No contention with inference thread. |
| `encode_observation` | Pure encode throughput. Encodes same observation 1000 times into pre-allocated buffer. Isolates encoder cost from state transitions. | Static state (no discard-history variation). Single safety config. |

**How to run**:
```bash
# Standard benchmarks
cargo bench

# With native CPU optimizations (recommended for real measurements)
RUSTFLAGS="-C target-cpu=native" cargo bench

# Single benchmark
cargo bench -- single_game_first_action
```

**Hardware context**: Numbers here are relative, not absolute. Benches ran on same machine before/after each optimization wave. Percent deltas matter, not raw microseconds. Your hardware will differ in absolute values but should show similar ratios.

**What these benchmarks do NOT measure**: NN inference latency, GPU transfer overhead, Python FFI cost, multi-process pipeline contention. Those belong in training-level benches, not engine-level benches.

---

## 3. Baseline Analysis (Pre-Optimization)

Before optimization, we profiled real hot spots in loop. Results drove every choice below.

### 3.1 The Allocation Problem

`get_observation()` = most expensive call per game step. Cost mostly memory allocation, not compute. Per call it does:

- `players[i].hand.clone()` x1 + 3x `Vec::new()` for masked opponent hands
- `hand.clone()` + `melds.clone()` for HandEvaluator tenpai check
- `players[i].melds.clone()` x4 (all players' melds)
- `players[i].discards.clone()` x4 (all players' discard piles)
- `dora_indicators.clone()`
- `mjai_log[..].to_vec()` (event-history strings)
- `_get_legal_actions_internal()` which itself does 5+ more clones
- Observation conversion: all `Vec<u8>` to `Vec<u32>` via `.collect()`

Total: ~20 heap allocs per call. One hanchan has 100-500 steps. Batch of 100 games = 10,000-50,000 `get_observation()` calls. That means 200,000-1,000,000 heap allocs only for observation.

Often Observation gets built only to read `legal_actions`, then dropped. All that alloc work wasted.

### 3.2 The Double-Validation Problem

`step()` validates submitted actions by calling `_get_legal_actions_internal(pid)` for every player in action map. Same expensive fn clones hand+melds 14 times. Then `get_observation()` calls it again to fill observation legal actions.

Per step: legal actions computed twice. In self-play this is pure waste because selector already chooses from legal mask. Actions are legal by construction.

### 3.3 The HandEvaluator Clone Storm

`_get_legal_actions_internal()` (498 lines) = most clone-heavy fn in engine. Riichi legality check at lines 96-110 = worst part: for each of ~14 tiles in hand, it clones hand + melds to build `HandEvaluator` and test whether discarding that tile leaves tenpai. So ~14 `HandEvaluator::new(hand.clone(), melds.clone())` calls only to decide riichi legality.

HandEvaluator ctor takes owned `Vec<u8>` and `Vec<Meld>`, forcing all callers to clone. Internally it immediately converts 136-format tiles to `[u8; 34]` counts and never stores original Vecs. Owned signature is API accident, not real need.

### 3.4 The Encoder's Computational Waste

Encoder waste has two classes:

**Transcendental functions**: `encode_discards` calls `(-DISCARD_DECAY * dt).exp()` per discard entry. With ~24 discards per player across 4 players, that is ~96 `exp()` calls per encode. Each `exp()` costs many cycles. precomputed lookup table indexed by integer `dt` (max ~30) replaces all 96 calls with array lookups.

**Branch-heavy safety encoding**: `encode_safety` does 3 opponents x 34 tiles x 4 bool checks = 408 conditional branches. With `[bool; 34]`, each check = byte load + branch. With `u64` bitmasks, same work becomes branchless bit extraction: `((mask >> tile) & 1) as f32`.

### 3.5 Missing Compiler Optimization

Workspace `Cargo.toml` had no `[profile.release]`. Rust defaults: no LTO, 16 codegen units, `panic=unwind`. That means:

- No cross-crate inlining (`hydra-engine` -> `hydra-core` boundaries stay opaque)
- Compiler splits code into 16 parallel codegen units (faster build, worse optimization)
- Unwind tables cost space and block some optimizations

Adding `lto = "fat"`, `codegen-units = 1`, and `panic = "abort"` = zero-risk, near-zero-effort change that commonly yields 25-45% improvement (per Rust compiler team measurements).

---

## 4. Technique Comparison Table

Every optimization considered, sorted by impact. Includes rejected ideas and reasons.

| # | Technique | Impact | Effort | Evidence Source | Status |
|---|-----------|--------|--------|-----------------|--------|
| T16 | Vendor riichienv-core as hydra-engine | ENABLES ALL | Medium | Needed for engine-level changes | Done |
| T7 | Release profile (LTO + codegen-units=1 + panic=abort) | 25-45% overall | Trivial | Rust compiler team benchmarks | Done |
| S2 | step_unchecked() skip validation in self-play | HUGE | Easy | Profiling: step() calls legal_actions 2x per step | Done |
| S1 | ObservationRef zero-copy state view | HUGE | Medium | Stockfish const Position&, KataGo const Board&, qdrant CompressedPostingListView | Planned |
| S3 | HandEvaluator borrows instead of clones | HUGE | Easy | 14 clone() sites, ctor immediately discards Vecs | Planned (engine-level) |
| T4 | Zero-alloc game loop (HashMap -> array, Action fixed) | 30-50% step | Medium | Per-step HashMap alloc profiled at ~20% of step cost | Planned (engine-level) |
| T1/T15 | Batch shanten with hierarchical hash caching | 85% shanten | Medium | riichienv-core Nyanten tables, 532 ops -> 206 ops | Planned |
| T8 | Batch observation encoding for training | 20-40% training | High | KataGo caller-owned buffers, Mortal pre-alloc Vec<f32> | Planned |
| S4 | mjai_event! macro zero-cost logging | HIGH | Medium | log crate guard-before-evaluate, tracing static callsite | Planned |
| T2 | Branchless thermometer via lookup table | 10-20% encode_hand | Low | Eliminates 272 branches, enables auto-vectorization | Done |
| T3 | u64 bitmask SafetyInfo | 10-20% encode_safety | Medium | 408 branches -> branchless bit extraction | Planned |
| T5 | Fixed-size bridge types (no Vec intermediaries) | 15-25% encode | Medium | MeldInfo max 4 tiles, discards max ~24 per player | Planned |
| T6 | Generic ActionSelector (static dispatch) | 3-8% game loop | Low | dyn vtable blocks inlining of `select_action()` | Done |
| T9 | #[inline] on 16+ hot-path functions | 5-15% cross-crate | Trivial | Public fns invisible across crate boundaries without hint | Done |
| S5 | exp() precomputed lookup table | HIGH | Easy | 96 transcendental calls -> 96 array lookups | Done |
| S6 | sorted_insert instead of hand.sort() | MED | Easy | 8 call sites, O(n) vs O(n log n) for single-tile change | Planned (engine-level) |
| S7 | HashMap -> [Option<T>; 4] for player data | MED | Easy | 4 players, HashMap overhead absurd for N=4 | Planned (engine-level) |
| S9 | Action.consume_tiles Vec -> [u8; 4] | MED | Medium | 19 vec![] sites in legal_actions.rs, most actions use 0 tiles | Planned (engine-level) |
| S10 | Branchless safety encoding | MED | Easy | `((mask >> tile) & 1) as f32`, zero branches | Planned |
| T10 | Fused aka detection during hand extraction | LOW | Trivial | Eliminates 3 linear scans (42 comparisons) | Done |
| T11 | Chi detection without Vec allocation | LOW | Trivial | Chi exactly 3 tiles, use fixed array | Done |
| T12 | Discard max via.last() instead of.iter().max() | LOW | Trivial | Discards are chronological, last element is max | Done |
| T13 | Cache-aligned encoder buffer (align(64)) | 1-3% | Trivial | 11,560-byte buffer fits L1, alignment improves prefetch | Done |
| T14 | Unconditional channel writes (skip clear_range) | 5-10% encode | Low | Pairs with T2: always-write removes prior zeroing pass | Done |
| PGO | Profile-Guided Optimization | 5-15% | Medium | cargo-pgo, needs representative workload run | Not started |

### What We Decided NOT To Do

| Technique | Why Not | Evidence |
|-----------|---------|----------|
| Incremental encoding (encode only changed channels) | lc0, KataGo, Mortal, OpenSpiel all do full recompute each time. Nobody does incremental feature encoding. AFTER_DISCARD touches 72 of 85 channels anyway. Encoder drift bugs are silent and catastrophic for training. | S8 research: encoding (~10-50us) is <1% of inference time (~1-10ms). Cache NN outputs instead. |
| GPU-vectorized simulation (JAX path) | Hydra 40-block SE-ResNet already saturates GPU with inference. Putting sim on GPU too creates contention. CPU still has spare cycles for branchy game logic. | SIMULATOR_PLAN.md Section 4.3: Scenario (CPU sim + GPU inference) wins for large models. |
| NNUE-style incremental accumulator updates | Stockfish NNUE works because inputs are sparse embeddings. Hydra uses dense 85x34 feature planes. Update pattern does not transfer. | Stockfish NNUE works on piece-square features, not full board tensors. |
| bumpalo arena allocator for per-step allocations | Extra dep, extra complexity, marginal gain when real fix is removing allocations entirely (`ObservationRef`, fixed-size types). | Treats symptom, not disease. |

---

## 5. Speed Lessons from Other Engines

Every fast game-AI engine converges on same patterns. Each one gives specific guidance.

### 5.1 Stockfish: Never Clone the Position

Stockfish passes `const Position&` everywhere. Board state never gets copied for reads. Search evaluates by borrowing reference. When it must change position, it uses `do_move()` / `undo_move()` on same object rather than cloning.

NNUE (Stockfish neural evaluator) uses incremental accumulator updates: when piece moves from square to B, accumulator subtracts embedding and adds B embedding. This works because piece-square features are sparse (only 2 features change per move). Hydra `85x34` dense feature planes lack that sparsity, so NNUE pattern does not transfer.

**Lesson for Hydra**: `ObservationRef<'a>` with `&'a [T]` slices is Rust equivalent of `const Position&`. Zero-copy observation, one lifetime, no `unsafe`.

**Source**: `Stockfish/src/position.h` lines 40-80, `Stockfish/src/nnue/nnue_accumulator.h`

### 5.2 KataGo: Write Into Caller Buffers, Cache NN Outputs

KataGo encoder writes directly into pre-allocated `float*` buffers (`rowBin`, `rowGlobal`) supplied by caller. Batch system pre-allocates contiguous buffer for N observations, then passes slices to each encode call. No per-observation allocation.

More important: KataGo caches NN outputs by board-position hash. If same position appears again (common in search), it skips both encoding and inference. Correct cache level. Encoding costs microseconds; inference costs milliseconds. Cache expensive part.

**Lesson for Hydra**: Batch encoding with caller-owned buffers. NN output hash cache for search/eval. Do not cache encoding results; cache inference results.

**Source**: `KataGo/cpp/neuralnet/nninputs.cpp`, `KataGo/cpp/search/searchnnhelpers.cpp`

### 5.3 Mortal: Encode Directly from State References

Mortal encoder (`libriichi/src/agent/obs_repr.rs`) writes into contiguous `Vec<f32>` buffer with shape `[channels, 4, 9]` (993 elements). It does not use incremental encoding. Full recompute every time. But it encodes directly from `&PlayerState` references, never cloning game state.

Mortal batch parallelism is simple: each game runs independently on its own thread. No shared state, no sync. Games produce MJAI event logs that later batch for inference. This matches Hydra rayon-based `Simulator::run_batch()`.

**Lesson for Hydra**: Full recompute encoding is fine. Win comes from removing allocations, not caching partial results. Encode from refs, not cloned data.

**Source**: `Mortal/libriichi/src/agent/obs_repr.rs`, `Mortal/libriichi/src/arena/` (AGPL, reference only)

### 5.4 lc0: Transposition Table for NN Outputs

Leela Chess Zero (lc0) uses transposition table to cache NN evaluations. When MCTS expands node mapping to previously seen board state, it reuses cached policy + value instead of rerunning inference. Hash is computed from board state, not encoded features.

lc0 does full recompute encoding too. Like KataGo, core idea is: cache expensive thing (inference), not cheap thing (encoding).

**Lesson for Hydra**: When inference-time search arrives (Hydra differentiator over Mortal), NN output cache keyed by state hash will be essential. This is training-level optimization, not engine-level.

**Source**: `LeelaChessZero/lc0/src/neural/cache.h`

### 5.5 The Universal Insight

Nobody does incremental feature encoding. Not Stockfish (it does incremental NNUE accumulators, different problem). Not KataGo. Not Mortal. Not lc0. Not OpenSpiel. All production systems use full recompute encoding and optimize at inference-cache layer.

Not coincidence. Feature-encoding bugs are silent and catastrophic in RL training. One wrong bit in observation tensor silently corrupts policy gradient. Full recompute gives correctness guarantee. Encoding cost (~10-50us) is tiny vs inference (~1-10ms), so there is no reason to risk correctness for sub-1% speedup.

---

## 6. Deep Dives

### 6.1 ObservationRef: Zero-Copy State View

**Pattern**: Replace `Vec<T>` with `&'a [T]`. One lifetime. No `unsafe`. Game state already owns data; observer only reads.

**Production evidence**: qdrant `CompressedPostingListView<'a>` (vector search), iggy `IggyMessageView<'a>` (message broker), tract `TensorView<'a>` (ML inference). All read structured data without cloning.

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

Ctor uses `&self` (immutable borrow), not `&mut self`. Critical: lets code observe multiple players from same state without mutable access. Encoder takes `&ObservationRef` and writes into its own pre-allocated buffer.

**Why not fix get_observation()?** Because `get_observation()` returns owned `Observation` with `Vec` fields. Replacing those with slices would break current API. `ObservationRef` is additive beside existing path.

### 6.2 Batch Shanten with Hierarchical Hash Caching

riichienv-core uses Nyanten/Cryolite lookup tables (~536KB binary data). Hash chain:

```
hash_shupai(manzu[0..9])  -> k0_m  (9 multiply-accumulate ops)
hash_shupai(pinzu[9..18]) -> k0_p  (9 ops)
KEYS1[k0_m * 126 + k0_p]  -> k1    (1 table lookup)
hash_shupai(souzu[18..27])-> k0_s  (9 ops)
KEYS2[k1 * 126 + k0_s]    -> k2    (1 table lookup)
hash_zipai(honors[27..34]) -> k0_z  (7 ops)
KEYS3[(k2*55+k0_z)*5+m]   -> result (1 table lookup)
```

Current path: run chain N+1 times (1 base + 1 per non-zero tile in hand = ~14 calls). Total: ~14 x 38 = 532 ops.

Optimized path: compute 4 suit hashes once. For each discard, rehash only touched suit (9 ops for suited, 7 for honors), then chain forward through KEYS1/KEYS2/KEYS3 with updated hash. Total: 38 base + ~14 x 12 = 206 ops. 61% reduction.

This needs vendored shanten tables (Apache-2.0, MIT-compatible) and `batch_discard_shanten()` returning all 34 discard-shanten values in one pass.

### 6.3 mjai_event! Macro: Zero-Cost Logging

Engine has 103 `serde_json`/`format!` sites building MJAI events. Current `_push_mjai_event()` returns early when `skip_mjai_logging=true`, but callers build `serde_json::Value` before calling it. So those Map+insert+String allocs still happen even when logging disabled. ~5-10 heap allocs per event, 22 events per step, all wasted.

Fix follows two known patterns:

**Pattern 1: `log` crate**. `log!` macro keeps arguments inside if-guard via macro expansion. If log level disabled, format string and arguments are never evaluated. Zero disabled cost.

**Pattern 2: `tracing` crate**. Static callsite interest caching. Each callsite checks cached flag showing whether any subscriber cares. If not, whole span/event build is skipped.

**Hydra's approach**: Two layers.

Layer 1, typed enum replaces `serde_json::Value`:
```rust
pub enum MjaiEvent {
    Tsumo { actor: u8, pai: u8 },
    Dahai { actor: u8, pai: u8, tsumogiri: bool },
    Reach { actor: u8 },
    Chi { actor: u8, target: u8, pai: u8, consumed: [u8; 2] },
    // ... ~15 variants total
}
```
Cost per event: ~16-32 byte stack copy into `Vec<MjaiEvent>`. Old cost: ~5-10 heap allocs for `Map<String, Value>`.

Layer 2, guard-before-evaluate macro:
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

Training binary built without `mjai-logging` feature keeps zero logging trace. With feature: one bool check per event, enum push only when enabled. JSON serialization deferred until game end via `impl Serialize for MjaiEvent`.

### 6.4 step_unchecked(): Skip Validation in Self-Play

`step()` does two expensive things: (1) validate submitted actions are legal, (2) execute game logic. Validation calls `_get_legal_actions_internal(pid)` for every player in action map. Same 498-line fn clones hand+melds 14 times. Then `get_observation()` calls it again.

In self-play, model picks from legal mask. Actions are legal by construction. Validation = pure overhead.

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

Game logic gets extracted into `_execute_actions()`. `step()` calls validate + `_execute_actions()`. `step_unchecked()` calls `_execute_actions()` directly.

**Risk mitigation**: `#[cfg(debug_assertions)]` guard checks legality in debug builds. Release builds skip check entirely. If illegal action somehow slips into production, state may corrupt silently. Acceptable because only caller is self-play loop, which selects from legal mask.

Companion `get_legal_actions(pid)` returns actions without building full Observation. This removes `get_observation()` entirely when caller only needs legal actions.

---

## 7. Results Table

Bench results by optimization wave. Same hardware, same Criterion config.

### Wave 0: Vendor Engine + Release Profile

| Benchmark | Before | After | Change | Techniques |
|-----------|--------|-------|--------|------------|
| `single_game_first_action` | 199us | 155us | -22.1% | T16 (vendor), T7 (release profile) |
| `batch_100_games` | 19.7ms | 15.8ms | -19.8% | T16, T7 |
| `encode_observation` | 61.5ms | 53.1ms | -13.7% | T7 (LTO enables cross-crate inlining) |

Release profile alone (LTO + codegen-units=1 + panic=abort) explains most of this wave. Free speed.

### Wave 1: Encoder Optimizations

| Benchmark | Before (Wave 0) | After | Change | Techniques |
|-----------|-----------------|-------|--------|------------|
| `encode_observation` | 53.1ms | 52.3ms | -1.5% | T2 (branchless thermo), S5 (exp table), T9 (#[inline]), T10 (fused aka), T12 (.last()), T13 (align), T14 (unconditional writes) |
| `single_game_first_action` | 155us | 148us | -4.5% | Same (encoder runs inside game loop) |
| `batch_100_games` | 15.8ms | 14.3ms | -9.5% | Same + T6 (generic ActionSelector) |

Each encoder technique is small alone, but effects compound. Batch bench improved more because rayon parallelism amplifies per-game savings across cores.

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

Engine-level changes (`ObservationRef`, `step_unchecked`, HandEvaluator refs) should have biggest single impact because they remove most per-call allocations.

---

## 8. Remaining Opportunities

Ordered by expected impact. Next targets after current waves.

### 8.1 Bypass get_observation() Entirely (Encode from ObservationRef)

Once `ObservationRef` exists, encoder can read borrowed slices directly from game state. Current path: `get_observation()` (clone everything) -> `bridge::extract_*` (build intermediate types) -> `encoder::encode_*` (write buffer). Target path: `state.observe(pid)` (zero-copy) -> `encoder.encode_ref(&obs_ref, &safety)` (direct writes). This removes both clone step and intermediate extraction.

### 8.2 Complete MJAI Typed Enum Migration

103 call sites in `hydra-engine` `state/mod.rs` build `serde_json::Value` maps. Each needs mechanical 1:1 conversion to typed `MjaiEvent` enum. Payoff: zero heap alloc per MJAI event on training path; JSON serialization delayed to game end for replay/debug only.

### 8.3 NN Output Hash Cache (KataGo Pattern)

When inference-time search lands (Hydra planned differentiator over Mortal), many search nodes will hit previously evaluated positions. state-hash cache avoids redundant inference. KataGo and lc0 both use this. Cache should hold one search tree worth of evals (~10K-100K entries).

### 8.4 Profile-Guided Optimization (PGO)

PGO uses execution profiles from representative workload to guide branch layout, inlining, code placement. Usually adds 5-15% on top of LTO. Workflow:

```bash
cargo install cargo-pgo
cargo pgo build                           # Build instrumented binary
cargo pgo run -- [benchmark command]      # Generate profile data
cargo pgo optimize                        # Rebuild with profile
```

PGO helps most on branch-heavy code, exactly what game engine is. Loop branches for calls, kan, furiten, scoring should benefit from profile-guided layout.

### 8.5 Batch Encoder for Training Pipeline

For training throughput, encode N observations into one contiguous buffer (`[batch_size, 85, 34]` row-major). Pre-allocate once, reuse across steps. Layout matches GPU tensor format, so buffer can be `memcpy`'d directly to GPU via burn-tch backend. KataGo and Mortal both use this caller-owned-buffer pattern.

### 8.6 2026-03 Optimization Checkpoint

This branch now contains long run of low-risk, green, benchmark-backed optimizations across `hydra-train` and `hydra-core`. Key point: not only code changed; each slice was verified on its directly dependent test surface before next slice.

#### Landed slices

Training-side / orchestration:
- BF16 BC rollout across training, probe, preflight, runtime-autotune, stage-two bench surfaces
- manifest-cache reuse across BC preflight / bootstrap / probe child paths
- replay-loader event-local opponent-target caching, placement precompute, sidecar/replay-key fast-path gating
- validation baseline-forward trim (same-model short-circuit and policy-only distinct baseline path)
- duplicate BC/validation batch materialization cleanup via owned collate helpers
- BC/RL scalar-sync cleanup
- model/inference CPU-bridge cleanup (`policy_cpu`, `value_cpu`, borrowed fill paths, CPU-side `infer_action`)
- RL batch scratch reuse and scalar GAE cleanup
- train-bin bookkeeping cleanup in `probe_search`, `probe_ladder`, `runtime_autotune`, `probe_summary`
- reporting helper cleanup (`delta_q_promotion`, `exit_validation`, `delta_q_validation`, `progress` fallback metrics)

Core simulation / search:
- `GameRunner` reuse in `hydra-core` batch sim paths
- `ObservationRef` discard-metadata parity with owned `Observation`
- zero-copy child observation encoding in hot self-play live-exit path

Replay correctness:
- replay `Tsumo` parity fixes in `hydra-engine` (hand-unique tile selection and forbidden-discard reset)

#### Current benchmark snapshot

Core benches (`cargo bench -p hydra-core --bench simulator_bench`, `ct_smc_bench`, `agari_bench`):

| Benchmark | Current snapshot |
|-----------|------------------|
| `single_game_first_action` | ~418 µs |
| `single_game_first_action_reuse` | ~472 µs |
| `batch_100_games` | ~4.05 ms |
| `encode_observation` | ~3.05 ms |
| `encode_observation_ref` | ~3.05 ms |
| `ct_smc_dp_128_samples` | ~2.12 ms |
| `hand_evaluator/calc_4p` | ~705 µs |
| `hand_evaluator/calc_3p` | ~290 µs |
| `calculate_score` | ~62.5 ns |

Training-side benches (`cargo bench -p hydra-train --bench train_hotpaths_bench`):

| Benchmark | Current snapshot |
|-----------|------------------|
| `loader/load_game_from_reader` | ~10.7 ms |
| `validation/collate_only` | ~33.7 µs |
| `validation/forward_loss_only` | ~1.33–1.40 s |
| `validation/collate_forward_loss` | ~0.55–1.10 s |
| `selfplay_batch/trajectories_to_rl_batch` | ~30–31 µs |
| `selfplay_batch/trajectories_to_rl_batch_reuse` | ~30.6 µs |
| `model_cpu_bridge/policy_value_cpu` | ~0.58–0.73 s |
| `model_cpu_bridge/policy_cpu` | ~0.93–0.99 s |
| `model_cpu_bridge/value_cpu` | ~0.95–1.00 s |
| `model_cpu_bridge/batch_policy_value_cpu_reuse` | ~2.46–3.50 s |
| `model_cpu_bridge/batch_value_cpu_reuse` | ~2.28–3.72 s |

#### Interpretation

- Corrected `batch_100_games` bench shows reused sim path truly wins once bench measures real production path.
- `ObservationRef` now good enough for hot zero-copy use, but `encode_observation_ref` microbench is not materially faster than owned encoding. Win is removal of allocations/copies in hot chains.
- `ct_smc` still costs nontrivially, but obvious stack-allocation experiment regressed and was reverted. Signal: stop guessing there; require stronger bench evidence.
- Training side: replay-loader and RL-batch collation are now relatively cheap vs validation forward/loss, which remains biggest measured CPU cost center in current harness.

#### Practical next choices

If work continues from this checkpoint, strongest evidence-backed directions:

1. **Deeper validation compute cost**
   - training bench now shows validation forward/loss dominates collation
   - future changes there should stay benchmark-driven and semantics-locked

2. **Broader `hydra-core` simulation / observation work**
   - only if guided by simulator benches, not intuition
   - likely around larger observation or loop architecture, not more helper churn

3. **Stop here and benchmark end-to-end training**
   - branch already contains many low-risk wins
   - end-to-end train/preflight bench now more valuable than more local helper edits

#### Experiments worth NOT repeating blindly

This branch also produced useful negative results. Record them so future passes do not waste time on same failed ideas.

- **`ct_smc` stack-array rewrite**
  - Replacing temp `Vec`s in hot sampling loop with fixed stack arrays looked good, but measured `ct_smc_dp_128_samples` regressed and change was reverted.
  - Lesson: this path needs benchmark-first changes, not assumed allocation folklore.

- **Approximate `TargetPresence` preservation in sliced targets**
  - Trying to carry scaled-down `TargetPresence` metadata through `HydraTargets::slice_batch()` broke contract and tests. Reverted.
  - Lesson: metadata reuse only matters when metadata stays exact.

- **Benchmark interpretations must match actual code path**
  - `batch_100_games` first appeared to regress until bench itself was updated to measure reused sim path rather than old fresh-runner path.
  - Lesson: late-stage optimization needs benchmark maintenance as much as code changes.

- **Zero-copy `ObservationRef` is mainly about allocation/copy reduction, not guaranteed encoder speedups**
  - `encode_observation_ref` is not materially faster than owned `encode_observation` in microbench.
  - Still worth using on hot sim/search paths because it removes owned `Observation` construction and copies, but should not be justified by encoder bench alone.

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
| arrayvec crate | https://docs.rs/arrayvec | Fixed-capacity stack vectors (alt to `Vec` for bounded collections) |
| criterion | https://github.com/bheisler/criterion.rs | Bench framework used for all measurements in this doc |
| cargo-pgo | https://github.com/Kobzol/cargo-pgo | PGO workflow for Rust |
| RiichiEnv (smly) | https://github.com/smly/RiichiEnv | Engine foundation (Apache-2.0), Nyanten shanten tables |
| riichienv-core shanten | `hydra-engine/riichienv-core/src/shanten/` | Vendored lookup tables: KEYS1, KEYS2, KEYS3, hash fns |
| Hydra encoder | `hydra-core/src/encoder.rs` | `192x34` fixed-superset tensor encoder; first 85 channels keep baseline prefix |
| Hydra benchmarks | `hydra-core/benches/simulator_bench.rs` | Criterion benches: single_game, batch_100, encode_observation |
| Hydra bridge | `hydra-core/src/bridge.rs` | Observation extraction layer (`extract_*` fns) |
| Hydra safety | `hydra-core/src/safety.rs` | Safety channel computation (genbutsu, suji, kabe) |
| HYDRA_FINAL.md | `research/design/HYDRA_FINAL.md` | Promoted architecture doctrine summary |
| INFRASTRUCTURE.md | `research/infrastructure/INFRASTRUCTURE.md` | Stack decisions, throughput targets, batch sim design |
| TESTING.md | `research/design/TESTING.md` | Golden encoder tests, property-based tests, correctness verification |