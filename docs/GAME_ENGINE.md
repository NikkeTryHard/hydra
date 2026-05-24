# Hydra Game Engine (`hydra-core` facade)

Ref doc for public `hydra-core` game-engine API after crate split.

## Overview

`hydra-core` is public facade + orchestration crate. It re-exports split impl crates for stable callers, owns bridge/simulator/game-loop/seeding, and wraps `riichienv-core` as rules engine.

Core responsibilities:

- Runtime tile/action API via `hydra-runtime-types`, exposed through `hydra-core`
- 192-channel x 34-tile fixed-superset observation encoder via `hydra-encoder`; first 85 channels preserve original public+safety baseline, Groups C/D add live search/belief + Hand-EV planes
- Tile safety analysis via `hydra-safety` (genbutsu, suji, kabe, one-chance)
- Belief/search feature helpers via `hydra-belief-search`
- Deterministic seeding via SHA-256 KDF + ChaCha8Rng
- Parallel batch sim with `rayon`
- Game loop abstraction with pluggable action selection

Hydra uses 100% Rust stack (see `research/infrastructure/INFRASTRUCTURE.md`). Training/runtime callers may depend on `hydra-core` facade; impl ownership lives in split crates below.

## Foundation: RiichiEnv

Game engine built on [smly/RiichiEnv](https://github.com/smly/RiichiEnv) (`riichienv-core` crate, Apache-2.0 license).

RiichiEnv provides:

- Full 4-player + 3-player Riichi Mahjong rules
- Red dora (aka-dora) support for all three suits
- All kan types: ankan (closed), daiminkan (open), shouminkan (added)
- Native MJAI protocol compatibility for game state repr
- Correctness verified by running MortalAgent (AGPL, black-box MJAI player, no code shared) over 1M+ hanchan with zero errors ([source: RiichiEnv README](https://github.com/smly/RiichiEnv#-features))

Hydra treats `riichienv-core` as black-box game engine. All game state progression, legality checks, rule enforcement happen inside RiichiEnv. Hydra handles encoding, analysis, orchestration only.

Because `riichienv-core` correctness already verified upstream: smly ran Mortal as black-box MJAI player (separate process, no linking) over 1M+ hanchan on RiichiEnv with zero errors ([source](https://github.com/smly/RiichiEnv)). Hydra needs no own cross-engine validation. Correctness inherited through dependency. No Mortal code exists in RiichiEnv or Hydra.

## Module / Crate Ownership

| Public route | impl owner | Role |
|--------------|----------------------|------|
| `hydra_core::tile` | `hydra-runtime-types::tile` | Tile constants/types, 136-format repr, aka-dora, suit permutation |
| `hydra_core::action` | bridge in `hydra-core` over `hydra-runtime-types::action` | 46-action `HydraAction`, riichienv conversion, legal mask builder |
| `hydra_core::encoder` | `hydra-encoder::encoder` | 192x34 fixed-superset tensor, `ObservationEncoder`, `DirtyFlags` |
| `hydra_core::batch_encoder` | `hydra-encoder::batch_encoder` | Pre-allocated contiguous batch encoding buffer |
| `hydra_core::safety` | `hydra-safety` | `SafetyInfo`: genbutsu, suji, kabe, one-chance |
| `hydra_core::{ct_smc, hand_ev, afbs, endgame, robust_opponent, shanten_batch, sinkhorn}` | `hydra-belief-search` | Belief/search, Hand-EV, shanten cache, Sinkhorn/SIB helpers |
| `hydra_core::bridge` | `hydra-core::bridge` | riichienv `Observation` -> encoder-ready data |
| `hydra_core::simulator` | `hydra-core::simulator` | `BatchSimulator`, `BatchConfig`, `GameResult` collection |
| `hydra_core::game_loop` | `hydra-core::game_loop` | `GameRunner`, `ActionSelector`, run loop |
| `hydra_core::seeding` | `hydra-core::seeding` | SHA-256 KDF, `SessionRng`, deterministic wall shuffle |
| `hydra_core::arena` | `hydra-core::arena` | Arena/runtime glue for core orchestration |


## Tile System (`hydra-runtime-types::tile`, via `hydra_core::tile`)

### TileType

All tiles use `TileType(u8)` newtype for 34 Mahjong tile kinds:

| Range | Tiles | Count |
|-------|-------|-------|
| 0-8 | 1m through 9m (manzu/characters) | 9 |
| 9-17 | 1p through 9p (pinzu/circles) | 9 |
| 18-26 | 1s through 9s (souzu/bamboo) | 9 |
| 27-33 | East, South, West, North, Haku, Hatsu, Chun | 7 |

Physical game uses 136 tiles (4 copies each type). 136-format index identifies specific physical tile; `TileType` identifies kind. Conversion = `tile136 / 4` truncation.

### Aka-Dora (Red Fives)

Three 136-format tiles are red dora (aka-dora):

- Red 5m (manzu)
- Red 5p (pinzu)
- Red 5s (souzu)

These are 0th copy (index 0 within each group of 4) of respective 5-tiles: 136-format indices 16 (5m), 52 (5p), 88 (5s). Extended tile type indices 34-36 represent aka variants in action space. Encoder + action space both treat aka-dora distinct from regular fives where needed.

### Suit Permutation

For training data aug, runtime tile API provides suit permutation fns. Six permutations exist for three numbered suits (manzu, pinzu, souzu); honor tiles stay unchanged. Given permutation index (0-5), module remaps all tile types in hand/observation to permuted suit assignment. This 6x aug helps model learn suit-invariant patterns.

## Action Space (`hydra-core::action`, uses `hydra-runtime-types::action`)

### 46-Action Space

Hydra uses fixed 46-action output space. Every decision point maps to one action index:

| Index | Action | Notes |
|-------|--------|-------|
| 0-33 | Discard tile type 0-33 | Standard discard (non-red) |
| 34-36 | Discard aka 5m, 5p, 5s | Discard specific red five |
| 37 | Declare riichi | Announces riichi; tile selection follows |
| 38-40 | Chi (3 variants) | Left/middle/right chi calls |
| 41 | Pon | Open pon call |
| 42 | Kan | Any kan type (ankan, daiminkan, shouminkan) |
| 43 | Agari | Win declaration (tsumo or ron) |
| 44 | Ryuukyoku | Abortive draw declaration (kyuushu kyuuhai, etc.) |
| 45 | Pass | Decline call opportunity |

### Two-Phase / Compact Actions
Riichi uses two-phase selection: model declares action 37, then phase-select mask limits follow-up to legal discard actions. Kan is compact in current bridge: Hydra action 42 maps to `Ankan` in `ActionPhase::Normal`, otherwise `Daiminkan`; inbound riichienv `Daiminkan | Ankan | Kakan` collapse to 42. Specific kan tile/type choice stays rules-engine/context concern, not extra Hydra action IDs.
Hydra compact action facade is 4-player. `riichienv-core`/`hydra-engine` supports sanma/Kita, but `hydra-core::action` does not represent `ActionType::Kita`; conversion skips/errors instead of assigning 46-action ID.

### HydraAction

`HydraAction` is validated newtype wrapper around `u8`:

```rust
pub struct HydraAction(u8);
```

It validates range 0-45 on construction via `HydraAction::new(id) -> Option<Self>`. Methods like `is_discard()`, `is_aka_discard()`, `discard_tile_type()` give type-safe access. Bidirectional conversion fns `hydra_to_riichienv()` and `riichienv_to_hydra()` translate between Hydra compact action space and riichienv-core `Action` struct, using `GameContext` to resolve context-dependent actions (chi consume tiles, tsumo vs ron, kan type).

### Legal Action Mask

`build_legal_mask(legal_actions: &[Action], phase: ActionPhase) -> [bool; 46]` converts current riichienv legal actions into Hydra compact mask. Each slot = `true` if action legal now. Phase-select states restrict follow-up to discard actions. Training pipeline uses this mask to zero illegal actions before softmax.

## Observation Encoder (`hydra-encoder::encoder`, via `hydra_core::encoder`)

### Tensor Shape

**Routing note:** this file records current runtime reality for live encoder/runtime, and current code wins if docs drift. For repo entry routing, trust/status vocabulary, crate ownership, read `README.md`. For active-path / staged-vs-reserve decisions, read `research/design/HYDRA_RECONCILIATION.md`. For compact compatibility contract, read `docs/COMPATIBILITY_SURFACE.md`. Original `85 x 34` tensor now means **baseline prefix** of live encoder, not full live encoder. Current impl already **fixed-shape 192 x 34 superset** with Groups C/D plus presence-mask channels.

Each observation is `192 x 34` float tensor (6,528 values). First 85 channels retain baseline public+safety encoding; rest provide fixed-shape search/belief + Hand-EV context with zero-fill plus explicit presence masks when dynamic features unavailable. Full shape feeds current SE-ResNet model input.

### Baseline Prefix Channel Layout (channels 0-84)

85 channels break into groups:

| Channels | Name | Encoding |
|----------|------|----------|
| 0-3 | Closed hand | Thresholded: ch N = 1.0 if tile count >= N+1 |
| 4-7 | Open meld hand | Same thresholding for tiles exposed in open melds |
| 8 | Drawn tile | One-hot: 1.0 at tile type drawn (tsumo only) |
| 9-10 | Shanten masks | Ch 9: keep-shanten (discard does not increase shanten). Ch 10: next-shanten (discard decreases shanten) |
| 11-13 | Player 0 discards | Presence (1.0 if discarded), tedashi flag (1.0 if from hand, not tsumogiri), temporal weight (`exp(-0.2 * age)`) |
| 14-16 | Player 1 discards | Same three channels, relative to seat |
| 17-19 | Player 2 discards | Same three channels, relative to seat |
| 20-22 | Player 3 discards | Same three channels, relative to seat |
| 23-25 | Player 0 melds | Chi (1.0 for tiles in chi melds), pon (tiles in pon), kan (tiles in kan) |
| 26-28 | Player 1 melds | Same three channels |
| 29-31 | Player 2 melds | Same three channels |
| 32-34 | Player 3 melds | Same three channels |
| 35-39 | Dora indicators | Thermometer encoding: ch N = 1.0 if N+1 or more dora indicators revealed |
| 40-42 | Aka dora flags | Per-suit plane: ch 40 = manzu red five, ch 41 = pinzu, ch 42 = souzu. 1.0 at 5-tile column if red five visible |
| 43-46 | Riichi flags | One channel per player. Whole plane = 1.0 if player declared riichi |
| 47-50 | Scores | One channel per player. Whole plane filled with score / 100,000 |
| 51-54 | Relative score gaps | One channel per player. Filled with (player_score - my_score) / 30,000 |
| 55-58 | Shanten one-hot | Ch 55 = tenpai (0), ch 56 = iishanten (1), ch 57 = ryanshanten (2), ch 58 = 3+ shanten. Whole plane = 1.0 for matching shanten |
| 59 | Round number | Whole plane filled with kyoku / 8.0 (normalized round index) |
| 60 | Honba count | Whole plane filled with honba / 10.0 |
| 61 | Kyotaku (riichi sticks) | Whole plane filled with kyotaku / 10.0 |
| 62-84 | Safety channels | 23 channels of per-opponent tile safety data (see Safety System section) |

**Safety channel breakdown (channels 62-84):**

| Channels | Name |
|----------|------|
| 62-64 | Genbutsu (all): 1.0 for tiles each opponent discarded (one ch per opponent) |
| 65-67 | Genbutsu (tedashi): only tiles discarded from hand (not tsumogiri) |
| 68-70 | Genbutsu (riichi-era): only tiles discarded after opponent's riichi |
| 71-73 | Suji: float 0.0-1.0 for suji-inferred safety vs each opponent |
| 74-76 | Half-suji indicator | 1.0 when tile is half-suji-safe vs that opponent |
| 77-79 | Matagi-suji danger | float danger signal for matagi-suji patterns vs that opponent |
| 80 | Kabe: 1.0 for tiles with all 4 copies visible (global, not per-opponent) |
| 81 | One-chance: 1.0 for tiles where exactly 3 of 4 copies are visible |
| 82-84 | Tenpai hints | Opponent tenpai hints (baseline impl: riichi or cached tenpai prediction threshold) |

### ObservationEncoder

`ObservationEncoder` is main struct for building observation tensors. Current impl holds pre-allocated `[f32; 192 * 34]` buffer marked `#[repr(C)]` for predictable layout. Baseline public+safety channels stay intact in first 85 planes; Groups C/D already present as fixed-shape extensions.

```rust
#[repr(C)]
    pub struct ObservationEncoder {
    buffer: [f32; 6528],  // 192 channels x 34 tiles, row-major
}
```

### Incremental Encoding with DirtyFlags

`DirtyFlags` is bitflags struct; each bit maps to channel group (hand, discards, melds, dora, scores, safety, etc.). When game state changes, only relevant flags are set. On next `encode()` call, only flagged groups recompute. Unchanged channels stay in buffer.

This matters for perf: one discard dirties only discard + safety channels, skipping more expensive hand/meld/dora re-encoding. In batch sim of thousands of games, savings compound.

## Safety System (`hydra-safety`, via `hydra_core::safety`)

Safety module computes per-opponent, per-tile safety info used to fill encoder channels 62-84 and guide defensive play.

### SafetyInfo

`SafetyInfo` holds safety data from one player's view against all 3 opponents:

```rust
#[repr(C)]
pub struct SafetyInfo {
    pub genbutsu_all: [[bool; 34]; 3],       // per-opponent
    pub genbutsu_tedashi: [[bool; 34]; 3],   // per-opponent
    pub genbutsu_riichi_era: [[bool; 34]; 3], // per-opponent
    pub suji: [[f32; 34]; 3],                // per-opponent, float 0.0-1.0
    pub kabe: [bool; 34],                     // global
    pub one_chance: [bool; 34],               // global
    pub visible_counts: [u8; 34],             // global tile visibility
    pub opponent_riichi: [bool; 3],           // per-opponent riichi status
}
```

**Genbutsu** (safe tiles) tracks tiles specific opponent cannot ron:

- `genbutsu_all`: any tile opponent discarded safe vs that player)
- `genbutsu_tedashi`: only tiles discarded from opponent hand (not tsumogiri), showing intentional discards
- `genbutsu_riichi_era`: only tiles discarded after opponent declared riichi, useful for post-riichi wait reading

**Suji** inference identifies tiles protected by 1-4-7 / 2-5-8 / 3-6-9 suji relation. If opponent discarded 4m, then 1m + 7m get suji safety (float 1.0) vs that opponent. Suji applies only to suited tiles (0-26); honors have none. Values update incrementally as new discards appear.

**Kabe** (wall block) marks tiles where all 4 copies are visible in info (discards, melds, own hand). Tile with all copies visible cannot be part of any opponent winning hand.

**One-chance** marks tiles where exactly 3 of 4 copies are visible, meaning one unknown copy remains. Reduced but nonzero risk.

All safety arrays update incrementally. When new discard or meld occurs, only affected opponent `SafetyInfo` recomputes.

## Batch Simulator (`hydra-core::simulator`)

### BatchSimulator

`BatchSimulator` runs many games in parallel using `rayon::ThreadPool`. Each game runs on its own thread with no shared mutable state between games.

```rust
pub struct BatchSimulator {
    pool: rayon::ThreadPool,
}
```

### BatchConfig

```rust
pub struct BatchConfig {
    pub num_games: usize,
    pub base_seed: Option<u64>,
    pub num_threads: Option<usize>,  // None = rayon default (num CPUs)
    pub game_mode: u8,               // 0 = hanchan, 1 = east only, 2 = single round
}
```

`hydra-core::BatchConfig` documents 4-player modes `0..2`. `hydra-engine::GameStateVariant` routes `game_mode >= 3` to 3-player engine modes: `3 = single`, `4 = east`, `5 = half`. Hydra compact 46-action bridge remains 4-player; sanma/Kita is engine-level surface.

Each game derives seed as `base_seed + game_index`. Two runs with same `BatchConfig` produce identical results regardless of thread scheduling.

### GameResult

`GameResult` collects outcome of one game: final scores for all four players, rounds played, total actions taken, seed used. Batch simulator returns `Vec<GameResult>`.

### Convenience Function

`run_batch_simple` is free fn using rayon global thread pool instead of dedicated one. Easiest entry point for scripts + benchmarks that need no custom pool config.


## Seeding (`hydra-core::seeding`)

Deterministic seeding is critical for reproducible training + eval. Seeding module provides hierarchical RNG system.

### Key Derivation

Session seed is `[u8; 32]`. `SessionRng` derives per-game seeds via `SHA-256(session_seed || game_index_le_bytes)`. `derive_kyoku_seed` further derives per-round seeds: `SHA-256(session_seed || nonce || kyoku || honba)`.

```
game_seed = SHA-256(session_seed || game_index)[0..32]
```

This ensures every game in batch gets unique deterministic seed from one session seed. Changing session seed changes all games. Changing game index changes only that game.

### SessionRng

`SessionRng` holds 32-byte seed + auto-incrementing game index counter. Each call to `next_game_seed()` derives new 32-byte seed and advances counter. Gives 2^64 independent game seeds from one session seed.

### Wall Generation

`generate_wall` takes session seed, nonce, kyoku number, honba count. It derives kyoku-specific seed, seeds fresh `ChaCha8Rng`, initializes sorted `[0..135]` wall, applies vendored Fisher-Yates shuffle. Vendored impl avoids dependence on `rand::seq::SliceRandom` internals that might change between `rand` versions.

### Determinism Guarantees

Given same session seed + batch config, `hydra-core` produces bit-identical results across:

- Different runs on same machine
- Different thread counts (`rayon` scheduling deterministic per-game)
- Different platforms (x86_64, aarch64) thanks to vendored shuffle

Only requirement: same Rust toolchain version, since floating-point encoder output depends on compiler codegen.

## Game Loop (`hydra-core::game_loop`)

### GameRunner

`GameRunner` orchestrates one game start to finish. It holds riichienv `GameState`, `[SafetyInfo; 4]` array (one per player view), and action/round counters.

Runner exposes three execution modes:

- `step_once(selector)`: advance game one step using given `ActionSelector`. Handles round transitions (auto-resets safety), WaitAct vs WaitResponse phases. Returns `false` when game over.
- `step_once_recording(selector, recorder)`: same real rules/legal path, but emits `DecisionRecord` for each player decision before applying actions. Record includes encoded obs `[192,34]`, legal mask `[bool;46]`, selected compact Hydra action, legal count, player/seat, and turn. This is smoke/boundary plumbing, not production actor-pool self-play.
- `run_to_completion(selector)`: play full game by calling checked steps in loop. Provides accessors for `scores()`, `total_actions()`, `rounds_played()`, and `safety(player)` after completion.

### ActionSelector Trait

```rust
pub trait ActionSelector {
    fn select_action(&mut self, player: u8, legal_actions: &[Action]) -> Action;
}
```

Any type implementing `ActionSelector` can drive game loop. `FirstActionSelector` is simple built-in choosing first legal action (useful for tests + benchmarks). Training pipeline provides its own selectors that call neural network.

### Safety Tracking

During play, `GameRunner` maintains `[SafetyInfo; 4]` array (one per player view). After every discard, call, and riichi event, runner `track_action` incrementally updates relevant safety data across all views. These feed encoder safety channels (62-84) on next observation request.

## Bridge (`hydra-core::bridge`)

Bridge module converts riichienv `Observation` into data encoder needs. Translation layer so encoder does not depend on riichienv types directly.

### Extract Functions

Each `extract_*` fn pulls one data category from riichienv observation:

- `extract_hand()`: closed hand tile counts + open meld tile counts
- `extract_discards()`: per-player discard sequences with tedashi + temporal info
- `extract_melds()`: per-player meld data (chi/pon/kan tile lists)
- `extract_dora()`: dora indicator tiles + aka-dora visibility
- `extract_metadata()`: scores, round number, honba, kyotaku, riichi states, shanten

### Entry Point

`encode_observation` is main bridge fn. Takes riichienv `Observation`, calls all `extract_*` fns, feeds results into `ObservationEncoder`. Returns filled 192x34 fixed-superset float buffer ready for model.

Current runtime-status note: bridge carries two live Hand-EV paths on same fixed surface. Default computes Hand-EV from public remaining counts. When search context supplies CT-SMC posterior, it upgrades to wall-weighted remaining counts while keeping same encoder/runtime interface. Bridge also fills fixed-shape Group C search/belief planes from live mixture/search/risk context when present, else zero-filled planes plus presence masks.

## Testing

Split impl crates keep inline unit tests (`#[cfg(test)]`). Runtime-facing integration tests:

| Test File | What It Covers |
|-----------|---------------|
| `crates/hydra-core/tests/golden_encoder.rs` | Encoder regression tests. Compares output against saved golden snapshots. Catches silent encoding drift when channel logic changes. |
| `crates/hydra-core/tests/mjai_replay.rs` | Replays recorded MJAI game logs through engine and verifies game state, actions, observations match expected sequence. Current regression surface explicitly covers replay round-reset correctness + kan-action legality matching so MJAI replay stays aligned with runtime legality checks. |
| `crates/hydra-core/tests/proptest_invariants.rs` | Property-based tests using `proptest`. Generates random game states, verifies invariants: legal mask consistency, encoder channel bounds, tile count conservation, action round-trip fidelity. |
| `crates/hydra-core/tests/game_loop_integration.rs` | End-to-end game loop tests. Runs complete games with `FirstActionSelector`, verifies termination, score consistency, result collection. |
| `crates/hydra-train/tests/integration_pipeline.rs` | Training-stack smoke/integration checks over model, loss, distill, GAE/DRDA, CT-SMC/AFBS, and replay sidecar compile surfaces. |

Current replay-status note: after fixing MJAI replay round-start reset semantics + kan replay matching in vendored engine layer, Hydra MJAI loader was re-audited against Tenhou Houou 2025 corpus (`178,897` files) with `0` skips.

### Benchmarks

Criterion benches by crate:

- `crates/hydra-core/benches/simulator_bench.rs`: `single_game_first_action`, `single_game_first_action_reuse`, `batch_100_games`, encoder variants, Hand-EV, shanten batches.
- `crates/hydra-core/benches/ct_smc_bench.rs`: `ct_smc_dp_128_samples`.
- `crates/hydra-engine/benches/agari_bench.rs`: 4p/3p agari evaluator and scoring fixture cases.
- `crates/hydra-train/benches/train_hotpaths_bench.rs`: loader, shard collation, validation grouping/stats, RL batch collation, model CPU bridge, self-play source generation.

Run benchmarks with `cargo bench -p hydra-core`, `cargo bench -p hydra-engine`, or `cargo bench -p hydra-train`.

## Direct Dependencies (`hydra-core`)

| Crate | Purpose |
|-------|---------|
| `riichienv-core` | Vendored rules/state/legality engine (`hydra-engine`) |
| `hydra-runtime-types` | Runtime tile/action rails re-exported by facade |
| `hydra-safety` | Safety analysis re-exported by facade |
| `hydra-encoder` | Observation + batch encoder re-exported by facade |
| `hydra-belief-search` | Belief/search helpers re-exported by facade |
| `rayon` | Parallel batch sim |
| `serde`, `serde_json` | Config/result/replay serialization |
| `rand`, `rand_chacha`, `sha2` | Deterministic seeding |
| `anyhow`, `thiserror` | Error handling |
| `dashmap`, `smallvec` | Runtime support containers |

### Dev / Test

| Crate | Purpose |
|-------|---------|
| `proptest` | Property-based testing framework |
| `criterion` | Benchmarking framework |

## License

`hydra-core` is BSL-1.1 (see [hydra-core/LICENSE](../crates/hydra-core/LICENSE)). `hydra-engine` is Apache-2.0 (vendored upstream). All dependencies use MIT, Apache-2.0, or BSD-compatible licenses.