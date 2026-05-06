# Hydra Game Engine (hydra-core)

Ref doc for `hydra-core` Rust crate, game engine powering Hydra Riichi Mahjong AI.

## Overview

`hydra-core` Rust lib gives Hydra training + runtime all game-side needs: full Riichi Mahjong sim, observation encoding, safety analysis, search/belief feature bridge, batch exec. Wraps `riichienv-core` as base engine; adds Hydra-specific encoding, seeding, orchestration.

Core responsibilities:

- Tile repr + suit permutation for data aug
- 46-action space with bidirectional conversion to/from `riichienv` actions
- Current **192-channel x 34-tile fixed-superset observation encoder**; first 85 channels preserve original public+safety baseline, Groups C/D add live search/belief + Hand-EV planes
- Tile safety analysis (genbutsu, suji, kabe, one-chance)
- Deterministic seeding via SHA-256 KDF + ChaCha8Rng
- Parallel batch sim with `rayon`
- Game loop abstraction with pluggable action selection

Hydra uses 100% Rust stack (see `research/infrastructure/RUST_STACK.md`). Training pipeline (`hydra-train`, Burn) consumes hydra-core directly; same process, same memory, zero IPC.

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

## Module Reference

| Module | File | Description |
|--------|------|-------------|
| `tile` | `tile.rs` | Tile types (0-33), 136-format repr, aka-dora handling, suit permutation |
| `action` | `action.rs` | 46-action space, `HydraAction` enum, bidirectional riichienv conversion, legal mask builder |
| `encoder` | `encoder.rs` | 192x34 fixed-superset observation tensor, `ObservationEncoder`, incremental encoding with `DirtyFlags` |
| `safety` | `safety.rs` | `SafetyInfo` per-opponent tile safety: genbutsu, suji, kabe, one-chance |
| `simulator` | `simulator.rs` | `BatchSimulator` with rayon thread pool, `BatchConfig`, `GameResult` collection |
| `seeding` | `seeding.rs` | SHA-256 KDF, `SessionRng`, deterministic wall generation, Fisher-Yates shuffle |
| `bridge` | `bridge.rs` | Converts riichienv `Observation` into encoder-ready data via `extract_*` fns |
| `game_loop` | `game_loop.rs` | `GameRunner`, `ActionSelector` trait, step-by-step or run-to-completion exec |
| `batch_encoder` | `batch_encoder.rs` | Pre-allocated contiguous buffer for encoding N observations without per-obs allocation |
| `shanten_batch` | `shanten_batch.rs` | Batch shanten with hierarchical hash caching (base + all 34 discards in one pass) |


## Tile System (`tile.rs`)

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

For training data aug, `tile.rs` provides suit permutation fns. Six permutations exist for three numbered suits (manzu, pinzu, souzu); honor tiles stay unchanged. Given permutation index (0-5), module remaps all tile types in hand/observation to permuted suit assignment. This 6x aug helps model learn suit-invariant patterns.

## Action Space (`action.rs`)

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

### Two-Phase Actions

Riichi + kan use two-phase selection. Model first outputs phase-1 action (37 for riichi, 42 for kan). Then game engine presents legal tile choices and model picks specific tile to discard (riichi) or specific kan to declare. Keeps action space compact at 46 while supporting full combinatorial range.

### HydraAction

`HydraAction` is validated newtype wrapper around `u8`:

```rust
pub struct HydraAction(u8);
```

It validates range 0-45 on construction via `HydraAction::new(id) -> Option<Self>`. Methods like `is_discard()`, `is_aka_discard()`, `discard_tile_type()` give type-safe access. Bidirectional conversion fns `hydra_to_riichienv()` and `riichienv_to_hydra()` translate between Hydra compact action space and riichienv-core `Action` struct, using `GameContext` to resolve context-dependent actions (chi consume tiles, tsumo vs ron, kan type).

### Legal Action Mask

`build_legal_action_mask` takes current riichienv game state, returns `[bool; 46]`. Each slot = `true` if action legal now. Training pipeline uses this mask to zero illegal actions before softmax, so model never selects impossible move.

## Observation Encoder (`encoder.rs`)

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

## Safety System (`safety.rs`)

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

## Batch Simulator (`simulator.rs`)

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
    pub game_mode: u8,               // 0 = hanchan, 1 = east only
}
```

Each game derives seed as `base_seed + game_index`. Two runs with same `BatchConfig` produce identical results regardless of thread scheduling.

### GameResult

`GameResult` collects outcome of one game: final scores for all four players, rounds played, total actions taken, seed used. Batch simulator returns `Vec<GameResult>`.

### Convenience Function

`run_batch_simple` is free fn using rayon global thread pool instead of dedicated one. Easiest entry point for scripts + benchmarks that need no custom pool config.

### Planned: Pre-Allocated Game Pools

Current each game in batch allocates fresh `GameState`. Future opt: pool pre-allocated game states and recycle between batches, removing per-game allocation overhead during high-throughput self-play.

## Seeding (`seeding.rs`)

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

## Game Loop (`game_loop.rs`)

### GameRunner

`GameRunner` orchestrates one game start to finish. It holds riichienv `GameState`, `[SafetyInfo; 4]` array (one per player view), and action/round counters.

Runner exposes two execution modes:

- `step_once(selector)`: advance game one step using given `ActionSelector`. Handles round transitions (auto-resets safety), WaitAct vs WaitResponse phases. Returns `false` when game over.
- `run_to_completion(selector)`: play full game by calling `step_once` in loop. Provides accessors for `scores()`, `total_actions()`, `rounds_played()`, and `safety(player)` after completion.

### ActionSelector Trait

```rust
pub trait ActionSelector {
    fn select_action(&mut self, player: u8, legal_actions: &[Action]) -> Action;
}
```

Any type implementing `ActionSelector` can drive game loop. `FirstActionSelector` is simple built-in choosing first legal action (useful for tests + benchmarks). Training pipeline provides its own selectors that call neural network.

### Safety Tracking

During play, `GameRunner` maintains `[SafetyInfo; 4]` array (one per player view). After every discard, call, and riichi event, runner `track_action` incrementally updates relevant safety data across all views. These feed encoder safety channels (62-84) on next observation request.

## Bridge (`bridge.rs`)

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

Current runtime-status note: bridge now carries two live Hand-EV paths on same fixed surface. Default computes Hand-EV from public remaining counts. When search context supplies CT-SMC posterior, it upgrades that path to use wall-weighted remaining counts from posterior while keeping same encoder/runtime interface. Same bridge surface also fills fixed-shape Group C search/belief planes from live mixture/search/risk context when present, else zero-filled planes plus presence masks. This file records runtime reality only; promoted sequencing/doctrine still lives in `research/design/HYDRA_FINAL.md` and `research/design/HYDRA_RECONCILIATION.md`.

## Testing

Every module in `hydra-core` has inline unit tests (`#[cfg(test)]` modules). Beyond unit tests, `tests/` dir contains integration tests:

| Test File | What It Covers |
|-----------|---------------|
| `golden_encoder.rs` | Encoder regression tests. Compares output against saved golden snapshots. Catches silent encoding drift when channel logic changes. |
| `mjai_replay.rs` | Replays recorded MJAI game logs through engine and verifies game state, actions, observations match expected sequence. Current regression surface explicitly covers replay round-reset correctness + kan-action legality matching so MJAI replay stays aligned with runtime legality checks. |
| `proptest_invariants.rs` | Property-based tests using `proptest`. Generates random game states, verifies invariants: legal mask consistency, encoder channel bounds, tile count conservation, action round-trip fidelity. |
| `game_loop_integration.rs` | End-to-end game loop tests. Runs complete games with `FirstActionSelector`, verifies termination, score consistency, result collection. |

Current replay-status note: after fixing MJAI replay round-start reset semantics + kan replay matching in vendored engine layer, Hydra MJAI loader was re-audited against Tenhou Houou 2025 corpus (`178,897` files) with `0` skips.

### Benchmarks

`benches/` dir uses `criterion` for perf benchmarks:

- `single_game`: time to run one complete game start to finish
- `batch_100`: time to run 100 games in parallel with `BatchSimulator`
- `encode_observation_1000x`: time to encode 1,000 observations (measures encoder throughput)

Run benchmarks with `cargo bench`.

## Dependencies

### Runtime

| Crate | Purpose |
|-------|---------|
| `riichienv-core` | Game engine (rules, state, legality) |
| `rayon` | Work-stealing thread pool for parallel batch sim |
| `serde` | Serialization for configs, game results, replay data |
| `ndarray` | N-dimensional array ops for observation tensors |
| `serde_json` | JSON serialization for MJAI protocol data |
| `chacha20` | ChaCha20 cipher (pinned version for determinism) |
| `rand` | RNG traits + distributions |
| `rand_chacha` | `ChaCha8Rng` for deterministic seeding |
| `sha2` | SHA-256 hashing for seed key derivation |
| `anyhow` | App-level error handling |
| `thiserror` | Derive macro for library error enums |

### Dev / Test

| Crate | Purpose |
|-------|---------|
| `proptest` | Property-based testing framework |
| `criterion` | Benchmarking framework |

## License

`hydra-core` is BSL-1.1 (see [hydra-core/LICENSE](../crates/hydra-core/LICENSE)). `hydra-engine` is Apache-2.0 (vendored upstream). All dependencies use MIT, Apache-2.0, or BSD-compatible licenses.