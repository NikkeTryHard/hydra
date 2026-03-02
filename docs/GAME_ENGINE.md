# Hydra Game Engine (hydra-core)

Reference documentation for the `hydra-core` Rust crate, the game engine powering the Hydra Riichi Mahjong AI.

## Overview

`hydra-core` is a Rust library that provides everything the Hydra training pipeline needs from the game side: a complete Riichi Mahjong simulator, observation encoding, safety analysis, and batch execution. It wraps `riichienv-core` as the underlying game engine and layers Hydra-specific encoding, seeding, and orchestration on top.

Core responsibilities:

- Tile representation and suit permutation for data augmentation
- A 46-action space with bidirectional conversion to/from `riichienv` actions
- An 85-channel x 34-tile observation encoder with incremental dirty-flag updates
- Tile safety analysis (genbutsu, suji, kabe, one-chance)
- Deterministic seeding via SHA-256 KDF + ChaCha8Rng
- Parallel batch simulation with `rayon`
- A game loop abstraction with pluggable action selection

Hydra uses a 100% Rust stack (see `research/RUST_STACK.md`). The training pipeline (hydra-train, using Burn framework) consumes hydra-core directly -- same process, same memory, zero IPC.

## Foundation: RiichiEnv

The game engine is built on top of [smly/RiichiEnv](https://github.com/smly/RiichiEnv) (`riichienv-core` crate, Apache-2.0 license).

RiichiEnv provides:

- Full 4-player and 3-player Riichi Mahjong rules
- Red dora (aka-dora) support for all three suits
- All kan types: ankan (closed), daiminkan (open), shouminkan (added)
- Native MJAI protocol compatibility for game state representation
- Correctness verified by running MortalAgent (AGPL, used as a black-box MJAI player -- no code shared) over 1M+ hanchan without errors ([source: RiichiEnv README](https://github.com/smly/RiichiEnv#-features))

Hydra treats `riichienv-core` as a black-box game engine. All game state progression, legality checks, and rule enforcement happen inside RiichiEnv. Hydra's own code handles encoding, analysis, and orchestration only.

Because riichienv-core's correctness is already verified upstream -- smly ran Mortal as a black-box MJAI player (separate process, no linking) over 1M+ hanchan on RiichiEnv with zero errors ([source](https://github.com/smly/RiichiEnv)) -- Hydra does not need its own cross-engine validation. The correctness guarantee is inherited through the dependency. No Mortal code exists in RiichiEnv or Hydra.

## Module Reference

| Module | File | Description |
|--------|------|-------------|
| `tile` | `tile.rs` | Tile types (0-33), 136-format representation, aka-dora handling, suit permutation |
| `action` | `action.rs` | 46-action space, `HydraAction` enum, bidirectional riichienv conversion, legal mask builder |
| `encoder` | `encoder.rs` | 85x34 observation tensor, `ObservationEncoder`, incremental encoding with `DirtyFlags` |
| `safety` | `safety.rs` | `SafetyInfo` per-opponent tile safety: genbutsu, suji, kabe, one-chance |
| `simulator` | `simulator.rs` | `BatchSimulator` with rayon thread pool, `BatchConfig`, `GameResult` collection |
| `seeding` | `seeding.rs` | SHA-256 KDF, `SessionRng`, deterministic wall generation, Fisher-Yates shuffle |
| `bridge` | `bridge.rs` | Converts riichienv `Observation` into encoder-ready data via `extract_*` functions |
| `game_loop` | `game_loop.rs` | `GameRunner`, `ActionSelector` trait, step-by-step or run-to-completion execution |
| `batch_encoder` | `batch_encoder.rs` | Pre-allocated contiguous buffer for encoding N observations without per-obs allocation |
| `shanten_batch` | `shanten_batch.rs` | Batch shanten with hierarchical hash caching (base + all 34 discards in one pass) |


## Tile System (`tile.rs`)

### TileType

All tiles use a `TileType(u8)` newtype representing the 34 distinct Mahjong tile kinds:

| Range | Tiles | Count |
|-------|-------|-------|
| 0-8 | 1m through 9m (manzu/characters) | 9 |
| 9-17 | 1p through 9p (pinzu/circles) | 9 |
| 18-26 | 1s through 9s (souzu/bamboo) | 9 |
| 27-33 | East, South, West, North, Haku, Hatsu, Chun | 7 |

The physical game uses 136 tiles (4 copies of each type). The 136-format index identifies a specific physical tile, while `TileType` identifies its kind. Converting between them is a simple `tile136 / 4` truncation.

### Aka-Dora (Red Fives)

Three tiles in the 136-format set are designated red dora (aka-dora):

- Red 5m (manzu)
- Red 5p (pinzu)
- Red 5s (souzu)

These are the 0th copy (index 0 within each group of 4) of the respective 5-tiles: 136-format indices 16 (5m), 52 (5p), 88 (5s). Extended tile type indices 34-36 represent aka variants in the action space. The encoder and action space both handle aka-dora as distinct from regular fives where needed.

### Suit Permutation

For data augmentation during training, `tile.rs` provides suit permutation functions. There are 6 permutations of the three numbered suits (manzu, pinzu, souzu), leaving honor tiles untouched. Given a permutation index (0-5), the module remaps all tile types in a hand/observation to the permuted suit assignment. This 6x data augmentation helps the model learn suit-invariant patterns.

## Action Space (`action.rs`)

### 46-Action Space

Hydra uses a fixed 46-action output space. Every decision point in the game maps to one of these action indices:

| Index | Action | Notes |
|-------|--------|-------|
| 0-33 | Discard tile type 0-33 | Standard discard (non-red) |
| 34-36 | Discard aka 5m, 5p, 5s | Discard a specific red five |
| 37 | Declare riichi | Announces riichi; tile selection follows |
| 38-40 | Chi (3 variants) | Left/middle/right chi calls |
| 41 | Pon | Open pon call |
| 42 | Kan | Any kan type (ankan, daiminkan, shouminkan) |
| 43 | Agari | Win declaration (tsumo or ron) |
| 44 | Ryuukyoku | Abortive draw declaration (kyuushu kyuuhai, etc.) |
| 45 | Pass | Decline a call opportunity |

### Two-Phase Actions

Riichi and kan use a two-phase selection process. The model first outputs a phase-1 action (index 37 for riichi, 42 for kan). Then the game engine presents the legal tile choices and the model picks which specific tile to discard (riichi) or which specific kan to declare. This keeps the action space compact at 46 while supporting the full combinatorial range.

### HydraAction

`HydraAction` is a validated newtype wrapper around `u8`:

```rust
pub struct HydraAction(u8);
```

It validates the index is in range 0-45 on construction via `HydraAction::new(id) -> Option<Self>`. Methods like `is_discard()`, `is_aka_discard()`, and `discard_tile_type()` provide type-safe access. Bidirectional conversion functions `hydra_to_riichienv()` and `riichienv_to_hydra()` translate between Hydra's compact action space and riichienv-core's `Action` struct, using a `GameContext` to resolve context-dependent actions (chi consume tiles, tsumo vs ron, kan type).

### Legal Action Mask

The `build_legal_action_mask` function takes the current riichienv game state and returns a `[bool; 46]` array. Each slot is `true` if that action is legal in the current state. The training pipeline uses this mask to zero out illegal actions before softmax, guaranteeing the model never selects an impossible move.

## Observation Encoder (`encoder.rs`)

### Tensor Shape

Each observation is an `85 x 34` float tensor (2,890 values). The first axis represents 85 channels of information. The second axis represents the 34 tile types. This shape feeds directly into the SE-ResNet model input.

### Channel Layout

The 85 channels break down into these groups:

| Channels | Name | Encoding |
|----------|------|----------|
| 0-3 | Closed hand | Thresholded: ch N is 1.0 if tile count >= N+1 |
| 4-7 | Open meld hand | Same thresholding for tiles exposed in open melds |
| 8 | Drawn tile | One-hot: 1.0 at the tile type just drawn (tsumo only) |
| 9-10 | Shanten masks | Ch 9: keep-shanten (tiles whose discard does not increase shanten). Ch 10: next-shanten (tiles whose discard decreases shanten) |
| 11-13 | Player 0 discards | Presence (1.0 if discarded), tedashi flag (1.0 if from hand, not tsumogiri), temporal weight (exp(-0.2 * age)) |
| 14-16 | Player 1 discards | Same three channels, relative to seat |
| 17-19 | Player 2 discards | Same three channels, relative to seat |
| 20-22 | Player 3 discards | Same three channels, relative to seat |
| 23-25 | Player 0 melds | Chi (1.0 for tiles in chi melds), pon (tiles in pon), kan (tiles in kan) |
| 26-28 | Player 1 melds | Same three channels |
| 29-31 | Player 2 melds | Same three channels |
| 32-34 | Player 3 melds | Same three channels |
| 35-39 | Dora indicators | Thermometer encoding: ch N is 1.0 if N+1 or more dora indicators revealed |
| 40-42 | Aka dora flags | Per-suit plane: ch 40 = manzu red five, ch 41 = pinzu, ch 42 = souzu. 1.0 at the 5-tile column if that red five is visible |
| 43-46 | Riichi flags | One channel per player. Entire plane is 1.0 if that player has declared riichi |
| 47-50 | Scores | One channel per player. Entire plane filled with score / 100,000 |
| 51-54 | Relative score gaps | One channel per player. Filled with (player_score - my_score) / 30,000 |
| 55-58 | Shanten one-hot | Ch 55 = tenpai (shanten 0), ch 56 = iishanten (1), ch 57 = ryanshanten (2), ch 58 = 3+ shanten. Entire plane is 1.0 for the matching shanten count |
| 59 | Round number | Entire plane filled with kyoku / 8.0 (normalized round index) |
| 60 | Honba count | Entire plane filled with honba / 10.0 |
| 61 | Kyotaku (riichi sticks) | Entire plane filled with kyotaku / 10.0 |
| 62-84 | Safety channels | 23 channels of per-opponent tile safety data (see Safety System section) |

**Safety channel breakdown (channels 62-84):**

| Channels | Name |
|----------|------|
| 62-64 | Genbutsu (all): 1.0 for tiles each opponent discarded (one ch per opponent) |
| 65-67 | Genbutsu (tedashi): restricted to tiles discarded from hand (not tsumogiri) |
| 68-70 | Genbutsu (riichi-era): restricted to tiles discarded after opponent's riichi |
| 71-73 | Suji: float 0.0-1.0 for suji-inferred safety against each opponent |
| 74-79 | Reserved suji context (zeros) |
| 80 | Kabe: 1.0 for tiles with all 4 copies visible (global, not per-opponent) |
| 81 | One-chance: 1.0 for tiles where exactly 3 of 4 copies are visible |
| 82-84 | Reserved for tenpai hints (zeros until opponent modeling heads are trained) |

### ObservationEncoder

`ObservationEncoder` is the main struct for building observation tensors. It holds a pre-allocated `[f32; 85 * 34]` buffer marked `#[repr(C)]` for predictable memory layout.

```rust
#[repr(C)]
    pub struct ObservationEncoder {
    buffer: [f32; 2890],  // 85 channels x 34 tiles, row-major
}
```

### Incremental Encoding with DirtyFlags

`DirtyFlags` is a bitflags struct where each bit corresponds to a channel group (hand, discards, melds, dora, scores, safety, etc.). When the game state changes, only the relevant flags are set. On the next `encode()` call, only flagged channel groups are recomputed. Unchanged channels keep their previous values in the buffer.

This matters for performance: a single discard only dirties the discard and safety channels, skipping the more expensive hand/meld/dora re-encoding. During batch simulation of thousands of games, these savings compound.

## Safety System (`safety.rs`)

The safety module computes per-opponent, per-tile safety information used to populate encoder channels 62-84 and to inform defensive play decisions.

### SafetyInfo

`SafetyInfo` holds safety data from one player's perspective against all 3 opponents:

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

**Genbutsu** (safe tiles) tracks tiles that a specific opponent cannot ron:

- `genbutsu_all`: any tile the opponent discarded (always safe against that player)
- `genbutsu_tedashi`: only tiles discarded from the opponent's hand (not tsumogiri), indicating intentional discards
- `genbutsu_riichi_era`: only tiles discarded after the opponent declared riichi, relevant for reading post-riichi waits

**Suji** inference identifies tiles protected by the 1-4-7 / 2-5-8 / 3-6-9 suji relationship. If an opponent discarded a 4m, then 1m and 7m get suji safety (float 1.0) against that opponent. Suji only applies to suited tiles (indices 0-26); honors have no suji. Values update incrementally as new discards appear.

**Kabe** (wall block) marks tiles where all 4 copies are accounted for in visible information (discards, melds, own hand). A tile with all copies visible can't be part of any opponent's winning hand.

**One-chance** marks tiles where exactly 3 of 4 copies are visible, meaning only one unknown copy remains. These tiles carry reduced but nonzero risk.

All safety arrays update incrementally. When a new discard or meld occurs, only the affected opponent's `SafetyInfo` is recomputed.

## Batch Simulator (`simulator.rs`)

### BatchSimulator

`BatchSimulator` runs many games in parallel using a `rayon::ThreadPool`. Each game runs on its own thread with no shared mutable state between games.

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

Each game derives its seed as `base_seed + game_index`. Two runs with the same `BatchConfig` produce identical results regardless of thread scheduling.

### GameResult

`GameResult` collects the outcome of a single game: final scores for all four players, rounds played, total actions taken, and the seed used. The batch simulator returns a `Vec<GameResult>`.

### Convenience Function

`run_batch_simple` is a free function that uses rayon's global thread pool instead of a dedicated one. It's the easiest entry point for scripts and benchmarks that don't need custom thread pool configuration.

### Planned: Pre-Allocated Game Pools

Currently each game in a batch allocates a fresh `GameState`. A future optimization is to maintain a pool of pre-allocated game states that get recycled between batches, eliminating per-game allocation overhead during high-throughput self-play.

## Seeding (`seeding.rs`)

Deterministic seeding is critical for reproducible training and evaluation. The seeding module provides a hierarchical RNG system.

### Key Derivation

The session seed is a `[u8; 32]` byte array. `SessionRng` derives per-game seeds via `SHA-256(session_seed || game_index_le_bytes)`. The `derive_kyoku_seed` function further derives per-round seeds: `SHA-256(session_seed || nonce || kyoku || honba)`.

```
game_seed = SHA-256(session_seed || game_index)[0..32]
```

This ensures every game in a batch gets a unique, deterministic seed derived from the single session seed. Changing the session seed changes all games. Changing the game index changes only that game.

### SessionRng

`SessionRng` holds a 32-byte seed and an auto-incrementing game index counter. Each call to `next_game_seed()` derives a new 32-byte seed and advances the counter. This gives 2^64 independent game seeds from a single session seed.

### Wall Generation

`generate_wall` takes a session seed, nonce, kyoku number, and honba count. It derives a kyoku-specific seed, seeds a fresh `ChaCha8Rng`, initializes a sorted `[0..135]` wall, and applies a vendored Fisher-Yates shuffle. The vendored implementation avoids dependence on `rand::seq::SliceRandom` internals that might change between rand versions.

### Determinism Guarantees

Given the same session seed and batch config, `hydra-core` produces bit-identical results across:

- Different runs on the same machine
- Different thread counts (rayon scheduling is deterministic per-game)
- Different platforms (x86_64, aarch64) thanks to the vendored shuffle

The only requirement is the same Rust toolchain version, since floating-point encoder output depends on compiler codegen.

## Game Loop (`game_loop.rs`)

### GameRunner

`GameRunner` orchestrates a single game from start to finish. It holds the riichienv `GameState`, a `[SafetyInfo; 4]` array (one per player perspective), and action/round counters.

The runner exposes two execution modes:

- `step_once(selector)`: advance the game by one step using the provided `ActionSelector`. Handles round transitions (auto-resets safety), WaitAct vs WaitResponse phases. Returns `false` when the game is over.
- `run_to_completion(selector)`: play an entire game by calling `step_once` in a loop. Provides accessor methods for `scores()`, `total_actions()`, `rounds_played()`, and `safety(player)` after completion.

### ActionSelector Trait

```rust
pub trait ActionSelector {
    fn select_action(&mut self, player: u8, legal_actions: &[Action]) -> Action;
}
```

Any type implementing `ActionSelector` can drive the game loop. `FirstActionSelector` is a simple built-in that picks the first legal action (useful for testing and benchmarks). The training pipeline provides its own selectors that call the neural network.

### Safety Tracking

During play, `GameRunner` maintains a `[SafetyInfo; 4]` array (one per player perspective). After every discard, call, and riichi event, the runner's `track_action` method incrementally updates the relevant safety data across all perspectives. These feed directly into the encoder's safety channels (62-84) on the next observation request.

## Bridge (`bridge.rs`)

The bridge module converts riichienv's `Observation` struct into the data the encoder needs. It acts as a translation layer so the encoder doesn't depend on riichienv types directly.

### Extract Functions

Each `extract_*` function pulls one category of data from the riichienv observation:

- `extract_hand()`: closed hand tile counts and open meld tile counts
- `extract_discards()`: per-player discard sequences with tedashi and temporal info
- `extract_melds()`: per-player meld data (chi/pon/kan tile lists)
- `extract_dora()`: dora indicator tiles and aka-dora visibility
- `extract_metadata()`: scores, round number, honba, kyotaku, riichi states, shanten

### Entry Point

`encode_observation` is the main bridge function. It takes a riichienv `Observation`, calls all `extract_*` functions, and feeds the results into the `ObservationEncoder`. Returns the filled 85x34 float buffer ready for the model.

## Testing

Every module in `hydra-core` has inline unit tests (`#[cfg(test)]` modules). Beyond unit tests, the `tests/` directory contains integration tests:

| Test File | What It Covers |
|-----------|---------------|
| `golden_encoder.rs` | Regression tests for the encoder. Compares encoder output against saved golden snapshots. Catches silent encoding drift when any channel logic changes. |
| `mjai_replay.rs` | Replays recorded MJAI game logs through the engine and verifies that game state, actions, and observations match the expected sequence. |
| `proptest_invariants.rs` | Property-based tests using `proptest`. Generates random game states and verifies invariants: legal mask consistency, encoder channel bounds, tile count conservation, action round-trip fidelity. |
| `game_loop_integration.rs` | End-to-end game loop tests. Runs complete games with `FirstActionSelector` and verifies termination, score consistency, and result collection. |

### Benchmarks

The `benches/` directory uses `criterion` for performance benchmarks:

- `single_game`: time to run one complete game from start to finish
- `batch_100`: time to run 100 games in parallel with `BatchSimulator`
- `encode_observation_1000x`: time to encode 1,000 observations (measuring encoder throughput)

Run benchmarks with `cargo bench`.

## Dependencies

### Runtime

| Crate | Purpose |
|-------|---------|
| `riichienv-core` | Game engine (rules, state, legality) |
| `rayon` | Work-stealing thread pool for parallel batch simulation |
| `serde` | Serialization for configs, game results, replay data |
| `ndarray` | N-dimensional array operations for observation tensors |
| `serde_json` | JSON serialization for MJAI protocol data |
| `chacha20` | ChaCha20 cipher (pinned version for determinism) |
| `rand` | RNG traits and distributions |
| `rand_chacha` | ChaCha8Rng for deterministic seeding |
| `sha2` | SHA-256 hashing for seed key derivation |
| `anyhow` | Application-level error handling |
| `thiserror` | Derive macro for library error enums |

### Dev / Test

| Crate | Purpose |
|-------|---------|
| `proptest` | Property-based testing framework |
| `criterion` | Benchmarking framework |

## License

hydra-core is BSL-1.1 (see [hydra-core/LICENSE](../hydra-core/LICENSE)). hydra-engine is Apache-2.0 (vendored upstream). All dependencies use MIT, Apache-2.0, or BSD-compatible licenses.
