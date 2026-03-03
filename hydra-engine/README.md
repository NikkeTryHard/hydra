# hydra-engine

Internal game engine for the Hydra Riichi Mahjong AI. Vendored from [smly/RiichiEnv](https://github.com/smly/RiichiEnv) (`riichienv-core` v0.3.4, Apache-2.0).

## Overview

`hydra-engine` provides the complete Riichi Mahjong simulation that Hydra's training pipeline runs on: game state management, hand evaluation, score calculation, and legal action generation for both 4-player and 3-player (sanma) variants.

The crate is vendored rather than used as an external dependency. Vendoring lets us make performance-critical modifications (zero-copy observations, stack-allocated actions, validation bypass for self-play) without waiting on upstream or maintaining a patch set. The lib name remains `riichienv_core` so that `hydra-core` imports don't need renaming.

This is a workspace-internal crate. It isn't published to crates.io.

## Origin and License

- Vendored from `riichienv-core` v0.3.4 by [smly](https://github.com/smly) (Apache-2.0).
- Correctness verified upstream against 1M+ hanchan using Mortal as a black-box MJAI player, zero errors ([source](https://github.com/smly/RiichiEnv)).
- The lib name stays `riichienv_core` for backward compatibility with `hydra-core` imports.
- Original license preserved (Apache-2.0). Hydra-specific additions (`ObservationRef`, `MjaiEvent`, `step_unchecked`, etc.) are BSL-1.1-licensed (same as `hydra-core`).

## Hydra Modifications

Changes from upstream `riichienv-core`, all targeting training throughput:

| Area | Change | Rationale |
|------|--------|-----------|
| Action type | `consume_tiles`: `Vec<u8>` -> `[u8; 4]` | `Action` is now `Copy`, zero heap allocation |
| HandEvaluator | `new()` takes `&[u8]` + `&[Meld]` (borrows) | Eliminates 30+ `clone()` calls per step |
| GameState | `step_unchecked()` | Skips redundant validation in self-play loops |
| GameState | `_execute_step` deleted | Single step impl via `_execute_step_array`, -905 lines |
| GameState | Extracted handler methods | `_handle_discard/riichi/ankan/kakan/tsumo/wait_response` |
| GameState | `observe()` -> `ObservationRef` | Zero-copy state access, no `Vec` allocations |
| GameState | `get_legal_actions_into()` | Buffer-reuse legal actions, zero alloc per step |
| GameState | `_get_claim_actions_into_claims()` | Zero-alloc claim resolution, writes directly to array |
| Hand maintenance | `sorted_insert_arr()` | Fixed-array sorted insert for stack-allocated hands |
| Player data | All fields stack-allocated | hand/melds/discards/pao/forbidden as fixed arrays |
| Wall data | `[u8; 136]` + cursor | Fixed array wall, O(1) draw via cursor index |
| Meld type | `[u8; 4]` + `tile_count` | Meld is Copy, zero heap allocation |
| Action type | `[u8; 4]` + `consume_count` | Action is Copy, zero heap allocation |
| HandEvaluator | `[Meld; 4]` + `get_waits_u8_into()` | Stack melds, buffer-reuse waits |
| Safety tracking | `u64` bitfields | Genbutsu/kabe/one-chance as bitsets, not bool arrays |
| MJAI logging | Gated by `skip_mjai_logging` | Zero-cost when disabled |
| Shanten tables | Made `pub` for batch shanten | Enables hierarchical hash caching in `hydra-core` |

## Module Reference

| Module | Description |
|--------|-------------|
| `action` | Action types (`Discard`, `Chi`, `Pon`, `Kan`, `Riichi`, `Ron`, `Tsumo`, `Kita`, etc.) and game phase tracking |
| `state` | Full game state management, wall handling, legal action validation, `step_unchecked()` (4-player) |
| `state_3p` | Game state management for 3-player games with Kita/BaBei support |
| `game_variant` | `GameStateVariant` enum dispatching between 4-player and 3-player game states |
| `observation` | Player-facing game state views with legal actions and MJAI event history (4-player) |
| `observation_3p` | Player-facing game state views for 3-player games |
| `observation_ref` | `ObservationRef`: zero-copy, borrow-based state access (Hydra addition) |
| `hand_evaluator` | Agari detection, tenpai checking, wait calculation, riichi candidate analysis (4-player) |
| `hand_evaluator_3p` | Hand evaluation for 3-player games |
| `shanten` | Shanten number calculation with pub tables for external caching |
| `score` | Han/fu-based score calculation |
| `rule` | Game rule configuration with Tenhou/MJSoul presets (4-player and sanma) |
| `types` | Core data types: `Hand`, `Wind`, `Meld`, `MeldType`, `Conditions`, `WinResult` |
| `parser` | MPSZ notation parsing for tiles and hands |
| `mjai_event` | `MjaiEvent` typed enum and `mjai_event!` macro for zero-cost logging (Hydra addition) |
| `yaku` | Yaku (winning hand pattern) definitions and detection (4-player) |
| `agari` | Agari (winning hand) table lookups |
| `replay` | MJAI and MJSoul replay parsing with step-by-step iteration (requires `python` feature) |
| `errors` | Error types (`RiichiError`) and result alias (`RiichiResult`) |

## Tile Representation

- **136-format**: Each of 34 tile types x 4 copies (indices 0-135), used for actual game state.
- **34-format**: Normalized tile type indices (0-33), used for calculations.
- **MPSZ notation**: `1m`-`9m` (man), `1p`-`9p` (pin), `1s`-`9s` (sou), `1z`-`7z` (honors).
- Red fives (aka-dora) are at indices 16, 52, 88 in 136-format.

## Benchmarks

Measured on Intel Core Ultra 7 265KF, 20 cores, `RAYON_NUM_THREADS=4`.
Trivial agent (first legal action), Criterion median. Full methodology
in [research/ENGINE_BENCHMARKS.md](../research/ENGINE_BENCHMARKS.md).

| Benchmark | hydra-engine | riichienv-core 0.3.4 | Delta |
|-----------|-------------|---------------------|-------|
| Single game (1 core) | 396us | 933us | **2.36x faster** |
| Batch 100 (1 core, seq) | 45.1ms (2,217/sec) | 94.1ms (1,063/sec) | **2.09x faster** |
| Batch 100 (4 cores, rayon) | 3.5ms (28,986/sec) | 28.0ms (3,571/sec) | **8.0x faster** |
| Observation encode | 422ns | n/a | -- |

Cross-engine comparison (single-threaded, first-action agent unless noted):

| Engine | Language | Per-Game | Games/sec |
|--------|----------|----------|-----------|
| hydra-engine | Rust | 396us | 2,525 |
| riichienv-core | Rust | 933us | 1,072 |
| mahjax | JAX/Python | 873us | 1,145 |
| Mjx | C++ | 17,498us | 57 |
| Mjai | Ruby | 86,883us | 12 |


## License

Apache-2.0 (original `riichienv-core` license). See the LICENSE file.
Hydra-specific additions (`ObservationRef`, `MjaiEvent`, `step_unchecked`, etc.) are BSL-1.1-licensed (same as `hydra-core`).
