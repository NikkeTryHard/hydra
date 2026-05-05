# hydra-engine

Internal game engine for Hydra Riichi Mahjong AI. Vendored from [smly/RiichiEnv](https://github.com/smly/RiichiEnv) (`riichienv-core` v0.4.7, Apache-2.0).

## Overview

`hydra-engine` gives full Riichi Mahjong sim Hydra training pipeline runs on: game state mgmt, hand eval, score calc, legal action gen for 4p + 3p (sanma).

Crate vendored, not external dep. Why: perf-critical mods (zero-copy obs, stack-alloc actions, validation bypass for self-play) without waiting upstream or carrying patch set. Lib name stays `riichienv_core` so `hydra-core` imports need no rename.

Workspace-internal crate. Not published to crates.io.

## Origin and License

- Vendored from `riichienv-core` v0.4.7 by [smly](https://github.com/smly) (Apache-2.0).
- Correctness verified upstream against 1M+ hanchan using Mortal as black-box MJAI player, zero errors ([source](https://github.com/smly/RiichiEnv)).
- Lib name stays `riichienv_core` for backward compat with `hydra-core` imports.
- Original license preserved (Apache-2.0). Hydra-specific additions (`ObservationRef`, `MjaiEvent`, `step_unchecked`, etc.) use BSL-1.1 (same as `hydra-core`).

## Hydra Modifications

Changes from upstream `riichienv-core`, all for training throughput:

| Area | Change | Rationale |
|------|--------|-----------|
| Action type | `consume_tiles`: `Vec<u8>` -> `[u8; 4]` | `Action` now `Copy`, zero heap alloc |
| HandEvaluator | `new()` takes `&[u8]` + `&[Meld]` (borrows) | Removes 30+ `clone()` calls per step |
| GameState | `step_unchecked()` | Skips redundant validation in self-play loops |
| GameState | `_execute_step` deleted | Single step impl via `_execute_step_array`, -905 lines |
| GameState | Extracted handler methods | `_handle_discard/riichi/ankan/kakan/tsumo/wait_response` |
| GameState | `observe()` -> `ObservationRef` | Zero-copy state access, no `Vec` allocs |
| GameState | `get_legal_actions_into()` | Buffer-reuse legal actions, zero alloc per step |
| GameState | `_get_claim_actions_into_claims()` | Zero-alloc claim resolution, writes direct to array |
| Hand maintenance | `sorted_insert_arr()` | Fixed-array sorted insert for stack-alloc hands |
| Player data | All fields stack-allocated | hand/melds/discards/pao/forbidden as fixed arrays |
| Wall data | `[u8; 136]` + cursor | Fixed array wall, O(1) draw via cursor index |
| Meld type | `[u8; 4]` + `tile_count` | Meld is Copy, zero heap alloc |
| Action type | `[u8; 4]` + `consume_count` | Action is Copy, zero heap alloc |
| HandEvaluator | `[Meld; 4]` + `get_waits_u8_into()` | Stack melds, buffer-reuse waits |
| Safety tracking | `u64` bitfields | Genbutsu/kabe/one-chance as bitsets, not bool arrays |
| MJAI logging | Gated by `skip_mjai_logging` | Zero-cost when disabled |
| Shanten tables | Made `pub` for batch shanten | Enables hierarchical hash caching in `hydra-core` |

## Module Reference

| Module | Description |
|--------|-------------|
| `action` | Action types (`Discard`, `Chi`, `Pon`, `Kan`, `Riichi`, `Ron`, `Tsumo`, `Kita`, etc.) + game phase tracking |
| `state` | Full game state mgmt, wall handling, legal action validation, `step_unchecked()` (4p) |
| `state_3p` | Game state mgmt for 3p games with Kita/BaBei support |
| `game_variant` | `GameStateVariant` enum dispatch between 4p and 3p game states |
| `observation` | Player-facing game state views with legal actions + MJAI event history (4p) |
| `observation_3p` | Player-facing game state views for 3p games |
| `observation_ref` | `ObservationRef`: zero-copy, borrow-based state access (Hydra addition) |
| `hand_evaluator` | Agari detect, tenpai check, wait calc, riichi candidate analysis (4p) |
| `hand_evaluator_3p` | Hand eval for 3p games |
| `shanten` | Shanten calc with pub tables for external caching |
| `score` | Han/fu-based score calc |
| `rule` | Game rule config with Tenhou/MJSoul presets (4p + sanma) |
| `types` | Core data types: `Hand`, `Wind`, `Meld`, `MeldType`, `Conditions`, `WinResult` |
| `parser` | MPSZ notation parsing for tiles + hands |
| `mjai_event` | `MjaiEvent` typed enum + `mjai_event!` macro for zero-cost logging (Hydra addition) |
| `yaku` | Yaku (winning hand pattern) defs + detection (4p) |
| `agari` | Agari (winning hand) table lookups |
| `replay` | MJAI + MJSoul replay parsing with step-by-step iteration |
| `errors` | Error types (`RiichiError`) + result alias (`RiichiResult`) |

## Tile Representation

- **136-format**: Each of 34 tile types x 4 copies (indices 0-135), used for real game state.
- **34-format**: Normalized tile type indices (0-33), used for calc.
- **MPSZ notation**: `1m`-`9m` (man), `1p`-`9p` (pin), `1s`-`9s` (sou), `1z`-`7z` (honors).
- Red fives (aka-dora) at indices 16, 52, 88 in 136-format.

## Benchmarks

Measured on Intel Core Ultra 7 265KF, 20 cores, `RAYON_NUM_THREADS=4`.
Trivial agent (first legal action), Criterion median. Full methodology
in [research/infrastructure/ENGINE_BENCHMARKS.md](../../research/infrastructure/ENGINE_BENCHMARKS.md).

| Benchmark | hydra-engine | riichienv-core 0.4.7 | Delta |
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

Apache-2.0 (original `riichienv-core` license). See `LICENSE` file.
Hydra-specific additions (`ObservationRef`, `MjaiEvent`, `step_unchecked`, etc.) use BSL-1.1 (same as `hydra-core`).