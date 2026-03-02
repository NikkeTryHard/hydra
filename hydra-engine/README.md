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
- Original license preserved (Apache-2.0). Hydra-specific additions (`ObservationRef`, `MjaiEvent`, `step_unchecked`, etc.) are MIT-licensed.

## Hydra Modifications

Changes from upstream `riichienv-core`, all targeting training throughput:

| Area | Change | Rationale |
|------|--------|-----------|
| Action type | `consume_tiles`: `Vec<u8>` -> `[u8; 4]` | `Action` is now `Copy`, zero heap allocation |
| HandEvaluator | `new()` takes `&[u8]` + `&[Meld]` (borrows) | Eliminates 30+ `clone()` calls per step |
| GameState | `step_unchecked()` | Skips redundant validation in self-play loops |
| GameState | `observe()` -> `ObservationRef` | Zero-copy state access, no `Vec` allocations |
| GameState | `get_legal_actions()` | Legal actions without full `Observation` construction |
| Hand maintenance | `sorted_insert()` | O(n) insert vs O(n log n) sort after every draw |
| Player data | `HashMap` -> `[Option<T>; 4]` | Direct array indexing for 4-player data |
| MJAI logging | `MjaiEvent` typed enum + `mjai_event!` macro | Zero-cost when disabled, stack-allocated when enabled |
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

## License

Apache-2.0 (original `riichienv-core` license). See the LICENSE file.
Hydra-specific additions (`ObservationRef`, `MjaiEvent`, `step_unchecked`, etc.) are MIT-licensed.
