# hydra-core

Core game/runtime crate for the Hydra Riichi Mahjong AI. It sits on top of `hydra-engine` / `riichienv-core` and provides the engine-side pieces used by both training and inference: observation encoding, safety analysis, search/belief feature bridging, deterministic seeding, and batch simulation.

## Overview

`hydra-core` transforms raw game states into neural network inputs. The current live encoder is a **fixed-superset 192x34 observation tensor**:

- channels `0..84`: baseline public + safety planes
- channels `85..149`: Group C search/belief context + presence masks + reserved slots
- channels `150..191`: Group D Hand-EV context + presence mask

The old `85x34` view is still useful as the baseline prefix, but it is no longer the full current encoder reality.

This crate also provides the batch simulation pipeline: run thousands of games in parallel via rayon, encode observations on the fly, and feed them directly into the training loop.

## Module Reference

| Module | Description |
|--------|-------------|
| `encoder` | 192x34 fixed-superset observation encoder; first 85 channels preserve the baseline public+safety planes |
| `bridge` | Converts `hydra-engine` `Observation`/`ObservationRef` into encoder input types |
| `safety` | Genbutsu, suji, kabe, one-chance safety calculations for the 23 safety channels (62-84) |
| `game_loop` | `GameRunner` with proper phase handling, `ActionSelector` trait, `FirstActionSelector` |
| `simulator` | Batch game simulation with rayon parallelism and configurable thread pools |
| `batch_encoder` | Pre-allocated contiguous buffer for encoding N observations without per-obs allocation |
| `shanten_batch` | Batch shanten computation with hierarchical hash caching (base + all 34 discards in one pass) |
| `seeding` | Deterministic RNG hierarchy: session seeds, per-game derivation, vendored Fisher-Yates shuffle |
| `tile` | 34-tile type system, aka-dora handling, 136-format conversion, suit permutation for data augmentation |
| `action` | 46-action space mapping (Mortal-compatible): discard, riichi, chi, pon, kan, pass, tsumo, ron |

## Observation Tensor (192x34 fixed superset)

The encoder produces an `[f32; 192 * 34]` flat array (row-major) with channels grouped:

| Channels | Content |
|----------|---------|
| 0-3 | Closed hand (thresholded tile counts) |
| 4-7 | Open meld hand counts |
| 8 | Drawn tile one-hot |
| 9-10 | Shanten masks (keep / next) |
| 11-22 | Discards per player (presence, tedashi, temporal) |
| 23-34 | Melds per player (chi, pon, kan) |
| 35-39 | Dora indicator thermometer |
| 40-42 | Aka dora flags (per suit plane) |
| 43-61 | Game metadata (riichi, scores, gaps, shanten, round, honba, kyotaku) |
| 62-84 | Safety channels (genbutsu, suji, kabe, one-chance, tenpai) |
| 85-149 | Search/belief context, mixture stats, delta-Q, opponent risk/stress, presence masks, reserved slots |
| 150-191 | Hand-EV context (tenpai / win / expected score / ukeire) plus presence mask |

If you need the active architecture/plan rather than the crate-local runtime summary, read:
- `research/design/HYDRA_FINAL.md`
- `research/design/HYDRA_RECONCILIATION.md`

## Benchmarks

Measured on Intel Core Ultra 7 265KF, Criterion median, `RAYON_NUM_THREADS=4`.
Full methodology in [research/ENGINE_BENCHMARKS.md](../../research/ENGINE_BENCHMARKS.md).

| Benchmark | Time |
|-----------|------|
| Single game (FirstActionSelector) | 396us |
| Batch 100 games (4 cores, rayon) | 3.5ms (28,986 games/sec) |
| Observation encode (baseline prefix + fixed superset write) | 422ns |

## License

Business Source License 1.1 (BSL). See [LICENSE](LICENSE) for full terms.

- Free for personal, non-commercial, and academic use
- Commercial mahjong AI services require a paid license from the Licensor
- Converts to Apache-2.0 on 2031-03-02

For commercial licensing inquiries, contact Sho Kaneko.
