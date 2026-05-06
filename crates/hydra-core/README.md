# hydra-core

Core game/runtime crate for Hydra Riichi Mahjong AI. Sits atop `hydra-engine` / `riichienv-core`. Provides engine-side parts for training + inference: observation encoding, safety analysis, search/belief feature bridge, deterministic seeding, batch simulation.

## Overview

`hydra-core` turns raw game states into neural net inputs. Current live encoder = **fixed-superset 192x34 observation tensor**:

- channels `0..84`: baseline public + safety planes
- channels `85..149`: Group C search/belief context + presence masks + reserved slots
- channels `150..191`: Group D Hand-EV context + presence mask

Old `85x34` view still useful as baseline prefix, but not full current encoder.

This crate also provides batch simulation pipeline: run thousands games in parallel via rayon, encode observations on fly, feed direct into training loop.

For full live runtime/channel contract, read [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md).
For compatibility-sensitive shape/runtime facts, read [`docs/COMPATIBILITY_SURFACE.md`](../../docs/COMPATIBILITY_SURFACE.md).

## Module Reference

| Module | Description |
|--------|-------------|
| `encoder` | 192x34 fixed-superset observation encoder; first 85 channels preserve baseline public+safety planes |
| `bridge` | Converts `hydra-engine` `Observation`/`ObservationRef` into encoder input types |
| `safety` | Genbutsu, suji, kabe, one-chance safety calcs for 23 safety channels (62-84) |
| `game_loop` | `GameRunner` with proper phase handling, `ActionSelector` trait, `FirstActionSelector` |
| `simulator` | Batch game simulation with rayon parallelism and configurable thread pools |
| `batch_encoder` | Pre-allocated contiguous buffer for encoding N observations without per-obs allocation |
| `shanten_batch` | Batch shanten computation with hierarchical hash caching (base + all 34 discards in one pass) |
| `seeding` | Deterministic RNG hierarchy: session seeds, per-game derivation, vendored Fisher-Yates shuffle |
| `tile` | 34-tile type system, aka-dora handling, 136-format conversion, suit permutation for data augmentation |
| `action` | 46-action space mapping (Mortal-compatible): discard, riichi, chi, pon, kan, pass, tsumo, ron |

## Observation Tensor (192x34 fixed superset)

Encoder produces `[f32; 192 * 34]` flat array (row-major) with three high-level regions:

- `0..84`: baseline public + safety prefix
- `85..149`: Group C search/belief context plus presence masks and reserved slots
- `150..191`: Group D Hand-EV context plus presence mask

For channel-by-channel breakdown and live runtime semantics, defer to `docs/GAME_ENGINE.md`.

## Benchmarks

Measured on Intel Core Ultra 7 265KF, Criterion median, `RAYON_NUM_THREADS=4`.
Full methodology in [research/infrastructure/ENGINE_BENCHMARKS.md](../../research/infrastructure/ENGINE_BENCHMARKS.md).

| Benchmark | Time |
|-----------|------|
| Single game (FirstActionSelector) | 396us |
| Batch 100 games (4 cores, rayon) | 3.5ms (28,986 games/sec) |
| Observation encode (baseline prefix + fixed superset write) | 422ns |

## License

Business Source License 1.1 (BSL). See [LICENSE](LICENSE) for full terms.

- Free for personal, non-commercial, academic use
- Commercial mahjong AI services require paid license from Licensor
- Converts to Apache-2.0 on 2031-03-02

For commercial licensing inquiries, contact Sho Kaneko.