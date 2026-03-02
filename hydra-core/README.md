# hydra-core

Training-layer crate for the Hydra Riichi Mahjong AI. Sits on top of `hydra-engine` and provides everything needed to generate training data: observation encoding, batch simulation, safety analysis, and deterministic seeding.

## Overview

`hydra-core` transforms raw game states from `hydra-engine` into neural network inputs. The 85x34 observation tensor encodes the full player-visible game state (hand, discards, melds, dora, scores, safety) into the format consumed by Hydra's SE-ResNet model.

This crate also provides the batch simulation pipeline: run thousands of games in parallel via rayon, encode observations on the fly, and feed them directly into the training loop.

## Module Reference

| Module | Description |
|--------|-------------|
| `encoder` | 85x34 observation tensor encoder: hand, discards, melds, dora, metadata, safety channels |
| `bridge` | Converts `hydra-engine` `Observation`/`ObservationRef` into encoder input types |
| `safety` | Genbutsu, suji, kabe, one-chance safety calculations for the 23 safety channels (62-84) |
| `game_loop` | `GameRunner` with proper phase handling, `ActionSelector` trait, `FirstActionSelector` |
| `simulator` | Batch game simulation with rayon parallelism and configurable thread pools |
| `batch_encoder` | Pre-allocated contiguous buffer for encoding N observations without per-obs allocation |
| `shanten_batch` | Batch shanten computation with hierarchical hash caching (base + all 34 discards in one pass) |
| `seeding` | Deterministic RNG hierarchy: session seeds, per-game derivation, vendored Fisher-Yates shuffle |
| `tile` | 34-tile type system, aka-dora handling, 136-format conversion, suit permutation for data augmentation |
| `action` | 46-action space mapping (Mortal-compatible): discard, riichi, chi, pon, kan, pass, tsumo, ron |

## Observation Tensor (85x34)

The encoder produces an `[f32; 85 * 34]` flat array (row-major) with channels grouped:

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

## Benchmarks

Measured on Intel Core Ultra 7 265KF, Criterion median, `RAYON_NUM_THREADS=4`.
Full methodology in [research/ENGINE_BENCHMARKS.md](../research/ENGINE_BENCHMARKS.md).

| Benchmark | Time |
|-----------|------|
| Single game (FirstActionSelector) | 417us |
| Batch 100 games (4 cores, rayon) | 12.2ms (8,170 games/sec) |
| Observation encode (85x34) | 405ns |

## License

Business Source License 1.1 (BSL). See [LICENSE](LICENSE) for full terms.

- Free for personal, non-commercial, and academic use
- Commercial mahjong AI services require a paid license from the Licensor
- Converts to Apache-2.0 on 2031-03-02

For commercial licensing inquiries, contact Sho Kaneko.