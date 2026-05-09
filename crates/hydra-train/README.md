# hydra-train

Crate-local map for Hydra training code. Not operator manual.

Operator docs:
- training modes/YAML and preflight/runtime authority: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md)
- current shipped/staged status: [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md)
- container, GHCR, Kaggle artifact, MJAI audit, coverage commands: [`docker/train/README.md`](../../docker/train/README.md)

## Owns

`hydra-train` sits above `hydra-core` and `hydra-engine`.

Owns:
- model/backbone/head defs
- losses and BC/RL optimize loops
- replay data loading, sample collation, BC shard consumption/build
- self-play batch gen and eval harnesses
- preflight/runtime autotune and resume-compat checks
- replay sidecar generation for ExIt / DeltaQ-style lanes
- training/data binaries listed below

Does not own:
- Riichi rules, scoring, legal actions, replay parsing core: `hydra-engine`
- runtime bridge, encoder, simulation, seeding, search/runtime feature plumbing: `hydra-core`
- operator truth for run commands: docs linked above

If rule/runtime semantics drift, code plus [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](../../docs/COMPATIBILITY_SURFACE.md) win.

## Current contract snapshot

- live encoder/model contract: `192x34`
- action space: 46 actions
- replay input support: loose MJAI dirs plus `.tar.zst` archives
- shipped baseline includes replay-derived `safety_residual`, stronger public-teacher belief semantics, ExIt carrier across live self-play and replay/sample sidecar-first lanes
- DeltaQ tooling exists but remains promotion-gated, not default-on

Read [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md) before treating staged code as live baseline.

## Entry binaries

| Binary | Purpose |
|---|---|
| `train` | main entrypoint: normal train, preflight, probe-only, DeltaQ promotion |
| `mjai_audit` | replay corpus/archive audit, failure buckets, optional failure inventories |
| `mjai_first_failure` | first bad MJAI entry in archive |
| `mjai_debug_failure` | detailed single-replay failure report |
| `build_bc_shards` | build BC shard datasets/manifests from replay corpora |
| `build_replay_exit_sidecar` | build replay-side ExIt sidecars |
| `build_replay_delta_q_sidecar` | build replay-side DeltaQ sidecars |
| `build_parsed_sample_cache` | build parsed sample cache artifact |
| `recompress` | replay/data artifact recompression util |
| `repack_tar` | tar replay corpus repack util |

Main entrypoint: [`src/bin/train.rs`](src/bin/train.rs). Train submodules under `src/bin/train/` own runtime/preflight selection, probe transport, autotune, resume/state persistence, and test support.

## Module map

| Module | Owns |
|---|---|
| `amp` | AMP/BF16 runtime helpers |
| `backbone` | network backbone blocks |
| `config` | shared train/runtime config parsing/types |
| `data` | replay scan/load, augmentation, batch/sample plumbing |
| `eval` | arena/eval helpers and metric summaries |
| `heads` | policy/value/aux heads |
| `inference` | train-side model inference helpers |
| `model` | `HydraModel` assembly and config surface |
| `preflight` | probe/preflight config and runtime selection helpers |
| `saf` | SAF-related train helpers |
| `selfplay` | self-play orchestration and mixed-policy execution |
| `selfplay_batch` | batched self-play data plumbing |
| `teacher` | teacher features/labels, incl belief surfaces |
| `training` | BC, RL, ACH, DRDA, ExIt, DeltaQ gates/orchestrators |

## License

Business Source License 1.1 (BSL). See repo-root [LICENSE](../../LICENSE).

- Free for personal, non-commercial, and academic use
- Commercial mahjong AI services require paid license from Licensor
- Converts to Apache-2.0 on 2031-03-02

Commercial licensing: Sho Kaneko.