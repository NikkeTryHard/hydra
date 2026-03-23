# hydra-train

Training crate for the Hydra Riichi Mahjong AI. It owns the model stack, replay/self-play data plumbing, target construction, evaluation harnesses, and the training/data utility binaries that turn `hydra-core` encoder/runtime signals into checkpoints or replay-side artifacts.

## Overview

`hydra-train` is the workspace layer that sits above `hydra-core` and `hydra-engine`.

- `hydra-engine` owns low-level Riichi rules and replay parsing
- `hydra-core` owns runtime bridging, encoding, simulation, seeding, and search/runtime feature plumbing
- `hydra-train` owns model definition, losses, BC/RL/self-play orchestration, sidecar generation, and training/evaluation utilities

The crate is built around Burn and the current Hydra training baseline. The shipped baseline already includes the live `192x34` encoder/model contract, replay-derived `safety_residual`, the stronger public-teacher belief semantics tranche, and the ExIt carrier across both live self-play and replay/sample sidecar-first lanes. Promotion-gated DeltaQ tooling also lives here, but it is not the default-on training lane.

For current shipped-vs-staged status, read [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md).
For active-path sequencing, read [`research/design/HYDRA_RECONCILIATION.md`](../../research/design/HYDRA_RECONCILIATION.md).

## What this crate owns

`hydra-train` is responsible for:

- model/backbone/head definitions
- BC and RL optimization loops
- replay data loading and sample collation
- self-play batch generation and evaluation harnesses
- preflight/runtime autotuning and resume compatibility checks
- replay sidecar generation for ExIt and DeltaQ-style lanes
- workspace binaries and utilities like `train`, `mjai_audit`, `recompress`, `repack_tar`, replay sidecar builders, and replay failure inspection tools

It does **not** own the Riichi rules engine itself. When rule semantics drift, `hydra-engine` and `docs/GAME_ENGINE.md` are the runtime authority.

## Module Reference

| Module | Description |
|--------|-------------|
| `backbone` | Backbone building blocks for Hydra's network stack |
| `config` | Shared training/runtime config types and parsing helpers |
| `data` | Replay loading, data-source scanning, augmentation, and batch/sample plumbing |
| `eval` | Arena/evaluation helpers and training/eval metric summaries |
| `heads` | Policy / value / auxiliary head definitions |
| `inference` | Train-side model inference helpers |
| `league` | League-style model coordination and related utilities |
| `model` | Top-level `HydraModel` assembly and config surface |
| `preflight` | Probe/preflight configuration for runtime selection and autotune flows |
| `saf` | SAF-related train-side helpers |
| `selfplay` | Self-play orchestration and mixed-policy game execution |
| `selfplay_batch` | Batched self-play data plumbing |
| `teacher` | Teacher-side feature/label helpers, including belief surfaces |
| `training` | BC, RL, ACH, DRDA, ExIt, DeltaQ promotion/validation, losses, gates, and orchestrators |

## Workspace binaries

The crate currently exposes these workspace binaries:

| Binary | Purpose |
|--------|---------|
| `train` | Main training entrypoint; supports normal training, preflight, probe, and DeltaQ-promotion modes |
| `mjai_audit` | Audits replay datasets and archives, including failure bucketing and optional failure inventories |
| `recompress` | Recompression utility for replay/data artifacts |
| `repack_tar` | Repack utility for tar-based replay corpora |
| `build_replay_delta_q_sidecar` | Builds replay-side DeltaQ sidecars |
| `build_replay_exit_sidecar` | Builds replay-side ExIt sidecars |
| `mjai_debug_failure` | Debug helper for replay failures |
| `mjai_first_failure` | Finds/inspects the first replay failure in a dataset |

The main training entrypoint lives at [`src/bin/train.rs`](src/bin/train.rs).

## Runtime and data contract

The training crate consumes the same live runtime surface as the rest of Hydra:

- encoder/model contract: `192x34`
- action space: 46 actions
- replay input support: flat MJAI directories plus `.tar.zst` archives
- default workspace test path: `cargo nextest run --release`

The Docker/container-facing training contract is documented in [`docker/train/README.md`](../../docker/train/README.md).

## Training flow at a glance

At a high level, `hydra-train` does four things:

1. reads config and chooses runtime/preflight behavior
2. loads replay or self-play data through `data::*`
3. builds targets/losses and runs BC/RL/update loops through `training::*`
4. evaluates, checkpoints, and reports metrics through the train binary and eval helpers

That split is intentional: runtime semantics stay below this crate, while optimization policy and target construction stay here.

## Where to read next

- Need runtime truth? Read [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](../../docs/COMPATIBILITY_SURFACE.md).
- Need current shipped/staged status? Read [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md).
- Need the active Hydra v1 roadmap? Read [`research/design/HYDRA_RECONCILIATION.md`](../../research/design/HYDRA_RECONCILIATION.md).
- Need container execution details? Read [`docker/train/README.md`](../../docker/train/README.md).

## License

Business Source License 1.1 (BSL). See the repo-root [LICENSE](../../LICENSE).

- Free for personal, non-commercial, and academic use
- Commercial mahjong AI services require a paid license from the Licensor
- Converts to Apache-2.0 on 2031-03-02

For commercial licensing inquiries, contact Sho Kaneko.
