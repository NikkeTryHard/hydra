# hydra-train

Training crate for Hydra Riichi Mahjong AI. Owns model stack, replay/self-play data plumbing, target build, eval harnesses, training/data bins turning `hydra-core` encoder/runtime signals into checkpoints or replay artifacts.

## Overview

`hydra-train` = workspace layer above `hydra-core` and `hydra-engine`.

- `hydra-engine` owns low-level Riichi rules and replay parsing
- `hydra-core` owns runtime bridge, encoding, simulation, seeding, and search/runtime feature plumbing
- `hydra-train` owns model defs, losses, active BC/RL/self-play orchestration, sidecar gen, and training/eval utils

Crate built around Burn and current Hydra training baseline. Shipped baseline already has live `192x34` encoder/model contract, replay-derived `safety_residual`, stronger public-teacher belief semantics tranche, and ExIt carrier across live self-play plus replay/sample sidecar-first lanes. Promotion-gated DeltaQ tooling also lives here, but not default-on training lane. Some internal modules stay for staged/reserve work, so use module table below as supported crate surface, not every file list.

For current shipped-vs-staged status, read [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md).
For active-path sequencing, read [`research/design/HYDRA_RECONCILIATION.md`](../../research/design/HYDRA_RECONCILIATION.md).

## What this crate owns

`hydra-train` responsible for:

- model/backbone/head defs
- BC and RL optimize loops
- replay data loading and sample collation
- self-play batch gen and eval harnesses
- preflight/runtime autotune and resume-compat checks
- replay sidecar gen for ExIt and DeltaQ-style lanes
- workspace bins and utils like `train`, `mjai_audit`, `recompress`, `repack_tar`, replay sidecar builders, and replay failure inspection tools

It does **not** own Riichi rules engine itself. If rule semantics drift, `hydra-engine` and `docs/GAME_ENGINE.md` are runtime authority.

## Module Reference

| Module | Description |
|--------|-------------|
| `amp` | AMP/BF16 runtime helpers shared by training flows |
| `backbone` | Backbone blocks for Hydra network stack |
| `config` | Shared training/runtime config types and parsing helpers |
| `data` | Replay loading, data-source scanning, augmentation, and batch/sample plumbing |
| `eval` | Arena/eval helpers and training/eval metric summaries |
| `heads` | Policy / value / aux head defs |
| `inference` | Train-side model inference helpers |
| `model` | Top-level `HydraModel` assembly and config surface |
| `preflight` | Probe/preflight config for runtime selection and autotune flows |
| `saf` | SAF-related train-side helpers |
| `selfplay` | Self-play orchestration and mixed-policy game execution |
| `selfplay_batch` | Batched self-play data plumbing |
| `teacher` | Teacher-side feature/label helpers, including belief surfaces |
| `training` | BC, RL, ACH, DRDA, ExIt, DeltaQ promotion/validation, losses, gates, and orchestrators |

## Workspace binaries

Crate currently exposes these workspace binaries:

| Binary | Purpose |
|--------|---------|
| `train` | Main training entrypoint; supports normal training, preflight, probe, and DeltaQ-promotion modes |
| `mjai_audit` | Audits replay datasets and archives, including failure bucketing and optional failure inventories |
| `recompress` | Recompression util for replay/data artifacts |
| `repack_tar` | Repack util for tar-based replay corpora |
| `build_replay_delta_q_sidecar` | Builds replay-side DeltaQ sidecars |
| `build_replay_exit_sidecar` | Builds replay-side ExIt sidecars |
| `mjai_debug_failure` | Debug helper for replay failures |
| `mjai_first_failure` | Finds/inspects first replay failure in dataset |
| `build_bc_shards` | Builds BC shard datasets and manifests from replay corpora |

Main training entrypoint lives at [`src/bin/train.rs`](src/bin/train.rs). Split into focused `src/bin/train/*` submodules for runtime/preflight selection, probe transport, autotune, resume/state persistence, and test support.

## Operator workflow docs

For concrete runbook-style docs, not crate ownership summary, read:

- [`docs/TRAINING_WORKFLOWS.md`](../../docs/TRAINING_WORKFLOWS.md) — training modes, YAML contract, BC/RL shape, and sidecar-enabled runs
- [`docs/PREFLIGHT_AND_RUNTIME_SELECTION.md`](../../docs/PREFLIGHT_AND_RUNTIME_SELECTION.md) — preflight cache, probe flows, runtime authority, and resume rules
- [`docs/REPLAY_SIDECARS.md`](../../docs/REPLAY_SIDECARS.md) — ExIt/DeltaQ sidecar gen and replay-time joins
- [`docs/MJAI_AUDIT_AND_FAILURE_TRIAGE.md`](../../docs/MJAI_AUDIT_AND_FAILURE_TRIAGE.md) — replay corpus audit, failure inventories, and replay-debug workflow
- [`docs/BC_SHARDS.md`](../../docs/BC_SHARDS.md) — BC shard production, manifest fields, and shard-backed training
- [`docs/DELTAQ_PROMOTION.md`](../../docs/DELTAQ_PROMOTION.md) — DeltaQ promotion mode, gates, and persisted artifact fields

## Runtime and data contract

Training crate consumes same live runtime surface as rest of Hydra:

- encoder/model contract: `192x34`
- action space: 46 actions
- replay input support: flat MJAI dirs plus `.tar.zst` archives
- default workspace test path: `cargo nextest run --release`

Docker/container-facing training contract documented in [`docker/train/README.md`](../../docker/train/README.md).

## Training flow at a glance

At high level, `hydra-train` does four things:

1. reads config and picks runtime/preflight behavior
2. loads replay or self-play data through `data::*`
3. builds targets/losses and runs BC/RL/update loops through `training::*`
4. evaluates, checkpoints, and reports metrics through train bin and eval helpers

That split intentional: runtime semantics stay below this crate, while optimize policy and target construction stay here.

## Where to read next

- Need runtime truth? Read [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](../../docs/COMPATIBILITY_SURFACE.md).
- Need current shipped/staged status? Read [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md).
- Need active Hydra v1 roadmap? Read [`research/design/HYDRA_RECONCILIATION.md`](../../research/design/HYDRA_RECONCILIATION.md).
- Need container execution details? Read [`docker/train/README.md`](../../docker/train/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [LICENSE](../../LICENSE).

- Free for personal, non-commercial, and academic use
- Commercial mahjong AI services require paid license from Licensor
- Converts to Apache-2.0 on 2031-03-02

For commercial licensing inquiries, contact Sho Kaneko.