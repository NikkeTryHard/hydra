# Hydra Training Workflows

Operator-facing guide to Hydra's current training entrypoints, config surface, and mode selection.

This document is intentionally workflow-first. For shipped-vs-staged status, read [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md). For runtime and compatibility contracts, read [`docs/COMPATIBILITY_SURFACE.md`](COMPATIBILITY_SURFACE.md) and [`docs/GAME_ENGINE.md`](GAME_ENGINE.md). For container execution, read [`docker/train/README.md`](../docker/train/README.md).

## What owns training

Hydra's main training entrypoint is:

- `crates/hydra-train/src/bin/train.rs`

That binary routes into four top-level behaviors:

1. normal training
2. preflight/runtime selection
3. probe-only runtime measurement
4. DeltaQ promotion evaluation

The YAML contract for all of them is owned by:

- `crates/hydra-train/src/bin/train/config.rs`

## CLI modes at a glance

The binary always starts with a YAML config path:

```bash
train <config.yaml>
```

Additional mode flags select specialized flows:

| Mode | Invocation shape | Purpose |
|---|---|---|
| Normal training | `train config.yaml` | Runs BC or RL training from config |
| Preflight | `train config.yaml --preflight` | Chooses runtime tuple and writes/reads preflight cache |
| Probe-only | `train config.yaml --probe-kind <train\|validation> --probe-candidate-microbatch <N> ...` | Measures a candidate microbatch without running full training |
| DeltaQ promotion | `train config.yaml --delta-q-promotion [--delta-q-baseline-checkpoint <path>]` | Evaluates a candidate checkpoint against a baseline using offline and arena-style gates |

There is also an internal child-process probe path used by preflight/probe orchestration. That path is not the normal operator entrypoint and should be treated as implementation detail.

## Choosing the right mode

Use this rule of thumb:

- Use normal training when you already trust the runtime tuple in config or resume state.
- Use preflight when you want Hydra to measure and choose runtime settings safely for the current machine and workload.
- Use probe-only when you are investigating runtime capacity or comparing a small set of candidate microbatches without paying full preflight/train cost.
- Use DeltaQ promotion only when a candidate checkpoint already exists and you want to decide whether the narrow DeltaQ lane should be promoted.

## YAML contract overview

`TrainConfig` is the real user-facing contract for Hydra training. The most important top-level fields are:

| Field | Meaning |
|---|---|
| `data_dir` | Replay source root. Supports loose MJAI files or a `.tar.zst` archive path. |
| `output_dir` | Training artifacts, checkpoints, logs, reports, and cache outputs. |
| `num_epochs` | Epoch count for BC-style training flows. |
| `batch_size` | Logical batch size before microbatching/accumulation. |
| `microbatch_size` | Optional explicit selected-runtime training microbatch override. |
| `validation_microbatch_size` | Optional explicit selected-runtime validation microbatch override. |
| `train_fraction` | Deterministic train/validation split fraction for replay sources. |
| `source_filters` | Include/exclude filtering for replay identities. |
| `augment` | Suit-permutation augmentation toggle for replay data. |
| `resume_checkpoint` | Resume from a previous checkpoint base. |
| `advanced_loss` | Optional activation/weights for auxiliary supervised lanes. |
| `rl` | Enables RL/self-play training path with RL-specific knobs. |
| `bc` | Behavioral-cloning optimizer hyperparameters. |
| `device` | Device label used to pick the train backend. |
| `precision_mode` | Precision dispatch, currently `fp32` or `bf16_autocast`. |
| `preflight` | Runtime-selection and autotuning knobs. |
| `exit_sidecar_path` | Optional replay ExIt sidecar index for replay-side supervision joins. |
| `delta_q_sidecar_path` | Optional replay DeltaQ sidecar index for replay-side supervision joins. |
| `bc_shards_manifest_path` | Optional prebuilt BC shard manifest input. |

## Minimal BC config

This is the smallest useful baseline shape for replay-driven BC training:

```yaml
data_dir: /data
output_dir: /output
num_epochs: 1
batch_size: 2048

bc:
  learning_rate: 0.00025
  min_learning_rate: 0.000001
  weight_decay: 0.00001
  grad_clip_norm: 1.0
  warmup_steps: 1000
```

Useful optional additions for normal BC runs:

```yaml
microbatch_size: 256
validation_microbatch_size: 128
precision_mode: bf16_autocast
train_fraction: 0.9
augment: true
tensorboard: true
```

## Minimal RL config

RL is configured by adding an `rl` block. The current RL phase enum is defined in `config.rs` as:

- `drda_ach_self_play`
- `exit_pondering`

Example:

```yaml
data_dir: /data
output_dir: /output
num_epochs: 1
batch_size: 2048

rl:
  games_per_batch: 4
  temperature: 1.0
  phase: drda_ach_self_play
```

Important current constraint:

- BF16/AMP is shipped for BC flows, but RL and DeltaQ promotion remain explicitly gated as staged surfaces rather than baseline-on defaults.

## Sidecar-enabled replay training

Hydra can join replay-side supervision lanes during replay loading when sidecar paths are configured.

Relevant config fields:

- `exit_sidecar_path`
- `delta_q_sidecar_path`

What those do:

- ExIt sidecars hydrate replay-time search-derived labels when provenance, source identity, source version, source net hash, and legal-mask digest all match.
- DeltaQ sidecars hydrate replay-time delta-Q labels under a stricter contract that also validates the action-mask/target shape.

For how to generate and validate those sidecars, read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).

## BC shards

If `bc_shards_manifest_path` is set, Hydra can consume prebuilt BC shard metadata rather than scanning loose replay sources directly.

Use this when:

- you have already materialized a shard dataset for repeated BC runs
- you want training input layout to be driven by a precomputed shard manifest rather than raw replay discovery

## Precision mode

Current precision modes in the training config:

- `fp32`
- `bf16_autocast`

Current repo status:

- BC training, preflight, probe flows, and stage-2 benchmark dispatch by precision mode.
- RL and DeltaQ promotion are not yet baseline BF16 surfaces.

## Training flow shape

At a high level, normal training does four things:

1. parse config and choose training mode
2. build loader/runtime settings
3. load replay or self-play data and collate targets
4. run training, validation, checkpointing, and reporting

The highest-signal boundary for operators is:

- runtime selection decides how large and fast training batches are allowed to be on this machine
- data pipeline decides what examples and sidecars actually get loaded
- training phase decides which optimization loop and target surfaces are active

## When to run preflight first

Run preflight before a new training run when:

- you are on a new machine or GPU layout
- you changed precision mode
- you changed microbatch assumptions or replay workload materially
- you do not trust old runtime selections to match the current hardware/workload pair

For the detailed cache and authority rules, read [`docs/PREFLIGHT_AND_RUNTIME_SELECTION.md`](PREFLIGHT_AND_RUNTIME_SELECTION.md).

## Container workflow

For container execution:

- mount config, data, and output paths explicitly
- keep `data_dir` and `output_dir` aligned with the mounted container paths
- keep the image entrypoint argument as the config path only

The container README is intentionally short; it assumes this document owns the workflow explanation and the Docker README owns the container contract.

## Where to read next

- Need runtime-selection and cache authority details? Read [`docs/PREFLIGHT_AND_RUNTIME_SELECTION.md`](PREFLIGHT_AND_RUNTIME_SELECTION.md).
- Need replay sidecar generation and join semantics? Read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).
- Need replay corpus validation and failure triage before training? Read [`docs/MJAI_AUDIT_AND_FAILURE_TRIAGE.md`](MJAI_AUDIT_AND_FAILURE_TRIAGE.md).
- Need the full DeltaQ promotion runbook and artifact interpretation? Read [`docs/DELTAQ_PROMOTION.md`](DELTAQ_PROMOTION.md).
- Need current shipped/staged truth? Read [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md).
- Need runtime and compatibility constraints? Read [`docs/COMPATIBILITY_SURFACE.md`](COMPATIBILITY_SURFACE.md).
