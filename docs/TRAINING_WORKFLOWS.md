# Hydra Training Workflows

Ops guide: current Hydra train entrypoints, config surface, mode choice.

Workflow-first doc. Shipped vs staged: read [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md). Runtime/compat contracts: read [`docs/COMPATIBILITY_SURFACE.md`](COMPATIBILITY_SURFACE.md) and [`docs/GAME_ENGINE.md`](GAME_ENGINE.md). Container exec: read [`docker/train/README.md`](../docker/train/README.md).

## What owns training

Main train entrypoint:

- `crates/hydra-train/src/bin/train.rs`

Binary routes to 4 behaviors:

1. normal training
2. preflight/runtime selection
3. probe-only runtime measurement
4. DeltaQ promotion evaluation

YAML contract owner:

- `crates/hydra-train/src/bin/train/config.rs`

## CLI modes at a glance

Binary starts with YAML config path:

```bash
train <config.yaml>
```

Extra flags select special flows:

| Mode | Invocation shape | Purpose |
|---|---|---|
| Normal training | `train config.yaml` | Run BC or RL from config |
| Preflight | `train config.yaml --preflight` | Pick runtime tuple; write/read preflight cache |
| Probe-only | `train config.yaml --probe-kind <train\|validation> --probe-candidate-microbatch <N> ...` | Measure candidate microbatch; no full train |
| DeltaQ promotion | `train config.yaml --delta-q-promotion [--delta-q-baseline-checkpoint <path>]` | Compare candidate checkpoint vs baseline with offline + arena-style gates |

Also internal child-process probe path for preflight/probe orchestration. Impl detail, not operator entrypoint.

## Choosing the right mode

Rule of thumb:

- Use normal training when runtime tuple in config/resume state already trusted.
- Use preflight when Hydra must safely measure and choose runtime settings for current machine/workload.
- Use probe-only when checking runtime capacity or comparing few microbatches without full preflight/train cost.
- Use DeltaQ promotion only when candidate checkpoint already exists and narrow DeltaQ lane needs promote/no-promote decision.

## YAML contract overview

`TrainConfig` = real user-facing training contract. Key top-level fields:

| Field | Meaning |
|---|---|
| `data_dir` | Replay source root. Loose MJAI files or `.tar.zst` archive path. |
| `output_dir` | Train artifacts, checkpoints, logs, reports, cache outputs. |
| `num_epochs` | Epoch count for BC-style flows. |
| `batch_size` | Logical batch size before microbatching/accumulation. |
| `microbatch_size` | Optional explicit selected-runtime train microbatch override. |
| `validation_microbatch_size` | Optional explicit selected-runtime validation microbatch override. |
| `train_fraction` | Deterministic train/validation split fraction for replay sources. |
| `source_filters` | Include/exclude replay identity filters. |
| `augment` | Suit-permutation augmentation toggle. |
| `resume_checkpoint` | Resume from prior checkpoint base. |
| `advanced_loss` | Optional aux supervised lane activation/weights. |
| `rl` | Enable RL/self-play path with RL knobs. |
| `bc` | BC optimizer hyperparams. |
| `device` | Device label for train backend pick. |
| `precision_mode` | Precision dispatch, `fp32` or `bf16_autocast`. |
| `preflight` | Runtime-selection + autotuning knobs. |
| `exit_sidecar_path` | Optional replay ExIt sidecar index for replay-side supervision joins. |
| `delta_q_sidecar_path` | Optional replay DeltaQ sidecar index for replay-side supervision joins. |
| `bc_shards_manifest_path` | Optional prebuilt BC shard manifest input. |
| `shard_prefetch_depth` | Optional shard host-batch prefetch depth. Default `2`; valid `1..64`. |
| `validation_gates` | Optional experiment gate. Disabled by default; when enabled, gates best-checkpoint promotion on validation sample/loss/agreement and sidecar-label presence. |

## Minimal BC config

Smallest useful baseline shape for replay-driven BC training:

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

Useful optional adds for normal BC runs:

```yaml
microbatch_size: 256
validation_microbatch_size: 128
precision_mode: bf16_autocast
train_fraction: 0.9
augment: true
tensorboard: true
```

Experimental gate example, off by default in baseline Hydra v1 configs:

```yaml
validation_gates:
  enabled: true
  min_validation_samples: 1024
  max_policy_loss_regression: 0.0
  min_policy_agreement_delta: 0.0
  fail_training_on_gate_failure: false
  require_sidecar_coverage_when_weighted: true
```

Gate affects `best_model` promotion only. `latest_model` still saves for resume safety.

## Minimal RL config

Add `rl` block for RL. Current RL phase enum in `config.rs`:

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

- BF16/AMP shipped for BC. RL and DeltaQ promotion still staged, not baseline defaults.

## Sidecar-enabled replay training

Hydra can join replay-side supervision lanes during replay load when sidecar paths set.

Relevant config fields:

- `exit_sidecar_path`
- `delta_q_sidecar_path`

What they do:

- ExIt sidecars hydrate replay-time search-derived labels when provenance, source identity, source version, source net hash, and legal-mask digest all match.
- DeltaQ sidecars hydrate replay-time delta-Q labels under stricter contract that also validates action-mask/target shape.

One easy-miss but ops-important identity detail:

- loose replay files join by replay file name
- archive-backed replay entries join by full archive-entry identity

So sidecar keyed to `game.json` not automatically valid for archive entry `replays.tar.zst/path/inside/game.json`.

For generation/validation, read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).

If `advanced_loss.exit` or `advanced_loss.delta_q` is positive, matching sidecar path is required. With `validation_gates.enabled: true`, Hydra also requires validation batches to contain hydrated labels before best-checkpoint promotion. This catches wrong hash/version/path joins early.

Experimental sidecar weights stay disabled for baseline Hydra v1 unless explicitly set:

```yaml
advanced_loss:
  exit: 0.0
  safety_residual: 0.0
  delta_q: 0.0
```

Use A/B runs: same shards, same split, one changed weight.

## BC shards

`bc_shards_manifest_path` = production BC input for repeated training and preflight runs. Build shards once from replay corpus, then point both train and validation at that manifest instead of making every run parse archives, replay game state, rebuild legal masks, and regenerate labels.

Use raw loose/archive replay loading as slow offline path for audit, debugging, one-off comparison, and shard building. Do not use as steady-state hot training path unless intentionally comparing transports.

Recommended operator workflow:

1. Audit replay corpus and sidecar inputs.
2. Build train+validation BC shards with `build_bc_shards`.
3. Set `bc_shards_manifest_path: /output/bc-shards/bc_shards_manifest.json` in BC config.
4. Run preflight and training against same manifest.
5. Rebuild shards only when replay corpus, sidecar inputs/provenance, train fraction/filtering, encoder/action contract, or shard format changes.

Operational changes when shard mode enabled:

- train path reads prebuilt shard rows, not rescanning replay archives each run
- validation path also switches to shard readers, not loose-replay validation stream
- validation-sample materialization disabled, so run does not build cached in-memory validation microbatch set up front even when `max_validation_samples` set
- shard-backed validation becomes sequential shard-row scan with small host-batch prefetch queue, not replay-stream microbatch iteration
- `shard_prefetch_depth` controls train+validation shard host-batch queue depth; default `2`, raise only to hide measured producer/H2D bubbles, not to change samples
- startup banner uses manifest sample counts from shard artifact, not raw replay scan counts

Hydra rejects stale shard manifests whose encoder/action/record contract does not match current binary. It does not silently fall back to raw replay when configured shard manifest is invalid.

So shard-backed runs change startup behavior and validation transport. For runtime/memory comparison vs loose-replay run, treat shard mode as different data path, not only faster input source.

For build/inspect/consume workflow and manifest meaning, read [`docs/BC_SHARDS.md`](BC_SHARDS.md).

## Precision mode

Current precision modes in training config:

- `fp32`
- `bf16_autocast`

Current repo status:

- BC training, preflight, probe flows, and stage-2 benchmark dispatch by precision mode.
- RL and DeltaQ promotion not yet baseline BF16 surfaces.

## CUDA shard transfer and graph probe path

Shard-backed BC can use pinned host staging, dedicated async H2D copy stream, and preallocated GPU tensors when all true:

- build enables Cargo feature `cuda-graph`
- runtime device is CUDA
- BC config sets `bc_shards_manifest_path`

This applies to shard train epochs, shard train probes, and shard validation. Without `cuda-graph`, or on CPU, Hydra still uses shard mmap + host prefetch but materializes from pageable host memory through normal tensor construction.

Current semantics:

- shard host batches collate pageable, then stage into pinned buffers
- policy targets are generated as bounded CPU f32 one-hot rows before transfer; invalid action IDs leave all-zero rows
- pinned/preallocated path mirrors same policy-target semantics and no longer uses lazy Burn/tch `IntTensor::one_hot`
- H2D remains single-buffered; compute stream waits on copy event before forward
- metric accumulation keeps CPU rare-action counts on CPU and reads back only GPU loss scalars at logical-batch stats boundary
- progress bar text is throttled to first/log/validation/checkpoint/final step boundaries; skipped progress refreshes do not finalize discarded aggregate stats

Operator build forms:

```bash
cargo run --release -p hydra-train --features cuda-graph --bin train -- /path/to/config.yaml
cargo build --release -p hydra-train --features cuda-graph
```

Pinned memory footprint is batch-size bounded and queue depth stays small. Increase `batch_size` only with host RAM and pinned-memory pressure in mind.

`HYDRA_CUDA_GRAPH_PROBE=1 train config.yaml` runs child-process CUDA graph feasibility probe. It reports JSON with `probe_mode=compute_capture_only`, warmup/parity/capture timings, replay repeats, and blocker labels. Production replay is intentionally off and reported as `cuda_graph_replay=production_off_probe_only` because Burn Adam needs fresh Rust-side `GradientsParams`; graph replay cannot feed optimizer state safely. Probe knobs:

- `HYDRA_CUDA_GRAPH_PROBE_REPLAYS=N` changes replay repeat count; default `16`, max `1024`
- `HYDRA_CUDA_GRAPH_PROBE_POST_REPLAY_PARITY=0` disables post-replay parity rerun; default on

Recent real 64-step shard slice benchmark after safe transport/metric cuts: profiled `1981.9 samples/s`, wall `4.63s`; vs prior plain shard mean `1888.9 samples/s`, about `+4.9%` profiled throughput in one run. Treat as single-run signal, not final distribution. Main remaining bottleneck remains model compute plus unfused Burn Adam.

## Runtime advisories

Hydra emits startup advisories to console and `bc/step_log.jsonl` as `runtime_advisories` records. Preflight also prints measured advisory lines beside selected runtime when probe data shows material throughput gap. Advisories are not failures; they mean run remains semantically valid but may under-use hardware.

Startup keys:

- `cpu_device_for_training`: CPU run; CUDA feeding optimizations unavailable.
- `steady_state_cuda_bc_uses_loose_replay`: CUDA BC uses loose/archive replay; build shards for repeated steady-state training.
- `cuda_shards_without_pinned_async_h2d`: CUDA shard run built without `cuda-graph`; pinned async H2D unavailable.
- `small_microbatch_high_accumulation_overhead`: microbatch below batch; accumulation overhead may dominate.
- `explicit_microbatch_blocks_faster_candidate_search`: explicit microbatch settings may stop preflight from choosing faster candidates.
- `logging_or_metric_sync_overhead`: `log_every_n_steps=1`; metric/log readback can hurt CUDA throughput.
- `validation_or_checkpoint_cadence_overhead`: validation/checkpoint cadence may dominate wall time.
- `selected_train_runtime_slower_than_best_probe_candidate`: selected train microbatch was at least 20% slower than best stable probe candidate.
- `selected_validation_runtime_slower_than_best_probe_candidate`: selected validation microbatch was at least 20% slower than best stable probe candidate.

Warnings say under-optimized, not wrong. Hard failures remain reserved for invalid contracts such as stale shard manifests or missing required sidecars.

## Training flow shape

High level, normal training does 4 things:

1. parse config and choose training mode
2. build loader/runtime settings
3. load replay or self-play data and collate targets
4. run training, validation, checkpointing, and reporting

Highest-signal operator boundary:

- runtime selection decides batch size/speed limits on current machine
- data pipeline decides which examples and sidecars load
- training phase decides which optimization loop and target surfaces are active

## When to run preflight first

Run preflight before new training run when:

- new machine or GPU layout
- precision mode changed
- microbatch assumptions or replay workload changed materially
- old runtime selections no longer trusted for current hardware/workload pair

Detailed cache + authority rules: read [`docs/PREFLIGHT_AND_RUNTIME_SELECTION.md`](PREFLIGHT_AND_RUNTIME_SELECTION.md).

## Container workflow

For container execution:

- mount config, data, output paths explicitly
- keep `data_dir` and `output_dir` aligned with mounted container paths
- keep image entrypoint argument as config path only

Container README intentionally short; this doc owns workflow explanation, Docker README owns container contract.

## Where to read next

- Need runtime-selection and cache authority details? Read [`docs/PREFLIGHT_AND_RUNTIME_SELECTION.md`](PREFLIGHT_AND_RUNTIME_SELECTION.md).
- Need replay sidecar generation and join semantics? Read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).
- Need replay corpus validation and failure triage before training? Read [`docs/MJAI_AUDIT_AND_FAILURE_TRIAGE.md`](MJAI_AUDIT_AND_FAILURE_TRIAGE.md).
- Need shard build/manifest workflow? Read [`docs/BC_SHARDS.md`](BC_SHARDS.md).
- Need full DeltaQ promotion runbook and artifact interpretation? Read [`docs/DELTAQ_PROMOTION.md`](DELTAQ_PROMOTION.md).
- Need current shipped/staged truth? Read [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md).
- Need runtime and compatibility constraints? Read [`docs/COMPATIBILITY_SURFACE.md`](COMPATIBILITY_SURFACE.md).