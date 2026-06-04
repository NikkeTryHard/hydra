# Hydra Training Runbook

`example.yaml` is launch/config source of truth. Keep it current when runtime, resume, validation, model/profile, checkpoint, data, PPO, or backend behavior changes.

Current code wins if this doc drifts.

## Read This First

Normal BC training streams raw MJAI unless `bc_shards_manifest_path` is set.

YAML is runtime authority. Preflight output is evidence only; it never mutates YAML and never chooses config for you.

Shards, sidecars, DeltaQ, CUDA graph paths, and promotion gates are opt-in/gated paths. They must fail closed on contract mismatch.

## Train Entrypoint

Training goes through Rust `hydra-train` binary:

```bash
pixi run cargo run -p hydra-train --no-default-features --features training --bin train -- <config.yaml>
```

Modes:

- `train <config.yaml>`: normal train from YAML.
- `train --preflight --pf-candidate-tuples ... --pf-output md`: benchmark exact runtime tuples. No config/data/manifest/cache read.
- `train <config.yaml> --probe-kind <kind> ...`: bounded microbatch/game-count probe.

`hydra-train` is glue. Config/preflight/probe/status contracts live in `hydra-train-runtime`; execution lives in `hydra-train-exec`.

## YAML Authority

Operators usually touch:

- `data_dir` / `raw_mjai_data_dirs`
- `output_dir`, `stage`, `run_name`
- `batch_size`, `microbatch_size`, `validation_microbatch_size`
- `device`
- `bc`
- `rl`
- `resume_checkpoint`, `resume_latest`
- `checkpoint_every_n_steps`, `keep_step_checkpoints`
- `tensorboard`, `launch_tensorboard`, `background`
- `bc_shards_manifest_path`
- `python_model_profile`, `python_backbone_profile`, `python_residual_profile`, `python_variant`
- `python_raw_mjai_transport`

Use [`../example.yaml`](../example.yaml) for current intended shape.

## Raw-MJAI BC

Raw-MJAI is default BC input path.

If `raw_mjai_data_dirs` is set, Hydra uses exactly those entries. Otherwise it uses `data_dir`.

Default transport is pinned PyO3 via `hydra-raw-mjai-pyo3`; `stdout` is fallback.

Resume is game-boundary based. Checkpoint load validates schema, model/runtime/optimizer/loss/EMA contracts, source identity, raw-MJAI progress metadata, and RNG metadata before continuing.

## Run Layout

`output_dir` is campaign root. Run artifacts live under:

```text
<output_dir>/stages/<stage>/runs/<run_id>/
```

Important files:

```text
config.yaml
launch_metadata.json
logs/events.jsonl
logs/train_steps.jsonl
logs/stdout.log
logs/stderr.log
train.pid
checkpoints/latest.pt
checkpoints/step_<global_step>.pt
python_learner_result.json
tensorboard/
exports/
rollouts/
eval/
```

`resume_latest: true` resolves through stage `latest_run` marker when available.

## Logging And Checkpoints

`logs/train_steps.jsonl` is main watch file.

```bash
tail -f <run_dir>/logs/train_steps.jsonl
```

TensorBoard starts on first free port at or above `tensorboard_port` when `launch_tensorboard: true`.

Checkpoints are data-only. `latest.pt` is newest/resumable state, not PPO “best”. Immutable step checkpoints require `keep_step_checkpoints: true`.

## Validation

Python validation uses fixed held-out window by default.

Train augmentation does not apply to validation unless explicitly passed by direct CLI.

Validation logs expose requested/actual batches and samples. Use actual validation sample count when judging plateaus or regressions.

`validation_gates` affect best-checkpoint promotion only; they do not gate resume checkpoint writes.

## BC Models

Current common long-run shape:

- profile: `large`
- hidden: `384`
- blocks: `16`
- SE bottleneck: `96`
- residual: `mish_se`
- variant: `compile_max_autotune`
- batch: `3072`
- microbatch: `1024`
- validation microbatch: `1024`
- device: `cuda:0`
- EMA: enabled

Backbones:

- `conv2d_local3`
- `tileformer_bias`
- `convnext_tile_k7`
- `global_pool_bias`

Residual profiles:

- `mish_se`
- `mish_no_se`
- `mish_eca`
- `silu_se`
- `relu_se`
- `relu_no_se`
- `relu_no_norm_no_se`

Do not use `global_pool_bias` until its previous compile-path NaN is debugged.

## BC Shards

BC shards are optional fixed/cache input.

Use shards when you need repeated runs, stable materialized train/validation data, or sidecar-baked labels.

Contracts:

- compact v3 only
- dense/v2 hard-errors
- manifest is dataset contract
- YAML is runtime contract
- preflight reads neither manifest nor YAML

Typical flow:

1. Audit replay input.
2. Build shards.
3. Validate manifest/header/report.
4. Set `bc_shards_manifest_path` in YAML.
5. Train.

## Sidecars And Advanced Labels

Sidecars are optional labels keyed by replay/action identity.

Missing key means absent label. Present-but-mismatched record hard-errors.

Sidecars are bound to source, version, legal-mask digest, schema, shape, and provenance. Valid JSONL alone is not enough.

Positive ExIt/safety/DeltaQ weights require matching sidecar path and compatible data.

DeltaQ promotion remains gated eval workflow, not default training.

## T1 PPO

Enable PPO with:

```yaml
rl:
  phase: ppo_control
  rollout_inference: mahjax-gpu
```

Routes:

- `mahjax-gpu`: current default for serial/depth-1 Python PPO rollout.
- `torch-callback`: compatibility/reference route.
- `rust-ort`: native ONNX reference route.

MahJAX PPO uses training CUDA device. Separate rollout device is rejected.

Useful knobs:

- `rl.games_per_batch`
- `rl.microbatch_size`
- `rl.learning_rate`
- `rl.target_kl`
- `rl.ppo_pipeline_depth`
- `rl.rollout_inference`

See [`MAHJAX_PPO.md`](MAHJAX_PPO.md) for limits.

## Preflight And Probes

Preflight benchmarks exact tuples and prints evidence.

It does not read YAML, data, shard manifest, dataset cache, or current run state.

It does not choose winners and does not edit config.

Operators update YAML by hand after accepting evidence.

## Precision

BC CUDA may use BF16 AMP by config/default. CPU stays FP32. Explicit `fp32` overrides.

RL training and DeltaQ promotion hard-error on BF16.

Loss, backward, optimizer, checkpoint, validation, and promotion authority stay FP32 unless specific path proves otherwise.

## CUDA Graph / Burn CUDA

CUDA graph and Burn CUDA paths are probe/feature-gated lanes.

Do not promote them from single throughput row. Need correctness, parity, and stable profile evidence first.

## Failure Rules

Hard-error on:

- replay legality mismatch
- shard manifest/header mismatch
- sidecar provenance/schema/shape mismatch
- checkpoint runtime/schema/source mismatch
- invalid tensor shape or action-mask width
- BF16 use in RL/DeltaQ gated paths

Do not suppress these errors for launch convenience.

## Read Next

- Current status: [`CURRENT_STATUS.md`](CURRENT_STATUS.md)
- Runtime/game rules: [`GAME_ENGINE.md`](GAME_ENGINE.md)
- Hard compatibility contracts: [`COMPATIBILITY_SURFACE.md`](COMPATIBILITY_SURFACE.md)
- MahJAX PPO: [`MAHJAX_PPO.md`](MAHJAX_PPO.md)
- Config SSOT: [`../example.yaml`](../example.yaml)
