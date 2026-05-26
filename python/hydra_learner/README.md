# Hydra PyTorch BC learner

Default plain-BC training backend. Rust owns replay parsing, raw-MJAI stream, BC shard build, manifest validation, launcher glue, legacy Rust/Burn reference path, and Python config conversion in `hydra-train-runtime::config::python`. Python owner is `hydra_learner.cli`; train loop/modules live under `hydra_learner`. `train_bc.py` and `scripts/hydra_pytorch_oracle.py` are compatibility entrypoints only.

Supported now:

- compact BC shard ingest from `--manifest <bc_shards_manifest.json>`
- raw-MJAI ingest from `--raw-mjai-data-dir <dir>`; default transport `pinned_pyo3`, fallback `stdout`
- full base heads/losses: policy, value, GRP, tenpai, opponent next discard, danger, score PDF/CDF
- default-off advanced labels already present in shard path: `oracle_critic`, `safety_residual`, ExIt target/mask, DeltaQ target/mask carrier
- BF16 autocast on CUDA
- `torch.compile` fullgraph clean for BC loss step
- resumable data-only checkpoint save/load for raw-MJAI and shard-backed runs with model + optimizer + RNG + config metadata; raw-MJAI uses game-boundary cursor restore
- balanced JSONL lifecycle/step logs under concrete run dir `logs/`
- TensorBoard event files under concrete run dir `tensorboard/`
- T1 PPO control long-run path via Rust YAML `rl.phase: ppo_control`; exports current checkpoint to ONNX, collects real-game rollouts through `hydra-raw-mjai-pyo3`, then runs masked PPO-GAE with run-local `exports/`, `rollouts/`, and `eval/`.

Not active in Python default yet:

- ExIt/search teacher training is default-off; ExIt loss only consumes validated compact-shard labels when explicitly weighted.
- DeltaQ target/mask carrier is default-off; positive DeltaQ loss fails closed until reviewed output-head contract exists.
- belief fields
- mixture weights
- opponent hand type

Use legacy Rust/Burn BC path only for feature-gated advanced modes or debugging.

## Environment

Use Pixi `py-train` env. It pins PyTorch `2.11.0+cu128`.

```bash
pixi run -e py-train python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
PY
```

Default Rust/Burn env remains root/default Pixi env with torch/libtorch `2.9.0` for Burn/tch compatibility. Do not use 2.12/nightly in normal workflow.

## Normal launch through Rust CLI

```bash
pixi run cargo run --quiet --package hydra-train --features training --bin train -- \
  --bc-shards-manifest path/to/bc_shards_manifest.json \
  --output-dir path/to/hydra-campaign \
  --device cuda:0 \
  --python-variant compile_max_autotune \
  --python-warmup 1 \
  --python-steps 3 \
  --python-compile-fullgraph-check \
  --python-residual-profile mish_se
```

This shells out to:

```bash
pixi run -e py-train python scripts/hydra_pytorch_oracle.py ...
```

`hydra_learner.cli` owns user-facing Python CLI/config. `hydra_learner.train_bc` and script path `scripts/hydra_pytorch_oracle.py` remain compatibility entrypoints; `--experimental-python-learner` remains accepted as deprecated Rust alias. Prefer default `--bc-shards-manifest` route, raw-MJAI YAML without `bc_shards_manifest_path`, or explicit `--bc-backend python`.

Residual profiles are checkpoint-stable architecture strings. Default `mish_se` is canonical SE-ResNet: Mish + GroupNorm + SE, 10 blocks, 256 hidden. Opt-ins: `silu_se`, `relu_se`, `mish_no_se`, `relu_no_se`, `relu_no_norm_no_se`. No-SE profiles are speed/ablation only. 5k equal-step raw-MJAI validation: `mish_no_se` had faster train loop but slightly worse validation than `mish_se`; keep opt-in, do not promote. Checkpoint resume requires exact profile match.

Python backbone profile is checkpoint-stable. Supported values: `conv2d_local3`, `tileformer_bias`, `convnext_tile_k7`, `global_pool_bias`; current default/canonical value is `conv2d_local3`. Non-default backbone profiles are training/checkpoint-supported only; ONNX/native-arena export supports `conv2d_local3` only for now.

Compile variants do not change model math, topology, checkpoint architecture, input/action shapes, residual profile, or losses; they only change TorchInductor strategy. Canonical production Python BC uses `compile_max_autotune`; use `compile_default` only for smoke/preflight/short debug.

If YAML omits `bc_shards_manifest_path`, Rust launcher streams raw MJAI from `raw_mjai_data_dirs` when set, otherwise from `data_dir`. Default bridge crate is `hydra-raw-mjai-pyo3` pinned PyO3; stdout remains fallback. Raw-MJAI resume is default-on and skips deterministic completed games from checkpoint progress.
## Run artifacts and resume

Rust launcher treats `output_dir` as campaign root, resolves concrete run dir, and passes that run dir to Python. Campaign root owns `campaign.json`, `registry/`, and `stages/`; Python writes only under run dir:

```text
<output_dir>/stages/<stage>/latest_run
<output_dir>/stages/<stage>/runs/<run_id>/config.yaml
<output_dir>/stages/<stage>/runs/<run_id>/launch_metadata.json
<output_dir>/stages/<stage>/runs/<run_id>/python_learner_result.json
<output_dir>/stages/<stage>/runs/<run_id>/logs/events.jsonl
<output_dir>/stages/<stage>/runs/<run_id>/logs/train_steps.jsonl
<output_dir>/stages/<stage>/runs/<run_id>/logs/stdout.log          # background mode only
<output_dir>/stages/<stage>/runs/<run_id>/logs/stderr.log          # background mode only
<output_dir>/stages/<stage>/runs/<run_id>/logs/tensorboard.log     # auto TensorBoard output
<output_dir>/stages/<stage>/runs/<run_id>/checkpoints/latest.pt
<output_dir>/stages/<stage>/runs/<run_id>/checkpoints/step_<global_step>.pt  # only with --python-keep-step-checkpoints
<output_dir>/stages/<stage>/runs/<run_id>/exports/                # ONNX/native export artifacts
<output_dir>/stages/<stage>/runs/<run_id>/rollouts/               # RL rollout batches/artifacts
<output_dir>/stages/<stage>/runs/<run_id>/eval/                   # arena/eval reports
<output_dir>/stages/<stage>/runs/<run_id>/summary.json
<output_dir>/stages/<stage>/runs/<run_id>/tensorboard/events.out.tfevents.*
```

`events.jsonl` is lifecycle/resume/validation/checkpoint log. `train_steps.jsonl`
is balanced step telemetry, written every `--python-log-every-steps` and final
step; avoid `1` on CUDA unless debugging sync/log overhead.

Stages are `T0_bc_baseline`, `T1_ppo_control`, `T2_direct_sampled_ach`, `T3_drda_residual_ach`, `T4_pbrs_beta_sweep`, `T5_exit_auxiliary`, `T6_delta_q_experiment`, and `T7_population_window`.

Resume with config `resume_checkpoint: <run_dir>/checkpoints/latest.pt` or CLI `--python-resume <checkpoint>`. `resume_latest: true` resolves through stage `latest_run` marker when stage metadata is available. Checkpoint load validates schema, model, optimizer, runtime, loss weights, source contract, and RNG metadata. Raw-MJAI resume supports game-boundary cursor restore when checkpoint/runtime metadata matches.

Periodic checkpoint controls:

```bash
--python-checkpoint-every-steps 200 \
--python-keep-step-checkpoints
```

Default periodic output refreshes `checkpoints/latest.pt`; keep-step mode also
retains immutable `step_<global_step>.pt` files. Direct CLI `--python-checkpoint-out`
keeps legacy single-file checkpoint path and is mutually exclusive with
checkpoint-dir/keep-step mode.

TensorBoard:

```bash
--python-launch-tensorboard \
--python-tensorboard-host 127.0.0.1 \
--python-tensorboard-port 6006
```

Launcher picks first free port at or above requested port, passes selected
URL into Python metrics, starts TensorBoard detached, and writes its output to
`logs/tensorboard.log`.

Background mode:

```bash
--python-background
```

Rust detaches learner, writes `train.pid`, redirects stdout/stderr logs, and
prints output/log/checkpoint/TensorBoard URL plus watch command:

```bash
tail -f <output_dir>/stages/<stage>/runs/<run_id>/logs/train_steps.jsonl
```


Direct Python accepts exactly one input:

```bash
pixi run python-bc-train -- --manifest path/to/bc_shards_manifest.json --variant compile_max_autotune --warmup 1 --steps 3 --out path/to/result.json
pixi run python-bc-train -- --raw-mjai-data-dir path/to/mjai --raw-mjai-transport pinned_pyo3 --variant compile_max_autotune --warmup 1 --steps 3 --out path/to/result.json
```

Pinned PyO3 lookup: `HYDRA_RAW_MJAI_PYO3_LIB`, then `target/release/libhydra_raw_mjai_pyo3.so`, then `target/debug/libhydra_raw_mjai_pyo3.so`. Build missing lib with:

```bash
pixi run cargo build -p hydra-raw-mjai-pyo3 --release --quiet
```

Compatibility entrypoints: `python -m hydra_learner.train_bc`, `scripts/hydra_pytorch_oracle.py`. New Python CLI docs/code should point at `hydra_learner.cli`.
