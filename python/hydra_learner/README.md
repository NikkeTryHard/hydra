# Hydra PyTorch BC learner

Default plain-BC training backend. Rust owns replay parsing, raw-MJAI stream, BC shard build, manifest validation, CLI orchestration, and legacy Rust/Burn reference path. Python owns BC model/loss/optimizer/AMP/`torch.compile`/checkpoint.

Supported now:

- compact BC shard ingest from `--manifest <bc_shards_manifest.json>`
- raw-MJAI ingest from `--raw-mjai-data-dir <dir>`; default transport `pinned_pyo3`, fallback `stdout`
- full base heads/losses: policy, value, GRP, tenpai, opponent next discard, danger, score PDF/CDF
- default-off advanced labels already present in shard path: `oracle_critic`, `safety_residual`
- BF16 autocast on CUDA
- `torch.compile` fullgraph clean for BC loss step
- resumable data-only checkpoint save/load with model + optimizer + RNG + config metadata
- balanced JSONL lifecycle/step logs under `output_dir/logs`
- TensorBoard event files under `output_dir/tensorboard`

Not supported in Python default yet:

- ExIt target/mask
- DeltaQ target/mask
- belief fields
- mixture weights
- opponent hand type

Use legacy Rust/Burn BC path only for those advanced modes or debugging.

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
  --output-dir path/to/output-dir \
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

Script path keeps `oracle` for compatibility; user-facing config/CLI path is Python BC learner. `--experimental-python-learner` remains accepted as deprecated alias. Prefer default `--bc-shards-manifest` route, raw-MJAI YAML without `bc_shards_manifest_path`, or explicit `--bc-backend python`.

Residual profiles are checkpoint-stable architecture strings. Default `mish_se` is canonical SE-ResNet: Mish + GroupNorm + SE, 10 blocks, 256 hidden. Opt-ins: `silu_se`, `relu_se`, `mish_no_se`, `relu_no_se`, `relu_no_norm_no_se`. No-SE profiles are speed/ablation only. 5k equal-step raw-MJAI validation: `mish_no_se` had faster train loop but slightly worse validation than `mish_se`; keep opt-in, do not promote. Checkpoint resume requires exact profile match.

Python backbone profile is checkpoint-stable and accepts only `conv2d_local3`: Conv2d over singleton height with local-3 tile kernels. `token_linear_local3` was probed and deleted: slower in repeated raw-MJAI timing, higher architecture risk, no concrete profiler reason to keep.

Compile variants do not change model math, topology, checkpoint architecture, input/action shapes, residual profile, or losses; they only change TorchInductor strategy. Canonical production Python BC uses `compile_max_autotune`; use `compile_default` only for smoke/preflight/short debug.

If YAML omits `bc_shards_manifest_path`, Rust launcher streams raw MJAI from `data_dir`. Default transport is pinned PyO3; stdout remains fallback.
## Run artifacts and resume

Rust launcher creates stable artifact dirs for every Python BC run:

```text
<output_dir>/python_learner_result.json
<output_dir>/logs/events.jsonl
<output_dir>/logs/train_steps.jsonl
<output_dir>/logs/stdout.log          # background mode only
<output_dir>/logs/stderr.log          # background mode only
<output_dir>/logs/tensorboard.log     # auto TensorBoard output
<output_dir>/checkpoints/latest.pt
<output_dir>/checkpoints/step_<global_step>.pt  # only with --python-keep-step-checkpoints
<output_dir>/tensorboard/events.out.tfevents.*
```

`events.jsonl` is lifecycle/resume/validation/checkpoint log. `train_steps.jsonl`
is balanced step telemetry, written every `--python-log-every-steps` and final
step; avoid `1` on CUDA unless debugging sync/log overhead.

Resume with config `resume_checkpoint: <output_dir>/checkpoints/latest.pt` or
CLI `--python-resume <checkpoint>`. Checkpoint load validates schema, model,
optimizer, runtime, loss weights, manifest/source contract, and RNG metadata.

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
tail -f <output_dir>/logs/train_steps.jsonl
```


Direct Python accepts exactly one input:

```bash
pixi run python-bc-train -- --manifest path/to/bc_shards_manifest.json --variant compile_max_autotune --warmup 1 --steps 3 --out path/to/result.json
pixi run python-bc-train -- --raw-mjai-data-dir path/to/mjai --raw-mjai-transport pinned_pyo3 --variant compile_max_autotune --warmup 1 --steps 3 --out path/to/result.json
```

Pinned PyO3 lookup: `HYDRA_RAW_MJAI_PINNED_LIB`, then `target/release/libhydra_raw_mjai_ffi.so`, then `target/debug/libhydra_raw_mjai_ffi.so`. Build missing lib with:

```bash
pixi run cargo build -p hydra-raw-mjai-ffi --release --quiet
```

Script name `scripts/hydra_pytorch_oracle.py` remains compatibility name.
