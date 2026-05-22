# Hydra PyTorch BC learner

Default BC shard training path. Rust owns replay parsing, BC shard build, manifest validation, CLI orchestration, and legacy Rust/Burn reference path. Python owns BC model/loss/optimizer/AMP/`torch.compile`/checkpoint.

Supported now:

- real compact BC shard ingest from `--manifest <bc_shards_manifest.json>`
- full base heads/losses: policy, value, GRP, tenpai, opponent next discard, danger, score PDF/CDF
- default-off advanced labels already present in shard path: `oracle_critic`, `safety_residual`
- BF16 autocast on CUDA
- `torch.compile` fullgraph clean for BC loss step
- data-only checkpoint save/load with model + optimizer + RNG + config metadata

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
  --bc-shards-manifest output/bc-feed-bench-shards-100k/bc_shards_manifest.json \
  --output-dir /home/cachybtw/tmp/hydra-py-default-bc-smoke \
  --device cuda:0 \
  --python-variant compile_default \
  --python-warmup 1 \
  --python-steps 3 \
  --python-compile-fullgraph-check \
  --python-residual-profile mish_se
```

This shells out to:

```bash
pixi run -e py-train python scripts/hydra_pytorch_oracle.py ...
```

script path keeps `oracle` for compatibility; user-facing config/CLI path is Python BC learner.

`--experimental-python-learner` remains accepted as deprecated alias. Prefer default `--bc-shards-manifest` route or explicit `--bc-backend python`.

Residual profiles are checkpoint-stable architecture strings. Default `mish_se` is canonical SE-ResNet: Mish + GroupNorm + SE, 10 blocks, 256 hidden. Opt-ins: `silu_se`, `relu_se`, `mish_no_se`, `relu_no_se`, `relu_no_norm_no_se`. No-SE profiles are speed/ablation only. 5k equal-step raw-MJAI validation: `mish_no_se` had faster train loop but slightly worse validation than `mish_se`; keep opt-in, do not promote. Checkpoint resume requires exact profile match.

Python backbone profile is checkpoint-stable and accepts only `conv2d_local3`: Conv2d over singleton height with local-3 tile kernels. `token_linear_local3` was probed and deleted: slower in repeated raw-MJAI timing, higher architecture risk, no concrete profiler reason to keep.

Compile variants do not change model math, topology, checkpoint architecture, input/action shapes, residual profile, or losses; they only change TorchInductor compile strategy. Default remains `compile_default` for smoke/preflight/short runs. `compile_max_autotune` is recommended for long same-architecture Python BC training. 200-step raw-MJAI run (`batch=2048`, `microbatch=1024`, `mish_se`, 10 blocks, warmup 10) measured `+8.7%` train throughput and `+7.7%` end-to-end throughput excluding compile versus `compile_default`. Compile/autotune overhead means short runs may be slower including compile; measured 200-step run remained below break-even when compile time was included. Inductor/autotune warnings during compile are diagnostic text, not training failure, unless process returns non-zero or writes `compile_error`. Do not promote Inductor env knobs: same-architecture `conv2d_local3`/`mish_se`/10-block/256-hidden `compile_max_autotune` probes rejected `warn_mix_layout`, `layout_optimization`, `max_autotune_pointwise`, `coordinate_descent_tuning`, and `max-autotune-no-cudagraphs`; e2e was `-0.07%` to `-1.30%`. Next same-architecture lane is fused GroupNorm+Mish(+SE), custom Triton or equivalent fusion.

If YAML omits `bc_shards_manifest_path`, Rust launcher streams raw MJAI from `data_dir`; shard manifests still use `--manifest`.

## Direct Python task

```bash
pixi run python-bc-train -- --manifest output/bc-feed-bench-shards-100k/bc_shards_manifest.json --variant compile_default --warmup 1 --steps 3 --out /home/cachybtw/tmp/hydra_py_direct.json
```

Script name `scripts/hydra_pytorch_oracle.py` remains compatibility name.
