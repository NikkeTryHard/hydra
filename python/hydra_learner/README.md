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
  --python-compile-fullgraph-check
```

This shells out to:

```bash
pixi run -e py-train python scripts/hydra_pytorch_oracle.py ...
```

`--experimental-python-learner` remains accepted as deprecated alias. Prefer default `--bc-shards-manifest` route or explicit `--bc-backend python`.

## Direct Python task

```bash
pixi run python-bc-train -- --manifest output/bc-feed-bench-shards-100k/bc_shards_manifest.json --variant compile_default --warmup 1 --steps 3 --out /home/cachybtw/tmp/hydra_py_direct.json
```

Script name `scripts/hydra_pytorch_oracle.py` remains compatibility name.
