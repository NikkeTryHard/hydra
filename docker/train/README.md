# Hydra training container

Image package Hydra training binaries for local + container GPU run.

## What is inside

- `train` -- training entrypoint from `crates/hydra-train/src/bin/train.rs`
- `mjai_audit` -- MJAI replay auditor from `crates/hydra-train/src/bin/mjai_audit.rs`
- Burn + `tch` / libtorch-compatible runtime via CUDA base image

Image does **not** add Jupyter, VS Code server, or RStudio.

## Build locally

From repo root:

```bash
docker build -f docker/train/Dockerfile -t hydra:local .
```

## Basic smoke check

No args should print binary usage contract:

```bash
docker run --rm hydra:local
```

## Runtime contract

Container entrypoint = `train`; expects 1 YAML config arg:

- `train <config.yaml>`
- mount config/data/output paths instead of baking datasets into image

Hydra current behavioral-cloning loader supports either:

- flat MJAI dir of `.json` / `.json.gz` files
- direct `.tar.zst` MJAI archive path

Keep mounted container paths aligned with config:

- `data_dir: /data` for extracted MJAI dir
- `data_dir: /data/dataset.tar.zst` for mounted archive file
- `output_dir: /output`

## Example run

```bash
docker run --rm \
  --gpus all \
  -v /host/config.yaml:/config/train.yaml:ro \
  -v /host/mjai:/data:ro \
  -v /host/output:/output \
  hydra:local \
  /config/train.yaml
```

YAML config should point at mounted container paths, example:

```yaml
data_dir: /data
output_dir: /output
num_epochs: 1
batch_size: 32
```

For archive-backed run, mount archive itself and point `data_dir` at that file path.

For full training-mode + config contract, read [`docs/TRAINING_WORKFLOWS.md`](../../docs/TRAINING_WORKFLOWS.md). For preflight/runtime-selection behavior, read [`docs/PREFLIGHT_AND_RUNTIME_SELECTION.md`](../../docs/PREFLIGHT_AND_RUNTIME_SELECTION.md).

## Publish to GHCR

Tag image:

```bash
docker tag hydra:local ghcr.io/nikketryhard/hydra:latest
```

Log in if needed:

```bash
gh auth token | docker login ghcr.io -u NikkeTryHard --password-stdin
```

Push:

```bash
docker push ghcr.io/nikketryhard/hydra:latest
```

If also want versioned tag:

```bash
docker tag hydra:local ghcr.io/nikketryhard/hydra:0.1.0
docker push ghcr.io/nikketryhard/hydra:0.1.0
```

## Note on image publishing

If publishing image to GHCR or another registry, make sure package visibility + pull path match environment that will run training.