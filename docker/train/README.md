# Hydra training container

This image packages Hydra's training binaries for local and containerized GPU execution.

## What is inside

- `train` -- training entrypoint from `crates/hydra-train/src/bin/train.rs`
- `mjai_audit` -- MJAI replay auditor from `crates/hydra-train/src/bin/mjai_audit.rs`
- Burn + `tch` / libtorch-compatible runtime via the CUDA base image

This image does **not** add Jupyter, VS Code server, or RStudio.

## Build locally

From the repo root:

```bash
docker build -f docker/train/Dockerfile -t hydra:local .
```

## Basic smoke check

No arguments should print the binary usage contract:

```bash
docker run --rm hydra:local
```

## Runtime contract

The container entrypoint is `train`, and it expects one YAML config argument:

- `train <config.yaml>`
- mounted config/data/output paths instead of baking datasets into the image

Hydra's current behavioral-cloning loader supports either:

- a flat MJAI directory of `.json` / `.json.gz` files
- a direct `.tar.zst` MJAI archive path

Keep mounted container paths aligned with the config:

- `data_dir: /data` for an extracted MJAI directory
- `data_dir: /data/dataset.tar.zst` for a mounted archive file
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

Your YAML config should point at the mounted container paths, for example:

```yaml
data_dir: /data
output_dir: /output
num_epochs: 1
batch_size: 32
```

For an archive-backed run, mount the archive itself and point `data_dir` at that file path.

## Publish to GHCR

Tag the image:

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

If you also want a versioned tag:

```bash
docker tag hydra:local ghcr.io/nikketryhard/hydra:0.1.0
docker push ghcr.io/nikketryhard/hydra:0.1.0
```

## Note on image publishing

If you publish the image to GHCR or another registry, make sure the package visibility and pull path match the environment that will run training.
