# Hydra training container

This image packages Hydra's current training entrypoint as a prebuilt binary for SCNet-style registry import and GPU execution.

## What is inside

- Hydra `train` binary from `crates/hydra-train/src/bin/train.rs`
- Burn + `tch` / libtorch-compatible runtime via PyTorch 2.9.0 CUDA image
- `sudo` and `openssh-server` installed to stay closer to SCNet's documented custom-image expectations for SSH-capable container usage

This first image is intentionally training-focused. It does **not** add Jupyter, VS Code server, or RStudio.

## Build locally

From the repo root:

```bash
docker build -f docker/train/Dockerfile -t hydra-train:local .
```

## Basic smoke check

No arguments should print the binary usage contract:

```bash
docker run --rm hydra-train:local
```

## Runtime contract

The binary expects:

- one JSON config argument: `train <config.json>`
- `HYDRA_TRAIN_DEVICE=cpu|cuda|cuda:<index>`
- mounted config/data/output paths instead of baking datasets into the image

Hydra's current behavioral-cloning loader reads a **flat** MJAI directory of `.json` / `.json.gz` files.

## Example run

```bash
docker run --rm \
  -e HYDRA_TRAIN_DEVICE=cuda:0 \
  -v /host/config:/config:ro \
  -v /host/mjai:/data:ro \
  -v /host/output:/output \
  hydra-train:local \
  /config/train.json
```

Your JSON config should point at the mounted container paths, for example:

```json
{
  "data_dir": "/data",
  "output_dir": "/output",
  "num_epochs": 1,
  "batch_size": 32
}
```

## Publish to GHCR

Tag the image:

```bash
docker tag hydra-train:local ghcr.io/nikketryhard/hydra-train:latest
```

Log in if needed:

```bash
gh auth token | docker login ghcr.io -u NikkeTryHard --password-stdin
```

Push:

```bash
docker push ghcr.io/nikketryhard/hydra-train:latest
```

If you also want a versioned tag:

```bash
docker tag hydra-train:local ghcr.io/nikketryhard/hydra-train:0.1.0
docker push ghcr.io/nikketryhard/hydra-train:0.1.0
```

## SCNet note

SCNet's external-registry flow expects a public image pull path. After pushing to GHCR, make sure the package visibility is public before trying to import it into SCNet.
