<p align="center">
<img src="assets/hydra.webp" alt="Hydra banner" width="720">
</p>

<h1 align="center">Hydra</h1>


<p align="center">
<img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-blue">
<img alt="Rust 2024" src="https://img.shields.io/badge/rust-2024-orange">
<img alt="License BSL 1.1" src="https://img.shields.io/badge/license-BSL--1.1-lightgrey">
<img alt="Training PyTorch + MahJAX" src="https://img.shields.io/badge/training-PyTorch%20%2B%20MahJAX-ee4c2c">
</p>

Hydra is Riichi Mahjong AI project focused on correct, reproducible train/eval. Its goal is to beat [LuckyJ](https://haobofu.github.io/) in strength.

## Setup

Hydra is developed on Linux. Install system basics first.

Arch/CachyOS:

```bash
sudo pacman -S --needed git curl base-devel
```

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y git curl build-essential ca-certificates
```

Install Pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Restart your shell, then check:

```bash
pixi --version
```

Clone and enter Hydra:

```bash
git clone https://github.com/NikkeTryHard/hydra.git
cd hydra
```

Let Pixi install environment:

```bash
pixi install
```

List available tasks:

```bash
pixi task list
```

Common checks:

```bash
pixi run check-lib
pixi run test-lib
pixi run gate
```

Use Pixi-owned commands from repo root. Avoid direct system `cargo` for normal work.

## Training

Start from [`example.yaml`](example.yaml), then use [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md) for launch, resume, checkpoint, validation, and T1 PPO details.

BC training:

```bash
pixi run cargo run --quiet --package hydra-train --features training --bin train -- example.yaml
```

T1 PPO uses same launcher after setting `rl.phase: ppo_control` in YAML:

```bash
pixi run cargo run --quiet --package hydra-train --features training --bin train -- example.yaml
```

When TensorBoard is enabled, launcher prints local TensorBoard URL. Run logs and checkpoints are written under `<output_dir>/stages/<stage>/runs/<run_id>/`, including `logs/train_steps.jsonl`, `logs/events.jsonl`, `logs/stdout.log`, `logs/stderr.log`, and `checkpoints/latest.pt`.

## Repository Map

- `crates/`: Rust engine, runtime, replay, shard, model, and train crates.
- `python/hydra_learner/`: Python learner package and tests.
- `docs/`: current operator/runtime docs.
- `research/`: design notes and evidence archive.
- `docker/train/`: container/Kaggle packaging.

## Read Next

- [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md): what is shipped, default, staged, or experimental.
- [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md): launch/resume/checkpoint/operator flow.
- [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md): engine/runtime behavior and rule authority.
- [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md): hard runtime/training contracts.
- [`docs/MAHJAX_PPO.md`](docs/MAHJAX_PPO.md): MahJAX PPO scope and limits.
- [`AGENTS.md`](AGENTS.md): contributor rules for this repo.

## Compute Support

<p align="center">
<img src="assets/delta.webp" alt="Delta GPU node" width="720">
</p>

This research used Delta advanced computing and data resource which is supported by National Science Foundation (award OAC 2005572) and State of Illinois. Delta is joint effort of University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications.

## License Boundaries

First-party crates use repo BSL 1.1 unless crate-specific license says otherwise. `hydra-engine` is vendored Apache-2.0.

