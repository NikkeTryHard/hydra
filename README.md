# Hydra

Hydra is Riichi Mahjong AI project focused on correct, reproducible train/eval.

Current authority is Tenhou/MJAI rules plus `hydra-engine` as executable reference. Speed and throughput matter only after correctness is proven.

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

`example.yaml` is launch/config source of truth. Keep it current when runtime, resume, validation, model/profile, checkpoint, data, PPO, or backend behavior changes.

Normal BC training uses Rust launcher and Python/PyTorch learner. Raw MJAI streaming is default; compact shards are optional cache/resume material.

T1 PPO uses MahJAX GPU rollout by default, with `torch-callback` and `rust-ort` kept as reference/compat routes.

Operator detail lives in [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md).

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

## License Boundaries

First-party crates use repo BSL 1.1 unless crate-specific license says otherwise. `hydra-engine` is vendored Apache-2.0.

`Mortal-Policy/` is AGPL. Do not copy, adapt, derive, port, link, or translate it. Do not add AGPL/GPL/LGPL dependencies.
