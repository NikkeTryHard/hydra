# Hydra

Hydra is open-source Riichi Mahjong AI project.

goal is reproducible system that can train, evaluate, and eventually play near LuckyJ-level mahjong while staying inspectable by researchers and engineers. Hydra is still under active development: runtime/game surfaces, encoders, replay tooling, training data formats, model crates, and LibTorch/CUDA-gated training paths already exist, but not every research path is default-on.

Current shipped/staged status lives in [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md).

> ## Compute support
> Research used Delta advanced computing/data resource, supported by National Science Foundation award OAC 2005572 and State of Illinois. Delta is joint effort of University of Illinois Urbana-Champaign and National Center for Supercomputing Applications.

## Quick start

### 1. Install system basics

Hydra is developed on Linux. Install Git, curl, and C/C++ toolchain first.

Arch/CachyOS:

```bash
sudo pacman -S --needed git curl base-devel
```

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y git curl build-essential ca-certificates
```

### 2. Install Pixi

Pixi provides Hydra's Rust toolchain, Python, PyTorch/libtorch, `cargo-nextest`, clang, mold, protobuf, and sccache environment.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Restart your shell, then check:

```bash
pixi --version
```

### 3. Clone Hydra

```bash
git clone https://github.com/NikkeTryHard/hydra.git
cd hydra
```

### 4. Let Pixi install the environment

First Pixi command downloads pinned tools and Python packages. This can be large because PyTorch/libtorch is included.

```bash
pixi run torch-check
```

Expected output includes default-env PyTorch version, CUDA visibility, and libtorch path. Production Python BC runs through separate `py-train` env with torch `2.11.0+cu128`.

### 5. Compile

Fast library-only check:

```bash
pixi run check-lib
```

Default workspace check:

```bash
pixi run check
```

Build default binaries/libraries:

```bash
pixi run build
```

### 6. Run tests

Fast library tests:

```bash
pixi run test-lib
```

Default workspace tests:

```bash
pixi run test
```

### 7. Run lint before sending changes

```bash
pixi run lint
```

## Running Hydra

Hydra's user-facing training binary is `train`. It is glue behind explicit `training` feature: parse/env-dispatch/delegate only. CLI/config conversion lives in `hydra-train-runtime::config` + `hydra-train-runtime::config::python`; execution lives in `hydra-train-exec`.

Public entrypoints:

```bash
pixi run check-training
pixi run cargo run -p hydra-train --no-default-features --features training --bin train -- --preflight --pf-candidate-tuples 1024:2:1:1 --pf-warmup-steps 10 --pf-measure-steps 100 --pf-repetitions 1 --pf-output md
pixi run cargo run -p hydra-train --no-default-features --features training --bin train -- path/to/config.yaml
pixi run check-cuda-graph
```

Operator details and YAML examples live in [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md). Compatibility contracts live in [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md).

Default plain-BC backend is Python/PyTorch through Rust launcher. Rust owns replay parsing, shard building, manifest validation, launcher glue, and config conversion via `hydra-train-runtime::config::python`; Python owner is `hydra_learner.cli`. Use `bc_backend: rust_burn` or CLI `--bc-backend rust-burn` only for feature-gated legacy/debug or advanced labels not supported by Python default yet: ExIt, DeltaQ, belief, mixture, opponent hand type.

Raw-MJAI training streams explicit `raw_mjai_data_dirs` when set, otherwise `data_dir`. Compact shards are optional cache/resume path via `bc_shards_manifest_path`. `output_dir` is campaign root; run artifacts live under `stages/<stage>/runs/<run_id>/` with run-local `logs/`, `checkpoints/`, TensorBoard events, and `python_learner_result.json`.

Default stage names / implemented phase mapping: `bc_baseline`, `T1_ppo_control`, `T2_direct_sampled_ach`, `T3_drda_residual_ach`, `T4_pbrs_beta_sweep`, `T5_exit_auxiliary`, `T6_deltaq_experiment`, `T7_population_window`. Top-level `stage:` overrides layout stage; `run_name:` overrides run id.

Background Python runs write run-local `train.pid`, redirect stdout/stderr to `logs/`, and print `tail -f <run_dir>/logs/train_steps.jsonl` probe. TensorBoard starts on first free port at or above configured port.

CUDA graph support is explicit and compile-checked with `pixi run check-cuda-graph`.

## Common commands

`pyproject.toml` `[tool.pixi.tasks]` is command SSOT. Quiet gates print nothing on success; on warnings/errors they print captured diagnostics only.

| Command | What it does |
|---|---|
| `pixi run gate` | One-command default gate: format check + lint + Rust/Python tests |
| `pixi run gate-full` | `gate` plus all-feature Rust tests |
| `pixi run check` | Quiet default compile check for no-heavy workspace targets |
| `pixi run check-lib` | Quiet fast compile check for workspace libraries |
| `pixi run test` | Quiet default Rust + Python tests |
| `pixi run test-lib` | Quiet fast Rust library tests |
| `pixi run lint` | Quiet fast lint gate; includes staged Markdown compression and anti-game scan |
| `pixi run quality` | Compatibility alias for `pixi run gate` |
| `pixi run build` | Fast default workspace build |
| `pixi run build-release` | Release workspace build |
| `pixi run check-training` | Compile training CLI path |
| `pixi run check-cuda-graph` | Compile explicit CUDA graph path |
| `pixi run train-cuda-shards -- <config.yaml>` | Run CUDA BC shard training with explicit cuda-graph feature path |
| `pixi run python-bc-train -- ...` | Run stable Python BC learner (`py-train`, torch `2.11.0+cu128`) |
| `pixi run python-bc-train-cu126 -- ...` | Run torch `2.12.0+cu126` target-machine probe |
| `pixi run python-bc-train-nightly -- ...` | Run torch `2.12` nightly cu128 local probe |
| `pixi run torch-check` / `torch-check-cu126` / `torch-check-nightly` | Print selected Torch environment info |

For focused Rust commands, keep Cargo inside Pixi:

```bash
pixi run cargo check -p hydra-core --no-default-features --quiet
pixi run scripts/nextest-quiet.sh run -p hydra-core --lib --no-default-features --cargo-profile dev --cargo-quiet
```

Avoid direct system `cargo` for normal Hydra work. It can pick host PyTorch/libtorch that does not match Hydra's pinned stack.

## Repository map

| Path | Purpose |
|---|---|
| `crates/hydra-engine` | Riichi rules engine |
| `crates/hydra-runtime-types` | Shared action/tile/runtime types |
| `crates/hydra-core` | Public runtime facade, simulator, action/tile API, seeding |
| `crates/hydra-encoder`, `crates/hydra-safety`, `crates/hydra-belief-search` | Encoder, safety, belief/search impl crates |
| `crates/hydra-data-core`, `crates/hydra-replay-loader`, `crates/hydra-replay-sidecar`, `crates/hydra-sample-cache` | Sample DTOs, replay loading, sidecars, parsed cache |
| `crates/hydra-bc-shards`, `crates/hydra-raw-mjai-pyo3` | BC shard format and raw-MJAI pinned PyO3 bridge |
| `crates/hydra-model`, `crates/hydra-train-algo`, `crates/hydra-train-types` | Model/loss/coordination types |
| `crates/hydra-train-runtime`, `crates/hydra-train-exec`, `crates/hydra-train` | Training config/contracts including Python option conversion, execution, train-bin glue |
| `python/hydra_learner` | Default Python/PyTorch BC learner |
| `docs/` | Current user/operator docs |
| `research/` | Research notes, design docs, evidence archive |
| `docker/train/` | Container/Kaggle/operator packaging docs |

## Key docs

| Need | Read |
|---|---|
| Current shipped/staged state | [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) |
| Training configs and operator commands | [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md) |
| Runtime/game behavior | [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) |
| Compatibility contracts | [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md) |
| Container/Kaggle workflow | `docker/train/` Dockerfiles |
| Hydra v1 roadmap | [`research/design/HYDRA_RECONCILIATION.md`](research/design/HYDRA_RECONCILIATION.md) |
| Long-term design ceiling | [`research/design/HYDRA_FINAL.md`](research/design/HYDRA_FINAL.md) |

## Compatibility facts

These are current public/runtime contracts:

- live encoder/model input: `192x34`
- action space: 46 actions
- legal action mask shape: `[bool; 46]`
- tile kinds: `0..33`
- suit augmentation: 6 numbered-suit permutations, honors unchanged
- old `85x34` encoder shape is historical baseline-prefix compatibility, not live full encoder

## License and source boundaries

Hydra first-party crates use Business Source License 1.1 unless crate says otherwise. `hydra-engine` is vendored Apache-2.0 rules-engine code.

[Mortal](https://github.com/Equim-chan/Mortal) is important public comparison point, but it is AGPL. Hydra does not copy, adapt, port, translate, or link Mortal code.

Do not add AGPL/GPL/LGPL dependencies.
