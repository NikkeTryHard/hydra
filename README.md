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

Hydra's user-facing training binary is `train`. It is behind explicit `training` feature because it pulls in Burn, LibTorch, and heavier training dependencies.

Compile training binary:

```bash
pixi run check-training
```

Run preflight benchmark:

```bash
pixi run cargo run -p hydra-train --no-default-features --features training --bin train -- --preflight --pf-candidate-tuples 1024:2:1:1 --pf-warmup-steps 10 --pf-measure-steps 100 --pf-repetitions 1 --pf-output md
```

Run default BC shard training through Rust launcher. Rust validates CLI/manifest contracts, then launches Python/PyTorch BC learner in Pixi `py-train`:

```bash
pixi run cargo run --quiet --package hydra-train --features training --bin train -- \
  --bc-shards-manifest path/to/bc_shards_manifest.json \
  --output-dir path/to/output-dir \
  --device cuda:0 \
  --python-variant compile_max_autotune \
  --python-warmup 1 \
  --python-steps 3 \
  --python-compile-fullgraph-check
```

Run config-backed training:

```bash
pixi run cargo run -p hydra-train --no-default-features --features training --bin train -- path/to/config.yaml
```

Notes:

- Config examples and operator details live in [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md).
- Plain BC with `bc_shards_manifest_path` trains compact shards. Without shard manifest, YAML `data_dir` streams raw MJAI through Python raw-MJAI input; default transport is pinned PyO3 with stdout fallback.
- Default plain-BC backend is Python/PyTorch. Rust owns replay parsing, shard building, manifest validation, and orchestration; Python owns BC model/loss/optimizer/compile/checkpoint.
- Use `bc_backend: rust_burn` or CLI `--bc-backend rust-burn` only for legacy Rust/Burn debugging or advanced labels not supported by Python default yet: ExIt, DeltaQ, belief, mixture, opponent hand type.
- `device: cpu` works for CPU Rust paths. Default Python BC expects CUDA for throughput.
- CUDA graph support is explicit and compile-checked with:

```bash
pixi run check-cuda-graph
```

## Common commands

| Command | What it does |
|---|---|
| `pixi run torch-check` | Print default-env PyTorch version, CUDA availability, and libtorch path |
| `pixi run check-lib` | Fast compile check for workspace libraries |
| `pixi run check` | Default compile check for no-heavy workspace targets |
| `pixi run build` | Fast default workspace build |
| `pixi run build-release` / `pixi run build-fast` | Release workspace build |
| `pixi run test-lib` | Fast workspace library tests |
| `pixi run test-fast` | Fast workspace test targets without benches/examples |
| `pixi run test` | Default workspace tests through nextest + Python tests |
| `pixi run lint` | Rust lint, Python lint, and Python typecheck |
| `pixi run quality` | Format check + lint + tests |
| `pixi run check-training` | Compile training CLI path |
| `pixi run check-cuda-graph` | Compile explicit CUDA graph path |
| `pixi run python-bc-train -- ...` | Run stable Python BC learner (`py-train`, torch `2.11.0+cu128`) |
| `pixi run python-bc-train-cu126 -- ...` | Run torch `2.12.0+cu126` target-machine probe |
| `pixi run python-bc-train-nightly -- ...` | Run torch `2.12` nightly cu128 local probe |
| `pixi run torch-check-cu126` / `pixi run torch-check-nightly` | Print probe-env Torch versions |
Use Pixi-owned Cargo for focused Rust commands:

```bash
pixi run cargo check -p hydra-core --no-default-features --quiet
pixi run cargo nextest run -p hydra-core --lib --no-default-features --cargo-profile dev --cargo-quiet
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
| `crates/hydra-bc-shards`, `crates/hydra-raw-mjai-ffi` | BC shard format and raw-MJAI pinned PyO3 bridge |
| `crates/hydra-model`, `crates/hydra-train-algo`, `crates/hydra-train-types` | Model/loss/coordination types |
| `crates/hydra-train-runtime`, `crates/hydra-train-exec`, `crates/hydra-train` | Training config/contracts, execution, user binaries |
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
| Container/Kaggle workflow | [`docker/train/README.md`](docker/train/README.md) |
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
