# Hydra training container

Ops doc for Hydra train image, GHCR image, Kaggle-compatible runtime artifact.

Truth split:
- training YAML/modes and preflight/runtime authority: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md)
- current shipped/staged status: [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md)
- crate/bin ownership: [`crates/hydra-train/README.md`](../../crates/hydra-train/README.md)

## Surfaces

| Surface | Use | Contract |
|---|---|---|
| Local image | build/run on local GPU host | `docker/train/Dockerfile`; entrypoint `train`; mount config/data/output |
| GHCR image | same image pulled by remote runner | tag/push `ghcr.io/nikketryhard/hydra:<tag>` |
| Kaggle compat artifact | Kaggle where local compile or host-built binary ABI fails | `scripts/build-kaggle-compatible-artifact.sh` exports `dist/kaggle-compat/` |

Image includes:
- `train` from `crates/hydra-train/src/bin/train.rs`
- `mjai_audit` from `crates/hydra-train/src/bin/mjai_audit.rs`
- Burn + `tch` / libtorch-compatible CUDA runtime

Image does not include Jupyter, VS Code server, or RStudio.

## Local image

Build from repo root:

```bash
docker build -f docker/train/Dockerfile -t hydra:local .
```

Smoke: no args should print train usage.

```bash
docker run --rm hydra:local
```

Run:

```bash
docker run --rm \
  --gpus all \
  -v /host/config.yaml:/config/train.yaml:ro \
  -v /host/mjai:/data:ro \
  -v /host/output:/output \
  hydra:local \
  /config/train.yaml
```

Archive-backed run: mount archive file and set `data_dir` to mounted file path.

## Config mount contract

Container entrypoint = `train`; invocation shape:

```bash
train <config.yaml>
```

Do not bake datasets into image. Mount config/data/output. YAML paths must be container paths, not host paths.

Replay inputs supported by current BC loader:
- loose MJAI dir with `.json` / `.json.gz`
- direct `.tar.zst` MJAI archive path

Common mounted paths:

```yaml
data_dir: /data              # dir, or /data/replays.tar.zst
output_dir: /output
num_epochs: 1
batch_size: 2048
```

## MJAI audit and failure triage

Audit before BC training, BC shard build, replay sidecar generation, or when skipped files exceed expectation.

Layered workflow:
1. `mjai_audit` checks loose dirs, loose files, or `.tar.zst` archives using same loader family as training.
2. `mjai_first_failure` isolates first bad archive entry.
3. `mjai_debug_failure` explains one failing replay file.

Commands:

```bash
cargo run -p hydra-train --bin mjai_audit -- /data/mjai --threads 16 --failure-examples 10
cargo run -p hydra-train --bin mjai_audit -- /data/replays.tar.zst --threads 8 --failure-examples 5 --failure-inventory-dir /tmp/hydra-audit-failures
cargo run -p hydra-train --bin mjai_first_failure -- /data/replays.tar.zst
cargo run -p hydra-train --bin mjai_debug_failure -- /tmp/failing_replay.json
```

`mjai_audit` summary fields:
- `loaded`: sources/archive entries parsed into games
- `skipped`: loader/archive rejects
- `samples`: training samples recovered from loaded games
- `total`: `loaded + skipped`

Failure inventories are JSONL with `source`, `identity`, `error`. Identity is loose path, `archive.tar.zst/path/inside/archive.json`, or `archive.tar.zst#entry[N]` when path inspection fails.

Read result as data-quality signal: few skips in huge corpus may be tolerable but real; high skip or low sample count means fix corpus before trusting train stats. For archive failure, isolate one entry first; do not debug opaque archive as whole.

## Coverage / regression safety

Hydra uses `cargo-llvm-cov`; day-to-day fast regression remains `cargo nextest run --release`.

Install once:

```bash
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov --locked
```

Workspace report:

```bash
./scripts/coverage.sh
```

Fast scoped coverage:

```bash
HYDRA_COVERAGE_FAST=1 \
HYDRA_COVERAGE_HTML=0 \
HYDRA_COVERAGE_LCOV=0 \
HYDRA_COVERAGE_NEXTTEST_FILTERS='-p hydra-core arena robust_opponent bridge' \
./scripts/coverage.sh
```

Useful knobs:
- `HYDRA_COVERAGE_PROFILE=release` overrides default workspace `coverage` profile.
- `HYDRA_COVERAGE_DIR=/absolute/path` changes output dir.
- `HYDRA_BUILD_JOBS` and `HYDRA_TEST_THREADS` tune build/test parallelism; defaults pin both to 16.

Default artifacts under `target/coverage/`:
- `summary.txt`, `summary.json`
- `timings.txt`, `run.log`
- sccache stats when available

Optional artifacts:
- `HYDRA_COVERAGE_HTML=1` writes `html/index.html`
- `HYDRA_COVERAGE_LCOV=1` writes `lcov.info`
- `HYDRA_COVERAGE_FAST=1` skips HTML, LCOV, and summary generation

Use coverage as safety signal, not correctness proof. Watch gaps near encoder channels, replay roundtrip, legal actions, scoring/state transitions, batch shaping, supervision gates.

## GHCR publish

```bash
docker tag hydra:local ghcr.io/nikketryhard/hydra:latest
gh auth token | docker login ghcr.io -u NikkeTryHard --password-stdin
docker push ghcr.io/nikketryhard/hydra:latest
```

Optional version tag:

```bash
docker tag hydra:local ghcr.io/nikketryhard/hydra:0.1.0
docker push ghcr.io/nikketryhard/hydra:0.1.0
```

Ensure package visibility and pull path match runtime environment.

## Kaggle-compatible artifact

Use when Kaggle cannot compile Hydra locally or cannot run host-built `train` because host binary requires newer `glibc` / `libstdc++` ABI than Kaggle exposes.

Builder strategy:
- build `train` in Ubuntu 22.04 / glibc 2.35 userspace
- source libtorch from Python PyTorch via `LIBTORCH_USE_PYTORCH=1`
- keep PyTorch `2.9.0+cu128` for current `tch 0.22.0` / `torch-sys` expectation
- install via `uv`; install `protobuf-compiler` because `tboard` build needs `protoc`

Build from repo root:

```bash
bash scripts/build-kaggle-compatible-artifact.sh
```

Exports `dist/kaggle-compat/`:
- `bin/train`
- `bin/mjai_audit`
- `bin/recompress`
- `lib/` exact shared-library closure resolved from `ldd` for shipped `bin/train`
- `runtime-manifest.json` producer-owned runtime contract
- `lib-manifest.tsv` per-library size + sha256
- `lib-summary.txt` total runtime payload + builder search roots
- `ldd-train.txt`
- `ldd-train-summary.txt`
- `abi-symbols.txt`

Validation before Kaggle upload: inspect manifest, `ldd-*`, and `abi-symbols.txt`; `train` must not require newer glibc/libstdc++ ABI than Kaggle. If ABI floor still too new, move builder image/toolchain older.

`runtime-manifest.json` is source of truth for notebook/bundle reuse of persisted Kaggle working dirs. Do not use shallow sentinel files.

Compat artifact production does not auto-update Kaggle notebook bundle. After validation, wire `bin/train`, exact matching `lib/` closure, and manifest metadata into bundle/launcher path.