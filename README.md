# Hydra

Hydra is open-source Riichi Mahjong AI project. goal is reproducible, permissively released system that can approach LuckyJ-level play while staying usable by researchers and engineers who want to inspect, train, and extend it.

Hydra is still under active development. repo already contains game/runtime surface, encoders, training data formats, model/training crates, replay tooling, and CUDA/LibTorch-gated training paths. Current shipped/staged status lives in [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md).

> ## Compute support
> Research used Delta advanced computing/data resource, supported by National Science Foundation award OAC 2005572 and State of Illinois. Delta is joint effort of University of Illinois Urbana-Champaign and National Center for Supercomputing Applications.

## What Hydra is trying to do

Hydra aims to:

- train strong Riichi Mahjong policy, ultimately near [LuckyJ](https://haobofu.github.io/) strength;
- release weights under permissive license;
- support reproducible training/evaluation;
- add opponent modeling and inference-time search;
- keep compatibility surfaces explicit: action space, encoder shape, legal masks, replay identity, and checkpoint/runtime contracts.

Useful context:

- [Mortal](https://github.com/Equim-chan/Mortal) is key public comparison point, but it is AGPL. Hydra does not copy or derive from Mortal code.
- LuckyJ is long-term strength target, not claim that Hydra is already there.

## Quick start

Install/use [Pixi](https://pixi.sh/) from repo root. Pixi owns Rust toolchain, PyTorch/libtorch, `cargo-nextest`, `clang`, `mold`, and `sccache` setup.

```bash
pixi run torch-check
pixi run check
pixi run test
pixi run lint
```

Fast default commands avoid LibTorch/CUDA-heavy paths unless explicitly requested.

| Command | Use |
|---|---|
| `pixi run check` | Fast workspace compile check |
| `pixi run build` | Fast workspace dev build |
| `pixi run test` | Fast workspace tests via nextest |
| `pixi run lint` | Fast lint: anti-game scan, rustfmt, clippy |
| `pixi run nextest-list` | List default test inventory |
| `pixi run timings-check` | Emit Cargo timing report for default graph |

Heavy paths are explicit:

| Command | Use |
|---|---|
| `pixi run check-training` | CPU LibTorch training compile path |
| `pixi run check-cuda-graph` | CUDA graph compile path |
| `pixi run test-exhaustive` | All-feature test gate |
| `pixi run lint-exhaustive` | All-feature/CUDA lint gate |
| `pixi run lint-cuda-graph` | Focused CUDA graph lint |
| `pixi run build-dist` | Fat-LTO final artifact build |
| `pixi run timings-full` | Timing report for full heavy graph |

For narrow work, prefer Pixi-owned Cargo:

```bash
pixi run cargo check -p hydra-core --no-default-features --quiet
pixi run cargo nextest run -p hydra-core --lib --no-default-features --cargo-profile dev --cargo-quiet
pixi run cargo nextest run golden_aka_flags --no-default-features --cargo-profile dev --cargo-quiet
```

Use direct system `cargo` only when Pixi is unavailable. Direct Cargo can pick host PyTorch/libtorch that does not match Hydra's pinned Burn/tch stack.

## Build environment notes

Hydra's default Pixi env is supported local build surface.

Pixi config sets:

- `RUSTC_WRAPPER=scripts/rustc-wrapper.sh`
- repo-local `SCCACHE_DIR=.pixi/sccache`
- Pixi `clang` linker driver
- Pixi `mold` linker
- conda sysroot startup-symbol workaround
- PyTorch/libtorch path for `torch-sys`

Current Burn LibTorch stack requires PyTorch/libtorch `2.9.0` through `burn-tch 0.21` -> `tch 0.22` -> `torch-sys 0.22`. Do not bypass that version check unless Burn/tch requirements changed.

For clean compile measurements:

```bash
CARGO_TARGET_DIR=$HOME/tmp/hydra-compile-bench/default-check \
SCCACHE_DISABLE=1 \
pixi run check
```

Do not compare cold-cache and warm-cache numbers.

## Repository map

| Path | Purpose |
|---|---|
| `crates/hydra-engine` | Vendored Apache-2.0 Riichi rules engine |
| `crates/hydra-runtime-types` | Shared runtime action/tile rails |
| `crates/hydra-safety` | Safety primitives |
| `crates/hydra-belief-search` | Belief/search primitives |
| `crates/hydra-encoder` | Observation encoders |
| `crates/hydra-core` | Public runtime facade, simulator, action/tile API, seeding |
| `crates/hydra-data-core` | Sample DTOs and scoring helpers |
| `crates/hydra-replay-sidecar` | Replay sidecar JSONL contracts |
| `crates/hydra-replay-loader` | MJAI replay loading and sample conversion |
| `crates/hydra-sample-cache` | Parsed sample cache format/storage |
| `crates/hydra-bc-shards` | Backend-agnostic behavior-cloning shard format |
| `crates/hydra-train-types` | Shared training scalar/coordination types |
| `crates/hydra-model` | Burn neural model components |
| `crates/hydra-train-algo` | Pure Burn losses/algorithms |
| `crates/hydra-selfplay` | Self-play coordination primitives |
| `crates/hydra-search-labels` | Search-label generation |
| `crates/hydra-train-runtime` | Training CLI/config/preflight/probe contracts |
| `crates/hydra-train-exec` | Training execution over model/algo/data/runtime crates |
| `crates/hydra-train` | User-facing binaries |
| `docs/` | Current user/operator docs |
| `research/` | Research/design/evidence archive |
| `docker/train/` | Container/Kaggle/operator packaging docs |

## Where to read next

| Need | Read |
|---|---|
| Current shipped/staged state | [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) |
| Runtime/game invariants | [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) |
| Compatibility contracts | [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md) |
| Training/operator commands | [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md) |
| Container/Kaggle workflow | [`docker/train/README.md`](docker/train/README.md) |
| Hydra v1 roadmap | [`research/design/HYDRA_RECONCILIATION.md`](research/design/HYDRA_RECONCILIATION.md) |
| Long-term architecture | [`research/design/HYDRA_FINAL.md`](research/design/HYDRA_FINAL.md) |
| Research evidence summary | [`research/evidence/RESEARCH_DIGEST.md`](research/evidence/RESEARCH_DIGEST.md) |
| Benchmarks | [`research/infrastructure/ENGINE_BENCHMARKS.md`](research/infrastructure/ENGINE_BENCHMARKS.md) |
| Reproducibility/seeding | [`research/design/SEEDING.md`](research/design/SEEDING.md) |

canonical research ledger is [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl). Derived roadmap/rendered files are helpers; JSONL wins if archive views drift.

## Testing

Default:

```bash
pixi run test
```

Common focused tests:

```bash
pixi run cargo nextest run -p hydra-core --no-default-features --cargo-profile dev --cargo-quiet
pixi run cargo nextest run -p hydra-core --test golden_encoder --no-default-features --cargo-profile dev --cargo-quiet
pixi run cargo nextest run -p hydra-core --lib --no-default-features --cargo-profile dev --cargo-quiet
pixi run cargo nextest run -p hydra-train-exec --lib --no-default-features --cargo-profile dev --cargo-quiet
```

Python scripts:

```bash
uv run python -m unittest discover -s scripts/tests
```

Coverage/container/Kaggle commands live in [`docker/train/README.md`](docker/train/README.md).

For impl work: start from [`research/design/HYDRA_RECONCILIATION.md`](research/design/HYDRA_RECONCILIATION.md), confirm status in [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md), then confirm runtime contracts in [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md), [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md), and current code.

Raw files under `research/agent_handoffs/combined_all_variants/` are provenance only, not impl doctrine.

## License

Hydra first-party crates use BSL 1.1 unless crate-specific license says otherwise. `hydra-engine` is Apache-2.0 vendored upstream rules-engine code.

Do not add AGPL/GPL/LGPL dependencies. Do not copy, adapt, port, or translate AGPL code from Mortal.
