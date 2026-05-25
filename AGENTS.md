# AGENTS.md -- Hydra agent guide

Hydra = Riichi Mahjong AI. Goal: LuckyJ-level strength, reproducible train/eval.

## Rules

- Fix root cause at owner layer. No warnings/fallbacks/silent clamps/shims hiding failure.
- Delete obsolete paths. No stale aliases, dead branches, TODO stubs, no-op impls.
- Keep API narrow. No `pub` for tests/convenience/future guesses.
- App errors use `anyhow::Result`; library boundaries use typed errors when callers need classify.
- No `unwrap()`/`expect()` in runtime/library. Tests only when panic cannot hide assertion.
- Hot paths: avoid needless `String`, `Vec`, clone, boxing, dyn dispatch, `format!`, per-turn alloc. Reuse caller buffers/scratch.
- Determinism: explicit seeds, stable ordering, no unordered-map output dependence.
- Feature flags additive/explicit. CUDA/libtorch-heavy paths opt-in unless documented default.
- Build scripts: exact `cargo:rerun-if-changed/env-changed`; no broad invalidation.
- Unsafe needs local safety invariant. Comments explain invariant/intent only.

## Compatibility facts

- Live encoder/model input `192x34`; old `85x34` = historical baseline-prefix channels `0..84`.
- Action space fixed `46`; legal mask `[bool; 46]`.
- Riichi two-phase. Kan bridge action `42`: normal -> `Ankan`, other phases -> `Daiminkan`; inbound kan variants collapse to `42`.
- Tile kinds `0..33`; aka/red fives distinct on 136-format/action surfaces where required.
- Compact action facade is 4-player. Sanma/Kita stays engine-level, not 46-action bridge.
- Suit augmentation exactly 6 numbered-suit permutations; honors unchanged.
- Sidecar/replay/checkpoint/action contracts hard-error on mismatch. Present-but-incomplete metadata is error.
- BC runtime authority: fresh config-derived; epoch resume may reuse matching tuple; partial-epoch resume requires identical runtime.
- Python BC default backend = PyTorch. Rust/Burn path legacy/debug/advanced.
- Native arena/RL inference default = Rust + ONNX Runtime CUDA. Inputs: ONNX export dir (`policy.onnx`, `policy.json`, `parity_fixture.safetensors`) or `.pt` checkpoint auto-exported to ONNX first. Legacy Python checkpoint arena only with `--python-checkpoints`.
- Arena defaults: `--games 1024`, `--arena-batch-decisions 1024`, `--device cuda:0`, native ONNX, auto worker threads.

## Crate owners

| Crate | Owns |
|---|---|
| `hydra-engine` | vendored Apache-2.0 RiichiEnv/rules engine |
| `hydra-runtime-types` | shared tile/action/runtime rails |
| `hydra-safety` | genbutsu/suji/kabe/one-chance |
| `hydra-belief-search` | AFBS/CT-SMC/Hand-EV/search |
| `hydra-encoder` | observation/batch encoder components |
| `hydra-core` | runtime facade, action/tile API, simulator, game loop, seeding, arena glue |
| `hydra-data-core` | pure sample DTOs/scoring helpers |
| `hydra-replay-sidecar` | replay sidecar JSONL contracts |
| `hydra-replay-loader` | MJAI replay loading/sample conversion |
| `hydra-sample-cache` | parsed sample cache format/storage |
| `hydra-bc-shards` | compact BC shard format/reader/writer/manifest |
| `hydra-train-types` | training scalar/coordination types |
| `hydra-model` | Burn model components; ONNX Runtime policy loader |
| `hydra-train-algo` | pure Burn losses/algorithms |
| `hydra-selfplay` | self-play coordination |
| `hydra-search-labels` | search-label generation |
| `hydra-train-runtime` | CLI/config/preflight/probe/status; Python launcher option conversion |
| `hydra-train-exec` | execution composition over runtime/model/algo/data; shard builder impl |
| `hydra-train` | user-facing bins/dispatch glue only |

Boundary: runtime/action/encoder/game semantics before training consumers. Config/preflight/probe/status in `hydra-train-runtime`. Execution in `hydra-train-exec`. Do not dump libraries into `hydra-train`.

## Build / gates

Use Pixi from repo root. Never system Cargo for normal work. `pyproject.toml` is tool/task SSOT; legacy `pixi.toml` removed.

Fast Rust:
```bash
pixi run cargo check -p <crate> --no-default-features --quiet
pixi run scripts/nextest-quiet.sh run -p <crate> --lib --no-default-features --cargo-profile dev --cargo-quiet
pixi run scripts/nextest-quiet.sh run <test-name> --no-default-features --cargo-profile dev --cargo-quiet
```

Fast Python:
```bash
pixi run ruff format --check <paths>
pixi run ruff check <paths>
pixi run pyrefly check <paths>
pixi run scripts/pytest-quiet.sh <tests>
```

Python gates mandatory for Python changes. Ruff only formatter/import sorter/linter. Pyrefly authoritative type checker.

Pixi/libtorch facts:
- Default env covers CPU/GPU. Device selects `cpu` or `cuda:0`.
- Burn LibTorch stack: `burn-tch 0.21` -> `tch 0.22` -> PyTorch/libtorch `2.9.0`.
- Required linker/env in `pyproject.toml`: `LIBTORCH_USE_PYTORCH`, `PROTOC`, `LD_LIBRARY_PATH`, `CUDA_HOME`, `CUDA_PATH`, `RUSTC_WRAPPER`, `SCCACHE_DIR`, clang/mold, conda sysroot.
- Do not hardcode local tool/CUDA paths unless proven and documented.
- `.cargo/config.toml` local/gitignored. `Cargo.lock` committed.
- Keep `.codebase-memory/`, `target/`, `output/`, notebooks/model/cache artifacts out of commits unless explicitly refreshing.

## Python/PyTorch backend

- Python owns BC training, model/loss/optimizer/AMP/`torch.compile`/profiler/checkpoint, ONNX export.
- Rust owns replay/shards/raw stream/runtime contracts, action count, legal mask width, encoder shape, launcher orchestration, checkpoint/runtime metadata validation, native ONNX arena, RL inference.
- Python owner: `hydra_learner.cli`; train modules under `hydra_learner`. `hydra_learner/train_bc.py` and `scripts/hydra_pytorch_oracle.py` compatibility only.
- ExIt/DeltaQ/belief/mixture/opponent-hand-type not in Python default yet; use legacy Rust/Burn for advanced/debug paths.
- Public Python funcs/classes/dataclasses/configs/boundaries need explicit param/return annotations. No implicit `Any`; casts/ignores need local reason.
- Validate tensor shape/dtype/device/layout/finite/batch, `192x34`, and 46 action/mask at process/file/FFI/data/model/loss/checkpoint boundaries. Avoid expensive validation in hot loops.
- Device movement explicit. No hidden `.to(device)` deep in helpers.
- No hidden sync in hot path (`.item()`, `.tolist()`, `.cpu()`, `.numpy()`, broad synchronize) except named metric/validation/checkpoint/debug/profile boundary.
- BF16/AMP explicit. Optimizer/master/checkpoint scalar state FP32 unless parity proves otherwise.
- `torch.compile` regions pure tensor code: no I/O/logging/config/side effects/dynamic object churn.
- Randomness explicit and recorded.
- Checkpoints data-only/versioned; never pickle modules/dataloaders/closures/compiled funcs/config objects. Load safely and validate schema/runtime/shape/dtype.
- Metrics structured JSONL/TensorBoard, not prints.

## Python source map

`python/hydra_learner/hydra_learner/`:
- `cli.py`: user BC train CLI entry; keep as dispatch/config glue.
- `train_bc.py`: compatibility entrypoint only.
- `train_loop.py`: Python BC training loop owner: fit/validate/checkpoint/log cadence.
- `model.py`: HydraPolicyNet, profiles, 192x34 -> 46 policy/value/head surfaces.
- `losses.py`: BC auxiliary loss math; no I/O/config.
- `checkpoint.py`: production checkpoint schema/save/load validation.
- `checkpointing.py`: runtime checkpoint helpers around train loop.
- `export_inference.py`: `.pt` -> ONNX policy export + metadata + parity fixture for Rust arena/RL.
- `arena_eval.py`: eval CLI; native ONNX default, `.pt` auto-export, legacy Python path behind `--python-checkpoints`.
- `raw_mjai*.py`: raw MJAI streaming/FFI wrappers; Rust remains contract/source authority.
- `shard_*`, `shards.py`: compact shard manifest/decode/reader contracts.
- `batches.py`, `constants.py`: typed batch/constants surfaces; keep 192x34 and 46-action contracts centralized.
- `validation.py`: validation loop/reporting.
- `hydra_logging.py`, `metrics.py`: structured JSONL/TensorBoard metrics.
- `optim.py`: optimizer/scheduler factories.
- `rl.py`, `ppo_*`, `ach_step.py`, `step.py`: Python-side RL/step experiments; production arena/RL inference should use Rust native ONNX path.


## Tests

- Extend existing relevant test file/module. Do not create new test file for convenience.
- Public contract tests: `crates/<crate>/tests/*.rs`; subsystem tests: `src/<subsystem>/tests/*.rs`; private leaf tests: `src/<leaf>/tests.rs` only when needed.
- No inline `#[cfg(test)] mod tests {}` inside production source bodies. Do not widen visibility only for tests.
- Every behavior change needs regression test when practical.
- Test behavior/invariants, not plumbing/default strings. No flaky sleeps/timeouts.
- Assert useful state before final success/exit. Never claim integration/perf/parity unless exact path ran.

Useful invariants:
- Non-terminal state has legal action; terminal state has empty legal mask.
- `[bool; 46]` / 46 action contract stable.
- Encoder output `192x34`; baseline prefix byte/shape compatible where promised.
- Visible tile kind counts <= 4; physical wall/accounting totals 136.
- Score conservation includes riichi deposits/kyotaku.
- Suit permutation preserves scores; identity permutation byte-identical.
- Sidecar/replay/checkpoint/action contract mismatch hard-errors.
- Shard manifest/header mismatch hard-errors.

## Runtime / data safety

- Train CLI/YAML/preflight/shards/sidecars: read `docs/TRAINING_RUNBOOK.md` before edits.
- RNG/seeding: read `research/design/SEEDING.md`; preserve deterministic replay/eval.
- Perf work: use benchmark/profile evidence before durable claims; update `research/infrastructure/ENGINE_BENCHMARKS.md` only for durable claims.
- Dataset root `/home/cachybtw/Downloads/dataset_bundle/`; known corpus `/home/cachybtw/Downloads/dataset_bundle/tenhou-houou-mjai-2025`. Do not broad `find`/glob/list there (~200k files). Use exact paths.
- Infra/checkpoint/container work: read `research/infrastructure/INFRASTRUCTURE.md` and `docker/train/README.md` as needed.
- Suspicious low sample/high skip replay data: audit with `mjai_audit` before trust.
- `advanced_loss.exit`/`delta_q` positive requires matching sidecar path; validation promotion must hydrate needed labels.
- Default training/perf uses `cuda:0` when GPU exists. CPU train only explicit debug/compat.
- `example.yaml` is local BC launch SSOT. Update when launch/resume/validation/model/profile/runtime/checkpoint/data behavior changes. `training/*.yaml` only for actual launch.
- Normal Python BC launch streams raw MJAI on-the-fly from `data_dir`/`raw_mjai_data_dirs`; compact shards optional cache/resume path.
- Raw-MJAI resume is rejected until cursor resume exists; shard runs support resumable `latest.pt`.
- Python BC LR schedule uses completed optimizer step; resumed bounded cosine at step N matches uninterrupted step N.
- Current local launch shape: large, `mish_se`, `compile_max_autotune`, batch 3072, microbatch 1024, validation microbatch 1024, `cuda:0`, EMA.

## CUDA / arena profiling

- Evidence first. Attribute by Hydra timing stage before kernel names.
- Python BC timing fields: `mean_step_ms`, `samples_per_s`, `mean_fwd_loss_ms`, `mean_backward_ms`, `mean_optimizer_ms`, `mean_fetch_decode_ms`, `mean_h2d_wall_ms`.
- Preferred Python BC profiler: built-in `torch.profiler`, real workload, warmup, one steady measured step. Nsight only for CUDA API/NVTX questions.
- Current Python BC bottleneck evidence (2026-05-22, RTX 5070, raw MJAI, `compile_max_autotune`, `mish_se`): ~42.7k samples/s, ~47.9ms/step. Bottleneck backbone backward/activation-normalization, not data/H2D/optimizer/heads.
- Native ONNX arena/RL profiling: use JSON `timing` fields. 16-thread CUDA ONNX with 1024 decision batch measured ~91 games/s at 4000/8000 actual games; inference and pending/encoder roughly tied.

## Licensing

- `Mortal-Policy/` is AGPL. Never copy/adapt/derive/port/link/translate. Black-box behavior/compat ideas only.
- Do not add AGPL/GPL/LGPL deps. Allowed deps: MIT, Apache-2.0, BSD-compatible.
- `hydra-engine` vendored Apache-2.0. First-party crates use repo BSL 1.1 unless crate-specific license says otherwise.

## Docs

- Public Rust items need useful `///`; public module surfaces need `//!` summary/invariants.
- Runtime/training surface changes update owner doc: `GAME_ENGINE`, `COMPATIBILITY_SURFACE`, `CURRENT_STATUS`, `TRAINING_RUNBOOK`, `SEEDING`, or `ENGINE_BENCHMARKS`.
- Markdown terse/caveman-compressed. No `*.original.md` backups.
