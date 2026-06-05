# AGENTS.md -- Hydra agent guide

Hydra = Riichi Mahjong AI. Goal: LuckyJ-level strength with reproducible train/eval.

## Philosophy

- Correctness beats speed, throughput, convenience, and cleverness.
- Speed matters only after behavior is perfectly correct and verified.
- Fix root cause at owner layer. Do not hide failures with warnings, silent clamps, fallback paths, shims, or TODO stubs.
- Current rules SSOT: Tenhou/MJAI rules plus `hydra-engine` as executable authority.
- 2025 Tenhou corpus SSOT: `/home/cachybtw/Downloads/dataset_bundle/tenhou-houou-mjai-2025`.
- Hydra engine strict audit target for that corpus: all valid, no errors, no false positives, and little to no false negatives. Do not broad list/glob that dataset; use exact paths or owner scanners.
- Test wall time matters. If focused test/gate takes unexpectedly long, treat that as bug or bad test shape; investigate and fix cause.
- `example.yaml` is training launch/config SSOT. Update it whenever runtime, launch, resume, validation, model/profile, checkpoint, data, PPO, or backend behavior changes and example can reasonably show current intended shape.
- Delta hardware/setup note SSOT: `delta_docs/README.md`. Read it before Delta login, storage, Pixi/cache, dataset, Slurm, or A100 work.

## Rust Rules

- No `unwrap()`/`expect()` in runtime/library code. Tests only when panic cannot hide assertion quality.
- Keep APIs narrow. Do not make items `pub` for tests, convenience, or future guesses.
- Hot paths: avoid needless `String`, `Vec`, clone, boxing, dyn dispatch, `format!`, and per-turn allocation. Reuse caller buffers/scratch.
- Determinism: explicit seeds, stable ordering, no unordered-map output dependence.
- Feature flags are additive and explicit. CUDA/libtorch-heavy paths stay opt-in unless documented default.
- Build scripts must use exact `cargo:rerun-if-changed` / `cargo:rerun-if-env-changed`; no broad invalidation.
- Unsafe code needs local safety invariant comment.
- App errors use `anyhow::Result`; library boundaries use typed errors when callers need classification.

## Python Rules

- Ruff is formatter/import sorter/linter. Pyrefly is authoritative type checker. Do not bypass either to make code pass.
- Public Python funcs/classes/dataclasses/configs/boundaries need explicit param/return annotations.
- No raw or implicit `Any` in normal logic. Dynamic JAX/MahJAX/PyO3/Torch/JSON/checkpoint/MJAI payloads need boundary aliases, Protocols, or validators.
- Do not use broad Ruff/Pyrefly ignores. `type: ignore` needs local reason.
- Prefer validation helpers or Protocols over `cast(...)` when they make type true.
- Validate tensor shape/dtype/device/layout/finite/batch, `192x34`, and `46` action/mask at process/file/FFI/data/model/loss/checkpoint boundaries. Avoid hot-loop validation.
- Device movement is explicit. Do not move `.cpu()`, `.item()`, `.numpy()`, `.to(device)`, or JAX host sync across PPO/MahJAX hot paths during refactor.
- Keep JAX/MahJAX imports isolated from default Python import paths; optional dependency failures stay local to MahJAX-only commands.
- Checkpoints are data-only/versioned. Never pickle modules, dataloaders, closures, compiled funcs, or config objects.

## Pixi Commands

Use Pixi from repo root. Never system Cargo for normal work. `pyproject.toml` `[tool.pixi.tasks]` is command/tool SSOT.

```bash
pixi run gate
pixi run gate-full
pixi run check
pixi run check-lib
pixi run test
pixi run test-lib
pixi run lint
```

Focused commands:

```bash
pixi run cargo check -p <crate> --no-default-features --quiet
pixi run scripts/nextest-quiet.sh run -p <crate> --lib --no-default-features --cargo-profile dev --cargo-quiet
pixi run scripts/nextest-quiet.sh run <test-name> --no-default-features --cargo-profile dev --cargo-quiet
pixi run ruff format --check <paths>
pixi run ruff check <paths>
pixi run scripts/pyrefly-quiet.sh <paths>
pixi run scripts/pytest-quiet.sh <tests>
```

## Training Shapes

BC model/profile knobs:

- Input/action contract: `192x34` observations, `46` actions.
- Common long-run shape: `large`, hidden `384`, blocks `16`, SE bottleneck `96`.
- Current local BC launch shape: `mish_se`, `compile_max_autotune`, batch `3072`, microbatch `1024`, validation microbatch `1024`, `cuda:0`, EMA.
- Backbones: `conv2d_local3`, `tileformer_bias`, `convnext_tile_k7`, `global_pool_bias`.
- Residual profiles: `mish_se`, `mish_no_se`, `mish_eca`, `silu_se`, `relu_se`, `relu_no_se`, `relu_no_norm_no_se`.
- `global_pool_bias` previously hit NaN on compile path; do not use until debugged.

T1 PPO rollout backends:

- `mahjax-gpu`: current default/superior PPO rollout path for serial and depth-1 PPO.
- `torch-callback`: compatibility/reference path; preserve semantics.
- `rust-ort`: native ONNX arena/reference path; preserve semantics.
- Native ONNX arena/RL default uses Rust + ONNX Runtime CUDA. Inputs are ONNX export dir (`policy.onnx`, `policy.json`, `parity_fixture.safetensors`) or `.pt` checkpoint auto-exported first.

## Source Tree

```text
python/hydra_learner/
  src/hydra_learner/
    cli.py                  # BC CLI dispatch
    train_bc.py             # compatibility entrypoint
    arena_eval.py           # eval CLI / ONNX export handoff
    export_inference.py     # .pt -> ONNX export + parity fixture
    ppo_control.py          # PPO-control CLI implementation
    constants.py
    typing_boundaries.py
    checkpointing/          # checkpoint schema/save/load/RNG/EMA/eval/training helpers
    data/                   # batches, BC shards, raw-MJAI transport
      raw_mjai/
    mahjax/                 # MahJAX adapters, PPO rollout, replay validation tools
      replay/
      tools/
    model/                  # policy, backbones, profiles, losses, optimizer helpers
    ppo/                    # PPO config/checkpoint/rollout/train-step/math helpers
    rl_experiments/         # ACH/DRDA/population/reward/objective experiments
    telemetry/              # JSONL/TensorBoard/phase/resource telemetry
    training/               # BC config/step/loop/validation
  tests/
```

## Rust Crates

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
| `hydra-raw-mjai-pyo3` | PyO3 raw-MJAI bridge |
| `hydra-train-runtime` | CLI/config/preflight/probe/status; Python launcher option conversion |
| `hydra-train-exec` | execution composition over runtime/model/algo/data; shard builder impl |
| `hydra-train` | user-facing bins/dispatch glue only |

## Licensing

- `Mortal-Policy/` is AGPL. Never copy, adapt, derive, port, link, or translate it. Black-box behavior/compat ideas only.
- Do not add AGPL/GPL/LGPL dependencies.
- Allowed dependency licenses: MIT, Apache-2.0, BSD-compatible.
- `hydra-engine` vendored Apache-2.0. First-party crates use repo BSL 1.1 unless crate-specific license says otherwise.

## Commit Exclusions

- Never commit files under `local/` or `training/`; they are local/run artifacts even when task creates or stages them.
- Before every commit/push, verify `git ls-files local training` is empty.
- If not empty, remove them from index with `git rm -r --cached --ignore-unmatch local training`.
- Keep `.codebase-memory/`, `target/`, `output/`, notebooks/model/cache artifacts out of commits unless explicitly refreshing.
