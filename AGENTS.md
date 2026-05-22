# AGENTS.md -- Hydra code-agent guide

Hydra = open-source Riichi Mahjong AI. Target LuckyJ-level strength and reproducible training.

## Implementation rules

- Fix root cause at owning layer. Do not hide failure with warnings, fallback defaults, broad `Ok(())`, silent clamping, or compatibility shims.
- Delete obsolete paths. No stale aliases, dead branches, TODO stubs, no-op implementations, or fake fallbacks.
- If something cannot be turned on for production and doesnt have any value, dont leave it off as trash code in repo. Talk with user and discuss whether it should be turned on or removed.
- Keep public API narrow. Do not make items `pub` for tests, convenience, or guessed future use.
- App-level errors use `anyhow::Result`; library boundaries use typed errors (`thiserror`) where callers need classification.
- No `unwrap()`/`expect()` in library/runtime code. Tests may use them only when failure cannot obscure assertion.
- Hot paths: no needless `String`, `Vec`, heap clone, boxing, dynamic dispatch, `format!`, or per-turn allocation. Prefer borrowed slices/iterators and caller-owned buffers.
- `clone()` must be intentional. Cheap scalar/Arc refcount OK; data clone needs local reason or refactor.
- Training/data paths must be deterministic: explicit seeds, stable ordering, no unordered map iteration dependence where output matters.
- Feature flags must be additive and explicit. Default dev path stays fast; CUDA/libtorch-heavy paths opt-in through commands/features.
- Build scripts/native code must use precise `cargo:rerun-if-changed` and `cargo:rerun-if-env-changed`; no broad env/filesystem invalidation.
- Unsafe blocks require short local safety invariant. Prefer safe Rust unless cost is material and measured.
- Comments explain invariants/intent. Delete comments that restate code or became stale.

## Codebase hygiene

- One concept has one owner. Shared constants, DTOs, and runtime contracts live in owner crates; consumers import them instead of copying numbers, shapes, strings, or tuple contracts.
- Validate before allocate. Binary/cache/shard readers must bound counts, lengths, and shapes before reserving `Vec` or reading payloads; writers must enforce same bounds readers enforce.
- Optional data must be explicit. Present-but-incomplete sidecars, provenance, manifests, checkpoint metadata, and runtime tuples hard-error; only truly absent optional inputs may become `None`.
- Reusable hot-path scratch keeps capacity. Do not use `mem::take`, `clone`, or temporary `Vec` handoff in per-turn/per-sample loops unless replacement preserves amortized allocation behavior or cost is proven irrelevant.
- Public graph/tree APIs must reject invalid indices, action ids, and cycles at boundary; traversal helpers must not rely on private callers keeping structures acyclic.
- Test attributes stay attached to tests. When inserting tests near cfg-gated cases, verify old test still runs under its feature gate.
- CLI parsers must preserve documented separators and harness paths (`--`, probe-child, libtest forwarding); add focused regression tests before tightening usage errors.

## Compatibility facts not casually changed

- Live encoder/model input = `192x34`.
- Old `85x34` = historical baseline-prefix channels `0..84`, not live full encoder.
- Action space = fixed 46 actions.
- Legal action mask shape = `[bool; 46]`.
- Riichi is two-phase. Kan uses compact bridge action `42`: normal phase maps to `Ankan`, other phases to `Daiminkan`; inbound kan variants collapse to `42`.
- Tile kinds = `0..33`; aka/red fives remain distinct in 136-format/action surfaces where required.
- Hydra compact action facade is 4-player; `hydra-engine` sanma/Kita support is engine-level and not represented in 46-action bridge.
- Suit augmentation = exactly 6 numbered-suit permutations; honors unchanged.
- BC selected-runtime authority: fresh run config-derived; epoch-boundary resume may reuse matching selected-runtime tuple; partial-epoch resume requires identical runtime.
- BC loader-runtime authority stays config-derived; matching preflight cache does not override checkpoint/runtime contract.
- BC CUDA LibTorch runs default to BF16 AMP when `precision_mode` omitted; explicit `fp32` stays FP32; CPU omission stays FP32; RL/DeltaQ BF16 hard-gated.
- CUDA graph feature ships pinned staging/preallocated tensors/probes; production graph replay remains off/probe-only until Burn optimizer gradient contract permits it.
- Plain BC default backend is Python/PyTorch. It trains compact BC shards from `bc_shards_manifest_path` or streams raw MJAI from `data_dir`; raw transport defaults to pinned PyO3 with stdout fallback.

## Crate ownership

Put code in crate that owns concern.

| Crate | Owns |
|---|---|
| `crates/hydra-engine` | vendored Apache-2.0 RiichiEnv/rules engine |
| `crates/hydra-runtime-types` | shared tile/action/runtime rails |
| `crates/hydra-safety` | genbutsu/suji/kabe/one-chance safety primitives |
| `crates/hydra-belief-search` | AFBS/CT-SMC/Hand-EV/search primitives |
| `crates/hydra-encoder` | observation and batch encoder components |
| `crates/hydra-core` | public runtime facade/bridge, action/tile API, simulator, game loop, seeding, arena glue |
| `crates/hydra-data-core` | pure sample DTOs/scoring helpers |
| `crates/hydra-replay-sidecar` | replay sidecar JSONL contracts |
| `crates/hydra-replay-loader` | MJAI replay loading/sample conversion |
| `crates/hydra-sample-cache` | parsed sample cache format/storage |
| `crates/hydra-bc-shards` | backend-agnostic BC shard format/reader/writer/manifest |
| `crates/hydra-train-types` | training scalar/coordination types |
| `crates/hydra-model` | Burn neural model components |
| `crates/hydra-train-algo` | pure Burn losses/algorithms |
| `crates/hydra-selfplay` | self-play coordination primitives |
| `crates/hydra-search-labels` | search-label generation |
| `crates/hydra-train-runtime` | CLI/config/preflight/probe/status contracts |
| `crates/hydra-train-exec` | execution composition over runtime/model/algo/data; shard builder impl |
| `crates/hydra-train` | user-facing binaries/package entrypoint; keep bin code as arg/env dispatch glue |

Boundary rules:
- `hydra-train` must not become library dumping ground.
- CLI/config/preflight/probe/status contract code belongs in `hydra-train-runtime`.
- Execution over model/algo/data belongs in `hydra-train-exec`.
- Runtime/action/encoder/game semantics belong before training consumers.

## Build and validation

Use Pixi from repo root. Do not use system Cargo for normal work; Pixi pins Rust, libtorch/PyTorch, clang/mold, sccache, protobuf, and linker env.

Core gates: read `pyproject.toml` (`tool.pixi`, Ruff, Pyrefly, pytest); legacy `pixi.toml` intentionally removed.

Compatibility aliases: `build-fast` -> `build-release`; `nextest` -> `test`.

Fast loops:

```bash
pixi run check-lib
pixi run test-lib
pixi run check
pixi run test
pixi run cargo check -p <crate> --no-default-features --quiet
pixi run scripts/nextest-quiet.sh run -p <crate> --lib --no-default-features --cargo-profile dev --cargo-quiet
pixi run scripts/nextest-quiet.sh run <test-name> --no-default-features --cargo-profile dev --cargo-quiet
pixi run scripts/nextest-quiet.sh run -p <crate> <ignored-test> --features <feature> --no-default-features --no-capture -- --ignored
```

Prefer faster nextest over cargo test. When invoking nextest directly, use `scripts/nextest-quiet.sh` through Pixi instead of `cargo nextest` to store full output in `target/nextest-quiet-output.log` and print only failure context.

Python fast loops (new PyTorch backend):

```bash
pixi run fmt-python-check
pixi run lint-python
pixi run typecheck-python
pixi run test-python
pixi run test-python-cuda  # only when task touches CUDA/compile path
```

Python gates are mandatory for Python changes. Tool deps/config live in `pyproject.toml`; do not add parallel Python tool config files unless tool requires it.
Pixi task surface stays small. Default env owns top-level tasks; probe envs (`py-train`, `py-train-torch212-cu126`, `py-train-torch212-nightly-cu128`) intentionally expose no duplicate task names. Use top-level aliases:

```bash
pixi run python-bc-train          # stable py-train torch 2.11 cu128
pixi run python-bc-train-cu126    # torch 2.12 cu126 target-machine probe
pixi run python-bc-train-nightly  # torch 2.12 nightly cu128 local probe
pixi run torch-check
pixi run torch-check-cu126
pixi run torch-check-nightly
```

Removed duplicate/noisy aliases (`check-all-targets`, `build-dist`, `test-release`, `timings-*`, `nextest-list*`, direct `rustc-wrapper`) unless needed again by real operator workflow.

## Pixi/libtorch/tooling contract

- Single default Pixi env covers CPU/GPU. Hydra config selects `device: cpu` or `device: cuda:0`.
- Current Burn LibTorch stack uses `burn-tch 0.21` -> `tch 0.22` -> `torch-sys 0.22`; requires exact PyTorch/libtorch `2.9.0`.
- Required env/linker settings live in `pyproject.toml` Pixi activation: `LIBTORCH_USE_PYTORCH`, `PROTOC`, `LD_LIBRARY_PATH`, `CUDA_HOME`, `CUDA_PATH`, `RUSTC_WRAPPER`, `SCCACHE_DIR`, Pixi clang/mold, conda sysroot `libc_nonshared.a`.
- Do not hardcode `/usr/bin/clang`, `/usr/bin/mold`, user-local tool paths, or local CUDA paths unless Pixi link failure is proven and documented.
- `.cargo/config.toml` is local-only/gitignored.
- `Cargo.lock` stays committed.
- `.codebase-memory/`, `target/`, `output/`, notebooks payloads, model/cache artifacts stay out of normal commits unless task explicitly refreshes them.

## Burn dependency decisions

- Burn stack is patched locally in `third_party/burn`: `burn`, `burn-autodiff`, `burn-backend`, `burn-tch`, `burn-flex`, `burn-ndarray`, `burn-optim` at `0.21.0`.
- Current default plain-BC backend is Python/PyTorch through Rust launcher (`py-train`, torch `2.11.0+cu128`). It supports compact BC shards and raw MJAI (`pinned_pyo3` default, `stdout` fallback). Probe env `py-train-torch212-cu126` pins torch `2.12.0+cu126` + torchvision `0.27.0+cu126` for CUDA 12.6 machines; local RTX 5070 `sm_120` cannot execute cu126 wheels, so throughput must be measured on target hardware. Rust/Burn remains legacy/reference path for advanced modes and debugging; keep Burn stack patched for compatibility.
- keep optimizer `.bin` contract unless full resume parity proof passes.
- Keep Burn Adam for legacy Rust/Burn path. Python BC uses AdamW in its data-only checkpoint contract. AdamW/AMSGrad/Adan remain Rust/Burn fresh-run experiments only; Muon requires parameter groups and is unsafe for global Hydra params; LBFGS does not fit streaming BC/RL. None fixes CUDA graph replay because Burn `GradientsParams` + module mapping remains blocker.
- For profiling, keep Hydra timings + NVTX + Nsight Systems/Compute.


## Python/PyTorch backend rules

- Python is default plain BC training backend for compact shards and raw MJAI. Rust remains source of truth for replay/shards/raw stream/runtime contracts, action count, legal mask width, encoder shape, CLI orchestration, and checkpoint/runtime metadata validation.
- Use Python for BC model/loss/optimizer/AMP/`torch.compile`/profiler/checkpoint. Keep Rust data/orchestration contracts narrow and explicit. ExIt/DeltaQ/belief/mixture/opponent-hand-type are not supported in Python default yet; use legacy Rust/Burn only for those advanced modes or debugging.
- Ruff format/check, Pyrefly, and pytest are required gates for Python code. Do not defer tool setup; Python tech debt compounds quickly.
- Ruff is only formatter/import sorter/linter unless project policy changes. Do not add Black/isort/Flake8 stacks beside Ruff.
- Pyrefly is authoritative Python type checker. New Python files must pass Pyrefly. If another checker is run, treat disagreement as review input, not reason for broad suppressions.
- All public Python functions, methods, dataclasses, configs, dataset/batch objects, model/loss boundaries, optimizer factories, checkpoint readers/writers, and Rust/Python boundary code require explicit parameter and return annotations.
- No implicit `Any`, bare containers, or untyped `Callable` in checked code. `Any`, `cast`, `# type: ignore`, `# pyrefly: ignore`, and Ruff `noqa` require exact diagnostic code plus short local reason.
- Do not globally silence missing imports or missing stubs. Add narrow package-scoped stub/allowlist with owner and removal plan when dependency types are incomplete.
- Tensor annotations are not tensor contracts. Validate shape, dtype, device, layout/contiguity, finite values, batch dimension, `192x34` encoder shape, and 46-wide action/legal-mask surfaces at process/file/FFI/data/model/loss/checkpoint boundaries.
- Boundary validation should be cheap and deliberate. Do not run Pydantic or expensive Python validation inside per-batch/per-microbatch hot loops; validate once, convert to typed runtime objects, then run hot code allocation-aware.
- Device movement is explicit. No hidden `.to(device)` deep in helpers; caller owns placement. function that moves tensors says so in its API/name and tests assert resulting device.
- No hidden synchronization in training hot paths: `.item()`, `.tolist()`, `.cpu()`, `.numpy()`, tensor `print`, and broad `torch.cuda.synchronize()` are forbidden except at named metric, validation, checkpoint, debug, or profiling boundaries with measured sync cost.
- BF16/AMP is explicit. Use `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)` only where intended. Optimizer/master/checkpoint scalar state stays FP32 unless exact resume/parity tests prove another contract.
- `torch.compile` regions must be pure tensor code: no file I/O, logging, config parsing, Python side effects, data-dependent Python branches, or dynamic object churn. Use compile diagnostics and fail tests on unexpected graph breaks/recompiles.
- Randomness is explicit and recorded: seed Python, NumPy, PyTorch CPU/CUDA as applicable; record seed, deterministic flags, PyTorch/CUDA/cuDNN versions, device, precision, compile mode, and shard manifest digest.
- Checkpoints are versioned data-only contracts: save model/optimizer/scheduler/scaler/RNG/config/runtime metadata as state-dict-like primitives with schema version and manifest digest. Never pickle modules, dataloaders, closures, compiled functions, or Python config objects as production checkpoints.
- Production checkpoint load uses safe loading (`weights_only=True` where possible) and validates schema/version/runtime/shape/dtype before accepting. Unknown or partial checkpoint metadata hard-errors.
- Metrics are structured records, not ad-hoc prints. Emit JSONL/typed metrics with step, samples, batch/microbatch, losses, LR, grad norm/overflow, throughput, CUDA memory, compile counters, sync points, and stage timings.
- Tests use pytest with real tensors/files/process boundaries. No mocks for model, optimizer, dataloader, checkpoint, compile, or CUDA behavior. Use tiny real datasets and `tmp_path`; monkeypatch only env/path isolation.
- Tensor assertions use `torch.testing.assert_close` or exact equality with explicit shape/dtype/device checks and dtype-appropriate tolerances. BF16 gets separate tolerances; never claim FP32 parity unless measured.
- CUDA/BF16/compile claims require CUDA-marked tests or exact benchmark/profiler run on that path. CPU tests do not prove CUDA behavior.
- Profiling evidence needs warmup, fixed inputs, named Hydra stages, and before/after comparison. Kernel names alone are not enough; tie decisions to data/H2D/forward/loss/backward/optimizer/metrics stages.
- Training speed gains matter only after correctness and strength are preserved. Do not promote faster profile/default when it weakens validation, strength, replay correctness, or runtime contracts; keep such changes opt-in/ablation until equal-step validation and, when relevant, arena evidence prove quality-per-wall-clock improves.
## Licensing and source boundaries

- `Mortal-Policy/` is AGPL. Never copy, adapt, derive, port line-by-line, link, or translate code from it. Black-box behavior/compatibility ideas only.
- Do not add AGPL/GPL/LGPL dependencies.
- Allowed dependency families: MIT, Apache-2.0, BSD-compatible.
- `hydra-engine` is Apache-2.0 vendored upstream.
- First-party Hydra crates use repo BSL 1.1 unless crate-specific license says otherwise.

## High-risk Hydra surfaces

Extra care + targeted tests required:

- `hydra-core` action/tile/encoder/bridge/simulator/game_loop/seeding: action legality, aka, score, wall, replay determinism.
- `hydra-engine` rules state: do not fork semantics casually; preserve upstream/license boundary.
- `hydra-replay-loader`: MJAI round reset, kan legality, sample conversion, target alignment.
- `hydra-bc-shards` + `hydra-train-exec` shard builder: manifest/header/layout are training contracts; invalid metadata hard-errors, no fallback.
- Sidecars: ExIt/DeltaQ sidecars bind replay identity, checkpoint source hash/version, action/legal-mask contract, and target shape. Valid JSONL alone is not enough.
- Preflight/resume: cache fingerprint authority and checkpoint runtime contract must not be weakened.
- Checkpoints: digest mismatch hard-fails for gate checkpoints; corrupt `latest` may try previous only while reporting every attempted path. Optimizer state stays fp32 even when model weights are bf16.
- CUDA/BF16: do not claim overlap, graph replay, or mixed-precision parity unless that exact path was measured.

## Tests

Default: extend existing relevant test file/module. Do not create new test file because convenient.

Placement:
- Public contract tests: `crates/<crate>/tests/*.rs`; public API only.
- Subsystem white-box tests: `src/<subsystem>/tests/*.rs`; real `pub(crate)`/`pub(super)` subsystem API.
- Leaf private tests: `src/<leaf>/tests.rs`; only when private funcs/types/fields/constants are required.

Rules:
- No inline `#[cfg(test)] mod tests { ... }` blocks inside production source bodies.
- Do not widen visibility only for tests.
- Every behavioral change needs regression test when practical, preferably one that fails before fix.
- Test behavior/invariants, not impl plumbing or current default strings.
- No flaky sleeps/timeouts as synchronization; wait on deterministic condition/event/fake input.
- Assert useful output/state before final success/exit assertion so failures explain bug.
- Never claim integration/perf/parity unless that exact path was run or measured.

Useful invariants for runtime/data changes:
- Non-terminal state has at least one legal action.
- Terminal state has empty legal mask.
- Legal mask/action mapping stays `[bool; 46]` / 46 action contract.
- Encoder output stays `192x34`; baseline prefix stays byte/shape compatible where promised.
- Tile kind counts never exceed 4 visible; physical wall/accounting totals 136.
- Score conservation includes riichi deposits/kyotaku.
- Suit permutation preserves scores; identity permutation is byte-identical.
- Sidecar mismatch on replay/checkpoint/action contract hard-errors.
- Shard manifest/header mismatch hard-errors.

## Runtime/training data safety

- Train CLI/YAML/preflight/shards/sidecars: read `docs/TRAINING_RUNBOOK.md` before editing.
- RNG/seeding changes: read `research/design/SEEDING.md`; preserve deterministic replay/eval behavior.
- Performance work: use auto benchmark first, then read/update `research/infrastructure/ENGINE_BENCHMARKS.md` if results become durable claims.
- Do not use `find`/broad glob discovery under `/home/cachybtw/Downloads/dataset_bundle/tenhou-houou-mjai-2025` or other raw dataset folders. They contain many replay files; broad enumeration can eat RAM and stall tools. Use exact known path directly in train/audit commands.
- Infrastructure/checkpoint/container work: read `research/infrastructure/INFRASTRUCTURE.md` and `docker/train/README.md` as applicable.
- Before trusting replay/shard/sidecar data with suspicious low sample count/high skip count, audit with `mjai_audit`; use failure inventory or focused loader tests for specific bad replays.
- If `advanced_loss.exit` or `advanced_loss.delta_q` is positive, matching sidecar path is required; validation promotion must hydrate labels when gates require them.

- Default training/perf runs use `device: cuda:0` (or `HYDRA_TRAIN_DEVICE=cuda:0`) when GPU exists. CPU train is super slow; use CPU only for explicit CPU-debug/compat checks. GPU accelerates model forward/backward/optimizer/H2D; raw replay, BC-shard decode, sample collation, and materialization still run on CPU workers.

- Preferred Python BC production training/perf shape on this machine: raw MJAI through pinned PyO3, `batch=2048`, `microbatch=1024`, `--python-variant compile_max_autotune`, default `python_residual_profile: mish_se`, `device cuda:0`. `compile_max_autotune` is canonical for long same-shape production runs because it changes TorchInductor kernel choice only, not model math. Use `compile_default` only for smoke/short debug where compile latency dominates. Treat sub-1% candidate differences as noise; choose smallest candidate within noise margin unless repeated long runs prove material gain.

### CUDA profiling quick start

- Evidence first. Do not optimize BC CUDA from kernel names alone; attribute by Hydra stage/source first.
- Preferred Python BC baseline: raw MJAI dataset folder through pinned PyO3 stream, not prebuilt BC shards. Use `/home/cachybtw/Downloads/dataset_bundle/tenhou-houou-mjai-2025`; `--raw-mjai-transport pinned_pyo3`; `batch=2048`; `microbatch=1024`; `compile_max_autotune`; default `mish_se`; CUDA BF16 autocast. BC-shard timing is experimental/diagnostic only, not primary baseline.
- Normal timing: run Python BC once without profiler. Read JSON summary fields: `mean_step_ms`, `samples_per_s`, `mean_fwd_loss_ms`, `mean_backward_ms`, `mean_optimizer_ms`, `mean_fetch_decode_ms`, `mean_h2d_wall_ms`. Compare data/H2D/forward+loss/backward/optimizer before any kernel-level claim.
- Preferred focused profiler for Python BC: use built-in scheduled `torch.profiler`, not Nsight. Keep real workload shape (`raw_mjai pinned_pyo3`, `batch=2048`, `microbatch=1024`, `warmup=10`, `steps=200`) and capture one steady measured step, e.g. `--torch-profiler-trace /home/cachybtw/tmp/hydra-profile/trace.json --torch-profiler-start-step 100 --torch-profiler-stop-step 101`. This preserves compile/warmup/queue behavior and avoids Nsight full-process slowdown.
- Parse profiler trace by GPU category/name buckets. Attribute to Hydra stage with JSON timing first; then use kernels to split dominant stage (conv fprop/dgrad/wgrad, Mish+GroupNorm, GroupNorm, memcpy, optimizer, loss/head). Do not optimize from raw kernel names alone.
- Nsight is fallback only for CUDA API/NVTX questions that `torch.profiler` cannot answer. Avoid full-process `nsys` on `compile_default`: it can time out before capture and perturb workload. If forced, use `--capture-range=cudaProfilerApi --capture-range-end=stop --wait=primary`, no sampling, tiny active range, then export/query reports immediately.
- Current raw-MJAI Python bottleneck evidence (2026-05-22, RTX 5070, `compile_default`, `batch=2048`, `microbatch=1024`, 200 steps, one steady-step torch trace): baseline `~40.3k samples/s`, `~50.9ms/step`; fetch/decode `~0.002ms`, H2D `~0.10ms`; forward+loss `~15.0ms`, backward `~34.7ms`, optimizer `~0.75ms`. `compile_max_autotune` with same `mish_se` measured `~42.7k samples/s`, `~47.9ms/step` and is canonical for production training. Steady-step GPU buckets: fused Mish+GroupNorm `~13.9ms`, conv weight-grad `~9.5ms`, conv data-grad `~8.1ms`, conv forward `~8.1ms`, GroupNorm `~4.9ms`, H2D memcpy `~1.1ms`, optimizer `~0.4ms`, loss/head `~0.05ms`. Biggest owner: backbone backward/activation-normalization, not data/H2D/optimizer/heads.

### Local throughput baseline

- Preferred baseline uses raw MJAI dataset folder with pinned PyO3 stream. Do not hand-write YAML.
- Do not `find`/glob inside dataset folder. Pass exact path directly.
- Python pinned raw-stream baseline:
```bash
pixi run -e py-train python-bc-train -- \
  --raw-mjai-data-dir /home/cachybtw/Downloads/dataset_bundle/tenhou-houou-mjai-2025 \
  --raw-mjai-transport pinned_pyo3 \
  --raw-mjai-worker-threads 20 \
  --raw-mjai-prefetch-batches 2 \
  --raw-mjai-queue-bound 8 \
  --raw-mjai-max-games 5000 \
  --variant compile_max_autotune \
  --batch 2048 \
  --microbatch 1024 \
  --warmup 10 \
  --steps 200 \
  --out /home/cachybtw/tmp/hydra-py-pinned-baseline/result.json \
  --quiet
```
- 2026-05-22 RTX 5070 Python pinned PyO3 refs (`compile_max_autotune`, `mish_se`, `batch=2048`, `microbatch=1024`, `warmup=10`, `steps=200`): GPU train `~42.7k samples/s`; end-to-end `~42.6k samples/s`; mean step `~47.9ms`; fetch/decode `~0.003ms`; H2D wall `~0.12ms`; compile/autotune overhead varies by cache/run and is larger than `compile_default`. Treat driver/thermal/codegen drift as noise unless repeated.
- Torch 2.12 cu126 probe env: `pixi run -e py-train-torch212-cu126 python-bc-train -- ...` or `pixi run -e py-train-torch212-cu126 torch-check`. Exact pins: `torch==2.12.0+cu126`, `torchvision==0.27.0+cu126`, index `https://download.pytorch.org/whl/cu126`. Local RTX 5070 cannot benchmark it because cu126 wheels support up to `sm_90`, not `sm_120`; benchmark on CUDA 12.6 target machine with same raw-MJAI command.
- Torch 2.12 nightly cu128 local probe env: `py-train-torch212-nightly-cu128`, pins `torch==2.12.0.dev20260329+cu128`, `torchvision==0.26.0.dev20260329+cu128`. Measured on RTX 5070 same raw-MJAI run: `~43.9k samples/s`, `~46.6ms/step`; `TORCHINDUCTOR_MAX_AUTOTUNE_DEFER_LAYOUT_FREEZING=1` slightly best and avoids huge compile cost in this run (`~1.7s` vs `~50.7s`). Nightly is probe-only, not production default.
- BC-shard runs are experimental/diagnostic only. They are useful for shard reader/materialization checks, not preferred training speed baseline.
Hydra binaries quick use:

- Cargo features are compile-time capability gates. Use `training` for LibTorch/Burn train/model binaries; `cuda-graph` implies `training` and checks CUDA pinned/prealloc/probe code; `data-tools` is lightweight data-conversion tooling. Omit `--features` only for bins with `Features=none`.
- Use Pixi from repo root: `pixi run cargo run --quiet --package hydra-train --features <features> --bin <bin> -- <args>`. `cargo run` samples/s from build tools is not GPU training throughput.

|Bin|Features|Purpose|Usage / key flags|
|---|---|---|---|
|`train`|`training` / `cuda-graph` for CUDA transport checks|Main config-driven train/probe/preflight/benchmark entry.|`-- <config.yaml>` normal train. `-- --list-devices`. `-- --benchmark-baseline --bench-source mjai|bc-shards|both (--data-dir DIR|--bc-shards-manifest PATH) [--output-dir DIR] [--device cuda:0]`. `-- --preflight [--device cpu|cuda:N] [--output-dir DIR] [--preflight-mode safe|unsafe] [--pf-candidate-tuples batch:ring:threads:prefetch,...] [--pf-warmup-steps N] [--pf-measure-steps N] [--pf-repetitions N] [--pf-output md]`. Config modes: `<config.yaml> --delta-q-promotion [--delta-q-baseline-checkpoint PATH]`; `<config.yaml> --probe-kind train|validation|rl_games|rl_microbatch --probe-candidate-microbatch N [--probe-warmup-steps N] [--probe-measure-steps N]`. Full training behavior lives in YAML/JSON config, not CLI overrides.|
|`mjai_audit`|`training`|Audit replay/cache ingestion; bucket loader failures.|`-- <data-dir> [--threads N] [--failure-examples N] [--failure-inventory-dir DIR]`. Use before trusting suspicious low sample/high skip corpora.|
|`build_bc_shards`|`training`|Raw MJAI/cache -> compact BC shards; validate manifests; materialization throughput proof.|Build: `-- --input <dir|archive|replay> --output-dir DIR [--manifest-name FILE] [--shard-samples N] [--train-fraction F] [--split train|val|both] [--num-threads N] [--queue-bound N] [--chunk-games N] [--max-games N] [--max-samples N] [--report-name FILE|--no-report] [--progress-jsonl FILE] [--dry-scan-only] [--resume] [--resume-dir DIR] [--max-error-examples N]`. Validate: `-- --validate-manifest PATH`. Sidecars require full triples: `--exit-sidecar PATH --exit-source-net-hash U64 --exit-source-version U32`; `--delta-q-sidecar PATH --delta-q-source-net-hash U64 --delta-q-source-version U32`.|
|`extract_timing_metrics`|none|Extract timing/throughput from step/training logs.|`-- (--step-log PATH|--training-log PATH)... [--run-id ID] [--skip-initial-rows N] [--min-global-step N] [--format json|csv]`. Use for benchmark claims.|
|`build_parsed_sample_cache`|`data-tools`|Loose MJAI `.json`/`.json.gz` -> parsed sample cache.|`-- --input <loose-file|dir> --output-dir DIR`. Archives rejected. Skips/reports per-file errors.|
|`build_replay_exit_sidecar`|`training`|Single replay + checkpoint -> ExIt JSONL sidecar + `.report.json`.|`-- --input <replay.json|.gz> --checkpoint <model_base> --output <sidecar.jsonl> --source-version U32 [--min-visits U32] [--hard-state-threshold F32] [--max-kl F32]`. CPU model load; source hash from checkpoint.|
|`build_replay_delta_q_sidecar`|`training`|Single replay + checkpoint -> DeltaQ JSONL sidecar + `.report.json`.|Same as ExIt, but `--source-version` must be `1` for train-side lookup.|

Common examples:

```bash
pixi run cargo run --quiet --package hydra-train --features training --bin build_bc_shards -- \
  --input data/mjai --output-dir output/bc-shards --num-threads 20 --max-games 5000 \
  --report-name report.json
pixi run cargo run --quiet --package hydra-train --bin extract_timing_metrics -- \
  --step-log output/bc/step_log.jsonl --skip-initial-rows 1 --format json
```

## Docs and Markdown

- Public Rust items need useful `///`; public module surfaces should have `//!` summary/invariants.
- When changing runtime/training surfaces, update owner doc: `GAME_ENGINE`, `COMPATIBILITY_SURFACE`, `CURRENT_STATUS`, `TRAINING_RUNBOOK`, `SEEDING`, or `ENGINE_BENCHMARKS`.
- Markdown stays caveman-compressed: terse, exact, low fluff, technical substance preserved.
- Do not commit `*.original.md` backups.
- `scripts/caveman-compress-hook.sh` must leave source unchanged on validation failure; report exact failure, no bluff.
