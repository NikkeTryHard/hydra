# AGENTS.md -- Hydra code-agent guide

Hydra = open-source Riichi Mahjong AI. Target LuckyJ-level strength and reproducible training. This file is compact operating contract for agents changing code; detailed doctrine lives in linked docs.

## Truth order

For doctrine/planning, read only as deep as needed:

1. `README.md` -- repo routing.
2. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` -- canonical research ledger.
3. Derived archive views as helpers only; JSONL wins on drift.
4. `research/design/HYDRA_RECONCILIATION.md` -- Hydra v1 active path.
5. `research/design/HYDRA_FINAL.md` -- Max Hydra ceiling, not immediate build license.
6. `docs/CURRENT_STATUS.md` -- shipped/staged/default-on state.
7. `docs/GAME_ENGINE.md`, `docs/COMPATIBILITY_SURFACE.md`, current code -- runtime truth; code wins on impl drift.

Rules:
- Do not implement from raw archive corpus like `research/agent_handoffs/combined_all_variants/` directly. Promote claims into design/status docs first.
- For impl work: pick lane from reconciliation, confirm status in current status, confirm runtime contract in game/compat docs + code.
- Status words (`shipped baseline`, `implemented but staged`, `implemented but not default-on`, `reserve shelf`, `historical`) are contracts, not decoration.
- If changing compatibility surface, update code + tests + owner doc in same cut.

## Compatibility facts not casually changed

- Live encoder/model input = `192x34`.
- Old `85x34` = historical baseline-prefix channels `0..84`, not live full encoder.
- Action space = fixed 46 actions.
- Legal action mask shape = `[bool; 46]`.
- Riichi and kan use two-phase handling.
- Tile kinds = `0..33`; aka/red fives remain distinct in 136-format/action surfaces where required.
- Suit augmentation = exactly 6 numbered-suit permutations; honors unchanged.
- BC selected-runtime authority: fresh run config-derived; epoch-boundary resume may reuse matching selected-runtime tuple; partial-epoch resume requires identical runtime.
- BC loader-runtime authority stays config-derived; matching preflight cache does not override checkpoint/runtime contract.
- BC CUDA LibTorch runs default to BF16 AMP when `precision_mode` omitted; explicit `fp32` stays FP32; CPU omission stays FP32; RL/DeltaQ BF16 hard-gated.
- CUDA graph feature ships pinned staging/preallocated tensors/probes; production graph replay remains off/probe-only until Burn optimizer gradient contract permits it.

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

Core gates:

| Command | Use |
|---|---|
| `pixi run torch-check` | Verify pinned PyTorch/libtorch and CUDA visibility |
| `pixi run check-lib` | Fastest compile gate: workspace libs, no default features |
| `pixi run test-lib` | Fast lib unit tests, no default features |
| `pixi run test-fast` | Fast test targets, no benches/examples |
| `pixi run check` | Default no-heavy all-target compile gate |
| `pixi run test` | Default no-heavy all-target tests through nextest |
| `pixi run lint` | Fast lint: anti-game scan, rustfmt, clippy |
| `pixi run fmt` | Rustfmt only |
| `pixi run build` | Fast dev build |
| `pixi run build-release` | Fast release build |
| `pixi run build-dist` | Fat-LTO prod artifact |
| `pixi run clean` | Cargo clean |

Explicit heavy/special gates:

| Command | Use |
|---|---|
| `pixi run check-training` | CPU LibTorch `hydra-train --features training` |
| `pixi run check-cuda-graph` | CUDA graph/pinned transport compile gate |
| `pixi run check-training-cuda` | Alias for serious CUDA shard training check |
| `pixi run train-cuda-shards -- <config>` | Release CUDA BC shard training with pinned/prealloc transport |
| `pixi run check-data-tools` | Data-tool bins (`data-tools`) |
| `pixi run check-debug-tools` | Replay-debug bins (`debug-tools`) |
| `pixi run test-exhaustive` | All-feature test gate |
| `pixi run lint-exhaustive` | All-feature/CUDA clippy gate |
| `pixi run lint-cuda-graph` | Focused CUDA graph clippy gate |
| `pixi run bench-core` | `hydra-core` benches |
| `pixi run bench-engine` | `hydra-engine` benches |
| `pixi run bench-train` | `hydra-train` benches |

Timing/discovery:

| Command | Use |
|---|---|
| `pixi run timings-check` | Cargo timing report for default check |
| `pixi run timings-nextest-list` | Timing report for test discovery |
| `pixi run timings-build-fast` | Timing report for default build |
| `pixi run timings-full` | Timing report for all-feature build |
| `pixi run nextest-list` | Discover default no-heavy tests |
| `pixi run nextest-list-all-features` | Discover all-feature tests |
| `pixi run nextest-list-exhaustive` | Alias for all-feature test discovery |

Compatibility aliases: `build-fast` -> `build-release`; `nextest` -> `test`.

Fast loops:

```bash
pixi run check-lib
pixi run test-lib
pixi run check
pixi run test
pixi run cargo check -p <crate> --no-default-features --quiet
pixi run cargo nextest run -p <crate> --lib --no-default-features --cargo-profile dev --cargo-quiet
pixi run cargo nextest run <test-name> --no-default-features --cargo-profile dev --cargo-quiet
pixi run cargo nextest run -p <crate> <ignored-test> --features <feature> --no-default-features --no-capture -- --ignored
```

Rules:
- Heavy paths are opt-in only: training, CUDA graph, exhaustive/all-features, data/debug tools, benches.
- Do not compare cold-cache compile timings to warm-cache timings.
- Clean compile measurements: `CARGO_TARGET_DIR=$HOME/tmp/hydra-compile-bench/<name> SCCACHE_DISABLE=1 pixi run ...`; normal dev leaves sccache on.
- Python script tests: `uv run python -m unittest discover -s scripts/tests`.
- Coverage/container/Kaggle commands live in `docker/train/README.md`.
- Perf claims must cite `research/infrastructure/ENGINE_BENCHMARKS.md` or exact command/output observed.

## Pixi/libtorch/tooling contract

- `pixi.toml` + `pixi.lock` are dev/build env source of truth. Commit `pixi.lock` when env changes.
- Single default Pixi env covers CPU/GPU. Hydra config selects `device: cpu` or `device: cuda:0`.
- Current Burn LibTorch stack uses `burn-tch 0.21` -> `tch 0.22` -> `torch-sys 0.22`; requires exact PyTorch/libtorch `2.9.0`.
- Do not use PyTorch `2.9.1`, newer libtorch, or `LIBTORCH_BYPASS_VERSION_CHECK` unless Burn dependency requirements changed.
- Required env/linker settings live in `pixi.toml`: `LIBTORCH_USE_PYTORCH`, `PROTOC`, `LD_LIBRARY_PATH`, `CUDA_HOME`, `CUDA_PATH`, `RUSTC_WRAPPER`, `SCCACHE_DIR`, Pixi clang/mold, conda sysroot `libc_nonshared.a`.
- Do not hardcode `/usr/bin/clang`, `/usr/bin/mold`, user-local tool paths, or local CUDA paths unless Pixi link failure is proven and documented.
- `.cargo/config.toml` is local-only/gitignored.
- `Cargo.lock` stays committed.
- `.codebase-memory/`, `target/`, `output/`, notebooks payloads, model/cache artifacts stay out of normal commits unless task explicitly refreshes them.

## Burn dependency decisions

- Burn stack is patched locally in `third_party/burn`: `burn`, `burn-autodiff`, `burn-backend`, `burn-tch`, `burn-flex`, `burn-ndarray`, `burn-optim` at `0.21.0`.
- Current training backend is LibTorch/tch. Do not replace with `burn-flex`: Flex is CPU-only and useful only for isolated CPU smoke/parity tests.
- Do not adopt `burn-dispatch` in production training yet. It changes device/tensor associated types (`DispatchDevice`/`DispatchTensor`) and is not throughput fix.
- Do not use `burn-store` for BC shard/sample materialization. It is model tensor storage, not Hydra compact sample decoding. If checkpoint time is proven material, benchmark Burnpack/SafeTensors as model-payload export only; keep optimizer `.bin` contract unless full resume parity proof passes.
- Keep Burn Adam as production optimizer. AdamW/AMSGrad/Adan are fresh-run experiments only; Muon requires parameter groups and is unsafe for global Hydra params; LBFGS does not fit streaming BC/RL. None fixes CUDA graph replay because Burn `GradientsParams` + module mapping remains blocker.
- For profiling, keep Hydra timings + NVTX + Nsight Systems/Compute. Do not add `hdrhistogram` unless long-run online tail summaries are needed and existing JSONL timing extraction is insufficient.

## Licensing and source boundaries

- `Mortal-Policy/` is AGPL. Never copy, adapt, derive, port line-by-line, link, or translate code from it. Black-box behavior/compatibility ideas only.
- Do not add AGPL/GPL/LGPL dependencies.
- Allowed dependency families: MIT, Apache-2.0, BSD-compatible.
- `hydra-engine` is Apache-2.0 vendored upstream.
- First-party Hydra crates use repo BSL 1.1 unless crate-specific license says otherwise.
- New dependency requires license check, compile-time cost check, and search for existing/simple in-repo alternative.

## Implementation rules

- Fix root cause at owning layer. Do not hide failure with warnings, fallback defaults, broad `Ok(())`, silent clamping, or compatibility shims.
- If neighboring code does something different, find why before deviating. Assume load-bearing until proven otherwise.
- Delete obsolete paths. No stale aliases, dead branches, TODO stubs, no-op implementations, or fake fallbacks.
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
- Performance work: read/update `research/infrastructure/ENGINE_BENCHMARKS.md`.
- Infrastructure/checkpoint/container work: read `research/infrastructure/INFRASTRUCTURE.md` and `docker/train/README.md` as applicable.
- Before trusting replay/shard/sidecar data with suspicious low sample count/high skip count, audit: `mjai_audit` -> `mjai_first_failure` -> `mjai_debug_failure`.
- If `advanced_loss.exit` or `advanced_loss.delta_q` is positive, matching sidecar path is required; validation promotion must hydrate labels when gates require them.

## Docs and Markdown

- Public Rust items need useful `///`; public module surfaces should have `//!` summary/invariants.
- When changing runtime/training surfaces, update owner doc: `GAME_ENGINE`, `COMPATIBILITY_SURFACE`, `CURRENT_STATUS`, `TRAINING_RUNBOOK`, `SEEDING`, or `ENGINE_BENCHMARKS`.
- Markdown stays caveman-compressed: terse, exact, low fluff, technical substance preserved.
- Do not commit `*.original.md` backups.
- `scripts/caveman-compress-hook.sh` must leave source unchanged on validation failure; report exact failure, no bluff.

## Self-check before yielding

- Correct layer? Owner crate updated first, facade/bin glue thin?
- Compatibility surface changed? Docs/tests updated?
- Obsolete code deleted rather than shimmed?
- Determinism/action/encoder/preflight/shard/checkpoint contracts preserved?
- Hot path still allocation-clean?
- Dependency/license/compile-time impact checked?
- Test covers behavior that can break?
- Claims match commands/docs observed?
