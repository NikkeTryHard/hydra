Hydra2 is a reproducible Riichi Mahjong AI research stack. Ambition: beat LuckyJ/Mortal at Tenhou four-player hanchan by measurement, never by claim. No superiority, Elo, rank, or throughput claim without measured evidence in an artifact record.

<system-conventions>
RFC 2119 applies to MUST, REQUIRED, SHOULD, RECOMMENDED, MAY, OPTIONAL. `NEVER` = `MUST NOT`; `AVOID` = `SHOULD NOT`.
</system-conventions>

## Stack (exact; Pixi is sole authority)

Pixi owns env+lock (`pixi.lock`; pins declared in `pyproject.toml` `[tool.pixi.*]`). You MUST NEVER create `uv.lock`, use a stray `.venv`, or run bare `pytest`/`python`/`ruff`. Pins: `torch==2.14.0` (CUDA 13.0, sm_120 kernels), `riichienv==0.4.10`, mahjax `cff90d1e68cf21464071864672a9618bb20f2551` (v0.1.3), `ruff==0.16.6`, `pyrefly==1.2.0`, `pytest==9.1.1`, `clearml==2.1.12`, python 3.12. `lightning`/`pytorch-lightning`/`lightning-fabric` MUST stay absent (fabric removed). `pixi run config-check` proves the running env matches these pins. Pyrefly MUST use pin `.pixi/envs/default/bin/python`.

## Commands (always through Pixi; never on host)

| Need | Command |
| --- | --- |
| Focused test | `pixi run test-unit <file>::<test>` (or the lane owning the file: `test-contracts`, `test-search`, `test-integration`; `test-serial` for `gpu`/`serial`-marked) |
| Parallel lanes | `pixi run test-cpu` + `pixi run test-serial` (REQUIRED split; append `--package <WP-ID>` where the task supports it) |
| Package gate | `pixi run test-{unit,integration,contracts,search,training} --package <WP-ID>`; `pixi run test-analysis` (WP-12 baked in); `pixi run test-conformance` (conformance/engines/parity trees, single-process) |
| Full suite | `pixi run test` (both lanes; only if shared contracts/runtime/data touched) |
| Lint / format / types | `pixi run lint`, `pixi run format-check`, `pixi run typecheck` |
| Env / probe | `pixi run config-check`, `pixi run env-manifest`, `pixi run runtime-probe` |
| WP exit | `pixi run hydra2 work-package verify <WP-ID> --artifact-root "$HYDRA2_ARTIFACT_ROOT"` |
| Lean file | `cd lean && lake env lean Formal/<Path>.lean` (full: `cd lean && lake build`) |
| Rust packager | `cargo nextest run` inside `crates/packager/` |
| Rust workspace tests | `PYO3_PYTHON` pinned via root `.cargo/config.toml` `[env]`; test binaries additionally need `LD_LIBRARY_PATH=$PWD/.pixi/envs/default/lib` (libpython link) — e.g. `LD_LIBRARY_PATH=$PWD/.pixi/envs/default/lib cargo nextest run --manifest-path crates/Cargo.toml -p hydra-bridge` |
| Bridge build | `pixi run build-ext` (portable max; ONLY source of published artifacts) |
| Bridge build, local max | `pixi run build-ext-max` (`target-cpu=native`; non-portable, local benchmarks ONLY — rebuild plain `build-ext` before any measured run) |

Capture once to a log file, then grep it. NEVER re-run a suite with different greps. Batch edits before building: one build, not build-per-edit.

## Correctness invariants (non-negotiable)

- `dora_indicators` shape is `(5,)`. NEVER pad `(4,)` with a shim; `(4,)` artifacts are incompatible.
- JSON identity artifacts MUST use RFC 8785 canonical bytes + SHA-256 (`python/hydra2/artifacts/canonical.py`, `python/hydra2/artifacts/`).
- Randomness MUST use semantic counter-based streams: every draw derives from a fully named stream key (purpose + experiment/game scope + attempt) hashed to its seed. NEVER `datetime.now()`, wall-clock seeds, global counters, or unseeded RNG in tests.
- `CUBLAS_WORKSPACE_CONFIG` MUST be set before any CUDA context (done in `tests/conftest.py`); inductor cache is version-keyed there. Do not move it.
- Simulator stays eager. Only pure-tensor model regions MAY compile (`torch.compile`/inductor); SDPA is the standard dense-attention path. Every speed claim needs fixed-corpus eager parity + cold-start/latency/throughput/memory/determinism evidence per device.
- MahJax is a quarantined accelerator at its pinned SHA until conformance passes. NEVER let accelerator trajectories leak into reference data.
- No Lightning Trainer: `lightning`/`pytorch-lightning` MUST stay absent (`TRAINER_FORBIDDEN_PACKAGES` in `python/hydra2/config.py`). Own the loop, optimizer, schedule, accumulation, checkpoint.
- `HYDRA2_ARTIFACT_ROOT` MUST live outside raw/confidential data roots. Dataset authority is confidential: use non-identifying source IDs with authorization attestation; NEVER publish raw samples, source identity, or sponsor identity.

```python
# CORRECT: fixed (5,) dora, actor-visible mask, counter-stream seed
obs = encode_actor_visible(packet, dora=packet.dora_indicators)  # shape (5,), exact
rng = counter_stream(seed=semantic_seed("WP-09A", block_id, seat))
# WRONG: padded dora, hidden info in encoder, wall-clock seed
dora = F.pad(dora4, (0, 1))  # NEVER — hides an incompatible artifact
```

## Architecture boundaries

Layered DAG, dependencies flow one way: `contracts` (stdlib-only Tenhou vocab) <- `artifacts` <- `engines` (riichienv 0.4.10 reference adapter; mahjax JAX shell) <- `runtime` (plain eager adapter; Fabric removed) + `data` (zstd ingest -> validate -> quarantine -> parquet) -> `models` (actor-visible encoder + SDPA transformer) -> `belief` (natural packets) -> `search` (candidate0/ISMCTS/DESPOT/PBRF/Gumbel/resolving) -> `eval` (duplicate-wall blocks, expected final placement) + `training`/`distillation` + `analysis`…

## Language policy (Python minimal, Rust preferred)

- **Python is last resort, NEVER default.** New logic MUST land in Rust (`crates/*`) behind a thin PyO3 boundary; Python stays ONLY where the runtime forces it (`torch` nn/optim/SDPA + owned loop, JAX `jit` shells, `pyarrow`/`zstandard` edges, Python-only SDKs, stdlib `contracts`, POSIX helpers).
- **Thin-shim rule.** New Python MUST stay typing + firewall + one batched FFI call (~100 LoC per bridge file). Bytes, loops, hashes, validators, frozen tables belong in Rust behind `OnceLock`/`PyOnceLock` + batch entry points. Per-row/per-node Python callbacks on hot paths are FORBIDDEN.
- **Bridge builds.** `build-ext` is the portable max (release + fat LTO + single CGU + stripped) and the ONLY source of published artifacts. `build-ext-max` adds `target-cpu=native` for local benchmarks; its binaries are non-portable and MUST NEVER back published measurements. RUSTFLAGS env REPLACES config rustflags (never merges) — any env preset MUST repeat the mold link-arg or the link silently falls back to bfd.

## Docs authority (conflicts)

1. Versioned canonical artifacts from completed packages. 2. `docs/BUILD_EXECUTION_PLAN.md` (order, gates, evidence). 3. `docs/IMPLEMENTATION_SPEC.md` (schemas, APIs, algorithms). 4. `docs/PROJECT_PLAN.md` (direction). 5. `docs/ALGORITHM_EXPERIMENT_BLUEPRINT.md` (candidate intent). 6. External refs (rationale only; NEVER override contracts). On conflict: stop the package, record the exact conflict, NEVER silent-pick. Missing spec blocks implementation; it NEVER authorizes guessing.

## Comment self-containment (all code comments: Lean + Rust + Python)

- **Self-contained comments.** All code comments (Lean, Rust, Python) MUST state invariant + failure mode inline.
- **Allowed refs.** Source-file paths (`file://python/...`, `python/...`, `crates/...`, `Formal/...`) + online URLs (`https://...`) ONLY; URLs stay full even if path contains `docs/`.
- **Banned pointers.** NEVER bare doc pointers in comments: `SPEC §`, `per SPEC/BUILD`, `Blueprint §`, `IMPLEMENTATION_SPEC.md`, `docs/...`, `ideas/...`, `>> SPEC >>`, `file://docs/...`.
- **No-benefit provenance.** Delete it; numbers/thresholds MUST stay inline.

## Testing rules

- Default: add the test to the existing file for the module you changed. New files only for new modules. Every test MUST defend observable behavior or an invariant and MUST fail without the change.
- Ladder: focused nodeid -> file -> `--package <WP-ID>` gate -> full suite only if shared code moved. A passing narrow test NEVER substitutes for the package exit gate.
- Determinism: fixed seeds, `tempDir`-style tmp dirs, `port: 0`, poll-with-deadline NEVER `sleep(N)`, no live internet (local harness doubles only), assert behavior before exit codes.
- Parallel lanes: `pixi run test-cpu` (`-n 16 --dist loadscope`, everything except `gpu`/`serial`) + `pixi run test-serial` (`-n 0`, `gpu or serial` only). Every test MUST land in exactly one lane. `-n`/`--dist` live ONLY in lane tasks, NEVER in `addopts`. `loadscope` (whole file/class per worker) is REQUIRED — per-test `--dist load` oversubscribes the single GPU (CUBLAS/OOM) and splits file-order-coupled tests.
- Lane isolation: `test-cpu` hides all GPUs (`CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`, no XLA prealloc) so the CPU lane uses zero VRAM — torch/JAX never init a CUDA context there (conftest thread clamp auto-scales `total // workers`). Any test needing CUDA driver/device (CUDA contexts, `pin_memory`, JAX init, fused kernels) MUST carry `gpu`/`serial`, never silently degrade: CUDA-unavailable in cpu lane skips/fails loud (see `require_cuda`, `test_pinned_ring.py` serial marks).
- Lane marks: `gpu` = needs CUDA (absence is hard failure); `serial` = fused GPU-kernel loads, JAX device init at import, CUDA contexts, shared-cache writers (single process ONLY — concurrent import aborts workers, CUBINs OOM under contention). Readers of another test's outputs MUST carry the writer's mark. `slow`/`soak` stay opt-in via `-m`, NEVER deselected by default.
- Parallel-safe tests: `tmp_path`/explicit roots ONLY (NEVER shared or hardcoded paths); fixtures read-only or worker-local; no sockets, no `sleep`, no wall-clock seeds; each test self-contained (NEVER read files another test writes).
- Float agreement: bitwise `torch.equal` ONLY for same-shape repeat eval. Cross-shape/bucket agreement MUST use `allclose(atol=1e-6, rtol=1e-5)` — thread count changes reduction order (workers run fewer threads than serial).
- Threads: conftest clamps torch/OMP per xdist worker. You MUST NEVER call `set_num_threads` in tests, or loosen the clamp to fit one test — mark that test `serial` instead.
- JAX hot loops: hoist `jax.jit(step-fn)` above the loop on one stable function object (NEVER re-jit per step, NEVER eager-step hot loops); persistent cache dir is version-keyed in conftest. Same kernels = decisions identical; anything changing compute needs requalification.
- Shared writers (reports, tokens): controller-only under xdist, unique-per-run paths. Designed producer→consumer chains MUST share one lane mark (order is only guaranteed inside it) — NEVER rely on unmarked cross-test order.
- Perf parity on ports: every bridge/accelerator port MUST prove bridge-vs-oracle TIMING on its hottest caller path alongside value parity — value-green tests hid a 532x slowdown (per-call census regen). Gate: replay benchmark of the hottest call shape, bit-identical outputs AND faster-or-equal wall time, or the port does not land.
- Frozen-once bridge state: pyfns MUST NEVER regenerate/rebuild tables per call — freeze process-once state (`OnceLock`/`LazyLock`) and expose batch variants for hot loops. Per-call FFI on hot paths needs the timing gate above.
- Lane speed budget: no single test past ~60s wall in the CPU lane — split mega-rollups per case-group into loadscope-spreadable files (consume results, never re-execute engine sims; coverage assertion stays); one shared import-only session fixture (NEVER per-module cargo builds or cdylib copies); serial-lane file order stays cheap→heavy with init-heavy files last.
- Be humble and honest: NEVER overstate what works in commits, PRs, or messages. Second related branch-condition finding -> stop, re-read the requirement, narrow the contract instead of adding machinery.

## Allowed / ask-first / never

- Allowed: focused `pixi run` commands, reading any file, `lake build`/`cargo nextest run` in their own trees.
- Ask-first: schema/contract changes (update all affected docs+hashes first), dependency adds, kernel/compile-arm changes (need per-device qualification), A100 hours (ledger entry first per D-015).
- NEVER: bare `pytest`/`python`/`ruff` on host; `uv.lock`; secrets/keys in repo or prompts; force-push (history stays linear: `pull.rebase=true` + `merge.ff=only` are set in repo config — pulls rebase, merges fast-forward-only); suppressing warnings/errors to hide failure; live Tenhou/Soul clients; benchmark claims without artifact evidence.

## Done means

`pixi run lint`, `pixi run format-check`, `pixi run typecheck`, plus the scoped package gate, plus `work-package verify` for WP-tracked work. If you did not run them, it does not work. Keep modules ~500 LoC; split past ~800; keep PRs reviewable.
