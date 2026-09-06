Hydra2 is a reproducible Riichi Mahjong AI research stack. Ambition: beat LuckyJ/Mortal at Tenhou four-player hanchan by measurement, never by claim. No superiority, Elo, rank, or throughput claim without measured evidence in an artifact record.

<system-conventions>
RFC 2119 applies to MUST, REQUIRED, SHOULD, RECOMMENDED, MAY, OPTIONAL. `NEVER` = `MUST NOT`; `AVOID` = `SHOULD NOT`.
</system-conventions>

## Stack (exact; Pixi is sole authority)

Pixi owns env+lock (`pixi.lock`). You MUST NEVER create `uv.lock`, use a stray `.venv`, or run bare `pytest`/`python`/`ruff`. Verified 2026-09-04 via `pixi run config-check`: `torch==2.14.0+cu130` (cuda 13.0, sm_120 available), `lightning-fabric==2.6.5`, `riichienv==0.4.8`, mahjax pinned `5222872`, `ruff==0.16.5`, `pyrefly==1.2.0`, `pytest==9.1.1`, python 3.12. Pyrefly MUST use pin `.pixi/envs/default/bin/python`.

## Commands (always through Pixi; never on host)

| Need | Command |
| --- | --- |
| Focused test | `pixi run test <file>::<test>` |
| Package gate | `pixi run test-{contracts,conformance,integration,unit,search,training,analysis} --package <WP-ID>` |
| Full suite | `pixi run test` (only if shared contracts/runtime/data touched) |
| Lint / format / types | `pixi run lint`, `pixi run format-check`, `pixi run typecheck` |
| Env / probe | `pixi run config-check`, `pixi run env-manifest`, `pixi run runtime-probe` |
| WP exit | `pixi run hydra2 work-package verify <WP-ID> --artifact-root "$HYDRA2_ARTIFACT_ROOT"` |
| Lean file | `cd lean && lake env lean Formal/<Path>.lean` (full: `lake build`) |
| Rust packager | `cargo nextest run` inside `tools/mjai-dataset-packager/` |

Capture once to a log file, then grep it. NEVER re-run a suite with different greps. Batch edits before building: one build, not build-per-edit.

## Correctness invariants (non-negotiable)

- `dora_indicators` shape is `(5,)`. NEVER pad `(4,)` with a shim; `(4,)` artifacts are incompatible.
- JSON identity artifacts MUST use RFC 8785 canonical bytes + SHA-256 (`src/hydra2/_canon.py`, `src/hydra2/artifacts/`).
- Randomness MUST use semantic counter-based streams from `IMPLEMENTATION_SPEC.md`. NEVER `datetime.now()` or unseeded RNG in tests.
- `CUBLAS_WORKSPACE_CONFIG` MUST be set before any CUDA context (done in `tests/conftest.py`); inductor cache is version-keyed there. Do not move it.
- Simulator stays eager. Only pure-tensor model regions MAY compile (`torch.compile`/inductor); SDPA is the standard dense-attention path. Every speed claim needs fixed-corpus eager parity + cold-start/latency/throughput/memory/determinism evidence per device.
- MahJax is a quarantined accelerator at its pinned SHA until conformance passes. NEVER let accelerator trajectories leak into reference data.
- No Lightning Trainer: `lightning`/`pytorch-lightning` MUST stay absent (`TRAINER_FORBIDDEN` in `src/hydra2/config.py`). Own the loop, optimizer, schedule, accumulation, checkpoint.
- `HYDRA2_ARTIFACT_ROOT` MUST live outside raw/confidential data roots. NEVER publish raw samples, source identity, or sponsor identity (D-017).

```python
# CORRECT: fixed (5,) dora, actor-visible mask, counter-stream seed
obs = encode_actor_visible(packet, dora=packet.dora_indicators)  # shape (5,), exact
rng = counter_stream(seed=semantic_seed("WP-09A", block_id, seat))
# WRONG: padded dora, hidden info in encoder, wall-clock seed
dora = F.pad(dora4, (0, 1))  # NEVER — hides an incompatible artifact
```

## Architecture boundaries

Layered DAG, dependencies flow one way: `contracts` (stdlib-only Tenhou vocab) <- `artifacts` <- `engines` (riichienv 0.4.8 reference adapter; mahjax JAX shell) <- `runtime` (plain eager / Fabric adapter) + `data` (zstd ingest -> validate -> quarantine -> parquet) -> `models` (actor-visible encoder + SDPA transformer) -> `belief` (natural packets) -> `search` (candidate0/ISMCTS/DESPOT/PBRF/Gumbel/resolving) -> `eval` (duplicate-wall blocks, expected final placement) + `training`/`distillation`. NEVER invert an edge (e.g. models importing search; workers touching raw stores). `lean/` is a manual-sync sidecar: no codegen either direction; no `sorry` in files called done. `tools/mjai-dataset-packager/` is isolated (clang+mold, nextest); behavior changes need compatibility evidence.

## Docs authority (conflicts)

1. Versioned canonical artifacts from completed packages. 2. `docs/BUILD_EXECUTION_PLAN.md` (order, gates, evidence). 3. `docs/IMPLEMENTATION_SPEC.md` (schemas, APIs, algorithms). 4. `docs/PROJECT_PLAN.md` (direction). 5. `docs/ALGORITHM_EXPERIMENT_BLUEPRINT.md` (candidate intent). 6. External refs (rationale only; NEVER override contracts). On conflict: stop the package, record the exact conflict, NEVER silent-pick. Missing spec blocks implementation; it NEVER authorizes guessing.

## Testing rules

- Default: add the test to the existing file for the module you changed. New files only for new modules. Every test MUST defend observable behavior or an invariant and MUST fail without the change.
- Ladder: focused nodeid -> file -> `--package <WP-ID>` gate -> full suite only if shared code moved. A passing narrow test NEVER substitutes for the package exit gate.
- Determinism: fixed seeds, `tempDir`-style tmp dirs, `port: 0`, poll-with-deadline NEVER `sleep(N)`, no live internet (local harness doubles only), assert behavior before exit codes.
- GPU lanes: `gpu` mark needs CUDA (absence is hard failure); `slow`/`soak` are opt-in via `-m`, never deselected by default.
- Be humble and honest: NEVER overstate what works in commits, PRs, or messages. Second related branch-condition finding -> stop, re-read the requirement, narrow the contract instead of adding machinery.

## Allowed / ask-first / never

- Allowed: focused `pixi run` commands, reading any file, `lake build`/`cargo nextest run` in their own trees.
- Ask-first: schema/contract changes (update all affected docs+hashes first), dependency adds, kernel/compile-arm changes (need per-device qualification), A100 hours (ledger entry first per D-015).
- NEVER: bare `pytest`/`python`/`ruff` on host; `uv.lock`; secrets/keys in repo or prompts; force-push; suppressing warnings/errors to hide failure; live Tenhou/Soul clients; benchmark claims without artifact evidence.

## Done means

`pixi run lint`, `pixi run format-check`, `pixi run typecheck`, plus the scoped package gate, plus `work-package verify` for WP-tracked work. If you did not run them, it does not work. Keep modules ~500 LoC; split past ~800; keep PRs reviewable.
