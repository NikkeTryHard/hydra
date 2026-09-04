<p align="center">
<img src="assets/hydra.webp" alt="Hydra2 banner" width="720">
</p>

<h1 align="center">Hydra2</h1>

Riichi mahjong AI research, done carefully. Correct training, honest evaluation,
no fake data in the records, everything reproducible. Tenhou 4-player hanchan
rules. Long-term goal: outplay LuckyJ.

Hydra1 never got as strong as Mortal, so Hydra2 started clean — only the
validated infrastructure lessons carried over, plus the Rust MJAI packager.

## Status

Working tree, torch 2.14 on CUDA. Type checks pass, linter is clean, unit
tests are 364 green with zero failing. The teacher pipeline refuses to run on
synthetic stand-ins (it errors instead of inventing data), candidate hashes
are bound to real digests, and the training-repeatability test actually
trains twice again.

- `docs/PROJECT_PLAN.md` — where this is going
- `docs/BUILD_EXECUTION_PLAN.md` — work order and what "done" means
- `docs/IMPLEMENTATION_SPEC.md` — schemas, APIs, contracts
- `docs/ALGORITHM_EXPERIMENT_BLUEPRINT.md` — candidates and promotion rules
- `lean/` — formal specs sidecar (Lean 4 + mathlib)

## Quick start

```sh
pixi install              # only supported env (don't use uv/venv here)
pixi run pytest tests/unit -q   # fast gate, ~1 min
pixi run ruff check src tests   # lint
pixi run pyrefly check src      # types
```

Full suite takes ~30 min (mostly conformance soak tests). `HYDRA2_SKIP_GPU_SOAK=1`
skips the GPU soak on machines without CUDA.

```sh
pixi run python -m hydra2.probe   # runtime environment report
```

## Repo map

- `src/hydra2/` — the stack: contracts, engines (RiichiEnv/MahJax), search,
  belief, training, eval, artifacts
- `configs/` — rules, contracts, model input specs, attestations
- `tools/mjai-dataset-packager/` — restart-safe Rust MJAI packager
- `tests/` — unit, contracts, search, integration, conformance

## Compute support

<p align="center">
<img src="assets/delta.webp" alt="Delta GPU node" width="720">
</p>

Compute from Delta (NCSA / UIUC), supported by the NSF (award OAC 2005572)
and the State of Illinois.
