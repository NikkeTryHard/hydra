# Hydra2 Lean specs (`lean/`)

Formal sidecar to the Python stack in `src/hydra2`. Lean states the rules and
experiment contracts; Python implements them. Keep both in sync by hand —
nothing here is generated from (or generates) Python.

Toolchain: `leanprover/lean4:v4.33.1`, mathlib `v4.33.1` (`lakefile.toml`,
`lean-toolchain`). Build artifacts live in `.lake/` (gitignored).

## Layout

- `Formal/Mahjong/` — Tile, Wall, Dora, Shanten, Yaku, Meld, ActorObservation,
  Scoring, State, Turn, Event, Action, Rule, Game, ObsWire.
- `Formal/Blueprint/` — Objective, Belief, PBRF, CQL, Curriculum, PPO,
  SearchHygiene, TrainingData, Evaluation, EvaluationAxioms,
  `Modules/` (RaoBlackwell, MIS, CRN, MLMC, SMC, RQMC, ForcedTarget, Gating,
  Gumbel, Acquisition).
- `Formal/Implementation/` — PPO, Evaluation, Training (mirrors the training
  surface, not a verified build of it).
- `Formal.lean` — root import list; `Formal/Basic.lean` is a stub.

## Prerequisites

```sh
# one-time: Lean version manager (provides `lean`, `lake`)
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
```

`elan` reads `lean-toolchain` and fetches the pinned toolchain automatically
on first `lake` run. No system Lean install needed.

## Quick start

```sh
cd lean

lake update        # first clone only: fetch mathlib + deps (see lake-manifest.json)
lake build         # build the Formal lib (defaultTargets = ["Formal"])

lake build Formal.Mahjong.Tile   # one module
lake env lean Formal/Mahjong/Tile.lean   # typecheck a single file with the built env

grep -rn "sorry" Formal/         # unfinished proofs
lake clean         # drop build outputs (.lake/ is disposable)
```

## Editor

VS Code + the `lean4` extension, opened at `lean/` (not the repo root), gives
goal states and inline errors. Restart the extension after `lake update`.

## CI

`.github/workflows/lean_action_ci.yml` builds on push. `update.yml`
bumps the toolchain; `create-release.yml` cuts releases.

## Conventions

- One concept per file under `Formal/Mahjong/`; new search/training ideas go
  under `Formal/Blueprint/Modules/`.
- No `sorry` in files you call done — `grep` above is the gate.
- Pin bumps (`lean-toolchain` + `lakefile.toml` mathlib rev) go together;
  rebuild from `lake clean` after a bump.
