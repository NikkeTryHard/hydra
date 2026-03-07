# Hydra Archive and Demoted Planning Surfaces

This document preserves weaker, older, broader, or reserve-only Hydra planning surfaces so they remain available for later review **without** competing with the current strongest doctrine.

## How to use this file

- Use `HYDRA_FINAL.md` for the architectural north star.
- Use `HYDRA_RECONCILIATION.md` for current execution doctrine.
- Use `docs/GAME_ENGINE.md` for current runtime reality.
- Use this file only when you intentionally want:
  - historical context,
  - reserve-shelf ideas,
  - or archived plan surfaces that should not drive current implementation by default.

## Archived / demoted categories

### 1. Legacy architecture and stale interfaces

These should not be treated as current defaults:
- old `85x34` full-tensor assumptions (current reality is 192x34 fixed superset)
- broad 40-block / PPO-era planning language
- old full-stack build ordering that ignores the reconciled active path

Primary homes:
- `HYDRA_SPEC.md`
- older sections of `IMPLEMENTATION_ROADMAP.md`
- older operational assumptions in `INFRASTRUCTURE.md`
- stale lower-level test or contributor docs that still describe the old baseline as current truth

### 2. Reserve-shelf techniques worth preserving

These ideas may matter later, but they are not the default active path right now:
- stronger endgame exactification
- robust-opponent search backups
- confidence-gated safe exploitation
- richer latent opponent posterior / opponent-plan inference
- deeper AFBS semantics and hard-state expansion policies
- structured / incremental belief alternatives if the current unified stack underdelivers
- DRDA/ACH as a stronger optimizer branch after the target pipeline is healthy

### 3. Demoted complexity that should not quietly return to the mainline

These ideas are not forbidden forever, but they should not re-enter current implementation by accident:
- broad “search everywhere” AFBS
- duplicated belief stacks
- adding more heads before existing advanced targets are credible and alive
- optimizer/game-theory side quests before target-generation is closed
- novelty-heavy deception / RSA / speculative ToM work without strong evidence and a clear insertion point

## Why this archive exists

Hydra has enough strong ideas that simply deleting older or reserve-only material would be wasteful.
But keeping all of it mixed into active docs creates doctrine drift.

This archive keeps the strong-but-not-current material visible **without** letting it compete with the strongest current Hydra.
