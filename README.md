# Hydra

Open-source Riichi Mahjong AI. The goal is to build an AI that rivals [LuckyJ](https://haobofu.github.io/) (Tencent, 10.68 stable dan on Tenhou) with open weights.

## Goal

Train a mahjong AI that:
- Surpasses [Mortal](https://github.com/Equim-chan/Mortal) (~7-dan) and approaches LuckyJ-level play (10+ dan) in head-to-head evaluation
- Releases weights under a permissive license
- Adds opponent modeling and inference-time search — the two capabilities that separate LuckyJ from all other mahjong AIs

## Architecture

Hydra's strongest current doctrine is:

1. [`research/design/HYDRA_FINAL.md`](research/design/HYDRA_FINAL.md) — target architecture north star
2. [`research/design/HYDRA_RECONCILIATION.md`](research/design/HYDRA_RECONCILIATION.md) — current execution doctrine
3. [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) — current runtime reality

Everything else should be read as implementation reference, rationale, reserve shelf, or archive material unless one of the three files above explicitly promotes it.

If you are deciding what to build next, read these in order:
- [`research/design/HYDRA_FINAL.md`](research/design/HYDRA_FINAL.md) — target architecture SSOT
- [`research/design/HYDRA_RECONCILIATION.md`](research/design/HYDRA_RECONCILIATION.md) — verified repo reality + best next action
- [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) — current game-engine/runtime baseline
- [`research/design/IMPLEMENTATION_ROADMAP.md`](research/design/IMPLEMENTATION_ROADMAP.md) — staged implementation plan
- [`research/design/HYDRA_ARCHIVE.md`](research/design/HYDRA_ARCHIVE.md) — archived / reserve-only planning surfaces

`research/design/HYDRA_SPEC.md` is historical context only.

## Research

| File | What's In It |
|------|-------------|
| [HYDRA_FINAL.md](research/design/HYDRA_FINAL.md) | Current target architecture SSOT |
| [HYDRA_RECONCILIATION.md](research/design/HYDRA_RECONCILIATION.md) | Current execution doctrine and active/reserve split |
| [HYDRA_ARCHIVE.md](research/design/HYDRA_ARCHIVE.md) | Archived / reserve-only planning surfaces |
| [HYDRA_SPEC.md](research/design/HYDRA_SPEC.md) | Historical architecture spec only |
| [MORTAL_ANALYSIS.md](research/MORTAL_ANALYSIS.md) | Mortal's architecture, training details, confirmed weaknesses |
| [OPPONENT_MODELING.md](research/OPPONENT_MODELING.md) | Opponent-modeling rationale; includes both active ideas and reserve/future extensions |
| [INFRASTRUCTURE.md](research/infrastructure/INFRASTRUCTURE.md) | Rust stack, data pipeline, training infra, hardware, deployment |
| [SEEDING.md](research/design/SEEDING.md) | RNG hierarchy, reproducibility, evaluation seed bank |
| [CHECKPOINTING.md](research/infrastructure/CHECKPOINTING.md) | Checkpoint format, save protocol, retention policy |
| [ECOSYSTEM.md](research/intel/ECOSYSTEM.md) | Useful repos, tooling, and framework references |
| [REWARD_DESIGN.md](research/design/REWARD_DESIGN.md) | Reward design and RVR notes |
| [COMMUNITY_INSIGHTS.md](research/intel/COMMUNITY_INSIGHTS.md) | Community observations and external signals |
| [REFERENCES.md](research/intel/REFERENCES.md) | Citation index |
| [TESTING.md](research/design/TESTING.md) | Testing strategy, correctness verification, property-based tests |
| [RUST_STACK.md](research/infrastructure/RUST_STACK.md) | 100% Rust decision and framework notes |

### Documentation Ownership (SSOT)

- `research/design/HYDRA_FINAL.md`: target architecture SSOT
- `research/design/HYDRA_RECONCILIATION.md`: current repo-wide decision memo and best-next-action guide
- `docs/GAME_ENGINE.md`: current game-engine/runtime baseline
- `research/design/OPPONENT_MODELING.md`: detailed opponent-modeling rationale
- `research/design/HYDRA_ARCHIVE.md`: archive / reserve-only doctrine
- `research/design/HYDRA_SPEC.md`: historical architecture summary only; do not use it as the current implementation SSOT

## Status

Active implementation. `hydra-core` is already built out as a real baseline engine/encoder crate, and `hydra-train` contains a substantial training/model scaffold with partial advanced integration. The immediate project need is reconciliation plus closure of the highest-leverage training/search supervision loops, not a restart from scratch.

## License

- **hydra-core** (encoder, training pipeline): [BSL 1.1](hydra-core/LICENSE) -- free for non-commercial use, converts to Apache-2.0 on 2031-03-02
- **hydra-engine** (game rules): Apache-2.0 (vendored from riichienv-core)
