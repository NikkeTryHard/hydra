# Hydra

Open-source Riichi Mahjong AI. The goal is to build an AI that rivals [LuckyJ](https://haobofu.github.io/) (Tencent, 10.68 stable dan on Tenhou) with open weights.

## Goal

Train a mahjong AI that:
- Surpasses [Mortal](https://github.com/Equim-chan/Mortal) (~7-dan) and approaches LuckyJ-level play (10+ dan) in head-to-head evaluation
- Releases weights under a permissive license
- Adds opponent modeling and inference-time search — the two capabilities that separate LuckyJ from all other mahjong AIs

## Architecture

Hydra uses a layered authority flow:

1. [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl) — canonical archive SSOT for upstream research conclusions
2. [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) and [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md) — derived planning/render views over that archive canon
3. [`research/design/HYDRA_FINAL.md`](research/design/HYDRA_FINAL.md) — promoted architecture doctrine summary
4. [`research/design/HYDRA_RECONCILIATION.md`](research/design/HYDRA_RECONCILIATION.md) — promoted execution doctrine summary and active-vs-reserve guide
5. [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) — runtime reality reference; current code wins when docs drift

Raw `answer_*_combined.md` files in `research/agent_handoffs/combined_all_variants/` remain raw archive corpus, not promoted doctrine.

If you are deciding what to build next, read these in order:
- [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl) — canonical archive SSOT / source ledger
- [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) — archive prioritization view
- [`research/design/HYDRA_FINAL.md`](research/design/HYDRA_FINAL.md) — promoted architecture doctrine summary
- [`research/design/HYDRA_RECONCILIATION.md`](research/design/HYDRA_RECONCILIATION.md) — promoted execution doctrine summary + best-next-action guide
- [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) — current game-engine/runtime baseline
- [`research/design/IMPLEMENTATION_ROADMAP.md`](research/design/IMPLEMENTATION_ROADMAP.md) — staged implementation reference
- [`research/design/HYDRA_ARCHIVE.md`](research/design/HYDRA_ARCHIVE.md) — reserve-only design/archive planning

`research/design/HYDRA_SPEC.md` is historical context only.

## Research

| File | What's In It |
|------|-------------|
| [ARCHIVE_CANONICAL_CLAIMS.jsonl](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl) | Canonical archive SSOT for upstream research intake |
| [ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) | Archive prioritization view over canonical archive claims |
| [ARCHIVE_CANONICAL_CLAIMS_RENDERED.md](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md) | Generated human-readable mirror of the canonical archive ledger |
| [HYDRA_FINAL.md](research/design/HYDRA_FINAL.md) | Promoted architecture doctrine summary |
| [HYDRA_RECONCILIATION.md](research/design/HYDRA_RECONCILIATION.md) | Promoted execution doctrine summary and active/reserve split |
| [HYDRA_ARCHIVE.md](research/design/HYDRA_ARCHIVE.md) | Reserve-only design/archive planning surfaces |
| [HYDRA_SPEC.md](research/design/HYDRA_SPEC.md) | Historical architecture spec only |
| [MORTAL_ANALYSIS.md](research/intel/MORTAL_ANALYSIS.md) | Mortal's architecture, training details, confirmed weaknesses |
| [OPPONENT_MODELING.md](research/design/OPPONENT_MODELING.md) | Opponent-modeling rationale; includes both active ideas and reserve/future extensions |
| [INFRASTRUCTURE.md](research/infrastructure/INFRASTRUCTURE.md) | Rust stack, data pipeline, training infra, hardware, deployment |
| [SEEDING.md](research/design/SEEDING.md) | RNG hierarchy, reproducibility, evaluation seed bank |
| [CHECKPOINTING.md](research/infrastructure/CHECKPOINTING.md) | Checkpoint format, save protocol, retention policy |
| [ECOSYSTEM.md](research/intel/ECOSYSTEM.md) | Useful repos, tooling, and framework references |
| [REWARD_DESIGN.md](research/design/REWARD_DESIGN.md) | Reward design and RVR notes |
| [COMMUNITY_INSIGHTS.md](research/intel/COMMUNITY_INSIGHTS.md) | Community observations and external signals |
| [REFERENCES.md](research/intel/REFERENCES.md) | Citation index |
| [TESTING.md](research/design/TESTING.md) | Testing strategy, correctness verification, property-based tests |
| [RUST_STACK.md](research/infrastructure/RUST_STACK.md) | 100% Rust decision and framework notes |

### Authority Layers and Promotion Flow

- `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`: canonical archive SSOT / source ledger for upstream research conclusions
- `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md`: archive prioritization view over the canonical archive
- `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md`: generated human-readable mirror of the canonical archive ledger
- `research/design/HYDRA_FINAL.md`: promoted architecture doctrine summary
- `research/design/HYDRA_RECONCILIATION.md`: promoted execution doctrine summary and best-next-action guide
- `docs/GAME_ENGINE.md`: current game-engine/runtime baseline
- `research/design/OPPONENT_MODELING.md`: detailed opponent-modeling rationale
- `research/design/HYDRA_ARCHIVE.md`: reserve-only design/archive planning
- `research/design/HYDRA_SPEC.md`: historical architecture summary only; do not use it as current implementation authority

## Status

Active implementation. `hydra-core` is already built out as a real baseline engine/encoder crate, and `hydra-train` contains a substantial training/model scaffold with partial advanced integration. The repo now has a real narrow replay-derived `safety_residual` supervision lane plus a replay/sample ExIt sidecar-first lane: offline replay ExIt labels can be generated as search-derived sidecar records, joined back into replay samples with provenance checks, and consumed by BC as a separate optional ExIt loss without polluting the replay action target. `delta_q` is now also closed across both live RL and replay/offline BC paths: the shared root-search producer emits a masked discard-compatible `Q(child)-Q(root)` target into RL batches, replay/offline `delta_q` sidecars can be generated and joined back into replay samples with provenance/version checks, and BC/train now has the narrow activation-hook + warmup-detach path needed to train on those labels without broadening other advanced heads. Immediate project needs are now stronger belief-teacher semantics and the next realism/strength tranche, not another `delta_q` closure pass.

## License

- **hydra-core** (encoder, training pipeline): [BSL 1.1](crates/hydra-core/LICENSE) -- free for non-commercial use, converts to Apache-2.0 on 2031-03-02
- **hydra-engine** (game rules): Apache-2.0 (vendored from riichienv-core)
