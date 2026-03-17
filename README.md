# Hydra

Open-source Riichi Mahjong AI. The goal is to build an AI that rivals [LuckyJ](https://haobofu.github.io/) (Tencent, 10.68 stable dan on Tenhou) with open weights.

## Goal

Train a mahjong AI that:
- Surpasses [Mortal](https://github.com/Equim-chan/Mortal) (~7-dan) and approaches LuckyJ-level play (10+ dan) in head-to-head evaluation
- Releases weights under a permissive license
- Adds opponent modeling and inference-time search — the two capabilities that separate LuckyJ from all other mahjong AIs

## Architecture

Hydra uses a layered authority flow built from the archive handoff canon upward:

1. [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl) — epistemic root / canonical archive SSOT for upstream research conclusions
2. [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) and [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md) — derived archive views over that canonical source ledger
3. [`research/design/HYDRA_FINAL.md`](research/design/HYDRA_FINAL.md) — promoted architecture doctrine built from archive canon plus repo validation
4. [`research/design/HYDRA_RECONCILIATION.md`](research/design/HYDRA_RECONCILIATION.md) — promoted operational doctrine / active-path owner built from archive canon plus repo validation
5. [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — promoted current-status snapshot for already-built repo surfaces
6. [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md) — runtime semantics and compatibility surfaces; current code wins when docs drift

Raw `answer_*_combined.md` files in `research/agent_handoffs/combined_all_variants/` remain raw archive corpus, not promoted doctrine.

## Fresh-agent routing

If you are entering Hydra with zero prior memory, use this order and stop when you have enough truth for the task.

| Question | Primary file | What it is | What it is not |
|---|---|---|---|
| Where should I start? | `README.md` | repo entry router | not the full status board |
| What upstream research claims survived intake, and what powers the rest of the repo's doctrine? | `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` | epistemic root / canonical archive intake / source ledger | not auto-promoted repo status by itself |
| What archive-derived triage survived intake before promotion? | `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` | derived archive prioritization view | not final runtime truth or promoted operational doctrine |
| What is Hydra trying to become? | `research/design/HYDRA_FINAL.md` | promoted architecture doctrine / north-star target | not the owner of current shipped status |
| What is active now vs staged/reserve/historical? | `research/design/HYDRA_RECONCILIATION.md` | promoted operational doctrine / operational status owner | not raw archive intake |
| What is already built in the repo today? | `docs/CURRENT_STATUS.md` | promoted current-status snapshot | not runtime compatibility truth by itself |
| What runtime semantics and invariants are true today? | `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md` | runtime truth and compatibility surface | not strategic architecture priority |
| What is preserved raw archive corpus? | `research/agent_handoffs/combined_all_variants/` | evidence/provenance archive | not current implementation doctrine |

## Trust and status vocabulary

### Trust map

| Surface | Role | Trust level for implementation work | Use it for | Do not use it for |
|---|---|---|---|---|
| `README.md` | repo router | entry-only | deciding where to read next | detailed runtime status |
| `ARCHIVE_CANONICAL_CLAIMS.jsonl` | epistemic root / canonical archive intake | highest for upstream research provenance | preserving upstream conclusions with provenance; refreshing promoted doctrine when it drifts | assuming something is already implemented without code/runtime validation |
| `ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` | derived archive prioritization view | high, but subordinate to the JSONL ledger | archive-derived do-now / phase-next triage | settling promoted doctrine or current runtime truth by itself |
| `HYDRA_FINAL.md` | promoted architecture doctrine | high for target architecture | north-star design direction | shipped-status ownership |
| `HYDRA_RECONCILIATION.md` | promoted operational doctrine | highest for active-path sequencing | active vs staged vs reserve decisions; best-next-task guidance | runtime compatibility details |
| `docs/CURRENT_STATUS.md` | promoted status snapshot | high for shipped/staged snapshot | checking what is already built today | replacing runtime/code truth |
| `docs/GAME_ENGINE.md` / `docs/COMPATIBILITY_SURFACE.md` | runtime truth | highest for runtime semantics | encoder/action/runtime contracts | archive promotion decisions |
| current code | live implementation truth | final runtime truth | settling doc drift | skipping the doc-routing model entirely |
| `combined_all_variants/` | raw archive corpus | evidence-only | provenance, archive archaeology | current Hydra doctrine |

### Status vocabulary

| Term | Meaning |
|---|---|
| `active path` | what Hydra should optimize/build now |
| `shipped baseline` | implemented and part of the current live baseline |
| `implemented but not default-on` | implemented in code, intentionally not the default path |
| `implemented but staged` | implemented enough to exist, but activation/promotion is intentionally deferred |
| `reserve shelf` | preserved later-work direction, not current mainline |
| `blocked` | not ready because a real dependency or semantic gap remains |
| `rejected` | not part of the current plan |
| `historical` | preserved context only; not governing truth |

## Crate ownership

| Crate | Owns | Does not own |
|---|---|---|
| `crates/hydra-engine` | vendored rules engine behavior | Hydra-specific runtime/training orchestration |
| `crates/hydra-core` | runtime bridge, encoder, simulator, seeding, search/runtime feature plumbing | Burn training logic or vendored rules ownership |
| `crates/hydra-train` | model, targets, losses, BC/RL/self-play orchestration, train binary | low-level rules engine behavior |

If you are deciding what to build next, read these in order:
- [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl) — epistemic root / canonical archive SSOT / source ledger
- [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) — derived archive prioritization view
- [`research/design/HYDRA_FINAL.md`](research/design/HYDRA_FINAL.md) — promoted architecture doctrine summary
- [`research/design/HYDRA_RECONCILIATION.md`](research/design/HYDRA_RECONCILIATION.md) — promoted operational doctrine summary + best-next-action guide
- [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — promoted current shipped/staged repo snapshot
- [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) — current game-engine/runtime baseline
- [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md) — current compatibility contract
- [`research/design/IMPLEMENTATION_ROADMAP.md`](research/design/IMPLEMENTATION_ROADMAP.md) — staged implementation reference
- [`research/design/HYDRA_ARCHIVE.md`](research/design/HYDRA_ARCHIVE.md) — reserve-only design/archive planning

`research/design/HYDRA_SPEC.md` is historical context only.

## Research

| File | What's In It |
|------|-------------|
| [ARCHIVE_CANONICAL_CLAIMS.jsonl](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl) | Epistemic root / canonical archive SSOT for upstream research intake |
| [ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) | Derived archive prioritization view over canonical archive claims |
| [ARCHIVE_CANONICAL_CLAIMS_RENDERED.md](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md) | Generated human-readable mirror of the canonical archive ledger |
| [HYDRA_FINAL.md](research/design/HYDRA_FINAL.md) | Promoted architecture doctrine summary |
| [HYDRA_RECONCILIATION.md](research/design/HYDRA_RECONCILIATION.md) | Promoted operational doctrine summary and active/reserve split |
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

- `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`: epistemic root / canonical archive SSOT / source ledger for upstream research conclusions
- `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md`: derived archive prioritization view over the canonical archive
- `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md`: generated human-readable mirror of the canonical archive ledger
- `research/design/HYDRA_FINAL.md`: promoted architecture doctrine summary
- `research/design/HYDRA_RECONCILIATION.md`: promoted operational doctrine summary and best-next-action guide
- `docs/CURRENT_STATUS.md`: promoted shipped/staged repo status snapshot for already-built surfaces
- `docs/GAME_ENGINE.md`: current game-engine/runtime baseline
- `docs/COMPATIBILITY_SURFACE.md`: compact compatibility contract for runtime/training-sensitive invariants
- `research/design/OPPONENT_MODELING.md`: detailed opponent-modeling rationale
- `research/design/HYDRA_ARCHIVE.md`: reserve-only design/archive planning
- `research/design/HYDRA_SPEC.md`: historical architecture summary only; do not use it as current implementation authority

## Status

Hydra is in active implementation. For the current shipped/staged repo snapshot, read [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md). For runtime semantics and compatibility-sensitive invariants, read [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md).

## License

- **hydra-core** (encoder, training pipeline): [BSL 1.1](crates/hydra-core/LICENSE) -- free for non-commercial use, converts to Apache-2.0 on 2031-03-02
- **hydra-engine** (game rules): Apache-2.0 (vendored from riichienv-core)
