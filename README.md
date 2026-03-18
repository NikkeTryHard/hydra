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
4. [`research/design/HYDRA_RECONCILIATION.md`](research/design/HYDRA_RECONCILIATION.md) — promoted operational doctrine and roadmap to Hydra v1 built from archive canon plus repo validation
5. [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — promoted current-status snapshot for already-built repo surfaces
6. [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md) — runtime semantics and compatibility surfaces; current code wins when docs drift

Hydra's documentation split is simple:

- `HYDRA_FINAL.md` describes the max-ceiling destination
- `HYDRA_RECONCILIATION.md` is the roadmap to Hydra v1
- `docs/CURRENT_STATUS.md` says what is already shipped or still staged today

Raw `answer_*_combined.md` files in `research/agent_handoffs/combined_all_variants/` remain raw archive corpus, not promoted doctrine.

## Fresh-agent routing

If you are entering Hydra with zero prior memory, use this order and stop when you have enough truth for the task:

1. `README.md` for repo routing
2. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` for canonical archive intake
3. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` for derived archive triage
4. `research/design/HYDRA_RECONCILIATION.md` for the roadmap to Hydra v1
5. `research/design/HYDRA_FINAL.md` for the long-term ceiling
6. `docs/CURRENT_STATUS.md` for shipped/staged truth
7. `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md` for runtime truth

`combined_all_variants/` remains raw archive corpus for provenance only.

## Status vocabulary

For implementation work, choose the next lane from
`research/design/HYDRA_RECONCILIATION.md`, confirm whether it already exists in
`docs/CURRENT_STATUS.md`, and confirm exact runtime contracts in
`docs/GAME_ENGINE.md` plus current code.

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

If you are deciding what to build next, follow the Fresh-agent routing order above.
`research/design/HYDRA_SPEC.md` remains historical context only.

## Research

| File | What's In It |
|------|-------------|
| [ARCHIVE_CANONICAL_CLAIMS.jsonl](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl) | Epistemic root / canonical archive SSOT for upstream research intake |
| [ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) | Derived archive prioritization view over canonical archive claims |
| [ARCHIVE_CANONICAL_CLAIMS_RENDERED.md](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md) | Generated human-readable mirror of the canonical archive ledger |
| [HYDRA_FINAL.md](research/design/HYDRA_FINAL.md) | Promoted architecture doctrine summary |
| [HYDRA_RECONCILIATION.md](research/design/HYDRA_RECONCILIATION.md) | Promoted operational doctrine summary and roadmap to Hydra v1 |
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

## Status

Hydra is in active implementation. For the current shipped/staged repo snapshot, read [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md). For runtime semantics and compatibility-sensitive invariants, read [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md).

## Testing and Coverage

Hydra uses `cargo nextest run --release` as the default workspace test path and `cargo-llvm-cov` for workspace-wide coverage reporting. For local coverage generation details, read [`docs/COVERAGE.md`](docs/COVERAGE.md).

## License

- **hydra-core** (encoder, training pipeline): [BSL 1.1](crates/hydra-core/LICENSE) -- free for non-commercial use, converts to Apache-2.0 on 2031-03-02
- **hydra-engine** (game rules): Apache-2.0 (vendored from riichienv-core)
