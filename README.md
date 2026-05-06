# Hydra

Open-source Riichi Mahjong AI. Goal: rival [LuckyJ](https://haobofu.github.io/) (Tencent, 10.68 stable dan on Tenhou) with open weights.

> ## Compute support
> Research used Delta advanced computing/data resource, supported by National Science Foundation (award OAC 2005572) and State of Illinois. Delta = joint effort of University of Illinois Urbana-Champaign and National Center for Supercomputing Applications.

## Goal

Train mahjong AI that:
- Beats [Mortal](https://github.com/Equim-chan/Mortal) (~7-dan), nears LuckyJ-level play (10+ dan) in head-to-head eval
- Releases weights under permissive license
- Adds opponent modeling and inference-time search — two capabilities separating LuckyJ from other mahjong AIs

## Architecture

Hydra uses layered authority flow, built upward from archive handoff canon:

1. [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl) — epistemic root / canonical archive SSOT for upstream research conclusions
2. [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) and [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md) — derived archive views over canonical source ledger
3. [`research/design/HYDRA_FINAL.md`](research/design/HYDRA_FINAL.md) — promoted architecture doctrine from archive canon + repo validation
4. [`research/design/HYDRA_RECONCILIATION.md`](research/design/HYDRA_RECONCILIATION.md) — promoted operational doctrine and Hydra v1 roadmap from archive canon + repo validation
5. [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — promoted current-status snapshot for shipped repo surfaces
6. [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md) — runtime semantics and compatibility surfaces; current code wins if docs drift

Hydra doc split:

- `HYDRA_FINAL.md` = max-ceiling destination
- `HYDRA_RECONCILIATION.md` = roadmap to Hydra v1
- `docs/CURRENT_STATUS.md` = shipped/staged now

Raw `answer_*_combined.md` files in `research/agent_handoffs/combined_all_variants/` stay raw archive corpus, not promoted doctrine.

## Fresh-agent routing

If entering Hydra with zero prior memory, use this order and stop when truth enough for task:

1. `README.md` for repo routing
2. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` for canonical archive intake
3. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` for derived archive triage
4. `research/design/HYDRA_RECONCILIATION.md` for Hydra v1 roadmap
5. `research/design/HYDRA_FINAL.md` for long-term ceiling
6. `docs/CURRENT_STATUS.md` for shipped/staged truth
7. `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md` for runtime truth

`combined_all_variants/` stays raw archive corpus for provenance only.

## Status vocabulary

For impl work, choose next lane from
`research/design/HYDRA_RECONCILIATION.md`, confirm shipped/staged status in
`docs/CURRENT_STATUS.md`, then confirm exact runtime contracts in
`docs/GAME_ENGINE.md` plus current code.

| Term | Meaning |
|---|---|
| `active path` | what Hydra should optimize/build now |
| `shipped baseline` | implemented, part of current live baseline |
| `implemented but not default-on` | implemented in code, intentionally not default path |
| `implemented but staged` | implemented enough to exist, activation/promotion intentionally deferred |
| `reserve shelf` | preserved later-work direction, not current mainline |
| `blocked` | not ready because real dependency or semantic gap remains |
| `rejected` | not part of current plan |
| `historical` | preserved context only; not governing truth |

## Crate ownership

| Crate | Owns | Does not own |
|---|---|---|
| `crates/hydra-engine` | vendored rules engine behavior | Hydra-specific runtime/training orchestration |
| `crates/hydra-core` | runtime bridge, encoder, simulator, seeding, search/runtime feature plumbing | Burn training logic or vendored rules ownership |
| `crates/hydra-train` | model, targets, losses, BC/RL/self-play orchestration, train binary | low-level rules engine behavior |

If deciding what to build next, follow Fresh-agent routing order above.
`research/design/HYDRA_SPEC.md` stays historical context only.

## Research

| File | What's In It |
|------|-------------|
| [ARCHIVE_CANONICAL_CLAIMS.jsonl](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl) | Epistemic root / canonical archive SSOT for upstream research intake |
| [ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) | Derived archive prioritization view over canonical archive claims |
| [ARCHIVE_CANONICAL_CLAIMS_RENDERED.md](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md) | Generated human-readable mirror of canonical archive ledger |
| [HYDRA_FINAL.md](research/design/HYDRA_FINAL.md) | Promoted architecture doctrine summary |
| [HYDRA_RECONCILIATION.md](research/design/HYDRA_RECONCILIATION.md) | Promoted operational doctrine summary and Hydra v1 roadmap |
| [HYDRA_ARCHIVE.md](research/design/HYDRA_ARCHIVE.md) | Reserve-only design/archive planning surfaces |
| [HYDRA_SPEC.md](research/design/HYDRA_SPEC.md) | Historical architecture spec only |
| [MORTAL_ANALYSIS.md](research/intel/MORTAL_ANALYSIS.md) | Mortal architecture, training details, confirmed weaknesses |
| [OPPONENT_MODELING.md](research/design/OPPONENT_MODELING.md) | Opponent-modeling rationale; includes active ideas and reserve/future extensions |
| [INFRASTRUCTURE.md](research/infrastructure/INFRASTRUCTURE.md) | Rust stack, data pipeline, training infra, hardware, deployment |
| [SEEDING.md](research/design/SEEDING.md) | RNG hierarchy, reproducibility, evaluation seed bank |
| [CHECKPOINTING.md](research/infrastructure/CHECKPOINTING.md) | Checkpoint format, save protocol, retention policy |
| [ECOSYSTEM.md](research/intel/ECOSYSTEM.md) | Useful repos, tooling, framework references |
| [REWARD_DESIGN.md](research/design/REWARD_DESIGN.md) | Reward design and RVR notes |
| [COMMUNITY_INSIGHTS.md](research/intel/COMMUNITY_INSIGHTS.md) | Community observations and external signals |
| [REFERENCES.md](research/intel/REFERENCES.md) | Citation index |
| [TESTING.md](research/design/TESTING.md) | Testing strategy, correctness verification, property-based tests |
| [RUST_STACK.md](research/infrastructure/RUST_STACK.md) | 100% Rust decision and framework notes |

## Status

Hydra in active impl. For current shipped/staged repo snapshot, read [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md). For runtime semantics and compatibility-sensitive invariants, read [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md).

## Operator docs

If need run/debug training stack rather than read architecture docs first, start here:

- [`docs/TRAINING_WORKFLOWS.md`](docs/TRAINING_WORKFLOWS.md) — training entry modes, YAML contract, BC/RL shape, sidecar-enabled training
- [`docs/PREFLIGHT_AND_RUNTIME_SELECTION.md`](docs/PREFLIGHT_AND_RUNTIME_SELECTION.md) — preflight cache, selected-runtime authority, probe flows, runtime reuse rules
- [`docs/REPLAY_SIDECARS.md`](docs/REPLAY_SIDECARS.md) — ExIt/DeltaQ sidecar generation and replay-time hydration contracts
- [`docs/MJAI_AUDIT_AND_FAILURE_TRIAGE.md`](docs/MJAI_AUDIT_AND_FAILURE_TRIAGE.md) — replay corpus validation, failure inventories, triage tooling
- [`docs/BC_SHARDS.md`](docs/BC_SHARDS.md) — BC shard production, manifest interpretation, training consumption
- [`docs/DELTAQ_PROMOTION.md`](docs/DELTAQ_PROMOTION.md) — DeltaQ promotion gates, arena confirmation, artifact interpretation
- [`docker/train/README.md`](docker/train/README.md) — container execution contract

## Testing and Coverage

Hydra uses `cargo nextest run --release` as default workspace test path and `cargo-llvm-cov` for workspace-wide coverage reporting. For local coverage generation details, read [`docs/COVERAGE.md`](docs/COVERAGE.md).

## License

- **hydra-core** (encoder, training pipeline): [BSL 1.1](crates/hydra-core/LICENSE) -- free for non-commercial use, converts to Apache-2.0 on 2031-03-02
- **hydra-engine** (game rules): Apache-2.0 (vendored from riichienv-core)