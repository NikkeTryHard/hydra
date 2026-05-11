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
2. [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) — derived archive triage view over canonical source ledger; full render archived/generated on demand
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
| `crates/hydra-engine` | vendored riichi rules engine | Hydra runtime/training orchestration |
| `crates/hydra-runtime-types` | shared runtime rails/types | rules, encoder, training |
| `crates/hydra-safety` | safety rail primitives | policy/model/training execution |
| `crates/hydra-belief-search` | belief/search primitives | neural labels or training loop |
| `crates/hydra-encoder` | observation encoder components | simulator or training ownership |
| `crates/hydra-core` | public runtime bridge over engine/types/safety/encoder/search; simulator, action/tile API, seeding | Burn model/training/data pipelines or vendored rules ownership |
| `crates/hydra-data-core` | sample DTOs/scoring helpers | replay IO, shard/cache storage |
| `crates/hydra-replay-sidecar` | replay sidecar JSONL contracts | replay loading/conversion |
| `crates/hydra-replay-loader` | MJAI replay load + sample conversion | model/training loop |
| `crates/hydra-sample-cache` | parsed-sample cache format | replay parsing authority |
| `crates/hydra-bc-shards` | backend-agnostic BC shard host format | optimizer/model runtime |
| `crates/hydra-train-types` | training scalar coordination types | algorithms, model layers, CLI |
| `crates/hydra-model` | Burn neural model components | training algorithms/runtime CLI |
| `crates/hydra-train-algo` | pure Burn training algorithms/loss math | CLI/config/preflight, model definition |
| `crates/hydra-selfplay` | self-play coordination primitives | train CLI/runtime contracts |
| `crates/hydra-search-labels` | search-label generation | base search/runtime primitives |
| `crates/hydra-train-runtime` | train CLI/config/preflight/probe/status contracts | execution/model/algo ownership |
| `crates/hydra-train-exec` | training execution composition over runtime/model/algo/data crates | CLI/config/preflight/probe/status contracts |
| `crates/hydra-train` | compatibility facade + bins during split migration | settled ownership for new model/algo/runtime/exec code |

If deciding what to build next, follow Fresh-agent routing order above.
Historical design/planning docs now live as compact entries in `research/design/HYDRA_ARCHIVE.md`.

## Research

| File | What's In It |
|------|-------------|
| [ARCHIVE_CANONICAL_CLAIMS.jsonl](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl) | Epistemic root / canonical archive SSOT for upstream research intake |
| [ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) | Derived archive prioritization view over canonical archive claims |
| [HYDRA_FINAL.md](research/design/HYDRA_FINAL.md) | Promoted Max Hydra architecture doctrine |
| [HYDRA_RECONCILIATION.md](research/design/HYDRA_RECONCILIATION.md) | Promoted Hydra v1 active-path roadmap |
| [HYDRA_ARCHIVE.md](research/design/HYDRA_ARCHIVE.md) | Historical/reserve design parking lot for retired docs |
| [RESEARCH_DIGEST.md](research/evidence/RESEARCH_DIGEST.md) | Consolidated evidence: value decomposition, belief/search limits, safe exploitation, mean-field caveats |
| [ALGORITHM_WATCH.md](research/evidence/ALGORITHM_WATCH.md) | ACH/LuckyJ, ExIt, R-NaD/DRDA, CFR variants, algorithm status |
| [MAHJONG_AI_INTEL.md](research/intel/MAHJONG_AI_INTEL.md) | Competitor/community/tactical-gap intel; AGPL boundary |
| [REFERENCES.md](research/intel/REFERENCES.md) | Citation/source ledger |
| [INFRASTRUCTURE.md](research/infrastructure/INFRASTRUCTURE.md) | Rust/Burn, artifacts, checkpoint essentials, compute doctrine |
| [ENGINE_BENCHMARKS.md](research/infrastructure/ENGINE_BENCHMARKS.md) | Measured benchmark ledger |
| [SEEDING.md](research/design/SEEDING.md) | RNG hierarchy, reproducibility, eval seed bank |
| [TESTING.md](research/design/TESTING.md) | Testing strategy and high-risk verification |

## Status

Hydra in active impl. For current shipped/staged repo snapshot, read [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md). For runtime semantics and compatibility-sensitive invariants, read [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md).

## Operator docs

If need run/debug training stack rather than architecture docs, start here:

|- [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md) — train CLI modes, YAML contract, preflight/runtime authority, BC shards, replay sidecars, DeltaQ promotion, precision/CUDA notes
|- [`docker/train/README.md`](docker/train/README.md) — container, GHCR, Kaggle-compatible artifact, MJAI audit, coverage commands

## Testing and Coverage

Hydra uses `cargo nextest run --release` as default workspace test path. Coverage commands now live in [`docker/train/README.md`](docker/train/README.md).

## License

- **hydra-core** (encoder, training pipeline): [BSL 1.1](crates/hydra-core/LICENSE) -- free for non-commercial use, converts to Apache-2.0 on 2031-03-02
- **hydra-engine** (game rules): Apache-2.0 (vendored from riichienv-core)