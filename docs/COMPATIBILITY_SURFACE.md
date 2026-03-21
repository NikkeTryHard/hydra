# Hydra Compatibility Surface

Compact compatibility contract for agents and developers touching runtime, training, or model-shape-sensitive code.

If you change any row in this file, you should assume matching docs, tests, and consumers need review.

Primary runtime owner: `docs/GAME_ENGINE.md`

## Compatibility table

| Surface | Current contract | Owner / source of truth | Notes |
|---|---|---|---|
| Encoder/model input shape | `192x34` | `docs/GAME_ENGINE.md`, `hydra-core/src/encoder.rs` | Live full contract |
| Baseline prefix | `85x34` (`channels 0..84`) | `docs/GAME_ENGINE.md` | Historical baseline-prefix only; not the full live encoder |
| Action space | `46` actions | `hydra-core/src/action.rs`, `docs/GAME_ENGINE.md` | Mortal-compatible action indexing |
| Riichi handling | two-phase | `hydra-core/src/action.rs`, `docs/GAME_ENGINE.md` | Declare riichi, then choose discard |
| Kan handling | two-phase | `hydra-core/src/action.rs`, `docs/GAME_ENGINE.md` | Declare kan, then choose specific kan when needed |
| Tile kind indices | `0..33` normalized tile indices | `hydra-core/src/tile.rs`, `docs/GAME_ENGINE.md` | 34 tile kinds |
| Aka tile behavior | aka tiles stay distinct in 136-format/action handling where needed | `hydra-core/src/tile.rs`, `hydra-core/src/action.rs` | Red 5m/5p/5s remain special cases |
| Legal action mask shape | `[bool; 46]` | `hydra-core/src/action.rs` | Training/inference must agree on mask semantics |
| Runtime/train entrypoint | `crates/hydra-train/src/bin/train.rs` | root `AGENTS.md`, crate docs | Main train binary entry surface |
| BC selected-runtime authority | fresh run = config-derived; epoch-boundary resume may reuse matching preflight-selected runtime; partial-epoch resume requires identical runtime | `crates/hydra-train/src/bin/train/bootstrap.rs`, `crates/hydra-train/src/bin/train/resume.rs` | Applies only to selected-runtime tuple (`train_microbatch_size`, `validation_microbatch_size`, `accum_steps`) |
| BC loader-runtime authority | config-derived | `crates/hydra-train/src/bin/train/bootstrap.rs`, `crates/hydra-train/src/bin/train/config_runtime.rs` | Matching BC preflight cache does not make loader-runtime authoritative |
| Runtime truth on drift | current code wins | `docs/GAME_ENGINE.md`, root `AGENTS.md` | Docs are compatibility aids, not stronger than code |

## Crate ownership quick reference

| Crate | Owns |
|---|---|
| `crates/hydra-engine` | vendored rules engine behavior |
| `crates/hydra-core` | runtime bridge, encoder, simulator, seeding, search/runtime feature plumbing |
| `crates/hydra-train` | model, targets, losses, training/inference orchestration |

## Read next

- Need the full runtime explanation? Read `docs/GAME_ENGINE.md`.
- Need the repo routing / trust map? Read `README.md`.
- Need active-vs-staged status? Read `research/design/HYDRA_RECONCILIATION.md` and `docs/CURRENT_STATUS.md`.
