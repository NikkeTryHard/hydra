# Hydra Compatibility Surface

Compact compat contract for agents/devs touching runtime, training, or model-shape-sensitive code.

If you change any row in this file, assume matching docs, tests, consumers need review.

Primary runtime owner: `docs/GAME_ENGINE.md`

## Compatibility table

| Surface | Current contract | Owner / source of truth | Notes |
|---|---|---|---|
| Encoder/model input shape | `192x34` | `docs/GAME_ENGINE.md`, `hydra-core/src/encoder.rs` | Live full contract |
| Baseline prefix | `85x34` (`channels 0..84`) | `docs/GAME_ENGINE.md` | Historical baseline-prefix only; not full live encoder |
| Action space | `46` actions | `hydra-core/src/action.rs`, `docs/GAME_ENGINE.md` | Mortal-compatible action indexing |
| Riichi handling | two-phase | `hydra-core/src/action.rs`, `docs/GAME_ENGINE.md` | Declare riichi, then choose discard |
| Kan handling | two-phase | `hydra-core/src/action.rs`, `docs/GAME_ENGINE.md` | Declare kan, then choose specific kan when needed |
| Tile kind indices | `0..33` normalized tile indices | `hydra-core/src/tile.rs`, `docs/GAME_ENGINE.md` | 34 tile kinds |
| Aka tile behavior | aka tiles stay distinct in 136-format/action handling where needed | `hydra-core/src/tile.rs`, `hydra-core/src/action.rs` | Red 5m/5p/5s stay special cases |
| Legal action mask shape | `[bool; 46]` | `hydra-core/src/action.rs` | Training/inference must match mask semantics |
| Runtime/train entrypoint | `crates/hydra-train/src/bin/train.rs` | root `AGENTS.md`, crate docs | Main train binary entry surface |
| BC selected-runtime authority | fresh run = config-derived; epoch-boundary resume may reuse matching preflight-selected runtime; partial-epoch resume requires identical runtime | `crates/hydra-train/src/bin/train/bootstrap.rs`, `crates/hydra-train/src/bin/train/resume.rs` | Applies only to selected-runtime tuple (`train_microbatch_size`, `validation_microbatch_size`, `accum_steps`) |
| BC loader-runtime authority | config-derived | `crates/hydra-train/src/bin/train/bootstrap.rs`, `crates/hydra-train/src/bin/train/config_runtime.rs` | Matching BC preflight cache does not make loader-runtime authoritative |
| Preflight cache key (v4) | hardware + workload + preflight config signature + explicit microbatch overrides | `crates/hydra-train/src/bin/train/preflight_fingerprint.rs`, `crates/hydra-train/src/preflight.rs` | `data_dir`, `seed`, `num_threads`, `buffer_games`, `buffer_samples` deliberately excluded from key |
| Preflight identical-run fast path | `run_preflight` and `run_rl_preflight` read cache before probing; cache hit on matching v4 key skips all probes | `crates/hydra-train/src/bin/train/preflight_runtime.rs` | Does not affect bootstrap authority; probe result vectors empty on cache hit |
| Precision mode dispatch (BF16/AMP) | BC training dispatches by `PrecisionMode`; RL training and DeltaQ promotion explicitly gated with hard errors | `crates/hydra-train/src/bin/train/modes.rs` | RL BF16 = staged surface, not shipped |
| Runtime truth on drift | current code wins | `docs/GAME_ENGINE.md`, root `AGENTS.md` | Docs aid compat, not stronger than code |

## Crate ownership quick reference

| Crate | Owns |
|---|---|
| `crates/hydra-engine` | vendored rules engine behavior |
| `crates/hydra-core` | runtime bridge, encoder, simulator, seeding, search/runtime feature plumbing |
| `crates/hydra-train` | model, targets, losses, training/inference orchestration |

## Read next

- Need full runtime explanation? Read `docs/GAME_ENGINE.md`.
- Need repo routing / trust map? Read `README.md`.
- Need active-vs-staged status? Read `research/design/HYDRA_RECONCILIATION.md` and `docs/CURRENT_STATUS.md`.