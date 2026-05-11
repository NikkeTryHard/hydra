# Hydra Compatibility Surface

Compat contract for agents/devs touching runtime, training, model-shape-sensitive code.

Change any row here -> review matching docs, tests, consumers.

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
| Runtime/train entrypoint | `crates/hydra-train/src/bin/train.rs` parses/env-dispatches/delegates through `hydra-train-runtime` + `hydra-train-exec`; `hydra-train` remains compat facade + bins during migration | root `AGENTS.md`, `crates/hydra-train-runtime/src/lib.rs`, `crates/hydra-train-exec/src/lib.rs`, crate manifests | New CLI/config/preflight/probe/status contract code belongs in `hydra-train-runtime`; execution composition belongs in `hydra-train-exec`; bin-local modules are test/wrapper seams |
| BC selected-runtime authority | fresh run = config-derived; epoch-boundary resume may reuse matching preflight-selected runtime; partial-epoch resume requires identical runtime | `crates/hydra-train-runtime/src/config_runtime.rs`, `crates/hydra-train-runtime/src/preflight.rs`, `crates/hydra-train-exec/src/bootstrap.rs`, `crates/hydra-train-exec/src/resume.rs` | Only selected-runtime tuple (`train_microbatch_size`, `validation_microbatch_size`, `accum_steps`) |
| BC loader-runtime authority | config-derived | `crates/hydra-train-runtime/src/config_runtime.rs`, `crates/hydra-train-exec/src/bootstrap.rs` | Matching BC preflight cache does not make loader-runtime authoritative |
| Preflight cache key (v4) | hardware + workload + preflight config signature + explicit microbatch overrides | `crates/hydra-train-runtime/src/preflight.rs`, `crates/hydra-train-runtime/src/probe_request.rs` | `data_dir`, `seed`, `num_threads`, `buffer_games`, `buffer_samples` intentionally excluded from key |
| Preflight identical-run fast path | `run_preflight` and `run_rl_preflight` read cache before probing; cache hit on matching v4 key skips all probes | `crates/hydra-train-runtime/src/preflight.rs` plus `crates/hydra-train-exec` execution callsites | Does not affect bootstrap authority; probe result vectors empty on cache hit |
| Precision mode dispatch (BF16/AMP) | BC training dispatches by `PrecisionMode`; RL training and DeltaQ promotion explicitly gated with hard errors | `crates/hydra-train-runtime/src/config.rs`, `crates/hydra-train-exec/src/modes.rs` | RL BF16 = staged surface, not shipped |
| Runtime truth on drift | current code wins | `docs/GAME_ENGINE.md`, root `AGENTS.md` | Docs aid compat, not stronger than code |

## Crate ownership quick reference

| Crate | Owns |
|---|---|
| `crates/hydra-engine` | vendored riichi rules engine |
| `crates/hydra-runtime-types` | shared runtime rails/types |
| `crates/hydra-safety` | safety rails |
| `crates/hydra-belief-search` | belief/search primitives |
| `crates/hydra-encoder` | observation encoder components |
| `crates/hydra-core` | public runtime bridge, simulator, action/tile API, seeding |
| `crates/hydra-data-core` | sample DTOs/scoring helpers |
| `crates/hydra-replay-sidecar` | replay sidecar JSONL contracts |
| `crates/hydra-replay-loader` | MJAI replay loading/sample conversion |
| `crates/hydra-sample-cache` | parsed-sample cache format |
| `crates/hydra-bc-shards` | backend-agnostic BC shard host format |
| `crates/hydra-train-types` | training scalar coordination types |
| `crates/hydra-model` | Burn neural model components |
| `crates/hydra-train-algo` | pure Burn training algorithms/loss math |
| `crates/hydra-selfplay` | self-play coordination primitives |
| `crates/hydra-search-labels` | search-label generation |
| `crates/hydra-train-runtime` | train CLI/config/preflight/probe/status contracts |
| `crates/hydra-train-exec` | training execution composition over runtime/model/algo/data crates |
| `crates/hydra-train` | compatibility facade + bins during migration |

## Read next

- Need full runtime explanation? Read `docs/GAME_ENGINE.md`.
- Need repo routing / trust map? Read `README.md`.
- Need active-vs-staged status? Read `research/design/HYDRA_RECONCILIATION.md` and `docs/CURRENT_STATUS.md`.