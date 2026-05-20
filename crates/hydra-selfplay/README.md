# hydra-selfplay

Crate-local map for self-play coordination primitives.

## Owns

- NN action selector used in live self-play loops.
- Observation encoding handoff to policy inference.
- Safety tracking during self-play decisions.
- Trajectory-to-RL-batch conversion helpers.
- Live ExIt/DeltaQ validation hooks over self-play trajectories.
- Cooperative state used by live search-label coordination.
- Allocation cleanup wave: active-player turn staging uses fixed 4-seat buffers; RL advantage scratch reused; misleading batched wrapper removed after caller audit.

## Does not own

- Model architecture/inference impl details: `hydra-model` owns.
- Pure training algorithms: `hydra-train-algo` owns.
- Search-label construction logic: `hydra-search-labels` owns.
- Runtime simulator/action API: `hydra-core` owns.
- Train CLI/execution loops: `hydra-train` / `hydra-train-exec` own.

## Critical invariants

| Surface | Contract |
|---|---|
| Default game mode | 4-player mode unless caller config says otherwise |
| Action space | Hydra 46-action mapping via `hydra-core` |
| Safety | per-player `SafetyInfo` reset/update across games |
| Max steps | bounded guard to avoid runaway games |
| Boundary | coordination primitives, not operator CLI |

## Read next

- Runtime semantics: [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md).
- Training runbook: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md).
- Search labels: [`crates/hydra-search-labels/README.md`](../hydra-search-labels/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
