# hydra-train-algo

Crate-local map for pure training algorithms shared by Hydra training crates.

## Owns

- ACH helpers.
- BC algorithm/loss helpers.
- Distillation helpers.
- DRDA helpers.
- GAE helpers.
- Shared loss math that is not execution/CLI-bound.

## Does not own

- Model architecture: `hydra-model` owns.
- Scalar/type contracts: `hydra-train-types` owns.
- CLI/config/preflight/status: `hydra-train-runtime` owns.
- Execution loops/artifacts/GPU adapters: `hydra-train-exec` owns.
- User-facing train binaries: `hydra-train` owns.

## Critical invariants

| Surface | Contract |
|---|---|
| Layer | pure algorithm/loss math |
| Framework | Burn where tensor math needed |
| No ownership | no CLI/env/preflight/artifact policy |
| Reuse | shared by exec/selfplay/search-label lanes |

## Read next

- Training runbook: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md).
- Model components: [`crates/hydra-model/README.md`](../hydra-model/README.md).
- Execution layer: [`crates/hydra-train-exec/README.md`](../hydra-train-exec/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
