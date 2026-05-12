# hydra-train-types

Crate-local map for shared training data-transfer/scalar coordination types.

## Owns

- Backend-independent scalar gate/config types.
- Burn tensor target/config types defining loss inputs.
- Checkpoint/eval/head-gate/loss/orchestrator/phase/RL/selfplay DTOs.
- Types that must sit below runtime/model/algo/exec without cycles.

## Does not own

- Pure algorithms/loss execution: `hydra-train-algo` owns.
- Model layers/forward DTOs: `hydra-model` owns.
- CLI/config/preflight/status contracts: `hydra-train-runtime` owns.
- Execution orchestration: `hydra-train-exec` owns.
- User-facing binaries: `hydra-train` owns.

## Critical invariants

| Surface | Contract |
|---|---|
| Layer | shared type rail below train crates |
| Dependency posture | no dependency on `hydra-train` runtime/exec/model facades |
| Burn use | target/config tensor types only where needed |
| Goal | avoid training dependency cycles |

## Read next

- Training runbook: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md).
- Algorithms: [`crates/hydra-train-algo/README.md`](../hydra-train-algo/README.md).
- Runtime contracts: [`crates/hydra-train-runtime/README.md`](../hydra-train-runtime/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
