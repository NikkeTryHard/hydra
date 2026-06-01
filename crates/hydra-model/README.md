# hydra-model

Crate-local map for Burn neural model components used by legacy/reference Rust path.

Default plain BC model/loss/checkpoint ownership lives in Python/PyTorch. This
crate remains for Rust/Burn fallback and debug runs plus advanced lanes not yet
owned by Python default path: ExIt, DeltaQ, belief fields, mixture weights,
opponent hand type, safety residual, oracle critic, and search-as-feature.

## Owns

- AMP compatibility helpers.
- SE-ResNet backbone modules.
- Model output heads.
- Inference facade plus server, pure CPU policy, and tensor utility modules.
- Full Hydra model facade plus init, CPU adapter, forward graph, and output DTO modules.
- Search-as-Feature adaptor modules.
- ONNX Runtime policy loader split across metadata, device parsing, runtime, and output extraction modules.

## Does not own

- Training algorithms/loss math: `hydra-train-algo` owns.
- Runtime CLI/config/preflight: `hydra-train-runtime` owns.
- Execution orchestration/artifacts/GPU adapters: `hydra-train-exec` owns.
- User-facing binaries: `hydra-train` owns.
- Runtime observation/action semantics: `hydra-core` + docs own.

## Critical invariants

| Surface | Contract |
|---|---|
| Framework | Burn model components |
| Public shape | follows runtime/model contract in docs/current code |
| Layer | model definition/inference helpers, not train loop |
| Search-as-feature | adapters here; label generation elsewhere |

## Read next

- Training operator contract: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md).
- Runtime shape contract: [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md).
- Training algos: [`crates/hydra-train-algo/README.md`](../hydra-train-algo/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
