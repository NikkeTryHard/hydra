# hydra-train-exec

Crate-local map for training execution composition migrated out of train binaries.

## Owns

- BC fixed-shape train/probe execution helpers.
- Bootstrap, modes, orchestration, epoch/RL runners.
- Heavy preflight/probe/validation execution.
- Data pipeline and Burn-facing sample adapters.
- GPU/libtorch/CUDA-graph/NVTX/pinned-transfer adapters where feature-gated.
- Artifacts, resume state, progress accumulation, presentation helpers.
- BC shard build execution helpers.

## Does not own

- CLI/config/preflight/probe/status contracts: `hydra-train-runtime` owns.
- Model components: `hydra-model` owns.
- Pure algorithms/loss math: `hydra-train-algo` owns.
- User-facing binary/process boundary: `hydra-train` owns.
- Data/shard/cache/sidecar schema authority: sibling data crates own.

## Critical invariants

| Surface | Contract |
|---|---|
| Layer | execution composition over runtime/model/algo/data crates |
| Feature gates | CUDA graph/pinned transfer behind feature flags |
| API posture | some migrated compatibility seams intentionally preserve old train facade shape |
| Boundary | execution heavy code here, operator docs in `docs/TRAINING_RUNBOOK.md` |

## Read next

- Operator contract: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md).
- Runtime contracts: [`crates/hydra-train-runtime/README.md`](../hydra-train-runtime/README.md).
- User-facing binaries: [`crates/hydra-train/README.md`](../hydra-train/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
