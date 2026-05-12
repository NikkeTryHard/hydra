# hydra-replay-loader

Crate-local map for MJAI replay loading and replay-to-sample conversion.

## Owns

- Loose MJAI replay loading helpers.
- Replay archive helpers.
- Replay-to-target/sample conversion entrypoints.
- Target helper internals for replay-derived training data.

## Does not own

- Pure sample DTO definitions: `hydra-data-core` owns.
- Parsed-sample cache format: `hydra-sample-cache` owns.
- Replay sidecar schemas: `hydra-replay-sidecar` owns.
- BC shard host format: `hydra-bc-shards` owns.
- Model/training execution: train/model/exec crates own.

## Critical invariants

| Surface | Contract |
|---|---|
| Input | loose MJAI files and replay archives |
| Output | Hydra sample/target data for downstream crates |
| Boundary | loader/converter, not storage-format owner |
| Rules source | engine/runtime semantics follow `docs/GAME_ENGINE.md` |

## Read next

- Replay/data operator contract: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md).
- Runtime semantics: [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md).
- Data DTOs: [`crates/hydra-data-core/README.md`](../hydra-data-core/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
