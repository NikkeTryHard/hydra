# hydra-data-core

Crate-local map for pure data sample contracts and scoring helpers.

## Owns

- `MjaiSample` DTO and optional target fields.
- Data manifest/source/filter DTOs.
- Score delta bins/CDF/PDF/value helpers.
- GRP placement label helpers.
- One-hot action helper for sample targets.

## Does not own

- Replay loading/parsing: `hydra-replay-loader` owns.
- Parsed-sample cache file format: `hydra-sample-cache` owns.
- BC shard host format: `hydra-bc-shards` owns.
- Replay sidecar schemas: `hydra-replay-sidecar` owns.
- Model/training execution: train/model/exec crates own.

## Critical invariants

| Surface | Contract |
|---|---|
| Layer | pure DTO/scoring helper layer |
| Replay IO | not here |
| Storage format | not here except manifest/source DTOs |
| Action targets | match Hydra action-space semantics from `hydra-core` |

## Read next

- Data flow/operator contract: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md).
- Replay loader: [`crates/hydra-replay-loader/README.md`](../hydra-replay-loader/README.md).
- BC shard format: [`crates/hydra-bc-shards/README.md`](../hydra-bc-shards/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
