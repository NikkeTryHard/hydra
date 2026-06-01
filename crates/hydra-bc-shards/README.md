# hydra-bc-shards

Crate-local map for backend-agnostic behavior-cloning shard host format.

## Owns

- Compact v3 BC shard manifest/header constants and validators.
- Host batch/scratch DTOs.
- BC shard reader/writer helpers.
- Split manifest/build totals descriptors.
- Optional target flags for safety residual, ExIt, DeltaQ, oracle/masks.

## Internal module map

- `manifest/`: frozen ABI constants, manifest DTOs, and contract validators.
- `reader/`: header validation, record decoding, observation reconstruction, and suit/action augmentation.
- `writer/`: header writing, record encoding, observation fact encoding, packed masks, and primitive writes.
- `host.rs`: backend-agnostic host batches and reusable scratch buffers.

## Does not own

- Replay parsing/sample creation: `hydra-replay-loader` and `hydra-data-core` own.
- Parsed-sample cache: `hydra-sample-cache` owns.
- Shard build CLI/execution orchestration: `hydra-train` / `hydra-train-exec` own.
- Burn tensor materialization/training loop: `hydra-train-exec` owns.

## Critical invariants

| Surface | Contract |
|---|---|
| Format | compact-only v3 host shard format; dense v2 invalid |
| Observation storage | replay-fact baseline obs only; reader rebuilds `192x34` f32 with advanced/search/Hand-EV channels absent/zero |
| Legal/action masks | 46-action Hydra space; masks packed on disk, expanded on read |
| Optional labels | gated by explicit flags; unsupported bits hard-error |
| Validators | manifest/header/descriptor byte/count contracts validate counts and byte lengths before allocation; drift hard-errors |

## Read next

- BC shard operator contract: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md).
- Data DTOs: [`crates/hydra-data-core/README.md`](../hydra-data-core/README.md).
- Execution adapters: [`crates/hydra-train-exec/README.md`](../hydra-train-exec/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
