# hydra-replay-sidecar

Crate-local map for pure replay sidecar JSONL contracts.

## Owns

- Replay-indexed sidecar lookup keys.
- ExIt replay sidecar records/indexes.
- DeltaQ replay sidecar records/indexes.
- Sidecar provenance/semantics constants.
- JSONL record reader helpers.
- Legal-mask digest/copy helpers.

## Does not own

- Replay parsing/conversion: `hydra-replay-loader` owns.
- Sidecar generation algorithms: `hydra-search-labels` owns.
- BC shard embedding of sidecar targets: `hydra-bc-shards` / `hydra-train-exec` own.
- Training promotion/default-on policy: docs + runtime/exec crates own.

## Critical invariants

| Surface | Contract |
|---|---|
| Layer | pure schema/index/hash/provenance contracts |
| Format | JSONL records keyed by replay decision identity |
| Model independence | no training runtime dependency |
| Validation | contract validators live with record/schema types |

## Read next

- Training runbook sidecar lanes: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md).
- Search-label producers: [`crates/hydra-search-labels/README.md`](../hydra-search-labels/README.md).
- Replay loader: [`crates/hydra-replay-loader/README.md`](../hydra-replay-loader/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
