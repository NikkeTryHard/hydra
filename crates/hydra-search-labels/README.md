# hydra-search-labels

Crate-local map for search-label generation and validation support.

## Owns

- ExIt target construction, losses, AFBS tree adapters.
- Live/root-decision search-label producers and adapter traits.
- Replay-indexed offline ExIt sidecar helpers.
- Replay-indexed offline DeltaQ sidecar helpers.
- DeltaQ and ExIt validation reports/collectors.
- Shared validation metric helpers for label harnesses.

## Does not own

- Base search/belief primitives: `hydra-belief-search` and `hydra-core` own.
- Replay sidecar schema contracts: `hydra-replay-sidecar` owns.
- Self-play coordination loop: `hydra-selfplay` owns.
- Training execution/promotion policy: `hydra-train-exec` / docs own.
- Model architecture: `hydra-model` owns.

## Critical invariants

| Surface | Contract |
|---|---|
| Labels | ExIt and DeltaQ search-label generation |
| Sidecars | produces/validates data matching `hydra-replay-sidecar` schemas |
| Sharing | used by self-play and replay producers without `hydra-train` internals |
| Default status | promotion/default-on governed by docs/runtime, not this crate alone |

## Read next

- Training runbook sidecar/search-label lanes: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md).
- Replay sidecar schema: [`crates/hydra-replay-sidecar/README.md`](../hydra-replay-sidecar/README.md).
- Self-play: [`crates/hydra-selfplay/README.md`](../hydra-selfplay/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
