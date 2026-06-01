# hydra-belief-search

Crate-local map for belief-state/search primitives. Keep independent of encoder impl details.

## Owns

- AFBS support modules: tree/PUCT, provenance cache, ponder queue, leaf batching.
- CT-SMC belief/search pieces.
- Endgame, hand-EV, robust-opponent, Sinkhorn support.
- Batch shanten search helpers.
- Search-side primitives used by runtime/training labels.

## Does not own

- Observation tensor encoding: `hydra-encoder` owns.
- Public runtime compatibility/re-exports: `hydra-core` owns.
- Search label construction for training: `hydra-search-labels` owns.
- Training loops/model/losses: train/model/algo crates own.

## Critical invariants

| Surface | Contract |
|---|---|
| Engine dependency | may use `hydra-engine` / `riichienv_core` for rules-facing state |
| Runtime deps | depends below encoder/train where possible |
| Encoder boundary | no ownership of channel ABI |
| Use | search/belief feature and label support, not CLI/runtime orchestration |

## Read next

- Runtime/search feature contract: [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md).
- Public bridge: [`crates/hydra-core/README.md`](../hydra-core/README.md).
- Search labels: [`crates/hydra-search-labels/README.md`](../hydra-search-labels/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
