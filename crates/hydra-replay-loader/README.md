# hydra-replay-loader

Crate-local map for MJAI replay loading and replay-to-sample conversion.

## Owns

- Loose MJAI replay loading helpers.
- Replay archive helpers.
- Replay-to-target/sample conversion entrypoints.
- Target helper internals for replay-derived training data.

## Internal map

- `mjai_loader.rs`: public MJAI loader API glue and replay materialization loop.
- `mjai_loader/stats.rs`: replay materialization stats/profiling counters.
- `mjai_loader/dataset.rs`: `MjaiGame`, `MjaiDataset`, train-fraction normalization/splits.
- `mjai_loader/sink.rs`: streaming sample sink records and vector sink internals.
- `mjai_loader/tile.rs`: tile conversion, safety updates, next-discard helpers.
- `mjai_loader/scores.rs` / `validation.rs`: final score derivation and terminal score validation.
- `mjai_loader/decisions.rs` / `implicit_pass.rs`: replay decision preparation and implicit-pass handling.
- `mjai_loader/sidecar.rs`: replay target policy/provenance and joined sidecar lookup.
- `mjai_loader/stream.rs`: reader/stream/path entrypoint glue and compression inspection.

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
