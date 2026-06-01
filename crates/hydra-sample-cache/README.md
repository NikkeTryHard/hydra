# hydra-sample-cache

Crate-local map for parsed-sample cache format. Used by BC raw-path reuse.

## Owns

- `.samples.cache` file extension/format.
- Cache metadata: original source path, source identity, sample count.
- Parsed game payload: `MjaiSample` list + final scores.
- Cache read/write helpers and metadata-only read path.
- Cache magic/version validation.


## Internal modules

- `path`: cache suffix detection and MJAI source filename rewrite.
- `header`/`limits`: binary header metadata, magic/version/sample-count validation, final scores, and format limits.
- `sample`: `MjaiSample` binary payload encoding, optional flags, and presence checks.
- `primitives`: private endian/string/bool/f32 binary helpers.

## Does not own

- MJAI replay parsing: `hydra-replay-loader` owns.
- Sample DTO semantics: `hydra-data-core` owns.
- BC shard storage format: `hydra-bc-shards` owns.
- Cache-building CLI execution: `hydra-train` binary + `hydra-train-exec` own.

## Critical invariants

| Surface | Contract |
|---|---|
| Extension | `.samples.cache` |
| Magic | `HPSCACHE` |
| Version | `1` |
| Payload | binary `MjaiSample` stream plus final scores |
| Purpose | reuse parsed samples; not canonical replay source |

## Read next

- Training cache path: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md).
- Sample DTOs: [`crates/hydra-data-core/README.md`](../hydra-data-core/README.md).
- Replay loader: [`crates/hydra-replay-loader/README.md`](../hydra-replay-loader/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
