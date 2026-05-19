# hydra-train-runtime

Crate-local map for training runtime contracts. Owns CLI/config/preflight/probe/status shape, not heavy execution.

## Owns

- Training config and runtime-materialized config DTOs.
- DeltaQ promotion contract types.
- Head gates and loss policy contracts.
- Preflight benchmark request/report contracts.
- Progress/schedule/status/validation DTOs.
- YAML/CLI-facing runtime shape consumed by execution layer; YAML remains runtime authority for normal training.

## Does not own

- Heavy train/preflight/probe execution: `hydra-train-exec` owns.
- Model architecture: `hydra-model` owns.
- Pure algorithms/loss math: `hydra-train-algo` owns.
- User-facing process/binary boundary: `hydra-train` owns.
- Data storage formats: data/shard/cache/sidecar crates own.

## Critical invariants

| Surface | Contract |
|---|---|
| Layer | config/CLI/preflight/probe/status contracts |
| Serialization | serde YAML/JSON where operator-facing |
| Boundary | no heavy execution ownership |
| Authority | runtime contract truth for `hydra-train-exec` and `hydra-train` |
| Preflight benchmark | exact tuple CLI input; markdown/report rows only; no config, manifest, dataset, cache authority, automatic winner, or YAML mutation |
| Shard workflow | build shards -> optional manifestless markdown preflight -> human edits YAML if desired -> train from `bc_shards_manifest_path` |

## Read next

- Operator contract: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md).
- Execution layer: [`crates/hydra-train-exec/README.md`](../hydra-train-exec/README.md).
- User-facing binaries: [`crates/hydra-train/README.md`](../hydra-train/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
