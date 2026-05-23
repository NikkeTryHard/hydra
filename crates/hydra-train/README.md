# hydra-train

Package-local map for Hydra training entry binaries. Not operator manual.

Operator docs:
- training modes/YAML and preflight/runtime authority: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md)
- current shipped/staged status: [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md)
- container, GHCR, Kaggle artifact, MJAI audit, coverage commands: [`docker/train/README.md`](../../docker/train/README.md)

## Owns

`hydra-train` is user-facing binary package plus marker library; it stays glue over runtime/exec crates.

Owns:
- training/data CLI entrypoints listed below
- env dispatch / process exit boundary in `src/bin/train.rs`
- tiny binary-specific glue where CLI shape requires it; Python/raw/shard execution delegates to `hydra-train-exec`

Does not own:
- CLI/config/preflight/probe/status contracts: `hydra-train-runtime`
- execution composition, bootstrap, validation, artifacts, GPU/NVTX, data pipeline, BC shard builder: `hydra-train-exec`
- model components: `hydra-model`
- pure training algorithms/loss math: `hydra-train-algo`
- replay load/archive helpers: `hydra-replay-loader`
- BC shard format: `hydra-bc-shards`
- parsed sample cache: `hydra-sample-cache`
- replay sidecar schema: `hydra-replay-sidecar`
- self-play coordination: `hydra-selfplay`
- search-label generation: `hydra-search-labels`
- Riichi rules, scoring, legal actions: `hydra-engine`
- runtime bridge, encoder, simulation, seeding, search/runtime feature plumbing: `hydra-core`

If rule/runtime semantics drift, code plus [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](../../docs/COMPATIBILITY_SURFACE.md) win.

## Current contract snapshot

- live encoder/model contract: `192x34`
- action space: 46 actions
- replay input support: loose MJAI dirs plus `.tar.zst` archives
- shipped baseline includes replay-derived `safety_residual`, stronger public-teacher belief semantics, ExIt carrier across live self-play and replay/sample sidecar-first lanes
- DeltaQ tooling exists but remains promotion-gated, not default-on
- BC shard workflow: build shards, optionally run manifestless markdown preflight on exact tuples for runtime-shape evidence, edit YAML by hand if accepting measured knobs, then train from `bc_shards_manifest_path`.

Read [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md) before treating staged code as live baseline.

## Entry binaries

Cargo target truth comes from Cargo metadata: explicit `[[bin]]` entries plus auto-discovered files under `src/bin/` when `autobins` stays enabled.

| Binary | Purpose |
|---|---|
| `train` | main glue entrypoint: normal train, markdown preflight benchmark, probe-only, DeltaQ promotion |
| `mjai_audit` | replay corpus/archive audit, failure buckets, optional failure inventories |
| `build_bc_shards` | build BC shard datasets/manifests from replay corpora |
| `build_replay_exit_sidecar` | build replay-side ExIt sidecars |
| `build_replay_delta_q_sidecar` | build replay-side DeltaQ sidecars |
| `build_parsed_sample_cache` | build parsed sample cache artifact |
| `extract_timing_metrics` | extract step/training timing metrics |

Main entrypoint: [`src/bin/train.rs`](src/bin/train.rs). Bin-local tests live under `src/bin/train/tests/`; production execution lives in canonical crates above.

## Source map

| Path | Role |
|---|---|
| `src/lib.rs` | marker library; no public training facade |
| `src/bin/train.rs` | train CLI entrypoint/env dispatch/delegation |
| `src/bin/train/tests/` | bin-local tests for train entrypoint integration seams |
| `src/bin/common/replay_sidecar_common.rs` | shared sidecar CLI flag parsing glue |
| `src/bin/*.rs` | user-facing binary entrypoints |

## License

Business Source License 1.1 (BSL). See repo-root [LICENSE](../../LICENSE).

- Free for personal, non-commercial, and academic use
- Commercial mahjong AI services require paid license from Licensor
- Converts to Apache-2.0 on 2031-03-02

Commercial licensing: Sho Kaneko.
