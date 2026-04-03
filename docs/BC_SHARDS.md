# BC Shards

Operator guide for building, inspecting, and consuming precomputed BC shard datasets in Hydra.

This document explains the `build_bc_shards` workflow, the manifest it emits, how optional replay sidecars affect the artifact contract, and how training consumes `bc_shards_manifest_path`. For the top-level training entrypoint, read [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md). For replay-side supervision artifacts, read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).

## What BC shards are for

BC shards are precomputed training artifacts for replay-driven behavioral cloning.

Instead of scanning and loading raw replay files on every run, Hydra can:

1. scan a replay corpus once
2. convert accepted samples into shard files
3. write a manifest describing those shard files
4. point training at the manifest with `bc_shards_manifest_path`

Use BC shards when:

- you want repeated BC runs to reuse a stable dataset layout
- you want train/validation splits fixed by a generated manifest rather than by repeated replay scanning
- you want replay-side supervision from ExIt or DeltaQ sidecars baked into the shard artifact
- you want the training path to consume a known-good prebuilt dataset rather than raw replay discovery every time

## Owning surfaces

Main builder CLI:

- `crates/hydra-train/src/bin/build_bc_shards.rs`

Manifest and shard data contract:

- `crates/hydra-train/src/data/bc_shards.rs`

Training consumption points:

- `crates/hydra-train/src/bin/train/validation.rs`
- `crates/hydra-train/src/bin/train/preflight_runtime.rs`
- broader config wiring through `bc_shards_manifest_path`

## CLI shape

The builder is a standalone binary:

```bash
build_bc_shards \
  --input <dir|archive|replay> \
  --output-dir <dir> \
  [--manifest-name <file>] \
  [--shard-samples <usize>] \
  [--train-fraction <f32>] \
  [--split train|val|both] \
  [--exit-sidecar <path> --exit-source-net-hash <u64> --exit-source-version <u32>] \
  [--delta-q-sidecar <path> --delta-q-source-net-hash <u64> --delta-q-source-version <u32>]
```

### Required flags

| Flag | Meaning |
|---|---|
| `--input` | Replay source root or direct replay/archive input |
| `--output-dir` | Directory where shard files and the manifest will be written |

### Important optional flags

| Flag | Meaning |
|---|---|
| `--manifest-name` | Name of the emitted JSON manifest file. Defaults to `bc_shards_manifest.json`. |
| `--shard-samples` | Target number of samples per shard file. Defaults to `10000`. Must be greater than 0. |
| `--train-fraction` | Deterministic train/validation split fraction for replay identities. Defaults to `0.9`. |
| `--split train|val|both` | Which split(s) to emit. Defaults to `both`. |

## Input types

The builder accepts the same broad replay-source shapes Hydra training already understands:

- a directory containing loose replay files
- a direct loose replay file path
- a direct archive path such as `.tar.zst`

Internally, the builder scans sources with `scan_data_sources_with_progress(...)` before building shards, so the operator experience is “scan first, then materialize.”

## Minimal example

Build both train and validation shards from a replay directory:

```bash
cargo run -p hydra-train --bin build_bc_shards -- \
  --input /data/replays \
  --output-dir /output/bc-shards
```

That emits:

- shard files inside `/output/bc-shards`
- a manifest at `/output/bc-shards/bc_shards_manifest.json`

## Split control

The split policy is deterministic and identity-driven.

Relevant knobs:

- `--train-fraction <f32>`
- `--split train|val|both`

Interpretation:

- `train_fraction` decides which identities map to the train or validation split
- `split` decides whether the builder emits both splits or only one side

Examples:

### Train-only shards

```bash
cargo run -p hydra-train --bin build_bc_shards -- \
  --input /data/replays \
  --output-dir /output/bc-shards \
  --split train
```

### Validation-only shards

```bash
cargo run -p hydra-train --bin build_bc_shards -- \
  --input /data/replays \
  --output-dir /output/bc-shards \
  --split val
```

### Custom train fraction

```bash
cargo run -p hydra-train --bin build_bc_shards -- \
  --input /data/replays \
  --output-dir /output/bc-shards \
  --train-fraction 0.95
```

## Sidecar-backed shard generation

BC shard production can incorporate replay-side supervision lanes at build time.

Supported optional inputs:

- ExIt sidecar index
- DeltaQ sidecar index

Each sidecar requires three pieces of information together:

- sidecar path
- `source_net_hash`
- `source_version`

Hydra rejects partial provenance input. A sidecar is either fully specified or absent.

### Why the provenance requirement matters

The builder is not blindly attaching labels from any JSONL file. It is preserving the same provenance-sensitive contract used by replay-side supervision in the loader path.

That means the shard manifest records sidecar provenance so later consumers can understand what optional supervision was baked into the artifact.

### Example: ExIt-backed shard build

```bash
cargo run -p hydra-train --bin build_bc_shards -- \
  --input /data/replays \
  --output-dir /output/bc-shards \
  --exit-sidecar /labels/exit.jsonl \
  --exit-source-net-hash 123456789 \
  --exit-source-version 7
```

### Example: ExIt + DeltaQ-backed shard build

```bash
cargo run -p hydra-train --bin build_bc_shards -- \
  --input /data/replays \
  --output-dir /output/bc-shards \
  --exit-sidecar /labels/exit.jsonl \
  --exit-source-net-hash 123456789 \
  --exit-source-version 7 \
  --delta-q-sidecar /labels/delta_q.jsonl \
  --delta-q-source-net-hash 123456789 \
  --delta-q-source-version 1
```

For how those sidecar indices are generated and what the provenance fields mean, read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).

## What gets written

The builder prints a summary like:

```text
Wrote <shard_count> shard(s), <sample_count> sample(s) to <output_dir> (manifest: <manifest_path>)
```

It also includes source scan context in the final line:

- `sources=<N>`
- `total_hint=<N>`

Treat the summary as a quick sanity check, not the authoritative record. The manifest is the durable contract.

## Manifest fields that matter most

The emitted `BcShardManifest` includes:

- format/version fields
- action/observation dimensions
- split policy
- input/output paths
- source counts/hints
- optional sidecar provenance
- aggregate totals
- one descriptor list per split

The most operator-relevant fields are:

| Field | Meaning |
|---|---|
| `manifest_version` / `shard_version` | Format versioning for the manifest and shard files |
| `train_fraction` | Split fraction used when building |
| `shard_samples` | Target sample count per shard |
| `input` | Original replay input root/path |
| `output_dir` | Output root holding shard files |
| `source_count` | Number of scanned replay sources |
| `source_total_games_hint` | Scan-derived game-count hint from replay discovery |
| `exit_sidecar` | Optional ExIt sidecar provenance baked into the dataset |
| `delta_q_sidecar` | Optional DeltaQ sidecar provenance baked into the dataset |
| `totals.sample_count` | Total samples successfully written |
| `totals.skipped_games` | Replay games skipped during shard production |
| `totals.empty_games` | Games that loaded but produced no training samples |
| `splits` | Per-split shard descriptors and sample counts |

### Per-split information

Each split manifest carries:

- split name (`train` or `validation`)
- shard count
- sample count
- feature flags
- record size
- list of shard descriptors

Each shard descriptor carries:

- shard index
- file name
- sample count
- first sample index
- byte length
- feature flags
- record size

This is enough to answer the two most common operator questions:

- Did I actually build the split I intended?
- How many samples and shard files did that split produce?

## How training consumes BC shards

Hydra training consumes the artifact through config:

```yaml
bc_shards_manifest_path: /output/bc-shards/bc_shards_manifest.json
```

When this field is set, Hydra uses prebuilt shard metadata instead of scanning raw replay files for normal BC data ingestion.

Use this when:

- the replay corpus has already been validated
- you want reproducible repeated BC runs on the same dataset layout
- startup/scanning cost on raw replay inputs is no longer worth paying each run

The current training docs already cover that field at a high level; this document is the missing production-side runbook explaining how to create the artifact it points to.

## Recommended operator workflow

1. Validate a new replay corpus first with [`docs/MJAI_AUDIT_AND_FAILURE_TRIAGE.md`](MJAI_AUDIT_AND_FAILURE_TRIAGE.md).
2. Decide whether you need raw replay loading every run or a prebuilt shard dataset.
3. Build shards with `build_bc_shards`.
4. Inspect the manifest for split counts, sample totals, and sidecar provenance.
5. Point training at `bc_shards_manifest_path`.
6. Rebuild shards if replay inputs or sidecar provenance change materially.

## When to rebuild shards

Rebuild when any of these change in a way you want reflected in the dataset:

- replay corpus contents
- split policy (`train_fraction`, `split`)
- shard sizing (`shard_samples`)
- ExIt sidecar provenance or path
- DeltaQ sidecar provenance or path

Do not assume old shards automatically reflect a new sidecar index or a new replay corpus snapshot.

## Common mistakes

- Supplying only part of the sidecar provenance tuple; Hydra requires all sidecar provenance fields together.
- Treating the manifest path as a magical cache key instead of a concrete dataset artifact.
- Forgetting that shard production bakes in whether ExIt or DeltaQ supervision was available at build time.
- Skipping corpus validation and then debugging shard failures downstream instead of fixing the replay input first.

## Relationship to other docs

- `docs/TRAINING_WORKFLOWS.md` explains when to set `bc_shards_manifest_path`.
- `docs/REPLAY_SIDECARS.md` explains how sidecar JSONL artifacts are generated and what their provenance means.
- `docs/MJAI_AUDIT_AND_FAILURE_TRIAGE.md` explains how to validate and debug the replay corpus before shard production.

## Where to read next

- Need the main training flow? Read [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md).
- Need replay sidecar generation? Read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).
- Need replay corpus validation before shard production? Read [`docs/MJAI_AUDIT_AND_FAILURE_TRIAGE.md`](MJAI_AUDIT_AND_FAILURE_TRIAGE.md).
