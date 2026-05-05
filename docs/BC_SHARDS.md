# BC Shards

Operator guide: build, inspect, consume precomputed BC shard datasets in Hydra.

Doc covers `build_bc_shards` flow, emitted manifest, optional replay sidecar effect on artifact contract, how training consumes `bc_shards_manifest_path`. Top-level training entrypoint: [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md). Replay-side supervision artifacts: [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).

## What BC shards are for

BC shards = precomputed training artifacts for replay-driven behavioral cloning.

Instead of rescanning/loading raw replay files every run, Hydra can:

1. scan replay corpus once
2. convert accepted samples into shard files
3. write manifest describing shard files
4. point training at manifest with `bc_shards_manifest_path`

Use BC shards when:

- want repeated BC runs reuse stable dataset layout
- want train/validation splits fixed by generated manifest, not repeated replay scanning
- want replay-side supervision from ExIt or DeltaQ sidecars baked into shard artifact
- want training path consume known-good prebuilt dataset, not raw replay discovery every time

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

Builder = standalone binary:

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
| `--manifest-name` | Name of emitted JSON manifest file. Default `bc_shards_manifest.json`. |
| `--shard-samples` | Target samples per shard file. Default `10000`. Must be > 0. |
| `--train-fraction` | Deterministic train/validation split fraction for replay identities. Default `0.9`. |
| `--split train|val|both` | Which split(s) to emit. Default `both`. |

## Input types

Builder accepts same broad replay-source shapes Hydra training already understands:

- directory containing loose replay files
- direct loose replay file path
- direct archive path such as `.tar.zst`

Internally, builder scans sources with `scan_data_sources_with_progress(...)` before shard build, so operator flow = scan first, then materialize.

## Minimal example

Build both train and validation shards from replay directory:

```bash
cargo run -p hydra-train --bin build_bc_shards -- \
  --input /data/replays \
  --output-dir /output/bc-shards
```

Emits:

- shard files inside `/output/bc-shards`
- manifest at `/output/bc-shards/bc_shards_manifest.json`

## Split control

Split policy = deterministic, identity-driven.

Relevant knobs:

- `--train-fraction <f32>`
- `--split train|val|both`

Interpretation:

- `train_fraction` decides which identities map to train or validation
- `split` decides whether builder emits both splits or only one side

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

BC shard production can bake replay-side supervision lanes at build time.

Supported optional inputs:

- ExIt sidecar index
- DeltaQ sidecar index

Each sidecar needs 3 pieces together:

- sidecar path
- `source_net_hash`
- `source_version`

Hydra rejects partial provenance input. Sidecar = fully specified or absent.

### Why the provenance requirement matters

Builder not blindly attaching labels from random JSONL. It preserves same provenance-sensitive contract used by replay-side supervision in loader path.

So shard manifest records sidecar provenance, and later consumers can see what optional supervision got baked into artifact.

### Identity rules are preserved when shards are built

BC shard production does not flatten replay identity semantics when joining sidecars before writing shard rows.

- Loose replay inputs use replay file name as sidecar identity key.
- Archive-backed replay inputs use full archive-entry identity string.

In practice: sidecar record keyed to `game.json` can hydrate loose replay `game.json`, but not archive entry identified as `replays.tar.zst/path/inside/game.json`.

Deliberate. Builder preserves same sidecar join contract Hydra uses in normal replay loader, so shard-backed BC runs and loose-replay BC runs do not silently disagree on which labels belong to which replay decisions.

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

For sidecar index generation and provenance field meaning, read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).

## What gets written

Builder prints summary like:

```text
Wrote <shard_count> shard(s), <sample_count> sample(s) to <output_dir> (manifest: <manifest_path>)
```

Final line also includes source scan context:

- `sources=<N>`
- `total_hint=<N>`

Use summary as quick sanity check, not authoritative record. Manifest = durable contract.

## Manifest fields that matter most

Emitted `BcShardManifest` includes:

- format/version fields
- action/observation dimensions
- split policy
- input/output paths
- source counts/hints
- optional sidecar provenance
- aggregate totals
- one descriptor list per split

Most operator-relevant fields:

| Field | Meaning |
|---|---|
| `manifest_version` / `shard_version` | Format versioning for manifest and shard files |
| `train_fraction` | Split fraction used when building |
| `shard_samples` | Target sample count per shard |
| `input` | Original replay input root/path |
| `output_dir` | Output root holding shard files |
| `source_count` | Number of scanned replay sources |
| `source_total_games_hint` | Scan-derived game-count hint from replay discovery |
| `exit_sidecar` | Optional ExIt sidecar provenance baked into dataset |
| `delta_q_sidecar` | Optional DeltaQ sidecar provenance baked into dataset |
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

Enough to answer 2 common operator questions:

- Did I build split I intended?
- How many samples and shard files did split produce?

## How training consumes BC shards

Hydra training consumes artifact through config:

```yaml
bc_shards_manifest_path: /output/bc-shards/bc_shards_manifest.json
```

When field set, Hydra uses prebuilt shard metadata instead of scanning raw replay files for normal BC data ingestion.

Use when:

- replay corpus already validated
- want reproducible repeated BC runs on same dataset layout
- startup/scanning cost on raw replay inputs no longer worth paying each run

Training does not only switch train loader here. Validation also runs against shard readers, meaning:

- validation no longer streams loose replay files from `data_dir`
- validation helper that normally pre-materializes bounded in-memory validation cache is bypassed
- validation sample limits still apply, but over shard rows rather than replay-stream microbatches

Important when comparing shard-backed vs loose-replay runs. Shard manifest changes full BC data path, including validation transport and validation memory/runtime shape.

Current training docs already cover field at high level; this doc is missing production-side runbook for creating artifact that field points to.

## Recommended operator workflow

1. Validate new replay corpus first with [`docs/MJAI_AUDIT_AND_FAILURE_TRIAGE.md`](MJAI_AUDIT_AND_FAILURE_TRIAGE.md).
2. Decide whether need raw replay loading every run or prebuilt shard dataset.
3. Build shards with `build_bc_shards`.
4. Inspect manifest for split counts, sample totals, sidecar provenance.
5. Point training at `bc_shards_manifest_path`.
6. Rebuild shards if replay inputs or sidecar provenance change materially.

## When to rebuild shards

Rebuild when any of these change and you want dataset reflect it:

- replay corpus contents
- split policy (`train_fraction`, `split`)
- shard sizing (`shard_samples`)
- ExIt sidecar provenance or path
- DeltaQ sidecar provenance or path
- encoder geometry or shard layout compatibility (`obs_size`, base record size, feature-flag-driven row layout)

Do not assume old shards auto-reflect new sidecar index or new replay corpus snapshot.

Hydra also rejects shard artifacts at load time when current binary and manifest/header contract no longer agree. Important operator-visible cases:

- manifest `obs_size` no longer matches current encoder geometry
- manifest `base_record_size` no longer matches current binary
- requested `train` or `validation` split missing from manifest
- individual shard header disagrees with manifest about split, feature flags, record size, or shard version

Treat failures as rebuild signal, not something to patch with manual manifest edits.

One more artifact detail matters operationally: shard files resolve relative to manifest file parent directory, not current shell working directory. Move manifest without shard files -> artifact breaks.

## Common mistakes

- Supplying only part of sidecar provenance tuple; Hydra requires all sidecar provenance fields together.
- Treating manifest path as magical cache key instead of concrete dataset artifact.
- Forgetting shard production bakes in whether ExIt or DeltaQ supervision was available at build time.
- Assuming manifest with only one split can still satisfy both train and validation consumers.
- Moving manifest file independently from shard files it references.
- Skipping corpus validation, then debugging shard failures downstream instead of fixing replay input first.

## Relationship to other docs

- `docs/TRAINING_WORKFLOWS.md` explains when to set `bc_shards_manifest_path`.
- `docs/REPLAY_SIDECARS.md` explains how sidecar JSONL artifacts are generated and what their provenance means.
- `docs/MJAI_AUDIT_AND_FAILURE_TRIAGE.md` explains how to validate and debug replay corpus before shard production.

## Where to read next

- Need main training flow? Read [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md).
- Need replay sidecar generation? Read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).
- Need replay corpus validation before shard production? Read [`docs/MJAI_AUDIT_AND_FAILURE_TRIAGE.md`](MJAI_AUDIT_AND_FAILURE_TRIAGE.md).