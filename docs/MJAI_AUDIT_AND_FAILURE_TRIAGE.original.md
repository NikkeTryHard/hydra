# MJAI Audit and Failure Triage

Operator guide for validating replay corpora before training and for drilling into broken MJAI inputs when Hydra rejects a file or archive entry.

This document covers three concrete binaries from `hydra-train`:

- `mjai_audit`
- `mjai_first_failure`
- `mjai_debug_failure`

Use this guide before BC training, before building BC shards, and before generating replay-side supervision artifacts from a new corpus. For the main training entrypoint, read [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md). For replay sidecar generation, read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).

## When to use this workflow

Run the audit/triage loop when:

- a new replay corpus has not been validated yet
- training is skipping more files than expected
- replay-sidecar generation or shard building appears to fail on a subset of files
- you need to isolate the first broken entry inside a `.tar.zst` archive
- you need a human-readable explanation of why a single replay fails loader validation

The workflow is intentionally layered:

1. `mjai_audit` tells you whether the corpus is healthy enough to use.
2. `mjai_first_failure` isolates the first bad archive entry when the source is an archive.
3. `mjai_debug_failure` explains a single failing replay in more detail.

## 1) `mjai_audit`: corpus-wide validation

Owning binary:

- `crates/hydra-train/src/bin/mjai_audit.rs`

CLI shape:

```bash
mjai_audit <data-dir> [--threads N] [--failure-examples N] [--failure-inventory-dir DIR]
```

### What counts as input

`mjai_audit` accepts:

- a directory containing loose replay files
- a direct loose replay file path
- a direct `.tar.zst` archive path

It uses the same replay loader family Hydra training relies on:

- loose files are validated with `load_game_from_path(...)`
- archive entries are validated with `load_game_from_stream(...)`

That means the audit is not a loose pre-check. It is exercising the same replay acceptance surface the training pipeline will later depend on.

### Important flags

| Flag | Meaning |
|---|---|
| `--threads N` | Number of rayon worker threads used for audit work. Must be greater than 0. |
| `--failure-examples N` | Maximum number of example failure payloads to print at the end. Defaults to 20. |
| `--failure-inventory-dir DIR` | If set, writes per-source JSONL inventories containing every failure encountered for that source. |

### Example: audit a loose replay directory

```bash
cargo run -p hydra-train --bin mjai_audit -- /data/mjai --threads 16 --failure-examples 10
```

### Example: audit an archive and persist machine-readable failure inventories

```bash
cargo run -p hydra-train --bin mjai_audit -- /data/replays.tar.zst \
  --threads 8 \
  --failure-examples 5 \
  --failure-inventory-dir /tmp/hydra-audit-failures
```

## Understanding `mjai_audit` output

At startup, the binary prints what it is auditing and how many worker threads it will use.

The summary lines you should care about are:

```text
Audit complete: loaded=<N> skipped=<N> samples=<N> total=<N>
Speed: elapsed=<secs> files_per_sec=<rate> samples_per_sec=<rate>
```

Interpretation:

- `loaded`: replay sources or archive entries Hydra successfully parsed into games
- `skipped`: replay sources or archive entries Hydra rejected due to loader or archive errors
- `samples`: total training samples recovered from successfully loaded games
- `total`: `loaded + skipped`

### Failure bucket summary

When failures exist, `mjai_audit` prints a ranked summary headed by:

```text
Top failure buckets:
```

Each bucket is the first line of a failure message, collapsed into a count. This is the fastest way to answer:

- Is the corpus mostly fine with one repeated schema issue?
- Is there one broken archive, or many unrelated failure classes?
- Are we dealing with replay corruption, parse drift, or unexpected runtime semantics?

### Failure examples

If `--failure-examples` is non-zero, the audit prints a bounded sample of raw failures after the bucket summary.

Use that output for quick triage. If you need exhaustive failure details, use `--failure-inventory-dir`.

## Failure inventories

If `--failure-inventory-dir DIR` is set, Hydra creates one JSONL inventory per source.

Each line contains:

- `source`
- `identity`
- `error`

This is useful when:

- you need to hand failures to another cleanup tool
- you want a durable machine-readable record rather than console output
- a large archive produces many bad entries and you do not want to lose them in terminal scrollback

### Identity format

Hydra records identities differently depending on source type:

- loose file: the file path itself
- archive entry by path: `archive.tar.zst/path/inside/archive.json`
- archive entry when path inspection fails: `archive.tar.zst#entry[<index>]`

That makes it possible to map failures back to the real container and entry without reverse-engineering the audit logs.

## 2) `mjai_first_failure`: find the first bad entry in an archive

Owning binary:

- `crates/hydra-train/src/bin/mjai_first_failure.rs`

CLI shape:

```bash
mjai_first_failure <archive.tar.zst>
```

This tool is intentionally narrow. It scans a single archive in order and stops on the first MJAI entry that fails replay loading.

On failure, it prints:

- archive path
- entry path
- how many MJAI entries were seen before the failure
- raw error text

Example:

```bash
cargo run -p hydra-train --bin mjai_first_failure -- /data/replays.tar.zst
```

Use it when:

- `mjai_audit` tells you an archive is bad, but you need the first reproducing entry fast
- you want a deterministic “first bad record” for debugging or CI reproduction

If the archive is healthy, Hydra prints:

```text
No failures found after scanning <N> MJAI entries.
```

## 3) `mjai_debug_failure`: explain one failing replay

Owning binary:

- `crates/hydra-train/src/bin/mjai_debug_failure.rs`

CLI shape:

```bash
mjai_debug_failure <replay.json>
```

This tool runs the focused replay-failure explainer on one replay file and prints a detailed report for the first failure found.

Example:

```bash
cargo run -p hydra-train --bin mjai_debug_failure -- /tmp/failing_replay.json
```

Use it when:

- you already isolated a single bad replay from `mjai_audit` or `mjai_first_failure`
- you need a deeper explanation than the one-line failure bucket or raw archive failure message

If the replay does not fail, the tool prints:

```text
No failure found.
```

## Recommended operator workflow

### For a new corpus

1. Run `mjai_audit` on the whole corpus.
2. If there are no failures, proceed to training or shard building.
3. If failures exist, inspect the failure buckets and a few examples.
4. If failures are widespread, persist failure inventories and hand them to a cleanup pipeline.

### For a broken archive

1. Run `mjai_audit` to confirm the archive is problematic.
2. Run `mjai_first_failure <archive.tar.zst>` to isolate the first bad entry.
3. Extract or locate that replay and run `mjai_debug_failure` on the single file.

### For CI or repeated data-quality checks

Use `mjai_audit` as the top-level gate because it measures:

- how many sources load at all
- how many samples survive loader validation
- what the dominant failure classes are

That is a much better signal than ad hoc file-count checks.

## Reading results correctly

- A few skipped files in a large corpus may be tolerable for experimentation, but they are still real loader mismatches.
- A large `samples` count with a high `skipped` count usually means the corpus is usable but dirty.
- A small `samples` count with many failures means training statistics and validation splits may be misleading; fix the corpus first.
- Archive failures are often easier to handle by first isolating one bad entry instead of treating the archive as opaque.

## Relationship to training and shard building

Run this workflow before:

- replay-driven BC training
- BC shard generation
- replay-sidecar generation for ExIt or DeltaQ labels

That order matters because all of those workflows assume the replay loader can consume the underlying corpus consistently.

## Where to read next

- Need the main training entrypoint? Read [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md).
- Need replay-side supervision generation? Read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).
- Need the current shipped/staged repo snapshot? Read [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md).
