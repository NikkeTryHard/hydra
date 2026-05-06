# MJAI Audit and Failure Triage

Operator guide: validate replay corpora before training. Also triage broken MJAI inputs when Hydra rejects file or archive entry.

This doc covers three concrete binaries from `hydra-train`:

- `mjai_audit`
- `mjai_first_failure`
- `mjai_debug_failure`

Use before BC training, before BC shard build, before replay-side supervision artifact generation from new corpus. For main training entrypoint, read [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md). For replay sidecar generation, read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).

## When to use this workflow

Run audit/triage loop when:

- new replay corpus unvalidated
- training skips more files than expected
- replay-sidecar generation or shard build fails on subset
- need isolate first broken entry inside `.tar.zst` archive
- need human-readable reason single replay fails loader validation

Workflow layered:

1. `mjai_audit` tells whether corpus healthy enough.
2. `mjai_first_failure` isolates first bad archive entry.
3. `mjai_debug_failure` explains single failing replay in more detail.

## 1) `mjai_audit`: corpus-wide validation

Owning binary:

- `crates/hydra-train/src/bin/mjai_audit.rs`

CLI shape:

```bash
mjai_audit <data-dir> [--threads N] [--failure-examples N] [--failure-inventory-dir DIR]
```

### What counts as input

`mjai_audit` accepts:

- directory with loose replay files
- direct loose replay file path
- direct `.tar.zst` archive path

It uses same replay loader family Hydra training uses:

- loose files validated with `load_game_from_path(...)`
- archive entries validated with `load_game_from_stream(...)`

So audit not loose pre-check. It hits same replay acceptance surface training later needs.

### Important flags

| Flag | Meaning |
|---|---|
| `--threads N` | Rayon worker count for audit work. Must be greater than 0. |
| `--failure-examples N` | Max example failure payloads printed at end. Default 20. |
| `--failure-inventory-dir DIR` | If set, writes per-source JSONL inventories with every failure for that source. |

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

At startup, binary prints target and worker-thread count.

Summary lines that matter:

```text
Audit complete: loaded=<N> skipped=<N> samples=<N> total=<N>
Speed: elapsed=<secs> files_per_sec=<rate> samples_per_sec=<rate>
```

Interpretation:

- `loaded`: replay sources or archive entries Hydra parsed into games
- `skipped`: replay sources or archive entries Hydra rejected from loader or archive errors
- `samples`: total training samples recovered from successfully loaded games
- `total`: `loaded + skipped`

### Failure bucket summary

When failures exist, `mjai_audit` prints ranked summary headed by:

```text
Top failure buckets:
```

Each bucket = first line of failure message, collapsed into count. Fast way answer:

- corpus mostly fine with one repeated schema issue?
- one broken archive, or many unrelated failure classes?
- replay corruption, parse drift, or unexpected runtime semantics?

### Failure examples

If `--failure-examples` non-zero, audit prints bounded raw-failure sample after bucket summary.

Use for quick triage. Need exhaustive failure detail -> use `--failure-inventory-dir`.

## Failure inventories

If `--failure-inventory-dir DIR` set, Hydra creates one JSONL inventory per source.

Each line contains:

- `source`
- `identity`
- `error`

Useful when:

- need hand failures to cleanup tool
- want durable machine-readable record, not console output
- large archive makes many bad entries, terminal scrollback not enough

### Identity format

Hydra records identities by source type:

- loose file: file path itself
- archive entry by path: `archive.tar.zst/path/inside/archive.json`
- archive entry when path inspection fails: `archive.tar.zst#entry[<index>]`

So failures map back to real container and entry without reverse-engineering audit logs.

## 2) `mjai_first_failure`: find the first bad entry in an archive

Owning binary:

- `crates/hydra-train/src/bin/mjai_first_failure.rs`

CLI shape:

```bash
mjai_first_failure <archive.tar.zst>
```

This tool narrow by design. It scans one archive in order, stops on first MJAI entry that fails replay load.

On failure, it prints:

- archive path
- entry path
- how many MJAI entries were seen before failure
- raw error text

Example:

```bash
cargo run -p hydra-train --bin mjai_first_failure -- /data/replays.tar.zst
```

Use when:

- `mjai_audit` says archive bad, need first reproducing entry fast
- want deterministic first bad record for debugging or CI reproduction

If archive healthy, Hydra prints:

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

This tool runs focused replay-failure explainer on one replay file. Prints detailed report for first failure found.

Example:

```bash
cargo run -p hydra-train --bin mjai_debug_failure -- /tmp/failing_replay.json
```

Use when:

- already isolated single bad replay from `mjai_audit` or `mjai_first_failure`
- need deeper explanation than one-line failure bucket or raw archive failure message

If replay does not fail, tool prints:

```text
No failure found.
```

## Recommended operator workflow

### For a new corpus

1. Run `mjai_audit` on whole corpus.
2. If no failures, proceed to training or shard build.
3. If failures exist, inspect failure buckets and few examples.
4. If failures widespread, persist failure inventories and hand to cleanup pipeline.

### For a broken archive

1. Run `mjai_audit` to confirm archive problematic.
2. Run `mjai_first_failure <archive.tar.zst>` to isolate first bad entry.
3. Extract or locate that replay and run `mjai_debug_failure` on single file.

### For CI or repeated data-quality checks

Use `mjai_audit` as top-level gate because it measures:

- how many sources load at all
- how many samples survive loader validation
- dominant failure classes

Better signal than ad hoc file-count checks.

## Reading results correctly

- few skipped files in large corpus may be tolerable for experimentation, but still real loader mismatches
- large `samples` count + high `skipped` count usually means corpus usable but dirty
- small `samples` count + many failures means training stats and validation splits may mislead; fix corpus first
- archive failures easier if first isolate one bad entry, not treat archive as opaque

## Relationship to training and shard building

Run this workflow before:

- replay-driven BC training
- BC shard generation
- replay-sidecar generation for ExIt or DeltaQ labels

Order matters because all those workflows assume replay loader can consume underlying corpus consistently.

## Where to read next

- Need main training entrypoint? Read [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md).
- Need replay-side supervision generation? Read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).
- Need current shipped/staged repo snapshot? Read [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md).