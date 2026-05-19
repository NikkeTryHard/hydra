# Hydra raw direct-sink loader plan

## Goal
Make loose raw MJAI training emit samples directly into reusable host-batch scratch, avoiding `MjaiGame.samples` / `Vec<MjaiSample>` construction on raw folder hot path, without changing training semantics.

## Baseline from completed P0.1/P0.2
- Raw folder: `/home/cachybtw/Downloads/dataset_bundle/tenhou-houou-mjai-2025`
- Clean config: `/home/cachybtw/tmp/hydra-raw-folder-bottleneck-p0-r1-clean.yaml`
- Clean log: `/home/cachybtw/tmp/hydra-raw-folder-bottleneck-p0-r1-clean/bc/step_log.jsonl`
- Median samples/s: `1343.844481539612`
- H2D materialize elapsed share: `0.43905725991249855%`
- Input starvation: `3.7970595585124944%`
- Current raw path now uses recycled host batches and staged H2D. Remaining target is loader/sample allocation and replay-to-host-batch overhead.

## Non-negotiables
- Preserve train/val split and `is_train_game` behavior.
- Preserve epoch shuffle seed behavior and buffer yield order.
- Preserve `samples_to_skip` continuation/resume behavior.
- Preserve replay order inside each game.
- Preserve augmentation row order exactly: identity then same 6 suit permutations as current raw sample path.
- Preserve sidecar policy and optional ExIt/safety/DeltaQ target semantics.
- Preserve loss weighting, optimizer step timing, precision mode, runtime labels, and checkpoint shape.
- CUDA graph replay remains `experimental_probe_only`.
- No mock-only proof. Row parity must use real sample structures and selected real MJAI games.

## Target files
- `crates/hydra-replay-loader/src/mjai_loader.rs`
- `crates/hydra-train-exec/src/data/sample.rs`
- `crates/hydra-train-exec/src/data_pipeline.rs`
- `crates/hydra-train-exec/src/epoch_runner.rs`
- Existing relevant tests under `crates/hydra-train-exec/src/**/tests.rs`

## Current useful building blocks
- `load_game_from_reader_into_sink` emits replay records to caller sink.
- `ReplayHostScratchSink` can write replay records into `BcShardHostScratch`.
- `collate_samples_into_recycled_host_batch` and raw P0.1/P0.2 training path already consume `BcShardHostBatch` with explicit row semantics and staged H2D.
- Keep P0.1/P0.2 path as semantic oracle until direct sink proves exact parity.

## P1.1 Discovery and seam design

### Change
1. Map current `stream_train_epoch` raw path: file enumeration, split, shuffle, buffering, skip accounting, sidecar hydration, and sample emission.
2. Identify smallest seam where direct-sink can replace only sample Vec materialization while preserving all upstream ordering decisions.
3. Define producer item that can carry either:
   - current `Vec<MjaiSample>` path, or
   - direct `BcShardHostBatch` plus equivalent accounting metadata.
4. Keep raw P0.1/P0.2 train loop semantics unchanged until direct host batch reaches same boundary.

### Acceptance
- short code comment or local design note documents which layer owns ordering/split/skip accounting and which layer owns direct host-batch emission.
- No code path bypasses `is_train_game`, shuffle, `samples_to_skip`, or sidecar policy.

## P1.2 Direct-sink parity harness

### Change
1. Add tests that load selected real MJAI games through both paths:
   - existing `MjaiGame.samples` / `MjaiSample` path,
   - direct `load_game_from_reader_into_sink` + `ReplayHostScratchSink` path.
2. Compare rows exactly or with existing tight project tolerance where float derivation requires it:
   - obs tensor/flat rows,
   - action,
   - legal mask,
   - masks/aux labels,
   - ExIt target/mask,
   - safety target/mask,
   - DeltaQ target/mask,
   - sample order.
3. Cover `augment=false` and `augment=true` and assert 6-permutation row order.
4. Cover optional target presence and absence, including transition across recycled host scratch/batches.

### Acceptance
- Test fails if one row is reordered, dropped, duplicated, or carries stale optional target data.
- Test uses real replay parsing/sample structures, not synthetic-only mocks.

## P1.3 Pipeline integration

### Change
1. Add direct-sink raw producer path behind internal impl seam, not user-facing feature flag.
2. Preserve existing public config and CLI surface.
3. Keep fallback to current sample path only for unsupported sidecar/edge cases during impl; by completion, supported production raw path must default to direct sink.
4. Ensure direct host batches flow into `train_logical_batch_from_host_batch` with `HostBatchRows::RawReplay { augment: config.augment }`.
5. Recycle host scratch/batch storage across buffer flushes and tail batches.

### Acceptance
- Supported loose raw MJAI production training defaults to direct-sink host-batch emission.
- No permanent duplicate production hot path remains unless one path is test-only/semantic oracle.
- P0.1/P0.2 staged H2D remains active for raw CUDA.

## P1.4 End-to-end semantic proof

### Tests
- Direct-sink vs current raw sample path train step parity on fixed samples/model seed:
  - same optimizer step count,
  - same effective sample/microbatch accounting,
  - same loss/stat fields within existing project tolerance,
  - same post-step model/logits within existing project tolerance,
  - same optimizer/checkpoint record key set, ranks, dtypes, shapes, and tensor values where practical.
- Epoch-level test for split/shuffle/skip:
  - same train/validation game decisions,
  - same buffer yield order,
  - same `samples_to_skip` behavior,
  - same tail behavior.
- Sidecar test:
  - same optional target rows/masks and same hard-error behavior on mismatch.

### Acceptance
- Tests would fail on changed order, changed augmentation order, changed loss weighting, changed optimizer step count, stale optional buffers, or checkpoint shape drift.

## P1.5 Performance proof

### Benchmark
Use same raw-folder benchmark family as P0.1/P0.2:
```bash
HYDRA_NVTX=1 HYDRA_BENCHMARK_QUIET=1 pixi run train-cuda-shards -- /home/cachybtw/tmp/hydra-raw-folder-bottleneck-direct-sink.yaml
pixi run cargo run -p hydra-train --bin extract_timing_metrics -- \
  --step-log /home/cachybtw/tmp/hydra-raw-folder-bottleneck-direct-sink/bc/step_log.jsonl \
  --run-id raw-folder-direct-sink \
  --skip-initial-rows 1 \
  --format json
```

### Acceptance
- Benchmark log is unambiguous one-run evidence; no appended duplicate run.
- Median post-cold samples/s is strictly greater than `1343.844481539612`, or if flat, direct-sink must show lower loader/collation allocation pressure with no throughput regression.
- Input starvation does not exceed P0.1/P0.2 baseline by more than 2 absolute percentage points.
- H2D materialize remains low; no regression above 2% elapsed share.
- No NaN/Inf, panic, fallback warning, stale target warning, or loss spike.

## Completion criteria

P1 direct-sink is complete only when all items below are true.

### Functional completeness
- Supported loose raw MJAI production path emits directly into reusable host scratch/batches.
- Direct path avoids `MjaiGame.samples` / `Vec<MjaiSample>` construction for supported raw training.
- Direct path is default-on for supported raw training; no user flag required.
- P0.1/P0.2 recycled host-batch and staged H2D behavior remains active.
- Unsupported edge cases hard-error or use clearly scoped non-production fallback; no silent semantic fallback.

### Semantic proof
- Exact/tight row parity against current sample path for selected real MJAI games.
- Train-step parity against current path: stats, logits, optimizer/checkpoint record.
- Epoch-level split/shuffle/skip/buffer/tail proof.
- Sidecar optional-target parity and mismatch hard-error behavior preserved.

### Performance proof
- Clean one-run benchmark evidence against P0.1/P0.2 baseline.
- Metrics extracted with preserved command/output.
- No benchmark artifact ambiguity.

### Required gates
- Focused direct-sink parity tests.
- Focused raw host-batch/staging regression tests from P0.1/P0.2.
- `pixi run cargo check -p hydra-train --bin train --no-default-features --features cuda-graph --quiet`
- `pixi run lint`
- Raw benchmark and timing extraction.

## Kill criteria
- Any changed train/val split, shuffle, replay order, or skip behavior.
- Any changed augmentation row order.
- Any changed legal mask/action/target row.
- Any changed loss weighting or optimizer step count.
- Any checkpoint shape/name drift.
- Any stale optional target leak.
- Any hidden fallback in production raw path.
- Any default build break.
