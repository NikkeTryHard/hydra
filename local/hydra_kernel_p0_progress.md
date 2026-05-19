# Hydra raw direct-sink loader progress

## P1.1 Discovery and seam design

### Change
- [ ] Map current `stream_train_epoch` raw path: file enumeration.
- [ ] Map current `stream_train_epoch` raw path: train/validation split.
- [ ] Map current `stream_train_epoch` raw path: shuffle and buffer order.
- [ ] Map current `stream_train_epoch` raw path: `samples_to_skip` accounting.
- [ ] Map current `stream_train_epoch` raw path: sidecar hydration and policy.
- [ ] Identify smallest seam replacing sample Vec materialization only.
- [ ] Define producer item/metadata for direct `BcShardHostBatch` emission.
- [ ] Document ordering/split/skip owner layer.
- [ ] Document direct host-batch emission owner layer.

### Acceptance
- [ ] No design bypasses `is_train_game`.
- [ ] No design bypasses shuffle behavior.
- [ ] No design bypasses `samples_to_skip`.
- [ ] No design bypasses sidecar policy.

## P1.2 Direct-sink parity harness

### Row parity tests
- [ ] Load selected real MJAI games through current `MjaiGame.samples` / `MjaiSample` path.
- [ ] Load same games through `load_game_from_reader_into_sink` + `ReplayHostScratchSink` path.
- [ ] Compare obs rows exactly or with existing tight tolerance.
- [ ] Compare action rows exactly.
- [ ] Compare legal-mask rows exactly.
- [ ] Compare masks/aux labels exactly.
- [ ] Compare ExIt target/mask rows exactly or with tight tolerance.
- [ ] Compare safety target/mask rows exactly or with tight tolerance.
- [ ] Compare DeltaQ target/mask rows exactly or with tight tolerance.
- [ ] Compare sample order exactly.
- [ ] Cover `augment=false`.
- [ ] Cover `augment=true`.
- [ ] Assert 6-permutation row order.
- [ ] Cover optional target presence.
- [ ] Cover optional target absence.
- [ ] Cover optional target presence changing across recycled host scratch/batches.

### Acceptance
- [ ] Test fails if one row is reordered.
- [ ] Test fails if one row is dropped.
- [ ] Test fails if one row is duplicated.
- [ ] Test fails if stale optional target data leaks.
- [ ] Test uses real replay parsing/sample structures, not synthetic-only mocks.

## P1.3 Pipeline integration

### Change
- [ ] Add direct-sink raw producer path behind internal impl seam.
- [ ] Preserve existing public config surface.
- [ ] Preserve existing CLI surface.
- [ ] Direct host batches flow into `train_logical_batch_from_host_batch`.
- [ ] Direct host batches use `HostBatchRows::RawReplay { augment: config.augment }`.
- [ ] Recycle host scratch/batch storage across buffer flushes.
- [ ] Recycle host scratch/batch storage across tail batches.
- [ ] Remove or narrow any superseded production sample-Vec hot path.
- [ ] Keep current sample path only as test/semantic oracle or clearly scoped fallback.

### Acceptance
- [ ] Supported loose raw MJAI production training defaults to direct-sink host-batch emission.
- [ ] No permanent duplicate production hot path remains.
- [ ] P0.1/P0.2 staged H2D remains active for raw CUDA.
- [ ] Unsupported edge cases hard-error or use scoped non-production fallback; no silent fallback.

## P1.4 End-to-end semantic proof

### Train-step parity
- [ ] Direct-sink vs current raw sample path uses fixed samples/model seed.
- [ ] Same optimizer step count.
- [ ] Same effective sample accounting.
- [ ] Same microbatch accounting.
- [ ] Same loss/stat fields within existing project tolerance.
- [ ] Same post-step model/logits within existing project tolerance.
- [ ] Same optimizer/checkpoint ParamId key set.
- [ ] Same optimizer/checkpoint ranks.
- [ ] Same optimizer/checkpoint dtypes.
- [ ] Same optimizer/checkpoint tensor shapes.
- [ ] Same optimizer/checkpoint tensor values where practical.

### Epoch-level proof
- [ ] Same train/validation game decisions.
- [ ] Same buffer yield order.
- [ ] Same `samples_to_skip` behavior.
- [ ] Same tail behavior.
- [ ] Same sidecar optional-target rows/masks.
- [ ] Same sidecar mismatch hard-error behavior.

### Acceptance
- [ ] Tests would fail on changed order.
- [ ] Tests would fail on changed augmentation row order.
- [ ] Tests would fail on changed loss weighting.
- [ ] Tests would fail on changed optimizer step count.
- [ ] Tests would fail on stale optional buffers.
- [ ] Tests would fail on checkpoint shape drift.

## P1.5 Performance proof

### Benchmark setup
- [ ] Benchmark uses dataset `/home/cachybtw/Downloads/dataset_bundle/tenhou-houou-mjai-2025`.
- [ ] Benchmark config has no `bc_shards_manifest_path`.
- [ ] Benchmark uses CUDA BF16 AMP default when `precision_mode` omitted.
- [ ] Benchmark uses `batch_size: 2048`.
- [ ] Benchmark uses `microbatch_size: 256`.
- [ ] Benchmark uses `augment: true`.
- [ ] Benchmark uses `max_train_steps: 30`.
- [ ] Benchmark log is unambiguous one-run evidence.
- [ ] Timing extraction skips first interval.

### Performance acceptance
- [ ] Median post-cold samples/s > `1343.844481539612`, or flat throughput has proven lower loader/collation allocation pressure with no regression.
- [ ] Median input starvation does not exceed P0.1/P0.2 baseline by more than 2 absolute percentage points.
- [ ] H2D materialize remains below 2% elapsed share.
- [ ] No NaN/Inf.
- [ ] No panic.
- [ ] No fallback warning.
- [ ] No stale target warning.
- [ ] No loss spike.

## Completion criteria

### Functional completeness
- [ ] Supported loose raw MJAI production path emits directly into reusable host scratch/batches.
- [ ] Direct path avoids `MjaiGame.samples` / `Vec<MjaiSample>` construction for supported raw training.
- [ ] Direct path is default-on for supported raw training; no user flag required.
- [ ] P0.1/P0.2 recycled host-batch and staged H2D behavior remains active.
- [ ] Unsupported edge cases hard-error or use clearly scoped non-production fallback; no silent semantic fallback.

### Semantic proof
- [ ] Exact/tight row parity against current sample path for selected real MJAI games.
- [ ] Train-step parity against current path: stats, logits, optimizer/checkpoint record.
- [ ] Epoch-level split/shuffle/skip/buffer/tail proof.
- [ ] Sidecar optional-target parity and mismatch hard-error behavior preserved.

### Performance proof
- [ ] Clean one-run benchmark evidence against P0.1/P0.2 baseline.
- [ ] Metrics extracted with preserved command/output.
- [ ] No benchmark artifact ambiguity.

### Required gates
- [ ] Focused direct-sink parity tests.
- [ ] Focused raw host-batch/staging regression tests from P0.1/P0.2.
- [ ] `pixi run cargo check -p hydra-train --bin train --no-default-features --features cuda-graph --quiet`
- [ ] `pixi run lint`
- [ ] Raw benchmark command run.
- [ ] Timing extraction command run.

## Kill criteria
- [ ] Do not claim complete if train/val split changes.
- [ ] Do not claim complete if shuffle order changes.
- [ ] Do not claim complete if replay order changes.
- [ ] Do not claim complete if skip behavior changes.
- [ ] Do not claim complete if augmentation row order changes.
- [ ] Do not claim complete if legal mask/action/target row changes.
- [ ] Do not claim complete if loss weighting changes.
- [ ] Do not claim complete if optimizer step count changes.
- [ ] Do not claim complete if checkpoint shape/name drifts.
- [ ] Do not claim complete if stale optional target data can leak.
- [ ] Do not claim complete if production raw path silently falls back.
- [ ] Do not claim complete if any default build breaks.
