# Hydra Training Runbook

Compact op entrypoint. Owns: train CLI modes, YAML shape, preflight authority, BC shards, replay sidecars, DeltaQ promotion gates, precision/CUDA shard notes.

Truth owners:
- CLI/router: `crates/hydra-train/src/bin/train.rs`
- YAML contract: `crates/hydra-train-runtime/src/config.rs`
- shard contract: `crates/hydra-bc-shards/src/lib.rs`
- shard builder CLI: `crates/hydra-train/src/bin/build_bc_shards.rs`; impl: `crates/hydra-train-exec/src/bc_shard_builder.rs`
- sidecar shared flags: `crates/hydra-train/src/bin/common/replay_sidecar_common.rs`
- current shipped/staged status: `docs/CURRENT_STATUS.md`
- replay audit before train/shard/sidecar: `docker/train/README.md`
- compat/runtime surface: `docs/COMPATIBILITY_SURFACE.md`

## Train CLI modes

Binary shape:

```bash
cargo run -p hydra-train --bin train -- <config.yaml> [flags]
```

Modes:

| Mode | Invoke | Use |
|---|---|---|
| normal BC shard train | `train --bc-shards-manifest <manifest> --output-dir <dir> --device cuda:0 ...` or `train config.yaml` with `bc_shards_manifest_path` | default Python/PyTorch BC learner via Rust launcher |
| normal Rust/Burn train | `train config.yaml` with `bc_backend: rust_burn` or CLI `--bc-backend rust-burn` | legacy/reference BC path; required for ExIt/DeltaQ/belief/mixture/opponent-hand-type until Python supports them |
| preflight | `train --preflight --pf-candidate-tuples ... --pf-output md` | benchmark exact runtime tuples against synthetic/in-memory work; no config, manifest, dataset, cache, choice, or YAML write |
| probe-only | `train config.yaml --probe-kind <train|validation|rl_games|rl_microbatch> --probe-candidate-microbatch <N> ...` | bounded candidate check, no full train |

Internal child probe path exists. Not op entrypoint.

Choose:
- normal: YAML already contains intended runtime knobs.
- preflight: new GPU/machine, precision change, workload/shard change, or explicit shard-throughput comparison.
- probe-only: test one candidate train/validation microbatch, RL microbatch, or RL game-count setting.
- DeltaQ promotion: candidate checkpoint exists; DeltaQ lane needs promote/no-promote evidence. Not normal train.

## YAML contract

`TrainConfig` top-level fields op usually touches:

| Field | Meaning |
|---|---|
| `data_dir` | replay root, loose file, or archive path |
| `output_dir` | checkpoints, logs, reports, caches |
| `num_epochs` | BC epoch count |
| `batch_size` | logical batch before microbatch/accum |
| `microbatch_size` | explicit train microbatch override |
| `validation_microbatch_size` | explicit validation microbatch override |
| `train_fraction` | deterministic train/validation split |
| `source_filters` | replay include/exclude identity filters |
| `augment` | suit permutation augmentation |
| `resume_checkpoint` | checkpoint base for resume |
| `precision_mode` | optional; omitted BC CUDA defaults to `bf16_autocast` / effective `bf16_amp`; explicit `fp32` overrides; CPU omission FP32; RL/DeltaQ BF16 hard-error |
| `device` | backend device label |
| `bc` | BC optimizer knobs |
| `rl` | RL/self-play enable + phase knobs |
| `advanced_loss` | optional ExIt/safety/DeltaQ supervised weights |
| `exit_sidecar_path` | optional ExIt sidecar index |
| `delta_q_sidecar_path` | optional DeltaQ sidecar index |
| `bc_shards_manifest_path` | prebuilt BC shard manifest input |
| `bc_backend` | BC shard train backend; default `python`; set `rust_burn` only for legacy/debug or advanced labels Python lacks |
| `shard_prefetch_depth` | shard host-batch queue depth; default `2`, valid `1..64` |
| `python_residual_profile` | Python BC residual profile; default `mish_se`; `mish_no_se` is opt-in throughput ablation with strength risk |
| `python_variant` | Python BC TorchInductor strategy only; default/canonical `compile_max_autotune`; use `compile_default` only for smoke/preflight/short debug |
| `validation_gates` | optional best-checkpoint gate; off by default |
| `python_raw_mjai_transport` | raw-MJAI input transport when `bc_shards_manifest_path` absent; default `pinned_pyo3`, fallback `stdout` |

Minimal raw-MJAI BC train, default Python/PyTorch backend:

```yaml
data_dir: /data/mjai              # raw replay root/file/archive; used directly when no BC shard manifest is set
output_dir: /output
num_epochs: 1
batch_size: 2048
microbatch_size: 1024
device: cuda:0
# bc_backend: python             # default plain-BC backend

bc:
  learning_rate: 0.00025
  min_learning_rate: 0.000001
  weight_decay: 0.00001
  grad_clip_norm: 1.0
  warmup_steps: 1000
```

Minimal BC shard train, default Python/PyTorch backend:

```yaml
data_dir: /data                 # still retained as config contract; shard path supplies samples
output_dir: /output
num_epochs: 1
batch_size: 2048
device: cuda:0
bc_shards_manifest_path: /data/bc_shards_manifest.json
# bc_backend: python            # default
```

If `bc_shards_manifest_path` is absent, Python learner receives `data_dir` and streams raw MJAI. Default transport is pinned PyO3 (`python_raw_mjai_transport: pinned_pyo3`); `stdout` remains fallback. Rust raw-MJAI helper runs in Pixi `default` env; Python training runs in Pixi `py-train`.

Legacy Rust/Burn BC path:

```yaml
bc_backend: rust_burn
```

Useful BC adds:

```yaml
microbatch_size: 256
validation_microbatch_size: 128
# precision_mode omitted on BC CUDA defaults to BF16 AMP; set fp32 explicitly for debug/repro FP32
train_fraction: 0.9
augment: true
tensorboard: true
```

BC head profile. Default `full` is canonical BC baseline. `policy_only` is explicit experiment/probe mode: train shard path runs backbone+policy head and policy CE only; value/GRP/tenpai/danger/opp-next/score losses log as zero because they are not computed. Do not use `policy_only` for default baseline comparisons.

```yaml
bc_head_profile: full        # default
# bc_head_profile: policy_only
```

Python residual profile. Default `mish_se` is canonical SE-ResNet Python BC: Mish activation, GroupNorm, squeeze-excitation every residual block, 10 blocks, 256 hidden channels. Profile name is checkpoint contract: checkpoints record exact string and reject mismatched resume.

Opt-ins:
- `silu_se`: SiLU activation, GroupNorm, SE; speed/activation probe, validation required before strength claim.
- `relu_se`: ReLU activation, GroupNorm, SE; speed/activation probe, higher semantic risk.
- `mish_no_se`: speed/ablation profile only; removes core SE, keeps Mish + GroupNorm. 5k equal-step raw-MJAI validation was faster in train loop but slightly worse than `mish_se`; keep opt-in, do not promote.
- `relu_no_se`: speed/ablation profile only; ReLU + GroupNorm, no SE. Never default without separate validation + arena evidence.
- `relu_no_norm_no_se`: speed/ablation profile only; ReLU without GroupNorm/SE. Validation-only, never default.

```yaml
python_residual_profile: mish_se          # default canonical SE-ResNet
# python_residual_profile: mish_no_se     # speed/ablation only; do not promote from throughput alone
# python_residual_profile: relu_no_norm_no_se # speed/ablation only; validation-only, never default
```

Python learner backbone profile is part of checkpoint metadata. Valid value now only `conv2d_local3`; default is `conv2d_local3`. Removed probe `token_linear_local3`: slower than Conv2d in raw-MJAI timing, higher architecture risk, no profiler reason to keep.

```bash
--backbone-profile conv2d_local3
```

Python compile variants do not change model math, topology, checkpoint architecture, input shape, action shape, residual profile, or losses; they only change TorchInductor strategy. Default production Python BC is `compile_max_autotune` for long same-architecture runs. Use `compile_default` for smoke/preflight/short debug when compile/autotune overhead dominates.

Raw-MJAI transport knobs:

```yaml
python_raw_mjai_transport: pinned_pyo3   # default; stdout fallback exists
```

Direct Python flags:

```bash
--raw-mjai-data-dir path/to/mjai --raw-mjai-transport pinned_pyo3 --raw-mjai-pinned-ffi target/release/libhydra_raw_mjai_ffi.so
```

Pinned PyO3 lookup: `HYDRA_RAW_MJAI_PINNED_LIB`, then `target/release/libhydra_raw_mjai_ffi.so`, then `target/debug/libhydra_raw_mjai_ffi.so`. If missing, build:

```bash
pixi run cargo build -p hydra-raw-mjai-ffi --release --quiet
```

Use `--raw-mjai-transport stdout` only for compat/debug fallback.

```bash
--python-variant compile_max_autotune
```

Torch 2.12 CUDA 12.6 probe env:

```bash
pixi run -e py-train-torch212-cu126 torch-check
pixi run -e py-train-torch212-cu126 python-bc-train -- \
  --raw-mjai-data-dir /path/to/mjai \
  --raw-mjai-transport pinned_pyo3 \
  --raw-mjai-worker-threads 20 \
  --raw-mjai-prefetch-batches 2 \
  --raw-mjai-queue-bound 8 \
  --variant compile_max_autotune \
  --batch 2048 \
  --microbatch 1024 \
  --warmup 10 \
  --steps 200 \
  --out /path/to/result.json \
  --quiet
```

Pins: `torch==2.12.0+cu126`, `torchvision==0.27.0+cu126`, PyTorch cu126 index. Use only on CUDA 12.6 target hardware. Local RTX 5070 `sm_120` cannot execute cu126 wheel kernels; local benchmark must stay on cu128/cu130-capable wheel.

Torch 2.12 nightly CUDA 12.8 local probe env:

```bash
TORCHINDUCTOR_MAX_AUTOTUNE_DEFER_LAYOUT_FREEZING=1 \
  pixi run -e py-train-torch212-nightly-cu128 python-bc-train -- \
  --raw-mjai-data-dir /home/cachybtw/Downloads/dataset_bundle/tenhou-houou-mjai-2025 \
  --raw-mjai-transport pinned_pyo3 \
  --raw-mjai-worker-threads 20 \
  --raw-mjai-prefetch-batches 2 \
  --raw-mjai-queue-bound 8 \
  --raw-mjai-max-games 5000 \
  --variant compile_max_autotune \
  --batch 2048 \
  --microbatch 1024 \
  --warmup 10 \
  --steps 200 \
  --out /home/cachybtw/tmp/hydra-torch212-nightly-cu128/result.json \
  --quiet
```

Pins: `torch==2.12.0.dev20260329+cu128`, `torchvision==0.26.0.dev20260329+cu128`. 2026-05-22 RTX 5070 result: `~43.96k samples/s`, `~46.59ms/step`, compile `~1.71s` with layout-defer env. Probe-only; do not production-pin nightly without repeat + validation evidence.
Experimental backbone profile. Default absent = canonical learner: 24 blocks, 256 hidden, Mish, SE every block, two GroupNorms per block. Research infra only: final op-count ablation did not show material throughput gain. Do not use for default training, throughput-win claims, or strength claims without separate evidence.

```yaml
experimental_backbone_profile:
  activation: relu          # mish | silu | relu
  se_every_n: 999           # 1 = every block; high value = none for current learner
  norm: first_only          # both | first_only
  num_blocks: 6             # optional architecture tradeoff
  hidden_channels: 256      # optional architecture tradeoff
```

Final fixed-shard ablation: reducing blocks 12 -> 6 did not improve throughput; best observed gain was ~1%, below material threshold.


Explicit CUDA FP32 override:

```yaml
device: cuda:0
precision_mode: fp32
```

Validation gate example. Affects `best_model` only; `latest_model` still saves for resume.

```yaml
validation_gates:
  enabled: true
  min_validation_samples: 1024
  max_policy_loss_regression: 0.0
  min_policy_agreement_delta: 0.0
  fail_training_on_gate_failure: false
  require_sidecar_coverage_when_weighted: true
```

Minimal RL add:

```yaml
rl:
  games_per_batch: 4
  temperature: 1.0
  phase: drda_ach_self_play
```

Current RL phase enum:
- `drda_ach_self_play`
- `exit_pondering`

Status: plain BC shard path defaults to Python/PyTorch through Rust launcher. Rust owns replay parsing, shard building, manifest validation, and CLI/config orchestration. Python owns BC model/loss/optimizer/BF16/`torch.compile`/checkpoint. ExIt/DeltaQ/belief/mixture/opponent-hand-type are not supported by Python default yet; use `bc_backend: rust_burn` only for those advanced modes or debugging.

## Preflight benchmark + YAML authority

Preflight is benchmark mode. It runs tuples operator supplies and emits markdown table. It is not config authority, does not read YAML, does not read shard manifest or dataset, does not read/write runtime cache, does not choose winner, and does not mutate YAML.

YAML `preflight:` is forbidden. `TrainConfig` rejects it. Preflight tuning is CLI-owned; training authority stays in YAML. Positional config paths and `--bc-shards-manifest-path` are not accepted in preflight mode.

Benchmark invocation shape:

```bash
train --preflight \
  --pf-candidate-tuples 1024:2:1:1,2048:4:2:2,4096:4:2:2,4096:8:4:2,8192:8:4:2 \
  --pf-warmup-steps 100 \
  --pf-measure-steps 1000 \
  --pf-repetitions 5 \
  --pf-output md
```

Tuple grammar is exact, not Cartesian:

```text
<batch>:<ring_batches>:<loader_threads>:<prefetch_batches>
```

Markdown table contract:

```md
| idx | status | device | mode | batch | ring | threads | prefetch | shuffle | codec | samples/s | MiB/s | p50 ms | p95 ms | producer wait % | consumer wait % | disk wait % | gpu input wait % | cpu user s | cpu sys s | error |
|---:|---|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
```

`status=pass` rows carry numeric throughput and wait ratios. Metrics that require real disk or GPU work and do not apply to this manifestless benchmark are emitted as numeric `0.0` by definition. `status=error` rows preserve tuple identity and put failure in `error`. Current benchmark codec is `none` only.

Operator contract:
- normal `train config.yaml` uses YAML fields only.
- YAML runtime fields are authority: `batch_size`, `microbatch_size`, `validation_microbatch_size`, `num_threads`, `buffer_games`, `buffer_samples`, `archive_queue_bound`, `shard_prefetch_depth`.
- unsafe math authority is YAML too: `bc.learning_rate`, `bc.min_learning_rate`, `bc.warmup_steps`.
- benchmark rows are evidence for human edit, not automatic training config.

Preflight benchmark behavior:
- no config file, YAML, dataset, or shard manifest is read.
- candidate tuple comes from each exact CLI tuple.
- synthetic/in-memory benchmark rows are not comparable to loose-replay or shard-reader throughput.
- benchmark output/report text must not contain authority words such as `selected`, `saved`, `recommended`, `runtime`, `cache_hit`, or `cache_key`.

Common preflight traps:
- passing config path or shard manifest to preflight; use only explicit benchmark CLI flags.
- treating fastest row as automatically applied; it is only evidence.
- editing YAML without preserving logical training intent (`batch_size`, LR schedule, split, precision, labels).
- comparing synthetic/in-memory preflight rows with shard-backed or loose/archive replay runs as if they were same data path.
## BC shards

BC shards = production steady-state input for replay-driven BC. Build once, consume many.

Use when:
- repeated BC preflight/train/validation.
- GPU should not wait on replay parse/decompress/engine replay.
- train/validation split must stay fixed by manifest.
- ExIt/DeltaQ labels should be baked into validated artifact.
- training should consume known-good dataset, not rediscover replay each run.

Raw loose/archive replay remains slow path for audit, shard production, debug, one-off transport comparison.

Build CLI:

```bash
pixi run cargo run -p hydra-train --bin build_bc_shards --no-default-features --features training -- \
  --input <dir|archive|replay> \
  --output-dir <dir> \
  [--manifest-name <file>] \
  [--shard-samples <usize>] \
  [--train-fraction <f32>] \
  [--max-games <usize>] \
  [--max-samples <usize>] \
  [--split train|val|both] \
  [--num-threads <usize>] \
  [--queue-bound <usize>] \
  [--resume] \
  [--resume-dir <dir>] \
  [--chunk-games <usize>] \
  [--report-name <file>|--no-report] \
  [--progress-jsonl <file>] \
  [--dry-scan-only] \
  [--exit-sidecar ...] \
  [--delta-q-sidecar ...]

pixi run cargo run -p hydra-train --bin build_bc_shards --no-default-features --features training -- --validate-manifest <path>
```

Defaults: manifest `bc_shards_manifest.json`, `--shard-samples 10000`, `--train-fraction 0.9`, `--split both`. Parallel/resume/report defaults are builder-owned; inspect `--help` for current binary defaults.
Shard storage is compact-only v3. Builder stores replay-fact baseline observation records only; old dense v2 shards are invalid and must be rebuilt from replay. No dense/v2 shard storage mode exists. Training still sees dense `192x34` f32 obs and 46-action legal masks after reader decode; replay BC shard advanced/search/Hand-EV observation tail is absent/zero.

Operator rules:
- use `--resume` for long corpus builds.
- never move manifest without shard files.
- report is op metadata; manifest is training contract.
- archive source counts may be hints; build report has loaded/skipped/empty actuals.
- resume fingerprint mismatch: use new output dir or intentionally delete resume state.
- non-resume output should be new/empty; do not mix stale shards with new manifest.
- `--dry-scan-only` uses build scan path; it must not become second scanner.

Phase 1 compact-shard proof:
1. Build bounded compact shards with `--split both`, `--progress-jsonl <file>`, and report enabled.
2. Validate: `pixi run cargo run -p hydra-train --bin build_bc_shards --no-default-features --features training -- --validate-manifest <proof-shards-dir>/bc_shards_manifest.json`.
3. Inspect report ABI, disk, split plan, output split stats, and rates.
4. Train from `bc_shards_manifest_path` with `shard_prefetch_depth: 2`.
5. Extract timing with `extract_timing_metrics`.
6. Tune `shard_prefetch_depth` only after proof, one value at time, same manifest/config.


CUDA shard no-starvation proof result: serious CUDA BC compact-shard runs should use `pixi run train-cuda-shards -- <config.yaml>` or equivalent explicit feature command `pixi run cargo run --release -p hydra-train --bin train --no-default-features --features cuda-graph -- <config.yaml>`. This enables pinned H2D staging and preallocated device tensors; production graph replay remains off/probe-only. Representative proof with batch 1024, microbatch 256, `shard_prefetch_depth: 2`, and fp32 passed input gates: producer wait 0.0004/0.0004%, H2D 0.602/0.627%, input starvation 0.605/0.627%, compute 98.97%, 2498.21 samples/s. `--features training` alone is semantically valid but does not enable CUDA pinned/prealloc transport and can be limited by pageable H2D materialization.
Prod workflow:
1. Audit replay corpus and sidecar inputs.
2. Build train+validation shards using same train fraction + sidecar provenance intended for training; use `--resume` for long builds.
3. Validate manifest and inspect split counts/totals/sidecar provenance plus build report skipped/empty/rates.
4. Run manifestless markdown preflight benchmark with exact `--pf-candidate-tuples` if choosing candidate runtime shapes.
5. Human edits YAML runtime fields if desired, preserving YAML as authority.
6. Train from `bc_shards_manifest_path` using same manifest.
7. Rebuild on dataset/contract change.

Shard consume semantics:
- train reads prebuilt shard rows, not raw replay scan.
- validation also reads shards, not loose replay stream.
- validation sample limits apply over shard rows.
- bounded validation sample materialization disabled.
- startup banner uses manifest counts.
- invalid manifest/header contract is hard error; no silent fallback.
- dense v2 shard magic hard-errors; no mixed reader or fallback.
- compact v3 is only shard format; no format flag, mixed reader, or dense fallback.
- shard files resolve relative to manifest parent. Do not move manifest alone.

Rebuild if any change:
- replay corpus contents.
- source filters or split policy (`train_fraction`, `split`).
- shard sizing (`shard_samples`).
- ExIt/DeltaQ sidecar path or provenance.
- encoder geometry, action space, shard version/layout, base record size, feature flags.
- requested train/validation split missing from artifact.

Manifest fields to inspect:
- `manifest_version`, `shard_version`
- `train_fraction`, `shard_samples`
- `input`, `output_dir`
- `source_count`, `source_total_games_hint`
- `exit_sidecar`, `delta_q_sidecar`
- `totals.sample_count`, `totals.skipped_games`, `totals.empty_games`
- `splits[*].sample_count`, shard count, feature flags, record size, descriptor list
- top-level `storage_layout` must be `compact`; v3 manifest/header/layout versions must match current binary

Build report fields to inspect:
- ABI, disk, planned splits, loaded/skipped/empty game counts
- shard/sample totals and rates
- error examples, bounded by CLI config
- resume reused/built chunks when enabled
- output `report_path`; manifest path remains training input
- output split `bytes_per_sample`, min/max shard bytes, feature flags, and record size show compact storage effect; dense equivalent is report-only

Sidecar-backed shard build requires full provenance tuple. Partial tuple rejected.

```bash
pixi run cargo run -p hydra-train --bin build_bc_shards --no-default-features --features training -- \
  --input /data/replays \
  --output-dir /output/bc-shards \
  --exit-sidecar /labels/exit.jsonl \
  --exit-source-net-hash 123456789 \
  --exit-source-version 7 \
  --delta-q-sidecar /labels/delta_q.jsonl \
  --delta-q-source-net-hash 123456789 \
  --delta-q-source-version 1
```

### BC shard benchmark protocol

Required baseline before code change: same semantic config as after run. Use placeholder paths; do not bake local artifact dirs into docs.

```bash
rm -rf path/to/baseline-out
/usr/bin/time -v \
  pixi run cargo run -p hydra-train --bin build_bc_shards --no-default-features --features training --quiet -- \
    --input path/to/replays \
    --output-dir path/to/baseline-out \
    --manifest-name manifest.json \
    --shard-samples 10000 \
    --train-fraction 0.9 \
    --split train
```

Record baseline:
- command line
- git ref
- hardware note: CPU, RAM, storage path, OS/kernel
- wall/user/sys from `/usr/bin/time -v`
- max RSS from `/usr/bin/time -v`
- source count / total hint / counts_exact from manifest
- input compressed bytes when cheap to compute
- skipped/empty games
- output shard count
- output sample count
- output bytes = sum descriptor byte_len
- samples/sec
- included files/sec if counts exact
- input MiB/sec if input bytes known
- output MiB/sec

Manual baseline shell report may live at example path:

```text
<output-dir>/bc_shard_build_report.baseline-shell.json
```

Use `schema_version: 0` or `captured_by: shell`; do not confuse with code report schema v1.

After impl: same input/config, new output dir, plus parallel/resume/report flags.

```bash
rm -rf path/to/after-out
/usr/bin/time -v \
  pixi run cargo run -p hydra-train --bin build_bc_shards --no-default-features --features training --quiet -- \
    --input path/to/replays \
    --output-dir path/to/after-out \
    --manifest-name manifest.json \
    --shard-samples 10000 \
    --train-fraction 0.9 \
    --split train \
    --num-threads 20 \
    --queue-bound 128 \
    --resume \
    --chunk-games 10000 \
    --report-name bc_shard_build_report.json \
    --progress-jsonl bc_shard_build_progress.jsonl
```

Compare semantic manifest fields, not `created_at`:

```text
totals.sample_count
totals.skipped_games
totals.empty_games
totals.shard_count
splits[*].sample_count
splits[*].shard_count
splits[*].feature_flags
splits[*].record_size
splits[*].shards[*].sample_count
splits[*].shards[*].first_sample_index
splits[*].shards[*].byte_len
```

Pass target:
- same semantic counts as serial baseline.
- shard reader loads produced manifest.
- speedup >= 5x on local loose-file subset, or report proves bottleneck is disk/decompression/writer.
- max RSS bounded by queue/chunk config, not source count.

Archive benchmark when tar/tar.zst corpus exists:

```bash
/usr/bin/time -v \
  pixi run cargo run -p hydra-train --bin build_bc_shards --no-default-features --features training --quiet -- \
    --input <path/to/replays.tar.zst> \
    --output-dir <tmp>/hydra-bc-shard-after-archive \
    --manifest-name manifest.json \
    --shard-samples 10000 \
    --train-fraction 0.9 \
    --split both \
    --num-threads 20 \
    --queue-bound 128 \
    --resume \
    --report-name bc_shard_build_report.json \
  2> <tmp>/hydra-bc-shard-after-archive.time.txt \
  | tee <tmp>/hydra-bc-shard-after-archive.stdout.txt
```

Archive acceptance:
- tar reader remains sequential.
- entry bytes queue bounded.
- worker parse parallelism visible in CPU user time / progress rates.
- output order matches serial archive order.

Full dataset dry scan after `--dry-scan-only` exists:

```bash
/usr/bin/time -v \
  pixi run cargo run -p hydra-train --bin build_bc_shards --no-default-features --features training --quiet -- \
    --input <full-replay-root> \
    --output-dir <tmp>/hydra-bc-shard-dry-scan \
    --train-fraction 0.9 \
    --split both \
    --dry-scan-only \
    --report-name bc_shard_scan_report.json \
  2> <tmp>/hydra-bc-shard-dry-scan.time.txt \
  | tee <tmp>/hydra-bc-shard-dry-scan.stdout.txt
```

Train-side benchmark from shards: if manifest exists, create short train YAML with `bc_shards_manifest_path`, short step count, stable batch config.

```bash
HYDRA_BENCHMARK_QUIET=1 \
/usr/bin/time -v \
  cargo run -p hydra-train --bin train --no-default-features --quiet -- \
    <tmp>/train-from-shards-bench.yaml \
  2> <output_dir>/train-from-shards.time.txt \
  | tee <output_dir>/train-from-shards.stdout.txt
```

Read `<output_dir>/bc/step_log.jsonl` for:
- `producer_wait`
- `data_load`
- `collation`
- `h2d_transfer`
- `train`
- `steps/s`
- derived `samples/s = steps/s * batch_size`

## Replay sidecars

Replay sidecar = offline JSONL keyed to replay decisions. Lets Hydra hydrate replay-time labels without recomputing search/DeltaQ during load.

Lanes:
- ExIt sidecar: `ReplayExitRecordV1`, root search-derived labels.
- DeltaQ sidecar: `ReplayDeltaQRecordV1`, child-minus-root value deltas for supported discard actions. Implemented, not default-on.

Builder shape:

```bash
cargo run -p hydra-train --bin build_replay_exit_sidecar -- \
  --input <replay.json|replay.json.gz> \
  --checkpoint <model_base> \
  --output <sidecar.jsonl> \
  --source-version <u32> \
  [--min-visits <u32>] \
  [--hard-state-threshold <f32>] \
  [--max-kl <f32>]

cargo run -p hydra-train --bin build_replay_delta_q_sidecar -- \
  --input <replay.json|replay.json.gz> \
  --checkpoint <model_base> \
  --output <sidecar.jsonl> \
  --source-version 1
```

Both emit sibling `.report.json`. Use report for label count, validation, coverage.

Provenance identity:
- `source_net_hash`: checkpoint identity.
- `source_version`: operator semantic version tag.
- Sidecar labels are not generic. They are bound to replay identity, checkpoint lineage, source version, legal mask/action contract.

Join checks before hydration:
- replay decision key / source identity match.
- action match.
- legal-mask digest match.
- `source_net_hash` match.
- `source_version` match.
- lane target/mask shape valid.
- DeltaQ additionally enforces legal discard-only mask/target semantics and `source_version == 1`.

Identity shape:
- loose replay joins by replay file name.
- archive replay joins by full archive-entry identity, e.g. `replays.tar.zst/path/inside/game.json`.
- Sidecar keyed to `game.json` matches loose `game.json`; not archive entry above.

Train YAML consume:

```yaml
exit_sidecar_path: /output/exit_index.jsonl
delta_q_sidecar_path: /output/delta_q_index.jsonl
```

If `advanced_loss.exit` or `advanced_loss.delta_q` > 0, matching sidecar path required. With `validation_gates.enabled: true`, validation batches must contain hydrated labels before best-checkpoint promotion.

Baseline v1 sidecar weights stay off unless explicit:

```yaml
advanced_loss:
  exit: 0.0
  safety_residual: 0.0
  delta_q: 0.0
```

Use A/B: same shards, same split, one changed weight.

Failure modes:
- missing sidecar key => no hydration for that decision.
- present sidecar record with wrong `source_version`, checkpoint hash, legal-mask digest, schema/shape, or provenance => hard error; do not treat as absent.
- identity mismatch loose-vs-archive blocks join.
- valid JSONL alone not enough.

## DeltaQ promotion

Status:
- DeltaQ implemented, intentionally not default-on.
- Promotion = gated eval workflow, not training default.
- BF16/autocast rejected in promotion mode.
- Baseline checkpoint mandatory.
- Artifact persists `arena_decision` and `arena_report` when arena runs.

Invoke:

```bash
cargo run -p hydra-train --bin train -- \
  /config/train.yaml \
  --delta-q-promotion \
  --delta-q-baseline-checkpoint /models/baseline/model_base
```

Hard gates:
- Missing `--delta-q-baseline-checkpoint` = invocation error.
- `precision_mode: bf16_autocast` = runtime error; DeltaQ promotion BF16 unsupported.

Decision flow:
1. Offline DeltaQ gate: candidate vs baseline on replay-derived DeltaQ decision quality.
2. Policy-transfer gate: ensures candidate does not transfer policy quality badly.
3. Arena confirmation: paired eval when pre-arena rec says candidate promising.

Offline metrics include eligible/compared states, top-1 agreement, mean regret, mean decision lift, negative lift fraction, candidate-beats-baseline rates, high-gap summaries.

Policy-transfer metrics include compared states, top-1-to-teacher rates, mean teacher regret, candidate-beats-baseline rate, negative transfer fraction.

Arena report includes compared games, baseline/candidate mean placement, delta mean placement, stable-dan scores, confidence bounds.

Persisted artifact:

```text
<output_dir>/delta_q_promotion.json
```

Fields to read first:
- `recommendation`: pre-arena rec from offline + transfer gates.
- `stage`: offline-only vs offline+transfer+arena.
- `arena_confirmation`: paired arena request, if constructed.
- `arena_decision`: arena-side decision, if present.
- `arena_report`: paired arena summary, if present.
- `policy_transfer`, `policy_transfer_result`: transfer behavior evidence.
- `report`, `result`: DeltaQ offline gate evidence.

rec states seen by ops:
- `RejectAtOfflineGate`: do not advance on current evidence.
- `RequiresArenaConfirmation`: candidate earned arena step; not accepted yet.

Common mistakes:
- Treating `RequiresArenaConfirmation` as acceptance.
- Reading console banner as durable truth; artifact is durable truth.
- Ignoring policy-transfer failure after offline DeltaQ improvement.
- Assuming DeltaQ lane default-on because tooling exists.

## Precision, CUDA shards, graph probe

Precision modes:
- `fp32`
- `bf16_autocast`

Current status:
- Omitted `precision_mode` on BC CUDA defaults to requested `bf16_autocast` and effective `bf16_amp`. Explicit `precision_mode: fp32` keeps FP32. CPU omission stays FP32.
- BF16 AMP wraps BC forward only. Loss construction, backward, optimizer state, checkpoints, and validation remain FP32.
- RL and DeltaQ promotion hard-error on BF16. On CUDA DeltaQ promotion, set explicit `precision_mode: fp32`. Do not claim CUDA graph BF16 support without exact measurement.
Shard CUDA fast path available only when all true:
- `hydra-train` built with `--features cuda-graph` (or run via `pixi run train-cuda-shards -- <config.yaml>`).
- runtime device CUDA.
- `bc_shards_manifest_path` set.

Build/run:

```bash
pixi run train-cuda-shards -- /path/to/config.yaml
pixi run cargo build --release -p hydra-train --bin train --no-default-features --features cuda-graph
```

Semantics:
- shard rows mmap/prefetch on CPU regardless.
- with `cuda-graph`, shard train/probe/validation use pinned host staging, async H2D copy stream, preallocated GPU tensors.
- without feature or on CPU, normal pageable tensor construction.
- current optimized path collates pageable host batches then copies to pinned staging before H2D.
- policy targets generated as bounded CPU f32 one-hot rows; invalid action IDs produce all-zero rows.
- H2D single-buffered; compute waits on copy event.
- CPU rare-action counts remain CPU; only GPU loss scalars read back at logical-batch stats boundary.
- pinned footprint batch-size bounded; keep `shard_prefetch_depth` small unless profiling shows producer/H2D bubbles.

CUDA graph probe:

```bash
HYDRA_CUDA_GRAPH_PROBE=1 train config.yaml
```

Reports JSON with `probe_mode=compute_capture_only`, warmup/parity/capture timings, replay repeats, blockers. Production replay intentionally off: `cuda_graph_replay=production_off_probe_only`, because Burn Adam needs fresh Rust-side `GradientsParams`; graph replay cannot safely feed optimizer state.

Probe knobs:
- `HYDRA_CUDA_GRAPH_PROBE_REPLAYS=N`; default `16`, max `1024`.
- `HYDRA_CUDA_GRAPH_PROBE_POST_REPLAY_PARITY=0`; disables post-replay parity rerun.

Recent shard slice was CUDA-graph transport/probe signal, not BF16 evidence. Single-run shard signals are not promotion evidence. Main bottleneck remains model compute + unfused Burn Adam.

Nsight/NVTX capture: `HYDRA_NVTX=1` needs Pixi NVTX library visible or `nsys stats --report nvtx_kern_sum:base` may report no NVTX data:

```bash
LD_LIBRARY_PATH=.pixi/envs/default/lib/python3.12/site-packages/nvidia/nvtx/lib:$LD_LIBRARY_PATH
```

Do not make source-attribution claims from Nsight reports until `nvtx_kern_sum` shows Hydra ranges.
## Runtime advisories

Hydra emits advisories to console and `bc/step_log.jsonl` as `runtime_advisories`. They mean valid-but-underoptimized unless paired with hard error.

Keys:
- `cpu_device_for_training`: CPU run; CUDA feeding optimizations unavailable.
- `steady_state_cuda_bc_uses_loose_replay`: CUDA BC uses loose/archive replay; build shards for repeated steady-state training.
- `cuda_shards_without_pinned_async_h2d`: CUDA shard run built without `cuda-graph`.
- `small_microbatch_high_accumulation_overhead`: accumulation overhead may dominate.
- `explicit_microbatch_blocks_faster_candidate_search`: explicit microbatch may block faster candidates.
- `logging_or_metric_sync_overhead`: `log_every_n_steps=1` may hurt CUDA throughput.
- `validation_or_checkpoint_cadence_overhead`: cadence may dominate wall.
- `selected_train_runtime_slower_than_best_probe_candidate`: selected train microbatch >=20% slower than best stable probe candidate.
- `selected_validation_runtime_slower_than_best_probe_candidate`: same for validation.

Hard failures remain for invalid contracts: stale shard manifest, missing required sidecars, unsupported precision in DeltaQ promotion, missing baseline.

## Operator checklist

New BC steady-state run:
1. Audit corpus with `mjai_audit`; command refs in `docker/train/README.md`.
2. Build sidecars if using ExIt/DeltaQ supervision; capture reports.
3. Build BC shards with matching sidecar provenance.
4. Inspect manifest split/sample totals and sidecar provenance.
5. Run manifestless markdown preflight with exact candidate tuples if choosing runtime shapes.
6. Edit YAML runtime fields by hand if accepting row.
7. Train with `bc_shards_manifest_path` pointing at same manifest.
8. Treat validation gates as best-checkpoint gate, not latest-checkpoint blocker.

Debug/audit run:
- Use raw replay loading intentionally.
- Compare shard-backed and loose-replay as different data paths.
- Keep same split/filter/sidecar provenance when comparing losses.

Promotion run:
1. Pick candidate and accepted baseline checkpoint.
2. On CUDA, set explicit `precision_mode: fp32`; omitted precision defaults to BC BF16 AMP and DeltaQ promotion rejects BF16.
3. Run DeltaQ promotion command with baseline.
4. Read console for quick path.
5. Archive `delta_q_promotion.json` as durable decision record.
6. Do not promote from `RequiresArenaConfirmation` alone.
