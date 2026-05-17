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
| normal train | `train config.yaml` | BC/RL per YAML |
| preflight | `train config.yaml --preflight --preflight-mode safe` | measure/select runtime tuple, cache repeated preflight result |
| probe-only | `train config.yaml --probe-kind <train|validation|rl_games|rl_microbatch> --probe-candidate-microbatch <N> ...` | bounded candidate check, no full train |
| DeltaQ promotion | `train config.yaml --delta-q-promotion --delta-q-baseline-checkpoint <path>` | candidate-vs-baseline gated eval |

Internal child probe path exists. Not op entrypoint.

Choose:
- normal: runtime tuple already trusted or config explicit.
- preflight: new GPU/machine, precision change, workload/shard change, old cache suspect.
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
| `microbatch_size` | explicit train selected-runtime override |
| `validation_microbatch_size` | explicit validation selected-runtime override |
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
| `shard_prefetch_depth` | shard host-batch queue depth; default `2`, valid `1..64` |
| `validation_gates` | optional best-checkpoint gate; off by default |

Minimal BC:

```yaml
data_dir: /data
output_dir: /output
num_epochs: 1
batch_size: 2048

bc:
  learning_rate: 0.00025
  min_learning_rate: 0.000001
  weight_decay: 0.00001
  grad_clip_norm: 1.0
  warmup_steps: 1000
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

Status: BC path is stable baseline. `bf16_autocast` runs BC CUDA AMP forward as effective `bf16_amp`; loss/backward/optimizer/checkpoint/validation stay FP32. RL and DeltaQ promotion hard-error on BF16.

## Preflight authority + cache

Preflight answers: what runtime tuple can this machine/workload sustain under selected tuning safety?

YAML `preflight:` is forbidden. `TrainConfig` rejects it. Preflight tuning is CLI-owned.

Default safe preflight:

```bash
train config.yaml --preflight --preflight-mode safe
```

`--preflight-mode` is mandatory with `--preflight`.

Safe mode tunes only math-preserving runtime knobs: train microbatch, validation microbatch, derived accumulation for unchanged `batch_size`, loader tuple (`num_threads`, `buffer_games`, `buffer_samples`, `archive_queue_bound`), and probe/benchmark search effort. It must not change logical optimizer batch, LR/schedule, precision, loss, model, data split, augmentation, sample set, labels, or targets.

Unsafe mode is explicit and may tune math-affecting performance knobs:

```bash
train config.yaml --preflight --preflight-mode unsafe \
  --pf-unsafe-batch-size 1024,2048,4096 \
  --pf-unsafe-lr-scale 0.5,1.0,1.5 \
  --pf-unsafe-warmup-steps 500-2000+500
```

Unsafe flags hard-error unless `--preflight-mode unsafe`. Reports include selected values and `lr_auto_scaled=false`; values are explicit candidates, not silent automatic scaling.

Operator contract:
- normal `train config.yaml` does not read preflight tuning config.
- normal training does not apply preflight cache.
- YAML runtime fields are authority: `microbatch_size`, `validation_microbatch_size`, `num_threads`, `buffer_games`, `buffer_samples`, `archive_queue_bound`.
- unsafe math authority is YAML too: `batch_size`, `bc.learning_rate`, `bc.min_learning_rate`, `bc.warmup_steps`.
- after preflight, copy selected runtime values into YAML before training if desired.

Selected-runtime tuple:
- `train_microbatch_size` -> copy to `microbatch_size`
- `validation_microbatch_size` -> copy to `validation_microbatch_size`
- derived `accum_steps`
- unsafe-only `unsafe_selected_batch_size` -> copy to `batch_size` only if intentionally accepting unsafe math
- unsafe-only selected LR/min LR/warmup fields -> copy to `bc.learning_rate`, `bc.min_learning_rate`, `bc.warmup_steps` only if intentionally accepting unsafe math

Loader-runtime tuple: data/replay loader knobs `num_threads`, `buffer_games`, `buffer_samples`, `archive_queue_bound`; copy into YAML when accepting result.

Cache key covers hardware, workload, CLI preflight config signature, explicit microbatch overrides. Manifest-cache scan reuse also requires replay-selection contract match: `train_fraction`, `source_filters`. Cache key deliberately excludes knobs not defining selected-runtime contract: `data_dir`, `seed`, `num_threads`, `buffer_games`, `buffer_samples`. Meaning: same runtime-selection problem, not byte-identical YAML.

Identical-run fast path:
- `run_preflight` / `run_rl_preflight` can hit v6 cache and skip probes.
- Probe result vectors empty on that path because no probe ran.
- Cache is preflight-identical-run only; normal BC/RL bootstrap does not consume it.

Preflight CLI knobs worth knowing:
- safety label: `--preflight-mode <safe|unsafe>`
- profile: `--pf-profile <default|fast-repeated-run>`
- candidate ladder: `--pf-candidate-microbatch`, `--pf-min-microbatch`, `--pf-allow-explicit-microbatch-override`
- probe stability: `--pf-warmup-steps`, `--pf-measure-steps`, `--pf-required-successes`, `--pf-noise-tolerance`
- loader search: `--pf-loader-rounds`, `--pf-loader-tuple-margin`, `--pf-loader-extra-samples`
- stage-2: `--pf-real-benchmark`, `--pf-real-benchmark-*`
- refinement: `--pf-local-refinement`, `--pf-local-refinement-*`, `--pf-search-coordinate-rounds`, `--pf-search-top-k`
- unsafe candidates: `--pf-unsafe-batch-size`, `--pf-unsafe-lr-scale`, `--pf-unsafe-warmup-steps`

Internal artifact/cache field names keep serialized names (`tuning_mode`, `candidate_microbatches`, `unsafe_candidate_batch_sizes`, etc.). Do not rename when reading reports.

Stage-2 benchmark may reuse bounded validation cache only when validation sample limit finite and finalists share loader-runtime + resolved validation limit. Materialization cost still counted. Shard-backed validation does not use this in-memory cache path.

Shard-backed preflight changes behavior:
- train + validation probes load shard readers directly.
- loader-runtime tuning collapses to config-derived loader tuple.
- stage-2 finalist benchmark skipped even if enabled.
- Not comparable to loose-replay preflight for replay-scan throughput.

Fast repeated shard profile only when hardware, requested/effective precision, shard manifest, prior runtime known-good. YAML keeps workload/runtime inputs only:

```yaml
bc_shards_manifest_path: /output/bc-shards/bc_shards_manifest.json
microbatch_size: 256
validation_microbatch_size: 128
```

CLI owns fast preflight tuning:

```bash
train config.yaml --preflight --preflight-mode safe --pf-profile fast-repeated-run
```

`fast-repeated-run` profile sets one-candidate fast probe (`fast_repeated_run_profile=true`, window `1`, successes/warmup/measure `1`, loader rounds/extra samples `0`, real benchmark off). All other preflight fields stay default unless explicit `--pf-*` flags override.

New hardware, new shard artifact, changed requested/effective precision, or odd throughput: use full default preflight.

Common preflight traps:
- explicit microbatch can still be rejected by safety/authority.
- cache hit can skip probes; absence of probe rows not failure.
- precision change invalidates assumptions.
- advisory `selected_*_runtime_slower_than_best_probe_candidate` means optimization gap, not wrong result.
- unsafe mode can change logical `batch_size` and selected LR/min LR/warmup when candidate fields are configured; apply only by copying reported values into YAML intentionally.
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
cargo run -p hydra-train --bin build_bc_shards -- \
  --input <dir|archive|replay> \
  --output-dir <dir> \
  [--manifest-name <file>] \
  [--shard-samples <usize>] \
  [--train-fraction <f32>] \
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

Prod workflow:
1. Audit replay corpus and sidecar inputs.
2. Build train+validation shards using same train fraction + sidecar provenance intended for training; use `--resume` for long builds.
3. Inspect manifest split counts/totals/sidecar provenance and build report skipped/empty/rates.
4. Set `bc_shards_manifest_path: /output/bc-shards/bc_shards_manifest.json` (example path).
5. Run preflight + training against same manifest.
6. Rebuild on dataset/contract change.

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
- loaded/skipped/empty game counts
- shard/sample totals and rates
- error examples, bounded by CLI config
- resume reused/built chunks when enabled
- output `report_path`; manifest path remains training input
- output `bytes_per_sample` and `savings_ratio_vs_dense_observation` show compact storage effect; dense equivalent is report-only

Sidecar-backed shard build requires full provenance tuple. Partial tuple rejected.

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

### BC shard benchmark protocol

Required serial baseline before code change: same semantic config as after run. Example paths from original plan only; do not assume they exist.

```bash
rm -rf /home/cachybtw/tmp/hydra-bc-shard-baseline-2019
/usr/bin/time -v \
  cargo run -p hydra-train --bin build_bc_shards --no-default-features --quiet -- \
    --input /home/cachybtw/Downloads/dataset_bundle/majsoul-jade-mjai-2019 \
    --output-dir /home/cachybtw/tmp/hydra-bc-shard-baseline-2019 \
    --manifest-name manifest.json \
    --shard-samples 10000 \
    --train-fraction 0.9 \
    --split train \
  2> /home/cachybtw/tmp/hydra-bc-shard-baseline-2019.time.txt \
  | tee /home/cachybtw/tmp/hydra-bc-shard-baseline-2019.stdout.txt
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
rm -rf /home/cachybtw/tmp/hydra-bc-shard-after-2019
/usr/bin/time -v \
  cargo run -p hydra-train --bin build_bc_shards --no-default-features --quiet -- \
    --input /home/cachybtw/Downloads/dataset_bundle/majsoul-jade-mjai-2019 \
    --output-dir /home/cachybtw/tmp/hydra-bc-shard-after-2019 \
    --manifest-name manifest.json \
    --shard-samples 10000 \
    --train-fraction 0.9 \
    --split train \
    --num-threads 20 \
    --queue-bound 128 \
    --resume \
    --chunk-games 10000 \
    --report-name bc_shard_build_report.json \
    --progress-jsonl bc_shard_build_progress.jsonl \
  2> /home/cachybtw/tmp/hydra-bc-shard-after-2019.time.txt \
  | tee /home/cachybtw/tmp/hydra-bc-shard-after-2019.stdout.txt
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
  cargo run -p hydra-train --bin build_bc_shards --no-default-features --quiet -- \
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
  cargo run -p hydra-train --bin build_bc_shards --no-default-features --quiet -- \
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
- wrong `source_version` or checkpoint hash => no hydration.
- legal-mask drift blocks join.
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
- `hydra-train` default features enabled (default includes `cuda-graph`; use `--no-default-features` to opt out).
- runtime device CUDA.
- `bc_shards_manifest_path` set.

Build/run:

```bash
cargo run --release -p hydra-train --bin train -- /path/to/config.yaml
cargo build --release -p hydra-train
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

Recent shard slice was CUDA-graph transport/probe signal, not BF16 evidence: `1981.9 samples/s`, wall `4.63s`; prior plain shard mean `1888.9 samples/s`; about `+4.9%` in one run. Single-run signal only. Main bottleneck still model compute + unfused Burn Adam.

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
5. Point config at `bc_shards_manifest_path`.
6. Run `train config.yaml --preflight --preflight-mode safe` on new/changed hardware/workload.
7. Copy selected runtime into YAML before training if desired; then train with same manifest/config.
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
