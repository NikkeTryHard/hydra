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
| preflight | `train config.yaml --preflight` | measure/select runtime tuple, cache result |
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
| `precision_mode` | `fp32` or `bf16_autocast` |
| `device` | backend device label |
| `bc` | BC optimizer knobs |
| `rl` | RL/self-play enable + phase knobs |
| `advanced_loss` | optional ExIt/safety/DeltaQ supervised weights |
| `preflight` | runtime autotune knobs |
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
precision_mode: bf16_autocast
train_fraction: 0.9
augment: true
tensorboard: true
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

Status: BC path is stable baseline. RL exists, but BC-first op surface remains safer. BF16/AMP shipped for BC; RL and DeltaQ promotion not baseline BF16 surfaces.

## Preflight authority + cache

Preflight answers one question: what runtime tuple can this machine/workload sustain?

Selected-runtime tuple:
- `train_microbatch_size`
- `validation_microbatch_size`
- derived accumulation/throughput effects

Loader-runtime tuple: data/replay loader knobs such as threads/buffers. Separate authority.

Authority rules:
- Fresh BC: selected-runtime config-derived unless preflight chooses; loader-runtime config-derived.
- Epoch-boundary resume: may reuse matching preflight-selected selected-runtime when authority + cache identity match; loader-runtime still config-derived.
- Partial-epoch resume: runtime must match prior compatible resume contract. Stricter by design.
- Matching BC preflight cache does not make loader-runtime authoritative.

Cache key covers hardware, workload, preflight config signature, explicit microbatch overrides. Manifest-cache scan reuse also requires replay-selection contract match: `train_fraction`, `source_filters`. Cache key deliberately excludes knobs not defining selected-runtime contract: `data_dir`, `seed`, `num_threads`, `buffer_games`, `buffer_samples`. Meaning: same runtime-selection problem, not byte-identical YAML.

Identical-run fast path:
- `run_preflight` / `run_rl_preflight` can hit v4 cache and skip probes.
- Probe result vectors empty on that path because no probe ran.

Preflight knobs worth knowing:
- candidate ladder: `candidate_microbatches`, `min_microbatch_size`, `allow_override_explicit_microbatch`
- probe stability: `warmup_steps`, `measure_steps`, `required_successes`, `measure_noise_tolerance_ratio`
- loader search: `loader_runtime_rounds`, `loader_tuple_margin_ratio`, `loader_tuple_extra_samples`
- stage-2: `real_benchmark_enabled`, candidate caps, `real_benchmark_max_finalists`
- refinement: `local_refinement_enabled`, `local_refinement_max_candidates`, `local_refinement_min_gap`, `search_coordinate_rounds`, `search_top_k`

Stage-2 benchmark may reuse bounded validation cache only when validation sample limit finite and finalists share loader-runtime + resolved validation limit. Materialization cost still counted. Shard-backed validation does not use this in-memory cache path.

Shard-backed preflight changes behavior:
- train + validation probes load shard readers directly.
- loader-runtime tuning collapses to config-derived loader tuple.
- stage-2 finalist benchmark skipped even if enabled.
- Not comparable to loose-replay preflight for replay-scan throughput.

Fast repeated shard profile only when hardware, precision, shard manifest, prior runtime known-good:

```yaml
bc_shards_manifest_path: /output/bc-shards/bc_shards_manifest.json
microbatch_size: 256
validation_microbatch_size: 128
preflight:
  fast_repeated_run_profile: true
  fast_repeated_run_candidate_window: 1
  required_successes: 1
  warmup_steps: 1
  measure_steps: 1
  loader_runtime_rounds: 0
  loader_tuple_extra_samples: 0
  real_benchmark_enabled: false
```

New hardware, new shard artifact, changed precision, or odd throughput: use full default preflight.

Common preflight traps:
- explicit microbatch can still be rejected by safety/authority.
- cache hit can skip probes; absence of probe rows not failure.
- precision change invalidates assumptions.
- advisory `selected_*_runtime_slower_than_best_probe_candidate` means optimization gap, not wrong result.

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
  [--exit-sidecar <path> --exit-source-net-hash <u64> --exit-source-version <u32>] \
  [--delta-q-sidecar <path> --delta-q-source-net-hash <u64> --delta-q-source-version <u32>]
```

Defaults: manifest `bc_shards_manifest.json`, `--shard-samples 10000`, `--train-fraction 0.9`, `--split both`.

Prod workflow:
1. Audit replay corpus and sidecar inputs.
2. Build train+validation shards using same train fraction + sidecar provenance intended for training.
3. Inspect manifest split counts, totals, sidecar provenance.
4. Set `bc_shards_manifest_path: /output/bc-shards/bc_shards_manifest.json`.
5. Run preflight + training against same manifest.
6. Rebuild on dataset/contract change.

Shard consume semantics:
- train reads prebuilt shard rows, not raw replay scan.
- validation also reads shards, not loose replay stream.
- validation sample limits apply over shard rows.
- bounded validation sample materialization disabled.
- startup banner uses manifest counts.
- invalid manifest/header contract is hard error; no silent fallback.
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
- BC training/preflight/probe/stage-2 dispatch by precision.
- RL and DeltaQ promotion not baseline BF16 surfaces.

Shard CUDA fast path available only when all true:
- Cargo feature `cuda-graph` enabled.
- runtime device CUDA.
- `bc_shards_manifest_path` set.

Build/run:

```bash
cargo run --release -p hydra-train --features cuda-graph --bin train -- /path/to/config.yaml
cargo build --release -p hydra-train --features cuda-graph
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

Observed recent 64-step shard slice: `1981.9 samples/s`, wall `4.63s`; prior plain shard mean `1888.9 samples/s`; about `+4.9%` profiled throughput in one run. Single-run signal only. Main bottleneck still model compute + unfused Burn Adam.

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
6. Run `train config.yaml --preflight` on new/changed hardware/workload.
7. Train with same manifest/config.
8. Treat validation gates as best-checkpoint gate, not latest-checkpoint blocker.

Debug/audit run:
- Use raw replay loading intentionally.
- Compare shard-backed and loose-replay as different data paths.
- Keep same split/filter/sidecar provenance when comparing losses.

Promotion run:
1. Pick candidate and accepted baseline checkpoint.
2. Ensure `precision_mode` not `bf16_autocast`.
3. Run DeltaQ promotion command with baseline.
4. Read console for quick path.
5. Archive `delta_q_promotion.json` as durable decision record.
6. Do not promote from `RequiresArenaConfirmation` alone.
