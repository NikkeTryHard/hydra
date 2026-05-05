# Hydra Preflight and Runtime Selection

Ops guide for Hydra preflight cache, runtime authority, probe-driven runtime selection.

Doc explains how Hydra picks train/validation microbatch settings, what preflight cache means, when cached results are authoritative vs informative. High-level training entrypoint: [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md). Compact compatibility contracts: [`docs/COMPATIBILITY_SURFACE.md`](COMPATIBILITY_SURFACE.md).

## What preflight is for

Hydra preflight answers one narrow safety question:

> What runtime tuple can this machine and this workload sustain for the current training run?

Selection surface:

- selected train microbatch size
- selected validation microbatch size
- derived accumulation behavior tied to those choices
- loader-runtime tuple evaluation for data-path throughput

Preflight not only benchmark. It is policy layer turning probes, cache entries, explicit settings into `EffectiveRuntimeConfig`.

## Terms that matter

### Selected-runtime

Selected-runtime tuple = operator-visible training runtime choice. Current baseline means:

- `train_microbatch_size`
- `validation_microbatch_size`
- any derived accumulation/throughput consequences tied to those values

### Loader-runtime

Loader-runtime = replay/data-path runtime tuple, incl knobs like loader threads, buffering. Affects ingest throughput, but governed differently from selected-runtime tuple.

### Probe

Probe = bounded runtime measurement pass testing one candidate microbatch or loader tuple. Results rank candidates and may feed stage-2 benchmark.

### Stage-2 benchmark

When enabled, Hydra runs fuller real benchmark over shortlisted finalists after cheaper probes. Reason: probe winner may not be best end-to-end training runtime once loader and validation behavior included.

## Authority rules

Rules operators need most.

### Fresh BC run

For fresh BC run:

- selected-runtime is config-derived unless preflight chooses runtime
- loader-runtime remains config-derived

### Epoch-boundary resume

For epoch-boundary resume:

- Hydra may reuse matching preflight-selected selected-runtime tuple when authority rules and cache identity match
- loader-runtime still remains config-derived

### Partial-epoch resume

For partial-epoch resume:

- runtime must remain identical to prior run's compatible resume contract
- intentionally stricter than epoch-boundary resume

### Loader-runtime rule

Most important non-obvious rule:

> A matching BC preflight cache does not automatically make loader-runtime authoritative.

Hydra keeps loader-runtime config-derived even when reusing matching selected-runtime results.

## What is in the cache key

Current docs and code agree preflight key covers:

- hardware
- workload
- preflight config signature
- explicit microbatch overrides

Manifest-cache identity used by probe/preflight scan reuse is narrower than full training config, but not only `data_dir`. Cache reuse path also requires replay-selection contract match, including:

- `train_fraction`
- `source_filters`

This matters when comparing two runs pointing at same replay root but using different include/exclude filters. Manifest cache hit means Hydra believes replay-selection problem is same; changing `source_filters` intentionally breaks identity.

And deliberately excludes some knobs not defining selected-runtime contract, such as:

- `data_dir`
- `seed`
- `num_threads`
- `buffer_games`
- `buffer_samples`

So cache hit means “same relevant runtime-selection problem,” not “same exact whole config file.”

## Identical-run fast path

Hydra current baseline supports identical-run fast path:

- `run_preflight` and `run_rl_preflight` consult cache before probes
- matching cache hit on current v4 key can skip all probes
- probe result vectors empty on that cache-hit path because no new probe pass ran

This speeds repeated runs on unchanged hardware/workload/settings without changing authority rules above.

## Preflight knobs that matter most

`PreflightConfig` large, but operators usually need smaller mental model.

### Candidate set and explicit override behavior

- `candidate_microbatches`
- `min_microbatch_size`
- `allow_override_explicit_microbatch`

These define search ladder and whether Hydra may move away from explicit requested microbatch in probe-driven flows.

### Probe stability and acceptance

- `warmup_steps`
- `measure_steps`
- `required_successes`
- `measure_noise_tolerance_ratio`

These control how noisy or brittle runtime search may be.

### Loader-runtime search

- `loader_runtime_rounds`
- `loader_tuple_margin_ratio`
- `loader_tuple_extra_samples`

These govern how hard Hydra searches data-path throughput rather than pure train-step throughput.

### Stage-2 benchmark gating

- `real_benchmark_enabled`
- `real_benchmark_train_candidates`
- `real_benchmark_validation_candidates`
- `real_benchmark_loader_candidates`
- `real_benchmark_max_finalists`

These control whether more expensive end-to-end finalist benchmark may refine winner.

### Stage-2 validation cache

Stage-2 benchmark can reuse pre-materialized validation samples across finalists when two conditions hold:

- validation sample limit is finite, so actual bounded validation cache exists
- multiple finalists share same loader-runtime tuple and resolved validation sample limit

When true, Hydra materializes validation samples once, reuses across finalists, records one-time materialization cost separately. That materialization time still counts in benchmark accounting so operators do not misread finalist as “free validation.”

This reuse applies only to loose-replay validation cache path. Shard-backed validation does not use this in-memory cached-sample route.

When `bc_shards_manifest_path` is set, preflight behavior changes beyond validation caching:

- BC train and validation probes load shard readers directly instead of replay-manifest scan/cache path for those probe kinds
- loader-runtime tuning collapses to config-derived loader tuple instead of normal loader finalist search
- stage-2 finalist benchmark is skipped entirely for shard-backed BC runs, even if `real_benchmark_enabled` is true

So shard-backed preflight results are not directly comparable to loose-replay preflight results when reasoning about replay-scan throughput or stage-2 winner selection.

### Local refinement and coordinate search

- `local_refinement_enabled`
- `local_refinement_max_candidates`
- `local_refinement_min_gap`
- `search_coordinate_rounds`
- `search_top_k`

These let Hydra do more local or coordinated search instead of accepting first coarse-ranked result.

## Practical defaults

Defaults intentionally conservative enough for baseline search policy:

- descending candidate ladder from large to small microbatches
- short warmup/measure phases for initial probes
- real benchmark enabled
- local refinement enabled
- small number of coordinate rounds

If not actively investigating runtime behavior, default recommendation:

- keep default preflight config
- only override preflight knobs when concrete reason exists

## When to disable or narrow preflight behavior

You may want lower preflight cost when:

- running repeated experiments on identical hardware
- already trust narrow candidate range
- debugging specific candidate microbatch or loader tuple

Examples of narrowing moves:

- reduce `candidate_microbatches`
- disable `real_benchmark_enabled`
- disable `local_refinement_enabled`
- lower `loader_runtime_rounds`

Treat these as debugging/iteration tools, not default long-term baseline.

## Probe-only workflow

Probe-only CLI path exists for targeted measurement without normal training. It takes:

- `--probe-kind <train|validation>`
- `--probe-candidate-microbatch <N>`
- optional warmup and measure step overrides

Use probe-only when you want answer:

- can this candidate microbatch run at all?
- is validation behaving differently from training?
- did a precision/device/config change alter runtime headroom?

## RL-specific note

Preflight config also carries RL-oriented memory and growth-safety knobs, but current shipped precision/runtime baseline is still BC-first. Treat RL preflight as real surface, but not yet most stable operator path compared with BC preflight.

## Failure modes to watch for

- Explicit microbatch settings may still be rejected when authority or safety rules require them fixed.
- Cache hit does not mean loader-runtime became authoritative.
- Cache hit may intentionally produce no probe results because Hydra skipped probing.
- Precision-mode changes may invalidate assumptions even if YAML looks mostly same.
- Stage-2 benchmark throughput may include one-time validation materialization cost when Hydra decides cache reuse is valid for finalist group.

## Recommended operator workflow

1. Start with config you actually want to train.
2. Run `train config.yaml --preflight` on new hardware or materially changed workloads.
3. Accept selected-runtime result unless explicitly investigating runtime policy.
4. For repeated runs on same setup, rely on cache-hit fast path rather than re-tuning manually.
5. If resuming, distinguish epoch-boundary resume from partial-epoch resume before assuming cached runtime reuse is allowed.

## Where to read next

- Need main training entrypoint and mode selection? Read [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md).
- Need replay-side supervision lanes depending on training config? Read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).
- Need compact authority table? Read [`docs/COMPATIBILITY_SURFACE.md`](COMPATIBILITY_SURFACE.md).