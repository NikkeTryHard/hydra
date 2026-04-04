# Hydra Preflight and Runtime Selection

Operational guide to Hydra's preflight cache, runtime authority rules, and probe-driven runtime selection.

This document explains how Hydra chooses train/validation microbatch settings, what the preflight cache actually means, and when cached results are authoritative versus merely informative. For the high-level training entrypoint, read [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md). For compact compatibility contracts, read [`docs/COMPATIBILITY_SURFACE.md`](COMPATIBILITY_SURFACE.md).

## What preflight is for

Hydra's preflight flow exists to answer a narrow question safely:

> What runtime tuple can this machine and this workload sustain for the current training run?

The selection surface includes:

- selected train microbatch size
- selected validation microbatch size
- derived accumulation behavior tied to those choices
- loader-runtime tuple evaluation for data-path throughput

Preflight is not just a benchmark. It is a policy decision layer that turns probes, cache entries, and explicit settings into an `EffectiveRuntimeConfig`.

## Terms that matter

### Selected-runtime

The selected-runtime tuple is the operator-visible training runtime choice. In the current baseline that means:

- `train_microbatch_size`
- `validation_microbatch_size`
- any derived accumulation/throughput consequences tied to those values

### Loader-runtime

Loader-runtime is the replay/data-path runtime tuple, including knobs like loader threads and buffering. It affects ingest throughput, but it is governed differently from the selected-runtime tuple.

### Probe

A probe is a bounded runtime measurement pass that tests one candidate microbatch or loader tuple. Probe results are used to rank candidates and optionally feed a stage-2 benchmark.

### Stage-2 benchmark

When enabled, Hydra performs a fuller real benchmark over shortlisted finalists after cheaper probe passes. This exists because a candidate that looks good in probe-only form may not be the best end-to-end training runtime once loader and validation behavior are included.

## Authority rules

These are the rules operators most often need.

### Fresh BC run

For a fresh BC run:

- selected-runtime is config-derived unless preflight is used to choose a runtime
- loader-runtime remains config-derived

### Epoch-boundary resume

For an epoch-boundary resume:

- Hydra may reuse a matching preflight-selected selected-runtime tuple when authority rules and cache identity match
- loader-runtime still remains config-derived

### Partial-epoch resume

For a partial-epoch resume:

- runtime must remain identical to the prior run's compatible resume contract
- this is intentionally stricter than epoch-boundary resume

### Loader-runtime rule

The most important non-obvious rule is:

> A matching BC preflight cache does not automatically make loader-runtime authoritative.

Hydra keeps loader-runtime config-derived even when it reuses matching selected-runtime results.

## What is in the cache key

Current docs and code agree that the preflight key covers:

- hardware
- workload
- preflight config signature
- explicit microbatch overrides

The manifest-cache identity used by probe/preflight scan reuse is narrower than the full training config, but it is not just `data_dir`. The cache reuse path also requires the replay-selection contract to match, including:

- `train_fraction`
- `source_filters`

That matters when you are comparing two runs that point at the same replay root but use different include/exclude filters. A manifest cache hit means Hydra believes the replay-selection problem is the same; changing `source_filters` intentionally breaks that identity.

And deliberately excludes some knobs that do not define the selected-runtime contract, such as:

- `data_dir`
- `seed`
- `num_threads`
- `buffer_games`
- `buffer_samples`

That is why a cache hit means “same relevant runtime-selection problem,” not “same exact whole config file.”

## Identical-run fast path

Hydra's current baseline supports an identical-run fast path:

- `run_preflight` and `run_rl_preflight` consult cache before running probes
- a matching cache hit on the current v4 key can skip all probes
- probe result vectors are empty on that cache-hit path because no new probe pass ran

This speeds up repeated runs on unchanged hardware/workload/settings without changing the authority rules above.

## Preflight knobs that matter most

`PreflightConfig` is large, but operators usually only need a smaller mental model.

### Candidate set and explicit override behavior

- `candidate_microbatches`
- `min_microbatch_size`
- `allow_override_explicit_microbatch`

These define the search ladder and whether Hydra is allowed to move away from an explicit requested microbatch in probe-driven flows.

### Probe stability and acceptance

- `warmup_steps`
- `measure_steps`
- `required_successes`
- `measure_noise_tolerance_ratio`

These control how noisy or brittle the runtime search is allowed to be.

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

These control whether the more expensive end-to-end finalist benchmark is allowed to refine the winner.

### Stage-2 validation cache

The stage-2 benchmark can reuse pre-materialized validation samples across finalists when two conditions hold:

- the validation sample limit is finite, so there is an actual bounded validation cache to materialize
- multiple finalists share the same loader-runtime tuple and resolved validation sample limit

When that happens, Hydra materializes the validation samples once, reuses them across those finalists, and records the one-time materialization cost separately. That materialization time is still charged into the benchmark accounting so operators do not misread the finalist as “free validation.”

This reuse only applies to the loose-replay validation cache path. Shard-backed validation does not use this in-memory cached-sample route.

When `bc_shards_manifest_path` is set, preflight behavior changes more broadly than just validation caching:

- BC train and validation probes load shard readers directly instead of using the replay-manifest scan/cache path for those probe kinds
- loader-runtime tuning collapses to the config-derived loader tuple instead of running the normal loader finalist search
- the stage-2 finalist benchmark is skipped entirely for shard-backed BC runs, even if `real_benchmark_enabled` is true

That means shard-backed preflight results are not directly comparable to loose-replay preflight results when you are reasoning about replay-scan throughput or stage-2 winner selection.

### Local refinement and coordinate search

- `local_refinement_enabled`
- `local_refinement_max_candidates`
- `local_refinement_min_gap`
- `search_coordinate_rounds`
- `search_top_k`

These let Hydra do a more local or coordinated search rather than accepting the first coarse-ranked result.

## Practical defaults

The defaults are intentionally conservative enough to work as a baseline search policy:

- a descending candidate ladder from large to small microbatches
- short warmup/measure phases for initial probes
- real benchmark enabled
- local refinement enabled
- a small number of coordinate rounds

If you are not actively investigating runtime behavior, the default recommendation is:

- keep the default preflight config
- only override preflight knobs when you have a concrete reason

## When to disable or narrow preflight behavior

You may want to reduce preflight cost when:

- you are running repeated experiments on identical hardware
- you already trust a narrow candidate range
- you are debugging a specific candidate microbatch or loader tuple

Examples of narrowing moves:

- reduce `candidate_microbatches`
- disable `real_benchmark_enabled`
- disable `local_refinement_enabled`
- lower `loader_runtime_rounds`

These should be treated as debugging/iteration tools, not as the default long-term baseline.

## Probe-only workflow

The probe-only CLI path exists for targeted measurement without running normal training. It takes:

- `--probe-kind <train|validation>`
- `--probe-candidate-microbatch <N>`
- optional warmup and measure step overrides

Use probe-only when you want to answer:

- can this candidate microbatch run at all?
- is validation behaving differently from training?
- did a precision/device/config change alter runtime headroom?

## RL-specific note

The preflight config also carries RL-oriented memory and growth-safety knobs, but the current shipped precision/runtime baseline is still BC-first. Treat RL preflight as a real surface, but not yet the most stable operator path compared with BC preflight.

## Failure modes to watch for

- Explicit microbatch settings may still be rejected when authority or safety rules say they must remain fixed.
- A cache hit does not mean loader-runtime became authoritative.
- A cache hit can intentionally produce no probe results because Hydra skipped probing.
- Precision-mode changes can invalidate assumptions even if the YAML looks mostly the same.
- Stage-2 benchmark throughput can include one-time validation materialization cost when Hydra decides that cache reuse is valid for a finalist group.

## Recommended operator workflow

1. Start with the config you actually want to train.
2. Run `train config.yaml --preflight` on new hardware or materially changed workloads.
3. Accept the selected-runtime result unless you are explicitly investigating runtime policy.
4. For repeated runs on the same setup, rely on the cache-hit fast path rather than re-tuning everything manually.
5. If resuming, distinguish epoch-boundary resume from partial-epoch resume before assuming cached runtime reuse is allowed.

## Where to read next

- Need the main training entrypoint and mode selection? Read [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md).
- Need replay-side supervision lanes that depend on training config? Read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).
- Need the compact authority table? Read [`docs/COMPATIBILITY_SURFACE.md`](COMPATIBILITY_SURFACE.md).
