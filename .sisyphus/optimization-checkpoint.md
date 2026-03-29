# Optimization checkpoint

Date: 2026-03-29

## Branch state

- Workspace is green on broad `cargo check`, `clippy -D warnings`, all tests, and integration
- Zero compiler warnings, zero `#[allow(dead_code)]` annotations
- 1128 crate tests (526 train-bin), 9 integration tests, all green
- `hydra-train` benchmark harness (`crates/hydra-train/benches/train_hotpaths_bench.rs`) compiles and passes

## Major landed slices (speed-profiling branch)

### Performance (same-logic speedups)
1. Fixed-shape clone reduction in BC hot path (`bc_fixed_shape.rs`)
2. Stage-two preflight benchmark validation-cache reuse
3. Final-cache-driven loader shortlist construction (`runtime_autotune.rs`)
4. Mixed fixed-shape prefix/tail execution for non-divisible logical batches
5. Routed probe-batch fan-out collapse across repeated attempts
6. Exact-key loader-autotune seed for current tuple
7. Count-aware runtime refine top-up for close tuples
8. Preflight-cache fingerprint widening (v4 key: preflight config signature + explicit microbatch overrides)
9. Preflight identical-run fast path (`run_preflight` + `run_rl_preflight` skip probing on cache hit)

### Infrastructure / quality
10. cfg(test) direct ProbeResult seam
11. BC runtime-authority reconciliation to documented contract
12. BF16/AMP precision dispatch (BC training, preflight, probe, autotune, stage-2 benchmark)
13. NVTX sub-stage markers for BC train microbatch (collation, forward, loss, backward, optimizer_step)
14. NVTX sub-stage markers extended to fixed-shape path (both prefix loop and tail-remainder)
15. NVTX scope order verification test

### Test improvements
- 14 epoch_runner tests switched from full learner to tiny model (~17% wall-time reduction)
- FP32/BF16 probe test split for parallel execution (critical path: ~130s -> ~88s)
- bc_fixed_shape edge case tests: split_divisible_prefix, empty batch, zero microbatch, microbatch > batch
- ProfilingEnvelope from_children and merge_assign coverage
- Preflight cache benchmark preservation test

### Docs
- CURRENT_STATUS.md: BF16/AMP, preflight cache, NVTX profiling
- COMPATIBILITY_SURFACE.md: preflight cache key v4 contract, precision mode dispatch

## Bench snapshot

### hydra-core
- `single_game_first_action`: ~418 us
- `single_game_first_action_reuse`: ~472 us
- `batch_100_games`: ~4.05 ms
- `encode_observation`: ~3.05 ms
- `encode_observation_ref`: ~3.05 ms
- `ct_smc_dp_128_samples`: ~2.12 ms
- `agari calc_4p`: ~705 us
- `agari calc_3p`: ~290 us

### hydra-train
- `loader/load_game_from_reader`: ~10.7-11.1 ms
- `validation/collate_only`: ~33-36 us
- `validation/forward_loss_only`: ~1.33-1.40 s
- `validation/collate_forward_loss`: ~0.55-1.10 s
- `selfplay_batch/trajectories_to_rl_batch`: ~30-31 us
- `selfplay_batch/trajectories_to_rl_batch_reuse`: ~30.6 us
- `model_cpu_bridge/policy_value_cpu`: ~0.58-0.73 s
- `model_cpu_bridge/policy_cpu`: ~0.93-0.99 s
- `model_cpu_bridge/value_cpu`: ~0.95-1.00 s

## Reverted / rejected ideas

- `ct_smc` stack-array rewrite regressed benchmark; reverted
- Approximate `TargetPresence` propagation through sliced targets was incorrect; reverted
- `live_exit` hard-state helper rewrite regressed labels; reverted
- cfg(test) preloaded-config probe execution lane: reverted (stopwatch data was bad despite clean code)
- CPU-bridge copy-elision experiment: explicitly rejected

## Current interpretation

- Same-logic speedups at the orchestration level are largely harvested
- BC BF16 is complete; RL BF16 is intentionally gated (not half-done)
- NVTX profiling covers orchestration + BC sub-stages; library internals need module refactor for deeper instrumentation
- Preflight cache with v4 key provides genuine identical-run fast path
- Fixed-shape path now has full parity with fallback path (NVTX scopes, edge case tests, exit loss)
- Test suite is well-covered with 1128 tests and clean CI surface
- Next worthwhile work would be either:
  1. Move NVTX module to library crate for model/loss/selfplay instrumentation (structural change)
  2. BF16 RL training enablement (new capability, separate branch)
  3. Deeper loader/data pipeline profiling (needs epoch loop restructuring)
