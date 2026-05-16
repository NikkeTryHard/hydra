# Hydra Current Status

Current shipped/staged snapshot for Hydra built surfaces.

Use file for: what shipped today, what impl but staged, what impl but not default-on.

File reports shipped/staged status only.

- For roadmap to Hydra v1, read `research/design/HYDRA_RECONCILIATION.md`.
- For runtime semantics and compatibility truth, read `docs/GAME_ENGINE.md`, `docs/COMPATIBILITY_SURFACE.md`, and current code.

If file and current code disagree, current code wins. If file and `HYDRA_RECONCILIATION.md` disagree on active vs reserve vs staged priority, refresh reconciliation, then refresh file. If reconciliation or current status drift from archive root, refresh promoted docs, not demote canonical archive source ledger.

## Status vocabulary

File uses status vocabulary from `research/design/HYDRA_RECONCILIATION.md`.

| Term | Meaning |
|---|---|
| `shipped baseline` | impl, part of current live baseline |
| `implemented but not default-on` | impl, validated enough to exist in-code, intentionally not default runtime/training path |
| `implemented but staged` | core code path exists, promotion/activation intentionally deferred |
| `reserve shelf` | documented later-work direction, not current mainline priority |
| `historical` | preserved context only; not current governing truth |

## Runtime and training snapshot

### Shipped baseline

- `hydra-core` = real first-party runtime/encoder/simulator crate.
- Live encoder/model contract = `192x34`; old `85x34` view = baseline-prefix only.
- Fixed runtime action space = 46 actions with two-phase riichi and kan handling.
- BC training supports **epoch-boundary-only** reuse of matching preflight-selected runtime for selected-runtime tuple (`train_microbatch_size`, `validation_microbatch_size`, derived `accum_steps`); fresh runs stay config-derived, partial-epoch resumes still require identical runtime, loader-runtime stays config-derived.
- Stronger public-teacher belief-semantics tranche shipped in current training baseline.
- Current Hand-EV realism upgrade shipped in live baseline surface.
- Replay-derived `safety_residual` shipped as narrow supervised lane.
- ExIt has end-to-end carrier across live self-play lane and replay/sample sidecar-first lane.
- Rare-action train/validation metrics shipped as observability only; no policy behavior change.

- BC shards are compact-only v3 on disk. No dense/v2 storage path is supported. Shard builder writes replay-fact baseline observation records, omits advanced/search/Hand-EV dense tails, and dense v2 shards hard-error with rebuild-from-replay message. Reader expands compact records back to unchanged `192x34` training tensors with advanced channels absent/zero for replay BC shards.
- BC shard CUDA path has reusable pinned H2D staging, preallocated GPU tensors, CPU f32 policy-target materialization, and child-process CUDA graph compute-capture probe. Production CUDA graph replay remains blocked by Burn `GradientsParams` optimizer contract; runtime labels say `cuda_graph_replay=production_off_probe_only`.
- BC CUDA LibTorch training defaults to BF16 AMP when `precision_mode` is omitted. Explicit `precision_mode: fp32` keeps CUDA BC FP32; CPU omission stays FP32. BF16 AMP wraps BC forward only; loss/backward/optimizer/checkpoints/validation remain FP32. RL and DeltaQ promotion hard-error on BF16.
### Implemented but not default-on

- Narrow DeltaQ supervision lane impl in code, promotion-gated through arena-confirmation path.
- DeltaQ promotion artifacts now persist explicit `arena_decision` plus `arena_report`, but lane still **not** default-on.
- `validation_gates` config exists for experiments; disabled by default and gates best-checkpoint promotion, not resume checkpoints.

### Implemented but staged

- `mixture_weight` promotion remains staged.
- Richer opponent-target closure remains staged.
- Representative-world / per-particle CT-SMC Hand-EV remains staged.
- Selective AFBS / endgame deepening remains staged.

### Reserve shelf

- Broader public-belief search as project identity remains reserve-shelf, not active-path.
- Deeper robust-opponent search backups remain reserve-shelf.
- Larger latent-opponent / richer auxiliary-head expansion remains reserve-shelf until existing target closure improves.

## Area-by-area summary

| Area | Current status | Notes |
|---|---|---|
| Runtime encoder / action semantics | shipped baseline | See `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md` |
| Hand-EV baseline surface | shipped baseline | Stronger local evaluator live; representative-world CT-SMC Hand-EV still staged |
| Belief semantics baseline | shipped baseline | Stronger public-teacher belief tranche in live baseline |
| BC runtime authority | shipped baseline | Fresh runs config-derived; epoch-boundary resumes may reuse matching preflight-selected runtime for selected-runtime only; partial-epoch resumes still require identical runtime; loader-runtime remains config-derived |
| BF16/AMP precision | shipped BC CUDA default; hard-gated RL/DeltaQ | Omitted `precision_mode` on BC CUDA LibTorch resolves to requested `bf16_autocast` / effective `bf16_amp`; explicit `fp32` overrides. CPU omission stays FP32. Loss/backward/optimizer/checkpoints/validation remain FP32. No CUDA graph BF16 claim. |
| Preflight cache system | shipped baseline | Fingerprint v4 key covers hardware, workload, preflight config, explicit microbatch overrides. Identical-run fast path skips probing on cache hit. BC and RL bootstrap read cache under documented authority rules. |
| NVTX profiling | shipped baseline | Orchestration-level fully instrumented (epoch, step, validation, checkpoint, logging, self-play, stage-2 benchmark). BC microbatch sub-stages (collation, forward, loss, backward, optimizer_step) instrumented. Library internals not yet instrumented. Gated by `HYDRA_NVTX` env var via dlopen. |
| CUDA BC shard throughput | shipped baseline (transport/metrics); probe-only (CUDA graph replay) | `cuda-graph` feature enables pinned staging/preallocated tensors for shard train/probe/validation. CPU f32 policy-target path avoids lazy `IntTensor::one_hot`; metric accumulation avoids discarded progress finalization and redundant agreement kernels. Child graph probe proves compute-only capture/replay parity, but production replay is blocked by Burn optimizer gradient extraction. |
| `safety_residual` | shipped baseline | Narrow replay-derived supervised lane |
| ExIt carrier | shipped baseline | Live self-play lane + replay/sample sidecar-first lane |
| DeltaQ lane | implemented but not default-on | Arena-confirmation path impl; promotion artifact now records pre-arena rec plus final `arena_decision`/`arena_report` |
| `mixture_weight` activation | implemented but staged | Surface exists, promotion remains deferred |
| `opponent_hand_type` activation | implemented but staged | Surface exists, target closure remains incomplete |
| AFBS broad default runtime | reserve shelf | Specialist / hard-state gated direction only |

## Where to read next

- Need current runtime contract? Read `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md`.
- Need roadmap to Hydra v1 or active-path / staged-vs-reserve decision? Read `research/design/HYDRA_RECONCILIATION.md`.
- Need north-star architecture, not current shipped status? Read `research/design/HYDRA_FINAL.md`.