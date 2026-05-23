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
- BC shard/raw-MJAI training default is Python/PyTorch through Rust launcher. Rust owns replay parsing, shard building, manifest validation, launcher glue, and Python config conversion (`hydra-train-runtime::config::python`). Python owner is `hydra_learner.cli`; train loop/modules live under `hydra_learner`; `train_bc.py` is compatibility entrypoint only. Python owns BC model/loss/optimizer/BF16/`torch.compile`/checkpoint-resume. Plain BC supports full base heads/losses plus default-off `safety_residual` and `oracle_critic` labels when present. Python run UX writes balanced JSONL logs, TensorBoard event files, periodic `latest.pt` checkpoints, optional step checkpoints, background pid/stdout/stderr logs, and metadata-validated shard checkpoints.
- Stronger public-teacher belief-semantics tranche shipped in current training baseline.
- Current Hand-EV realism upgrade shipped in live baseline surface.
- Replay-derived `safety_residual` shipped as narrow supervised lane.
- ExIt has end-to-end carrier across live self-play lane and replay/sample sidecar-first lane.
- Rare-action train/validation metrics shipped as observability only; no policy behavior change.

- BC shards are compact-only v3 on disk. No dense/v2 storage path is supported. Workflow: build shards, then train/resume via Python BC learner from `--bc-shards-manifest` or YAML `bc_shards_manifest_path`; or omit shard manifest and stream raw MJAI from YAML `raw_mjai_data_dirs` (explicit multiple roots) or fallback `data_dir` into fresh output dir. Shard builder writes replay-fact baseline observation records, omits advanced/search/Hand-EV dense tails, and dense v2 shards hard-error with rebuild-from-replay message. Python reader expands compact records to unchanged `192x34` training tensors and full base labels.
- Legacy Rust/Burn BC shard CUDA path remains explicit feature-gated fallback/debug path (`bc_backend: rust_burn` / `--bc-backend rust-burn`). It has reusable pinned H2D staging, preallocated GPU tensors, CPU f32 policy-target materialization, and child-process CUDA graph compute-capture probe. Production CUDA graph replay remains blocked by Burn `GradientsParams` optimizer contract; runtime labels say `cuda_graph_replay=production_off_probe_only`.
- Python BC CUDA uses `py-train` env with PyTorch `2.11.0+cu128` plus TensorBoard. Raw-MJAI input uses `hydra-raw-mjai-pyo3` pinned PyO3 transport by default with stdout fallback; pinned PyO3 supports loose replay files and tar/tar.zst archives. Raw-MJAI resume fails closed until stream cursor resume exists; shard resume is supported. ExIt/DeltaQ/belief/mixture/opponent-hand-type are not supported in Python default yet; use legacy Rust/Burn path for those advanced modes or debugging.

### Implemented but not default-on

- Narrow DeltaQ supervision lane impl in code, promotion-gated through arena-confirmation path.
- DeltaQ promotion artifacts now persist explicit `arena_decision` plus `arena_report`, but lane still **not** default-on.
- `validation_gates` config exists for experiments; disabled by default and gates best-checkpoint promotion, not resume checkpoints.

### Experimental / parked

- `burn-cuda-probe` is explicit feature-gated, operator-selected BC-shard FP32 probe only. It never runs unless built with feature `burn-cuda-probe` and selected with `--experimental-backend burn-cuda`. LibTorch remains default/production backend. Not current throughput lane; no more Burn native CUDA throughput work.
- `experimental_backbone_profile` exists for BC throughput ablation only. It is YAML/benchmark-CLI gated, default absent, and preserves `192x34` input plus all BC full/policy head tensor contracts. Park as research infra only unless explicitly built/used for throughput experiments; do not treat as throughput fix or strength claim.
- Python `compile_max_autotune` is canonical for production Python BC. It does not change model math/topology, checkpoint architecture, input/action shapes, residual profile, or losses versus `compile_default`; it only changes TorchInductor compile strategy. Use `compile_default` only for smoke/preflight/short debug when compile/autotune overhead dominates.
- Python BC logging/checkpoint UX is shipped: `logs/events.jsonl`, `logs/train_steps.jsonl`, `checkpoints/latest.pt`, optional `checkpoints/step_<global_step>.pt`, TensorBoard event files, auto TensorBoard launch with upward port scan, and `background` detached mode. `full_epoch: true` plus `max_train_steps: null` is Python raw-MJAI full-train-split run length; explicit `max_train_steps` is bounded probe/ablation length. `num_epochs` remains Rust/Burn full-loop authority.

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
| BC runtime authority | shipped baseline | Normal training is YAML-derived. Benchmark rows are evidence for human YAML edit; normal BC/RL bootstrap does not consume benchmark artifacts. |
| BF16/AMP precision | shipped BC CUDA default; hard-gated RL/DeltaQ | Omitted `precision_mode` on BC CUDA LibTorch resolves to requested `bf16_autocast` / effective `bf16_amp`; explicit `fp32` overrides. CPU omission stays FP32. Loss/backward/optimizer/checkpoints/validation remain FP32. No CUDA graph BF16 claim. |
| Preflight benchmark | shipped baseline | `--preflight` runs exact `--pf-candidate-tuples` and emits markdown benchmark table with numeric throughput and wait ratios. It has no config input, dataset read, shard manifest read, cache authority, automatic winner, or YAML mutation. Non-applicable disk/GPU-only metrics are numeric `0.0`. |
| NVTX profiling | shipped baseline | Orchestration-level fully instrumented (epoch, step, validation, checkpoint, logging, self-play, stage-2 benchmark). BC microbatch sub-stages (collation, forward, loss, backward, optimizer_step) instrumented. Library internals not yet instrumented. Gated by `HYDRA_NVTX` env var via dlopen. |
| CUDA BC shard throughput | shipped baseline (transport/metrics); probe-only (CUDA graph replay) | `cuda-graph` feature enables pinned staging/preallocated tensors for shard train/probe/validation. CPU f32 policy-target path avoids lazy `IntTensor::one_hot`; metric accumulation avoids discarded progress finalization and redundant agreement kernels. Child graph probe proves compute-only capture/replay parity, but production replay is blocked by Burn optimizer gradient extraction. |
| `safety_residual` | shipped baseline | Narrow replay-derived supervised lane |
| ExIt carrier | shipped baseline | Live self-play lane + replay/sample sidecar-first lane; replay sidecar missing keys remain absent, but present contract mismatches hard-error instead of silently disabling hydration. |
| DeltaQ lane | implemented but not default-on | Arena-confirmation path impl; promotion artifact now records pre-arena rec plus final `arena_decision`/`arena_report` |
| `mixture_weight` activation | implemented but staged | Surface exists, promotion remains deferred |
| `opponent_hand_type` activation | implemented but staged | Surface exists, target closure remains incomplete |
| AFBS broad default runtime | reserve shelf | Specialist / hard-state gated direction only |

## Where to read next

- Need current runtime contract? Read `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md`.
- Need roadmap to Hydra v1 or active-path / staged-vs-reserve decision? Read `research/design/HYDRA_RECONCILIATION.md`.
- Need north-star architecture, not current shipped status? Read `research/design/HYDRA_FINAL.md`.