# Hydra Infrastructure

Infra router + compact reference. Truth priority:

1. Live code + runtime docs: `docs/GAME_ENGINE.md`, `docs/COMPATIBILITY_SURFACE.md`, crate sources.
2. Current execution doctrine: `research/design/HYDRA_RECONCILIATION.md`, `research/design/HYDRA_FINAL.md`.
3. This file: infra rationale, artifact routing, preserved historical/reserve planning.
4. Benchmark ledger: `research/infrastructure/ENGINE_BENCHMARKS.md`.

If conflict: code/runtime wins. Later PPO/oracle/league sections here = reserve/historical unless promoted by reconciliation doc.

## Crate ownership

| Crate | Owns | Notes |
|---|---|---|
| `hydra-engine` | vendored/owned riichi game engine, MJAI replay/state transition, scoring/shanten bridge | No Mortal/AGPL code. Engine behavior truth in code + `docs/GAME_ENGINE.md`. |
| `hydra-core` | tile/action types, 46-action mapping, `192x34` fixed-superset encoder, safety features, `ObservationRef`, batch sim/search glue, seeding helpers | First 85 channels keep old public+safety baseline prefix; compatibility truth in `docs/COMPATIBILITY_SURFACE.md`. |
| `hydra-train` | Burn model, inference, BC/replay pipeline, target/sidecar builders, preflight/runtime selection, training binaries | Training workflow truth in `docs/TRAINING_RUNBOOK.md`. |

Core contracts:

- MJAI line JSON is log/bot compatibility format.
- Tile ids use standard 34-type mapping plus 136-format/red-five handling where runtime docs specify.
- Action space stays Mortal-compatible 46 actions for policy/mask compatibility, without copying Mortal impl.
- Encoder emits fixed-shape `192x34`; dynamic/search/belief features use zero-fill + presence masks when unavailable.
- Suit augmentation uses full 3! = 6 suit permutations; apply at event/tile layer, one permutation per game.

## Rust/Burn decision

Hydra uses 100% Rust for engine, encoding, sim, inference, and training. No Python/PyO3 training path.

Selected stack:

| Layer | Decision | Why |
|---|---|---|
| DL framework | Burn `0.21+` | Rust-native modules, autodiff, optimizer/scheduler, dataloading, records. |
| Production backend | `burn-tch` | libtorch/cuDNN/cuBLAS kernels; proven CUDA allocator/autograd. Same GPU kernels as PyTorch, no Python IPC/GIL. |
| Future backend | `burn-cuda` / CubeCL | JIT fusion/autotuning path. Adopt only after Hydra-shape forward/backward/grad/memory benchmarks pass. |
| Precision | bf16 on CUDA where available | Tensor-core speed, fp32 exponent range, no fp16 GradScaler fragility. |
| Norm | GroupNorm(32) | No running stats; safer across BC -> self-play distribution shift than BatchNorm. |

Burn capability mapping:

| Need | Burn/Rust route |
|---|---|
| model modules | `#[derive(Module)]`, Conv/Linear/GroupNorm/activation primitives |
| AdamW + clipping | `burn-optim` |
| LR schedules | Burn scheduler configs; compose warmup + cosine where needed |
| data loading | `burn-dataset` / `DataLoaderBuilder` + Rust loaders/rayon |
| records/checkpoints | Burn Record system; model/optimizer/scheduler records |
| inference | non-autodiff backend; direct Burn inference |
| W&B | REST via `reqwest` or tensorboard-rs; no Python SDK needed |

Migration rule: stay on `burn-tch` until `burn-cuda` beats/equal on Hydra active shapes and passes value/gradient comparison. Backend swap mid-run may change numerics; treat as deliberate run boundary, not silent checkpoint continuation.

## License boundary

Allowed deps/patterns: MIT, Apache-2.0, BSD-compatible. `xiangting`, `rayon`, `serde`, `serde_json`, `ndarray`, `rand`, Burn ecosystem fit this boundary.

Forbidden for Hydra impl:

| Source/license | Boundary |
|---|---|
| AGPL/GPL code | Do not copy, port line-by-line, link, or derive impl. |
| LGPL static-link ambiguity | Avoid unless relinking obligations explicitly handled. |
| Mortal code/policy restrictions | Reference behavior/bench ideas only; do not use code. |

Mortal/libriichi may be cited as external reference for action-space compatibility and high-level patterns. impl must remain original or permissively licensed.

## Runtime + artifact routing

Read these instead of old parallel docs:

| Question | Read |
|---|---|
| Engine/runtime semantics | `docs/GAME_ENGINE.md` |
| Compatibility surface | `docs/COMPATIBILITY_SURFACE.md` |
| Bench ledger / perf claims | `research/infrastructure/ENGINE_BENCHMARKS.md` |
| Training workflow / BC shards / sidecars / preflight | `docs/TRAINING_RUNBOOK.md` |
| RNG doctrine | `research/design/SEEDING.md` |
| Active roadmap | `research/design/HYDRA_RECONCILIATION.md` |

Artifact owners:

| Artifact | Owner | Notes |
|---|---|---|
| raw MJAI `.json/.json.gz` | data source | Keep raw logs as source of truth. |
| BC shards | `hydra-train` builders/docs | Pre-encoded production path; rebuild when encoder/features change. |
| replay sidecars | `hydra-train` sidecar builders/docs | Exit/delta-Q sidecar docs own schema/commands. |
| model records | Burn Record files | Training/inference native format. |
| run dirs | training binaries | One run id should encode timestamp + master seed when practical. |
| eval outputs | eval binaries/docs | Append/audit; do not overwrite claims silently. |

Data doctrine:

- Default BC path: on-the-fly Rust parsing can work at current scale; pre-encoded shards are production throughput artifact, not source of truth.
- Avoid HDF5/Parquet/WebDataset for dense game tensors unless live evidence overturns current path.
- Filtering should be manifest-based: scan once, train from filtered list; do not waste per-epoch CPU on repeated quality filtering.
- Three-level shuffle remains useful: file order, in-worker buffer shuffle, reserve mixing.

## Checkpoint essentials

Checkpoint must tell truth on resume. No plausible-success fallback.

Record contents:

| Key | Required content |
|---|---|
| model | Burn `Module::record()` for active model weights; dtype preserved. |
| optimizer | AdamW/optimizer record where training can resume in same phase. |
| scheduler | LR scheduler position/state. |
| rng | backend/system/data/sim RNG state or deterministic seed lineage sufficient to resume intended stream. |
| counters | global step, epoch/shard/cursor where relevant. |
| phase/config | full config + phase/schema/checkpoint version. |
| metrics | best/current metric snapshot used for selection. |
| timestamp | UTC timestamp. |

Dtype rule: model weights may be bf16; AdamW moment buffers stay fp32. Do not cast optimizer state to bf16.

Atomic save protocol:

1. Serialize complete record to memory/buffer before touching final path.
2. Compute SHA-256 digest of exact bytes.
3. Write `{target}.tmp` on same filesystem.
4. Flush file buffers.
5. `fsync` file.
6. Atomic rename tmp -> final.
7. Write `.sha256` sidecar in GNU `sha256sum` format.
8. `fsync` parent dir.

Load policy:

| Condition | Behavior |
|---|---|
| sidecar present + digest matches | Load. |
| sidecar present + mismatch | Hard fail before deserialize. |
| sidecar missing | Warn; may load for backward compatibility / crash-after-rename case. |
| latest corrupt | May try previous retained checkpoint for training resume; report every attempted path. |
| gate checkpoint corrupt | Hard fail; no silent fallback. |

Retention reference:

- Training checkpoints: FIFO-prune non-protected files; keep latest/best/gates.
- Gate artifacts (`bc_best`, future `distill_best`) must be independent copies, not symlinks into pruned dirs.
- Symlinks like `latest`/`best` must update only after verified save completes.
- Future opponent-pool models, if revived, are inference-only stripped records plus metadata sidecars; anchors are never evicted.

Stage transition rule: reset optimizer/scheduler at major objective changes unless active doctrine says otherwise. Keep global step for logging continuity. Re-seed from documented stage child seed; do not reuse accidental RNG stream.

## Compute-budget doctrine

Current planning target: about 2,000 Delta GPU A100-hours on `gpuA100x4` with 1 shared A100 reservation.

Delta assumptions:

- `gpuA100x4` = shared quad-A100 node shape.
- Current Hydra target = 1 shared A100, not exclusive full node.
- Accounting: 1 SU corresponds to 1 A100, 16 reserved CPU cores, or 62.5 GB reserved host memory for 1 hour; charge by largest reserved fraction.
- Delta GB means `1e9` bytes, not `2^30`.
- Older RTX 5000/Frontera normalization is historical only.

Doctrine:

- 2,000 A100-hours buys one disciplined Hydra v1 push, not open-ended LuckyJ-scale search.
- Maximize BC + offline/replay reuse before expensive online RL.
- Use proven configs; avoid broad hyperparameter sweeps.
- Treat strong amateur/low-dan or ~70-80% expert agreement as plausible target; 10+ dan / stable LuckyJ parity is not credible on this budget alone.
- If results justify more, apply for more compute after evidence, not before.

Triangulation:

| System | Public compute signal | Lesson |
|---|---|---|
| Suphx | ~2,112 old-GPU hours per RL agent; 44 GPUs x 48h; 1.5M self-play games | 10-dan possible with strong method + large old cluster, but not comparable one-to-one. |
| LuckyJ/JueJong | no full public budget; public components use V100s + thousands CPUs | Stable 10+ dan likely far beyond 2,000 A100-hours. |
| Mortal | no total public budget; 4090+7950X throughput reported; likely hundreds-low-thousands GPU-hours | Strong play possible with efficient offline/RL/data reuse. AGPL reference only. |
| LsAc*-MJ | 51.4h small low-resource model | Low-resource methods help, but result not Suphx/Mortal-class. |

Budget sketch remains planning, not promise:

| Work | GPU-hours | Role |
|---|---:|---|
| BC / supervised launch | 250-600 | Cheapest competence. |
| Offline/replay RL or target reuse | 800-1,300 | Main efficiency lever. |
| Limited online self-play/search calibration | 400-1,000 | Spend only after BC/replay signal. |
| Eval/debug/ablation buffer | 130-270 | Keep claims honest. |

## Performance doctrine

Benchmarks live in `research/infrastructure/ENGINE_BENCHMARKS.md`; do not fork numbers here.

Rules:

- Optimize measured hot paths only. Benchmark maintenance is part of optimization.
- Prefer zero-copy/borrowed views (`ObservationRef`) and caller-owned buffers over caching partial encodings.
- Full recompute encoding is safer than incremental feature-cache tricks unless current code/tests/benches prove otherwise; silent feature drift corrupts training.
- Cache expensive NN/search outputs by state hash when search needs it; do not cache cheap encoded tensors by default.
- Release profile/LTO/codegen settings are legitimate infra, but claims must be benchmark-backed.
- Avoid repeating known negative experiments without new evidence: stack-array `ct_smc` rewrite regressed; approximate `TargetPresence` preservation broke contracts; `ObservationRef` wins by removing allocations/copies, not by guaranteed encoder microbench speedup.

Current perf routing:

- Engine/sim microbench truth: `ENGINE_BENCHMARKS.md` + bench files.
- End-to-end training bottlenecks: training benches and preflight/runtime docs.
- If local microbench conflicts with end-to-end throughput, investigate; do not extrapolate.

## Hardware minima/reference

| Use | Minimum | Recommended current target |
|---|---|---|
| Training | CUDA GPU with enough VRAM for active batch/model; 8 CPU cores; 32 GB RAM | 1 shared A100 40 GB on Delta `gpuA100x4`, 16 reserved cores, 62.5 GB host memory. |
| Inference | RTX 3060-class 6 GB, 4 CPU cores, 8 GB RAM | RTX 4070-class 12 GB, 8 CPU cores, 16 GB RAM. |

Model footprint is small relative to A100 memory; GPU compute and data/validation path dominate, not capacity.

## Historical/reserve notes

Old docs had detailed Phase 2/3 PPO, oracle distillation, league opponent-pool, multi-stream GPU, and OpenSkill plans. Preserve only as reserve unless re-promoted:

- Oracle teacher + student distillation: teacher frozen bf16/eval; student gets fresh AdamW; KL anchor can protect BC behavior.
- League self-play: Rust rayon game workers + batched Burn inference; opponent pool with frozen anchors; duplicate 1v3 seat rotation for eval.
- Evaluation: 1v3 duplicate format, seat-rotated seeds, conservative statistics; full claims need large game counts.
- No DDP/FSDP needed for active single-GPU plan; model/batch fit easily. Multi-node only if future throughput evidence demands it.

Do not implement reserve machinery merely because it exists here. Active tranche comes from reconciliation/runtime docs and code.