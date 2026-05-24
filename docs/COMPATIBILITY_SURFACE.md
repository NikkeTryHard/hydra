# Hydra Compatibility Surface

Compat contract for agents/devs touching runtime, training, model-shape-sensitive code.

Change any row here -> review matching docs, tests, consumers.

Primary runtime owner: `docs/GAME_ENGINE.md`

## Compatibility table

| Surface | Current contract | Owner / source of truth | Notes |
|---|---|---|---|
| Encoder/model input shape | `192x34` | `docs/GAME_ENGINE.md`, `hydra-core/src/encoder.rs` | Live full contract |
| Baseline prefix | `85x34` (`channels 0..84`) | `docs/GAME_ENGINE.md` | Historical baseline-prefix only; not full live encoder |
| Action space | `46` actions | `hydra-core/src/action.rs`, `docs/GAME_ENGINE.md` | 4-player facade; `ActionType::Kita`/sanma is engine-only, not represented in Hydra compact actions |
| Riichi handling | two-phase | `hydra-core/src/action.rs`, `docs/GAME_ENGINE.md` | Declare riichi, then choose discard |
| Kan handling | compact bridge | `hydra-core/src/action.rs`, `docs/GAME_ENGINE.md` | Action `42`; bridge maps normal phase to `Ankan`, other phases to `Daiminkan`; inbound kan variants collapse to `42` |
| Tile kind indices | `0..33` normalized tile indices | `hydra-core/src/tile.rs`, `docs/GAME_ENGINE.md` | 34 tile kinds |
| Aka tile behavior | aka tiles stay distinct in 136-format/action handling where needed | `hydra-core/src/tile.rs`, `hydra-core/src/action.rs` | Red 5m/5p/5s stay special cases |
| Legal action mask shape | `[bool; 46]` | `hydra-core/src/action.rs` | Training/inference must match mask semantics |
| Game modes | core 4p simulator modes `0` hanchan, `1` east, `2` single; engine sanma modes `3` single, `4` east, `5` half | `hydra-core/src/simulator.rs`, `hydra-engine/src/game_variant.rs` | Hydra 46-action bridge remains 4p; sanma/Kita stays engine surface |
| BC shard storage | compact-only v3; old dense v2 hard-errors and must be rebuilt from replay; no alternate shard format is supported | `crates/hydra-bc-shards/src/manifest.rs`, `crates/hydra-bc-shards/src/reader.rs` | Training API remains `[batch, 192*34]` f32 obs + `[batch, 46]` legal mask. Shards store replay-fact baseline obs only; advanced/search/Hand-EV obs tail is absent/zero, not feature-gated. |
| Runtime/train entrypoint | `crates/hydra-train/src/bin/train.rs` parses/env-dispatches/delegates through `hydra-train-runtime` + `hydra-train-exec`; `hydra-train` is package/binary entrypoint glue only | root `AGENTS.md`, `crates/hydra-train-runtime/src/lib.rs`, `crates/hydra-train-exec/src/lib.rs`, crate manifests | New CLI/config/preflight/probe/status contract code belongs in `hydra-train-runtime`; Python config conversion belongs in `hydra-train-runtime::config::python`; execution composition belongs in `hydra-train-exec`; keep train-bin code as entrypoint glue only |
| BC runtime authority | YAML-derived only for normal training; epoch-boundary resume follows checkpoint/resume runtime contract | `crates/hydra-train-runtime/src/config_runtime.rs`, `crates/hydra-train-exec/src/bootstrap.rs`, `crates/hydra-train-exec/src/resume.rs` | Benchmark rows are evidence only; operators edit YAML fields by hand when accepting measured knobs. |
| BC loader authority | YAML-derived only | `crates/hydra-train-runtime/src/config_runtime.rs`, `crates/hydra-train-exec/src/bootstrap.rs` | Loader knobs (`num_threads`, buffers, archive queue, shard prefetch/ring tuple) are not inferred by normal training. |
| Python BC run UX | `output_dir` owns `logs/events.jsonl`, `logs/train_steps.jsonl`, `checkpoints/latest.pt`, optional `checkpoints/step_<global_step>.pt`, TensorBoard event files, `python_learner_result.json`, and background `train.pid`; TensorBoard port scans upward from configured port | `crates/hydra-train-runtime/src/config/python.rs`, `crates/hydra-train-exec/src/python_learner.rs`, `python/hydra_learner/hydra_learner/cli.py`, `docs/TRAINING_RUNBOOK.md` | Python owner is `hydra_learner.cli`; train loop/modules live under `hydra_learner`; `train_bc.py` is compatibility entrypoint only. Python BC `max_train_steps` is run length today; `num_epochs` remains Rust/Burn full-loop authority. Resume validates model/runtime/optimizer/loss/source/RNG metadata; raw-MJAI resume skips completed deterministic game prefix before streaming. |
| Preflight benchmark contract | exact tuple input; markdown table output; no config, YAML, dataset, shard manifest, cache, winner, or YAML mutation | `crates/hydra-train-runtime/src/preflight.rs`, `crates/hydra-train-runtime/src/config.rs`, `crates/hydra-train-exec/src/preflight_runtime.rs`, `crates/hydra-train-exec/src/presentation.rs` | `--pf-candidate-tuples <batch:ring:threads:prefetch,...>` emits rows with numeric throughput/wait ratios. Non-applicable disk/GPU-only metrics are numeric `0.0`; current codec is `none`; compression remains gated by measurements. |
| Shard workflow authority | build shards -> optional manifestless markdown preflight for runtime-shape evidence -> human edits YAML if desired -> train from `bc_shards_manifest_path` when explicitly requested | `docs/TRAINING_RUNBOOK.md`, `crates/hydra-train/src/bin/build_bc_shards.rs`, `crates/hydra-train/src/bin/train.rs` | Manifest is training dataset contract; YAML is training/runtime contract. Preflight reads neither. Raw-MJAI is default; resume skips deterministic completed games from checkpoint progress. |
| Replay sidecar hydration | missing replay/action keys are absent labels; present records with source/version/legal-mask/schema/shape/provenance mismatch hard-error | `crates/hydra-replay-sidecar/src/error.rs`, `crates/hydra-replay-sidecar/src/exit.rs`, `crates/hydra-replay-sidecar/src/delta_q.rs`, `docs/TRAINING_RUNBOOK.md` | Valid JSONL alone is not enough; sidecars are bound to replay identity, checkpoint hash/version, legal-mask digest, and action contract. |
| Precision mode dispatch (BF16/AMP) | Omitted `precision_mode` resolves by mode/device: BC CUDA LibTorch defaults to requested `bf16_autocast` / effective `bf16_amp`; explicit `fp32` overrides; CPU omission stays FP32; RL training and DeltaQ promotion hard-error on BF16 | `crates/hydra-train-runtime/src/config.rs`, `crates/hydra-train-runtime/src/config_runtime.rs`, `crates/hydra-train-exec/src/preflight_runtime.rs`, `crates/hydra-train-exec/src/modes.rs`, `crates/hydra-model/src/amp.rs` | BF16 AMP wraps BC forward only. Loss/backward/optimizer/checkpoint/validation remain FP32. No CUDA graph BF16 claim. Proof-only `bf16-autocast-proof` diagnostics are not production plumbing. |
| Experimental Burn-CUDA backend | Parked probe only: build feature `burn-cuda-probe` plus `--experimental-backend burn-cuda`; BC shards only; FP32 only; no BF16, CUDA graph, LibTorch tensor/optimizer mixing, or optimizer-state sharing | `crates/hydra-train/Cargo.toml`, `crates/hydra-train-exec/src/modes.rs`, `third_party/burn/README.hydra.md` | Feature-gated legacy/debug/advanced path only; not active/default throughput lane. |
| Runtime truth on drift | current code wins | `docs/GAME_ENGINE.md`, root `AGENTS.md` | Docs aid compat, not stronger than code |

## Crate ownership quick reference

| Crate | Owns |
|---|---|
| `crates/hydra-engine` | vendored riichi rules engine |
| `crates/hydra-runtime-types` | shared runtime rails/types |
| `crates/hydra-safety` | safety rails |
| `crates/hydra-belief-search` | belief/search primitives |
| `crates/hydra-encoder` | observation encoder components |
| `crates/hydra-core` | public runtime bridge, simulator, action/tile API, seeding |
| `crates/hydra-data-core` | sample DTOs/scoring helpers |
| `crates/hydra-replay-sidecar` | replay sidecar JSONL contracts |
| `crates/hydra-replay-loader` | MJAI replay loading/sample conversion |
| `crates/hydra-sample-cache` | parsed-sample cache format |
| `crates/hydra-bc-shards` | backend-agnostic BC shard host format |
| `crates/hydra-train-types` | training scalar coordination types |
| `crates/hydra-model` | Burn neural model components |
| `crates/hydra-train-algo` | pure Burn training algorithms/loss math |
| `crates/hydra-selfplay` | self-play coordination primitives |
| `crates/hydra-search-labels` | search-label generation |
| `crates/hydra-train-runtime` | train CLI/config/preflight/probe/status contracts; Python config conversion in `config::python` |
| `crates/hydra-train-exec` | training execution composition over runtime/model/algo/data crates |
| `crates/hydra-train` | user-facing training binaries/package marker; bin code is glue |

## Read next

- Need full runtime explanation? Read `docs/GAME_ENGINE.md`.
- Need repo routing / trust map? Read `README.md`.
- Need active-vs-staged status? Read `research/design/HYDRA_RECONCILIATION.md` and `docs/CURRENT_STATUS.md`.