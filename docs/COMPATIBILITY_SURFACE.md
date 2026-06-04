# Hydra Compatibility Surface

Current code wins if this file drifts. Primary runtime owner: [`GAME_ENGINE.md`](GAME_ENGINE.md).

This page records hard contracts for code that touches runtime, training, model-shape-sensitive paths, replay, checkpoints, or rollout.

## Runtime Shapes

- Live encoder/model input is `192x34`.
- Historical `85x34` means baseline-prefix channels `0..84`, not full live input.
- Action space is fixed at `46`.
- Legal mask shape is `[bool; 46]`.
- Tile kind indices are `0..33`.
- Aka/red fives stay distinct on 136-format/action surfaces where needed.
- Suit augmentation has exactly 6 numbered-suit permutations; honors unchanged.

## Action Semantics

- Riichi is two-phase: declare riichi, then choose discard.
- Compact kan bridge uses action `42`.
- Normal phase maps `42` to `Ankan`; other phases map it to `Daiminkan`.
- Inbound kan variants collapse to `42`.
- Compact Hydra action facade is 4-player. Sanma/Kita stays engine-level, not 46-action bridge.
- Open-kan dora event order follows Tenhou/Mahjong Soul/MJAI: dora before `dahai`. Legacy after-discard order exists only behind `open_kan_dora_after_discard = true`.

## Training Data

- BC shards are compact-only v3.
- Dense/v2 shard formats hard-error and must be rebuilt from replay.
- Training API remains `[batch, 192*34]` f32 obs plus `[batch, 46]` legal mask.
- Shards store replay-fact baseline obs. Advanced/search/Hand-EV tail is absent/zero, not feature-gated.
- Replay sidecars fail closed on source, version, legal-mask, schema, shape, or provenance mismatch.
- Missing sidecar replay/action keys mean absent labels. Present-but-mismatched records hard-error.

## Runtime Authority

- `hydra-train` binary is entrypoint glue only.
- Config/preflight/probe/status contracts live in `hydra-train-runtime`.
- Python option conversion lives in `hydra-train-runtime::config::python`.
- Execution composition lives in `hydra-train-exec`.
- Normal BC runtime and loader authority are YAML-derived.
- Benchmark/preflight rows are evidence only; operators edit YAML by hand when accepting measured knobs.
- `example.yaml` is launch/config SSOT for intended training shape.

## Python Training UX

Python run owns run-local:

- `logs/events.jsonl`
- `logs/train_steps.jsonl`
- `checkpoints/latest.pt`
- optional `checkpoints/step_<global_step>.pt`
- TensorBoard event files
- `python_learner_result.json`
- `train.pid` for background runs

TensorBoard scans upward from configured port. Full layout and commands live in [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md).

## Evaluation And PPO

- Native ONNX arena is default eval/arena path.
- Inputs: ONNX export dir with `policy.onnx`, `policy.json`, `parity_fixture.safetensors`, or `.pt` checkpoint auto-exported first.
- Legacy Python checkpoint arena requires `--python-checkpoints`.
- Python owns training/export. Rust owns arena/RL inference.
- MahJAX GPU is default T1 PPO rollout route for serial and depth-1 Python PPO.
- MahJAX replay scanner is experimental validation infra and not GPU-throughput claim.

## Precision And Experimental Backends

- Omitted BC CUDA precision resolves to BF16 AMP when requested/effective by config.
- Explicit `fp32` overrides.
- CPU omission stays FP32.
- RL training and DeltaQ promotion hard-error on BF16.
- BF16 AMP wraps BC forward only; loss/backward/optimizer/checkpoint/validation remain FP32.
- Burn-CUDA backend is probe-only: feature `burn-cuda-probe`, FP32, BC shards only. It is not active/default throughput lane.

## Read Next

- Runtime details: [`GAME_ENGINE.md`](GAME_ENGINE.md)
- Current status: [`CURRENT_STATUS.md`](CURRENT_STATUS.md)
- Training/operator flow: [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md)
- MahJAX PPO scope: [`MAHJAX_PPO.md`](MAHJAX_PPO.md)
