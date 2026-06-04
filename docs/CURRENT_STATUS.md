# Hydra Current Status

Current code wins if this file drifts.

Use this page to see what is default, gated, experimental, or parked.

## Default / Shipped

- **Rules authority:** Tenhou/MJAI semantics through `hydra-engine`.
- **Encoder/action contract:** live input `192x34`, action space `46`, legal mask `[bool; 46]`.
- **BC training:** Python/PyTorch learner launched by Rust.
- **BC input:** raw MJAI streaming by default; compact shards are optional fixed/cache input.
- **Native arena:** Rust + ONNX Runtime CUDA default. `.pt` checkpoints auto-export to ONNX before arena/eval.
- **T1 PPO:** Python PPO-control with MahJAX GPU rollout by default.
- **BF16/AMP:** BC CUDA can use BF16 AMP by default. CPU stays FP32. RL/DeltaQ BF16 hard-errors.

## Gated / Opt-In

- **BC shards:** compact v3 only. Dense/v2 hard-errors.
- **Validation gates:** affect best-checkpoint promotion only; do not gate resume checkpoints.
- **Python ExIt targets:** available only with matching compact-shard sidecar provenance.
- **DeltaQ:** carrier metadata exists, but positive DeltaQ loss/head remains fail-closed.
- **Legacy Rust/Burn BC:** debug/advanced path only via explicit backend selection.
- **PBRS/GRP shaping:** default-off; nonzero beta needs strict validation artifact authorization.

## Experimental / Parked

- MahJAX replay scanner is validation infrastructure, not fast corpus validator yet.
- Burn CUDA graph replay is probe-only.
- Backbone/profile throughput probes are not strength claims.
- Belief/mixture/opponent-hand-type support is not Python default.
- Broader public-belief search, deeper robust-opponent backups, and larger opponent latent heads are reserve work.

## Read Next

- Runtime semantics: [`GAME_ENGINE.md`](GAME_ENGINE.md)
- Hard compatibility contracts: [`COMPATIBILITY_SURFACE.md`](COMPATIBILITY_SURFACE.md)
- Training/operator flow: [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md)
- MahJAX PPO scope: [`MAHJAX_PPO.md`](MAHJAX_PPO.md)
- Repo rules: [`../AGENTS.md`](../AGENTS.md)
