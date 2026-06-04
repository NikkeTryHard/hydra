# MahJAX PPO

MahJAX is default rollout simulator for T1 Python PPO-control.

It is not replacement for `hydra-engine`, Rust replay loading, checkpoint schema, native ONNX arena/eval, or broad replay validation.

## Current Split

- MahJAX/JAX simulates batched single-round PPO rollouts on training CUDA device.
- Torch owns model forward, action sampling, PPO batch construction, optimizer step, checkpoints, and metrics.
- Rust/Hydra remains authority for MJAI replay semantics, encoder/action contracts, launcher config conversion, raw-MJAI transport, and native ONNX arena/eval.

Hard contracts stay unchanged: `192x34` observations, `46` logits/actions, `[46]` legal masks, recorded deterministic seeds, and hard errors on checkpoint/replay/schema mismatch.

## Routes

For `rl.phase: ppo_control`, rollout defaults to `mahjax-gpu`.

Reference routes stay available:

- `torch-callback`: Rust rollout collector calls Torch policy inference.
- `rust-ort`: Rust rollout collector uses exported ONNX/ORT inference.

`mahjax-gpu` requires CUDA training. It uses same CUDA device as training. separate PPO rollout device is rejected today.

## One PPO Update

1. Current policy snapshot is prepared when rollout path needs it.
2. MahJAX starts `games_per_update` environments from deterministic JAX keys.
3. JAX builds Hydra-shaped obs/masks and maps Hydra actions back into MahJAX actions.
4. Torch reads JAX arrays through DLPack, runs policy/value inference, and samples masked legal actions.
5. JAX steps active games until all finish.
6. Torch builds PPO rows with terminal placement utility, masked GAE, old logits/logprobs, and row metadata.
7. Torch trains configured PPO epochs/microbatches and writes metrics/checkpoints.

## Main Knobs

YAML/CLI:

- `rl.games_per_batch` / `--games-per-update`: parallel MahJAX games and JAX kernel shape.
- `rl.microbatch_size` / `--microbatch-size`: Torch PPO train microbatch.
- `device` / `--device`: CUDA device for both Torch train and MahJAX rollout.
- `--rollout-inference`: `mahjax-gpu`, `torch-callback`, or `rust-ort`.
- `--ppo-pipeline-depth`: `0` or `1`.

Environment:

- `HYDRA_MAHJAX_AOT=1|0`: defaults on; `0` is diagnostic.
- `HYDRA_MAHJAX_JAX_CACHE_DIR=<path>`: persistent JAX compilation cache.
- `HYDRA_MAHJAX_COMPLETION_SYNC_INTERVAL=<N>`: positive completion polling interval; default `32`.
- `HYDRA_MAHJAX_SYNC_TIMING=1`: diagnostic only because it synchronizes device work.

## Validation Limit

Replay scanner path:

```bash
python -m hydra_learner.mahjax.replay.scan --full ...
```

Treat scan as strict correctness evidence only when:

- `mismatch_count == 0`
- `unsupported_count == 0`
- every replay stops at `authority_exhausted`
- no row/event-limit stop is accepted

GPU replay validation remains experimental until trace export is transition-complete and batched JAX/GPU validation is faster than Hydra CPU authority.

## Checkpoints

PPO checkpoints remain data-only Python checkpoints. Resume validates model, runtime, optimizer, loss, source, RNG, and rollout snapshot metadata before training continues.

Do not relax checkpoint or replay hard errors for MahJAX convenience.
