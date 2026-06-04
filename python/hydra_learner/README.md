# Hydra Python Learner

Python owns PyTorch learner: BC train loop, model/loss/optimizer, checkpoints, metrics, ONNX export, and PPO train step.

Rust owns replay parsing, raw-MJAI transport, shard build/validation, launcher conversion, runtime/action/encoder contracts, native ONNX arena, and RL inference authority.

## Entrypoints

Primary entrypoint:

```bash
python -m hydra_learner.cli
```

Compatibility entrypoints to keep working:

```bash
python -m hydra_learner.train_bc
python scripts/hydra_pytorch_oracle.py
```

## Source Layout

```text
python/hydra_learner/
  src/hydra_learner/
    cli.py
    train_bc.py
    arena_eval.py
    export_inference.py
    ppo_control.py
    checkpointing/
    data/
      raw_mjai/
    mahjax/
    model/
    ppo/
    rl_experiments/
    telemetry/
    training/
  tests/
```

## Read Next

- [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md): training, resume, checkpoints, TensorBoard, shards, PPO launch.
- [`docs/MAHJAX_PPO.md`](../../docs/MAHJAX_PPO.md): MahJAX PPO contract and limits.
- [`docs/COMPATIBILITY_SURFACE.md`](../../docs/COMPATIBILITY_SURFACE.md): hard runtime/training contracts.
- [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md): shipped/staged/default status.
- [`AGENTS.md`](../../AGENTS.md): repo rules.
