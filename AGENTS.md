# Hydra TPU -- Agent Knowledge Base

## Project Overview

Riichi Mahjong AI with a **hybrid Rust + Python/JAX** stack.
Rust handles game engine, observation encoding, and data export.
Python/JAX handles the neural network (Flax), training loop, and TPU execution.

**Current phase**: Phase 0 -- Behavioral Cloning (BC) from expert replays.

## Architecture

```
crates/
  hydra-core/       # Game engine, action space (46 actions), encoder (192x34 obs)
  hydra-engine/     # Match orchestration, selfplay
  hydra-train/      # Rust-side model (Burn), bridge, data pipeline
    src/bin/export_bc_phase0/   # Streaming shard exporter (MJAI -> .npy)
    src/bridge_import.rs        # Weight import from JAX -> Burn (verified parity)

python/hydra_phase0_tpu/
  hydra_phase0_tpu/
    model.py         # HydraModel (Flax) -- 24-block ResNet, SE, multi-head
    train.py         # Main training loop (CLI entry point)
    train_step.py    # JIT-compiled train/eval steps
    train_state.py   # TrainState with optax AdamW + cosine schedule
    dataset.py       # ExportDataset, ShardRecord, streaming batch iterator
    config.py        # Pydantic config (ExperimentConfig)
    checkpoints.py   # Orbax checkpoint save/restore with resume state
    augment.py       # Train-time suit augmentation (6x)
    losses.py        # Multi-task loss (policy, value, GRP, tenpai, danger, etc.)
```

## Data Pipeline

1. Raw data: `.tar.zst` archives of MJAI JSON replay files (jade, throne, tenhou houou)
2. Rust exporter (`export_bc_phase0`): reads archives, runs game engine to produce observations, writes `.npy` shards
3. Python dataloader (`dataset.py`): reads shards, shuffles, yields batches

**Exporter uses streaming architecture** -- processes one game at a time, flushes shards to disk when buffer limits are hit. This keeps memory proportional to one shard (~131K samples, ~4GB) instead of the entire corpus.

**Exporter config** (`export_config.yaml`):
```yaml
data_dir: /path/to/tar.zst/archives
output_dir: /path/to/output
train_fraction: 0.9
seed: 42
max_samples_per_shard: 131072
max_games_per_shard: 4096
```

**Train/val split** is deterministic: `fnv1a_hash(game_identity) % 1000 < threshold`. No need to see all games first -- each game is classified as it's loaded.

**Shard hash verification**: the manifest stores SHA-256 per file. A sentinel value `{"shard.json": "skip"}` in the hash map bypasses verification (useful for interrupted exports that were reconstructed from shard metadata).

## Google Colab TPU Setup

### TPU v6e-1 (Trillium) Environment Variables

The Colab kernel process has the correct TPU config in its environment. Read it from `/proc/<kernel_pid>/environ`. The critical variables:

```bash
export TPU_SKIP_MDS_QUERY=1
export TPU_ACCELERATOR_TYPE=v6e-1
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_WORKER_HOSTNAMES=localhost
export TPU_WORKER_ID=0
export PJRT_DEVICE=TPU
export VBAR_CONTROL_SERVICE_URL=172.28.0.1:8353   # NOT localhost -- this is the host VM
unset XRT_TPU_CONFIG                                # conflicts with PJRT path
```

**Key discovery**: `VBAR_CONTROL_SERVICE_URL` must point to the host gateway (`172.28.0.1:8353`), not `localhost` or `[::]:8353`. The vbar control service runs outside the container. Without this, libtpu hangs for 60s then fails with `DEADLINE_EXCEEDED`.

### TPU v5e-1 Environment Variables

v5e uses an older init path. The `XRT_TPU_CONFIG` env var works directly and no `VBAR_CONTROL_SERVICE_URL` is needed.

### Python Environment on Colab

Colab ships Python 3.12. The system `pkg_resources` is broken (missing `pkgutil.ImpImporter`). Fix:
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
pip install setuptools==68.2.2
uv pip install -e "."
uv pip install "jax[tpu]>=0.9.1"
```

The `jax[tpu]` extra installs `libtpu` which provides the PJRT plugin for TPU backend detection.

### Setup Sequence (new Colab instance)

1. Run the colab_ssh cell in the notebook (installs `colab_ssh`, starts cloudflared tunnel)
2. SSH in via the provided `*.trycloudflare.com` hostname
3. Sync code: `rsync -avz --exclude='.git' --exclude='target' --exclude='.venv' ...`
4. Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y`
5. Build exporter: `cargo build --release --bin export_bc_phase0`
6. Set up Python venv (see above)
7. If exported shards exist on Drive, copy or point config at them
8. If not, run exporter against the Drive-mounted archives
9. Launch training with the TPU env vars set

### Drive Persistence

Survives Colab restarts:
- `/content/drive/MyDrive/dataset/` -- raw .tar.zst replay archives (~19GB)
- `/content/drive/MyDrive/hydra-phase0-export-full/` -- exported .npy shards (~117GB)
- `/content/drive/MyDrive/hydra-colab-output-ready/` -- training checkpoints, logs, TensorBoard
- `/content/drive/MyDrive/hydra-colab-setup.sh` -- env var setup script

Does NOT survive restarts:
- Local disk (`/root/`) -- code, venv, Rust toolchain, local shard copies

## Training

**Entry point**: `python -m hydra_phase0_tpu.train --config <yaml>`

**Config file** (`colab_current.yaml`):
```yaml
data:
  export_root: /path/to/bc_phase0_export
run:
  output_dir: /content/drive/MyDrive/hydra-colab-output-ready
  seed: 42
  num_epochs: 20
  batch_size: 2048
  microbatch_size: 256
  required_backend: tpu
  xla_cache_dir: /content/drive/MyDrive/hydra-xla-cache
optimizer:
  learning_rate: 2.5e-4
  warmup_steps: 1000
model:
  num_blocks: 24
  compute_dtype: bfloat16
```

**Output** goes to `<output_dir>/bc_tpu_phase0/`:
- `training_log.jsonl` -- end-of-epoch summaries
- `step_log.jsonl` -- per-N-step metrics (controlled by `log_every_n_steps`)
- `latest_orbax/` -- Orbax checkpoint (params + optimizer state)
- `latest_state.json` -- resume metadata
- `best/` -- best validation checkpoint
- `tb/` -- TensorBoard events

**No stdout during training** except a final summary. All metrics go to JSONL files. Use `PYTHONUNBUFFERED=1` and redirect to a log file.

## Known Issues and Gotchas

### JIT Compilation Time
The full `train_logical_batch` (microbatch gradient accumulation, 8 microsteps for batch=2048/micro=256) takes 15-20 minutes to JIT-compile on TPU v6e-1. A single `train_step` compiles in ~30s. Set `xla_cache_dir` in the run config to persist compiled executables to Drive -- subsequent launches skip recompilation entirely.

### Dirty Replays
A small percentage of MJAI replays have red-five desync issues. The exporter skips these automatically (prints "Skipping ... Replay desync" to stderr).

### Device Contention
If a previous training process held `/dev/vfio/0`, a new process may hang during TPU init. Always `kill -9` old Python processes before relaunching.

### Drive FUSE Latency
Reading large files from `/content/drive/` has FUSE overhead. For training, copy exported shards to local disk first for 10-100x faster IO. The exporter reads archives from Drive at acceptable speed since it streams sequentially.

## Cross-Stack Parity

The Rust model (Burn) and Python model (JAX/Flax) produce numerically identical outputs on the same weights and inputs (max drift ~4e-5). Verified via `bridge_import.rs` which imports JAX weights into Burn format.

## Testing

```bash
# Rust tests
cargo test -p hydra-train --bin export_bc_phase0
cargo test -p hydra-core

# Python tests
cd python/hydra_phase0_tpu
uv run pytest tests/
```
