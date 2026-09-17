"""Streaming checkpoints: files, sidecars, prune, history, holdout eval.

Owns atomic publication of ``ckpt-<update>.pt`` plus its ResumeExactPlan
sidecar (payload hash, RNG anchors, shuffle and buffer snapshots),
checkpoint pruning, resume-safe metrics append, and the observer-only
held-out eval (failures log, never abort training). The shuffle epoch-seed
entry is bridge-direct via the ``resume`` judge (``_epoch_seed``; missing
bridge raises ContractError with a build-ext hint, never an oracle);
torch owns the RNG capture and the payload tensors (no Rust GPU math,
Burn/Candle out).
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import sys
import time
from typing import TYPE_CHECKING, Any

import torch

from hydra2.artifacts.atomic import atomic_replace_bytes as atomic_replace_bytes
from hydra2.artifacts.digest import sha256_digest as sha256_digest
from hydra2.contracts.common import ContractError as ContractError
from hydra2.data.stream import GameStream as GameStream
from hydra2.runtime.checkpoint import capture_rng_state as capture_rng_state
from hydra2.training.dataset import encode_observation_rows as encode_observation_rows
from hydra2.training.stream_build import (
    _prefix_hashes_name as _prefix_hashes_name,
)
from hydra2.training.stream_build import (
    _rng_anchors as _rng_anchors,
)
from hydra2.training.stream_build import _sidecar_cursor as _sidecar_cursor
from hydra2.training.stream_expand import (
    _FEATURE_FOLD_DIM as _FEATURE_FOLD_DIM,
)
from hydra2.training.stream_expand import (
    _SINGLETON_RANK as _SINGLETON_RANK,
)
from hydra2.training.stream_expand import _expand_game_planes as _expand_game_planes
from hydra2.training.stream_expand import _expand_game_rows as _expand_game_rows
from hydra2.training.stream_expand import _row_to_dict as _row_to_dict

if TYPE_CHECKING:
    from pathlib import Path

    from hydra2.data.stream import StreamManifest as StreamManifest
    from hydra2.training.loop import SupervisedLoop as SupervisedLoop
    from hydra2.training.run_config import RunConfig as RunConfig
    from hydra2.training.stream_dataset_buffer import _StreamDataset as _StreamDataset

__all__ = [
    "_append_new_history",
    "_ckpt_names",
    "_epoch_seed",
    "_prune_checkpoints",
    "_run_holdout_eval",
    "_write_streaming_checkpoint",
]

# ---------------------------------------------------------------------------
# Bridge-resume glue (resume/RNG-restore entries only; torch owns GPU/RNG).
# ---------------------------------------------------------------------------
# The bridge (`hydra2_replay_rs.resume`, Rust) owns the shuffle epoch-seed
# judge used on the checkpoint write path: `epoch_seed` (wrapping
# `data_seed + epoch`, mirroring the sidecar `epoch_seed`). Torch CPU/CUDA
# RNG capture/restore stays Python (`capture_rng_state` /
# `_restore_rng_state` — no Rust GPU math, Burn/Candle out); the RNG-anchor
# compare (`_verify_rng_anchors`) stays Python per `resume.rs`. The helper
# below is bridge-direct (no oracle body: bridge==oracle proven by the
# throwaway parity script — wrapping `+` identical for all reachable seeds;
# missing bridge or any bridge error raises ContractError with a build-ext
# hint — mismatch=raise, never a silent pass).
_RESUME_MOD: Any = None
_RESUME_PROBED: bool = False

#: Injectable native backend (tests monkeypatch this; production leaves None
#: so `_resume_native()` imports the compiled extension).
_RESUME_NATIVE_OVERRIDE: Any = None


def _resume_native() -> Any | None:
    """Import the built `resume` submodule once; None → fail-closed for counting.

    ``None`` means the extension (or its ``resume`` surface) is not
    importable; `_epoch_seed` raises ContractError with a build-ext hint on
    ``None`` (no oracle). Any other import-time failure raises ContractError
    (mismatch=raise, never a silent pass).
    """
    global _RESUME_MOD, _RESUME_PROBED
    if _RESUME_NATIVE_OVERRIDE is not None:
        return _RESUME_NATIVE_OVERRIDE
    if _RESUME_MOD is not None:
        return _RESUME_MOD
    late = sys.modules.get("hydra2_replay_rs")
    if late is not None:
        mod = getattr(late, "resume", None)
        if mod is not None:
            _RESUME_MOD = mod
            return _RESUME_MOD
    if not _RESUME_PROBED:
        _RESUME_PROBED = True
        try:
            _RESUME_MOD = importlib.import_module("hydra2_replay_rs").resume
        except (ImportError, AttributeError):
            _RESUME_MOD = None
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"resume bridge unusable: {type(exc).__name__}: {exc}") from exc
    return _RESUME_MOD


def _epoch_seed(*, data_seed: int, epoch: int) -> int:
    """Shuffle epoch seed: bridge-direct `resume.epoch_seed`.

    `epoch_seed = data_seed + epoch` (wrapping add u64 bridge-side; the
    stream builder owns the draw, the sidecar only pins it). Inputs are
    validated (bool/non-int/negative → ContractError, mirroring the sidecar
    `epoch_seed >= 0` gate — the gate stays Python because the u64 FFI
    accepts bool-as-int). Missing bridge or any bridge error raises
    ContractError with a build-ext hint (mismatch=raise, never an oracle).
    """
    if isinstance(data_seed, bool) or not isinstance(data_seed, int) or data_seed < 0:
        raise ContractError(f"epoch_seed data_seed must be a non-negative int, got {data_seed!r}")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ContractError(f"epoch_seed epoch must be a non-negative int, got {epoch!r}")
    resume = _resume_native()
    if resume is None:
        raise ContractError(
            "resume bridge unavailable; run `pixi run build-ext` to build "
            "hydra2_replay_rs before deriving an epoch seed"
        )
    try:
        return int(resume.epoch_seed(int(data_seed), int(epoch)))
    except ContractError:
        raise
    except (ImportError, AttributeError) as exc:
        raise ContractError(
            "resume epoch_seed bridge surface missing; rebuild the bridge "
            f"(`pixi run build-ext`): {type(exc).__name__}: {exc}"
        ) from exc
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    except Exception as exc:
        raise ContractError(f"resume epoch_seed failed: {type(exc).__name__}: {exc}") from exc


def _write_streaming_checkpoint(
    *,
    run_dir: Path,
    update: int,
    run_digest: str,
    stream_digest: str,
    config: RunConfig,
    loop: SupervisedLoop,
    dataset: _StreamDataset,
    batch_size: int,
) -> Path:
    """Atomically publish ``ckpt-<update>.pt`` + ResumeExactPlan sidecar.

    Resume/RNG-restore entries are bridge-direct: ``epoch_seed`` crosses the
    bridge ``resume`` judge (wrapping ``data_seed + epoch``; missing bridge
    raises — mismatch=raise, never an oracle), while ``rng_state`` capture
    and the ``_rng_anchors()`` bundle stay torch/Python (no Rust GPU math).
    """
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scheduler_state: Any = {}
    if loop.scheduler is not None and hasattr(loop.scheduler, "state_dict"):
        with contextlib.suppress(Exception):
            scheduler_state = loop.scheduler.state_dict()
    dataset_snapshot = dataset.buffer_snapshot()
    shuffle_entries: list[dict[str, Any]] = []
    shuffle_rng: dict[str, Any] = {}
    shuffle_keys: list[str] = []
    if config.data.shuffle_buffer_size > 0 and dataset._stream is not None:
        snap = dataset._stream.shuffle_snapshot()
        if snap is not None:
            entries, rng_state = snap
            shuffle_entries = [dict(e) for e in entries]
            shuffle_rng = dict(rng_state)
            shuffle_keys = [str(e["key"]) for e in shuffle_entries]
    prefix_hashes: list[str] = []
    if dataset._stream is not None:
        prefix_hashes = dataset._stream.prefix_hashes_snapshot()
    prefix_blob = "".join(f"{sha}\n" for sha in prefix_hashes).encode("utf-8")
    prefix_name = _prefix_hashes_name(update)
    atomic_replace_bytes(checkpoint_dir / prefix_name, prefix_blob)
    prefix_record = {
        "file": prefix_name,
        "count": len(prefix_hashes),
        "sha256": str(sha256_digest(prefix_blob)),
    }
    payload: dict[str, Any] = {
        "model_state": loop.model.state_dict(),
        "optimizer_state": loop.optimizer.state_dict(),
        "scheduler_state": scheduler_state,
        "training_state": loop.state.to_dict(),
        "sampler_state": dataset.get_sampler_state(),
        "rng_state": capture_rng_state(),
        "loss_history": [dict(entry) for entry in loop.loss_history],
        "stream_epoch": dataset.epoch,
        "microbatch_size": batch_size,
        "rows_consumed_in_epoch": dataset.rows_consumed_in_epoch,
        "dataset_buffer": dataset_snapshot,
        "shuffle_buffer": {
            "buffer_keys": shuffle_keys,
            "buffer_entries": shuffle_entries,
            "buffer_rng_state": shuffle_rng,
            "epoch_seed": _epoch_seed(data_seed=config.seeds.data_seed, epoch=dataset.epoch),
            "buffer_size": config.data.shuffle_buffer_size,
            "prefix_hashes": prefix_record,
        },
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    blob = buffer.getvalue()
    payload_digest = str(sha256_digest(blob))
    ckpt_path = checkpoint_dir / f"ckpt-{update:06d}.pt"
    atomic_replace_bytes(ckpt_path, blob)

    data_cursor = dataset.stream_cursor()
    sidecar: dict[str, Any] = {
        "global_update": update,
        "run_digest": run_digest,
        "stream_cursor": _sidecar_cursor(data_cursor),
        "stream_shuffle_pos": data_cursor.shuffle_pos,
        "stream_epoch": dataset.epoch,
        "microbatch_size": batch_size,
        "rows_consumed_in_epoch": dataset.rows_consumed_in_epoch,
        "rng_state": _rng_anchors(),
        "shuffle": {
            "buffer_keys": shuffle_keys,
            "epoch_seed": _epoch_seed(data_seed=config.seeds.data_seed, epoch=dataset.epoch),
            "buffer_size": config.data.shuffle_buffer_size,
            "buffer_rng_state": shuffle_rng,
            "buffer_entries": shuffle_entries,
            "prefix_hashes": prefix_record,
        },
        "dataset_buffer": dataset_snapshot,
        "accum": {"yielded_batches": update, "micro_in_update": 0},
        "worker_plan": {
            "num_workers": config.data.num_workers,
            "world_size": config.data.world_size,
            "rank": _SINGLETON_RANK,
        },
        "manifests": {
            "stream_manifest_hash": stream_digest,
            "dataset_manifest_hash": stream_digest,
        },
        "payload_sha256": payload_digest,
    }
    atomic_replace_bytes(
        ckpt_path.with_suffix(".json"),
        json.dumps(sidecar, indent=2, sort_keys=True).encode("utf-8"),
    )
    return ckpt_path


def _ckpt_names(run_dir: Path) -> list[str]:
    """``ckpt-<update>.pt`` names in numeric update order (torn names skipped)."""
    found: list[tuple[int, str]] = []
    for path in (run_dir / "checkpoints").glob("ckpt-*.pt"):
        try:
            found.append((int(path.stem.split("-")[1]), path.name))
        except (ValueError, IndexError):
            continue
    return [name for _, name in sorted(found)]


def _prune_checkpoints(run_dir: Path, *, keep: int | None) -> None:
    """Bound retained ``ckpt-<update>.pt`` files (null keeps all)."""
    if keep is None:
        return
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ContractError(f"keep_last_checkpoints must be a positive int, got {keep!r}")
    names = _ckpt_names(run_dir)
    for stale_name in names[: max(0, len(names) - keep)]:
        stale = run_dir / "checkpoints" / stale_name
        try:
            stale_update = int(stale_name.split("-")[1].split(".")[0])
        except (ValueError, IndexError):
            stale_update = -1
        with contextlib.suppress(OSError):
            stale.unlink()
        with contextlib.suppress(OSError):
            stale.with_suffix(".json").unlink()
        if stale_update >= 0:
            with contextlib.suppress(OSError):
                (run_dir / "checkpoints" / _prefix_hashes_name(stale_update)).unlink()


def _append_new_history(run_dir: Path, history: list[dict[str, float]]) -> int:
    """Append per-update rows for entries not yet in ``metrics.jsonl``.

    Resume-safe: already-logged updates (by ``global_update``) are skipped,
    so a resumed run never duplicates rows. Returns appended count.
    """
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    train_log = run_dir / "logs" / "train.log"
    logged: set[int] = set()
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip() == "":
                continue
            try:
                logged.add(int(json.loads(line).get("global_update", -1)))
            except (ValueError, AttributeError):
                continue
    appended = 0
    with (
        metrics_path.open("a", encoding="utf-8") as metrics_handle,
        train_log.open("a", encoding="utf-8") as log_handle,
    ):
        for entry in history:
            update = int(entry.get("global_update", -1))
            if update in logged:
                continue
            payload = json.dumps(entry, sort_keys=True) + "\n"
            _ = metrics_handle.write(payload)  # intentionally discarded: byte count unneeded
            _ = log_handle.write(  # intentionally discarded: byte count unneeded
                f"update={update:06d} total={entry.get('total', 0.0):.6f} "
                f"policy={entry.get('policy', 0.0):.6f} "
                f"top1={entry.get('top1', 0.0):.4f}\n"
            )
            logged.add(update)
            appended += 1
    return appended


def _run_holdout_eval(
    *,
    manifest: StreamManifest,
    ratios: dict[str, float],
    config: RunConfig,
    loop: SupervisedLoop,
    run_dir: Any,
    update: int,
) -> dict[str, Any] | None:
    """Held-out eval on val-split games (observer: never touches train state).

    Builds a throwaway val stream (fixed seed/epoch, so every eval sees
    identical batches), expands + encodes ``eval.num_batches`` microbatches,
    and runs :meth:`SupervisedLoop.evaluate_report` (no grad; the model is
    restored to train mode inside). Appends one row to ``eval/eval.jsonl``
    plus a ``train.log`` marker. ANY failure is logged and returns None --
    eval never aborts training. Resume-safe: fully stateless, and
    already-recorded updates are skipped, never duplicated.
    """
    eval_path = run_dir / "eval" / "eval.jsonl"
    train_log = run_dir / "logs" / "train.log"
    try:
        recorded: set[int] = set()
        if eval_path.is_file():
            for line in eval_path.read_text(encoding="utf-8").splitlines():
                if line.strip() == "":
                    continue
                try:
                    recorded.add(int(json.loads(line).get("update", -1)))
                except (ValueError, AttributeError):
                    continue
        if update in recorded:
            return None
        micro = config.eval.microbatch_size
        if micro is None:
            micro = config.loop.microbatch_size
        if isinstance(micro, bool) or not isinstance(micro, int) or micro <= 0:
            raise ContractError(f"eval microbatch invalid: {micro!r}")
        need_batches = config.eval.num_batches
        if isinstance(need_batches, bool) or not isinstance(need_batches, int) or need_batches <= 0:
            raise ContractError(f"eval num_batches invalid: {need_batches!r}")
        stream = GameStream(
            manifest,
            seed=config.seeds.data_seed,
            ratios=ratios,
            epoch=0,
            split=config.data.val_split,
            shuffle_buffer=0,
        )
        rows: list[dict[str, Any]] = []
        games_touched = 0
        expand_start = time.perf_counter()
        for streamed in stream:
            if len(rows) >= need_batches * micro:
                break
            games_touched += 1
            try:
                if config.data.replay_backend == "rust":
                    row_dicts, _ = _expand_game_planes(streamed.game, streamed.split, streamed.raw)
                else:
                    actor_rows, _ = _expand_game_rows(
                        streamed.game, streamed.split, config.data.replay_backend
                    )
                    row_dicts = [_row_to_dict(row) for row in actor_rows]
            except ContractError:
                continue
            for row_dict in row_dicts:
                rows.append(row_dict)
                if len(rows) >= need_batches * micro:
                    break
        expand_wall = time.perf_counter() - expand_start
        if len(rows) == 0:
            raise ContractError("no val rows for held-out eval")
        encode_start = time.perf_counter()
        if config.data.replay_backend == "rust":
            from hydra2.training.rust_batch import assemble_slim_batch

            batches = [
                assemble_slim_batch(
                    rows[start : start + micro], action_count=config.model.action_count
                )
                for start in range(0, need_batches * micro, micro)
            ]
        else:
            batches = [
                encode_observation_rows(
                    rows[start : start + micro],
                    num_actions=config.model.action_count,
                    feature_dim=_FEATURE_FOLD_DIM,
                )
                for start in range(0, need_batches * micro, micro)
            ]
        encode_wall = time.perf_counter() - encode_start
        report = loop.evaluate_report(batches)
        entry: dict[str, Any] = {"update": update}
        for key, value in report.items():
            entry[key] = float(value) if isinstance(value, (int, float)) else value
        with eval_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(
                f"eval:update={update:06d} nll={float(report.get('masked_nll', 0.0)):.6f} "
                f"top1={float(report.get('top1', 0.0)):.4f} "
                f"ece={float(report.get('calibration_ece', 0.0)):.6f} "
                f"batches={int(report.get('num_eval_batches', 0))} "
                f"games={games_touched} expand_s={expand_wall:.2f} encode_s={encode_wall:.2f}\n"
            )
        return entry
    except Exception as exc:
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(f"eval:update={update:06d} skipped ({type(exc).__name__})\n")
        return None
