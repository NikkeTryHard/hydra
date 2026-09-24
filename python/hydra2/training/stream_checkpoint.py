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
from hydra2.data.stream_iter import GameStream as GameStream
from hydra2.runtime.checkpoint import capture_rng_state as capture_rng_state
from hydra2.runtime.checkpoint_publish import _snapshot_payload_to_cpu as _snapshot_payload_to_cpu
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

if TYPE_CHECKING:
    from pathlib import Path

    from hydra2.data.stream_manifest import StreamManifest as StreamManifest
    from hydra2.training._rc_sections import RunConfig as RunConfig
    from hydra2.training.loop_train import SupervisedLoop as SupervisedLoop
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
# The bridge (`hydra2._native.resume`, Rust) owns the shuffle epoch-seed
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
    late = sys.modules.get("hydra2._native")
    if late is not None:
        mod = getattr(late, "resume", None)
        if mod is not None:
            _RESUME_MOD = mod
            return _RESUME_MOD
    if not _RESUME_PROBED:
        _RESUME_PROBED = True
        try:
            _RESUME_MOD = importlib.import_module("hydra2._native").resume
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
            "hydra2._native before deriving an epoch seed"
        )
    try:
        seeded: int = resume.epoch_seed(data_seed, epoch)
        return seeded
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
    writer: Any = None,
    span_sink: Any = None,
) -> Path:
    """Atomically publish ``ckpt-<update>.pt`` + ResumeExactPlan sidecar.

    Resume/RNG-restore entries are bridge-direct: ``epoch_seed`` crosses the
    bridge ``resume`` judge (wrapping ``data_seed + epoch``; missing bridge
    raises — mismatch=raise, never an oracle), while ``rng_state`` capture
    and the ``_rng_anchors()`` bundle stay torch/Python (no Rust GPU math).

    Every live read (model/optimizer/scheduler state, dataset and shuffle
    snapshots, RNG) happens synchronously before returning; GPU tensors are
    snapshotted to detached CPU clones. With ``writer`` given, the
    ``torch.save`` serialization plus the three atomic file publishes run
    on the background worker while training continues and this returns the
    reserved path immediately (the caller owns draining before eval and
    closing at run end). Without a writer the publish runs inline, exactly
    as before. Failure mode: a worker error raises at the caller's next
    poll/drain (fail closed); the returned path is reserved, never a
    promise that bytes already landed.
    """
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    import time as _time

    _asm_t0 = _time.perf_counter()
    _asm_wall = _time.time()
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
    prefix_target = checkpoint_dir / prefix_name
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
    # Snapshot GPU tensors to detached CPU clones BEFORE any background
    # handoff: state_dict() aliases live parameters and the next update
    # mutates them. The plain leaves above are already fresh copies
    # (buffer/shuffle snapshots, to_dict, sampler state), and the snapshot
    # rebuilds their containers anyway, so the worker owns everything it
    # serializes. Inline path keeps the live payload (byte-identical to the
    _snap_t0 = _time.perf_counter()

    _snap_wall = _time.time()
    publish_payload = _snapshot_payload_to_cpu(payload) if writer is not None else payload
    _snap_ms = (_time.perf_counter() - _snap_t0) * 1000.0 if writer is not None else 0.0
    _main_spans: list[tuple[str, float, float]] = []
    if span_sink is not None:
        _main_spans.append(("assemble", _asm_wall, (_snap_wall - _asm_wall) * 1000.0))
        _main_spans.append(("snapshot", _snap_wall, _snap_ms))
    data_cursor = dataset.stream_cursor()
    sidecar_body: dict[str, Any] = {
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
    }
    ckpt_path = checkpoint_dir / f"ckpt-{update:06d}.pt"

    def _publish() -> Path:
        buffer = io.BytesIO()
        torch.save(publish_payload, buffer)
        blob = buffer.getvalue()
        payload_digest = str(sha256_digest(blob))
        # Generation order preserved on the single worker: prefix record,
        # then payload, then the sidecar naming the payload digest — the
        # same order as the historical inline publish.
        atomic_replace_bytes(prefix_target, prefix_blob)
        atomic_replace_bytes(ckpt_path, blob)
        sidecar = dict(sidecar_body)
        sidecar["payload_sha256"] = payload_digest
        atomic_replace_bytes(
            ckpt_path.with_suffix(".json"),
            json.dumps(sidecar, indent=2, sort_keys=True).encode("utf-8"),
        )
        return ckpt_path

    if writer is not None:
        writer.submit(
            _publish,
            generation=int(update),
            description=f"stream-ckpt-{update:06d}",
            span_sink=span_sink,
            main_spans=tuple(_main_spans),
        )
    else:
        _publish()
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
                logged.add(int(json.loads(line).get("global_update", -1)))  # pyrefly: ignore[unknown-argument-type] # JSON line mapping dynamic; int owns the gate
            except (ValueError, AttributeError):
                continue
    appended = 0
    with (
        metrics_path.open("a", encoding="utf-8") as metrics_handle,
        train_log.open("a", encoding="utf-8") as log_handle,
    ):
        for entry in history:
            update: int = int(entry.get("global_update", -1))
            if update in logged:
                continue
            # Wall stamp at write time only (logging-only: the in-memory
            # entry stays byte-identical for checkpoint payloads, digests,
            # and resume — a hashed wall clock would break identity).
            row = dict(entry)
            row["t_wall_s"] = time.time()
            payload = json.dumps(row, sort_keys=True) + "\n"
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
    span_sink: Any = None,
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
    eval_path: Path = run_dir / "eval" / "eval.jsonl"
    train_log: Path = run_dir / "logs" / "train.log"
    try:
        recorded: set[int] = set()
        if eval_path.is_file():
            for line in eval_path.read_text(encoding="utf-8").splitlines():
                if line.strip() == "":
                    continue
                try:
                    recorded.add(int(json.loads(line).get("update", -1)))  # pyrefly: ignore[unknown-argument-type] # JSON line mapping dynamic; int owns the gate
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
        expand_wall_start = time.time()
        for streamed in stream:
            if len(rows) >= need_batches * micro:
                break
            games_touched += 1
            try:
                row_dicts, _ = _expand_game_planes(streamed.game, streamed.split, streamed.raw)
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
        encode_wall_start = time.time()
        from hydra2.training.rust_batch import assemble_slim_batch

        batches = [
            assemble_slim_batch(
                rows[start : start + micro],
                action_count=config.model.action_count,
                pack_histories=config.loop.pack_histories,
            )
            for start in range(0, need_batches * micro, micro)
        ]
        encode_wall = time.perf_counter() - encode_start
        report_t0 = time.perf_counter()
        report_wall_start = time.time()
        report: dict[str, Any] = loop.evaluate_report(batches)
        report_ms = (time.perf_counter() - report_t0) * 1000.0
        if span_sink is not None:
            # Stage rows are observer-only (never hashed, never in the
            # eval entry): wall time must not leak into eval.jsonl.
            span_sink.emit(
                stage="eval-expand",
                dur_ms=expand_wall * 1000.0,
                update=update,
                t_start_s=expand_wall_start,
            )
            span_sink.emit(
                stage="eval-encode",
                dur_ms=encode_wall * 1000.0,
                update=update,
                t_start_s=encode_wall_start,
            )
            span_sink.emit(
                stage="eval-report",
                dur_ms=report_ms,
                update=update,
                t_start_s=report_wall_start,
            )
        entry: dict[str, Any] = {"update": update}
        for key, value in report.items():
            entry[key] = float(value) if isinstance(value, (int, float)) else value
        with eval_path.open("a", encoding="utf-8") as handle:
            _ = handle.write(json.dumps(entry, sort_keys=True) + "\n")
        with train_log.open("a", encoding="utf-8") as handle:
            _ = handle.write(
                f"eval:update={update:06d} nll={float(report.get('masked_nll', 0.0)):.6f} "
                f"top1={float(report.get('top1', 0.0)):.4f} "
                f"ece={float(report.get('calibration_ece', 0.0)):.6f} "
                f"batches={int(report.get('num_eval_batches', 0))} "
                f"games={games_touched} expand_s={expand_wall:.2f} encode_s={encode_wall:.2f}\n"
            )
        return entry
    except Exception as exc:
        with train_log.open("a", encoding="utf-8") as handle:
            _ = handle.write(f"eval:update={update:06d} skipped ({type(exc).__name__})\n")
        return None
