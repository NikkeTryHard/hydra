"""Resume application and checkpoint-to-checkpoint segment execution.

Owns applying verified payloads to live runtime objects (mutate only
after verify), reservoir prime snapshots, resume-envelope verification
(read-only), the caller-owned overlap feed, profiler captures, and the
train stepper that advances one checkpoint segment at a time.
Resume/RNG-restore entries are bridge-direct via the ``resume`` judges
(``_pack_cursor`` / ``_unpack_cursor`` / ``_check_buffer_entries`` /
``_verify_prefix_blob``; missing bridge or any bridge error raises
ContractError with a build-ext hint, never an oracle fallback); torch owns
the RNG restore and the model/optimizer tensors (no Rust GPU math,
Burn/Candle out).
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from typing import TYPE_CHECKING, Any, Literal

import torch

from hydra2.contracts.common import ContractError as ContractError
from hydra2.data.stream_read import StreamCursor as DataStreamCursor
from hydra2.training.loop_state import TrainingState as TrainingState

try:
    from hydra2._native import contracts as _resume_cfg_bridge  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal, same text as call-site
    _resume_cfg_bridge = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from hydra2.data.stream_manifest import StreamManifest as StreamManifest
    from hydra2.training._rc_sections import ResumePlan as ResumePlan
    from hydra2.training.loop_train import SupervisedLoop as SupervisedLoop
    from hydra2.training.stream_dataset_buffer import _StreamDataset as _StreamDataset

__all__ = [
    "_GatedOverlapFeed",
    "_apply_resume_payload",
    "_backward_pass_autocast_for",
    "_check_buffer_entries",
    "_open_gated_feed",
    "_pack_cursor",
    "_profile_single_update",
    "_train_segment",
    "_unpack_cursor",
    "_verify_prefix_blob",
]


# ---------------------------------------------------------------------------
# Bridge-resume glue (resume/RNG-restore entries only; torch owns GPU/RNG).
# ---------------------------------------------------------------------------
#
# The bridge (`hydra2._native.resume`, Rust) owns the pure resume-side
# judges used on the read/verify path: the prefix-hash chain verify
# (`verify_prefix_blob`: blob sha + count + per-entry `sha256:` shape +
# sorted order, all detached, bridge-direct), cursor pack/unpack
# (YAML-resume round-trip, bool-rejecting, bridge-direct) and buffer-entry
# shape checks (`check_buffer_entries`, bridge-direct).
# Torch CPU/CUDA RNG restore stays Python (`_restore_rng_state` — no Rust
# GPU math, Burn/Candle out); the RNG-anchor compare (`_verify_rng_anchors`
# in `stream_build`) stays Python per `resume.rs`. The counting helpers below
# are bridge-direct (no oracle bodies: bridge==oracle proven by the
# throwaway parity script — pack key order alphabetical both sides, unpack
# rejects identical, buffer gates identical; any bridge-missing or
# bridge-present error raises ContractError — mismatch=raise, never silent).
_RESUME_MOD: Any = None
_RESUME_PROBED: bool = False

#: Injectable native backend (tests monkeypatch this; production leaves None
#: so `_resume_native()` imports the compiled extension). Mirrors the
#: `_RING_NATIVE_OVERRIDE` hook in `models/encoder.py`.
_RESUME_NATIVE_OVERRIDE: Any = None


def _resume_native() -> Any | None:
    """Import the built `resume` submodule once; None → fail-closed for counting.

    ``None`` means the extension (or its ``resume`` surface) is not
    importable. All callers (`_pack_cursor` / `_unpack_cursor` /
    `_check_buffer_entries` / `_verify_prefix_blob`) raise ContractError
    with a build-ext hint on ``None`` (no oracle). Any other import-time
    failure raises ContractError (mismatch=raise, never a silent pass).
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


def _pack_cursor(*, cursor: DataStreamCursor) -> dict[str, int]:
    """Cursor pack: six frontier fields → YAML-resume mapping (bridge-direct).

    `resume.pack_cursor` owns the mapping (byte-identical to the retired
    `StreamCursor.to_dict` oracle). Missing bridge or any bridge error raises
    ContractError (fail-closed with a build-ext hint, never an oracle).
    """
    resume = _resume_native()
    if resume is None:
        raise ContractError(
            "resume bridge unavailable; run `pixi run build-ext` to build "
            "hydra2._native before packing a cursor"
        )
    try:
        packed: dict[str, int] = resume.pack_cursor(
            file_index=cursor.file_index,
            byte_offset=cursor.byte_offset,
            games_seen=cursor.games_seen,
            seed=cursor.seed,
            epoch=cursor.epoch,
            shuffle_pos=cursor.shuffle_pos,
        )
        return dict(packed)
    except ContractError:
        raise
    except (ImportError, AttributeError) as exc:
        raise ContractError(
            "resume pack_cursor bridge surface missing; rebuild the bridge "
            f"(`pixi run build-ext`): {type(exc).__name__}: {exc}"
        ) from exc
    except (ValueError, OverflowError) as exc:
        raise ContractError(str(exc)) from exc
    except Exception as exc:
        raise ContractError(f"resume pack_cursor failed: {type(exc).__name__}: {exc}") from exc


def _unpack_cursor(raw: object) -> DataStreamCursor:
    """Cursor unpack: YAML-resume mapping → frontier fields (bridge-direct).

    `resume.unpack_cursor` owns the gates (bool-rejecting, non-negative-int —
    identical to the retired `StreamCursor.from_dict` oracle). Missing bridge
    or any bridge error raises ContractError (fail-closed with a build-ext
    hint, never an oracle).
    """
    resume = _resume_native()
    if resume is None:
        raise ContractError(
            "resume bridge unavailable; run `pixi run build-ext` to build "
            "hydra2._native before unpacking a cursor"
        )
    try:
        if not isinstance(raw, dict):
            raise ContractError(f"cursor payload must be a mapping, got {type(raw).__name__}")
        mapping: dict[str, Any] = raw
        fields: tuple[int, int, int, int, int, int] = resume.unpack_cursor(mapping)
        file_index, byte_offset, games_seen, seed, epoch, shuffle_pos = fields
        return DataStreamCursor(
            file_index=file_index,
            byte_offset=byte_offset,
            games_seen=games_seen,
            seed=seed,
            epoch=epoch,
            shuffle_pos=shuffle_pos,
        )
    except ContractError:
        raise
    except (ImportError, AttributeError) as exc:
        raise ContractError(
            "resume unpack_cursor bridge surface missing; rebuild the bridge "
            f"(`pixi run build-ext`): {type(exc).__name__}: {exc}"
        ) from exc
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    except Exception as exc:
        raise ContractError(f"resume unpack_cursor failed: {type(exc).__name__}: {exc}") from exc


def _check_buffer_entries(entries: object, *, ckpt: Path) -> int:
    """Buffer-entry shape gate (bridge-direct `check_buffer_entries`).

    Each entry carries exactly `{key, path, offset, split, rows}` (unknown
    keys, empty key/path/split, negative offset, rows <= 0 all raise —
    identical to the retired oracle). Missing bridge or any bridge error
    raises ContractError (fail-closed with a build-ext hint, never an
    oracle). Returns the entry count.
    """
    resume = _resume_native()
    if resume is None:
        raise ContractError(
            "resume bridge unavailable; run `pixi run build-ext` to build "
            f"hydra2._native before checking buffer entries: {ckpt}"
        )
    try:
        return int(resume.check_buffer_entries(list(entries)))  # type: ignore[arg-type]
    except ContractError:
        raise
    except (ImportError, AttributeError) as exc:
        raise ContractError(
            "resume check_buffer_entries bridge surface missing; rebuild the "
            f"bridge (`pixi run build-ext`): {ckpt} ({type(exc).__name__}: {exc})"
        ) from exc
    except ValueError as exc:
        raise ContractError(f"checkpoint buffer entries invalid: {ckpt} ({exc})") from exc
    except Exception as exc:
        raise ContractError(
            f"resume check_buffer_entries failed: {ckpt} ({type(exc).__name__}: {exc})"
        ) from exc


def _verify_prefix_blob(*, data: bytes, count: int, digest: str, ckpt: Path) -> list[str]:
    """Prefix-hash chain verify (bridge-direct `verify_prefix_blob`).

    Blob sha + line count + per-entry `sha256:` shape + sorted order — the
    same gates `_load_prefix_hashes` (in `stream_build`) enforces; the blob
    bytes here ride straight through instead of a file read. Missing bridge
    or any bridge error raises ContractError (fail-closed with a build-ext
    hint, never an oracle).
    """
    resume = _resume_native()
    if resume is None:
        raise ContractError(
            "resume bridge unavailable; run `pixi run build-ext` to build "
            f"hydra2._native before verifying prefix hashes: {ckpt}"
        )
    try:
        hashes: list[str] = resume.verify_prefix_blob(data, count, digest)
        return list(hashes)
    except ContractError:
        raise
    except (ImportError, AttributeError) as exc:
        raise ContractError(
            "resume verify_prefix_blob bridge surface missing; rebuild the bridge "
            f"(`pixi run build-ext`): {ckpt} ({type(exc).__name__}: {exc})"
        ) from exc
    except ValueError as exc:
        raise ContractError(f"checkpoint prefix_hashes digest mismatch: {ckpt}") from exc
    except Exception as exc:
        raise ContractError(
            f"resume verify_prefix_blob failed: {ckpt} ({type(exc).__name__}: {exc})"
        ) from exc


def _apply_resume_payload(
    *,
    loop: SupervisedLoop,
    dataset: _StreamDataset,
    payload: Any,
    ckpt: Path,
) -> None:
    """Apply a verified payload to live runtime objects (post-verify mutate).

    RNG-restore entry: torch CPU/CUDA RNG state restores via
    `_restore_rng_state` (Python, torch-owned — no Rust GPU math); the
    sidecar RNG anchors were already compared pre-mutate by
    `_verify_rng_anchors` (Python per `resume.rs`). Mismatch anywhere above
    raises before any object mutates — never a silent resume.
    """
    if not isinstance(payload, dict):
        raise ContractError(f"checkpoint payload must be a mapping: {ckpt}")
    for key in ("model_state", "optimizer_state", "training_state", "loss_history"):
        if key not in payload:
            raise ContractError(f"checkpoint payload missing {key!r}: {ckpt}")
    loop.model.load_state_dict(payload["model_state"])
    loop.optimizer.load_state_dict(payload["optimizer_state"])
    scheduler_state: dict[str, Any] | None = payload.get("scheduler_state")
    if (
        loop.scheduler is not None
        and scheduler_state is not None
        and len(scheduler_state) > 0  # {}-skip: writers store {} for scheduler-less runs
    ):
        try:
            loop.scheduler.load_state_dict(scheduler_state)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ContractError(f"scheduler state incompatible: {exc}") from exc
    raw_training: dict[str, Any] = payload["training_state"]
    if not isinstance(raw_training, dict):
        raise ContractError(f"checkpoint training_state malformed: {ckpt}")
    _restored = TrainingState.from_dict(raw_training)
    if _restored.precision != loop.config.precision:
        raise ContractError(
            f"checkpoint precision {_restored.precision!r} != "
            f"loop precision {loop.config.precision!r}; refusing cross-regime resume: {ckpt}"
        )
    loop.state = _restored
    loop.state.sampler_cursor = dataset.get_sampler_state()
    history: list[dict[str, Any]] = payload["loss_history"]
    if not isinstance(history, list):
        raise ContractError(f"checkpoint loss_history malformed: {ckpt}")
    loop.loss_history = [dict(entry) for entry in history]
    from hydra2.runtime.checkpoint import _restore_rng_state

    rng_state: Mapping[str, Any] | None = payload.get("rng_state")
    if rng_state is not None:
        _restore_rng_state(rng_state)
    loop.model.to(loop.device)
    loop.model.train()


class _GatedOverlapFeed:
    """Caller-owned gated overlap feed for :class:`SupervisedLoop` (driver lifecycle).

    Duck-types the ``pinned_ring.py`` contract (``next``/``stats``/``close``);
    the loop never opens or closes the handle. Schema-field tensors under
    ``batch["actor_batch"].features`` ride the single caller-owned
    :class:`PinnedRing` (fixed max-bucket-T layout, exact shape match, so
    batches stay byte-identical); every other leaf — folded ``features`` /
    ``legal_mask`` / ``chosen_action_id``, ``_``-prefixed passthroughs,
    joined oracle targets, metadata — moves synchronously. A batch whose
    history bucket differs from the ring ``T`` falls back to the synchronous
    move for its schema leaves (still byte-identical, counted in ``stats``).
    The input batch is never mutated.
    """

    def __init__(self, ring: Any, device: Any) -> None:
        self._ring = ring
        self._device = torch.device(device)
        self._batches = 0
        self._fallback_batches = 0

    def _sync_tensor(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self._device, non_blocking=True)
        return value

    def _sync_packed(self, packed: Any) -> Any:
        # Packed stream bypasses the fixed-layout ring (variable total):
        # small tensors ride the synchronous H2D while the bulk overlaps.
        if packed is None:
            return None
        from hydra2.models.encoder import PackedHistories

        return PackedHistories(
            packed_kind=self._sync_tensor(packed.packed_kind),
            cu_seqlens=self._sync_tensor(packed.cu_seqlens),
            row_lengths=tuple(packed.row_lengths),
            max_len=int(packed.max_len),
        )

    def _sync_value(self, value: Any) -> Any:
        from hydra2.models.encoder import ActorTensorBatch

        if isinstance(value, torch.Tensor):
            return value.to(self._device, non_blocking=True)
        if isinstance(value, dict):
            return {
                sub_key: (
                    sub_value.to(self._device, non_blocking=True)
                    if isinstance(sub_value, torch.Tensor)
                    else sub_value
                )
                for sub_key, sub_value in value.items()
            }
        if isinstance(value, ActorTensorBatch):
            moved_features = {
                name: self._sync_tensor(tensor) for name, tensor in dict(value.features).items()
            }
            return ActorTensorBatch(
                features=moved_features,
                history_mask=moved_features["history_mask"],
                legal_mask=moved_features["legal_mask"],
                observation_hashes=value.observation_hashes,
                actor_seats=moved_features["actor_seats"],
                packed=self._sync_packed(getattr(value, "packed", None)),
            )
        return value

    def next(self, cpu_batch: Mapping[str, Any]) -> dict[str, Any]:
        """Move one CPU batch host-to-device (ring fast path + sync remainder)."""
        from hydra2.models.encoder import ActorTensorBatch

        self._batches += 1
        actor_batch = cpu_batch.get("actor_batch")
        moved_actor: Any | None = None
        if isinstance(actor_batch, ActorTensorBatch):
            schema = dict(actor_batch.features)
            try:
                ring_out: dict[str, torch.Tensor] = self._ring.next(schema)
            except ContractError:
                # Off-bucket batch (T != ring T): synchronous move, same bytes.
                self._fallback_batches += 1
                ring_out = {name: self._sync_tensor(tensor) for name, tensor in schema.items()}
            moved_actor = ActorTensorBatch(
                features=ring_out,
                history_mask=ring_out["history_mask"],
                legal_mask=ring_out["legal_mask"],
                observation_hashes=actor_batch.observation_hashes,
                actor_seats=ring_out["actor_seats"],
                packed=self._sync_packed(getattr(actor_batch, "packed", None)),
            )
        moved: dict[str, Any] = {}
        for key, value in dict(cpu_batch).items():
            if key == "actor_batch" and moved_actor is not None:
                moved[key] = moved_actor
            elif key.startswith("_"):
                moved[key] = value
            else:
                moved[key] = self._sync_value(value)
        return moved

    def stats(self) -> dict[str, Any]:
        """Ring counters plus feed-level batch/fallback counts."""
        ring_stats: dict[str, Any] = self._ring.stats()
        base: dict[str, Any] = dict(ring_stats)
        base["feed_batches"] = self._batches
        base["feed_fallback_batches"] = self._fallback_batches
        return base

    def close(self) -> None:
        """Release the ring (idempotent, single lifecycle)."""
        self._ring.close()


def _open_gated_feed(
    *, microbatch: int, action_count: int, device: Any, depth: int = 3
) -> _GatedOverlapFeed | None:
    """Open the caller-owned overlap feed, or ``None`` for the sync fallback.

    Gated opt-in: CUDA target with CUDA available and pinnable host memory
    only; anything else (CPU runs, CPU-only tests, failed alloc) returns
    ``None`` and the loop keeps its synchronous path (queue_wait stays 0.0).
    Depth must cover one accumulation window plus one spare overlap slot;
    the caller sizes it as ``accumulation_steps + 2`` (accum 1 keeps the
    historical depth 3 exactly). Undersized rings stall on slot recycle
    (visible as sustained queue_wait_ms per the telemetry scaling rule).
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 2:
        raise ContractError(f"gated feed ring depth must be an int >= 2, got {depth!r}")
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError, ValueError):
        return None
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        from hydra2.models.schema import HISTORY_BUCKET_LENGTHS
        from hydra2.training.pinned_ring import PinnedRing, slot_layout

        layout = slot_layout(microbatch, max(HISTORY_BUCKET_LENGTHS), action_count=action_count)
        ring = PinnedRing.open(layout, depth=depth, device=resolved)
    except Exception:
        return None
    return _GatedOverlapFeed(ring, resolved)


#: Bridge leaf for the backward-shim derivation (single source:
#: ``hydra2._native.contracts`` ``backward_autocast_for``; the stale-.so
#: fallback is the byte-identical oracle below).
_backward_autocast_bridge = getattr(_resume_cfg_bridge, "backward_autocast_for", None)


def _backward_pass_autocast_for(*, precision: str, compile_mode: str) -> Literal["off"] | None:
    """Derive the functorch backward shim for a runtime precision/compile pair.

    Mirrors the fail-closed gate in :func:`protocol.build_runtime`: compiled
    non-fp32 requires ``'off'``; every other pair keeps ``None`` so existing
    runtime identities are byte-identical. Pure derivation, no hardware touch.
    """
    if _backward_autocast_bridge is not None:
        try:
            return _backward_autocast_bridge(precision, compile_mode)
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
    if precision != "fp32" and compile_mode != "eager":
        return "off"
    return None


def _profile_single_update(*, loop: Any, update: int, profiler_dir: Path, train_log: Path) -> None:
    """Train exactly one update under torch.profiler; warn-only on failure.

    Observer-only: the trace lands under ``profiler/capture-<update>/`` and
    is never read back into training, checkpoints, or resume.
    """
    capture_dir = profiler_dir / f"capture-{update:06d}"
    try:
        import gzip
        import shutil

        import torch

        capture_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = capture_dir / "trace.json"
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities, record_shapes=False, with_stack=False, profile_memory=False
        ) as prof:
            loop.train(max_updates=1)
        prof.export_chrome_trace(str(tmp_path))
        with open(tmp_path, "rb") as src, gzip.open(capture_dir / "trace.json.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        with contextlib.suppress(Exception):
            tmp_path.unlink()
        with train_log.open("a", encoding="utf-8") as handle:
            _ = handle.write(f"profiler:update={update:06d} captured {capture_dir.name}\n")
    except Exception as exc:
        with contextlib.suppress(Exception), train_log.open("a", encoding="utf-8") as handle:
            _ = handle.write(f"profiler:update={update:06d} skipped ({type(exc).__name__})\n")


def _train_segment(
    *,
    loop: Any,
    done: int,
    step: int,
    captures: set[int],
    profiler_dir: Path,
    train_log: Path,
    cuda_ok: bool,
) -> int:
    """Train ``step`` updates from ``done``; capture updates go via profiler.

    Pure stepping helper (no math): plain updates train untouched, and each
    capture update trains exactly once — under the profiler on CUDA, plain
    with a log marker elsewhere. Returns the new ``done`` cursor.
    """
    cursor = done
    target = done + step
    for capture in sorted(captures):
        if not (cursor < capture <= target):
            continue
        plain = capture - 1 - cursor
        if plain > 0:
            loop.train(max_updates=plain)
            cursor += plain
        if cuda_ok:
            _profile_single_update(
                loop=loop, update=capture, profiler_dir=profiler_dir, train_log=train_log
            )
        else:
            with contextlib.suppress(Exception), train_log.open("a", encoding="utf-8") as handle:
                _ = handle.write(f"profiler:update={capture:06d} skipped (cpu-device)\n")
            loop.train(max_updates=1)
        cursor += 1
    rest = target - cursor
    if rest > 0:
        loop.train(max_updates=rest)
        cursor += rest
    return cursor
