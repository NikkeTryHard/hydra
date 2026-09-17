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
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from hydra2.artifacts.atomic import atomic_replace_bytes as atomic_replace_bytes
from hydra2.artifacts.digest import sha256_digest as sha256_digest
from hydra2.contracts.common import ContractError as ContractError
from hydra2.data.stream import StreamCursor as DataStreamCursor
from hydra2.training.loop_state import TrainingState as TrainingState

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hydra2.data.stream import StreamManifest as StreamManifest
    from hydra2.training._rc_sections import ResumePlan as ResumePlan
    from hydra2.training.loop_train import SupervisedLoop as SupervisedLoop
    from hydra2.training.stream_dataset_buffer import _StreamDataset as _StreamDataset

__all__ = [
    "_GatedOverlapFeed",
    "_ResumeEnvelope",
    "_apply_resume_payload",
    "_backward_pass_autocast_for",
    "_capture_prime_snapshot",
    "_check_buffer_entries",
    "_load_prime_snapshot",
    "_load_resume_envelope",
    "_open_gated_feed",
    "_pack_cursor",
    "_prime_cache_path",
    "_profile_single_update",
    "_save_prime_snapshot",
    "_train_segment",
    "_unpack_cursor",
    "_verify_prefix_blob",
]


# ---------------------------------------------------------------------------
# Bridge-resume glue (resume/RNG-restore entries only; torch owns GPU/RNG).
# ---------------------------------------------------------------------------
#
# The bridge (`hydra2_replay_rs.resume`, Rust) owns the pure resume-side
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
            "hydra2_replay_rs before packing a cursor"
        )
    try:
        packed = resume.pack_cursor(
            file_index=int(cursor.file_index),
            byte_offset=int(cursor.byte_offset),
            games_seen=int(cursor.games_seen),
            seed=int(cursor.seed),
            epoch=int(cursor.epoch),
            shuffle_pos=int(cursor.shuffle_pos),
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
            "hydra2_replay_rs before unpacking a cursor"
        )
    try:
        fields = resume.unpack_cursor(dict(raw))  # type: ignore[arg-type]
        file_index, byte_offset, games_seen, seed, epoch, shuffle_pos = (
            int(fields[0]),
            int(fields[1]),
            int(fields[2]),
            int(fields[3]),
            int(fields[4]),
            int(fields[5]),
        )
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
            f"hydra2_replay_rs before checking buffer entries: {ckpt}"
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
            f"hydra2_replay_rs before verifying prefix hashes: {ckpt}"
        )
    try:
        return list(resume.verify_prefix_blob(bytes(data), int(count), str(digest)))
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
    _ = loop.model.load_state_dict(payload["model_state"])
    _ = loop.optimizer.load_state_dict(payload["optimizer_state"])
    scheduler_state = payload.get("scheduler_state")
    if (
        loop.scheduler is not None
        and scheduler_state is not None
        and len(scheduler_state) > 0  # {}-skip: writers store {} for scheduler-less runs
    ):
        try:
            loop.scheduler.load_state_dict(scheduler_state)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ContractError(f"scheduler state incompatible: {exc}") from exc
    raw_training = payload["training_state"]
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
    history = payload["loss_history"]
    if not isinstance(history, list):
        raise ContractError(f"checkpoint loss_history malformed: {ckpt}")
    loop.loss_history = [dict(entry) for entry in history]
    from hydra2.runtime.checkpoint import _restore_rng_state

    rng_state = payload.get("rng_state")
    if rng_state is not None:
        _restore_rng_state(rng_state)
    _ = loop.model.to(loop.device)
    _ = loop.model.train()


def _capture_prime_snapshot(
    *,
    dataset: Any,
    index_path: Path,
    stream_digest: str,
    buffer_size: int,
) -> Path | None:
    """Capture the post-join reservoir into a snapshot (fresh runs only).

    Reads the live stream's buffer entries + RNG (post-eager-join: the full
    prime buffer, verbatim order), pulls the matching raw bytes out of the
    live buffered games keyed by entry order, writes the zstd blob beside
    the index, and saves the index atomically. Miss-on-anything: returns
    the blob path on success, None on any anomaly (normal fill continues).
    """
    from hydra2.data.stream import write_reservoir_blob

    if buffer_size <= 0:
        return None
    stream = getattr(dataset, "_stream", None)
    if stream is None:
        return None
    snap = stream.shuffle_snapshot()
    if snap is None:
        return None
    entries, rng_state = snap
    if len(entries) == 0:
        return None
    # LIVE-STATE OWNERSHIP: same hazard as the resume path — the fill thread
    # mutates _active_buf while we read it. Main-thread snapshot barrier
    # only (call sites: post-eager-join); the fill thread is joined there.
    buf = list(getattr(stream, "_active_buf", None) or [])
    raws = [bytes(game.raw) for game in buf]
    if len(raws) != len(entries):
        return None
    live_entries = [
        {
            "key": str(game.game.raw_bytes_sha256),
            "path": game.path.as_posix(),
            "offset": int(game.offset),
        }
        for game in buf
    ]
    if [str(e.get("key")) for e in entries] != [e["key"] for e in live_entries]:
        return None
    live_cursor = stream.cursor()
    live_prefix = stream.prefix_hashes_snapshot()
    blob_path = index_path.with_name(index_path.stem + ".zst")
    try:
        index = write_reservoir_blob(raws, blob_path)
    except OSError:
        return None
    _save_prime_snapshot(
        path=index_path,
        payload={
            "resolver": "reservoir-v1",
            "stream_digest": stream_digest,
            "buffer_entries": live_entries,
            "buffer_rng_state": rng_state,
            # Bridge-direct cursor pack (bridge `resume.pack_cursor`; missing
            # bridge raises ContractError — never an oracle mapping).
            "stream_cursor": _pack_cursor(cursor=live_cursor),
            "prefix_hashes": live_prefix,
            "blob_path": str(blob_path),
            "blob_sha256": index["blob_sha256"],
            "blob": index,
        },
    )
    return blob_path


def _prime_cache_path(
    *,
    manifest: StreamManifest,
    stream_digest: str,
    seed: int,
    ratios: dict[str, float],
    train_split: str,
    val_split: str,
    buffer_size: int,
) -> Path:
    """Machine-local reservoir-snapshot index path (same key family as scan)."""
    base_raw = os.environ.get("HYDRA2_SCAN_CACHE_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "hydra2", "scan"
    )
    stamps: list[bytes] = []
    for entry in manifest.files:
        try:
            fingerprint_stat = entry.path.stat()
            stamp = f"{fingerprint_stat.st_size}:{fingerprint_stat.st_mtime_ns}"
        except OSError:
            stamp = "missing"
        stamps.append(f"{entry.path.as_posix()}:{stamp}\n".encode())
    files_hex = str(sha256_digest(b"".join(stamps))).removeprefix("sha256:")
    fingerprint = repr(
        (
            "reservoir-v1",
            stream_digest,
            seed,
            sorted(ratios.items()),
            train_split,
            val_split,
            buffer_size,
        )
    )
    key = str(sha256_digest((fingerprint + files_hex).encode())).removeprefix("sha256:")[:32]
    return Path(base_raw) / f"reservoir-{key}.json"


def _save_prime_snapshot(*, path: Path, payload: Mapping[str, Any]) -> None:
    """Best-effort atomic snapshot-index write (never fails training on I/O)."""
    # Best-effort write: I/O failure skips the cache, the run continues warm.
    try:
        blob = json.dumps(dict(payload), indent=2, sort_keys=True).encode("utf-8")
        atomic_replace_bytes(path, blob)
    except OSError:
        pass


def _load_prime_snapshot(*, path: Path, stream_digest: str) -> dict[str, Any] | None:
    """Load a prime snapshot index on exact key match, else None (miss)."""
    # Exact-key match fails safe: a stale cache returns miss, so the run
    # re-primes instead of diverging the reservoir on reordered rows.
    try:
        raw_text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        raw: object = json.loads(raw_text)
    except ValueError:
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("stream_digest") != stream_digest:
        return None
    entries = raw.get("buffer_entries")
    rng_state = raw.get("buffer_rng_state")
    blob_path = raw.get("blob_path")
    blob_sha = raw.get("blob_sha256")
    if (
        not isinstance(entries, list)
        or len(entries) == 0
        or not isinstance(rng_state, dict)
        or not isinstance(blob_path, str)
        or not isinstance(blob_sha, str)
        or blob_sha == ""
    ):
        return None
    blob = Path(blob_path)
    try:
        data = blob.read_bytes()
    except OSError:
        return None
    if str(sha256_digest(data)) != blob_sha:
        return None
    cursor_raw = raw.get("stream_cursor")
    prefix_raw = raw.get("prefix_hashes")
    try:
        # Bridge-direct cursor unpack; rejecting input (or a missing bridge)
        # is a miss either way (fail-safe cache path: the run re-primes).
        cursor = _unpack_cursor(cursor_raw)
    except ContractError:
        return None
    # Prefix entries keep the oracle shape gates (per-entry `sha256:` form)
    # plus the sorted-order gate the sidecar path enforces — the prime index
    # carries no stored blob digest/count record, so the full chain verify
    # (`_verify_prefix_blob`) stays on the sidecar path only. Any tamper is
    # a miss, never a partial restore.
    if not isinstance(prefix_raw, list) or any(
        not isinstance(sha, str) or not sha.startswith("sha256:") or sha == "sha256:"
        for sha in prefix_raw
    ):
        return None
    if list(prefix_raw) != sorted(prefix_raw):
        return None
    return {
        "buffer_entries": entries,
        "buffer_rng_state": rng_state,
        "blob_path": blob_path,
        "stream_cursor": cursor,
        "prefix_hashes": list(prefix_raw),
    }


@dataclass(slots=True)
class _ResumeEnvelope:
    """Verified resume inputs: sidecar extras, payload bytes, drain position."""

    sidecar: dict[str, Any]
    blob: bytes
    start_epoch: int
    drain_microbatches: int
    has_fast_path: bool = False


def _load_resume_envelope(
    *, resume: ResumePlan, run_digest: str, microbatch: int
) -> _ResumeEnvelope:
    """Read and verify the checkpoint sidecar + payload bytes (no mutation).

    Resume-entry hardening: sidecar/payload identity gates below are
    fail-closed (any mismatch raises before live objects mutate — never a
    silent resume). The prefix-hash chain check is bridge-direct via the
    bridge `resume` judge (`_verify_prefix_blob` — missing bridge raises);
    the cursor/buffer-entry shape checks are bridge-direct
    (`_unpack_cursor` / `_check_buffer_entries`, no oracle — missing bridge
    raises); torch RNG restore stays Python (`_restore_rng_state` via
    `_apply_resume_payload` — no Rust GPU math). Evidence: resume blob
    parity shapes; seed derivation byte-identical.
    """
    sidecar_path = resume.checkpoint.with_suffix(".json")
    try:
        raw: Any = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError(f"checkpoint sidecar unreadable: {sidecar_path} ({exc})") from exc
    if not isinstance(raw, dict):
        raise ContractError(f"checkpoint sidecar must be a mapping: {sidecar_path}")
    sidecar: dict[str, Any] = raw
    if sidecar.get("run_digest") != run_digest:
        raise ContractError(f"checkpoint run_digest mismatch: {sidecar_path}")
    if int(sidecar.get("global_update", -1)) != resume.global_update:
        raise ContractError(f"checkpoint global_update mismatch: {sidecar_path}")
    if sidecar.get("microbatch_size") != microbatch:
        raise ContractError(
            f"checkpoint microbatch_size {sidecar.get('microbatch_size')} != "
            f"config loop.microbatch_size {microbatch}"
        )
    epoch_raw = sidecar.get("stream_epoch", 0)
    buffer_raw = sidecar.get("dataset_buffer")
    drain_raw = buffer_raw.get("microbatches_in_epoch", 0) if isinstance(buffer_raw, dict) else 0
    if (
        isinstance(epoch_raw, bool)
        or not isinstance(epoch_raw, int)
        or epoch_raw < 0
        or isinstance(drain_raw, bool)
        or not isinstance(drain_raw, int)
        or drain_raw < 0
    ):
        raise ContractError(f"checkpoint stream progress malformed: {sidecar_path}")
    try:
        blob = resume.checkpoint.read_bytes()
    except OSError as exc:
        raise ContractError(f"checkpoint unreadable: {resume.checkpoint} ({exc})") from exc
    if sidecar.get("payload_sha256") != str(sha256_digest(blob)):
        raise ContractError(f"checkpoint payload hash mismatch: {resume.checkpoint}")
    # Seek-state presence: sidecars carry ``dataset_buffer`` (whole-game tail
    # index + counters + row hash). Absent → pre-simplification sidecar →
    # hard failure at resume (no full-replay drain remains).
    has_fast = isinstance(sidecar.get("dataset_buffer"), dict)
    if has_fast:
        # Bridge-direct buffer-entry shape gate (bridge
        # `resume.check_buffer_entries`, no oracle — missing bridge raises
        # before any live object mutates). Row-hash and counter gates stay
        # downstream (row hash is recomputed at restore; counters are
        # compared in `_verify_fast_snapshots`).
        raw_entries = sidecar["dataset_buffer"].get("entries")
        if raw_entries is not None:
            _check_buffer_entries(raw_entries, ckpt=resume.checkpoint)
    return _ResumeEnvelope(
        sidecar=sidecar,
        blob=blob,
        start_epoch=epoch_raw,
        drain_microbatches=drain_raw,
        has_fast_path=has_fast,
    )


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
                ring_out = self._ring.next(schema)
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
        base = dict(self._ring.stats())
        base["feed_batches"] = self._batches
        base["feed_fallback_batches"] = self._fallback_batches
        return base

    def close(self) -> None:
        """Release the ring (idempotent, single lifecycle)."""
        self._ring.close()


def _open_gated_feed(
    *, microbatch: int, action_count: int, device: Any
) -> _GatedOverlapFeed | None:
    """Open the caller-owned overlap feed, or ``None`` for the sync fallback.

    Gated opt-in: CUDA target with CUDA available and pinnable host memory
    only; anything else (CPU runs, CPU-only tests, failed alloc) returns
    ``None`` and the loop keeps its synchronous path (queue_wait stays 0.0).
    """
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
        ring = PinnedRing.open(layout, depth=3, device=resolved)
    except Exception:
        return None
    return _GatedOverlapFeed(ring, resolved)


def _backward_pass_autocast_for(*, precision: str, compile_mode: str) -> Literal["off"] | None:
    """Derive the functorch backward shim for a runtime precision/compile pair.

    Mirrors the fail-closed gate in :func:`protocol.build_runtime`: compiled
    non-fp32 requires ``'off'``; every other pair keeps ``None`` so existing
    runtime identities are byte-identical. Pure derivation, no hardware touch.
    """
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
            _ = loop.train(max_updates=1)
        prof.export_chrome_trace(str(tmp_path))
        with open(tmp_path, "rb") as src, gzip.open(capture_dir / "trace.json.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        with contextlib.suppress(Exception):
            tmp_path.unlink()
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(f"profiler:update={update:06d} captured {capture_dir.name}\n")
    except Exception as exc:
        with contextlib.suppress(Exception), train_log.open("a", encoding="utf-8") as handle:
            handle.write(f"profiler:update={update:06d} skipped ({type(exc).__name__})\n")


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
            _ = loop.train(max_updates=plain)
            cursor += plain
        if cuda_ok:
            _profile_single_update(
                loop=loop, update=capture, profiler_dir=profiler_dir, train_log=train_log
            )
        else:
            with contextlib.suppress(Exception), train_log.open("a", encoding="utf-8") as handle:
                handle.write(f"profiler:update={capture:06d} skipped (cpu-device)\n")
            _ = loop.train(max_updates=1)
        cursor += 1
    rest = target - cursor
    if rest > 0:
        _ = loop.train(max_updates=rest)
        cursor += rest
    return cursor
