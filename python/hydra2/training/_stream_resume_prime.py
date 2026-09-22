"""Resume prime snapshots and envelopes (split from stream_resume; single
home for the prime snapshot path).

Single home for `_capture_prime_snapshot`, `_prime_cache_path`,
`_save_prime_snapshot`, `_load_prime_snapshot`, `_ResumeEnvelope`, and
`_load_resume_envelope`. Bridge-direct cursor and buffer gates stay in
`stream_resume`; this file imports them one way with no cycle. Failure mode
is miss-or-raise (cache miss returns None, sidecar mismatch raises
`ContractError`), never silent resume.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra2.artifacts.atomic import atomic_replace_bytes as atomic_replace_bytes
from hydra2.artifacts.digest import sha256_digest as sha256_digest
from hydra2.contracts.common import ContractError as ContractError
from hydra2.training.stream_resume import _check_buffer_entries as _check_buffer_entries
from hydra2.training.stream_resume import _pack_cursor as _pack_cursor
from hydra2.training.stream_resume import _unpack_cursor as _unpack_cursor

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hydra2.data.stream_manifest import StreamManifest as StreamManifest
    from hydra2.data.stream_read import StreamCursor as DataStreamCursor
    from hydra2.training._rc_sections import ResumePlan as ResumePlan


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
    from hydra2.data.stream_manifest import write_reservoir_blob

    if buffer_size <= 0:
        return None
    stream = getattr(dataset, "_stream", None)
    if stream is None:
        return None
    snap: tuple[list[dict[str, object]], dict[str, object]] | None = stream.shuffle_snapshot()
    if snap is None:
        return None
    snapshot: tuple[list[dict[str, object]], dict[str, object]] = snap
    entries, rng_state = snapshot
    if len(entries) == 0:
        return None
    # LIVE-STATE OWNERSHIP: same hazard as the resume path — the fill thread
    # mutates _active_buf while we read it. Main-thread snapshot barrier
    # only (call sites: post-eager-join); the fill thread is joined there.
    active: Any | None = getattr(stream, "_active_buf", None)
    buf: list[Any] = list(active) if active is not None else []
    raws: list[bytes] = [bytes(game.raw) for game in buf]  # pyrefly: ignore[unknown-argument-type] # live buffered games dynamic; bytes owns the copy
    if len(raws) != len(entries):
        return None
    live_entries = [
        {
            "key": game.game.raw_bytes_sha256,
            "path": game.path.as_posix(),
            "offset": game.offset,
        }
        for game in buf
    ]
    if [str(e.get("key")) for e in entries] != [e["key"] for e in live_entries]:
        return None
    live_cursor: DataStreamCursor = stream.cursor()
    live_prefix: list[str] = stream.prefix_hashes_snapshot()
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
    cache_dir: str | None = os.environ.get("HYDRA2_SCAN_CACHE_DIR")
    base_raw: str = (
        cache_dir
        if cache_dir is not None
        else os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "hydra2", "scan"
        )
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
        raw_entries: list[dict[str, Any]] | None = sidecar["dataset_buffer"].get("entries")
        if raw_entries is not None:
            _ = _check_buffer_entries(raw_entries, ckpt=resume.checkpoint)
    return _ResumeEnvelope(
        sidecar=sidecar,
        blob=blob,
        start_epoch=epoch_raw,
        drain_microbatches=drain_raw,
        has_fast_path=has_fast,
    )
