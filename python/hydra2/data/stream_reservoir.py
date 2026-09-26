"""Stream reservoir blob: shuffle-buffer snapshot format (stdlib + zstd).

Split from ``stream_manifest`` (LOC gate): the length-prefixed per-game
zstd snapshot (writer + bridge-decoded reader) lives here so the manifest
module keeps vocabulary, records, build, and digest. ``stream_manifest``
re-exports every public name, so existing import sites keep working
unchanged. Snapshots stay advisory: corruption fails closed to a miss.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import zstandard as zstd

from hydra2.artifacts.digest import sha256_file
from hydra2.contracts.common import ContractError

__all__ = [
    "RESERVOIR_BLOB_VERSION",
    "read_reservoir_blob",
    "write_reservoir_blob",
]


#: Reservoir-blob layout version (bump on format change; mismatch → miss).
RESERVOIR_BLOB_VERSION = 1
_RESERVOIR_MAGIC = b"HYDRARS1"


def _require_resume() -> Any:
    """Import the built ``resume`` bridge surface (fail closed)."""
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with resume not importable; "
            "build the bridge with `pixi run build-ext` before reading reservoir blobs"
        ) from exc
    try:
        return _ext.resume
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.resume submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def write_reservoir_blob(
    raws: list[bytes], path: Path | str, *, level: int = 1
) -> dict[str, object]:
    """Write length-prefixed per-game zstd frames for a shuffle buffer.

    Layout: magic(8) + version u32le + count u32le, then per game
    ``[u32le frame-len][frame]`` in buffer order. Returns the index record
    ``{version, count, uncompressed_bytes, blob_sha256}``. Corrupt/truncated
    blobs fail closed in :func:`read_reservoir_blob` (miss, never partial).

    Framing stays the v1 Python writer (byte-identical frames); the blob
    digest is minted by the hard-Rust digest owner (``sha256_file`` over
    the emitted file — ``ImportError`` with a ``build-ext`` hint when the
    extension is not built, NO oracle fallback). Stays off
    ``resume.encode_reservoir``: the bridge writer emits generation v2
    with different zstd frames, which would change blob bytes, the index
    ``blob_sha256``, and the ``reservoir-v1`` sidecar contract.
    """
    out_path = Path(path)
    cctx = zstd.ZstdCompressor(level=level)
    uncompressed = 0
    with open(out_path, "wb") as fh:
        header = _RESERVOIR_MAGIC + struct.pack("<II", RESERVOIR_BLOB_VERSION, len(raws))
        _ = fh.write(header)
        for raw in raws:
            frame = cctx.compress(raw)
            _ = fh.write(struct.pack("<I", len(frame)))
            _ = fh.write(frame)
            uncompressed += len(raw)
    return {
        "version": RESERVOIR_BLOB_VERSION,
        "count": len(raws),
        "uncompressed_bytes": uncompressed,
        "blob_sha256": str(sha256_file(out_path)),
    }


def read_reservoir_blob(path: Path | str) -> list[bytes]:
    """Read + decompress a reservoir blob into per-game raw bytes (in order).

    Raises :class:`ContractError` on any corruption (callers treat as a
    snapshot miss and fall back to the normal fill). Transient peak is the
    decompressed buffer (~600MB for a full 10k prime); freed after decode.

    Decode runs through ``resume.decode_reservoir`` (``ImportError`` with
    a ``build-ext`` hint when the extension is not built, NO oracle
    fallback); bridge rejects (``ValueError`` → :class:`ContractError`
    here) on bad magic, unknown version, truncated frames, and trailing
    bytes. The bridge dual-reads v1+v2 generations while this module
    still writes v1 only.
    """
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ContractError(f"reservoir blob unreadable: {path} ({exc})") from exc
    resume = _require_resume()
    try:
        raws: list[bytes] = resume.decode_reservoir(data)
    except ValueError as exc:
        raise ContractError(f"reservoir blob corrupt: {path} ({exc})") from exc
    return list(raws)
