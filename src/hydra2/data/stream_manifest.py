"""Stream manifest: partition vocabulary, file list, digests, and caches.

Owns the scan-plane vocabulary shared by the reader and the training
driver: partition order/grouping constants, the ``FileEntry``/``StreamManifest``
records with the sha256-hex ordering builder, the byte-identical manifest
digest, the reservoir-blob snapshot format, the scan-cache envelope, the
shuffle-RNG codec, and the partition/identity math. Anything that must stay
byte-identical across checkpoints and scan caches lives here.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import zstandard as zstd

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError, CorruptArtifactError

if TYPE_CHECKING:
    import random
    from collections.abc import Mapping

__all__ = [
    "DEFAULT_GROUPING_KEYS",
    "PARTITION_ORDER",
    "RESERVOIR_BLOB_VERSION",
    "SCAN_CACHE_VERSION",
    "FileEntry",
    "SplitName",
    "StreamManifest",
    "build_manifest",
    "load_scan_cache",
    "manifest_digest",
    "parse_shuffle_rng",
    "read_reservoir_blob",
    "save_scan_cache",
    "scan_cache_path",
    "serialize_shuffle_rng",
    "write_reservoir_blob",
]

#: Partition names in :mod:`partition.py` cumulative-threshold order.
PARTITION_ORDER: tuple[str, ...] = ("train", "validation", "test", "decision_eval", "block_eval")

#: Default grouping; mirrors ``grouping_keys=("source", "time")``.
DEFAULT_GROUPING_KEYS: tuple[str, ...] = ("source", "time")

SplitName = Literal["train", "validation", "test", "decision_eval", "block_eval"]


@dataclass(frozen=True, slots=True)
class FileEntry:
    """One corpus file: path + compressed bytes; counts populated by scan pass."""

    path: Path
    bytes: int
    game_count: int | None = None
    wall_hashes: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class StreamManifest:
    """Deterministic file list; order is sha256-hex of the relative path."""

    files: tuple[FileEntry, ...]
    root: Path

    def __len__(self) -> int:
        return len(self.files)

    def paths(self) -> tuple[Path, ...]:
        """File paths in stream order."""
        return tuple(entry.path for entry in self.files)


def build_manifest(root: Path | str, pattern: str = "*.mjai.json.zst") -> StreamManifest:
    """Collect files under ``root`` in sha256-hex relative-path order.

    The hash orders (never splits): split assignment uses
    :func:`assign_split` exclusively.
    """
    base = Path(root)
    if not base.is_dir():
        raise ContractError(f"stream root not a directory: {base}")
    found = [path for path in base.rglob(pattern) if path.is_file()]

    def _order_key(path: Path) -> str:
        return hashlib.sha256(path.relative_to(base).as_posix().encode()).hexdigest()

    found.sort(key=_order_key)
    entries = tuple(FileEntry(path=path, bytes=path.stat().st_size) for path in found)
    return StreamManifest(files=entries, root=base)


def manifest_digest(manifest: StreamManifest) -> str:
    """Bind the file list for RunSpec provenance.

    Byte-identical to ``sha256(canonical_bytes([{bytes, path}, ...]))`` by
    construction in both paths below (the canonical list encoding is ``[``
    + comma-joined elements + ``]``; element key order is ``bytes`` <
    ``path`` by UTF-16BE). Fast path assembles the same bytes directly
    when every path is escape-free ASCII (single C-speed scan proves it);
    anything else takes the element-wise canonical path. Any divergence
    breaks scan-cache keys loudly (miss → full scan), never silently.
    """

    files = manifest.files
    paths = [entry.path.as_posix() for entry in files]
    if paths and re.search(r"[^\x20\x21\x23-\x5b\x5d-\x7e]", "".join(paths)) is None:
        # Escape-free: no `"`, `\`, controls, or non-ASCII anywhere, so raw
        # interpolation equals the canonical string encoding on every path
        # (only `"`/`\` are escaped in ASCII; controls/non-ASCII take the
        # element-wise path below).
        h = hashlib.sha256()
        h.update(b"[")
        for i, entry in enumerate(files):
            if i:
                h.update(b",")
            h.update(b'{"bytes":')
            h.update(str(entry.bytes).encode("ascii"))
            h.update(b',"path":"')
            h.update(paths[i].encode("ascii"))
            h.update(b'"}')
        h.update(b"]")
        return "sha256:" + h.hexdigest()
    h = hashlib.sha256()
    h.update(b"[")
    for i, entry in enumerate(files):
        if i:
            h.update(b",")
        h.update(canonical_bytes({"bytes": entry.bytes, "path": entry.path.as_posix()}))
    h.update(b"]")
    return "sha256:" + h.hexdigest()


#: Reservoir-blob layout version (bump on format change; mismatch → miss).
RESERVOIR_BLOB_VERSION = 1
_RESERVOIR_MAGIC = b"HYDRARS1"


def write_reservoir_blob(
    raws: list[bytes], path: Path | str, *, level: int = 1
) -> dict[str, object]:
    """Write length-prefixed per-game zstd frames for a shuffle buffer.

    Layout: magic(8) + version u32le + count u32le, then per game
    ``[u32le frame-len][frame]`` in buffer order. Returns the index record
    ``{version, count, uncompressed_bytes, blob_sha256}``. Corrupt/truncated
    blobs fail closed in :func:`read_reservoir_blob` (miss, never partial).
    """
    out_path = Path(path)
    cctx = zstd.ZstdCompressor(level=level)
    h = hashlib.sha256()
    uncompressed = 0
    with open(out_path, "wb") as fh:
        header = _RESERVOIR_MAGIC + struct.pack("<II", RESERVOIR_BLOB_VERSION, len(raws))
        fh.write(header)
        h.update(header)
        for raw in raws:
            frame = cctx.compress(raw)
            fh.write(struct.pack("<I", len(frame)))
            fh.write(frame)
            h.update(struct.pack("<I", len(frame)))
            h.update(frame)
            uncompressed += len(raw)
    return {
        "version": RESERVOIR_BLOB_VERSION,
        "count": len(raws),
        "uncompressed_bytes": uncompressed,
        "blob_sha256": "sha256:" + h.hexdigest(),
    }


def read_reservoir_blob(path: Path | str) -> list[bytes]:
    """Read + decompress a reservoir blob into per-game raw bytes (in order).

    Raises :class:`ContractError` on any corruption (callers treat as a
    snapshot miss and fall back to the normal fill). Transient peak is the
    decompressed buffer (~600MB for a full 10k prime); freed after decode.
    """
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ContractError(f"reservoir blob unreadable: {path} ({exc})") from exc
    try:
        dctx = zstd.ZstdDecompressor()
        off = 0
        if data[off : off + 8] != _RESERVOIR_MAGIC:
            raise ContractError(f"reservoir blob bad magic: {path}")
        off += 8
        (version, count) = struct.unpack_from("<II", data, off)
        off += 8
        if version != RESERVOIR_BLOB_VERSION:
            raise ContractError(
                f"reservoir blob version {version} != {RESERVOIR_BLOB_VERSION}: {path}"
            )
        raws: list[bytes] = []
        for _ in range(count):
            (flen,) = struct.unpack_from("<I", data, off)
            off += 4
            frame = data[off : off + flen]
            if len(frame) != flen:
                raise ContractError(f"reservoir blob truncated: {path}")
            raws.append(dctx.decompress(frame))
            off += flen
        if off != len(data):
            raise ContractError(f"reservoir blob trailing bytes: {path}")
    except (ContractError, CorruptArtifactError):
        raise
    except Exception as exc:
        raise ContractError(f"reservoir blob corrupt: {path} ({exc})") from exc
    return raws


#: Scan-cache envelope version (bump on schema change; mismatch → miss).
SCAN_CACHE_VERSION = 1


def scan_cache_path(run_dir: Path | str) -> Path:
    """Cache file for the manifest/split pre-scan under ``run_dir``."""
    return Path(run_dir) / "cache" / "scan-cache.json"


def _ratios_match(cached: object, expected: Mapping[str, float]) -> bool:
    """Exact-ish ratio comparison (JSON round-trip safe, fail-closed to miss).

    1e-9 tolerates JSON round-trip without admitting a different split;
    bool is excluded because bool is an int subclass (True == 1 would pass
    a count check).
    """
    if not isinstance(cached, dict):
        return False
    if set(cached) != set(expected):
        return False
    for key, value in expected.items():
        raw = cached.get(key)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return False
        try:
            if abs(float(raw) - value) > 1e-9:
                return False
        except (TypeError, ValueError):
            return False
    return True


def load_scan_cache(
    path: Path | str,
    *,
    manifest_digest: str,
    seed: int,
    ratios: Mapping[str, float],
    train_split: str,
    val_split: str,
) -> dict[str, object] | None:
    """Load a cached scan report on exact key match, else ``None`` (miss).
    Corrupt/stale entries fail closed to miss (full scan), never raise.
    """
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
    try:
        version: object = raw.get("version", -1)
        if version != SCAN_CACHE_VERSION:
            return None
        if raw.get("manifest_digest") != manifest_digest:
            return None
        if raw.get("data_seed") != seed:
            return None
        if raw.get("train_split") != train_split or raw.get("val_split") != val_split:
            return None
        if not _ratios_match(raw.get("ratios"), ratios):
            return None
        scan = raw.get("scan")
        if not isinstance(scan, dict):
            return None
        train_walls = scan.get("train_walls")
        val_walls = scan.get("val_walls")
        if not isinstance(train_walls, list) or not isinstance(val_walls, list):
            return None
        if any(not isinstance(w, str) for w in train_walls):
            return None
        if any(not isinstance(w, str) for w in val_walls):
            return None
        # Re-checks disjointness on load: a cache written before a wall fix
        # must not resurrect cross-split walls — fail closed to miss.
        if len(set(train_walls) & set(val_walls)) > 0:
            return None
        counts: dict[str, object] = {}
        for key in (
            "train_games",
            "val_games",
            "train_sim_games",
            "val_sim_games",
            "framed",
            "emitted",
            "quarantined",
            "duplicates",
        ):
            value = scan.get(key)
            # bool excluded (isinstance(True, int)): counts are true ints.
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return None
            counts[key] = value
        return {
            "train_walls": sorted(train_walls),
            "val_walls": sorted(val_walls),
            **counts,
        }
    except (TypeError, ValueError, AttributeError):
        return None


def save_scan_cache(
    path: Path | str,
    *,
    manifest_digest: str,
    seed: int,
    ratios: Mapping[str, float],
    train_split: str,
    val_split: str,
    scan: Mapping[str, object],
) -> None:
    """Best-effort atomic cache write (never fails training on I/O error).

    tmp+replace publishes atomically: a crash leaves old or new, never
    torn. Sorted walls make the payload deterministic; every failure
    returns silently because cache is advisory.
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SCAN_CACHE_VERSION,
            "manifest_digest": manifest_digest,
            "data_seed": seed,
            "ratios": dict(ratios),
            "train_split": train_split,
            "val_split": val_split,
            "scan": {
                # reason: type arg-type on scan payload object; sorted/int
                # re-validated by isinstance guards on load, fail-closed miss.
                "train_walls": sorted(scan["train_walls"]),  # type: ignore[arg-type]
                "val_walls": sorted(scan["val_walls"]),  # type: ignore[arg-type]
                "train_games": int(scan["train_games"]),  # type: ignore[arg-type]
                "val_games": int(scan["val_games"]),  # type: ignore[arg-type]
                "train_sim_games": int(scan["train_sim_games"]),  # type: ignore[arg-type]
                "val_sim_games": int(scan["val_sim_games"]),  # type: ignore[arg-type]
                "framed": int(scan["framed"]),  # type: ignore[arg-type]
                "emitted": int(scan["emitted"]),  # type: ignore[arg-type]
                "quarantined": int(scan["quarantined"]),  # type: ignore[arg-type]
                "duplicates": int(scan["duplicates"]),  # type: ignore[arg-type]
            },
        }
        tmp = target.with_suffix(".tmp")
        # Atomic cache write/publish is the effect; counts/paths discarded.
        _ = tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        _ = tmp.replace(target)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return


def serialize_shuffle_rng(rng: random.Random) -> dict[str, object]:
    """JSON-safe snapshot of a shuffle ``random.Random`` (fail-closed on misuse).

    Stores the version+internal+gauss_next getstate triple; misuse fails
    closed via ContractError so a corrupt sidecar resumes from origin.
    """
    state = rng.getstate()
    version, internal, gauss_next = state[0], state[1], state[2]
    if not isinstance(version, int) or not isinstance(internal, tuple):
        raise ContractError("shuffle RNG state malformed")
    if gauss_next is not None and not isinstance(gauss_next, float):
        raise ContractError("shuffle RNG gauss state malformed")
    internal_seq: tuple[int, ...] = internal
    return {"version": version, "state": list(internal_seq), "gauss_next": gauss_next}


def parse_shuffle_rng(raw: object) -> tuple[int, tuple[int, ...], float | None]:
    """Inverse of :func:`serialize_shuffle_rng`; raises on any mismatch."""
    if not isinstance(raw, dict):
        raise ContractError("shuffle buffer_rng_state must be a mapping")
    version = raw.get("version")
    internal = raw.get("state")
    gauss_next = raw.get("gauss_next")
    raw_map: dict[str, object] = raw
    unknown = sorted(k for k in raw_map if k not in ("version", "state", "gauss_next"))
    if len(unknown) > 0:
        raise ContractError(f"shuffle buffer_rng_state unknown keys {unknown}")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ContractError("shuffle buffer_rng_state.version must be an int")
    if not isinstance(internal, list) or len(internal) == 0:
        raise ContractError("shuffle buffer_rng_state.state must be a non-empty int list")
    for value in internal:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError("shuffle buffer_rng_state.state must hold non-negative ints")
    if gauss_next is not None and (
        isinstance(gauss_next, bool) or not isinstance(gauss_next, (int, float))
    ):
        raise ContractError("shuffle buffer_rng_state.gauss_next must be a float or null")
    gauss: float | None = None if gauss_next is None else float(gauss_next)
    internal_list: list[int] = internal
    return (version, tuple(internal_list), gauss)
