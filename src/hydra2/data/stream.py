"""Streaming-first game reader: train straight from ``.mjai.json.zst``.

Ephemeral rows, identical contracts: every framed game passes through
:func:`decode_game_object` (strict line rules) and :func:`validate_game`
((5,) dora, tile conservation, red identity); decode failures, invalid
games, and exact-hash duplicates are quarantined and counted, never
silently skipped. Split assignment reuses the :mod:`partition.py`
``sha256(seed|group_key)`` math identically — never path hashing, never
FNV — with the default ``(source, time)`` grouping (player grouping is
attested-weak, so it is not a default).

Scale honesty (SplitScalePlan): real Tenhou MJAI carries no wall field
(wall: the 136-tile deal order), so ``wall_hash`` is null across the corpus
and wall-disjointness reduces to game-disjointness (filename-stem identity,
one game per file) plus exact decoded-hash dedup. :func:`check_wall_disjoint`
still enforces the cross-split wall predicate wherever walls exist
(synthetic fixtures). ``FileEntry.game_count`` / ``wall_hashes`` stay ``None``
here: populating them at 6.8M-file scale is the header-scan pass's job, not
the reader's.

Actor/privileged firewall: :func:`actor_payload` projects the safe subset
(no events — ``tehais`` are private hands) and
:func:`verify_no_privileged_leakage` enforces ``FORBIDDEN_IN_ACTOR`` from
:mod:`parquet.py` on any actor-bound mapping.

Determinism: file order is the sha256-hex of the relative path (uniform,
stable; ordering only — split assignment never sees the path). Shuffle
uses a ``random.Random`` keyed by ``sha256(seed|epoch)`` — counter-derived,
no wall-clock. ``StreamCursor`` records the emitted frontier; resume
replays the un-emitted tail idempotently, and shuffle resume replays the
prefix (uncounted) to rebuild buffer/RNG state, so same seed + cursor is
the same sequence. Cursor checkpoints are only consistent at emission
boundaries or origin; a shuffle fill-phase cursor raises on resume.

Throughput sizing (ThroughputPlan): the dominant downstream cost is the
per-row Python encode plus the per-event inner loop, so rows must arrive
pre-batched — :meth:`GameStream.iter_batches` yields fixed-size game
lists (one stack + ``pin_memory`` per batch downstream, per the
``dataset.py`` idiom) and :func:`slice_microbatches` cuts microbatches as
shallow slices with no re-decode. :class:`PrefetchGameStream` decodes in
background threads, yields strictly in sequence-number order (never
completion order), and counts consumer ``waits`` as the starvation signal.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import re
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context as _mp_get_context
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import zstandard as zstd

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.data.decode import GameRecord, decode_game_object, decode_json_line
from hydra2.data.parquet import FORBIDDEN_IN_ACTOR
from hydra2.data.validate import validate_game

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from concurrent.futures import Future

__all__ = [
    "DEFAULT_GROUPING_KEYS",
    "PARTITION_ORDER",
    "RESERVOIR_BLOB_VERSION",
    "FileEntry",
    "GameStream",
    "PrefetchGameStream",
    "StreamCursor",
    "StreamGame",
    "StreamManifest",
    "StreamStats",
    "ZstdLineStream",
    "actor_payload",
    "assign_split",
    "build_manifest",
    "check_wall_disjoint",
    "compute_wall_hash",
    "count_decisions",
    "fetch_game_at",
    "group_key_for",
    "group_key_for_path",
    "load_scan_cache",
    "manifest_digest",
    "parse_shuffle_rng",
    "read_reservoir_blob",
    "save_scan_cache",
    "scan_cache_path",
    "serialize_shuffle_rng",
    "stem_of",
    "verify_no_privileged_leakage",
    "write_reservoir_blob",
]

#: Partition names in :mod:`partition.py` cumulative-threshold order.
PARTITION_ORDER: tuple[str, ...] = ("train", "validation", "test", "decision_eval", "block_eval")

#: Default grouping; mirrors ``grouping_keys=("source", "time")``.
DEFAULT_GROUPING_KEYS: tuple[str, ...] = ("source", "time")

SplitName = Literal["train", "validation", "test", "decision_eval", "block_eval"]

_START_TYPES = frozenset({"start_game", "startGame", "game_start", "start"})
_END_TYPES = frozenset({"end_game", "endGame", "game_end", "end"})
_CHUNK_SIZE = 65536
_DIGIT_RUN = re.compile(r"[0-9]+")


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


def fetch_game_at(
    path: Path | str,
    game_offset: int,
    *,
    seed: int,
    ratios: Mapping[str, float],
    expected_sha: str | None = None,
) -> StreamGame:
    """Fetch one game by decompressed-byte ``game_offset`` (fail-closed).
    Verifies ``expected_sha`` (raw_bytes_sha256) when given; split assignment
    uses ``seed``/``ratios`` identically to the live stream.
    """
    from hydra2.data.decode import decode_game_object as _decode
    from hydra2.data.validate import validate_game as _validate

    fpath = Path(path)
    if type(game_offset) is not int or game_offset < 0:
        raise ContractError(f"game offset must be a non-negative int, got {game_offset!r}")
    found: bytes | None = None
    for offset, game_bytes in ZstdLineStream(fpath).iter_games():
        if offset == game_offset:
            found = game_bytes
            break
        if offset > game_offset:
            break
    if found is None:
        raise ContractError(f"game offset {game_offset} not on a game boundary: {fpath}")
    try:
        game = _decode(
            object_id=stem_of(fpath),
            packaged_object_id=stem_of(fpath),
            decoded_bytes=found,
        )
    except (ContractError, CorruptArtifactError, ValueError) as exc:
        raise ContractError(f"buffered game undecodable at {fpath}:{game_offset} ({exc})") from exc
    try:
        validation_hash: str | None = _validate(game).validation_hash
    except (ContractError, CorruptArtifactError, ValueError) as exc:
        raise ContractError(f"buffered game invalid at {fpath}:{game_offset} ({exc})") from exc
    if expected_sha is not None and game.raw_bytes_sha256 != expected_sha:
        raise ContractError(f"buffered game key mismatch at {fpath}:{game_offset}")
    wall_hash = compute_wall_hash(game)
    assigned = assign_split(group_key=group_key_for_path(fpath), seed=seed, ratios=ratios)
    return StreamGame(
        path=fpath,
        offset=game_offset,
        game=game,
        wall_hash=wall_hash,
        validation_hash=validation_hash,
        split=assigned,
        raw=found,
    )


def stem_of(path: Path) -> str:
    """Game identity stem: ``<stem>.mjai.json.zst`` -> ``<stem>``."""
    name = path.name
    for suffix in (".mjai.json.zst", ".mjai.json", ".zst"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def group_key_for(*, source: str, time: str) -> str:
    """Group key identical to partition ``(source, time)`` grouping join."""
    return f"{source}|{time}"


def group_key_for_path(path: Path) -> str:
    """Derive the ``(source, time)`` group from a corpus path.

    Source is the parent directory name (mount-independent); time is the
    leading digit run of the filename stem (Tenhou ``YYYYMMDDHH...``),
    else ``"unknown"``.
    """
    source = path.parent.name if path.parent.name not in ("", ".") else "unknown"
    match = _DIGIT_RUN.match(stem_of(path))
    time = match.group(0) if match is not None else "unknown"
    return group_key_for(source=source, time=time)


def compute_wall_hash(game: GameRecord) -> str | None:
    """Wall hash identical to partition identity; ``None`` when no wall.

    Real MJAI has no 136-list wall field, so this is null corpus-wide.
    """
    if game.wall_tiles is None:
        return None
    return "sha256:" + hashlib.sha256(canonical_bytes(list(game.wall_tiles))).hexdigest()


def assign_split(*, group_key: str, seed: int, ratios: Mapping[str, float]) -> str:
    """Assign a group to a partition; math identical to ``assign_partitions``.

    Cumulative thresholds over :data:`PARTITION_ORDER` on
    ``sha256(f"{seed}|{group_key}")`` scaled to ``[0, 1)``.
    """
    if len(ratios) == 0:
        raise ContractError("split ratios must not be empty")
    weights: dict[str, float] = {}
    for name, value in ratios.items():
        # reason: type arg-type on ratios mapping value; float() guarded
        # below, ContractError on non-numeric.
        try:
            weight = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ContractError(f"split ratio for {name!r} not numeric: {value!r}") from exc
        if weight < 0.0:
            raise ContractError(f"split ratio for {name!r} negative: {weight}")
        weights[name] = weight
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ContractError(f"split ratios must sum to 1.0, got {total}")
    cumulative: list[tuple[str, float]] = []
    running = 0.0
    for part in PARTITION_ORDER:
        if part in weights and weights[part] > 0.0:
            running += weights[part]
            cumulative.append((part, running))
    if len(cumulative) == 0:
        raise ContractError("no active partitions in ratios")
    digest = hashlib.sha256(f"{seed}|{group_key}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    for part, threshold in cumulative:
        if draw < threshold:
            return part
    return cumulative[-1][0]


@dataclass(frozen=True, slots=True)
class StreamCursor:
    """Resumable frontier: file seek + byte resume + accounting.

    ``byte_offset`` is a decompressed-byte offset (stable across zstd
    re-encodes); resume skips framed games with ``offset < byte_offset``
    (game-prefix catch-up). ``shuffle_pos`` counts shuffled emissions.
    """

    file_index: int
    byte_offset: int
    games_seen: int
    seed: int
    epoch: int
    shuffle_pos: int = 0

    def to_dict(self) -> dict[str, int]:
        """Plain mapping for YAML persistence under the artifact root."""
        return {
            "byte_offset": self.byte_offset,
            "epoch": self.epoch,
            "file_index": self.file_index,
            "games_seen": self.games_seen,
            "seed": self.seed,
            "shuffle_pos": self.shuffle_pos,
        }

    @staticmethod
    def from_dict(raw: Mapping[str, object]) -> StreamCursor:
        """Inverse of :meth:`to_dict`; rejects non-int or negative fields."""
        values: dict[str, int] = {}
        for key in ("file_index", "byte_offset", "games_seen", "seed", "epoch", "shuffle_pos"):
            value = raw.get(key)
            if type(value) is not int or value < 0:
                raise ContractError(f"cursor field {key!r} must be a non-negative int")
            values[key] = value
        return StreamCursor(
            file_index=values["file_index"],
            byte_offset=values["byte_offset"],
            games_seen=values["games_seen"],
            seed=values["seed"],
            epoch=values["epoch"],
            shuffle_pos=values["shuffle_pos"],
        )


@dataclass(frozen=True, slots=True)
class StreamGame:
    """One validated, split-assigned game: the ephemeral row."""

    path: Path
    offset: int
    game: GameRecord
    wall_hash: str | None
    validation_hash: str | None
    split: str = "train"
    #: Raw framed game bytes (verbatim decompressed payload, trailing newline
    #: included). Lets the Rust pull path walk the original bytes instead of
    #: re-serializing parsed events; ephemeral like the record itself (buffer
    #: entries refetch by path/offset, never by value).
    raw: bytes = b""


@dataclass(slots=True)
class StreamStats:
    """Run-scoped counters; fresh on every ``__iter__`` pass."""

    framed: int = 0
    emitted: int = 0
    quarantined: int = 0
    duplicates: int = 0
    skipped_split: int = 0
    waits: int = 0


class ZstdLineStream:
    """Incremental zstd frame splitter with decompressed-byte offsets.

    Yields ``(offset, game_bytes)`` framed on start/end event boundaries
    while reusing :mod:`decode.py` line rules implicitly: blank lines,
    non-object lines, and unbalanced boundaries still yield byte ranges so
    the consumer quarantines them instead of skipping silently. Offsets are
    decompressed-byte offsets of each game's first byte.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Source file."""
        return self._path

    def iter_games(self) -> Iterator[tuple[int, bytes]]:
        """Yield framed games in file order."""
        pending: list[bytes] = []
        pending_start = 0
        in_game = False

        def _peek_type(line: bytes) -> str | None:
            # Hot-path superset prefilter: every boundary name in _START_TYPES /
            # _END_TYPES contains b"start" or b"end", so lines without either
            # cannot frame and skip the parse. A \u-escaped type value would
            # defeat the substring test, so such lines fall through to the
            # exact parse below; decode_json_line stays the authority in all cases.
            if b"\\u" not in line and b"start" not in line and b"end" not in line:
                return None
            try:
                value = decode_json_line(line)
            except ValueError:
                return None
            if not isinstance(value, dict):
                return None
            found = value.get("type")
            return found if isinstance(found, str) else None

        def _feed(line: bytes, line_start: int, *, final: bool) -> list[tuple[int, bytes]]:
            nonlocal pending, pending_start, in_game
            done: list[tuple[int, bytes]] = []
            raw = line if final else line + b"\n"
            if len(line.strip()) == 0 and not final:
                if in_game:
                    pending.append(raw)
                else:
                    done.append((line_start, raw))
                return done
            peaked = _peek_type(line)
            if peaked in _START_TYPES:
                if in_game:
                    done.append((pending_start, b"".join(pending)))
                pending = [raw]
                pending_start = line_start
                in_game = True
            elif peaked in _END_TYPES:
                if in_game:
                    pending.append(raw)
                    done.append((pending_start, b"".join(pending)))
                    pending = []
                    in_game = False
                else:
                    done.append((line_start, raw))
            elif not in_game:
                pending = [raw]
                pending_start = line_start
                in_game = True
            else:
                pending.append(raw)
            return done

        decoder = zstd.ZstdDecompressor()
        try:
            handle = self._path.open("rb")
        except OSError as exc:
            raise CorruptArtifactError(f"cannot open stream file {self._path}: {exc}") from exc
        buf = bytearray()
        buf_start = 0
        try:
            with handle, decoder.stream_reader(handle) as reader:
                while True:
                    chunk = reader.read(_CHUNK_SIZE)
                    if chunk == b"":
                        break
                    buf += chunk
                    while True:
                        newline = buf.find(b"\n")
                        if newline == -1:
                            break
                        line = bytes(buf[:newline])
                        line_start = buf_start
                        del buf[: newline + 1]
                        buf_start += newline + 1
                        for item in _feed(line, line_start, final=False):
                            yield item
        except zstd.ZstdError as exc:
            raise CorruptArtifactError(f"zstd decode failed for {self._path}: {exc}") from exc
        if len(buf) > 0:
            # No trailing newline: frame the remainder so decode rejects it.
            for item in _feed(bytes(buf), buf_start, final=True):
                yield item
        if in_game and len(pending) > 0:
            yield (pending_start, b"".join(pending))


@dataclass(slots=True)
class _Run:
    """Mutable per-pass position, counters, and dedup set."""

    file_index: int
    byte_offset: int
    games_seen: int
    stats: StreamStats | None
    hashes: set[str]


@dataclass(slots=True)
class _ShuffleState:
    """Shuffle buffer + deterministic RNG."""

    buf: list[StreamGame]
    rng: random.Random


def _shuffle_key(seed: int, epoch: int) -> int:
    """Counter-derived shuffle material; no wall-clock anywhere."""
    raw = hashlib.sha256(f"{seed}|{epoch}".encode()).digest()
    return int.from_bytes(raw[:8], "big")


def _check_int(name: str, value: int) -> int:
    if type(value) is not int or value < 0:
        raise ContractError(f"{name} must be a non-negative int, got {value!r}")
    return value


class GameStream:
    """Validated, split-filtered game iterator with cursor resume.

    ``split=None`` emits every valid game (with its assignment attached);
    otherwise only that partition is emitted. ``ratios`` are explicit —
    owned by RunSpec selection, never defaulted here.
    """

    def __init__(
        self,
        manifest: StreamManifest,
        *,
        seed: int,
        ratios: Mapping[str, float],
        epoch: int = 0,
        split: str | None = None,
        shuffle_buffer: int = 0,
        start: StreamCursor | None = None,
        drop_duplicates: bool = True,
        shuffle_restore_entries: list[dict[str, object]] | None = None,
        shuffle_restore_rng: dict[str, object] | None = None,
        shuffle_restore_prefix_hashes: list[str] | tuple[str, ...] | None = None,
        snapshot_blob: Path | str | None = None,
    ) -> None:
        self._manifest = manifest
        self._seed = _check_int("seed", seed)
        self._epoch = _check_int("epoch", epoch)
        if split is not None and split not in PARTITION_ORDER:
            raise ContractError(f"unknown split {split!r}")
        self._split = split
        # Eager ratio validation: raise before any I/O on bad specs.
        # Assigned split discarded.
        _ = assign_split(group_key="__validate__", seed=self._seed, ratios=ratios)
        self._ratios = dict(ratios)
        self._shuffle_buffer = _check_int("shuffle_buffer", shuffle_buffer)
        if not isinstance(drop_duplicates, bool):
            raise ContractError("drop_duplicates must be bool")
        self._drop_duplicates = drop_duplicates
        self._file_index = 0
        self._byte_offset = 0
        self._games_seen = 0
        self._shuffle_pos = 0
        self._stats = StreamStats()
        # Live shuffle iteration state (tail phase) for checkpoint snapshots.
        # Points at the generator-owned ``_ShuffleState``/``_Run`` while the
        # epoch pass is suspended at a yield; ``None`` before first pull.
        self._active_buf: list[StreamGame] | None = None
        self._active_rng: random.Random | None = None
        self._active_run: _Run | None = None
        # Verbatim shuffle restore (fast resume): entries + RNG, no prefix replay.
        if (shuffle_restore_entries is None) != (shuffle_restore_rng is None):
            raise ContractError("shuffle restore needs both entries and RNG state")
        self._shuffle_restore_entries: list[dict[str, object]] | None = None
        self._shuffle_restore_rng: dict[str, object] | None = None
        if shuffle_restore_entries is not None and shuffle_restore_rng is not None:
            if self._shuffle_buffer <= 0:
                raise ContractError("shuffle restore requires a positive shuffle_buffer")
            if not isinstance(shuffle_restore_entries, list):
                raise ContractError("shuffle restore entries must be a list")
            if len(shuffle_restore_entries) > self._shuffle_buffer:
                raise ContractError("shuffle restore overflows buffer_size")
            for entry in shuffle_restore_entries:
                if not isinstance(entry, dict):
                    raise ContractError("shuffle restore entry must be a mapping")
                entry_map: dict[str, object] = entry
                unknown = sorted(k for k in entry_map if k not in ("key", "path", "offset"))
                if len(unknown) > 0:
                    raise ContractError(f"shuffle restore entry unknown keys {unknown}")
                key, path, offset = entry.get("key"), entry.get("path"), entry.get("offset")
                if not isinstance(key, str) or key == "":
                    raise ContractError("shuffle restore entry key must be a non-empty str")
                if not isinstance(path, str) or path == "":
                    raise ContractError("shuffle restore entry path must be a non-empty str")
                if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                    raise ContractError("shuffle restore entry offset must be non-negative int")
            # Fail fast on malformed RNG (parsed again at iter time).
            # Parsed triple discarded.
            _ = parse_shuffle_rng(shuffle_restore_rng)
            self._shuffle_restore_entries = list(shuffle_restore_entries)
            self._shuffle_restore_rng = dict(shuffle_restore_rng)
        # Reservoir snapshot blob: per-game raw bytes in entry order, replacing
        # corpus refetch in _restore_shuffle_state (same decode+validate tail).
        # Requires restore entries (unmappable bytes otherwise); entries alone
        # keep the corpus-refetch path (no blob → fetch by path/offset).
        self._snapshot_blob: Path | None = (
            Path(snapshot_blob) if snapshot_blob is not None else None
        )
        if self._snapshot_blob is not None and self._shuffle_restore_entries is None:
            raise ContractError("snapshot_blob needs shuffle restore entries")
        # Prefix dedup seed (S1 hardening): verified count+digest upstream; shape
        # checked here, seeded into the resume _Run so future membership matches
        # the full-replay prefix set even when duplicates exist. Independent of
        # the entries/RNG pair (ordered resume seeds prefix only).
        self._shuffle_restore_prefix_hashes: tuple[str, ...] | None = None
        if shuffle_restore_prefix_hashes is not None:
            if not isinstance(shuffle_restore_prefix_hashes, (list, tuple)):
                raise ContractError("shuffle restore prefix_hashes must be a list")
            for sha in shuffle_restore_prefix_hashes:
                if not isinstance(sha, str) or not sha.startswith("sha256:") or sha == "sha256:":
                    raise ContractError("shuffle restore prefix entries must be sha256 strings")
            self._shuffle_restore_prefix_hashes = tuple(shuffle_restore_prefix_hashes)
        if start is not None:
            self.skip_to(start)

    def cursor(self) -> StreamCursor:
        """Current emitted frontier for YAML persistence."""
        return StreamCursor(
            file_index=self._file_index,
            byte_offset=self._byte_offset,
            games_seen=self._games_seen,
            seed=self._seed,
            epoch=self._epoch,
            shuffle_pos=self._shuffle_pos,
        )

    @property
    def stats(self) -> StreamStats:
        """Counters for the most recent pass."""
        return self._stats

    def skip_to(self, cursor: StreamCursor) -> None:
        """Seek to a cursor; seed/epoch mismatch is a hard failure."""
        if cursor.seed != self._seed or cursor.epoch != self._epoch:
            raise ContractError("cursor seed/epoch does not match this stream")
        if cursor.file_index > len(self._manifest.files):
            raise ContractError(f"cursor file_index {cursor.file_index} beyond manifest")
        # Cursor validation is the effect; checked values discarded.
        _ = _check_int("cursor.byte_offset", cursor.byte_offset)
        _ = _check_int("cursor.games_seen", cursor.games_seen)
        _ = _check_int("cursor.shuffle_pos", cursor.shuffle_pos)
        self._file_index = cursor.file_index
        self._byte_offset = cursor.byte_offset
        self._games_seen = cursor.games_seen
        self._shuffle_pos = cursor.shuffle_pos

    def _sync_pos(self, run: _Run) -> None:
        self._file_index = run.file_index
        self._byte_offset = run.byte_offset
        self._games_seen = run.games_seen

    def _framed_from(self, run: _Run) -> Iterator[tuple[int, int, bytes]]:
        """Yield ``(file_index, end_offset, game_bytes)`` from the run position.

        Read-only over ``run``: position advances in :meth:`_finish_decode`
        (in-order processing), never on submit-ahead framing.
        """
        files = self._manifest.files
        start_file = run.file_index
        start_base = run.byte_offset
        for file_index in range(start_file, len(files)):
            entry = files[file_index]
            base = start_base if file_index == start_file else 0
            for offset, game_bytes in ZstdLineStream(entry.path).iter_games():
                if offset < base:
                    continue
                yield file_index, offset + len(game_bytes), game_bytes

    def _finish_decode(
        self,
        run: _Run,
        file_index: int,
        end: int,
        game_bytes: bytes,
        game: GameRecord | None,
        validation_hash: str | None,
        precomputed: tuple[str | None, str | None] | None = None,
    ) -> StreamGame | None:
        """Shared validate/quarantine/dedup/split tail; advances the run."""
        run.file_index = file_index
        run.byte_offset = end
        run.games_seen += 1
        stats = run.stats
        if stats is not None:
            stats.framed += 1
        if game is None or validation_hash is None:
            if stats is not None:
                stats.quarantined += 1
            return None
        if self._drop_duplicates:
            if game.raw_bytes_sha256 in run.hashes:
                if stats is not None:
                    stats.quarantined += 1
                    stats.duplicates += 1
                return None
            run.hashes.add(game.raw_bytes_sha256)
        entry = self._manifest.files[file_index]
        if precomputed is not None:
            wall_hash, assigned_maybe = precomputed
            # Worker-computed split is always str for valid games (assign_split
            # returns str; None-games return before this point). Cast pins the
            # invariant the type cannot express.
            assigned = cast("str", assigned_maybe)
        else:
            wall_hash = compute_wall_hash(game)
            assigned = assign_split(
                group_key=group_key_for_path(entry.path), seed=self._seed, ratios=self._ratios
            )
        if self._split is not None and assigned != self._split:
            if stats is not None:
                stats.skipped_split += 1
            return None
        if stats is not None:
            stats.emitted += 1
        return StreamGame(
            path=entry.path,
            offset=end - len(game_bytes),
            game=game,
            wall_hash=wall_hash,
            validation_hash=validation_hash,
            split=assigned,
            raw=game_bytes,
        )

    def _decode_inline(
        self, object_id: str, game_bytes: bytes
    ) -> tuple[GameRecord | None, str | None]:
        try:
            game = decode_game_object(
                object_id=object_id, packaged_object_id=object_id, decoded_bytes=game_bytes
            )
        except (ContractError, CorruptArtifactError, ValueError):
            return None, None
        return game, validate_game(game).validation_hash

    def _ordered_source(self, run: _Run) -> Iterator[StreamGame]:
        for file_index, end, game_bytes in self._framed_from(run):
            entry = self._manifest.files[file_index]
            game, validation_hash = self._decode_inline(stem_of(entry.path), game_bytes)
            emitted = self._finish_decode(run, file_index, end, game_bytes, game, validation_hash)
            if emitted is not None:
                yield emitted

    def _shuffled(self, run: _Run, state: _ShuffleState) -> Iterator[StreamGame]:
        """Fill-sample-replace shuffle; deterministic in seed+epoch."""
        width = self._shuffle_buffer
        for game in self._ordered_source(run):
            if len(state.buf) < width:
                state.buf.append(game)
                continue
            index = state.rng.randrange(width)
            outgoing = state.buf[index]
            state.buf[index] = game
            yield outgoing
        state.rng.shuffle(state.buf)
        while len(state.buf) > 0:
            yield state.buf.pop()

    def shuffle_snapshot(self) -> tuple[list[dict[str, object]], dict[str, object]] | None:
        """Live shuffle buffer + RNG for checkpoint sidecars (``None`` when ordered).
        Returns ``(entries, rng_state)`` with entries in buffer order; each entry
        carries deterministic ``key`` (raw_bytes_sha256) plus ``path``/``offset``
        for verbatim refetch. Empty buffer yields ``([], rng)`` (valid).
        """
        if self._shuffle_buffer <= 0:
            return None
        if self._active_buf is None or self._active_rng is None:
            rng = random.Random(_shuffle_key(self._seed, self._epoch))
            return ([], serialize_shuffle_rng(rng))
        entries: list[dict[str, object]] = []
        for game in self._active_buf:
            entries.append(
                {
                    "key": game.game.raw_bytes_sha256,
                    "path": game.path.as_posix(),
                    "offset": game.offset,
                }
            )
        return (entries, serialize_shuffle_rng(self._active_rng))

    def prefix_hashes_snapshot(self) -> list[str]:
        """Live dedup-prefix hashes for checkpoint sidecars (sorted, possibly empty).
        Sourced from the suspended pass ``_Run``; ``[]`` before first pull.
        """
        if self._active_run is None:
            return []
        return sorted(self._active_run.hashes)

    def _materialize_from_snapshot_blob(self) -> list[StreamGame]:
        """Materialize the shuffle buffer from snapshot raw bytes (no refetch).

        Reads :attr:`_snapshot_blob` (length-prefixed per-game zstd frames in
        buffer order), decodes each through the identical inline path as live
        pulls, and enforces the same gates (sha vs entry key, split match).
        Any corruption raises (the caller treats it like a fill failure, and
        the driver only selects this path after hash-verifying the snapshot).
        No stats touched and no dedup membership added here — like the
        refetch path, dedup seeds from the prefix record (done by caller).
        """
        assert self._snapshot_blob is not None
        assert self._shuffle_restore_entries is not None
        raws = read_reservoir_blob(self._snapshot_blob)
        entries = self._shuffle_restore_entries
        if len(raws) != len(entries):
            raise ContractError(f"snapshot blob holds {len(raws)} games for {len(entries)} entries")
        buf: list[StreamGame] = []
        for entry, raw in zip(entries, raws, strict=True):
            key = str(entry["key"])
            fpath = Path(str(entry["path"]))
            # reason: type arg-type on restore-entry object; int() guarded
            # by offset validation at construction, raises on misuse.
            game_offset = int(entry["offset"])  # type: ignore[arg-type]
            game, validation_hash = self._decode_inline(stem_of(fpath), raw)
            if game is None or validation_hash is None:
                raise ContractError(f"snapshot game undecodable at {fpath}:{game_offset}")
            if game.raw_bytes_sha256 != key:
                raise ContractError(f"snapshot game key mismatch at {fpath}:{game_offset}")
            wall_hash = compute_wall_hash(game)
            assigned = assign_split(
                group_key=group_key_for_path(fpath), seed=self._seed, ratios=self._ratios
            )
            if self._split is not None and assigned != self._split:
                raise ContractError("restored shuffle game split mismatch")
            buf.append(
                StreamGame(
                    path=fpath,
                    offset=game_offset,
                    game=game,
                    wall_hash=wall_hash,
                    validation_hash=validation_hash,
                    split=assigned,
                    raw=raw,
                )
            )
        return buf

    def _restore_shuffle_state(self, run: _Run) -> _ShuffleState:
        """Materialize the verbatim shuffle buffer + RNG (fast resume, no replay)."""
        assert self._shuffle_restore_entries is not None
        assert self._shuffle_restore_rng is not None
        version, internal, gauss_next = parse_shuffle_rng(self._shuffle_restore_rng)
        rng = random.Random()
        rng.setstate((version, tuple(internal), gauss_next))
        buf: list[StreamGame] = []
        if self._snapshot_blob is not None:
            buf = self._materialize_from_snapshot_blob()
        else:
            for entry in self._shuffle_restore_entries:
                key = str(entry["key"])
                fpath = Path(str(entry["path"]))
                # reason: type arg-type on restore-entry object; non-neg-int
                # validated at construction, int() raises on misuse.
                game_offset = int(entry["offset"])  # type: ignore[arg-type]
                fetched = fetch_game_at(
                    fpath, game_offset, seed=self._seed, ratios=self._ratios, expected_sha=key
                )
                if self._split is not None and fetched.split != self._split:
                    raise ContractError("restored shuffle game split mismatch")
                buf.append(fetched)
        # Dedup set seeds from the verified prefix record (count+digest checked
        # upstream before construction), so future membership matches the
        # full-replay prefix set even when duplicates exist.
        saved_prefix = self._shuffle_restore_prefix_hashes
        prefix: set[str] = set(saved_prefix if saved_prefix is not None else ())
        run.hashes.update(prefix)
        return _ShuffleState(buf=buf, rng=rng)

    def __iter__(self) -> Iterator[StreamGame]:
        stats = StreamStats()
        self._stats = stats
        if self._shuffle_buffer <= 0:
            if self._shuffle_restore_entries is not None:
                raise ContractError("shuffle restore requires a positive shuffle_buffer")
            saved = self._shuffle_restore_prefix_hashes
            run = _Run(
                file_index=self._file_index,
                byte_offset=self._byte_offset,
                games_seen=self._games_seen,
                stats=stats,
                hashes=set(saved if saved is not None else ()),
            )
            for game in self._ordered_source(run):
                self._shuffle_pos += 1
                self._sync_pos(run)
                yield game
            self._sync_pos(run)
            return
        snap = (self._file_index, self._byte_offset, self._games_seen, self._shuffle_pos)
        if self._shuffle_restore_entries is not None and self._shuffle_restore_rng is not None:
            # Seek resume: verbatim buffer + RNG + prefix, no prefix replay.
            run = _Run(
                file_index=self._file_index,
                byte_offset=self._byte_offset,
                games_seen=self._games_seen,
                stats=stats,
                hashes=set(),
            )
            state = self._restore_shuffle_state(run)
            self._active_buf = state.buf
            self._active_rng = state.rng
            self._active_run = run
            try:
                for game in self._shuffled(run, state):
                    self._shuffle_pos += 1
                    self._sync_pos(run)
                    yield game
                self._sync_pos(run)
            finally:
                self._active_buf = None
                self._active_rng = None
                self._active_run = None
            return
        if snap != (0, 0, 0, 0):
            raise ContractError(
                "shuffled stream resume requires shuffle restore state "
                "(buffer_entries/buffer_rng_state/prefix_hashes); prefix replay removed"
            )
        state = _ShuffleState(buf=[], rng=random.Random(_shuffle_key(self._seed, self._epoch)))
        run = _Run(file_index=0, byte_offset=0, games_seen=0, stats=stats, hashes=set())
        self._active_buf = state.buf
        self._active_rng = state.rng
        self._active_run = run
        try:
            for game in self._shuffled(run, state):
                self._shuffle_pos += 1
                self._sync_pos(run)
                yield game
            self._sync_pos(run)
        finally:
            self._active_buf = None
            self._active_rng = None
            self._active_run = None

    def iter_batches(self, batch_size: int) -> Iterator[list[StreamGame]]:
        """Yield pre-batched game lists; one stack + pin_memory per batch downstream."""
        if type(batch_size) is not int or batch_size <= 0:
            raise ContractError(f"batch_size must be a positive int, got {batch_size!r}")
        batch: list[StreamGame] = []
        for game in self:
            batch.append(game)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch


#: Cap for spawn decode workers (bounded pool; decode is pure Python+C
#: without torch, so workers stay light but numerous).
_DECODE_PROC_MAX_WORKERS = 16

#: One worker file-batch result: per game ``(file_index, end, game_bytes,
#: game, validation_hash, wall_hash, assigned_split)`` (``None`` payload
#: for undecodable games; the tail ignores hashes when the game is ``None``).
_WorkerBatchResult = list[
    tuple[int, int, bytes, GameRecord | None, str | None, str | None, str | None]
]
#: File-batch size bounding in-flight work (prefetch sequences x this many
#: files); sized with the prefetch default below so emission stays ordered.
_DECODE_BATCH_GAMES = 16


#: Decode tasks between periodic worker collections. Bounds cyclic-garbage
#: growth (autograd-free data: rare cycles) while keeping each pause ~10ms on
#: one worker — absorbed by the other seven behind prefetch depth.
_DECODE_WORKER_COLLECT_EVERY = 32


def _decode_worker_init() -> None:
    """Freeze import-time heap in spawn decode workers (initializer).

    Moves everything allocated during spawn import (pyarrow, msgspec, zstd,
    module state) to the permanent generation so per-task nursery scans stay
    fast; cyclic task garbage is reclaimed by the periodic collect below.
    """
    import gc as _gc

    with contextlib.suppress(Exception):
        _gc.collect()
        _gc.freeze()


_DECODE_WORKER_TASKS = 0


def _decode_frames_worker(
    files: list[tuple[int, str, int]],
    *,
    seed: int,
    ratios: dict[str, float],
) -> _WorkerBatchResult:
    """Frame+decode+validate one ordered file batch in a spawn worker.

    Input ``(file_index, path, base)`` mirrors :meth:`GameStream._framed_from`
    offset semantics; output per game ``(file_index, end, game_bytes, game,
    validation_hash, wall_hash, assigned_split)`` in file order (``None``
    payload trio for undecodable games, exactly like :meth:`_decode_inline`).
    ``wall_hash``/``assigned_split`` use the identical pure functions the
    shared :meth:`_finish_decode` tail applies, so main-thread results are
    bit-identical with ~0.06ms/game of hashing moved off the consumer.
    """
    # No local imports: spawn re-imports this module, so module globals
    # (decode_game_object, validate_game, stem_of, Path, contracts) are present.
    global _DECODE_WORKER_TASKS
    _DECODE_WORKER_TASKS += 1
    if _DECODE_WORKER_TASKS % _DECODE_WORKER_COLLECT_EVERY == 0:
        import gc as _gc

        with contextlib.suppress(Exception):
            _gc.collect()
    out: _WorkerBatchResult = []
    for file_index, path_str, base in files:
        fpath = Path(path_str)
        stem = stem_of(fpath)
        group_key = group_key_for_path(fpath)
        for offset, game_bytes in ZstdLineStream(fpath).iter_games():
            if offset < base:
                continue
            end = offset + len(game_bytes)
            try:
                game = decode_game_object(
                    object_id=stem,
                    packaged_object_id=stem,
                    decoded_bytes=game_bytes,
                )
            except (ContractError, CorruptArtifactError, ValueError):
                out.append((file_index, end, game_bytes, None, None, None, None))
                continue
            vhash = validate_game(game).validation_hash
            out.append(
                (
                    file_index,
                    end,
                    game_bytes,
                    game,
                    vhash,
                    compute_wall_hash(game),
                    assign_split(group_key=group_key, seed=seed, ratios=ratios),
                )
            )
    return out


class PrefetchGameStream(GameStream):
    """Game stream with spawn-process background decode.

    prefetch=64 sequences x _DECODE_BATCH_GAMES=16 files bounds in-flight
    work. Frames are submitted in order with sequence numbers; results are
    processed strictly in sequence order, never completion order, so the
    emitted sequence is bit-identical to :class:`GameStream`. ``waits``
    counts how often the consumer blocked on the next sequence — the
    GPU-starvation signal (raise prefetch/workers while it climbs).
    """

    def __init__(
        self,
        manifest: StreamManifest,
        *,
        seed: int,
        ratios: Mapping[str, float],
        epoch: int = 0,
        split: str | None = None,
        shuffle_buffer: int = 0,
        start: StreamCursor | None = None,
        drop_duplicates: bool = True,
        prefetch: int = 64,
        max_workers: int | None = None,
        shuffle_restore_entries: list[dict[str, object]] | None = None,
        shuffle_restore_rng: dict[str, object] | None = None,
        shuffle_restore_prefix_hashes: list[str] | tuple[str, ...] | None = None,
        snapshot_blob: Path | str | None = None,
    ) -> None:
        super().__init__(
            manifest,
            seed=seed,
            ratios=ratios,
            epoch=epoch,
            split=split,
            shuffle_buffer=shuffle_buffer,
            start=start,
            drop_duplicates=drop_duplicates,
            shuffle_restore_entries=shuffle_restore_entries,
            shuffle_restore_rng=shuffle_restore_rng,
            shuffle_restore_prefix_hashes=shuffle_restore_prefix_hashes,
            snapshot_blob=snapshot_blob,
        )
        if type(prefetch) is not int or prefetch < 1:
            raise ContractError(f"prefetch must be a positive int, got {prefetch!r}")
        if max_workers is not None and (type(max_workers) is not int or max_workers < 1):
            raise ContractError(f"max_workers must be a positive int, got {max_workers!r}")
        self._prefetch = prefetch
        self._max_workers = max_workers

    def _ordered_source(self, run: _Run) -> Iterator[StreamGame]:
        # Spawn-process frame+decode: CPython's json holds the GIL through the
        # whole parse, so decode threads serialize (8-thread pull == serial
        # speed, measured); framing threads only add GIL/future overhead
        # (rejected twice, measured). Spawn workers frame+decode truly in
        # parallel; file batches keep sequence numbers so the emitted order
        # is bit-identical to GameStream. Spawn (never fork: the caller may
        # hold pool threads already).
        pending: dict[int, Future[_WorkerBatchResult]] = {}
        waits = 0
        sequence = 0
        due = 0
        inflight = 0
        files = self._manifest.files
        nxt = run.file_index
        first_base = run.byte_offset
        exhausted = False
        workers = self._max_workers or os.cpu_count() or 8
        workers = max(1, min(int(workers), _DECODE_PROC_MAX_WORKERS))
        ctx = _mp_get_context("spawn")
        worker_seed: int = self._seed
        worker_ratios: dict[str, float] = dict(self._ratios)
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=_decode_worker_init
        ) as pool:
            while True:
                while inflight < self._prefetch and not exhausted:
                    batch: list[tuple[int, str, int]] = []
                    while len(batch) < _DECODE_BATCH_GAMES and nxt < len(files):
                        base = first_base if nxt == run.file_index else 0
                        batch.append((nxt, files[nxt].path.as_posix(), base))
                        nxt += 1
                    if nxt >= len(files):
                        exhausted = True
                    if len(batch) == 0:
                        break
                    pending[sequence] = pool.submit(
                        _decode_frames_worker, batch, seed=worker_seed, ratios=worker_ratios
                    )
                    inflight += len(batch)
                    sequence += 1
                if due not in pending:
                    break
                future = pending.pop(due)
                if not future.done():
                    waits += 1
                results = future.result()
                inflight -= len(results)
                due += 1
                for item in results:
                    fi, end, gbytes, game, vhash, whash, assigned = item
                    emitted = self._finish_decode(
                        run, fi, end, gbytes, game, vhash, precomputed=(whash, assigned)
                    )
                    if emitted is not None:
                        yield emitted
        if run.stats is not None:
            run.stats.waits += waits


def slice_microbatches(
    batch: Sequence[StreamGame], microbatch_size: int
) -> Iterator[list[StreamGame]]:
    """Cut shallow microbatch slices with no re-decode (same objects)."""
    if type(microbatch_size) is not int or microbatch_size <= 0:
        raise ContractError(f"microbatch_size must be a positive int, got {microbatch_size!r}")
    for start in range(0, len(batch), microbatch_size):
        yield list(batch[start : start + microbatch_size])


def count_decisions(games: Iterable[StreamGame]) -> int:
    """Decision proxy for throughput math: total events across games."""
    return sum(len(game.game.events) for game in games)


def check_wall_disjoint(games: Iterable[StreamGame]) -> None:
    """Fail if one wall is shared by two splits; vacuous when hashes are null."""
    seen: dict[str, str] = {}
    for game in games:
        if game.wall_hash is None:
            continue
        previous = seen.get(game.wall_hash)
        if previous is None:
            seen[game.wall_hash] = game.split
        elif previous != game.split:
            raise ContractError(f"wall {game.wall_hash[:16]} in splits {previous} and {game.split}")


def verify_no_privileged_leakage(payload: Mapping[str, object]) -> None:
    """Hard failure if an actor-bound mapping carries a privileged key."""
    bad = sorted(key for key in payload if key in FORBIDDEN_IN_ACTOR)
    if len(bad) > 0:
        raise ContractError(f"privileged keys in actor payload: {bad}")


def actor_payload(game: StreamGame) -> dict[str, object]:
    """Firewall-safe projection: identity + counts + hashes, never events.

    Events carry private hands (``tehais``), so they stay privileged.
    """
    payload: dict[str, object] = {
        "event_count": len(game.game.events),
        "game_id": game.game.game_id,
        "source_object_id": game.game.object_id,
        "split": game.split,
        "validation_hash": game.validation_hash,
        "wall_hash": game.wall_hash,
    }
    verify_no_privileged_leakage(payload)
    return payload
