"""Stream reads: identity math, framing, and cursor-anchored fetch.

Owns the read-plane records the iterator walks: the split-assignment and
identity helpers (``stem_of``, ``group_key_for[_path]``, ``compute_wall_hash``,
``assign_split``), the ``StreamCursor``/``StreamGame``/``StreamStats`` records,
the ``ZstdLineStream`` framer, the ``_Run``/``_ShuffleState`` pass state, and
the ``fetch_game_at`` fail-closed single-game fetch.
"""

from __future__ import annotations

import hashlib
import random  # noqa: TC003  # reason: TC003 random.Random builds RNGs at runtime in _ShuffleState
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import zstandard as zstd

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.data.decode import decode_json_line
from hydra2.data.stream_manifest import PARTITION_ORDER

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from hydra2.data.decode import GameRecord

__all__ = [
    "StreamCursor",
    "StreamGame",
    "StreamStats",
    "ZstdLineStream",
    "assign_split",
    "compute_wall_hash",
    "fetch_game_at",
    "group_key_for",
    "group_key_for_path",
    "stem_of",
]

_START_TYPES = frozenset({"start_game", "startGame", "game_start", "start"})
_END_TYPES = frozenset({"end_game", "endGame", "game_end", "end"})
_CHUNK_SIZE = 65536
_DIGIT_RUN = re.compile(r"[0-9]+")

#: Framing authority: the byte-exact Python oracle below is the live path
#: (packet 283/283 games + raw sha every game). The deferred top-level
#: ``frame_games`` hasattr gate was deleted (Wave 0 — dead two ways: the
#: bridge registers framing only on the ``packet`` submodule, and its
#: signature takes ``(compressed, file_idx, base)``, not a path). A future
#: packet cutover must add a new correctly-shaped ``packet``-submodule call
#: here, not restore the probe.


def fetch_game_at(
    path: Path | str,
    game_offset: int,
    *,
    seed: int,
    ratios: Mapping[str, float],
    expected_sha: str | None = None,
) -> StreamGame:
    """Fetch one game by decompressed-byte ``game_offset`` (fail-closed).

    Framing is the byte-exact Python oracle (:class:`ZstdLineStream`;
    packet 283/283 games + raw sha every game). Verifies ``expected_sha``
    (raw_bytes_sha256) when given; split assignment uses ``seed``/``ratios``
    identically to the live stream.
    """
    from hydra2.data.decode import decode_game_object as _decode
    from hydra2.data.validate import validate_game as _validate

    fpath = Path(path)
    if type(game_offset) is not int or game_offset < 0:
        raise ContractError(f"game offset must be a non-negative int, got {game_offset!r}")
    found: bytes | None = None
    frame_source = ZstdLineStream(fpath).iter_games()
    for offset, game_bytes in frame_source:
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
        """Yield framed games in file order (byte-exact Python oracle)."""
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
