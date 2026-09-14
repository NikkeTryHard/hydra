"""Game stream iterator: ordered/shuffled emission with cursor resume.

Owns the ``GameStream`` pass: framing, the shared validate/quarantine/dedup
tail, deterministic shuffle, shuffle snapshot/restore, and batching.
``PrefetchGameStream`` subclasses it from :mod:`hydra2.data.stream_decode`;
emission order is bit-identical between the two.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, cast

from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.data.decode import GameRecord, decode_game_object
from hydra2.data.stream_manifest import (
    PARTITION_ORDER,
    parse_shuffle_rng,
    read_reservoir_blob,
    serialize_shuffle_rng,
)
from hydra2.data.stream_read import (
    StreamCursor,
    StreamGame,
    StreamStats,
    ZstdLineStream,
    _check_int,
    _Run,
    _shuffle_key,
    _ShuffleState,
    assign_split,
    compute_wall_hash,
    fetch_game_at,
    group_key_for_path,
    stem_of,
)
from hydra2.data.validate import validate_game

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from hydra2.data.stream_manifest import StreamManifest

__all__ = [
    "GameStream",
]


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
