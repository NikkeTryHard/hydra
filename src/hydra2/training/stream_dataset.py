"""Loop-facing streaming dataset over a lazily-pulled game stream.

Owns the :class:`_StreamDatasetCore` pull/expand/fill machinery (serial
pulls, the bounded parallel-expansion pool, microbatch fills); the
buffered-window half (grouped takes, sampler state, snapshots, verbatim
restore) rides :class:`_StreamDatasetBufferMixin`. Single-pass with the
epoch pinned at 0: exhaustion while rows are still demanded fails closed.

Fill path: the bounded spawn pool (:meth:`_fill_parallel`) plus the serial
pull/batch-expansion tails (:meth:`_pull_game` / :meth:`_pull_game_batch`);
the H2D staging is the caller-owned ``PinnedRing``/``_GatedOverlapFeed`` —
never a ``DataLoader``. :meth:`_StreamDatasetCore._fill` runs the verbatim
Python pool path below (counted in ``feed_fallbacks``). Orchestration —
order, counters, privileged joins, quarantine classes, buffer index — is
untouched.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context as _mp_get_context
from typing import TYPE_CHECKING, Any

from hydra2.contracts.common import ContractError as ContractError
from hydra2.data.replay_expand import expand_privileged_rows as expand_privileged_rows
from hydra2.data.stream import StreamCursor as DataStreamCursor
from hydra2.training.stream_expand import (
    _PARALLEL_EXPAND_MAX_WORKERS as _PARALLEL_EXPAND_MAX_WORKERS,
)
from hydra2.training.stream_expand import (
    _QUARANTINE_REASON_CLASSES as _QUARANTINE_REASON_CLASSES,
)
from hydra2.training.stream_expand import _expand_game_planes as _expand_game_planes
from hydra2.training.stream_expand import _expand_game_rows as _expand_game_rows
from hydra2.training.stream_expand import _expand_streamed_chunk as _expand_streamed_chunk
from hydra2.training.stream_expand import _pool_worker_init as _pool_worker_init
from hydra2.training.stream_expand import _quarantine_class as _quarantine_class
from hydra2.training.stream_expand import _require_replay_backend as _require_replay_backend
from hydra2.training.stream_expand import _row_to_dict as _row_to_dict
from hydra2.training.stream_expand import _slim_row_dicts as _slim_row_dicts

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from hydra2.data.stream import GameStream as GameStream
    from hydra2.data.stream import StreamGame as StreamGame

__all__ = [
    "_StreamDatasetCore",
]


class _StreamDatasetCore:
    """Pull/expand/fill half of the streaming dataset (base for the buffer mixin)."""

    def __init__(
        self,
        *,
        stream_factory: Callable[[], GameStream],
        num_actions: int,
        feature_dim: int,
        seed: int,
        drop_last: bool,
        need_privileged: bool = True,
        replay_backend: str = "rust",
        expand_workers: int = 0,
        expand_batch_games: int = 64,
        pin_memory: bool = True,
        homogeneous_buckets: bool = False,
    ) -> None:
        if drop_last is not True:
            raise ContractError(f"single-pass requires drop_last=true, got {drop_last!r}")
        if not isinstance(homogeneous_buckets, bool):
            raise ContractError(f"homogeneous_buckets must be a bool, got {homogeneous_buckets!r}")
        if (
            isinstance(expand_workers, bool)
            or not isinstance(expand_workers, int)
            or expand_workers < 0
        ):
            raise ContractError(f"expand_workers invalid: {expand_workers!r} (non-negative int)")
        # Bounded pool: larger requests clamp to _PARALLEL_EXPAND_MAX_WORKERS.
        self._expand_workers = min(expand_workers, _PARALLEL_EXPAND_MAX_WORKERS)
        if (
            isinstance(expand_batch_games, bool)
            or not isinstance(expand_batch_games, int)
            or not (1 <= expand_batch_games <= 1024)
        ):
            raise ContractError(
                f"expand_batch_games invalid: {expand_batch_games!r} (int in [1, 1024])"
            )
        self._expand_batch_games = expand_batch_games
        self._expand_pool: ProcessPoolExecutor | None = None
        self._replay_backend = _require_replay_backend(replay_backend)
        self._need_privileged = need_privileged
        self._factory = stream_factory
        self._num_actions = num_actions
        self._feature_dim = feature_dim
        self._seed = seed
        self._drop_last = drop_last
        self._epoch = 0  # single-pass: epoch pinned 0 (no wrap, no start_epoch)
        self._rows: list[dict[str, Any]] = []
        self._offset = 0
        self._dropped = 0
        self._microbatches_in_epoch = 0
        self._stream: GameStream | None = None
        self._iter: Iterator[StreamGame] | None = None
        self.privileged: dict[str, dict[str, Any]] = {}
        self.replayed = 0
        self.sim_replayed = 0
        self.expand_quarantined = 0
        self.expand_quarantine_reasons: dict[str, int] = {}
        # Fast-resume buffer index: whole-game aligned entries for the live
        # ``_rows`` window (``_dropped`` is the logical index of ``_rows[0]``).
        # Each entry pins ``key`` (raw_bytes_sha256, deterministic id) plus
        # ``path``/``offset``/``split``/``rows`` for verbatim refetch +
        # re-expansion (kilobytes, never rows). ``sum(rows) == len(_rows)``.
        # Caller-owned transfer owns pinning: when a pinned-ring feed stages
        # the H2D copy, encode-side pin_memory() calls (29 page-locks per
        # microbatch) are pure overhead — the ring copies into its own pinned
        # slots. The driver clears this once the feed is open; the sync path
        # keeps it set so non_blocking H2D still overlaps.
        self.pin_memory = pin_memory
        self._homogeneous_buckets = homogeneous_buckets
        self._buffered_entries: list[dict[str, Any]] = []
        # Python pool fill count (telemetry only).
        self.feed_fallbacks = 0

    @property
    def epoch(self) -> int:
        """Current epoch (matches the live stream's epoch)."""
        return self._epoch

    @property
    def rows_consumed_in_epoch(self) -> int:
        """Rows consumed (plus tail-dropped) in the live epoch."""
        return self._offset

    @property
    def microbatches_in_epoch(self) -> int:
        """Microbatches consumed in the live epoch (resume-drain count)."""
        return self._microbatches_in_epoch

    def stream_cursor(self) -> DataStreamCursor:
        """Pulled-game frontier of the live stream (resume anchor)."""
        if self._stream is None:
            return DataStreamCursor(
                file_index=0,
                byte_offset=0,
                games_seen=0,
                seed=self._seed,
                epoch=self._epoch,
            )
        return self._stream.cursor()

    def _ensure_iter(self) -> None:
        # Single-pass: the stream is built once and consumed to exhaustion.
        # ``_stream`` is never reset, so a consumed iterator (``_iter`` None
        # with ``_stream`` set) stays consumed — recreating it here would
        # silently re-read the corpus from scratch (duplicated rows, backward
        # resume frontier). Post-exhaustion pulls return ``False`` and the
        # terminal ``_fill`` below fails closed.
        if self._iter is None and self._stream is None:
            self._stream = self._factory()
            self._iter = iter(self._stream)

    def _pull_game(self) -> bool:
        """Pull, expand, and buffer one game; ``False`` at stream end.

        Every game expands through :func:`_expand_game_rows` on this
        dataset's ``replay_backend`` (the ``"python"`` oracle). Privileged rows
        expand only when the dataset was built with ``need_privileged``
        (positive auxiliary weights); BC-only runs skip them so label-less
        games still train. Either path raising :class:`ContractError`
        quarantines-and-counts the game (fail closed, never a silent drop);
        only fully expanded games buffer rows.
        This serial pull is the bit-identity reference: order, cursors, dedup,
        and split assignment are untouched from pull to buffer (the parallel
        batch path in :meth:`_fill_parallel` merges in this same pull order).
        """
        self._ensure_iter()
        if self._iter is None:
            return False
        while True:
            try:
                streamed = next(self._iter)
            except StopIteration:
                self._iter = None
                return False
            try:
                if self._replay_backend == "rust":
                    row_dicts, sim_path = _expand_game_planes(
                        streamed.game, streamed.split, streamed.raw
                    )
                else:
                    actor_rows, sim_path = _expand_game_rows(
                        streamed.game, streamed.split, self._replay_backend
                    )
                    row_dicts = [_row_to_dict(row) for row in actor_rows]
                if self._need_privileged:
                    priv_rows = expand_privileged_rows(streamed.game, split=streamed.split)
                else:
                    # BC-only weights read no privileged labels: skip expansion so
                    # games whose labels are unavailable still train.
                    priv_rows = []
            except ContractError as exc:
                self._count_quarantine(exc)
                continue
            if sim_path:
                self.sim_replayed += 1
            else:
                self.replayed += 1
            for row_dict in row_dicts:
                self._rows.append(row_dict)
            self._track_buffered(streamed, len(row_dicts))
            for priv in priv_rows:
                label = dict(priv.privileged_label)
                # intentionally discarded: existing label wins
                _ = self.privileged.setdefault(str(priv.decision_id), label)
            return True

    def _pull_game_batch(self, limit: int) -> bool:
        """Pull up to ``limit`` games; expand rust games in one bridge call.

        Bit-identical merge to serial :meth:`_pull_game` game-for-game
        (order, replayed/sim counters, privileged joins, quarantine classes,
        buffer index); only the Rust handoff is batched (one FFI crossing
        and one GIL release per batch instead of per game). ``True`` once at
        least one game buffers, ``False`` at stream end. Python-backend
        datasets never take this path (the oracle stays per-game).
        """
        from hydra2.training.rust_batch import expand_game_batch

        while True:
            batch = self._pull_streamed_batch(limit)
            if len(batch) == 0:
                return False
            walls: list[list[int] | None] = []
            pre_err: dict[int, ContractError] = {}
            for pos, streamed in enumerate(batch):
                tiles = streamed.game.wall_tiles
                if tiles is None:
                    walls.append(None)
                else:
                    wall = [int(t) for t in tiles]
                    if len(wall) != 136:
                        pre_err[pos] = ContractError(
                            f"wall_tiles must carry 136 tiles, got {len(wall)}"
                        )
                        walls.append(None)
                    else:
                        walls.append(wall)
            idxs = [
                pos for pos, streamed in enumerate(batch) if streamed.raw and pos not in pre_err
            ]
            results = (
                expand_game_batch([(batch[i].raw, 0, walls[i]) for i in idxs])
                if len(idxs) > 0
                else []
            )
            if len(results) != len(idxs):
                raise ContractError(
                    f"bridge batch drift: {len(results)} results for {len(idxs)} games"
                )
            by_pos = dict(zip(idxs, results, strict=True))
            buffered_any = False
            for pos, streamed in enumerate(batch):
                try:
                    if pos in pre_err:
                        raise pre_err[pos]
                    if pos in by_pos:
                        planes, rows, t_len, quarantined, reason, event_idx = by_pos[pos]
                        game_id = str(streamed.game.game_id)
                        if quarantined:
                            raise ContractError(
                                f"rust walk quarantined game {game_id!r}: "
                                f"{reason} at event {event_idx}"
                            )
                        row_dicts = _slim_row_dicts(game_id, planes, rows, t_len)
                        sim_path = streamed.game.wall_tiles is None
                    else:
                        # Hand-built games (tests) carry no framed bytes:
                        # ``_expand_game_planes`` rebuilt path, same rows as the serial pull.
                        row_dicts, sim_path = _expand_game_planes(
                            streamed.game, streamed.split, b""
                        )
                    if self._need_privileged:
                        priv_rows = expand_privileged_rows(streamed.game, split=streamed.split)
                    else:
                        priv_rows = []
                except ContractError as exc:
                    self._count_quarantine(exc)
                    continue
                if sim_path:
                    self.sim_replayed += 1
                else:
                    self.replayed += 1
                self._rows.extend(row_dicts)
                self._track_buffered(streamed, len(row_dicts))
                for priv in priv_rows:
                    label = dict(priv.privileged_label)
                    # intentionally discarded: existing label wins
                    _ = self.privileged.setdefault(str(priv.decision_id), label)
                buffered_any = True
            if buffered_any:
                return True

    def _get_expand_pool(self) -> ProcessPoolExecutor:
        """Lazily-built spawn pool for parallel expansion (bounded, persistent).

        Spawn (never fork): the parent has torch threads alive and
        fork-with-threads deadlocks racily; spawn re-imports clean workers
        (same rationale as :func:`_scan_corpus_parallel`). One pool per
        dataset, built once per run and reused across fills — never rebuilt
        per fill or per epoch (single-pass pins epoch 0, so no epoch boundary
        exists to respawn at; post-first-fill spawn cost is zero by
        construction). Workers start under :func:`_pool_worker_init` (one
        torch thread each); :meth:`close` shuts the pool down.
        """
        pool = self._expand_pool
        if pool is None:
            ctx = _mp_get_context("spawn")
            pool = ProcessPoolExecutor(
                max_workers=self._expand_workers,
                mp_context=ctx,
                initializer=_pool_worker_init,
            )
            self._expand_pool = pool
        return pool

    def close(self) -> None:
        """Shut down the parallel expansion pool, if any (idempotent)."""
        pool, self._expand_pool = self._expand_pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def _pull_streamed_batch(self, limit: int) -> list[Any]:
        """Pull up to ``limit`` raw games in stream order; ``[]`` at stream end."""
        batch: list[Any] = []
        while len(batch) < limit:
            self._ensure_iter()
            if self._iter is None:
                break
            try:
                batch.append(next(self._iter))
            except StopIteration:
                self._iter = None
                break
        return batch

    def _merge_expanded(
        self,
        streamed: Any,
        result: tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]], bool],
    ) -> None:
        """Buffer one pool-expanded game (merge tail mirrors :meth:`_pull_game`).

        Counters, row order, privileged joins, and the whole-game buffer
        index update exactly as the serial pull does (including the
        fail-closed empty-row raise in :meth:`_track_buffered`); only the
        row projection already happened in the worker.
        """
        row_dicts, priv_pairs, sim_path = result
        if sim_path:
            self.sim_replayed += 1
        else:
            self.replayed += 1
        self._rows.extend(row_dicts)
        self._track_buffered(streamed, len(row_dicts))
        for decision_id, label in priv_pairs:
            # intentionally discarded: existing label wins
            _ = self.privileged.setdefault(decision_id, label)

    def _fill_parallel(self, need: int) -> None:
        """Buffer at least ``need`` rows via ordered parallel expansion.

        Pulls raw games in stream order in ``_expand_batch_games``-sized
        rounds, shards each round into contiguous per-worker chunks, expands
        chunks in the bounded spawn pool (:func:`_expand_streamed_chunk` —
        one bridge FFI per chunk on rust, per-game oracle inside the chunk
        on python), and merges strictly in pull order via per-chunk futures
        (strict pairing keeps chunk↔future↔row alignment — silent skew would
        surface only as a restore_buffer hash failure); never ``pool.map``
        (one game's failure cancels the batch tail there). Rows, counters,
        and quarantine classes match the serial
        :meth:`_fill` game-for-game; per-game :class:`ContractError`
        quarantines-and-counts (fail closed, never a silent drop) while a
        chunk-level failure quarantines each game it carried (same reason)
        and any other worker failure propagates. Terminal exhaustion raises
        the serial messages verbatim.
        """
        _t_pool = time.perf_counter() if self._offset == 0 else 0.0
        pool = self._get_expand_pool()
        workers = max(1, self._expand_workers)
        debug_fill = os.environ.get("HYDRA2_FILL_DEBUG") == "1"
        _t_fill = time.perf_counter() if debug_fill else 0.0
        _fill_rounds = 0
        _fill_games = 0
        _fill_rows = 0
        _fill_pull = 0.0
        _fill_expand = 0.0
        _fill_submit = 0.0
        _fill_maxchunk = 0.0
        while self._live_count() < need:
            first_fill = self._offset == 0
            timed = first_fill or debug_fill
            if first_fill and _t_pool > 0.0:
                print(f"phase: pool init={time.perf_counter() - _t_pool:.1f}s", flush=True)
                _t_pool = 0.0
            _t_round = time.perf_counter() if timed else 0.0
            batch = self._pull_streamed_batch(self._expand_batch_games)
            if len(batch) == 0:
                if self._live_count() == 0 and self._offset == 0:
                    raise ContractError("stream yielded zero train rows (empty split?)")
                raise ContractError(
                    f"stream exhausted with {self._live_count()} buffered rows, need {need} "
                    "(single-pass: stream end with updates remaining; "
                    "rescope loop.max_updates to supply)"
                )
            _t_pull = time.perf_counter() if timed else 0.0
            size = max(1, (len(batch) + workers - 1) // workers)
            chunks = [batch[i : i + size] for i in range(0, len(batch), size)]
            _t_submit = time.perf_counter() if timed else 0.0
            futures = [
                pool.submit(
                    _expand_streamed_chunk,
                    [
                        (
                            streamed.game,
                            streamed.split,
                            self._replay_backend,
                            self._need_privileged,
                            streamed.raw,
                        )
                        for streamed in chunk
                    ],
                )
                for chunk in chunks
            ]
            _t_submitted = time.perf_counter() if timed else 0.0
            for chunk, future in zip(chunks, futures, strict=True):
                _t_chunk = time.perf_counter() if timed else 0.0
                try:
                    results = future.result()
                except ContractError as exc:
                    for _streamed in chunk:
                        self._count_quarantine(exc)
                    continue
                if timed:
                    _dt_chunk = time.perf_counter() - _t_chunk
                    if _dt_chunk > _fill_maxchunk:
                        _fill_maxchunk = _dt_chunk
                    if _dt_chunk >= 0.15:
                        _ev = sum(len(s.game.events) for s in chunk)
                        print(
                            f"phase: slow-chunk games={len(chunk)} events={_ev} t={_dt_chunk:.2f}s",
                            flush=True,
                        )
                for streamed, result in zip(chunk, results, strict=True):
                    if result[0] == "quarantine":
                        self._count_quarantine(ContractError(str(result[1])))
                        continue
                    _, row_dicts, priv_pairs, sim_path = result
                    self._merge_expanded(streamed, (row_dicts, priv_pairs, sim_path))
                    if timed:
                        _fill_rows += len(row_dicts)
            if timed:
                _now = time.perf_counter()
                _fill_rounds += 1
                _fill_games += len(batch)
                _fill_pull += _t_pull - _t_round
                _fill_expand += _now - _t_pull
                _fill_submit += _t_submitted - _t_pull
                if first_fill:
                    print(
                        f"phase: fill games={len(batch)} pull={_t_pull - _t_round:.1f}s "
                        f"expand={_now - _t_pull:.1f}s",
                        flush=True,
                    )
        if debug_fill:
            print(
                f"phase: fill-done need={need} rounds={_fill_rounds} games={_fill_games} "
                f"rows={_fill_rows} pull={_fill_pull:.1f}s expand={_fill_expand:.1f}s "
                f"submit={_fill_submit:.1f}s maxchunk={_fill_maxchunk:.2f}s "
                f"total={time.perf_counter() - _t_fill:.1f}s",
                flush=True,
            )

    def _track_buffered(self, streamed: Any, row_count: int) -> None:
        """Index one buffered game for fast-resume verbatim rebuild (whole-game)."""
        try:
            key = str(streamed.game.raw_bytes_sha256)
            path = streamed.path.as_posix()
            offset = int(streamed.offset)
            split = str(streamed.split)
        except (AttributeError, TypeError, ValueError) as err:
            raise ContractError("buffered game identity malformed") from err
        if row_count <= 0:
            raise ContractError("buffered game must contribute at least one row")
        self._buffered_entries.append(
            {"key": key, "path": path, "offset": offset, "split": split, "rows": row_count}
        )

    def _count_quarantine(self, exc: ContractError) -> None:
        """Count one quarantined game under its normalized reason class."""
        self.expand_quarantined += 1
        reason = _quarantine_class(exc)
        if reason in self.expand_quarantine_reasons:
            self.expand_quarantine_reasons[reason] += 1
        elif len(self.expand_quarantine_reasons) < _QUARANTINE_REASON_CLASSES:
            self.expand_quarantine_reasons[reason] = 1
        else:
            overflow = self.expand_quarantine_reasons.get("_other", 0)
            self.expand_quarantine_reasons["_other"] = overflow + 1

    def _live_count(self) -> int:
        """Buffered-but-unconsumed rows (list length minus compacted prefix)."""
        return len(self._rows) - (self._offset - self._dropped)

    def _compact(self) -> None:
        """Drop whole consumed games once the prefix outgrows the bound.
        Logical counters (``_offset`` and everything derived) are untouched;
        only list storage is reclaimed. Whole-game alignment keeps
        ``_buffered_entries`` (``sum(rows) == len(_rows)``) verbatim so a
        checkpoint can re-expand the live window game-for-game. At most one
        partially-consumed game's prefix stays buffered past the bound.
        """
        consumed = self._offset - self._dropped
        # Bound lives on the public facade: resolve it there at call time so
        # facade-level overrides keep applying to this split-home method.
        from hydra2.training.stream_train import _BUFFER_COMPACT_ROWS

        if consumed < _BUFFER_COMPACT_ROWS:
            return
        remaining = consumed
        drop_games = 0
        drop_rows = 0
        for entry in self._buffered_entries:
            count = int(entry["rows"])
            if count <= remaining:
                remaining -= count
                drop_games += 1
                drop_rows += count
            else:
                break
        if drop_games <= 0:
            return
        del self._rows[:drop_rows]
        del self._buffered_entries[:drop_games]
        self._dropped += drop_rows

    def _fill(self, need: int) -> None:
        """Buffer at least ``need`` consumable rows (single-pass: exhaustion is terminal)."""
        self.feed_fallbacks += 1
        if self._expand_workers > 0:
            self._fill_parallel(need)
            return
        if self._replay_backend == "rust":
            while self._live_count() < need:
                if self._pull_game_batch(self._expand_batch_games):
                    continue
                if self._live_count() == 0 and self._offset == 0:
                    raise ContractError("stream yielded zero train rows (empty split?)")
                raise ContractError(
                    f"stream exhausted with {self._live_count()} buffered rows, need {need} "
                    "(single-pass: stream end with updates remaining; "
                    "rescope loop.max_updates to supply)"
                )
            return
        while self._live_count() < need:
            if self._pull_game():
                continue
            if self._live_count() == 0 and self._offset == 0:
                raise ContractError("stream yielded zero train rows (empty split?)")
            raise ContractError(
                f"stream exhausted with {self._live_count()} buffered rows, need {need} "
                "(single-pass: stream end with updates remaining; "
                "rescope loop.max_updates to supply)"
            )
