"""Stream background decode and actor-plane projections.

Owns the spawn-process frame/decode worker (pool constants, initializer,
batch entry point), the ``PrefetchGameStream`` spawn-decode subclass, the
shallow microbatch slicer, the wall-disjointness predicate, and the
privileged-leakage firewall (``verify_no_privileged_leakage`` /
``actor_payload``).
"""

from __future__ import annotations

import contextlib
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context as _mp_get_context
from pathlib import Path
from typing import TYPE_CHECKING

from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.data.decode import GameRecord, decode_game_object
from hydra2.data.parquet import FORBIDDEN_IN_ACTOR
from hydra2.data.stream_iter import GameStream
from hydra2.data.stream_read import (
    StreamCursor,
    StreamGame,
    ZstdLineStream,
    _Run,
    assign_split,
    compute_wall_hash,
    group_key_for_path,
    stem_of,
)
from hydra2.data.validate import validate_game

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from concurrent.futures import Future

    from hydra2.data.stream_manifest import StreamManifest

__all__ = [
    "PrefetchGameStream",
    "actor_payload",
    "check_wall_disjoint",
    "count_decisions",
    "slice_microbatches",
    "verify_no_privileged_leakage",
]

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
