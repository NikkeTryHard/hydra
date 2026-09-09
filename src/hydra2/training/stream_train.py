"""Streaming-first supervised training driver: stream → expand → encode → loop.

Consumes ``.mjai.json.zst`` straight from ``data.root`` (no parquet shards)
yet carries the identical contracts as the materialized path: strict decode,
validate-then-quarantine, actor/privileged split, wall-disjoint manifests plus
the leakage check, counter-based RNG, and ``(5,)`` dora.

Pipeline per call to :func:`run_stream_training`:

1. Resolve the run layout (idempotent) and the stream manifest over
   ``data.root`` (provenance pin verified before the first game is read).
2. Scan the corpus once (ordered, ``split=None``) to enforce wall-disjoint
   train/val assignment and to collect the evaluation-wall ledger plus the
   quarantine counters. Any wall shared across the two splits fails closed
   before any runtime object is built.
3. Pull train-split games through :class:`GameStream` (or the bit-identical
   :class:`PrefetchGameStream` when ``data.num_workers > 0``), expand each
   game to actor :class:`DecisionRow` rows — wall-bound games through
   :func:`expand_game` unchanged, wall-less games through the simulator's
   own wall-less replay (:func:`replay_game`), quarantining-and-counting
   expansion failures per game — encode microbatches through the
   real actor-visible encoder, join privileged ranks by opaque decision id
   (train-split rows only; the loop gates the join on config weights), and
   drive :meth:`SupervisedLoop.train` with the config weights.
4. Write ``ckpt-<update>.pt`` + ResumeExactPlan sidecars every
   ``loop.checkpoint_frequency_updates`` updates, append ``metrics.jsonl``
   rows plus ``train.log`` lines, prune to ``keep_last_checkpoints``, and
   leave best-checkpoint promotion to the manual gate
   (:meth:`SupervisedLoop.evaluate_selection` /
   :meth:`SupervisedLoop.maybe_promote_best` are never called here).

Resume is row-exact: checkpoints record the pulled-game frontier cursor plus
the consumed-microbatch count of the live epoch. Resume re-verifies sidecar
identity (run digest, seed, worker plan, payload hash, RNG anchors) BEFORE
mutating any runtime object, drains the identical microbatch prefix (the
stream re-emits the same sequence from the same seed, so the drain is a
catch-up slice, never a refill), verifies the drained frontier against the
recorded cursor, restores model/optimizer/scheduler/loop/RNG state, and
continues. Same seed plus same cursor is the same sequence, so resumed
histories match fresh histories.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context as _mp_get_context
from typing import TYPE_CHECKING, Any, cast

import torch

from hydra2.artifacts.atomic import atomic_replace_bytes
from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.data.decode import decode_game_object
from hydra2.data.replay_expand import (
    expand_game,
    expand_privileged_rows,
)
from hydra2.data.stream import (
    GameStream,
    PrefetchGameStream,
    ZstdLineStream,
    assign_split,
    build_manifest,
    compute_wall_hash,
    group_key_for_path,
    load_scan_cache,
    manifest_digest,
    save_scan_cache,
    scan_cache_path,
    stem_of,
)
from hydra2.data.stream import (
    StreamCursor as DataStreamCursor,
)
from hydra2.data.validate import validate_game
from hydra2.runtime.checkpoint import capture_rng_state
from hydra2.training.dataset import encode_observation_rows
from hydra2.training.loop import SupervisedLoop, TrainingLoopConfig, TrainingState
from hydra2.training.run_config import create_run_layout, run_config_digest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path

    from hydra2.data.decode import GameRecord
    from hydra2.data.parquet import DecisionRow
    from hydra2.data.stream import StreamGame, StreamManifest
    from hydra2.training.run_config import ResumePlan, RunConfig

__all__ = ["SPLIT_RATIOS", "run_stream_training"]

#: Provisional v1 train/validation group fractions for
#: :func:`~hydra2.data.stream.assign_split`. RunConfig pins the split *names*
#: but not the fractions yet; until it does, the driver splits every group
#: deterministically 80/20 under ``seeds.data_seed``. Both partitions stay
#: active so every corpus game lands in exactly one of them; wall-disjointness
#: is enforced by construction (same seed/math for both streams) and verified
SPLIT_RATIOS: dict[str, float] = {"train": 0.8, "validation": 0.2}

#: legacy dict path byte-identical for stub consumers.
_FEATURE_FOLD_DIM = 64

#: Consumed-row prefix dropped at a time (memory bound). 8192 rows ≈ 140MB
#: worst case at ~17KB measured per buffered row dict; the live buffer then
#: stays near in-flight games only, no matter how long the epoch runs.
_BUFFER_COMPACT_ROWS = 8192

#: Max distinct quarantine reason classes retained (overflow merges onward).
_QUARANTINE_REASON_CLASSES = 32


def _quarantine_class(exc: ContractError) -> str:
    """Normalize a quarantine reason to a stable, game-identity-free class."""
    text = str(exc)
    text = re.sub(r"game '[^']*'", "game '<id>'", text)
    text = re.sub(r'game "[^"]*"', "game '<id>'", text)
    text = re.sub(r"game-[0-9a-f]{4,}", "game-<id>", text)
    # Rust formats tokens with Debug (``"5mr"``) where Python uses repr
    # (``'5mr'``): unify spaceless quoted tokens so both backends share one
    # class. Spaced prose quotes are untouched.
    text = re.sub(r'"([A-Za-z0-9_\-]+)"', r"'\1'", text)
    text = re.sub(r"kyoku \d+", "kyoku <n>", text)
    text = re.sub(r"seat \d+", "seat <n>", text)
    return text[:100]


#: Wall-less replay backends (Slice 5, flag-gated). ``"python"`` is the
#: default and preserves the pre-flag behavior byte-identically;
#: ``"rust_json"`` routes wall-less games through the Slice-4 ``rust_stream``
#: JSON handoff. Wall-bound games never leave :func:`expand_game`.
_REPLAY_BACKENDS: tuple[str, ...] = ("python", "rust_json")

#: Provenance hash bound into Rust-fed rows as ``rules_hash``/``adapter_hash``
#: (the stream mints no hashes of its own). Training drops these hashes before
#: the encoder (:func:`_row_to_dict` keeps decision/choice/observation only),
#: so a constant suffices until Slice 6 wires RunConfig-digest plumbing.
_RUST_JSON_SPEC_HASH = "rust_json-v1"


def _require_replay_backend(backend: str) -> str:
    """Validate a replay backend id (fail closed on unknown)."""
    if backend not in _REPLAY_BACKENDS:
        raise ContractError(
            f"replay_backend must be one of {list(_REPLAY_BACKENDS)}, got {backend!r}"
        )
    return backend


def _expand_game_rows(
    game: GameRecord, split: str, backend: str = "python"
) -> tuple[list[DecisionRow], bool]:
    """Expand one game to actor rows on the selected backend.

    Wall-bound games always expand through :func:`expand_game` (the Rust path
    never binds walls); wall-less games replay through ``log_replay`` on the
    ``"python"`` backend or through the Slice-4 ``rust_stream`` JSON handoff
    on ``"rust_json"``. Returns ``(actor_rows, sim_path)`` with ``sim_path``
    true for wall-less games on either backend, so the
    ``replayed``/``sim_replayed``/``expand_quarantined`` counters and the
    :func:`_quarantine_class` normalization apply verbatim to both backends.
    """
    _require_replay_backend(backend)
    if game.wall_tiles is not None:
        return (expand_game(game, split=split), False)
    if backend == "rust_json":
        return (_expand_game_rust_json(game, split), True)
    from hydra2.engines.riichienv.log_replay import replay_game as _replay_sim_game

    return (_replay_sim_game(game, split=split), True)


def _expand_game_rust_json(game: GameRecord, split: str) -> list[DecisionRow]:
    """Replay one wall-less game through the Rust JSON handoff.

    Frames ``game.events`` to single-game JSONL scratch (temp dir only: the
    corpus stays read-only, nothing lands under the artifact root), drains one
    Slice-4 ``RustJsonStream`` over it, and projects each 13-field row onto
    :class:`DecisionRow`. Whole-game Rust quarantines raise
    :class:`ContractError` carrying the Rust ``Quarantine.detail`` so callers
    count them under :func:`_quarantine_class` exactly like Python replay
    failures. A missing compiled extension raises loudly (never a silent
    quarantine). Rust expansion stashes no live observations, so one process
    must not mix backends across an encode.
    """
    import tempfile
    from pathlib import Path

    from hydra2.data.parquet import DecisionRow
    from hydra2.training.rust_stream import RustJsonStream

    try:
        lines = [json.dumps(dict(event), sort_keys=True) for event in game.events]
    except (TypeError, ValueError) as exc:
        raise ContractError(f"rust replay cannot frame game {game.game_id!r}: {exc}") from exc
    stem = str(game.object_id).replace("/", "_")
    if stem in ("", ".", ".."):
        raise ContractError(f"rust replay object_id unusable: {game.object_id!r}")
    with tempfile.TemporaryDirectory(prefix="hydra2-rust-game-") as tmp:
        (Path(tmp) / f"{stem}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        stream = RustJsonStream([tmp], batch=4096, split=split, spec_hash=_RUST_JSON_SPEC_HASH)
        try:
            rows: list[dict[str, Any]] = []
            batch = stream.next()
            while batch:
                rows.extend(batch)
                batch = stream.next()
            quarantines = stream.quarantines()
        finally:
            stream.close()
    if not rows:
        if quarantines:
            # Verbatim detail: Rust desync sentences share the Python
            # ``sim replay desync game … kyoku …`` shape (Rust Debug quotes
            # the id with double quotes, normalized above), so both backends
            # count the same game under the same reason class.
            raise ContractError(quarantines[0].detail)
        raise ContractError(f"rust replay returned no rows for game {game.game_id!r}")
    actor_rows: list[DecisionRow] = []
    for row in rows:
        try:
            actor_doc = json.loads(row["actor_observation"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(
                f"rust replay actor_observation unparseable for {row.get('decision_id')!r}: {exc}"
            ) from exc
        if not isinstance(actor_doc, dict):
            raise ContractError(
                f"rust replay actor_observation not a mapping for {row.get('decision_id')!r}"
            )
        chosen = row.get("chosen_action_id")
        if isinstance(chosen, bool) or not isinstance(chosen, int):
            raise ContractError(
                f"rust replay left chosen unresolved for {row.get('decision_id')!r}"
            )
        actor_rows.append(
            DecisionRow(
                game_id=str(row["game_id"]),
                round_id=str(row["round_id"]),
                decision_id=str(row["decision_id"]),
                seat=int(row["seat"]),
                source_object_id=str(row["source_object_id"]),
                split=str(row["split"]),
                rules_hash=str(row["rules_hash"]),
                adapter_hash=str(row["adapter_hash"]),
                observation_hash=str(row["observation_hash"]),
                action_table_hash=str(row["action_table_hash"]),
                derivation_hash=str(row["derivation_hash"]),
                actor_observation=dict(actor_doc),
                chosen_action_id=int(chosen),
            )
        )
    return actor_rows


#: v1 is single-process: only rank 0 exists, so any other world size fails
#: closed instead of silently training on a rank-0 shard.
_SINGLETON_WORLD = 1
_SINGLETON_RANK = 0

#: Model parameters this driver forwards to
#: :class:`~hydra2.models.model.Hydra2BaselineModel`. Anything else raises
#: instead of being silently ignored.
_MODEL_PARAMETERS = ("d_model", "n_layers", "n_heads", "d_ff", "dropout")


def _needs_privileged_labels(config: RunConfig) -> bool:
    """Whether any positive auxiliary weight consumes privileged labels.

    Mirrors the positive-requires-binding rule in ``run_config``: policy-only
    (BC) runs must not expand privileged rows at all, so a game whose
    privileged labels are unavailable (e.g. wall-less logs without terminal
    scores) still trains instead of quarantining on labels nobody reads.
    """
    return any(
        [
            config.weights.w_placement,
            config.weights.w_value,
            *(config.weights.w_event or {}).values(),
            *(config.weights.w_belief or {}).values(),
        ]
    )


def _split_ratios(config: RunConfig) -> dict[str, float]:
    """Concrete split fractions with the config's split names bound."""
    train_split = config.data.train_split
    val_split = config.data.val_split
    if train_split == val_split:
        raise ContractError(f"data.train_split and data.val_split must differ, got {train_split!r}")
    return {train_split: SPLIT_RATIOS["train"], val_split: SPLIT_RATIOS["validation"]}


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Project a :class:`DecisionRow` onto the real-encoder input shape."""
    return {
        "decision_id": str(row.decision_id),
        "chosen_action_id": int(row.chosen_action_id),
        "actor_observation": dict(row.actor_observation),
    }


#: Max parallel game-expansion workers (bounded pool: larger requests clamp
#: here, never spawn unbounded processes; 16 keeps spawn RSS + file
#: descriptors predictable next to the scan pool).
#:
#: Joint worker budget — size ``config.data.num_workers`` against cores, not
#: against one pool: the same knob sizes the one-shot pre-train scan pool
#: (processes, released before training starts), the live ``PrefetchGameStream``
#: decode threads, AND this expansion pool (processes, alive for the whole
#: run). Scan never overlaps training, but prefetch threads + expansion procs
#: + the main torch threads train concurrently, so keep
#: ``2 * num_workers + main torch threads <= cores``. :func:`_pool_worker_init`
#: clamps each worker to one thread (fixes thread oversubscription, not process
#: oversubscription). Oversubscribed symptoms: fills slower than the serial
#: path (thrash), spawn RSS + fd pressure (every proc re-imports torch), and
#: climbing ``PrefetchGameStream.waits``. When fills stop scaling, lower
#: ``num_workers`` before raising it.
_PARALLEL_EXPAND_MAX_WORKERS = 16


def _pool_worker_init() -> None:
    """Clamp per-worker thread pools (spawn-pool initializer, both pools).

    A fresh spawn worker inherits a full-size torch/OpenMP thread pool
    (defaults scale with host cores), so ``N`` workers x ``C`` threads
    oversubscribe ``C`` cores with ``N*C`` runnable threads — context-switch
    thrash that fills slower than serial. Clamping to one thread per worker
    keeps the pool's parallelism at exactly ``max_workers`` processes; the env
    defaults cover native (OpenMP/OpenBLAS/MKL) regions entered before torch
    is first touched in the worker. Runs in workers only — parent threads
    are untouched.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    with contextlib.suppress(Exception):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)


def _expand_streamed_game(
    payload: tuple[Any, str, str, bool],
) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]], bool]:
    """Pure per-game expansion for the process pool (no shared mutable state).

    ``payload`` is ``(game, split, replay_backend, need_privileged)``;
    returns ``(row_dicts, privileged_pairs, sim_path)`` with actor rows
    already projected through :func:`_row_to_dict` and privileged labels as
    plain ``(decision_id, label)`` pairs, so the parent merges plain data
    only. Runs the identical entry points as the serial pull
    (:func:`_expand_game_rows` plus :func:`expand_privileged_rows`), hence
    identical rows and identical :class:`ContractError` messages; failures
    propagate to the parent, which quarantines-and-counts per game exactly
    like :meth:`_StreamDataset._pull_game` (reason classes verbatim, order
    by pull sequence). Anything that is not a :class:`ContractError`
    propagates fail-closed, as in serial.
    """
    game, split, backend, need_privileged = payload
    actor_rows, sim_path = _expand_game_rows(game, split, backend)
    priv_rows = expand_privileged_rows(game, split=split) if need_privileged else []
    row_dicts = [_row_to_dict(row) for row in actor_rows]
    priv_pairs = [(str(priv.decision_id), dict(priv.privileged_label)) for priv in priv_rows]
    return (row_dicts, priv_pairs, sim_path)


class _StreamDataset:
    """Loop-facing microbatch source over a lazily-pulled game stream.

    Exposes the :class:`SupervisedLoop` dataset surface (``next_batch``,
    ``get_sampler_state``/``set_sampler_state``, ``__len__``, ``cursor``).
    Games are pulled on demand, expanded to actor rows, and encoded per
    microbatch through the real encoder; privileged rows ride along into
    :attr:`privileged` (train-split rows only — this dataset never sees
    validation games). Expansion dispatches on wall presence: games carrying
    ``wall_tiles`` expand through :func:`expand_game` unchanged, wall-less
    games replay through :func:`_expand_game_rows` on the configured
    ``replay_backend`` (``"python"`` wall-less replay or the ``"rust_json"``
    handoff); expansion failures quarantine-and-count per game
    (never fail-soft) in :attr:`expand_quarantined`, with per-path game
    counts in :attr:`replayed` / :attr:`sim_replayed`. ``expand_workers > 0``
    expands pulled games in a bounded spawn pool (cache-free pure map,
    ordered merge by pull sequence: rows, counters, and quarantine classes
    match the serial path game-for-game); ``0`` is the serial reference.
    Single-pass: the stream is consumed once, epoch pinned 0; exhaustion
    with rows still demanded fails closed (rescope ``max_updates`` to
    supply), and resume seeks to the recorded frontier with verbatim buffer
    + RNG + dedup-prefix restore.
    """

    def __init__(
        self,
        *,
        stream_factory: Callable[[], GameStream],
        num_actions: int,
        feature_dim: int,
        seed: int,
        drop_last: bool,
        need_privileged: bool = True,
        replay_backend: str = "python",
        expand_workers: int = 0,
    ) -> None:
        if drop_last is not True:
            raise ContractError(f"single-pass requires drop_last=true, got {drop_last!r}")
        if (
            isinstance(expand_workers, bool)
            or not isinstance(expand_workers, int)
            or expand_workers < 0
        ):
            raise ContractError(f"expand_workers invalid: {expand_workers!r} (non-negative int)")
        # Bounded pool: larger requests clamp to _PARALLEL_EXPAND_MAX_WORKERS.
        self._expand_workers = min(expand_workers, _PARALLEL_EXPAND_MAX_WORKERS)
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
        self._buffered_entries: list[dict[str, Any]] = []

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

        Wall-bound games expand through :func:`expand_game` unchanged;
        wall-less games replay through :func:`_expand_game_rows` on this
        dataset's ``replay_backend``. Privileged rows expand only when the dataset
        was built with ``need_privileged`` (positive auxiliary weights);
        BC-only runs skip them so label-less games still train. Either path
        raising :class:`ContractError` quarantines-and-counts the game
        (fail closed, never a silent drop); only fully expanded games buffer rows.
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
                actor_rows, sim_path = _expand_game_rows(
                    streamed.game, streamed.split, self._replay_backend
                )
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
            for row in actor_rows:
                self._rows.append(_row_to_dict(row))
            self._track_buffered(streamed, len(actor_rows))
            for priv in priv_rows:
                label = dict(priv.privileged_label)
                self.privileged.setdefault(str(priv.decision_id), label)
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
            self.privileged.setdefault(decision_id, label)

    def _fill_parallel(self, need: int) -> None:
        """Buffer at least ``need`` rows via ordered parallel expansion.

        Pulls raw games in stream order, expands each batch in the bounded
        spawn pool (:func:`_expand_streamed_game` pure map, no shared mutable
        state), and merges strictly in pull order via per-sequence futures —
        never ``pool.map`` (one game's failure cancels the batch tail there).
        Rows, counters, and quarantine classes match the serial :meth:`_fill`
        game-for-game; per-game :class:`ContractError` quarantines-and-counts
        (fail closed, never a silent drop) while any other worker failure
        propagates. Terminal exhaustion raises the serial messages verbatim.
        """
        pool = self._get_expand_pool()
        while self._live_count() < need:
            batch = self._pull_streamed_batch(self._expand_workers)
            if not batch:
                if self._live_count() == 0 and self._offset == 0:
                    raise ContractError("stream yielded zero train rows (empty split?)")
                raise ContractError(
                    f"stream exhausted with {self._live_count()} buffered rows, need {need} "
                    "(single-pass: stream end with updates remaining; "
                    "rescope loop.max_updates to supply)"
                )
            payloads = [
                (streamed.game, streamed.split, self._replay_backend, self._need_privileged)
                for streamed in batch
            ]
            futures = [pool.submit(_expand_streamed_game, payload) for payload in payloads]
            for streamed, future in zip(batch, futures, strict=True):
                try:
                    result = future.result()
                except ContractError as exc:
                    self._count_quarantine(exc)
                    continue
                self._merge_expanded(streamed, result)

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
        if self._expand_workers > 0:
            self._fill_parallel(need)
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

    def _consume_microbatch(self, batch_size: int) -> list[dict[str, Any]]:
        """Advance exactly one microbatch through the fill machine."""
        self._fill(batch_size)
        start = self._offset - self._dropped
        taken = self._rows[start : start + batch_size]
        self._offset += len(taken)
        self._microbatches_in_epoch += 1
        self._compact()
        return taken

    def next_batch(self, batch_size: int) -> dict[str, Any]:
        """Encode and return the next microbatch, advancing the cursor."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ContractError(f"batch_size must be a positive int, got {batch_size!r}")
        taken = self._consume_microbatch(batch_size)
        if len(taken) == 0:
            raise ContractError("stream dataset produced an empty microbatch")
        batch = encode_observation_rows(
            taken, num_actions=self._num_actions, feature_dim=self._feature_dim
        )
        batch["_decision_ids"] = [str(row["decision_id"]) for row in taken]
        batch["_epoch"] = torch.tensor(self._epoch)
        return batch

    def get_sampler_state(self) -> dict[str, Any]:
        return {
            "offset": self._offset,
            "seed": self._seed,
            "total": len(self._rows),
            "epoch": self._epoch,
            "dropped": self._dropped,
        }

    def set_sampler_state(self, state: Any) -> None:
        if isinstance(state, dict):
            offset = state.get("offset", 0)
            epoch = state.get("epoch", 0)
        else:
            offset = getattr(state, "offset", 0)
            epoch = getattr(state, "epoch", 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ContractError(f"sampler offset invalid: {offset!r}")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch != 0:
            raise ContractError(f"sampler epoch must be 0 (single-pass), got {epoch!r}")
        if offset < self._dropped:
            raise ContractError(
                f"sampler offset {offset} precedes compacted prefix {self._dropped}"
            )
        if offset > len(self._rows) + self._dropped:
            raise ContractError(f"sampler offset {offset} beyond buffered rows {len(self._rows)}")
        self._offset = offset
        self._epoch = 0

    def buffered_row_hash(self) -> str:
        """Deterministic hash of the live ``_rows`` window (decision_id order)."""
        h = hashlib.sha256()
        for row in self._rows:
            h.update(str(row.get("decision_id", "")).encode())
            h.update(b"\x00")
        return "sha256:" + h.hexdigest()

    def buffer_snapshot(self) -> dict[str, Any]:
        """Fast-resume snapshot: whole-game entries + counters + row hash."""
        total = sum(int(entry["rows"]) for entry in self._buffered_entries)
        if total != len(self._rows):
            raise ContractError(
                f"buffer index drift: {total} indexed rows != {len(self._rows)} buffered"
            )
        return {
            "entries": [dict(entry) for entry in self._buffered_entries],
            "offset": self._offset,
            "dropped": self._dropped,
            "epoch": self._epoch,
            "microbatches_in_epoch": self._microbatches_in_epoch,
            "replayed": self.replayed,
            "sim_replayed": self.sim_replayed,
            "expand_quarantined": self.expand_quarantined,
            "expand_quarantine_reasons": dict(self.expand_quarantine_reasons),
            "row_hash": self.buffered_row_hash(),
            "total_rows": len(self._rows),
            "replay_backend": self._replay_backend,
        }

    def restore_buffer(self, snapshot: Mapping[str, Any]) -> None:
        """Rebuild ``_rows`` verbatim from snapshot entries (fail-closed).
        Re-expands each buffered game through the identical serial entry points
        as live pulls, then verifies length + row hash before restoring logical
        counters. Only the buffered tail is re-expanded (``O(buffer)``), never
        the epoch.
        """
        from hydra2.data.stream import fetch_game_at as _fetch

        if not isinstance(snapshot, dict):
            raise ContractError("dataset buffer snapshot must be a mapping")
        entries = snapshot.get("entries")
        if not isinstance(entries, list):
            raise ContractError("dataset buffer entries must be a list")
        recorded_backend = snapshot.get("replay_backend", "python")
        if recorded_backend != self._replay_backend:
            raise ContractError(
                f"dataset buffer replay_backend {recorded_backend!r} != "
                f"live backend {self._replay_backend!r} (cross-backend restore refused)"
            )
        for entry in entries:
            if not isinstance(entry, dict):
                raise ContractError("dataset buffer entry must be a mapping")
            unknown = sorted(
                k for k in entry if k not in ("key", "path", "offset", "split", "rows")
            )
            if unknown:
                raise ContractError(f"dataset buffer entry unknown keys {unknown}")
            key, path, offset, split, rows = (
                entry.get("key"),
                entry.get("path"),
                entry.get("offset"),
                entry.get("split"),
                entry.get("rows"),
            )
            if not isinstance(key, str) or key == "":
                raise ContractError("dataset buffer entry key must be a non-empty str")
            if not isinstance(path, str) or path == "":
                raise ContractError("dataset buffer entry path must be a non-empty str")
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise ContractError("dataset buffer entry offset must be non-negative int")
            if not isinstance(split, str) or split == "":
                raise ContractError("dataset buffer entry split must be a non-empty str")
            if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
                raise ContractError("dataset buffer entry rows must be a positive int")
        offset = snapshot.get("offset")
        dropped = snapshot.get("dropped")
        epoch = snapshot.get("epoch")
        micro = snapshot.get("microbatches_in_epoch")
        for name, value in (
            ("offset", offset),
            ("dropped", dropped),
            ("epoch", epoch),
            ("microbatches_in_epoch", micro),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"dataset buffer {name} must be a non-negative int")
        expected_hash = snapshot.get("row_hash")
        if not isinstance(expected_hash, str) or not expected_hash.startswith("sha256:"):
            raise ContractError("dataset buffer row_hash must be a sha256 string")
        rebuilt_rows: list[dict[str, Any]] = []
        rebuilt_priv: dict[str, dict[str, Any]] = {}
        replayed = 0
        sim_replayed = 0
        for entry in entries:
            recorded_split = str(entry["split"])
            fetched = _fetch(
                entry["path"],
                int(entry["offset"]),  # type: ignore[arg-type]
                seed=self._seed,
                ratios={recorded_split: 1.0},
                expected_sha=str(entry["key"]),
            )
            try:
                actor_rows, sim_path = _expand_game_rows(
                    fetched.game, recorded_split, self._replay_backend
                )
                priv_rows = (
                    expand_privileged_rows(fetched.game, split=recorded_split)
                    if self._need_privileged
                    else []
                )
            except ContractError as exc:
                raise ContractError(f"buffered game failed to re-expand: {exc}") from exc
            if len(actor_rows) != int(entry["rows"]):
                raise ContractError("buffered game row count mismatch on restore")
            rebuilt_rows.extend(_row_to_dict(row) for row in actor_rows)
            for priv in priv_rows:
                rebuilt_priv.setdefault(str(priv.decision_id), dict(priv.privileged_label))
            if sim_path:
                sim_replayed += 1
            else:
                replayed += 1
        # Verify verbatim before mutating live state.
        h = hashlib.sha256()
        for row in rebuilt_rows:
            h.update(str(row.get("decision_id", "")).encode())
            h.update(b"\x00")
        actual_hash = "sha256:" + h.hexdigest()
        if actual_hash != expected_hash:
            raise ContractError("dataset buffer row hash mismatch on restore")
        if len(rebuilt_rows) != int(snapshot.get("total_rows", len(rebuilt_rows))):
            raise ContractError("dataset buffer total_rows mismatch on restore")
        if not (dropped <= offset <= dropped + len(rebuilt_rows)):  # type: ignore[operator]
            raise ContractError("dataset buffer offset outside live window on restore")
        # Commit (verify-then-mutate): rows + privileged + index + counters.
        self._rows = rebuilt_rows
        self.privileged = rebuilt_priv
        self._buffered_entries = [dict(entry) for entry in entries]  # type: ignore[union-attr]
        self._offset = int(offset)  # type: ignore[arg-type]
        self._dropped = int(dropped)  # type: ignore[arg-type]
        self._microbatches_in_epoch = int(micro)  # type: ignore[arg-type]
        if int(epoch) != 0:  # type: ignore[arg-type]
            raise ContractError(f"dataset buffer epoch {epoch!r} != 0 (single-pass)")
        self._epoch = 0
        # Prefix totals (fail-closed when absent/malformed; never recomputed).
        for name in ("replayed", "sim_replayed", "expand_quarantined"):
            value = snapshot.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"dataset buffer {name} must be a non-negative int")
        reasons = snapshot.get("expand_quarantine_reasons")
        if not isinstance(reasons, dict):
            raise ContractError("dataset buffer quarantine reasons must be a mapping")
        self.replayed = int(snapshot["replayed"])  # type: ignore[arg-type]
        self.sim_replayed = int(snapshot["sim_replayed"])  # type: ignore[arg-type]
        self.expand_quarantined = int(snapshot["expand_quarantined"])  # type: ignore[arg-type]
        self.expand_quarantine_reasons = {str(k): int(v) for k, v in reasons.items()}

    @property
    def cursor(self) -> int:
        return self._offset

    def __len__(self) -> int:
        return len(self._rows)


@dataclass(slots=True)
class _ScanReport:
    """Pre-train corpus scan: split walls, eval ledger, quarantine counts."""

    train_walls: frozenset[str]
    val_walls: frozenset[str]
    train_games: int
    val_games: int
    train_sim_games: int
    val_sim_games: int
    framed: int
    emitted: int
    quarantined: int
    duplicates: int


def _scan_file_games(path_str: str) -> list[tuple[str, str | None, bool] | None]:
    """Decode+validate every game in one file (scan-worker pure function).

    Returns per game, in file order: ``None`` when undecodable/invalid,
    else ``(raw_bytes_sha256, wall_hash_or_None, wall_tiles_is_None)``.
    Mirrors :meth:`GameStream._decode_inline` failure classes exactly;
    file-level I/O errors propagate like the serial scan (fail closed).
    """
    from pathlib import Path

    out: list[tuple[str, str | None, bool] | None] = []
    for _, game_bytes in ZstdLineStream(path_str).iter_games():
        try:
            game = decode_game_object(
                object_id=stem_of(Path(path_str)),
                packaged_object_id=stem_of(Path(path_str)),
                decoded_bytes=game_bytes,
            )
        except (ContractError, CorruptArtifactError, ValueError):
            out.append(None)
            continue
        try:
            validate_game(game)
        except (ContractError, CorruptArtifactError, ValueError):
            out.append(None)
            continue
        out.append((game.raw_bytes_sha256, compute_wall_hash(game), game.wall_tiles is None))
    return out


def _scan_corpus(
    manifest: StreamManifest, *, config: RunConfig, ratios: dict[str, float]
) -> _ScanReport:
    """One ordered pass: wall sets per split, eval ledger, quarantine counts.

    Uses ``split=None`` so every valid game is emitted with its assignment
    attached, then partitions by the config split names. Raises before any
    training state exists when one wall lands in both splits.
    """
    if config.data.num_workers > 0:
        return _scan_corpus_parallel(manifest, config=config, ratios=ratios)
    common: dict[str, Any] = {
        "seed": config.seeds.data_seed,
        "ratios": ratios,
        "epoch": 0,
        "split": None,
        "shuffle_buffer": 0,
    }
    stream = GameStream(manifest, **common)
    train_walls: set[str] = set()
    val_walls: set[str] = set()
    train_games = 0
    val_games = 0
    train_sim_games = 0
    val_sim_games = 0
    for game in stream:
        if game.split == config.data.train_split:
            train_games += 1
            if game.wall_hash is not None:
                train_walls.add(game.wall_hash)
            if game.game.wall_tiles is None:
                train_sim_games += 1
        elif game.split == config.data.val_split:
            val_games += 1
            if game.wall_hash is not None:
                val_walls.add(game.wall_hash)
            if game.game.wall_tiles is None:
                val_sim_games += 1
    stats = stream.stats
    return _scan_report(
        config,
        train_walls,
        val_walls,
        train_games,
        val_games,
        train_sim_games,
        val_sim_games,
        stats.framed,
        stats.emitted,
        stats.quarantined,
        stats.duplicates,
    )


def _scan_report(
    config: RunConfig,
    train_walls: set[str],
    val_walls: set[str],
    train_games: int,
    val_games: int,
    train_sim_games: int,
    val_sim_games: int,
    framed: int,
    emitted: int,
    quarantined: int,
    duplicates: int,
) -> _ScanReport:
    """Shared scan tail: wall-disjoint gate plus the report record."""
    overlap = sorted(train_walls & val_walls)
    if overlap:
        raise ContractError(
            f"wall {overlap[0][:16]} in splits "
            f"{config.data.train_split} and {config.data.val_split}"
        )
    return _ScanReport(
        train_walls=frozenset(train_walls),
        val_walls=frozenset(val_walls),
        train_games=train_games,
        val_games=val_games,
        train_sim_games=train_sim_games,
        val_sim_games=val_sim_games,
        framed=framed,
        emitted=emitted,
        quarantined=quarantined,
        duplicates=duplicates,
    )


def _scan_corpus_parallel(
    manifest: StreamManifest, *, config: RunConfig, ratios: dict[str, float]
) -> _ScanReport:
    """Process-parallel scan; bit-identical to the serial pass by construction.

    Decode+validate fan out over files (pure per-file work); the merge below
    replays :meth:`GameStream._finish_decode` accounting line-for-line in
    manifest order (``pool.map`` preserves order, so exact-hash dedup keeps
    first-seen-wins). Split assignment is pure per game
    (:func:`assign_split`), so parallel decode cannot change partitions.
    """
    paths = [str(entry.path) for entry in manifest.files]
    keys = [group_key_for_path(entry.path) for entry in manifest.files]
    train_walls: set[str] = set()
    val_walls: set[str] = set()
    train_games = 0
    val_games = 0
    train_sim_games = 0
    val_sim_games = 0
    framed = 0
    emitted = 0
    quarantined = 0
    duplicates = 0
    hashes: set[str] = set()
    # Spawn (never fork): the parent has torch OMP threads alive and
    # fork-with-threads deadlocks racily; spawn re-imports clean workers.
    # Built once per run (the cached scan runs a single pass before any
    # training state exists — never per epoch); workers start single-threaded
    # under _pool_worker_init (see the joint budget at
    # _PARALLEL_EXPAND_MAX_WORKERS).
    ctx = _mp_get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=config.data.num_workers, mp_context=ctx, initializer=_pool_worker_init
    ) as pool:
        for file_index, games in enumerate(pool.map(_scan_file_games, paths, chunksize=32)):
            group_key = keys[file_index]
            for item in games:
                framed += 1
                if item is None:
                    quarantined += 1
                    continue
                raw_sha, wall_hash, is_sim = item
                if raw_sha in hashes:
                    quarantined += 1
                    duplicates += 1
                    continue
                hashes.add(raw_sha)
                assigned = assign_split(
                    group_key=group_key, seed=config.seeds.data_seed, ratios=ratios
                )
                emitted += 1
                if assigned == config.data.train_split:
                    train_games += 1
                    if wall_hash is not None:
                        train_walls.add(wall_hash)
                    if is_sim:
                        train_sim_games += 1
                elif assigned == config.data.val_split:
                    val_games += 1
                    if wall_hash is not None:
                        val_walls.add(wall_hash)
                    if is_sim:
                        val_sim_games += 1
    return _scan_report(
        config,
        train_walls,
        val_walls,
        train_games,
        val_games,
        train_sim_games,
        val_sim_games,
        framed,
        emitted,
        quarantined,
        duplicates,
    )


def _scan_corpus_cached(
    manifest: StreamManifest,
    *,
    config: RunConfig,
    ratios: dict[str, float],
    stream_digest: str,
    run_dir: Path,
) -> _ScanReport:
    """Pre-train scan with content-hash cache (miss → full scan + save).
    Cache key is ``(manifest digest, seed, ratios, split names)``; reload
    re-verifies the digest and wall-disjointness. Stale/corrupt entries fail
    closed to a full scan (never raise, never partial).
    """
    cache_file = scan_cache_path(run_dir)
    cached = load_scan_cache(
        cache_file,
        manifest_digest=stream_digest,
        seed=config.seeds.data_seed,
        ratios=ratios,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
    )
    if cached is not None:
        try:
            return _scan_report(
                config,
                set(cached["train_walls"]),  # type: ignore[arg-type]
                set(cached["val_walls"]),  # type: ignore[arg-type]
                int(cached["train_games"]),  # type: ignore[arg-type]
                int(cached["val_games"]),  # type: ignore[arg-type]
                int(cached["train_sim_games"]),  # type: ignore[arg-type]
                int(cached["val_sim_games"]),  # type: ignore[arg-type]
                int(cached["framed"]),  # type: ignore[arg-type]
                int(cached["emitted"]),  # type: ignore[arg-type]
                int(cached["quarantined"]),  # type: ignore[arg-type]
                int(cached["duplicates"]),  # type: ignore[arg-type]
            )
        except (ContractError, ValueError, TypeError, KeyError, AttributeError):
            pass
    report = _scan_corpus(manifest, config=config, ratios=ratios)
    save_scan_cache(
        cache_file,
        manifest_digest=stream_digest,
        seed=config.seeds.data_seed,
        ratios=ratios,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        scan={
            "train_walls": sorted(report.train_walls),
            "val_walls": sorted(report.val_walls),
            "train_games": report.train_games,
            "val_games": report.val_games,
            "train_sim_games": report.train_sim_games,
            "val_sim_games": report.val_sim_games,
            "framed": report.framed,
            "emitted": report.emitted,
            "quarantined": report.quarantined,
            "duplicates": report.duplicates,
        },
    )
    return report


def _verify_fast_snapshots(*, sidecar: dict[str, Any], payload: Any, ckpt: Path) -> None:
    """Cross-check sidecar vs payload fast snapshots (tamper → raise)."""
    if not isinstance(payload, dict):
        raise ContractError(f"checkpoint payload must be a mapping: {ckpt}")
    side_dataset = sidecar.get("dataset_buffer")
    payload_dataset = payload.get("dataset_buffer")
    if not isinstance(side_dataset, dict) or not isinstance(payload_dataset, dict):
        raise ContractError(f"checkpoint fast snapshots missing: {ckpt}")
    if side_dataset != payload_dataset:
        raise ContractError(f"checkpoint dataset buffer mismatch (tampered?): {ckpt}")
    side_shuffle = sidecar.get("shuffle")
    payload_shuffle = payload.get("shuffle_buffer")
    if not isinstance(side_shuffle, dict) or not isinstance(payload_shuffle, dict):
        raise ContractError(f"checkpoint shuffle snapshots missing: {ckpt}")
    for key in ("buffer_keys", "buffer_rng_state", "epoch_seed", "buffer_size", "prefix_hashes"):
        if side_shuffle.get(key) != payload_shuffle.get(key):
            raise ContractError(f"checkpoint shuffle {key} mismatch (tampered?): {ckpt}")
    side_entries = side_shuffle.get("buffer_entries", [])
    payload_entries = payload_shuffle.get("buffer_entries", [])
    if side_entries != payload_entries:
        raise ContractError(f"checkpoint shuffle entries mismatch (tampered?): {ckpt}")


def _parse_dataset_buffer_sidecar(raw: Any, *, ckpt: Path) -> dict[str, Any]:
    """Strict validation of the sidecar ``dataset_buffer`` (tamper → raise)."""
    if not isinstance(raw, dict):
        raise ContractError(f"checkpoint dataset_buffer must be a mapping: {ckpt}")
    entries = raw.get("entries")
    if not isinstance(entries, list):
        raise ContractError(f"checkpoint dataset_buffer entries must be a list: {ckpt}")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ContractError(f"checkpoint dataset_buffer entry must be a mapping: {ckpt}")
        unknown = sorted(k for k in entry if k not in ("key", "path", "offset", "split", "rows"))
        if unknown:
            raise ContractError(f"checkpoint dataset_buffer entry unknown keys {unknown}: {ckpt}")
    for name in ("offset", "dropped", "epoch", "microbatches_in_epoch", "total_rows"):
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"checkpoint dataset_buffer {name} invalid: {ckpt}")
    for name in ("replayed", "sim_replayed", "expand_quarantined"):
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"checkpoint dataset_buffer {name} invalid: {ckpt}")
    reasons = raw.get("expand_quarantine_reasons")
    if not isinstance(reasons, dict):
        raise ContractError(f"checkpoint dataset_buffer quarantine reasons invalid: {ckpt}")
    row_hash = raw.get("row_hash")
    if not isinstance(row_hash, str) or not row_hash.startswith("sha256:"):
        raise ContractError(f"checkpoint dataset_buffer row_hash invalid: {ckpt}")
    exhausted = raw.get("stream_exhausted", False)
    if not isinstance(exhausted, bool):
        raise ContractError(f"checkpoint dataset_buffer stream_exhausted invalid: {ckpt}")
    return dict(raw)


def _build_model(config: RunConfig) -> Any:
    """Real baseline model bound to the config (config-driven, fail closed)."""
    from hydra2.models.model import Hydra2BaselineModel
    from hydra2.models.schema import BASELINE_ACTION_COUNT

    if config.model.architecture_id != "hydra2_baseline_transformer_v1":
        raise ContractError(
            f"stream training supports architecture "
            f"'hydra2_baseline_transformer_v1', got {config.model.architecture_id!r}"
        )
    if config.model.action_count != BASELINE_ACTION_COUNT:
        raise ContractError(
            f"model.action_count {config.model.action_count} != baseline {BASELINE_ACTION_COUNT}"
        )
    params = dict(config.model.parameters)
    unknown = sorted(k for k in params if k not in _MODEL_PARAMETERS)
    if unknown:
        raise ContractError(f"model.parameters unknown keys {unknown}")
    for key in ("d_model", "n_layers", "n_heads", "d_ff"):
        if key in params and (isinstance(params[key], bool) or not isinstance(params[key], int)):
            raise ContractError(f"model.parameters[{key!r}] must be an int")
    if "dropout" in params and (
        isinstance(params["dropout"], bool)
        or not isinstance(params["dropout"], (int, float))
        or not 0.0 <= float(params["dropout"]) < 1.0
    ):
        raise ContractError("model.parameters['dropout'] must lie in [0, 1)")
    return Hydra2BaselineModel(action_count=config.model.action_count, **params)


def _build_optimizer(config: RunConfig, model: Any) -> Any:
    """Registered optimizer id plus the config battery (no silent defaults)."""
    if config.optimizer.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
        )
    if config.optimizer.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
        )
    if config.optimizer.name == "sgd":
        # v1 config carries no momentum knob: plain SGD (momentum 0).
        return torch.optim.SGD(
            model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay
        )
    raise ContractError("optimizer.id must be one of ['adamw', 'adam', 'sgd']")


def _build_scheduler(config: RunConfig, optimizer: Any) -> Any:
    """Warmup + registered decay, stepped once per global update by the loop."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR

    name = config.scheduler.name
    warmup = config.scheduler.warmup_updates
    horizon = config.loop.max_updates
    parameters = dict(config.scheduler.parameters)
    if name == "cosine":
        unknown = sorted(k for k in parameters if k != "T_max")
        if unknown:
            raise ContractError(f"scheduler.parameters unknown keys {unknown}")
        main_span = max(1, horizon - warmup) if warmup < horizon else horizon
        main = CosineAnnealingLR(optimizer, T_max=int(parameters.get("T_max", main_span)))
    elif name == "constant":
        if parameters:
            raise ContractError(f"scheduler.parameters unknown keys {sorted(parameters)}")
        main = LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    elif name == "linear":
        unknown = sorted(k for k in parameters if k != "end_factor")
        if unknown:
            raise ContractError(f"scheduler.parameters unknown keys {unknown}")
        end_factor = float(parameters.get("end_factor", 0.0))
        if not 0.0 <= end_factor <= 1.0:
            raise ContractError(f"scheduler end_factor must lie in [0, 1], got {end_factor}")
        span = max(1, horizon - warmup) if warmup < horizon else horizon
        main = LinearLR(optimizer, start_factor=1.0, end_factor=end_factor, total_iters=span)
    else:
        raise ContractError("scheduler.id must be one of ['cosine', 'constant', 'linear']")
    if warmup <= 0 or warmup >= horizon:
        if warmup <= 0:
            return main
        # Warmup covers the whole horizon: single warmup ramp, no decay tail.
        return LinearLR(optimizer, start_factor=0.01, total_iters=horizon)
    warm = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
    return SequentialLR(optimizer, schedulers=[warm, main], milestones=[warmup])


def _manifest_hashes_for_loop(
    config: RunConfig, *, model: Any, manifest: StreamManifest
) -> dict[str, str]:
    """Derive all ten loop manifest digests from authoritative sources.

    Never constant hashes: run/manifest digests come from the resolved config
    and stream manifest, model/utility digests from the live model, rules and
    schema digests from the pinned repo artifacts, the environment digest from
    the engine identity, and optimizer/scheduler digests from the pinned
    config sections. Derivation failure raises loudly.
    """
    from hydra2.config import repo_root
    from hydra2.contracts.action import load_action_table
    from hydra2.contracts.observation import observation_schema_digest
    from hydra2.engines.riichienv.identity import ENGINE_IDENTITY

    repo = repo_root()
    try:
        rules_bytes = (repo / "configs" / "rules" / "tenhou_4p_hanchan_v1.json").read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot derive rules_hash for training: {exc}") from exc
    try:
        action_table_path = repo / "configs" / "contracts" / "action_table_v1.json"
        action_digest = str(load_action_table(action_table_path).digest)
    except (ContractError, OSError, ValueError) as exc:
        raise ContractError(f"cannot derive action_schema_hash for training: {exc}") from exc
    try:
        observation_digest = str(observation_schema_digest())
    except (ContractError, ValueError) as exc:
        raise ContractError(f"cannot derive observation_schema_hash for training: {exc}") from exc
    optimizer_doc = {
        "id": config.optimizer.name,
        "lr": config.optimizer.lr,
        "betas": list(config.optimizer.betas),
        "weight_decay": config.optimizer.weight_decay,
    }
    scheduler_doc = {
        "id": config.scheduler.name,
        "warmup_updates": config.scheduler.warmup_updates,
        "parameters": dict(config.scheduler.parameters),
    }
    stream_digest = manifest_digest(manifest)
    return {
        "run_spec_hash": run_config_digest(config),
        "model_spec_hash": str(model.model_identity),
        "optimizer_spec_hash": str(of_canonical(optimizer_doc)),
        "scheduler_spec_hash": str(of_canonical(scheduler_doc)),
        "environment_hash": str(ENGINE_IDENTITY.environment_hash),
        "rules_hash": "sha256:" + hashlib.sha256(rules_bytes).hexdigest(),
        "utility_manifest_hash": str(model.utility_manifest_hash),
        "action_schema_hash": action_digest,
        "observation_schema_hash": observation_digest,
        "dataset_manifest_hash": stream_digest,
    }


def _rng_anchors() -> dict[str, Any]:
    """JSON-safe anchors over the live torch RNG states (sidecar bundle)."""
    state = capture_rng_state()
    anchors: dict[str, Any] = {
        "torch_cpu_sha256": "sha256:" + hashlib.sha256(bytes(state["cpu"])).hexdigest()
    }
    if "cuda" in state and state["cuda"] is not None:
        cuda = state["cuda"]
        if isinstance(cuda, list):
            anchors["torch_cuda_sha256"] = [
                "sha256:" + hashlib.sha256(bytes(device_state)).hexdigest() for device_state in cuda
            ]
        else:
            anchors["torch_cuda_sha256"] = "sha256:" + hashlib.sha256(bytes(cuda)).hexdigest()
    else:
        anchors["torch_cuda_sha256"] = None
    return anchors


def _verify_rng_anchors(payload_rng: Any, anchors: Any, *, ckpt: Path) -> None:
    """Recompute RNG anchors from the loaded payload; raise on any mismatch."""
    if not isinstance(payload_rng, dict) or "cpu" not in payload_rng:
        raise ContractError(f"checkpoint payload rng_state malformed: {ckpt}")
    if not isinstance(anchors, dict):
        raise ContractError(f"checkpoint sidecar rng_state must be a mapping: {ckpt}")
    expected_cpu = anchors.get("torch_cpu_sha256")
    actual_cpu = "sha256:" + hashlib.sha256(bytes(payload_rng["cpu"])).hexdigest()
    if expected_cpu != actual_cpu:
        raise ContractError(f"checkpoint RNG state mismatch (torch cpu): {ckpt}")
    payload_cuda = payload_rng.get("cuda")
    expected_cuda = anchors.get("torch_cuda_sha256")
    if payload_cuda is None and expected_cuda is None:
        return
    if payload_cuda is None or expected_cuda is None:
        raise ContractError(f"checkpoint RNG state mismatch (torch cuda): {ckpt}")
    actual_list = (
        ["sha256:" + hashlib.sha256(bytes(s)).hexdigest() for s in payload_cuda]
        if isinstance(payload_cuda, list)
        else "sha256:" + hashlib.sha256(bytes(payload_cuda)).hexdigest()
    )
    if actual_list != expected_cuda:
        raise ContractError(f"checkpoint RNG state mismatch (torch cuda): {ckpt}")


def _sidecar_cursor(data_cursor: DataStreamCursor) -> dict[str, int]:
    """Project the stream cursor onto the 5-field resume envelope."""
    return {
        "file_index": data_cursor.file_index,
        "byte_offset": data_cursor.byte_offset,
        "games_seen": data_cursor.games_seen,
        "seed": data_cursor.seed,
        "epoch": data_cursor.epoch,
    }


def _prefix_hashes_name(update: int) -> str:
    """Sidecar-adjacent prefix-hash filename for ``ckpt-<update>``."""
    return f"prefix-hashes-{update:06d}.txt"


def _load_prefix_hashes(
    checkpoint_dir: Path, record: Mapping[str, Any], *, ckpt: Path
) -> list[str]:
    """Load + verify the sidecar-adjacent prefix-hash file (tamper → raise)."""
    from pathlib import Path

    name = record.get("file")
    count = record.get("count")
    digest = record.get("sha256")
    if not isinstance(name, str) or name == "" or "/" in name or name.startswith("."):
        raise ContractError(f"checkpoint prefix_hashes file invalid: {ckpt}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ContractError(f"checkpoint prefix_hashes count invalid: {ckpt}")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ContractError(f"checkpoint prefix_hashes digest invalid: {ckpt}")
    try:
        blob = (Path(checkpoint_dir) / name).read_bytes()
    except OSError as exc:
        raise ContractError(f"checkpoint prefix_hashes unreadable: {ckpt} ({exc})") from exc
    if "sha256:" + hashlib.sha256(blob).hexdigest() != digest:
        raise ContractError(f"checkpoint prefix_hashes digest mismatch: {ckpt}")
    lines = blob.decode("utf-8").splitlines()
    if len(lines) != count:
        raise ContractError(f"checkpoint prefix_hashes count mismatch: {ckpt}")
    for sha in lines:
        if not sha.startswith("sha256:") or sha == "sha256:":
            raise ContractError(f"checkpoint prefix_hashes entry invalid: {ckpt}")
    if lines != sorted(lines):
        raise ContractError(f"checkpoint prefix_hashes order invalid: {ckpt}")
    return lines


def _write_streaming_checkpoint(
    *,
    run_dir: Path,
    update: int,
    run_digest: str,
    stream_digest: str,
    config: RunConfig,
    loop: SupervisedLoop,
    dataset: _StreamDataset,
    batch_size: int,
) -> Path:
    """Atomically publish ``ckpt-<update>.pt`` + ResumeExactPlan sidecar."""
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scheduler_state: Any = {}
    if loop.scheduler is not None and hasattr(loop.scheduler, "state_dict"):
        with contextlib.suppress(Exception):
            scheduler_state = loop.scheduler.state_dict()
    dataset_snapshot = dataset.buffer_snapshot()
    shuffle_entries: list[dict[str, Any]] = []
    shuffle_rng: dict[str, Any] = {}
    shuffle_keys: list[str] = []
    if config.data.shuffle_buffer_size > 0 and dataset._stream is not None:
        snap = dataset._stream.shuffle_snapshot()
        if snap is not None:
            entries, rng_state = snap
            shuffle_entries = [dict(e) for e in entries]
            shuffle_rng = dict(rng_state)
            shuffle_keys = [str(e["key"]) for e in shuffle_entries]
    prefix_hashes: list[str] = []
    if dataset._stream is not None:
        prefix_hashes = dataset._stream.prefix_hashes_snapshot()
    prefix_blob = "".join(f"{sha}\n" for sha in prefix_hashes).encode("utf-8")
    prefix_name = _prefix_hashes_name(update)
    atomic_replace_bytes(checkpoint_dir / prefix_name, prefix_blob)
    prefix_record = {
        "file": prefix_name,
        "count": len(prefix_hashes),
        "sha256": "sha256:" + hashlib.sha256(prefix_blob).hexdigest(),
    }
    payload: dict[str, Any] = {
        "model_state": loop.model.state_dict(),
        "optimizer_state": loop.optimizer.state_dict(),
        "scheduler_state": scheduler_state,
        "training_state": loop.state.to_dict(),
        "sampler_state": dataset.get_sampler_state(),
        "rng_state": capture_rng_state(),
        "loss_history": [dict(entry) for entry in loop.loss_history],
        "stream_epoch": dataset.epoch,
        "microbatch_size": batch_size,
        "rows_consumed_in_epoch": dataset.rows_consumed_in_epoch,
        "dataset_buffer": dataset_snapshot,
        "shuffle_buffer": {
            "buffer_keys": shuffle_keys,
            "buffer_entries": shuffle_entries,
            "buffer_rng_state": shuffle_rng,
            "epoch_seed": config.seeds.data_seed + dataset.epoch,
            "buffer_size": config.data.shuffle_buffer_size,
            "prefix_hashes": prefix_record,
        },
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    blob = buffer.getvalue()
    payload_digest = "sha256:" + hashlib.sha256(blob).hexdigest()
    ckpt_path = checkpoint_dir / f"ckpt-{update:06d}.pt"
    atomic_replace_bytes(ckpt_path, blob)

    data_cursor = dataset.stream_cursor()
    sidecar: dict[str, Any] = {
        "global_update": update,
        "run_digest": run_digest,
        "stream_cursor": _sidecar_cursor(data_cursor),
        "stream_shuffle_pos": data_cursor.shuffle_pos,
        "stream_epoch": dataset.epoch,
        "microbatch_size": batch_size,
        "rows_consumed_in_epoch": dataset.rows_consumed_in_epoch,
        "rng_state": _rng_anchors(),
        "shuffle": {
            "buffer_keys": shuffle_keys,
            "epoch_seed": config.seeds.data_seed + dataset.epoch,
            "buffer_size": config.data.shuffle_buffer_size,
            "buffer_rng_state": shuffle_rng,
            "buffer_entries": shuffle_entries,
            "prefix_hashes": prefix_record,
        },
        "dataset_buffer": dataset_snapshot,
        "accum": {"yielded_batches": update, "micro_in_update": 0},
        "worker_plan": {
            "num_workers": config.data.num_workers,
            "world_size": config.data.world_size,
            "rank": _SINGLETON_RANK,
        },
        "manifests": {
            "stream_manifest_hash": stream_digest,
            "dataset_manifest_hash": stream_digest,
        },
        "payload_sha256": payload_digest,
    }
    atomic_replace_bytes(
        ckpt_path.with_suffix(".json"),
        json.dumps(sidecar, indent=2, sort_keys=True).encode("utf-8"),
    )
    return ckpt_path


def _ckpt_names(run_dir: Path) -> list[str]:
    """``ckpt-<update>.pt`` names in numeric update order (torn names skipped)."""
    found: list[tuple[int, str]] = []
    for path in (run_dir / "checkpoints").glob("ckpt-*.pt"):
        try:
            found.append((int(path.stem.split("-")[1]), path.name))
        except (ValueError, IndexError):
            continue
    return [name for _, name in sorted(found)]


def _prune_checkpoints(run_dir: Path, *, keep: int | None) -> None:
    """Bound retained ``ckpt-<update>.pt`` files (null keeps all)."""
    if keep is None:
        return
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ContractError(f"keep_last_checkpoints must be a positive int, got {keep!r}")
    names = _ckpt_names(run_dir)
    for stale_name in names[: max(0, len(names) - keep)]:
        stale = run_dir / "checkpoints" / stale_name
        try:
            stale_update = int(stale_name.split("-")[1].split(".")[0])
        except (ValueError, IndexError):
            stale_update = -1
        with contextlib.suppress(OSError):
            stale.unlink()
        with contextlib.suppress(OSError):
            stale.with_suffix(".json").unlink()
        if stale_update >= 0:
            with contextlib.suppress(OSError):
                (run_dir / "checkpoints" / _prefix_hashes_name(stale_update)).unlink()


def _append_new_history(run_dir: Path, history: list[dict[str, float]]) -> int:
    """Append per-update rows for entries not yet in ``metrics.jsonl``.

    Resume-safe: already-logged updates (by ``global_update``) are skipped,
    so a resumed run never duplicates rows. Returns appended count.
    """
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    train_log = run_dir / "logs" / "train.log"
    logged: set[int] = set()
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip() == "":
                continue
            try:
                logged.add(int(json.loads(line).get("global_update", -1)))
            except (ValueError, AttributeError):
                continue
    appended = 0
    with (
        metrics_path.open("a", encoding="utf-8") as metrics_handle,
        train_log.open("a", encoding="utf-8") as log_handle,
    ):
        for entry in history:
            update = int(entry.get("global_update", -1))
            if update in logged:
                continue
            metrics_handle.write(json.dumps(entry, sort_keys=True) + "\n")
            log_handle.write(
                f"update={update:06d} total={entry.get('total', 0.0):.6f} "
                f"policy={entry.get('policy', 0.0):.6f} "
                f"top1={entry.get('top1', 0.0):.4f}\n"
            )
            logged.add(update)
            appended += 1
    return appended


def _run_holdout_eval(
    *,
    manifest: StreamManifest,
    ratios: dict[str, float],
    config: RunConfig,
    loop: SupervisedLoop,
    run_dir: Any,
    update: int,
) -> dict[str, Any] | None:
    """Held-out eval on val-split games (observer: never touches train state).

    Builds a throwaway val stream (fixed seed/epoch, so every eval sees
    identical batches), expands + encodes ``eval.num_batches`` microbatches,
    and runs :meth:`SupervisedLoop.evaluate_report` (no grad; the model is
    restored to train mode inside). Appends one row to ``eval/eval.jsonl``
    plus a ``train.log`` marker. ANY failure is logged and returns None --
    eval never aborts training. Resume-safe: fully stateless, and
    already-recorded updates are skipped, never duplicated.
    """
    eval_path = run_dir / "eval" / "eval.jsonl"
    train_log = run_dir / "logs" / "train.log"
    try:
        recorded: set[int] = set()
        if eval_path.is_file():
            for line in eval_path.read_text(encoding="utf-8").splitlines():
                if line.strip() == "":
                    continue
                try:
                    recorded.add(int(json.loads(line).get("update", -1)))
                except (ValueError, AttributeError):
                    continue
        if update in recorded:
            return None
        micro = config.eval.microbatch_size
        if micro is None:
            micro = config.loop.microbatch_size
        if isinstance(micro, bool) or not isinstance(micro, int) or micro <= 0:
            raise ContractError(f"eval microbatch invalid: {micro!r}")
        need_batches = config.eval.num_batches
        if isinstance(need_batches, bool) or not isinstance(need_batches, int) or need_batches <= 0:
            raise ContractError(f"eval num_batches invalid: {need_batches!r}")
        stream = GameStream(
            manifest,
            seed=config.seeds.data_seed,
            ratios=ratios,
            epoch=0,
            split=config.data.val_split,
            shuffle_buffer=0,
        )
        rows: list[dict[str, Any]] = []
        games_touched = 0
        expand_start = time.perf_counter()
        for streamed in stream:
            if len(rows) >= need_batches * micro:
                break
            games_touched += 1
            try:
                actor_rows, _ = _expand_game_rows(
                    streamed.game, streamed.split, config.data.replay_backend
                )
            except ContractError:
                continue
            for row in actor_rows:
                rows.append(_row_to_dict(row))
                if len(rows) >= need_batches * micro:
                    break
        expand_wall = time.perf_counter() - expand_start
        if not rows:
            raise ContractError("no val rows for held-out eval")
        encode_start = time.perf_counter()
        batches = [
            encode_observation_rows(
                rows[start : start + micro],
                num_actions=config.model.action_count,
                feature_dim=_FEATURE_FOLD_DIM,
            )
            for start in range(0, need_batches * micro, micro)
        ]
        encode_wall = time.perf_counter() - encode_start
        report = loop.evaluate_report(batches)
        entry: dict[str, Any] = {"update": update}
        for key, value in report.items():
            entry[str(key)] = float(value) if isinstance(value, (int, float)) else value
        with eval_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(
                f"eval:update={update:06d} nll={float(report.get('masked_nll', 0.0)):.6f} "
                f"top1={float(report.get('top1', 0.0)):.4f} "
                f"ece={float(report.get('calibration_ece', 0.0)):.6f} "
                f"batches={int(report.get('num_eval_batches', 0))} "
                f"games={games_touched} expand_s={expand_wall:.2f} encode_s={encode_wall:.2f}\n"
            )
        return entry
    except Exception as exc:
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(f"eval:update={update:06d} skipped ({type(exc).__name__})\n")
        return None


def _apply_resume_payload(
    *,
    loop: SupervisedLoop,
    dataset: _StreamDataset,
    payload: Any,
    ckpt: Path,
) -> None:
    """Apply a verified payload to live runtime objects (post-verify mutate)."""
    if not isinstance(payload, dict):
        raise ContractError(f"checkpoint payload must be a mapping: {ckpt}")
    for key in ("model_state", "optimizer_state", "training_state", "loss_history"):
        if key not in payload:
            raise ContractError(f"checkpoint payload missing {key!r}: {ckpt}")
    _ = loop.model.load_state_dict(payload["model_state"])
    _ = loop.optimizer.load_state_dict(payload["optimizer_state"])
    scheduler_state = payload.get("scheduler_state")
    if loop.scheduler is not None and scheduler_state:
        try:
            loop.scheduler.load_state_dict(scheduler_state)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ContractError(f"scheduler state incompatible: {exc}") from exc
    raw_training = payload["training_state"]
    if not isinstance(raw_training, dict):
        raise ContractError(f"checkpoint training_state malformed: {ckpt}")
    _restored = TrainingState.from_dict(raw_training)
    if str(_restored.precision) != str(loop.config.precision):
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
    """Read and verify the checkpoint sidecar + payload bytes (no mutation)."""
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
    if sidecar.get("payload_sha256") != "sha256:" + hashlib.sha256(blob).hexdigest():
        raise ContractError(f"checkpoint payload hash mismatch: {resume.checkpoint}")
    # Seek-state presence: sidecars carry ``dataset_buffer`` (whole-game tail
    # index + counters + row hash). Absent → pre-simplification sidecar →
    # hard failure at resume (no full-replay drain remains).
    has_fast = isinstance(sidecar.get("dataset_buffer"), dict)
    return _ResumeEnvelope(
        sidecar=sidecar,
        blob=blob,
        start_epoch=epoch_raw,
        drain_microbatches=drain_raw,
        has_fast_path=has_fast,
    )


def run_stream_training(config: RunConfig, resume: ResumePlan | None = None) -> dict[str, Any]:
    """Train a supervised run from the stream and return the run summary.

    ``resume`` is the read-only plan from :func:`resolve_resume_plan`
    (``None`` for a fresh start). Raises :class:`ContractError` fail-closed
    on wall overlap, empty train splits, config drift, or any checkpoint
    identity mismatch — always before the mismatched state is applied.
    Selection is never called: best-checkpoint promotion stays a manual,
    post-hoc gate.
    """
    if config.run.kind != "supervised":
        raise ContractError(f"stream training supports kind 'supervised', got {config.run.kind!r}")
    if config.data.world_size != _SINGLETON_WORLD:
        raise ContractError(
            f"stream training is single-process v1 (world_size 1), got {config.data.world_size}"
        )
    run_digest = run_config_digest(config)
    if resume is not None:
        # Verify-before-mutate: every identity gate BEFORE layout/stream/loop.
        if resume.run_digest != run_digest:
            raise ContractError(
                f"resume run_digest {resume.run_digest} != run spec {run_digest} (config drift)"
            )
        if resume.cursor.seed != config.seeds.data_seed:
            raise ContractError(
                f"resume cursor seed={resume.cursor.seed} != "
                f"config seeds.data_seed={config.seeds.data_seed}"
            )
        if resume.worker_plan.num_workers != config.data.num_workers:
            raise ContractError("resume worker_plan.num_workers != config data.num_workers")
        if resume.worker_plan.world_size != config.data.world_size:
            raise ContractError("resume worker_plan.world_size != config data.world_size")
        if not resume.checkpoint.is_file():
            raise ContractError(f"resume checkpoint missing: {resume.checkpoint}")

    run_dir = create_run_layout(config)
    manifest = build_manifest(config.data.root)
    if len(manifest) == 0:
        raise ContractError(f"stream manifest empty under {config.data.root}")
    stream_digest = manifest_digest(manifest)
    pinned = config.data.dataset_manifest_hash
    if pinned is not None and pinned != stream_digest:
        raise ContractError(
            f"stream manifest {stream_digest} != config dataset_manifest_hash pin {pinned}"
        )
    ratios = _split_ratios(config)

    # Pre-train scan: wall-disjointness + eval ledger + quarantine counts,
    # all before any runtime object exists (content-hash cache; miss → scan).
    scan = _scan_corpus_cached(
        manifest, config=config, ratios=ratios, stream_digest=stream_digest, run_dir=run_dir
    )
    if scan.train_games == 0:
        raise ContractError(f"train split empty: no games in {config.data.train_split!r}")

    start_update = resume.global_update if resume is not None else 0
    if start_update > config.loop.max_updates:
        raise ContractError(
            f"resume update {start_update} beyond loop.max_updates {config.loop.max_updates}"
        )
    remaining = config.loop.max_updates - start_update
    microbatch = config.loop.microbatch_size

    # Deterministic roots: counter-based seeds only, never wall-clock.
    _ = torch.manual_seed(config.seeds.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seeds.train_seed)
    with contextlib.suppress(Exception):
        import numpy as np

        np.random.seed(config.seeds.train_seed % (2**32))

    envelope: _ResumeEnvelope | None = None
    if resume is not None:
        envelope = _load_resume_envelope(
            resume=resume, run_digest=run_digest, microbatch=microbatch
        )

    need_privileged = _needs_privileged_labels(config)
    # Single seek path (single-pass, epoch pinned 0): fresh runs start at the
    # origin; resumes seek to the recorded frontier with verbatim shuffle
    # buffer + RNG + dedup-prefix restore. No prefix replay, no epoch wrap.
    seek_start: DataStreamCursor | None = None
    seek_entries: list[dict[str, Any]] | None = None
    seek_rng: dict[str, Any] | None = None
    seek_prefix: list[str] | None = None
    if envelope is not None and resume is not None:
        if not envelope.has_fast_path:
            raise ContractError(
                f"checkpoint lacks seek state (dataset_buffer): {resume.checkpoint} "
                "(pre-simplification sidecars cannot resume; re-run from a fresh id)"
            )
        if envelope.start_epoch != 0:
            raise ContractError(
                f"checkpoint stream epoch {envelope.start_epoch} != 0: {resume.checkpoint} "
                "(single-pass pins epoch 0)"
            )
        if resume.cursor.epoch != 0:
            raise ContractError(
                f"resume cursor epoch {resume.cursor.epoch} != 0: {resume.checkpoint}"
            )
        seek_start = DataStreamCursor(
            file_index=resume.cursor.file_index,
            byte_offset=resume.cursor.byte_offset,
            games_seen=resume.cursor.games_seen,
            seed=resume.cursor.seed,
            epoch=0,
            shuffle_pos=int(envelope.sidecar.get("stream_shuffle_pos", 0)),
        )
        shuffle_raw = envelope.sidecar.get("shuffle")
        if not isinstance(shuffle_raw, dict):
            raise ContractError(f"checkpoint shuffle missing: {resume.checkpoint}")
        prefix_rec = shuffle_raw.get("prefix_hashes")
        if not isinstance(prefix_rec, dict):
            raise ContractError(f"checkpoint shuffle prefix_hashes missing: {resume.checkpoint}")
        seek_prefix = _load_prefix_hashes(
            resume.checkpoint.parent, prefix_rec, ckpt=resume.checkpoint
        )
        if config.data.shuffle_buffer_size > 0:
            entries = shuffle_raw.get("buffer_entries", [])
            rng_state = shuffle_raw.get("buffer_rng_state", {})
            if not isinstance(entries, list) or not isinstance(rng_state, dict):
                raise ContractError(f"checkpoint shuffle restore malformed: {resume.checkpoint}")
            # Keys must mirror entries (tamper → raise, never silent refill).
            keys = shuffle_raw.get("buffer_keys", [])
            if [str(e.get("key")) for e in entries if isinstance(e, dict)] != list(keys):
                raise ContractError(
                    f"checkpoint shuffle keys/entries mismatch: {resume.checkpoint}"
                )
            seek_entries = entries
            seek_rng = rng_state

    def _train_stream_factory() -> GameStream:
        common: dict[str, Any] = {
            "seed": config.seeds.data_seed,
            "ratios": ratios,
            "epoch": 0,
            "split": config.data.train_split,
            "shuffle_buffer": config.data.shuffle_buffer_size,
        }
        if config.data.num_workers > 0:
            return PrefetchGameStream(
                manifest,
                **common,
                max_workers=config.data.num_workers,
                start=seek_start,
                shuffle_restore_entries=seek_entries,  # type: ignore[arg-type]
                shuffle_restore_rng=seek_rng,  # type: ignore[arg-type]
                shuffle_restore_prefix_hashes=seek_prefix,  # type: ignore[arg-type]
            )
        return GameStream(
            manifest,
            **common,
            start=seek_start,
            shuffle_restore_entries=seek_entries,  # type: ignore[arg-type]
            shuffle_restore_rng=seek_rng,  # type: ignore[arg-type]
            shuffle_restore_prefix_hashes=seek_prefix,  # type: ignore[arg-type]
        )

    dataset = _StreamDataset(
        stream_factory=_train_stream_factory,
        num_actions=config.model.action_count,
        feature_dim=_FEATURE_FOLD_DIM,
        seed=config.seeds.data_seed,
        drop_last=config.data.drop_last,
        need_privileged=need_privileged,
        replay_backend=config.data.replay_backend,
        # Parallel game expansion rides the existing prefetch/parallel path:
        # num_workers > 0 already selects PrefetchGameStream above; the same
        # knob sizes the bounded (16-max) expansion pool, 0 stays serial.
        expand_workers=config.data.num_workers,
    )
    payload: Any = None
    if resume is not None and envelope is not None:
        try:
            payload = torch.load(io.BytesIO(envelope.blob), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ContractError(f"checkpoint payload unloadable: {resume.checkpoint}") from exc
        _verify_rng_anchors(payload.get("rng_state"), resume.rng_state, ckpt=resume.checkpoint)
        assert seek_start is not None
        # Verbatim tail rebuild (O(buffer) re-expansions), never the epoch.
        _verify_fast_snapshots(sidecar=envelope.sidecar, payload=payload, ckpt=resume.checkpoint)
        parsed_snapshot = _parse_dataset_buffer_sidecar(
            envelope.sidecar.get("dataset_buffer"), ckpt=resume.checkpoint
        )
        if int(parsed_snapshot["epoch"]) != 0:
            raise ContractError(
                f"checkpoint dataset epoch {parsed_snapshot['epoch']} != 0: {resume.checkpoint}"
            )
        if int(parsed_snapshot["microbatches_in_epoch"]) != envelope.drain_microbatches:
            raise ContractError(f"checkpoint microbatch count mismatch: {resume.checkpoint}")
        shuffle_raw = envelope.sidecar.get("shuffle")
        assert isinstance(shuffle_raw, dict)
        if int(shuffle_raw.get("epoch_seed", -1)) != config.seeds.data_seed + dataset.epoch:
            raise ContractError(f"checkpoint shuffle epoch_seed mismatch: {resume.checkpoint}")
        if int(shuffle_raw.get("buffer_size", -1)) != config.data.shuffle_buffer_size:
            raise ContractError(f"checkpoint shuffle buffer_size mismatch: {resume.checkpoint}")
        dataset.restore_buffer(parsed_snapshot)
        # Seek the live stream to the recorded frontier (no prefix drain).
        dataset._stream = dataset._factory()  # type: ignore[attr-defined]
        dataset._iter = iter(dataset._stream)  # type: ignore[attr-defined]
        drained = dataset.stream_cursor()
        if drained != seek_start:
            raise ContractError(
                f"resume seek landed on {drained!r} but checkpoint records "
                f"{seek_start!r}: {resume.checkpoint}"
            )

    model = _build_model(config)
    optimizer = _build_optimizer(config, model)
    scheduler = _build_scheduler(config, optimizer)

    from hydra2.runtime.plain import PlainPytorchAdapter
    from hydra2.runtime.protocol import RuntimeSpec, build_runtime, require_device_available

    require_device_available(config.runtime.device)
    if config.runtime.adapter_id == "plain_pytorch":
        adapter: Any = PlainPytorchAdapter()
    elif config.runtime.adapter_id == "fabric_2.6.5":
        from hydra2.runtime.fabric import FabricRuntimeAdapter

        adapter = FabricRuntimeAdapter()
    else:
        raise ContractError(f"unknown runtime adapter_id {config.runtime.adapter_id!r}")
    spec = RuntimeSpec(
        adapter_id=config.runtime.adapter_id,  # type: ignore[arg-type]
        device=config.runtime.device,
        precision=config.loop.precision,  # type: ignore[arg-type]
        compile_mode=config.runtime.compile_mode,  # type: ignore[arg-type]
    )
    handle = build_runtime(adapter=adapter, model=model, optimizer=optimizer, spec=spec)

    loop_config = TrainingLoopConfig(
        w_policy=config.weights.w_policy,
        w_placement=config.weights.w_placement,
        w_value=config.weights.w_value,
        w_event=dict(config.weights.w_event) if config.weights.w_event is not None else None,
        w_belief=dict(config.weights.w_belief) if config.weights.w_belief is not None else None,
        microbatch_size=microbatch,
        accumulation_steps=config.loop.accumulation_steps,
        gradient_clip_norm=config.loop.gradient_clip_norm,
        max_updates=config.loop.max_updates,
        checkpoint_frequency_updates=config.loop.checkpoint_frequency_updates,
        seed=config.seeds.train_seed,
        precision=config.loop.precision,  # type: ignore[arg-type]
    )
    from hydra2.tracking.clearml_mirror import make_mirror

    loop_manifest_hashes = _manifest_hashes_for_loop(config, model=handle.model, manifest=manifest)

    mirror = make_mirror(
        enabled=config.mirror.enabled,
        project=config.mirror.project,
        task_name=config.mirror.task_name,
        offline_dir=config.mirror.offline_dir,
        manifest_hashes=loop_manifest_hashes,
        loop_config={
            "microbatch_size": microbatch,
            "accumulation_steps": config.loop.accumulation_steps,
            "max_updates": config.loop.max_updates,
            "checkpoint_frequency_updates": config.loop.checkpoint_frequency_updates,
            "seed": config.seeds.train_seed,
            "w_policy": config.weights.w_policy,
            "w_placement": config.weights.w_placement,
            "w_value": config.weights.w_value,
            "precision": config.loop.precision,
        },
    )
    loop = SupervisedLoop(
        model=cast("Any", handle.model),
        optimizer=cast("Any", handle.optimizer),
        scheduler=scheduler,
        dataset=dataset,
        config=loop_config,
        checkpoint_dir=run_dir / "checkpoints",
        manifest_hashes=loop_manifest_hashes,
        handle=handle,
        device=cast("Any", handle.device),
        privileged_source=dataset.privileged,
        evaluation_wall_ids=set(scan.val_walls),
        mirror=mirror,
        runtime_spec=spec,
    )
    if resume is not None:
        # Payload identity fully verified above; now mutate live objects.
        _apply_resume_payload(loop=loop, dataset=dataset, payload=payload, ckpt=resume.checkpoint)

    ckpt_every = config.loop.checkpoint_frequency_updates
    eval_every = config.eval.frequency_updates
    done = 0
    evals: list[dict[str, Any]] = []
    while done < remaining:
        step = min(ckpt_every, remaining - done)
        loop.train(max_updates=step)
        done += step
        _write_streaming_checkpoint(
            run_dir=run_dir,
            update=loop.state.global_update,
            run_digest=run_digest,
            stream_digest=stream_digest,
            config=config,
            loop=loop,
            dataset=dataset,
            batch_size=microbatch,
        )
        _append_new_history(run_dir, loop.loss_history)
        _prune_checkpoints(run_dir, keep=config.loop.keep_last_checkpoints)
        update = loop.state.global_update
        if eval_every > 0 and update % eval_every == 0:
            report = _run_holdout_eval(
                manifest=manifest,
                ratios=ratios,
                config=config,
                loop=loop,
                run_dir=run_dir,
                update=update,
            )
            if report is not None:
                evals.append(report)
    if remaining == 0:
        _append_new_history(run_dir, loop.loss_history)
    # Training pulled its last row: release expansion workers (no-op serial).
    dataset.close()
    checkpoints = _ckpt_names(run_dir)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_digest": run_digest,
        "stream_manifest_hash": stream_digest,
        "start_update": start_update,
        "end_update": loop.state.global_update,
        "updates_run": loop.state.global_update - start_update,
        "train_games": scan.train_games,
        "val_games": scan.val_games,
        "train_sim_games": scan.train_sim_games,
        "val_sim_games": scan.val_sim_games,
        "quarantined": scan.quarantined,
        "duplicates": scan.duplicates,
        "replayed": dataset.replayed,
        "sim_replayed": dataset.sim_replayed,
        "expand_quarantined": dataset.expand_quarantined,
        "expand_quarantined_by_reason": dict(dataset.expand_quarantine_reasons),
        "privileged_labels": "joined" if need_privileged else "skipped-no-auxiliary-weights",
        "checkpoints": checkpoints,
        "evals": evals,
        "loss_history": [dict(entry) for entry in loop.loss_history],
        "metrics_path": str(run_dir / "logs" / "metrics.jsonl"),
    }
    return summary
