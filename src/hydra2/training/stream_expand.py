"""Stream row expansion: games to actor rows plus buffer-window helpers.

Owns the replay-backend dispatch (rust plane walk versus the python oracle
shim), quarantine classification, the bounded spawn-pool expansion worker,
and the game-keyed window hash plus history-bucket grouping keys the
dataset buffer builds on. Row identity (decision_id format) and counter
semantics match on both backends, so privileged joins and sidecar hashes
agree regardless of path.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import time
from typing import TYPE_CHECKING, Any

import torch

from hydra2.contracts.common import ContractError as ContractError
from hydra2.data.replay_expand import expand_game as expand_game
from hydra2.data.replay_expand import expand_privileged_rows as expand_privileged_rows
from hydra2.data.replay_expand import pop_live_observation as pop_live_observation

if TYPE_CHECKING:
    from hydra2.data.decode import GameRecord as GameRecord
    from hydra2.data.parquet import DecisionRow as DecisionRow
    from hydra2.training.run_config import RunConfig as RunConfig

__all__ = [
    "SPLIT_RATIOS",
    "_ACTION_KINDS_BY_ID",
    "_BUFFER_COMPACT_ROWS",
    "_FEATURE_FOLD_DIM",
    "_MODEL_PARAMETERS",
    "_PARALLEL_EXPAND_MAX_WORKERS",
    "_QUARANTINE_REASON_CLASSES",
    "_REPLAY_BACKENDS",
    "_SINGLETON_RANK",
    "_SINGLETON_WORLD",
    "_action_kind_for_id",
    "_expand_game_planes",
    "_expand_game_rows",
    "_expand_streamed_chunk",
    "_history_bucket_of",
    "_needs_privileged_labels",
    "_pool_worker_init",
    "_quarantine_class",
    "_require_replay_backend",
    "_row_to_dict",
    "_sidecar_window_hash",
    "_slim_row_dicts",
    "_split_ratios",
]


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


#: Replay backends. ``"rust"`` is the default (Rust per-game walk over raw
#: framed bytes + tensor-native batch assembly, no per-row Python);
#: ``"python"`` is the oracle shim (byte-identical rows, parity only). No
#: JSON handoff exists: the rust plane path is the sole handoff, and an
#: unknown backend id fails closed in ``_require_replay_backend``.
_REPLAY_BACKENDS: tuple[str, ...] = ("python", "rust")


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
    """Expand one game to actor rows on the ``"python"`` oracle backend.

    Wall-bound games expand through :func:`expand_game`; wall-less games
    replay through ``log_replay``. Returns ``(actor_rows, sim_path)`` with
    ``sim_path`` true for wall-less games, so the
    ``replayed``/``sim_replayed``/``expand_quarantined`` counters and the
    :func:`_quarantine_class` normalization apply verbatim.
    """
    # intentionally discarded: validation only, backend arg already bound
    _ = _require_replay_backend(backend)
    if game.wall_tiles is not None:
        return (expand_game(game, split=split), False)

    from hydra2.engines.riichienv.log_replay import replay_game as _replay_sim_game

    return (_replay_sim_game(game, split=split), True)


def _expand_game_planes(
    game: GameRecord, split: str, raw: bytes = b""
) -> tuple[list[dict[str, Any]], bool]:
    """Expand one game through the Rust per-game walk (``replay_backend="rust"``).

    Returns ``(slim_row_dicts, sim_path)`` with ``sim_path`` true for
    wall-less games (same predicate as :func:`_expand_game_rows`, so
    replayed/sim counters match). Row dicts carry ``decision_id``,
    ``chosen_action_id``, ``action_kind`` plus the shared game planes blob
    (``_planes`` name→bytes, ``_row`` offset, ``_t_len``) for tensor
    assembly in ``next_batch`` — no ``DecisionRow`` objects, no observations.
    ``decision_id`` format (``<game_id>:d<seq:04d>``) matches the python path
    exactly, so privileged joins and sidecar hashes agree. Quarantines raise
    :class:`ContractError` (feed reason string) for the caller to
    quarantine-and-count exactly like oracle rejects.

    ``raw`` is the verbatim framed game bytes (``StreamGame.raw``): the walk
    runs on the original bytes with the wall spliced in Rust, instead of
    re-serializing every parsed event in Python (which costs more than the
    walk itself). Walled games bind ``game.wall_tiles`` (136 ints, fail
    closed otherwise); wall-less games walk the embedded content verbatim.
    Empty ``raw`` (hand-built games in tests) falls back to the
    ``_expand_game_planes`` re-serialization, which parses to identical events.
    """
    from hydra2.training.rust_batch import replay_game_planes_raw

    sim_path = game.wall_tiles is None
    # intentionally discarded: split rides the buffer entry parent-side
    _ = split
    wall: list[int] | None = None
    if game.wall_tiles is not None:
        # Walled regime follows wall content (the framer binds the record
        # wall, mirroring decode_game_object): the record wall overrides
        # whatever the log embedded.
        wall = [int(t) for t in game.wall_tiles]
        if len(wall) != 136:
            raise ContractError(f"wall_tiles must carry 136 tiles, got {len(wall)}")
    if raw:
        text = raw
    elif sim_path:
        text = ("\n".join(json.dumps(event) for event in game.events) + "\n").encode("utf-8")
    else:
        assert wall is not None
        head = dict(game.events[0])
        head["wall"] = wall
        body = "\n".join(json.dumps(e) for e in game.events[1:])
        text = (json.dumps(head) + "\n" + body + "\n").encode("utf-8")
    planes, rows, _t_len, quarantined, reason, event_idx = replay_game_planes_raw(text, 0, wall)
    if quarantined:
        raise ContractError(
            f"rust walk quarantined game {game.game_id!r}: {reason} at event {event_idx}"
        )

    return (_slim_row_dicts(str(game.game_id), planes, int(rows), int(_t_len)), sim_path)


def _slim_row_dicts(
    game_id: str, planes: dict[str, bytes], rows: int, t_len: int
) -> list[dict[str, Any]]:
    """Project staged planes into slim per-row dicts (shared pull paths).

    Parses the committed ``chosen_action_id`` column once and fans out one
    dict per row (``decision_id``/``chosen_action_id``/``action_kind`` plus
    the shared game planes blob for tensor assembly). Raises the identical
    row-drift :class:`ContractError` as the inline path on shape mismatch.
    """
    chosen_ids: list[int] = [
        int(v) for v in torch.frombuffer(planes["chosen_action_id"], dtype=torch.int64).tolist()
    ]
    if len(chosen_ids) != int(rows):
        raise ContractError(
            f"rust walk row drift on game {game_id!r}: {len(chosen_ids)} chosen vs {int(rows)} rows"
        )
    out: list[dict[str, Any]] = []
    for seq, chosen_id in enumerate(chosen_ids):
        out.append(
            {
                "decision_id": f"{game_id}:d{seq:04d}",
                "chosen_action_id": chosen_id,
                "action_kind": _action_kind_for_id(chosen_id),
                "_planes": planes,
                "_row": seq,
                "_t_len": int(t_len),
            }
        )
    return out


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
            *(config.weights.w_event if config.weights.w_event is not None else {}).values(),
            *(config.weights.w_belief if config.weights.w_belief is not None else {}).values(),
        ]
    )


def _split_ratios(config: RunConfig) -> dict[str, float]:
    """Concrete split fractions with the config's split names bound."""
    train_split = config.data.train_split
    val_split = config.data.val_split
    if train_split == val_split:
        raise ContractError(f"data.train_split and data.val_split must differ, got {train_split!r}")
    return {train_split: SPLIT_RATIOS["train"], val_split: SPLIT_RATIOS["validation"]}


_ACTION_KINDS_BY_ID: list[str] | None = None


def _action_kind_for_id(chosen_id: int) -> str:
    """Honest kind for one table index (``"unknown"`` when unmapped).

    The 6792 action table is authoritative (``configs/contracts/
    action_table_v1.json`` index == action id); out-of-range or unloadable
    tables fall back to ``"unknown"`` (never synthesized, never raised) so
    per-type scorecards stay honest until the table binds.
    """
    global _ACTION_KINDS_BY_ID
    cached = _ACTION_KINDS_BY_ID
    if cached is None:
        try:
            import json as _json
            from pathlib import Path as _Path

            _table_path = (
                _Path(__file__).resolve().parents[3]
                / "configs"
                / "contracts"
                / "action_table_v1.json"
            )
            raw = _json.loads(_table_path.read_bytes().decode("utf-8"))
            actions = raw.get("payload", {}).get("actions", [])
            cached = [
                str(a.get("kind", "unknown")) if isinstance(a, dict) else "unknown" for a in actions
            ]
            if len(cached) == 0:
                return "unknown"
            _ACTION_KINDS_BY_ID = cached
        except Exception:
            return "unknown"
    if isinstance(chosen_id, bool) or not isinstance(chosen_id, int):
        return "unknown"
    if 0 <= chosen_id < len(cached):
        kind = cached[chosen_id]
        return kind if isinstance(kind, str) and kind != "" else "unknown"
    return "unknown"


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Project a :class:`DecisionRow` onto the real-encoder input shape.

    Carries the live :class:`ActorObservation` when the expanding call just
    built it (same object serially; pickled copy across the pool boundary),
    so the encoder consumes it directly instead of re-serializing,
    re-parsing, and re-validating the document (~0.3ms/row saved). Misses
    (foreign rows) keep the validated document form.
    :func:`_resolve_live_or_parse` owns the fallback plus the
    observation-hash guard either way, so content is identical in both
    cases — this changes transport, never content.
    """
    chosen = int(row.chosen_action_id)
    decision_id = str(row.decision_id)
    live = pop_live_observation(decision_id)
    return {
        "decision_id": decision_id,
        "chosen_action_id": chosen,
        "actor_observation": live if live is not None else dict(row.actor_observation),
        "observation_hash": str(row.observation_hash),
        "action_kind": _action_kind_for_id(chosen),
    }


#: Max parallel game-expansion workers (bounded pool: larger requests clamp
#: here, never spawn unbounded processes; 16 keeps spawn RSS + file
#: descriptors predictable next to the scan pool).
#:
#: Thread budget is joint: the scan pool's spawn workers (processes, released
#: before training starts), the live ``PrefetchGameStream`` decode threads,
#: AND this expansion pool (processes, alive for the whole run). Scan never
#: overlaps training, but prefetch threads + expansion procs + the main torch
#: threads train concurrently, so keep ``2 * num_workers + main torch threads
#: <= cores``. :func:`_pool_worker_init` clamps each worker to one thread
#: (fixes thread oversubscription, not process oversubscription).
#: Oversubscribed symptoms: fills slower than the serial path (thrash), spawn
#: RSS + fd pressure (every proc re-imports torch), and climbing
#: ``PrefetchGameStream.waits``. When fills stop scaling, lower
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
    # intentionally discarded: env value unneeded after ensure
    _ = os.environ.setdefault("OMP_NUM_THREADS", "1")
    _ = os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _ = os.environ.setdefault("MKL_NUM_THREADS", "1")
    with contextlib.suppress(Exception):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)


def _expand_streamed_chunk(
    payloads: list[tuple[Any, str, str, bool, bytes]],
) -> list[tuple[str, Any, Any, Any]]:
    """Expand a contiguous chunk of games; one bridge call for rust backend.

    ``payloads`` are ``(game, split, replay_backend, need_privileged, raw)``
    in pull order. Returns per game, in order: ``("ok", row_dicts,
    priv_pairs, sim_path)`` or ``("quarantine", message)``. Per-game
    :class:`ContractError` rides as data so the parent quarantines-and-counts
    exactly like the serial pull (reason classes verbatim — the classifier
    is message-based); anything else propagates fail-closed. No shared
    mutable state (spawn-pool safe).
    """
    _t_task = time.perf_counter()
    if len(payloads) == 0:
        return []
    backend = payloads[0][2]
    if backend == "rust":
        from hydra2.training.rust_batch import expand_game_batch

        # Wall pre-check mirrors _pull_game_batch (fail-closed 136 tiles).
        walls: list[list[int] | None] = []
        pre_err: dict[int, str] = {}
        for pos, (game, _split, _b, _n, _raw) in enumerate(payloads):
            tiles = game.wall_tiles
            if tiles is None:
                walls.append(None)
            else:
                wall = [int(t) for t in tiles]
                if len(wall) != 136:
                    pre_err[pos] = f"wall_tiles must carry 136 tiles, got {len(wall)}"
                    walls.append(None)
                else:
                    walls.append(wall)
        idxs = [
            pos
            for pos, (_game, _s, _b, _n, raw) in enumerate(payloads)
            if raw and pos not in pre_err
        ]
        results = expand_game_batch([(payloads[i][4], 0, walls[i]) for i in idxs]) if idxs else []
        if len(results) != len(idxs):
            raise ContractError(f"bridge batch drift: {len(results)} results for {len(idxs)} games")
        by_pos = dict(zip(idxs, results, strict=True))
        out: list[tuple[str, Any, Any, Any]] = []
        for pos, (game, split, _b, need_priv, _raw) in enumerate(payloads):
            try:
                if pos in pre_err:
                    raise ContractError(pre_err[pos])
                if pos in by_pos:
                    planes, rows, t_len, quarantined, reason, event_idx = by_pos[pos]
                    if quarantined:
                        raise ContractError(
                            f"rust walk quarantined game {game.game_id!r}: "
                            f"{reason} at event {event_idx}"
                        )
                    row_dicts = _slim_row_dicts(str(game.game_id), planes, rows, t_len)
                    sim_path = game.wall_tiles is None
                else:
                    # Hand-built games (tests) carry no framed bytes:
                    # ``_expand_game_planes`` rebuilt path, same rows as the serial pull.
                    row_dicts, sim_path = _expand_game_planes(game, split, b"")
                priv_rows = expand_privileged_rows(game, split=split) if need_priv else []
                priv_pairs = [(str(p.decision_id), dict(p.privileged_label)) for p in priv_rows]
                out.append(("ok", row_dicts, priv_pairs, sim_path))
            except ContractError as exc:
                out.append(("quarantine", str(exc), None, None))
        # phase:-log contract (observer-only): no tool parses stdout and
        # stdout-redirect silences all of it; the 0.3s/0.15s tripwires below
        # are debug thresholds, not SLOs. Never env-gate these prints.
        if os.environ.get("HYDRA2_FILL_DEBUG") == "1":
            _dt_task = time.perf_counter() - _t_task
            if _dt_task >= 0.3:
                print(f"phase: slow-task games={len(payloads)} t={_dt_task:.2f}s", flush=True)
        return out
    out = []
    for game, split, _b, need_priv, _raw in payloads:
        try:
            actor_rows, sim_path = _expand_game_rows(game, split, "python")
            row_dicts = [_row_to_dict(row) for row in actor_rows]
            priv_rows = expand_privileged_rows(game, split=split) if need_priv else []
            priv_pairs = [(str(p.decision_id), dict(p.privileged_label)) for p in priv_rows]
            out.append(("ok", row_dicts, priv_pairs, sim_path))
        except ContractError as exc:
            out.append(("quarantine", str(exc), None, None))
    return out


def _sidecar_window_hash(entries: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    """Game-keyed integrity hash over a buffered row window (K4 sidecar).

    Binds each whole-game sidecar record (``key`` + ``rows``) plus that
    game's row payload (``chosen_action_id`` + ``action_kind`` per row, in
    window order). Per-row ``decision_id`` strings never enter the digest:
    identity rides the per-game ``key`` (``raw_bytes_sha256``), content
    rides the chosen/kind payload. Snapshot and restore compute the
    identical digest, so any game-identity, count, or content drift fails
    the ``row_hash`` comparison (re-homed, never dropped).
    """
    total = 0
    for entry in entries:
        count_any = entry["rows"]
        if isinstance(count_any, bool) or not isinstance(count_any, int):
            raise ContractError("sidecar window entry rows must be an int")
        total += count_any
    if total != len(rows):
        raise ContractError(f"buffer index drift: {total} indexed rows != {len(rows)} buffered")
    h = hashlib.sha256()
    pos = 0
    for entry in entries:
        key = entry["key"]
        if not isinstance(key, str) or key == "":
            raise ContractError("sidecar window entry key must be a non-empty str")
        count = int(entry["rows"])
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        h.update(count.to_bytes(8, "little"))
        for row in rows[pos : pos + count]:
            chosen_any = row.get("chosen_action_id", 0)
            if isinstance(chosen_any, bool) or not isinstance(chosen_any, int):
                raise ContractError("sidecar window row chosen_action_id must be an int")
            kind_any = row.get("action_kind", "")
            if not isinstance(kind_any, str):
                raise ContractError("sidecar window row action_kind must be a str")
            h.update(chosen_any.to_bytes(8, "little", signed=True))
            h.update(kind_any.encode("utf-8"))
            h.update(b"\x00")
        pos += count
    return "sha256:" + h.hexdigest()


def _history_bucket_of(row: dict[str, Any]) -> int:
    """History-bucket ceil for one buffered row (homogeneous-take grouping key).

    Reads ``len(row["actor_observation"]["visible_history"])`` (live
    :class:`ActorObservation` or its validated-dict form) and returns the
    smallest ``HISTORY_BUCKET_LENGTHS`` entry covering it. Missing,
    malformed, or over-cap lengths map to the max bucket: grouping must
    never mask errors — over-cap rows still fail closed at encode time with
    the proper error. Rows without an actor observation (Rust slim rows
    carry ``_t_len`` planes instead) group at max bucket, so the
    default-Rust feed keeps its legacy order bit-for-bit.
    """
    from hydra2.models.schema import HISTORY_BUCKET_LENGTHS

    buckets = HISTORY_BUCKET_LENGTHS
    try:
        obs = row.get("actor_observation")
        hist = (
            obs.get("visible_history")
            if isinstance(obs, dict)
            else getattr(obs, "visible_history", None)
        )
        length = len(hist)  # type: ignore[arg-type]
    except (AttributeError, TypeError):
        return buckets[-1]
    if isinstance(length, bool) or not isinstance(length, int):
        return buckets[-1]
    for bucket in buckets:
        if length <= bucket:
            return bucket
    return buckets[-1]
