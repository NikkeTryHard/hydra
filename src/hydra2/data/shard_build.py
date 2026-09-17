"""Offline tensor shards + mmap-ready manifest (Phase 4 builder).

Pipeline: parallel expand (bounded spawn pool, same bound as
``training.stream_train``: clamp, never unbounded) -> firewall /
checks -> encode once -> per-plane contiguous files.

Layout: one UNCOMPRESSED Arrow IPC Tensor file per (plane, chunk). The hot
path stays uncompressed on purpose: C++ IPC reads off mmap/BufferReader are
zero-copy, while compressed buffers force a transform + copy on every read
(per internal mmap/zstd read-path analysis). Cold archives may stay
Parquet-ZSTD; that is not this module. Chunks default to 8192 rows (~74MB
@9KB/row, inside the 64-256MB / ~7-28k-row LitData guidance); small builds
collapse to a single chunk.

Varlen history is stored fixed ``[N, 256]`` (the frozen bucket cap) plus an
explicit ``history_len`` plane: ``ListArray`` cannot ``to_numpy``-view, so
ragged storage is quarantined at build time, never silently (decided, no
revisit). Boolean planes are stored as ``uint8`` (Arrow booleans are
bit-packed and cannot view); the manifest records the logical ``dtype`` plus
the ``storage_dtype`` so the reader views ``uint8`` and reinterprets without
a copy.

Manifest contract (owned here; the reader consumes read-only)::

    {
      "version": 1,
      "planes": [{"name","dtype","storage_dtype","shape","chunk","file","sha256"}],
      "rows": int, "chunks": int, "chunk_rows": int,
      "schema_digest": str, "dataset_hash": str, "split": str,
      "order": {"kind": "canonical"|"perm", "seed": int},
      "attestation": {"attestation_id": str, "kind": str},
      "decision_ids_sha256": str, "observation_hashes_sha256": str,
      "build": {"games","games_replayed","games_sim_replayed",
                "games_quarantined","quarantine_reasons","rows","chunks",
                "chunk_rows","elapsed_s","games_per_s","rows_per_s",
                "mib_per_s","uncompressed_bytes"},
    }

``dataset_hash`` is ``sha256:`` over the newline-joined SORTED decision ids
(order-independent identity); ``order`` records how rows were sequenced.
Resume cursors (``{epoch, sample_in_epoch}``, MosaicML deterministic
resumption) live with the reader; the builder only guarantees the row order
the manifest attests.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context as _mp_get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import torch

from hydra2.contracts.common import ContractError
from hydra2.data.attestation import (
    SYNTHETIC_ATTESTATION,
    Attestation,
    require_attestation,
)

if TYPE_CHECKING:
    from hydra2.data.decode import GameRecord
    from hydra2.models.encoder import ActorTensorBatch
from hydra2.data.parquet import FORBIDDEN_IN_ACTOR
from hydra2.data.replay_expand import expand_game
from hydra2.models.encoder import input_schema_hash
from hydra2.models.schema import BASELINE_ACTION_COUNT

__all__ = [
    "DECISION_IDS_FILENAME",
    "FIXED_HISTORY_T",
    "HISTORY_LEN_PLANE",
    "OBSERVATION_HASHES_FILENAME",
    "build_shards",
    "validate_manifest",
]

#: Fixed history width: the frozen bucket cap. Encoded batches arrive with a
#: bucketed ``T`` (32/64/128/256); the builder pads every history plane up to
#: this width so every shard row is fixed-shape.
FIXED_HISTORY_T = 256

#: Extra plane carrying per-row true history lengths (mask sums at build).
HISTORY_LEN_PLANE = "history_len"

#: Row-identity sidecars (string columns cannot ride the zero-copy tensor
#: chain, so they travel as JSON with their sha256 pinned in the manifest).
DECISION_IDS_FILENAME = "decision_ids.json"
OBSERVATION_HASHES_FILENAME = "observation_hashes.json"

#: Max parallel game-expansion workers (bounded at 16 workers: larger
#: requests clamp here, never spawn unbounded processes).
_SHARD_MAX_WORKERS = 16

#: Order kinds accepted by :func:`build_shards`.
_ORDER_KINDS = ("canonical", "perm")


def _pool_worker_init() -> None:
    """Clamp per-worker thread pools (spawn-pool initializer).

    A fresh spawn worker inherits a full-size torch/OpenMP thread pool, so
    ``N`` workers x ``C`` threads oversubscribe ``C`` cores. One thread per
    spawn worker keeps parallelism at exactly ``max_workers`` processes.
    Runs in workers only.
    """
    # Env clamp is the effect; prior values discarded.
    _ = os.environ.setdefault("OMP_NUM_THREADS", "1")
    _ = os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _ = os.environ.setdefault("MKL_NUM_THREADS", "1")
    with contextlib.suppress(Exception):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)


def _expand_shard_game(
    payload: tuple[GameRecord, str],
) -> tuple[list[dict[str, Any]] | None, str | None, bool]:
    """Pure per-game expansion for the process pool (top-level: picklable).

    ``payload`` is ``(game, split)``. Wall-bound games expand through
    :func:`expand_game`; wall-less games replay through the simulator's own
    wall-less replay. Returns ``(row_dicts, None, sim_path)`` on success or
    ``(None, reason, False)`` when the game quarantines (:class:`ContractError`
    captured in-worker so the parent can merge strictly in pull order with
    plain ``executor.map`` — no per-sequence future plumbing needed because a
    quarantined game is data, never an exception). Anything that is not a
    :class:`ContractError` propagates fail-closed, as in serial.
    """
    from hydra2.contracts.common import ContractError as _ContractError

    game, split = payload
    try:
        if game.wall_tiles is not None:
            actor_rows = expand_game(game, split=split)
            sim_path = False
        else:
            from hydra2.engines.riichienv._lr_end import replay_game as _replay_sim_game

            actor_rows = _replay_sim_game(game, split=split)
            sim_path = True
    except _ContractError as exc:
        return (None, _quarantine_class(exc), False)
    row_dicts = [
        {
            "decision_id": row.decision_id,
            "chosen_action_id": row.chosen_action_id,
            "actor_observation": dict(row.actor_observation),
        }
        for row in actor_rows
    ]
    return (row_dicts, None, sim_path)


def _quarantine_class(exc: ContractError) -> str:
    """Normalize a quarantine reason to a stable, game-identity-free class."""
    text = str(exc)
    text = re.sub(r"game '[^']*'", "game '<id>'", text)
    text = re.sub(r'game "[^"]*"', "game '<id>'", text)
    text = re.sub(r"game-[0-9a-f]{4,}", "game-<id>", text)
    text = re.sub(r'"([A-Za-z0-9_\-]+)"', r"'\1'", text)
    text = re.sub(r"kyoku \d+", "kyoku <n>", text)
    text = re.sub(r"seat \d+", "seat <n>", text)
    return text[:100]


def _firewall_check_row(row: dict[str, Any]) -> None:
    """Fail closed on privileged leakage or a non-(5,) dora binding.

    Mirrors the actor-shard firewall in ``data.parquet`` at row granularity:
    privileged keys (top level or one level nested) and padded ``(4,)`` dora
    shapes never reach the encoder. Raises :class:`ContractError`; the whole
    build fails (leakage is a code bug, not a game bug — never a quarantine).
    """
    did = str(row.get("decision_id", ""))
    obs = row.get("actor_observation")
    if not isinstance(obs, dict):
        raise ContractError(f"actor_observation not a mapping for {did!r}")
    obs_map: dict[str, object] = obs
    for key in obs_map:
        if key in FORBIDDEN_IN_ACTOR:
            raise ContractError(f"privileged field leakage into actor observation: {key!r}")
        nested: object = obs_map[key]
        if isinstance(nested, dict):
            nested_map: dict[str, object] = nested
            for sub in nested_map:
                if sub in FORBIDDEN_IN_ACTOR:
                    raise ContractError(f"privileged nested field leakage: {key}.{sub}")
    dora = obs.get("dora_indicators")
    if not isinstance(dora, list) or len(dora) != 5:
        raise ContractError(
            f"dora_indicators must bind (5,) for {did!r}, got {type(dora).__name__}"
        )


def _require_workers(expand_workers: int) -> int:
    """Validate the worker knob (non-negative int) and clamp to the bound."""
    if (
        isinstance(expand_workers, bool)
        or not isinstance(expand_workers, int)
        or expand_workers < 0
    ):
        raise ContractError(f"expand_workers invalid: {expand_workers!r} (non-negative int)")
    return min(expand_workers, _SHARD_MAX_WORKERS)


def _expand_all(
    games: list[GameRecord], *, split: str, expand_workers: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand every game (serial reference or ordered spawn pool).

    Returns ``(rows, stats)`` with rows in pull order and stats carrying
    ``games_replayed`` / ``games_sim_replayed`` / ``games_quarantined`` /
    ``quarantine_reasons``. Quarantined games count under normalized reason
    classes; anything else fails closed.
    """
    rows: list[dict[str, Any]] = []
    replayed = 0
    sim_replayed = 0
    quarantined = 0
    reasons: dict[str, int] = {}
    if expand_workers == 0:
        for game in games:
            row_dicts, reason, sim_path = _expand_shard_game((game, split))
            if reason is not None or row_dicts is None:
                quarantined += 1
                reasons[str(reason)] = reasons.get(str(reason), 0) + 1
                continue
            if sim_path:
                sim_replayed += 1
            else:
                replayed += 1
            rows.extend(row_dicts)
    else:
        ctx = _mp_get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=expand_workers,
            mp_context=ctx,
            initializer=_pool_worker_init,
        ) as pool:
            payloads = [(game, split) for game in games]
            for row_dicts, reason, sim_path in pool.map(_expand_shard_game, payloads):
                if reason is not None or row_dicts is None:
                    quarantined += 1
                    reasons[str(reason)] = reasons.get(str(reason), 0) + 1
                    continue
                if sim_path:
                    sim_replayed += 1
                else:
                    replayed += 1
                rows.extend(row_dicts)
    stats = {
        "games_replayed": replayed,
        "games_sim_replayed": sim_replayed,
        "games_quarantined": quarantined,
        "quarantine_reasons": reasons,
    }
    return (rows, stats)


def _plane_name(entry: dict[str, object]) -> str:
    """Manifest plane name as stored (opaque str)."""
    name: object = entry["name"]
    return str(name)


def _plane_chunk(entry: dict[str, object]) -> int:
    """Manifest chunk index (validated int upstream; no coercion)."""
    raw: object = entry["chunk"]
    if not isinstance(raw, int):
        raise ContractError(f"manifest plane chunk invalid: {raw!r}")
    return raw


def _plane_trailing(entry: dict[str, object]) -> tuple[int, ...]:
    """Manifest plane trailing shape (row dim excluded)."""
    raw: object = entry["shape"]
    if not isinstance(raw, list):
        raise ContractError(f"manifest plane shape invalid: {raw!r}")
    dims: list[int] = []
    for value in raw:
        if not isinstance(value, int):
            raise ContractError(f"manifest plane shape invalid: {raw!r}")
        dims.append(value)
    return tuple(dims[1:])


def _decision_id_key(row: dict[str, object]) -> str:
    """Canonical order key: opaque decision id as stored."""
    return str(row["decision_id"])


def _order_rows(rows: list[dict[str, Any]], *, order: str, seed: int) -> list[dict[str, Any]]:
    """Sequence rows: canonical (sorted decision_id) or seeded perm."""
    if order == "canonical":
        return sorted(rows, key=_decision_id_key)
    perm = list(range(len(rows)))
    random.Random(seed).shuffle(perm)
    return [rows[i] for i in perm]


def _pad_history_batch(
    batch: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Pad encoded history planes to ``FIXED_HISTORY_T``; return planes.

    Returns ``(planes, decision_ids, observation_hashes)`` where ``planes``
    maps plane name -> CPU contiguous tensor (bool stays torch.bool here;
    uint8 storage conversion happens at write time). Adds the
    ``history_len`` plane (true lengths = pre-pad mask sums). Raises
    :class:`ContractError` when the bucketed width already exceeds the cap
    (rows are never truncated).
    """
    from hydra2.contracts.common import ContractError as _ContractError

    actor_batch: ActorTensorBatch = batch["actor_batch"]
    features: dict[str, torch.Tensor] = dict(actor_batch.features)
    kind = features["history_event_kind"]
    mask = features["history_mask"]
    width = kind.shape[1]
    if width > FIXED_HISTORY_T:
        raise _ContractError(
            f"encoded history width {width} exceeds fixed cap {FIXED_HISTORY_T}; "
            "rows are never truncated"
        )
    history_len = mask.sum(dim=1).to(torch.int32).contiguous()
    if width < FIXED_HISTORY_T:
        pad = FIXED_HISTORY_T - width
        kind = torch.cat(
            [kind, torch.zeros((kind.shape[0], pad), dtype=torch.int64)], dim=1
        ).contiguous()
        mask = torch.cat(
            [mask, torch.zeros((mask.shape[0], pad), dtype=torch.bool)], dim=1
        ).contiguous()
        features["history_event_kind"] = kind
        features["history_mask"] = mask
    planes: dict[str, torch.Tensor] = {
        name: tensor.contiguous() for name, tensor in features.items()
    }
    planes["chosen_action_id"] = torch.as_tensor(
        batch["chosen_action_id"], dtype=torch.int64
    ).contiguous()
    planes[HISTORY_LEN_PLANE] = history_len
    decision_ids = [str(d) for d in batch.get("decision_ids", [])]
    observation_hashes = [str(h) for h in actor_batch.observation_hashes]
    return (planes, decision_ids, observation_hashes)


def _write_plane_tensor(path: Path, tensor: torch.Tensor) -> tuple[str, str, str, int]:
    """Write one plane chunk as an uncompressed Arrow IPC Tensor file.

    Returns ``(logical_dtype, storage_dtype, sha256_hex, byte_size)``. Bool
    planes travel as ``uint8`` (Arrow booleans are bit-packed and cannot
    ``to_numpy``-view); every other dtype travels natively. No compression,
    ever, on this path.
    """
    np_arr: np.ndarray = np.ascontiguousarray(tensor.detach().to("cpu").numpy())
    logical_dtype = str(np_arr.dtype)
    if np_arr.dtype == np.bool_:
        np_arr = np.ascontiguousarray(np_arr.astype(np.uint8))
    storage_dtype = str(np_arr.dtype)
    with pa.OSFile(str(path), "wb") as sink:
        pa_ipc.write_tensor(pa.Tensor.from_numpy(np_arr), sink)
    raw = path.read_bytes()
    return (logical_dtype, storage_dtype, hashlib.sha256(raw).hexdigest(), len(raw))


def build_shards(
    games: list[GameRecord] | tuple[GameRecord, ...],
    *,
    out_dir: Path | str,
    split: str = "train",
    expand_workers: int = 0,
    chunk_rows: int = 8192,
    order: str = "canonical",
    seed: int = 0,
    attestation: Attestation | None = None,
    num_actions: int = BASELINE_ACTION_COUNT,
    allow_narrow: bool = False,
) -> dict[str, Any]:
    """Expand, check, encode once, and write per-plane shard files.

    Args:
        games: input games (read-only corpus records; never mutated).
        out_dir: shard directory (created; ``manifest.json`` + plane files +
            row-identity sidecars land here).
        split: split games expand under (recorded in the manifest).
        expand_workers: ``0`` is the serial reference; ``>0`` expands in a
            bounded spawn pool with strictly ordered merge (identical rows,
            counters, and quarantine classes either way).
        chunk_rows: rows per chunk file (positive; one chunk when the build
            is smaller).
        order: ``"canonical"`` (sorted decision_id) or ``"perm"`` (seeded
            permutation; ``seed`` recorded).
        seed: shuffle seed for ``order="perm"`` (also recorded for canonical).
        attestation: D-017 binding (defaults to synthetic; fail closed when
            absent/unusable via :func:`require_attestation`).
        num_actions: action vocab width (full baseline by default; the stored
            ``legal_mask`` plane is sliced exactly like the live path; any
            non-baseline width is test-only and requires ``allow_narrow``).
            The width is recorded as ``action_width`` in the manifest.

    Returns the manifest dict (also written to ``manifest.json``).
    """
    from hydra2.training.dataset_encode import encode_observation_rows

    if not isinstance(games, (list, tuple)) or len(games) == 0:
        detail = str(len(games)) if isinstance(games, (list, tuple)) else type(games).__name__
        raise ContractError(f"build_shards requires at least one game, got {detail}")
    if order not in _ORDER_KINDS:
        raise ContractError(f"order must be one of {list(_ORDER_KINDS)}, got {order!r}")
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ContractError(f"chunk_rows must be a positive int, got {chunk_rows!r}")
    workers = _require_workers(expand_workers)
    att = require_attestation(attestation if attestation is not None else SYNTHETIC_ATTESTATION)

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    game_list = list(games)
    total_games = len(game_list)
    start = time.perf_counter()
    rows, stats = _expand_all(game_list, split=split, expand_workers=workers)
    if len(rows) == 0:
        raise ContractError(f"build_shards quarantined all {total_games} games (no rows to encode)")
    for row in rows:
        _firewall_check_row(row)
    ordered = _order_rows(rows, order=order, seed=seed)
    batch = encode_observation_rows(ordered, num_actions=num_actions, allow_narrow=allow_narrow)
    batch = dict(batch)
    batch["decision_ids"] = [str(row["decision_id"]) for row in ordered]
    planes, decision_ids, observation_hashes = _pad_history_batch(batch)
    total_rows = len(ordered)
    if len(decision_ids) != total_rows or len(observation_hashes) != total_rows:
        raise ContractError("row identity sidecars disagree with encoded row count")

    dataset_hash = "sha256:" + hashlib.sha256("\n".join(sorted(decision_ids)).encode()).hexdigest()
    schema_digest = str(input_schema_hash())

    plane_names = sorted(planes.keys())
    full = {name: planes[name] for name in plane_names}
    manifest_planes: list[dict[str, Any]] = []
    uncompressed_bytes = 0
    chunks = (total_rows + chunk_rows - 1) // chunk_rows
    for chunk in range(chunks):
        lo = chunk * chunk_rows
        hi = min(lo + chunk_rows, total_rows)
        for name in plane_names:
            chunk_tensor = full[name][lo:hi].contiguous()
            filename = f"{name}.chunk-{chunk:05d}.arrow.ipc"
            logical_dtype, storage_dtype, sha, size = _write_plane_tensor(
                dest / filename, chunk_tensor
            )
            manifest_planes.append(
                {
                    "name": name,
                    "dtype": logical_dtype,
                    "storage_dtype": storage_dtype,
                    "shape": [hi - lo, *full[name].shape[1:]],
                    "chunk": chunk,
                    "file": filename,
                    "sha256": sha,
                }
            )
            uncompressed_bytes += size

    ids_bytes = json.dumps(decision_ids).encode()
    hashes_bytes = json.dumps(observation_hashes).encode()
    # Sidecar writes are the effect; byte counts discarded.
    _ = (dest / DECISION_IDS_FILENAME).write_bytes(ids_bytes)
    _ = (dest / OBSERVATION_HASHES_FILENAME).write_bytes(hashes_bytes)

    elapsed = time.perf_counter() - start
    mib = uncompressed_bytes / float(2**20)
    manifest: dict[str, Any] = {
        "version": 1,
        "planes": manifest_planes,
        "rows": total_rows,
        "chunks": chunks,
        "chunk_rows": chunk_rows,
        "action_width": num_actions,
        "schema_digest": schema_digest,
        "dataset_hash": dataset_hash,
        "split": split,
        "order": {"kind": order, "seed": seed},
        "attestation": {"attestation_id": att.attestation_id, "kind": att.kind},
        "decision_ids_sha256": hashlib.sha256(ids_bytes).hexdigest(),
        "observation_hashes_sha256": hashlib.sha256(hashes_bytes).hexdigest(),
        "build": {
            "games": total_games,
            "games_replayed": stats["games_replayed"],
            "games_sim_replayed": stats["games_sim_replayed"],
            "games_quarantined": stats["games_quarantined"],
            "quarantine_reasons": stats["quarantine_reasons"],
            "rows": total_rows,
            "chunks": chunks,
            "chunk_rows": chunk_rows,
            "elapsed_s": elapsed,
            "games_per_s": (total_games / elapsed) if elapsed > 0 else 0.0,
            "rows_per_s": (total_rows / elapsed) if elapsed > 0 else 0.0,
            "mib_per_s": (mib / elapsed) if elapsed > 0 else 0.0,
            "uncompressed_bytes": uncompressed_bytes,
        },
    }
    # Manifest write is the effect; byte count discarded.
    _ = (dest / "manifest.json").write_bytes(
        json.dumps(manifest, sort_keys=True, indent=2).encode()
    )
    return manifest


def validate_manifest(manifest: dict[str, Any], *, out_dir: Path | str) -> None:
    """Validate a shard manifest against its files (fail closed).

    Re-checks every plane sha256, row-count agreement, the schema digest
    against the live encoder, the dataset hash against the decision-id
    sidecar, canonical ordering when attested, and the attestation binding.
    Raises :class:`ContractError` on the first mismatch.
    """
    dest = Path(out_dir)
    if not isinstance(manifest, dict):
        raise ContractError("manifest must be a mapping")
    if manifest.get("version") != 1:
        raise ContractError(f"manifest version unsupported: {manifest.get('version')!r}")
    planes = manifest.get("planes")
    if not isinstance(planes, list) or len(planes) == 0:
        raise ContractError("manifest planes must be a non-empty list")
    rows = manifest.get("rows")
    if not isinstance(rows, int) or rows <= 0:
        raise ContractError(f"manifest rows invalid: {rows!r}")
    action_width = manifest.get("action_width")
    if not isinstance(action_width, int) or action_width <= 0:
        raise ContractError(f"manifest action_width invalid: {action_width!r}")

    per_chunk: dict[int, int] = {}
    for entry in planes:
        if not isinstance(entry, dict):
            raise ContractError("manifest plane entry must be a mapping")
        for key in ("name", "dtype", "storage_dtype", "shape", "chunk", "file", "sha256"):
            if key not in entry:
                raise ContractError(f"manifest plane entry missing {key!r}")
        shape: list[int] = entry["shape"]
        if not isinstance(shape, list) or len(shape) == 0 or shape[0] <= 0:
            raise ContractError(f"manifest plane shape invalid: {shape!r}")
        chunk: int = entry["chunk"]
        if not isinstance(chunk, int) or chunk < 0:
            raise ContractError(f"manifest plane chunk invalid: {chunk!r}")
        per_chunk[chunk] = per_chunk.get(chunk, 0) + shape[0]
        file_obj: object = entry["file"]
        plane_path = dest / str(file_obj)
        data = plane_path.read_bytes() if plane_path.is_file() else None
        if data is None:
            raise ContractError(f"manifest plane file missing: {entry['file']!r}")
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ContractError(f"manifest plane sha256 mismatch: {entry['file']!r}")
    plane_names = {_plane_name(e) for e in planes if isinstance(e, dict)}
    if sum(per_chunk.values()) != rows * len(plane_names):
        raise ContractError("manifest plane row totals disagree with rows")
    # Every chunk must carry the same plane set; one plane keeps one trailing
    # shape across chunks (only the row dim varies).
    by_chunk: dict[int, list[dict[str, Any]]] = {}
    for entry in planes:
        if isinstance(entry, dict):
            by_chunk.setdefault(_plane_chunk(entry), []).append(entry)
    reference = sorted(str(e["name"]) for e in by_chunk[min(by_chunk)])
    for chunk, entries in by_chunk.items():
        if sorted(str(e["name"]) for e in entries) != reference:
            raise ContractError(f"manifest chunk {chunk} plane set mismatch")
    trailing: dict[str, set[tuple[int, ...]]] = {}
    for entry in planes:
        if isinstance(entry, dict):
            shapes = trailing.setdefault(_plane_name(entry), set())
            shapes.add(_plane_trailing(entry))
    for name, shapes in trailing.items():
        if len(shapes) != 1:
            raise ContractError(f"manifest plane {name!r} trailing shapes disagree")
    legal_shapes = trailing.get("legal_mask")
    if legal_shapes is not None:
        (legal_tail,) = legal_shapes
        if len(legal_tail) != 1 or legal_tail[0] != action_width:
            raise ContractError(
                f"manifest legal_mask width {legal_tail!r} != action_width {action_width!r}"
            )
    if manifest.get("chunks") != len(by_chunk):
        raise ContractError("manifest chunks disagree with plane files")

    if manifest.get("schema_digest") != str(input_schema_hash()):
        raise ContractError("manifest schema_digest disagrees with the live encoder")

    ids_raw = dest / DECISION_IDS_FILENAME
    hashes_raw = dest / OBSERVATION_HASHES_FILENAME
    if not ids_raw.is_file() or not hashes_raw.is_file():
        raise ContractError("manifest row-identity sidecars missing")
    ids_bytes = ids_raw.read_bytes()
    hashes_bytes = hashes_raw.read_bytes()
    if hashlib.sha256(ids_bytes).hexdigest() != manifest.get("decision_ids_sha256"):
        raise ContractError("decision_ids sidecar sha256 mismatch")
    if hashlib.sha256(hashes_bytes).hexdigest() != manifest.get("observation_hashes_sha256"):
        raise ContractError("observation_hashes sidecar sha256 mismatch")
    decision_ids: list[object] = json.loads(ids_bytes.decode())
    observation_hashes: list[object] = json.loads(hashes_bytes.decode())
    if not isinstance(decision_ids, list) or len(decision_ids) != rows:
        raise ContractError("decision_ids sidecar row count mismatch")
    if not isinstance(observation_hashes, list) or len(observation_hashes) != rows:
        raise ContractError("observation_hashes sidecar row count mismatch")
    recomputed = (
        "sha256:"
        + hashlib.sha256("\n".join(sorted(str(d) for d in decision_ids)).encode()).hexdigest()
    )
    if recomputed != manifest.get("dataset_hash"):
        raise ContractError("manifest dataset_hash mismatch")
    order_obj: object = manifest.get("order")
    if not isinstance(order_obj, dict):
        raise ContractError(f"manifest order invalid: {manifest.get('order')!r}")
    order: dict[str, object] = order_obj
    if order.get("kind") not in _ORDER_KINDS:
        raise ContractError(f"manifest order invalid: {order!r}")
    if order.get("kind") == "canonical" and [str(d) for d in decision_ids] != sorted(
        str(d) for d in decision_ids
    ):
        raise ContractError("manifest attests canonical order but decision_ids are unsorted")
    attestation_obj: object = manifest.get("attestation")
    if not isinstance(attestation_obj, dict):
        raise ContractError(f"manifest attestation invalid: {manifest.get('attestation')!r}")
    attestation: dict[str, object] = attestation_obj
    attestation_id: object = attestation.get("attestation_id")
    attestation_kind: object = attestation.get("kind")
    if bool(attestation_id) is False or attestation_kind not in ("synthetic", "real"):
        raise ContractError(f"manifest attestation invalid: {attestation!r}")
    build = manifest.get("build")
    if not isinstance(build, dict):
        raise ContractError("manifest build report missing")
    for key in ("games", "rows", "rows_per_s", "games_per_s", "mib_per_s", "uncompressed_bytes"):
        if key not in build:
            raise ContractError(f"manifest build report missing {key!r}")
    if build.get("rows") != rows:
        raise ContractError("manifest build report rows disagree")
