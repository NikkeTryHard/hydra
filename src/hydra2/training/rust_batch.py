"""Tensor-native training batch assembly over Rust plane slots (no Python rows).

Assembles :class:`ActorTensorBatch` + loss targets straight from the device
tensors :meth:`RustPlaneStream.next_into_planes` returns — no
``DecisionRow`` dicts, no ``ActorObservation`` objects, no per-row Python.
Every op below is a whole-batch torch op; the only host sync is the
nonterminal legal-mask validation (the Python encoder path syncs per row
and the model syncs the same predicate before forward, so this is fewer).

Value contract (proven by ``tests/unit/test_rust_batch.py`` against
:func:`encode_observations` on shared rows): every feature tensor is
element-identical to the encoder's output. Rust planes already carry
encoder-mapped ids (seat/round winds pre-mapped 27-30 to 0-3, furiten 0-3,
phase/riichi kind ids, history kind ids); only dtypes need casts and the
packed legal ids need a scatter. History width is the fill ``t_len`` rather
than the encoder's per-batch bucket ceil — values on the real prefix are
identical and both pad with 0/False, which the model masks out exactly
(verified bitwise-equal logits in the parity test).

``observation_hashes`` is empty: row identity never enters training math
(the loop only passes it through; loss reads logits + ``chosen_action_id``
+ ``legal_mask``). Scope is supervised policy training; privileged joins
and telemetry that need decision ids stay on the Python path.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import torch

from hydra2.contracts.common import ContractError
from hydra2.models.encoder import ActorTensorBatch
from hydra2.models.schema import BASELINE_ACTION_COUNT, HISTORY_BUCKET_LENGTHS

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "assemble_slim_batch",
    "assemble_training_batch",
    "expand_game_batch",
    "replay_game_planes",
    "replay_game_planes_raw",
]


_FRESH_CHECKED: set[str] = set()

#: Timestamp granularity slack for the freshness check (coarse filesystems).
_FRESH_EPSILON_S = 1.0


def _source_newest_mtime(root: str) -> float | None:
    """Newest mtime under a source tree (``target/`` excluded), or None."""
    from pathlib import Path

    try:
        return max(
            (
                p.stat().st_mtime
                for p in Path(root).rglob("*")
                if p.is_file() and "target" not in p.parts
            ),
            default=None,
        )
    except OSError:
        return None


def _check_fresh(ext_file: str) -> None:
    """Fail closed when an installed ext predates its Rust sources.

    ``pixi run build-ext`` writes ``hydra2_replay_rs.build.json`` beside the
    installed ``.so`` recording the source tree and its newest mtime.
    Ad-hoc staged copies (tests, harness scratch dirs) carry no sidecar and
    skip the check — they build their extension in the same session. When
    the sidecar exists but the tree is gone (source-less deploy), there is
    nothing to compare against and the import is allowed.
    """
    from pathlib import Path

    if ext_file in _FRESH_CHECKED:
        return
    sidecar = Path(ext_file).with_name("hydra2_replay_rs.build.json")
    try:
        import json

        record = json.loads(sidecar.read_text(encoding="utf-8"))
        root = str(record["source_root"])
        pinned = float(record["source_mtime"])
    except (OSError, ValueError, KeyError, TypeError):
        return
    newest = _source_newest_mtime(root)
    if newest is None:
        return
    if newest > pinned + _FRESH_EPSILON_S:
        raise ContractError(
            "hydra2_replay_rs extension is stale (Rust sources newer than the "
            "installed build); re-run `pixi run build-ext`"
        )
    _FRESH_CHECKED.add(ext_file)


def _load_extension() -> Any:
    """Import the compiled ``hydra2_replay_rs`` module (fail closed)."""
    try:
        ext = importlib.import_module("hydra2_replay_rs")
    except ImportError as exc:
        raise ContractError(
            "hydra2_replay_rs extension not importable; build it with "
            "`cargo build -p hydra2-replay-rs` and put the resulting "
            "`hydra2_replay_rs` shared object on sys.path"
        ) from exc
    _check_fresh(str(getattr(ext, "__file__", "")))
    return ext


def assemble_training_batch(
    planes: Mapping[str, torch.Tensor],
    *,
    action_count: int = BASELINE_ACTION_COUNT,
    buckets: tuple[int, ...] = HISTORY_BUCKET_LENGTHS,
) -> dict[str, Any]:
    """Assemble a loop-ready batch dict from one Rust plane fill (device kept).

    Input is the ``out`` mapping of :meth:`RustPlaneStream.next_into_planes`
    (valid-prefix slices already applied): ``legal_ids`` ``[B,32]`` int32,
    ``legal_len`` ``[B]`` int64, ``history_event_kind`` ``[B,T]`` int64,
    ``history_mask`` ``[B,T]`` bool, ``chosen_action_id`` ``[B]`` int64, plus
    the scalar/count planes. Returns ``{"actor_batch", "chosen_action_id",
    "legal_mask"}`` — the exact keys :func:`compute_supervised_loss` reads.
    """
    try:
        chosen = planes["chosen_action_id"]
        legal_ids = planes["legal_ids"]
        legal_len = planes["legal_len"]
        hist_kind = planes["history_event_kind"]
        hist_mask = planes["history_mask"]
    except KeyError as exc:
        raise ContractError(f"rust batch missing plane {exc}") from exc
    batch_size = int(chosen.shape[0])
    if batch_size == 0:
        raise ContractError("assemble_training_batch requires at least one row")
    t_len = int(hist_kind.shape[1])
    if t_len > buckets[-1]:
        raise ContractError(
            f"rust history width {t_len} exceeds model bucket cap {buckets[-1]}; "
            "rows are never truncated"
        )
    device = chosen.device

    # Legal unpack: scatter valid ids into [B, A+1] (padding ids park in the
    # dropped extra column, so a real pass id 0 never collides with filler).
    ids = legal_ids.to(torch.int64).clamp(0, action_count)
    valid = torch.arange(32, device=device).unsqueeze(0) < legal_len.reshape(batch_size, 1)
    safe = ids.masked_fill(~valid, action_count)
    wide = torch.zeros((batch_size, action_count + 1), dtype=torch.bool, device=device)
    wide.scatter_(1, safe, valid)
    legal_mask = wide[:, :action_count]
    if not bool(legal_mask.any(dim=1).all().item()):
        raise ContractError("legal_mask must contain at least one True at a decision")

    def req(name: str) -> torch.Tensor:
        try:
            value = planes[name]
        except KeyError as exc:
            raise ContractError(f"rust batch missing plane {name}") from exc
        if not isinstance(value, torch.Tensor):
            raise ContractError(f"rust plane {name} must be a Tensor")
        if int(value.shape[0]) != batch_size:
            raise ContractError(f"rust plane {name} batch {value.shape} != {batch_size}")
        return value

    actor = req("actor")
    concealed = req("concealed_hand_counts").to(torch.int32)
    discards = req("visible_discards_counts").to(torch.int32)
    features: dict[str, torch.Tensor] = {
        "actor": actor,
        "actor_can_riichi": req("actor_can_riichi"),
        "actor_can_tsumo": req("actor_can_tsumo"),
        "actor_furiten": req("actor_furiten"),
        "actor_seats": actor,
        "concealed_hand_counts": concealed,
        "dealer": req("dealer"),
        "dora_indicators": req("dora_indicators"),
        "hand_number": req("hand_number"),
        "history_event_kind": hist_kind,
        "history_mask": hist_mask,
        "honba": req("honba"),
        "ippatsu_active": req("ippatsu_active"),
        "kan_count": req("kan_count"),
        "legal_mask": legal_mask,
        "live_wall_tiles_remaining": req("live_wall_tiles_remaining"),
        "own_drawn_tile": req("own_drawn_tile"),
        "phase": req("phase"),
        "riichi_states": req("riichi_states"),
        "riichi_sticks": req("riichi_sticks"),
        "round_index": req("round_index"),
        "round_wind": req("round_wind"),
        "scores": req("scores"),
        "seat_winds": req("seat_winds"),
        "turn_actor": req("turn_actor"),
        "visible_discards_counts": discards,
    }
    actor_batch = ActorTensorBatch(
        features=features,
        history_mask=hist_mask,
        legal_mask=legal_mask,
        observation_hashes=(),
        actor_seats=actor,
    )
    return {
        "actor_batch": actor_batch,
        "chosen_action_id": chosen,
        "legal_mask": legal_mask,
    }


def _plane_schema() -> dict[str, tuple[Any, Any]]:
    """Name → (torch dtype, fixed width or ``"T"``) for game plane blobs."""
    from hydra2.training.rust_stream import _HOT_PLANES

    schema: dict[str, tuple[Any, Any]] = {}
    for name, cols, dtype_name in _HOT_PLANES:
        schema[name] = (getattr(torch, dtype_name), cols)
    return schema


def _slice_game_planes(
    blob: dict[str, bytes],
    start: int,
    count: int,
    t_len: int,
    t_pad: int,
    names: list[str],
    schema: dict[str, tuple[Any, Any]],
) -> dict[str, torch.Tensor]:
    """Slice ``[start:start+count]`` rows from one game blob (zero-copy views).

    History planes pad to ``t_pad`` (zeros/False — model-masked, values
    preserved); ``legal_ids``/``legal_len`` arrive unpacked from the bridge.
    """
    out: dict[str, torch.Tensor] = {}
    for name in names:
        data = blob[name]
        if name == "legal_ids":
            arr = torch.frombuffer(data, dtype=torch.int32).reshape(-1, 32)
            out[name] = arr[start : start + count]
        elif name == "legal_len":
            arr = torch.frombuffer(data, dtype=torch.int64)
            out[name] = arr[start : start + count]
        elif name in ("history_event_kind", "history_mask"):
            dtype = torch.int64 if name == "history_event_kind" else torch.bool
            arr = torch.frombuffer(data, dtype=dtype).reshape(-1, t_len)
            piece = arr[start : start + count]
            if t_len == t_pad:
                out[name] = piece
            else:
                padded = torch.zeros((count, t_pad), dtype=dtype)
                padded[:, :t_len] = piece
                out[name] = padded
        else:
            dtype, cols = schema[name]
            arr = torch.frombuffer(data, dtype=dtype)
            if isinstance(cols, int):
                arr = arr.reshape(-1, cols)
            out[name] = arr[start : start + count]
    return out


def assemble_slim_batch(
    rows: list[dict[str, Any]],
    *,
    action_count: int = BASELINE_ACTION_COUNT,
) -> dict[str, Any]:
    """Assemble a loop-ready batch from slim Rust-plane rows (no encoder).

    Rows carry a shared per-game planes blob (``_planes`` name→bytes) plus
    ``_row`` offset and ``_t_len``. Consecutive rows sharing one blob form
    one zero-copy slice group; groups concat (histories padded to the batch
    max-T) and :func:`assemble_training_batch` finishes. Returns
    ``{"actor_batch", "chosen_action_id", "legal_mask", "_decision_ids",
    "_action_kinds"}`` — the exact training/eval batch contract. Any row
    without ``_planes`` fails closed (a mixed-backend buffer would silently
    mis-assemble).
    """
    if len(rows) == 0:
        raise ContractError("assemble_slim_batch requires at least one row")
    for row in rows:
        if "_planes" not in row or "_row" not in row or "_t_len" not in row:
            raise ContractError("slim row lacks Rust planes (mixed-backend buffer?)")
    schema = _plane_schema()
    names = sorted(rows[0]["_planes"].keys())
    t_max = 0
    for row in rows:
        t_len = int(row["_t_len"])
        if t_len > t_max:
            t_max = t_len
    groups: list[tuple[dict[str, bytes], int, int, int]] = []
    for row in rows:
        blob = row["_planes"]
        if groups and groups[-1][0] is blob:
            start, count, t_len = groups[-1][1], groups[-1][2], groups[-1][3]
            groups[-1] = (blob, start, count + 1, t_len)
        else:
            groups.append((blob, int(row["_row"]), 1, int(row["_t_len"])))
    parts = [
        _slice_game_planes(blob, start, count, t_len, t_max, names, schema)
        for blob, start, count, t_len in groups
    ]
    if len(parts) == 1:
        planes = parts[0]
    else:
        planes = {name: torch.cat([part[name] for part in parts], dim=0) for name in names}
    batch = assemble_training_batch(planes, action_count=action_count)
    batch["_decision_ids"] = [str(row["decision_id"]) for row in rows]
    batch["_action_kinds"] = [str(row["action_kind"]) for row in rows]
    return batch


def replay_game_planes_raw(
    events: bytes, game_idx: int, wall: list[int] | None
) -> tuple[dict[str, bytes], int, int, bool, str, int]:
    """Frame + gate + walk one game in Rust with an overriding wall.

    ``events`` is the raw framed game bytes (no Python re-serialization);
    ``wall`` (136 ints) replaces the first event's wall in Rust when given,
    ``None`` walks the embedded content verbatim. Returns the same
    ``(planes, rows, t_len, quarantined, reason, event_idx)`` shape as
    :func:`replay_game_planes`.
    """
    ext = _load_extension()
    try:
        out = ext.replay_game_planes_wall(events, game_idx, wall)
    except Exception as exc:
        raise ContractError(f"rust game walk failed: {exc}") from exc
    planes, rows, t_len, quarantined, reason, event_idx = out
    return (dict(planes), int(rows), int(t_len), bool(quarantined), str(reason), int(event_idx))


def replay_game_planes(
    events: bytes, game_idx: int
) -> tuple[dict[str, bytes], int, int, bool, str, int]:
    """Frame + gate + walk one game in Rust; serve staged planes.

    ``events`` is the raw game bytes (newline-separated mjai lines, trailing
    newline included); ``game_idx`` is the caller lineage key. Returns
    ``(planes, rows, t_len, quarantined, reason, event_idx)`` where
    ``planes`` maps §7 names to raw LE bytes (fixed planes ``[rows *
    stride]``, history planes dense ``[rows * t_len]``) with the
    ``legal_ids``/``legal_len`` entries unpacked (never ``legal_packed``),
    ready for :func:`assemble_training_batch` after a zero-copy
    ``torch.frombuffer`` wrap. Walled regime follows wall content, exactly
    like the file fill. Quarantine is data (``quarantined=True`` + feed
    reason string); only argument/host failures raise.
    """
    ext = _load_extension()
    try:
        out = ext.replay_game_planes(events, game_idx)
    except Exception as exc:
        raise ContractError(f"rust game walk failed: {exc}") from exc
    planes, rows, t_len, quarantined, reason, event_idx = out
    return (dict(planes), int(rows), int(t_len), bool(quarantined), str(reason), int(event_idx))


def expand_game_batch(
    items: list[tuple[bytes, int, list[int] | None]],
) -> list[tuple[dict[str, bytes], int, int, bool, str, int]]:
    """Frame + gate + walk many games in one Rust call; per-game planes.

    ``items`` is ``(raw_bytes, game_idx, wall)`` per game in pull order
    (``wall`` splices an override in Rust, ``None`` walks embedded
    content). Returns one ``(planes, rows, t_len, quarantined, reason,
    event_idx)`` tuple per input game, in order, with the exact shapes
    :func:`replay_game_planes_raw` serves. One FFI crossing and one GIL
    release stage the whole batch; per-item stage failures ride as
    quarantine data (same text the serial path raises), never as a
    call-level error.
    """
    ext = _load_extension()
    try:
        outs = ext.expand_games(items)
    except Exception as exc:
        raise ContractError(f"rust game walk failed: {exc}") from exc
    got = []
    for out in outs:
        planes, rows, t_len, quarantined, reason, event_idx = out
        got.append(
            (dict(planes), int(rows), int(t_len), bool(quarantined), str(reason), int(event_idx))
        )
    return got
