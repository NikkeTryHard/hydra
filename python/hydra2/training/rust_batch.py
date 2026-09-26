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
packed legal ids need a scatter. History width is the batch max-``t_len``
snapped UP to the model bucket ceil (same ``bucket_for_length`` the encoder
uses) — values on the real prefix are identical and both pad with 0/False,
which the model masks out exactly (verified bitwise-equal logits in the
parity test). Snapping bounds torch.compile to four T shapes and lets the
H2D ring shape-match bucketed batches instead of falling back per batch.

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
from hydra2.models.encoder import ActorTensorBatch, bucket_for_length
from hydra2.models.encoder import PackedHistories as PackedHistories
from hydra2.models.schema import BASELINE_ACTION_COUNT, HISTORY_BUCKET_LENGTHS

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "assemble_slim_batch",
    "assemble_training_batch",
    "build_packed_batch",
    "expand_game_batch",
    "replay_game_planes",
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

    ``pixi run build-ext`` writes ``hydra2._native.build.json`` beside the
    installed ``.so`` recording the source tree and its newest mtime.
    Ad-hoc staged copies (tests, harness scratch dirs) carry no sidecar and
    skip the check — they build their extension in the same session. When
    the sidecar exists but the tree is gone (source-less deploy), there is
    nothing to compare against and the import is allowed. When the sidecar
    points at a DIFFERENT checkout than the caller's (xdist/controller
    resolving another worktree's env, e.g. ``/home/cachybtw/dev/hydra2`` vs
    ``/home/cachybtw/dev/hydra2-mp-endstate``), the comparison is meaningless
    and skipped — each checkout's own lane gates its own build.
    """
    from pathlib import Path

    if ext_file in _FRESH_CHECKED:
        return
    sidecar = Path(ext_file).with_name("hydra2._native.build.json")
    try:
        import json

        record: dict[str, Any] = json.loads(sidecar.read_text(encoding="utf-8"))
        root: str = str(record["source_root"])
        pinned: float = float(record["source_mtime"])
    except (OSError, ValueError, KeyError, TypeError):
        return
    try:
        here = str(Path.cwd().resolve())
        # Same-checkout test: sidecar root must live under the caller's
        # checkout (worktree root = parent of the first ``.pixi`` hit, else
        # the cwd itself). Cross-checkout imports skip the check. Narrowed
        # to the REAL stale-bridge shape: the sidecar must sit beside an
        # installed ``.so`` inside a ``.pixi`` env (``pixi run build-ext``
        # output). Tmp-fixture sidecars (bare ``build.json`` next to a fake
        # ``.so`` with no ``.pixi`` ancestor) always get the full freshness
        # comparison — otherwise the staleness unit test cannot fail closed.
        anchor = here
        for parent in [here, *list(Path(here).parents)]:
            if (Path(parent) / ".pixi").is_dir():
                anchor = parent
                break
        root_path = Path(root)
        anchor_path = Path(anchor, "crates")
        ext_in_pixi = ".pixi" in Path(ext_file).parts
        if (
            ext_in_pixi
            and root_path.is_absolute()
            and anchor_path.is_absolute()
            and root_path.resolve() != anchor_path.resolve()
        ):
            # Different checkout's installed env (or relocated tree): not our
            # build to judge.
            _FRESH_CHECKED.add(ext_file)
            return
    except OSError:
        return
    newest = _source_newest_mtime(root)
    if newest is None:
        return
    if newest > pinned + _FRESH_EPSILON_S:
        raise ContractError(
            "hydra2._native extension is stale (Rust sources newer than the "
            "installed build); re-run `pixi run build-ext`"
        )
    _FRESH_CHECKED.add(ext_file)


def _load_extension() -> Any:
    """Import the compiled ``hydra2._native`` module (fail closed)."""
    try:
        ext = importlib.import_module("hydra2._native")
    except ImportError as exc:
        raise ContractError(
            "hydra2._native extension not importable; run `pixi run build-ext` first"
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
        chosen: torch.Tensor = planes["chosen_action_id"]
        legal_ids: torch.Tensor = planes["legal_ids"]
        legal_len: torch.Tensor = planes["legal_len"]
        hist_kind: torch.Tensor = planes["history_event_kind"]
        hist_mask: torch.Tensor = planes["history_mask"]
    except KeyError as exc:
        raise ContractError(f"rust batch missing plane {exc}") from exc
    batch_size: int = chosen.shape[0]
    if batch_size == 0:
        raise ContractError("assemble_training_batch requires at least one row")
    t_len = hist_kind.shape[1]
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
    _ = wide.scatter_(1, safe, valid)
    legal_mask = wide[:, :action_count]
    if not bool(legal_mask.any(dim=1).all().item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # single host sync for legality gate; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise ContractError("legal_mask must contain at least one True at a decision")

    def req(name: str) -> torch.Tensor:
        try:
            value = planes[name]
        except KeyError as exc:
            raise ContractError(f"rust batch missing plane {name}") from exc
        if not isinstance(value, torch.Tensor):
            raise ContractError(f"rust plane {name} must be a Tensor")
        if value.shape[0] != batch_size:
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


def build_packed_batch(
    rows: list[dict[str, Any]],
    batch: dict[str, Any],
) -> PackedHistories:
    """Pack one assembled microbatch's real history prefixes (CPU).

    Rows stay in pull order (never sorted — order is determinism): for each
    row ``i`` the real length ``L_i`` is its history-mask popcount, and the
    packed stream concatenates ``history_event_kind[i, :L_i]``. Lengths
    cross-check the staged ``_hist_len`` the Rust pull stamps per row (exact
    mask popcount at stage time): any disagreement fails closed instead of
    silently shifting a window. Masks must be prefix-contiguous (real block
    then pad tail); a hole fails closed. Rows without a staged length fall
    back to the max bucket path at the caller (grouping must never mask
    errors), so this function requires the stamp. An all-empty batch
    (total 0) packs to empty bounds; the model renders the zero-vector
    pooling the bucketed path produces for empty histories.
    """
    try:
        actor = batch["actor_batch"]
        kind: torch.Tensor = actor.features["history_event_kind"]
        mask: torch.Tensor = actor.history_mask
    except (KeyError, AttributeError) as exc:
        raise ContractError(f"packed batch needs assembled history planes: {exc}") from exc
    batch_size = int(mask.shape[0])
    if len(rows) != batch_size:
        raise ContractError(
            f"packed rows {len(rows)} != batch rows {batch_size} (order is determinism)"
        )
    if kind.shape != mask.shape:
        raise ContractError(f"packed kind {tuple(kind.shape)} != mask {tuple(mask.shape)}")
    pad_width = int(mask.shape[1])
    # Vectorized lengths: one popcount kernel, one host transfer (CPU
    # planes, so no device syncs) — never a per-row Python loop.
    lengths_t: torch.Tensor = mask.sum(dim=1).to(torch.int32)
    lengths: list[int] = lengths_t.tolist()
    # Staged cross-check in one vector compare (exact mask popcount stamped
    # at pull time): any disagreement fails closed naming the first bad
    # row instead of silently shifting a window.
    staged: list[int] = []
    for pos, row in enumerate(rows):
        stamp = row.get("_hist_len")
        if isinstance(stamp, bool) or not isinstance(stamp, int):
            raise ContractError(
                f"packed row {pos} lacks staged _hist_len (grouping never masks errors)"
            )
        staged.append(int(stamp))
    staged_t: torch.Tensor = torch.tensor(staged)
    if bool((lengths_t != staged_t).any().item()):
        bad = int((lengths_t != staged_t).nonzero()[0].item())
        raise ContractError(
            f"packed row {bad} staged length {staged[bad]} != mask popcount {lengths[bad]}"
        )
    # Prefix-contiguity in one op: a real block followed by pad tail has no
    # 0->1 transition anywhere (empty and full rows pass trivially).
    if pad_width > 1 and bool((mask[:, 1:] & ~mask[:, :-1]).any().item()):
        raise ContractError("packed history mask not prefix-contiguous")
    # Single boolean-select kernel in row-major order = pull order.
    packed_kind: torch.Tensor = kind[mask]
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    cu_seqlens = torch.tensor(bounds, dtype=torch.int32)
    return PackedHistories(
        packed_kind=packed_kind,
        cu_seqlens=cu_seqlens,
        row_lengths=tuple(lengths),
        max_len=max(lengths) if len(lengths) > 0 else 0,
    )


def assemble_slim_batch(
    rows: list[dict[str, Any]],
    *,
    action_count: int = BASELINE_ACTION_COUNT,
    pack_histories: bool = False,
) -> dict[str, Any]:
    """Assemble a loop-ready batch from slim Rust-plane rows (no encoder).

    Rows carry a shared per-game planes blob (``_planes`` name→bytes) plus
    ``_row`` offset and ``_t_len``. Consecutive rows sharing one blob form
    one zero-copy slice group; groups concat (histories padded to the batch
    max-T snapped UP to the model bucket ceil) and :func:`assemble_training_batch`
    finishes. Returns
    ``{"actor_batch", "chosen_action_id", "legal_mask", "_decision_ids",
    "_action_kinds"}`` — the exact training/eval batch contract. Any row
    without ``_planes`` fails closed (a mixed-backend buffer would silently
    mis-assemble). With ``pack_histories`` the batch additionally carries
    the packed stream on ``actor_batch.packed`` (padded planes stay: the
    model consumes packed instead of ``[B,T]`` when present).
    """
    if len(rows) == 0:
        raise ContractError("assemble_slim_batch requires at least one row")
    for row in rows:
        if "_planes" not in row or "_row" not in row or "_t_len" not in row:
            raise ContractError("slim row lacks Rust planes (mixed-backend buffer?)")
    schema = _plane_schema()
    first_planes: dict[str, bytes] = rows[0]["_planes"]
    names: list[str] = sorted(first_planes.keys())
    t_max = 0
    for row in rows:
        t_len = int(row["_t_len"])
        if t_len > t_max:
            t_max = t_len
    if t_max > HISTORY_BUCKET_LENGTHS[-1]:
        raise ContractError(
            f"rust history width {t_max} exceeds model bucket cap {HISTORY_BUCKET_LENGTHS[-1]}; "
            "rows are never truncated"
        )
    # Bucket-snap: pad histories to the bucket ceil, not the raw batch max.
    # Every batch then carries one of (32, 64, 128, 256), so torch.compile sees
    # at most four T shapes (no per-length recompile drizzle) and the H2D ring
    # shape-matches more often. Values on the real prefix are identical and
    # padding is model-masked, matching the encoder path exactly.
    t_pad = bucket_for_length(t_max)
    # Slice groups key on exact _row continuity, never blob adjacency: the
    # homogeneous-bucket take reorders rows, so same-blob neighbors in take
    # order are routinely non-consecutive decisions (e.g. seq 5 then 17).
    # Merging those as [start, start+count) silently pairs decision 17's
    # identity/labels with decision 5's planes (packed cross-check caught
    # staged 3 vs mask 34 on exactly this shape). Per-row slices on any
    # mismatch are always correct; slicing is zero-copy views, padding copies
    # dominate either way.
    groups: list[tuple[dict[str, bytes], int, int, int]] = []
    for row in rows:
        blob: dict[str, bytes] = row["_planes"]
        row_start = int(row["_row"])
        row_t = int(row["_t_len"])
        if (
            len(groups) > 0
            and groups[-1][0] is blob
            and groups[-1][3] == row_t
            and row_start == groups[-1][1] + groups[-1][2]
        ):
            start, count, t_len = groups[-1][1], groups[-1][2], groups[-1][3]
            groups[-1] = (blob, start, count + 1, t_len)
        else:
            groups.append((blob, row_start, 1, row_t))
    parts: list[dict[str, torch.Tensor]] = [
        _slice_game_planes(blob, start, count, t_len, t_pad, names, schema)
        for blob, start, count, t_len in groups
    ]
    if len(parts) == 1:
        planes = parts[0]
    else:
        planes = {name: torch.cat([part[name] for part in parts], dim=0) for name in names}  # pyrefly: ignore[unknown-argument-type] # part slices dynamic; cat owns the stacking
    batch = assemble_training_batch(planes, action_count=action_count)
    batch["_decision_ids"] = [str(row["decision_id"]) for row in rows]
    batch["_action_kinds"] = [str(row["action_kind"]) for row in rows]
    if pack_histories:
        import dataclasses

        packed = build_packed_batch(rows, batch)
        batch["actor_batch"] = dataclasses.replace(batch["actor_batch"], packed=packed)
    return batch


def replay_game_planes(
    events: bytes, game_idx: int
) -> tuple[dict[str, bytes], int, int, bool, str, int]:
    """Frame + gate + walk one game in Rust; serve staged planes.

    Bridge-plane consume (already wired — no new surfaces this slice):
    ``replay_game_planes`` serves the staged planes the file fill commits
    (feed 172 green); only argument/host failures raise.

    ``events`` is the raw game bytes (newline-separated mjai lines, trailing
    ``torch.frombuffer`` wrap. Walled regime follows wall content, exactly
    like the file fill. Quarantine is data (``quarantined=True`` + feed
    reason string); only argument/host failures raise.
    """
    ext = _load_extension()
    try:
        out: tuple[dict[str, bytes], int, int, bool, str, int] = ext.replay_game_planes(
            events, game_idx
        )
    except Exception as exc:
        raise ContractError(f"rust game walk failed: {exc}") from exc
    staged: tuple[dict[str, bytes], int, int, bool, str, int] = out
    planes, rows, t_len, quarantined, reason, event_idx = staged
    return (dict(planes), rows, t_len, quarantined, reason, event_idx)


def expand_game_batch(
    items: list[tuple[bytes, int, list[int] | None]],
) -> list[tuple[dict[str, bytes], int, int, bool, str, int]]:
    """Frame + gate + walk many games in one Rust call; per-game planes.

    Bridge-plane consume (already wired — no new surfaces this slice):
    ``expand_games`` serves per-game staged planes under one GIL release
    (feed 172 green); per-item failures ride as quarantine data.

    ``items`` is ``(raw_bytes, game_idx, wall)`` per game in pull order
    (``wall`` splices an override in Rust, ``None`` walks embedded
    content). Returns one ``(planes, rows, t_len, quarantined, reason,
    event_idx)`` tuple per input game, in order, with the exact shapes
    :func:`replay_game_planes` serves. One FFI crossing and one GIL
    release stage the whole batch; per-item stage failures ride as
    quarantine data (same text the serial path raises), never as a
    call-level error.
    """
    ext = _load_extension()
    try:
        outs: list[tuple[dict[str, bytes], int, int, bool, str, int]] = ext.expand_games(items)
    except Exception as exc:
        raise ContractError(f"rust game walk failed: {exc}") from exc
    got: list[tuple[dict[str, bytes], int, int, bool, str, int]] = []
    for staged in outs:
        planes, rows, t_len, quarantined, reason, event_idx = staged
        got.append((dict(planes), rows, t_len, quarantined, reason, event_idx))
    return got
