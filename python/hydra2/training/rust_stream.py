"""Thin plane-filling handoff over the PyO3 boundary (sole replay path).

Bridge-plane consume (already wired — no new surfaces this slice):
``PyHydraStream`` stages framed per-game inputs and fills caller-owned
pinned plane slots DIRECTLY via ``data_ptr`` (feed 172 + shard 311 green).
``next_into`` takes 13 base pointers + 13 byte caps (``tensor.nbytes``) in
§7 plane order and returns the committed ``(rows, games, t_len)``; the
valid prefix moves to device on a single transfer stream with a slot-local
``wait_event`` (no global sync, no ``.item``).

Open pins are opaque non-empty digests (``source_hash``/``rules_hash``/
``action_table_hash``); the closed-form walk ids are pinned entry-wise to
the compiled action-table digest inside Rust, which fails closed at open
when the pin itself drifts. Quarantines cross as whole-game records with
feed reason codes (never a new code on this path).

Scratch discipline mirrors the Rust parity gate: when
``HYDRA2_ARTIFACT_ROOT`` is set it must sit outside the raw corpus roots
(``HYDRA2_DATA_ROOT``, ``HYDRA2_TENHOU_MOUNT``); the corpus stays read-only.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

__all__ = [
    "RustPlaneFill",
    "RustPlaneQuarantine",
    "RustPlaneStream",
    "RustStreamStats",
    "assert_artifact_root_outside_raw_roots",
    "open_rust_plane_stream",
]

_RAW_ROOT_KEYS = ("HYDRA2_DATA_ROOT", "HYDRA2_TENHOU_MOUNT")


def assert_artifact_root_outside_raw_roots() -> None:
    """Fail closed when artifacts would land inside a raw corpus mount."""
    artifact = os.environ.get("HYDRA2_ARTIFACT_ROOT", "")
    if artifact == "":
        return
    for key in _RAW_ROOT_KEYS:
        raw = os.environ.get(key, "")
        if raw == "":
            continue
        if artifact.startswith(raw) or raw.startswith(artifact):
            raise ValueError(
                f"{key} raw root must sit outside HYDRA2_ARTIFACT_ROOT: "
                f"artifact={artifact} raw={raw}"
            )


def _load_extension() -> Any:
    """Import the compiled ``hydra2._native`` module (fail closed)."""
    try:
        return importlib.import_module("hydra2._native")
    except ImportError as exc:
        raise RuntimeError(
            "hydra2._native extension not importable; run `pixi run build-ext` first"
        ) from exc


@dataclass(frozen=True, slots=True)
class RustStreamStats:
    open_count: int
    games_ok: int
    games_quarantined: int
    rows_out: int


_T_BUCKETS = (32, 64, 128, 256)

#: Hot plane order (§7 #0-25; ptrs order = table order). Each entry is
#: ``(name, cols, dtype_name)`` with ``cols`` None (scalar), an int (fixed
#: ``"legal"`` (per-row stride ``[ids128][len8]``, 136B/row, carried as raw
#: bytes and unpacked per row on device — never SoA across rows).
#: resolved lazily in :class:`RustPlaneStream`.
_HOT_PLANES: tuple[tuple[str, int | str | None, str], ...] = (
    ("concealed_hand_counts", 34, "uint8"),
    ("visible_discards_counts", 34, "uint8"),
    ("dora_indicators", 5, "int32"),
    ("scores", 4, "int32"),
    ("history_event_kind", "T", "int64"),
    ("history_mask", "T", "bool"),
    ("chosen_action_id", None, "int64"),
    ("actor", None, "int64"),
    ("dealer", None, "int64"),
    ("round_wind", None, "int64"),
    ("phase", None, "int64"),
    ("seat_winds", 4, "int64"),
    ("legal_packed", "legal", "uint8"),
    ("turn_actor", None, "int64"),
    ("actor_furiten", None, "int64"),
    ("honba", None, "int32"),
    ("riichi_sticks", None, "int32"),
    ("live_wall_tiles_remaining", None, "int32"),
    ("kan_count", None, "int32"),
    ("round_index", None, "int32"),
    ("hand_number", None, "int32"),
    ("own_drawn_tile", None, "int32"),
    ("ippatsu_active", 4, "bool"),
    ("riichi_states", 4, "int64"),
    ("actor_can_riichi", None, "bool"),
    ("actor_can_tsumo", None, "bool"),
)


@dataclass(frozen=True, slots=True)
class RustPlaneFill:
    rows: int
    games_consumed: int
    games_quarantined: int
    t_len: int


@dataclass(frozen=True, slots=True)
class RustPlaneQuarantine:
    game_id: str
    reason_code: str
    detail: str
    obs_hash: int


class RustPlaneStream:
    """Plane-filling stream over caller-pinned slots (sole replay handoff).
    Rust fills pinned ring slots DIRECTLY via ``data_ptr`` — no JSON, no
    per-row copies — and ``next_into_planes`` moves the valid prefix to
    device on a single transfer stream with slot-local ``wait_event``
    (the :mod:`pinned_ring` pattern: no global sync, no ``.item``).

    Per §6.5 caller: rings are built ONCE (``depth`` slots x 13 planes,
    ``pin_memory=True``), ``byte_caps`` are ``tensor.nbytes``, the GIL is
    released for the entire fill, and ``slot_events[cur]`` is recorded
    AFTER the ``.to()``s on the transfer stream. History planes transfer
    ``[:rows, :t_len]`` (the fill commits ``t_len`` entries per history
    row; beyond-``t_len`` slot bytes are untouched filler, never rows).
    """

    def __init__(
        self,
        data_dirs: list[str],
        batch: int,
        *,
        split: str = "train",
        source_hash: str = "",
        rules_hash: str = "",
        action_table_hash: str = "",
        slot_rows: int = 256,
        t_max: int = 256,
        depth: int = 2,
        device: str = "cuda",
    ) -> None:
        import time
        from collections import deque

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("rust plane stream needs torch") from exc
        assert_artifact_root_outside_raw_roots()
        if len(data_dirs) == 0:
            raise ValueError("rust plane stream data_dirs must not be empty")
        if batch < 1:
            raise ValueError(f"rust plane stream batch must be >= 1, got {batch}")
        if split not in ("train", "validation"):
            raise ValueError(f"rust plane stream split must be train|validation, got {split!r}")
        for name, value in (
            ("source_hash", source_hash),
            ("rules_hash", rules_hash),
            ("action_table_hash", action_table_hash),
        ):
            if value == "":
                raise ValueError(f"rust plane stream {name} must not be empty")
        if slot_rows < 1:
            raise ValueError(f"rust plane stream slot_rows must be >= 1, got {slot_rows}")
        if t_max not in _T_BUCKETS:
            raise ValueError(f"rust plane stream t_max must be one of {_T_BUCKETS}, got {t_max}")
        if depth < 1:
            raise ValueError(f"rust plane stream depth must be >= 1, got {depth}")
        self._torch = torch
        self._time = time
        self._t_max = t_max
        self._batch = batch
        resolved = torch.device(device)
        self._device = resolved
        self._cuda = resolved.type == "cuda" and torch.cuda.is_available()
        dtypes = {n: getattr(torch, n) for _, _, n in _HOT_PLANES}

        def _shape(cols: int | str | None) -> tuple[int, ...]:
            if cols is None:
                return (slot_rows,)
            if cols == "T":
                return (slot_rows, t_max)
            if cols == "legal":
                return (slot_rows * 136,)
            if isinstance(cols, int):
                return (slot_rows, cols)
            raise ValueError(f"rust plane stream bad plane width {cols!r}")

        try:
            self._rings = [
                [
                    torch.empty(_shape(cols), dtype=dtypes[dtype], pin_memory=True)
                    for _, cols, dtype in _HOT_PLANES
                ]
                for _ in range(depth)
            ]
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"rust plane stream pinned alloc failed (fail closed): {exc}") from exc
        unpinned = sorted(
            f"slot-{s}/plane-{p}"
            for s, slot in enumerate(self._rings)
            for p, tensor in enumerate(slot)
            if not tensor.is_pinned()
        )
        if len(unpinned) > 0:
            self._rings = []
            raise ValueError(f"rust plane stream unpinnable for {unpinned} (fail closed)")
        # Event/stream handles hold torch CUDA objects on GPU runs and None
        # on CPU runs; typed Any so the CUDA-only use sites (guarded by
        # self._cuda) narrow without Optional-element assertions.
        self._transfer: Any
        self._consume: list[Any]
        self._start: list[Any]
        self._end: list[Any]
        if self._cuda:
            self._transfer = torch.cuda.Stream(device=resolved)
            self._consume = [torch.cuda.Event(enable_timing=True) for _ in range(depth)]
            self._start = [torch.cuda.Event(enable_timing=True) for _ in range(depth)]
            self._end = [torch.cuda.Event(enable_timing=True) for _ in range(depth)]
        else:
            self._transfer = None
            self._consume = [None] * depth
            self._start = [None] * depth
            self._end = [None] * depth
        self._depth = depth
        self._cursor = 0
        self._closed = False
        self._slot_used = [False] * depth
        self._sync_ms: deque[float] = deque(maxlen=1024)
        self._h2d_ms: deque[float] = deque(maxlen=1024)
        ext = _load_extension()
        self._handle: Any = ext.PyHydraStream.open(
            list(data_dirs), batch, split, source_hash, rules_hash, action_table_hash
        )

    def next_into_planes(self) -> tuple[dict[str, Any], RustPlaneFill]:
        """Fill one slot and move its valid prefix to device (slot-local wait)."""
        torch = self._torch
        if self._closed:
            raise ValueError("hydra2 replay stream is closed")
        cur = self._cursor
        slot = self._rings[cur]
        if self._cuda:
            # Reuse gate (pinned_ring pattern): stall here is the overlap
            # signal — near-zero once fills + transfers pipeline at depth>=2.
            wait_began = self._time.perf_counter()
            self._consume[cur].synchronize()
            self._sync_ms.append((self._time.perf_counter() - wait_began) * 1000.0)
            if self._slot_used[cur]:
                elapsed: float = self._start[cur].elapsed_time(self._end[cur])
                self._h2d_ms.append(elapsed)
        ptrs: list[int] = [t.data_ptr() for t in slot]
        byte_caps: list[int] = [t.nbytes for t in slot]
        raw: Any = self._handle.next_into(ptrs, byte_caps)  # pyrefly: ignore[unknown-argument-type] # data_ptr/nbytes lists dynamic; native owns the gates
        fill_rows: int = raw.rows
        fill_games: int = raw.games_consumed
        fill_quarantined: int = raw.games_quarantined
        fill_t_len: int = raw.t_len
        fill = RustPlaneFill(
            rows=fill_rows,
            games_consumed=fill_games,
            games_quarantined=fill_quarantined,
            t_len=fill_t_len,
        )
        rows, t_len = fill.rows, fill.t_len
        self._cursor = (cur + 1) % self._depth
        if rows == 0:
            return {}, fill
        if t_len > self._t_max:
            raise ValueError(
                f"rust plane stream fill t_len {t_len} exceeds slot t_max {self._t_max}; "
                "reopen with a bigger t_max"
            )
        out: dict[str, Any] = {}
        if self._cuda:
            with torch.cuda.stream(self._transfer):
                self._start[cur].record()
                for (name, cols, _), tensor in zip(_HOT_PLANES, slot, strict=True):
                    if cols == "T":
                        out[name] = tensor[:rows, :t_len].to(self._device, non_blocking=True)
                    elif cols == "legal":
                        out[name] = tensor[: rows * 136].to(self._device, non_blocking=True)
                    else:
                        out[name] = tensor[:rows].to(self._device, non_blocking=True)
                packed = out.pop("legal_packed")
                # Per-row stride, not SoA: view rows x 136B, then split. The
                # slices copy ~rows*136B once (35KB at 256 rows) — negligible
                # next to the H2D that follows on this stream.
                strided: torch.Tensor = packed.view(rows, 136)
                out["legal_ids"] = strided[:, :128].contiguous().view(torch.int32).reshape(rows, 32)
                out["legal_len"] = strided[:, 128:].contiguous().view(torch.int64).reshape(rows)
                self._end[cur].record()
                self._consume[cur].record()
            torch.cuda.current_stream().wait_event(self._consume[cur])
        else:
            for (name, cols, _), tensor in zip(_HOT_PLANES, slot, strict=True):
                if cols == "T":
                    out[name] = tensor[:rows, :t_len].clone()
                elif cols == "legal":
                    packed = tensor[: rows * 136].clone()
                    strided = packed.view(rows, 136)
                    out["legal_ids"] = (
                        strided[:, :128].contiguous().view(torch.int32).reshape(rows, 32)
                    )
                    out["legal_len"] = strided[:, 128:].contiguous().view(torch.int64).reshape(rows)
                else:
                    out[name] = tensor[:rows].clone()
        self._slot_used[cur] = True
        return out, fill

    def timings(self) -> tuple[list[float], list[float]]:
        """(sync_ms, h2d_ms) samples: reuse stalls vs transfer durations."""
        return list(self._sync_ms), list(self._h2d_ms)

    def stats(self) -> RustStreamStats:
        """Cumulative counters (``open_count == 1`` on a live stream)."""
        raw: Any = self._handle.stats()
        open_count: int = raw.open_count
        games_ok: int = raw.games_ok
        games_quarantined: int = raw.games_quarantined
        rows_out: int = raw.rows_out
        return RustStreamStats(
            open_count=open_count,
            games_ok=games_ok,
            games_quarantined=games_quarantined,
            rows_out=rows_out,
        )

    def quarantines(self) -> list[RustPlaneQuarantine]:
        """Whole-game quarantines in load order (feed reason codes)."""
        quarantined: list[Any] = self._handle.quarantines()
        return [
            RustPlaneQuarantine(
                game_id=item.game_id,  # pyrefly: ignore[unknown-argument-type] # native quarantine record dynamic; constructor owns the gates
                reason_code=item.reason_code,  # pyrefly: ignore[unknown-argument-type] # native quarantine record dynamic; constructor owns the gates
                detail=item.detail,  # pyrefly: ignore[unknown-argument-type] # native quarantine record dynamic; constructor owns the gates
                obs_hash=item.obs_hash,  # pyrefly: ignore[unknown-argument-type] # native quarantine record dynamic; constructor owns the gates
            )
            for item in quarantined
        ]

    def close(self) -> None:
        """Idempotent close (second call is a no-op success)."""
        if self._closed:
            return
        self._closed = True
        self._handle.close()

    def __enter__(self) -> RustPlaneStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_rust_plane_stream(
    data_dirs: list[str],
    batch: int,
    *,
    split: str = "train",
    source_hash: str = "",
    rules_hash: str = "",
    action_table_hash: str = "",
    slot_rows: int = 256,
    t_max: int = 256,
    depth: int = 2,
    device: str = "cuda",
) -> RustPlaneStream:
    """Open a plane-filling stream (see :class:`RustPlaneStream`).
    Sole replay handoff: no JSON path exists on this bridge.
    """
    return RustPlaneStream(
        data_dirs=data_dirs,
        batch=batch,
        split=split,
        source_hash=source_hash,
        rules_hash=rules_hash,
        action_table_hash=action_table_hash,
        slot_rows=slot_rows,
        t_max=t_max,
        depth=depth,
        device=device,
    )
