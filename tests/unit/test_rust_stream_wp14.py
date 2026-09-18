"""Thin plane-filling handoff over the PyO3 boundary (K1 cutover, serial lane).

Drives ``hydra2.training.rust_stream`` (thin plane bridge) over the compiled
``hydra2_replay_rs`` extension: framed per-game inputs in, caller-pinned
plane slots out. Proves the K1 acceptance set:

- serial drain green over ``next_into_planes`` with the 26 §7 planes in
  order (unpacked ``legal_ids``/``legal_len``), exact dtypes, and the
  committed history width (``t_len`` bucket) on every call;
- determinism: two full passes commit identical plane bytes at the FFI
  boundary;
- stats ``open_count == 1`` with ok/quarantine/rows counters exact and
  ``games_ok + games_quarantined == games_consumed`` on every drain;
- quarantine classes stay in the closed feed vocabulary (``double-ron``
  only here; ``past-terminal`` admitted for hora tails without end_kyoku);
- fail-closed edges: tiny caller slot (``BufferError``, nothing drained,
  counters unmoved), raw too-small plane reporting the failing plane
  index, bad open args, empty digest pins, forged-but-well-formed pins
  draining identically (pins are opaque on this path — the closed-form
  walk ids pin to the compiled table digest inside Rust), and
  ``HYDRA2_ARTIFACT_ROOT`` inside a raw root;
- CUDA overlap mechanics: depth-2 ring drains through record + slot-local
  wait with recycle timing samples (quantitative ``sync < h2d`` proof
  lives on the F2 primary bench, not this serial lane).

The shared ``rust_extension`` session fixture (``tests/unit/conftest.py``)
imports the lane-built extension (``pixi run build-ext`` first; zero cargo
here); every test takes the fixture so the import path is live while the
stream opens.
"""

from __future__ import annotations

import ctypes
import json
from typing import TYPE_CHECKING, Any

import pytest
import torch

import hydra2.training.rust_stream as rust_stream

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.contract_package("WP-14"), pytest.mark.serial]

#: Opaque open pins (non-empty digests; the closed-form walk ids pin to the
#: compiled action-table digest inside Rust, never to these values).
PINS = {
    "source_hash": "wp14-source-v1",
    "rules_hash": "wp14-rules-v1",
    "action_table_hash": "wp14-action-table-v1",
}

#: Closed feed quarantine vocabulary: gate render names
#: (``crates/hydra-feed/src/gate.rs::reason_name``) plus walk render names
#: (``crates/hydra-feed/src/ledger.rs::walk_reason_name``). ``"other"`` is
#: the fail-loud canary and must never appear here.
FEED_QUARANTINE_CODES = frozenset(
    {
        "framing",
        "wall-bearing",
        "bare-dora",
        "double-ron",
        "unknown-event",
        "tile-conservation",
        "turn-order",
        "claim-no-offer",
        "draw-past-wall",
        "kyushu-ambiguous",
        "unmapped-ryukyoku-reason",
        "engine-desync",
        "action-id-unresolved",
        "past-terminal",
    }
)

#: Expected device-batch planes in §7 order (#0-25), with torch dtypes.
EXPECTED_PLANES: tuple[tuple[str, torch.dtype], ...] = (
    ("concealed_hand_counts", torch.uint8),
    ("visible_discards_counts", torch.uint8),
    ("dora_indicators", torch.int32),
    ("scores", torch.int32),
    ("history_event_kind", torch.int64),
    ("history_mask", torch.bool),
    ("chosen_action_id", torch.int64),
    ("actor", torch.int64),
    ("dealer", torch.int64),
    ("round_wind", torch.int64),
    ("phase", torch.int64),
    ("seat_winds", torch.int64),
    ("legal_ids", torch.int32),
    ("legal_len", torch.int64),
    ("turn_actor", torch.int64),
    ("actor_furiten", torch.int64),
    ("honba", torch.int32),
    ("riichi_sticks", torch.int32),
    ("live_wall_tiles_remaining", torch.int32),
    ("kan_count", torch.int32),
    ("round_index", torch.int32),
    ("hand_number", torch.int32),
    ("own_drawn_tile", torch.int32),
    ("ippatsu_active", torch.bool),
    ("riichi_states", torch.int64),
    ("actor_can_riichi", torch.bool),
    ("actor_can_tsumo", torch.bool),
)

_TILES = [
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
]


def _tile_at(slot: int) -> str:
    return _TILES[slot % len(_TILES)]


def _tehais() -> list[list[str]]:
    return [[_tile_at(seat * 13 + i) for i in range(13)] for seat in range(4)]


def _winning_tehais(seat: int) -> list[list[str]]:
    """Tehais where `seat` is tenpai (1112223334445m) for an engine-honest win.

    All four hands share one conservation budget (<=4 copies per tile):
    the winner's manzu are exclusive to it; other seats hold honors/sou
    only. Intermediate tsumogiri draws (1s/2s/3s/4s) stay within budget.
    """
    others = [
        ["E", "E", "E", "S", "S", "S", "W", "W", "W", "N", "N", "N", "P"],
        ["F", "F", "F", "C", "C", "C", "1s", "1s", "1s", "2s", "2s", "2s", "3s"],
        ["4s", "4s", "4s", "5s", "5s", "5s", "6s", "6s", "6s", "7s", "7s", "7s", "8s"],
    ]
    hands: list[list[str]] = []
    pool = list(others)
    for index in range(4):
        if index == seat:
            hands.append(
                [
                    "1m",
                    "1m",
                    "1m",
                    "2m",
                    "2m",
                    "2m",
                    "3m",
                    "3m",
                    "3m",
                    "4m",
                    "4m",
                    "4m",
                    "5m",
                ]
            )
        else:
            hands.append(pool.pop(0))
    return hands


def _framed_game(
    draws_before: int,
    tail: list[str],
    *,
    tehais: list[list[str]] | None = None,
) -> str:
    lines = [
        '{"type":"start_game"}',
        json.dumps(
            {
                "type": "start_kyoku",
                "bakaze": "E",
                "dora_marker": "3m",
                "kyoku": 1,
                "honba": 0,
                "kyotaku": 0,
                "oya": 0,
                "scores": [25000, 25000, 25000, 25000],
                "tehais": tehais if tehais is not None else _tehais(),
            }
        ),
    ]
    for turn in range(draws_before):
        seat = turn % 4
        pai = _tile_at(52 + turn)
        lines.append(json.dumps({"type": "tsumo", "actor": seat, "pai": pai}))
        lines.append(json.dumps({"type": "dahai", "actor": seat, "pai": pai, "tsumogiri": True}))
    lines.extend(tail)
    lines.append('{"type":"end_game"}')
    return "\n".join(lines) + "\n"


def _tsumo_win_game(draws_before: int) -> str:
    seat = draws_before % 4
    # Engine-honest win: tenpai 1112223334445m + tsumo 5m (tanyao+tsumo).
    return _framed_game(
        draws_before,
        [
            json.dumps({"type": "tsumo", "actor": seat, "pai": "5m"}),
            json.dumps(
                {
                    "type": "hora",
                    "actor": seat,
                    "target": seat,
                    "tsumo": True,
                    "deltas": [8000, -2000, -2000, -4000],
                }
            ),
            '{"type":"end_kyoku"}',
        ],
        tehais=_winning_tehais(seat),
    )


def _double_ron_game() -> str:
    return _framed_game(
        0,
        [
            json.dumps({"type": "tsumo", "actor": 0, "pai": "E"}),
            json.dumps({"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": True}),
            json.dumps(
                {
                    "type": "hora",
                    "actor": 1,
                    "target": 0,
                    "deltas": [8000, 8000, -8000, -8000],
                }
            ),
            json.dumps(
                {
                    "type": "hora",
                    "actor": 2,
                    "target": 0,
                    "deltas": [8000, -8000, 8000, -8000],
                }
            ),
        ],
    )


def _write_inputs(root: Path) -> Path:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "gA-good.jsonl").write_text(_tsumo_win_game(3))
    (inputs / "gB-good.jsonl").write_text(_tsumo_win_game(4))
    (inputs / "gC-double-ron.jsonl").write_text(_double_ron_game())
    return inputs


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Artifact root on scratch, raw roots unset (pass path of the gate)."""
    monkeypatch.setenv("HYDRA2_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.delenv("HYDRA2_DATA_ROOT", raising=False)
    monkeypatch.delenv("HYDRA2_TENHOU_MOUNT", raising=False)
    return tmp_path


def _drain_all(
    inputs: Path, batch: int, **kwargs: Any
) -> tuple[list[tuple[Any, dict[str, torch.Tensor]]], Any, list[Any]]:
    """Drain one full pass via ``next_into_planes`` (CPU, deterministic)."""
    params: dict[str, Any] = {"slot_rows": 64, "t_max": 64, "depth": 2, "device": "cpu"}
    params.update(kwargs)
    snapshots: list[tuple[Any, dict[str, torch.Tensor]]] = []
    with rust_stream.open_rust_plane_stream(
        [str(inputs)], batch, split="train", **PINS, **params
    ) as stream:
        while True:
            batch_out, fill = stream.next_into_planes()
            snapshots.append((fill, batch_out))
            if fill.rows == 0 and fill.games_consumed == 0:
                break
        stats = stream.stats()
        quars = stream.quarantines()
    return snapshots, stats, quars


def _snapshot_key(snapshots: list[tuple[Any, dict[str, torch.Tensor]]]) -> list[Any]:
    """Fill tuples plus per-plane bytes (cross-pass determinism key)."""
    keyed: list[Any] = []
    for fill, batch_out in snapshots:
        planes = tuple(
            (name, tuple(t.shape), str(t.dtype), bytes(t.cpu().numpy().tobytes()))
            for name, t in batch_out.items()
        )
        keyed.append(((fill.rows, fill.games_consumed, fill.games_quarantined, fill.t_len), planes))
    return keyed


def test_plane_drain_stats_quarantine_envelope(
    rust_extension: Any, clean_env: Path, tmp_path: Path
) -> None:
    inputs = _write_inputs(tmp_path)
    snapshots, stats, quars = _drain_all(inputs, 2)
    assert stats.open_count == 1
    total_rows = sum(fill.rows for fill, _ in snapshots if fill.rows > 0)
    assert total_rows == 9, "walk sampling drift: gA(3-draw win)=4 + gB(4-draw win)=5 rows"
    consumed = sum(fill.games_consumed for fill, _ in snapshots)
    quarantined = sum(fill.games_quarantined for fill, _ in snapshots)
    assert stats.rows_out == total_rows
    assert stats.games_ok + stats.games_quarantined == consumed == 3
    assert stats.games_quarantined == quarantined == 1
    codes = {q.reason_code for q in quars}
    assert codes <= FEED_QUARANTINE_CODES, f"new reason codes: {codes - FEED_QUARANTINE_CODES}"
    assert "other" not in codes
    by_id = {q.game_id: q for q in quars}
    assert by_id["gC-double-ron"].reason_code == "double-ron"
    for fill, batch_out in snapshots:
        if fill.rows == 0:
            assert batch_out == {}
            continue
        assert fill.t_len in (32, 64, 128, 256)
        assert [n for n, _ in EXPECTED_PLANES] == list(batch_out.keys())
        for name, dtype in EXPECTED_PLANES:
            assert batch_out[name].dtype == dtype, name
        rows = fill.rows
        assert batch_out["dora_indicators"].shape == (rows, 5)
        assert batch_out["history_event_kind"].shape == (rows, fill.t_len)
        assert batch_out["history_mask"].shape == (rows, fill.t_len)
        assert batch_out["legal_ids"].shape == (rows, 32)
        assert batch_out["legal_len"].shape == (rows,)


def test_two_passes_planes_identical(rust_extension: Any, clean_env: Path, tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    first, _, _ = _drain_all(inputs, 2)
    second, _, _ = _drain_all(inputs, 2)
    assert _snapshot_key(first) == _snapshot_key(second)


def test_open_pins_fail_closed(rust_extension: Any, clean_env: Path, tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="source_hash"):
        rust_stream.open_rust_plane_stream(
            [str(inputs)], 2, split="train", rules_hash="r", action_table_hash="a"
        )
    with pytest.raises(ValueError, match="rules_hash"):
        rust_stream.open_rust_plane_stream(
            [str(inputs)], 2, split="train", source_hash="s", action_table_hash="a"
        )
    with pytest.raises(ValueError, match="action_table_hash"):
        rust_stream.open_rust_plane_stream(
            [str(inputs)], 2, split="train", source_hash="s", rules_hash="r"
        )


def test_forged_pins_drain_identically(
    rust_extension: Any, clean_env: Path, tmp_path: Path
) -> None:
    """Well-formed pins are opaque: forged values drain identical planes.

    The closed-form walk ids pin to the compiled action-table digest inside
    Rust (fail-closed at open when that pin drifts); the passed digests bind
    nothing on this path, so a forged-but-non-empty triple must not change
    a single plane byte.
    """
    inputs = _write_inputs(tmp_path)
    ref, _, _ = _drain_all(inputs, 2)
    forged_pins = {
        "source_hash": "forged-source",
        "rules_hash": "forged-rules",
        "action_table_hash": "forged-action",
    }
    got: list[tuple[Any, dict[str, torch.Tensor]]] = []
    with rust_stream.open_rust_plane_stream(
        [str(inputs)],
        2,
        split="train",
        slot_rows=64,
        t_max=64,
        depth=2,
        device="cpu",
        **forged_pins,
    ) as stream:
        while True:
            batch_out, fill = stream.next_into_planes()
            got.append((fill, batch_out))
            if fill.rows == 0 and fill.games_consumed == 0:
                break
    assert _snapshot_key(got) == _snapshot_key(ref)


def test_small_slot_fails_closed_drains_nothing(
    rust_extension: Any, clean_env: Path, tmp_path: Path
) -> None:
    inputs = _write_inputs(tmp_path)
    stream = rust_stream.open_rust_plane_stream(
        [str(inputs)], 64, split="train", slot_rows=1, t_max=32, depth=2, device="cpu", **PINS
    )
    try:
        with pytest.raises(BufferError):
            stream.next_into_planes()
        stats = stream.stats()
        assert stats.rows_out == 0, "failed fill drained rows"
        assert stats.games_ok == 0, "failed fill moved games_ok"
        assert stats.games_quarantined == 0, "failed fill moved games_quarantined"
        assert stream.quarantines() == [], "failed fill listed quarantines"
    finally:
        stream.close()
    # Nothing drained by the failed call: a full-size pass still yields all rows.
    snapshots, stats, _ = _drain_all(inputs, 2)
    assert stats.rows_out == sum(fill.rows for fill, _ in snapshots if fill.rows > 0) > 0


def test_raw_small_plane_reports_failing_index(
    rust_extension: Any, clean_env: Path, tmp_path: Path
) -> None:
    """Too-small plane cap raises ``BufferError`` naming the short plane."""
    inputs = _write_inputs(tmp_path)
    handle = rust_extension.PyHydraStream.open([str(inputs)], 16, "train", *PINS.values())
    try:
        # Generous caps at T256 strides, except plane #4 (history kinds).
        strides = [
            34,
            34,
            20,
            16,
            2048,
            256,
            8,
            8,
            8,
            8,
            8,
            32,
            136,
            8,
            8,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            32,
            1,
            1,
        ]
        caps = [16 * s + 64 for s in strides]
        caps[4] = 1
        bufs = [bytearray(c) for c in caps]
        ptrs = [ctypes.addressof(ctypes.c_char.from_buffer(b)) for b in bufs]
        with pytest.raises(BufferError, match="plane 4"):
            handle.next_into(ptrs, caps)
        stats = handle.stats()
        assert stats.rows_out == 0, "failed fill drained rows"
        assert handle.quarantines() == [], "failed fill listed quarantines"
    finally:
        handle.close()


def test_open_args_fail_closed(rust_extension: Any, clean_env: Path, tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="batch"):
        rust_stream.open_rust_plane_stream([str(inputs)], 0, split="train", **PINS)
    with pytest.raises(ValueError, match="split"):
        rust_stream.open_rust_plane_stream([str(inputs)], 2, split="test", **PINS)
    with pytest.raises(ValueError, match="data_dirs"):
        rust_stream.open_rust_plane_stream([], 2, split="train", **PINS)
    with pytest.raises(ValueError, match="slot_rows"):
        rust_stream.open_rust_plane_stream([str(inputs)], 2, split="train", slot_rows=0, **PINS)
    with pytest.raises(ValueError, match="t_max"):
        rust_stream.open_rust_plane_stream([str(inputs)], 2, split="train", t_max=33, **PINS)
    with pytest.raises(ValueError, match="depth"):
        rust_stream.open_rust_plane_stream([str(inputs)], 2, split="train", depth=0, **PINS)
    with pytest.raises(ValueError, match="not a directory"):
        rust_stream.open_rust_plane_stream([str(tmp_path / "missing")], 2, split="train", **PINS)


def test_artifact_root_inside_raw_root_fail_closed(
    rust_extension: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _write_inputs(tmp_path)
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    monkeypatch.setenv("HYDRA2_ARTIFACT_ROOT", str(artifact))
    monkeypatch.setenv("HYDRA2_DATA_ROOT", str(artifact))
    with pytest.raises(ValueError, match="outside HYDRA2_ARTIFACT_ROOT"):
        rust_stream.open_rust_plane_stream([str(inputs)], 2, split="train", **PINS)
    monkeypatch.setenv("HYDRA2_DATA_ROOT", str(artifact / "nested"))
    with pytest.raises(ValueError, match="outside HYDRA2_ARTIFACT_ROOT"):
        rust_stream.open_rust_plane_stream([str(inputs)], 2, split="train", **PINS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs one CUDA device")
def test_cuda_depth2_ring_records_and_recycles(
    rust_extension: Any, clean_env: Path, tmp_path: Path
) -> None:
    """Depth-2 CUDA drain flows through record + slot-local wait.

    Structural overlap check on the serial lane: every call records the
    consume event after the transfers and orders the calling stream after
    it, so recycled slots carry transfer-timing samples from the third
    call on (first full ring turn). The quantitative
    ``sync_wait_p99 < h2d_ms_p50`` proof runs on the F2 primary bench.
    """
    inputs = _write_inputs(tmp_path)
    with rust_stream.open_rust_plane_stream(
        [str(inputs)], 1, split="train", slot_rows=64, t_max=64, depth=2, device="cuda", **PINS
    ) as stream:
        ncalls = 0
        total_rows = 0
        while True:
            batch_out, fill = stream.next_into_planes()
            ncalls += 1
            if fill.rows == 0 and fill.games_consumed == 0:
                break
            if fill.rows == 0:
                assert batch_out == {}, "zero-row fill must move no prefix"
                continue
            total_rows += fill.rows
            assert batch_out["legal_ids"].is_cuda
        sync_ms, h2d_ms = stream.timings()
        assert len(sync_ms) == ncalls, "reuse stall sampled once per call"
        assert len(h2d_ms) == max(0, ncalls - 2), "transfer sample per recycled slot"


def test_legal_planes_per_row_stride_not_soa(
    rust_extension: Any, clean_env: Path, tmp_path: Path
) -> None:
    """Legal offers unpack per 136B row stride (``[ids128][len8]`` per row).

    The fill writes one 136-byte ``[ids128][len8]`` block per row; reading
    the buffer as structure-of-arrays (all ids, then all lens) silently
    shifts every row past row 0 and fabricates lens. Every drained row must
    therefore carry a non-empty, in-range, sorted offer set containing its
    committed choice — the trainability invariant the model requires.
    """
    inputs = _write_inputs(tmp_path)
    snapshots, _, _ = _drain_all(inputs, 2)
    seen = 0
    for fill, batch_out in snapshots:
        if fill.rows == 0:
            continue
        ids = batch_out["legal_ids"].tolist()
        lens = batch_out["legal_len"].tolist()
        chosen = batch_out["chosen_action_id"].tolist()
        for i in range(fill.rows):
            offers = ids[i][: lens[i]]
            assert 0 < lens[i] <= 32, f"row {i} legal_len {lens[i]}"
            assert all(0 <= v < 6792 for v in offers), f"row {i} offers out of range"
            assert offers == sorted(offers), f"row {i} offers not sorted"
            assert chosen[i] in offers, f"row {i} choice {chosen[i]} not offered"
            seen += 1
    assert seen == 9, f"fixture drift: drained {seen} rows, want 9"
