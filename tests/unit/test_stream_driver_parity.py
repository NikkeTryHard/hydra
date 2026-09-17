"""Stream driver parity — frozen oracle pins vs the Rust StreamDriver D4.

The future Rust feed driver (inversion backlog prize #2 / D4: owns spawn,
order, delivery over hydra-feed/hydra-shard/stream replay) MUST reproduce
the Python oracle's outputs bit-identically (parity FIRST per the inversion
backlog); throughput SECOND. Goldens below were frozen from the Python
oracle on its last green run: semantic seed derivation, manifest order,
ordered emission (game ids/offsets/hashes), take sequences, snapshot
scalars, restore round-trips, resume-seek continuation, cursor/stats
equality, and the actor-plane firewall projection.

Fixture: six hand-built golden games (alternating wall-less sim path /
identity-wall replay path) under ``tenhou/``-style stems pinned to the
train split by the semantic data seed. Every seed is rooted in
:mod:`hydra2.contracts.randomness` (semantic counter-based streams); no
wall-clock, no ``random`` module, no ``torch.manual_seed``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import zstandard as zstd

from hydra2.contracts.common import ContractError
from hydra2.contracts.randomness import RandomStream, make_random_stream_key, semantic_seed
from hydra2.data.stream import (
    GameStream,
    actor_payload,
    build_manifest,
    verify_no_privileged_leakage,
)
from hydra2.data.stream_read import StreamCursor
from hydra2.training import stream_train as driver

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract_package("WP-14")

_MASTER = b"stream-driver-parity-v1"
_EXPERIMENT = "stream-driver-parity"
_SPLIT = "oracle"
_RATIOS = {"train": 0.8, "validation": 0.2}

_SEED_HEX = "5b5a0368fd1205e732cb49bea9bf0cecd0d134c3e2a17b82fa6b0b134a1349ba"
_DATA_SEED = 11818523353586753573

_WALL = list(range(136))
_WALL_HASH = "sha256:87ef3e03a99fdd08632d1e74dda2c6287b549293c6d781c0693789f1095ae25c"

_STEMS = [
    "2024030100gm-00a9-0000-0000001",
    "2024030200gm-00a9-0000-0000002",
    "2024030600gm-00a9-0000-0000006",
    "2024030800gm-00a9-0000-0000008",
    "2024030900gm-00a9-0000-0000009",
    "2024031000gm-00a9-0000-0000010",
]

#: Manifest order: sha256-hex of the relative path (oracle-frozen).
_MANIFEST_ORDER = [
    "2024030900gm-00a9-0000-0000009.mjai.json.zst",
    "2024030800gm-00a9-0000-0000008.mjai.json.zst",
    "2024030100gm-00a9-0000-0000001.mjai.json.zst",
    "2024031000gm-00a9-0000-0000010.mjai.json.zst",
    "2024030200gm-00a9-0000-0000002.mjai.json.zst",
    "2024030600gm-00a9-0000-0000006.mjai.json.zst",
]

_TEHAIS = [
    ["1m", "1m", "1m", "1m", "5mr", "5m", "5m", "5m", "9m", "9m", "9m", "9m", "4p"],
    ["2m", "2m", "2m", "2m", "6m", "6m", "6m", "6m", "1p", "1p", "1p", "1p", "4p"],
    ["3m", "3m", "3m", "3m", "7m", "7m", "7m", "7m", "2p", "2p", "2p", "2p", "4p"],
    ["4m", "4m", "4m", "4m", "8m", "8m", "8m", "8m", "3p", "3p", "3p", "3p", "4p"],
]

#: (game_id, raw_bytes_sha256, wall_hash, validation_hash, raw_len); offsets all 0.
_EMISSION = [
    (
        "parity-game-4",
        "sha256:e8489095a18547ad7ed300bdf2a03e6ac0152880d148a5c6ed820d6a8a5155db",
        None,
        "sha256:97dc89284b66eaf8f445904438e724f1c232024a5a131b9ff6a8c269150f392b",
        937,
    ),
    (
        "parity-game-3",
        "sha256:5f4a81d2479799574598847b66d1854d679cbb1229b12a9411819776ba0f758a",
        _WALL_HASH,
        "sha256:248cbd1a1c69896c7cb3671b05fe1ac0be5eb2b599063949adc13447275b42f0",
        1517,
    ),
    (
        "parity-game-0",
        "sha256:751ba61a40fe618934b9ace50237a8f108f9719424631546845f4ceb337f77df",
        None,
        "sha256:7f7955de4fd28b5ae1258ef1f4487903f1e64ac7092726a4d7630190bbf6f70f",
        937,
    ),
    (
        "parity-game-5",
        "sha256:c562b0472df58388ecea755e0a4d8f9c5f13b531194f22af11431367a139ff40",
        _WALL_HASH,
        "sha256:1be35b44020366be946b813e3c4ced88c0ee0a5c18e605c213d9d5467ba40c70",
        1517,
    ),
    (
        "parity-game-1",
        "sha256:b399015a35ac7216c049ed925a6220d92ff2aec65fad401f0288fbe0c26478a2",
        _WALL_HASH,
        "sha256:0d68a37fed606301eed5b776727f91690e7dde2b14aededba46cfb1b10fc6cc4",
        1517,
    ),
    (
        "parity-game-2",
        "sha256:d59b1a6ec92de2be850ba61657f8af7bc8cb2a42cffc576cc9cad879bbce0f0e",
        None,
        "sha256:928c7e9a06b07d2f25dfc22e45cb114f8d40cf9df1843a84550d6694bb830acf",
        937,
    ),
]

_FULL_IDS = [
    "parity-game-4:d0000",
    "parity-game-4:d0001",
    "parity-game-4:d0002",
    "parity-game-3:d0000",
    "parity-game-3:d0001",
    "parity-game-3:d0002",
    "parity-game-0:d0000",
    "parity-game-0:d0001",
    "parity-game-0:d0002",
    "parity-game-5:d0000",
    "parity-game-5:d0001",
    "parity-game-5:d0002",
    "parity-game-1:d0000",
    "parity-game-1:d0001",
    "parity-game-1:d0002",
    "parity-game-2:d0000",
    "parity-game-2:d0001",
    "parity-game-2:d0002",
]

_CHOSEN = [192, 193, 193, 192, 193, 194, 192, 193, 193, 192, 193, 194, 192, 193, 194, 192, 193, 193]

_TAKE1 = [
    "parity-game-4:d0000",
    "parity-game-4:d0001",
    "parity-game-4:d0002",
    "parity-game-3:d0000",
    "parity-game-3:d0001",
]
_TAKE2 = [
    "parity-game-3:d0002",
    "parity-game-0:d0000",
    "parity-game-0:d0001",
    "parity-game-0:d0002",
    "parity-game-5:d0000",
]
_TAKE3 = [
    "parity-game-5:d0001",
    "parity-game-5:d0002",
    "parity-game-1:d0000",
    "parity-game-1:d0001",
    "parity-game-1:d0002",
]
_TAKE4 = ["parity-game-2:d0000", "parity-game-2:d0001", "parity-game-2:d0002"]

_HASH15 = "sha256:f8854a248e523d920222eb1f0735968a2ec5e44117b81add588777c570ebae64"
_HASH18 = "sha256:d968986d5da0a51547e2a26bf989fa56f5be088f65d34531fbd30fb49ed0992e"

_SEEK_AFTER_15 = {
    "byte_offset": 1517,
    "epoch": 0,
    "file_index": 4,
    "games_seen": 5,
    "shuffle_pos": 5,
}
_CURSOR_FINAL = {"byte_offset": 937, "epoch": 0, "file_index": 5, "games_seen": 6, "shuffle_pos": 6}


def _key() -> Any:
    return make_random_stream_key(
        purpose="training_shuffle",
        experiment_id=_EXPERIMENT,
        split_id=_SPLIT,
        replicate_id=0,
        attempt_id=0,
    )


def _data_seed() -> int:
    return int.from_bytes(RandomStream(semantic_seed(_MASTER, key=_key())).get_bytes(8), "big")


def _golden_events(game_id: str, *, wall: list[int] | None) -> list[dict[str, object]]:
    start: dict[str, object] = {"type": "start_game", "game_id": game_id}
    if wall is not None:
        start["wall"] = list(wall)
    return [
        start,
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "F",
            "honba": 0,
            "kyoku": 1,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": _TEHAIS,
        },
        {"type": "tsumo", "actor": 0, "pai": "5pr"},
        {"type": "dahai", "actor": 0, "pai": "5pr", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "5p"},
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "5p"},
        {"type": "dahai", "actor": 2, "pai": "5p", "tsumogiri": True},
        {"type": "end_game", "game_id": game_id, "scores": [30000, 25000, 20000, 15000]},
    ]


def _corpus(tmp_path: Path) -> Path:
    """Six golden games; even indices wall-less (sim path), odd identity-wall."""
    corpus = tmp_path / "corpus" / "tenhou"
    corpus.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(_STEMS):
        wall = _WALL if index % 2 == 1 else None
        raw = "\n".join(json.dumps(e) for e in _golden_events(f"parity-game-{index}", wall=wall))
        (corpus / f"{stem}.mjai.json.zst").write_bytes(
            zstd.ZstdCompressor().compress((raw + "\n").encode())
        )
    return corpus


def _stream(manifest: Any, seed: int, *, start: Any = None) -> GameStream:
    return GameStream(
        manifest,
        seed=seed,
        ratios=dict(_RATIOS),
        epoch=0,
        split="train",
        shuffle_buffer=0,
        start=start,
    )


def _dataset(manifest: Any, seed: int, *, start: Any = None) -> Any:
    return driver._StreamDataset(
        stream_factory=lambda: _stream(manifest, seed, start=start),
        num_actions=6792,
        feature_dim=64,
        seed=seed,
        drop_last=True,
        need_privileged=False,
        replay_backend="python",
    )


def test_semantic_seed_pinned() -> None:
    """The parity data seed derives from the semantic stream (no wall-clock)."""
    assert semantic_seed(_MASTER, key=_key()).hex() == _SEED_HEX
    assert _data_seed() == _DATA_SEED


def test_manifest_order_pinned(tmp_path: Path) -> None:
    """Manifest order is sha256-hex of the relative path (oracle-frozen)."""
    manifest = build_manifest(_corpus(tmp_path))
    assert [entry.path.name for entry in manifest.files] == _MANIFEST_ORDER


def test_ordered_emission_pinned(tmp_path: Path) -> None:
    """Ordered delivery: game ids, offsets, hashes, splits, event counts."""
    manifest = build_manifest(_corpus(tmp_path))
    games = list(_stream(manifest, _DATA_SEED))
    assert [g.game.game_id for g in games] == [row[0] for row in _EMISSION]
    for streamed, (_game_id, sha, wall_hash, validation_hash, raw_len) in zip(
        games, _EMISSION, strict=True
    ):
        assert streamed.offset == 0
        assert streamed.game.raw_bytes_sha256 == sha
        assert streamed.wall_hash == wall_hash
        assert streamed.validation_hash == validation_hash
        assert len(streamed.raw) == raw_len
        assert streamed.split == "train"
        assert len(streamed.game.events) == 9


def test_cursor_stats_and_resume_tail_pinned(tmp_path: Path) -> None:
    """Cursor frontier + stats equality; resume from the final cursor is empty."""
    manifest = build_manifest(_corpus(tmp_path))
    stream = _stream(manifest, _DATA_SEED)
    games = list(stream)
    assert len(games) == 6
    cursor = stream.cursor()
    assert cursor.to_dict() == {"seed": _DATA_SEED, **_CURSOR_FINAL}
    assert StreamCursor.from_dict(cursor.to_dict()) == cursor
    stats = stream.stats
    assert (stats.framed, stats.emitted, stats.quarantined) == (6, 6, 0)
    assert (stats.duplicates, stats.skipped_split, stats.waits) == (0, 0, 0)
    assert len(list(_stream(manifest, _DATA_SEED, start=cursor))) == 0


def test_dataset_rows_chosen_and_hash_pinned(tmp_path: Path) -> None:
    """Pull/expand/fill rows: decision ids, choices, kinds, counters, window hash."""
    manifest = build_manifest(_corpus(tmp_path))
    dataset = _dataset(manifest, _DATA_SEED)
    while dataset._pull_game():
        pass
    assert [row["decision_id"] for row in dataset._rows] == _FULL_IDS
    assert [int(row["chosen_action_id"]) for row in dataset._rows] == _CHOSEN
    assert {row.get("action_kind") for row in dataset._rows} == {"tsumogiri"}
    assert (dataset.replayed, dataset.sim_replayed) == (3, 3)
    assert dataset.expand_quarantined == 0
    assert [entry["key"] for entry in dataset._buffered_entries] == [row[1] for row in _EMISSION]
    assert [entry["rows"] for entry in dataset._buffered_entries] == [3] * 6
    assert dataset.buffered_row_hash() == _HASH18
    assert dataset.get_sampler_state() == {
        "offset": 0,
        "seed": _DATA_SEED,
        "total": 18,
        "epoch": 0,
        "dropped": 0,
    }


def test_take_sequences_pinned(tmp_path: Path) -> None:
    """Grouped takes consume the contiguous-prefix sequence 5/5/5."""
    manifest = build_manifest(_corpus(tmp_path))
    dataset = _dataset(manifest, _DATA_SEED)
    assert [r["decision_id"] for r in dataset._consume_microbatch(5)] == _TAKE1
    assert [r["decision_id"] for r in dataset._consume_microbatch(5)] == _TAKE2
    assert [r["decision_id"] for r in dataset._consume_microbatch(5)] == _TAKE3
    assert dataset.get_sampler_state() == {
        "offset": 15,
        "seed": _DATA_SEED,
        "total": 15,
        "epoch": 0,
        "dropped": 0,
    }
    assert dataset.buffered_row_hash() == _HASH15


def test_snapshot_scalars_pinned(tmp_path: Path) -> None:
    """Snapshot scalars pin the buffer frontier (entries re-homed, never dropped)."""
    manifest = build_manifest(_corpus(tmp_path))
    dataset = _dataset(manifest, _DATA_SEED)
    for _i in range(3):
        dataset._consume_microbatch(5)
    snap = dataset.buffer_snapshot()
    for name, want in (
        ("offset", 15),
        ("dropped", 0),
        ("epoch", 0),
        ("microbatches_in_epoch", 3),
        ("replayed", 3),
        ("sim_replayed", 2),
        ("expand_quarantined", 0),
        ("total_rows", 15),
        ("replay_backend", "python"),
    ):
        assert snap[name] == want, name
    assert snap["expand_quarantine_reasons"] == {}
    assert snap["row_hash"] == _HASH15
    assert [entry["rows"] for entry in snap["entries"]] == [3] * 5
    assert dataset.stream_cursor().to_dict() == {"seed": _DATA_SEED, **_SEEK_AFTER_15}


def test_restore_round_trip_pinned(tmp_path: Path) -> None:
    """Restore rebuilds rows verbatim (hash-checked before live state mutates)."""
    manifest = build_manifest(_corpus(tmp_path))
    dataset = _dataset(manifest, _DATA_SEED)
    for _i in range(3):
        dataset._consume_microbatch(5)
    snap = dataset.buffer_snapshot()
    fresh = _dataset(manifest, _DATA_SEED)
    fresh.restore_buffer(snap)
    assert [r["decision_id"] for r in fresh._rows] == [r["decision_id"] for r in dataset._rows]
    assert fresh.buffered_row_hash() == dataset.buffered_row_hash() == _HASH15
    assert fresh.get_sampler_state() == dataset.get_sampler_state()
    forged = json.loads(json.dumps(snap))
    forged["entries"][0]["key"] = "sha256:" + "0" * 64
    with pytest.raises(ContractError):
        _dataset(manifest, _DATA_SEED).restore_buffer(forged)


def test_resume_seek_continuation_pinned(tmp_path: Path) -> None:
    """Restore + seek to the recorded frontier continues the live sequence."""
    manifest = build_manifest(_corpus(tmp_path))
    live = _dataset(manifest, _DATA_SEED)
    for _i in range(3):
        live._consume_microbatch(5)
    snap = live.buffer_snapshot()
    seek = live.stream_cursor()
    fresh = _dataset(manifest, _DATA_SEED)
    fresh.restore_buffer(snap)
    # Resume-seek: rebind the live stream at the recorded frontier (driver
    # shape in run_stream_training); the gate below mirrors its seek check.
    fresh._stream = _stream(manifest, _DATA_SEED, start=seek)
    fresh._iter = iter(fresh._stream)
    assert fresh.stream_cursor() == seek
    assert [r["decision_id"] for r in live._consume_microbatch(3)] == _TAKE4
    assert [r["decision_id"] for r in fresh._consume_microbatch(3)] == _TAKE4
    assert [r["decision_id"] for r in fresh._rows] == [r["decision_id"] for r in live._rows]
    assert fresh.buffered_row_hash() == live.buffered_row_hash() == _HASH18
    assert fresh.get_sampler_state() == live.get_sampler_state()


def test_actor_firewall_pinned(tmp_path: Path) -> None:
    """Actor-plane projection carries identity + counts + hashes, never events."""
    manifest = build_manifest(_corpus(tmp_path))
    games = list(_stream(manifest, _DATA_SEED))
    assert actor_payload(games[0]) == {
        "event_count": 9,
        "game_id": "parity-game-4",
        "source_object_id": "2024030900gm-00a9-0000-0000009",
        "split": "train",
        "validation_hash": _EMISSION[0][3],
        "wall_hash": None,
    }
    assert actor_payload(games[1]) == {
        "event_count": 9,
        "game_id": "parity-game-3",
        "source_object_id": "2024030800gm-00a9-0000-0000008",
        "split": "train",
        "validation_hash": _EMISSION[1][3],
        "wall_hash": _WALL_HASH,
    }
    for game in games:
        verify_no_privileged_leakage(actor_payload(game))
    with pytest.raises(ContractError, match="privileged keys"):
        verify_no_privileged_leakage({**actor_payload(games[0]), "hidden_tiles": [1]})


def _driver_job(game: Any) -> tuple[bytes, str, str, list[int] | None]:
    """Driver push tuple for one streamed game (file-IO envelope stays Python)."""
    wall = game.game.wall_tiles
    return (
        bytes(game.raw),
        str(game.game.game_id),
        str(game.game.raw_bytes_sha256),
        [int(t) for t in wall] if wall is not None else None,
    )


def _driver_take_buffers(take: int) -> tuple[list[Any], list[int], list[int]]:
    """Flat T-max caller buffers for one driver take (torch collate stays Python).

    History planes are flat at T-max so the take fills with the committed
    ``t_len`` stride; content is ignored here (envelope+hash pins only).
    """
    import torch

    from hydra2.training.rust_stream import _HOT_PLANES

    widths = {"uint8": 1, "bool": 1, "int32": 4, "int64": 8}
    caps: list[int] = []
    for _name, cols, dtype in _HOT_PLANES:
        size = widths[dtype]
        if cols is None:
            caps.append(take * size)
        elif cols == "T":
            caps.append(take * 256 * size)
        elif cols == "legal":
            caps.append(take * 136)
        else:
            caps.append(take * int(cols) * size)
    assert len(caps) == 26
    bufs = [torch.empty(cap, dtype=torch.uint8) for cap in caps]
    return bufs, [int(b.data_ptr()) for b in bufs], [int(b.nbytes) for b in bufs]


def test_driver_push_receipt_and_snapshot_pinned(rust_extension: Any, tmp_path: Path) -> None:
    """Driver push: ordered delivery, snapshot scalars, seed passthrough."""
    _ = rust_extension
    manifest = build_manifest(_corpus(tmp_path))
    games = list(_stream(manifest, _DATA_SEED))
    drv = rust_extension.PyStreamDriver.open(_DATA_SEED, threads=0)
    try:
        out = drv.push_games([_driver_job(game) for game in games])
        assert (out.games_ok, out.games_quarantined, out.rows_added) == (6, 0, 18)
        snap = drv.snapshot()
        assert [(key, int(rows)) for key, rows in snap.entries] == [
            (row[1], 3) for row in _EMISSION
        ]
        assert snap.row_hash == _HASH18
        assert (int(snap.replayed), int(snap.sim_replayed)) == (3, 3)
        assert int(snap.quarantined) == 0
        assert list(snap.classes) == []
        assert int(snap.total_rows) == 18
        assert int(drv.seed()) == _DATA_SEED
    finally:
        drv.close()


def test_driver_takes_match_oracle_pinned(rust_extension: Any, tmp_path: Path) -> None:
    """Driver takes: 5/5/5/3 envelopes, frontier, hash, stats bit-exact."""
    _ = rust_extension
    manifest = build_manifest(_corpus(tmp_path))
    oracle = _dataset(manifest, _DATA_SEED)
    drv = rust_extension.PyStreamDriver.open(_DATA_SEED, threads=0)
    try:
        pushed = iter(_stream(manifest, _DATA_SEED))
        live = 0
        taken = 0

        def _fill_to(need: int) -> None:
            nonlocal live
            while live - taken < need:
                game = next(pushed, None)
                if game is None:
                    break
                live += int(drv.push_games([_driver_job(game)]).rows_added)

        for want in (_TAKE1, _TAKE2, _TAKE3):
            oracle_take = oracle._consume_microbatch(5)
            assert [row["decision_id"] for row in oracle_take] == want
            _fill_to(5)
            bufs, ptrs, caps = _driver_take_buffers(5)
            take = drv.next_microbatch(ptrs, caps, 5)
            assert list(take.ids) == want
            assert [int(c) for c in take.chosen] == [
                int(row["chosen_action_id"]) for row in oracle_take
            ]
            assert list(take.kinds) == [str(row.get("action_kind")) for row in oracle_take]
            taken += 5
            assert int(take.offset) == taken
            del bufs
        snap = drv.snapshot()
        assert (int(snap.offset), int(snap.dropped), int(snap.microbatches)) == (15, 0, 3)
        assert int(snap.total_rows) == 15
        assert snap.row_hash == _HASH15 == oracle.buffered_row_hash()
        assert (
            (int(snap.replayed), int(snap.sim_replayed))
            == (
                oracle.replayed,
                oracle.sim_replayed,
            )
            == (3, 2)
        )
        oracle_take4 = oracle._consume_microbatch(3)
        assert [row["decision_id"] for row in oracle_take4] == _TAKE4
        _fill_to(3)
        bufs, ptrs, caps = _driver_take_buffers(3)
        take4 = drv.next_microbatch(ptrs, caps, 3)
        assert list(take4.ids) == _TAKE4
        assert [int(c) for c in take4.chosen] == [
            int(row["chosen_action_id"]) for row in oracle_take4
        ]
        del bufs
        assert drv.snapshot().row_hash == _HASH18 == oracle.buffered_row_hash()
        stats = drv.stats()
        assert (int(stats.games_seen), int(stats.games_ok), int(stats.games_quarantined)) == (
            6,
            6,
            0,
        )
        assert (int(stats.rows_in), int(stats.rows_out), int(stats.microbatches)) == (
            18,
            18,
            4,
        )
        cursor = oracle.stream_cursor().to_dict()
        assert cursor == {"seed": _DATA_SEED, **_CURSOR_FINAL}
        assert int(stats.games_seen) == cursor["games_seen"] == 6
    finally:
        drv.close()
        oracle.close()


def test_driver_restore_repush_and_failclosed_pinned(rust_extension: Any, tmp_path: Path) -> None:
    """Driver restore: re-push in snapshot order rebuilds the hash; bad input fails."""
    _ = rust_extension
    manifest = build_manifest(_corpus(tmp_path))
    oracle = _dataset(manifest, _DATA_SEED)
    for _i in range(3):
        oracle._consume_microbatch(5)
    snap = oracle.buffer_snapshot()
    assert snap["row_hash"] == _HASH15
    by_key = {
        str(game.game.raw_bytes_sha256): _driver_job(game) for game in _stream(manifest, _DATA_SEED)
    }
    drv = rust_extension.PyStreamDriver.open(_DATA_SEED, threads=0)
    try:
        for entry in snap["entries"][:5]:
            drv.push_games([by_key[str(entry["key"])]])
        assert drv.snapshot().row_hash == _HASH15
        with pytest.raises(ValueError):
            drv.push_games([(b"x", "game", "", None)])
    finally:
        drv.close()
        oracle.close()
