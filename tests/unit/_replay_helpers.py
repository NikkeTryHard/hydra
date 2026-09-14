"""Shared WP-14 replay golden fixture (test-only).

One hand-built MJAI game over the identity wall (tile ids 0..135 in order,
fixed so expansion is deterministic) shared by the replay expansion tests
and the Rust batch parity tests. A single definition keeps the golden
decision digests identical everywhere they are asserted.
"""

from __future__ import annotations

from hydra2.data.decode import GameRecord

GAME_ID = "replay-golden-01"
OBJECT_ID = "test-object-01"
WALL = tuple(range(136))


def _golden_events() -> list[dict[str, object]]:
    return [
        {"type": "start_game"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "F",
            "honba": 0,
            "kyoku": 1,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [
                ["1m", "1m", "1m", "1m", "5mr", "5m", "5m", "5m", "9m", "9m", "9m", "9m", "4p"],
                ["2m", "2m", "2m", "2m", "6m", "6m", "6m", "6m", "1p", "1p", "1p", "1p", "4p"],
                ["3m", "3m", "3m", "3m", "7m", "7m", "7m", "7m", "2p", "2p", "2p", "2p", "4p"],
                ["4m", "4m", "4m", "4m", "8m", "8m", "8m", "8m", "3p", "3p", "3p", "3p", "4p"],
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "5pr"},
        {"type": "dahai", "actor": 0, "pai": "5pr", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "5p"},
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "5p"},
        {"type": "dahai", "actor": 2, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "5p"},
        {"type": "dahai", "actor": 3, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "6p"},
        {"type": "dahai", "actor": 0, "pai": "6p", "tsumogiri": True},
        {"type": "end_game", "scores": [30000, 25000, 20000, 15000]},
    ]


def _golden_game(
    events: list[dict[str, object]] | None = None,
    wall_tiles: tuple[int, ...] | None = WALL,
) -> GameRecord:
    return GameRecord(
        game_id=GAME_ID,
        object_id=OBJECT_ID,
        packaged_object_id="test-packaged-01",
        events=tuple(events if events is not None else _golden_events()),
        raw_bytes_sha256="sha256:" + "0" * 64,
        wall_tiles=wall_tiles,
        source={"type": "start_game"},
    )
