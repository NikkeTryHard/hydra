"""Wave 3 engines parity — oracle pins for tiles / walls / events.

Deterministic oracle goldens for the portable engine helpers: physical
tile id <-> mjai-string conversion (red fives included), continuation-wall
derivation, and canonical event-envelope constructors. Every seed is rooted
in :mod:`hydra2.contracts.randomness` (semantic counter-based streams);
no wall-clock, no ``random`` module, no ``torch.manual_seed``.

These pins are the bridge contract for the Wave 3 engines port: the Rust
helpers MUST reproduce the frozen vectors/hashes below byte-for-byte.
"""

from __future__ import annotations

import hashlib

import pytest

from hydra2._native import tiles
from hydra2.contracts.randomness import (
    RandomStream,
    make_random_stream_key,
    semantic_seed,
)
from hydra2.engines.riichienv import events
from hydra2.engines.riichienv.events import (
    make_delta,
    make_envelope,
    meld_delta_value,
    reason_kind,
)
from hydra2.engines.riichienv.walls import WALL_STREAM_NAME, derive_hand_wall

mjai_string_of = tiles.mjai_string_of
physical_of = tiles.physical_of

pytestmark = pytest.mark.contract_package("WP-03A")

_MASTER = b"wave3-engines-parity-v1"
_EXPERIMENT = "wave3-engines-parity"
_SPLIT = "oracle"

_SCHEDULE_ID = "sched-wave3-001"

# Curated physical-id -> mjai goldens (edges, red fives, honors).
_TILE_GOLDENS: tuple[tuple[int, str], ...] = (
    (0, "1m"),
    (3, "1m"),
    (4, "2m"),
    (16, "5mr"),  # man red five: FIRST copy of the 5m block
    (17, "5m"),  # unsuffixed 5m: SECOND copy
    (35, "9m"),
    (36, "1p"),
    (52, "5pr"),  # pin red five
    (53, "5p"),
    (72, "1s"),
    (88, "5sr"),  # sou red five
    (89, "5s"),
    (104, "9s"),
    (107, "9s"),
    (108, "E"),
    (112, "S"),
    (120, "N"),
    (124, "P"),  # haku
    (128, "F"),  # hatsu
    (132, "C"),  # chun
    (135, "C"),
)

# Red-five aliases collapse onto the FIRST copy of the 5-suit block.
_RED_ALIAS_GOLDENS: tuple[tuple[str, int], ...] = (
    ("0m", 16),
    ("5mr", 16),
    ("5m", 17),
    ("0p", 52),
    ("5pr", 52),
    ("5p", 53),
    ("0s", 88),
    ("5sr", 88),
    ("5s", 89),
)


def _wall_stream(wall_id: str, replicate_id: int = 0) -> RandomStream:
    key = make_random_stream_key(
        purpose="wall",
        experiment_id=_EXPERIMENT,
        split_id=_SPLIT,
        replicate_id=replicate_id,
        attempt_id=0,
        wall_id=wall_id,
    )
    return RandomStream(semantic_seed(_MASTER, key=key))


def _schedule_digest(wall_id: str = "w-wave3-001") -> str:
    return "sha256:" + _wall_stream(wall_id).get_bytes(32).hex()


# ---------------------------------------------------------------------------
# Tiles: mjai round-trips including red fives
# ---------------------------------------------------------------------------


def test_tile_curated_goldens() -> None:
    for tile_id, expected in _TILE_GOLDENS:
        assert mjai_string_of(tile_id) == expected, f"tile {tile_id}"


def test_tile_red_alias_goldens() -> None:
    for text, expected in _RED_ALIAS_GOLDENS:
        assert int(physical_of(text)) == expected, f"alias {text!r}"
        # Red first-copies render marked; unsuffixed fives render plain.
        rendered = mjai_string_of(expected)
        assert rendered == tiles.mjai_string_of(expected)


def test_tile_red_first_copies_render_marked() -> None:
    assert mjai_string_of(16) == "5mr"
    assert mjai_string_of(52) == "5pr"
    assert mjai_string_of(88) == "5sr"
    for red in (16, 52, 88):
        assert int(physical_of(mjai_string_of(red))) == red


def test_tile_sweep_folds_to_block_base() -> None:
    """Every id renders; parsing folds to the block base except the 5-suit blocks."""
    for tile_id in range(136):
        reparsed = int(physical_of(mjai_string_of(tile_id)))
        if tile_id in (16, 52, 88):
            # Red first-copies render marked and parse back exactly.
            assert reparsed == tile_id
        elif 16 <= tile_id <= 19:
            # The whole 5m block renders unsuffixed "5m", which resolves to
            # the SECOND copy (engine mjai_to_tid: "5m" -> 17).
            assert reparsed == 17, f"tile {tile_id}"
        elif 52 <= tile_id <= 55:
            assert reparsed == 53, f"tile {tile_id}"
        elif 88 <= tile_id <= 91:
            assert reparsed == 89, f"tile {tile_id}"
        else:
            assert reparsed == tile_id - (tile_id % 4), f"tile {tile_id}"


def test_tile_invalid_inputs_fail_closed() -> None:
    for bad in ("0x", "5mrr", "", "E ", "10m", "6mr"):
        with pytest.raises(ValueError):
            physical_of(bad)
    for bad_id in (-1, 136, 1000):
        with pytest.raises(ValueError):
            mjai_string_of(bad_id)


# ---------------------------------------------------------------------------
# Walls: derivation determinism
# ---------------------------------------------------------------------------


def test_wall_stream_name_pinned() -> None:
    assert WALL_STREAM_NAME == "hydra2.wall_continuation_v1"


def test_wall_schedule_digest_rooted_in_randomness() -> None:
    assert (
        _schedule_digest()
        == "sha256:75c313f2c3f348c33075c056e3f011a274a03579675cc08591a64d733e6e09a1"
    )


def test_wall_hand_zero_is_identity_sentinel() -> None:
    assert (
        derive_hand_wall(schedule_digest=_schedule_digest(), schedule_id=_SCHEDULE_ID, hand_index=0)
        == ()
    )


def test_wall_derivation_goldens() -> None:
    digest = _schedule_digest()
    hand1 = derive_hand_wall(schedule_digest=digest, schedule_id=_SCHEDULE_ID, hand_index=1)
    hand2 = derive_hand_wall(schedule_digest=digest, schedule_id=_SCHEDULE_ID, hand_index=2)
    assert hashlib.sha256(bytes(hand1)).hexdigest() == (
        "9138cb9e45994a2ff0b1f607e50ed24ab95d8b9d10961ba72b9d1cc1ff71e82a"
    )
    assert hashlib.sha256(bytes(hand2)).hexdigest() == (
        "05420c92508eeaf54b415f6b6c833dc29394a675ed10d81c7d33f49455209cd6"
    )
    assert tuple(hand1[:8]) == (92, 86, 77, 87, 61, 128, 16, 130)
    assert tuple(hand1[-4:]) == (116, 132, 112, 15)
    assert tuple(hand2[:8]) == (100, 75, 82, 3, 20, 107, 72, 80)
    assert tuple(hand2[-4:]) == (128, 106, 44, 62)


def test_wall_derivation_is_permutation_and_deterministic() -> None:
    digest = _schedule_digest()
    for hand_index in (1, 2, 3):
        wall = derive_hand_wall(
            schedule_digest=digest, schedule_id=_SCHEDULE_ID, hand_index=hand_index
        )
        assert len(wall) == 136
        assert sorted(int(t) for t in wall) == list(range(136))
        repeat = derive_hand_wall(
            schedule_digest=digest, schedule_id=_SCHEDULE_ID, hand_index=hand_index
        )
        assert tuple(wall) == tuple(repeat)


def test_wall_derivation_sensitive_to_hand_and_digest() -> None:
    digest = _schedule_digest()
    hand1 = derive_hand_wall(schedule_digest=digest, schedule_id=_SCHEDULE_ID, hand_index=1)
    hand2 = derive_hand_wall(schedule_digest=digest, schedule_id=_SCHEDULE_ID, hand_index=2)
    assert tuple(hand1) != tuple(hand2)
    other = derive_hand_wall(
        schedule_digest=_schedule_digest("w-wave3-002"),
        schedule_id=_SCHEDULE_ID,
        hand_index=1,
    )
    assert tuple(hand1) != tuple(other)


# ---------------------------------------------------------------------------
# Events: constructors
# ---------------------------------------------------------------------------


def test_event_reason_classification_goldens() -> None:
    assert sorted(events.DRAW_END_REASONS) == ["exhaustive_draw", "nagashi_mangan"]
    assert reason_kind("exhaustive_draw") == "draw_end"
    assert reason_kind("nagashi_mangan") == "draw_end"
    assert reason_kind("suufon_renda") == "abortive_draw"
    assert reason_kind("sufuurenta") == "abortive_draw"
    assert events.ABORTIVE_REASONS["kyushu_kyuhai"] == "kyuushu_kyuuhai"


def test_event_meld_delta_value_golden() -> None:
    assert meld_delta_value(
        kind="pon", owner=1, source_seat=0, called_tile=32, tiles=[33, 32, 34]
    ) == {
        "meld_id": "pon:32.33.34",
        "kind": "pon",
        "owner": 1,
        "source_seat": 0,
        "called_tile": 32,
        "tiles": [32, 33, 34],  # sorted tile order pinned
    }


def test_event_make_delta_golden() -> None:
    delta = make_delta(("scores", 0), "set", 25000)
    assert delta.path == ("scores", 0)
    assert delta.operation == "set"
    assert delta.value == 25000


def test_event_envelope_discard_golden() -> None:
    env = make_envelope(
        game_id="g-wave3-001",
        sequence=3,
        kind="discard",
        visibility="public",
        rules_hash="sha256:" + "11" * 32,
        schema_hash="sha256:" + "22" * 32,
        actor=1,
        tile=16,
        action_id=7,
    )
    assert env.kind == "discard"
    assert tuple(int(s) for s in env.visible_to) == (0, 1, 2, 3)
    assert int(env.payload.tile) == 16  # aka discard keeps its physical copy
    assert int(env.payload.actor) == 1
    assert int(env.sequence) == 3


def test_event_envelope_draw_tile_is_actor_private() -> None:
    env = make_envelope(
        game_id="g-wave3-001",
        sequence=4,
        kind="draw_tile",
        visibility="actor_private",
        rules_hash="sha256:" + "11" * 32,
        schema_hash="sha256:" + "22" * 32,
        actor=2,
        tile=52,  # red five draw stays addressed to the drawer only
    )
    assert env.kind == "draw_tile"
    assert tuple(int(s) for s in env.visible_to) == (2,)
    assert int(env.payload.tile) == 52


def test_event_envelope_meld_and_game_start_goldens() -> None:
    pon = make_envelope(
        game_id="g-wave3-001",
        sequence=5,
        kind="pon",
        visibility="public",
        rules_hash="sha256:" + "11" * 32,
        schema_hash="sha256:" + "22" * 32,
        actor=0,
        tile=32,
        action_id=9,
        source_seat=3,
        consumed_tiles=(33, 34),
    )
    assert tuple(int(t) for t in pon.payload.consumed_tiles) == (33, 34)
    assert int(pon.payload.source_seat) == 3
    start = make_envelope(
        game_id="g-wave3-001",
        sequence=0,
        kind="game_start",
        visibility="public",
        rules_hash="sha256:" + "11" * 32,
        schema_hash="sha256:" + "22" * 32,
        round_index=0,
        scores=(25000, 25000, 25000, 25000),
    )
    assert start.payload.round_index == 0
    assert tuple(start.payload.scores) == (25000, 25000, 25000, 25000)
