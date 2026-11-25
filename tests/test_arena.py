"""Wire-level checks for the native full-match arena."""

import struct
from array import array

import pytest
from hydra3._arena import Arena
from hydra3.model_schema import (
    ACTION_WIDTH,
    INDICATOR_COUNT,
    INDICATOR_FIRST,
    TILE_TOKENS,
    TOKEN_COUNT,
    TOKEN_WIDTH,
    VISIBLE_COUNT,
)

SEED = 0x5EED_0001


def _decode(batch):
    slots_buffer, seats_buffer, tokens_buffer, descriptors_buffer = batch
    count = len(seats_buffer)
    if count == 0:
        assert not slots_buffer and not tokens_buffer and not descriptors_buffer
        return [], [], bytearray(), 0, []
    assert len(slots_buffer) == count * 4
    assert len(tokens_buffer) == count * TOKEN_COUNT * TOKEN_WIDTH * 4
    stride = count * ACTION_WIDTH * 4
    assert descriptors_buffer and len(descriptors_buffer) % stride == 0
    width = len(descriptors_buffer) // stride
    assert width > 0
    slots = list(struct.unpack(f"<{count}i", slots_buffer))
    actions = struct.unpack(f"<{count * width * ACTION_WIDTH}i", descriptors_buffer)
    rows = [
        actions[index * width * ACTION_WIDTH : (index + 1) * width * ACTION_WIDTH]
        for index in range(count)
    ]
    return slots, list(seats_buffer), tokens_buffer, width, rows


def _first_legal_ids(batch):
    _, _, _, width, rows = _decode(batch)
    ids = []
    for row in rows:
        for action in range(width):
            candidate = row[action * ACTION_WIDTH : (action + 1) * ACTION_WIDTH]
            if candidate[0] != -1:
                ids.append(candidate[0])
                break
        else:
            raise AssertionError("row has no legal candidate")
    return ids


def test_empty_arena_returns_empty_wire_buffers():
    arena = Arena([])
    batch = arena.batch()
    assert _decode(batch) == ([], [], bytearray(), 0, [])
    assert arena.outcomes() == []
    arena.step(array("i", []))


def test_duplicate_seeds_and_zero_workers_are_rejected():
    with pytest.raises(ValueError):
        Arena([1, 1])
    with pytest.raises(ValueError):
        Arena([1], workers=0)


def test_invalid_ids_leave_the_staged_batch_usable():
    arena = Arena([SEED], workers=1)
    batch = arena.batch()
    ids = _first_legal_ids(batch)
    assert len(ids) == 1
    two_dimensional = memoryview(array("i", ids)).cast("B").cast("i", shape=[1, len(ids)])
    with pytest.raises(ValueError):
        arena.step(two_dimensional)
    for invalid in (
        array("i", [-1]),
        array("i", []),
        array("i", ids + ids),
        array("i", [10**9]),
    ):
        with pytest.raises(ValueError):
            arena.step(invalid)
    arena.step(array("i", ids))
    next_batch = arena.batch()
    slots, seats, _, width, _ = _decode(next_batch)
    assert len(slots) == len(seats)
    assert all(slot == 0 for slot in slots)
    if seats:
        assert width > 0


def test_opening_exposes_only_the_first_dora_indicator():
    _, _, tokens, _, _ = _decode(Arena([SEED], workers=1).batch())
    faces = struct.unpack(f"<{TOKEN_COUNT * TOKEN_WIDTH}f", tokens)
    indicators = [
        [faces[face * TOKEN_WIDTH + INDICATOR_FIRST + slot] for face in range(TILE_TOKENS)]
        for slot in range(INDICATOR_COUNT)
    ]
    assert [sum(slot) for slot in indicators] == [1, 0, 0, 0, 0]
    revealed_face = indicators[0].index(1)
    assert faces[revealed_face * TOKEN_WIDTH + VISIBLE_COUNT] == 1


def test_batches_keep_one_stable_row_per_scheduled_slot():
    arena = Arena([SEED, SEED ^ 0xA5A5_A5A5], workers=2)
    slots, seats, tokens, width, rows = _decode(arena.batch())
    assert slots == [0, 1]
    assert len(seats) == 2
    assert len(rows) == 2
    assert width >= 1
    assert len(tokens) == 2 * TOKEN_COUNT * TOKEN_WIDTH * 4
    for row in rows:
        assert any(row[action * ACTION_WIDTH] != -1 for action in range(width))


def test_same_seed_replays_byte_identical_matches_to_terminal():
    left = Arena([SEED], workers=1)
    right = Arena([SEED], workers=1)
    for _ in range(30_000):
        left_batch = left.batch()
        right_batch = right.batch()
        for left_buffer, right_buffer in zip(left_batch, right_batch, strict=True):
            assert left_buffer == right_buffer
        if not left_batch[1]:
            break
        ids = _first_legal_ids(left_batch)
        left.step(array("i", ids))
        right.step(array("i", ids))
    else:
        raise AssertionError("deterministic match did not terminate")

    outcomes = left.outcomes()
    assert outcomes == right.outcomes()
    assert len(outcomes) == 1
    slot, scores, ranks = outcomes[0]
    assert slot == 0
    assert len(scores) == 4
    assert sorted(ranks) == [1, 2, 3, 4]
