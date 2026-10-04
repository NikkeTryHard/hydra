"""WP-14 streaming replay expansion: GameRecord -> DecisionRow goldens.

Hand-built one-game MJAI log over the identity wall (0..135): five tsumogiri
discards (seats 0,1,2,3,0) with one engine claim window auto-passed
mechanically between them (no row, never synthesized). Golden pins decision
count, seats, and chosen ids; replay identity is checked across fresh runs.
CPU lane, fixed wall, no wall-clock seeds; throughput prints informationally.
"""

from __future__ import annotations

import hashlib
import time

import pytest

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError, IllegalActionError
from hydra2.data.replay_expand import (
    ReplayExpander,
    expand_game,
    expand_privileged_rows,
    iter_microbatches,
)
from tests.unit._replay_helpers import GAME_ID, OBJECT_ID, _golden_events, _golden_game

pytestmark = pytest.mark.contract_package("WP-14")


EXPECTED_SEATS = [0, 1, 2, 3, 0]
EXPECTED_CHOSEN = [192, 193, 194, 195, 196]


def test_golden_decision_count_chosen_ids_and_hashes_stable() -> None:
    game = _golden_game()
    rows = expand_game(game)

    assert len(rows) == 5
    assert [r.seat for r in rows] == EXPECTED_SEATS
    assert [r.chosen_action_id for r in rows] == EXPECTED_CHOSEN
    assert [r.decision_id for r in rows] == [f"{GAME_ID}:d{i:04d}" for i in range(5)]
    assert all(r.round_id == f"{GAME_ID}:h00" for r in rows)
    assert all(r.game_id == GAME_ID and r.split == "train" for r in rows)
    assert all(r.source_object_id == OBJECT_ID for r in rows)
    assert all(r.privileged_label_ref == r.decision_id for r in rows)
    for row in rows:
        identity = {
            key: value for key, value in row.actor_observation.items() if key != "observation_hash"
        }
        expected = "sha256:" + hashlib.sha256(canonical_bytes(identity)).hexdigest()
        assert row.observation_hash == row.actor_observation["observation_hash"] == expected
        mask = row.actor_observation["legal_mask"]
        assert isinstance(mask, list) and mask[row.chosen_action_id] is True
        dora = row.actor_observation["dora_indicators"]
        assert isinstance(dora, list) and len(dora) == 5

    # Deterministic resume: same game re-expanded on a fresh engine replays
    # the identical sequence (same seed material + same cursor == same rows).
    replayed = ReplayExpander().expand(game)
    assert [(r.decision_id, r.chosen_action_id, r.observation_hash) for r in replayed] == [
        (r.decision_id, r.chosen_action_id, r.observation_hash) for r in rows
    ]

    # Privileged twin: ranks from final scores, decision ids join 1:1.
    privileged = expand_privileged_rows(game)
    assert len(privileged) == len(rows)
    assert [p.decision_id for p in privileged] == [r.decision_id for r in rows]
    for entry in privileged:
        assert list(entry.privileged_label["ranks"]) == [1, 2, 3, 4]
        assert entry.privileged_label["split"] == "train"
        assert isinstance(entry.privileged_label["wall_id"], str)

    # Microbatch slicing without re-decode: slices cover the rows exactly.
    batches = list(iter_microbatches(rows, 2))
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [r.decision_id for batch in batches for r in batch] == [r.decision_id for r in rows]
    with pytest.raises(ContractError):
        list(iter_microbatches(rows, 0))


def test_unmapped_event_raises() -> None:
    events = _golden_events()
    events.insert(-1, {"type": "bogus_kind"})
    with pytest.raises(ContractError, match="unmapped"):
        expand_game(_golden_game(events))


def test_illegal_logged_action_raises() -> None:
    events = _golden_events()
    # Seat 0 holds 1m/5m/9m/4p plus the drawn red 5p: discarding 1s is illegal.
    events[3] = {"type": "dahai", "actor": 0, "pai": "1s", "tsumogiri": False}
    with pytest.raises(IllegalActionError):
        expand_game(_golden_game(events))


def test_reports_decisions_per_second_informational() -> None:
    game = _golden_game()
    start = time.perf_counter()
    rows = expand_game(game)
    elapsed = time.perf_counter() - start
    assert len(rows) == 5
    print(
        f"[replay_expand] {len(rows)} decisions in {elapsed * 1e3:.2f} ms "
        f"-> {len(rows) / max(elapsed, 1e-9):.1f} decisions/sec (informational, no assert)"
    )
