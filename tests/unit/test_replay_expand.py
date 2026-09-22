"""WP-14 streaming replay expansion: GameRecord -> DecisionRow goldens.

Hand-built one-game MJAI log over the identity wall (0..135): five tsumogiri
discards (seats 0,1,2,3,0) with one engine claim window auto-passed
mechanically between them (no row, never synthesized). Golden pins decision
count, seats, chosen ids, and observation hashes; error paths fail closed.
CPU lane, fixed wall, no wall-clock seeds; throughput prints informationally.
"""

from __future__ import annotations

import time

import pytest

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
EXPECTED_OBS_HASHES = [
    "sha256:7514e75f61a8581257f62f04512f8b962d2d5ff757cd8b751f8fc2631767228a",
    "sha256:0197814741e6e4431d6a257fed55e50764145b688d6e2b6082ad36fbf60c01a6",
    "sha256:ea91c247bc7a3431c55a7ec149c5037f3ec054e89b4dd76faa5f16d30d10c06e",
    "sha256:e82e9eaa2efc6625ce5a9dcba0a4ac061aa958c33b3344c8bf959d9251a442c2",
    "sha256:a0d177335cf6aba60eb1e9ba0c0de44805b85e49c433ad2946e11d78206db3d8",
]


def test_golden_decision_count_chosen_ids_and_hashes_stable() -> None:
    game = _golden_game()
    rows = expand_game(game)

    assert len(rows) == 5
    assert [r.seat for r in rows] == EXPECTED_SEATS
    assert [r.chosen_action_id for r in rows] == EXPECTED_CHOSEN
    assert [r.observation_hash for r in rows] == EXPECTED_OBS_HASHES
    assert [r.decision_id for r in rows] == [f"{GAME_ID}:d{i:04d}" for i in range(5)]
    assert all(r.round_id == f"{GAME_ID}:h00" for r in rows)
    assert all(r.game_id == GAME_ID and r.split == "train" for r in rows)
    assert all(r.source_object_id == OBJECT_ID for r in rows)
    assert all(r.privileged_label_ref == r.decision_id for r in rows)
    for row in rows:
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
