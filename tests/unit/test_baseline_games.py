"""WP-05C reference-games pole (FIX-02a split).

Second half of ``test_baseline.py``: the 4-game reference evaluation
pole lives here so ``--dist loadscope`` can spread the file-pinned workers.
No behavior change (num_games=4 gate kept verbatim).
"""

from __future__ import annotations

import pytest

from hydra2.eval.baseline_eval import evaluate_reference_games

pytestmark = pytest.mark.contract_package("WP-05C")


def test_reference_games_zero_illegal_timeouts() -> None:
    """Complete reference games: zero illegal actions / timeouts."""
    summary = evaluate_reference_games(num_games=4, seed=0)
    assert summary["illegal_actions"] == 0
    assert summary["timeouts"] == 0
    assert summary["num_games"] == 4
    assert len(summary["game_hashes"]) == 4
