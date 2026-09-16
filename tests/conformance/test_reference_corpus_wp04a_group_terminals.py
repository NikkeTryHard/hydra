"""WP-04A reference corpus: terminals report fragment (12-14c).

FIX-01 split home for the abortive/match-end group (triple-ron abort, rank +
uma accounting, all-last branches). Replays the cases through a group-local
worker runner (counterexamples + fragment report under tmp_path) so the old
17-case rollup no longer serialises all groups on one worker. Case logic lives
in test_reference_corpus_terminals_wp04a (imported, never copied). The wave-C
disposition summary stays in that module with the cases it aggregates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.conformance import test_reference_corpus_terminals_wp04a as _cases
from tests.conformance.test_reference_corpus_wp04a import _publish_group_fragment

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract_package("WP-04A")

GROUP_CASE_IDS: tuple[str, ...] = (
    "WP04A-12",
    "WP04A-13",
    "WP04A-14a",
    "WP04A-14b",
    "WP04A-14c",
)


def test_wp04a_terminals_report_fragment(tmp_path: Path) -> None:
    """Replay the terminals group and publish its intersection-report fragment."""
    _publish_group_fragment(
        group="terminals",
        case_ids=GROUP_CASE_IDS,
        cases=(
            _cases.test_wp04a_12_sanchahou_triple_ron_abort,
            _cases.test_wp04a_13_rank_tie_break_and_uma_utility,
            _cases.test_wp04a_14a_all_last_dealer_tenpai_stop_yame,
            _cases.test_wp04a_14b_west_entry_sudden_death_expected_mismatch,
            _cases.test_wp04a_14c_tobi_score_injection_unavailable_blocked,
        ),
        tmp_path=tmp_path,
    )
