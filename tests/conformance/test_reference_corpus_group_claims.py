"""WP-04A reference corpus: claim-window report fragment (03-06).

FIX-01 split home for the kuikae/furiten/multi-ron group. Replays the cases
through a group-local worker runner (counterexamples + fragment report under
tmp_path) so the old 17-case rollup no longer serialises all groups on one
worker. Case logic lives in test_reference_corpus_claims (imported,
never copied).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.conformance import test_reference_corpus_claims as _cases
from tests.conformance.test_reference_corpus import _publish_group_fragment

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract_package("WP-04A")

GROUP_CASE_IDS: tuple[str, ...] = (
    "WP04A-03",
    "WP04A-04a",
    "WP04A-04b",
    "WP04A-05",
    "WP04A-06",
)


def test_wp04a_claims_report_fragment(tmp_path: Path) -> None:
    """Replay the claims group and publish its intersection-report fragment."""
    _publish_group_fragment(
        group="claims",
        case_ids=GROUP_CASE_IDS,
        cases=(
            _cases.test_wp04a_03_kuikae_post_pon_same_meld_swap_barred,
            _cases.test_wp04a_04a_temp_furiten_clears_then_ron_lands,
            _cases.test_wp04a_04b_permanent_furiten_after_riichi_miss,
            _cases.test_wp04a_05_double_ron_priority_packets_upstream_first,
            _cases.test_wp04a_06_multi_ron_sticks_upstream_with_dealer_co_winner,
        ),
        tmp_path=tmp_path,
    )
