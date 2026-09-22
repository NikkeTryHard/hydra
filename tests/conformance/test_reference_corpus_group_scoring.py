"""WP-04A reference corpus: scoring report fragment (07-10).

FIX-01 split home for the red-five/pao/abortive/exhaustive-draw group.
Replays the cases through a group-local worker runner (counterexamples +
fragment report under tmp_path) so the old 17-case rollup no longer
serialises all groups on one worker. Case logic lives in
test_reference_corpus_scoring (imported, never copied).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.conformance import test_reference_corpus_scoring as _cases
from tests.conformance.test_reference_corpus import _publish_group_fragment

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract_package("WP-04A")

GROUP_CASE_IDS: tuple[str, ...] = ("WP04A-07", "WP04A-08", "WP04A-09", "WP04A-10")


def test_wp04a_scoring_report_fragment(tmp_path: Path) -> None:
    """Replay the scoring group and publish its intersection-report fragment."""
    _publish_group_fragment(
        group="scoring",
        case_ids=GROUP_CASE_IDS,
        cases=(
            _cases.test_wp04a_07_red_five_scoring,
            _cases.test_wp04a_08_pao_liability_split_and_kazoe,
            _cases.test_wp04a_09_kyuushu_kyuuhai_abort,
            _cases.test_wp04a_10_exhaustive_draw_noten_split,
        ),
        tmp_path=tmp_path,
    )
