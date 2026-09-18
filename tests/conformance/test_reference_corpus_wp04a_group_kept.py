"""WP-04A reference corpus: kept-cases report fragment (01, 02, 11).

FIX-01 split home for the three cases defined in
test_reference_corpus_wp04a itself. Replays them through a group-local worker
runner (counterexamples + fragment report under tmp_path) so the old 17-case
rollup no longer serialises all groups on one worker. Case logic is imported,
never copied (module alias: importing the test functions by name would make
pytest collect them here a second time).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.conformance import test_reference_corpus_wp04a as _cases
from tests.conformance.test_reference_corpus_wp04a import _publish_group_fragment

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract_package("WP-04A")

GROUP_CASE_IDS: tuple[str, ...] = ("WP04A-01", "WP04A-02", "WP04A-11")


def test_wp04a_kept_report_fragment(tmp_path: Path) -> None:
    """Replay the kept group and publish its intersection-report fragment."""
    _publish_group_fragment(
        group="kept",
        case_ids=GROUP_CASE_IDS,
        cases=(
            _cases.test_wp04a_01_fifth_dora_and_kan_ura_timing,
            _cases.test_wp04a_02_chankan_and_rinshan_payout,
            _cases.test_wp04a_11_suufon_renda_documented_unsupported,
        ),
        tmp_path=tmp_path,
    )
