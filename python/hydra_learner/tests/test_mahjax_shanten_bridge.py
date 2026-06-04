from __future__ import annotations

import pytest

from hydra_learner.mahjax.shanten_bridge import (
    exact_discard_shanten_masks,
    exact_shanten_mask_planes,
    has_shanten_bridge,
)


def test_exact_discard_shanten_bridge_returns_34_wide_masks() -> None:
    if not has_shanten_bridge():
        pytest.skip("hydra_raw_mjai_pyo3 extension is not built")
    counts = [0] * 34
    for tile in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10, 22):
        counts[tile] += 1

    base, non_increase, decrease = exact_discard_shanten_masks(counts)

    assert isinstance(base, int)
    assert len(non_increase) == 34
    assert len(decrease) == 34
    assert all(isinstance(value, bool) for value in non_increase)
    assert all(isinstance(value, bool) for value in decrease)
    assert not non_increase[3]
    assert not decrease[3]
    assert any(non_increase)


def test_exact_shanten_mask_planes_match_bridge_masks() -> None:
    if not has_shanten_bridge():
        pytest.skip("hydra_raw_mjai_pyo3 extension is not built")
    counts = [0] * 34
    for tile in (0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 27, 28, 28, 28):
        counts[tile] += 1

    _, non_increase, decrease = exact_discard_shanten_masks(counts)
    ch9, ch10 = exact_shanten_mask_planes(counts)

    assert ch9 == [1.0 if value else 0.0 for value in non_increase]
    assert ch10 == [1.0 if value else 0.0 for value in decrease]
