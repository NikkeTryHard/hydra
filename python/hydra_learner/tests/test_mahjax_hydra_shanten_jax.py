from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp: Any = importlib.import_module("jax.numpy")

from hydra_learner.mahjax.shanten import hydra_discard_shanten_masks_jax
from hydra_learner.mahjax.shanten_bridge import exact_discard_shanten_masks, has_shanten_bridge


def _counts(tiles: tuple[int, ...]) -> list[int]:
    counts = [0] * 34
    for tile in tiles:
        counts[tile] += 1
    return counts


@pytest.mark.parametrize(
    "counts",
    [
        _counts((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10, 22)),
        _counts((0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 33)),
        _counts((0, 0, 1, 1, 9, 9, 10, 10, 18, 18, 19, 19, 27, 27)),
        _counts((0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 28, 28, 28)),
    ],
)
def test_jax_hydra_shanten_matches_rust_bridge(counts: list[int]) -> None:
    if not has_shanten_bridge():
        pytest.skip("hydra_raw_mjai_pyo3 extension is not built")

    rust_base, rust_non_increase, rust_decrease = exact_discard_shanten_masks(counts)
    jax_base, jax_non_increase, jax_decrease = hydra_discard_shanten_masks_jax(jnp.asarray(counts, dtype=jnp.int32))

    assert int(jax_base) == rust_base
    np.testing.assert_array_equal(np.asarray(jax_non_increase), np.asarray(rust_non_increase))
    np.testing.assert_array_equal(np.asarray(jax_decrease), np.asarray(rust_decrease))


def test_jitted_jax_hydra_shanten_matches_eager() -> None:
    counts = _counts((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10, 22))
    eager = hydra_discard_shanten_masks_jax(jnp.asarray(counts, dtype=jnp.int32))
    compiled = jax.jit(hydra_discard_shanten_masks_jax)(jnp.asarray(counts, dtype=jnp.int32))

    assert int(compiled[0]) == int(eager[0])
    np.testing.assert_array_equal(np.asarray(compiled[1]), np.asarray(eager[1]))
    np.testing.assert_array_equal(np.asarray(compiled[2]), np.asarray(eager[2]))
