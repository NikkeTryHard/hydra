"""Unit-tree marker registration (WP-03B).

Registers the ``contract_package`` marker so ``--strict-markers`` stays green
for marked unit tests regardless of which conftest owns ``--package``
selection. Intentionally defines NO options: package filtering lives in one
place only (tests/conftest.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hydra2.data.parquet import write_actor_shards
from tests.unit._supervised_loop_helpers import NUM_ACTIONS_SMALL, _make_actor_rows

if TYPE_CHECKING:
    from pathlib import Path


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "contract_package(wp_id): test belongs to this work package's gate",
    )


@pytest.fixture(scope="session")
def rust_extension() -> Any:
    """Import the lane-built Rust extension (import-only, never builds).

    Lane precondition: run ``pixi run build-ext`` first. This fixture runs
    ZERO cargo and copies NO ``.so`` into tmp dirs. Single install target:
    the ``hydra2._native`` submodule (``build-ext``'s editable install);
    missing fails loud. Staleness stays fail-closed at the USE sites
    (``rust_batch._check_fresh`` over the ``build.json`` sidecar), never
    here. The ``sys.modules`` pop on teardown is kept so a late-session
    import never pins a shadowed copy.
    """
    import importlib
    import sys

    try:
        try:
            yield importlib.import_module("hydra2._native")
        except ImportError as exc:
            raise ImportError(
                "hydra2._native bridge not importable; run `pixi run build-ext` first"
            ) from exc
    finally:
        # Pop the module itself: a shadowed copy must never leak to later
        # importers in this worker — import succeeds but attributes miss.
        sys.modules.pop("hydra2._native", None)


@pytest.fixture(scope="session")
def actor_parquet_factory(tmp_path_factory):
    """Build each (num_rows, num_actions) synthetic variant ONCE per session.

    Shared dirs are READ-ONLY inputs: datasets verify + tensorize from them
    while SupervisedLoop / model / optimizer / checkpoint_dir stay per-test
    via tmp_path. Corrupt-input tests (privileged / dora-shim) keep building
    their own parquet — they assert the writer/loader rejects.
    """
    cache: dict[tuple[int, int], Path] = {}

    def get(num_rows: int = 20, num_actions: int = NUM_ACTIONS_SMALL) -> Path:
        key = (num_rows, num_actions)
        hit = cache.get(key)
        if hit is None:
            dest = (
                tmp_path_factory.mktemp("actor_parquet") / f"rows-{num_rows}-actions-{num_actions}"
            )
            rows = _make_actor_rows(num_rows=num_rows, num_actions=num_actions)
            write_actor_shards(
                destination=dest,
                rows=rows,
                dataset_hash="sha256:" + "e" * 64,
                split_manifest_hash="sha256:" + "f" * 64,
            )
            cache[key] = dest
            return dest
        return hit

    return get
