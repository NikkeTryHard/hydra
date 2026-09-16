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


@pytest.fixture(scope="session", autouse=True)
def _s8_rust_extension(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Build the Rust replay extension once per unit-tree session."""
    import importlib
    import importlib.machinery
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    crate = root / "tools" / "hydra2-replay-rs"
    env = {**os.environ, "PYO3_PYTHON": sys.executable}
    proc = subprocess.run(
        ["cargo", "build", "--offline", "-p", "hydra2-replay-rs"],
        cwd=crate,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"cargo build failed:\n{proc.stderr[-4000:]}"
    built = crate / "target" / "debug" / "libhydra2_replay_rs.so"
    assert built.is_file(), f"expected cdylib at {built}"
    ext_dir = tmp_path_factory.mktemp("hydra2_replay_rs")
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    shutil.copy(built, ext_dir / f"hydra2_replay_rs{suffix}")
    sys.path.insert(0, str(ext_dir))
    try:
        yield importlib.import_module("hydra2_replay_rs")
    finally:
        sys.path.remove(str(ext_dir))
        # Pop the module itself: the tmp copy may predate bridge surfaces
        # (packet_decode etc.) added later in the session. A stale entry in
        # sys.modules shadows the fresh installed .so for every later
        # importer in this worker — import succeeds but attributes miss.
        sys.modules.pop("hydra2_replay_rs", None)


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
