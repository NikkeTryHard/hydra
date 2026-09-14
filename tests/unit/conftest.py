"""Unit-tree marker registration (WP-03B).

Registers the ``contract_package`` marker so ``--strict-markers`` stays green
for marked unit tests regardless of which conftest owns ``--package``
selection. Intentionally defines NO options: package filtering lives in one
place only (tests/conftest.py).
"""

from __future__ import annotations

from typing import Any

import pytest


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
