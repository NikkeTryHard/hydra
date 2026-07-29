"""Integration-tree marker registration (WP-03B).

Mirrors tests/unit/conftest.py: registers the ``contract_package`` marker
only; package selection lives solely in tests/conftest.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "contract_package(wp_id): test belongs to this work package's gate",
    )
