"""Unit-tree marker registration (WP-03B).

Registers the ``contract_package`` marker so ``--strict-markers`` stays green
for marked unit tests regardless of which conftest owns ``--package``
selection. Intentionally defines NO options: package filtering lives in one
place only (tests/conftest.py per WP-03A cutover).
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
