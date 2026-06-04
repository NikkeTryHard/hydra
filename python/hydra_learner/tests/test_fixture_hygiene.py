from __future__ import annotations

from pathlib import Path

NORMAL_TEST_ROOT = Path(__file__).resolve().parent
FORBIDDEN_NORMAL_TEST_STRINGS = (
    "/home/cachybtw",
    "training/",
    "target/python-parity-fixture",
    "ppo_smoke_fixture",
)
ALLOWLIST = {Path(__file__).name}


def test_normal_python_tests_do_not_reference_local_paths() -> None:
    offenders = [
        f"{path.relative_to(NORMAL_TEST_ROOT)} contains {forbidden}"
        for path in sorted(NORMAL_TEST_ROOT.glob("test_*.py"))
        if path.name not in ALLOWLIST
        for text in [path.read_text(encoding="utf-8")]
        for forbidden in FORBIDDEN_NORMAL_TEST_STRINGS
        if forbidden in text
    ]

    assert offenders == []
