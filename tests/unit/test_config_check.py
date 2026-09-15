"""WP-01 environment probes: config-check/pure-python probe helpers (CPU lane).

Covers the hermetic, GPU-free surface of ``hydra2.config_check`` /
``hydra2.probe`` / ``hydra2._probe_support``: stdlib sha helper shape,
parity-tolerance sanity, trainer-absence evidence shape, and the pyproject
pin reader. GPU/network probes (sm120, frozen install, full suite) stay in
the serial/GPU lane — this file never spawns subprocesses, never touches
CUDA, and uses fixed inputs only.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract_package("WP-01")


def test_probe_sha_of_matches_stdlib() -> None:
    """Local probe sha helper is exact (stdlib sha256, sha256: prefix)."""
    import hashlib

    from hydra2.probe import sha_of

    assert sha_of(b"") == "sha256:" + hashlib.sha256(b"").hexdigest()
    assert sha_of(b"hydra2") == "sha256:" + hashlib.sha256(b"hydra2").hexdigest()


def test_parity_tolerance_sane() -> None:
    """Frozen parity tolerances satisfy the config-check invariant.

    Guards the ``0 < rtol < 1e-3 and 0 <= atol <= rtol`` gate in
    config_check.main: a tolerance edit that widens rtol or inverts
    atol/rtol order must fail here, or GPU parity gates would pass on
    meaningless comparisons.
    """
    from hydra2.config import PARITY_ABS_TOL, PARITY_REL_TOL

    assert 0 < PARITY_REL_TOL < 1e-3
    assert 0 <= PARITY_ABS_TOL <= PARITY_REL_TOL


def test_trainer_absence_reports_fabric() -> None:
    """Trainer-absence probe returns (ok, evidence) naming lightning-fabric.

    Guards check_trainer_absence's evidence contract: the boolean must be a
    real bool and the evidence string must name the standalone runtime.
    A regression that swallows the check (always-True, empty evidence)
    fails here instead of silently blessing a Trainer install.
    """
    from hydra2._probe_support import check_trainer_absence

    ok, detail = check_trainer_absence()
    assert isinstance(ok, bool)
    assert "lightning-fabric" in detail
    assert "forbidden" in detail


def test_probe_trainer_absence_agrees_with_helper() -> None:
    """probe_trainer_absence never contradicts the shared helper.

    Guards the two-layer defense (metadata check + fresh-import check):
    when the helper reports forbidden packages present, the probe must
    also fail. A regression that decouples the two layers fails here.
    """
    from hydra2._probe_support import check_trainer_absence
    from hydra2.probe import probe_trainer_absence

    helper_ok, _ = check_trainer_absence()
    if not helper_ok:
        ok, _ = probe_trainer_absence()
        assert ok is False


def test_pypi_pins_reader_covers_declared_deps() -> None:
    """Pin reader parses the pixi dependency contract (non-empty, sane).

    Guards config_check._pypi_pins: the repo's own declared pins must
    include the locked runtime (torch, pytest) with non-empty versions.
    A regression that mis-parses pyproject (wrong table, stripped names)
    returns an empty/wrong map here instead of passing config-check
    against garbage.
    """
    from hydra2.config_check import _pypi_pins

    pins = _pypi_pins()
    assert isinstance(pins, dict) and len(pins) > 0
    for name in ("torch", "pytest"):
        assert name in pins, f"missing pin for {name}"
        assert isinstance(pins[name], str) and pins[name] != ""


def test_probe_manifest_round_trip_stable() -> None:
    """Environment manifest round-trips canonically (CPU-only, hermetic).

    Guards probe_environment_manifest_round_trip's core claim: two
    captures digest identically and the digest equals the canonical-bytes
    sha. A regression that leaks wall-clock nondeterminism into the
    manifest fails here (digests diverge between captures).
    """
    from hydra2.probe import probe_environment_manifest_round_trip

    ok, detail = probe_environment_manifest_round_trip()
    assert ok is True, detail
    assert "digest_stable=True" in detail
    assert "canonical_roundtrip=True" in detail
