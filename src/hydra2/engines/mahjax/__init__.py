"""Hydra2 MahJax quarantine shell (BUILD WP-03C).

Public surface: the fail-closed shell, the qualification state machine, and
the environment-tuple capture used to bind WP-04C qualification tokens.
"""

from __future__ import annotations

from hydra2.engines.mahjax.capture import (
    FRAGMENT_ARTIFACT_NAME,
    DeviceFingerprint,
    MahJaxEnvironmentTuple,
    capture_mahjax_tuple,
    installed_origin_commit_id,
    relevant_xla_flags,
    verify_installed_origin,
    write_mahjax_environment_fragment,
)
from hydra2.engines.mahjax.quarantine import (
    ADAPTER_VERSION,
    OBSERVATION_MODE,
    AdapterState,
    QualificationToken,
    fabricate_test_only_token,
)
from hydra2.engines.mahjax.shell import MahJaxQuarantineShell

__all__ = [
    "ADAPTER_VERSION",
    "FRAGMENT_ARTIFACT_NAME",
    "OBSERVATION_MODE",
    "AdapterState",
    "DeviceFingerprint",
    "MahJaxEnvironmentTuple",
    "MahJaxQuarantineShell",
    "QualificationToken",
    "capture_mahjax_tuple",
    "fabricate_test_only_token",
    "installed_origin_commit_id",
    "relevant_xla_flags",
    "verify_installed_origin",
    "write_mahjax_environment_fragment",
]
