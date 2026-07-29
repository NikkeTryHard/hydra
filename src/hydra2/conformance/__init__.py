"""Hydra2 reference conformance corpus (WP-04A).

Versioned Tenhou edge-case cases, the reference trace runner that replays
them through :class:`~hydra2.engines.riichienv.RiichiEnvExactSimulator`, and
the supported-rule intersection report. BUILD lines 397-420.
"""

from hydra2.conformance.runner import (
    CaseResult,
    ReferenceTraceRunner,
    TraceExpectation,
)

__all__ = ["CaseResult", "ReferenceTraceRunner", "TraceExpectation"]
