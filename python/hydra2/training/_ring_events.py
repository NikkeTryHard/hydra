"""Pinned-ring event and percentile helpers (perf telemetry only).

Single home for `_CpuEvent` plus `_percentile` used by `PinnedRing.stats`.
No CUDA, no torch tensors here; production CUDA never uses `_CpuEvent`.
Failure mode is fail-closed `ContractError` upstream, never silent fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def _percentile(xs: Sequence[float], q: float) -> float | None:
    if len(xs) == 0:
        return None
    ordered = sorted(xs)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


class _CpuEvent:
    """Host-side event stand-in keeping the record/synchronize call order."""

    def __init__(self, label: str, log: list[str] | None = None) -> None:
        self._label = label
        self._log = log

    def record(self, stream: Any = None) -> None:
        del stream
        if self._log is not None:
            self._log.append(f"record:{self._label}")

    def synchronize(self) -> None:
        if self._log is not None:
            self._log.append(f"synchronize:{self._label}")

    def elapsed_time(self, other: Any) -> float:
        del other
        return 0.0

    def query(self) -> bool:
        return True
