"""Qual gate pure leaves (bridge-first eligibility, reasons, summaries).

Single home for `_gate_eligible`, `_truncate_reason`, and `_summarize_gates`.
Bridge leaves decide first when built; pure-Python fallbacks are
byte-identical. Failure mode is fail-closed `ContractError` on bad bridge
values, never silent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from hydra2._native import search as _gates_bridge  # pyrefly: ignore[missing-import]
except ImportError:
    _gates_bridge = None  # type: ignore[assignment]

from hydra2.contracts.common import ContractError as ContractError

if TYPE_CHECKING:
    from hydra2.analysis.qual_gates import AnalysisGateRecord


def _gate_eligible(compute_only: bool, deterministic_ok: bool, privileged_leak: bool) -> bool:
    """Eligibility closed form, bridge-first (``qual_gate_eligible``)."""
    leaf: Any = getattr(_gates_bridge, "qual_gate_eligible", None)
    if leaf is None:
        return compute_only and deterministic_ok and not privileged_leak
    try:
        eligible: bool = leaf(compute_only, deterministic_ok, privileged_leak)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    return eligible


def _truncate_reason(exc: BaseException | str) -> str:
    """Error-reason truncation, bridge-first (``qual_gate_reason_truncate``)."""
    text = exc if isinstance(exc, str) else str(exc)
    leaf: Any = getattr(_gates_bridge, "qual_gate_reason_truncate", None)
    if leaf is None:
        return text[:240]
    try:
        truncated: str = leaf(text)
    except (ValueError, TypeError) as err:
        raise ContractError(str(err)) from err
    return truncated


def _summarize_gates(gates: list[AnalysisGateRecord]) -> dict[str, int]:
    """Report summary counts, bridge-first (``qual_gate_summary``)."""
    leaf: Any = getattr(_gates_bridge, "qual_gate_summary", None)
    if leaf is None:
        return {
            "total": len(gates),
            "eligible": sum(1 for g in gates if g.eligible),
            "ineligible": sum(1 for g in gates if not g.eligible),
            "compute_only_pass": sum(1 for g in gates if g.compute_only),
            "deterministic_pass": sum(1 for g in gates if g.deterministic_replay_ok),
        }
    try:
        counts: tuple[int, int, int, int, int] = leaf(
            [g.compute_only for g in gates],
            [g.deterministic_replay_ok for g in gates],
            [g.eligible for g in gates],
        )
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    total, eligible, ineligible, compute_only_pass, deterministic_pass = counts
    return {
        "total": total,
        "eligible": eligible,
        "ineligible": ineligible,
        "compute_only_pass": compute_only_pass,
        "deterministic_pass": deterministic_pass,
    }
