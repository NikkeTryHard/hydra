"""SPEC 18.2 resource telemetry schema and missing-telemetry policy.

Counters include all planner overhead named by the resource view. Missing
required telemetry invalidates a block according to the predeclared tolerance
and is NEVER imputed silently: :func:`telemetry_invalid_reason` names every
missing field, and :func:`block_missing_telemetry_report` aggregates the
violations per game so the exclusion surfaces in the evaluation report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from hydra2.artifacts.digest import validate_digest

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "REQUIRED_CORE_FIELDS",
    "ResourceTelemetry",
    "TelemetryTolerance",
    "block_missing_telemetry_report",
    "make_resource_telemetry",
    "telemetry_invalid_reason",
]


@dataclass(frozen=True, slots=True)
class ResourceTelemetry:
    """SPEC 18.2 telemetry row; field order matches the specification."""

    mode: str
    wall_id: str | None
    case_id: str | None
    candidate_spec_hash: str
    hardware_hash: str
    environment_hash: str
    cold_start: bool
    synchronized_elapsed_ms: float
    model_calls: int
    exact_transitions: int
    particles: int
    fallback_used: bool
    timeout: bool
    illegal_action: bool
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None
    host_peak_bytes: int | None
    energy_joules: float | None
    graph_breaks: int | None
    recompiles: int | None
    invalid_reason: str | None


TELEMETRY_FIELDS: tuple[str, ...] = tuple(item.name for item in fields(ResourceTelemetry))

#: Fields every telemetry row MUST carry regardless of resource view.
REQUIRED_CORE_FIELDS: tuple[str, ...] = (
    "mode",
    "candidate_spec_hash",
    "hardware_hash",
    "environment_hash",
    "cold_start",
    "synchronized_elapsed_ms",
    "model_calls",
    "exact_transitions",
    "particles",
    "fallback_used",
    "timeout",
    "illegal_action",
)

#: Extra fields a resource view turns from optional into required.
MODE_REQUIRED_EXTRAS: dict[str, tuple[str, ...]] = {
    "cuda_eager": ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"),
    "torch_compile": (
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "graph_breaks",
        "recompiles",
    ),
    "energy_metered": ("energy_joules",),
}


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _require_nonneg_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a nonnegative int")
    return value


def _require_nonneg_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise TypeError(f"{name} must be finite and >= 0")
    return number


def _require_optional_int(name: str, value: object) -> int | None:
    return None if value is None else _require_nonneg_int(name, value)


def _require_optional_float(name: str, value: object) -> float | None:
    return None if value is None else _require_nonneg_float(name, value)


def make_resource_telemetry(**kwargs: object) -> ResourceTelemetry:
    """Validate and construct a :class:`ResourceTelemetry` row."""
    unknown = set(kwargs) - set(TELEMETRY_FIELDS)
    if len(unknown) != 0:
        raise TypeError(f"unknown ResourceTelemetry fields: {sorted(unknown)}")
    missing = [name for name in TELEMETRY_FIELDS if name not in kwargs]
    if len(missing) != 0:
        raise TypeError(f"missing ResourceTelemetry fields: {missing}")

    mode = kwargs["mode"]
    if not isinstance(mode, str) or mode == "":
        raise TypeError("mode must be a nonempty str")
    for digest_name in ("candidate_spec_hash", "hardware_hash", "environment_hash"):
        validate_digest(kwargs[digest_name])  # type: ignore[arg-type]
    for identifier in ("wall_id", "case_id"):
        value = kwargs[identifier]
        if value is not None and (not isinstance(value, str) or value == ""):
            raise TypeError(f"{identifier} must be None or a nonempty str")
    invalid_reason = kwargs["invalid_reason"]
    if invalid_reason is not None and (not isinstance(invalid_reason, str) or invalid_reason == ""):
        raise TypeError("invalid_reason must be None or a nonempty str")

    return ResourceTelemetry(
        mode=mode,
        wall_id=kwargs["wall_id"],  # type: ignore[arg-type]
        case_id=kwargs["case_id"],  # type: ignore[arg-type]
        candidate_spec_hash=kwargs["candidate_spec_hash"],  # type: ignore[arg-type]
        hardware_hash=kwargs["hardware_hash"],  # type: ignore[arg-type]
        environment_hash=kwargs["environment_hash"],  # type: ignore[arg-type]
        cold_start=_require_bool("cold_start", kwargs["cold_start"]),
        synchronized_elapsed_ms=_require_nonneg_float(
            "synchronized_elapsed_ms", kwargs["synchronized_elapsed_ms"]
        ),
        model_calls=_require_nonneg_int("model_calls", kwargs["model_calls"]),
        exact_transitions=_require_nonneg_int("exact_transitions", kwargs["exact_transitions"]),
        particles=_require_nonneg_int("particles", kwargs["particles"]),
        fallback_used=_require_bool("fallback_used", kwargs["fallback_used"]),
        timeout=_require_bool("timeout", kwargs["timeout"]),
        illegal_action=_require_bool("illegal_action", kwargs["illegal_action"]),
        cuda_peak_allocated_bytes=_require_optional_int(
            "cuda_peak_allocated_bytes", kwargs["cuda_peak_allocated_bytes"]
        ),
        cuda_peak_reserved_bytes=_require_optional_int(
            "cuda_peak_reserved_bytes", kwargs["cuda_peak_reserved_bytes"]
        ),
        host_peak_bytes=_require_optional_int("host_peak_bytes", kwargs["host_peak_bytes"]),
        energy_joules=_require_optional_float("energy_joules", kwargs["energy_joules"]),
        graph_breaks=_require_optional_int("graph_breaks", kwargs["graph_breaks"]),
        recompiles=_require_optional_int("recompiles", kwargs["recompiles"]),
        invalid_reason=invalid_reason,
    )


@dataclass(frozen=True, slots=True)
class TelemetryTolerance:
    """Predeclared tolerance deciding which gaps invalidate a block.

    ``allow_missing`` may excuse only genuinely optional fields; excusing a
    required-core or mode-required field is rejected at construction so a
    silent gap cannot hide behind tolerance.
    """

    allow_missing: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        forbidden = set(self.allow_missing) & (set(REQUIRED_CORE_FIELDS) | {"invalid_reason"})
        if len(forbidden) != 0:
            raise ValueError(f"tolerance cannot excuse required fields: {sorted(forbidden)}")

    def required_for(self, mode: str) -> tuple[str, ...]:
        extras = MODE_REQUIRED_EXTRAS.get(mode, ())
        tolerated_extras = tuple(name for name in extras if name in self.allow_missing)
        if len(tolerated_extras) != 0:
            raise ValueError(f"tolerance cannot excuse mode-required fields: {tolerated_extras}")
        return REQUIRED_CORE_FIELDS + extras

def telemetry_invalid_reason(row: ResourceTelemetry, tolerance: TelemetryTolerance) -> str | None:
    """Return why this row invalidates its block, or None when usable."""
    if row.invalid_reason is not None:
        return f"row marked invalid: {row.invalid_reason}"
    missing = [name for name in tolerance.required_for(row.mode) if getattr(row, name) is None]
    if len(missing) != 0:
        return f"missing required telemetry (never imputed): {missing}"
    return None

def block_missing_telemetry_report(
    rows_by_game: Mapping[str, ResourceTelemetry], tolerance: TelemetryTolerance
) -> dict[str, str]:
    """Per-game invalidity report; empty mapping means fully usable rows."""
    report: dict[str, str] = {}
    for game_id in sorted(rows_by_game):
        reason = telemetry_invalid_reason(rows_by_game[game_id], tolerance)
        if reason is not None:
            report[game_id] = reason
    return report
