"""Shared stdlib-only contract validators (SPEC 2.1 helpers).

Single home for the `_require_*` integer, string, bool, enum, float, quad,
and canonical-JSON-domain checks duplicated across `contracts` modules.
Importing modules keep their public dataclasses and digests; this file owns
only pure validation with no bridge, artifact, or engine imports, so the
layer arrow stays one way. Failure mode is fail-closed `ContractError`
naming the field, never silent default.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from hydra2.contracts.common import ContractError

_MAX_SAFE_INTEGER = 2**53 - 1


def _require_int(value: int, *, name: str, minimum: int, maximum: int | None) -> int:
    # bool MUST NOT pass integer validation (bool subclasses int).
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "∞" if maximum is None else str(maximum)
        raise ContractError(f"{name}={value} outside [{minimum}, {upper}]")
    return value


def _require_str(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a str, got {type(value).__name__}")
    return value


def _require_bool(value: bool, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_enum(value: str, *, name: str, allowed: tuple[str, ...] | frozenset[str]) -> str:
    text = _require_str(value, name=name)
    if text not in allowed:
        raise ContractError(f"{name}={text!r} must be one of {sorted(allowed)}")
    return text


def _require_finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a finite number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{name} must be finite, got {number!r}")
    return number


def _require_nonempty_str(value: str, *, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ContractError(f"{name} must be a non-empty str")
    return value


def _validate_quad_ints(
    values: Sequence[int], *, name: str, minimum: int, maximum: int | None
) -> tuple[int, int, int, int]:
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        raise ContractError(f"{name} must be a sequence of exactly 4 ints")
    first, second, third, fourth = (
        _require_int(item, name=f"{name}[{i}]", minimum=minimum, maximum=maximum)
        for i, item in enumerate(values)
    )
    return (first, second, third, fourth)


_require_quad_ints = _validate_quad_ints


def _validate_json_value(value: Any, *, where: str) -> None:
    """Canonical JSON domain check (finite numbers only, string-keyed objects)."""
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ContractError(f"{where}: integer {value} exceeds the IEEE 754 double-safe range")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{where}: non-finite number {value!r}")
        return
    if isinstance(value, str):
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _validate_json_value(item, where=f"{where}[{i}]")  # pyrefly: ignore[unknown-argument-type] # JSON domain is honestly Any; recursion validates
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{where}: object keys must be strings")
            _validate_json_value(item, where=f"{where}.{key}")  # pyrefly: ignore[unknown-argument-type] # JSON domain is honestly Any; recursion validates
        return
    raise ContractError(f"{where}: type {type(value).__name__} is outside the JSON domain")
