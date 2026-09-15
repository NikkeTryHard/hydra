"""Run-config strict-loading primitives: interpolation and field gates.

Owns ``${VAR}`` interpolation over the allowlisted environment set,
base+override deep merge, unknown-key rejection, and the scalar gate
helpers every section parser validates through. Failures raise
:class:`ContractError` fail-closed; nothing here reads files.
"""

from __future__ import annotations

import re
from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.training._rc_sections import _INTERPOLATION_RE as _INTERPOLATION_RE
from hydra2.training._rc_sections import INTERPOLATION_ALLOWLIST as INTERPOLATION_ALLOWLIST

__all__ = [
    "_DIGEST_RE",
    "_digest_pin_or_none",
    "_interpolate_string",
    "_interpolate_tree",
    "_positive_float_map",
    "_reject_unknown",
    "_require_bounded_float",
    "_require_bounded_int",
    "_require_loop_bool",
    "_require_nonempty_str",
    "_require_nonnegative_float",
    "_require_nonnegative_int",
    "_require_positive_int",
    "_weight_map",
    "deep_merge",
]

# ---------------------------------------------------------------------------
# Strict YAML loading: interpolation, base+override merge, unknown-key reject
# ---------------------------------------------------------------------------


def _interpolate_string(value: str, *, environ: Any, where: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in INTERPOLATION_ALLOWLIST:
            raise ContractError(
                f"{where}: ${{{name}}} is not allowlisted for interpolation; "
                f"allowed={sorted(INTERPOLATION_ALLOWLIST)}"
            )
        resolved = environ.get(name)
        if resolved is None or str(resolved) == "":
            raise ContractError(f"{where}: environment variable {name!r} is unset or empty")
        return str(resolved)

    return _INTERPOLATION_RE.sub(_replace, value)


def _interpolate_tree(node: Any, *, environ: Any, where: str) -> Any:
    if isinstance(node, str):
        return _interpolate_string(node, environ=environ, where=where)
    if isinstance(node, dict):
        return {
            key: _interpolate_tree(item, environ=environ, where=f"{where}.{key}")
            for key, item in node.items()
        }
    if isinstance(node, list):
        return [
            _interpolate_tree(item, environ=environ, where=f"{where}[{i}]")
            for i, item in enumerate(node)
        ]
    return node


def deep_merge(base: Any, override: Any) -> Any:
    """Deep-merge ``override`` onto ``base`` (base+override YAML composition).

    Mappings merge key-wise; every other type (including lists) is replaced
    wholesale by the override. Neither input is mutated.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged: dict[Any, Any] = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _reject_unknown(raw: Any, allowed: tuple[str, ...], *, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ContractError(f"{where} must be a mapping, got {type(raw).__name__}")
    unknown = sorted(k for k in raw if k not in allowed)
    if len(unknown) > 0:
        raise ContractError(f"{where} has unknown keys {unknown}; allowed={sorted(allowed)}")
    return dict(raw)


def _require_nonempty_str(raw: dict[str, Any], key: str, *, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise ContractError(f"{where}.{key} must be a non-empty string")
    return value


def _require_positive_int(raw: dict[str, Any], key: str, *, where: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{where}.{key} must be a positive int, got {value!r}")
    return value


def _require_bounded_int(raw: dict[str, Any], key: str, *, where: str, lo: int, hi: int) -> int:
    """Tuning knob in ``[lo, hi]`` (fail-closed; bounds are memory sanity, not tuning)."""
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not (lo <= value <= hi):
        raise ContractError(f"{where}.{key} must be an int in [{lo}, {hi}], got {value!r}")
    return value


def _require_nonnegative_int(raw: dict[str, Any], key: str, *, where: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{where}.{key} must be a non-negative int, got {value!r}")
    return value


def _require_nonnegative_float(raw: dict[str, Any], key: str, *, where: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}.{key} must be a non-negative number, got {value!r}")
    number = float(value)
    if not (number == number and number not in (float("inf"), float("-inf"))) or number < 0.0:
        raise ContractError(f"{where}.{key} must be finite and non-negative, got {value!r}")
    return number


def _weight_map(raw: dict[str, Any], key: str, *, where: str) -> dict[str, float] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ContractError(f"{where}.{key} must be a mapping or null, got {type(value).__name__}")
    out: dict[str, float] = {}
    for head, weight in value.items():
        if not isinstance(head, str) or head == "":
            raise ContractError(f"{where}.{key} keys must be non-empty strings")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not (float(weight) == float(weight))
            or float(weight) in (float("inf"), float("-inf"))
            or float(weight) < 0.0
        ):
            raise ContractError(f"{where}.{key}[{head!r}] must be finite non-negative")
        out[head] = float(weight)
    return out


def _require_bounded_float(
    raw: dict[str, Any],
    key: str,
    *,
    where: str,
    lo: float,
    hi: float,
    lo_open: bool = False,
    hi_open: bool = False,
) -> float:
    """Strict ``[lo, hi]`` float (``lo_open``/``hi_open`` select open ends)."""
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}.{key} must be a number, got {value!r}")
    number = float(value)
    if not (number == number and number not in (float("inf"), float("-inf"))):
        raise ContractError(f"{where}.{key} must be finite, got {value!r}")
    lo_ok = lo < number if lo_open else lo <= number
    hi_ok = number < hi if hi_open else number <= hi
    if not (lo_ok and hi_ok):
        bound = f"{'(' if lo_open else '['}{lo}, {hi}{')' if hi_open else ']'}"
        raise ContractError(f"{where}.{key} must lie in {bound}, got {value!r}")
    return number


def _positive_float_map(raw: dict[str, Any], key: str, *, where: str) -> dict[str, float] | None:
    """Optional ``{name: positive-finite-mult}`` map (null when absent)."""
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ContractError(f"{where}.{key} must be a mapping or null, got {type(value).__name__}")
    out: dict[str, float] = {}
    for head, mult in value.items():
        if not isinstance(head, str) or head == "":
            raise ContractError(f"{where}.{key} keys must be non-empty strings")
        if (
            isinstance(mult, bool)
            or not isinstance(mult, (int, float))
            or not (float(mult) == float(mult))
            or float(mult) in (float("inf"), float("-inf"))
            or float(mult) <= 0.0
        ):
            raise ContractError(f"{where}.{key}[{head!r}] must be positive and finite")
        out[head] = float(mult)
    return out


def _require_loop_bool(raw: dict[str, Any], key: str, *, default: bool) -> bool:
    """Strict bool knob (absent/null → ``default``; non-bool raises)."""
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ContractError(f"loop.{key} must be a bool, got {value!r}")
    return value


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _digest_pin_or_none(raw: dict[str, Any], key: str, *, where: str) -> str | None:
    """Optional ``sha256:<64 hex>`` provenance pin (null when absent)."""
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ContractError(f"{where}.{key} must be null or sha256:<64 hex>, got {value!r}")
    return value
