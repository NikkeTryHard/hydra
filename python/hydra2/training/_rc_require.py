"""Run-config strict-loading primitives: interpolation and field gates.

Owns ``${VAR}`` interpolation over the allowlisted environment set,
base+override deep merge, unknown-key rejection, and the scalar gate
helpers every section parser validates through. Failures raise
:class:`ContractError` fail-closed; nothing here reads files.

The eleven field gates delegate to ``hydra2._native.contracts`` (``rc_*``);
this module keeps the :class:`ContractError` shaping, the ``__all__`` names,
interpolation, merge, and the digest literal.
"""

from __future__ import annotations

import re
from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.training._rc_sections import _INTERPOLATION_RE as _INTERPOLATION_RE
from hydra2.training._rc_sections import INTERPOLATION_ALLOWLIST as INTERPOLATION_ALLOWLIST

try:
    from hydra2._native import contracts as _rc_bridge  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal, same text as call-site
    _rc_bridge = None  # type: ignore[assignment]


def _require_rc_bridge() -> Any:
    """Resolve the bridge, fail closed when the extension is not built."""
    if _rc_bridge is None or not hasattr(_rc_bridge, "rc_require_positive_int"):
        raise ImportError(
            "hydra2._native extension with contracts not importable; "
            "run `pixi run build-ext` to build the extension before use"
        )
    return _rc_bridge


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
        raw_value: object = environ.get(name)
        if raw_value is None or str(raw_value) == "":
            raise ContractError(f"{where}: environment variable {name!r} is unset or empty")
        return str(raw_value)

    return _INTERPOLATION_RE.sub(_replace, value)


def _interpolate_tree(node: Any, *, environ: Any, where: str) -> Any:
    if isinstance(node, str):
        return _interpolate_string(node, environ=environ, where=where)
    if isinstance(node, dict):
        mapping: dict[str, Any] = node
        return {
            key: _interpolate_tree(item, environ=environ, where=f"{where}.{key}")
            for key, item in mapping.items()
        }
    if isinstance(node, list):
        sequence: list[Any] = node
        return [
            _interpolate_tree(item, environ=environ, where=f"{where}[{i}]")
            for i, item in enumerate(sequence)
        ]
    return node


def deep_merge(base: Any, override: Any) -> Any:
    """Deep-merge ``override`` onto ``base`` (base+override YAML composition).

    Mappings merge key-wise; every other type (including lists) is replaced
    wholesale by the override. Neither input is mutated.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged: dict[Any, Any] = dict(base)
        override_map: dict[str, Any] = override
        for key, value in override_map.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _reject_unknown(raw: Any, allowed: tuple[str, ...], *, where: str) -> dict[str, Any]:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_reject_unknown`` decides."""
    try:
        return _require_rc_bridge().rc_reject_unknown(raw, list(allowed), where)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def _require_nonempty_str(raw: dict[str, Any], key: str, *, where: str) -> str:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_require_nonempty_str`` decides."""
    try:
        return _require_rc_bridge().rc_require_nonempty_str(raw, key, where)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def _require_positive_int(raw: dict[str, Any], key: str, *, where: str) -> int:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_require_positive_int`` decides."""
    try:
        return _require_rc_bridge().rc_require_positive_int(raw, key, where)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def _require_bounded_int(raw: dict[str, Any], key: str, *, where: str, lo: int, hi: int) -> int:
    """Tuning knob in ``[lo, hi]`` (fail-closed; bounds are memory sanity, not tuning)."""
    try:
        return _require_rc_bridge().rc_require_bounded_int(raw, key, where, lo, hi)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def _require_nonnegative_int(raw: dict[str, Any], key: str, *, where: str) -> int:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_require_nonnegative_int`` decides."""
    try:
        return _require_rc_bridge().rc_require_nonnegative_int(raw, key, where)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def _require_nonnegative_float(raw: dict[str, Any], key: str, *, where: str) -> float:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_require_nonnegative_float`` decides."""
    try:
        return _require_rc_bridge().rc_require_nonnegative_float(raw, key, where)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def _weight_map(raw: dict[str, Any], key: str, *, where: str) -> dict[str, float] | None:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_weight_map`` decides."""
    try:
        return _require_rc_bridge().rc_weight_map(raw, key, where)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


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
    try:
        return _require_rc_bridge().rc_require_bounded_float(
            raw, key, where, lo, hi, lo_open, hi_open
        )
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def _positive_float_map(raw: dict[str, Any], key: str, *, where: str) -> dict[str, float] | None:
    """Optional ``{name: positive-finite-mult}`` map (null when absent)."""
    try:
        return _require_rc_bridge().rc_positive_float_map(raw, key, where)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def _require_loop_bool(raw: dict[str, Any], key: str, *, default: bool) -> bool:
    """Strict bool knob (absent/null → ``default``; non-bool raises)."""
    try:
        return _require_rc_bridge().rc_require_loop_bool(raw, key, default)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _digest_pin_or_none(raw: dict[str, Any], key: str, *, where: str) -> str | None:
    """Optional ``sha256:<64 hex>`` provenance pin (null when absent)."""
    try:
        return _require_rc_bridge().rc_digest_pin_or_none(raw, key, where)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
