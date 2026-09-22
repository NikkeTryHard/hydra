"""Rust-judged utility fixed-point helpers (bridge-first, oracle fallback).

Single home for the `canon_rng` fixed judges plus the exact `Fraction`
fallback behind `_rank_values_sum_is_zero`. Importing `contracts.utility`
keeps dataclasses and digests; this file owns only judge plumbing with no
engine or training imports. Failure mode is fail-closed `ContractError` on
mismatch, `ImportError`-only oracle fallback, never silent.
"""

from __future__ import annotations

import importlib
from fractions import Fraction
from typing import Any

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError

#: Injectable native backend for the fixed-point judge (tests monkeypatch
#: this; production leaves None so `_fixed_native()` imports the compiled
#: extension).
_FIXED_NATIVE_OVERRIDE: Any = None

#: i64 bounds for the bridge fixed surface (integer-only quads).
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1


def _fixed_native() -> Any | None:
    """Import the built ``canon_rng`` fixed-point surface once; None → oracle."""
    if _FIXED_NATIVE_OVERRIDE is not None:
        return _FIXED_NATIVE_OVERRIDE
    try:
        return importlib.import_module("hydra2._native").canon_rng
    except (ImportError, AttributeError):
        return None


def _as_i64_quad(values: tuple[float, ...]) -> list[int] | None:
    """Integral i64 view of a 4-quad, else None (oracle decides alone)."""
    out: list[int] = []
    for item in values:
        number: float = item
        if not number.is_integer():
            return None
        intval = int(number)
        if intval < _I64_MIN or intval > _I64_MAX:
            return None
        out.append(intval)
    return out


def _exact_total(values: tuple[float, ...]) -> Fraction:
    """Exact sum over floats via Fraction (ImportError-only fallback core)."""
    total = Fraction(0)
    for item in values:
        fraction = Fraction(item)
        total += fraction
    return total


def _judge_ranks_and_values(
    *,
    rank_values: tuple[float, float, float, float],
    ranks: tuple[int, int, int, int],
    values: tuple[float, ...],
) -> None:
    """Rust judge over the ranks gate + values indexing (mismatch=raise)."""
    native = _fixed_native()
    if native is None:
        return
    rank_list: list[int] = list(ranks)
    try:
        native.validate_ranks(rank_list)
    except ImportError:
        return
    except Exception as exc:
        raise ContractError(f"utility ranks rejected by Rust validate_ranks gate: {exc}") from exc
    fixed_values = _as_i64_quad(tuple(rank_values))
    if fixed_values is None:
        return
    try:
        rust_values: list[int] = native.utility_for_ranks_fixed(fixed_values, rank_list)
    except ImportError:
        return
    except Exception as exc:
        raise ContractError(
            f"utility values rejected by Rust utility_for_ranks_fixed: {exc}"
        ) from exc
    if tuple(float(item) for item in rust_values) != tuple(values):
        raise ContractError(
            "utility Rust/Python mismatch: "
            f"rust {tuple(float(item) for item in rust_values)} != python {tuple(values)}"
        )


def _rank_values_sum_is_zero(values: tuple[float, ...]) -> bool:
    """Bridge-first exact zero-sum (Fraction oracle on ImportError only)."""
    native = _fixed_native()
    if native is not None:
        try:
            is_zero: bool = native.exact_total_is_zero(list(values))
            return is_zero
        except ImportError:
            pass
    return _exact_total(values) == 0


def _judge_values_finite(values: tuple[float, ...]) -> None:
    """Rust finiteness judge for value vectors (mismatch=raise)."""
    native = _fixed_native()
    if native is None:
        return
    try:
        native.exact_total_is_zero(list(values))
    except ImportError:
        return
    except Exception as exc:
        raise ContractError(f"utility values rejected by Rust finiteness judge: {exc}") from exc


def _judge_utility_values(
    *,
    rank_values: tuple[float, float, float, float],
    ranks: tuple[int, int, int, int],
    values: tuple[float, ...],
) -> None:
    """Rust judge over rank-to-score indexing (mismatch=raise)."""
    judge = getattr(_bridge_contracts, "utility_values_for_ranks", None)
    if judge is None:
        return
    try:
        rust_values: list[float] = judge(list(rank_values), list(ranks))
    except Exception as exc:
        raise ContractError(
            f"utility values rejected by Rust utility_values_for_ranks: {exc}"
        ) from exc
    if tuple(rust_values) != tuple(values):
        raise ContractError(
            f"utility Rust/Python mismatch: rust {tuple(rust_values)} != python {tuple(values)}"
        )


def _judge_root_scalar(*, values: tuple[float, ...], seat: int, scalar: float) -> None:
    """Rust judge over acting-seat index selection (mismatch=raise)."""
    judge = getattr(_bridge_contracts, "utility_root_scalar", None)
    if judge is None:
        return
    try:
        rust_scalar: float = judge(list(values), seat)
    except Exception as exc:
        raise ContractError(f"utility scalar rejected by Rust utility_root_scalar: {exc}") from exc
    if rust_scalar != scalar:
        raise ContractError(f"utility Rust/Python mismatch: rust {rust_scalar} != python {scalar}")
