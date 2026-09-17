"""SPEC 5.2 terminal-outcome and expected-final-placement utility contracts.

Raw scores/ranks/deltas survive training targets, backups, and logs;
:class:`UtilityVector` is derived strictly through a :class:`UtilityManifest`
that carries its own identity ``digest`` (RFC 8785 canonical bytes of the
manifest document minus the digest field, computed contract-locally because
contracts never import artifacts — SPEC §1).

Zero-sum is descriptive only and true only when the manifest proves it; no
code hard-codes a zero-sum vector (BUILD WP-02B).

Rust-judged fixed surface: ``utility()`` ranks/values and the exact-total
zero-sum check delegate first to the ``hydra2_replay_rs.canon_rng`` fixed
fns (``validate_ranks`` gate + ``utility_for_ranks_fixed`` indexing +
``exact_total_is_zero``); the Fraction oracle in this file remains the
fallback (ImportError-only) and the comparator (mismatch=raise, fail-closed).
Evidence: ranks/indexing agree both sides; m8 exact-total True both sides
via the in-feed exact stack big-int fallback (Fraction-equivalent; Overflow
defensive-only); pyfn probes live.
"""

from __future__ import annotations

import hmac
import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256 as _sha256
from types import MappingProxyType
from typing import Any

from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.contracts.common import (
    ContractError,
    DigestMismatchError,
    DigestText,
    RulesMismatchError,
    SchemaVersion,
    Seat,
    make_schema_version,
)
from hydra2.contracts.rules_canonical import canonical_contract_json_bytes

__all__ = [
    "UTILITY_OBJECTIVE",
    "UTILITY_TIE_POLICY",
    "RawOutcome",
    "SettlementFact",
    "UtilityManifest",
    "UtilityVector",
    "make_utility_manifest",
    "root_scalar",
    "utility",
    "utility_manifest_digest_document",
    "utility_manifest_from_payload",
    "utility_manifest_to_payload",
]

#: Primary objective: expected final placement (SPEC 5.2).
UTILITY_OBJECTIVE = "expected_final_placement"
#: Ties never reach utility: rules-resolved ranks are required up front.
UTILITY_TIE_POLICY = "use_rules_resolved_rank"

_MAX_SAFE_INTEGER = 2**53 - 1
_SCORE_LIMIT = 10**12


#: Injectable native backend for the fixed-point judge (tests monkeypatch
#: this; production leaves None so `_fixed_native()` imports the compiled
#: extension). Mirrors `_NATIVE_OVERRIDE` in `_rust_bridge.py` /
#: `_rust_columnar.py` / `_rust_search.py` and `_RING_NATIVE_OVERRIDE` in
#: `models/encoder.py`.
_FIXED_NATIVE_OVERRIDE: Any = None

#: i64 bounds for the bridge fixed surface (integer-only quads).
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1


def _fixed_native() -> Any | None:
    """Import the built ``canon_rng`` fixed-point surface once; None → oracle.

    ImportError-only fallback: a missing/unbuilt bridge means the Fraction
    oracle below decides (byte-identical). Any other bridge failure (missing
    submodule, reject, Overflow, mismatch) raises — never silent.
    """
    if _FIXED_NATIVE_OVERRIDE is not None:
        return _FIXED_NATIVE_OVERRIDE
    try:
        return importlib.import_module("hydra2_replay_rs").canon_rng
    except (ImportError, AttributeError):
        # Stale/shadowed .so (fixture-copied debug builds predate the fixed
        # surface): oracle decides, same as unbuilt bridge.
        return None


def _as_i64_quad(values: tuple[float, ...]) -> list[int] | None:
    """Integral i64 view of a 4-quad, else None (oracle decides alone).

    The bridge fixed surface is integer-only: non-integral or out-of-i64-range
    rank_values cannot cross, so the Fraction oracle decides those alone (no
    deletion, no behavior change).
    """
    out: list[int] = []
    for item in values:
        number = float(item)
        if not number.is_integer():
            return None
        intval = int(number)
        if intval < _I64_MIN or intval > _I64_MAX:
            return None
        out.append(intval)
    return out


def _judge_ranks_and_values(
    *,
    rank_values: tuple[float, float, float, float],
    ranks: tuple[int, int, int, int],
    values: tuple[float, ...],
) -> None:
    """Rust judge over the ranks gate + values indexing (mismatch=raise).

    ``validate_ranks`` gates the oracle-validated permutation and
    ``utility_for_ranks_fixed`` (mirrors ``rank_values[rank-1]`` indexing)
    recomputes the values; any divergence from the oracle ``values`` raises
    :class:`ContractError` (fail-closed). Evidence: ranks/indexing agree both
    sides; pyfn probes live. ImportError (or non-i64 rank_values) → the
    Fraction oracle decides alone.
    """
    native = _fixed_native()
    if native is None:
        return
    rank_list = [int(rank) for rank in ranks]
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
        rust_values = native.utility_for_ranks_fixed(fixed_values, rank_list)
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
    """Bridge-first exact zero-sum (Fraction oracle on ImportError only).

    ``canon_rng.exact_total_is_zero`` decides: integer-only, Fraction-exact,
    no float accumulation, no epsilon (m8 True both sides via the in-feed
    exact stack big-int fallback; Overflow defensive-only, unreachable for
    finite f64). Evidence: 4.0-4.7x faster than the Fraction loop with
    byte-identical bools on the zero/non-zero/exotic-exponent matrix. A
    missing/unbuilt bridge falls back to the exact Fraction loop
    (byte-identical); any other bridge failure raises — never silent.
    """
    native = _fixed_native()
    if native is not None:
        try:
            return bool(native.exact_total_is_zero([float(item) for item in values]))
        except ImportError:
            pass
    return _exact_total(values) == 0


def _judge_values_finite(values: tuple[float, ...]) -> None:
    """Rust finiteness judge for value vectors (mismatch=raise).

    ``exact_total_is_zero`` rejects any non-finite lane (NonFiniteValue)
    exactly like the oracle finiteness loop, so its reject arm cross-checks
    finiteness here; the zero-bool is discarded (a UtilityVector need not sum
    to zero — zero-sum is manifest-descriptive only). ImportError → the oracle
    decides alone.
    """
    native = _fixed_native()
    if native is None:
        return
    try:
        native.exact_total_is_zero([float(item) for item in values])
    except ImportError:
        return
    except Exception as exc:
        raise ContractError(f"utility values rejected by Rust finiteness judge: {exc}") from exc


def _require_int(value: int, *, name: str, minimum: int, maximum: int | None) -> int:
    # bool MUST NOT pass integer validation (bool subclasses int).
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "∞" if maximum is None else str(maximum)
        raise ContractError(f"{name}={value} outside [{minimum}, {upper}]")
    return value


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
            _validate_json_value(item, where=f"{where}[{i}]")  # pyrefly: ignore[unknown-argument-type]  # reason: isinstance branches narrow only the container; item stays object
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{where}: object keys must be strings")
            _validate_json_value(item, where=f"{where}.{key}")  # pyrefly: ignore[unknown-argument-type]  # reason: isinstance branches narrow only the container; item stays object
        return
    raise ContractError(f"{where}: type {type(value).__name__} is outside the JSON domain")


@dataclass(frozen=True, slots=True, kw_only=True)
class SettlementFact:
    """One atomic settlement movement inside a terminal hand.

    ``point_deltas`` is the net four-seat delta produced by this fact alone;
    ``detail`` must stay inside the canonical JSON domain so settlements can be
    serialized into identity artifacts verbatim.
    """

    kind: str
    from_seat: Seat | None
    to_seats: tuple[Seat, ...]
    point_deltas: tuple[int, int, int, int]
    detail: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _require_nonempty_str(self.kind, name="settlement kind"))
        payer = self.from_seat
        if payer is not None:
            payer = _bridge_contracts.make_seat(
                _require_int(payer, name="from_seat", minimum=0, maximum=3)
            )
        payees = self.to_seats
        if not isinstance(payees, (tuple, list)) or len(payees) == 0:
            raise ContractError("to_seats must be a non-empty tuple of seats")
        validated_payees = tuple(
            _bridge_contracts.make_seat(
                _require_int(seat, name=f"to_seats[{i}]", minimum=0, maximum=3)
            )
            for i, seat in enumerate(payees)
        )
        if len(set(validated_payees)) != len(validated_payees):
            raise ContractError("to_seats must not repeat seats")
        if payer is not None and payer in validated_payees:
            raise ContractError(f"from_seat {payer} must not appear among to_seats")
        object.__setattr__(self, "to_seats", validated_payees)
        object.__setattr__(
            self,
            "point_deltas",
            _validate_quad_ints(
                self.point_deltas, name="point_deltas", minimum=-_SCORE_LIMIT, maximum=_SCORE_LIMIT
            ),
        )
        if not isinstance(self.detail, Mapping):
            raise ContractError("detail must be a string-keyed mapping")
        _validate_json_value(self.detail, where="detail")
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))


def _validated_ranks(values: Sequence[int]) -> tuple[int, int, int, int]:
    ranks = _validate_quad_ints(values, name="ranks", minimum=1, maximum=4)
    if sorted(ranks) != [1, 2, 3, 4]:
        raise ContractError(
            f"ranks must be a strict permutation of 1..4 with no ties or gaps, got {tuple(ranks)}"
        )
    return ranks


@dataclass(frozen=True, slots=True, kw_only=True)
class RawOutcome:
    """Terminal raw outcome: four-seat vectors that survive everywhere."""

    final_scores: tuple[int, int, int, int]
    ranks: tuple[int, int, int, int]
    point_deltas: tuple[int, int, int, int]
    settlements: tuple[SettlementFact, ...]
    rules_id: str
    rules_hash: DigestText

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "final_scores",
            _validate_quad_ints(
                self.final_scores, name="final_scores", minimum=-_SCORE_LIMIT, maximum=_SCORE_LIMIT
            ),
        )
        object.__setattr__(self, "ranks", _validated_ranks(self.ranks))
        object.__setattr__(
            self,
            "point_deltas",
            _validate_quad_ints(
                self.point_deltas, name="point_deltas", minimum=-_SCORE_LIMIT, maximum=_SCORE_LIMIT
            ),
        )
        settlements = self.settlements
        if not isinstance(settlements, (tuple, list)):
            raise ContractError("settlements must be a tuple of SettlementFact")
        for i, fact in enumerate(settlements):
            if not isinstance(fact, SettlementFact):
                raise ContractError(
                    f"settlements[{i}] must be SettlementFact, got {type(fact).__name__}"
                )
        object.__setattr__(self, "settlements", tuple(settlements))
        object.__setattr__(self, "rules_id", _require_nonempty_str(self.rules_id, name="rules_id"))
        object.__setattr__(self, "rules_hash", _bridge_contracts.make_digest_text(self.rules_hash))


@dataclass(frozen=True, slots=True, kw_only=True)
class UtilityManifest:
    """Expected-final-placement conversion bound to one rules hash.

    ``digest`` is computed here from the canonical bytes of the manifest
    document excluding the digest itself, so the identity always binds content.
    """

    utility_id: str
    schema_version: SchemaVersion
    rules_id: str
    rules_hash: DigestText
    objective: str
    rank_values: tuple[float, float, float, float]
    tie_policy: str
    value_min: float
    value_max: float
    zero_sum: bool
    digest: DigestText

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "utility_id", _require_nonempty_str(self.utility_id, name="utility_id")
        )
        object.__setattr__(self, "schema_version", make_schema_version(self.schema_version))
        object.__setattr__(self, "rules_id", _require_nonempty_str(self.rules_id, name="rules_id"))
        object.__setattr__(self, "rules_hash", _bridge_contracts.make_digest_text(self.rules_hash))
        if self.objective != UTILITY_OBJECTIVE:
            raise ContractError(f"objective must be {UTILITY_OBJECTIVE!r}, got {self.objective!r}")
        raw_values = self.rank_values
        if not isinstance(raw_values, (tuple, list)) or len(raw_values) != 4:
            raise ContractError("rank_values must hold exactly 4 numbers indexed by rank 1..4")
        rank_values = tuple(
            _require_finite_float(item, name=f"rank_values[{i}]")
            for i, item in enumerate(raw_values)
        )
        object.__setattr__(self, "rank_values", rank_values)
        if self.tie_policy != UTILITY_TIE_POLICY:
            raise ContractError(
                f"tie_policy must be {UTILITY_TIE_POLICY!r}, got {self.tie_policy!r}"
            )
        value_min = _require_finite_float(self.value_min, name="value_min")
        value_max = _require_finite_float(self.value_max, name="value_max")
        if value_min > value_max:
            raise ContractError(f"value_min {value_min} must not exceed value_max {value_max}")
        for rank, item in enumerate(rank_values, start=1):
            if item < value_min or item > value_max:
                raise ContractError(
                    f"rank_values[{rank - 1}]={item} outside declared bounds "
                    f"[{value_min}, {value_max}]"
                )
        object.__setattr__(self, "value_min", value_min)
        object.__setattr__(self, "value_max", value_max)
        if not isinstance(self.zero_sum, bool):
            raise ContractError(f"zero_sum must be a bool, got {type(self.zero_sum).__name__}")
        if self.zero_sum and not _rank_values_sum_is_zero(rank_values):
            raise ContractError(
                "zero_sum=true requires the rank_values total to be exactly zero; "
                "zero-sum is never assumed"
            )
        computed = _bridge_contracts.make_digest_text(
            "sha256:"
            + _sha256(
                canonical_contract_json_bytes(utility_manifest_digest_document(self))
            ).hexdigest()
        )
        if not hmac.compare_digest(
            str(_bridge_contracts.make_digest_text(self.digest)), str(computed)
        ):
            raise DigestMismatchError(
                f"utility manifest digest mismatch: recorded {self.digest} != recomputed {computed}"
            )


def _exact_total(values: tuple[float, ...]) -> Fraction:
    """Exact sum over floats via Fraction (ImportError-only fallback core).

    Kept as the byte-identical fallback behind :func:`_rank_values_sum_is_zero`
    when the bridge is missing/unbuilt; the bridge decides whenever present.
    """
    total = Fraction(0)
    for item in values:
        fraction = Fraction(item)
        total += fraction
    return total


def utility_manifest_digest_document(
    source: UtilityManifest | Mapping[str, Any],
) -> dict[str, Any]:
    """Canonical identity document: every field except ``digest``."""

    def get(name: str) -> Any:
        if isinstance(source, Mapping):
            return source[name]
        return getattr(source, name)

    return {
        "objective": get("objective"),
        "rank_values": list(get("rank_values")),
        "rules_hash": get("rules_hash"),
        "rules_id": get("rules_id"),
        "schema_version": get("schema_version"),
        "tie_policy": get("tie_policy"),
        "utility_id": get("utility_id"),
        "value_max": get("value_max"),
        "value_min": get("value_min"),
        "zero_sum": get("zero_sum"),
    }


_MANIFEST_FIELD_NAMES = (
    "utility_id",
    "schema_version",
    "rules_id",
    "rules_hash",
    "objective",
    "rank_values",
    "tie_policy",
    "value_min",
    "value_max",
    "zero_sum",
)


def make_utility_manifest(**fields: Any) -> UtilityManifest:
    """Build a manifest whose identity digest is computed from its content."""
    missing = [name for name in _MANIFEST_FIELD_NAMES if name not in fields]
    if len(missing) > 0:
        raise ContractError(f"make_utility_manifest is missing fields {missing}")
    extras = sorted(set(fields) - set(_MANIFEST_FIELD_NAMES))
    if len(extras) > 0:
        raise ContractError(f"make_utility_manifest got undeclared fields {extras}")
    computed = _bridge_contracts.make_digest_text(
        "sha256:"
        + _sha256(
            canonical_contract_json_bytes(utility_manifest_digest_document(fields))
        ).hexdigest()
    )
    return UtilityManifest(digest=computed, **fields)


@dataclass(frozen=True, slots=True, kw_only=True)
class UtilityVector:
    """Per-seat placement utilities derived through a UtilityManifest."""

    values: tuple[float, float, float, float]
    utility_id: str
    utility_manifest_hash: DigestText
    rules_hash: DigestText

    def __post_init__(self) -> None:
        raw_values = self.values
        if not isinstance(raw_values, (tuple, list)) or len(raw_values) != 4:
            raise ContractError("values must hold exactly 4 floats, one per seat")
        values = tuple(
            _require_finite_float(item, name=f"values[{i}]") for i, item in enumerate(raw_values)
        )
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self, "utility_id", _require_nonempty_str(self.utility_id, name="utility_id")
        )
        object.__setattr__(
            self,
            "utility_manifest_hash",
            _bridge_contracts.make_digest_text(self.utility_manifest_hash),
        )
        object.__setattr__(self, "rules_hash", _bridge_contracts.make_digest_text(self.rules_hash))


def utility(outcome: RawOutcome, manifest: UtilityManifest) -> UtilityVector:
    """Map a raw terminal outcome onto placement utilities (SPEC 5.2).

    Rejects rules-id/rules-hash mismatches (:class:`RulesMismatchError`),
    unresolved/tied or non-permutation ranks, and any computed value outside
    the declared bounds (:class:`ContractError`). Sets
    ``utility_manifest_hash=manifest.digest``.

    Rust-first via the bridge ``canon_rng`` fixed fns (``validate_ranks`` gate
    + ``utility_for_ranks_fixed`` values; ``utility_fixed`` is canonical-points
    only, so caller manifests go through the indexed fn): the oracle mapping
    below still decides types/rules/bounds verbatim, then the Rust judge
    recomputes ranks+values and any mismatch raises (fail-closed, never
    silent); ImportError-only Fraction oracle fallback. Evidence:
    ranks/indexing agree both sides; pyfn probes live.
    """
    if not isinstance(outcome, RawOutcome):
        raise ContractError("outcome must be a RawOutcome")
    if not isinstance(manifest, UtilityManifest):
        raise ContractError("manifest must be a UtilityManifest")
    if outcome.rules_id != manifest.rules_id:
        raise RulesMismatchError(
            f"outcome rules_id {outcome.rules_id!r} != manifest rules_id {manifest.rules_id!r}"
        )
    if outcome.rules_hash != manifest.rules_hash:
        raise RulesMismatchError(
            f"outcome rules hash {outcome.rules_hash} does not match manifest rules hash "
            f"{manifest.rules_hash}"
        )
    try:
        ranks = _validated_ranks(outcome.ranks)
    except ContractError as exc:
        raise ContractError(
            f"outcome carries unresolved/tied or non-permutation ranks and has no utility: {exc}"
        ) from exc
    values = tuple(manifest.rank_values[rank - 1] for rank in ranks)
    for i, item in enumerate(values):
        if not math.isfinite(item):
            raise ContractError(f"computed utility for seat {i} is non-finite")
        if item < manifest.value_min or item > manifest.value_max:
            raise ContractError(
                f"computed utility {item} outside declared bounds "
                f"[{manifest.value_min}, {manifest.value_max}]"
            )
    _judge_ranks_and_values(rank_values=manifest.rank_values, ranks=ranks, values=values)
    first, second, third, fourth = values
    return UtilityVector(
        values=(first, second, third, fourth),
        utility_id=manifest.utility_id,
        utility_manifest_hash=manifest.digest,
        rules_hash=outcome.rules_hash,
    )


def root_scalar(value: UtilityVector, seat: Seat) -> float:
    """Acting-seat root scalar: vector index selection (SPEC 5.2).

    Rust-judged finiteness via ``exact_total_is_zero`` (reject arm only; the
    zero-bool is discarded — vectors need not sum to zero); ImportError-only
    oracle fallback, mismatch=raise. Evidence: m8 True both sides; pyfn probe
    live.
    """
    if not isinstance(value, UtilityVector):
        raise ContractError("value must be a UtilityVector")
    seat_index = _bridge_contracts.make_seat(_require_int(seat, name="seat", minimum=0, maximum=3))
    for i, item in enumerate(value.values):
        if not math.isfinite(item):
            raise ContractError(f"utility values[{i}] is non-finite")
    _judge_values_finite(value.values)
    return value.values[seat_index]


# ---------------------------------------------------------------------------
# Payload codec for UtilityManifest documents (identity artifacts).
# ---------------------------------------------------------------------------


def utility_manifest_to_payload(manifest: UtilityManifest) -> dict[str, Any]:
    payload = utility_manifest_digest_document(manifest)
    payload["digest"] = manifest.digest
    return payload


def utility_manifest_from_payload(payload: Mapping[str, Any]) -> UtilityManifest:
    """Rebuild a manifest from JSON and verify its recorded digest."""
    if not isinstance(payload, Mapping):
        raise ContractError("utility manifest payload must be a JSON object")
    required = {
        "utility_id",
        "schema_version",
        "rules_id",
        "rules_hash",
        "objective",
        "rank_values",
        "tie_policy",
        "value_min",
        "value_max",
        "zero_sum",
        "digest",
    }
    missing = sorted(required - set(payload))
    if len(missing) > 0:
        raise ContractError(f"utility manifest payload is missing fields {missing}")
    extras = sorted(set(payload) - required)
    if len(extras) > 0:
        raise ContractError(f"undeclared extra utility-manifest payload keys {extras}")
    # UtilityManifest.__post_init__ recomputes the digest from content and
    # raises DigestMismatchError when the recorded digest does not match.
    return UtilityManifest(**payload)
