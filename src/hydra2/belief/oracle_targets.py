"""WP-07B oracle targets — deterministic teacher targets from privileged rows.

Owns the materialized target record and the derivation helpers shared by
the store and join paths: belief/value targets from privileged labels
through the canonical utility manifest, and teacher-logit inversion.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from hydra2.contracts.utility import UtilityManifest


@dataclass(frozen=True, slots=True)
class OracleTarget:
    """Deterministic teacher target derived from privileged row."""

    decision_id: str
    wall_id: str
    # Belief target: distribution over hidden tile types (34-dim, sum=1)
    belief_target: tuple[float, ...]
    # Value target: 4-seat UtilityVector.values via utility()
    # (ranks -> rank_values -> values; e.g. (20,10,-10,-20) permuted,
    # zero-sum, NOT a distribution). Opaque join on decision_id only —
    # actor batch carries decision_id + observation_hash, never privileged.
    value_target: tuple[float, ...]
    # Event target: next event kind id (0..19) for belief model
    event_target: int
    # Teacher soft logits (pre-softmax, for KL distillation)
    teacher_belief_logits: tuple[float, ...]
    teacher_value_logits: tuple[float, ...]
    # Provenance
    split: str
    observation_hash: str


def _belief_target_from_privileged(
    privileged_label: dict[str, Any] | None,
    decision_id: str,
    *,
    allow_synthetic: bool = False,
) -> tuple[float, ...]:
    """Deterministic 34-dim belief target from privileged hidden tiles.

    Real path: ``hidden_tiles`` / ``hidden_tile_counts`` (34-list) or
    ``wait_tiles``. When no real signal is present the legacy deterministic
    hash fallback applies ONLY with ``allow_synthetic=True`` (synthetic-only
    opt-in, byte-identical to the pre-flag behavior); otherwise raises
    :class:`ContractError` (fail closed — missing labels never silently
    hash-synthesize on real paths).
    """
    if isinstance(privileged_label, dict):
        # Try to extract hidden counts if present
        _h_a: Any | None = privileged_label.get("hidden_tiles")
        _h_b: Any | None = privileged_label.get("hidden_tile_counts")
        hidden: Any | None = _h_a if _h_a is not None else _h_b
        if isinstance(hidden, list) and len(hidden) == 34:
            _raw_total: float = float(sum(hidden))  # type: ignore[unknown-argument-type]  # reason: Any from privileged dict; float() validates. Evidence: https://docs.python.org/3/library/functions.html#float
            total: float = _raw_total if _raw_total != 0.0 else 1.0
            return tuple(float(x) / total for x in hidden)  # type: ignore[unknown-argument-type]  # reason: Any element intentional; float() validates
        # Try wait tiles
        waits: Any | None = privileged_label.get("wait_tiles")
        if isinstance(waits, list) and len(waits) > 0:
            vec: list[float] = [0.0] * 34
            for t in waits:
                if isinstance(t, int) and 0 <= t < 34:
                    vec[t] += 1.0
            _vec_total: float = float(sum(vec))
            total = _vec_total if _vec_total != 0.0 else 1.0
            return tuple(v / total for v in vec)
    if not allow_synthetic:
        raise ContractError(
            f"belief target: missing privileged hidden tiles for {decision_id!r} "
            "(fail closed; synthetic opt-in via allow_synthetic=True)"
        )
    h = hashlib.sha256(decision_id.encode()).digest()
    raw = (
        [float(b) + 1.0 for b in h[:34]]
        if len(h) >= 34
        else [float(b & 0xFF) + 1.0 for b in (h * 3)[:34]]
    )
    total = sum(raw)
    return tuple(v / total for v in raw)


def _oracle_utility_manifest() -> UtilityManifest:
    """Canonical day-one utility manifest (same golden as models/model.py)."""
    from hydra2.contracts.rules import RULES_ID
    from hydra2.contracts.utility import (
        UTILITY_OBJECTIVE,
        UTILITY_TIE_POLICY,
        make_utility_manifest,
    )

    return make_utility_manifest(
        utility_id="expected_final_placement_tenhou_4p_hanchan_v1",
        schema_version="1.0.0",
        rules_id=RULES_ID,
        rules_hash="sha256:3042a493280224f533d831f371275b1c96585cf1db5a2e5fb86ec259f403286b",
        objective=UTILITY_OBJECTIVE,
        rank_values=(20.0, 10.0, -10.0, -20.0),
        tie_policy=UTILITY_TIE_POLICY,
        value_min=-100.0,
        value_max=100.0,
        zero_sum=True,
    )


def _value_from_ranks_via_utility(ranks_in: Any) -> tuple[float, ...] | None:
    """Map ranks permutation 1..4 through utility() to UtilityVector.values.

    Returns None when input is not a strict 1..4 permutation (caller falls
    through to legacy paths). Synthesizes a RawOutcome whose final_scores
    are consistent with the ranks (rank 1 -> 40000, 2 -> 30000, 3 -> 20000,
    4 -> 10000) so utility()'s ranks -> rank_values -> values mapping is
    exact; utility() itself remains the fixed point (never duplicated).
    """
    if not isinstance(ranks_in, (list, tuple)) or len(ranks_in) != 4:
        return None
    if any(isinstance(x, bool) for x in ranks_in) or not all(isinstance(x, int) for x in ranks_in):
        return None
    # Guards prove 4 ints (bool rejected), so int() would be identity.
    _ranks_int: list[int] = list(ranks_in)
    ranks = tuple(_ranks_int)
    if sorted(ranks) == [0, 1, 2, 3]:
        # Zero-based seat convention -> 1..4 for utility().
        ranks = tuple(r + 1 for r in ranks)
    if sorted(ranks) != [1, 2, 3, 4]:
        return None
    from hydra2.contracts.utility import RawOutcome, utility

    manifest = _oracle_utility_manifest()
    score_for_rank = {1: 40000, 2: 30000, 3: 20000, 4: 10000}
    final_scores = tuple(score_for_rank[r] for r in ranks)  # type: ignore[index]  # ranks validated permutation above
    point_deltas = tuple(s - 25000 for s in final_scores)
    outcome = RawOutcome(
        final_scores=final_scores,  # type: ignore[arg-type]  # validated quad above
        ranks=ranks,  # type: ignore[arg-type]  # validated permutation above
        point_deltas=point_deltas,  # type: ignore[arg-type]  # derived from validated scores
        settlements=(),
        rules_id=manifest.rules_id,
        rules_hash=manifest.rules_hash,
    )
    return tuple(utility(outcome, manifest).values)


def _value_target_from_privileged(
    privileged_label: dict[str, Any] | None,
    decision_id: str,
    *,
    allow_synthetic: bool = False,
) -> tuple[float, ...]:
    """Deterministic 4-dim value target from privileged ranks (utility scale).

    Real paths (in order): ``ranks`` / 4-list ``final_placement`` permutation
    via ``utility()``; explicit ``value_vector`` / ``utility_vector`` /
    ``placement`` 4-list; legacy single-int ``final_placement`` / ``rank``
    0..3 mapped through the utility manifest (0-based rank +1 -> 1..4 selects
    ``rank_values[rank]`` at the rank index — utility scale, never 0/1
    one-hot). When no real signal is present the legacy deterministic hash
    fallback applies ONLY with ``allow_synthetic=True`` (synthetic-only
    opt-in, byte-identical to the pre-flag behavior); otherwise raises
    :class:`ContractError` (fail closed).
    """
    if isinstance(privileged_label, dict):
        # Day-one authoritative path: ranks -> utility() -> UtilityVector.values.
        # Accepts "ranks" permutation 1..4, or "final_placement" as a 4-list
        # permutation 1..4 (distinct from legacy single-int 0..3 below).
        # Opaque join: actor side supplies decision_id only; privileged ranks
        # never enter the actor batch (firewall: validate_actor_batch_no_privileged).
        _ranks_candidate: Any | None = privileged_label.get("ranks")
        if _ranks_candidate is None:
            _fp: Any | None = privileged_label.get("final_placement")
            if isinstance(_fp, (list, tuple)) and len(_fp) == 4:
                _ranks_candidate = _fp
        _via_utility = (
            _value_from_ranks_via_utility(_ranks_candidate)
            if _ranks_candidate is not None
            else None
        )
        if _via_utility is not None:
            return _via_utility
        _v_a: Any | None = privileged_label.get("value_vector")
        _v_b: Any | None = privileged_label.get("utility_vector")
        _v_c: Any | None = privileged_label.get("placement")
        _v_tmp: Any | None = _v_a if _v_a is not None else _v_b
        v: Any | None = _v_tmp if _v_tmp is not None else _v_c
        if isinstance(v, list) and len(v) == 4:
            # Explicit 4-list value claim: strict validation against the
            # manifest (never silently passed through). Bool is not a number.
            _vals_list: list[float] = []
            for _i, _x in enumerate(v):
                if isinstance(_x, bool) or not isinstance(_x, (int, float)):
                    raise ContractError(
                        f"value target: entry[{_i}] must be a number for {decision_id!r}, "
                        f"got {_x!r}"
                    )
                if not math.isfinite(float(_x)):
                    raise ContractError(
                        f"value target: entry[{_i}] must be finite for {decision_id!r}"
                    )
                # _x narrowed to int | float here; float() keeps int case exact.
                _vals_list.append(float(_x))
            _vals = tuple(_vals_list)
            _manifest = _oracle_utility_manifest()
            # UtilityManifest bounds are float already; float() would be identity.
            _lo = _manifest.value_min
            _hi = _manifest.value_max
            for _i, _x in enumerate(_vals):
                if _x < _lo or _x > _hi:
                    raise ContractError(
                        f"value target: entry[{_i}]={_x!r} outside manifest bounds "
                        f"[{_lo}, {_hi}] for {decision_id!r}"
                    )
            if bool(getattr(_manifest, "zero_sum", False)):
                _total = math.fsum(_vals)
                if not math.isclose(_total, 0.0, rel_tol=0.0, abs_tol=1e-9):
                    raise ContractError(
                        "value target: zero-sum manifest requires values summing to 0 "
                        f"for {decision_id!r}, got sum {_total!r}"
                    )
            return _vals
        # Single placement rank (legacy shape; utility scale, never one-hot):
        # 0-based rank +1 -> 1..4 selects manifest.rank_values[rank], the same
        # entry utility() itself uses (rank_values[rank - 1]), stored at the
        # rank index preserving the legacy 4-vector shape. Full-permutation
        # labels remain preferred (exact utility() call above).
        _r_a: Any | None = privileged_label.get("final_placement")
        _r_b: Any | None = privileged_label.get("rank")
        rank: Any | None = _r_a if _r_a is not None else _r_b
        if isinstance(rank, int) and not isinstance(rank, bool) and 0 <= rank < 4:
            manifest = _oracle_utility_manifest()
            vec = [0.0] * 4
            vec[rank] = manifest.rank_values[rank]
            return tuple(vec)
    if not allow_synthetic:
        raise ContractError(
            f"value target: missing privileged ranks for {decision_id!r} "
            "(fail closed; synthetic opt-in via allow_synthetic=True)"
        )
    # Hash-fallback synthetic target — synthetic-only opt-in (allow_synthetic=True),
    # byte-identical to the pre-flag behavior; never mixed with real utility()
    # targets except under explicit opt-in.
    # Deterministic pseudo value from hash
    h = int(hashlib.sha256((decision_id + "_value").encode()).hexdigest()[:8], 16)
    # 4-seat softmax-like values
    scores = [((h >> (i * 4)) & 0xF) / 15.0 for i in range(4)]
    _score_sum: float = float(sum(scores))
    total: float = _score_sum if _score_sum != 0.0 else 1.0
    return tuple(s / total for s in scores)


def _teacher_logits_from_targets(
    belief_target: tuple[float, ...], value_target: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    import math as _math

    # Invert softmax with small epsilon: logits = log(p+eps)
    eps = 1e-6
    belief_logits = tuple(_math.log(max(p, eps)) for p in belief_target)
    value_logits = tuple(_math.log(max(p, eps)) for p in value_target)
    return belief_logits, value_logits


__all__ = [
    "OracleTarget",
    "_belief_target_from_privileged",
    "_oracle_utility_manifest",
    "_teacher_logits_from_targets",
    "_value_from_ranks_via_utility",
    "_value_target_from_privileged",
]
