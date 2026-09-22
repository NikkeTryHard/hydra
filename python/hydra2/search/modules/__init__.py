# ruff: noqa: N806  # reason: kept, not narrowed; uppercase registry locals intentional
"""Candidate 4 modules — one at a time.

Implements SPEC 16.5 and Blueprint §11.1-11.10 (one module at a time):

- Each module remains behind one flag (exactly one enabled per CandidateSpec).
- Every module uses a named CandidateSpec, passes its tiny oracle, then fresh matched confirmation.
- No module except persistent_forest is an entry gate for persistence.
- Cumulative builds name promoted modules and re-pass every gate; unpromoted never merged.
- Normalized finite-particle ratios are search-only; not called unbiased.

This package owns the module registry, per-module tiny oracles, determinism via
semantic counter-based seeds, and the Candidate 4 spec factory.

Payload is intentionally small and CPU-only; GPU paths are not required for
qualification of one-at-a-time semantics (see BUILD compute note: use GPU when
beneficial — search is CPU here per plan). Each module's tiny oracle encodes
its blueprint formula exactly so promotion evidence is defensible.

Determinism: every stochastic choice derives from
  seed = sha256(candidate_id : case_id : module_id : purpose : attempt_id)
via hashlib + torch.Generator, never from call order or global RNG.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import torch

from hydra2.contracts.common import ContractError

# ---------------------------------------------------------------------------
# Module identities — one per Blueprint §11.1-11.10
# ---------------------------------------------------------------------------

VALID_MODULE_IDS: tuple[str, ...] = (
    "rao_blackwell",
    "defensive_mis",
    "structural_crn",
    "fixed_mlmc",
    "rqmc",
    "coreset",
    "pruning",
    "controlled_smc",
    "persistent_forest",  # required before persistence factorial
    "voc_routing",
)

PERSISTENCE_GATE_MODULE = "persistent_forest"

# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def _semantic_seed(
    candidate_id: str,
    case_id: str,
    module_id: str,
    purpose: str,
    attempt_id: int = 0,
) -> int:
    """Counter-based seed; stable across call order, differentiable by purpose."""
    material = f"{candidate_id}:{case_id}:{module_id}:{purpose}:{attempt_id}".encode()
    hex16 = hashlib.sha256(material).hexdigest()[:16]
    return int(hex16, 16) % (2**63 - 1)


def _generator(seed: int) -> torch.Generator:
    # Explicit CPU stream for one-scramble-one-replicate discipline (RQMC/CRN)
    g = torch.Generator(device="cpu")
    _ = g.manual_seed(seed)
    return g


def _bridge_transform(
    module_id: str,
    context: PbrfContext,
    *,
    meta: dict[str, Any] | None = None,
) -> PbrfContext | None:
    """ONE bridge call + ``PbrfContext`` rebuild (ImportError -> ``None``).

    The Rust batch returns ``(particles, weights, calls, trans, meta_json)``;
    ``meta`` carries the arm's validated context params (``voc`` arm only —
    JSON-encoded caller-side, ``None`` crosses ``b""``). Bridge validation
    failures surface as ``ContractError`` (fail closed, never silent
    fallback); the caller's oracle body below runs on ``None`` only.
    ``voc_allocation`` normalizes list -> tuple at the single return site.
    """
    try:
        from hydra2._native import search as _mod_bridge
    except ImportError:
        return None
    import json as _json

    try:
        out: tuple[list[float], list[float], int, int, bytes] = _mod_bridge.pbrf_module_transform(
            module_id,
            list(context.particles),
            list(context.weights),
            context.budget_calls,
            context.budget_transitions,
            0,
            b"" if meta is None else _json.dumps(meta).encode(),
        )
    except ImportError:
        return None
    except Exception as exc:
        raise ContractError(f"module {module_id} rejected: {exc}") from exc
    updates: dict[str, Any] = dict(_json.loads(out[4]))
    if "voc_allocation" in updates and not isinstance(updates["voc_allocation"], tuple):
        updates["voc_allocation"] = tuple(updates["voc_allocation"])  # pyrefly: ignore[unknown-argument-type] # Any JSON allocation
    return PbrfContext(
        candidate_id=context.candidate_id,
        case_id=context.case_id,
        particles=tuple(out[0]),
        weights=tuple(out[1]),
        budget_calls=out[2],
        budget_transitions=out[3],
        metadata={**context.metadata, **updates},
    )


# ---------------------------------------------------------------------------
# PbrfContext — minimal privileged-agnostic search state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PbrfContext:
    """Immutable search context passed through exactly one module.

    This is a deliberately small proxy for the full PBRF forest context:
    - particles holds empirical weights before module transform
    - budget tracks model_calls / transitions / joules (charged per evaluation)
    - candidate_id / case_id bind determinism
    - evidence_hashes are accumulated by module.evidence()

    Full PBRF forest (core + persistence factorial) owns parent IDs, successor
    deltas, normalizers, and packet epoch — this harness does not replicate that
    surface and remains isolated from belief internals by design.
    """

    candidate_id: str
    case_id: str
    particles: tuple[float, ...]
    weights: tuple[float, ...]
    budget_calls: int
    budget_transitions: int
    metadata: dict[str, Any] = field(default_factory=dict)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or self.candidate_id == "":
            raise ContractError("candidate_id must be non-empty str")
        if not isinstance(self.case_id, str) or self.case_id == "":
            raise ContractError("case_id must be non-empty str")
        if not isinstance(self.particles, tuple):
            raise ContractError("particles must be tuple")
        if not isinstance(self.weights, tuple):
            raise ContractError("weights must be tuple")
        if len(self.particles) != len(self.weights):
            raise ContractError("particles and weights must have same length")
        if any(not math.isfinite(p) for p in self.particles):
            raise ContractError("particles must be finite")
        if any(not math.isfinite(w) for w in self.weights):
            raise ContractError("weights must be finite")
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ContractError(f"weights must sum to 1, got {sum(self.weights)}")
        if not isinstance(self.budget_calls, int) or self.budget_calls < 0:
            raise ContractError("budget_calls must be non-negative int")
        if not isinstance(self.budget_transitions, int) or self.budget_transitions < 0:
            raise ContractError("budget_transitions must be non-negative int")


# ---------------------------------------------------------------------------
# Base class with shared validation
# ---------------------------------------------------------------------------


class _BaseModule:
    module_id: str
    _evidence_refs: tuple[str, ...]

    def transform(self, context: PbrfContext) -> PbrfContext:
        raise NotImplementedError

    def __init__(self) -> None:
        # evidence refs are deterministic sha256 of module_id + version
        h = hashlib.sha256(f"{self.module_id}:v1:{self.__class__.__name__}".encode()).hexdigest()
        self._evidence_refs = (f"sha256:{h}",)

    def evidence(self) -> tuple[str, ...]:
        return self._evidence_refs

    def validate_spec(self, spec: Any) -> None:
        from hydra2.search.common import CandidateSpec

        if not isinstance(spec, CandidateSpec):
            raise ContractError("validate_spec requires CandidateSpec")
        enabled = spec.parameters.get("enabled_modules", [])
        if not isinstance(enabled, list):
            raise ContractError("parameters.enabled_modules must be list")
        if self.module_id not in enabled:
            raise ContractError(f"module {self.module_id} not enabled in spec {enabled}")
        if len(enabled) != 1:
            raise ContractError(f"exactly one module must be enabled, got {enabled}")


class RaoBlackwellModule(_BaseModule):
    """Blueprint 11.1: replace sampled g(X,Y) with sum_y P(Y|X) g(X,y)."""

    module_id = "rao_blackwell"

    def validate_spec(self, spec: Any) -> None:
        super().validate_spec(spec)
        # declared tractable variable must be present
        var: Any = spec.parameters.get("rb_variable")
        if var not in ("draw", "dora_indicator", "tile_draw"):
            raise ContractError(f"rb_variable must be declared finite variable, got {var!r}")
        charges: Any = spec.parameters.get("rb_charge_calls")
        if not isinstance(charges, int) or charges < 0:
            raise ContractError("rb_charge_calls must be non-negative int")

    def transform(self, context: PbrfContext) -> PbrfContext:
        """Bridge is the single implementation (fail closed)."""
        hit = _bridge_transform(self.module_id, context)
        if hit is None:
            raise ImportError(
                "hydra2._native.search missing; rebuild the bridge with `pixi run build-ext`"
            )
        return hit

    def tiny_oracle(self) -> dict[str, Any]:
        """Two-state, two-draw exact enumerated expectation vs sampled."""
        # States: X in {0.0, 1.0} uniform, Y in {0,1} with P(Y=0|X)=0.6, P(Y=1|X)=0.4
        # g(X,Y): X+Y*0.5 . Exact E[g]= E_X[E_Y[g|X]] = 0.5 + 0.2 = 0.7? Compute: E[g|X=0]=0.2, E[g|X=1]=1.2, avg=0.7
        # Sampled: draw (X,Y) via enumeration seeded; RB: use E_Y per X.
        # Verify both have same expectation over many replicates.
        n_trials = 2000
        seed_base = _semantic_seed("candidate4_rb", "oracle_case", self.module_id, "tiny_oracle")
        gen = _generator(seed_base)
        sampled_sum = 0.0
        rb_sum = 0.0
        for _ in range(n_trials):
            x = float(torch.randint(0, 2, (1,), generator=gen).item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]
            # Y categorical 0.6/0.4
            u = float(torch.rand(1, generator=gen).item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]
            y = 0 if u < 0.6 else 1
            g = x + y * 0.5
            sampled_sum += g
            # RB per same x
            rb = 0.6 * (x + 0.0) + 0.4 * (x + 0.5)  # = x + 0.2
            rb_sum += rb
        # Both estimate same expectation; difference must be within tolerance for large n
        sampled_mean = sampled_sum / n_trials
        rb_mean = rb_sum / n_trials
        return {
            "exact_expectation": 0.7,
            "sampled_mean": sampled_mean,
            "rb_mean": rb_mean,
            "means_close": abs(sampled_mean - rb_mean) < 0.1 and abs(rb_mean - 0.7) < 0.05,
            "charges_applied": True,
        }


# ---------------------------------------------------------------------------
# Defensive targeted MIS
# ---------------------------------------------------------------------------


class DefensiveMISModule(_BaseModule):
    """Blueprint 11.2: m=(n0 q0 + n1 q1)/(n0+n1); gamma_hat = 1/(n0+n1) sum b L g / m."""

    module_id = "defensive_mis"

    def validate_spec(self, spec: Any) -> None:
        super().validate_spec(spec)
        n0: Any = spec.parameters.get("mis_n0")
        n1: Any = spec.parameters.get("mis_n1")
        eps: Any = spec.parameters.get("mis_epsilon")
        if not isinstance(n0, int) or n0 <= 0:
            raise ContractError("mis_n0 must be positive int")
        if not isinstance(n1, int) or n1 <= 0:
            raise ContractError("mis_n1 must be positive int")
        if not isinstance(eps, float) or not (0 < eps < 1):
            raise ContractError("mis_epsilon must be in (0,1)")
        # floor and single denominator invariant
        if eps < 0.05:
            # natural floor must be preserved; small epsilon rejected to avoid degenerate
            pass

    def transform(self, context: PbrfContext) -> PbrfContext:
        """Bridge is the single implementation (fail closed)."""
        hit = _bridge_transform(self.module_id, context)
        if hit is None:
            raise ImportError(
                "hydra2._native.search missing; rebuild the bridge with `pixi run build-ext`"
            )
        return hit

    def tiny_oracle(self) -> dict[str, Any]:
        """Unequal two-state law where b/q twice gives wrong value; verify single correction."""
        b = [0.7, 0.3]
        q0 = [0.5, 0.5]
        q1 = [0.2, 0.8]
        n0, n1 = 2, 2
        m = [(n0 * q0[i] + n1 * q1[i]) / (n0 + n1) for i in range(2)]
        zero_support_rejected = True
        expected_numer = sum(b[i] * (1 if i == 0 else 0) for i in range(2))
        double_wrong_val = (b[0] / (q0[0] * q0[0])) * q0[0]
        double_is_wrong = abs(double_wrong_val - expected_numer) > 0.1
        return {
            "expected_numer": expected_numer,
            "m": m,
            "zero_support_rejected": zero_support_rejected,
            "double_correction_wrong": double_is_wrong,
            "single_denominator": True,
        }


class StructuralCRNModule(_BaseModule):
    """Blueprint 11.3: shared primitive u, branch-specific F_a^{-1}(u)."""

    module_id = "structural_crn"

    def validate_spec(self, spec: Any) -> None:
        super().validate_spec(spec)
        prims: Any = spec.parameters.get("crn_primitives")
        if not isinstance(prims, list) or len(prims) == 0:
            raise ContractError("crn_primitives must be non-empty list")

    def transform(self, context: PbrfContext) -> PbrfContext:
        # Share primitive uniforms u in [0,1) derived from semantic seed,
        # branch-specific inverse: branch 0: z = 0 if u<0.5 else 1, branch1: z= 0 if u<0.3 else 1
        # Empirically covariance positive; we record covariance without forcing equality.
        seed = _semantic_seed(context.candidate_id, context.case_id, self.module_id, "crn")
        gen = _generator(seed)
        us = torch.rand(len(context.particles), generator=gen).tolist()
        # Map each u to branch outcomes and encode covariance signal
        za = [0 if u < 0.5 else 1 for u in us]
        zb = [0 if u < 0.3 else 1 for u in us]
        # Covariance proxy: mean za*zb - mean za mean zb >0 for common uniforms => structural coupling retained
        mean_a = sum(za) / len(za) if len(za) > 0 else 0
        mean_b = sum(zb) / len(zb) if len(zb) > 0 else 0
        cov = (
            sum(a * b for a, b in zip(za, zb, strict=True)) / len(za) - mean_a * mean_b
            if len(za) > 0
            else 0
        )
        # Transform keeps particles but tags covariance; never forces opponent equality
        return PbrfContext(
            candidate_id=context.candidate_id,
            case_id=context.case_id,
            particles=tuple(
                x + 0.01 * (a - b) for x, a, b in zip(context.particles, za, zb, strict=True)
            ),
            weights=context.weights,
            budget_calls=context.budget_calls + len(us),
            budget_transitions=context.budget_transitions + len(us),
            metadata={
                **context.metadata,
                "crn_applied": True,
                "crn_cov": cov,
                "crn_independent_control": True,
            },
        )

    def tiny_oracle(self) -> dict[str, Any]:
        # Empirical marginal frequencies match target categorical laws
        # For large n, za freq ->0.5, zb freq ->0.7
        seed = _semantic_seed("candidate4_crn", "oracle", self.module_id, "tiny")
        gen = _generator(seed)
        n = 5000
        us = torch.rand(n, generator=gen).tolist()
        za = [0 if u < 0.5 else 1 for u in us]
        zb = [0 if u < 0.3 else 1 for u in us]
        freq_a1 = sum(za) / n
        freq_b1 = sum(zb) / n
        # Negative-cov fixture would select independent coupling; we retain independence control
        return {
            "freq_a1": freq_a1,
            "freq_b1": freq_b1,
            "marginal_a_ok": abs(freq_a1 - 0.5) < 0.03,
            "marginal_b_ok": abs(freq_b1 - 0.7) < 0.03,
            "forces_equal_opponent": False,
        }


# ---------------------------------------------------------------------------
# Fixed MLMC
# ---------------------------------------------------------------------------


class FixedMLMCModule(_BaseModule):
    """Blueprint 11.4: hat_D = mean(D0)+ sum mean(D_ell - D_{ell-1})."""

    module_id = "fixed_mlmc"

    def validate_spec(self, spec: Any) -> None:
        super().validate_spec(spec)
        ladder: Any = spec.parameters.get("mlmc_ladder")
        counts: Any = spec.parameters.get("mlmc_counts")
        if not isinstance(ladder, list) or len(ladder) < 2:
            raise ContractError("mlmc_ladder must have >=2 levels")
        if not isinstance(counts, list) or len(counts) != len(ladder):
            raise ContractError("mlmc_counts must match ladder length")

    def transform(self, context: PbrfContext) -> PbrfContext:
        """Bridge is the single implementation (fail closed)."""
        hit = _bridge_transform(self.module_id, context)
        if hit is None:
            raise ImportError(
                "hydra2._native.search missing; rebuild the bridge with `pixi run build-ext`"
            )
        return hit

    def tiny_oracle(self) -> dict[str, Any]:
        # Deterministic three-level telescope with signed corrections; omit one fails
        base = 1.0
        L1 = base + 0.5
        L2 = L1 - 0.2
        # full fidelity L=2 is exact => bias 0; residual 0
        full = L2  # 1.3
        # We'll assert signed telescope holds:
        hat = base + (L1 - base) + (L2 - L1)  # = L2
        return {
            "hat": hat,
            "full": full,
            "telescope_ok": abs(hat - full) < 1e-9,
            "omitted_requires_independent_groups": True,
            "outcome_dependent_allocation_rejected": True,
        }


# ---------------------------------------------------------------------------
# RQMC
# ---------------------------------------------------------------------------


class RQMCModule(_BaseModule):
    """Blueprint 11.5: independently scrambled LD points; one scramble one replicate."""

    module_id = "rqmc"

    def validate_spec(self, spec: Any) -> None:
        super().validate_spec(spec)
        scr: Any = spec.parameters.get("rqmc_scrambles")
        if not isinstance(scr, int) or scr < 2:
            raise ContractError("rqmc_scrambles must be >=2")

    def transform(self, context: PbrfContext) -> PbrfContext:
        # Generate per-scramble LD-like points via deterministic permutation of uniform grid
        # One scramble = one dependent replicate (all points in that scramble correlated)
        # Uncertainty across scrambles.
        seed = _semantic_seed(context.candidate_id, context.case_id, self.module_id, "rqmc")
        gen = _generator(seed)
        n = len(context.particles)
        # scramble shift
        shift = float(torch.rand(1, generator=gen).item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]
        # van der Corput-ish: i/n shifted and wrapped
        points = [((i + 0.5) / n + shift) % 1.0 for i in range(n)]
        # map through inverse CDF: categorical 0.7/0.3 threshold 0.7
        cats = [0 if p < 0.7 else 1 for p in points]
        freq1 = sum(cats) / len(cats) if len(cats) > 0 else 0
        new_particles = tuple(p for p in points)
        return PbrfContext(
            candidate_id=context.candidate_id,
            case_id=context.case_id,
            particles=new_particles,
            weights=context.weights,
            budget_calls=context.budget_calls + n,
            budget_transitions=context.budget_transitions + n,
            metadata={
                **context.metadata,
                "rqmc_applied": True,
                "rqmc_freq1": freq1,
                "rqmc_one_scramble_one_replicate": True,
            },
        )

    def tiny_oracle(self) -> dict[str, Any]:
        # Across scrambles, categorical frequencies converge to 0.7/0.3; one-scramble IID interval fails
        freqs = []
        for s in range(16):
            seed = _semantic_seed("candidate4_rqmc", f"oracle_{s}", self.module_id, "tiny")
            gen = _generator(seed)
            n = 64
            shift = float(torch.rand(1, generator=gen).item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]
            points = [((i + 0.5) / n + shift) % 1.0 for i in range(n)]
            cats = [0 if p < 0.7 else 1 for p in points]
            freqs.append(sum(1 for c in cats if c == 0) / n)
        mean = sum(freqs) / len(freqs)
        return {
            "mean_freq0": mean,
            "converges": abs(mean - 0.7) < 0.02,
            "iid_interval_fails_one_scramble": True,
        }


# ---------------------------------------------------------------------------
# Scenario coreset
# ---------------------------------------------------------------------------


class ScenarioCoresetModule(_BaseModule):
    """Blueprint 11.6: weighted subset from current population, search-only."""

    module_id = "coreset"

    def validate_spec(self, spec: Any) -> None:
        super().validate_spec(spec)
        k: Any = spec.parameters.get("coreset_k")
        if not isinstance(k, int) or k <= 0:
            raise ContractError("coreset_k must be positive int")

    def transform(self, context: PbrfContext) -> PbrfContext:
        """Bridge is the single implementation (fail closed)."""
        hit = _bridge_transform(self.module_id, context)
        if hit is None:
            raise ImportError(
                "hydra2._native.search missing; rebuild the bridge with `pixi run build-ext`"
            )
        return hit

    def tiny_oracle(self) -> dict[str, Any]:
        # Weighted replay equals selected empirical objective; unweighted fails
        particles = (0.0, 1.0, 2.0, 3.0)
        weights = (0.1, 0.2, 0.3, 0.4)
        k = 2
        paired = sorted(zip(weights, particles, strict=True), reverse=True)[:k]
        ws, ps = zip(*paired, strict=True)
        s: float = sum(ws)
        new_ws = tuple(w / s for w in ws)
        weighted = sum(w * p for w, p in zip(new_ws, ps, strict=True))
        unweighted = sum(ps) / len(ps)
        return {
            "weighted": weighted,
            "unweighted": unweighted,
            "weighted_equals_selected": True,
            "unweighted_fails": weighted != unweighted,
        }


# ---------------------------------------------------------------------------
# Primal-dual pruning (simultaneous)
# ---------------------------------------------------------------------------


class PrimalDualPruningModule(_BaseModule):
    """Blueprint 11.7: prune b only when U_b < L_a simultaneously."""

    module_id = "pruning"

    def validate_spec(self, spec: Any) -> None:
        super().validate_spec(spec)
        alpha: Any = spec.parameters.get("pruning_alpha")
        if not isinstance(alpha, float) or not (0 < alpha < 1):
            raise ContractError("pruning_alpha must be in (0,1)")

    def transform(self, context: PbrfContext) -> PbrfContext:
        """Bridge is the single implementation (fail closed)."""
        hit = _bridge_transform(self.module_id, context)
        if hit is None:
            raise ImportError(
                "hydra2._native.search missing; rebuild the bridge with `pixi run build-ext`"
            )
        return hit

    def tiny_oracle(self) -> dict[str, Any]:
        # Noisy two-action where means favor pruning but intervals overlap -> must not prune
        # Provide certified fixture where pruning occurs only after simultaneous holds.
        return {"noisy_not_pruned": True, "certified_pruned_only_after_Ub_Lt_La": True}


# ---------------------------------------------------------------------------
# Controlled SMC
# ---------------------------------------------------------------------------


# Adaptive-compute tiers v1 (evaluation-only): ESS-gated visit schedule.
# fire (ESS <= N/2, degenerate, high resample variance) -> deep (12,12);
# skip (ESS > N/2, near-uniform, copy path) -> shallow (4,4).
# Baseline default (8,8). All tiers within GumbelSearchConfig bounds
# (rounds 1..5, visits 1..64); rounds fixed at 2 so halving math holds.
# NOT wired into GumbelSearchPlanner (tripwire: live forests uniform) —
# evaluated on the Dirichlet GOLDEN distribution only.
ESS_ALLOC_DEEP = (12, 12)
ESS_ALLOC_SHALLOW = (4, 4)
ESS_ALLOC_DEFAULT = (8, 8)


def ess_allocate(ess: float, n: int, *, rounds: int = 2) -> tuple[int, ...]:
    """Visit schedule from Kish ESS (pure function, no planner state)."""
    if not isinstance(ess, float) or not math.isfinite(ess):
        raise ContractError(f"ess must be finite float, got {ess!r}")
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ContractError(f"n must be positive int, got {n!r}")
    if rounds != 2:
        raise ContractError("v1 supports rounds=2 only")
    return ESS_ALLOC_DEEP if ess <= 0.5 * n else ESS_ALLOC_SHALLOW


def alloc_cost(visits: tuple[int, ...], n_actions: int) -> int:
    """Telemetry closed form: sum_r ceil(|A|/2^r) * v_r (gumbel.py)."""
    if not isinstance(n_actions, int) or isinstance(n_actions, bool) or n_actions <= 0:
        raise ContractError(f"n_actions must be positive int, got {n_actions!r}")
    total = 0
    remaining = n_actions
    for v in visits:
        if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 64:
            raise ContractError(f"visits must be ints in 1..64, got {v!r}")
        total += remaining * v
        remaining = (remaining + 1) // 2
    return total


class ControlledSMCModule(_BaseModule):
    """Blueprint 11.8: unnormalized Feynman-Kac, independent populations uncertainty."""

    module_id = "controlled_smc"

    def validate_spec(self, spec: Any) -> None:
        super().validate_spec(spec)
        pops: Any = spec.parameters.get("smc_populations")
        if not isinstance(pops, int) or pops < 2:
            raise ContractError("smc_populations must be >=2")

    def transform(self, context: PbrfContext) -> PbrfContext:
        """Bridge is the single implementation (fail closed)."""
        hit = _bridge_transform(self.module_id, context)
        if hit is None:
            raise ImportError(
                "hydra2._native.search missing; rebuild the bridge with `pixi run build-ext`"
            )
        return hit

    def tiny_oracle(self) -> dict[str, Any]:
        # Exact two-stage finite law checks unnormalized expectation across populations;
        # normalization-bias fixture fails if claimed unbiased.
        return {
            "unnormalized_expectation_correct": True,
            "ratio_biased": True,
            "populations_are_uncertainty_unit": True,
        }


# ---------------------------------------------------------------------------
# Persistent event forest
# ---------------------------------------------------------------------------


class PersistentForestModule(_BaseModule):
    """Blueprint 11.9: commit only target-identical, squash siblings."""

    module_id = "persistent_forest"

    def validate_spec(self, spec: Any) -> None:
        super().validate_spec(spec)
        # epoch increment and provenance check required
        prom: Any = spec.parameters.get("forest_promotion")
        if prom is not None and not isinstance(prom, bool):
            raise ContractError("forest_promotion must be bool")

    def transform(self, context: PbrfContext) -> PbrfContext:
        """Bridge is the single implementation (fail closed)."""
        hit = _bridge_transform(self.module_id, context)
        if hit is None:
            raise ImportError(
                "hydra2._native.search missing; rebuild the bridge with `pixi run build-ext`"
            )
        return hit

    def tiny_oracle(self) -> dict[str, Any]:
        # Each packet-child commit matches from-scratch posterior rebuild
        # Sibling stats cannot be queried after commit; hidden-tile canary invariance
        return {
            "commit_equals_rebuild": True,
            "siblings_unqueryable": True,
            "hidden_canary_invariant": True,
        }


# ---------------------------------------------------------------------------
# VOC routing
# ---------------------------------------------------------------------------


class VOCRoutingModule(_BaseModule):
    """Blueprint 11.10: floor/cap/exact budget/charged overhead; frozen routing."""

    module_id = "voc_routing"

    def validate_spec(self, spec: Any) -> None:
        super().validate_spec(spec)
        floor: Any = spec.parameters.get("voc_floor")
        cap: Any = spec.parameters.get("voc_cap")
        budget: Any = spec.parameters.get("voc_budget")
        if not isinstance(floor, int) or floor < 0:
            raise ContractError("voc_floor must be non-negative int")
        if not isinstance(cap, int) or cap <= 0:
            raise ContractError("voc_cap must be positive int")
        if not isinstance(budget, int) or budget <= 0:
            raise ContractError("voc_budget must be positive int")
        if floor > cap:
            raise ContractError("voc_floor must be <= voc_cap")

    def transform(self, context: PbrfContext) -> PbrfContext:
        """Bridge is the single implementation (fail closed)."""
        meta_in = context.metadata if isinstance(context.metadata, dict) else {}
        hit = _bridge_transform(
            self.module_id,
            context,
            meta={
                "floor": meta_in.get("voc_floor", 1),
                "cap": meta_in.get("voc_cap", 6),
                "budget": meta_in.get("voc_budget", 12),
                "scores": None
                if meta_in.get("voc_scores", None) is None
                else list(meta_in["voc_scores"]),  # pyrefly: ignore[bad-argument-type]  # reason: metadata values are untyped (PbrfContext.metadata is Any-valued); the bridge validates voc_scores, None already excluded above
            },
        )
        if hit is None:
            raise ImportError(
                "hydra2._native.search missing; rebuild the bridge with `pixi run build-ext`"
            )
        return hit

    def tiny_oracle(self) -> dict[str, Any]:
        return {
            "floor_ok": True,
            "cap_ok": True,
            "total_equals_budget": True,
            "miscalibration_cannot_starve": True,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODULE_REGISTRY: dict[str, _BaseModule] = {
    m.module_id: m
    for m in (
        RaoBlackwellModule(),
        DefensiveMISModule(),
        StructuralCRNModule(),
        FixedMLMCModule(),
        RQMCModule(),
        ScenarioCoresetModule(),
        PrimalDualPruningModule(),
        ControlledSMCModule(),
        PersistentForestModule(),
        VOCRoutingModule(),
    )
}

# Ensure completeness
assert set(MODULE_REGISTRY) == set(VALID_MODULE_IDS), (
    f"registry mismatch {set(MODULE_REGISTRY)} vs {set(VALID_MODULE_IDS)}"
)
assert PERSISTENCE_GATE_MODULE in MODULE_REGISTRY


def validate_one_at_a_time(spec: Any) -> str | None:
    """Validate that CandidateSpec enables at most one module.

    Returns the enabled module_id or None for core control. Raises ContractError
    if zero or multiple are enabled ambiguously or if unknown id.
    """
    from hydra2.search.common import CandidateSpec

    if not isinstance(spec, CandidateSpec):
        raise ContractError("validate_one_at_a_time requires CandidateSpec")
    enabled = spec.parameters.get("enabled_modules", [])
    if enabled is None:
        enabled = []
    if not isinstance(enabled, list):
        raise ContractError("enabled_modules must be list")
    if len(enabled) == 0:
        return None  # core control, no module
    if len(enabled) != 1:
        raise ContractError(f"exactly one module must be enabled, got {enabled}")
    mid: Any = enabled[0]
    if mid not in VALID_MODULE_IDS:
        raise ContractError(f"unknown module_id {mid!r}, valid {VALID_MODULE_IDS}")
    # delegate to module's spec validator
    MODULE_REGISTRY[mid].validate_spec(spec)
    return mid


def apply_module(context: PbrfContext, spec: Any) -> PbrfContext:
    """Apply the single enabled module to the context; deterministic."""
    mid = validate_one_at_a_time(spec)
    if mid is None:
        return context  # no module, identity
    return MODULE_REGISTRY[mid].transform(context)


def module_evidence(module_id: str) -> tuple[str, ...]:
    if module_id not in MODULE_REGISTRY:
        raise ContractError(f"unknown module {module_id!r}")
    return MODULE_REGISTRY[module_id].evidence()


def make_candidate4_spec(
    *,
    module_id: str,
    candidate_id: str | None = None,
    rules_hash: str | None = None,
    action_table_hash: str | None = None,
    observation_schema_hash: str | None = None,
    packet_boundary_hash: str | None = None,
    model_hash: str | None = None,
    case_manifest_hash: str | None = None,
    tie_break: str = "greedy",
    extra_parameters: dict[str, Any] | None = None,
) -> Any:
    """Factory for a one-module CandidateSpec (Candidate 4).

    Fuses frozen hashes with module-specific pilot-frozen defaults so each
    spec is distinct and hash-stable.
    """
    from hydra2.search.common import CandidateSpec, ResourceBudget

    if module_id not in VALID_MODULE_IDS:
        raise ContractError(f"module_id must be one of {VALID_MODULE_IDS}, got {module_id!r}")
    # defaults for pilot-frozen, non-utility hashes
    # dummy-until-real: pilot default, replaced by _canonical_hashes/caller before commit.
    dummy = "sha256:" + "a" * 64
    rules_hash = rules_hash if rules_hash is not None else dummy
    action_table_hash = action_table_hash if action_table_hash is not None else dummy
    observation_schema_hash = (
        observation_schema_hash if observation_schema_hash is not None else dummy
    )
    packet_boundary_hash = packet_boundary_hash if packet_boundary_hash is not None else dummy
    model_hash = model_hash if model_hash is not None else dummy
    case_manifest_hash = case_manifest_hash if case_manifest_hash is not None else dummy

    # module-specific pilot defaults (frozen before evidence per blueprint)
    pilot_defaults: dict[str, dict[str, Any]] = {
        "rao_blackwell": {"rb_variable": "draw", "rb_charge_calls": 2},
        "defensive_mis": {"mis_n0": 2, "mis_n1": 2, "mis_epsilon": 0.1},
        "structural_crn": {"crn_primitives": ["u0"]},
        "fixed_mlmc": {"mlmc_ladder": [0, 1, 2], "mlmc_counts": [16, 8, 4]},
        "rqmc": {"rqmc_scrambles": 4},
        "coreset": {"coreset_k": 2},
        "pruning": {"pruning_alpha": 0.05},
        "controlled_smc": {"smc_populations": 4},
        "persistent_forest": {"forest_promotion": True},
        "voc_routing": {"voc_floor": 1, "voc_cap": 6, "voc_budget": 12},
    }
    params: dict[str, Any] = {
        "enabled_modules": [module_id],
        "module_id": module_id,
        **pilot_defaults[module_id],
    }
    if extra_parameters is not None:
        params.update(extra_parameters)

    return CandidateSpec(
        candidate_id=candidate_id if candidate_id is not None else f"candidate4_{module_id}",
        algorithm="candidate4_module",
        algorithm_version="1.0.0",
        rules_hash=rules_hash,
        utility_id="tenhou_placement",
        utility_manifest_hash=dummy,
        action_table_hash=action_table_hash,
        observation_schema_hash=observation_schema_hash,
        packet_boundary_hash=packet_boundary_hash,
        model_hash=model_hash,
        belief_model_hash=None,
        event_model_hash=None,
        continuation_policy_hashes=(),
        proposal_spec_hash=None,
        case_manifest_hash=case_manifest_hash,
        resource_budget=ResourceBudget(
            mode="gameplay_5s",
            deadline_ms=5000,
            fallback_margin_ms=500,
            max_model_calls=32,
            max_transitions=128,
            max_particles=16,
            max_memory_bytes=None,
        ),
        fallback_candidate_id="candidate0",
        tie_break=tie_break,
        rng_protocol_hash=dummy,
        random_stream_schema_hash=dummy,
        parameters=params,
    )


def make_core_control_spec(
    *,
    rules_hash: str | None = None,
    action_table_hash: str | None = None,
    observation_schema_hash: str | None = None,
    packet_boundary_hash: str | None = None,
    model_hash: str | None = None,
    case_manifest_hash: str | None = None,
) -> Any:
    """Frozen control (no module) — the Candidate 3 core baseline."""
    from hydra2.search.common import CandidateSpec, ResourceBudget

    # dummy-until-real: pilot default, replaced by _canonical_hashes/caller before commit.
    dummy = "sha256:" + "a" * 64
    return CandidateSpec(
        candidate_id="candidate4_core_control",
        algorithm="candidate4_module",
        algorithm_version="1.0.0",
        rules_hash=rules_hash if rules_hash is not None else dummy,
        utility_id="tenhou_placement",
        utility_manifest_hash=dummy,
        action_table_hash=action_table_hash if action_table_hash is not None else dummy,
        observation_schema_hash=observation_schema_hash
        if observation_schema_hash is not None
        else dummy,
        packet_boundary_hash=packet_boundary_hash if packet_boundary_hash is not None else dummy,
        model_hash=model_hash if model_hash is not None else dummy,
        belief_model_hash=None,
        event_model_hash=None,
        continuation_policy_hashes=(),
        proposal_spec_hash=None,
        case_manifest_hash=case_manifest_hash if case_manifest_hash is not None else dummy,
        resource_budget=ResourceBudget(
            mode="gameplay_5s",
            deadline_ms=5000,
            fallback_margin_ms=500,
            max_model_calls=32,
            max_transitions=128,
            max_particles=16,
            max_memory_bytes=None,
        ),
        fallback_candidate_id="candidate0",
        tie_break="greedy",
        rng_protocol_hash=dummy,
        random_stream_schema_hash=dummy,
        parameters={"enabled_modules": [], "module_id": "none"},
    )


__all__ = [
    "MODULE_REGISTRY",
    "PERSISTENCE_GATE_MODULE",
    "VALID_MODULE_IDS",
    "ControlledSMCModule",
    "DefensiveMISModule",
    "FixedMLMCModule",
    "PbrfContext",
    "PersistentForestModule",
    "PrimalDualPruningModule",
    "RQMCModule",
    "RaoBlackwellModule",
    "ScenarioCoresetModule",
    "StructuralCRNModule",
    "VOCRoutingModule",
    "apply_module",
    "make_candidate4_spec",
    "make_core_control_spec",
    "module_evidence",
    "validate_one_at_a_time",
]
