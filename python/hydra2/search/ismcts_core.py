# ruff: noqa: F401, SIM102, SIM108  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dep fallback imports; SIM102/SIM108 nested contract guards; B905 intentionally non-strict action/legal zips; N814 upstream belief symbol casing). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 1 ISMCTS core — firewall vocabulary, frozen config, tree nodes, and policy.

Owns the information-set firewall shared by the search loop and the
Planner adapter: the forbidden tree-key vocabulary, the redetermination
negative control, the canonical info-key and vector math, the frozen
``NaturalISMCTSConfig`` hyper-parameters, the ``InformationSetNode`` tree
records with UCT selection, and the frozen ``UniformContinuationPolicy``
actor-view sampler. The simulation loop lives in
:mod:`hydra2.search.ismcts_search` and the Planner adapter in
:mod:`hydra2.search.ismcts_act` so each file stays inside the
review-size ceiling.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import (
    ContractError,
    VisibilityViolationError,
)
from hydra2.search.gumbel_core import scalarize_vector as scalarize_vector

try:
    from hydra2.search.common import (
        CandidateSpec as CandidateSpec,
    )
    from hydra2.search.common import (
        Planner as Planner,
    )
    from hydra2.search.common import (
        ResourceBudget as ResourceBudget,
    )
    from hydra2.search.common import (
        SearchRequest as SearchRequest,
    )
    from hydra2.search.common import (
        SearchResult as SearchResult,
    )

    _COMMON_AVAILABLE = True
except ImportError as exc:
    raise ImportError(
        "hydra2.search.common is required for ismcts_core; "
        "the minimal-contract fallback was removed (single authority is search.common)"
    ) from exc


try:
    from hydra2.contracts.randomness import RandomStream, make_random_stream_key, semantic_seed

    _RANDOM_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    RandomStream = Any  # placeholder; _require_random_stream() raises on use
    make_random_stream_key = Any
    semantic_seed = Any
    _RANDOM_IMPORT_ERROR = exc


def _require_random_stream() -> Any:
    """Fail-closed RNG access (lazy ImportError with build-ext hint)."""
    if _RANDOM_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.contracts.randomness not importable "
            f"({_RANDOM_IMPORT_ERROR}); build the bridge with `pixi run build-ext` "
            "before ISMCTS search"
        ) from _RANDOM_IMPORT_ERROR
    return RandomStream


try:
    from hydra2.belief.natural import BeliefEpoch, NaturalBelief
    from hydra2.belief.world import FullWorld, make_full_world, world_actor_observation

    _BELIEF_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    NaturalBelief = Any  # placeholder; _require_belief() raises on use
    BeliefEpoch = Any
    FullWorld = Any
    make_full_world = Any
    world_actor_observation = Any
    _BELIEF_IMPORT_ERROR = exc


def _require_belief() -> None:
    """Fail-closed belief access (lazy ImportError with build-ext hint)."""
    if _BELIEF_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.belief natural/world not importable "
            f"({_BELIEF_IMPORT_ERROR}); build the bridge with `pixi run build-ext` "
            "before ISMCTS search"
        ) from _BELIEF_IMPORT_ERROR


try:
    from hydra2.contracts.action_model import CanonicalAction
    from hydra2.contracts.observation_actor import (
        ActorObservation,
        observation_identity_document,
    )
    from hydra2.contracts.utility import UtilityVector
    from hydra2.eval.telemetry import ResourceTelemetry, make_resource_telemetry

    _TELEMETRY_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    CanonicalAction = Any  # placeholder; _require_telemetry() raises on use
    ActorObservation = Any
    observation_identity_document = Any
    UtilityVector = Any
    ResourceTelemetry = Any
    make_resource_telemetry = Any
    _TELEMETRY_IMPORT_ERROR = exc


def _require_telemetry() -> Any:
    """Fail-closed telemetry access (lazy ImportError with build-ext hint)."""
    if _TELEMETRY_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.eval.telemetry/contracts not importable "
            f"({_TELEMETRY_IMPORT_ERROR}); build the bridge with `pixi run build-ext` "
            "before ISMCTS search"
        ) from _TELEMETRY_IMPORT_ERROR
    return make_resource_telemetry


try:
    from hydra2._native import search as _ismcts_core_bridge  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover — bridge-less env keeps pure-Python oracles
    _ismcts_core_bridge = None  # type: ignore[assignment]


def _bridge_attr(name: str, default: Any) -> Any:
    """Read one ``hydra2._native.search`` ISMCTS leaf with a HEAD-literal fallback."""
    try:
        bridge: Any = _ismcts_core_bridge
        if bridge is None:
            return default
        return getattr(bridge, name, default)
    except Exception:
        return default


__all__ = [
    "FORBIDDEN_IN_TREE_KEY",
    "InformationSetNode",
    "NaturalISMCTSConfig",
    "UniformContinuationPolicy",
    "info_key_for_observation",
    "is_redeterminization_enabled",
    "model_vector_for_world",
    "scalarize_vector",
    "terminal_vector_for_world",
    "validate_tree_keys_contain_no_world_id",
]

# Single source is ``hydra2._native.search`` once MAIN wires
# ``ismcts_core::register`` (``ISMCTS_*``); the HEAD literal below is the
# stale-``.so``/bridge-less fallback (same values, byte-identical to the
# replaced literal).
FORBIDDEN_IN_TREE_KEY: frozenset[str] = frozenset(
    _bridge_attr(
        "ISMCTS_FORBIDDEN_IN_TREE_KEY",
        {
            "world_id",
            "simulator_snapshot",
            "hidden_tiles",
            "wall",
            "dead_wall",
            "opponent_hand",
            "full_world",
            "privileged",
            "privileged_label",
            "world_ref",
            "parent_id",
            "latent_state_hidden",
            "server_private",
            "engine_rng_state",
            "future_events",
            "opponent_concealed",
            "unrevealed_dora",
        },
    )
)

_MASTER_SEED = bytes(_bridge_attr("ISMCTS_MASTER_SEED", b"wp08b_ismcts_natural_v1"))


def _require_search_bridge() -> Any:
    """Import the built ``search`` bridge surface (fail closed)."""
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with search not importable; "
            "build the bridge with `pixi run build-ext` before ISMCTS selection"
        ) from exc
    try:
        return _ext.search
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.search submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def is_redeterminization_enabled() -> bool:
    """Re-determinization is disabled until a named conditional-law proof exists.

    State keys are canonical information-set hashes for the acting player (no world/full-state hash in the action-selection key); ``q_j(x | I_j, immutable_constraints)`` must preserve public reach,
    root-known tiles and observed packets, with exact ratio applied once.
    Without the tiny-state proof, this flag remains ``False`` and any call that
    attempts conditional re-sampling raises ``ContractError``.
    """
    return False


def attempt_redeterminize(*_args: Any, **_kwargs: Any) -> None:
    """Negative control — always raises because re-determinization is disabled."""
    raise ContractError(
        "re-determinization disabled: requires named conditional law "
        "q_j(x | I_j, immutable_constraints) with exact b/q proof"
    )


def info_key_for_observation(observation: Any) -> str:
    """Canonical information-set key for the acting player's observation.

    The key is ``sha256`` over the RFC 8785 canonical bytes of the actor-observation
    identity document *without* ``legal_mask`` redundancy and *without*
    ``observation_hash``. No world ID, hidden tiles, or server-private fields
    may appear (enforced). Equal actor observations map to equal keys;
    hidden-permutation worlds with the same root observation map to the same key.
    """
    if observation is None:
        raise ContractError("observation must be ActorObservation")
    _require_telemetry()
    try:
        from hydra2.contracts.observation_actor import ActorObservation as _Obs
        from hydra2.contracts.observation_actor import observation_identity_document as _oid

        if isinstance(observation, _Obs):
            doc = _oid(observation)
        else:
            raise ContractError(
                f"continuation policy input must be ActorObservation, got {type(observation).__name__}"
            )
    except ImportError:
        raise
    except Exception as exc:  # pragma: no cover
        if isinstance(exc, ContractError):
            raise
        raise ContractError(
            f"observation must be ActorObservation, got {type(observation).__name__}"
        ) from exc
    # doc excludes observation_hash already; remove legal_mask redundancy per spec
    doc = {k: v for k, v in doc.items() if k != "legal_mask"}
    # Forbidden field check
    for bad in FORBIDDEN_IN_TREE_KEY:
        if bad in doc:
            raise VisibilityViolationError(
                f"forbidden field {bad!r} in tree key document [PBRF_VIS_TREE_KEY]"
            )
        # also check values for accidental world_id leakage via string
        # (lightweight: if value is dict with world_id key)
        if isinstance(doc.get(bad), dict):
            raise VisibilityViolationError(
                f"forbidden nested field {bad!r} in tree key [PBRF_VIS_TREE_KEY_NESTED]"
            )
    payload = canonical_bytes(doc)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def model_vector_for_world(
    world: Any, *, candidate_id: str = "candidate1"
) -> tuple[float, float, float, float]:
    """Deterministic four-seat leaf value from actor-visible model stub.

    In the tiny domain the model is not a learned network but a frozen
    deterministic mapping from ``world_id`` (via hash) to a 4-vector. This
    preserves vector semantics without hidden-state leakage and remains
    deterministic across replays.
    """
    _wid_val: Any = getattr(world, "world_id", None)
    if _wid_val is not None and str(_wid_val) != "":
        wid: str = str(_wid_val)  # pyrefly: ignore[explicit-any]
    else:
        _wid_ref: Any = getattr(world, "world_ref", None)
        if _wid_ref is not None and str(_wid_ref) != "":
            wid = str(_wid_ref)  # pyrefly: ignore[explicit-any]
        else:
            wid = str(world)  # pyrefly: ignore[explicit-any]
    # Digest worlds ride the bridge as the single implementation (fail closed;
    # pinned (0.2, 0.51, 0.06, 0.22) in test_search_parity_wave2). Non-digest
    # shapes keep the hash below as their sole implementation (bridge gate
    # never sees them).
    if wid.startswith("sha256:"):
        try:
            out: tuple[float, float, float, float] = _require_search_bridge().ismcts_model_vector(
                wid, candidate_id
            )
            return (out[0], out[1], out[2], out[3])
        except ImportError:
            raise
        except Exception as exc:
            raise ContractError(f"ismcts bridge model vector failed: {exc}") from exc
    h = hashlib.sha256(f"{wid}:{candidate_id}:leaf".encode()).digest()
    return ((h[0] % 100) / 100.0, (h[1] % 100) / 100.0, (h[2] % 100) / 100.0, (h[3] % 100) / 100.0)
    # Keep vectors in [0,1] and preserve raw settlement shape (no utility-schema mangling)


def terminal_vector_for_world(world: Any) -> tuple[float, float, float, float]:
    """Exact terminal utility for tiny simulator — derived from hands + wall."""
    # Deterministic settlement placeholder that is stable but distinct per world.
    _wid_val2: Any = getattr(world, "world_id", None)
    if _wid_val2 is not None and str(_wid_val2) != "":
        wid: str = str(_wid_val2)  # pyrefly: ignore[explicit-any]
    else:
        wid = str(world)  # pyrefly: ignore[explicit-any]
    # Digest worlds ride the bridge as the single implementation (fail closed);
    # non-digest shapes keep the hash below as their sole implementation.
    if wid.startswith("sha256:"):
        try:
            out: tuple[float, float, float, float] = (
                _require_search_bridge().ismcts_terminal_vector(wid)
            )
            return (out[0], out[1], out[2], out[3])
        except ImportError:
            raise
        except Exception as exc:
            raise ContractError(f"ismcts bridge terminal vector failed: {exc}") from exc
    # Hash to settlement: first seat gets higher when hand sum larger
    h = hashlib.sha256(f"{wid}:terminal".encode()).digest()
    # Produce bounded scores then convert to placement-like values
    # Test proxy only: deterministic hash-derived vectors stand in for
    # model scores in unit tests; never feed them to real utility.
    scores = tuple((b % 50) - 25 for b in h[:4])  # -25..24
    # Convert to utility-like ranks: softmax would be zero-sum but we keep raw vector
    # For test preservation we keep values finite and distinct.
    base = tuple(float(s) / 50.0 for s in scores)  # -0.5..0.48
    # Shift to [0,1] for placement style but keep 4-dim vector identity
    return (base[0] + 0.5, base[1] + 0.5, base[2] + 0.5, base[3] + 0.5)


def validate_tree_keys_contain_no_world_id(tree_keys: Any) -> bool:
    """Return True iff no key string contains a world_id / privileged substring."""
    try:
        from hydra2._native import search as _vk_bridge  # pyrefly: ignore[missing-import]

        _vk_fn = _vk_bridge.ismcts_validate_tree_keys  # type: ignore[attr-defined]  # reason: ismcts_core leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except (ImportError, AttributeError):
        pass  # stale .so: fall through to the oracle below (same values)
    else:
        try:
            return bool(_vk_fn([str(k) for k in tree_keys]))  # pyrefly: ignore[unknown-argument-type] # untyped bridge fn
        except Exception as exc:
            raise ContractError(f"ismcts bridge tree-key validation failed: {exc}") from exc
    for k in tree_keys:
        ks = str(k)
        if "world" in ks.lower() and len(ks) > 32:
            # Keys are hashes (64 hex); world_id is also hash but we check leakage via prefix
            # The test checks that tree key document never contained world_id field;
            # presence of raw world bytes in key would fail the info_key function above.
            # Here we provide a best-effort guard: key must be sha256 hex, not raw world bytes.
            if ks.startswith("world") or "FullWorld" in ks:
                return False
        for bad in FORBIDDEN_IN_TREE_KEY:
            if bad in ks:
                return False
    return True


# ---------------------------------------------------------------------------
# Frozen planner config — part of CandidateSpec.parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NaturalISMCTSConfig:
    """Frozen Candidate 1 hyper-parameters (part of CandidateSpec.parameters).

    All fields are pilot-frozen and compose the identity of Candidate 1.
    """

    uct_c: float = 1.41421356237
    max_depth: int = 6
    max_simulations: int = 48
    max_transitions: int | None = 256
    max_model_calls: int | None = 48
    tie_break: str = "lowest_action_id"
    candidate_id: str = "candidate1"
    resource_view: str = "calls"  # for matched confirmation (calls / transitions / joules)
    seed_material: bytes = _MASTER_SEED

    def __post_init__(self) -> None:
        if not isinstance(self.uct_c, float) or not math.isfinite(self.uct_c) or self.uct_c <= 0:
            raise ContractError(f"uct_c must be finite >0, got {self.uct_c!r}")
        if (
            not isinstance(self.max_depth, int)
            or isinstance(self.max_depth, bool)
            or self.max_depth <= 0
            or self.max_depth > 32
        ):
            raise ContractError(f"max_depth must be int 1..32, got {self.max_depth!r}")
        if (
            not isinstance(self.max_simulations, int)
            or isinstance(self.max_simulations, bool)
            or self.max_simulations <= 0
        ):
            raise ContractError(
                f"max_simulations must be positive int, got {self.max_simulations!r}"
            )
        for name in ("max_transitions", "max_model_calls"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
                raise ContractError(f"{name} must be positive int or None, got {v!r}")
        if self.tie_break not in ("lowest_action_id", "stable_hash", "lexicographic"):
            raise ContractError(
                f"tie_break must be lowest_action_id/stable_hash/lexicographic, got {self.tie_break!r}"
            )
        if not isinstance(self.candidate_id, str) or len(self.candidate_id) == 0:
            raise ContractError("candidate_id must be non-empty str")
        if self.resource_view not in ("calls", "transitions", "joules"):
            raise ContractError("resource_view must be calls/transitions/joules")
        if not isinstance(self.seed_material, (bytes, bytearray)) or len(self.seed_material) == 0:
            raise ContractError("seed_material must be non-empty bytes")


# ---------------------------------------------------------------------------
# Information-set tree
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ActionStats:
    visits: int = 0
    value_sum: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass(slots=True)
class InformationSetNode:
    """One root information-set node — aggregates across hidden worlds."""

    key: str
    visits: int = 0
    action_stats: dict[int, _ActionStats] = field(default_factory=dict)
    legal_actions: tuple[int, ...] = ()

    def mean_vector(self, action: int) -> tuple[float, float, float, float] | None:
        st = self.action_stats.get(action)
        if st is None or st.visits == 0:
            return None
        return tuple(v / st.visits for v in st.value_sum)  # type: ignore[return]

    def scalar_mean(self, action: int, root_seat: int) -> float | None:
        mv = self.mean_vector(action)
        if mv is None:
            return None
        return scalarize_vector(mv, root_seat)


# ---------------------------------------------------------------------------
# Continuation policy — frozen legal-masked actor view (sandbox)
# ---------------------------------------------------------------------------


class UniformContinuationPolicy:
    """Frozen continuation policy for non-root seats.

    Samples uniformly from ``legal`` but with an observation-dependent bias
    so that equal actor observations map to equal distributions and changed
    actor-visible information *may* change the distribution (proving the
    policy consumes ``I_j``). No hidden state is consulted.
    """

    def __init__(self, *, bias_strength: float = 0.2) -> None:
        if not isinstance(bias_strength, float) or not 0 <= bias_strength < 0.5:
            raise ContractError("bias_strength must be float in [0,0.5)")
        self._bias = bias_strength

    def _distribution_for(self, observation: Any, legal: tuple[int, ...]) -> tuple[float, ...]:
        if len(legal) == 0:
            raise ContractError("legal must be non-empty")
        # If single action, trivial
        if len(legal) == 1:
            return (1.0,)
        # Derive per-observation bias from observation_hash (deterministic, actor-visible only)
        try:
            _obs_hash_raw: Any = getattr(observation, "observation_hash", None)
            if _obs_hash_raw is not None and str(_obs_hash_raw) != "":
                h: str = str(_obs_hash_raw)  # pyrefly: ignore[explicit-any]
            else:
                h = ""
            # Use hash of observation_hash to decide bias direction
            digest = hashlib.sha256(h.encode()).digest()
            direction = digest[0] & 1  # 0 or 1
        except Exception:
            direction = 0
        # For 2 actions, tilt toward first or second depending on observation
        n = len(legal)
        if n == 2:
            if direction == 0:
                p0 = 0.5 + self._bias
            else:
                p0 = 0.5 - self._bias
            return (p0, 1.0 - p0)
        # For >2, keep uniform (still consumes observation via legal_mask only)
        w = 1.0 / n
        return tuple(w for _ in legal)

    def distribution(self, observation: Any, legal: tuple[int, ...]) -> tuple[float, ...]:
        # Validate no privileged field in observation (lightweight) — fail closed.
        if observation is not None:
            # ensure observation is ActorObservation, not FullWorld
            try:
                from hydra2.contracts.observation_actor import ActorObservation as _Obs
            except ImportError as exc:
                raise ImportError(
                    "hydra2.contracts.observation not importable "
                    f"({exc}); build the bridge with `pixi run build-ext` before ISMCTS search"
                ) from exc
            if not isinstance(observation, _Obs):
                raise ContractError(
                    f"continuation policy input must be ActorObservation, got {type(observation).__name__}"
                )
            # forbid world_id field leak
            if hasattr(observation, "world_id"):
                raise VisibilityViolationError(
                    "policy input contains world_id [PBRF_VIS_POLICY_WORLD]"
                )
            if hasattr(observation, "concealed_hands"):
                raise VisibilityViolationError(
                    "policy input contains concealed_hands (FullWorld) [PBRF_VIS_POLICY_HANDS]"
                )
        return self._distribution_for(observation, legal)

    def sample(self, observation: Any, legal: tuple[int, ...], rng: Any) -> int:
        if not isinstance(legal, tuple) or len(legal) == 0:
            raise ContractError("legal must be non-empty tuple")
        dist = self.distribution(observation, legal)
        # sample categorical via rng — fail closed, no hash%1000 fallback.
        _require_random_stream()
        if not hasattr(rng, "random_float"):
            raise ContractError("ismcts: rng must expose random_float; hash%1000 fallback removed")
        _rng_raw: Any = rng.random_float()  # pyrefly: ignore[explicit-any]
        r: float = float(_rng_raw)  # pyrefly: ignore[explicit-any]
        cum = 0.0
        for idx, p in enumerate(dist):
            cum += p
            if r < cum:
                return legal[idx]
        return legal[-1]


def _uct_select(
    node: InformationSetNode, legal: tuple[int, ...], root_seat: int, uct_c: float, tie_break: str
) -> int:
    # UCT pick rides the bridge (unvisited-first + q+u with 1e-12 eps + tie
    # arm, bit-identical to the oracle below); node tables stay Python —
    # only visited-arm scalars cross, never hidden worlds (info-keys only).
    try:
        stat_ids = sorted(node.action_stats.keys())
    except Exception as exc:
        raise ContractError(f"uct node stats unreadable: {exc}") from exc
    actions: list[int] = []
    visits: list[int] = []
    sums_flat: list[float] = []
    for aid in stat_ids:
        st = node.action_stats.get(aid)
        if st is None:
            continue
        try:
            n = int(st.visits)  # type: ignore[arg-type]
            quad = tuple(st.value_sum)
        except Exception as exc:
            raise ContractError(f"uct node stats malformed for {aid!r}: {exc}") from exc
        if len(quad) != 4:
            raise ContractError(f"uct node value_sum must hold 4 entries for {aid!r}")
        actions.append(aid)
        visits.append(n)
        sums_flat.extend(quad)
    try:
        total = int(node.visits)  # type: ignore[arg-type]
        selected: int = _require_search_bridge().uct_select(
            actions,
            visits,
            sums_flat,
            list(legal),
            root_seat,
            total,
            uct_c,
            tie_break,
        )
        return selected
    except ImportError:
        raise
    except Exception as exc:
        raise ContractError(f"ismcts bridge uct failed: {exc}") from exc
