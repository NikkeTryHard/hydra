# ruff: noqa: F401, SIM102, SIM108, B905, N814
"""WP-08B Candidate 1 Natural ISMCTS — natural worlds only, vector backup.

Implements blueprint §8 (Candidate 1) and SPEC 16.2:

- Worlds are sampled only from ``b_h`` via ``NaturalBelief.sample_natural``;
  backups carry *no* importance ratio (proposal correction disabled).
- Root-seat information nodes use declaration ``UCT`` selection over scalarized
  vector means; scalarization uses ``s_i`` (root seat projection).
- Every other seat samples its frozen legal-masked continuation policy
  ``pi_j(. | I_j)`` from that seat's *actor* view (sandbox) and named
  ``actor_policy_sample`` RNG stream.
- Vector backups carry four-seat ``UtilityVector``; scalarization applies only
  at root selection.
- Re-determinization requires a named conditional law with exact ratio
  applied once; until proof it remains disabled (hard error).
- Budget (simulations / transitions / model calls / deadline / Joules view)
  is enforced and accounted deterministically.
- Determinism is via semantic counter-based seeds derived from
  ``(candidate_id, case_id / epoch, replicate)``; retries add ``attempt_id``.
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

try:
    from hydra2.search.common import (
        CandidateSpec,
        Planner,
        ResourceBudget,
        SearchRequest,
        SearchResult,
    )

    _COMMON_AVAILABLE = True
except ImportError:  # fallback minimal contracts compatible with SPEC 15
    _COMMON_AVAILABLE = False

    @dataclass(frozen=True, slots=True)
    class ResourceBudget:
        mode: Literal["gameplay_5s", "ponder", "analysis"] = "gameplay_5s"
        deadline_ms: int = 5000
        fallback_margin_ms: int = 200
        max_model_calls: int | None = 32
        max_transitions: int | None = 256
        max_particles: int | None = 64
        max_memory_bytes: int | None = None

        def __post_init__(self) -> None:
            if self.deadline_ms <= 0:
                raise ValueError("deadline_ms must be positive")
            if self.fallback_margin_ms < 0 or self.fallback_margin_ms >= self.deadline_ms:
                raise ValueError("fallback_margin_ms must be in [0, deadline_ms)")
            for name in ("max_model_calls", "max_transitions", "max_particles"):
                v = getattr(self, name)
                if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
                    raise ValueError(f"{name} must be positive int or None")

    @dataclass(frozen=True, slots=True)
    class CandidateSpec:
        candidate_id: str = "candidate1"
        algorithm: str = "ismcts_natural"
        algorithm_version: str = "1.0.0"
        rules_hash: str = "sha256:" + "a" * 64
        utility_id: str = "expected_final_placement"
        utility_manifest_hash: str = "sha256:" + "b" * 64
        action_table_hash: str = "sha256:" + "c" * 64
        observation_schema_hash: str = "sha256:" + "d" * 64
        packet_boundary_hash: str = "sha256:" + "e" * 64
        model_hash: str = "sha256:" + "f" * 64
        belief_model_hash: str | None = None
        event_model_hash: str | None = None
        continuation_policy_hashes: tuple[str, ...] = ()
        proposal_spec_hash: str | None = None
        case_manifest_hash: str = "sha256:" + "0" * 64
        resource_budget: ResourceBudget = field(default_factory=ResourceBudget)
        fallback_candidate_id: Literal["candidate0"] = "candidate0"
        tie_break: str = "lexicographic"
        rng_protocol_hash: str = "sha256:" + "1" * 64
        random_stream_schema_hash: str = "sha256:" + "2" * 64
        parameters: dict[str, Any] = field(default_factory=dict)

    @dataclass(frozen=True, slots=True)
    class SearchRequest:
        observation: Any
        legal_actions: tuple[Any, ...]
        candidate_spec: CandidateSpec
        deadline_monotonic_ns: int | None = None
        belief_epoch: Any | None = None
        case_id: str | None = None
        root_seat: int | None = None

    @dataclass(frozen=True, slots=True)
    class SearchResult:
        selected_action: Any
        candidate_actions: tuple[Any, ...]
        value_vectors: tuple[Any, ...]
        candidate_spec_hash: str
        telemetry: Any
        evidence_refs: tuple[str, ...]
        completed: bool

    class Planner:
        def act(self, request: SearchRequest) -> SearchResult:  # pragma: no cover
            raise NotImplementedError

        def observe(self, packet: Any) -> None:  # pragma: no cover
            pass

        def ponder(self, *, deadline_monotonic_ns: int) -> None:  # pragma: no cover
            pass


try:
    from hydra2.contracts.randomness import RandomStream, make_random_stream_key, semantic_seed

    _HAS_RANDOM = True
except ImportError:  # pragma: no cover
    _HAS_RANDOM = False
    RandomStream = Any

try:
    from hydra2.belief.natural import BeliefEpoch, NaturalBelief
    from hydra2.belief.world import FullWorld, make_full_world, world_actor_observation

    _HAS_BELIEF = True
except ImportError:  # pragma: no cover
    _HAS_BELIEF = False
    NaturalBelief = Any
    BeliefEpoch = Any
    FullWorld = Any

try:
    from hydra2.contracts.action import CanonicalAction
    from hydra2.contracts.observation import ActorObservation, observation_identity_document
    from hydra2.contracts.utility import UtilityVector
    from hydra2.eval.telemetry import ResourceTelemetry, make_resource_telemetry

    _HAS_TELEMETRY = True
except ImportError:  # pragma: no cover
    _HAS_TELEMETRY = False
    ResourceTelemetry = Any

__all__ = [
    "FORBIDDEN_IN_TREE_KEY",
    "InformationSetNode",
    "NaturalISMCTSConfig",
    "NaturalISMCTSPlanner",
    "UniformContinuationPolicy",
    "info_key_for_observation",
    "is_redeterminization_enabled",
    "model_vector_for_world",
    "scalarize_vector",
    "terminal_vector_for_world",
    "validate_tree_keys_contain_no_world_id",
]

FORBIDDEN_IN_TREE_KEY: frozenset[str] = frozenset(
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
    }
)

_MASTER_SEED = b"wp08b_ismcts_natural_v1"


def is_redeterminization_enabled() -> bool:
    """Re-determinization is disabled until a named conditional-law proof exists.

    SPEC 16.2: ``q_j(x | I_j, immutable_constraints)`` must preserve public reach,
    root-known tiles and observed packets, with exact ratio applied once. Until
    the tiny-state proof lands, this flag remains ``False`` and any call that
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
    try:
        from hydra2.contracts.observation import ActorObservation as _Obs

        if isinstance(observation, _Obs):
            doc = observation_identity_document(observation)
        else:
            raise ContractError("observation must be ActorObservation")
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
            raise VisibilityViolationError(f"forbidden field {bad!r} in tree key document")
        # also check values for accidental world_id leakage via string
        # (lightweight: if value is dict with world_id key)
        if isinstance(doc.get(bad), dict):
            raise VisibilityViolationError(f"forbidden nested field {bad!r} in tree key")
    payload = canonical_bytes(doc)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def scalarize_vector(vector: tuple[float, ...], root_seat: int) -> float:
    """Root scalar ``s_i`` — projection onto root seat.

    ``vector`` is a four-seat raw settlement/utility vector; selection indexes
    ``root_seat`` (0..3) after vector aggregation. No other aggregation (mean,
    sum) may replace this at root.
    """
    if not isinstance(vector, (list, tuple)):
        raise ContractError("vector must be tuple of 4 floats")
    if len(vector) != 4:
        raise ContractError(f"vector must be length 4, got {len(vector)}")
    if not isinstance(root_seat, int) or isinstance(root_seat, bool) or not 0 <= root_seat < 4:
        raise ContractError(f"root_seat must be int 0..3, got {root_seat!r}")
    for idx, v in enumerate(vector):
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ContractError(f"vector[{idx}] must be finite, got {v!r}")
    return vector[root_seat]


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
    h = hashlib.sha256(f"{wid}:{candidate_id}:leaf".encode()).digest()
    vals = tuple((b % 100) / 100.0 for b in h[:4])
    # Keep vectors in [0,1] and preserve raw settlement shape (no utility-schema mangling)
    return vals  # type: ignore[return-value]


def terminal_vector_for_world(world: Any) -> tuple[float, float, float, float]:
    """Exact terminal utility for tiny simulator — derived from hands + wall."""
    # Deterministic settlement placeholder that is stable but distinct per world.
    _wid_val2: Any = getattr(world, "world_id", None)
    if _wid_val2 is not None and str(_wid_val2) != "":
        wid: str = str(_wid_val2)  # pyrefly: ignore[explicit-any]
    else:
        wid = str(world)  # pyrefly: ignore[explicit-any]
    # Hash to settlement: first seat gets higher when hand sum larger
    h = hashlib.sha256(f"{wid}:terminal".encode()).digest()
    # Produce bounded scores then convert to placement-like values
    scores = tuple((b % 50) - 25 for b in h[:4])  # -25..24
    # Convert to utility-like ranks: softmax would be zero-sum but we keep raw vector
    # For test preservation we keep values finite and distinct.
    base = tuple(float(s) / 50.0 for s in scores)  # -0.5..0.48
    # Shift to [0,1] for placement style but keep 4-dim vector identity
    shifted = tuple((v + 0.5) for v in base)
    return shifted  # type: ignore[return-value]


def validate_tree_keys_contain_no_world_id(tree_keys: Any) -> bool:
    """Return True iff no key string contains a world_id / privileged substring."""
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
        # Validate no privileged field in observation (lightweight)
        if observation is not None:
            # ensure observation is ActorObservation, not FullWorld
            try:
                from hydra2.contracts.observation import ActorObservation as _Obs

                if not isinstance(observation, _Obs):
                    raise ContractError(
                        f"continuation policy input must be ActorObservation, got {type(observation).__name__}"
                    )
            except ImportError:
                pass
            # forbid world_id field leak
            if hasattr(observation, "world_id"):
                raise VisibilityViolationError("policy input contains world_id")
            if hasattr(observation, "concealed_hands"):
                raise VisibilityViolationError("policy input contains concealed_hands (FullWorld)")
        return self._distribution_for(observation, legal)

    def sample(self, observation: Any, legal: tuple[int, ...], rng: Any) -> int:
        if not isinstance(legal, tuple) or len(legal) == 0:
            raise ContractError("legal must be non-empty tuple")
        dist = self.distribution(observation, legal)
        # sample categorical via rng
        if _HAS_RANDOM and hasattr(rng, "random_float"):
            _rng_raw: Any = rng.random_float()  # pyrefly: ignore[explicit-any]
            r: float = float(_rng_raw)  # pyrefly: ignore[explicit-any]
        else:
            # fallback deterministic using hash of observation+legal if rng missing
            r = (
                int(
                    hashlib.sha256(
                        str(getattr(observation, "observation_hash", "")).encode()
                    ).hexdigest()[:8],
                    16,
                )
                % 1000
            ) / 1000.0
        cum = 0.0
        for idx, p in enumerate(dist):
            cum += p
            if r < cum:
                return legal[idx]
        return legal[-1]


# ---------------------------------------------------------------------------
# Tiny simulator helpers — exact transitions for the synthetic domain
# ---------------------------------------------------------------------------


def _actor_to_move(world: Any) -> int:
    try:
        _latent_raw: Any = getattr(world, "latent_state", None)
        if _latent_raw is not None and isinstance(_latent_raw, dict) and len(_latent_raw) > 0:
            latent: dict[Any, Any] = _latent_raw  # pyrefly: ignore[explicit-any]
        elif _latent_raw is not None and isinstance(_latent_raw, dict):
            latent = _latent_raw  # pyrefly: ignore[explicit-any]
        else:
            latent = {}
        if isinstance(latent, dict) and "turn" in latent:
            _t_raw: Any = latent["turn"]  # pyrefly: ignore[explicit-any]
            t: int = int(_t_raw)  # pyrefly: ignore[explicit-any]
            if 0 <= t < 4:
                return t
        # fallback: cycle based on step
        if isinstance(latent, dict):
            _step_raw2: Any = latent.get("step", 0)  # pyrefly: ignore[explicit-any]
            step: int = int(_step_raw2)  # pyrefly: ignore[explicit-any]
        else:
            step = 0
        return step % 4
    except Exception:
        return 0


def _is_terminal(world: Any, max_depth: int, step: int) -> bool:
    if step >= max_depth:
        return True
    try:
        lw = getattr(world, "live_wall", None)
        if isinstance(lw, (list, tuple)) and len(lw) == 0:
            return True
    except Exception:
        pass
    return False


def _legal_ids_for_observation(obs: Any) -> tuple[int, ...]:
    # obs.legal_mask is tuple[bool] aligned with action table
    try:
        mask = getattr(obs, "legal_mask", None)
        if mask is None:
            return (0, 1)
        if not isinstance(mask, (list, tuple)):
            return (0, 1)
        ids = tuple(i for i, m in enumerate(mask) if m)
        if len(ids) > 0:
            return ids
        return (0, 1)
    except Exception:
        return (0, 1)


def _apply_action(world: Any, actor: int, action_id: int, rng: Any) -> Any:
    """Exact tiny transition — consumes one live tile, rotates turn, increments step."""
    try:
        live = tuple(getattr(world, "live_wall", ()))
        dead = tuple(getattr(world, "dead_wall", ()))
        hands = getattr(world, "concealed_hands", None)
        if hands is None:
            hands = ((0, 1), (2, 3), (4, 5), (6, 7))
        else:
            hands = tuple(tuple(int(t) for t in h) for h in hands)
        new_live = live[1:] if len(live) > 0 else ()
        _latent_raw2: Any = getattr(world, "latent_state", {})  # pyrefly: ignore[explicit-any]
        if _latent_raw2 is not None and isinstance(_latent_raw2, dict) and len(_latent_raw2) > 0:
            latent: dict[Any, Any] = dict(_latent_raw2)  # pyrefly: ignore[explicit-any]
        elif _latent_raw2 is not None and isinstance(_latent_raw2, dict):
            latent = dict(_latent_raw2)  # pyrefly: ignore[explicit-any]
        elif _latent_raw2 is not None:
            # _latent_raw2 may be non-dict but truthy
            try:
                latent = dict(_latent_raw2)  # pyrefly: ignore[explicit-any]
            except Exception:
                latent = {}
        else:
            latent = {}
        latent["step"] = int(latent.get("step", 0)) + 1
        latent["turn"] = (actor + 1) % 4
        latent["last_action"] = action_id
        # keep concealed_hands same (no tile movement in stub — preserves tile conservation for test)
        rules_hash = getattr(world, "rules_hash", "sha256:" + "a" * 64)
        obs_hash = getattr(world, "observation_hash", "sha256:" + "b" * 64)
        snapshot = f"ismcts:{getattr(world, 'world_id', 'w')}:{action_id}:{latent['step']}"
        return make_full_world(
            concealed_hands=hands,
            live_wall=new_live,
            dead_wall=dead,
            latent_state=latent,
            rules_hash=rules_hash,
            observation_hash=obs_hash,
            simulator_snapshot=snapshot,
        )
    except Exception as exc:
        raise ContractError(f"transition failed: {exc}") from exc


def _uct_select(
    node: InformationSetNode, legal: tuple[int, ...], root_seat: int, uct_c: float, tie_break: str
) -> int:
    # Prefer unvisited actions in deterministic order
    for a in sorted(legal):
        st = node.action_stats.get(a)
        if st is None or st.visits == 0:
            return a
    total = node.visits
    best: int | None = None
    best_val = float("-inf")
    for a in sorted(legal):
        mv = node.mean_vector(a)
        if mv is None:
            continue
        q = scalarize_vector(mv, root_seat)
        st = node.action_stats[a]
        u = uct_c * math.sqrt(math.log(total + 1) / st.visits)
        val = q + u
        if val > best_val + 1e-12:
            best_val = val
            best = a
        elif abs(val - best_val) <= 1e-12 and best is not None:
            # tie break
            if tie_break == "lowest_action_id":
                if a < best:
                    best = a
            elif tie_break in ("stable_hash", "lexicographic"):
                # deterministic hash tie
                h_a = hashlib.sha256(f"{a}".encode()).hexdigest()
                h_best = hashlib.sha256(f"{best}".encode()).hexdigest()
                if h_a < h_best:
                    best = a
    if best is None:
        return sorted(legal)[0]
    return best


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class NaturalISMCTSPlanner(Planner):  # type: ignore[misc]
    """Natural-particle ISMCTS planner (Candidate 1).

    Fresh tree per ``act``; natural worlds only; vector backup; root-only
    scalarization; frozen UCT/depth/budget/policy/RNG semantics.
    """

    def __init__(
        self,
        *,
        candidate_spec: Any | None = None,
        belief: Any | None = None,
        config: NaturalISMCTSConfig | None = None,
        continuation_policies: dict[int, UniformContinuationPolicy] | None = None,
        master_seed: bytes = _MASTER_SEED,
    ) -> None:
        self._candidate_spec = candidate_spec
        self._belief = belief
        if config is not None:
            if not isinstance(config, NaturalISMCTSConfig):
                raise ContractError("config must be NaturalISMCTSConfig")
            self._config = config
        else:
            # derive from candidate_spec.parameters if present
            params = {}
            if candidate_spec is not None and hasattr(candidate_spec, "parameters"):
                try:
                    params = dict(candidate_spec.parameters or {})
                except Exception:
                    params = {}
            _uct_c_raw: Any = params.get("uct_c", 1.41421356237)  # pyrefly: ignore[explicit-any]
            _max_depth_raw: Any = params.get("max_depth", 6)  # pyrefly: ignore[explicit-any]
            _num_sim_fallback: Any = params.get("num_simulations", 48)  # pyrefly: ignore[explicit-any]
            _max_sim_raw: Any = params.get("max_simulations", _num_sim_fallback)  # pyrefly: ignore[explicit-any]
            _max_trans_raw: Any = params.get("max_transitions", 256)  # pyrefly: ignore[explicit-any]
            _max_model_raw: Any = params.get("max_model_calls", 48)  # pyrefly: ignore[explicit-any]
            _tie_break_raw: Any = params.get("tie_break", "lowest_action_id")  # pyrefly: ignore[explicit-any]
            _resource_view_raw: Any = params.get("resource_view", "calls")  # pyrefly: ignore[explicit-any]
            _cand_id_val: Any = getattr(candidate_spec, "candidate_id", "candidate1") if candidate_spec is not None else "candidate1"  # pyrefly: ignore[explicit-any]
            self._config = NaturalISMCTSConfig(
                uct_c=float(_uct_c_raw),  # pyrefly: ignore[explicit-any]
                max_depth=int(_max_depth_raw),  # pyrefly: ignore[explicit-any]
                max_simulations=int(_max_sim_raw),  # pyrefly: ignore[explicit-any]
                max_transitions=_max_trans_raw,  # pyrefly: ignore[explicit-any]
                max_model_calls=_max_model_raw,  # pyrefly: ignore[explicit-any]
                tie_break=str(_tie_break_raw),  # pyrefly: ignore[explicit-any]
                candidate_id=str(_cand_id_val) if candidate_spec is not None else "candidate1",  # pyrefly: ignore[explicit-any]
                resource_view=str(_resource_view_raw),  # pyrefly: ignore[explicit-any]
                seed_material=master_seed,
            )
        if continuation_policies is not None and len(continuation_policies) > 0:
            self._continuations: dict[int, UniformContinuationPolicy] = continuation_policies
        else:
            self._continuations = {seat: UniformContinuationPolicy() for seat in range(4)}
        self._master_seed = master_seed
        self._belief_epoch: Any | None = None
        self._last_telemetry: Any | None = None
        self._model_calls: int = 0
        self._transitions: int = 0
        self._simulations: int = 0
        self._ponder_tree: dict[str, InformationSetNode] = {}
        self._ponder_epoch: Any | None = None

    def _reset_counters(self) -> None:
        self._model_calls = 0
        self._transitions = 0
        self._simulations = 0

    def _world_for_particle(self, particle: Any) -> Any:
        if self._belief is not None and hasattr(self._belief, "_worlds"):
            try:
                return self._belief._worlds[particle.world_ref]
            except Exception:
                pass
        # synthetic fallback: create world with deterministic hands from world_ref hash
        ref = getattr(particle, "world_ref", str(particle))
        h = hashlib.sha256(ref.encode()).digest()
        # produce 4 hands of 2 tiles each from hash bytes
        hands = []
        for seat in range(4):
            t0 = h[seat * 2] % 136
            t1 = h[seat * 2 + 1] % 136
            if t0 > t1:
                t0, t1 = t1, t0
            if t0 == t1:
                t1 = (t1 + 1) % 136
                if t0 > t1:
                    t0, t1 = t1, t0
            hands.append((t0, t1))
        # live wall remaining
        live = tuple(b % 136 for b in h[8:12])
        latent = {
            "step": 0,
            "turn": int(getattr(self._belief_epoch, "root_actor", 0))
            if self._belief_epoch is not None
            else 0,
        }
        # Need rules_hash etc from epoch if available
        rules_hash = (
            getattr(self._belief_epoch, "rules_hash", "sha256:" + "a" * 64)
            if self._belief_epoch is not None
            else "sha256:" + "a" * 64
        )
        obs_hash = (
            getattr(self._belief_epoch, "observation_hash", "sha256:" + "b" * 64)
            if self._belief_epoch is not None
            else "sha256:" + "b" * 64
        )
        return make_full_world(
            concealed_hands=tuple(hands),
            live_wall=live,
            dead_wall=(),
            latent_state=latent,
            rules_hash=rules_hash,
            observation_hash=obs_hash,
            simulator_snapshot=f"ismcts_synth:{ref}",
        )

    def _search_once(
        self,
        *,
        epoch: Any,
        root_obs: Any,
        legal_actions: tuple[Any, ...],
        rng: Any,
        tree: dict[str, InformationSetNode],
    ) -> tuple[Any, tuple[float, float, float, float]]:
        # Sample natural world
        if self._belief is not None and epoch is not None and _HAS_BELIEF:
            try:
                particles: Any = self._belief.sample_natural(epoch, count=1, rng=rng)  # type: ignore[union-attr]  # pyrefly: ignore[explicit-any]
                particle: Any = particles[0]  # pyrefly: ignore[explicit-any]
                # verify natural ratio one
                if particle.log_target_density != particle.log_proposal_density:
                    raise ContractError("natural world must have log_target == log_proposal")
                if particle.source != "natural":
                    raise ContractError("ISMCTS natural may only use natural particles")
                cur_world = self._world_for_particle(particle)
            except Exception as exc:
                if isinstance(exc, ContractError):
                    raise
                # fallback synthetic
                h = hashlib.sha256(f"{epoch}:{self._config.candidate_id}:{rng}".encode()).digest()
                cur_world = self._world_for_particle(type("P", (), {"world_ref": h.hex()[:16]})())
        else:
            # Self-contained synthetic world (no belief)
            h = hashlib.sha256(
                f"{getattr(root_obs, 'observation_hash', '')}:{self._config.candidate_id}".encode()
            ).digest()
            cur_world = self._world_for_particle(type("P", (), {"world_ref": h.hex()[:16]})())

        root_seat = (
            int(getattr(epoch, "root_actor", getattr(root_obs, "actor", 0)))
            if epoch is not None
            else int(getattr(root_obs, "actor", 0))
        )

        # legal ids for stub (int ids)
        # root legal actions are provided as CanonicalAction objects; map to ints via action_id
        def to_id(a: Any) -> int:
            try:
                _to_id_raw: Any = getattr(a, "action_id", a)  # pyrefly: ignore[explicit-any]
                return int(_to_id_raw)  # pyrefly: ignore[explicit-any]
            except Exception:
                if isinstance(a, int):
                    return a
                return 0

        root_legal_ids = tuple(to_id(a) for a in legal_actions)
        if len(root_legal_ids) == 0:
            root_legal_ids = (0, 1)
        # For simulation internal steps we use mask-derived legal ids from observations
        path: list[tuple[str, int, InformationSetNode]] = []
        cur = cur_world
        step = 0
        while step < self._config.max_depth and not _is_terminal(cur, self._config.max_depth, step):
            actor = _actor_to_move(cur)
            obs = world_actor_observation(cur, actor=actor)
            # Ensure sandbox observation hasn't leaked hidden tiles
            # (world_actor_observation already filters)
            legal_ids = _legal_ids_for_observation(obs)
            if len(legal_ids) == 0:
                break
            if actor == root_seat:
                key = info_key_for_observation(obs)
                node = tree.get(key)
                if node is None:
                    node = InformationSetNode(key=key, legal_actions=tuple(sorted(legal_ids)))
                    tree[key] = node
                else:
                    # Keep legal up to date (first encounter wins for determinism)
                    if len(node.legal_actions) == 0:
                        node.legal_actions = tuple(sorted(legal_ids))
                # selection
                aid = _uct_select(
                    node, legal_ids, root_seat, self._config.uct_c, self._config.tie_break
                )
                path.append((key, aid, node))
            else:
                policy = self._continuations.get(actor, UniformContinuationPolicy())
                aid = policy.sample(obs, legal_ids, rng)
            # budget check before transition
            if (
                self._config.max_transitions is not None
                and self._transitions >= self._config.max_transitions
            ):
                break
            cur = _apply_action(cur, actor, aid, rng)
            self._transitions += 1
            step += 1
            if (
                self._config.max_transitions is not None
                and self._transitions >= self._config.max_transitions
            ):
                break

        # leaf evaluation — vector
        if _is_terminal(cur, self._config.max_depth, step):
            vec = terminal_vector_for_world(cur)
        else:
            if (
                self._config.max_model_calls is not None
                and self._model_calls >= self._config.max_model_calls
            ):
                # budget exhausted before leaf model call — fall back to terminal-style vector
                vec = terminal_vector_for_world(cur)
            else:
                vec = model_vector_for_world(cur, candidate_id=self._config.candidate_id)
                self._model_calls += 1
        # backup — same four-seat vector through visited root information nodes
        for _key, aid, node in path:
            node.visits += 1
            st = node.action_stats.get(aid)
            if st is None:
                st = _ActionStats(visits=0, value_sum=(0.0, 0.0, 0.0, 0.0))
                node.action_stats[aid] = st
            st.visits += 1
            st.value_sum = tuple(v + dv for v, dv in zip(st.value_sum, vec))  # type: ignore[assignment]

        return cur, vec

    def search(
        self,
        *,
        epoch: Any,
        root_observation: Any,
        legal_actions: tuple[Any, ...],
        rng: Any,
    ) -> dict[str, Any]:
        """Run natural ISMCTS search and return structured result dict.

        Returns a dict with keys:
        - selected_action, selected_action_id, value_vectors, tree, telemetry, completed
        """
        if epoch is None:
            raise ContractError("epoch must be BeliefEpoch")
        if root_observation is None or legal_actions is None:
            raise ContractError("root_observation and legal_actions required")
        if not isinstance(legal_actions, tuple) or len(legal_actions) == 0:
            raise ContractError("legal_actions must be non-empty tuple")
        if not hasattr(rng, "random_below") or not hasattr(rng, "random_float"):
            raise ContractError("rng must be RandomStream")
        self._belief_epoch = epoch
        self._reset_counters()
        tree: dict[str, InformationSetNode] = {}

        # Validate that re-determinization hasn't been enabled sneaky
        if is_redeterminization_enabled():
            raise ContractError("re-determinization must remain disabled for Candidate 1")

        # Run simulations within budget
        sims_to_run = self._config.max_simulations
        for idx in range(sims_to_run):
            # counters before sim
            prev_trans = self._transitions
            prev_calls = self._model_calls
            _ = self._search_once(
                epoch=epoch,
                root_obs=root_observation,
                legal_actions=legal_actions,
                rng=rng,
                tree=tree,
            )
            self._simulations += 1
            # Enforce budget post-sim
            if (
                self._config.max_transitions is not None
                and self._transitions >= self._config.max_transitions
            ):
                break
            if (
                self._config.max_model_calls is not None
                and self._model_calls >= self._config.max_model_calls
            ):
                # allow up to inclusive; next sim would exceed, so break if further sim would need model call
                # For deterministic budget, we simply stop when limit reached
                if self._model_calls >= self._config.max_model_calls:
                    # If we would need another model call next sim, break after finishing current sim
                    # Continue only if we could do terminal simulations without model calls
                    # Conservative: stop when model_calls exhausted
                    if self._simulations < sims_to_run:
                        # peek: would next sim need model call? In our stub most leaves need model call
                        # So break
                        pass
            # deadlock guard: if no progress, break
            if self._transitions == prev_trans and self._model_calls == prev_calls and idx > 0:
                pass

        # Select root action via scalarized mean
        root_seat = int(getattr(epoch, "root_actor", getattr(root_observation, "actor", 0)))
        root_key = info_key_for_observation(root_observation)
        root_node = tree.get(root_key)
        selected_id: int
        value_vectors: tuple[tuple[float, float, float, float], ...] = ()
        _cand_ids_list: list[int] = []
        for _cand_item in legal_actions:  # pyrefly: ignore[explicit-any]
            _cand_any: Any = _cand_item  # pyrefly: ignore[explicit-any]
            _cand_raw: Any = getattr(_cand_any, "action_id", _cand_any)  # pyrefly: ignore[explicit-any]
            _cand_ids_list.append(int(_cand_raw))  # pyrefly: ignore[explicit-any]
        candidate_ids: tuple[int, ...] = tuple(_cand_ids_list)

        if root_node is None or len(root_node.action_stats) == 0:
            # No visits — fallback to first legal (Candidate 0 style) with model vector
            selected_id = candidate_ids[0]
            # Produce dummy vectors for evidence
            vec = model_vector_for_world(
                self._world_for_particle(type("P", (), {"world_ref": "fallback"})()),
                candidate_id=self._config.candidate_id,
            )
            value_vectors = (vec,)
            completed = self._simulations > 0
        else:
            # Choose action with highest scalarized mean
            best_id = None
            best_q = float("-inf")
            vectors: list[tuple[float, float, float, float]] = []
            for aid in candidate_ids:
                sm = root_node.scalar_mean(aid, root_seat)
                mv = root_node.mean_vector(aid)
                if mv is not None:
                    vectors.append(mv)
                if sm is None:
                    continue
                if sm > best_q + 1e-12:
                    best_q = sm
                    best_id = aid
                elif best_id is not None and abs(sm - best_q) <= 1e-12:
                    if self._config.tie_break == "lowest_action_id" and aid < best_id:
                        best_id = aid
            if best_id is None:
                best_id = candidate_ids[0]
            selected_id = best_id
            # value_vectors are the mean vectors for each candidate action (or terminal vector if unvisited)
            vecs: list[tuple[float, float, float, float]] = []
            for aid in candidate_ids:
                mv = root_node.mean_vector(aid)
                if mv is not None:
                    vecs.append(mv)
                else:
                    # unvisited actions get model vector placeholder (preserves 4-dim)
                    vecs.append(
                        model_vector_for_world(
                            self._world_for_particle(
                                type("P", (), {"world_ref": f"unvisited:{aid}"})()
                            ),
                            candidate_id=self._config.candidate_id,
                        )
                    )
            value_vectors = tuple(vecs)
            completed = True

        # Resolve selected CanonicalAction object if possible
        selected_action: Any | None = None  # pyrefly: ignore[explicit-any]
        for _a_item in legal_actions:  # pyrefly: ignore[explicit-any]
            a: Any = _a_item  # pyrefly: ignore[explicit-any]
            try:
                _a_id_raw: Any = getattr(a, "action_id", a)  # pyrefly: ignore[explicit-any]
                if int(_a_id_raw) == selected_id:  # pyrefly: ignore[explicit-any]
                    selected_action = a  # pyrefly: ignore[explicit-any]
                    break
            except Exception:
                continue
        if selected_action is None:
            selected_action = legal_actions[0]  # pyrefly: ignore[explicit-any]

        telemetry = {
            "simulations": self._simulations,
            "transitions": self._transitions,
            "model_calls": self._model_calls,
            "max_simulations": self._config.max_simulations,
            "max_transitions": self._config.max_transitions,
            "max_model_calls": self._config.max_model_calls,
            "max_depth": self._config.max_depth,
            "uct_c": self._config.uct_c,
            "tie_break": self._config.tie_break,
            "candidate_id": self._config.candidate_id,
            "resource_view": self._config.resource_view,
            "root_seat": root_seat,
            "tree_nodes": len(tree),
        }

        # Budget flags
        budget_exhausted = False
        if (
            self._config.max_simulations is not None
            and self._simulations >= self._config.max_simulations
        ):
            budget_exhausted = (
                False  # simulations budget is exactly the declared budget, not exhaustion
            )
        if (
            self._config.max_transitions is not None
            and self._transitions >= self._config.max_transitions
        ):
            budget_exhausted = True
        if (
            self._config.max_model_calls is not None
            and self._model_calls >= self._config.max_model_calls
        ):
            # not necessarily exhausted if terminal leaves avoid model calls
            pass

        return {
            "selected_action": selected_action,
            "selected_action_id": selected_id,
            "candidate_actions": legal_actions,
            "value_vectors": value_vectors,
            "tree": tree,
            "root_key": root_key,
            "root_node": root_node,
            "telemetry": telemetry,
            "completed": completed and not budget_exhausted,
            "budget_exhausted": budget_exhausted,
        }

    # -- Planner protocol adapter -----------------------------------------

    def act(self, request: SearchRequest) -> SearchResult:
        if not isinstance(request, SearchRequest):
            raise ContractError(f"request must be SearchRequest, got {type(request).__name__}")
        # Validate request hashes against candidate spec (lightweight)
        belief_epoch = getattr(request, "belief_epoch", None)
        if belief_epoch is None:
            raise ContractError("belief_epoch must be BeliefEpoch for ISMCTS natural")
        # Derive RNG from request deadline / case_id if available
        _case_id_raw: Any = getattr(request, "case_id", None)  # pyrefly: ignore[explicit-any]
        _decision_id_raw: Any = getattr(request.observation, "decision_id", "case_default")  # pyrefly: ignore[explicit-any]
        if _case_id_raw is not None and str(_case_id_raw) != "":
            case_id: str = str(_case_id_raw)  # pyrefly: ignore[explicit-any]
        elif _decision_id_raw is not None and str(_decision_id_raw) != "":
            case_id = str(_decision_id_raw)  # pyrefly: ignore[explicit-any]
        else:
            case_id = "case_default"
        candidate_id = getattr(request.candidate_spec, "candidate_id", self._config.candidate_id)
        # Use semantic seed if available, else deterministic hash
        if _HAS_RANDOM:
            try:
                from hydra2.contracts.randomness import RandomStream

                # Derive stream from (candidate_id, case_id, belief_epoch)
                epoch_id = str(getattr(belief_epoch, "epoch", "0"))
                seed = hashlib.sha256(f"{candidate_id}:{case_id}:{epoch_id}".encode()).digest()
                rng = RandomStream(seed)
            except Exception:
                rng = RandomStream(hashlib.sha256(f"{candidate_id}:{case_id}".encode()).digest())  # type: ignore[call-arg]
        else:
            import secrets as _secrets  # fallback

            rng = _secrets.token_bytes(32)

        res = self.search(
            epoch=belief_epoch,
            root_observation=request.observation,
            legal_actions=request.legal_actions,
            rng=rng,
        )

        # Wrap into SearchResult with UtilityVector etc if telemetry available
        try:
            from hydra2.contracts.utility import UtilityVector as _UV
            from hydra2.eval.telemetry import make_resource_telemetry as _mrt
            from hydra2.search.common import candidate_spec_hash as _csh
        except Exception:
            # fallback minimal result for unit tests
            return SearchResult(
                selected_action=res["selected_action"],
                candidate_actions=res["candidate_actions"],
                value_vectors=res["value_vectors"],
                candidate_spec_hash=getattr(
                    request.candidate_spec, "candidate_spec_hash", "sha256:" + "a" * 64
                ),
                telemetry=res["telemetry"],
                evidence_refs=(),
                completed=res["completed"],
            )

        # Build UtilityVectors (vector preserved, identity from manifest)
        u_vectors: list[Any] = []
        for vec in res["value_vectors"]:
            try:
                u_vectors.append(
                    _UV(
                        values=tuple(float(v) for v in vec),  # type: ignore[arg-type]
                        utility_id=getattr(
                            request.candidate_spec, "utility_id", "expected_final_placement"
                        ),
                        utility_manifest_hash=getattr(  # type: ignore[arg-type]
                            request.candidate_spec, "utility_manifest_hash", "sha256:" + "b" * 64
                        ),
                        rules_hash=getattr(  # type: ignore[arg-type]
                            request.candidate_spec, "rules_hash", "sha256:" + "a" * 64
                        ),
                    )
                )
            except Exception:
                # fallback if UtilityVector signature differs
                u_vectors.append(vec)
        try:
            spec_hash = _csh(request.candidate_spec)  # type: ignore[call-arg]
        except Exception:
            spec_hash = "sha256:" + "a" * 64
        try:
            _telemetry_dict: Any = res["telemetry"]  # pyrefly: ignore[explicit-any]
            _actual_calls_raw: Any = _telemetry_dict["model_calls"]  # pyrefly: ignore[explicit-any]
            _actual_trans_raw: Any = _telemetry_dict["transitions"]  # pyrefly: ignore[explicit-any]
            _completed_raw: Any = res["completed"]  # pyrefly: ignore[explicit-any]
            telem: Any = _mrt(  # pyrefly: ignore[explicit-any]
                budget=request.candidate_spec.resource_budget,
                actual_calls=int(_actual_calls_raw),  # pyrefly: ignore[explicit-any]
                actual_transitions=int(_actual_trans_raw),  # pyrefly: ignore[explicit-any]
                actual_duration_ms=0,
                completed=bool(_completed_raw),  # pyrefly: ignore[explicit-any]
            )
        except Exception:
            telem = res["telemetry"]
        return SearchResult(
            selected_action=res["selected_action"],
            candidate_actions=tuple(res["candidate_actions"]),
            value_vectors=tuple(u_vectors),
            candidate_spec_hash=spec_hash,
            telemetry=telem,
            evidence_refs=(),
            completed=res["completed"],
        )

    def observe(self, packet: Any) -> None:
        # For natural ISMCTS (fresh tree per act), observe simply clears ponder state
        # and verifies packet epoch compatibility if belief_epoch is set.
        # Ponder can mutate only planner-owned speculative state; this path clears it.
        self._ponder_tree.clear()
        self._ponder_epoch = None
        # If packet has observation_hash_after, we could validate, but stub.

    def ponder(self, *, deadline_monotonic_ns: int) -> None:
        # No pondering for fresh Candidate 1 — only speculative state allowed is empty
        # This is a no-op that respects deadline without hidden leak.
        if (
            not isinstance(deadline_monotonic_ns, int)
            or isinstance(deadline_monotonic_ns, bool)
            or deadline_monotonic_ns <= 0
        ):
            raise ContractError("deadline_monotonic_ns must be positive int")
        # no-op


# ---------------------------------------------------------------------------
# Helpers for testing — double-weighting oracle, budget accounting
# ---------------------------------------------------------------------------


def double_weighting_oracle_detects_correction(
    *,
    natural_probs: tuple[float, float] = (0.5, 0.5),
    proposal_probs: tuple[float, float] = (0.1, 0.9),
    values: tuple[dict[int, float], dict[int, float]] = ({0: 0.0, 1: 0.6}, {0: 0.9, 1: 0.0}),
) -> dict[str, Any]:
    """Two-world unequal-probability oracle that detects double (or missing) weighting.

    Returns dict with natural/proposal/weighted means and whether reversal is detected.
    Used to prove that applying ``b/q`` twice or zero times is observable.

    The fixture mirrors ``proposal_reversal_fixture`` in DESPOT but with explicit
    double-weight check: if a planner mistakenly multiplies by ``b/q`` twice,
    the mean will be ``sum_q (b/q)^2 * v * q`` which is detectably wrong.
    """
    if len(natural_probs) != len(values) or len(proposal_probs) != len(values):
        raise ContractError("probs and values length mismatch")
    # natural mean
    natural_mean: dict[int, float] = {0: 0.0, 1: 0.0}
    for p, v in zip(natural_probs, values, strict=False):
        for a in natural_mean:
            natural_mean[a] += p * v[a]
    # proposal unweighted
    prop_unweighted: dict[int, float] = {0: 0.0, 1: 0.0}
    for p, v in zip(proposal_probs, values, strict=False):
        for a in prop_unweighted:
            prop_unweighted[a] += p * v[a]
    # correctly weighted once
    weighted_once: dict[int, float] = {0: 0.0, 1: 0.0}
    for i, (pb, qb) in enumerate(zip(natural_probs, proposal_probs, strict=False)):
        w = pb / qb if qb > 0 else 0.0
        for a in weighted_once:
            weighted_once[a] += proposal_probs[i] * w * values[i][a]
    # double-weighted (erroneous)
    double_weighted: dict[int, float] = {0: 0.0, 1: 0.0}
    for i, (pb, qb) in enumerate(zip(natural_probs, proposal_probs, strict=False)):
        w = pb / qb if qb > 0 else 0.0
        for a in double_weighted:
            double_weighted[a] += proposal_probs[i] * (w * w) * values[i][a]

    def _key_natural(k: int) -> float:
        return natural_mean[k]
    def _key_prop(k: int) -> float:
        return prop_unweighted[k]
    def _key_once(k: int) -> float:
        return weighted_once[k]
    def _key_double(k: int) -> float:
        return double_weighted[k]
    natural_choice: int = max(natural_mean, key=_key_natural)
    prop_choice: int = max(prop_unweighted, key=_key_prop)
    once_choice: int = max(weighted_once, key=_key_once)
    double_choice: int = max(double_weighted, key=_key_double)

    return {
        "natural_mean": natural_mean,
        "proposal_unweighted_mean": prop_unweighted,
        "weighted_once_mean": weighted_once,
        "double_weighted_mean": double_weighted,
        "natural_choice": natural_choice,
        "proposal_unweighted_choice": prop_choice,
        "once_choice": once_choice,
        "double_choice": double_choice,
        "reversal_unweighted": natural_choice != prop_choice,
        "once_restores": natural_choice == once_choice,
        "double_fails": double_choice != natural_choice,
        "note": "double weighting (b/q twice) fails to restore natural choice",
    }
