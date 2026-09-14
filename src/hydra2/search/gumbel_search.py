# ruff: noqa: SIM102, B905  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dep fallback imports; SIM102 nested contract guards; B905 intentionally non-strict action/legal zips; N814 upstream belief symbol casing). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 6 Gumbel search loop — continuation policy and sequential halving.

Owns the frozen ``UniformContinuationPolicy`` actor-view sampler beside
its only callers, and the construction plus sequential-halving search
half of :class:`GumbelSearchPlanner`: particle-to-world materialization,
single-rollout exact descent with vector backup, and the budgeted search
driver returning the structured result dict. The Planner protocol adapter
arrives via the act-mixin subclass in :mod:`hydra2.search.gumbel_act` so
each file stays inside the review-size ceiling.
"""

from __future__ import annotations

import hashlib
from typing import Any

from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.common import VisibilityViolationError as VisibilityViolationError
from hydra2.search.gumbel_config import GumbelSearchConfig as GumbelSearchConfig
from hydra2.search.gumbel_config import _ActionStats as _ActionStats
from hydra2.search.gumbel_core import _HAS_BELIEF as _HAS_BELIEF
from hydra2.search.gumbel_core import _HAS_RANDOM as _HAS_RANDOM
from hydra2.search.gumbel_core import _MASTER_SEED as _MASTER_SEED
from hydra2.search.gumbel_core import _actor_to_move as _actor_to_move
from hydra2.search.gumbel_core import _is_terminal as _is_terminal
from hydra2.search.gumbel_core import _legal_ids_for_observation as _legal_ids_for_observation
from hydra2.search.gumbel_core import deterministic_root_gumbels as deterministic_root_gumbels
from hydra2.search.gumbel_core import exact_transition as exact_transition
from hydra2.search.gumbel_core import make_full_world as make_full_world
from hydra2.search.gumbel_core import model_vector_for_world as model_vector_for_world
from hydra2.search.gumbel_core import scalarize_vector as scalarize_vector
from hydra2.search.gumbel_core import terminal_vector_for_world as terminal_vector_for_world
from hydra2.search.gumbel_core import world_actor_observation as world_actor_observation

__all__ = [
    "GumbelSearchPlannerSearchMixin",
    "UniformContinuationPolicy",
]


# ---------------------------------------------------------------------------
# Continuation policy — frozen legal-masked actor view (sandbox)
# ---------------------------------------------------------------------------


class UniformContinuationPolicy:
    """Frozen continuation for non-root seats — actor observation only."""

    def __init__(self, *, bias_strength: float = 0.2) -> None:
        if not isinstance(bias_strength, float) or not 0 <= bias_strength < 0.5:
            raise ContractError("bias_strength must be float in [0,0.5)")
        self._bias = bias_strength

    def _distribution_for(self, observation: Any, legal: tuple[int, ...]) -> tuple[float, ...]:
        if len(legal) == 0:
            raise ContractError("legal must be non-empty")
        if len(legal) == 1:
            return (1.0,)
        try:
            _h2: Any | None = getattr(observation, "observation_hash", None)
            h: str = _h2 if isinstance(_h2, str) and _h2 != "" else ""
            digest = hashlib.sha256(h.encode()).digest()
            direction = digest[0] & 1
        except Exception:
            direction = 0
        n = len(legal)
        if n == 2:
            p0 = 0.5 + self._bias if direction == 0 else 0.5 - self._bias
            return (p0, 1.0 - p0)
        w = 1.0 / n
        return tuple(w for _ in legal)

    def distribution(self, observation: Any, legal: tuple[int, ...]) -> tuple[float, ...]:
        if observation is not None:
            try:
                from hydra2.contracts.observation import ActorObservation as _Obs

                if not isinstance(observation, _Obs):
                    raise ContractError(
                        f"policy input must be ActorObservation, got {type(observation).__name__}"
                    )
            except ImportError:
                pass
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
        if _HAS_RANDOM and hasattr(rng, "random_float"):
            r: float = float(rng.random_float())  # type: ignore[explicit-any]
        else:
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
# Gumbel Search Planner
# ---------------------------------------------------------------------------


class GumbelSearchPlannerSearchMixin:
    """Search half of :class:`GumbelSearchPlanner`.

    Split host for construction and the sequential-halving search driver;
    the Planner protocol surface arrives via the act-mixin subclass, which
    adds no overrides. Attribute access is duck-typed through the subclass.
    """

    _config: GumbelSearchConfig
    _candidate_spec: Any | None
    _belief: Any | None
    _belief_epoch: Any | None
    _continuations: dict[int, UniformContinuationPolicy]
    _master_seed: bytes
    _model_calls: int
    _transitions: int
    _simulations: int

    def __init__(
        self,
        *,
        candidate_spec: Any | None = None,
        belief: Any | None = None,
        config: GumbelSearchConfig | None = None,
        continuation_policies: dict[int, UniformContinuationPolicy] | None = None,
        master_seed: bytes = _MASTER_SEED,
    ) -> None:
        self._candidate_spec = candidate_spec
        self._belief = belief
        if config is not None:
            if not isinstance(config, GumbelSearchConfig):
                raise ContractError("config must be GumbelSearchConfig")
            self._config = config
        else:
            params: dict[str, Any] = {}
            if candidate_spec is not None and hasattr(candidate_spec, "parameters"):
                try:
                    params = dict(candidate_spec.parameters or {})
                except Exception:
                    params = {}
            # Derive halving schedule from params or defaults
            halving_rounds = int(params.get("halving_rounds", 2))
            # visits_per_round may be stored as list
            vpr = params.get("visits_per_round", (8, 8))
            if isinstance(vpr, list):
                vpr = tuple(vpr)
            if not isinstance(vpr, tuple):
                # fallback: single visit count repeated
                try:
                    v = int(params.get("visits_per_action", 8))
                    vpr = tuple(v for _ in range(halving_rounds))
                except Exception:
                    vpr = (8,) * halving_rounds
            # Ensure length matches rounds
            if len(vpr) != halving_rounds:
                # pad/truncate deterministically
                if len(vpr) < halving_rounds:
                    vpr = tuple(list(vpr) + [vpr[-1]] * (halving_rounds - len(vpr)))
                else:
                    vpr = vpr[:halving_rounds]
            self._config = GumbelSearchConfig(
                halving_rounds=halving_rounds,
                visits_per_round=vpr,
                max_depth=int(params.get("max_depth", 6)),
                max_model_calls=params.get("max_model_calls", 32),
                max_transitions=params.get("max_transitions", 64),
                tie_break=str(params.get("tie_break", "lowest_action_id")),
                candidate_id=str(getattr(candidate_spec, "candidate_id", "candidate6"))
                if candidate_spec is not None
                else "candidate6",
                resource_view=str(params.get("resource_view", "calls")),
                seed_material=master_seed,
            )
        self._continuations: dict[int, UniformContinuationPolicy] = (
            continuation_policies
            if continuation_policies is not None
            else {seat: UniformContinuationPolicy() for seat in range(4)}
        )
        self._master_seed = master_seed
        self._belief_epoch: Any | None = None
        self._model_calls: int = 0
        self._transitions: int = 0
        self._simulations: int = 0

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
        ref = getattr(particle, "world_ref", str(particle))
        h = hashlib.sha256(ref.encode()).digest()
        hands: list[tuple[int, int]] = []
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
        live = tuple(b % 136 for b in h[8:12])
        latent = {
            "step": 0,
            "turn": int(getattr(self._belief_epoch, "root_actor", 0))
            if self._belief_epoch is not None
            else 0,
        }
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
            simulator_snapshot=f"gumbel_synth:{ref}",
        )

    def _rollout(
        self,
        *,
        start_world: Any,
        root_action_id: int,
        root_seat: int,
        rng: Any,
    ) -> tuple[Any, tuple[float, float, float, float]]:
        """Exact rollout starting with forced root action, then continuation policies."""
        cur = exact_transition(start_world, root_seat, root_action_id)
        self._transitions += 1
        step = 1
        while step < self._config.max_depth and not _is_terminal(cur, self._config.max_depth, step):
            actor = _actor_to_move(cur)
            obs = world_actor_observation(cur, actor=actor)
            legal_ids = _legal_ids_for_observation(obs)
            if len(legal_ids) == 0:
                break
            policy = self._continuations.get(actor, UniformContinuationPolicy())
            aid = policy.sample(obs, legal_ids, rng)
            if (
                self._config.max_transitions is not None
                and self._transitions >= self._config.max_transitions
            ):
                break
            cur = exact_transition(cur, actor, aid)
            self._transitions += 1
            step += 1
            if (
                self._config.max_transitions is not None
                and self._transitions >= self._config.max_transitions
            ):
                break
        if _is_terminal(cur, self._config.max_depth, step):
            vec = terminal_vector_for_world(cur)
        else:
            if (
                self._config.max_model_calls is not None
                and self._model_calls >= self._config.max_model_calls
            ):
                vec = terminal_vector_for_world(cur)
            else:
                vec = model_vector_for_world(cur, candidate_id=self._config.candidate_id)
                self._model_calls += 1
        return cur, vec

    def _action_id_for(self, action: Any) -> int:
        aid = getattr(action, "action_id", None)
        if isinstance(aid, int) and not isinstance(aid, bool):
            return aid
        # Deterministic fallback: hash of canonical fields
        try:
            kind = str(getattr(action, "kind", ""))
            tile = getattr(action, "tile", None)
            called = getattr(action, "called_tile", None)
            consumed = getattr(action, "consumed_tiles", ())
            source = getattr(action, "source_seat", None)
            riichi = getattr(action, "declares_riichi", False)
            payload = f"{kind}:{tile}:{called}:{tuple(consumed) if isinstance(consumed, (list, tuple)) else consumed}:{source}:{riichi}".encode()
            h = hashlib.sha256(payload).digest()
            return int.from_bytes(h[:4], "big") & 0x7FFFFFFF  # 0 .. 2^31-1
        except Exception:
            # salted sha256 — hash() is per-process seeded (PYTHONHASHSEED), not deterministic
            h = hashlib.sha256(b"gumbel_aid_v1" + str(action).encode()).digest()
            return int.from_bytes(h[:4], "big") & 0x7FFFFFFF

    def search(
        self,
        *,
        epoch: Any,
        root_observation: Any,
        legal_actions: tuple[Any, ...],
        rng: Any,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        """Run Gumbel sequential-halving search and return structured result."""
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

        legal_ids: tuple[int, ...] = tuple(self._action_id_for(a) for a in legal_actions)  # type: ignore[explicit-any]
        if len(set(legal_ids)) != len(legal_ids):
            # Fallback to index-disambiguation for duplicate hash collisions (tile collision edge case)
            # Deterministically perturb colliding ids via index mix
            seen: dict[int, int] = {}
            new_ids: list[int] = []
            for idx, aid in enumerate(legal_ids):
                if aid in seen:
                    # mix index into hash
                    aid = (aid + idx * 1000003) & 0x7FFFFFFF
                    while aid in seen:
                        aid = (aid + 1) & 0x7FFFFFFF
                seen[aid] = 1
                new_ids.append(aid)
            legal_ids = tuple(new_ids)
        # Stable sort by id for determinism, but keep original objects mapping
        id_to_action: dict[int, Any] = {}
        for a, aid in zip(legal_actions, legal_ids, strict=False):  # type: ignore[explicit-any]
            if aid not in id_to_action:
                id_to_action[aid] = a  # type: ignore[unknown-argument-type]
        sorted_ids = tuple(sorted(legal_ids))
        # Resolve root seat and case
        root_seat = int(getattr(epoch, "root_actor", getattr(root_observation, "actor", 0)))
        if not 0 <= root_seat < 4:
            root_seat = int(getattr(root_observation, "actor", 0)) % 4
        _case_tmp: Any | None = case_id
        if _case_tmp is not None and isinstance(_case_tmp, str) and _case_tmp != "":
            cid_raw: Any = _case_tmp
        else:
            _epoch_val: Any | None = getattr(epoch, "epoch", None)
            if _epoch_val is not None and isinstance(_epoch_val, str) and _epoch_val != "":
                cid_raw = _epoch_val
            else:
                _dec_val: Any | None = getattr(root_observation, "decision_id", None)
                if _dec_val is not None and isinstance(_dec_val, str) and _dec_val != "":
                    cid_raw = _dec_val
                else:
                    cid_raw = "case_default"
        cid = str(cid_raw)
        candidate_id = getattr(self._config, "candidate_id", "candidate6")

        # Deterministic root Gumbels — SPEC 16.7: (case_id, root_seat, candidate_id, action_id)
        gumbels = deterministic_root_gumbels(
            case_id=cid, root_seat=root_seat, candidate_id=candidate_id, legal_action_ids=sorted_ids
        )

        # Per-action vector stats (four-seat)
        stats: dict[int, _ActionStats] = {aid: _ActionStats() for aid in sorted_ids}
        survivors: tuple[int, ...] = sorted_ids

        # Sequential halving rounds
        for round_idx in range(self._config.halving_rounds):
            if len(survivors) <= 1:
                break
            visits = self._config.visits_per_round[round_idx]
            # Synthetic worlds are loop-invariant in visits (observation hash,
            # candidate, aid only) and consume no rng/counters: build once per
            # (aid, round). Bit-identical: _rollout never mutates start_world.
            use_synth = not (self._belief is not None and epoch is not None and _HAS_BELIEF)
            # For each survivor, allocate visits rollouts
            for aid in survivors:
                synth_world: Any = None
                if use_synth:
                    h0 = hashlib.sha256(
                        f"{getattr(root_observation, 'observation_hash', '')}:{candidate_id}:{aid}".encode()
                    ).digest()
                    synth_world = self._world_for_particle(
                        type("P", (), {"world_ref": h0.hex()[:16]})()
                    )
                for _ in range(visits):
                    # Budget checks before rollout
                    if (
                        self._config.max_transitions is not None
                        and self._transitions >= self._config.max_transitions
                    ):
                        break
                    if (
                        self._config.max_model_calls is not None
                        and self._model_calls >= self._config.max_model_calls
                    ):
                        # Need model call for non-terminal leaf; if terminal heavy, could still continue
                        # But we enforce hard budget for determinism
                        # Allow terminal rollouts that avoid model calls
                        # So we only break if we would definitely need model call and already exhausted
                        # For simplicity, break when both budgets exhausted
                        if self._transitions >= (
                            self._config.max_transitions
                            if self._config.max_transitions is not None
                            else 10**9
                        ):
                            break
                    # Sample natural world
                    if self._belief is not None and epoch is not None and _HAS_BELIEF:
                        try:
                            particles: Any = self._belief.sample_natural(epoch, count=1, rng=rng)  # type: ignore[union-attr]
                            particle: Any = particles[0]  # type: ignore[explicit-any]
                            if particle.log_target_density != particle.log_proposal_density:
                                raise ContractError("natural world must have ratio 1")
                            if particle.source != "natural":
                                raise ContractError("gumbel natural may only use natural particles")
                            cur_world = self._world_for_particle(particle)  # type: ignore[unknown-argument-type]
                        except Exception as exc:
                            if isinstance(exc, ContractError):
                                raise
                            h = hashlib.sha256(f"{epoch}:{candidate_id}:{rng}".encode()).digest()
                            cur_world = self._world_for_particle(
                                type("P", (), {"world_ref": h.hex()[:16]})()
                            )
                    else:
                        cur_world = synth_world
                    _, vec = self._rollout(
                        start_world=cur_world, root_action_id=aid, root_seat=root_seat, rng=rng
                    )
                    # Vector backup — accumulate four-seat sum
                    st = stats[aid]
                    st.visits += 1
                    st.value_sum = tuple(v + dv for v, dv in zip(st.value_sum, vec))  # type: ignore[assignment]
                    self._simulations += 1
                    # Enforce per-round budget
                    if (
                        self._config.max_transitions is not None
                        and self._transitions >= self._config.max_transitions
                    ):
                        break
                if (
                    self._config.max_transitions is not None
                    and self._transitions >= self._config.max_transitions
                ):
                    break
            # Score survivors by gumbel + scalarized mean (vector backup -> scalarize at root only)
            scored: list[tuple[float, int]] = []
            for aid in survivors:
                st = stats[aid]
                mv = st.mean_vector()
                q = scalarize_vector(mv, root_seat) if mv is not None else float("-inf")
                g = gumbels[aid]
                # Gumbel score rule: g + q (MuZero style uses g + logits + sigma(q); we use g+q)
                score = g + q
                scored.append((score, aid))
            # Sort descending by score, tie_break deterministic
            scored.sort(
                key=lambda x: (
                    -x[0],
                    x[1]
                    if self._config.tie_break == "lowest_action_id"
                    else hashlib.sha256(f"{x[1]}".encode()).hexdigest(),
                )
            )
            # Keep ceil(n/2) survivors (sequential halving)
            keep = (len(survivors) + 1) // 2
            if keep < 1:
                keep = 1
            # If all scores are -inf (no visits), keep original order
            survivors = tuple(aid for _, aid in scored[:keep])
            # Budget exhausted -> break early
            if (
                self._config.max_transitions is not None
                and self._transitions >= self._config.max_transitions
            ):
                break
            if (
                self._config.max_model_calls is not None
                and self._model_calls >= self._config.max_model_calls
            ):
                # Budget-exhausted rounds fall through to terminal fallback
                # vectors; the return rule below picks the max-Gumbel survivor.
                pass

        # Final selection: survivor with max gumbel score
        best_id: int | None = None
        best_score = float("-inf")
        for aid in survivors:
            st = stats[aid]
            mv = st.mean_vector()
            q = scalarize_vector(mv, root_seat) if mv is not None else float("-inf")
            score = gumbels[aid] + q
            if score > best_score + 1e-12:
                best_score = score
                best_id = aid
            elif best_id is not None and abs(score - best_score) <= 1e-12:
                if self._config.tie_break == "lowest_action_id" and aid < best_id:
                    best_id = aid
                elif self._config.tie_break in ("stable_hash", "lexicographic"):
                    ha = hashlib.sha256(f"{aid}".encode()).hexdigest()
                    hb = hashlib.sha256(f"{best_id}".encode()).hexdigest()
                    if ha < hb:
                        best_id = aid
        if best_id is None:
            # Fallback: highest gumbel alone (no visits)
            best_id = (
                max(survivors, key=lambda aid: gumbels[aid])  # type: ignore[unknown-argument-type,explicit-any]
                if len(survivors) > 0
                else sorted_ids[0]
            )

        # Value vectors for each legal action (mean vectors, or placeholder for unvisited)
        vecs: list[tuple[float, float, float, float]] = []
        for aid in sorted_ids:
            mv = stats[aid].mean_vector()
            if mv is not None:
                vecs.append(mv)
            else:
                vecs.append(
                    model_vector_for_world(
                        self._world_for_particle(
                            type("P", (), {"world_ref": f"unvisited:{aid}"})()
                        ),
                        candidate_id=candidate_id,
                    )
                )
        value_vectors = tuple(vecs)

        # Resolve selected action object
        selected_action: Any = id_to_action.get(best_id, legal_actions[0])  # type: ignore[unknown-argument-type]

        telemetry = {
            "simulations": self._simulations,
            "transitions": self._transitions,
            "model_calls": self._model_calls,
            "max_simulations": sum(
                len(sorted_ids) // (2**r) * self._config.visits_per_round[r]
                if r == 0
                else ((len(sorted_ids) + (2**r - 1)) // (2**r)) * self._config.visits_per_round[r]
                for r in range(self._config.halving_rounds)
            ),
            "max_transitions": self._config.max_transitions,
            "max_model_calls": self._config.max_model_calls,
            "max_depth": self._config.max_depth,
            "halving_rounds": self._config.halving_rounds,
            "visits_per_round": self._config.visits_per_round,
            "tie_break": self._config.tie_break,
            "candidate_id": self._config.candidate_id,
            "resource_view": self._config.resource_view,
            "root_seat": root_seat,
            "gumbels": gumbels,
            "survivors": survivors,
        }

        return {
            "selected_action": selected_action,
            "selected_action_id": best_id,
            "candidate_actions": legal_actions,
            "value_vectors": value_vectors,
            "stats": stats,
            "gumbels": gumbels,
            "survivors": survivors,
            "telemetry": telemetry,
            "completed": True,
        }
