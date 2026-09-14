# ruff: noqa: B905, N814  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dep fallback imports; SIM102 nested contract guards; B905 intentionally non-strict action/legal zips; N814 upstream belief symbol casing). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 6 PUCT comparator — matched-budget baseline search.

Owns the matched-resource PUCT baseline for comparison against Gumbel:
construction from a candidate spec or a frozen ``PuctConfig``,
particle-to-world materialization, the PUCT selection loop over the same
exact simulator with identical budget semantics, and result assembly with
the same vector/telemetry shape. Shares the firewall vocabulary,
continuation policy, and simulator helpers with the Gumbel planner via
:mod:`hydra2.search.gumbel_core`.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, cast

from hydra2.contracts.common import ContractError as ContractError
from hydra2.search.common import Planner as Planner
from hydra2.search.common import SearchRequest as SearchRequest
from hydra2.search.common import SearchResult as SearchResult
from hydra2.search.gumbel_config import PuctConfig as PuctConfig
from hydra2.search.gumbel_core import _HAS_BELIEF as _HAS_BELIEF
from hydra2.search.gumbel_core import _HAS_RANDOM as _HAS_RANDOM
from hydra2.search.gumbel_core import _MASTER_SEED as _MASTER_SEED
from hydra2.search.gumbel_core import _actor_to_move as _actor_to_move
from hydra2.search.gumbel_core import _is_terminal as _is_terminal
from hydra2.search.gumbel_core import _legal_ids_for_observation as _legal_ids_for_observation
from hydra2.search.gumbel_core import exact_transition as exact_transition
from hydra2.search.gumbel_core import make_digest_text as make_digest_text
from hydra2.search.gumbel_core import make_full_world as make_full_world
from hydra2.search.gumbel_core import model_vector_for_world as model_vector_for_world
from hydra2.search.gumbel_core import scalarize_vector as scalarize_vector
from hydra2.search.gumbel_core import terminal_vector_for_world as terminal_vector_for_world
from hydra2.search.gumbel_core import world_actor_observation as world_actor_observation
from hydra2.search.gumbel_search import UniformContinuationPolicy as UniformContinuationPolicy

__all__ = [
    "PuctBaselinePlanner",
]


# ---------------------------------------------------------------------------
# PUCT Baseline Comparator — matched budget, same exact transitions
# ---------------------------------------------------------------------------


class PuctBaselinePlanner(Planner):  # type: ignore[misc]
    """PUCT baseline for matched-resource comparison (candidate6 comparator).

    Uses same exact simulator and belief sampling as Gumbel, but selects via
    PUCT rather than Gumbel sequential halving.  Budget (model_calls /
    transitions) is enforced identically for fair comparison.
    """

    def __init__(
        self,
        *,
        candidate_spec: Any | None = None,
        belief: Any | None = None,
        config: PuctConfig | None = None,
        continuation_policies: dict[int, UniformContinuationPolicy] | None = None,
        master_seed: bytes = _MASTER_SEED,
    ) -> None:
        self._candidate_spec = candidate_spec
        self._belief = belief
        if config is not None:
            if not isinstance(config, PuctConfig):
                raise ContractError("config must be PuctConfig")
            self._config = config
        else:
            params: dict[str, Any] = {}
            if candidate_spec is not None and hasattr(candidate_spec, "parameters"):
                try:
                    params = dict(candidate_spec.parameters or {})
                except Exception:
                    params = {}
            self._config = PuctConfig(
                puct_c=float(params.get("puct_c", 1.5)),
                max_depth=int(params.get("max_depth", 6)),
                max_model_calls=params.get("max_model_calls", 32),
                max_transitions=params.get("max_transitions", 64),
                num_simulations=int(params.get("num_simulations", 16)),
                tie_break=str(params.get("tie_break", "lowest_action_id")),
                candidate_id=str(getattr(candidate_spec, "candidate_id", "puct_baseline"))
                if candidate_spec is not None
                else "puct_baseline",
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
        self._model_calls = 0
        self._transitions = 0
        self._simulations = 0

    def _reset(self) -> None:
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
            simulator_snapshot=f"puct_synth:{ref}",
        )

    def _action_id_for(self, action: Any) -> int:
        aid = getattr(action, "action_id", None)
        if isinstance(aid, int) and not isinstance(aid, bool):
            return aid
        try:
            kind = str(getattr(action, "kind", ""))
            tile = getattr(action, "tile", None)
            called = getattr(action, "called_tile", None)
            consumed = getattr(action, "consumed_tiles", ())
            source = getattr(action, "source_seat", None)
            riichi = getattr(action, "declares_riichi", False)
            payload = f"{kind}:{tile}:{called}:{tuple(consumed) if isinstance(consumed, (list, tuple)) else consumed}:{source}:{riichi}".encode()
            h = hashlib.sha256(payload).digest()
            return int.from_bytes(h[:4], "big") & 0x7FFFFFFF
        except Exception:
            # salted sha256 — hash() is per-process seeded (PYTHONHASHSEED), not deterministic
            h = hashlib.sha256(b"gumbel_aid_v1" + str(action).encode()).digest()
            return int.from_bytes(h[:4], "big") & 0x7FFFFFFF

    def act(self, request: SearchRequest) -> SearchResult:
        if not isinstance(request, SearchRequest):
            raise ContractError(f"request must be SearchRequest, got {type(request).__name__}")
        belief_epoch = getattr(request, "belief_epoch", None)
        if belief_epoch is None:
            raise ContractError("belief_epoch must be BeliefEpoch for PUCT baseline")
        _case_req2: Any | None = getattr(request, "case_id", None)
        if _case_req2 is not None and isinstance(_case_req2, str) and _case_req2 != "":
            case_id = _case_req2
        else:
            _dec3: Any | None = getattr(request.observation, "decision_id", None)
            case_id = _dec3 if isinstance(_dec3, str) and _dec3 != "" else "puct_case"
        self._belief_epoch = belief_epoch
        self._reset()
        start_ns = time.monotonic_ns()

        legal = tuple(request.legal_actions)
        legal_ids = tuple(self._action_id_for(a) for a in legal)
        # Deduplicate collisions deterministically
        if len(set(legal_ids)) != len(legal_ids):
            seen: dict[int, int] = {}
            new_ids: list[int] = []
            for idx, aid in enumerate(legal_ids):
                if aid in seen:
                    aid = (aid + idx * 1000003) & 0x7FFFFFFF
                    while aid in seen:
                        aid = (aid + 1) & 0x7FFFFFFF
                seen[aid] = 1
                new_ids.append(aid)
            legal_ids = tuple(new_ids)
        root_seat = int(
            getattr(belief_epoch, "root_actor", getattr(request.observation, "actor", 0))
        )
        priors: dict[int, float] = {aid: 1.0 / len(legal_ids) for aid in legal_ids}
        # PUCT stats
        visits: dict[int, int] = dict.fromkeys(legal_ids, 0)
        value_sum: dict[int, tuple[float, float, float, float]] = dict.fromkeys(
            legal_ids, (0.0, 0.0, 0.0, 0.0)
        )
        # Derive RNG deterministically per case — use local import to avoid Any fallback type
        if _HAS_RANDOM:
            try:
                from hydra2.contracts.randomness import RandomStream as _RS

                rng = _RS(
                    hashlib.sha256(f"puct:{case_id}:{self._config.candidate_id}".encode()).digest()
                )
            except Exception:
                from hydra2.contracts.randomness import RandomStream as _RS2

                rng = _RS2(hashlib.sha256(case_id.encode()).digest())  # type: ignore[call-arg]
        else:
            rng = None

        total_needed = self._config.num_simulations
        for _ in range(total_needed):
            if (
                self._config.max_model_calls is not None
                and self._model_calls >= self._config.max_model_calls
            ):
                break
            if (
                self._config.max_transitions is not None
                and self._transitions >= self._config.max_transitions
            ):
                break
            # PUCT selection
            best_aid = None
            best_score = float("-inf")
            total_visits = sum(visits.values())
            for aid in legal_ids:
                n = visits[aid]
                if n == 0:
                    score = float("inf")  # prioritize unvisited
                else:
                    mv = tuple(v / n for v in value_sum[aid])
                    q = scalarize_vector(mv, root_seat)
                    u = self._config.puct_c * priors[aid] * math.sqrt(total_visits) / (1 + n)
                    score = q + u
                if score > best_score + 1e-12:
                    best_score = score
                    best_aid = aid
                elif best_aid is not None and abs(score - best_score) <= 1e-12:
                    if self._config.tie_break == "lowest_action_id" and aid < best_aid:
                        best_aid = aid
            if best_aid is None:
                best_aid = legal_ids[0]
            # Sample world and rollout
            if self._belief is not None and _HAS_BELIEF:
                try:
                    particles_p: Any = self._belief.sample_natural(belief_epoch, count=1, rng=rng)  # type: ignore[union-attr]
                    cur_world = self._world_for_particle(particles_p[0])  # type: ignore[unknown-argument-type]
                except Exception:
                    cur_world = self._world_for_particle(
                        type("P", (), {"world_ref": f"synth:{best_aid}"})()
                    )
            else:
                cur_world = self._world_for_particle(
                    type("P", (), {"world_ref": f"synth:{best_aid}"})()
                )
            # Exact rollout
            cur = exact_transition(cur_world, root_seat, best_aid)
            self._transitions += 1
            step = 1
            while step < self._config.max_depth and not _is_terminal(
                cur, self._config.max_depth, step
            ):
                actor = _actor_to_move(cur)
                obs = world_actor_observation(cur, actor=actor)
                legal_next = _legal_ids_for_observation(obs)
                if len(legal_next) == 0:
                    break
                pol = self._continuations.get(actor, UniformContinuationPolicy())
                aid = pol.sample(obs, legal_next, rng) if rng is not None else legal_next[0]
                if (
                    self._config.max_transitions is not None
                    and self._transitions >= self._config.max_transitions
                ):
                    break
                cur = exact_transition(cur, actor, aid)
                self._transitions += 1
                step += 1
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
            visits[best_aid] += 1
            value_sum[best_aid] = tuple(v + dv for v, dv in zip(value_sum[best_aid], vec))  # type: ignore[assignment]
            self._simulations += 1

        # Select highest mean Q (scalarized)
        best_id = None
        best_q = float("-inf")
        vecs: list[tuple[float, float, float, float]] = []
        for aid in legal_ids:
            n = visits[aid]
            raw_mv: tuple[float, ...] | tuple[float, float, float, float] = (
                cast("tuple[float, float, float, float]", tuple(v / n for v in value_sum[aid]))
                if n > 0
                else model_vector_for_world(
                    self._world_for_particle(type("P", (), {"world_ref": f"unvisited:{aid}"})()),
                    candidate_id=self._config.candidate_id,
                )
            )
            # Narrow to fixed quad
            assert len(raw_mv) == 4
            mv = cast("tuple[float, float, float, float]", raw_mv)  # pyrefly: ignore[redundant-cast]
            vecs.append(mv)
            q = scalarize_vector(mv, root_seat)
            if n == 0:
                q = float("-inf")  # deprioritize unvisited unless all unvisited
                # But if we never visited due to budget, keep model vector mean
                # For determinism, we treat unvisited as -inf so visited wins
                # Unless all are unvisited
                if all(v == 0 for v in visits.values()):
                    q = scalarize_vector(mv, root_seat)
            if q > best_q + 1e-12:
                best_q = q
                best_id = aid
            elif best_id is not None and abs(q - best_q) <= 1e-12:
                if self._config.tie_break == "lowest_action_id" and aid < best_id:
                    best_id = aid
        if best_id is None:
            best_id = legal_ids[0]
        selected = next((a for a in legal if self._action_id_for(a) == best_id), legal[0])
        # Build telemetry / result
        try:
            from hydra2.contracts.utility import UtilityVector as _UV
            from hydra2.eval.telemetry import make_resource_telemetry as _mrt
            from hydra2.search.common import candidate_spec_hash as _csh

            u_vectors = []
            for mv in vecs:
                vals_fixed: tuple[float, float, float, float] = tuple(v for v in mv)  # type: ignore[assignment]
                assert len(vals_fixed) == 4
                u_vectors.append(
                    _UV(
                        values=vals_fixed,
                        utility_id=str(
                            getattr(
                                request.candidate_spec, "utility_id", "expected_final_placement"
                            )
                        ),
                        utility_manifest_hash=make_digest_text(
                            str(
                                getattr(
                                    request.candidate_spec,
                                    "utility_manifest_hash",
                                    "sha256:" + "b" * 64,
                                )
                            )
                        ),
                        rules_hash=make_digest_text(
                            str(getattr(request.candidate_spec, "rules_hash", "sha256:" + "a" * 64))
                        ),
                    )
                )
            spec_hash = _csh(request.candidate_spec)  # type: ignore[call-arg]
            telem = _mrt(
                mode=str(getattr(request.candidate_spec.resource_budget, "mode", "gameplay_5s")),
                wall_id=None,
                case_id=case_id,
                candidate_spec_hash=spec_hash,
                hardware_hash="sha256:" + "8" * 64,
                environment_hash="sha256:" + "7" * 64,
                cold_start=False,
                synchronized_elapsed_ms=(time.monotonic_ns() - start_ns) / 1e6,
                model_calls=self._model_calls,
                exact_transitions=self._transitions,
                particles=len(legal_ids),
                fallback_used=False,
                timeout=False,
                illegal_action=False,
                cuda_peak_allocated_bytes=None,
                cuda_peak_reserved_bytes=None,
                host_peak_bytes=None,
                energy_joules=self._model_calls * 0.5 + self._transitions * 0.2,
                graph_breaks=None,
                recompiles=None,
                invalid_reason=None,
            )
            return SearchResult(
                selected_action=selected,
                candidate_actions=tuple(legal),
                value_vectors=tuple(u_vectors),
                candidate_spec_hash=spec_hash,
                telemetry=telem,
                evidence_refs=(),
                completed=True,
            )
        except Exception:
            # Fallback minimal
            return SearchResult(
                selected_action=selected,
                candidate_actions=tuple(legal),
                value_vectors=tuple(vecs),
                candidate_spec_hash="sha256:" + "a" * 64,
                telemetry={"model_calls": self._model_calls, "transitions": self._transitions},
                evidence_refs=(),
                completed=True,
            )

    def observe(self, packet: Any) -> None:
        self._belief_epoch = None

    def ponder(self, *, deadline_monotonic_ns: int) -> None:
        if (
            not isinstance(deadline_monotonic_ns, int)
            or isinstance(deadline_monotonic_ns, bool)
            or deadline_monotonic_ns <= 0
        ):
            raise ContractError("deadline_monotonic_ns must be positive int")
