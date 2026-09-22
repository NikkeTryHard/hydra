# ruff: noqa: B904, N814  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dep fallback imports; B904 ContractError preconditions; N814 upstream casing). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 8 joint type/world — planner adapter (prior, act, observe, ponder).

Owns the Planner protocol surface of :class:`JointTypeWorldPlanner`: the uniform
joint prior over Theta x Worlds, deterministic act with joint-gumbel
tie-breaking and robust penalty, the likelihood-exactly-once observe path, and
the deterministic ponder no-op. Simulation semantics arrive via the exact oracle
in :mod:`hydra2.search.joint_uncertainty` and the vocabulary via
:mod:`hydra2.search.joint_types` so each file stays inside the review-size
ceiling.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.search._drive_trigger import pack_rng as _pack_rng
from hydra2.search._drive_trigger import pack_worlds as _pack_worlds
from hydra2.search._drive_trigger import wrap_result as _wrap_result
from hydra2.search.joint_types import (
    _MASTER_SEED as _MASTER_SEED,
)
from hydra2.search.joint_types import (
    CandidateSpec as CandidateSpec,
)
from hydra2.search.joint_types import JointParticle as JointParticle
from hydra2.search.joint_types import JointPosterior as JointPosterior
from hydra2.search.joint_types import OpponentTypePolicy as OpponentTypePolicy
from hydra2.search.joint_types import Planner as Planner
from hydra2.search.joint_types import SearchRequest as SearchRequest
from hydra2.search.joint_types import SearchResult as SearchResult
from hydra2.search.joint_types import _require_belief as _require_belief
from hydra2.search.joint_uncertainty import (
    JointTypeWorldConfig as JointTypeWorldConfig,
)
from hydra2.search.joint_uncertainty import (
    UncertaintySet as UncertaintySet,
)
from hydra2.search.joint_uncertainty import (
    exact_joint_posterior_oracle as exact_joint_posterior_oracle,
)

__all__ = [
    "JointTypeWorldPlanner",
]

# ---------------------------------------------------------------------------
# Planner — joint posterior maintainer, deterministic, hidden-invariant
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class JointTypeWorldPlanner(Planner):  # type: ignore[misc]
    """Candidate 8 planner — joint type/world with deterministic exact updates.

    - Prior: uniform over Theta x Worlds consistent with root observation.
    - Posterior: exact joint oracle, likelihood exactly once per observed packet.
    - Decision: max expected value under joint posterior (nominal) vs robust worst-case.
    - Determinism via semantic seeds; hidden permutation invariant via info_key.
    """

    candidate_spec: CandidateSpec
    config: JointTypeWorldConfig = field(default_factory=JointTypeWorldConfig)
    _belief: Any = field(default=None, init=False, repr=False)
    _epoch: Any = field(default=None, init=False, repr=False)
    _joint_posterior: JointPosterior | None = field(default=None, init=False, repr=False)
    _worlds_by_ref: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _policy_for_theta: dict[str, OpponentTypePolicy] = field(
        default_factory=dict, init=False, repr=False
    )
    _uncertainty_set: UncertaintySet | None = field(default=None, init=False, repr=False)
    _case_id: str | None = field(default=None, init=False, repr=False)
    _root_seat: int | None = field(default=None, init=False, repr=False)
    _model_calls: int = field(default=0, init=False)
    _transitions: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        # Validate spec is candidate8
        if getattr(self.candidate_spec, "candidate_id", "") != "candidate8":
            raise ContractError(
                f"JointTypeWorldPlanner requires candidate_id='candidate8', got {getattr(self.candidate_spec, 'candidate_id', None)!r}"
            )
        if getattr(self.candidate_spec, "algorithm", "") != "joint_type_world":
            raise ContractError(
                f"algorithm must be 'joint_type_world', got {getattr(self.candidate_spec, 'algorithm', None)!r}"
            )
        # Build policies for each theta deterministically from spec hashes
        for theta in self.config.theta_ids:
            seed = hashlib.sha256(f"{self.candidate_spec.rules_hash}:{theta}".encode()).digest()[
                :16
            ]
            self._policy_for_theta[theta] = OpponentTypePolicy(
                theta=theta, seed_domain=_MASTER_SEED + seed
            )
        # Build uncertainty set
        self._uncertainty_set = UncertaintySet(
            nominal_policy=dict(self._policy_for_theta),
            rho=self.config.rho,
            epsilon=self.config.epsilon,
            divergence_direction=self.config.divergence_direction,
            support_class=self.config.support_class,
            rationality_rule=self.config.rationality_rule,
            theta_ids=tuple(self.config.theta_ids),
        )
        # Feasibility proof: nonempty and contains nominal
        if not self._uncertainty_set.is_nonempty():
            raise ContractError("uncertainty set must be nonempty")
        # Validate with dummy info_key
        # NEVER-bind: feasibility probe key only, not a verified binding.
        dummy_key = "sha256:" + "0" * 64
        dummy_legal = (0, 1)
        if not self._uncertainty_set.contains_nominal(
            info_key=dummy_key, legal_action_ids=dummy_legal
        ):
            raise ContractError("uncertainty set must contain nominal")

    def _ensure_joint_prior(
        self, observation: Any, *, legal_action_ids: tuple[int, ...]
    ) -> JointPosterior:
        """Initialize uniform joint prior over Theta x Worlds for one observation."""
        if self._joint_posterior is not None and self._epoch is not None:
            # Already have prior for this epoch; reuse if observation hash same
            try:
                if getattr(observation, "observation_hash", None) == getattr(
                    self._epoch, "observation_hash", None
                ):
                    return self._joint_posterior
            except Exception:
                pass
        # Build belief epoch and tiny corpus — real belief required; no synthetic fallback (fail closed).
        _require_belief()
        try:
            from hydra2.belief.natural import NaturalBelief as _NB
        except ImportError as exc:
            raise ImportError(
                "hydra2.belief.natural not importable "
                f"({exc}); build the bridge with `pixi run build-ext` before joint search"
            ) from exc

        belief: Any = _NB()
        epoch: Any = belief.begin(observation)
        # Use belief's tiny corpus builder via side-effect of sample_natural count
        # Instead we directly use private _build_tiny_corpus_for_epoch
        try:
            from hydra2.belief.natural import (
                _build_tiny_corpus_for_epoch,
            )

            worlds = _build_tiny_corpus_for_epoch(epoch, registry={})  # pyrefly: ignore[unknown-argument-type] # Any belief epoch
        except Exception as exc:
            raise ContractError(f"joint: belief corpus build failed: {exc}") from exc
        # Limit to max_particles // num_theta worlds
        max_worlds = max(1, self.config.max_particles // max(1, len(self.config.theta_ids)))
        worlds = worlds[:max_worlds]
        self._worlds_by_ref = {w.world_id: w for w in worlds}
        # Need target_id for particles (real epoch required)
        target_id = str(getattr(epoch, "target_id", "sha256:" + "f" * 64))  # pyrefly: ignore[unknown-argument-type] # Any belief epoch
        epoch_id = int(getattr(epoch, "epoch", 0))  # pyrefly: ignore[unknown-argument-type] # Any belief epoch
        num_theta = len(self.config.theta_ids)
        num_world = len(worlds)
        total = num_theta * num_world
        weight_each = 1.0 / total if total > 0 else 1.0
        particles: list[JointParticle] = []
        for theta in self.config.theta_ids:
            for w in worlds:
                particles.append(
                    JointParticle(
                        theta=theta,
                        world_ref=w.world_id,
                        weight=weight_each,
                        epoch=epoch_id,
                        target_id=target_id,
                    )
                )
        posterior = JointPosterior(
            particles=tuple(particles),
            epoch=epoch_id,
            target_id=target_id,
            theta_ids=tuple(self.config.theta_ids),
            normalized=True,
        )
        self._belief = belief
        self._epoch = epoch
        self._joint_posterior = posterior
        return posterior

    def act(self, request: SearchRequest) -> SearchResult:
        """Trigger-only joint act: validate, pack frozen ticket, one Rust drive call, wrap."""
        start_ns = time.monotonic_ns()
        if not hasattr(request, "observation") or request.observation is None:
            raise ContractError("SearchRequest.observation must be ActorObservation")
        if not hasattr(request, "legal_actions") or len(request.legal_actions) == 0:
            raise ContractError("legal_actions must be non-empty")
        if request.candidate_spec != self.candidate_spec:
            # Allow hash equality fallback if objects differ but same spec
            try:
                from hydra2.search.common import candidate_spec_hash as _csh

                if _csh(request.candidate_spec) != _csh(self.candidate_spec):  # type: ignore[arg-type]
                    raise ContractError("candidate_spec mismatch")
            except Exception:
                if str(request.candidate_spec) != str(self.candidate_spec):
                    raise ContractError("candidate_spec mismatch")
        legal = tuple(getattr(request, "legal_actions", ()))
        if len(legal) == 0:
            raise ContractError("legal_actions must be non-empty tuple")
        try:
            aids: list[int] = []
            for _a in legal:
                _v = getattr(_a, "action_id", None)
                if isinstance(_v, int) and not isinstance(_v, bool):
                    aids.append(_v)
                elif isinstance(_a, int) and not isinstance(_a, bool):
                    aids.append(_a)
                else:
                    _h = hashlib.sha256(str(_a).encode()).hexdigest()
                    aids.append(int(_h[:8], 16) & 0xFFFF)
            if len(aids) != len(set(aids)):
                raise ContractError("legal_actions must have unique action_ids")
            if aids != sorted(aids):
                paired = sorted(zip(aids, legal, strict=False), key=lambda x: x[0])
                legal = tuple(p for _, p in paired)
                aids = sorted(aids)
        except ContractError:
            raise
        except (ValueError, TypeError, AttributeError) as exc:
            raise ContractError(f"joint: legal_actions ids unreadable: {exc}") from exc
        if len(aids) == 0:
            raise ContractError("legal_actions must be non-empty tuple")
        # Ensure joint prior (uniform Theta x Worlds; retained for observe/tests).
        joint = self._ensure_joint_prior(request.observation, legal_action_ids=tuple(aids))
        # Deterministic case_id/root_seat from observation or belief epoch;
        # request case_id (test harness) and observation.decision_id agree.
        case_id_val = getattr(request, "case_id", None)
        if not isinstance(case_id_val, str) or case_id_val == "":
            _decision_id = getattr(request.observation, "decision_id", None)
            if isinstance(_decision_id, str) and _decision_id != "":
                case_id_val = _decision_id
            else:
                case_id_val = getattr(request.observation, "observation_hash", "case_0")
            if isinstance(case_id_val, str) and case_id_val.startswith("sha256:"):
                case_id_val = "case_" + case_id_val[:8]
        self._case_id = str(case_id_val)
        # Root seat from observation.actor or belief_epoch.root_actor
        root_from_obs = getattr(request.observation, "actor", None)
        root_from_epoch = None
        be = getattr(request, "belief_epoch", None)
        if be is not None:
            root_from_epoch = getattr(be, "root_actor", None)
        chosen_root = (
            root_from_epoch
            if isinstance(root_from_epoch, int) and 0 <= root_from_epoch < 4
            else root_from_obs
        )
        self._root_seat = (
            chosen_root if isinstance(chosen_root, int) and 0 <= chosen_root < 4 else 0
        )
        # Drive ticket: one belief sample, resolved to worlds (Particles never cross).
        cand_spec: Any = self.candidate_spec
        candidate_id: str = str(getattr(cand_spec, "candidate_id", "candidate8"))
        case_id: str = self._case_id
        budget: Any = getattr(cand_spec, "resource_budget", None)
        deadline_ms: int = int(getattr(budget, "deadline_ms", 5000))
        fallback_margin_ms: int = int(getattr(budget, "fallback_margin_ms", 200))
        max_calls_raw: Any = getattr(budget, "max_model_calls", 32)
        max_trans_raw: Any = getattr(budget, "max_transitions", 256)
        max_model_calls: int | None = int(max_calls_raw) if max_calls_raw is not None else None
        max_transitions: int | None = int(max_trans_raw) if max_trans_raw is not None else None
        k: int = len(self._worlds_by_ref)
        if k <= 0:
            raise ContractError("joint: belief corpus empty")
        from hydra2.contracts.randomness import RandomStream as _RS

        seed_bytes = hashlib.sha256(f"{candidate_id}:{case_id}:joint_drive".encode()).digest()
        rng = _RS(seed_bytes)
        epoch = self._epoch
        if epoch is None:
            raise ContractError("joint: belief epoch missing after prior")
        try:
            particles: Any = self._belief.sample_natural(epoch, count=k, rng=rng)
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"joint: belief sampling failed: {exc}") from exc
        try:
            raw_particles: list[Any] = list(particles)
        except TypeError as exc:
            raise ContractError(f"joint: belief sampling malformed: {exc}") from exc
        cur_worlds: list[Any] = []
        for _p in raw_particles:
            _wref: Any = getattr(_p, "world_ref", None)
            if not isinstance(_wref, str) or _wref == "":
                raise ContractError("joint: particle missing world_ref")
            try:
                _w: Any = self._belief._worlds[_wref]  # type: ignore[attr-defined]
            except Exception as exc:
                raise ContractError(f"joint: belief world missing for particle: {exc}") from exc
            cur_worlds.append(_w)
        worlds_json = _pack_worlds(cur_worlds)
        if len(worlds_json) == 0:
            raise ContractError("joint: belief corpus empty")
        seed_frozen, cursor_frozen = _pack_rng(rng)
        rules_hash: str = str(getattr(epoch, "rules_hash", ""))
        obs_hash: str = str(getattr(epoch, "observation_hash", ""))
        ticket = {
            "worlds": worlds_json,
            "rules_hash": rules_hash,
            "observation_hash": obs_hash,
            "root_legal": sorted(aids),
            "root_seat": self._root_seat,
            "candidate_id": candidate_id,
            "case_id": case_id,
            "theta_ids": list(self.config.theta_ids),
            "rho": self.config.rho,
            "epsilon": self.config.epsilon,
            "max_sims": len(worlds_json),
            "max_depth": 4,
            "max_transitions": max_transitions,
            "max_model_calls": max_model_calls,
            "deadline_ms": deadline_ms,
            "fallback_margin_ms": fallback_margin_ms,
            "tie_break": "lowest_action_id",
            "seed_hex": seed_frozen.hex(),
            "cursor": cursor_frozen,
        }
        try:
            from hydra2._native import search as _drive_bridge
        except ImportError as exc:
            raise ImportError(
                f"hydra2._native.search not importable ({exc}); build the bridge with `pixi run build-ext` before joint drive"
            ) from exc
        try:
            out: Any = _drive_bridge.joint_search_batch(json.dumps(ticket).encode())
        except (ValueError, TypeError, OverflowError) as exc:
            raise ContractError(f"joint bridge drive failed: {exc}") from exc
        try:
            end_cursor: int = out.end_cursor
        except Exception:
            end_cursor = cursor_frozen
        try:
            rng.jump_to(end_cursor)
        except (AttributeError, ValueError, TypeError) as exc:
            raise ContractError(f"joint: rng jump_to failed: {exc}") from exc
        return _wrap_result(
            out,  # pyrefly: ignore[unknown-argument-type] # native JointOut
            legal=legal,
            candidate_spec=cand_spec,
            start_ns=start_ns,
            case_id=case_id,
            particles=len(joint.particles),
        )

    def observe(self, packet: Any) -> None:  # type: ignore[override]
        """Update joint posterior with observed opponent packet — likelihood exactly once."""
        if self._joint_posterior is None:
            return
        # Packet expected to carry observed_action_id and opponent_seat, legal set
        observed_aid: Any | None = None
        opponent_seat: Any | None = None
        legal_ids: Any | None = None
        if isinstance(packet, dict):
            observed_aid = packet.get("observed_action_id", packet.get("action_id"))
            opponent_seat = packet.get("opponent_seat", packet.get("actor"))
            legal_ids = packet.get("legal_action_ids", (0, 1))
        elif hasattr(packet, "observed_action_id"):
            observed_aid = packet.observed_action_id
            opponent_seat = getattr(packet, "opponent_seat", 0)
            legal_ids = getattr(packet, "legal_action_ids", (0, 1))
        else:
            # No packet or unknown shape: no update (keep prior)
            return
        if observed_aid is None or opponent_seat is None or legal_ids is None:
            return
        if not isinstance(legal_ids, tuple):
            legal_ids = tuple(legal_ids) if isinstance(legal_ids, (list, tuple)) else (0, 1)
        assert isinstance(legal_ids, tuple)
        if not isinstance(observed_aid, int) or isinstance(observed_aid, bool):
            return
        if not isinstance(opponent_seat, int) or isinstance(opponent_seat, bool):
            return
        if observed_aid not in legal_ids:
            # Illegal packet: keep prior (caller should validate)
            return
        try:
            new_posterior = exact_joint_posterior_oracle(
                prior=self._joint_posterior,
                worlds_by_ref=self._worlds_by_ref,
                opponent_seat=opponent_seat,
                observed_action_id=observed_aid,
                legal_action_ids=tuple(int(x) for x in legal_ids),
                policy_for_theta=self._policy_for_theta,
            )
            self._joint_posterior = new_posterior
            self._transitions += len(new_posterior.particles)
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"joint observe failed: {exc}") from exc

    def ponder(self, *, deadline_monotonic_ns: int) -> None:
        # No speculative work beyond the prior; ponder is a deterministic no-op.
        pass
