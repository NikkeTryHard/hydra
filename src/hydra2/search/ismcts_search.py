# ruff: noqa: B905  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (B905 intentionally non-strict zips; SIM102 nested contract guards). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 1 ISMCTS search driver — Rust-batch descent (inversion vertical).

The Rust ``ismcts_descent`` driver owns UCT descent, exact tiny-domain
transitions, leaf vectors, vector backup, and scalarized selection with the
GIL released. Python builds ONE batch per search (sampled worlds, per-step
info keys, per-step policy directions, CTR floats) and makes ONE bridge
call; there is no Python sim loop, no per-step crossing, and no Python
backup. The retired oracle's outputs are reproduced bit-identically
(parity file frozen goldens).

Batch precompute mirrors the retired oracle's RNG consumption exactly
(sample-then-floats per sim, action-independent skeleton): natural world
samples ride ``belief.sample_natural`` on the caller's stream, and
continuation floats ride ``rng.random_float`` in (sim, step) order for the
non-root steps actually visited. Actor sequence, live-wall depletion, and
budget stops are action-independent (hands never move, one live tile pops
per transition, turn rotates deterministically), so the skeleton needs no
actions and no transitions — only the starting world per sim.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError
from hydra2.contracts.observation_actor import observation_identity_document
from hydra2.search.ismcts_core import _MASTER_SEED as _MASTER_SEED
from hydra2.search.ismcts_core import InformationSetNode as InformationSetNode
from hydra2.search.ismcts_core import NaturalISMCTSConfig as NaturalISMCTSConfig
from hydra2.search.ismcts_core import (
    UniformContinuationPolicy as UniformContinuationPolicy,
)
from hydra2.search.ismcts_core import _ActionStats as _ActionStats
from hydra2.search.ismcts_core import _require_belief as _require_belief
from hydra2.search.ismcts_core import info_key_for_observation as info_key_for_observation
from hydra2.search.ismcts_core import is_redeterminization_enabled as is_redeterminization_enabled

__all__ = [
    "NaturalISMCTSPlannerSearchMixin",
]

# ---------------------------------------------------------------------------
# Action-independent trajectory predicates — batch precompute only.
#
# These mirror the retired oracle's actor/terminal/legal-mask semantics so
# the precomputed envelope (info keys, policy directions, float alignment)
# matches the descent bit-identically. Descent, transitions, vectors, and
# backup live in Rust; the retired Python ``_apply_action`` is deleted.
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


def _require_driver_bridge() -> Any:
    """Import the built ``search`` bridge surface (fail closed, no fallback)."""
    try:
        import hydra2_replay_rs as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2_replay_rs extension with search not importable; "
            "build the bridge with `pixi run build-ext` before ISMCTS search"
        ) from exc
    try:
        mod = _ext.search
    except AttributeError as exc:
        raise ImportError(
            "hydra2_replay_rs.search submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc
    if not hasattr(mod, "ismcts_descent"):
        raise ImportError(
            "hydra2_replay_rs.search.ismcts_descent missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        )
    return mod


def _observation_hash_for_doc(doc: dict[str, Any]) -> str:
    """Actor-observation hash for an identity doc (mirrors the policy input)."""
    return "sha256:" + hashlib.sha256(canonical_bytes(doc)).hexdigest()


def _policy_direction_for_hash(obs_hash: str) -> int:
    """Continuation tilt direction (mirrors UniformContinuationPolicy)."""
    return hashlib.sha256(obs_hash.encode()).digest()[0] & 1


class NaturalISMCTSPlannerSearchMixin:
    """Simulation half of :class:`NaturalISMCTSPlanner`.

    Split host for construction and the Rust-batch search driver; the Planner
    protocol surface arrives via the act-mixin subclass, which adds no
    overrides. Attribute access is duck-typed through the subclass.
    """

    _config: NaturalISMCTSConfig
    _belief: Any | None
    _belief_epoch: Any | None
    _candidate_spec: Any | None
    _continuations: dict[int, UniformContinuationPolicy]
    _master_seed: bytes
    _last_telemetry: Any | None
    _model_calls: int
    _transitions: int
    _simulations: int
    _ponder_tree: dict[str, InformationSetNode]
    _ponder_epoch: Any | None

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
            _cand_id_val: Any = (
                getattr(candidate_spec, "candidate_id", "candidate1")
                if candidate_spec is not None
                else "candidate1"
            )  # pyrefly: ignore[explicit-any]
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
            except Exception as exc:
                raise ContractError(
                    "ismcts: belief world missing for particle; real belief required"
                ) from exc
        raise ContractError("ismcts: belief world required; synthetic worlds removed")

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
        _require_belief()
        if self._belief is None or epoch is None:
            raise ContractError("ismcts: belief and epoch required; synthetic worlds removed")
        self._belief_epoch = epoch
        self._reset_counters()

        # Validate that re-determinization hasn't been enabled sneaky
        if is_redeterminization_enabled():
            raise ContractError("re-determinization must remain disabled for Candidate 1")

        root_seat = int(getattr(epoch, "root_actor", getattr(root_observation, "actor", 0)))
        root_key = info_key_for_observation(root_observation)
        _cand_ids_list: list[int] = []
        for _cand_item in legal_actions:  # pyrefly: ignore[explicit-any]
            _cand_any: Any = _cand_item  # pyrefly: ignore[explicit-any]
            _cand_raw: Any = getattr(_cand_any, "action_id", _cand_any)  # pyrefly: ignore[explicit-any]
            _cand_ids_list.append(int(_cand_raw))  # pyrefly: ignore[explicit-any]
        candidate_ids: tuple[int, ...] = tuple(_cand_ids_list)
        sorted_legal = sorted(candidate_ids)
        if len(set(sorted_legal)) != len(sorted_legal):
            raise ContractError("legal_actions must hold distinct ids")

        # Identity-doc template for action-independent per-step observations.
        # Only concealed_hand / live_wall_tiles_remaining / actor / turn_actor /
        # decision_id vary across steps (hands never move, one live tile pops
        # per transition, turn rotates deterministically); everything else is
        # constant for the search. Proven exact against constructed
        # observations (cheap==real on all continuation steps).
        try:
            _template_doc: dict[str, Any] = dict(
                observation_identity_document(root_observation)  # type: ignore[arg-type]
            )
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"root_observation must be ActorObservation: {exc}") from exc

        cfg = self._config
        sims_to_run = cfg.max_simulations
        max_depth = cfg.max_depth
        max_trans = cfg.max_transitions

        worlds_json: list[dict[str, Any]] = []
        keys_rows: list[list[str]] = []
        dirs_rows: list[list[int]] = []
        draws: list[float] = []
        transitions = 0

        for _ in range(sims_to_run):
            try:
                particles: Any = self._belief.sample_natural(epoch, count=1, rng=rng)  # type: ignore[union-attr]  # pyrefly: ignore[explicit-any]
                particle: Any = particles[0]  # pyrefly: ignore[explicit-any]
                if particle.log_target_density != particle.log_proposal_density:
                    raise ContractError("natural world must have log_target == log_proposal")
                if particle.source != "natural":
                    raise ContractError("ISMCTS natural may only use natural particles")
                cur_world = self._world_for_particle(particle)
            except ContractError:
                raise
            except Exception as exc:
                raise ContractError(f"ismcts: belief sampling failed: {exc}") from exc

            try:
                hands = tuple(tuple(int(t) for t in h) for h in cur_world.concealed_hands)
                live_start = tuple(int(t) for t in cur_world.live_wall)
                dead = tuple(int(t) for t in cur_world.dead_wall)
                _lat: Any = getattr(cur_world, "latent_state", {}) or {}
                _step0: Any = _lat.get("step", None) if isinstance(_lat, dict) else None
                _turn0: Any = _lat.get("turn", None) if isinstance(_lat, dict) else None
                _corp0: Any = _lat.get("corpus_idx", None) if isinstance(_lat, dict) else None
            except Exception as exc:
                raise ContractError(f"ismcts: sampled world malformed: {exc}") from exc
            live_len = len(live_start)
            actor = _actor_to_move(cur_world)

            worlds_json.append(
                {
                    "world_id": str(cur_world.world_id),
                    "hands": [list(h) for h in hands],
                    "live": list(live_start),
                    "dead": list(dead),
                    "step": (None if _step0 is None else int(_step0)),
                    "turn": (None if _turn0 is None else int(_turn0)),
                    "corpus_idx": (None if _corp0 is None else int(_corp0)),
                    "snapshot": str(cur_world.simulator_snapshot),
                }
            )
            row_keys = [""] * max_depth
            row_dirs = [0] * max_depth
            step = 0
            stepped = 0 if _step0 is None else int(_step0)
            while step < max_depth and stepped < max_depth and live_len > 0:
                if actor == root_seat:
                    if step == 0:
                        row_keys[step] = root_key
                    else:
                        row_keys[step] = _info_key_for_step(_template_doc, hands, live_len, actor)
                else:
                    d = _policy_dir_for_step(_template_doc, hands, live_len, actor)
                    row_dirs[step] = d
                    draws.append(float(rng.random_float()))
                # Budget gate before the transition (mirrors the retired loop
                # and the Rust dry-run/replay: sample-then-gate, so breaking
                # reads count; no transition on the break).
                if max_trans is not None and transitions >= max_trans:
                    break
                transitions += 1
                step += 1
                stepped += 1
                live_len -= 1
                actor = (actor + 1) % 4
                if max_trans is not None and transitions >= max_trans:
                    break
            keys_rows.append(row_keys)
            dirs_rows.append(row_dirs)
            if max_trans is not None and transitions >= max_trans:
                break

        batch = {
            "worlds": worlds_json,
            "rules_hash": str(getattr(epoch, "rules_hash", "")),
            "observation_hash": str(getattr(epoch, "observation_hash", "")),
            "root_key": root_key,
            "root_legal": sorted_legal,
            "root_seat": root_seat,
            "step_keys": keys_rows,
            "policy_dirs": dirs_rows,
            "rng_floats": draws,
            "uct_c": cfg.uct_c,
            "max_depth": max_depth,
            "max_sims": len(worlds_json),
            "max_transitions": max_trans,
            "max_model_calls": cfg.max_model_calls,
            "candidate_id": cfg.candidate_id,
            "domain": "ismcts",
            "tie_break": cfg.tie_break,
            "leaf_overrides": [],
        }
        bridge = _require_driver_bridge()
        try:
            out = bridge.ismcts_descent(json.dumps(batch).encode())
        except ImportError:
            raise
        except Exception as exc:
            raise ContractError(f"ismcts bridge descent failed: {exc}") from exc

        self._simulations = int(out.sims_run)
        self._transitions = int(out.transitions)
        self._model_calls = int(out.model_calls)

        # Rebuild the information-set tree from the Rust dump. All nodes are
        # root-seat infosets; legal masks are uniform in the tiny domain, so
        # every node carries the sorted root legal set exactly as the retired
        # loop assigned on first encounter.
        tree: dict[str, InformationSetNode] = {}
        try:
            dump = json.loads(out.tree_json)
        except Exception as exc:
            raise ContractError(f"ismcts: descent tree dump malformed: {exc}") from exc
        for entry in dump:
            try:
                key = str(entry["key"])
                node = InformationSetNode(
                    key=key,
                    visits=int(entry["visits"]),
                    legal_actions=tuple(sorted_legal),
                )
                for arm in entry["arms"]:
                    aid = int(arm["action"])
                    quad = tuple(float(v) for v in arm["sum"])
                    if len(quad) != 4:
                        raise ContractError(
                            f"ismcts: descent arm sum must hold 4 entries for {aid!r}"
                        )
                    node.action_stats[aid] = _ActionStats(
                        visits=int(arm["visits"]),
                        value_sum=(quad[0], quad[1], quad[2], quad[3]),
                    )
            except Exception as exc:
                raise ContractError(f"ismcts: descent tree dump malformed: {exc}") from exc
            tree[key] = node

        root_node = tree.get(root_key)
        rust_cands = [int(a) for a in out.candidate_ids]
        rust_vecs: list[tuple[float, float, float, float]] = []
        for vec in out.value_vectors:
            quad = tuple(float(v) for v in vec)
            if len(quad) != 4:
                raise ContractError("ismcts: descent value vector must hold 4 entries")
            rust_vecs.append((quad[0], quad[1], quad[2], quad[3]))
        if len(rust_cands) != len(rust_vecs):
            raise ContractError("ismcts: descent vectors length mismatch")
        _vec_by_aid = dict(zip(rust_cands, rust_vecs))

        if root_node is None or len(root_node.action_stats) == 0:
            # No visits — the retired loop consulted belief here and always
            # raised for the synthetic refs; fail closed the same way the
            # descent reports it (first legal wins, zero visits).
            if len(candidate_ids) == 0:
                raise ContractError("legal_actions must be non-empty tuple")
            selected_id = candidate_ids[0]
            value_vectors: tuple[tuple[float, float, float, float], ...] = tuple(
                _vec_by_aid.get(aid, (0.0, 0.0, 0.0, 0.0)) for aid in candidate_ids
            )
            completed = self._simulations > 0
        else:
            for aid in candidate_ids:
                if aid not in root_node.action_stats:
                    raise ContractError(
                        f"ismcts: candidate action {aid} unvisited after "
                        f"{self._simulations} simulations; real belief required"
                    )
            selected_id = int(out.selected_id)
            value_vectors = tuple(_vec_by_aid[aid] for aid in candidate_ids)
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
            "max_simulations": cfg.max_simulations,
            "max_transitions": cfg.max_transitions,
            "max_model_calls": cfg.max_model_calls,
            "max_depth": cfg.max_depth,
            "uct_c": cfg.uct_c,
            "tie_break": cfg.tie_break,
            "candidate_id": cfg.candidate_id,
            "resource_view": cfg.resource_view,
            "root_seat": root_seat,
            "tree_nodes": len(tree),
        }

        # Budget flags
        budget_exhausted = False
        if cfg.max_simulations is not None and self._simulations >= cfg.max_simulations:
            budget_exhausted = (
                False  # simulations budget is exactly the declared budget, not exhaustion
            )
        if cfg.max_transitions is not None and self._transitions >= cfg.max_transitions:
            budget_exhausted = True
        if cfg.max_model_calls is not None and self._model_calls >= cfg.max_model_calls:
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


def _doc_for_step(
    template: dict[str, Any],
    hands: tuple[tuple[int, ...], ...],
    live_len: int,
    actor: int,
) -> dict[str, Any]:
    """Action-independent observation identity doc for one descent step."""
    doc = dict(template)
    hand = hands[actor]
    doc["concealed_hand"] = sorted(hand)
    doc["live_wall_tiles_remaining"] = live_len
    doc["actor"] = actor
    doc["turn_actor"] = actor
    doc["decision_id"] = f"dec_hand_{'_'.join(str(t) for t in hand)}_{actor}"
    return doc


def _policy_dir_for_step(
    template: dict[str, Any],
    hands: tuple[tuple[int, ...], ...],
    live_len: int,
    actor: int,
) -> int:
    """Continuation tilt direction for one step (mirrors the policy)."""
    doc = _doc_for_step(template, hands, live_len, actor)
    obs_hash = _observation_hash_for_doc(doc)
    return _policy_direction_for_hash(obs_hash)


def _info_key_for_step(
    template: dict[str, Any],
    hands: tuple[tuple[int, ...], ...],
    live_len: int,
    actor: int,
) -> str:
    """Information-set key for one step (mirrors info_key_for_observation)."""
    doc = _doc_for_step(template, hands, live_len, actor)
    payload = canonical_bytes({k: v for k, v in doc.items() if k != "legal_mask"})
    return "sha256:" + hashlib.sha256(payload).hexdigest()
