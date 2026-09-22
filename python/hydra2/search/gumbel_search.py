# ruff: noqa: B905  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dep fallback imports; B905 intentionally non-strict action/legal zips; N814 upstream belief symbol casing). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 6 Gumbel search loop — continuation policy and sequential halving.

Owns the frozen ``UniformContinuationPolicy`` actor-view sampler beside
its only callers, and the construction plus Rust-batch search half of
:class:`GumbelSearchPlanner`: particle-to-world materialization and the
one-bridge-call halving driver returning the structured result dict. The
halving rounds/aids/visits loops, rollout descent, vector backup, cuts,
and final selection live in ``hydra-search:gumbel_halving_batch`` (GIL
released); Python builds ONE batch per search (sampled worlds in oracle
visit order, per-rollout policy directions, CTR policy floats, root
Gumbels) and makes ONE bridge call. There is no Python sim loop, no
per-step crossing, and no Python backup. The Planner protocol adapter
arrives via the act-mixin subclass in :mod:`hydra2.search.gumbel_act` so
each file stays inside the review-size ceiling.
"""

from __future__ import annotations

import hashlib
import json
import types
from typing import Any

from hydra2.contracts.common import (
    ContractError as ContractError,
)
from hydra2.search._drive_trigger import pack_rng as _pack_rng
from hydra2.search._drive_trigger import pack_worlds as _pack_worlds
from hydra2.search._gumbel_precompute import UniformContinuationPolicy as UniformContinuationPolicy
from hydra2.search._gumbel_precompute import _dry_float_need as _dry_float_need
from hydra2.search._gumbel_precompute import _exact_sum_for_mean as _exact_sum_for_mean
from hydra2.search._gumbel_precompute import _halving_slots as _halving_slots
from hydra2.search._gumbel_precompute import _require_driver_bridge as _require_driver_bridge
from hydra2.search.gumbel_config import (
    GumbelSearchConfig as GumbelSearchConfig,
)
from hydra2.search.gumbel_config import (
    _ActionStats as _ActionStats,
)
from hydra2.search.gumbel_core import (
    _MASTER_SEED as _MASTER_SEED,
)
from hydra2.search.gumbel_core import _legal_ids_for_observation as _legal_ids_for_observation
from hydra2.search.gumbel_core import _require_belief as _require_belief
from hydra2.search.gumbel_core import deterministic_root_gumbels as deterministic_root_gumbels

__all__ = [
    "GumbelSearchPlannerSearchMixin",
    "UniformContinuationPolicy",
]


# ---------------------------------------------------------------------------
# Gumbel Search Planner
# ---------------------------------------------------------------------------


class GumbelSearchPlannerSearchMixin:
    """Search half of :class:`GumbelSearchPlanner`.

    Split host for construction and the Rust-batch search driver; the Planner
    protocol surface arrives via the act-mixin subclass, which adds no
    overrides. Attribute access is duck-typed through the subclass.
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
        _require_belief()
        if self._belief is not None and hasattr(self._belief, "_worlds"):
            try:
                return self._belief._worlds[particle.world_ref]
            except Exception as exc:
                raise ContractError(
                    "gumbel: belief world missing for particle; real belief required"
                ) from exc
        raise ContractError("gumbel: belief world required; synthetic worlds removed")

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
        """Run Gumbel sequential-halving search and return structured result.

        Rust-batch driver: Python builds ONE batch per search (sampled worlds
        in oracle visit order, per-rollout policy directions, CTR policy
        floats, root Gumbels) and makes ONE ``search.gumbel_halving`` call
        with the GIL released. Rounds/aids/visits loops, rollout descent,
        vector backup, cuts, and final selection live in Rust bit-identically
        (parity file frozen goldens).
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

        # Rust covers only the frozen 0.2 tilt (``CONTINUATION_BIAS``); custom
        # tilts have no pyfn — fail closed with a named reason instead of
        # forking a Python fallback (no dual implementations at rest).
        for seat, pol in self._continuations.items():
            bias = getattr(pol, "_bias", 0.2)
            if not isinstance(bias, float) or abs(bias - 0.2) > 1e-15:
                raise ContractError(
                    f"gumbel: continuation bias {bias!r} for seat {seat!r} has no "
                    "gumbel_halving envelope (Rust covers 0.2 only); fail closed"
                )

        # Real belief required; synthetic worlds removed (fail closed).
        _require_belief()
        if self._belief is None or epoch is None:
            raise ContractError("gumbel: belief and epoch required; synthetic worlds removed")
        try:
            from hydra2.belief.world import world_actor_observation as _wao
        except ImportError as exc:
            raise ImportError(
                "hydra2.belief.world not importable "
                f"({exc}); build the bridge with `pixi run build-ext` before Gumbel search"
            ) from exc

        cfg = self._config
        max_depth = cfg.max_depth
        max_trans = cfg.max_transitions
        slots = _halving_slots(len(sorted_ids), cfg.halving_rounds)

        worlds_json: list[dict[str, Any]] = []
        dirs_rows: list[list[int]] = []
        draws: list[float] = []
        worlds_live_lens: list[int] = []
        first_cont_legal: tuple[int, ...] | None = None
        transitions_sim = 0
        # Memo: continuation obs depends on (world, actor, live_len) only, NOT
        # visit order (``_wao(fake, actor)`` + legal + direction touch no RNG).
        # Corpus K=4 with ~16 visits => each world repeats ~4x — same class as
        # the joint memo 1.94x win. Byte-exact: same inputs, same dir bytes.
        # RNG floats still consumed per visit (stream order preserved).
        _cont_memo: dict[tuple[str, int, int], tuple[tuple[int, ...], int]] = {}

        # Freeze-check the CTR stream before drawing (fail closed, never a draw):
        # belief indices and policy floats interleave below in oracle visit
        # order, so the stream must be freezable upfront.
        _ = _pack_rng(rng)

        # Precompute walk in oracle visit order (round-major, slot-minor,
        # visit-minor); budget truncation stops the tail only. Belief draws +
        # policy floats interleave on the caller's stream exactly as the
        # oracle consumed them, so later belief samples pick identical corpus
        # indices. Continuation observations are action-independent (hands
        # never move, one live tile pops per transition, turn rotates), so
        # directions need no transitions — only hands/live/actor.
        for round_idx in range(cfg.halving_rounds):
            if slots[round_idx] <= 1:
                break
            visits = cfg.visits_per_round[round_idx]
            for _slot in range(slots[round_idx]):
                for _visit in range(visits):
                    if max_trans is not None and transitions_sim >= max_trans:
                        break
                    try:
                        particles: Any = self._belief.sample_natural(epoch, count=1, rng=rng)  # type: ignore[union-attr]
                        particle: Any = particles[0]  # type: ignore[explicit-any]
                        if particle.log_target_density != particle.log_proposal_density:
                            raise ContractError("natural world must have ratio 1")
                        if particle.source != "natural":
                            raise ContractError("gumbel natural may only use natural particles")
                        cur_world = self._world_for_particle(particle)  # type: ignore[unknown-argument-type]
                    except ContractError:
                        raise
                    except Exception as exc:
                        raise ContractError(f"gumbel: belief sampling failed: {exc}") from exc
                    try:
                        hands = tuple(tuple(int(t) for t in h) for h in cur_world.concealed_hands)  # pyrefly: ignore[unknown-argument-type] # Any world tiles
                        live_start = tuple(int(t) for t in cur_world.live_wall)  # pyrefly: ignore[unknown-argument-type] # Any world tiles
                    except Exception as exc:
                        raise ContractError(f"gumbel: sampled world malformed: {exc}") from exc
                    worlds_json.extend(_pack_worlds([cur_world]))
                    worlds_live_lens.append(len(live_start))
                    # Forced root transition (unconditional once entered).
                    transitions_sim += 1
                    step = 1
                    live_len = len(live_start) - 1 if len(live_start) > 0 else 0
                    actor = (root_seat + 1) % 4
                    row_dirs = [0] * max_depth
                    cstep = 0
                    while step < max_depth and live_len > 0:
                        _ckey = (str(cur_world.world_id), actor, live_len)  # pyrefly: ignore[unknown-argument-type] # Any world id
                        _hit = _cont_memo.get(_ckey)
                        if _hit is None:
                            fake = types.SimpleNamespace(
                                concealed_hands=hands,
                                live_wall=tuple([0] * live_len),
                                rules_hash=str(cur_world.rules_hash),  # pyrefly: ignore[unknown-argument-type] # Any world hash
                            )
                            try:
                                obs = _wao(fake, actor=actor)  # pyrefly: ignore[bad-argument-type]
                            except ImportError:
                                raise
                            except Exception as exc:
                                raise ContractError(
                                    f"gumbel: continuation observation failed: {exc}"
                                ) from exc
                            legal_next = _legal_ids_for_observation(obs)
                            if len(legal_next) == 0:
                                _cont_memo[_ckey] = ((), 0)
                                break
                            if first_cont_legal is None:
                                first_cont_legal = legal_next
                            elif legal_next != first_cont_legal:
                                raise ContractError(
                                    "gumbel: varying continuation legal has no batch envelope; fail closed"
                                )
                            try:
                                _h: Any | None = getattr(obs, "observation_hash", None)
                                hs: str = _h if isinstance(_h, str) and _h != "" else ""
                                direction = hashlib.sha256(hs.encode()).digest()[0] & 1
                            except Exception as exc:
                                raise ContractError(
                                    f"gumbel: continuation direction failed: {exc}"
                                ) from exc
                            _cont_memo[_ckey] = (legal_next, direction)
                        else:
                            legal_next, direction = _hit
                            if len(legal_next) == 0:
                                break
                            if first_cont_legal is None:
                                first_cont_legal = legal_next
                            elif legal_next != first_cont_legal:
                                raise ContractError(
                                    "gumbel: varying continuation legal has no batch envelope; fail closed"
                                )
                        row_dirs[cstep] = direction
                        try:
                            draws.append(float(rng.random_float()))  # pyrefly: ignore[unknown-argument-type] # Any RNG draw
                        except ImportError:
                            raise
                        except Exception as exc:
                            raise ContractError(f"gumbel: rng random_float failed: {exc}") from exc
                        # Sample-then-gate: the float above counts even when the cap breaks here.
                        if max_trans is not None and transitions_sim >= max_trans:
                            break
                        transitions_sim += 1
                        step += 1
                        cstep += 1
                        live_len -= 1
                        actor = (actor + 1) % 4
                        if max_trans is not None and transitions_sim >= max_trans:
                            break
                    dirs_rows.append(row_dirs)
                    if max_trans is not None and transitions_sim >= max_trans:
                        break
                if max_trans is not None and transitions_sim >= max_trans:
                    break
            if max_trans is not None and transitions_sim >= max_trans:
                break

        # Pad trailing dummies to the Rust dry-run demand (see helper doc).
        # Real policy floats stay first in consumption order; the tail rides
        # unused (``floats_used`` reports real consumption).
        need = _dry_float_need(worlds_live_lens, slots, cfg.visits_per_round, max_depth, max_trans)
        if len(draws) < need:
            draws = draws + [0.5] * (need - len(draws))
        continuation_legal = [0, 2] if first_cont_legal is None else sorted(first_cont_legal)

        batch = {
            "worlds": worlds_json,
            "rules_hash": str(getattr(epoch, "rules_hash", "")),
            "observation_hash": str(getattr(epoch, "observation_hash", "")),
            "root_legal": sorted(sorted_ids),
            "root_seat": root_seat,
            "gumbels": [[a, gumbels[a]] for a in sorted(sorted_ids)],
            "halving_rounds": cfg.halving_rounds,
            "visits_per_round": list(cfg.visits_per_round),
            "continuation_legal": list(continuation_legal),
            "policy_dirs": dirs_rows,
            "rng_floats": draws,
            "max_depth": max_depth,
            "max_transitions": (None if cfg.max_transitions is None else cfg.max_transitions),
            "max_model_calls": (None if cfg.max_model_calls is None else cfg.max_model_calls),
            "candidate_id": cfg.candidate_id,
            "domain": "gumbel",
            "tie_break": cfg.tie_break,
            "leaf_overrides": [],
        }
        bridge = _require_driver_bridge()
        try:
            out: Any = bridge.gumbel_halving(json.dumps(batch).encode())
        except ImportError:
            raise
        except Exception as exc:
            raise ContractError(f"gumbel bridge halving failed: {exc}") from exc

        try:
            rust_cands: list[int] = out.candidate_ids
            rust_visits: list[int] = out.visits
            rust_vecs: list[tuple[float, float, float, float]] = []
            rust_raw_vecs: list[list[float]] = out.value_vectors
            for vec in rust_raw_vecs:
                quad: tuple[float, ...] = tuple(vec)
                if len(quad) != 4:
                    raise ContractError("gumbel: halving value vector must hold 4 entries")
                for v in quad:
                    if not isinstance(v, float) or v != v or v in (float("inf"), float("-inf")):
                        raise ContractError("gumbel: halving vector must be finite")
                rust_vecs.append((quad[0], quad[1], quad[2], quad[3]))
            survivors_list: list[int] = out.survivors
            rust_survivors = tuple(survivors_list)
            selected_id: int = out.selected_id
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"gumbel: halving outcome malformed: {exc}") from exc
        if rust_cands != sorted(sorted_ids):
            raise ContractError("gumbel: halving candidates mismatch sorted legal")
        if len(rust_cands) != len(rust_vecs) or len(rust_cands) != len(rust_visits):
            raise ContractError("gumbel: halving vectors length mismatch")
        if selected_id not in rust_cands:
            raise ContractError("gumbel: halving selection not a candidate")

        self._simulations: int = out.sims_run
        self._transitions: int = out.transitions
        self._model_calls: int = out.model_calls

        # Per-action stats: visits ride Rust; sums reconstruct via
        # _exact_sum_for_mean so stats[].mean_vector() returns the Rust mean
        # bit-exactly for every schedule (value_vectors below carry the same
        # Rust means directly).
        vec_by_aid = dict(zip(rust_cands, rust_vecs))
        visit_by_aid = dict(zip(rust_cands, rust_visits))
        stats: dict[int, _ActionStats] = {}
        for aid in sorted_ids:
            n = visit_by_aid.get(aid, 0)
            if n > 0:
                mv = vec_by_aid[aid]
                stats[aid] = _ActionStats(
                    visits=n,
                    value_sum=tuple(_exact_sum_for_mean(m, n) for m in mv),  # type: ignore[assignment]
                )
            else:
                stats[aid] = _ActionStats(visits=0, value_sum=(0.0, 0.0, 0.0, 0.0))
        value_vectors = tuple(vec_by_aid[aid] for aid in sorted_ids)

        # Resolve selected action object
        selected_action: Any = id_to_action.get(selected_id, legal_actions[0])  # type: ignore[unknown-argument-type]

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
            "survivors": rust_survivors,
        }

        return {
            "selected_action": selected_action,
            "selected_action_id": selected_id,
            "candidate_actions": legal_actions,
            "value_vectors": value_vectors,
            "stats": stats,
            "gumbels": gumbels,
            "survivors": rust_survivors,
            "telemetry": telemetry,
            "completed": True,
        }
