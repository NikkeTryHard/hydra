# reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dependency fallback shims + re-exported spec symbols). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 5 local resolving — search: Rust-batch resolving driver.

Owns the construction plus Rust-batch search half of
:class:`LocalResolvingPlanner`: belief-world materialization and the
one-bridge-call resolving driver returning the structured result dict. The
per-iteration/per-depth traversal, regret/hedge/fictitious-play updates,
averaging accumulation, path sampling, and next-node mixing live in
``hydra-search:local_resolving_batch`` (GIL released); Python builds ONE
batch per search (sampled worlds in oracle iteration order, memoized
per-(world, actor) base info keys, validated leaf vectors, seeded init-table
entries, per-step sampling floats) and makes ONE bridge call. Tie-break
selection, concrete mapping, counters, and telemetry stay Python (once per
search, zero hotspot). Planner construction and act live in
:mod:`hydra2.search.local_act`; the spec factory lives in
:mod:`hydra2.search.local_spec`.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, cast

from hydra2.contracts.common import ContractError
from hydra2.search.local_abstraction import PublicSubgame as PublicSubgame
from hydra2.search.local_abstraction import build_public_subgame as build_public_subgame
from hydra2.search.local_abstraction import (
    info_key_for_actor_observation as info_key_for_actor_observation,
)
from hydra2.search.local_abstraction import preserves_vector_returns as preserves_vector_returns
from hydra2.search.local_shared import _require_belief as _require_belief
from hydra2.search.local_spec import LocalResolvingConfig as LocalResolvingConfig
from hydra2.search.local_spec import (
    _build_abstraction_from_config as _build_abstraction_from_config,
)
from hydra2.search.local_strategy import StrategyTable as StrategyTable
from hydra2.search.local_strategy import leaf_vector_replay as leaf_vector_replay
from hydra2.search.local_strategy import make_uniform_strategy as make_uniform_strategy

__all__ = [
    "LocalResolvingPlannerSearchMixin",
]


def _require_driver_bridge() -> Any:
    """Import the built ``search`` bridge surface with ``local_resolving_batch`` (fail closed)."""
    try:
        import hydra2_replay_rs as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2_replay_rs extension not importable "
            f"({exc}); rebuild the bridge with `pixi run build-ext` before local resolving search"
        ) from exc
    try:
        mod = _ext.search
    except AttributeError as exc:
        raise ImportError(
            f"hydra2_replay_rs.search missing ({exc}); rebuild the bridge with `pixi run build-ext`"
        ) from exc
    if not hasattr(mod, "local_resolving_batch"):
        raise ImportError(
            "hydra2_replay_rs.search.local_resolving_batch missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        )
    return mod


class LocalResolvingPlannerSearchMixin:
    """Resolving loop for :class:`LocalResolvingPlanner`.

    Split host for construction, table seeding, deterministic RNG, and the
    core search loop; act and result assembly live in the ``local_act``
    mixin and the final subclass adds nothing and no overrides.
    """

    config: LocalResolvingConfig
    candidate_spec: Any | None
    belief: Any | None
    warm_start_prior: dict[tuple[int, str], tuple[float, ...]] | None
    _model_calls: int
    _transitions: int
    _particles: int
    _last_subgame: PublicSubgame | None
    _last_table: StrategyTable | None
    _last_avg_table: StrategyTable | None

    def _init_tables(
        self,
        subgame: PublicSubgame,
        root_observation: Any,
        legal_ids: tuple[int, ...],
    ) -> StrategyTable: ...

    def _deterministic_rng(self, case_id: str, root_seat: int, attempt: int = 0) -> Any: ...
    def __init__(
        self,
        *,
        belief: Any | None = None,
        config: LocalResolvingConfig | None = None,
        candidate_spec: Any | None = None,
        warm_start_prior: dict[tuple[int, str], tuple[float, ...]] | None = None,
    ) -> None:
        if config is None and candidate_spec is not None:
            try:
                params = dict(getattr(candidate_spec, "parameters", {}))
                config = LocalResolvingConfig.from_parameters(params)
            except Exception:
                config = LocalResolvingConfig()
        if config is None:
            config = LocalResolvingConfig()
        self.config = config
        self.candidate_spec = candidate_spec
        self.belief = belief
        self.warm_start_prior = warm_start_prior
        # Telemetry counters
        self._model_calls = 0
        self._transitions = 0
        self._particles = 0
        self._last_subgame: PublicSubgame | None = None
        self._last_table: StrategyTable | None = None
        self._last_avg_table: StrategyTable | None = None

    def observe(self, packet: Any) -> None:  # type: ignore[override]
        return None

    def ponder(self, *, deadline_monotonic_ns: int) -> None:
        return None

    def search(
        self,
        *,
        epoch: Any | None,
        root_observation: Any,
        legal_actions: tuple[Any, ...],
        rng: Any | None = None,
        case_id: str = "case_tiny_001",
        root_seat: int | None = None,
    ) -> dict[str, Any]:
        """Core resolving loop — deterministic, returns telemetry and tables.

        Returns dict with keys: selected_abstract, selected_concrete, tables, subgame,
        telemetry, completed, vectors, avg_tables
        """
        t0 = time.monotonic_ns()
        # Reset per-search telemetry counters for deterministic reporting (not cumulative)
        self._model_calls = 0
        self._transitions = 0
        self._particles = 0
        # Resolve legal concrete ids — support both int/DummyAction and CanonicalAction
        legal_ids: list[int] = []
        for idx, a in enumerate(legal_actions):
            if hasattr(a, "kind") and hasattr(a, "actor"):
                cid: int = idx % 4
            else:
                try:
                    cid = int(getattr(a, "action_id", a))  # type: ignore[arg-type]
                except Exception:
                    cid = int(a)  # type: ignore[arg-type]
            legal_ids.append(cid)
        legal_ids_t = tuple(legal_ids)
        if len(legal_ids_t) == 0:
            raise ContractError("legal_actions must be non-empty")
        # Build subgame
        ab = _build_abstraction_from_config(self.config, legal_ids_t)
        subgame = build_public_subgame(
            epoch,
            horizon=self.config.horizon,
            abstraction=ab,
            iteration_count=self.config.iterations,
            averaging=self.config.averaging,
            update_rule=self.config.update_rule,
            leaf_model=self.config.leaf_model,
            public_history_seed=self.config.public_history_seed,
        )
        self._last_subgame = subgame
        # Prepare RNG
        if rng is None:
            seat = (
                root_seat if root_seat is not None else int(getattr(root_observation, "actor", 0))  # type: ignore[arg-type]
            )
            rng = self._deterministic_rng(case_id, seat)
        # Strategy tables
        table = self._init_tables(subgame, root_observation, legal_ids_t)

        # Need worlds for leaf evaluation — real belief required (fail closed, no synthetic).
        _require_belief()
        if self.belief is None or epoch is None:
            raise ContractError("local: belief and epoch required; synthetic worlds removed")
        try:
            # natural sample count = iterations (one world per iteration)
            particles = self.belief.sample_natural(epoch, count=self.config.iterations, rng=rng)  # type: ignore[call-arg]
        except ImportError:
            raise
        except Exception as exc:
            raise ContractError(f"local: belief sampling failed: {exc}") from exc
        worlds: list[Any] = []
        # worlds behind particle refs — resolve via belief registry (no synthetic fallback).
        for p in particles:
            p_any: Any = p
            w1_raw: Any | None = getattr(self.belief, "_worlds", None)
            w2_raw: Any | None = getattr(self.belief, "worlds", None)
            if w1_raw is not None:
                store: Any = w1_raw
            elif w2_raw is not None:
                store = w2_raw
            else:
                raise ContractError(
                    "local: belief worlds registry missing; synthetic worlds removed"
                )
            w: Any | None = None
            if isinstance(store, dict):
                store_dict: dict[Any, Any] = store  # type: ignore[assignment]
                store_key: str = str(getattr(p_any, "world_ref", ""))
                w = store_dict.get(store_key)
            if w is None:
                raise ContractError(
                    "local: belief world missing for particle; synthetic worlds removed"
                )
            worlds.append(w)
        if len(worlds) == 0:
            raise ContractError(
                "local: belief sampling returned no worlds; synthetic worlds removed"
            )
        self._particles = len(worlds)
        self._model_calls += len(worlds)
        # Map abstract ids to indices for distribution ordering
        ab_order = tuple(sorted(ab.abstract_ids))
        # Iterative traversal runs in Rust (single bridge call below); the
        # retired oracle state dicts (regrets/q/avg accumulators/visit copies)
        # live driver-side now. Only the seeded init entries cross — the root
        # entry is ensured driver-side without a visit-count entry, exactly
        # like the retired seeding below used to do table-only.
        root_actor = int(getattr(root_observation, "actor", 0))
        root_info = info_key_for_actor_observation(root_observation)

        # Batch precompute (ONE envelope per search): sampled worlds in oracle
        # iteration order with memoized per-(world, actor) base info keys and
        # validated leaf vectors, seeded init-table entries, and per-step
        # sampling floats. Only actor-visible strings/floats cross (firewall:
        # ActorObservation construction stays Python, info-keys-only cross).
        horizon = self.config.horizon
        iterations = self.config.iterations
        leaf_memo: dict[str, tuple[float, ...]] = {}
        base_memo: dict[str, list[str | None]] = {}
        iters_json: list[dict[str, Any]] = []
        for w in worlds:
            wid = str(w.world_id)
            leaf = leaf_memo.get(wid)
            if leaf is None:
                leaf = leaf_vector_replay(w, self.config.leaf_model)
                if not preserves_vector_returns(leaf):
                    raise ContractError(f"leaf vector invalid {leaf!r}")
                leaf_memo[wid] = leaf
            bases: list[str | None] | None = base_memo.get(wid)
            if bases is None:
                fresh: list[str | None] = []
                for actor_idx in range(4):
                    try:
                        from hydra2.belief.world import world_actor_observation as _wao

                        obs = _wao(w, actor=actor_idx)  # pyrefly: ignore[bad-argument-type]
                        fresh.append(info_key_for_actor_observation(obs))
                    except Exception:
                        fresh.append(None)
                base_memo[wid] = fresh
                bases = fresh
            iters_json.append({"world_id": wid, "leaf": list(leaf), "base": bases})
        # Per-step sampling floats in iteration-major order (mirrors the
        # retired duck-type exactly; RandomStream has no `random`, so the
        # common path consumes nothing from the stream).
        sampling: list[float] = []
        if hasattr(rng, "random"):
            for it in range(1, iterations + 1):
                for depth in range(horizon):
                    try:
                        r = float(rng.random())  # type: ignore[attr-defined]
                    except Exception:
                        r = ((it * 997 + depth * 13) % 100) / 100.0
                    sampling.append(r)
        else:
            sampling = [0.5] * (iterations * horizon)
        init_json = [
            {"actor": actor, "info": info, "dist": list(dist)}
            for (actor, info), dist in table.table.items()
        ]
        batch = {
            "ab_order": list(ab_order),
            "horizon": horizon,
            "iterations": iterations,
            "update_rule": self.config.update_rule,
            "averaging": self.config.averaging,
            "root_actor": root_actor,
            "root_info": root_info,
            "iters": iters_json,
            "init": init_json,
            "sampling": sampling,
            "public_history_hash": subgame.public_history_hash,
        }
        bridge = _require_driver_bridge()
        try:
            out = bridge.local_resolving_batch(json.dumps(batch).encode())
        except ImportError:
            raise
        # Loop counters match the retired loop exactly (one model call and
        # `horizon` transitions per iteration, plus the bulk sampling charge
        # already applied above).
        self._model_calls += iterations
        self._transitions += iterations * horizon
        # Rebuild the strategy tables from the Rust dumps. Shapes fail
        # closed; values ride through unchecked exactly as the retired loop
        # assigned them (no `.set` validation anywhere on this path).
        n_actions = len(ab_order)

        def _take_dist(raw: Any) -> tuple[float, ...]:
            if not isinstance(raw, (list, tuple)) or len(raw) != n_actions:
                raise ContractError("local: resolving dump dist malformed")
            vals: list[float] = []
            for v in raw:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ContractError("local: resolving dump dist malformed")
                vals.append(float(v))
            return tuple(vals)

        def _take_key(raw: Any) -> tuple[int, str]:
            if not isinstance(raw, (list, tuple)) or len(raw) != 4:
                raise ContractError("local: resolving dump row malformed")
            actor_r, info_r, _dist_ignored, _visits_ignored = raw
            if isinstance(actor_r, bool) or not isinstance(actor_r, int) or not 0 <= actor_r <= 3:
                raise ContractError("local: resolving dump actor malformed")
            if not isinstance(info_r, str) or info_r == "":
                raise ContractError("local: resolving dump info malformed")
            return actor_r, info_r

        try:
            table_rows = json.loads(out.table_json)
            avg_rows = json.loads(out.avg_json)
        except Exception as exc:
            raise ContractError(f"local: resolving dump malformed: {exc}") from exc
        if not isinstance(table_rows, list) or not isinstance(avg_rows, list):
            raise ContractError("local: resolving dump malformed")
        table = StrategyTable(abstraction=ab)
        for row in table_rows:
            actor_r, info_r = _take_key(row)
            _dist_r: Any = row[2]
            _visits_r: Any = row[3]
            if isinstance(_visits_r, bool) or (
                _visits_r is not None and (not isinstance(_visits_r, int) or _visits_r < 0)
            ):
                raise ContractError("local: resolving dump visits malformed")
            table.table[(actor_r, info_r)] = _take_dist(_dist_r)
            if _visits_r is not None:
                table.visit_counts[(actor_r, info_r)] = _visits_r
        avg_table = StrategyTable(abstraction=ab)
        for row in avg_rows:
            actor_r, info_r = _take_key(row)
            _dist_r = row[2]
            _visits_r = row[3]
            avg_table.table[(actor_r, info_r)] = _take_dist(_dist_r)
            if _visits_r is not None:
                if isinstance(_visits_r, bool) or not isinstance(_visits_r, int) or _visits_r < 0:
                    raise ContractError("local: resolving dump visits malformed")
                avg_table.visit_counts[(actor_r, info_r)] = _visits_r
        self._last_table = table
        self._last_avg_table = avg_table
        # Select root action from averaged marginal for root info
        root_avg = avg_table.table.get((root_actor, root_info))
        if root_avg is None:
            root_avg = table.table.get((root_actor, root_info), make_uniform_strategy(ab))
        # Tie break handling (once per search over the returned root
        # average — zero hotspot; the temperature softmax `exp` stays in
        # Python libm so it cannot fork 1 ulp across implementations).
        selected_abstract_idx: int
        if self.config.tie_break == "greedy":
            max_p = max(root_avg)
            candidates = [i for i, p in enumerate(root_avg) if abs(p - max_p) < 1e-9]
            selected_abstract_idx = min(
                candidates
            )  # deterministic greedy smallest abstract id among ties
        elif self.config.tie_break.startswith("temperature"):
            # deterministic temperature sampling via hash of root_info
            temp = 0.5 if "0.5" in self.config.tie_break else 1.0
            # softmax with temp
            logits = [p / temp for p in root_avg]
            m = max(logits)
            exps = [math.exp(v - m) for v in logits]
            s = sum(exps)
            probs = [e / s for e in exps]
            # deterministic sample via hash
            h_frac: float = int(hashlib.sha256(root_info.encode()).hexdigest()[:8], 16) / (2**32)
            cum = 0.0
            selected_abstract_idx = len(probs) - 1
            for i, pr in enumerate(probs):
                cum += pr
                if h_frac < cum:
                    selected_abstract_idx = i
                    break
        else:  # value_break
            # Use leaf vector for tie: pick action with highest offset-adjusted value
            # Recompute utilities for root
            world0_any: Any = worlds[0]
            leaf0 = leaf_vector_replay(world0_any, self.config.leaf_model)
            utilities: list[float] = []
            for aid in ab_order:
                world0_id: str = str(getattr(world0_any, "world_id", "w0"))
                off = (
                    int(
                        hashlib.sha256(f"{world0_id}:{aid}:value".encode()).hexdigest()[:4],
                        16,
                    )
                    % 100
                ) / 1000.0
                utilities.append(leaf0[root_actor] + off)
            # Among max prob actions, pick max utility
            max_p = max(root_avg)
            cand: list[int] = [i for i, p in enumerate(root_avg) if abs(p - max_p) < 1e-9]
            selected_abstract_idx = cand[0] if len(cand) > 0 else 0
            best_util: float = (
                utilities[selected_abstract_idx]
                if len(utilities) > selected_abstract_idx
                else float("-inf")
            )
            for idx_c in cand[1:]:
                if utilities[idx_c] > best_util:
                    best_util = utilities[idx_c]
                    selected_abstract_idx = idx_c
        selected_abstract = ab_order[selected_abstract_idx]
        # Map abstract to representative concrete legal action
        # Find legal concrete whose abstract equals selected_abstract
        candidates_concrete: list[int] = [
            c for c, a in ab.concrete_to_abstract if a == selected_abstract and c in legal_ids_t
        ]
        if len(candidates_concrete) == 0:
            # Fallback: smallest legal maps to selected_abstract via mod? Use first legal
            candidates_concrete = [legal_ids_t[0]]
        selected_concrete = min(candidates_concrete)
        # Find concrete action object from legal_actions — handle CanonicalAction via same deterministic mapping
        selected_action = None
        for idx, a in enumerate(legal_actions):
            if hasattr(a, "kind") and hasattr(a, "actor"):
                cid = idx % 4
            else:
                try:
                    cid = int(getattr(a, "action_id", a))  # type: ignore[arg-type]
                except Exception:
                    cid = int(a)
            if cid == selected_concrete:
                selected_action = a
                break
        if selected_action is None:
            selected_action = legal_actions[0]
        elapsed_ms = (time.monotonic_ns() - t0) / 1e6
        telemetry: dict[str, Any] = {
            "mode": "gameplay_5s",
            "model_calls": self._model_calls,
            "exact_transitions": self._transitions,
            "particles": self._particles,
            "elapsed_ms": elapsed_ms,
            "completed": True,
            "selected_abstract": selected_abstract,
            "selected_concrete": selected_concrete,
            "subgame_nodes": len(subgame.nodes),
            "subgame_edges": len(subgame.edges),
            "iterations": self.config.iterations,
            "horizon": self.config.horizon,
            "update_rule": self.config.update_rule,
            "averaging": self.config.averaging,
            "leaf_model": self.config.leaf_model,
            "resource_view": self.config.resource_view,
            "warm_start": (
                bool(
                    cast("Any", getattr(cast("Any", self.candidate_spec), "parameters", {})).get(
                        "warm_start", False
                    )
                    if isinstance(getattr(cast("Any", self.candidate_spec), "parameters", {}), dict)
                    else False
                )
                if self.candidate_spec is not None
                else False
            ),
        }
        return {
            "selected_action": selected_action,
            "selected_abstract": selected_abstract,
            "selected_concrete": selected_concrete,
            "subgame": subgame,
            "tables": table,
            "avg_tables": avg_table,
            "telemetry": telemetry,
            "completed": True,
            "vectors": [leaf_vector_replay(w, self.config.leaf_model) for w in worlds[:2]],
            "root_info": root_info,
            "root_avg": root_avg,
        }
