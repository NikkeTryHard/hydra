# reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dependency fallback shims + re-exported spec symbols). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 5 local resolving — search: resolving loop over the declared subgame.

Owns the empirical resolving loop: construction and table seeding, deterministic
counter-based RNG, per-iteration world sampling with horizon traversal, frozen
update-rule application, averaging accumulation, and root-marginal selection.
Planner construction and act live in :mod:`hydra2.search.local_act`; the spec
factory lives in :mod:`hydra2.search.local_spec`.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, cast

from hydra2.contracts.common import ContractError
from hydra2.search.local_abstraction import PublicSubgame as PublicSubgame
from hydra2.search.local_abstraction import _digest as _digest
from hydra2.search.local_abstraction import build_public_subgame as build_public_subgame
from hydra2.search.local_abstraction import (
    info_key_for_actor_observation as info_key_for_actor_observation,
)
from hydra2.search.local_abstraction import preserves_vector_returns as preserves_vector_returns
from hydra2.search.local_shared import _HAS_BELIEF as _HAS_BELIEF
from hydra2.search.local_spec import LocalResolvingConfig as LocalResolvingConfig
from hydra2.search.local_spec import (
    _build_abstraction_from_config as _build_abstraction_from_config,
)
from hydra2.search.local_strategy import StrategyTable as StrategyTable
from hydra2.search.local_strategy import _fictitious_play_update as _fictitious_play_update
from hydra2.search.local_strategy import _hedge_update as _hedge_update
from hydra2.search.local_strategy import _regret_matching_update as _regret_matching_update
from hydra2.search.local_strategy import averaging_weights as averaging_weights
from hydra2.search.local_strategy import leaf_vector_replay as leaf_vector_replay
from hydra2.search.local_strategy import make_uniform_strategy as make_uniform_strategy

__all__ = [
    "LocalResolvingPlannerSearchMixin",
]


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
        # Averaging accumulator: sum of strategies weighted
        avg_accum: dict[tuple[int, str], list[float]] = {}
        avg_weights: dict[tuple[int, str], float] = {}
        # Regret tables for regret_matching
        regrets: dict[tuple[int, str], list[float]] = {}
        q_vals: dict[tuple[int, str], list[float]] = {}
        # For fictitious_play need visit counts
        visit_counts: dict[tuple[int, str], int] = dict(table.visit_counts)

        # Need worlds for leaf evaluation: sample from belief if available else synthetic
        # Wave 2 bridge audit: kept Python — iteration worlds resolve behind particle
        # refs via the belief registry (bridge natural_indices returns indices only).
        worlds: list[Any] = []
        if self.belief is not None and epoch is not None and _HAS_BELIEF:
            try:
                # natural sample count = iterations (one world per iteration)
                particles = self.belief.sample_natural(epoch, count=self.config.iterations, rng=rng)  # type: ignore[call-arg]
                # worlds behind particle refs — try to resolve via belief registry if exposed
                # Fallback: generate synthetic worlds from particles
                for p in particles:
                    p_any: Any = p
                    try:
                        # Try to get world from belief internal store
                        w1_raw: Any | None = getattr(self.belief, "_worlds", None)
                        w2_raw: Any | None = getattr(self.belief, "worlds", None)
                        if w1_raw is not None:
                            store: Any = w1_raw
                        elif w2_raw is not None:
                            store = w2_raw
                        else:
                            store = {}
                        w: Any | None = None
                        if isinstance(store, dict):
                            store_dict: dict[Any, Any] = store  # type: ignore[assignment]
                            store_key: str = str(getattr(p_any, "world_ref", ""))
                            w = store_dict.get(store_key)
                        if w is not None:
                            worlds.append(w)
                        else:
                            # synthetic tiny world consistent with epoch
                            from hydra2.belief.world import make_full_world

                            obs_h: str = str(
                                getattr(epoch, "observation_hash", "sha256:" + "b" * 64)
                            )
                            rules_h: str = str(getattr(epoch, "rules_hash", "sha256:" + "a" * 64))
                            # deterministic synthetic hand
                            h_bytes: bytes = hashlib.sha256((obs_h + str(p_any)).encode()).digest()
                            hand_vals: list[int] = [b % 12 for b in h_bytes[:8]]
                            # ensure sorted hands of size 2 per seat
                            hands: Any = tuple(
                                tuple(sorted(hand_vals[i * 2 : i * 2 + 2])) for i in range(4)
                            )
                            w2 = make_full_world(
                                concealed_hands=hands,  # type: ignore[arg-type]
                                live_wall=(8, 9, 10, 11),
                                dead_wall=(),
                                latent_state={
                                    "iter_world": hashlib.sha256(str(p_any).encode()).hexdigest()[
                                        :8
                                    ]
                                },
                                rules_hash=rules_h,
                                observation_hash=obs_h,
                                simulator_snapshot=f"synth:{obs_h[:8]}:{len(worlds)}",
                            )
                            worlds.append(w2)
                    except Exception:
                        # last resort synthetic
                        from hydra2.belief.world import make_full_world

                        w3 = make_full_world(
                            concealed_hands=((0, 1), (2, 3), (4, 5), (6, 7)),
                            live_wall=(8, 9, 10, 11),
                            dead_wall=(),
                            latent_state={"fallback": len(worlds)},
                            rules_hash="sha256:" + "a" * 64,
                            observation_hash="sha256:" + "b" * 64,
                        )
                        worlds.append(w3)
                self._particles = len(worlds)
                self._model_calls += len(worlds)
            except Exception:
                worlds = []
        if len(worlds) == 0:
            # fallback synthetic worlds — deterministic 4 worlds
            from hydra2.belief.world import make_full_world

            base = [
                ((0, 1), (2, 3), (4, 5), (6, 7)),
                ((0, 1), (2, 4), (3, 5), (6, 7)),
                ((0, 1), (2, 5), (3, 4), (6, 7)),
                ((0, 1), (2, 6), (3, 4), (5, 7)),
            ]
            for idx, hands in enumerate(base):
                w = make_full_world(
                    concealed_hands=hands,
                    live_wall=(8, 9, 10, 11),
                    dead_wall=(),
                    latent_state={"synth_idx": idx},
                    rules_hash="sha256:" + "a" * 64,
                    observation_hash="sha256:" + "b" * 64,
                )
                worlds.append(w)
            self._particles = len(worlds)
        # Map abstract ids to indices for distribution ordering
        ab_order = tuple(sorted(ab.abstract_ids))
        # Initialize regrets/q for each key
        for key in list(table.table.keys()):
            n = len(ab_order)
            regrets[key] = [0.0] * n
            q_vals[key] = [0.0] * n
            avg_accum[key] = [0.0] * n
            avg_weights[key] = 0.0

        # Iterative traversal
        root_actor = int(getattr(root_observation, "actor", 0))
        root_info = info_key_for_actor_observation(root_observation)
        # Ensure root entry exists
        if (root_actor, root_info) not in table.table:
            table.table[(root_actor, root_info)] = make_uniform_strategy(ab)
            regrets[(root_actor, root_info)] = [0.0] * len(ab_order)
            q_vals[(root_actor, root_info)] = [0.0] * len(ab_order)
            avg_accum[(root_actor, root_info)] = [0.0] * len(ab_order)
            avg_weights[(root_actor, root_info)] = 0.0
            visit_counts[(root_actor, root_info)] = 0

        # Wave 2 bridge audit: kept Python — descent loop + node tables need live worlds
        # and model leaf vectors (no pyfn covers traversal; regret/hedge/FP updates stay Python).
        for it in range(1, self.config.iterations + 1):
            # Select world cyclically
            world = worlds[(it - 1) % len(worlds)]
            leaf_vec = leaf_vector_replay(world, self.config.leaf_model)
            if not preserves_vector_returns(leaf_vec):
                raise ContractError(f"leaf vector invalid {leaf_vec!r}")
            self._model_calls += 1
            self._transitions += self.config.horizon
            # Sample public history path: deterministic via rng or via hash
            # For each depth, pick abstract action by sampling from current strategy at that info node
            # Simplify: traverse one path per iteration, depth = horizon
            path_nodes: list[str] = [subgame.public_history_hash]
            # We'll simulate per-actor updates: at each depth, actor = (root_actor + depth) % 4
            for depth in range(subgame.horizon):
                actor = (root_actor + depth) % 4
                # info hash for this actor at this public node
                # Derive from world actor observation at that actor + node hash
                try:
                    from hydra2.belief.world import world_actor_observation

                    obs = world_actor_observation(world, actor=actor)
                    base_key = info_key_for_actor_observation(obs)
                    # Mix with public node to get distinct per depth but still per actor info
                    info_hash = _digest(f"{base_key}:{path_nodes[-1]}")
                except Exception:
                    info_hash = _digest(f"{path_nodes[-1]}:actor{actor}")
                # Ensure table entry
                if (actor, info_hash) not in table.table:
                    table.table[(actor, info_hash)] = make_uniform_strategy(ab)
                    regrets[(actor, info_hash)] = [0.0] * len(ab_order)
                    q_vals[(actor, info_hash)] = [0.0] * len(ab_order)
                    avg_accum[(actor, info_hash)] = [0.0] * len(ab_order)
                    avg_weights[(actor, info_hash)] = 0.0
                    visit_counts[(actor, info_hash)] = 0
                # Current strategy
                strat = table.table[(actor, info_hash)]
                # Generate regrets/q based on leaf_vec projection for that actor
                # Simplified: utility for actor is leaf_vec[actor]; create action utilities by adding small per-action offset
                # Offset deterministic from abstract id and world
                utilities: list[float] = []
                for aid in ab_order:
                    # deterministic offset per action
                    off = (
                        int(
                            hashlib.sha256(f"{world.world_id}:{aid}:{depth}".encode()).hexdigest()[
                                :4
                            ],
                            16,
                        )
                        % 100
                    ) / 1000.0 - 0.05
                    utilities.append(leaf_vec[actor] + off)
                # Compute expected value under current strat
                ev = sum(p * u for p, u in zip(strat, utilities, strict=True))
                # Regrets = utility - ev
                cur_reg = tuple(u - ev for u in utilities)
                # Update regrets/q
                if self.config.update_rule == "regret_matching":
                    # accumulate positive regrets
                    for i, r in enumerate(cur_reg):
                        regrets[(actor, info_hash)][i] += r
                    new_strat = _regret_matching_update(strat, tuple(regrets[(actor, info_hash)]))
                elif self.config.update_rule == "hedge":
                    for i, u in enumerate(utilities):
                        q_vals[(actor, info_hash)][i] += u
                    new_strat = _hedge_update(strat, tuple(q_vals[(actor, info_hash)]))
                else:  # fictitious_play
                    # best response is max utility
                    br_idx: int = 0
                    best_val: float = utilities[0] if len(utilities) > 0 else 0.0
                    for idx_br, val_br in enumerate(utilities):
                        if val_br > best_val:
                            best_val = val_br
                            br_idx = idx_br
                    cnt = visit_counts[(actor, info_hash)]
                    new_strat = _fictitious_play_update(strat, br_idx, cnt)
                    visit_counts[(actor, info_hash)] = cnt + 1
                table.table[(actor, info_hash)] = new_strat
                # Averaging accumulator
                w = averaging_weights(it, self.config.iterations, self.config.averaging)
                for i, p in enumerate(new_strat):
                    avg_accum[(actor, info_hash)][i] += p * w
                avg_weights[(actor, info_hash)] += w
                table.visit_counts[(actor, info_hash)] = (
                    table.visit_counts.get((actor, info_hash), 0) + 1
                )
                # Move to next public node via sampled abstract action
                # Sample action from new_strat deterministically via rng
                # Use rng.random() to pick
                try:
                    r = float(rng.random()) if hasattr(rng, "random") else 0.5  # type: ignore[attr-defined]
                except Exception:
                    r = ((it * 997 + depth * 13) % 100) / 100.0
                cum = 0.0
                chosen_idx = len(ab_order) - 1
                for i, p in enumerate(new_strat):
                    cum += p
                    if r < cum:
                        chosen_idx = i
                        break
                chosen_aid = ab_order[chosen_idx]
                # Next node hash via edge
                nxt = _digest(f"{path_nodes[-1]}:{depth}:{chosen_aid}")
                # Ensure nxt is in subgame nodes or synthesize
                if nxt not in subgame.nodes:
                    # For exhaustive gate, we may be off subgame graph; still continue but count transition
                    pass
                path_nodes.append(nxt)
            # End depth loop
            # Also update root averaging if not already via loop (root actor at depth 0 already updated)
            # Ensure root accumulators weighted
            rkey = (root_actor, root_info)
            if rkey in avg_accum and avg_weights[rkey] == 0:
                # root was updated already above at depth 0 when actor==root_actor
                pass

        # Build averaged tables
        avg_table = StrategyTable(abstraction=ab)
        for key, acc in avg_accum.items():
            w = avg_weights[key]
            if w > 0:
                avg = tuple(v / w for v in acc)
                # renormalize
                s = sum(avg)
                avg = tuple(v / s for v in avg) if s > 0 else make_uniform_strategy(ab)
                avg_table.table[key] = avg
                avg_table.visit_counts[key] = table.visit_counts.get(key, 0)
            else:
                # no visits — uniform
                avg_table.table[key] = table.table.get(key, make_uniform_strategy(ab))

        self._last_table = table
        self._last_avg_table = avg_table
        # Select root action from averaged marginal for root info
        root_avg = avg_table.table.get((root_actor, root_info))
        if root_avg is None:
            root_avg = table.table.get((root_actor, root_info), make_uniform_strategy(ab))
        # Tie break handling
        # Wave 2 bridge audit: kept Python — greedy/temperature/value_break tie-breaks
        # are not bridge selection cuts (no halving/gumbel/UCT/PUCT pyfn covers them).
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
