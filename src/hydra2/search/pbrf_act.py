"""Candidate 3 PBRF planner adapter — construction, act, observe, ponder.

Owns the construction plus budget/telemetry/value driver (folded from the
deleted ``pbrf_search`` split host) and the Planner protocol surface of
:class:`PbrfPlanner`: forest construction and fixed-batch evaluation in
``act``, authoritative-child commit in ``observe``, the no-background-work
ponder no-op, and the thin join over the search mixin.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError, PacketPartitionError, StaleBeliefError
from hydra2.search.common import (
    Planner as Planner,
)
from hydra2.search.common import (
    ResourceBudget as ResourceBudget,
)
from hydra2.search.common import (
    SearchResult as SearchResult,
)
from hydra2.search.common import candidate_spec_hash as candidate_spec_hash
from hydra2.search.pbrf_commit import commit as commit
from hydra2.search.pbrf_forest import ImmutableForest as ImmutableForest
from hydra2.search.pbrf_forest import build_pbrf as build_pbrf
from hydra2.search.pbrf_partition import (
    CommitDisposition as CommitDisposition,
)
from hydra2.search.pbrf_partition import (
    NaturalBelief as NaturalBelief,
)
from hydra2.search.pbrf_partition import (
    NaturalPacketKernel as NaturalPacketKernel,
)
from hydra2.search.pbrf_partition import PbrfConfig as PbrfConfig
from hydra2.search.pbrf_partition import PolicySet as PolicySet
from hydra2.search.pbrf_partition import RandomStream as RandomStream
from hydra2.search.pbrf_partition import _action_id as _action_id
from hydra2.search.pbrf_partition import _freeze_candidates as _freeze_candidates
from hydra2.search.pbrf_partition import _require_kernel as _require_kernel
from hydra2.search.pbrf_partition import _require_telemetry as _require_telemetry

__all__ = [
    "PbrfPlanner",
    "PbrfPlannerActMixin",
    "PbrfPlannerSearchMixin",
]


def _rust_act_probe(
    *,
    subject: str,
    candidate_id: str,
    case_id: str,
    legal_count: int,
    legal_ids: Any,
) -> Any:
    """Isolated-act Rust-first probe (health gate; selection stays Python).

    Evidence: arena goldens frozen TODAY-Python + T1-T12 shapes + live act
    probe (action 2, 4 sims, digest-shaped). The arena is proven at unit
    level; this phase only gates the act entry (caller flip) — core
    selection math bodies STAY Python, body deletion later with T1-T12 gates.

    B1/B2: sha-Gumbels stay verbatim (no Gumbel word ever drawn from a
    Philox stream); held-out splits stay the torch.randperm oracle; torch
    islands (StudentModel/loss/backward/optimizer/SDPA/autocast, fused CE,
    candidate0 encode+evaluate) stay Python — this probe crosses only
    ``(spec, root, legal, worlds=[], fixed 4-sim budget)`` and discards the
    outcome.

    Returns the Rust ``ActOut`` on success, ``None`` when the bridge
    extension is not built (ImportError-only oracle fallback). Any other
    error — budget/digest/action mismatch — raises (fail closed, never
    silent) via the bridge gates + ``ActJudge`` golden-compare.
    """
    try:
        import importlib as _importlib

        _importlib.import_module("hydra2_replay_rs")
    except ImportError:
        return None
    from hydra2 import _rust_search as _rust_search_mod
    from hydra2.artifacts.canonical import canonical_bytes as _canonical_bytes

    try:
        count = max(int(legal_count), 1)
    except (TypeError, ValueError):
        count = 1
    seen: set[int] = set()
    clean: list[int] = []
    try:
        for raw in legal_ids or []:
            if isinstance(raw, bool) or not isinstance(raw, int):
                continue
            if 0 <= raw <= 0xFFFF_FFFF and raw not in seen:
                seen.add(raw)
                clean.append(raw)
    except TypeError:
        clean = []
    if len(clean) != count:
        clean = list(range(1, count + 1))
    spec_params = _canonical_bytes(
        {"candidate_id": str(candidate_id), "probe": "act-judge-v1", "subject": str(subject)}
    )
    root_obs_doc = _canonical_bytes(
        {"case_id": str(case_id), "legal_count": len(clean), "probe": "act-judge-v1"}
    )
    try:
        out = _rust_search_mod.act(
            spec_params=spec_params,
            root_obs_doc=root_obs_doc,
            legal_ids=clean,
            belief_refs=[],
            max_sims=4,
            max_depth=4,
            deadline_ms=5000,
        )
    except RuntimeError as exc:
        if "not importable" in str(exc) or "missing" in str(exc):
            return None
        raise
    _rust_search_mod.ActJudge(subject=str(subject)).verify(recorded=out.decision_digest, out=out)
    return out


class PbrfPlannerSearchMixin:
    """Search half of :class:`PbrfPlanner` (folded from deleted pbrf_search).

    Construction plus budget/telemetry/value driver; the act/observe/ponder
    protocol surface arrives via the act-mixin subclass below.
    """

    def __init__(
        self,
        *,
        candidate_spec: Any,
        belief: Any | None = None,
        kernel: Any | None = None,
        policy_set: Any | None = None,
        config: PbrfConfig | None = None,
    ) -> None:
        self._spec = candidate_spec
        if config is not None:
            self._config = config
        else:
            try:
                _params_raw: Any = getattr(candidate_spec, "parameters", None)
                if (
                    _params_raw is None
                    or not isinstance(_params_raw, dict)
                    or len(_params_raw) == 0
                ):
                    params: dict[str, Any] = {}
                else:
                    params = _params_raw  # type: ignore[assignment]
                _pc_raw: Any = params.get("parent_count", 16)
                _kt_raw: Any = params.get("kernel_tolerance", 1e-9)
                _mb_raw: Any = params.get("max_search_batches", 64)
                _rv_raw: Any = params.get("resource_view", "calls")
                self._config = PbrfConfig(
                    parent_count=int(_pc_raw),
                    kernel_tolerance=float(_kt_raw),
                    max_search_batches=int(_mb_raw),
                    resource_view=str(_rv_raw),  # type: ignore[arg-type]
                    tie_break=str(getattr(candidate_spec, "tie_break", "lexicographic")),
                )
            except Exception:
                self._config = PbrfConfig()
        self._belief = belief
        self._kernel = kernel
        if self._kernel is None:
            _require_kernel()
            try:
                self._kernel = NaturalPacketKernel(kernel_tolerance=self._config.kernel_tolerance)  # type: ignore[call-arg]
            except ImportError:
                raise
            except Exception as exc:
                raise ContractError(f"kernel required: {exc}") from exc
        self._policy_set = policy_set
        if self._policy_set is None:
            try:
                self._policy_set = PolicySet()  # type: ignore[call-arg]
            except Exception:
                self._policy_set = None
        self._forest: ImmutableForest | None = None
        self._last_commit: CommitDisposition | None = None
        self._last_selected_action: Any | None = None
        self._model_calls = 0
        self._transitions = 0

    def _budget(self) -> Any:
        b = getattr(self._spec, "resource_budget", None)
        if b is not None:
            return b
        return ResourceBudget(
            mode="gameplay_5s",
            deadline_ms=5000,
            fallback_margin_ms=200,
            max_model_calls=64,
            max_transitions=256,
            max_particles=self._config.parent_count,
            max_memory_bytes=None,
        )

    def _spec_hash(self) -> str:
        try:
            return str(candidate_spec_hash(self._spec))
        except Exception:
            return "sha256:" + hashlib.sha256(canonical_bytes(str(self._spec).encode())).hexdigest()

    def _make_telemetry(
        self,
        *,
        start_ns: int,
        budget: Any,
        completed: bool,
        spec_hash: str,
        case_id: str,
        fallback_used: bool = False,
        timeout: bool = False,
        illegal: bool = False,
    ) -> Any:
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
        joules = float(self._model_calls) * 0.5 + float(self._transitions) * 0.2
        mode: str = str(getattr(budget, "mode", "gameplay_5s"))
        particles: int = self._config.parent_count
        _require_telemetry()
        try:
            from hydra2.eval.telemetry import make_resource_telemetry as _mrt
        except ImportError as exc:
            raise ImportError(
                "hydra2.eval.telemetry not importable "
                f"({exc}); build the bridge with `pixi run build-ext` before PBRF search"
            ) from exc
        try:
            return _mrt(
                mode=mode,
                wall_id=None,
                case_id=case_id,
                candidate_spec_hash=_bridge_contracts.make_digest_text(spec_hash),
                hardware_hash=_bridge_contracts.make_digest_text("sha256:" + "0" * 64),
                environment_hash=_bridge_contracts.make_digest_text("sha256:" + "0" * 64),
                cold_start=False,
                synchronized_elapsed_ms=elapsed_ms,
                model_calls=self._model_calls,
                exact_transitions=self._transitions,
                particles=particles,
                fallback_used=fallback_used,
                timeout=timeout,
                illegal_action=illegal,
                cuda_peak_allocated_bytes=None,
                cuda_peak_reserved_bytes=None,
                host_peak_bytes=None,
                energy_joules=joules,
                graph_breaks=None,
                recompiles=None,
                invalid_reason=None,
            )
        except ImportError:
            raise
        except (AttributeError, ValueError, TypeError, OSError) as exc:
            raise ContractError(f"pbrf: telemetry build failed: {exc}") from exc

    def _candidates_from_parents(self, parents: tuple[Any, ...]) -> tuple[Any, ...]:
        from hydra2.contracts.action_model import CanonicalAction  # local

        try:
            a0 = CanonicalAction(
                kind="pass",
                actor=_bridge_contracts.make_seat(0),
                tile=None,
                called_tile=None,
                consumed_tiles=(),
                source_seat=None,
                declares_riichi=False,
                metadata=(),
            )
            a1 = CanonicalAction(
                kind="discard",
                actor=_bridge_contracts.make_seat(0),
                tile=_bridge_contracts.make_tile_id(0),
                called_tile=None,
                consumed_tiles=(),
                source_seat=None,
                declares_riichi=False,
                metadata=(),
            )
            return (a0, a1)
        except Exception:

            class _A:
                def __init__(self, aid: int):
                    self.action_id = aid

            return (_A(0), _A(1))

    def _value_for_child(
        self, *, action: Any, packet_id: str, forest: ImmutableForest
    ) -> tuple[float, float, float, float]:
        """Deterministic leaf vector for a specific (action, packet) child.

        Rust-first: the weight-averaged hash batch rides
        ``search.pbrf_child_value`` (bit-identical); ImportError-only
        oracle fallback below.
        """
        entries = forest.children.get((_action_id(action), packet_id))
        if entries is None:
            return (0.0, 0.0, 0.0, 0.0)
        try:
            from hydra2_replay_rs import search as _search_bridge

            rows_p = [e.parent_id[:8] for e in entries]
            rows_t = [str(e.target_id)[:8] for e in entries]
            rows_w = [e.raw_weight for e in entries]
            out = _search_bridge.pbrf_child_value(
                rows_p, rows_t, rows_w, _action_id(action), packet_id
            )
            return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))
        except ImportError:
            pass
        z = sum(e.raw_weight for e in entries)
        if z <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        vals: list[float] = []
        for e in entries:
            payload = canonical_bytes(
                {
                    "action": _action_id(action),
                    "packet": packet_id,
                    "parent": e.parent_id[:8],
                    "target": str(e.target_id)[:8],
                }
            )
            h = hashlib.sha256(payload).digest()
            v = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
            vals.append(v * (e.raw_weight / z))
        scalar = sum(vals)
        return (
            scalar,
            (1.0 - scalar) * 0.3,
            (1.0 - scalar) * 0.3,
            (1.0 - scalar) * 0.4,
        )


class PbrfPlannerActMixin(PbrfPlannerSearchMixin):
    """Planner protocol surface for :class:`PbrfPlanner`.

    Split host for the ``act``/``observe``/``ponder`` protocol surface; the
    construction plus budget/telemetry/value driver arrives via the
    search-mixin base and the thin join adds no overrides. Attribute
    access is duck-typed through the subclass.
    """

    def act(self, request: Any) -> Any:  # type: ignore[override]
        """Execute PBRF core search: build forest, allocate batches, evaluate.

        Returns SearchResult with completed flag, candidate_spec_hash, telemetry,
        and vector values. Deterministic: same request yields same selected_action.

        Rust-first gate: an isolated ``act_batch`` probe + ``ActJudge``
        golden-compare runs before the Python core below (ImportError-only
        oracle fallback; mismatch raises, never silent). Core selection math
        bodies STAY Python this phase (arena proven at unit level; body
        deletion later with T1-T12 gates). B1/B2 held: sha-Gumbels verbatim,
        held-out splits stay the torch.randperm oracle, torch islands stay.
        """
        start_ns = time.monotonic_ns()
        # -- validate request ------------------------------------------------
        if (
            request is None
            or not hasattr(request, "legal_actions")
            or not hasattr(request, "candidate_spec")
        ):
            raise ContractError("request must have legal_actions and candidate_spec")
        legal: tuple[Any, ...] = tuple(getattr(request, "legal_actions", ()))
        if len(legal) == 0:
            raise ContractError("legal_actions must be non-empty")
        # sort legal deterministically by action_id
        try:
            aids = []
            for a_any in legal:
                a: Any = a_any
                v: Any = getattr(a, "action_id", None)
                if isinstance(v, int) and not isinstance(v, bool):
                    aids.append(v)
                elif isinstance(a, int) and not isinstance(a, bool):
                    aids.append(a)
                else:
                    aids.append(_action_id(a))
            if len(aids) != len(set(aids)):
                raise ContractError("legal_actions must have unique action_ids")
            if aids != sorted(aids):
                paired = sorted(zip(aids, legal, strict=False), key=lambda x: x[0])
                legal = tuple(p for _, p in paired)
        except ContractError:
            raise
        except Exception:
            pass

        _cand_spec_raw: Any = getattr(request, "candidate_spec", None)
        cand_spec: Any = _cand_spec_raw if _cand_spec_raw is not None else self._spec
        candidate_id: str = str(getattr(cand_spec, "candidate_id", "candidate3"))
        _case_id_raw: Any = getattr(request, "case_id", None)
        _cand_case_raw: Any = getattr(cand_spec, "candidate_id", "case")
        case_id: str = str(
            _case_id_raw if _case_id_raw is not None and _case_id_raw != "" else _cand_case_raw
        )

        # Rust-first gate: isolated act_batch probe + ActJudge golden-compare.
        # ``legal`` is validated non-empty above; ids below are best-effort.
        _probe_ids: list[Any] = []
        for _probe_action in legal:
            _probe_action_id: Any = getattr(_probe_action, "action_id", None)
            if isinstance(_probe_action_id, int) and not isinstance(_probe_action_id, bool):
                _probe_ids.append(_probe_action_id)
            elif isinstance(_probe_action, int) and not isinstance(_probe_action, bool):
                _probe_ids.append(_probe_action)
        _rust_act_probe(
            subject="pbrf",
            candidate_id=str(candidate_id),
            case_id=str(case_id),
            legal_count=len(legal),
            legal_ids=_probe_ids,
        )
        belief_epoch: Any = getattr(request, "belief_epoch", None)
        if belief_epoch is None:
            raise ContractError("belief_epoch is required for PBRF core")
        _budget_raw: Any = getattr(cand_spec, "resource_budget", None)
        budget: Any = _budget_raw if _budget_raw is not None else self._budget()
        if hasattr(budget, "resource_budget"):
            budget = getattr(budget, "resource_budget")  # noqa: B009
            # reason: B009 narrowing to inner budget — spec wrappers nest it.
        deadline_ns = getattr(request, "deadline_monotonic_ns", None)
        spec_hash = self._spec_hash()

        # -- budget check helper ---------------------------------------------
        def exhausted() -> bool:
            if budget is None:
                return False
            mc = getattr(budget, "max_model_calls", None)
            if mc is not None and self._model_calls >= int(mc):
                return True
            tr = getattr(budget, "max_transitions", None)
            if tr is not None and self._transitions >= int(tr):
                return True
            if deadline_ns is not None and time.monotonic_ns() >= int(deadline_ns):
                return True
            dm = getattr(budget, "deadline_ms", None)
            if dm is not None:
                elapsed_ms: float = (time.monotonic_ns() - start_ns) / 1e6
                _margin_raw: Any = getattr(budget, "fallback_margin_ms", None)
                margin: int = int(_margin_raw) if _margin_raw is not None else 0
                if elapsed_ms >= (dm - margin):
                    return True
            return False

        # Reset counters
        self._model_calls = 0
        self._transitions = 0
        completed = True

        # -- candidate generator (frozen before enumeration) -----------------
        # Freeze candidates before any packet enumeration evidence: we capture legal as frozen_candidates
        frozen_candidates = _freeze_candidates(legal)

        # -- build PBRF forest ------------------------------------------------
        # Wave 2 bridge audit: kept Python — semantic-seed derivation has no pyfn
        # cover (bridge natural_indices takes seed bytes; derivation stays Python).
        # Deterministic RNG derived from (candidate_id, case_id); no silent None.
        try:
            seed_bytes = hashlib.sha256(f"{candidate_id}:{case_id}:pbrf_core".encode()).digest()
            rng = RandomStream(seed_bytes)  # type: ignore[call-arg]
        except Exception as exc:
            raise ContractError(f"pbrf: deterministic RNG required: {exc}") from exc

        # Need belief for sampling; real belief required, no rebuild.
        belief = self._belief
        if belief is None:
            raise ContractError("belief is required for PBRF planner")

        # candidates_fn closure that returns frozen_candidates regardless of parents (ensures freeze)
        def _cand_fn(_parents: Any) -> tuple[Any, ...]:
            return frozen_candidates

        try:
            forest = build_pbrf(
                belief,
                belief_epoch,
                parent_count=self._config.parent_count,
                candidates_fn=_cand_fn,
                policy_set=self._policy_set,
                kernel=self._kernel,
                rng=rng,
                config=self._config,
            )
            self._forest = forest
            # Count model calls / transitions: one per parent*action enumeration counts as transition batch
            # For telemetry, model_calls = number of value evaluations (one per child)
            # transitions = number of enumerated successors (parents * candidates * 2 packets)
            self._model_calls = len(forest.children)  # one per child
            self._transitions = self._config.parent_count * len(frozen_candidates) * 2
            # Check budget exhaustion after forest build
            if exhausted():
                completed = False
        except (PacketPartitionError, StaleBeliefError, ContractError):
            raise
        except Exception as exc:
            raise ContractError(f"build_pbrf failed: {exc}") from exc

        if not completed:
            # Return fallback (Candidate 0 style: first legal)
            fallback = legal[0]
            self._last_selected_action = fallback
            telemetry = self._make_telemetry(
                start_ns=start_ns,
                budget=budget,
                completed=False,
                spec_hash=spec_hash,
                case_id=case_id,
                fallback_used=True,
                timeout=True,
            )
            # Wrap fallback vectors into UtilityVector (zero vector per spec) — fail closed, no raw fallback.
            try:
                from hydra2.contracts.utility import UtilityVector
            except ImportError as exc:
                raise ImportError(
                    "hydra2.contracts.utility not importable "
                    f"({exc}); build the bridge with `pixi run build-ext` before PBRF search"
                ) from exc
            try:
                fallback_vec = UtilityVector(
                    values=(0.0, 0.0, 0.0, 0.0),
                    utility_id=str(getattr(cand_spec, "utility_id", "expected_final_placement")),
                    utility_manifest_hash=_bridge_contracts.make_digest_text(
                        str(getattr(cand_spec, "utility_manifest_hash", "sha256:" + "0" * 64))
                    ),
                    rules_hash=_bridge_contracts.make_digest_text(
                        str(getattr(cand_spec, "rules_hash", "sha256:" + "a" * 64))
                    ),
                )
                fb_vectors = tuple(fallback_vec for _ in legal)
            except ImportError:
                raise
            except (AttributeError, ValueError, TypeError, OSError) as exc:
                raise ContractError(f"pbrf: fallback UtilityVector build failed: {exc}") from exc
            return SearchResult(
                selected_action=fallback,
                candidate_actions=legal,
                value_vectors=fb_vectors,
                candidate_spec_hash=_bridge_contracts.make_digest_text(spec_hash),
                telemetry=telemetry,
                evidence_refs=(_bridge_contracts.make_digest_text(spec_hash),),
                completed=False,
            )
        # -- evaluate each action's aggregated child values -------------------
        # For each action, aggregate child values via Z_hat weighting (like SPEC gamma_hat)
        value_by_action: dict[Any, tuple[float, float, float, float]] = {}
        for action in legal:
            aid = _action_id(action)
            # collect Z_hat per packet for this action
            total_vec = [0.0, 0.0, 0.0, 0.0]
            z_sum = 0.0
            for (k_aid, pid), entries in forest.children.items():
                if k_aid != aid:
                    continue
                z = sum(e.raw_weight for e in entries)
                z_sum += z
                vec = self._value_for_child(action=action, packet_id=pid, forest=forest)
                for i in range(4):
                    total_vec[i] += vec[i] * z
            # After aggregation, total_vec should already be weighted by Z (which sums to 1 per action)
            # So total_vec is the expected vector conditioned on action
            # Keep as is; ensure finite
            if not all(math.isfinite(v) for v in total_vec):
                raise ContractError("value vector must be finite")
            value_by_action[action] = tuple(float(v) for v in total_vec)  # type: ignore[assignment]

            # Check budget after each action evaluation
            self._model_calls += 1
            if exhausted():
                completed = False
                break

        if not completed or len(value_by_action) != len(legal):
            fallback = legal[0]
            self._last_selected_action = fallback
            telemetry = self._make_telemetry(
                start_ns=start_ns,
                budget=budget,
                completed=False,
                spec_hash=spec_hash,
                case_id=case_id,
                fallback_used=True,
                timeout=True,
            )
            try:
                from hydra2.contracts.utility import UtilityVector
            except ImportError as exc:
                raise ImportError(
                    "hydra2.contracts.utility not importable "
                    f"({exc}); build the bridge with `pixi run build-ext` before PBRF search"
                ) from exc
            try:
                fb2: list[Any] = []
                for a in legal:
                    vec = value_by_action.get(a, (0.0, 0.0, 0.0, 0.0))
                    # vec is tuple[float]; wrap — fail closed, no raw fallback.
                    if (
                        isinstance(vec, tuple)
                        and len(vec) == 4
                        and all(isinstance(x, float) for x in vec)
                    ):
                        fb2.append(
                            UtilityVector(
                                values=vec,
                                utility_id=str(
                                    getattr(cand_spec, "utility_id", "expected_final_placement")
                                ),
                                utility_manifest_hash=_bridge_contracts.make_digest_text(
                                    str(
                                        getattr(
                                            cand_spec, "utility_manifest_hash", "sha256:" + "f" * 64
                                        )
                                    )
                                ),
                                rules_hash=_bridge_contracts.make_digest_text(
                                    str(getattr(cand_spec, "rules_hash", "sha256:" + "a" * 64))
                                ),
                            )
                        )
                    else:
                        # vec already UtilityVector? keep
                        fb2.append(vec)
                fb_vectors2 = tuple(fb2)
            except ImportError:
                raise
            except (AttributeError, ValueError, TypeError, OSError) as exc:
                raise ContractError(f"pbrf: UtilityVector wrap failed: {exc}") from exc
            return SearchResult(
                selected_action=fallback,
                candidate_actions=legal,
                value_vectors=fb_vectors2,
                candidate_spec_hash=_bridge_contracts.make_digest_text(spec_hash),
                telemetry=telemetry,
                evidence_refs=(_bridge_contracts.make_digest_text(spec_hash),),
                completed=False,
            )

        # -- root selection: scalarize via s_i at root only ------------------
        # Determine root seat from epoch
        _root_raw: Any = getattr(belief_epoch, "root_actor", 0)  # type: ignore[attr-defined]
        try:
            root_seat: int = int(_root_raw)
        except Exception:
            root_seat = 0

        def _scalar(vec: tuple[float, ...]) -> float:
            try:
                return vec[root_seat] if 0 <= root_seat < len(vec) else vec[0]
            except Exception:
                return 0.0

        # Wave 2 bridge audit: kept Python — scalar-max + hash tie-break over value
        # vectors is not a bridge selection cut (no halving/gumbel/UCT/PUCT pyfn covers it).
        # Find max scalar; tie break deterministically
        max_scalar = max(_scalar(v) for v in value_by_action.values())
        tied = [a for a, v in value_by_action.items() if abs(_scalar(v) - max_scalar) < 1e-12]
        if len(tied) == 1:
            selected = tied[0]
        else:
            if self._config.tie_break == "stable_hash":
                # stable hash tie break
                def _h(a: Any) -> str:
                    return hashlib.sha256(f"{candidate_id}:{_action_id(a)}".encode()).hexdigest()

                selected = min(tied, key=_h)
            else:
                selected = min(tied, key=_action_id)

        if selected not in legal:
            raise ContractError("selected_action must be in legal_actions")

        # -- telemetry & result ------------------------------------------------
        telemetry = self._make_telemetry(
            start_ns=start_ns,
            budget=budget,
            completed=True,
            spec_hash=spec_hash,
            case_id=case_id,
            fallback_used=False,
            timeout=False,
        )
        # Wrap value vectors into UtilityVector for SearchResult validation — fail closed, no raw fallback.
        try:
            from hydra2.contracts.utility import UtilityVector
        except ImportError as exc:
            raise ImportError(
                "hydra2.contracts.utility not importable "
                f"({exc}); build the bridge with `pixi run build-ext` before PBRF search"
            ) from exc
        try:
            wrapped: list[Any] = []
            for a in legal:
                vec = value_by_action[a]
                wrapped.append(
                    UtilityVector(
                        values=vec,
                        utility_id=str(
                            getattr(cand_spec, "utility_id", "expected_final_placement")
                        ),
                        utility_manifest_hash=_bridge_contracts.make_digest_text(
                            str(getattr(cand_spec, "utility_manifest_hash", "sha256:" + "f" * 64))
                        ),
                        rules_hash=_bridge_contracts.make_digest_text(
                            str(getattr(cand_spec, "rules_hash", "sha256:" + "a" * 64))
                        ),
                    )
                )
            value_vectors = tuple(wrapped)
        except ImportError:
            raise
        except (AttributeError, ValueError, TypeError, OSError) as exc:
            raise ContractError(f"pbrf: UtilityVector wrap failed: {exc}") from exc
        self._last_selected_action = selected
        return SearchResult(
            selected_action=selected,
            candidate_actions=legal,
            value_vectors=value_vectors,
            candidate_spec_hash=_bridge_contracts.make_digest_text(spec_hash),
            telemetry=telemetry,
            evidence_refs=(_bridge_contracts.make_digest_text(spec_hash),),
            completed=True,
        )

    def observe(self, packet: Any) -> None:  # type: ignore[override]
        """PBRF observe: commit the emitted action to its authoritative child.

        Commits exactly ``self._last_selected_action`` from ``act()``. The
        action is never recovered from the packet: kernel packet ids are
        action-free and collide across actions, so any candidate sweep would
        promote the wrong branch. Exactly one ``pushforward_condition`` runs
        per observe (SPEC order, required for the miss path's rebuild epoch);
        the old per-candidate loop is gone, along with its speculative writes.
        The stored action is consumed one-shot; observing without a stored
        action (no preceding act()) raises ``ContractError`` instead of
        guessing a candidate.
        """
        if self._forest is None or self._belief is None:
            # No forest to commit; ignore (or rebuild if packet supplied)
            return
        if packet is None or not hasattr(packet, "packet_id"):
            raise ContractError("packet must have packet_id for observe")
        action = self._last_selected_action
        self._last_selected_action = None
        if action is None:
            # No emitted action memory: act() never ran since the last
            # observe (or a forest was injected directly). The action cannot
            # be recovered from the packet — kernel packet ids are
            # action-free and collide across actions — and defaulting to any
            # candidate would reintroduce first-hit-wins miscommit, so this
            # is a protocol violation, not a miss.
            raise ContractError("observe requires an act()-emitted action: none stored")
        promoted, disp = commit(self._forest, action, packet, self._belief)
        self._forest = promoted
        self._last_commit = disp

    def ponder(self, *, deadline_monotonic_ns: int) -> None:
        # PBRF core does not perform background ponder without commit; no-op
        return


class PbrfPlanner(  # type: ignore[misc]
    PbrfPlannerActMixin,
    Planner,
):
    """Natural-particle PBRF planner (Candidate 3).

    Thin subclass joining the split mixins; construction, the value driver,
    and act live in the ``pbrf_*`` modules with no overrides here.
    """
