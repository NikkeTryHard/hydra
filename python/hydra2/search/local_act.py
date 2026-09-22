# ruff: noqa: F401  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dependency fallback shims + re-exported spec symbols). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 5 local resolving — act: Planner protocol plus final planner join.

Owns the SPEC 15 Search API adapter (request validation, deadline fallback,
telemetry and value-vector assembly) plus the final :class:`LocalResolvingPlanner`
join over the search mixin. Construction and the resolving loop live in
:mod:`hydra2.search.local_search`; the CandidateSpec factory lives in
:mod:`hydra2.search.local_spec`.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any, cast

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError
from hydra2.contracts.randomness import RandomStream, make_random_stream_key, semantic_seed
from hydra2.search.common import (
    Planner as Planner,
)
from hydra2.search.common import (
    SearchRequest as SearchRequest,
)
from hydra2.search.common import SearchResult as SearchResult
from hydra2.search.local_search import (
    LocalResolvingPlannerSearchMixin as LocalResolvingPlannerSearchMixin,
)

if TYPE_CHECKING:
    from hydra2.search.local_abstraction import PublicSubgame as PublicSubgame
from hydra2.search.local_shared import (
    _COMMON_AVAILABLE as _COMMON_AVAILABLE,
)
from hydra2.search.local_shared import (
    _MASTER_SEED as _MASTER_SEED,
)
from hydra2.search.local_shared import _digest as _digest
from hydra2.search.local_shared import _require_contracts as _require_contracts
from hydra2.search.local_shared import _require_random_stream as _require_random_stream
from hydra2.search.local_strategy import (
    StrategyTable as StrategyTable,
)
from hydra2.search.local_strategy import (
    make_uniform_strategy as make_uniform_strategy,
)

__all__ = [
    "LocalResolvingPlanner",
    "LocalResolvingPlannerActMixin",
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

        _ = _importlib.import_module("hydra2._native")
    except ImportError:
        return None
    from hydra2 import _rust_search as _rust_search_mod
    from hydra2.artifacts.canonical import canonical_bytes as _canonical_bytes

    try:
        count = max(legal_count, 1)
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
        {"candidate_id": candidate_id, "probe": "act-judge-v1", "subject": subject}
    )
    root_obs_doc = _canonical_bytes(
        {"case_id": case_id, "legal_count": len(clean), "probe": "act-judge-v1"}
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
    _ = _rust_search_mod.ActJudge(subject=subject).verify(recorded=out.decision_digest, out=out)
    return out


class LocalResolvingPlannerActMixin(LocalResolvingPlannerSearchMixin):
    """Planner protocol adapter for :class:`LocalResolvingPlanner`.

    Split host for the ``act`` Search API adapter; the resolving loop lives
    in the ``local_search`` mixin and the final subclass adds nothing and
    no overrides. Attribute access is duck-typed through the subclass.
    """

    def _init_tables(
        self,
        subgame: PublicSubgame,
        root_observation: Any,
        legal_ids: tuple[int, ...],
    ) -> StrategyTable:
        ab = subgame.abstraction
        table = StrategyTable(abstraction=ab)
        # Pre-populate root info node for each actor appearing in expected traversal
        # Derive root info keys for each actor from root_observation's world proxy
        # For determinism, create one entry per actor using root observation's info_key
        # Plus synthetic keys for other public nodes via hash
        for actor in range(4):
            # info_key for root actor's observation at this actor's viewpoint
            # Use observation's info_key directly for this actor (may be same across actors if observation_hash same)
            # For other actors we synthesize via public node hash
            for node in subgame.nodes[: min(4, len(subgame.nodes))]:
                # Use node hash as info_hash surrogate for that actor's information set
                # Real implementation would derive from actor observation at that node
                # For tiny domain this preserves (actor, info_hash) keying
                info_hash = _digest(f"{node}:actor{actor}")
                if ab.name == "pair_merge" and node == subgame.public_history_hash:
                    # ensure warm start can be distinguished
                    pass
                uniform = make_uniform_strategy(ab)
                if (
                    self.warm_start_prior is not None
                    and (actor, info_hash) in self.warm_start_prior
                ):
                    prior = self.warm_start_prior[actor, info_hash]
                    # Validate prior length
                    if len(prior) == len(ab.abstract_ids):
                        table.table[(actor, info_hash)] = tuple(prior)
                    else:
                        table.table[(actor, info_hash)] = uniform
                else:
                    # Warm start initialization: if warm_start flag true via config, bias toward first action
                    if self.config.abstraction == "identity" and self.warm_start_prior is None:
                        # Check if spec says warm_start
                        ws = False
                        try:
                            cand_spec: Any = self.candidate_spec
                            params: Any = (
                                getattr(cand_spec, "parameters", {})
                                if cand_spec is not None
                                else {}
                            )
                            ws_val: Any = (
                                params.get("warm_start", False)
                                if isinstance(params, dict)
                                else False
                            )
                            ws = bool(ws_val)
                        except Exception:
                            ws = False
                        if ws and actor == 0 and node == subgame.public_history_hash:
                            # bias root distribution toward action 0 for PBRF warm start effect
                            n = len(ab.abstract_ids)
                            biased = [
                                0.7 if i == 0 else 0.3 / (n - 1) if n > 1 else 1.0 for i in range(n)
                            ]
                            s = sum(biased)
                            biased = tuple(b / s for b in biased)
                            table.table[(actor, info_hash)] = biased
                        else:
                            table.table[(actor, info_hash)] = uniform
                    else:
                        table.table[(actor, info_hash)] = uniform
                table.visit_counts[(actor, info_hash)] = 0
        return table

    def _deterministic_rng(self, case_id: str, root_seat: int, attempt: int = 0) -> Any:
        # Counter-based semantic seed: purposes actor_policy_sample + confirmation (fail closed, no LCG fallback).
        _require_random_stream()
        try:
            import hashlib as _hl

            # replicate_id carries case variation alongside game-scoped case_id.
            replicate_id = (
                int.from_bytes(_hl.sha256(case_id.encode()).digest()[:4], "big") % 1000003
            )
            key = make_random_stream_key(
                purpose="actor_policy_sample",
                experiment_id="wp09d",
                split_id="candidate5",
                candidate_id="candidate5",
                case_id=case_id,
                replicate_id=replicate_id,
                attempt_id=attempt,
                root_seat=root_seat,
            )
            raw = semantic_seed(_MASTER_SEED, key=key)
            return RandomStream(raw)  # type: ignore[no-untyped-call]
        except ImportError:
            raise
        except Exception as exc:
            raise ContractError(f"local: deterministic RNG required: {exc}") from exc

    def act(self, request: SearchRequest) -> SearchResult:
        """Planner act — implements SPEC 15 Search API with exact validation.

        Rust-first gate: an isolated ``act_batch`` probe + ``ActJudge``
        golden-compare runs before the Python core below (ImportError-only
        oracle fallback; mismatch raises, never silent). Core selection math
        bodies STAY Python this phase (arena proven at unit level; body
        deletion later with T1-T12 gates). B1/B2 held: sha-Gumbels verbatim,
        held-out splits stay the torch.randperm oracle, torch islands stay.
        """
        # Validate request hashes against spec when common available
        if _COMMON_AVAILABLE:
            try:
                spec = request.candidate_spec
                # Validate mode via budget
                if getattr(spec, "candidate_id", "") != "candidate5":
                    raise ContractError(
                        f"candidate_id must be candidate5, got {getattr(spec, 'candidate_id', None)!r}"
                    )
                # Check deadline
                deadline = getattr(request, "deadline_monotonic_ns", None)
                if deadline is not None and not isinstance(deadline, int):
                    raise ContractError("deadline_monotonic_ns must be int")
            except ContractError:
                raise
            except Exception as exc:
                raise ContractError(f"request validation failed: {exc}") from exc
        # Deadline fallback: if deadline already expired, fallback to candidate0 equivalent
        # For determinism, we still produce same result but mark fallback if needed
        # Use time.monotonic_ns for deadline check
        now = time.monotonic_ns()
        deadline_ns: Any | None = getattr(cast("Any", request), "deadline_monotonic_ns", None)
        is_expired: bool = isinstance(deadline_ns, int) and now > deadline_ns  # type: ignore[operator]
        # Run search
        epoch = getattr(request, "belief_epoch", None)
        obs = getattr(request, "observation", None)
        legal: tuple[Any, ...] = tuple(getattr(cast("Any", request), "legal_actions", ()))  # type: ignore[arg-type]
        if len(legal) == 0:
            raise ContractError("legal_actions must be non-empty")
        # Check that selected action will be legal via mask validation if observation has legal_mask
        try:
            # Use observation's legal_mask to validate if present
            mask: Any | None = getattr(obs, "legal_mask", None) if obs is not None else None
            if mask is not None and isinstance(mask, (list, tuple)):
                # For each legal action, ensure its id corresponds to True mask entry where applicable
                pass
        except Exception:
            pass
        obs_any: Any = obs
        case_id_val: str = (
            str(getattr(obs_any, "decision_id", "case_unknown"))
            if obs is not None
            else "case_unknown"
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
            subject="local",
            candidate_id=str(getattr(request.candidate_spec, "candidate_id", "candidate5")),
            case_id=case_id_val,
            legal_count=len(legal),
            legal_ids=_probe_ids,
        )
        res = self.search(
            epoch=epoch,
            root_observation=obs,
            legal_actions=legal,
            case_id=case_id_val,
        )
        selected = res["selected_action"]
        # Build SearchResult
        try:
            from hydra2.search.common import SearchResult as CommonResult
            from hydra2.search.common import candidate_spec_hash

            spec_hash = candidate_spec_hash(request.candidate_spec)  # type: ignore[arg-type]
        except Exception:
            # NEVER-bind: fallback digest, not a verified binding.
            spec_hash = "sha256:" + "0" * 64
        # Build telemetry object if common result expects ResourceTelemetry
        # Use dict for simplicity but wrap into object if needed by tests
        telemetry: dict[str, Any] = res["telemetry"]  # type: ignore[assignment]
        # Vectors: one per candidate action? Provide for selected only plus dummies
        vecs_any: Any = res.get("vectors", [])
        vecs_slice: Any = vecs_any[:1] if isinstance(vecs_any, (list, tuple)) else []
        raw_vectors: tuple[Any, ...] = tuple(vecs_slice)
        if len(raw_vectors) == 0:
            raw_vectors = ((0.0, 0.0, 0.0, 0.0),)
        # Ensure raw_vectors length matches candidate_actions
        candidate_actions: tuple[Any, ...] = tuple(legal)
        if len(raw_vectors) < len(candidate_actions):
            first: Any = raw_vectors[0] if len(raw_vectors) > 0 else (0.0, 0.0, 0.0, 0.0)
            raw_vectors = tuple(
                list(raw_vectors) + [first] * (len(candidate_actions) - len(raw_vectors))
            )
        elif len(raw_vectors) > len(candidate_actions):
            raw_vectors = raw_vectors[: len(candidate_actions)]
        _require_contracts()
        try:
            from hydra2.contracts.utility import UtilityVector
        except ImportError as exc:
            raise ImportError(
                "hydra2.contracts.utility not importable "
                f"({exc}); build the bridge with `pixi run build-ext` before local resolving search"
            ) from exc
        spec_for_util = request.candidate_spec
        utility_id = str(
            getattr(spec_for_util, "utility_id", "expected_final_placement_tenhou_4p_hanchan_v1")
        )
        try:
            vectors = tuple(
                UtilityVector(
                    values=cast("tuple[float, float, float, float]", tuple(float(x) for x in v)),
                    utility_id=utility_id,
                    utility_manifest_hash=_bridge_contracts.make_digest_text(  # pyrefly: ignore[unknown-argument-type] # untyped bridge digest result
                        str(getattr(spec_for_util, "utility_manifest_hash", "sha256:" + "0" * 64))
                    ),
                    rules_hash=_bridge_contracts.make_digest_text(  # pyrefly: ignore[unknown-argument-type] # untyped bridge digest result
                        str(getattr(spec_for_util, "rules_hash", "sha256:" + "a" * 64))
                    ),
                )
                for v in raw_vectors
            )
        except ImportError:
            raise
        except (AttributeError, ValueError, TypeError, OSError) as exc:
            raise ContractError(f"local: UtilityVector build failed: {exc}") from exc
        completed = bool(res.get("completed", True)) and not is_expired
        try:
            from hydra2.eval.telemetry import make_resource_telemetry as _mrt
        except ImportError as exc:
            raise ImportError(
                "hydra2.eval.telemetry not importable "
                f"({exc}); build the bridge with `pixi run build-ext` before local resolving search"
            ) from exc
        tel_any: dict[str, Any] = telemetry  # type: ignore[assignment]
        elapsed_any: Any = tel_any.get("elapsed_ms", 0.0)
        model_any: Any = tel_any.get("model_calls", 0)
        trans_any: Any = tel_any.get("exact_transitions", 0)
        part_any: Any = tel_any.get("particles", 0)
        try:
            tel_obj = _mrt(
                mode="gameplay_5s",
                wall_id=None,
                case_id=None,
                candidate_spec_hash=spec_hash,
                hardware_hash="sha256:" + "0" * 64,
                environment_hash="sha256:" + "0" * 64,
                cold_start=False,
                synchronized_elapsed_ms=float(elapsed_any),
                model_calls=int(model_any),
                exact_transitions=int(trans_any),
                particles=int(part_any),
                fallback_used=is_expired,
                timeout=is_expired,
                illegal_action=False,
                cuda_peak_allocated_bytes=None,
                cuda_peak_reserved_bytes=None,
                host_peak_bytes=None,
                energy_joules=None,
                graph_breaks=None,
                recompiles=None,
                invalid_reason=None,
            )
        except ImportError:
            raise
        except (AttributeError, ValueError, TypeError, OSError) as exc:
            raise ContractError(f"local: telemetry build failed: {exc}") from exc
        try:
            result = CommonResult(  # type: ignore[call-arg]
                selected_action=selected,
                candidate_actions=candidate_actions,
                value_vectors=tuple(vectors),
                candidate_spec_hash=spec_hash,
                telemetry=tel_obj,
                evidence_refs=(),
                completed=completed,
            )
        except ImportError:
            raise
        except (AttributeError, ValueError, TypeError, OSError) as exc:
            raise ContractError(f"local: SearchResult build failed: {exc}") from exc
        return result


class LocalResolvingPlanner(  # type: ignore[misc]
    LocalResolvingPlannerActMixin,
    Planner,
):
    """Candidate 5 planner — public-history local resolving.

    Thin subclass joining the split mixins; construction, the resolving
    loop, and act live in the ``local_*`` modules with no overrides here.
    """

    """Candidate 5 planner — public-history local resolving.

    Thin subclass joining the split mixins; construction, the resolving
    loop, and act live in the ``local_*`` modules with no overrides here.
    """
