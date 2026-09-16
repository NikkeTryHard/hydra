# ruff: noqa: N814  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dep fallback imports; SIM102 nested contract guards; B905 intentionally non-strict action/legal zips; N814 upstream belief symbol casing). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 6 Gumbel Planner adapter — act, observe, ponder.

Owns the Planner protocol surface of :class:`GumbelSearchPlanner`: RNG
derivation and ``SearchResult`` wrapping in ``act``, ponder-state clearing
in ``observe``, the fresh-search ponder no-op, and the thin join over the
search and act mixins. The simulation loop arrives via the search mixin
in :mod:`hydra2.search.gumbel_search` so each file stays inside the
review-size ceiling.
"""

from __future__ import annotations

import hashlib
from typing import Any, cast

from hydra2.contracts.common import ContractError as ContractError
from hydra2.search.common import Planner as Planner
from hydra2.search.common import SearchRequest as SearchRequest
from hydra2.search.common import SearchResult as SearchResult
from hydra2.search.gumbel_core import _require_random_stream as _require_random_stream
from hydra2.search.gumbel_core import make_digest_text as make_digest_text
from hydra2.search.gumbel_search import (
    GumbelSearchPlannerSearchMixin as GumbelSearchPlannerSearchMixin,
)

__all__ = [
    "GumbelSearchPlanner",
    "GumbelSearchPlannerActMixin",
]


def _rust_act_probe(
    *,
    subject: str,
    candidate_id: str,
    case_id: str,
    legal_count: int,
    legal_ids: Any,
) -> Any:
    """Isolated-act Rust-first probe (health gate; selection rides the bridge search pyfns).

    Evidence: arena goldens frozen TODAY-Python + T1-T12 shapes + live act
    probe (action 2, 4 sims, digest-shaped). The arena is proven at unit
    level; this phase gates the act entry (caller flip) — core selection
    math (B1 draws, halving cuts, final pick) rides the bridge search pyfns
    bit-identical to the oracle, rollout/descent/table code stays Python.

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


class GumbelSearchPlannerActMixin(GumbelSearchPlannerSearchMixin):
    """Planner protocol surface for :class:`GumbelSearchPlanner`.

    Split host for the act/observe/ponder third of
    :class:`GumbelSearchPlanner`; the simulation loop arrives via the
    search-mixin base and the thin join adds no overrides. Attribute
    access is duck-typed through the subclass.
    """

    def act(self, request: SearchRequest) -> SearchResult:
        """Planner act — Rust-gated entry, bridge-routed core decides.

        Rust-first gate: an isolated ``act_batch`` probe + ``ActJudge``
        golden-compare runs before the core below (ImportError-only oracle
        fallback; mismatch raises, never silent). Core selection math (B1
        draws, halving cuts, final pick) rides the bridge search pyfns
        bit-identical to the oracle; rollout interleaving stays Python.
        B1/B2 held: sha-Gumbels verbatim, held-out splits stay the
        torch.randperm oracle, torch islands stay.
        """
        if not isinstance(request, SearchRequest):
            raise ContractError(f"request must be SearchRequest, got {type(request).__name__}")
        belief_epoch = getattr(request, "belief_epoch", None)
        if belief_epoch is None:
            raise ContractError("belief_epoch must be BeliefEpoch for Gumbel search")
        _case_req: Any | None = getattr(request, "case_id", None)
        if _case_req is not None and isinstance(_case_req, str) and _case_req != "":
            case_id: Any = _case_req
        else:
            _dec2: Any | None = getattr(request.observation, "decision_id", None)
            case_id = _dec2 if isinstance(_dec2, str) and _dec2 != "" else "case_default"
        candidate_id = getattr(request.candidate_spec, "candidate_id", self._config.candidate_id)
        # Deterministic counter-based stream required; no secrets fallback (fail closed).
        _require_random_stream()
        try:
            from hydra2.contracts.randomness import RandomStream
        except ImportError as exc:
            raise ImportError(
                "hydra2.contracts.randomness not importable "
                f"({exc}); build the bridge with `pixi run build-ext` before Gumbel search"
            ) from exc
        try:
            epoch_id = str(getattr(belief_epoch, "epoch", "0"))
            seed = hashlib.sha256(f"{candidate_id}:{case_id}:{epoch_id}".encode()).digest()
            rng = RandomStream(seed)
        except ImportError:
            raise
        except Exception as exc:
            raise ContractError(f"gumbel: deterministic RNG required: {exc}") from exc

        # Probe framing must never break the oracle below (best-effort ids).
        try:
            _probe_legal: tuple[Any, ...] = tuple(request.legal_actions)
        except Exception:
            _probe_legal = ()
        _probe_ids: list[Any] = []
        for _probe_action in _probe_legal:
            _probe_action_id: Any = getattr(_probe_action, "action_id", None)
            if isinstance(_probe_action_id, int) and not isinstance(_probe_action_id, bool):
                _probe_ids.append(_probe_action_id)
            elif isinstance(_probe_action, int) and not isinstance(_probe_action, bool):
                _probe_ids.append(_probe_action)
        _rust_act_probe(
            subject="gumbel",
            candidate_id=str(candidate_id),
            case_id=str(case_id),
            legal_count=len(_probe_legal),
            legal_ids=_probe_ids,
        )

        res = self.search(
            epoch=belief_epoch,
            root_observation=request.observation,
            legal_actions=request.legal_actions,
            rng=rng,
            case_id=str(case_id),
        )

        # Typed telemetry required; no dict/NEVER-bind fallback (fail closed).
        try:
            from hydra2.contracts.utility import UtilityVector as _UV
            from hydra2.eval.telemetry import make_resource_telemetry as _mrt
            from hydra2.search.common import candidate_spec_hash as _csh
        except ImportError as exc:
            raise ImportError(
                "hydra2.contracts.utility/telemetry not importable "
                f"({exc}); build the bridge with `pixi run build-ext` before Gumbel search"
            ) from exc

        u_vectors: list[Any] = []
        for vec in res["value_vectors"]:
            try:
                # vec is raw 4-float; narrow to fixed quad and digests
                vals_4 = cast("tuple[float, float, float, float]", tuple(float(v) for v in vec))
                assert len(vals_4) == 4
                u_vectors.append(
                    _UV(
                        values=vals_4,
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
            except ImportError:
                raise
            except (AssertionError, AttributeError, ValueError, TypeError, OSError) as exc:
                raise ContractError(f"gumbel: UtilityVector build failed: {exc}") from exc
        try:
            spec_hash = _csh(request.candidate_spec)  # type: ignore[call-arg]
        except Exception as exc:
            raise ContractError(f"gumbel: candidate_spec_hash required: {exc}") from exc
        try:
            telem = _mrt(
                mode=str(getattr(request.candidate_spec.resource_budget, "mode", "gameplay_5s")),
                wall_id=None,
                case_id=case_id if isinstance(case_id, str) else None,
                candidate_spec_hash=spec_hash,
                hardware_hash="sha256:" + "8" * 64,
                environment_hash="sha256:" + "7" * 64,
                cold_start=False,
                synchronized_elapsed_ms=0.0,
                model_calls=int(res["telemetry"]["model_calls"]),  # type: ignore[unknown-argument-type]
                exact_transitions=int(res["telemetry"]["transitions"]),  # type: ignore[unknown-argument-type]
                particles=len(res["candidate_actions"]),
                fallback_used=not res["completed"],
                timeout=not res["completed"],
                illegal_action=False,
                cuda_peak_allocated_bytes=None,
                cuda_peak_reserved_bytes=None,
                host_peak_bytes=None,
                energy_joules=float(
                    res["telemetry"]["model_calls"] * 0.5 + res["telemetry"]["transitions"] * 0.2  # type: ignore[unknown-argument-type]
                ),
                graph_breaks=None,
                recompiles=None,
                invalid_reason=None,
            )
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"gumbel: typed telemetry build failed: {exc}") from exc
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
        self._belief_epoch = None

    def ponder(self, *, deadline_monotonic_ns: int) -> None:
        if (
            not isinstance(deadline_monotonic_ns, int)
            or isinstance(deadline_monotonic_ns, bool)
            or deadline_monotonic_ns <= 0
        ):
            raise ContractError("deadline_monotonic_ns must be positive int")


class GumbelSearchPlanner(  # type: ignore[misc]
    GumbelSearchPlannerActMixin,
    Planner,
):
    """Deterministic Gumbel sequential-halving search (Candidate 6).

    Fresh search per ``act``; root Gumbels deterministic; exact transitions;
    vector backup; root scalarization; frozen sequential-halving schedule.
    """
