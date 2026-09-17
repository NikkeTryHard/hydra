"""Thin isolated-act judge over the ``hydra2_replay_rs.search`` boundary.

Rust runs ONE isolated arena act per call (detached whole-call, never
per-node); Python moves bytes across and fail-closes. No selection math,
node storage, RNG streams, belief sampling, or digest math lives here —
this module only validates arguments attached, forwards one native call,
and asserts the outcome (golden-digest compare + budget-counter asserts).

Resolution notes:
- Legacy entry point stays ``import hydra2_replay_rs`` (crate/lib names
  untouched); the ``hydra_bridge._native`` rename is a Phase-6 maturin
  ``module-name`` cutover and MUST NOT be attempted here.
- ``belief_refs`` are opaque ``u64`` world handles, never Python objects:
  the M1 belief producer closure runs entirely arena-side, so this shim
  makes NO per-sim Python callback (the single native call per
  :func:`act` is the whole shape — per-node attach is barred).
- ``completed=False`` carries the frontier action + counters only; the
  judge asserts counters without presenting the selection as complete.
- Canon-wins (B3): bytes/hash identity is canon-owned; the search side
  owns arena tables + selection. This shim compares digests, never mints.
- ``BATCH`` stays owned by ``columnar::BATCH``; no second literal
  appears here (widths cross as plain args when needed).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.artifacts.digest import require_digest_match
from hydra2.contracts.common import ContractError, DigestText

__all__ = [
    "ActJudge",
    "ActOut",
    "act",
    "act_stats",
]

#: Document bytes the shim accepts (bytes-like only; never ``str``).
_ACT_DOC_TYPES = (bytes, bytearray, memoryview)

#: ``u32`` / ``u64`` ceilings for the handle/id lanes.
_U32_MAX = 0xFFFF_FFFF
_U64_MAX = 0xFFFF_FFFF_FFFF_FFFF

#: Deadline mirror of ``search/common.py::_require_deadline_ms`` (`(0,60000]`);
#: the arena owns enforcement, the shim fails closed first so a bad budget
#: never crosses the detach. Mirrors ``search::MAX_DEADLINE_MS`` Rust-side.
_MAX_DEADLINE_MS = 60_000

#: Injectable native backend (tests monkeypatch this; production leaves None
#: so ``_search()`` imports the compiled extension).
_NATIVE_OVERRIDE: Any = None


def _native() -> Any:
    """Import the compiled bridge (fail closed, no fallback search)."""
    try:
        return importlib.import_module("hydra2_replay_rs")
    except ImportError as exc:
        raise RuntimeError(
            "hydra2_replay_rs extension with search not importable; "
            "build the bridge before using the search judge"
        ) from exc


def _search() -> Any:
    if _NATIVE_OVERRIDE is not None:
        return _NATIVE_OVERRIDE
    ext = _native()
    try:
        return ext.search
    except AttributeError as exc:
        raise RuntimeError("hydra2_replay_rs.search submodule missing; rebuild the bridge") from exc


def _require_u32(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"search {name} must be an int, got {type(value).__name__}")
    if value < 0 or value > _U32_MAX:
        raise ValueError(f"search {name} {value} outside u32 range")
    return value


def _require_u64(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"search {name} must be an int, got {type(value).__name__}")
    if value < 0 or value > _U64_MAX:
        raise ValueError(f"search {name} {value} outside u64 range")
    return value


def _require_budget(*, max_sims: object, max_depth: object, deadline_ms: object) -> None:
    if isinstance(max_sims, bool) or not isinstance(max_sims, int) or max_sims < 1:
        raise ValueError(f"search budget max_sims must be an int >= 1, got {max_sims!r}")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError(f"search budget max_depth must be an int >= 1, got {max_depth!r}")
    if (
        isinstance(deadline_ms, bool)
        or not isinstance(deadline_ms, int)
        or deadline_ms <= 0
        or deadline_ms > _MAX_DEADLINE_MS
    ):
        raise ValueError(
            f"search budget deadline_ms {deadline_ms!r} outside (0,{_MAX_DEADLINE_MS}]"
        )


@dataclass(frozen=True, slots=True)
class ActOut:
    """ONE isolated arena act outcome (owned values, detached compute).

    ``decision_digest`` is canon-form ``sha256:<hex>`` minted arena-side
    over canon-owned bytes. ``completed=False`` means the frontier action
    + counters only — never a complete selection.
    """

    action: int
    completed: bool
    sims_run: int
    nodes_visited: int
    decision_digest: DigestText


def act(
    *,
    spec_params: bytes | bytearray | memoryview,
    root_obs_doc: bytes | bytearray | memoryview,
    legal_ids: Sequence[int],
    belief_refs: Sequence[int],
    max_sims: int,
    max_depth: int,
    deadline_ms: int,
) -> ActOut:
    """Run ONE isolated arena act (exactly one native call, detached whole).

    ``spec_params``/``root_obs_doc`` are opaque bytes crossed once;
    ``legal_ids`` are distinct ``u32`` action ids (non-empty);
    ``belief_refs`` are opaque ``u64`` world handles (M1: sampled
    arena-side, never a per-sim Python callback). Budget counters are
    asserted on return: ``sims_run <= max_sims``, the action stays in
    ``legal_ids``, and a completed act ran at least one sim.
    """
    if not isinstance(spec_params, _ACT_DOC_TYPES):
        raise TypeError(f"search spec_params must be bytes-like, got {type(spec_params).__name__}")
    if not isinstance(root_obs_doc, _ACT_DOC_TYPES):
        raise TypeError(
            f"search root_obs_doc must be bytes-like, got {type(root_obs_doc).__name__}"
        )
    spec = bytes(spec_params)
    root = bytes(root_obs_doc)
    if len(spec) == 0:
        raise ValueError("search spec_params must be non-empty")
    if len(root) == 0:
        raise ValueError("search root_obs_doc must be non-empty")
    legal = [_require_u32(v, name="legal_ids entry") for v in legal_ids]
    if len(legal) == 0:
        raise ValueError("search legal_ids must be non-empty")
    if len(set(legal)) != len(legal):
        raise ValueError("search legal_ids must hold distinct ids")
    worlds = [_require_u64(v, name="belief_refs entry") for v in belief_refs]
    _require_budget(max_sims=max_sims, max_depth=max_depth, deadline_ms=deadline_ms)
    # ONE crossing per act (single-consume discipline): the full payload
    # goes in a single native call; no per-node callback exists.
    action, completed, sims_run, nodes_visited, decision_digest = _search().act_batch(
        spec, root, legal, worlds, int(max_sims), int(max_depth), int(deadline_ms)
    )
    action = _require_u32(action, name="outcome action")
    done = bool(completed)
    sims = int(sims_run)
    nodes = int(nodes_visited)
    digest = _bridge_contracts.make_digest_text(str(decision_digest))
    if sims < 0 or sims > int(max_sims):
        raise ContractError(f"search budget overrun: sims_run {sims} outside [0,{max_sims}]")
    if nodes < 0:
        raise ContractError(f"search nodes_visited {nodes} must be >= 0")
    if action not in set(legal):
        raise ContractError(f"search outcome action {action} outside legal_ids {sorted(legal)!r}")
    if done and sims < 1:
        raise ContractError("search completed act must have sims_run >= 1")
    return ActOut(
        action=action,
        completed=done,
        sims_run=sims,
        nodes_visited=nodes,
        decision_digest=digest,
    )


@dataclass(frozen=True, slots=True)
class ActJudge:
    """Fail-close comparator: assert recomputed == recorded, else raise."""

    subject: str = "search"

    def verify(self, *, recorded: str, out: ActOut) -> ActOut:
        """Compare the arena decision digest against the golden, fail close."""
        require_digest_match(
            recorded=recorded, recomputed=out.decision_digest, subject=self.subject
        )
        return out


def act_stats() -> tuple[int, int]:
    """Judge observability: ``(acts, completed)`` (never identity)."""
    acts, completed = _search().act_stats()
    return (int(acts), int(completed))


class TestSearchJudge:
    """Judge-contract unit asserts (in-module; no perf; pytest style)."""

    def test_verify_mismatch_raises(self):
        import hydra2._rust_search as search
        from hydra2.contracts.common import DigestMismatchError

        judge = search.ActJudge(subject="t")
        out = search.ActOut(
            action=1,
            completed=True,
            sims_run=48,
            nodes_visited=96,
            decision_digest=DigestText("sha256:" + "a" * 64),
        )
        judge.verify(recorded="sha256:" + "a" * 64, out=out)
        try:
            judge.verify(recorded="sha256:" + "b" * 64, out=out)
        except DigestMismatchError:
            return
        raise AssertionError("mismatch must raise DigestMismatchError")

    def test_completed_false_carries_frontier_counters(self, monkeypatch):
        import hydra2._rust_search as search

        class FakeNative:
            def act_batch(self, spec, root, legal, worlds, max_sims, max_depth, deadline_ms):
                return (7, False, 0, 0, "sha256:" + "c" * 64)

        monkeypatch.setattr(search, "_NATIVE_OVERRIDE", FakeNative())
        out = search.act(
            spec_params=b'{"algo":"ismcts"}',
            root_obs_doc=b'{"seat":0}',
            legal_ids=[3, 7, 9],
            belief_refs=[11, 22],
            max_sims=48,
            max_depth=6,
            deadline_ms=5000,
        )
        assert out.completed is False
        assert (out.sims_run, out.nodes_visited) == (0, 0)
        assert out.action == 7
        search.ActJudge(subject="t").verify(recorded="sha256:" + "c" * 64, out=out)

    def test_single_consume_single_native_call(self, monkeypatch):
        import hydra2._rust_search as search

        calls: list = []

        class FakeNative:
            def act_batch(self, spec, root, legal, worlds, max_sims, max_depth, deadline_ms):
                calls.append(
                    (
                        bytes(spec),
                        bytes(root),
                        list(legal),
                        list(worlds),
                        max_sims,
                        max_depth,
                        deadline_ms,
                    )
                )
                return (3, True, 48, 96, "sha256:" + "d" * 64)

        monkeypatch.setattr(search, "_NATIVE_OVERRIDE", FakeNative())
        out = search.act(
            spec_params=b'{"algo":"gumbel"}',
            root_obs_doc=b'{"seat":1}',
            legal_ids=[1, 3],
            belief_refs=[5],
            max_sims=48,
            max_depth=6,
            deadline_ms=5000,
        )
        assert len(calls) == 1, "one act must cross the boundary exactly once"
        spec, root, legal, worlds, max_sims, max_depth, deadline_ms = calls[0]
        assert (spec, root, legal, worlds) == (b'{"algo":"gumbel"}', b'{"seat":1}', [1, 3], [5])
        assert (max_sims, max_depth, deadline_ms) == (48, 6, 5000)
        assert out.completed is True
        assert out.sims_run == 48

    def test_budget_overrun_and_illegal_action_fail_closed(self, monkeypatch):
        import hydra2._rust_search as search
        from hydra2.contracts.common import ContractError

        class OverrunNative:
            def act_batch(self, spec, root, legal, worlds, max_sims, max_depth, deadline_ms):
                return (1, True, 49, 10, "sha256:" + "e" * 64)

        monkeypatch.setattr(search, "_NATIVE_OVERRIDE", OverrunNative())
        try:
            search.act(
                spec_params=b'{"a":1}',
                root_obs_doc=b'{"s":0}',
                legal_ids=[1, 2],
                belief_refs=[],
                max_sims=48,
                max_depth=6,
                deadline_ms=5000,
            )
        except ContractError:
            pass
        else:
            raise AssertionError("sims_run > max_sims must raise ContractError")

        class IllegalNative:
            def act_batch(self, spec, root, legal, worlds, max_sims, max_depth, deadline_ms):
                return (99, True, 10, 10, "sha256:" + "e" * 64)

        monkeypatch.setattr(search, "_NATIVE_OVERRIDE", IllegalNative())
        try:
            search.act(
                spec_params=b'{"a":1}',
                root_obs_doc=b'{"s":0}',
                legal_ids=[1, 2],
                belief_refs=[],
                max_sims=48,
                max_depth=6,
                deadline_ms=5000,
            )
        except ContractError:
            return
        raise AssertionError("action outside legal_ids must raise ContractError")
