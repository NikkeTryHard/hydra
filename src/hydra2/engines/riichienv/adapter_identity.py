"""RiichiEnvExactSimulator: the WP-03A reference ExactSimulator.

Design decisions recorded here (full rationale in work_packages/WP-03A):

D-WP03A-1 Per-hand engines with adapter-driven chaining. RiichiEnv 0.4.10
    honours ``reset(wall=...)`` for the first hand only; later hands come from
    engine-internal RNG (verified: identical injected walls diverge at kyoku
    2). The adapter therefore plays each hand on a fresh engine instance fed
    with ``reset(oya/honba/kyotaku/scores/round_wind/wall=...)``. Carry
    parameters between hands are read from the ENGINE's own native
    ``start_kyoku`` advance (emitted inside the same step batch that closed
    the previous hand), so renchan/honba/stick/agari-yame/sudden-death/tobi
    logic stays engine-faithful while walls stay fully injected. The engine's
    RNG-dealt follow-up hand is discarded unplayed; nothing derived from it
    reaches any public or private surface. Continuation walls derive from the
    pinned WallSchedule via :mod:`hydra2.engines.riichienv.walls` (named
    stream ``hydra2.wall_continuation_v1``); the seed parameter is never
    touched on formal paths.

D-WP03A-5 Buffered response windows. RiichiEnv resolves claims from ONE
    simultaneous ``step`` over all responders (verified: partial submission
    resolves immediately and silently drops other claimants). The adapter
    buffers individual responder decisions and submits one combined step;
    ``call_window`` opens a discard-offered window and one server-private
    ``call_resolved`` closes it ahead of the outcome envelopes (accepted id
    taken from what the engine actually executed). Kan-offered windows
    (chankan) emit no window pair because the grammar routes ``kakan -> ron``
    directly.

D-WP03A-6 Multi-ron attribution. Concurrent hora events of one resolution
    merge into a single ``ron`` envelope: the first winner owns
    actor/action-id (matching the engine's stick rule), deltas sum, and
    SettlementFact entries carry every winner.

D-WP03A-8 Hand-scoped observation builders. One ObservationBuilder per hand
    keeps ``visible_history`` inside the current hand's public stream and
    avoids unresettable per-seat caches leaking across hands; determinism is
    unaffected because hands chain through injected walls.

Owns the shared caches and gates beside their only readers: the wind map
and engine action-type alias, the action-table and event-schema caches
with their loaders, and the rules support gate plus rules-identity helper
the simulator constructor calls. The simulator itself lives in
:mod:`hydra2.engines.riichienv.adapter_core`, the decision core in
:mod:`hydra2.engines.riichienv.adapter_step`, and mjai translation in
:mod:`hydra2.engines.riichienv.adapter_events_a` and
:mod:`hydra2.engines.riichienv.adapter_events_b`, so each file stays
inside the review-size ceiling.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import riichienv

from hydra2.config import repo_root as repo_root
from hydra2.contracts.action import ACTION_TABLE_RELPATH as ACTION_TABLE_RELPATH
from hydra2.contracts.action import load_action_table as load_action_table
from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.common import UnsupportedRuleError as UnsupportedRuleError
from hydra2.contracts.event_schema import EVENT_SCHEMA_RELPATH as EVENT_SCHEMA_RELPATH
from hydra2.contracts.event_schema import parse_event_schema as parse_event_schema

if TYPE_CHECKING:
    from hydra2.contracts.action import ActionTable as ActionTable
    from hydra2.contracts.rules import RulesManifest as RulesManifest

__all__ = [
    "_AT",
    "_BAKAZE_TO_TILE_TYPE",
    "_EVENT_SCHEMA_CACHE",
    "_TABLE_CACHE",
    "_action_table",
    "_event_schema_hash",
    "_rules_identity",
    "_validate_rules",
]

_BAKAZE_TO_TILE_TYPE = {"E": 27, "S": 28, "W": 29, "N": 30}
_AT = riichienv.ActionType

_TABLE_CACHE: dict[str, ActionTable] = {}
_EVENT_SCHEMA_CACHE: dict[str, str] = {}


def _action_table() -> ActionTable:
    root = str(repo_root())
    if root not in _TABLE_CACHE:
        _TABLE_CACHE[root] = load_action_table(Path(root) / ACTION_TABLE_RELPATH)
    return _TABLE_CACHE[root]


def _event_schema_hash() -> str:
    root = str(repo_root())
    if root not in _EVENT_SCHEMA_CACHE:
        document: dict[str, Any] = cast(
            "dict[str, Any]", parse_event_schema((Path(root) / EVENT_SCHEMA_RELPATH).read_bytes())
        )
        payload: Any = document["payload"]
        if not isinstance(payload, dict) or "digest" not in payload:
            raise ContractError("event schema artifact lacks a digest")
        payload_dict: dict[str, Any] = cast("dict[str, Any]", payload)
        _EVENT_SCHEMA_CACHE[root] = str(cast("Any", payload_dict["digest"]))
    return _EVENT_SCHEMA_CACHE[root]


def _validate_rules(rules: RulesManifest) -> None:
    """Structural support gate; failures happen BEFORE any game starts."""
    if rules.players != 4:
        raise UnsupportedRuleError(
            f"reference adapter supports 4-player games, got {rules.players}"
        )
    if rules.match_length != "hanchan":
        raise UnsupportedRuleError(
            f"reference adapter pins match_length='hanchan', got {rules.match_length!r}"
        )
    if tuple(rules.red_tile_ids) != (16, 52, 88):
        raise UnsupportedRuleError(f"unsupported red-five encoding {rules.red_tile_ids!r}")
    if rules.kuikae_policy != "forbidden":
        raise UnsupportedRuleError(
            f"RiichiEnv hard-forbids kuikae; manifest declares {rules.kuikae_policy!r}"
        )
    for entry in rules.adapter_compatibility:
        if entry.adapter_id == "riichienv" and entry.status not in ("supported", "qualified"):
            raise UnsupportedRuleError(
                f"manifest marks adapter riichienv as {entry.status!r}; refusing to run"
            )


def _rules_identity(manifest: RulesManifest, recomputed: str) -> str:
    """Published artifact bytes win when present (D-WP03A-4 refinement).

    The published configs/rules file is the authority its digest was recorded
    from; the payload recompute stays as the fallback for manifests without a
    published artifact.
    """
    published = Path(repo_root()) / "configs" / "rules" / f"{manifest.rules_id}.json"
    if not published.is_file():
        return recomputed
    return "sha256:" + hashlib.sha256(published.read_bytes()).hexdigest()
