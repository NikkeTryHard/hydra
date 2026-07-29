"""WP-04A reference trace runner.

Drives the :class:`~hydra2.engines.riichienv.RiichiEnvExactSimulator` through
a scripted policy over an injected wall, freezes an *expected* trace derived
from the rules manifest + Tenhou evidence (NEVER from engine output), and
compares adapter behaviour against it. The first counterexample of a case is
persisted with full inputs, outputs and hashes under
``$HYDRA2_ARTIFACT_ROOT/counterexamples/WP-04A/<case_id>.json``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra2.artifacts.atomic import atomic_replace_bytes
from hydra2.artifacts.digest import of_bytes, of_canonical
from hydra2.config import artifact_root, repo_root
from hydra2.contracts.common import Seat, TileId
from hydra2.contracts.rules import resolve_final_ranks
from hydra2.engines.protocol import WallSchedule, wall_schedule_digest
from hydra2.engines.riichienv import RiichiEnvExactSimulator

if TYPE_CHECKING:
    from collections.abc import Callable

    from hydra2.contracts.action import CanonicalAction
    from hydra2.contracts.event import EventEnvelope
    from hydra2.contracts.rules import RulesManifest
__all__ = [
    "CaseResult",
    "ExpectationStep",
    "ScriptedDecision",
    "TraceExpectation",
    "TraceRunnerError",
    "expect_event_kinds",
    "expect_predicate",
    "expect_terminal_scores",
    "wall_schedule_for",
]


class TraceRunnerError(RuntimeError):
    """A case could not be executed as scripted (construction bug, not DUT)."""


def wall_schedule_for(case_id: str, tiles: tuple[int, ...]) -> WallSchedule:
    """Named deterministic schedule for one corpus case."""
    if len(tiles) != 136:
        raise TraceRunnerError(f"{case_id}: wall must hold exactly 136 tiles")
    schedule_id = f"wp04a-{case_id}"
    frozen = tuple(TileId(t) for t in tiles)
    return WallSchedule(
        schedule_id=schedule_id,
        physical_tiles=frozen,
        digest=wall_schedule_digest(schedule_id, frozen),
    )


@dataclass(frozen=True, slots=True)
class ScriptedDecision:
    """One decision point: which canonical kind the next actor accepts.

    ``tile`` disambiguates several legal actions of one kind (physical id for
    discards/claims/kakan). ``kind="pass"`` answers a claim window with pass;
    ``kind="auto"`` answers pass when offered, else tsumogiris/discards the
    drawn tile - the neutral continuation used between scripted highlights.
    ``negate=True`` inverts the step into an expectation: the named action
    MUST NOT be legal for the acting seat (kuikae/furiten bars); the driver
    then applies the neutral action. A violation is recorded, never raised,
    so the case resolves to a persisted mismatch instead of a crash.
    """

    kind: str
    tile: int | None = None
    negate: bool = False


def _auto_action(sim: RiichiEnvExactSimulator, actor: int) -> CanonicalAction:
    actions = sim.legal_actions(Seat(actor))
    passes = [a for a in actions if a.kind == "pass"]
    if len(passes) != 0:
        return passes[0]
    assert sim._env is not None
    drawn = sim._env.drawn_tile if sim._mode == "draw" else None
    if drawn is not None:
        tsumogiri = [
            a
            for a in actions
            if a.kind == "tsumogiri"
            and a.tile is not None
            and int(a.tile) == drawn
        ]
        if len(tsumogiri) != 0:
            return tsumogiri[0]
    discards = [a for a in actions if a.kind in ("discard", "tsumogiri")]
    if len(discards) != 0:
        return discards[0]
    raise TraceRunnerError(
        f"seat {actor}: auto policy found no neutral action (legals {[(a.kind,) for a in actions]})"
    )


def _find_action(
    sim: RiichiEnvExactSimulator,
    actor: int,
    decision: ScriptedDecision,
) -> CanonicalAction:
    actions = sim.legal_actions(Seat(actor))
    if decision.kind == "pass":
        passes = [a for a in actions if a.kind == "pass"]
        if len(passes) != 0:
            return passes[0]
        raise TraceRunnerError(
            f"seat {actor}: no pass offered (legals "
            + str([(a.kind, a.tile, a.called_tile) for a in actions])
        )
    candidates = [a for a in actions if a.kind == decision.kind]
    if len(candidates) == 0:
        raise TraceRunnerError(
            f"seat {actor}: scripted {decision.kind} not offered "
            f"(legals {[(a.kind, int(a.tile) if a.tile is not None else None) for a in actions]})"
        )
    if decision.tile is not None:
        exact = [
            a
            for a in candidates
            if (a.tile is not None and int(a.tile) == decision.tile)
            or (a.called_tile is not None and int(a.called_tile) == decision.tile)
        ]
        if len(exact) == 0:
            raise TraceRunnerError(
                f"seat {actor}: scripted {decision.kind} tile {decision.tile} "
                f"not among " + str([(a.kind, a.tile, a.called_tile) for a in candidates])
            )
        return exact[0]
    return candidates[0]


@dataclass(slots=True)
class _DriverState:
    steps_consumed: int = 0
    applied_log: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


def _drive(
    sim: RiichiEnvExactSimulator,
    script: tuple[ScriptedDecision, ...],
    state: _DriverState,
    *,
    max_applies: int = 4000,
) -> None:
    """Apply scripted decisions until the simulation is terminal.

    One scripted step per ``apply()``; a decision consumed by an unexpected
    actor is a construction bug, so the driver records the actor per step.
    """
    while not sim._terminal:
        actor = sim._expected_actor_or_none()
        if actor is None:
            raise TraceRunnerError("simulation stalled without a pending decision")
        index = state.steps_consumed
        if index >= len(script):
            raise TraceRunnerError(
                f"script exhausted at seat {actor} before terminal (applied {state.applied_log})"
            )
        decision = script[index]
        if decision.negate:
            offered = [
                a
                for a in sim.legal_actions(Seat(actor))
                if a.kind == decision.kind
                and (
                    decision.tile is None
                    or (a.tile is not None and int(a.tile) == decision.tile)
                )
            ]
            if len(offered) != 0:
                state.violations.append(
                    f"seat {actor}: forbidden {decision.kind} was offered "
                    f"(step {index}); the DUT must bar it"
                )
            action = _auto_action(sim, actor)
        elif decision.kind == "auto":
            action = _auto_action(sim, actor)
        else:
            action = _find_action(sim, actor, decision)
        _ = sim.apply(action)
        state.steps_consumed += 1
        state.applied_log.append(_action_row(action, actor))
        if state.steps_consumed > max_applies:
            raise TraceRunnerError("runaway script loop")


def _action_row(action: CanonicalAction, actor: int) -> dict[str, Any]:
    return {
        "kind": str(action.kind),
        "actor": actor,
        "tile": None if action.tile is None else int(action.tile),
        "called_tile": None if action.called_tile is None else int(action.called_tile),
        "consumed": [int(t) for t in action.consumed_tiles],
        "source_seat": None if action.source_seat is None else int(action.source_seat),
    }


# ---------------------------------------------------------------------------
# Expectations (manifest-derived; never computed from DUT output)
# ---------------------------------------------------------------------------


def _fallback_policy(sim: RiichiEnvExactSimulator, rng_seed: int = 0):
    """Deterministic end-of-game filler: prefer pass, else tsumogiri/discard
    the drawn tile, else the first legal action. Keeps scripted cases short:
    the script covers only the decision window under test."""
    import random as _random

    rng = _random.Random(rng_seed)
    while not sim._terminal:
        actor = sim._expected_actor_or_none()
        if actor is None:
            raise TraceRunnerError("simulation stalled during fallback policy")
        actions = sim.legal_actions(Seat(actor))
        passes = [a for a in actions if a.kind == "pass"]
        if len(passes) != 0:
            choice = passes[0]
        else:
            assert sim._env is not None
            drawn = sim._env.drawn_tile if sim._mode == "draw" else None
            tsumogiri = (
                [
                    a
                    for a in actions
                    if a.kind == "tsumogiri"
                    and a.tile is not None
                    and int(a.tile) == drawn
                ]
                if drawn is not None
                else []
            )
            discards = [a for a in actions if a.kind in ("discard", "tsumogiri")]
            if len(tsumogiri) != 0:
                choice = tsumogiri[0]
            elif len(discards) != 0:
                choice = discards[0]
            else:  # pragma: no cover - terminal-only games never reach this
                choice = rng.choice(list(actions))
        _ = sim.apply(choice)


# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExpectationStep:
    """One expected-trace element evaluated against the executed run."""

    label: str
    predicate: Callable[[RiichiEnvExactSimulator], str | None]


def expect_event_kinds(kinds: tuple[str, ...]) -> ExpectationStep:
    """The emitted stream contains at least these kinds, in this order."""

    def check(sim: RiichiEnvExactSimulator) -> str | None:
        events = sim._events
        cursor = 0
        for envelope in events:
            if cursor < len(kinds) and envelope.kind == kinds[cursor]:
                cursor += 1
        if cursor != len(kinds):
            return f"expected ordered kinds {list(kinds)}; stream shows {[e.kind for e in events]}"
        return None

    return ExpectationStep(label="ordered event kinds", predicate=check)


def expect_predicate(
    label: str, check: Callable[[RiichiEnvExactSimulator], str | None]
) -> ExpectationStep:
    return ExpectationStep(label=label, predicate=check)


def expect_terminal_scores(expected_scores: tuple[int, int, int, int]) -> ExpectationStep:
    """Final canonical scores equal manifest-derived settlement arithmetic."""

    def check(sim: RiichiEnvExactSimulator) -> str | None:
        outcome = sim._raw_outcome
        if outcome is None:
            return "no RawOutcome at terminal"
        if outcome.final_scores != expected_scores:
            return (
                f"final scores {outcome.final_scores} != expected {expected_scores} "
                "(derived from rules-manifest settlement rules)"
            )
        expected_ranks = resolve_final_ranks(expected_scores)
        if outcome.ranks != expected_ranks:
            return f"ranks {outcome.ranks} != resolved {expected_ranks}"
        if outcome.point_deltas != tuple(outcome.final_scores[i] - 25000 for i in range(4)):
            return f"point_deltas {outcome.point_deltas} != final - starting"
        return None

    return ExpectationStep(label="terminal outcome", predicate=check)


@dataclass(slots=True)
class TraceExpectation:
    """Frozen expectation bundle metadata of one case."""

    case_id: str
    title: str
    rule_fields: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(slots=True)
class CaseResult:
    """Execution result of one corpus case."""

    case_id: str
    title: str
    status: str  # supported | mismatch | blocked
    rule_fields: tuple[str, ...]
    evidence: tuple[str, ...]
    error_detail: str | None = None
    counterexample_path: str | None = None
    schedule_digest: str | None = None


class ReferenceTraceRunner:
    """Runs corpus cases and persists the FIRST counterexample per case."""

    def __init__(
        self, *, manifest: RulesManifest, artifact_root_path: Path | None = None
    ) -> None:
        self._manifest: RulesManifest = manifest
        self._artifact_root = Path(
            artifact_root_path if artifact_root_path is not None else artifact_root()
        )

    # -- public API -------------------------------------------------------

    def run_case(
        self,
        case_id: str,
        title: str,
        rule_fields: tuple[str, ...],
        evidence: tuple[str, ...],
        wall_tiles: tuple[int, ...],
        script: tuple[ScriptedDecision, ...],
        expectations: list[ExpectationStep],
        finish_to_terminal: bool = True,
        seat_permutation: tuple[Seat, ...] = (Seat(0), Seat(1), Seat(2), Seat(3)),
    ) -> CaseResult:
        schedule = wall_schedule_for(case_id, wall_tiles)
        sim = RiichiEnvExactSimulator()
        # D-WP04A-FIX2: the driver assumes a reset game (haipai dealt from the
        # case wall); skipping reset leaves the engine stalled at step 0.
        sim.reset(
            rules=self._manifest,
            wall=schedule,
            seat_permutation=seat_permutation,
        )
        state = _DriverState()
        construction_error: str | None = None
        try:
            try:
                _drive(sim, script, state)
            except TraceRunnerError as exc:
                # D-WP04A-FIX3: short scripts are the documented contract
                # ("the script covers only the decision window under test").
                # A pure exhaustion error (script consumed while the game
                # continues) falls back to the deterministic filler when the
                # case opted into terminal driving; any other driver failure,
                # or exhaustion with the flag off, stays a construction bug.
                exhausted = "script exhausted" in str(exc)
                if not finish_to_terminal or not exhausted:
                    raise
                while not sim._terminal:
                    fallback_actor = sim._expected_actor_or_none()
                    if fallback_actor is None:
                        raise TraceRunnerError(
                            "simulation stalled during fallback without a pending decision"
                        ) from exc
                    _ = sim.apply(_auto_action(sim, fallback_actor))
            if finish_to_terminal and not sim._terminal:
                _fallback_policy(sim, rng_seed=sum(t for t in wall_tiles) % (2**32))
        except TraceRunnerError as exc:
            construction_error = str(exc)

        failures: list[str] = [*state.violations]
        if construction_error is None:
            for step in expectations:
                problem = step.predicate(sim)
                if problem is not None:
                    failures.append(f"[{step.label}] {problem}")
        else:
            failures.append(f"[script-construction] {construction_error}")

        status = "supported" if len(failures) == 0 else "mismatch"
        detail = "; ".join(failures) if len(failures) != 0 else None
        counterexample_path: str | None = None
        if len(failures) != 0:
            try:
                counterexample_path = self._persist_counterexample(
                    case_id=case_id,
                    title=title,
                    rule_fields=rule_fields,
                    evidence=evidence,
                    failures=failures,
                    schedule=schedule,
                    sim=sim,
                    state=state,
                )
            except OSError as exc:
                detail = f"{detail}; counterexample persistence failed: {exc}"
        return CaseResult(
            case_id=case_id,
            title=title,
            status=status,
            rule_fields=rule_fields,
            evidence=evidence,
            error_detail=detail,
            counterexample_path=counterexample_path,
            schedule_digest=str(schedule.digest),
        )

    # -- internals --------------------------------------------------------

    def _rules_manifest_sha256(self) -> str:
        path = repo_root() / "configs" / "rules" / "tenhou_4p_hanchan_v1.json"
        from hydra2.artifacts.digest import sha256_file

        return sha256_file(path)

    def _persist_counterexample(
        self,
        *,
        case_id: str,
        title: str,
        rule_fields: tuple[str, ...],
        evidence: tuple[str, ...],
        failures: list[str],
        schedule: WallSchedule,
        sim: RiichiEnvExactSimulator,
        state: _DriverState,
    ) -> str:
        document: dict[str, Any] = {
            "artifact_type": "hydra2.wp04a_counterexample",
            "schema_version": "1.0.0",
            "case_id": case_id,
            "title": title,
            "rule_fields": list(rule_fields),
            "evidence_refs": list(evidence),
            "failures": list(failures),
            "inputs": {
                "rules_id": self._manifest.rules_id,
                "rules_manifest_sha256": self._rules_manifest_sha256(),
                "schedule_id": schedule.schedule_id,
                "schedule_digest": str(schedule.digest),
                "schedule_physical_tiles": [int(t) for t in schedule.physical_tiles],
                "scripted_actions": state.applied_log,
                "steps_executed": state.steps_consumed,
            },
            "outputs": {
                "event_stream": [_envelope_row(e) for e in sim._events],
                "settlements": [
                    {
                        "kind": fact.kind,
                        "from_seat": None if fact.from_seat is None else int(fact.from_seat),
                        "to_seats": [int(s) for s in fact.to_seats],
                        "point_deltas": list(fact.point_deltas),
                        "detail": dict(fact.detail),
                    }
                    for fact in (
                        sim._raw_outcome.settlements if sim._raw_outcome is not None else ()
                    )
                ],
                "final_scores": None
                if sim._raw_outcome is None
                else list(sim._raw_outcome.final_scores),
                "ranks": None
                if sim._raw_outcome is None
                else list(sim._raw_outcome.ranks),
            },
            "hashes": {},
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        body = json.dumps(document, sort_keys=True, indent=1).encode()
        document["hashes"]["body_sha256"] = str(of_bytes(body))
        document["hashes"]["inputs_sha256"] = str(of_canonical(document["inputs"]))
        document["hashes"]["outputs_sha256"] = str(of_canonical(document["outputs"]))
        destination = self._artifact_root / "counterexamples" / "WP-04A" / f"{case_id}.json"
        final_bytes = json.dumps(document, sort_keys=True, indent=1).encode()
        digest = "sha256:" + hashlib.sha256(final_bytes).hexdigest()
        document["hashes"]["document_sha256"] = digest
        final_bytes = json.dumps(document, sort_keys=True, indent=1).encode()
        atomic_replace_bytes(destination, final_bytes)
        return str(destination)


def _envelope_row(envelope: EventEnvelope) -> dict[str, Any]:
    return {
        "sequence": int(envelope.sequence),
        "kind": envelope.kind,
        "actor": None if envelope.actor is None else int(envelope.actor),
        "visibility": envelope.visibility,
        "tile": None if envelope.payload.tile is None else int(envelope.payload.tile),
        "action_id": None
        if envelope.payload.action_id is None
        else int(envelope.payload.action_id),
        "source_seat": None
        if envelope.payload.source_seat is None
        else int(envelope.payload.source_seat),
        "consumed": sorted(int(t) for t in envelope.payload.consumed_tiles),
        "scores": None
        if envelope.payload.scores is None
        else list(envelope.payload.scores),
        "reason": envelope.payload.reason,
        "public_delta": [
            {"path": list(d.path), "operation": d.operation, "value": d.value}
            for d in envelope.public_delta
        ],
    }
