# ruff: noqa: N814  # reason: _UV/_aid aliases match repo bridge-translator casing; renaming churns KAT-pinned imports.
"""Trigger-only helpers for search driving.

Owns exactly three helpers with no logic beyond packing: the WorldDoc
extraction (verbatim from the retired per-sim walk), the CTR seed/cursor
freeze (fail closed, never a draw), and the DTO wrap (codec lookup,
finite-gated vectors, telemetry from Rust counters plus one elapsed
measure for the report only).

Rust owns all loops, clocks, cursors, sums, and forest mutation; this
module never loops over sims, never advances a stream, never sums.
"""

from __future__ import annotations

import time
from typing import Any

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError, DigestText

__all__ = [
    "pack_rng",
    "pack_worlds",
    "wrap_result",
]

MAX_SAFE_INTEGER = 2**53 - 1


def pack_worlds(worlds: list[Any]) -> list[dict[str, Any]]:
    """Freeze sampled worlds into WorldDoc dicts (no logic, only reads).

    Keys verbatim from the retired ISMCTS envelope: world_id, hands,
    live, dead, step, turn, corpus_idx, snapshot. Hands as list[list[int]],
    live/dead as list[int], step/turn/corpus_idx as int | None.
    """
    docs: list[dict[str, Any]] = []
    for cur_world in worlds:
        try:
            hands = tuple(
                tuple(int(t) for t in h)  # pyrefly: ignore[unknown-argument-type] # Any world tile
                for h in cur_world.concealed_hands
            )
            live_start = tuple(
                int(t)  # pyrefly: ignore[unknown-argument-type] # Any world tile
                for t in cur_world.live_wall
            )
            dead = tuple(
                int(t)  # pyrefly: ignore[unknown-argument-type] # Any world tile
                for t in cur_world.dead_wall
            )
            lat_raw: Any = getattr(cur_world, "latent_state", {})
            lat: Any = lat_raw if isinstance(lat_raw, dict) else {}
            step0: Any = lat.get("step", None) if isinstance(lat, dict) else None
            turn0: Any = lat.get("turn", None) if isinstance(lat, dict) else None
            corp0: Any = lat.get("corpus_idx", None) if isinstance(lat, dict) else None
        except Exception as exc:
            raise ContractError(f"drive: sampled world malformed: {exc}") from exc
        # Closed-domain staging: bool-before-int, safe-int gate, no tuples out.
        for row in hands:
            for tile in row:
                if isinstance(tile, bool) or not isinstance(tile, int):
                    raise ContractError("drive: hand tile must be int")
                if abs(tile) > MAX_SAFE_INTEGER:
                    raise ContractError("drive: hand tile outside safe integer")
        for tile in (*live_start, *dead):
            if isinstance(tile, bool) or not isinstance(tile, int):
                raise ContractError("drive: wall tile must be int")
            if abs(tile) > MAX_SAFE_INTEGER:
                raise ContractError("drive: wall tile outside safe integer")
        docs.append(
            {
                "world_id": str(cur_world.world_id),  # pyrefly: ignore[unknown-argument-type] # Any world id
                "hands": [list(h) for h in hands],
                "live": list(live_start),
                "dead": list(dead),
                "step": (None if step0 is None else int(step0)),
                "turn": (None if turn0 is None else int(turn0)),
                "corpus_idx": (None if corp0 is None else int(corp0)),
                "snapshot": str(cur_world.simulator_snapshot),  # pyrefly: ignore[unknown-argument-type] # Any world snapshot
            }
        )
    return docs


def pack_rng(rng: Any) -> tuple[bytes, int]:
    """Freeze the CTR stream to (seed_bytes, cursor) without drawing."""
    try:
        from hydra2.belief.natural import _ctr_seed_cursor as _freeze
    except ImportError as exc:
        raise ImportError(
            f"hydra2.belief.natural not importable ({exc}); "
            "build the bridge with `pixi run build-ext` before drive trigger"
        ) from exc
    try:
        frozen: tuple[bytes, int] = _freeze(rng)
        seed, cursor = frozen
    except (ContractError, ValueError, TypeError, AttributeError) as exc:
        raise ContractError(f"drive: rng freeze failed: {exc}") from exc
    if not isinstance(seed, (bytes, bytearray)) or len(seed) == 0:
        raise ContractError("drive: rng seed must be non-empty bytes")
    if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
        raise ContractError("drive: rng cursor must be nonnegative int")
    return seed, cursor


def _require_finite_row(row: Any, *, aid: Any) -> tuple[float, float, float, float]:
    """Finite-gate one 4-vector row (no sum, no mean, only the gate)."""
    try:
        quad = tuple(float(v) for v in row)
    except Exception as exc:
        raise ContractError(f"drive: value vector for {aid!r} malformed: {exc}") from exc
    if len(quad) != 4:
        raise ContractError(f"drive: value vector for {aid!r} must hold 4 entries")
    for v in quad:
        if not isinstance(v, float) or v != v or v in (float("inf"), float("-inf")):
            raise ContractError(f"drive: value vector for {aid!r} must be finite")
    return (quad[0], quad[1], quad[2], quad[3])


def _aid_for_trigger(item: Any) -> int:
    """Int id for trigger mapping: action_id else hash fallback (never empty)."""
    raw: Any = getattr(item, "action_id", None)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if isinstance(item, int) and not isinstance(item, bool):
        return item
    import hashlib as _hl2

    return int(_hl2.sha256(str(item).encode()).hexdigest()[:8], 16) & 0xFFFF


def wrap_result(
    out: Any,
    *,
    legal: tuple[Any, ...],
    candidate_spec: Any,
    start_ns: int,
    case_id: str | None = None,
    energy_joules: float | None = None,
    particles: int | None = None,
) -> Any:
    """Wrap one detached Rust OUT into a SearchResult (no recompute).

    Maps selected_id via action_id equality (candidate0 decode path shape),
    builds UtilityVector per candidate from OUT rows, telemetry from OUT
    counters plus one elapsed measure (report only, never pinned in KAT).
    """
    from hydra2.eval.telemetry import make_resource_telemetry as _mrt
    from hydra2.search.common import SearchResult, candidate_spec_hash

    if not isinstance(legal, tuple) or len(legal) == 0:
        raise ContractError("legal must be non-empty tuple")
    try:
        selected_id: int = out.selected_id
        candidate_ids: list[int] = out.candidate_ids
        raw_vecs: list[list[float]] = out.value_vectors
        _sims_run: int = out.sims_run
        transitions: int = out.transitions
        model_calls: int = out.model_calls
        digest: str = out.decision_digest
        completed: bool = out.completed
    except Exception as exc:
        raise ContractError(f"drive: OUT malformed: {exc}") from exc
    _ = _sims_run
    if len(candidate_ids) != len(raw_vecs):
        raise ContractError("drive: OUT vectors length mismatch")
    # Map selected id to the CanonicalAction object (action_id equality else hash fallback).
    selected_action: Any | None = None
    for item in legal:
        try:
            if _aid_for_trigger(item) == selected_id:  # pyrefly: ignore[unknown-argument-type] # Any legal action
                selected_action = item
                break
        except Exception:
            continue
    if selected_action is None:
        raise ContractError("drive: OUT selection not in legal set")
    # Map candidate ids to legal order (OUT rides sorted; legal may be sorted already).
    id_to_action: dict[int, Any] = {}
    for item in legal:
        try:
            id_to_action[_aid_for_trigger(item)] = item  # pyrefly: ignore[unknown-argument-type] # Any legal action
        except Exception as exc:
            raise ContractError(f"drive: legal action_id unreadable: {exc}") from exc
    for aid in candidate_ids:
        if aid not in id_to_action:
            raise ContractError("drive: OUT candidate not in legal set")
    vec_by_aid: dict[int, tuple[float, float, float, float]] = {}
    for aid, row in zip(candidate_ids, raw_vecs, strict=True):
        vec_by_aid[aid] = _require_finite_row(row, aid=aid)
    # Order vectors by legal order for the SearchResult.
    ordered_ids: list[int] = []
    for item in legal:
        ordered_ids.append(_aid_for_trigger(item))  # pyrefly: ignore[unknown-argument-type] # Any legal action
    try:
        from hydra2.contracts.utility import UtilityVector as _UV
    except ImportError as exc:
        raise ImportError(
            f"hydra2.contracts.utility not importable ({exc}); "
            "build the bridge with `pixi run build-ext` before drive trigger"
        ) from exc
    value_vectors: list[Any] = []
    for aid in ordered_ids:
        quad = vec_by_aid.get(aid, (0.0, 0.0, 0.0, 0.0))
        try:
            manifest_digest: DigestText = _bridge_contracts.make_digest_text(
                str(getattr(candidate_spec, "utility_manifest_hash", "sha256:" + "b" * 64))
            )
            rules_digest: DigestText = _bridge_contracts.make_digest_text(
                str(getattr(candidate_spec, "rules_hash", "sha256:" + "a" * 64))
            )
            value_vectors.append(
                _UV(
                    values=quad,
                    utility_id=str(
                        getattr(candidate_spec, "utility_id", "expected_final_placement")
                    ),
                    utility_manifest_hash=manifest_digest,
                    rules_hash=rules_digest,
                )
            )
        except (AttributeError, ValueError, TypeError, OSError) as exc:
            raise ContractError(f"drive: UtilityVector build failed: {exc}") from exc
    try:
        spec_hash = str(candidate_spec_hash(candidate_spec))
    except Exception as exc:
        raise ContractError(f"drive: candidate_spec_hash failed: {exc}") from exc
    elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
    if elapsed_ms < 0:
        elapsed_ms = 0.0
    joules = (
        energy_joules
        if energy_joules is not None
        else float(model_calls) * 0.5 + float(transitions) * 0.2
    )
    if joules != joules or joules in (float("inf"), float("-inf")):
        raise ContractError("drive: energy_joules must be finite")
    try:
        telemetry = _mrt(
            mode=str(getattr(candidate_spec.resource_budget, "mode", "gameplay_5s")),  # pyrefly: ignore[unknown-argument-type] # Any candidate spec
            wall_id=None,
            case_id=case_id,
            candidate_spec_hash=spec_hash,
            hardware_hash="sha256:" + "8" * 64,
            environment_hash="sha256:" + "7" * 64,
            cold_start=False,
            synchronized_elapsed_ms=elapsed_ms,
            model_calls=model_calls,
            exact_transitions=transitions,
            particles=particles if particles is not None else len(legal),
            fallback_used=not completed,
            timeout=not completed,
            illegal_action=False,
            cuda_peak_allocated_bytes=None,
            cuda_peak_reserved_bytes=None,
            host_peak_bytes=None,
            energy_joules=joules,
            graph_breaks=None,
            recompiles=None,
            invalid_reason=None,
        )
    except (AttributeError, ValueError, TypeError, OSError) as exc:
        raise ContractError(f"drive: telemetry build failed: {exc}") from exc
    try:
        evidence = (_bridge_contracts.make_digest_text(digest),)
        spec_digest: DigestText = _bridge_contracts.make_digest_text(spec_hash)
    except (ValueError, TypeError, OSError) as exc:
        raise ContractError(f"drive: digest shape failed: {exc}") from exc
    return SearchResult(
        selected_action=selected_action,  # pyrefly: ignore[unknown-argument-type] # Any legal action
        candidate_actions=tuple(id_to_action[aid] for aid in ordered_ids),
        value_vectors=tuple(value_vectors),
        candidate_spec_hash=spec_digest,
        telemetry=telemetry,
        evidence_refs=evidence,
        completed=completed,
    )
