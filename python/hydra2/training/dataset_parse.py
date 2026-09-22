"""Dataset parse: actor-observation JSON documents back into validated records.

Owns the ``actor_observation`` document boundary shared by the encoder and
the parquet store: the ``VisibleMeld`` / ``PublicStateDelta`` /
``EventPayload`` / ``EventEnvelope`` rebuilders, the
:class:`ActorObservation` revalidation constructor, the validating row
parse, and the live-or-parse resolver that prefers an already-validated
in-memory observation (replay-capture stash or replay-handoff attachment)
when its digest matches. Every path returns an identical validated object
or raises :class:`ContractError` — unparseable rows never fall back to a
hash stand-in.
"""

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.contracts.event_envelope import EventEnvelope, EventPayload, PublicStateDelta
from hydra2.contracts.observation_actor import (
    ActorObservation,
)
from hydra2.contracts.observation_types import (
    VisibleMeld,
)
from hydra2.data.parquet import verify_no_privileged_leakage

__all__ = [
    "_actor_observation_from_json_dict",
    "_delta_from_json",
    "_envelope_from_json",
    "_parse_actor_observation",
    "_payload_from_json",
    "_require_actor_parquet_dir",
    "_resolve_live_or_parse",
    "_verify_shards",
    "_visible_meld_from_json",
]


def _parse_bridge() -> Any:
    """Resolve the columnar dataset-parse bridge, fail closed when not built.

    Hard-dependency rule (mirrors ``data.rows._seal_bridge``): a missing
    extension or stale ``.so`` without the dataset-parse pyfns raises
    ``ImportError`` with a ``build-ext`` hint — NO oracle fallback, never
    silent. Compute rejects from Rust surface as ``ValueError`` and are
    mapped to :class:`ContractError` / :class:`CorruptArtifactError` at the
    call sites below (require-gate rejects map by call site, same type).
    """
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with columnar not importable; "
            "run `pixi run build-ext` to build the bridge before use"
        ) from exc
    sub = getattr(_ext, "columnar", None)
    if sub is None:
        raise ImportError(
            "hydra2._native.columnar submodule missing (stale .so); "
            "run `pixi run build-ext` to rebuild the bridge"
        )
    for fn_name in (
        "dataset_parse_check_actor_columns",
        "dataset_parse_require_actor_columns",
        "dataset_parse_check_actor_dir_names",
    ):
        if getattr(sub, fn_name, None) is None:
            raise ImportError(
                f"hydra2._native.columnar.{fn_name} missing (stale .so); "
                "run `pixi run build-ext` to rebuild the bridge"
            )
    return sub


def _visible_meld_from_json(raw: Any, *, where: str) -> VisibleMeld:
    """Rebuild one :class:`VisibleMeld` from its ``to_json`` document."""
    if not isinstance(raw, dict):
        raise ContractError(f"unparseable visible_meld for {where!r}: not a mapping")
    try:
        tiles_raw: Any = raw.get("tiles", ())
        tiles: tuple[int, ...] = tuple(int(t) for t in tiles_raw)  # type: ignore[union-attr]
        return VisibleMeld(
            meld_id=raw.get("meld_id"),  # type: ignore[arg-type]
            kind=raw.get("kind"),  # type: ignore[arg-type]
            owner=raw.get("owner"),  # type: ignore[arg-type]
            source_seat=raw.get("source_seat"),  # type: ignore[arg-type]
            called_tile=raw.get("called_tile"),  # type: ignore[arg-type]
            tiles=tiles,  # type: ignore[arg-type]
        )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"unparseable visible_meld for {where!r}: {exc}") from exc


def _delta_from_json(raw: Any, *, where: str) -> PublicStateDelta:
    if not isinstance(raw, dict):
        raise ContractError(f"unparseable public_delta for {where!r}: not a mapping")
    try:
        path_raw: Any = raw.get("path", [])
        path: tuple[str | int, ...] = tuple(path_raw)  # type: ignore[arg-type]
        return PublicStateDelta(
            path=path,  # type: ignore[arg-type]
            operation=raw.get("operation"),  # type: ignore[arg-type]
            value=raw.get("value"),
        )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"unparseable public_delta for {where!r}: {exc}") from exc


def _payload_from_json(raw: Any, *, where: str) -> EventPayload:
    if not isinstance(raw, dict):
        raise ContractError(f"unparseable event payload for {where!r}: not a mapping")
    try:
        scores_raw: Any = raw.get("scores")
        scores: tuple[int, int, int, int] | None = (
            None if scores_raw is None else tuple(int(s) for s in scores_raw)  # type: ignore[union-attr]
        )
        return EventPayload(
            kind=raw.get("kind"),  # type: ignore[arg-type]
            actor=raw.get("actor"),  # type: ignore[arg-type]
            tile=raw.get("tile"),  # type: ignore[arg-type]
            action_id=raw.get("action_id"),  # type: ignore[arg-type]
            source_seat=raw.get("source_seat"),  # type: ignore[arg-type]
            consumed_tiles=tuple(raw.get("consumed_tiles", ())),  # type: ignore[arg-type]
            offered_action_ids=tuple(raw.get("offered_action_ids", ())),  # type: ignore[arg-type]
            accepted_action_ids=tuple(raw.get("accepted_action_ids", ())),  # type: ignore[arg-type]
            round_index=raw.get("round_index"),  # type: ignore[arg-type]
            scores=scores,  # type: ignore[arg-type]
            reason=raw.get("reason"),  # type: ignore[arg-type]
        )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"unparseable event payload for {where!r}: {exc}") from exc


def _envelope_from_json(raw: Any, *, where: str) -> EventEnvelope:
    if not isinstance(raw, dict):
        raise ContractError(f"unparseable history event for {where!r}: not a mapping")
    try:
        payload = _payload_from_json(raw.get("payload"), where=where)
        deltas_raw: Any = raw.get("public_delta", ())
        deltas: tuple[PublicStateDelta, ...] = tuple(
            _delta_from_json(d, where=where)
            for d in deltas_raw  # type: ignore[union-attr]
        )
        return EventEnvelope(
            game_id=str(raw.get("game_id")),
            sequence=int(raw.get("sequence")),  # type: ignore[arg-type]
            kind=raw.get("kind"),  # type: ignore[arg-type]
            actor=raw.get("actor"),  # type: ignore[arg-type]
            visibility=raw.get("visibility"),  # type: ignore[arg-type]
            visible_to=tuple(raw.get("visible_to", ())),  # type: ignore[arg-type]
            payload=payload,
            public_delta=deltas,
            rules_hash=_bridge_contracts.make_digest_text(str(raw.get("rules_hash"))),  # pyrefly: ignore[unknown-argument-type] # JSON mapping value dynamic; bridge validates shape
            schema_hash=_bridge_contracts.make_digest_text(str(raw.get("schema_hash"))),  # pyrefly: ignore[unknown-argument-type] # JSON mapping value dynamic; bridge validates shape
        )
    except Exception as exc:
        raise ContractError(f"unparseable history event for {where!r}: {exc}") from exc


def _actor_observation_from_json_dict(doc: Any, *, decision_id: str) -> ActorObservation:
    """Rebuild one :class:`ActorObservation` from its ``to_json`` document.

    Raises :class:`ContractError` on any unparseable content — never falls
    back to hashing.  The stored ``observation_hash`` (when present) is
    revalidated by :class:`ActorObservation` itself.
    """
    if not isinstance(doc, dict):
        raise ContractError(f"unparseable actor_observation for {decision_id!r}: not a mapping")
    # Bridge ingress (W3-A): the Rust JSON handoff emits a 13-key string
    # projection (marked by its ``projection`` tag) as bridge INPUT, never as
    # encoder input. The plane feed carries no projections; any ``projection``
    # tag reaching the encoder path fails closed here (named reason, never
    # synthesized into a full doc).
    if "projection" in doc:
        raise ContractError(
            f"unexpanded-projection-row for {decision_id!r}: "
            f"projection tag {doc.get('projection')!r} has no plane-feed expansion"
        )
    where = decision_id
    try:
        meld_rows_raw: Any = doc.get("visible_melds", ())
        meld_rows: tuple[tuple[VisibleMeld, ...], ...] = tuple(
            tuple(_visible_meld_from_json(m, where=where) for m in row)  # type: ignore[union-attr]
            for row in meld_rows_raw  # type: ignore[union-attr]
        )
        history_raw: Any = doc.get("visible_history", ())
        history: tuple[EventEnvelope, ...] = tuple(
            _envelope_from_json(e, where=where)
            for e in history_raw  # type: ignore[union-attr]
        )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"unparseable actor_observation for {decision_id!r}: {exc}") from exc
    kwargs: dict[str, Any] = dict(doc)
    kwargs["visible_melds"] = meld_rows
    kwargs["visible_history"] = history
    try:
        return ActorObservation(**kwargs)  # type: ignore[arg-type]
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"unparseable actor_observation for {decision_id!r}: {exc}") from exc


def _parse_actor_observation(row: dict[str, Any]) -> ActorObservation:
    decision_id = str(row.get("decision_id", ""))
    raw: Any = row.get("actor_observation")
    if isinstance(raw, dict):
        doc: Any = raw
    elif isinstance(raw, str):
        try:
            doc = json.loads(raw)
        except Exception as exc:
            raise ContractError(
                f"unparseable actor_observation for {decision_id!r}: not JSON"
            ) from exc
    else:
        raise ContractError(
            f"unparseable actor_observation for {decision_id!r}: "
            f"expected str/dict, got {type(raw).__name__}"
        )
    return _actor_observation_from_json_dict(doc, decision_id=decision_id)


def _resolve_live_or_parse(row: dict[str, Any]) -> ActorObservation:
    """Return the row's live observation when stashed, else parse+validate.

    Perf-C P1a: replay capture stashes the already-validated
    :class:`ActorObservation` out of band keyed by ``decision_id`` (see
    :mod:`hydra2.data.replay_expand`). A cache hit whose digest matches the
    row's recorded ``observation_hash`` skips the serialize/re-parse/
    re-validate round trip entirely; anything else (parquet rows, cache
    misses, digest mismatch) takes the unchanged validating parse path, so
    the returned object is always identical either way.

    Phase 2A/B.1: rows already carrying the live :class:`ActorObservation`
    (replay handoff without a JSON boundary) are consumed directly — no
    stash lookup, no parse, no revalidation, no re-hash. As with the JSON
    path (which builds the object from the document, ignoring the row's
    own ``decision_id``), content wins: the sim-replay path stamps a
    stream-local ``decision_id`` inside the observation that differs from
    the row's canonical id by construction. When the row carries a
    top-level ``observation_hash`` it must match the object (same rule as
    the stash handoff — a divergent attachment never silently wins);
    anything else falls through to the validating parse path, which owns
    the error.
    """
    from hydra2.data.replay_expand import pop_live_observation

    raw: Any = row.get("actor_observation")
    if isinstance(raw, ActorObservation):
        recorded = row.get("observation_hash")
        if raw.observation_hash is not None and (
            not isinstance(recorded, str) or str(raw.observation_hash) == recorded
        ):
            return raw
    if isinstance(raw, dict):
        live = pop_live_observation(str(row.get("decision_id", "")))
        if (
            isinstance(live, ActorObservation)
            and live.observation_hash is not None
            and str(live.observation_hash) == str(raw.get("observation_hash"))
        ):
            return live
        if live is not None:
            # Key collision with divergent content must never silently win:
            # fall through to the validating parse (which owns the error).
            pass
    return _parse_actor_observation(row)


def _require_actor_parquet_dir(path: Path) -> list[Path]:
    path = Path(path)
    if not path.is_dir():
        raise ContractError(f"actor parquet path is not a directory: {path}")
    shards = sorted(path.glob("actor-*.parquet"))
    if len(shards) == 0:
        raise ContractError(f"no actor shards found in {path} (expected actor-*.parquet)")
    # Reject privileged shards masquerading as actor: filename gates are pure
    # string checks owned by the bridge (fail-closed ContractError text);
    # globbing (IO) stays here, order-preserving so the named offender matches.
    try:
        _parse_bridge().dataset_parse_check_actor_dir_names(
            str(path),
            [p.name for p in path.glob("privileged*")],
            [p.name for p in path.glob("*.parquet")],
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    return shards


def _verify_shards(shards: list[Path]) -> None:
    for shard in shards:
        # Hard failure: privileged leakage in actor shard
        verify_no_privileged_leakage(shard)
        # Perf-A HIGH: memory_map read + schema-only verification avoids
        # full materialization when possible.
        # Evidence:
        #  https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html  # noqa: E501 -- URL cannot be wrapped without breaking link; alternative (shortened URL) loses precision
        #  + https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html  # noqa: E501 -- URL cannot be wrapped without breaking link; alternative loses precision
        # Use ParquetFile to inspect schema without loading all batches if
        # available; fallback to read_table.
        try:
            pf = pq.ParquetFile(shard, memory_map=True)
            column_names = pf.schema.names
            num_rows = pf.metadata.num_rows
        except Exception:
            table = pq.read_table(shard, memory_map=True, pre_buffer=True, use_threads=True)
            column_names = table.column_names
            num_rows = table.num_rows
        bridge = _parse_bridge()
        try:
            bridge.dataset_parse_check_actor_columns(shard.name, list(column_names))
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        try:
            bridge.dataset_parse_require_actor_columns(shard.name, list(column_names))
        except ValueError as exc:
            raise CorruptArtifactError(str(exc)) from exc
        if num_rows == 0:
            raise ContractError(f"actor shard {shard.name} contains zero rows")
