"""WP-05B authoritative data integration: synthetic parquet dataset.

Wraps the WP-04B actor parquet shards with deterministic ordering,
privileged-field rejection, legal-mask verification, and sampler-cursor
tracking for resume.

Authoritative checks enforced on construction and on every batch yielded:
 - actor parquet column names never contain privileged keys (FORBIDDEN_IN_ACTOR)
 - verify_no_privileged_leakage called per shard
 - dora shape (5,) is enforced (no (4,) shim)
 - no privileged parquet path may be supplied; any such leakage raises
   ContractError before training sees data.
 - legal_mask rows must have at least one legal action and chosen action
   must be legal.

The dataset is synthetic-qualified: it reads the tiny shards written by
``write_actor_shards`` in tests.  The same code path would read the real
corpus after D-017 attestation; the synthetic qualifier is the data, not
the loader path.

Determinism: ordering is canonical (sorted by decision_id) then optionally
permuted by a seeded generator.  Sampler cursor is a plain ``{"offset": int,
"seed": int, "total": int}`` JSON value stored in TrainingState and the
checkpoint ``sampler_state`` section, enabling bitwise resume.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
import torch

from hydra2.contracts.common import ContractError, CorruptArtifactError, make_digest_text
from hydra2.contracts.event import EventEnvelope, EventPayload, PublicStateDelta
from hydra2.contracts.observation import ActorObservation, VisibleMeld
from hydra2.data.parquet import (
    ACTOR_FIELDS,
    FORBIDDEN_IN_ACTOR,
    verify_no_privileged_leakage,
)
from hydra2.models.encoder import encode_observations
from hydra2.models.schema import BASELINE_ACTION_COUNT

__all__ = [
    "DEFAULT_STRATIFIED_RATIOS",
    "RARE_ACTION_KINDS",
    "AuthoritativeParquetDataset",
    "SamplerState",
    "build_stratified_order",
    "encode_observation_rows",
    "tensorize_actor_row",
]
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SamplerState:
    offset: int
    seed: int
    total: int
    epoch: int = 0


def _require_actor_parquet_dir(path: Path) -> list[Path]:
    path = Path(path)
    if not path.is_dir():
        raise ContractError(f"actor parquet path is not a directory: {path}")
    shards = sorted(path.glob("actor-*.parquet"))
    if len(shards) == 0:
        raise ContractError(f"no actor shards found in {path} (expected actor-*.parquet)")
    # Reject privileged shards masquerading as actor: any file with privileged name
    for p in path.glob("privileged*"):
        raise ContractError(
            f"privileged shard present in actor dataset directory {path}: {p.name} — "
            "actor loader must never touch privileged parquet (WP-05B no privileged fields)"
        )
    for p in path.glob("*.parquet"):
        if "privileged" in p.name.lower():
            raise ContractError(f"privileged parquet detected in actor dir: {p.name}")
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
        for col in column_names:
            if col in FORBIDDEN_IN_ACTOR:
                raise ContractError(f"actor shard {shard.name} contains privileged column {col!r}")
            if col not in ACTOR_FIELDS:
                raise ContractError(f"actor shard {shard.name} has unexpected column {col!r}")
        for required in ("decision_id", "chosen_action_id", "actor_observation"):
            if required not in column_names:
                raise CorruptArtifactError(
                    f"actor shard {shard.name} missing required column {required!r}"
                )
        if num_rows == 0:
            raise ContractError(f"actor shard {shard.name} contains zero rows")


def _lexicographic_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def _require_action_width(num_actions: int, *, allow_narrow: bool, where: str) -> None:
    """Fail closed unless the vocab width is the frozen baseline or narrow is explicit.

    ``num_actions != BASELINE_ACTION_COUNT`` is a test-only configuration and
    requires ``allow_narrow=True``; production always trains the full 6792
    table so aliased (modulo-remapped) labels can never flow silently.
    """
    if num_actions != BASELINE_ACTION_COUNT and not allow_narrow:
        raise ContractError(
            f"{where}: num_actions {num_actions} != baseline {BASELINE_ACTION_COUNT} "
            "requires allow_narrow=True (test-only narrow vocab)"
        )


def tensorize_actor_row(
    row: dict[str, Any],
    *,
    num_actions: int,
    feature_dim: int = 16,
    seed: int = 0,
    allow_narrow: bool = False,
) -> dict[str, Any]:
    """Deterministic tensorization of one actor row for tests/synthetic data.

    The real WP-05A encoder would parse ``actor_observation`` JSON and produce
    per ``model_input_v1`` tensors.  This helper is the WP-05B synthetic
    stand-in that is deterministic, actor-visible only, and never touches
    privileged data.

    Produces:
      features: FloatTensor [feature_dim] hashed from decision_id
      legal_mask: BoolTensor [num_actions]
      chosen_action_id: LongTensor scalar (exact id in production; modulo
        ``num_actions`` only under ``allow_narrow`` for small test vocabs,
        ensured legal)
    """
    _require_action_width(num_actions, allow_narrow=allow_narrow, where="tensorize_actor_row")
    decision_id: str = str(row["decision_id"])
    chosen_raw: int = int(row["chosen_action_id"])
    # Deterministic features from decision_id hash
    h = hashlib.sha256(f"{decision_id}:{seed}".encode()).digest()
    # Expand to feature_dim floats via hash bytes
    vals: list[float] = []
    for i in range(feature_dim):
        # cycle through hash bytes
        b = h[i % len(h)]
        vals.append((b / 255.0) * 2 - 1)  # in [-1,1]
    features = torch.tensor(vals, dtype=torch.float32)
    # Deterministic legal mask: ensure chosen is legal, plus random other legals
    gen = torch.Generator().manual_seed(_lexicographic_hash(decision_id) ^ seed)
    # Randomly decide legal count 1..min(8, num_actions)
    legal_count = int(torch.randint(1, min(8, num_actions) + 1, (1,), generator=gen).item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for synthetic row; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
    legal_mask = torch.zeros(num_actions, dtype=torch.bool)
    # Always include chosen: exact label in production (out-of-range is a
    # hard error, never silently aliased); modulo remap only under the
    # test-only narrow flag.
    if allow_narrow:
        chosen = chosen_raw % num_actions
    else:
        if not 0 <= chosen_raw < num_actions:
            raise ContractError(
                f"chosen_action_id {chosen_raw} out of range for baseline "
                f"vocab {num_actions} (aliased labels are never trained)"
            )
        chosen = chosen_raw
    legal_mask[chosen] = True
    # Fill remaining
    candidates: list[int] = list(range(num_actions))
    candidates.remove(chosen)
    perm_any: Any = torch.randperm(len(candidates), generator=gen).tolist()
    perm: list[int] = [int(x) for x in perm_any]
    for idx in perm[: legal_count - 1]:
        cand: int = candidates[idx]
        legal_mask[cand] = True
    return {
        "features": features,
        "legal_mask": legal_mask,
        "chosen_action_id": torch.tensor(chosen, dtype=torch.long),
        "decision_id": decision_id,
    }


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
            rules_hash=make_digest_text(str(raw.get("rules_hash"))),
            schema_hash=make_digest_text(str(raw.get("schema_hash"))),
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


_REAL_COUNT_KEYS: tuple[str, ...] = (
    "hand_number",
    "honba",
    "riichi_sticks",
    "kan_count",
    "round_index",
)


def _real_features_from_encoder_batch(batch: Any, *, feature_dim: int) -> torch.Tensor:
    """Fold real encoder tensors into a fixed ``[B, feature_dim]`` float matrix.

    Every column derives from actor-visible encoder content (tile counts,
    dora, scores, seats, scalars, history kinds); no decision_id hash enters.
    Folding is a strided sum scaled by the bin fill so any single wide-column
    change moves exactly one output bin.
    """
    feats: dict[str, torch.Tensor] = batch.features
    parts: list[torch.Tensor] = []
    parts.append(feats["concealed_hand_counts"].to(torch.float32) / 4.0)
    parts.append(feats["visible_discards_counts"].to(torch.float32) / 4.0)
    parts.append(feats["dora_indicators"].to(torch.float32) / 34.0)
    parts.append(feats["scores"].to(torch.float32) / 40000.0)
    parts.append(feats["seat_winds"].to(torch.float32) / 3.0)
    parts.append(feats["ippatsu_active"].to(torch.float32))
    parts.append(feats["riichi_states"].to(torch.float32) / 2.0)
    parts.append(((feats["own_drawn_tile"].to(torch.float32) + 1.0) / 136.0).unsqueeze(1))
    for key, scale in (
        ("actor", 3.0),
        ("dealer", 3.0),
        ("turn_actor", 3.0),
        ("phase", 8.0),
        ("actor_furiten", 3.0),
        ("round_wind", 3.0),
    ):
        parts.append((feats[key].to(torch.float32) / scale).unsqueeze(1))
    parts.extend((feats[key].to(torch.float32) / 8.0).unsqueeze(1) for key in _REAL_COUNT_KEYS)
    parts.append((feats["live_wall_tiles_remaining"].to(torch.float32) / 136.0).unsqueeze(1))
    parts.append(feats["actor_can_riichi"].to(torch.float32).unsqueeze(1))
    parts.append(feats["actor_can_tsumo"].to(torch.float32).unsqueeze(1))
    parts.append(feats["history_event_kind"].to(torch.float32) / 16.0)
    parts.append(feats["history_mask"].to(torch.float32))
    wide = torch.cat([p.reshape(p.shape[0], -1) for p in parts], dim=1)
    batch_size = wide.shape[0]
    if feature_dim <= 0:
        raise ContractError(f"feature_dim must be positive, got {feature_dim}")
    out = torch.zeros((batch_size, feature_dim), dtype=torch.float32)
    width = wide.shape[1]
    if width == 0:
        return out
    idx = torch.arange(width) % feature_dim
    # intentionally discarded: in-place accumulation returns self
    expanded = idx.unsqueeze(0).expand(batch_size, width)
    _ = out.scatter_add_(1, expanded, wide)
    denom = (width + feature_dim - 1) // feature_dim
    return out / float(denom)


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


def encode_observation_rows(
    rows: Sequence[dict[str, Any]],
    *,
    num_actions: int,
    feature_dim: int = 16,
    allow_narrow: bool = False,
    pin_memory: bool = True,
) -> dict[str, Any]:
    """Tensorize rows through the real actor-visible encoder.

    Parses each row's ``actor_observation`` JSON into an
    :class:`ActorObservation` and encodes the batch with
    :func:`encode_observations`.  ``features`` folds real encoder content
    (never a decision_id hash), ``legal_mask`` is the observation's own mask
    (sliced to ``num_actions`` only under the test-only ``allow_narrow`` flag
    for a small vocab), and ``chosen_action_id`` is the record's exact choice
    in production (modulo ``num_actions`` only under ``allow_narrow``),
    validated legal.  The encoded :class:`ActorTensorBatch` is also carried
    under ``actor_batch`` for the real-model input bridge (loop routes
    ``model.evaluate(batch['actor_batch'])``); flat keys stay byte-identical
    for compat.  Rows whose validated observation was stashed live at
    capture time skip the re-parse (identical objects either way); rows
    already carrying the live object skip it outright.  Any unparseable row raises
    :class:`ContractError` — never falls back to the synthetic hash
    stand-in.
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ContractError(f"rows must be a sequence of mappings, got {type(rows).__name__}")
    row_list: list[dict[str, Any]] = list(rows)
    if len(row_list) == 0:
        raise ContractError("encode_observation_rows requires at least one row")
    if num_actions <= 0:
        raise ContractError(f"num_actions must be positive, got {num_actions}")
    if num_actions > BASELINE_ACTION_COUNT:
        raise ContractError(
            f"num_actions {num_actions} exceeds baseline {BASELINE_ACTION_COUNT} in real mode"
        )
    _require_action_width(num_actions, allow_narrow=allow_narrow, where="encode_observation_rows")
    if feature_dim <= 0:
        raise ContractError(f"feature_dim must be positive, got {feature_dim}")
    chosen_raws: list[int] = []
    for row in row_list:
        if not isinstance(row, dict):
            raise ContractError(f"row must be a mapping, got {type(row).__name__}")
        try:
            chosen_raw = int(row["chosen_action_id"])
        except (KeyError, TypeError, ValueError) as exc:
            did = row.get("decision_id")
            raise ContractError(f"unparseable chosen_action_id for {did!r}") from exc
        chosen_raws.append(chosen_raw)
    observations: list[ActorObservation] = [_resolve_live_or_parse(row) for row in row_list]
    try:
        encoded = encode_observations(observations, pin_memory=pin_memory)
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"unparseable actor_observation batch: {exc}") from exc
    full_legal = encoded.legal_mask
    if full_legal.dim() != 2 or full_legal.shape[0] != len(row_list):
        raise ContractError(f"encoder legal_mask batch mismatch {tuple(full_legal.shape)}")
    if num_actions == BASELINE_ACTION_COUNT:
        legal_mask = full_legal.to(torch.bool).contiguous()
    else:
        legal_mask = full_legal[:, :num_actions].to(torch.bool).contiguous()
    # Vectorized legality: one any/all reduction (single host sync) instead
    # of a per-row .any().item() sync in the row loop. First-bad index via
    # nonzero keeps the error identical to the row-loop version.
    row_has_legal = legal_mask.any(dim=1)
    if not bool(row_has_legal.all().item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for fail-closed legality guard; branching needs host value. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        bad_row = int(torch.nonzero(~row_has_legal, as_tuple=False)[0, 0].item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for identical error index; alternative loses error parity. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise ContractError(
            f"real legal_mask has no legal action for {observations[bad_row].decision_id!r} "
            f"(sliced to {num_actions})"
        )
    if allow_narrow:
        chosen_ids: list[int] = [raw % num_actions for raw in chosen_raws]
    else:
        for raw in chosen_raws:
            if not 0 <= raw < num_actions:
                raise ContractError(
                    f"chosen_action_id {raw} out of range for baseline vocab "
                    f"{num_actions} (aliased labels are never trained)"
                )
        chosen_ids = list(chosen_raws)
    chosen_t = torch.tensor(chosen_ids, dtype=torch.long)
    legal_chosen = legal_mask[torch.arange(len(row_list)), chosen_t]
    if not bool(legal_chosen.all().item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for fail-closed legality guard; branching needs host value. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        bad_idx = int(torch.nonzero(~legal_chosen, as_tuple=False)[0, 0].item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for identical error index; alternative loses error parity. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise ContractError(
            f"chosen action {chosen_ids[bad_idx]} (raw {chosen_raws[bad_idx]}) illegal "
            f"for {observations[bad_idx].decision_id!r}"
        )
    features = _real_features_from_encoder_batch(encoded, feature_dim=feature_dim)
    chosen_action_id = torch.tensor(chosen_ids, dtype=torch.long)
    if pin_memory and torch.cuda.is_available():
        try:
            features = features.pin_memory()
            legal_mask = legal_mask.pin_memory()
            chosen_action_id = chosen_action_id.pin_memory()
        except Exception as exc:
            logger.warning("dataset pin_memory failed, using pageable fallback: %s", exc)
    result = {
        "features": features,
        "legal_mask": legal_mask,
        "chosen_action_id": chosen_action_id,
        "actor_batch": encoded,
    }

    return result


#: Rare-action families oversampled by the stratified sampler (SOTA R5).
#: winning-call kinds.  Ratios are caller-supplied; the r1 recipe uses 3x.
RARE_ACTION_KINDS: tuple[str, ...] = ("daiminkan", "ankan", "kakan", "ron", "tsumo")

#: Starter oversample ratios (SOTA R5 quotes ``2``-``4``x for rare kan/ron;
#: mechanism default when the caller enables stratification without ratios).
DEFAULT_STRATIFIED_RATIOS: dict[str, float] = {
    "daiminkan": 3.0,
    "ankan": 3.0,
    "kakan": 3.0,
    "ron": 3.0,
}


def _row_action_kind(row: dict[str, Any]) -> str:
    """Best-effort action kind for one raw row (sampler bucketing only).

    Explicit ``action_kind``/``decision_kind`` string fields win (the stream
    layer threads these in a follow-up); otherwise the bucket is
    ``"unknown"``.  Never parses observations and never touches the
    ingress/encode path — bucketing labels only.
    """
    for key in ("action_kind", "decision_kind", "kind"):
        value = row.get(key)
        if isinstance(value, str) and value != "":
            return value
    return "unknown"


def _validate_sampling_ratios(ratios: Mapping[str, Any] | None) -> dict[str, float] | None:
    """Strict ``{kind: positive-finite-mult}`` validation (null stays null)."""
    if ratios is None:
        return None
    if not isinstance(ratios, Mapping):
        raise ContractError(
            f"sampling_ratios must be a mapping or null, got {type(ratios).__name__}"
        )
    out: dict[str, float] = {}
    for kind, mult in ratios.items():
        if not isinstance(kind, str) or kind == "":
            raise ContractError("sampling_ratios keys must be non-empty strings")
        if (
            isinstance(mult, bool)
            or not isinstance(mult, (int, float))
            or not (float(mult) == float(mult))
            or float(mult) in (float("inf"), float("-inf"))
            or float(mult) <= 0.0
        ):
            raise ContractError(f"sampling_ratios[{kind!r}] must be positive and finite")
        out[kind] = float(mult)
    return out


def _validate_kind_by_id(kind_by_id: Mapping[str, str] | None) -> dict[str, str] | None:
    """Strict ``{decision_id: kind}`` validation (null stays null)."""
    if kind_by_id is None:
        return None
    if not isinstance(kind_by_id, Mapping):
        raise ContractError(
            f"kind_by_id must be a mapping or null, got {type(kind_by_id).__name__}"
        )
    out: dict[str, str] = {}
    for did, kind in kind_by_id.items():
        if not isinstance(did, str) or did == "":
            raise ContractError("kind_by_id keys must be non-empty decision ids")
        if not isinstance(kind, str) or kind == "":
            raise ContractError(f"kind_by_id[{did!r}] must be a non-empty kind string")
        out[did] = kind
    return out


def build_stratified_order(
    kinds: Sequence[str],
    ratios: Mapping[str, float] | None,
    *,
    seed: int = 0,
) -> list[int]:
    """Deterministic stratified/oversampled row order over ``range(len(kinds))``.

    Each kind ``k`` with count ``c`` and ratio ``r`` (default ``1.0``)
    contributes ``max(1, round(c * r))`` positions (cycled deterministically
    through its rows when ``r > 1``); kinds interleave round-robin in sorted
    kind order so rare kinds spread evenly through the epoch instead of
    clustering.  ``seed`` is accepted for API symmetry with the cursor
    sampler (the construction is fully determined by ``(kinds, ratios)``;
    the caller's base permutation already carries the seed entropy).
    """
    kind_list = list(kinds)
    for kind in kind_list:
        if not isinstance(kind, str) or kind == "":
            raise ContractError(f"stratified kinds must be non-empty strings, got {kind!r}")
    resolved = _validate_sampling_ratios(ratios if ratios is not None else {})
    by_kind: dict[str, list[int]] = {}
    for pos, kind in enumerate(kind_list):
        by_kind.setdefault(kind, []).append(pos)
    expanded: dict[str, list[int]] = {}
    for kind in sorted(by_kind):
        members = by_kind[kind]
        ratio = (resolved if resolved is not None else {}).get(kind, 1.0)
        target = max(1, round(len(members) * ratio))
        expanded[kind] = [members[i % len(members)] for i in range(target)]
    order: list[int] = []
    pending = True
    cursor = 0
    ordered_kinds = sorted(expanded)
    while pending:
        pending = False
        for kind in ordered_kinds:
            members = expanded[kind]
            if cursor < len(members):
                order.append(members[cursor])
                pending = True
        cursor += 1
    _ = seed  # documented no-op: determinism comes from (kinds, ratios)
    return order


class AuthoritativeParquetDataset:
    """Deterministic authoritative parquet dataset for WP-05B.

    Args:
        parquet_dir: directory containing ``actor-*.parquet`` shards written
            by :func:`hydra2.data.parquet.write_actor_shards`.
        feature_dim: synthetic feature dimensionality (used when observation
            is not pre-tensorized).  In ``'real'`` mode it sets the folded
            real-feature width.
        num_actions: canonical action vocab size.  Defaults to the frozen
            action table size (6792) when ``None`` is passed; any other value
            is test-only and requires ``allow_narrow=True``.  In ``'real'``
            mode a narrow value slices the observation's own legal mask.
        allow_narrow: test-only flag permitting ``num_actions != 6792``
            (with modulo-remapped labels).  Production defaults to ``False``.
        seed: deterministic shuffle seed.  ``None`` disables shuffling
            (canonical lexicographic order).  When set, the permutation is
            computed once from the seed and the cursor tracks offset into
            the permuted order.
        verify: when ``True`` (default) shard verification runs on init;
            ``False`` skips verification for unit probes that inject bad rows
            directly.
        tensorize: ``'synthetic'`` (default) keeps the deterministic
            decision_id-hash stand-in; ``'real'`` parses ``actor_observation``
            JSON through :func:`encode_observations` and raises
            :class:`ContractError` on unparseable rows (never hash-falls-back).
        stratified: when ``True`` the sampler oversamples rare action kinds
            per ``sampling_ratios`` (deterministic interleaved order, same
            cursor/epoch resume contract over the expanded order).
            ``False`` (default) preserves the canonical cursor order
            bit-identically.
        sampling_ratios: ``{kind: multiplier}`` oversample map (values MUST
            be positive and finite); ``None`` with ``stratified=True`` uses
            :data:`DEFAULT_STRATIFIED_RATIOS` (kan/ron families ``3x``).
        kind_by_id: optional ``{decision_id: kind}`` bucket labels (explicit
            wins; rows missing from the map fall back to the row's own
            ``action_kind``/``decision_kind`` field, else ``"unknown"``).
            Batches carry the resolved labels under ``"_action_kinds"`` for
            per-type scorecards.
    """

    def __init__(
        self,
        *,
        parquet_dir: Path,
        feature_dim: int = 16,
        num_actions: int | None = 6792,
        seed: int | None = 0,
        verify: bool = True,
        tensorize: Literal["synthetic", "real"] = "synthetic",
        stratified: bool = False,
        sampling_ratios: Mapping[str, float] | None = None,
        kind_by_id: Mapping[str, str] | None = None,
        allow_narrow: bool = False,
    ) -> None:
        if tensorize not in ("synthetic", "real"):
            raise ContractError(f"tensorize must be 'synthetic' or 'real', got {tensorize!r}")
        self.tensorize: Literal["synthetic", "real"] = tensorize
        self.parquet_dir = Path(parquet_dir)
        self.feature_dim = feature_dim
        self.num_actions = num_actions if num_actions is not None else 6792
        _require_action_width(
            self.num_actions, allow_narrow=allow_narrow, where="AuthoritativeParquetDataset"
        )
        self.allow_narrow = allow_narrow
        self.seed = seed
        self._rows: list[dict[str, Any]] = []
        self._cursor: int = 0
        self._epoch: int = 0

        shards = _require_actor_parquet_dir(self.parquet_dir)
        if verify:
            _verify_shards(shards)
        for shard in shards:
            # Perf-A HIGH to_pylist break: memory_map + projection
            # + batched to_pydict replaces per-cell as_py loop.
            # Evidence:
            #  https://arrow.apache.org/docs/python/generated/
            #  pyarrow.parquet.read_table.html
            #  (memory_map=True zero-copy mmap)
            #  + https://arrow.apache.org/docs/python/generated/
            #  pyarrow.Table.html#pyarrow.Table.to_batches
            #  (max_chunksize bounds per-batch to_pylist
            #  to batch_size not full table)
            #  + https://arrow.apache.org/docs/python/dataset.html
            #  (Scanner iter_batches batch_size=2048 use_threads=True
            #  pre_buffer=True for same effect;
            #  pq.read_table is equivalent for single-file)
            #  + https://arrow.apache.org/docs/python/generated/
            #  pyarrow.Table.html#pyarrow.Table.to_pydict
            #  (zero-copy-ish per-batch conversion; pyarrow 25 idiom)
            # Previous O(N*C) dict cols reconstruct via
            # {name: table.column(name).to_pylist()} kept full-table
            # string actor_observation in memory.
            # Now: table.to_batches(max_chunksize=8192)
            # + batch.to_pydict() keeps per-batch to_pylist
            # bounded to 8192 rows.
            table = pq.read_table(
                shard,
                memory_map=True,
                columns=list(ACTOR_FIELDS),
                pre_buffer=True,
                use_threads=True,
            )
            for batch in table.to_batches(max_chunksize=8192):
                cols = batch.to_pydict()
                if not cols:
                    continue
                col_names = list(cols.keys())
                for row_tuple in zip(*cols.values(), strict=True):
                    raw: dict[str, Any] = dict(zip(col_names, row_tuple, strict=True))
                    # Verify no privileged field in the raw dict (defense in depth)
                    for bad in FORBIDDEN_IN_ACTOR:
                        if bad in raw:
                            raise ContractError(
                                f"privileged field {bad!r} in raw row {raw.get('decision_id')!r}"
                            )
                    # Also verify actor_observation JSON does not contain privileged keys
                    obs_raw: Any = raw.get("actor_observation")
                    if isinstance(obs_raw, str):
                        try:
                            obs_any: Any = json.loads(obs_raw)
                        except Exception as exc:
                            raise CorruptArtifactError(
                                f"actor_observation not JSON for {raw.get('decision_id')!r}"
                            ) from exc
                        if isinstance(obs_any, dict):
                            obs: dict[str, Any] = dict(obs_any)
                            did: Any = raw.get("decision_id")
                            for k_any in obs:
                                if not isinstance(k_any, str):
                                    continue
                                k: str = k_any
                                if k in FORBIDDEN_IN_ACTOR:
                                    raise ContractError(
                                        f"privileged field {k!r} inside "
                                        f"actor_observation for {did!r}"
                                    )
                            # dora shape check
                            for dk in ("dora_indicators", "dora", "indicators"):
                                v: Any = obs.get(dk)
                                if isinstance(v, list) and len(v) == 4:
                                    raise ContractError(
                                        f"(4,) dora shim in actor_observation[{dk!r}] for {did!r}"
                                    )
                    self._rows.append(raw)
        if len(self._rows) == 0:
            raise ContractError("authoritative dataset contains zero rows after loading")

        # Canonical order: sorted by decision_id lexicographically
        def _sort_key(r: dict[str, Any]) -> str:
            return str(r.get("decision_id", ""))

        self._rows.sort(key=_sort_key)
        # Preserve canonical order for deterministic reseeding on resume
        self._canonical_rows: list[dict[str, Any]] = list(self._rows)
        # Apply deterministic permutation if seed is not None
        if self.seed is not None:
            gen = torch.Generator().manual_seed(self.seed)
            perm_any: Any = torch.randperm(len(self._rows), generator=gen).tolist()
            perm: list[int] = [int(x) for x in perm_any]
            self._rows = [self._canonical_rows[i] for i in perm]
        if not isinstance(stratified, bool):
            raise ContractError(f"stratified must be a bool, got {stratified!r}")
        self._stratified: bool = stratified
        self._sampling_ratios: dict[str, float] | None = _validate_sampling_ratios(sampling_ratios)
        self._kind_by_id: dict[str, str] | None = _validate_kind_by_id(kind_by_id)
        self._kinds: list[str] | None = None
        self._order: list[int] = []
        self._total: int = 0
        self._rebuild_sampler_order()

    def _rebuild_sampler_order(self) -> None:
        """(Re)build kinds + order after (re)permutation; total follows order.

        Default (non-stratified, no labels): identity order, kinds ``None``,
        total ``len(rows)`` — bit-identical to the pre-stratification path.
        Stratified: kinds resolve per row (``kind_by_id`` wins, else the
        row's own kind field, else ``"unknown"``) and the order expands per
        ``sampling_ratios`` (or :data:`DEFAULT_STRATIFIED_RATIOS`).
        """
        if self._kind_by_id is not None or self._stratified:
            by_id = self._kind_by_id
            self._kinds = [
                by_id.get(str(r.get("decision_id", "")), _row_action_kind(r))
                if by_id is not None
                else _row_action_kind(r)
                for r in self._rows
            ]
        else:
            self._kinds = None
        if self._stratified:
            assert self._kinds is not None
            if self._sampling_ratios is not None and len(self._sampling_ratios) > 0:
                ratios = self._sampling_ratios
            else:
                ratios = dict(DEFAULT_STRATIFIED_RATIOS)
            self._order = build_stratified_order(
                self._kinds, ratios, seed=self.seed if self.seed is not None else 0
            )
        else:
            self._order = list(range(len(self._rows)))
        self._total = len(self._order)

    # ------------------------------------------------------------------
    # Cursor / sampler state
    # ------------------------------------------------------------------

    def get_sampler_state(self) -> dict[str, Any]:
        return {
            "offset": self._cursor,
            "seed": -1 if self.seed is None else self.seed,
            "total": self._total,
            "epoch": self._epoch,
            "stratified": self._stratified,
            "sampling_ratios": None
            if self._sampling_ratios is None
            else dict(self._sampling_ratios),
        }

    def set_sampler_state(self, state: dict[str, Any] | SamplerState) -> None:
        if isinstance(state, dict):
            offset = int(state.get("offset", 0))
            epoch = int(state.get("epoch", 0))
            seed_raw = state.get("seed", None)
        else:
            offset = state.offset
            epoch = state.epoch
            seed_raw = state.seed
        reseeded = False
        if seed_raw is not None:
            try:
                s_int = int(seed_raw)
            except Exception:
                s_int = None
            if s_int is not None:
                new_seed: int | None = None if s_int == -1 else s_int
                if new_seed != self.seed:
                    self.seed = new_seed
                    if self.seed is None:
                        self._rows = list(self._canonical_rows)
                    else:
                        gen = torch.Generator().manual_seed(self.seed)
                        perm_any: Any = torch.randperm(
                            len(self._canonical_rows), generator=gen
                        ).tolist()
                        perm: list[int] = [int(x) for x in perm_any]
                        self._rows = [self._canonical_rows[i] for i in perm]
                    reseeded = True
        if isinstance(state, dict):
            # Checkpoints written with stratification carry it; older states
            # (or SamplerState) keep this object's construction settings.
            if "stratified" in state and state["stratified"] is not None:
                s_new = state["stratified"]
                if not isinstance(s_new, bool):
                    raise ContractError(f"sampler stratified must be a bool, got {s_new!r}")
                if s_new != self._stratified:
                    self._stratified = s_new
                    reseeded = True
            if "sampling_ratios" in state:
                r_new = _validate_sampling_ratios(state["sampling_ratios"])
                if r_new != self._sampling_ratios:
                    self._sampling_ratios = r_new
                    reseeded = True
        if reseeded:
            # Order (and total) follows rows + stratified settings
            self._rebuild_sampler_order()
        if not (0 <= offset <= self._total):
            raise ContractError(f"sampler offset {offset} out of range [0,{self._total}]")
        self._cursor = offset
        self._epoch = epoch

    def sampler_cursor_for_checkpoint(self) -> dict[str, Any]:
        return self.get_sampler_state()

    def __len__(self) -> int:
        return self._total

    @property
    def cursor(self) -> int:
        return self._cursor

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def _tensorize_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Batch tensorization — deterministic, no privileged inputs.

        ``'synthetic'`` (default) uses the decision_id-hash stand-in and never
        carries ``actor_batch``; ``'real'`` parses ``actor_observation`` JSON
        through the real encoder (fail-closed :class:`ContractError`, never
        hash-falls-back) and also carries the encoded
        :class:`ActorTensorBatch` under ``actor_batch`` (flat keys
        byte-identical for compat).
        """
        if self.tensorize == "real":
            return encode_observation_rows(
                rows,
                num_actions=self.num_actions,
                feature_dim=self.feature_dim,
                allow_narrow=self.allow_narrow,
            )
        batch_features: list[torch.Tensor] = []
        batch_legal: list[torch.Tensor] = []
        batch_chosen: list[torch.Tensor] = []
        for r in rows:
            t = tensorize_actor_row(
                r,
                num_actions=self.num_actions,
                feature_dim=self.feature_dim,
                seed=self.seed if self.seed is not None else 0,
                allow_narrow=self.allow_narrow,
            )
            batch_features.append(t["features"])
            batch_legal.append(t["legal_mask"])
            batch_chosen.append(t["chosen_action_id"])
        features = torch.stack(batch_features, dim=0)  # [B,F]
        legal_mask = torch.stack(batch_legal, dim=0)  # [B,A] bool
        chosen_action_id = torch.stack(batch_chosen, dim=0)  # [B] long
        # Perf-A §4.4: pin_memory when cuda so
        # _move_batch_to_device(non_blocking=True) can overlap H2D.
        # Evidence: torch.Tensor.pin_memory docs
        # + torch/utils/data/_utils/pin_memory.py;
        # non_blocking requires pinned source.
        # Pin is no-op overhead on cpu-only hosts; guarded by cuda availability.
        if torch.cuda.is_available():
            try:
                features = features.pin_memory()
                legal_mask = legal_mask.pin_memory()
                chosen_action_id = chosen_action_id.pin_memory()
            except Exception as exc:
                logger.warning("dataset pin_memory failed, using pageable fallback: %s", exc)
        return {
            "features": features,
            "legal_mask": legal_mask,
            "chosen_action_id": chosen_action_id,
        }

    def next_batch(self, batch_size: int) -> dict[str, Any] | None:
        """Return next microbatch and advance cursor; wraps to next epoch.

        Deterministic: batches are slices of the sampler order (the seeded
        permutation, or the stratified interleaved order when enabled).  At
        end of epoch the cursor wraps to 0 and epoch increments.
        Returns ``None`` only when the dataset is empty (never for non-empty).
        """
        if batch_size <= 0:
            raise ContractError(f"batch_size must be positive, got {batch_size}")
        if self._cursor >= self._total:
            # Wrap epoch boundary: reset cursor and increment epoch (deterministic)
            self._cursor = 0
            self._epoch += 1
        end = min(self._cursor + batch_size, self._total)
        order_slice = self._order[self._cursor : end]
        rows = [self._rows[i] for i in order_slice]
        # Short tail batch is allowed; caller handles drop_last if desired
        batch: dict[str, Any] = self._tensorize_rows(rows)
        # Attach metadata for debugging (not used by model)
        batch["_decision_ids"] = [str(r["decision_id"]) for r in rows]
        batch["_epoch"] = torch.tensor(self._epoch)
        if self._kinds is not None:
            # Per-type scorecard labels: "_" prefix keeps the loop's H2D
            # mover passing them through untouched (never tensorized).
            batch["_action_kinds"] = [self._kinds[i] for i in order_slice]
        self._cursor = end
        # If we consumed exactly total, next call will wrap at top
        if self._cursor == self._total:
            # Do not auto-wrap here; allow caller to observe epoch boundary via next call
            pass
        return batch

    def peek_batch(self, batch_size: int, cursor: int | None = None) -> dict[str, Any]:
        """Non-advancing peek — useful for tests without mutating cursor."""
        cur = self._cursor if cursor is None else cursor
        end = min(cur + batch_size, self._total)
        order_slice = self._order[cur:end]
        rows = [self._rows[i] for i in order_slice]
        batch = self._tensorize_rows(rows)
        if self._kinds is not None:
            batch["_action_kinds"] = [self._kinds[i] for i in order_slice]
        return batch

    def iter_batches(self, batch_size: int, max_batches: int | None = None):
        """Generator yielding up to max_batches batches, advancing cursor."""
        yielded = 0
        while max_batches is None or yielded < max_batches:
            if self._cursor >= self._total:
                self._cursor = 0
                self._epoch += 1
            batch = self.next_batch(batch_size)
            if batch is None:
                break
            yield batch
            yielded += 1
            if (
                self._cursor >= self._total
                and yielded >= (self._total + batch_size - 1) // batch_size
            ):
                # Completed an epoch; continue wrapping if max_batches demands more
                pass
