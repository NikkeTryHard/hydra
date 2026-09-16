"""Dataset encode: the shared row-to-tensor path for training and shards.

Owns the single encode path consumed by the parquet dataset, the streaming
driver, and the offline shard builder: the deterministic synthetic
``decision_id``-hash stand-in (:func:`tensorize_actor_row`), the real
encoder fold (:func:`_real_features_from_encoder_batch`), and the
fail-closed real-mode batch entry (:func:`encode_observation_rows`) that
parses rows through :mod:`hydra2.training.dataset_parse`, encodes them
with the actor-visible encoder, and validates the legal mask and the
chosen label. Any unparseable row raises :class:`ContractError` — the
synthetic stand-in is a separate explicit entry, never a fallback.

Hardening (fail-closed): the pinned bulk stage below routes through
:func:`hydra2.models.encoder._stage_pinned_batch` (bridge ``ring`` first,
ImportError-only oracle inside; ``ContractError`` propagates — mismatch
raises, never a silent pass). Torch owns the fold math and the pageable
fallback (warn, byte-identical either way); held-out ``torch.randperm``
splits stay the torch oracle by Wave5 B2 (never Philox).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from hydra2.contracts.common import ContractError
from hydra2.models.encoder import _stage_pinned_batch, encode_observations
from hydra2.models.schema import BASELINE_ACTION_COUNT
from hydra2.training.dataset_parse import _resolve_live_or_parse

if TYPE_CHECKING:
    from hydra2.contracts.observation import ActorObservation

__all__ = [
    "_REAL_COUNT_KEYS",
    "_lexicographic_hash",
    "_real_features_from_encoder_batch",
    "_require_action_width",
    "encode_observation_rows",
    "tensorize_actor_row",
]

logger = logging.getLogger(__name__)


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
            # Bridge-ring bulk stage (buffer+ring over per-tensor pin
            # copies; the torch from_numpy views above stay untouched):
            # geometry is the encoding batch's own — batch rows, pure-Python
            # history lens (no extra sync), encoder bucket width.
            # Fail-closed: ContractError (bridge mismatch) propagates; only a
            # pin-resource failure warns down to the pageable fallback below.
            staged = _stage_pinned_batch(
                {
                    "chosen_action_id": chosen_action_id,
                    "features": features,
                    "legal_mask": legal_mask,
                },
                batch_size=len(row_list),
                max_history_len=max(len(o.visible_history) for o in observations),
                bucket_t=int(encoded.history_mask.shape[1]),
            )
            features = staged["features"]
            legal_mask = staged["legal_mask"]
            chosen_action_id = staged["chosen_action_id"]
        except ContractError:
            raise
        except Exception as exc:
            logger.warning("dataset pin_memory failed, using pageable fallback: %s", exc)
    result = {
        "features": features,
        "legal_mask": legal_mask,
        "chosen_action_id": chosen_action_id,
        "actor_batch": encoded,
    }

    return result
