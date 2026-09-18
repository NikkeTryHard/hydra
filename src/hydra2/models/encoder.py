"""Actor-visible tensor encoder — padded/bucketed histories with masks.

Converts :class:`ActorObservation` into :class:`ActorTensorBatch` tensors.
All inputs originate from ``ActorObservation`` (actor-visible boundary);
no hidden world, wall, or privileged label is consulted.
Padding values never carry semantics without their mask.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.contracts.common import ContractError, DigestText
from hydra2.contracts.event_vocab import EVENT_KINDS
from hydra2.contracts.observation_actor import (
    ActorObservation,
)
from hydra2.contracts.observation_types import (
    PHASES,
)
from hydra2.models.schema import (
    _BASELINE_FIELDS,
    BASELINE_ACTION_COUNT,
    HISTORY_BUCKET_LENGTHS,
    model_input_schema_digest,
)

logger = logging.getLogger(__name__)

# Visible enumerations for encoding.
_FURIKEN_STATES = ("none", "temporary", "riichi", "discard")
_RIICHI_STATES = ("none", "declared", "accepted")

_EVENT_KIND_TO_ID: dict[str, int] = {kind: idx for idx, kind in enumerate(EVENT_KINDS)}
_PHASE_TO_ID: dict[str, int] = {p: i for i, p in enumerate(PHASES)}
_FURITEN_TO_ID: dict[str, int] = {s: i for i, s in enumerate(_FURIKEN_STATES)}
_RIICHI_TO_ID: dict[str, int] = {s: i for i, s in enumerate(_RIICHI_STATES)}

_WIND_TO_ID: dict[int, int] = {27: 0, 28: 1, 29: 2, 30: 3}

# ---------------------------------------------------------------------------
# Bridge-ring glue (Phase 5 ModelLoss-owned; Python stays oracle).
# ---------------------------------------------------------------------------
#
# The bridge (`hydra2_replay_rs.ring`, Rust) owns: `ring_fill_batch` (bulk
# plane copy under detach — buffer+ring over per-row fill+pin, kills the 29
# pin copies), `validate_encoder_batch` (frozen-geometry judge: dora (5,)
# sentinel + [N,T_bucket] + history_len ceil + legal/mask widths,
# never-truncate), `bucket_for_length` (SINGLE ceil fn), and the ring
# accounting (`PyRing`: cursor/slot_used + 512-window h2d/sync p50/p99).
# Every helper below is Rust-first with an identical-messages oracle
# fallback, so CPU/test lanes without a built extension behave byte-
# identically. Fused-CE stays forced-Python (M7: per-bucket bakeoff wall +
# 1e-4 + NaN parity gate; D5 triton_op recipe gated) — gate notes live in
# ring.rs, which also carries the M5/M8/m3/m6 ledger.
#
#: Hardening (fail-closed): the oracle twins below run ONLY when the bridge
#: surface is absent (``ImportError``/missing attr → ``_ring_native()`` is
#: ``None``). Any bridge-present error (geometry ``ValueError`` text mapped
#: 1:1 to ``ContractError``, every other ``Exception`` wrapped as
#: ``ContractError``) raises — mismatch=raise, never a silent oracle
#: fallback. Evidence: canon arrays 2.8-3.8x + flats 1.4x Rust-faster linear
#: + digest parity every doc; ring padding byte-identity pinned by
#: ``ring_tests::padding_identity_zero_and_neg1_tails`` (0x00 tails on
#: 0/False planes, 0xFF tails on -1 planes). Torch owns the GPU: torch
#: allocates/owns the pinned slots physically; Rust only bulk-copies raw
#: bytes under detach (no GPU math, Burn/Candle out).
_RING_MOD: Any = None
_RING_PROBED: bool = False

#: Injectable native backend (tests monkeypatch this; production leaves None
#: so `_ring_native()` imports the compiled extension). Mirrors the
#: `_NATIVE_OVERRIDE` hook in `_rust_bridge.py` / `_rust_columnar.py` /
#: `_rust_search.py`.
_RING_NATIVE_OVERRIDE: Any = None


def _ring_native() -> Any | None:
    """Import the built `ring` submodule once; None → oracle fallback.

    ImportError-only oracle: ``None`` means the extension (or its ``ring``
    surface) is not importable — CPU/test lanes stay byte-identical via the
    twins below. Any other import-time failure raises (fail-closed).
    """
    global _RING_MOD, _RING_PROBED
    if _RING_NATIVE_OVERRIDE is not None:
        return _RING_NATIVE_OVERRIDE
    if _RING_MOD is not None:
        return _RING_MOD
    # Cheap late-arrival path: someone else (conftest, bench harness) may
    # import the built bridge after our first probe — a sys.modules hit
    # costs one dict lookup, no path rescan.
    late = sys.modules.get("hydra2_replay_rs")
    if late is not None:
        mod = getattr(late, "ring", None)
        if mod is not None:
            _RING_MOD = mod
            return _RING_MOD
    if not _RING_PROBED:
        _RING_PROBED = True
        try:
            # Established judge pattern (cf. `_rust_bridge._native` +
            # `_canon_rng`): import the top-level extension, then take the
            # submodule as an attribute (`add_submodule` publishes it).
            _RING_MOD = importlib.import_module("hydra2_replay_rs").ring
        except (ImportError, AttributeError):
            # Unbuilt extension / bridge without the ring surface only: the
            # pure-Python twins below decide (byte-identical). Anything else
            # is a broken bridge → raise, never a silent pass.
            _RING_MOD = None
    return _RING_MOD


def _bucket_length(actual: int, buckets: tuple[int, ...] = HISTORY_BUCKET_LENGTHS) -> int:
    """Ceil ``actual`` to the next bucket; over-cap callers fail closed."""
    for bucket in buckets:
        if actual <= bucket:
            return bucket
    return buckets[-1]


def _concealed_counts(observation: ActorObservation) -> list[int]:
    """Count the actor's own concealed hand; drawn tile stays separate."""
    counts = [0] * 34
    for tile in observation.concealed_hand:
        counts[int(tile) // 4] += 1
    # drawn tile stays separate per baseline — not merged into counts.
    return counts


def _visible_discards_counts(observation: ActorObservation) -> list[int]:
    """Discards plus public meld tiles as visible; dora excluded, clamped to 4."""
    counts = [0] * 34
    for river in observation.visible_discards:
        for tile in river:
            counts[int(tile) // 4] += 1
    for row in observation.visible_melds:
        for meld in row:
            # Public meld tiles count once via meld and aggregate as visible
            # alongside discards (dora excluded below, clamped to 4 per type).
            for tile in meld.tiles:
                counts[int(tile) // 4] += 1
    # Dora indicators are not discards — excluded. Clamp to 4 per tile type max for schema.
    return [min(c, 4) for c in counts]


@dataclass(frozen=True, slots=True)
class ActorTensorBatch:
    """Batched actor-visible tensors (SPEC 11.1).

    Mask polarity: ``history_mask``/``legal_mask`` use ``True`` = participate,
    while ``key_padding_mask`` in models/model.py uses ``True`` = padding
    (single ``~`` inversion at the encode-to-model boundary).
    """

    features: dict[str, torch.Tensor]
    history_mask: torch.Tensor  # [B,T] bool True=participate
    legal_mask: torch.Tensor  # [B,A] bool
    observation_hashes: tuple[DigestText, ...]
    actor_seats: torch.Tensor  # [B] int64


def encode_observations(
    observations: list[ActorObservation],
    *,
    buckets: tuple[int, ...] = HISTORY_BUCKET_LENGTHS,
    pin_memory: bool = True,
) -> ActorTensorBatch:
    """Encode a list of observations into a padded, bucketed batch.

    Validates observations via their dataclass invariant and ensures
    ``legal_mask`` has at least one True per row (nonterminal).
    """
    if len(observations) == 0:
        raise ContractError("encode_observations requires at least one observation")

    for obs in observations:
        if not isinstance(obs, ActorObservation):
            raise ContractError(f"expected ActorObservation, got {type(obs).__name__}")
        if obs.observation_hash is None:
            raise ContractError("observation_hash must be bound")
        _digest: DigestText = _bridge_contracts.make_digest_text(obs.observation_hash)

    batch_size = len(observations)

    # History bucketing: each history length → bucket ceil. Over-cap rows
    # fail closed here (replay paths quarantine earlier); tensors are NEVER
    # silently truncated to fit the model.
    history_lengths = [len(o.visible_history) for o in observations]
    max_len = max(history_lengths) if len(history_lengths) != 0 else 0
    if max_len > buckets[-1]:
        raise ContractError(
            f"visible history {max_len} exceeds model bucket cap {buckets[-1]}; "
            "rows are never truncated"
        )
    bucket_len = _bucket_length(max_len, buckets)
    # Bridge-ring geometry judge (frozen buckets/dora/legal widths + ceil
    # agreement + never-truncate, oracle-identical messages; no-op on
    # success). Custom-bucket test probes skip the frozen judge and stay pure
    # oracle. Without a built bridge the guards above/below decide alone.
    if tuple(buckets) == tuple(HISTORY_BUCKET_LENGTHS):
        validate_encoder_batch(
            batch_size=batch_size,
            max_history_len=max_len,
            bucket_t=bucket_len,
        )
    # Vectorized alloc: single numpy buffer per field outside the loop, scalar
    # slice fill inside (no per-row torch alloc), then one torch.from_numpy
    # per field (zero-copy view). Padding is 0/False, filled once up front so
    # unwritten tails stay valid without per-row init.
    # Evidence: https://docs.pytorch.org/docs/2.13/generated/torch.from_numpy.html
    # Byte-identical padding proof note (bridge-ring contract): every plane's
    # padding is a full-buffer pre-fill — 0 (int kinds), False (bool masks),
    # -1 (dora (B,5) + own_drawn int32 sentinel, 0xFF per byte) — and the
    # fill loop only overwrites valid-prefix slices. The ring bulk copy
    # (`_stage_pinned_batch` → `ring_fill_batch`) moves these bytes verbatim
    # (shape-agnostic `copy_nonoverlapping`, never re-inits), so staged slots
    # are bit-for-bit the oracle views: 0x00 tails on 0/False planes, 0xFF
    # tails on -1 planes. Pinned by `ring_tests::padding_identity_zero_and_neg1_tails`.
    history_event_kind_np = np.empty((batch_size, bucket_len), dtype=np.int64)
    history_event_kind_np.fill(0)
    history_mask_np = np.empty((batch_size, bucket_len), dtype=np.bool_)
    history_mask_np.fill(False)
    legal_mask_np = np.empty((batch_size, BASELINE_ACTION_COUNT), dtype=np.bool_)
    legal_mask_np.fill(False)
    # Scalar / categorical 1-D backing
    actor_np = np.empty((batch_size,), dtype=np.int64)
    actor_seats_np = np.empty((batch_size,), dtype=np.int64)
    actor_can_riichi_np = np.empty((batch_size,), dtype=np.bool_)
    actor_can_tsumo_np = np.empty((batch_size,), dtype=np.bool_)
    actor_furiten_np = np.empty((batch_size,), dtype=np.int64)
    concealed_hand_counts_np = np.empty((batch_size, 34), dtype=np.int32)
    concealed_hand_counts_np.fill(0)
    dealer_np = np.empty((batch_size,), dtype=np.int64)
    # Dora = bonus-indicator tile: public indicators only (padding -1);
    # wall contents/order never encoded.
    dora_indicators_np = np.empty((batch_size, 5), dtype=np.int32)
    dora_indicators_np.fill(-1)
    hand_number_np = np.empty((batch_size,), dtype=np.int32)
    honba_np = np.empty((batch_size,), dtype=np.int32)
    ippatsu_active_np = np.empty((batch_size, 4), dtype=np.bool_)
    kan_count_np = np.empty((batch_size,), dtype=np.int32)
    # Live wall = undealt count: remaining-tile COUNT (public), not wall
    # contents/order (privileged, never encoded).
    live_wall_tiles_remaining_np = np.empty((batch_size,), dtype=np.int32)
    own_drawn_tile_np = np.empty((batch_size,), dtype=np.int32)
    own_drawn_tile_np.fill(-1)
    phase_np = np.empty((batch_size,), dtype=np.int64)
    riichi_states_np = np.empty((batch_size, 4), dtype=np.int64)
    riichi_sticks_np = np.empty((batch_size,), dtype=np.int32)
    round_index_np = np.empty((batch_size,), dtype=np.int32)
    round_wind_np = np.empty((batch_size,), dtype=np.int64)
    scores_np = np.empty((batch_size, 4), dtype=np.int32)
    seat_winds_np = np.empty((batch_size, 4), dtype=np.int64)
    turn_actor_np = np.empty((batch_size,), dtype=np.int64)
    visible_discards_counts_np = np.empty((batch_size, 34), dtype=np.int32)
    visible_discards_counts_np.fill(0)

    # Hoist map lookups for inner loop (avoid dict global lookup per row)
    _wind_to_id = _WIND_TO_ID
    _furiten_to_id = _FURITEN_TO_ID
    _riichi_to_id = _RIICHI_TO_ID
    _phase_to_id = _PHASE_TO_ID
    _event_to_id = _EVENT_KIND_TO_ID

    observation_hashes: list[DigestText] = []

    for idx, obs in enumerate(observations):
        observation_hashes.append(DigestText(str(obs.observation_hash)))

        # Scalar / categorical — slice assignment (no torch.tensor alloc)
        actor_np[idx] = int(obs.actor)
        actor_seats_np[idx] = int(obs.actor)
        dealer_np[idx] = int(obs.dealer)
        turn_actor_np[idx] = int(obs.turn_actor)
        actor_can_riichi_np[idx] = obs.actor_can_riichi
        actor_can_tsumo_np[idx] = obs.actor_can_tsumo
        actor_furiten_np[idx] = _furiten_to_id[obs.actor_furiten]
        hand_number_np[idx] = obs.hand_number
        round_index_np[idx] = obs.round_index
        round_wind_np[idx] = _wind_to_id[int(obs.round_wind)]
        # seat_winds: row fill via list comp → slice (no per-row temporaries)
        seat_winds_np[idx] = [_wind_to_id[int(w)] for w in obs.seat_winds]
        honba_np[idx] = obs.honba
        riichi_sticks_np[idx] = obs.riichi_sticks
        scores_np[idx] = obs.scores
        phase_np[idx] = _phase_to_id[obs.phase]
        live_wall_tiles_remaining_np[idx] = obs.live_wall_tiles_remaining
        kan_count_np[idx] = obs.kan_count
        ippatsu_active_np[idx] = obs.ippatsu_active
        riichi_states_np[idx] = [_riichi_to_id[s] for s in obs.riichi_states]

        # Tiles (sequence assignment into the preallocated row — no per-row temp)
        concealed_hand_counts_np[idx] = _concealed_counts(obs)

        if obs.own_drawn_tile is not None:
            own_drawn_tile_np[idx] = int(obs.own_drawn_tile)
        # else already -1 fill

        dora_indicators_np[idx] = obs.dora_indicators

        visible_discards_counts_np[idx] = _visible_discards_counts(obs)

        # History — one slice fill per row (variable length, no per-event loop;
        # padding already 0/False)
        hist_len = len(obs.visible_history)
        if hist_len > 0:
            history_event_kind_np[idx, :hist_len] = [
                _event_to_id.get(event.kind, 0) for event in obs.visible_history
            ]
            history_mask_np[idx, :hist_len] = True

        # Legal mask — validate before numpy fill
        if len(obs.legal_mask) != BASELINE_ACTION_COUNT:
            raise ContractError(
                f"legal_mask length {len(obs.legal_mask)} != baseline {BASELINE_ACTION_COUNT}"
            )
        if not any(obs.legal_mask):
            raise ContractError("legal_mask must contain at least one True at a decision")
        legal_mask_np[idx] = obs.legal_mask

    # Zero-copy convert numpy → torch (from_numpy shares memory, no copy)
    # Note: torch.from_numpy zero-copy for CPU; pin_memory later enables async H2D.
    history_event_kind = torch.from_numpy(history_event_kind_np)
    history_mask = torch.from_numpy(history_mask_np)
    legal_mask = torch.from_numpy(legal_mask_np)
    actor = torch.from_numpy(actor_np)
    actor_seats = torch.from_numpy(actor_seats_np)
    actor_can_riichi = torch.from_numpy(actor_can_riichi_np)
    actor_can_tsumo = torch.from_numpy(actor_can_tsumo_np)
    actor_furiten = torch.from_numpy(actor_furiten_np)
    concealed_hand_counts = torch.from_numpy(concealed_hand_counts_np)
    dealer = torch.from_numpy(dealer_np)
    dora_indicators = torch.from_numpy(dora_indicators_np)
    hand_number = torch.from_numpy(hand_number_np)
    honba = torch.from_numpy(honba_np)
    ippatsu_active = torch.from_numpy(ippatsu_active_np)
    kan_count = torch.from_numpy(kan_count_np)
    live_wall_tiles_remaining = torch.from_numpy(live_wall_tiles_remaining_np)
    own_drawn_tile = torch.from_numpy(own_drawn_tile_np)
    phase = torch.from_numpy(phase_np)
    riichi_states = torch.from_numpy(riichi_states_np)
    riichi_sticks = torch.from_numpy(riichi_sticks_np)
    round_index = torch.from_numpy(round_index_np)
    round_wind = torch.from_numpy(round_wind_np)
    scores = torch.from_numpy(scores_np)
    seat_winds = torch.from_numpy(seat_winds_np)
    turn_actor = torch.from_numpy(turn_actor_np)
    visible_discards_counts = torch.from_numpy(visible_discards_counts_np)

    features: dict[str, torch.Tensor] = {
        "actor": actor,
        "actor_can_riichi": actor_can_riichi,
        "actor_can_tsumo": actor_can_tsumo,
        "actor_furiten": actor_furiten,
        "actor_seats": actor_seats,
        "concealed_hand_counts": concealed_hand_counts,
        "dealer": dealer,
        "dora_indicators": dora_indicators,
        "hand_number": hand_number,
        "history_event_kind": history_event_kind,
        "history_mask": history_mask,
        "honba": honba,
        "ippatsu_active": ippatsu_active,
        "kan_count": kan_count,
        "legal_mask": legal_mask,
        "live_wall_tiles_remaining": live_wall_tiles_remaining,
        "own_drawn_tile": own_drawn_tile,
        "phase": phase,
        "riichi_states": riichi_states,
        "riichi_sticks": riichi_sticks,
        "round_index": round_index,
        "round_wind": round_wind,
        "scores": scores,
        "seat_winds": seat_winds,
        "turn_actor": turn_actor,
        "visible_discards_counts": visible_discards_counts,
    }
    # Pin CPU memory for CUDA H2D overlap: non_blocking=True in
    # _move_batch_to_device requires pinned memory to overlap; without it
    # the flag is a no-op. Keep the pure CPU path for cpu-only tests (pin
    # only when CUDA is available). pin_memory() is out-of-place (returns
    # a pinned copy), so the results must be rebound.
    # Evidence: https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.pin_memory.html
    if pin_memory and torch.cuda.is_available():
        try:
            features = (
                _stage_pinned_batch(
                    features,
                    batch_size=batch_size,
                    max_history_len=max_len,
                    bucket_t=bucket_len,
                )
                if tuple(buckets) == tuple(HISTORY_BUCKET_LENGTHS)
                else {
                    name: tensor.pin_memory()  # type: ignore[attr-defined]  # reason: custom-bucket test probe skips the frozen Ring judge; pageable fallback still owns unpinnable
                    for name, tensor in features.items()
                }
            )
            history_mask = features["history_mask"]
            legal_mask = features["legal_mask"]
            actor_seats = features["actor_seats"]
        except ContractError:
            raise
        except Exception as exc:  # why-broad: any pin failure falls back to pageable
            logger.warning("encoder pin_memory failed, using pageable fallback: %s", exc)

    # Schema guard: every field in _BASELINE_FIELDS must be present, no extras besides
    # those fields. Check canonical order is respected by caller via sorted keys.
    expected = {f.name for f in _BASELINE_FIELDS}
    produced = set(features.keys())
    if expected != produced:
        missing = sorted(expected - produced)
        extra = sorted(produced - expected)
        raise ContractError(f"feature mismatch missing={missing} extra={extra}")

    return ActorTensorBatch(
        features=features,
        history_mask=history_mask,
        legal_mask=legal_mask,
        observation_hashes=tuple(observation_hashes),
        actor_seats=actor_seats,
    )


def bucket_for_length(actual: int) -> int:
    """Public helper: bucket length for a given history length.

    SINGLE ceil fn both sides call (scan-side + ring-side): Rust-first via
    ``hydra2_replay_rs.ring.bucket_for_length``; the oracle ``_bucket_length``
    runs ONLY when the bridge surface is absent (ImportError/missing attr).
    Bridge-present errors raise ContractError (ValueError text preserved
    1:1) — mismatch=raise, never a silent fallback. Over-cap returns the
    max bucket here; callers fail closed before calling (never truncate).
    """
    ring = _ring_native()
    if ring is not None:
        try:
            return int(ring.bucket_for_length(int(actual)))
        except ContractError:
            raise
        except (ImportError, AttributeError):
            pass  # bridge surface missing → oracle twin decides
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        except Exception as exc:
            raise ContractError(
                f"ring bucket_for_length failed: {type(exc).__name__}: {exc}"
            ) from exc
    return _bucket_length(actual)


def validate_encoder_batch(
    *,
    batch_size: int,
    max_history_len: int,
    bucket_t: int,
    dora_width: int = 5,
    num_actions: int = BASELINE_ACTION_COUNT,
) -> None:
    """Frozen-geometry judge pre-export (Rust-first, oracle-identical messages).

    Pins the T-geom gate: non-empty batch + ``bucket_t`` member + ceil
    agreement + dora ``(5,)`` sentinel + ``A == 6792`` + never-truncate.
    No-op on success; raises :class:`ContractError` with identical text on
    both paths (Rust ``ValueError`` text is mapped 1:1). The oracle below
    runs ONLY when the bridge surface is absent; any bridge-present error
    raises — mismatch=raise, never a silent pass.
    """
    ring = _ring_native()
    if ring is not None:
        try:
            ring.validate_encoder_batch(
                int(batch_size),
                int(max_history_len),
                int(dora_width),
                int(num_actions),
                int(bucket_t),
            )
            return
        except ContractError:
            raise
        except (ImportError, AttributeError):
            pass  # bridge surface missing → oracle twin decides below
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        except Exception as exc:
            # Bridge present but unusable (overflow, descriptor drift, ...):
            # mismatch=raise — never fall through to the oracle silently.
            raise ContractError(
                f"ring validate_encoder_batch failed: {type(exc).__name__}: {exc}"
            ) from exc
    if batch_size == 0:
        raise ContractError("encode_observations requires at least one observation")
    if bucket_t not in HISTORY_BUCKET_LENGTHS:
        raise ContractError(
            f"ring bucket_t {bucket_t} not in buckets {list(HISTORY_BUCKET_LENGTHS)}"
        )
    if max_history_len > HISTORY_BUCKET_LENGTHS[-1]:
        raise ContractError(
            f"visible history {max_history_len} exceeds model bucket cap "
            f"{HISTORY_BUCKET_LENGTHS[-1]}; rows are never truncated"
        )
    expect = _bucket_length(max_history_len)
    if bucket_t != expect:
        raise ContractError(f"ring bucket_t {bucket_t} != ceil({max_history_len}) = {expect}")
    if dora_width != 5:
        raise ContractError(f"ring dora width {dora_width} != 5 sentinel (padding -1, no 4-shim)")
    if num_actions != BASELINE_ACTION_COUNT:
        raise ContractError(
            f"ring num_actions {num_actions} != baseline {BASELINE_ACTION_COUNT} "
            "(narrow slicing is Python allow_narrow test-only)"
        )


def _oracle_pin_mapping(mapping: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """ImportError-only oracle pin path (single shared copy).

    Torch owns the pins physically (``Tensor.pin_memory`` per plane);
    Rust ``ring_fill_batch`` bulk-copies the same bytes when the bridge
    surface is present (no GPU math, verbatim under detach). Runs ONLY
    when the bridge surface is absent or as a shape precondition.
    """
    return {
        name: tensor.pin_memory()  # type: ignore[attr-defined]  # reason: CPU Tensor.pin_memory; stubs miss
        for name, tensor in mapping.items()
    }


def _stage_pinned_batch(
    mapping: dict[str, torch.Tensor],
    *,
    batch_size: int,
    max_history_len: int,
    bucket_t: int,
) -> dict[str, torch.Tensor]:
    """Bulk pinned stage: buffer+ring over per-row fill+pin (Rust-first).

    Geometry is judged first (fail closed, nothing copied), then ALL planes
    cross in ONE detached ``ring_fill_batch`` bulk copy into pre-pinned
    slots — one release instead of the 29 per-tensor ``pin_memory()`` copies.
    Torch owns the GPU: torch allocates/owns the pinned slots physically
    (pinned tensors, streams, events — no Rust GPU math); Rust moves
    shape-agnostic bytes verbatim under detach (never re-inits, never
    interprets padding). The oracle ``pin_memory()`` dict-comp runs ONLY
    when the bridge surface is absent; non-contiguous/non-tensor inputs take
    it explicitly (shape precondition, not a bridge verdict). Any
    bridge-present copy error raises ContractError (Rust ValueError text
    preserved 1:1) — mismatch=raise, never a silent fallback. Padding
    pre-fill (0/False/-1) rides along untouched (see proof note at the numpy
    alloc site).
    """
    validate_encoder_batch(
        batch_size=batch_size, max_history_len=max_history_len, bucket_t=bucket_t
    )
    ring = _ring_native()
    if ring is None:
        # ImportError-only oracle (bridge surface absent).
        return _oracle_pin_mapping(mapping)
    names = list(mapping.keys())
    srcs = [mapping[name] for name in names]
    for tensor in srcs:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_contiguous():
            # Shape precondition, not a bridge mismatch: strided views stay
            # on the identical oracle path (byte-identical either way).
            return _oracle_pin_mapping(mapping)
    # Resource alloc stays outside the Rust try: OOM/RuntimeError propagates
    # raw so the caller's pageable fallback (warn) still owns unpinnable.
    slots = [torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True) for tensor in srcs]
    try:
        ring.ring_fill_batch(
            [int(slot.data_ptr()) for slot in slots],
            [int(slot.nbytes) for slot in slots],
            [int(tensor.data_ptr()) for tensor in srcs],
            [int(tensor.nbytes) for tensor in srcs],
            int(batch_size),
            int(max_history_len),
            5,
            int(BASELINE_ACTION_COUNT),
            int(bucket_t),
        )
    except ContractError:
        raise
    except (ImportError, AttributeError):
        # Bridge surface vanished mid-call → identical oracle pin path.
        return _oracle_pin_mapping(mapping)
    except (ValueError, BufferError) as exc:
        raise ContractError(str(exc)) from exc
    except Exception as exc:
        raise ContractError(f"ring ring_fill_batch failed: {type(exc).__name__}: {exc}") from exc
    return dict(zip(names, slots, strict=True))


def validate_batch_against_schema(batch: ActorTensorBatch) -> None:
    """Runtime shape/dtype/range validation against the frozen schema.

    Padding values are allowed outside valid ranges only where masked;
    unmasked positions must satisfy valid_min/max.
    """
    field_map = {f.name: f for f in _BASELINE_FIELDS}
    # 3-sync kill: the frozen schema holds exactly one masked field
    # (history_event_kind min+max) plus the legal-row gate = 3 fail-scalars,
    # collected below and crossed in ONE stacked tolist (cap: this path
    # performs exactly 1 host sync; never one .item() per check). Order is
    # preserved (loop order, then legal), so single-fault messages are
    # identical; multi-fault precedence is dtype/shape-first (deterministic).
    # Same single-sync shape as loop_batch._window_means.
    sync_checks: list[tuple[str, torch.Tensor]] = []
    for name, tensor in batch.features.items():
        spec = field_map.get(name)
        if spec is None:
            raise ContractError(f"batch field {name!r} not in schema")
        # Dtype check (bool/int32/int64/float32)
        _dtype_map = {
            torch.bool: "bool",
            torch.int32: "int32",
            torch.int64: "int64",
            torch.float32: "float32",
        }
        dtype_name = _dtype_map.get(tensor.dtype, str(tensor.dtype))
        if dtype_name != spec.dtype:
            raise ContractError(f"field {name}: dtype {dtype_name} != spec {spec.dtype}")
        # Shape check: first dim B must match batch size, rest must match spec tail.
        batch_size = batch.actor_seats.shape[0]
        seq_len = batch.history_mask.shape[1]
        expected_shape = tuple(
            batch_size
            if dim == "B"
            else seq_len
            if dim == "T"
            else BASELINE_ACTION_COUNT
            if dim == "A"
            else dim
            for dim in spec.shape
        )
        if tensor.dim() != len(spec.shape):
            raise ContractError(f"field {name}: rank {tensor.dim()} != spec {spec.shape}")
        if tensor.shape != expected_shape:
            raise ContractError(
                f"field {name}: shape {tuple(tensor.shape)} != spec {expected_shape}"
            )
        # Validate unmasked values when mask_field applies — skip padding positions.
        if spec.mask_field is not None:
            mask = batch.features.get(spec.mask_field, batch.history_mask)
            # mask True => participate; validate only those positions.
            if spec.valid_min is not None or spec.valid_max is not None:
                valid_mask = mask
                # Range check applies only where mask shape == tensor shape.
                if valid_mask.shape == tensor.shape:
                    values = tensor[valid_mask]
                    if values.numel() > 0:
                        if spec.valid_min is not None:
                            sync_checks.append(
                                (
                                    f"field {name} below valid_min",
                                    (values < spec.valid_min).any().reshape(()),
                                )
                            )
                        if spec.valid_max is not None:
                            sync_checks.append(
                                (
                                    f"field {name} above valid_max",
                                    (values > spec.valid_max).any().reshape(()),
                                )
                            )
    # Legal mask at least one true per row (fail-scalar, same single sync).
    sync_checks.append(
        (
            "legal_mask must have at least one True per batch row",
            (~batch.legal_mask.any(dim=1).all()).reshape(()),
        )
    )
    # Single host sync for every range/legal check on this path (cap: 3
    # scalars on the frozen schema — min, max, legal — one tolist total).
    if len(sync_checks) != 0:
        failed = torch.stack([flag for _, flag in sync_checks]).tolist()
        for (message, _), is_failed in zip(sync_checks, failed, strict=True):
            if bool(is_failed):
                raise ContractError(message)
    # History mask shape must match history_event_kind.
    if batch.history_mask.shape != batch.features["history_event_kind"].shape:
        raise ContractError("history_mask shape must match history_event_kind")
    # Observation hashes length matches batch.
    if len(batch.observation_hashes) != batch.actor_seats.shape[0]:
        raise ContractError("observation_hashes length mismatch")


def input_schema_hash() -> DigestText:
    return model_input_schema_digest()
