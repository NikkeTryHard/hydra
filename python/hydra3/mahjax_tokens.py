"""Actor-visible MahJax wire: one 39 x 48 float32 token block per decision.

``WIRE_ID == "mahjax.red_4p.v1"`` is deliberately distinct from the native
``hydra3`` wire.  It reuses the native row geometry and tile/seat/match column
semantics, but it is fed by a MahJax 0.3.3 ``State`` and therefore cannot make
the native wire's provenance claims.  In particular, MahJax has no physical
copy ids, no canonical action ids in the actor observation, and no recorded
draw or dora-reveal events.  Nothing here treats a MahJax action-history index,
a river slot, or a meld slot as a native event ordinal.

Rows: 0..33 are tile types; 34..37 are seats relative to the actor in turn
order (0 self, 1 next, 2 opposite, 3 previous); 38 is the match token.

Tile row columns follow ``model_schema``: 0 own count including the current
draw, 1 own red held, 2 guarded own drawn face, 3 guarded own drawn red,
4 public visible copies excluding own concealed copies, 5 public visible red
copies, 6..10 one-hot revealed indicator slots 0..4, 11..46 four nine-field
copy records, and 47 reserved zero.

Copy records use the native field order: present, origin (1 own concealed,
2 own drawn, 3 public discard, 4 public meld, 5 dora indicator), owner relative
to the actor (4 table), source relative to the actor (4 absent), exposure
position, call position, red, drawn, exposing kind.  Records are sorted by the
native provenance key.  Exposing kind is the frozen ``ActionKind`` ordinal
where MahJax's action has the same meaning: 1 discard, 2 tsumogiri,
3 riichi discard, 4 chi, 5 pon, 6 daiminkan, 7 ankan, 8 kakan, 13 indicator,
and 0 for private own-copy records.  This is a semantic kind, not an event id.

Position semantics in this wire:
- A discard or call position is a 1-based index in
  ``round_state.action_history`` divided by 201.  That is the true public
  action-history order for this wire.  MahJax omits PASS and delays RONs until
  every ron answer is queued; action-history entries are therefore not the
  native event stream and not absolute arena sequence numbers.
- Own concealed copies have exposure 0.  MahJax does not record draw events in
  ``action_history``, so an own drawn copy also has exposure 0 even though its
  origin and drawn flag are set.
- A dora indicator's exposure is its reveal-slot ordinal (slot + 1) divided by
  201, because MahJax records only the revealed indicator array, not a reveal
  event.
- A called copy's exposure is the action-history position of its source river
  discard when that discard can be identified.  MahJax stores the caller only
  indirectly through the meld and leaves the called discard gray in the
  discarder's river.  When one source's identical discard and meld type are
  ambiguous, the latest matching source-local discard before the call is used.
  If no match exists the exposure is 0.
- A kakan reuses the pon's meld slot.  MahJax does not retain the old pon call
  position or per-copy provenance, so the consumed and added copies use the
  kakan call position and the called copy uses its river discard position.
  Native kakan records preserve the pre-kakan exposures.  Ankan's four copies
  all use the call position, matching native.

Guarded own draw: ``last_draw`` is round-level, so it is exposed only when the
actor is not terminated and is at a discard node or may tsumo, exactly the
``env.observe`` gate but without constructing waits or shanten.  Response and
robbing-kan windows, and terminal states, therefore never reveal a stale draw.

Visible counts exclude called-away (gray) river slots and count every meld tile
once, including own melds; a called tile is counted only in its meld.  Own
concealed copies are excluded.  Revealed indicators are included.  Visible red
uses the same rule: river reds only for unclaimed discards, meld reds inferred
from ``Meld.contains_red`` (called red, red chi, red pon, or a kan of fives),
and red indicators.  MahJax meld encoding does not retain every consumed
tile's red identity, so a meld contributes at most one red copy, on its five
face.

Seat columns 0..19 use the native meaning.  Seat 15 furiten is actor-only:
0 none, 1 any furiten while not in riichi, 2 furiten while in riichi.  Seat
16/17 are the actor's current TSUMO/RIICHI legality and are zero when inactive.
Seats 20..47 are reserved zero.  ``riichi_accepted`` (seat column 13) is zero:
MahJax accepts riichi implicitly on the next draw and records no acceptance
action.  ``last_public``, ``last_discard``, ``last_call`` and
``riichi_declared`` are public action-history positions divided by 201.

Match columns 0..15 use the native meaning: 0 present, 1 round wind / 3,
2 hand number / 16, 3 honba / 8, 4 riichi sticks / 4, 5 dealer offset / 3,
6 live wall / 70, 7 kan count / 4, 8 phase / 5, 9 actor dealer, 10 actor wind
/ 3, 11 round index / 16, 12 score spread / 100000, 13 actor score / 100000,
14 actor rank minus one / 3, 15 turn-actor offset / 3.  Columns 16..47 are
reserved zero.  MahJax stores scores in 100-point units, so scores are
multiplied by 100 before the native 100000-point normalization.  Round wind is
``round // 4``, hand number is ``round % 4 + 1``, and round index is the
0-based ``round``.  Rank ties use the native rule: one plus the number of
strictly higher scores, with no seat-wind tiebreak.  Phase is 0 inactive,
1 actor's own decision (draw or post-call discard), 2 discard response, and
3 kan response; 4 and 5 are not produced.

Only actor-visible or public state is read: the actor's hand and red flags, the
actor's legal mask for the draw gate and can-flags, public rivers, melds,
indicators, riichi/ippatsu/meld/kan flags, action history, wall counters, score,
round fields, and the guarded round-level draw.  This function never reads the
deck, opponent concealed hands, opponent waits/yaku, ``pending_winners`` or
``pending_rewards``, and never calls ``env.observe``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .model_schema import (
    COPY_RECORD_COUNT,
    COPY_RECORD_FIRST,
    COPY_RECORD_WIDTH,
    DRAWN_FACE,
    DRAWN_RED,
    INDICATOR_COUNT,
    INDICATOR_FIRST,
    MATCH_TOKEN,
    OWN_COUNT,
    OWN_RED,
    SELF_SEAT_TOKEN,
    TILE_FACES,
    TOKEN_COUNT,
    TOKEN_WIDTH,
    VISIBLE_COUNT,
    VISIBLE_RED,
)

__all__ = ["MAHJAX_WIRE_VERSION", "TOKEN_SHAPE", "WIRE_ID", "actor_tokens"]

WIRE_ID = "mahjax.red_4p.v1"
MAHJAX_WIRE_VERSION = 1
TOKEN_SHAPE = (TOKEN_COUNT, TOKEN_WIDTH)

# MahJax 0.3.3 constants.  Action values are the public state/action-stream ids.
_ACTION_HISTORY_CAP = 200
_NUM_TILE_TYPES_WITH_RED = 37
_MAX_RIVER = 24
_MAX_MELDS = 4
_MAX_MELD_TILES = 4
_MAX_DORA = 5
_OWN_COPIES = 4
_RED_FIVE_FACES = (4, 13, 22)

_ACTION_SELF_KAN_FIRST = 37
_ACTION_SELF_KAN_LAST = 71
_ACTION_TSUMOGIRI = 71
_ACTION_RIICHI = 72
_ACTION_TSUMO = 73
_ACTION_PON = 75
_ACTION_PON_RED = 76
_ACTION_OPEN_KAN = 77
_ACTION_CHI_L = 78
_ACTION_CHI_R_RED = 83

_OWN_CANDIDATES = TILE_FACES * _OWN_COPIES
_DISCARD_CANDIDATES = 4 * _MAX_RIVER
_MELD_CANDIDATES = 4 * _MAX_MELDS * _MAX_MELD_TILES
_POSITION_SCALE = float(_ACTION_HISTORY_CAP + 1)
_SORT_POSITION_BASE = _ACTION_HISTORY_CAP + 2

# River bit layout in mahjax.common.tile.
_TILE_MASK = 0x3F
_BIT_RIICHI = 0x40
_BIT_GRAY = 0x80
_BIT_TSUMOGIRI = 0x100
_MT_SHIFT = 11
_EMPTY_RIVER = 0xFFFF
_EMPTY_MELD = 0xFFFF


def _relative(seat: jax.Array, actor: jax.Array) -> jax.Array:
    return jnp.mod(seat - actor, 4).astype(jnp.int32)


def _to_tile_type(tile: jax.Array) -> jax.Array:
    tile = jnp.asarray(tile, dtype=jnp.int32)
    return jnp.where(
        tile == 34,
        4,
        jnp.where(tile == 35, 13, jnp.where(tile == 36, 22, tile)),
    )


def _to_red_tile(tile_type: jax.Array) -> jax.Array:
    return jnp.where(
        tile_type == 4,
        34,
        jnp.where(tile_type == 13, 35, jnp.where(tile_type == 22, 36, tile_type)),
    )


def _face_sum(faces: jax.Array, mask: jax.Array) -> jax.Array:
    safe = jnp.clip(jnp.where(mask, faces, 0), 0, TILE_FACES - 1).reshape(-1)
    values = jnp.where(mask, 1, 0).reshape(-1).astype(jnp.int32)
    return jnp.zeros((TILE_FACES,), dtype=jnp.int32).at[safe].add(values)


def _rank_of(scores: jax.Array) -> jax.Array:
    better = (scores[None, :] > scores[:, None]).sum(axis=1)
    return better.astype(jnp.int32) + 1


def _event_positions(player: jax.Array, event: jax.Array, capacity: int) -> jax.Array:
    """Map each seat's local event number to its one-based public history position."""
    seats = jnp.arange(4, dtype=jnp.int32)
    by_player = event[:, None] & (player[:, None] == seats)
    prior = jnp.cumsum(by_player, axis=0) - 1
    # Non-events and slots beyond capacity have no destination. Clamping them
    # onto the last slot would overwrite a real discard or meld timestamp.
    slot = jnp.where(by_player, prior, capacity)
    position = jnp.arange(1, player.size + 1, dtype=jnp.int32)[:, None]
    return jnp.zeros((4, capacity), jnp.int32).at[seats, slot].set(position, mode="drop")


def actor_tokens(state) -> jax.Array:
    """Return the acting seat's MahJax token wire as float32 ``[39, 48]``.

    ``state`` is a MahJax 0.3.3 ``State``.  The acting seat is
    ``state.current_player``.  The function is pure JAX and is intended to be
    called directly or under ``jax.vmap``; it performs no host-side branching
    on state values.
    """

    capacity = state.round_state.action_history.shape[1]
    if capacity != _ACTION_HISTORY_CAP:
        raise ValueError(
            f"{WIRE_ID} expects a {_ACTION_HISTORY_CAP}-step MahJax action history, got {capacity}"
        )

    actor = jnp.asarray(state.current_player, dtype=jnp.int32)
    players = state.players
    round_state = state.round_state
    scores = round_state.score.astype(jnp.int32)
    terminated = state.terminated | state.truncated
    active = ~terminated
    legal = state.legal_action_mask

    # ---- own hand, red flags, and the guarded own draw -------------------
    hand = players.hand[actor].astype(jnp.int32)
    red_counts = players.hand_with_red[actor, TILE_FACES : TILE_FACES + 3].astype(jnp.int32)
    own_red = (
        jnp.zeros((TILE_FACES,), dtype=jnp.bool_)
        .at[jnp.asarray(_RED_FIVE_FACES, dtype=jnp.int32)]
        .set(red_counts > 0)
    )

    last_draw = round_state.last_draw.astype(jnp.int32)
    is_discard_node = active & (legal[:_NUM_TILE_TYPES_WITH_RED].any() | legal[_ACTION_TSUMOGIRI])
    holds_draw = active & (is_discard_node | legal[_ACTION_TSUMO])
    drawn_present = holds_draw & (last_draw >= 0) & (last_draw < _NUM_TILE_TYPES_WITH_RED)
    drawn_type = _to_tile_type(last_draw)
    drawn_face = jnp.where(drawn_present, drawn_type, -1)
    drawn_red = drawn_present & (last_draw >= TILE_FACES)

    own_face = jnp.repeat(jnp.arange(TILE_FACES, dtype=jnp.int32), _OWN_COPIES)
    own_slot = jnp.tile(jnp.arange(_OWN_COPIES, dtype=jnp.int32), TILE_FACES)
    own_present = own_slot < hand[own_face]
    own_drawn = own_present & drawn_present & (own_face == drawn_face) & (own_slot == 0)
    own_origin = jnp.where(own_drawn, 2, 1)
    own_red_slot = jnp.where(drawn_present & ~drawn_red & (own_face == drawn_face), 1, 0)
    own_record_red = own_present & own_red[own_face] & (own_slot == own_red_slot)

    # ---- public rivers ---------------------------------------------------
    river = players.river.astype(jnp.int32)
    river_empty = river == _EMPTY_RIVER
    river_tile = jnp.where(river_empty, -1, river & _TILE_MASK)
    river_gray = (~river_empty) & ((river & _BIT_GRAY) != 0)
    river_tsumogiri = (~river_empty) & ((river & _BIT_TSUMOGIRI) != 0)
    river_riichi = (~river_empty) & ((river & _BIT_RIICHI) != 0)
    river_meld_type = jnp.where(river_empty, 0, (river >> _MT_SHIFT) & 0x7)
    river_face = _to_tile_type(river_tile)
    river_red = (~river_empty) & (river_tile >= TILE_FACES)
    river_open = (~river_empty) & ~river_gray

    # ---- public action history ------------------------------------------
    history = round_state.action_history.astype(jnp.int32)
    hist_player = history[0]
    hist_action = history[1]
    hist_valid = hist_player >= 0
    hist_is_discard = hist_valid & (hist_action >= 0) & (hist_action < _NUM_TILE_TYPES_WITH_RED)
    hist_is_chi = hist_valid & (hist_action >= _ACTION_CHI_L) & (hist_action <= _ACTION_CHI_R_RED)
    hist_is_pon = hist_valid & ((hist_action == _ACTION_PON) | (hist_action == _ACTION_PON_RED))
    hist_is_open_kan = hist_valid & (hist_action == _ACTION_OPEN_KAN)
    hist_is_self_kan = hist_valid & (
        (hist_action >= _ACTION_SELF_KAN_FIRST) & (hist_action < _ACTION_SELF_KAN_LAST)
    )
    hist_is_riichi = hist_valid & (hist_action == _ACTION_RIICHI)

    history_index = jnp.arange(_ACTION_HISTORY_CAP, dtype=jnp.int32)
    history_one = history_index + 1
    player_ids = jnp.arange(4, dtype=jnp.int32)
    discard_position = _event_positions(hist_player, hist_is_discard, _MAX_RIVER)

    # ---- public melds ----------------------------------------------------
    melds = players.melds.astype(jnp.int32)
    meld_empty = melds == _EMPTY_MELD
    meld_action = jnp.where(meld_empty, -1, melds & 0x7F)
    meld_target = jnp.where(meld_empty, -1, (melds >> 7) & 0x3F)
    meld_src = jnp.where(meld_empty, -1, (melds >> 13) & 0x3)
    meld_target_red = (~meld_empty) & (((melds >> 15) & 0x1) != 0)

    meld_chi = (meld_action >= _ACTION_CHI_L) & (meld_action <= _ACTION_CHI_R_RED)
    meld_pon = (meld_action == _ACTION_PON) | (meld_action == _ACTION_PON_RED)
    meld_open_kan = meld_action == _ACTION_OPEN_KAN
    meld_self_kan = (meld_action >= _ACTION_SELF_KAN_FIRST) & (meld_action < _ACTION_SELF_KAN_LAST)
    meld_ankan = meld_self_kan & (meld_src == 0)
    meld_kakan = meld_self_kan & (meld_src != 0)
    meld_kan = meld_open_kan | meld_self_kan
    meld_chi_index = jnp.where(
        (meld_action == _ACTION_CHI_L) | (meld_action == 79),
        0,
        jnp.where(
            (meld_action == 80) | (meld_action == 81),
            1,
            jnp.where((meld_action == 82) | (meld_action == _ACTION_CHI_R_RED), 2, -1),
        ),
    )
    meld_low = meld_target - meld_chi_index
    meld_kind = jnp.where(
        meld_chi,
        4,
        jnp.where(
            meld_pon,
            5,
            jnp.where(
                meld_open_kan,
                6,
                jnp.where(meld_ankan, 7, jnp.where(meld_kakan, 8, 0)),
            ),
        ),
    )
    meld_count = jnp.where(meld_chi | meld_pon, 3, jnp.where(meld_kan, 4, 0))
    meld_tile_slot = jnp.arange(_MAX_MELD_TILES, dtype=jnp.int32)
    meld_face = jnp.where(
        meld_chi[..., None],
        meld_low[..., None] + meld_tile_slot,
        meld_target[..., None],
    )
    meld_present = (
        (~meld_empty)[..., None]
        & (meld_tile_slot < meld_count[..., None])
        & (meld_face >= 0)
        & (meld_face < TILE_FACES)
    )
    meld_called = meld_present & jnp.where(
        meld_chi[..., None],
        meld_tile_slot == meld_chi_index[..., None],
        jnp.where(
            meld_pon[..., None] | meld_open_kan[..., None] | meld_kakan[..., None],
            meld_tile_slot == 0,
            False,
        ),
    )
    meld_target_tile = jnp.where(meld_target_red, _to_red_tile(meld_target), meld_target)

    meld_p = jnp.repeat(jnp.arange(4, dtype=jnp.int32), _MAX_MELDS)
    meld_action_flat = meld_action.reshape(-1)
    meld_present_flat = (~meld_empty).reshape(-1)
    meld_kakan_flat = meld_kakan.reshape(-1)
    meld_target_tile_flat = meld_target_tile.reshape(-1)
    meld_src_flat = meld_src.reshape(-1)

    # A call position is the public history entry that created (or, for a
    # kakan, last reclassified) this meld.  Kakan entries are excluded from
    # the creation sequence because they reuse an existing meld slot.
    kakan_match = (
        meld_kakan_flat[:, None]
        & hist_valid[None, :]
        & (meld_p[:, None] == hist_player[None, :])
        & (meld_action_flat[:, None] == hist_action[None, :])
    )
    kakan_occurrence = kakan_match.any(axis=0)
    creation_event = hist_valid & (
        hist_is_chi | hist_is_pon | hist_is_open_kan | (hist_is_self_kan & ~kakan_occurrence)
    )
    creation_position = _event_positions(hist_player, creation_event, _MAX_MELDS)

    kakan_position = jnp.max(jnp.where(kakan_match, history_one[None, :], 0), axis=1)
    creation_position_flat = creation_position.reshape(-1)
    call_raw_flat = jnp.where(
        meld_kakan_flat & (kakan_position > 0),
        kakan_position,
        creation_position_flat,
    )

    # Called-copy exposure: find the source's gray river discard that the meld
    # consumed.  The river stores the called tile and the meld family but not
    # the caller, so an exact duplicate consumed discard is resolved to the
    # latest matching source-local discard before the call history position.
    called_flat = meld_called.reshape(_MELD_CANDIDATES // _MAX_MELD_TILES, _MAX_MELD_TILES)
    source_abs_flat = jnp.where(
        meld_present_flat,
        jnp.mod(meld_p + meld_src_flat, 4),
        0,
    )
    meld_chi_index_flat = meld_chi_index.reshape(-1)
    expected_meld_type = jnp.where(
        meld_chi.reshape(-1),
        3 + meld_chi_index_flat,
        jnp.where(
            meld_pon.reshape(-1) | meld_kakan_flat,
            1,
            jnp.where(meld_open_kan.reshape(-1), 2, 0),
        ),
    )
    # Only one tile of a meld can come from the source river. Resolve that
    # discard once per meld; uncalled copies retain their own call timestamp.
    source_position = discard_position[source_abs_flat]
    called_match = (
        river_gray[source_abs_flat]
        & (river_tile[source_abs_flat] == meld_target_tile_flat[:, None])
        & (river_meld_type[source_abs_flat] == expected_meld_type[:, None])
        & (source_position > 0)
        & ((call_raw_flat[:, None] <= 0) | (source_position < call_raw_flat[:, None]))
    )
    called_select = jnp.argmax(jnp.where(called_match, source_position, -1), axis=1)
    called_has_match = called_match.any(axis=1)
    selected_position = source_position[jnp.arange(meld_p.size), called_select]
    called_exposure = jnp.where(called_has_match, selected_position, 0)[:, None]
    called_drawn = (called_has_match & river_tsumogiri[source_abs_flat, called_select])[:, None]

    meld_exposure = jnp.where(called_flat, called_exposure, call_raw_flat[:, None])
    meld_drawn = jnp.where(called_flat, called_drawn, False)
    meld_source = jnp.where(
        called_flat,
        jnp.where(source_abs_flat[:, None] < 4, source_abs_flat[:, None], 4),
        4,
    )

    is_five = (meld_target == 4) | (meld_target == 13) | (meld_target == 22)
    red_chi = (meld_action == 79) | (meld_action == 81) | (meld_action == _ACTION_CHI_R_RED)
    meld_contains_red = (
        meld_target_red | (meld_action == _ACTION_PON_RED) | red_chi | (meld_kan & is_five)
    )
    red_face = jnp.where(meld_chi, meld_low + 1, meld_target)
    red_slot = jnp.where(
        meld_chi[..., None],
        meld_face == red_face[..., None],
        jnp.where(
            meld_ankan[..., None],
            meld_tile_slot == 0,
            jnp.where(
                meld_target_red[..., None],
                meld_tile_slot == 0,
                meld_tile_slot == 1,
            ),
        ),
    )
    meld_red = meld_present & meld_contains_red[..., None] & red_slot

    # ---- revealed indicators --------------------------------------------
    dora = round_state.dora_indicators.astype(jnp.int32)
    indicator_present = dora >= 0
    indicator_face = _to_tile_type(dora)
    indicator_red = indicator_present & (dora >= TILE_FACES)
    indicator_exposure = jnp.arange(1, _MAX_DORA + 1, dtype=jnp.int32)

    indicator_columns = jnp.zeros((TILE_FACES, INDICATOR_COUNT), dtype=jnp.float32)
    for slot in range(INDICATOR_COUNT):
        hit = indicator_present[slot] & (
            jnp.arange(TILE_FACES, dtype=jnp.int32) == indicator_face[slot]
        )
        indicator_columns = indicator_columns.at[:, slot].set(hit.astype(jnp.float32))

    # ---- candidate copy records -----------------------------------------
    own_owner = jnp.broadcast_to(actor, (_OWN_CANDIDATES,))
    own_source = jnp.full((_OWN_CANDIDATES,), 4, dtype=jnp.int32)
    own_exposure = jnp.zeros((_OWN_CANDIDATES,), dtype=jnp.int32)
    own_call = jnp.zeros((_OWN_CANDIDATES,), dtype=jnp.int32)
    own_kind = jnp.zeros((_OWN_CANDIDATES,), dtype=jnp.int32)

    discard_face = river_face.reshape(-1)
    discard_present = river_open.reshape(-1)
    discard_owner = jnp.repeat(jnp.arange(4, dtype=jnp.int32), _MAX_RIVER)
    discard_exposure = discard_position.reshape(-1)
    discard_call = jnp.zeros((_DISCARD_CANDIDATES,), dtype=jnp.int32)
    discard_red = river_red.reshape(-1)
    discard_drawn = river_tsumogiri.reshape(-1)
    discard_kind = jnp.where(river_riichi, 3, jnp.where(river_tsumogiri, 2, 1)).reshape(-1)

    meld_face_flat = meld_face.reshape(-1)
    meld_present_flat_full = meld_present.reshape(-1)
    meld_owner = jnp.repeat(meld_p, _MAX_MELD_TILES)
    meld_exposure_flat = meld_exposure.reshape(-1)
    meld_call_flat = call_raw_flat[:, None] * jnp.ones((1, _MAX_MELD_TILES), dtype=jnp.int32)
    meld_call_flat = meld_call_flat.reshape(-1)
    meld_red_flat = meld_red.reshape(-1)
    meld_drawn_flat = meld_drawn.reshape(-1)
    meld_kind_flat = jnp.repeat(meld_kind.reshape(-1), _MAX_MELD_TILES)
    meld_source_flat = meld_source.reshape(-1)

    faces = jnp.concatenate((own_face, discard_face, meld_face_flat, indicator_face))
    present = jnp.concatenate(
        (
            own_present,
            discard_present,
            meld_present_flat_full,
            indicator_present,
        )
    )
    origin = jnp.concatenate(
        (
            own_origin,
            jnp.full((_DISCARD_CANDIDATES,), 3, dtype=jnp.int32),
            jnp.full((_MELD_CANDIDATES,), 4, dtype=jnp.int32),
            jnp.full((_MAX_DORA,), 5, dtype=jnp.int32),
        )
    )
    owner_abs = jnp.concatenate(
        (
            own_owner,
            discard_owner,
            meld_owner,
            jnp.full((_MAX_DORA,), 4, dtype=jnp.int32),
        )
    )
    source_abs = jnp.concatenate(
        (
            own_source,
            jnp.full((_DISCARD_CANDIDATES,), 4, dtype=jnp.int32),
            meld_source_flat,
            jnp.full((_MAX_DORA,), 4, dtype=jnp.int32),
        )
    )
    exposure_raw = jnp.concatenate(
        (
            own_exposure,
            discard_exposure,
            meld_exposure_flat,
            indicator_exposure,
        )
    )
    call_raw = jnp.concatenate(
        (
            own_call,
            discard_call,
            meld_call_flat,
            jnp.zeros((_MAX_DORA,), dtype=jnp.int32),
        )
    )
    red = jnp.concatenate(
        (
            own_record_red,
            discard_red,
            meld_red_flat,
            indicator_red,
        )
    )
    drawn = jnp.concatenate(
        (
            own_drawn,
            discard_drawn,
            meld_drawn_flat,
            jnp.zeros((_MAX_DORA,), dtype=jnp.bool_),
        )
    )
    kind = jnp.concatenate(
        (
            own_kind,
            discard_kind,
            meld_kind_flat,
            jnp.full((_MAX_DORA,), 13, dtype=jnp.int32),
        )
    )

    sort_key = exposure_raw.astype(jnp.int32)
    sort_key = sort_key * _SORT_POSITION_BASE + call_raw
    sort_key = sort_key * 6 + origin
    sort_key = sort_key * 5 + owner_abs
    sort_key = sort_key * 5 + source_abs
    sort_key = sort_key * 14 + kind
    sort_key = sort_key * 2 + drawn.astype(jnp.int32)
    sort_key = sort_key * 2 + red.astype(jnp.int32)

    face_ids = jnp.arange(TILE_FACES, dtype=jnp.int32)[:, None]
    match = present[None, :] & (faces[None, :] == face_ids)
    # Only four records can survive per face. A GPU top_k on the full candidate
    # axis lowers to segmented sorts, doing far more work than four reductions.
    # argmax keeps the first input index on ties, preserving the wire order.
    missing = jnp.iinfo(jnp.int32).min
    scores_by_face = jnp.where(match, -sort_key[None, :], missing)
    indices = []
    valid_slots = []
    for _ in range(COPY_RECORD_COUNT):
        index = jnp.argmax(scores_by_face, axis=1)
        indices.append(index)
        valid_slots.append(scores_by_face[face_ids[:, 0], index] != missing)
        scores_by_face = scores_by_face.at[face_ids[:, 0], index].set(missing)
    top = jnp.stack(indices, axis=1)
    selected = jnp.stack(valid_slots, axis=1)

    def masked_int(values: jax.Array) -> jax.Array:
        return jnp.where(selected, values[top], 0).astype(jnp.int32)

    def masked_bool(values: jax.Array) -> jax.Array:
        return jnp.where(selected, values[top], False)

    selected_owner = masked_int(owner_abs)
    owner_relative = jnp.where(
        selected, jnp.where(selected_owner < 4, _relative(selected_owner, actor), 4), 0
    )
    selected_source = masked_int(source_abs)
    source_relative = jnp.where(
        selected, jnp.where(selected_source < 4, _relative(selected_source, actor), 4), 0
    )
    exposed_value = masked_int(exposure_raw).astype(jnp.float32)
    call_value = masked_int(call_raw).astype(jnp.float32)
    exposed_value = jnp.where(exposed_value > 0.0, exposed_value / _POSITION_SCALE, 0.0)
    call_value = jnp.where(call_value > 0.0, call_value / _POSITION_SCALE, 0.0)

    record_fields = jnp.stack(
        (
            selected.astype(jnp.float32),
            masked_int(origin).astype(jnp.float32),
            owner_relative.astype(jnp.float32),
            source_relative.astype(jnp.float32),
            exposed_value,
            call_value,
            masked_bool(red).astype(jnp.float32),
            masked_bool(drawn).astype(jnp.float32),
            masked_int(kind).astype(jnp.float32),
        ),
        axis=-1,
    )

    # ---- public visible counts and copy records -------------------------
    public_count = (
        _face_sum(discard_face, discard_present)
        + _face_sum(meld_face_flat, meld_present_flat_full)
        + _face_sum(indicator_face, indicator_present)
    )
    public_red = (
        _face_sum(discard_face, discard_present & discard_red)
        + _face_sum(meld_face_flat, meld_present_flat_full & meld_red_flat)
        + _face_sum(indicator_face, indicator_present & indicator_red)
    )

    # ---- tile rows -------------------------------------------------------
    tokens = jnp.zeros((TOKEN_COUNT, TOKEN_WIDTH), dtype=jnp.float32)
    tiles = jnp.arange(TILE_FACES, dtype=jnp.int32)
    tokens = tokens.at[:TILE_FACES, OWN_COUNT].set(hand.astype(jnp.float32))
    tokens = tokens.at[:TILE_FACES, OWN_RED].set(own_red.astype(jnp.float32))
    tokens = tokens.at[:TILE_FACES, DRAWN_FACE].set(
        jnp.where(drawn_present, tiles == drawn_face, False).astype(jnp.float32)
    )
    tokens = tokens.at[:TILE_FACES, DRAWN_RED].set(
        (drawn_red & (tiles == drawn_face)).astype(jnp.float32)
    )
    tokens = tokens.at[:TILE_FACES, VISIBLE_COUNT].set(public_count.astype(jnp.float32))
    tokens = tokens.at[:TILE_FACES, VISIBLE_RED].set(public_red.astype(jnp.float32))
    tokens = tokens.at[
        :TILE_FACES,
        INDICATOR_FIRST : INDICATOR_FIRST + INDICATOR_COUNT,
    ].set(indicator_columns)
    tokens = tokens.at[
        :TILE_FACES,
        COPY_RECORD_FIRST : COPY_RECORD_FIRST + COPY_RECORD_COUNT * COPY_RECORD_WIDTH,
    ].set(record_fields.reshape(TILE_FACES, -1))

    # ---- seat rows -------------------------------------------------------
    relative_seats = jnp.arange(4, dtype=jnp.int32)
    absolute_seats = jnp.mod(actor + relative_seats, 4)
    seat_wind = round_state.seat_wind.astype(jnp.int32)
    dealer = round_state.dealer.astype(jnp.int32)
    round_index = round_state.round.astype(jnp.int32)
    round_wind = round_index // 4
    rank = _rank_of(scores)

    last_public_raw = jnp.max(
        jnp.where(
            hist_valid[:, None] & (hist_player[:, None] == player_ids[None, :]),
            history_one[:, None],
            0,
        ),
        axis=0,
    )
    last_discard_raw = jnp.max(
        jnp.where(
            hist_is_discard[:, None] & (hist_player[:, None] == player_ids[None, :]),
            history_one[:, None],
            0,
        ),
        axis=0,
    )
    is_call_history = hist_is_chi | hist_is_pon | hist_is_open_kan | hist_is_self_kan
    last_call_raw = jnp.max(
        jnp.where(
            is_call_history[:, None] & (hist_player[:, None] == player_ids[None, :]),
            history_one[:, None],
            0,
        ),
        axis=0,
    )
    riichi_declared_raw = jnp.max(
        jnp.where(
            hist_is_riichi[:, None] & (hist_player[:, None] == player_ids[None, :]),
            history_one[:, None],
            0,
        ),
        axis=0,
    )

    def position(raw: jax.Array) -> jax.Array:
        values = raw[absolute_seats].astype(jnp.float32)
        return jnp.where(values > 0.0, values / _POSITION_SCALE, 0.0)

    seat_block = jnp.zeros((4, TOKEN_WIDTH), dtype=jnp.float32)
    seat_block = seat_block.at[:, 0].set(1.0)
    seat_block = seat_block.at[:, 1].set(relative_seats.astype(jnp.float32) / 3.0)
    seat_block = seat_block.at[:, 2].set(seat_wind[absolute_seats].astype(jnp.float32) / 3.0)
    seat_block = seat_block.at[:, 3].set((dealer == absolute_seats).astype(jnp.float32))
    seat_block = seat_block.at[:, 4].set(
        scores[absolute_seats].astype(jnp.float32) * 100.0 / 100_000.0
    )
    seat_block = seat_block.at[:, 5].set((rank[absolute_seats] - 1).astype(jnp.float32) / 3.0)
    riichi_state = players.riichi[absolute_seats]
    riichi_declared_state = players.riichi_declared[absolute_seats]
    seat_block = seat_block.at[:, 6].set(
        jnp.where(
            riichi_state,
            2.0,
            jnp.where(riichi_declared_state, 1.0, 0.0),
        )
        / 2.0
    )
    seat_block = seat_block.at[:, 7].set(players.ippatsu[absolute_seats].astype(jnp.float32))
    seat_block = seat_block.at[:, 8].set(
        players.meld_counts[absolute_seats].astype(jnp.float32) / 4.0
    )
    seat_block = seat_block.at[:, 9].set(position(last_public_raw))
    seat_block = seat_block.at[:, 10].set(position(last_discard_raw))
    seat_block = seat_block.at[:, 11].set(position(last_call_raw))
    seat_block = seat_block.at[:, 12].set(position(riichi_declared_raw))
    # Seat column 13 (riichi acceptance) is unavailable in MahJax.
    seat_block = seat_block.at[:, 14].set(
        (state.current_player == absolute_seats).astype(jnp.float32)
    )
    furiten = players.furiten_by_discard[actor] | players.furiten_by_pass[actor]
    furiten_ordinal = jnp.where(furiten, jnp.where(players.riichi[actor], 2, 1), 0)
    seat_block = seat_block.at[:, 15].set(
        jnp.where(
            relative_seats == 0,
            furiten_ordinal.astype(jnp.float32) / 3.0,
            0.0,
        )
    )
    seat_block = seat_block.at[:, 16].set(
        jnp.where(
            relative_seats == 0,
            active & legal[_ACTION_TSUMO],
            False,
        ).astype(jnp.float32)
    )
    seat_block = seat_block.at[:, 17].set(
        jnp.where(
            relative_seats == 0,
            active & legal[_ACTION_RIICHI],
            False,
        ).astype(jnp.float32)
    )
    seat_block = seat_block.at[:, 18].set(
        jnp.mod(dealer - absolute_seats, 4).astype(jnp.float32) / 3.0
    )
    seat_block = seat_block.at[:, 19].set(
        jnp.mod(seat_wind[absolute_seats] - round_wind, 4).astype(jnp.float32) / 3.0
    )
    tokens = tokens.at[SELF_SEAT_TOKEN : SELF_SEAT_TOKEN + 4, :].set(seat_block)

    # ---- match row -------------------------------------------------------
    phase = jnp.where(
        active,
        jnp.where(
            is_discard_node | legal[_ACTION_TSUMO],
            1,
            jnp.where(round_state.kan_declared, 3, 2),
        ),
        0,
    )
    live_wall = jnp.maximum(
        round_state.next_deck_ix.astype(jnp.int32) - round_state.last_deck_ix.astype(jnp.int32) + 1,
        0,
    )
    kan_count = players.n_kan.astype(jnp.int32).sum()
    match_row = jnp.zeros((TOKEN_WIDTH,), dtype=jnp.float32)
    match_row = match_row.at[0].set(1.0)
    match_row = match_row.at[1].set(round_wind.astype(jnp.float32) / 3.0)
    match_row = match_row.at[2].set(((round_index % 4) + 1).astype(jnp.float32) / 16.0)
    match_row = match_row.at[3].set(round_state.honba.astype(jnp.float32) / 8.0)
    match_row = match_row.at[4].set(round_state.kyotaku.astype(jnp.float32) / 4.0)
    match_row = match_row.at[5].set(jnp.mod(dealer - actor, 4).astype(jnp.float32) / 3.0)
    match_row = match_row.at[6].set(live_wall.astype(jnp.float32) / 70.0)
    match_row = match_row.at[7].set(kan_count.astype(jnp.float32) / 4.0)
    match_row = match_row.at[8].set(phase.astype(jnp.float32) / 5.0)
    match_row = match_row.at[9].set((dealer == actor).astype(jnp.float32))
    match_row = match_row.at[10].set(seat_wind[actor].astype(jnp.float32) / 3.0)
    match_row = match_row.at[11].set(round_index.astype(jnp.float32) / 16.0)
    match_row = match_row.at[12].set(
        (scores.max() - scores.min()).astype(jnp.float32) * 100.0 / 100_000.0
    )
    match_row = match_row.at[13].set(scores[actor].astype(jnp.float32) * 100.0 / 100_000.0)
    match_row = match_row.at[14].set((rank[actor] - 1).astype(jnp.float32) / 3.0)
    match_row = match_row.at[15].set(
        jnp.mod(state.current_player - actor, 4).astype(jnp.float32) / 3.0
    )
    tokens = tokens.at[MATCH_TOKEN, :].set(match_row)

    return tokens
