"""MahJax 0.3.3 four-player action descriptors on the shared candidate wire.

MahJax exposes one fixed 87-slot action table per four-player state
(``mahjax/common/action.py:22-40``), so this adapter keeps every slot: row ``i``
describes MahJax action ``i``, and ``descriptors[i, ACTION_ID]`` is ``i``
exactly when ``state.legal_action_mask[i]`` is true and ``-1`` otherwise.
MahJax has neither canonical event ordinals nor canonical action ids, so the
id column carries MahJax action indices only; it never aliases a native
canonical id. Array order is the engine table order, so the runtime can take a
selected index as the action id directly.

Field semantics follow the native candidate writer
(``native/src/features/candidates.rs:11-68``): kind, relative source seat,
primary tile, drawn/riichi/claim bits, four actor-owned removal slots and their
red bits. As native pads, an absent primary or removal face is ``-1``, an
absent source is ``SOURCE_ABSENT``, absent red bits are ``0``, and a row the
engine masks out is all ``-1`` (the policy's ``ACTION_ID >= 0`` gate).

Two MahJax shapes differ from native and are handled explicitly:

* Riichi is a declaration decision (action 72) followed by a forced discard.
  Action 72 is an adapter-only candidate: kind ``RIICHI_DISCARD`` with no tile
  and no removals, so the declaration itself edits no hand. The discard that
  completes the declaration is kind ``RIICHI_DISCARD`` with its real tile,
  removal, drawn and riichi bits, which is the native descriptor for that
  discard. Later forced draws in accepted riichi are ordinary ``TSUMOGIRI``
  (kind 2), matching the native tsumogiri candidate for the drawn tile.
* Ankan and kakan share actions 37..70. The prior pon of the acting seat
  selects kakan; the removal slots use the exact copies
  ``Hand.closed_kan``/``Hand.added_kan`` remove, including the red five.

Only the acting seat's hand and pon array, the public offer target, the public
last-discard tsumogiri flag and the acting seat's legal mask are read. The
wall, other hands and ura indicators never influence the output.
"""

from typing import Protocol

import jax.numpy as jnp
from jax import Array

from .model_schema import (
    ACTION_CLAIM_INTENT,
    ACTION_DECLARE_RIICHI,
    ACTION_DRAWN_DISCARD,
    ACTION_ID,
    ACTION_KIND,
    ACTION_PRIMARY_FACE,
    ACTION_PRIMARY_RED,
    ACTION_REMOVAL_COUNT,
    ACTION_REMOVAL_FIRST,
    ACTION_REMOVAL_RED_FIRST,
    ACTION_SOURCE,
    ACTION_WIDTH,
    KIND_ANKAN,
    KIND_DISCARD,
    KIND_PASS,
    KIND_RIICHI_DISCARD,
    KIND_TSUMOGIRI,
    SOURCE_ABSENT,
)

# Wire revisions are explicit: any change to a slot's meaning bumps this.
MAHJAX_ACTION_WIRE_VERSION = 1
MAHJAX_ACTION_COUNT = 87

# Frozen native kind ordinals outside the names model_schema exports
# (vendor/hydra2/crates/feed/src/census.rs:46-61).
_KIND_CHI = 4
_KIND_PON = 5
_KIND_DAIMINKAN = 6
_KIND_KAKAN = 8
_KIND_RON = 9
_KIND_TSUMO = 10
_KIND_ABORT_NINE_TERMINALS = 11
_KIND_ACCEPT_ABORTIVE_DRAW = 12

# Frozen MahJax four-player action indices (mahjax/common/action.py:22-40).
_ACTION_TSUMOGIRI = 71
_ACTION_RIICHI = 72
_ACTION_TSUMO = 73
_ACTION_RON = 74
_ACTION_PON = 75
_ACTION_PON_RED = 76
_ACTION_OPEN_KAN = 77
_ACTION_CHI_FIRST = 78
_ACTION_CHI_LAST = 83
_ACTION_PASS = 84
_ACTION_KYUUSHU = 85

_TILE_TYPES = 34
_SUITED_FACES = 27
_TILE_TYPES_WITH_RED = 37
_NUM_PLAYERS = 4
_RIVER_RIICHI_BIT = 1 << 6
_RIVER_TSUMOGIRI_BIT = 1 << 8
_EMPTY_RIVER = 0xFFFF

# MahJax tile types 34/35/36 are the red five of each suit
# (mahjax/common/tile.py:31-45); their suit faces are 5m/5p/5s.
_RED_FACE = jnp.array((4, 13, 22), dtype=jnp.int32)

# Chi hand tiles relative to the called tile: CHI_L, CHI_L_RED, CHI_M,
# CHI_M_RED, CHI_R, CHI_R_RED (mahjax/common/hand.py:266-298).
_CHI_FIRST_OFFSET = jnp.array((1, 1, -1, -1, -2, -2), dtype=jnp.int32)
_CHI_SECOND_OFFSET = jnp.array((2, 2, 1, 1, -1, -1), dtype=jnp.int32)
_CHI_RED_ACTION = jnp.array((False, True, False, True, False, True))


class _PlayerArrays(Protocol):
    hand_with_red: Array
    pon: Array
    river: Array
    discard_counts: Array
    riichi_declared: Array


class _RoundArrays(Protocol):
    target: Array
    last_player: Array
    last_draw: Array
    kan_declared: Array


class _MahJaxState(Protocol):
    current_player: Array
    legal_action_mask: Array
    players: _PlayerArrays
    round_state: _RoundArrays


def _tile_face(tile: Array) -> Array:
    """Face 0..33 for a MahJax tile type 0..36; ``-1`` stays absent."""
    red = (tile >= _TILE_TYPES) & (tile < _TILE_TYPES_WITH_RED)
    index = jnp.clip(tile - _TILE_TYPES, 0, _RED_FACE.shape[0] - 1)
    face = jnp.where(red, _RED_FACE[index], tile)
    return jnp.where(tile < 0, -1, face)


def _tile_is_red(tile: Array) -> Array:
    return (tile >= _TILE_TYPES) & (tile < _TILE_TYPES_WITH_RED)


def _is_five(face: Array) -> Array:
    return (face >= 0) & (face < _SUITED_FACES) & ((face % 9) == 4)


def _own_red(hand: Array, face: Array) -> Array:
    """Copies of the red five of ``face`` in the acting seat's hand."""
    red = jnp.where(
        _is_five(face),
        _TILE_TYPES + jnp.clip(face, 0, _SUITED_FACES - 1) // 9,
        -1,
    )
    return jnp.where(red >= 0, hand[jnp.clip(red, 0, _TILE_TYPES_WITH_RED - 1)], 0)


def _own_black(hand: Array, face: Array) -> Array:
    """Plain (non-red) copies of ``face`` in the acting seat's hand."""
    valid = (face >= 0) & (face < _TILE_TYPES)
    return jnp.where(valid, hand[jnp.clip(face, 0, _TILE_TYPES - 1)], 0)


def action_descriptors(state: _MahJaxState) -> Array:
    """Return int32 ``[87, ACTION_WIDTH]`` rows for one MahJax 4p state."""
    if state.legal_action_mask.shape != (MAHJAX_ACTION_COUNT,):
        raise ValueError(
            "action_descriptors expects a four-player MahJax state with "
            f"{MAHJAX_ACTION_COUNT} actions, got mask shape "
            f"{tuple(state.legal_action_mask.shape)}"
        )
    if state.players.hand_with_red.shape[-1] != _TILE_TYPES_WITH_RED:
        raise ValueError(
            "action_descriptors expects a 37-tile MahJax hand, got "
            f"{tuple(state.players.hand_with_red.shape)}"
        )

    action = jnp.arange(MAHJAX_ACTION_COUNT, dtype=jnp.int32)
    actor = state.current_player.astype(jnp.int32)
    hand = state.players.hand_with_red[actor].astype(jnp.int32)
    legal = state.legal_action_mask.astype(jnp.bool_)

    target = state.round_state.target.astype(jnp.int32)
    last_player = state.round_state.last_player.astype(jnp.int32)
    last_draw = state.round_state.last_draw.astype(jnp.int32)
    kan_declared = state.round_state.kan_declared.astype(jnp.bool_)
    # Only the declaration discard phase uses riichi_discard; once accepted,
    # the forced draw discard is the native tsumogiri candidate.
    riichi_discard_phase = state.players.riichi_declared[actor].astype(jnp.bool_)
    pon = state.players.pon[actor].astype(jnp.int32)

    # The offered discard is the last public river tile of the source seat
    # (mahjax/common/tile.py:100-115, 154-179). Native from_draw for a claim is
    # the discard event's kind: true only for a tsumogiri, and false for a
    # riichi declaration discard even when that discard was the drawn tile
    # (native/src/features/draw.rs:79-93 checks the observer's own draw, which
    # cannot be the discarder of another seat).
    source_seat = jnp.clip(last_player, 0, _NUM_PLAYERS - 1)
    discards = state.players.discard_counts[source_seat].astype(jnp.int32)
    river_index = jnp.clip(discards - 1, 0, state.players.river.shape[-1] - 1)
    discard = state.players.river[source_seat, river_index].astype(jnp.int32)
    discard_valid = (last_player >= 0) & (discards > 0) & (discard != _EMPTY_RIVER)
    offered_drawn = (
        discard_valid
        & ((discard & _RIVER_TSUMOGIRI_BIT) != 0)
        & ((discard & _RIVER_RIICHI_BIT) == 0)
    )

    hand_discard = action < _TILE_TYPES_WITH_RED
    tsumogiri = action == _ACTION_TSUMOGIRI
    self_kan = (action >= _TILE_TYPES_WITH_RED) & (action < _ACTION_TSUMOGIRI)
    kan_face = action - _TILE_TYPES_WITH_RED
    riichi = action == _ACTION_RIICHI
    tsumo = action == _ACTION_TSUMO
    ron = action == _ACTION_RON
    pon_action = (action == _ACTION_PON) | (action == _ACTION_PON_RED)
    open_kan = action == _ACTION_OPEN_KAN
    chi = (action >= _ACTION_CHI_FIRST) & (action <= _ACTION_CHI_LAST)
    pass_action = action == _ACTION_PASS
    kyuushu = action == _ACTION_KYUUSHU
    # Chi/pon/daiminkan/ron are the native contested kinds; a self kan is not.
    claim = pon_action | open_kan | chi | ron
    discard_family = hand_discard | tsumogiri

    # Kinds (native census ordinals). Action 86 (DUMMY) is the adapter's
    # round-end acknowledgement: native has no information-sharing action, so
    # it borrows the tile-less AcceptAbortiveDraw kind and carries no fields.
    has_pon = pon[jnp.clip(kan_face, 0, _TILE_TYPES - 1)] > 0
    kan_kind = jnp.where(has_pon, _KIND_KAKAN, KIND_ANKAN)
    discard_kind = jnp.where(
        riichi_discard_phase,
        KIND_RIICHI_DISCARD,
        jnp.where(tsumogiri, KIND_TSUMOGIRI, KIND_DISCARD),
    )
    kind = jnp.full_like(action, _KIND_ACCEPT_ABORTIVE_DRAW)
    for mask, value in (
        (kyuushu, _KIND_ABORT_NINE_TERMINALS),
        (pass_action, KIND_PASS),
        (chi, _KIND_CHI),
        (open_kan, _KIND_DAIMINKAN),
        (pon_action, _KIND_PON),
        (ron, _KIND_RON),
        (tsumo, _KIND_TSUMO),
        (riichi, KIND_RIICHI_DISCARD),
        (self_kan, kan_kind),
        (discard_family, discard_kind),
    ):
        kind = jnp.where(mask, value, kind)

    # Relative source in turn order (0 self, 1 next, 2 opposite, 3 previous).
    # Claims source from the discarder or kan declarer; everything else is
    # absent, which native writes as SOURCE_ABSENT, not -1.
    source = jnp.where(
        claim,
        jnp.mod(last_player - actor, _NUM_PLAYERS),
        SOURCE_ABSENT,
    ).astype(jnp.int32)

    # Primary tile: called tile for claims, own tile for discards/kakan,
    # winning tile for ron/tsumo, absent for ankan/pass/aborts/riichi.
    kakan = self_kan & (kind == _KIND_KAKAN)
    ankan = self_kan & (kind == KIND_ANKAN)
    primary = jnp.full_like(action, -1)
    for mask, value in (
        (hand_discard, action),
        (tsumogiri, last_draw),
        (kakan, kan_face),
        (claim, target),
        (tsumo, last_draw),
    ):
        primary = jnp.where(mask, value, primary)
    primary_face = _tile_face(primary)

    # A kakan adds the only remaining copy of the type: black when one is left,
    # red when the red copy is the last one (mahjax/common/logic.py:1210-1221).
    kakan_red = _is_five(kan_face) & (_own_red(hand, kan_face) > 0)
    kakan_red = kakan_red & (_own_black(hand, kan_face) == 0)
    primary_red = jnp.zeros(MAHJAX_ACTION_COUNT, dtype=jnp.bool_)
    for mask, value in (
        (hand_discard, _tile_is_red(action)),
        (tsumogiri, _tile_is_red(last_draw)),
        (kakan, kakan_red),
        (claim, _tile_is_red(target)),
        (tsumo, _tile_is_red(last_draw)),
    ):
        primary_red = jnp.where(mask, value, primary_red)

    # Actor-owned removals. The native writer sorts consumed physical tiles,
    # and a red five is copy 0 of its block (vendor/hydra2/crates/bridge/src/
    # rl/actions.rs:34), so a same-face red copy fills the earlier slot.
    target_face = _tile_face(target)
    target_red = _tile_is_red(target)
    chi_slot = jnp.clip(action - _ACTION_CHI_FIRST, 0, _CHI_FIRST_OFFSET.shape[0] - 1)
    chi_first = target_face + _CHI_FIRST_OFFSET[chi_slot]
    chi_second = target_face + _CHI_SECOND_OFFSET[chi_slot]
    chi_red_action = _CHI_RED_ACTION[chi_slot]
    chi_red_first = chi_red_action & _is_five(chi_first) & (_own_red(hand, chi_first) > 0)
    chi_red_second = chi_red_action & _is_five(chi_second) & (_own_red(hand, chi_second) > 0)
    # A red pon is one black plus the hand's red five; a plain pon is two black
    # copies (mahjax/common/hand.py:306-320).
    pon_red = (action == _ACTION_PON_RED) & _is_five(target_face)
    pon_red = pon_red & (_own_red(hand, target_face) > 0)
    # An open kan of a five keeps the red copy only when the called tile is
    # black; a called red five consumes three black copies
    # (mahjax/common/hand.py:322-342).
    open_kan_red = _is_five(target_face) & ~target_red & (_own_red(hand, target_face) > 0)
    # An ankan of a five always consumes the hand's red copy
    # (mahjax/common/hand.py:354-366).
    ankan_red = _is_five(kan_face) & (_own_red(hand, kan_face) > 0)

    removal0 = jnp.full_like(action, -1)
    for mask, value in (
        (hand_discard, _tile_face(action)),
        (tsumogiri, _tile_face(last_draw)),
        (self_kan, kan_face),
        (chi, chi_first),
        (pon_action | open_kan, target_face),
    ):
        removal0 = jnp.where(mask, value, removal0)
    removal1 = jnp.full_like(action, -1)
    for mask, value in (
        (ankan, kan_face),
        (chi, chi_second),
        (pon_action | open_kan, target_face),
    ):
        removal1 = jnp.where(mask, value, removal1)
    removal2 = jnp.full_like(action, -1)
    for mask, value in (
        (ankan, kan_face),
        (open_kan, target_face),
    ):
        removal2 = jnp.where(mask, value, removal2)
    removal3 = jnp.where(ankan, kan_face, -1)

    removal_red0 = jnp.zeros(MAHJAX_ACTION_COUNT, dtype=jnp.bool_)
    for mask, value in (
        (hand_discard, _tile_is_red(action)),
        (tsumogiri, _tile_is_red(last_draw)),
        (kakan, kakan_red),
        (ankan, ankan_red),
        (chi, chi_red_first),
        (pon_action, pon_red),
        (open_kan, open_kan_red),
    ):
        removal_red0 = jnp.where(mask, value, removal_red0)
    removal_red1 = jnp.where(chi, chi_red_second, False)
    removal_red2 = jnp.zeros(MAHJAX_ACTION_COUNT, dtype=jnp.bool_)
    removal_red3 = jnp.zeros(MAHJAX_ACTION_COUNT, dtype=jnp.bool_)

    # Native from_draw: true for tsumogiri; for riichi_discard true only when
    # the discarded tile is the observer's own current draw; for claims true
    # only when the offered discard is a plain tsumogiri (a riichi declaration
    # discard is excluded). A robbed added kan is never the observer's draw.
    claim_drawn = jnp.where(ron & kan_declared, False, offered_drawn)
    from_draw = jnp.where(tsumogiri, True, jnp.where(claim, claim_drawn, False))

    declares_riichi = kind == KIND_RIICHI_DISCARD
    claim_intent = (
        (kind == _KIND_CHI) | (kind == _KIND_PON) | (kind == _KIND_DAIMINKAN) | (kind == _KIND_RON)
    )

    removal_faces = jnp.stack((removal0, removal1, removal2, removal3), axis=-1)
    removal_reds = jnp.stack(
        (removal_red0, removal_red1, removal_red2, removal_red3), axis=-1
    ).astype(jnp.int32)
    row = jnp.full((MAHJAX_ACTION_COUNT, ACTION_WIDTH), -1, dtype=jnp.int32)
    row = row.at[:, ACTION_ID].set(jnp.where(legal, action, -1))
    row = row.at[:, ACTION_KIND].set(kind)
    row = row.at[:, ACTION_SOURCE].set(source)
    row = row.at[:, ACTION_PRIMARY_FACE].set(primary_face)
    row = row.at[:, ACTION_PRIMARY_RED].set(primary_red.astype(jnp.int32))
    row = row.at[:, ACTION_DRAWN_DISCARD].set(from_draw.astype(jnp.int32))
    row = row.at[:, ACTION_DECLARE_RIICHI].set(declares_riichi.astype(jnp.int32))
    faces = slice(ACTION_REMOVAL_FIRST, ACTION_REMOVAL_FIRST + ACTION_REMOVAL_COUNT)
    reds = slice(ACTION_REMOVAL_RED_FIRST, ACTION_REMOVAL_RED_FIRST + ACTION_REMOVAL_COUNT)
    row = row.at[:, faces].set(removal_faces)
    row = row.at[:, reds].set(removal_reds)
    row = row.at[:, ACTION_CLAIM_INTENT].set(claim_intent.astype(jnp.int32))
    return jnp.where(legal[:, None], row, -1).astype(jnp.int32)
