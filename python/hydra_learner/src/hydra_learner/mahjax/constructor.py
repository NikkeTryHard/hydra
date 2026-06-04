from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from hydra_learner.mahjax.compat import (
    MAHJAX_CHI_LEFT,
    MAHJAX_CHI_LEFT_RED,
    MAHJAX_CHI_MID,
    MAHJAX_CHI_MID_RED,
    MAHJAX_CHI_RIGHT,
    MAHJAX_CHI_RIGHT_RED,
    MAHJAX_OPEN_KAN,
    MAHJAX_PASS,
    MAHJAX_PON,
    MAHJAX_PON_RED,
    MAHJAX_RIICHI,
    MAHJAX_RON,
    MAHJAX_SELF_KAN_START,
    MAHJAX_TSUMO,
    MAHJAX_TSUMOGIRI,
)

if TYPE_CHECKING:
    from hydra_learner.typing_boundaries import JaxArray, JaxModule, MahjaxEnv, MahjaxState

_RED_TILE_STR_TO_ID = {"5mr": 34, "5pr": 35, "5sr": 36}
_HONOR_BASE = {"E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33}
_SUIT_OFFSETS = {"m": 0, "p": 9, "s": 18}


def _jnp() -> JaxModule:
    return importlib.import_module("jax.numpy")


def _jax() -> JaxModule:
    return importlib.import_module("jax")


def mjai_tile_to_mahjax_id(tile: str) -> int:
    if tile in _RED_TILE_STR_TO_ID:
        return _RED_TILE_STR_TO_ID[tile]
    if tile in _HONOR_BASE:
        return _HONOR_BASE[tile]
    if len(tile) != 2 or tile[1] not in _SUIT_OFFSETS:
        raise ValueError(f"unsupported MJAI tile: {tile}")
    number = int(tile[0])
    if number < 1 or number > 9:
        raise ValueError(f"unsupported MJAI tile: {tile}")
    return _SUIT_OFFSETS[tile[1]] + number - 1


def _tile_type(tile_id: int) -> int:
    if tile_id == 34:
        return 4
    if tile_id == 35:
        return 13
    if tile_id == 36:
        return 22
    return tile_id


def mahjax_action_from_mjai_dahai(state: MahjaxState, *, pai: str, tsumogiri: bool) -> int:
    """Translate an MJAI dahai event into the MahJAX action id for the current state."""
    tile_id = mjai_tile_to_mahjax_id(pai)
    legal_mask = state.legal_action_mask
    last_draw = int(state.round_state.last_draw)
    if tsumogiri:
        if tile_id != last_draw:
            raise ValueError("MJAI tsumogiri tile must match MahJAX last_draw")
        if not bool(legal_mask[MAHJAX_TSUMOGIRI]):
            raise ValueError("MahJAX TSUMOGIRI is not legal in current state")
        return MAHJAX_TSUMOGIRI
    if not bool(legal_mask[tile_id]):
        raise ValueError("MJAI tedashi tile is not legal in current MahJAX state")
    return tile_id


def mahjax_action_from_mjai_none(state: MahjaxState) -> int:
    """Translate an MJAI no-call response into MahJAX PASS."""
    if not bool(state.legal_action_mask[MAHJAX_PASS]):
        raise ValueError("MahJAX PASS is not legal in current state")
    return MAHJAX_PASS


def _require_legal_action(state: MahjaxState, action: int, message: str) -> int:
    if not bool(state.legal_action_mask[action]):
        raise ValueError(message)
    return action


def _contains_red_five(tiles: Sequence[str]) -> bool:
    return any(tile in _RED_TILE_STR_TO_ID for tile in tiles)


def _event_int(event: Mapping[str, object], key: str) -> int:
    value = event[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"MJAI event {key} must be an int")
    return value


def _event_consumed(event: Mapping[str, object]) -> Sequence[str]:
    value = event.get("consumed", ())
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError("MJAI event consumed must be a sequence")
    for item in value:
        if not isinstance(item, str):
            raise TypeError("MJAI event consumed entries must be strings")
    return value


def mahjax_action_from_mjai_pon(state: MahjaxState, *, pai: str, consumed: Sequence[str]) -> int:
    """Translate an MJAI pon event into the MahJAX PON/PON_RED action."""
    if mjai_tile_to_mahjax_id(pai) != int(state.round_state.target):
        raise ValueError("MJAI pon target must match MahJAX response target")
    action = MAHJAX_PON_RED if _contains_red_five(consumed) else MAHJAX_PON
    return _require_legal_action(state, action, "MahJAX PON action is not legal in current state")


def mahjax_action_from_mjai_kan(state: MahjaxState, *, pai: str, consumed: Sequence[str]) -> int:
    """Translate an MJAI daiminkan event into the MahJAX OPEN_KAN action."""
    if mjai_tile_to_mahjax_id(pai) != int(state.round_state.target):
        raise ValueError("MJAI kan target must match MahJAX response target")
    if len(consumed) != 3:
        raise ValueError("MJAI daiminkan event must consume exactly three tiles")
    return _require_legal_action(state, MAHJAX_OPEN_KAN, "MahJAX OPEN_KAN is not legal in current state")


def mahjax_action_from_mjai_chi(state: MahjaxState, *, pai: str, consumed: Sequence[str]) -> int:
    """Translate an MJAI chi event into the MahJAX CHI action variant."""
    target = mjai_tile_to_mahjax_id(pai)
    if target != int(state.round_state.target):
        raise ValueError("MJAI chi target must match MahJAX response target")
    tile_type = _tile_type(target)
    consumed_types = sorted(_tile_type(mjai_tile_to_mahjax_id(tile)) for tile in consumed)
    sequence = sorted([tile_type, *consumed_types])
    if len(sequence) != 3 or sequence[0] + 1 != sequence[1] or sequence[1] + 1 != sequence[2]:
        raise ValueError("MJAI chi tiles must form a sequence with the target")
    if sequence[0] // 9 != sequence[2] // 9 or sequence[2] >= 27:
        raise ValueError("MJAI chi sequence must be suited")
    chi_index = sequence.index(tile_type)
    red = _contains_red_five(consumed)
    action = (
        (MAHJAX_CHI_LEFT_RED if red else MAHJAX_CHI_LEFT)
        if chi_index == 0
        else (MAHJAX_CHI_MID_RED if red else MAHJAX_CHI_MID)
        if chi_index == 1
        else (MAHJAX_CHI_RIGHT_RED if red else MAHJAX_CHI_RIGHT)
    )
    return _require_legal_action(state, action, "MahJAX CHI action is not legal in current state")


def mahjax_action_from_mjai_self_kan(state: MahjaxState, *, pai: str | None, consumed: Sequence[str]) -> int:
    """Translate an MJAI ankan/kakan event into the MahJAX self-kan action."""
    tile = pai if pai is not None else next(iter(consumed), None)
    if tile is None:
        raise ValueError("MJAI self-kan event requires pai or consumed tiles")
    action = MAHJAX_SELF_KAN_START + _tile_type(mjai_tile_to_mahjax_id(tile))
    return _require_legal_action(state, action, "MahJAX self-kan action is not legal in current state")


def mahjax_action_from_mjai_reach(state: MahjaxState) -> int:
    """Translate an MJAI reach declaration into MahJAX RIICHI."""
    return _require_legal_action(state, MAHJAX_RIICHI, "MahJAX RIICHI is not legal in current state")


def mahjax_action_from_mjai_hora(state: MahjaxState, *, actor: int, target: int) -> int:
    """Translate an MJAI hora event into MahJAX TSUMO/RON."""
    action = MAHJAX_TSUMO if actor == target else MAHJAX_RON
    return _require_legal_action(state, action, "MahJAX win action is not legal in current state")


def mahjax_state_from_mjai_ryukyoku(state: MahjaxState) -> MahjaxState:
    """Accept an MJAI ryukyoku only after MahJAX has already ended the round."""
    if not bool(getattr(state.round_state, "terminated_round", False)):
        raise ValueError("MJAI ryukyoku requires an already terminated MahJAX round")
    return state


def mahjax_state_from_mjai_tsumo(state: MahjaxState, *, actor: int, pai: str) -> MahjaxState:
    """Validate an MJAI tsumo event against MahJAX's already-advanced draw state."""
    tile_id = mjai_tile_to_mahjax_id(pai)
    if int(state.current_player) != actor:
        raise ValueError("MJAI tsumo actor must match MahJAX current_player")
    if int(state.round_state.last_draw) != tile_id:
        raise ValueError("MJAI tsumo tile must match MahJAX last_draw")
    return state


def mahjax_action_from_mjai_event(state: MahjaxState, event: Mapping[str, object]) -> int | None:
    """Translate an MJAI event to a MahJAX action, or validate no-action events."""
    event_type = event.get("type")
    if event_type == "dahai":
        if "pai" not in event or "tsumogiri" not in event:
            raise ValueError("MJAI dahai event requires pai and tsumogiri")
        return mahjax_action_from_mjai_dahai(state, pai=str(event["pai"]), tsumogiri=bool(event["tsumogiri"]))
    if event_type == "pon":
        return mahjax_action_from_mjai_pon(state, pai=str(event["pai"]), consumed=_event_consumed(event))
    if event_type == "chi":
        return mahjax_action_from_mjai_chi(state, pai=str(event["pai"]), consumed=_event_consumed(event))
    if event_type in {"kan", "daiminkan"}:
        return mahjax_action_from_mjai_kan(state, pai=str(event["pai"]), consumed=_event_consumed(event))
    if event_type == "ankan":
        return mahjax_action_from_mjai_self_kan(state, pai=None, consumed=_event_consumed(event))
    if event_type == "kakan":
        return mahjax_action_from_mjai_self_kan(state, pai=str(event["pai"]), consumed=())
    if event_type == "reach":
        return mahjax_action_from_mjai_reach(state)
    if event_type == "reach_accepted":
        return None
    if event_type == "hora":
        return mahjax_action_from_mjai_hora(state, actor=_event_int(event, "actor"), target=_event_int(event, "target"))
    if event_type == "tsumo":
        mahjax_state_from_mjai_tsumo(state, actor=_event_int(event, "actor"), pai=str(event["pai"]))
        return None
    if event_type == "none":
        return mahjax_action_from_mjai_none(state)
    if event_type == "ryukyoku":
        mahjax_state_from_mjai_ryukyoku(state)
        return None
    raise ValueError(f"unsupported MJAI event for MahJAX parity harness: {event_type}")


def apply_mjai_event_to_mahjax_state(env: MahjaxEnv, state: MahjaxState, event: Mapping[str, object]) -> MahjaxState:
    """Apply the MJAI events currently supported by the MahJAX parity harness."""
    action = mahjax_action_from_mjai_event(state, event)
    if action is None:
        return state
    return env.step(state, _jnp().asarray(action, dtype=_jnp().int32))


def _counts37(tile_ids: Sequence[int]) -> JaxArray:
    jnp = _jnp()
    ids = jnp.asarray(tile_ids, dtype=jnp.int32)
    return jnp.bincount(ids, length=37).astype(jnp.int8)


def _counts34_from_37(counts37: JaxArray) -> JaxArray:
    jnp = _jnp()
    return (
        jnp.asarray(counts37, dtype=jnp.int8)[:34]
        .at[4]
        .add(counts37[34])
        .at[13]
        .add(counts37[35])
        .at[22]
        .add(counts37[36])
    )


def _hand_ids(tile_ids: Sequence[int]) -> JaxArray:
    jnp = _jnp()
    arr = jnp.full((14,), -1, dtype=jnp.int16)
    ids = jnp.asarray(tile_ids, dtype=jnp.int16)
    return arr.at[: ids.shape[0]].set(ids)


def _deck_with_required_tiles(draw_tile: int, dora_marker: int) -> JaxArray:
    jnp = _jnp()
    return (
        jnp.zeros((136,), dtype=jnp.int8)
        .at[83]
        .set(jnp.asarray(draw_tile, dtype=jnp.int8))
        .at[9]
        .set(jnp.asarray(dora_marker, dtype=jnp.int8))
        .at[8]
        .set(jnp.asarray(dora_marker, dtype=jnp.int8))
    )


def mahjax_state_from_start_kyoku(
    *,
    tehais: Sequence[Sequence[str]],
    scores: Sequence[int],
    dora_marker: str,
    oya: int,
    kyoku: int,
    honba: int,
    kyotaku: int,
    first_draw: str | None = None,
) -> MahjaxState:
    if len(tehais) != 4:
        raise ValueError("MahJAX constructor currently supports 4 players only")
    if len(scores) != 4:
        raise ValueError("scores must have length 4")
    if oya < 0 or oya >= 4:
        raise ValueError("oya must be 0..3")
    jnp = _jnp()
    mahjax = importlib.import_module("mahjax")
    env = mahjax.make("red_mahjong", observe_type="dict")
    env_mod = importlib.import_module("mahjax.red_mahjong.env")
    shanten_mod = importlib.import_module("mahjax.red_mahjong.shanten")
    hand_mod = importlib.import_module("mahjax.red_mahjong.hand")
    state = env.init(_jax().random.PRNGKey(0))
    hand_ids = []
    hand_with_red = []
    hands34 = []
    hand_counts = []
    for hand in tehais:
        if len(hand) != 13:
            raise ValueError("each start_kyoku hand must contain 13 tiles")
        ids = [mjai_tile_to_mahjax_id(tile) for tile in hand]
        counts37 = _counts37(ids)
        hand_ids.append(_hand_ids(ids))
        hand_with_red.append(counts37)
        hands34.append(_counts34_from_37(counts37))
        hand_counts.append(len(ids))
    can_ron = env_mod.v_can_win(jnp.stack(hands34), env_mod.TILE_RANGE)

    draw_id = mjai_tile_to_mahjax_id(first_draw) if first_draw is not None else mjai_tile_to_mahjax_id(tehais[oya][0])
    dora_id = mjai_tile_to_mahjax_id(dora_marker)
    players = state.players.replace(
        hand=jnp.stack(hands34),
        hand_with_red=jnp.stack(hand_with_red),
        hand_ids=jnp.stack(hand_ids),
        hand_counts=jnp.asarray(hand_counts, dtype=jnp.int8),
        drawn_tile=jnp.full((4,), -1, dtype=jnp.int16),
    )
    round_state = state.round_state.replace(
        action_history=jnp.full((3, 200), -1, dtype=jnp.int8),
        round=jnp.asarray(kyoku - 1, dtype=jnp.int8),
        honba=jnp.asarray(honba, dtype=jnp.int8),
        kyotaku=jnp.asarray(kyotaku, dtype=jnp.int8),
        dealer=jnp.asarray(oya, dtype=jnp.int8),
        score=jnp.asarray([score // 100 for score in scores], dtype=jnp.int32),
        deck=_deck_with_required_tiles(draw_id, dora_id),
        next_deck_ix=jnp.asarray(83, dtype=jnp.int32),
        last_deck_ix=jnp.asarray(14, dtype=jnp.int8),
        dora_indicators=jnp.asarray([dora_id, -1, -1, -1, -1], dtype=jnp.int8),
        ura_dora_indicators=jnp.asarray([dora_id, -1, -1, -1, -1], dtype=jnp.int8),
        last_draw=jnp.asarray(-1, dtype=jnp.int8),
        last_player=jnp.asarray(-1, dtype=jnp.int8),
        shanten_current_player=shanten_mod.Shanten.number(hands34[oya]).astype(jnp.int8),
    )
    state = state.replace(
        current_player=jnp.asarray(oya, dtype=jnp.int8),
        players=players,
        round_state=round_state,
        step_count=jnp.asarray(0, dtype=jnp.int32),
        rewards=jnp.zeros((4,), dtype=jnp.float32),
        terminated=jnp.asarray(False),
        truncated=jnp.asarray(False),
    )
    hand_with_draw = players.hand_with_red.at[oya].set(hand_mod.Hand.add(players.hand_with_red[oya], draw_id))
    hand34_with_draw = hand_mod.Hand.to_34(hand_with_draw[oya])
    draw_eval_state = env_mod._replace_state(
        state, last_draw=jnp.asarray(draw_id, dtype=jnp.int8), is_haitei=jnp.asarray(False)
    )
    _, yakuman_num, _ = env_mod.Yaku.judge_yakuman(
        players.hand_with_red[oya],
        env_mod.FALSE,
        jnp.asarray(oya, dtype=jnp.int8),
        draw_eval_state,
    )
    _, draw_fan, draw_fu = env_mod.Yaku.judge(
        hand_with_draw[oya],
        env_mod.FALSE,
        jnp.asarray(oya, dtype=jnp.int8),
        draw_eval_state.replace(players=players.replace(hand_with_red=hand_with_draw)),
    )
    has_draw_yaku = (yakuman_num > 0) | (draw_fan > 0)
    draw_fan = jnp.where(yakuman_num > 0, yakuman_num, draw_fan)
    draw_fu = jnp.where(yakuman_num > 0, 0, draw_fu)
    players = players.replace(
        hand=players.hand.at[oya].set(hand34_with_draw),
        hand_with_red=hand_with_draw,
        hand_ids=players.hand_ids.at[oya, 13].set(jnp.asarray(draw_id, dtype=jnp.int16)),
        hand_counts=players.hand_counts.at[oya].set(jnp.asarray(14, dtype=jnp.int8)),
        can_win=can_ron,
        has_yaku=players.has_yaku.at[oya, 0]
        .set(can_ron[oya, hand_mod.Tile.to_tile_type(draw_id)])
        .at[oya, 1]
        .set(has_draw_yaku),
        fan=players.fan.at[oya, 0]
        .set(jnp.asarray(yakuman_num, dtype=jnp.int32))
        .at[oya, 1]
        .set(jnp.asarray(draw_fan, dtype=jnp.int32)),
        fu=players.fu.at[oya, 0]
        .set(jnp.asarray(0, dtype=jnp.int32))
        .at[oya, 1]
        .set(jnp.asarray(draw_fu, dtype=jnp.int32)),
    )
    mask_state = draw_eval_state.replace(players=players)
    legal_action_mask = env_mod._make_legal_action_mask_after_draw(
        mask_state,
        hand_with_draw,
        jnp.asarray(oya, dtype=jnp.int8),
        jnp.asarray(draw_id, dtype=jnp.int8),
    )
    players = players.replace(legal_action_mask=players.legal_action_mask.at[oya].set(legal_action_mask))
    round_state = round_state.replace(
        next_deck_ix=jnp.asarray(82, dtype=jnp.int32),
        last_draw=jnp.asarray(draw_id, dtype=jnp.int8),
        target=jnp.asarray(-1, dtype=jnp.int8),
        shanten_current_player=shanten_mod.Shanten.number(hand34_with_draw).astype(jnp.int8),
    )
    return state.replace(
        players=players,
        round_state=round_state,
        legal_action_mask=legal_action_mask,
    )
