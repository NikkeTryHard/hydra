from __future__ import annotations

from collections.abc import Sequence
from typing import Final

HYDRA_ACTION_SPACE: Final = 46
MAHJAX_RED_ACTION_SPACE: Final = 87

HYDRA_AKA_5M: Final = 34
HYDRA_AKA_5P: Final = 35
HYDRA_AKA_5S: Final = 36
HYDRA_RIICHI: Final = 37
HYDRA_CHI_LEFT: Final = 38
HYDRA_CHI_MID: Final = 39
HYDRA_CHI_RIGHT: Final = 40
HYDRA_PON: Final = 41
HYDRA_KAN: Final = 42
HYDRA_AGARI: Final = 43
HYDRA_RYUUKYOKU: Final = 44
HYDRA_PASS: Final = 45

MAHJAX_SELF_KAN_START: Final = 37
MAHJAX_SELF_KAN_END: Final = 70
MAHJAX_TSUMOGIRI: Final = 71
MAHJAX_RIICHI: Final = 72
MAHJAX_TSUMO: Final = 73
MAHJAX_RON: Final = 74
MAHJAX_PON: Final = 75
MAHJAX_PON_RED: Final = 76
MAHJAX_OPEN_KAN: Final = 77
MAHJAX_CHI_LEFT: Final = 78
MAHJAX_CHI_LEFT_RED: Final = 79
MAHJAX_CHI_MID: Final = 80
MAHJAX_CHI_MID_RED: Final = 81
MAHJAX_CHI_RIGHT: Final = 82
MAHJAX_CHI_RIGHT_RED: Final = 83
MAHJAX_PASS: Final = 84
MAHJAX_KYUUSHU: Final = 85
MAHJAX_DUMMY: Final = 86

_RED_FIVE_TO_BASE: Final = {HYDRA_AKA_5M: 4, HYDRA_AKA_5P: 13, HYDRA_AKA_5S: 22}
_CHI_TO_HYDRA: Final = {
    MAHJAX_CHI_LEFT: HYDRA_CHI_LEFT,
    MAHJAX_CHI_LEFT_RED: HYDRA_CHI_LEFT,
    MAHJAX_CHI_MID: HYDRA_CHI_MID,
    MAHJAX_CHI_MID_RED: HYDRA_CHI_MID,
    MAHJAX_CHI_RIGHT: HYDRA_CHI_RIGHT,
    MAHJAX_CHI_RIGHT_RED: HYDRA_CHI_RIGHT,
}


def _check_mahjax_action(action: int) -> None:
    if action < 0 or action >= MAHJAX_RED_ACTION_SPACE:
        raise ValueError(f"MahJAX action id out of range: {action}")


def _check_hydra_action(action: int) -> None:
    if action < 0 or action >= HYDRA_ACTION_SPACE:
        raise ValueError(f"Hydra action id out of range: {action}")


def _check_mahjax_mask(mask: Sequence[object]) -> None:
    if len(mask) != MAHJAX_RED_ACTION_SPACE:
        raise ValueError(f"MahJAX legal mask width must be {MAHJAX_RED_ACTION_SPACE}, got {len(mask)}")


def _is_legal(mask: Sequence[object] | None, action: int) -> bool:
    return mask is None or bool(mask[action])


def tile_to_hydra_discard(tile: int) -> int:
    """Map MahJAX local tile id 0..36 to Hydra discard id 0..36."""
    if tile < 0 or tile > HYDRA_AKA_5S:
        raise ValueError(f"MahJAX tile id out of range for Hydra discard: {tile}")
    return tile


def hydra_discard_base_tile(action: int) -> int:
    """Return 34-tile base type for a Hydra discard id."""
    _check_hydra_action(action)
    if action <= 33:
        return action
    if action in _RED_FIVE_TO_BASE:
        return _RED_FIVE_TO_BASE[action]
    raise ValueError(f"Hydra action is not a discard: {action}")


def mahjax_action_to_hydra(action: int, *, last_draw: int | None = None) -> int | None:
    """Project a MahJAX red action id into Hydra's compact 46-action facade.

    `None` means MahJAX `DUMMY`, which is a control-plane action and has no Hydra
    policy action. MahJAX `TSUMOGIRI` needs `last_draw` because Hydra encodes the
    discarded tile, not the tsumogiri flag.
    """
    _check_mahjax_action(action)
    if action <= HYDRA_AKA_5S:
        return action
    if MAHJAX_SELF_KAN_START <= action <= MAHJAX_SELF_KAN_END:
        return HYDRA_KAN
    if action == MAHJAX_TSUMOGIRI:
        if last_draw is None:
            raise ValueError("MahJAX TSUMOGIRI requires last_draw to project to Hydra discard")
        return tile_to_hydra_discard(last_draw)
    if action == MAHJAX_RIICHI:
        return HYDRA_RIICHI
    if action in (MAHJAX_TSUMO, MAHJAX_RON):
        return HYDRA_AGARI
    if action in (MAHJAX_PON, MAHJAX_PON_RED):
        return HYDRA_PON
    if action == MAHJAX_OPEN_KAN:
        return HYDRA_KAN
    if action in _CHI_TO_HYDRA:
        return _CHI_TO_HYDRA[action]
    if action == MAHJAX_PASS:
        return HYDRA_PASS
    if action == MAHJAX_KYUUSHU:
        return HYDRA_RYUUKYOKU
    if action == MAHJAX_DUMMY:
        return None
    raise AssertionError("unreachable MahJAX action")


def mahjax_mask_to_hydra(mask: Sequence[object], *, last_draw: int | None = None) -> list[bool]:
    """OR-project a MahJAX red legal mask to Hydra width 46.

    Collapsed groups preserve Hydra policy ABI: self/open kan -> KAN, tsumo/ron ->
    AGARI, red/non-red pon and chi variants -> compact call ids. DUMMY is ignored.
    """
    _check_mahjax_mask(mask)
    hydra = [False] * HYDRA_ACTION_SPACE
    for action, legal in enumerate(mask):
        if not bool(legal):
            continue
        projected = mahjax_action_to_hydra(action, last_draw=last_draw)
        if projected is not None:
            hydra[projected] = True
    return hydra


def hydra_action_to_mahjax(
    action: int,
    *,
    legal_mask: Sequence[object] | None = None,
    last_draw: int | None = None,
    kan_tile_type: int | None = None,
    prefer_red_call: bool = False,
    response_phase: bool = False,
) -> int:
    """Choose a MahJAX red action for a compact Hydra action.

    The reverse direction is context-dependent because Hydra collapses kan/agari
    and red/no-red call variants. `legal_mask`, `last_draw`, and optional sidecars
    make the choice explicit and deterministic.
    """
    _check_hydra_action(action)
    if legal_mask is not None:
        _check_mahjax_mask(legal_mask)

    if action <= HYDRA_AKA_5S:
        if (
            last_draw is not None
            and action == tile_to_hydra_discard(last_draw)
            and _is_legal(legal_mask, MAHJAX_TSUMOGIRI)
        ):
            return MAHJAX_TSUMOGIRI
        if not _is_legal(legal_mask, action):
            raise ValueError(f"Hydra discard {action} is not legal in MahJAX mask")
        return action
    if action == HYDRA_RIICHI:
        if not _is_legal(legal_mask, MAHJAX_RIICHI):
            raise ValueError("Hydra RIICHI is not legal in MahJAX mask")
        return MAHJAX_RIICHI
    if action == HYDRA_CHI_LEFT:
        return _choose_pair(legal_mask, MAHJAX_CHI_LEFT_RED, MAHJAX_CHI_LEFT, prefer_first=prefer_red_call)
    if action == HYDRA_CHI_MID:
        return _choose_pair(legal_mask, MAHJAX_CHI_MID_RED, MAHJAX_CHI_MID, prefer_first=prefer_red_call)
    if action == HYDRA_CHI_RIGHT:
        return _choose_pair(legal_mask, MAHJAX_CHI_RIGHT_RED, MAHJAX_CHI_RIGHT, prefer_first=prefer_red_call)
    if action == HYDRA_PON:
        return _choose_pair(legal_mask, MAHJAX_PON_RED, MAHJAX_PON, prefer_first=prefer_red_call)
    if action == HYDRA_KAN:
        if _is_legal(legal_mask, MAHJAX_OPEN_KAN):
            return MAHJAX_OPEN_KAN
        if kan_tile_type is not None:
            if kan_tile_type < 0 or kan_tile_type > 33:
                raise ValueError(f"kan_tile_type must be 0..33, got {kan_tile_type}")
            candidate = MAHJAX_SELF_KAN_START + kan_tile_type
            if _is_legal(legal_mask, candidate):
                return candidate
        if legal_mask is not None:
            for candidate in range(MAHJAX_SELF_KAN_START, MAHJAX_SELF_KAN_END + 1):
                if bool(legal_mask[candidate]):
                    return candidate
        raise ValueError("Hydra KAN is not legal in MahJAX mask")
    if action == HYDRA_AGARI:
        first, second = (MAHJAX_RON, MAHJAX_TSUMO) if response_phase else (MAHJAX_TSUMO, MAHJAX_RON)
        return _choose_pair(legal_mask, first, second, prefer_first=True)
    if action == HYDRA_RYUUKYOKU:
        if not _is_legal(legal_mask, MAHJAX_KYUUSHU):
            raise ValueError("Hydra RYUUKYOKU is not legal in MahJAX mask")
        return MAHJAX_KYUUSHU
    if action == HYDRA_PASS:
        if not _is_legal(legal_mask, MAHJAX_PASS):
            raise ValueError("Hydra PASS is not legal in MahJAX mask")
        return MAHJAX_PASS
    raise AssertionError("unreachable Hydra action")


def _choose_pair(mask: Sequence[object] | None, first: int, second: int, *, prefer_first: bool) -> int:
    choices = (first, second) if prefer_first else (second, first)
    for action in choices:
        if _is_legal(mask, action):
            return action
    raise ValueError(f"Neither MahJAX action {first} nor {second} is legal")
