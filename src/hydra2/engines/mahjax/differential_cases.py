"""WP-04C MahJax differential — scenario registry.

Pure-move part of :mod:`hydra2.engines.mahjax.differential`; import from
that path. Covers the script shortcuts and the four differential cases with
convergent indicator types.
"""

from __future__ import annotations

from hydra2.conformance.runner import ScriptedDecision
from hydra2.engines.mahjax.differential_projection import Scenario

__all__ = [
    "SCENARIO_REGISTRY",
]


# Helper to create ScriptedDecision shortcuts
def _do(kind: str, tile: int | None = None) -> ScriptedDecision:
    return ScriptedDecision(kind, tile=tile)


def _neg(kind: str, tile: int | None = None) -> ScriptedDecision:
    return ScriptedDecision(kind, tile=tile, negate=True)


# WP04C-01: fifth dora / kan dora reveal (single ankan, checks dora_indicator_slots)
_SCENARIO_01 = Scenario(
    case_id="WP04C-01-fifth-dora",
    title="fifth dora indicator + kan-dora reveal (single ankan, convergent indicators)",
    rule_fields=(
        "kan_dora_reveal_policy",
        "dora_indicator_slots",
        "live_draw_order",
        "rinshan_draw_order",
        "shanten_parity",
    ),
    evidence=(
        "tenhou.net/man YAKU L1246 kan dora",
        "WP04A-01 geometry adapted with convergent indicator types",
        "wall translation live 52+k -> 83-k, dora 131-2k -> 9-2k",
    ),
    hands={
        0: {
            88: 1,
            89: 1,
            90: 1,
            40: 1,
            41: 1,
            42: 1,
            116: 1,
            117: 1,
            118: 1,
            67: 1,
            68: 1,
            69: 1,
            0: 1,
        },
        1: {
            72: 1,
            76: 1,
            80: 1,
            84: 1,
            44: 1,
            45: 1,
            46: 1,
            47: 1,
            48: 1,
            49: 1,
            50: 1,
            51: 1,
            85: 1,
        },
        2: {
            52: 1,
            53: 1,
            54: 1,
            55: 1,
            60: 1,
            61: 1,
            62: 1,
            63: 1,
            100: 1,
            101: 1,
            102: 1,
            103: 1,
            107: 1,
        },
        3: {4: 1, 8: 1, 12: 1, 16: 1, 20: 1, 24: 1, 28: 1, 32: 1, 36: 1, 70: 1, 71: 1, 64: 1, 1: 1},
    },
    live_draws={
        52: 91,
        53: 108,
        54: 112,
        55: 120,
        56: 119,
        57: 124,
        58: 125,
        59: 126,
        60: 128,
        61: 130,
        62: 133,
        63: 134,
    },
    dead_wall={131: 2, 130: 3, 129: 6, 128: 7, 127: 10, 126: 11, 135: 43},
    script=(
        _do("ankan"),
        _do("auto"),
        _do("auto"),
        _do("auto"),
        _do("auto"),
    ),
)

# WP04C-02: chankan (kakan + ron window)
_SCENARIO_02 = Scenario(
    case_id="WP04C-02-chankan",
    title="chankan window via kakan (WP04A-02 ryanmen → kakan → ron)",
    rule_fields=(
        "chankan_window",
        "rinshan_draw_order",
        "discard_legality_projection",
        "win_offer_flags",
        "shanten_parity",
    ),
    evidence=(
        "tenhou.net/man YAKU L1177 chankan",
        "WP04A-02 geometry adapted with convergent dora indicators",
        "kakan opens ron window for waiting riichi player",
    ),
    hands={
        3: {
            0: 1,
            4: 1,
            8: 1,
            12: 1,
            16: 1,
            20: 1,
            60: 1,
            64: 1,
            68: 1,
            92: 1,
            96: 1,
            120: 1,
            121: 1,
        },
        1: {
            88: 1,
            89: 1,
            36: 1,
            37: 1,
            38: 1,
            39: 1,
            45: 1,
            48: 1,
            49: 1,
            50: 1,
            51: 1,
            110: 1,
            119: 1,
        },
        0: {
            108: 1,
            109: 1,
            112: 1,
            113: 1,
            116: 1,
            117: 1,
            122: 1,
            124: 1,
            125: 1,
            128: 1,
            129: 1,
            132: 1,
            133: 1,
        },
        2: {
            126: 1,
            127: 1,
            130: 1,
            131: 1,
            134: 1,
            135: 1,
            114: 1,
            115: 1,
            118: 1,
            111: 1,
            28: 1,
            32: 1,
            24: 1,
        },
    },
    live_draws={
        52: 13,
        53: 14,
        54: 90,
        55: 29,
        56: 30,
        57: 31,
        58: 33,
        59: 34,
        60: 105,
        61: 21,
        62: 17,
        63: 25,
        64: 26,
        65: 27,
        66: 91,
    },
    dead_wall={131: 2, 129: 6, 127: 10},
    script=(
        _do("tsumogiri", 13),
        _do("tsumogiri", 14),
        _do("tsumogiri", 90),
        _do("pon"),
        _do("pass"),
        _do("discard", 45),
        _do("tsumogiri", 29),
        _do("tsumogiri", 30),
        _do("tsumogiri", 31),
        _do("tsumogiri", 33),
        _do("pass"),
        _do("tsumogiri", 34),
        _do("riichi_discard", 105),
        _do("tsumogiri", 21),
        _do("tsumogiri", 17),
        _do("tsumogiri", 25),
        _do("tsumogiri", 26),
        _do("tsumogiri", 27),
        _do("kakan"),
    ),
)

# WP04C-03: kuikae (post-pon same-type swap barred)
_SCENARIO_03 = Scenario(
    case_id="WP04C-03-kuikae",
    title="kuikae post-pon same-meld swap barred (WP04A-03)",
    rule_fields=(
        "kuikae_policy_forbidden",
        "discard_legality_projection",
        "shanten_parity",
    ),
    evidence=(
        "manifest kuikae_policy=forbidden 2007-11-29",
        "WP04A-03 geometry seat1 holds 5p {53,54,55}, dealer tedashi 52",
        "pon consumes {53,54}, leaving 55 as barred discard",
    ),
    hands={
        0: {52: 1, 4: 1, 5: 1, 8: 1, 9: 1, 12: 1, 13: 1, 16: 1, 20: 1, 21: 1, 24: 1, 28: 1, 32: 1},
        1: {
            53: 1,
            54: 1,
            55: 1,
            72: 1,
            73: 1,
            76: 1,
            77: 1,
            84: 1,
            85: 1,
            88: 1,
            89: 1,
            96: 1,
            97: 1,
        },
        2: {
            108: 1,
            109: 1,
            112: 1,
            113: 1,
            116: 1,
            117: 1,
            120: 1,
            121: 1,
            124: 1,
            125: 1,
            128: 1,
            129: 1,
            132: 1,
        },
        3: {
            110: 1,
            111: 1,
            114: 1,
            115: 1,
            118: 1,
            119: 1,
            122: 1,
            123: 1,
            126: 1,
            127: 1,
            130: 1,
            131: 1,
            134: 1,
        },
    },
    live_draws={52: 6, 53: 40, 54: 44, 55: 48, 56: 36, 57: 60, 58: 64},
    dead_wall={131: 2, 129: 10},
    script=(
        _do("discard", 52),
        _do("pon", 52),
        _neg("discard", 55),
        _do("auto"),
    ),
)

# WP04C-04: shanten parity (tenpai progression)
_SCENARIO_04 = Scenario(
    case_id="WP04C-04-shanten-parity",
    title="shanten parity across discards and melds (ordinary + open)",
    rule_fields=(
        "shanten_parity",
        "discard_legality_projection",
        "live_draw_order",
    ),
    evidence=(
        "mahjax Shanten.number vs reference hand-derived shanten",
        "tenpai progression via tsumogiri and pon",
    ),
    hands={
        0: {0: 1, 1: 1, 4: 1, 8: 1, 12: 1, 16: 1, 20: 1, 24: 1, 28: 1, 32: 1, 36: 1, 40: 1, 44: 1},
        1: {2: 1, 3: 1, 5: 1, 6: 1, 9: 1, 10: 1, 13: 1, 14: 1, 17: 1, 18: 1, 21: 1, 22: 1, 25: 1},
        2: {
            26: 1,
            27: 1,
            29: 1,
            30: 1,
            33: 1,
            34: 1,
            37: 1,
            38: 1,
            41: 1,
            42: 1,
            45: 1,
            46: 1,
            48: 1,
        },
        3: {
            49: 1,
            50: 1,
            52: 1,
            53: 1,
            56: 1,
            57: 1,
            60: 1,
            61: 1,
            64: 1,
            65: 1,
            68: 1,
            69: 1,
            72: 1,
        },
    },
    live_draws={
        52: 73,
        53: 74,
        54: 75,
        55: 76,
        56: 77,
        57: 78,
        58: 79,
        59: 80,
        60: 81,
        61: 82,
        62: 83,
        63: 84,
    },
    dead_wall={131: 62, 129: 66},
    script=(
        _do("auto"),
        _do("auto"),
        _do("auto"),
        _do("auto"),
        _do("auto"),
        _do("auto"),
        _do("auto"),
        _do("auto"),
    ),
)

SCENARIO_REGISTRY: tuple[Scenario, ...] = (
    _SCENARIO_01,
    _SCENARIO_02,
    _SCENARIO_03,
    _SCENARIO_04,
)

# Map for quick lookup
_SCENARIO_BY_ID: dict[str, Scenario] = {s.case_id: s for s in SCENARIO_REGISTRY}
