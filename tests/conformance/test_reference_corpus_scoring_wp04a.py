"""WP-04A reference corpus: scoring and terminal settlements (cases 07-10).

Red-tile scoring, pao liability with kazoe, and the nine-terminal and
exhaustive-draw settlements replayed through the WP-03A reference adapter.
"""

from __future__ import annotations

import pytest

from hydra2.conformance.runner import expect_predicate
from tests.conformance.test_reference_corpus_wp04a import (
    _do,
    _expect_abortive,
    _expect_scores_delta,
    _first_event,
    _run,
    assert_supported,
)

pytestmark = pytest.mark.contract_package("WP-04A")


def test_wp04a_07_red_five_scoring() -> None:
    """s3 (child) holds red 5m (16) + red 5p (52) concealed, pons red 5s (88)
    with two normal copies, completes 456p with dealer's discarded 6p.
    Yaku/dora/fu derivation (Tenhou tables):
      yaku   yakuhai (pon haku)                    = 1 han
      dora   aka x3: 16 red5m + 52 red5p concealed
             + 88 red5s inside the open triplet     = 3 han
             indicator slot 131 pinned to F (128) -> dora = C, held by nobody
      total  4 han
      fu     20 open-ron base + 2 open simple triplet (sou5 pon)
             + 4 open honor triplet (haku pon) + 0 ryanmen wait
             + 0 guest-wind EE pair = 26 -> rounded up 30 fu
      points basic 30 * 2^(2+4) = 1920 (< mangan cap); child ron = 1920 * 4
             = 7680 -> rounded up 7700; dealer discards => dealer pays all.
    Expected hora delta [-7700, 0, 0, +7700]."""
    hands = {
        0: {
            125: 1,
            0: 1,
            4: 1,
            8: 1,
            32: 1,
            36: 1,
            40: 1,
            44: 1,
            68: 1,
            76: 1,
            104: 1,
            112: 1,
            116: 1,
        },
        1: {
            89: 1,
            1: 1,
            5: 1,
            9: 1,
            33: 1,
            37: 1,
            41: 1,
            45: 1,
            69: 1,
            74: 1,
            80: 1,
            84: 1,
            100: 1,
        },
        2: {
            96: 1,
            97: 1,
            101: 1,
            21: 1,
            25: 1,
            29: 1,
            57: 1,
            61: 1,
            65: 1,
            72: 1,
            73: 1,
            120: 1,
            117: 1,
        },
        3: {
            88: 1,
            90: 1,
            124: 1,
            126: 1,
            12: 1,
            16: 1,
            20: 1,
            48: 1,
            52: 1,
            108: 1,
            109: 1,
            77: 1,
            105: 1,
        },
    }
    live = {52: 24, 53: 28, 54: 64, 55: 56}
    dead_wall = {131: 128}
    script = (
        _do("discard", 125),  # s0 sheds the third haku copy...
        _do("pon"),  # ...s3 calls haku (meld 1/2)
        _do("discard", 105),
        _do("tsumogiri", 28),
        _do("discard", 89),  # s1 sheds a normal sou5 copy...
        _do("pon"),  # ...s3 calls sou5 WITH the red copy 88 (meld 2/2)
        _do("discard", 77),
        _do("discard", 56),  # s0 draws live55=56 (6p) and deals in
        _do("ron"),
    )

    def winning_tile_is_scenario_six_pin(sim) -> str | None:
        ron = _first_event(sim, "ron")
        if ron is None:
            return "no ron event"
        if int(ron.payload.tile) != 56:
            return f"winning tile {ron.payload.tile} != scenario 6p (56)"
        return None

    result = _run(
        "WP04A-07",
        "red five scoring: aka dora x3 incl. ponned red sou5",
        ("red_tile_ids", "kuitan"),
        (
            "tenhou.net/man RULE red_tile_ids [16,52,88]; each aka five is a "
            "permanent dora whether concealed or inside a called meld",
            "computation cited in docstring: yakuhai 1 + aka 3 = 4 han; "
            "26 fu -> 30 fu; basic 1920; child ron 7680 -> 7700",
            "wall discipline: consumed live indices 52-55 all pinned; "
            "per-type copies <=4 across hands+pins",
        ),
        hands=hands,
        live_draws=live,
        script=script,
        expectations=[
            _expect_scores_delta((-7700, 0, 0, 7700)),
            expect_predicate("winning tile is scenario 6p", winning_tile_is_scenario_six_pin),
        ],
        dead_wall=dead_wall,
        finish_to_terminal=True,
    )
    assert_supported(result, "WP04A-07")


def test_wp04a_08_pao_liability_split_and_kazoe() -> None:
    """Dealer builds daisangen from three pons; the THIRD dragon meld is fed
    by s2 (discarding F copy 131), making s2 the pao bearer. s3 then discards
    sou9 (105) and the dealer rons. Manifest
    pao_policy=daisangen_daisuishi_tsumo_full_ron_half: on RON the pao bearer
    pays HALF of the win, the discarder the other half. Dealer yakuman ron =
    48000 -> expected delta [48000, 0, -24000, -24000].

    Kazoe documentation (manifest kazoe_policy=counted_yakuman_at_13_han):
    probe of the pinned engine score core calculate_score(han, fu=30, ron):
    12 han child = 24000 (sanbaiman cap) but 13 han child = 32000 and
    13 han dealer = 48000, i.e. counted-yakuman values start exactly at 13+ han;
    26 han doubles (child 64000). The same engine pays this case's 48000
    dealer yakuman ron end-to-end, so the manifest cap matches the engine."""
    hands = {
        # dealer/winner: CC PP FF pairs + 456m run + sou9 tanki + 3 floaters
        0: {
            132: 1,
            133: 1,
            124: 1,
            125: 1,
            128: 1,
            129: 1,
            12: 1,
            16: 1,
            20: 1,
            104: 1,
            0: 1,
            40: 1,
            60: 1,
        },
        # bystanders hold honour singles + non-adjacent simples: no claim
        # windows ever open for them, keeping the decision stream fixed
        1: {
            134: 1,
            108: 1,
            112: 1,
            120: 1,
            4: 1,
            24: 1,
            44: 1,
            68: 1,
            84: 1,
            100: 1,
            17: 1,
            52: 1,
            36: 1,
        },
        2: {
            127: 1,
            131: 1,
            109: 1,
            117: 1,
            121: 1,
            8: 1,
            28: 1,
            48: 1,
            72: 1,
            96: 1,
            37: 1,
            64: 1,
            81: 1,
        },
        3: {
            111: 1,
            115: 1,
            119: 1,
            135: 1,
            5: 1,
            25: 1,
            41: 1,
            56: 1,
            85: 1,
            101: 1,
            123: 1,
            130: 1,
            126: 1,
        },
    }
    live = {52: 14, 53: 15, 54: 18, 55: 19, 56: 21, 57: 22, 58: 23, 59: 26, 60: 105}
    script = (
        _do("tsumogiri", 14),
        _do("discard", 134),  # s1 feeds third C...
        _do("pon"),  # dragon meld 1/3
        _do("discard", 0),
        _do("auto"),  # s1 junk draw
        _do("discard", 127),  # s2 feeds third P...
        _do("pon"),  # dragon meld 2/3
        _do("discard", 40),
        _do("auto"),  # s1 junk draw
        _do("auto"),  # s1 junk draw
        _do("discard", 131),  # s2 feeds third F -> PAO BEARER becomes s2
        _do("pon"),  # daisangen complete
        _do("discard", 60),  # dealer tenpai: 456m + sou9 tanki
        _do("auto"),  # s1 junk draw
        _do("auto"),  # s2 junk draw
        _do("discard", 105),  # s3 draws live60=105 (sou9) and discards it
        _do("ron"),
    )
    result = _run(
        "WP04A-08",
        "daisangen pao half-split on ron + kazoe boundary",
        ("pao_policy", "kazoe_policy", "yakuman_policy"),
        (
            "tenhou.net/man RULE L1035-1036: daisangen/daisuishi pao; tsumo = "
            "full amount from pao bearer, ron = half; honba billed to bearer",
            "rules manifest pao_policy=daisangen_daisuishi_tsumo_full_ron_half; "
            "third dragon meld fed by s2 -> s2 liable for half of 48000",
            "kazoe probe: engine score core pays counted yakuman from 13 han "
            "(child ron 12han=24000 vs 13han=32000; dealer 13han=48000; "
            "26han=64000) matching kazoe_policy=counted_yakuman_at_13_han",
            "wall discipline: consumed live indices 52-60 pinned; bystander "
            "hands claim-free by construction (single honours, gap >=2 simples)",
        ),
        hands=hands,
        live_draws=live,
        script=script,
        expectations=[
            _expect_scores_delta((48000, 0, -24000, -24000)),
        ],
        finish_to_terminal=True,
    )
    assert_supported(result, "WP04A-08")


def test_wp04a_09_kyuushu_kyuuhai_abort() -> None:
    """Dealer's opening draw offers 10 distinct terminal/honor kinds; the
    canonical grammar exposes the abort as action kind
    ``abort_nine_terminals`` (probed at step 0). Applying it must emit
    abortive_draw reason=kyuushu_kyuuhai with ZERO payment ([0,0,0,0]
    scores delta, scores still [25000]*4), nothing settled before it, and
    the match continuing (a later round_start exists). Every abortive draw
    renchants per Tenhou RULE L1029-1030."""
    hands = {
        # dealer: E S W N P F C + 1m + 9m + 1p = 10 distinct terminal/honour kinds
        0: {
            108: 1,
            112: 1,
            116: 1,
            120: 1,
            124: 1,
            128: 1,
            132: 1,
            0: 1,
            32: 1,
            36: 1,
            41: 1,
            45: 1,
            61: 1,
        },
        1: {
            104: 1,
            105: 1,
            106: 1,
            107: 1,
            20: 1,
            21: 1,
            22: 1,
            23: 1,
            64: 1,
            65: 1,
            66: 1,
            67: 1,
            68: 1,
        },
        2: {
            72: 1,
            73: 1,
            74: 1,
            75: 1,
            80: 1,
            81: 1,
            82: 1,
            83: 1,
            100: 1,
            101: 1,
            102: 1,
            103: 1,
            69: 1,
        },
        3: {
            76: 1,
            77: 1,
            78: 1,
            79: 1,
            84: 1,
            85: 1,
            86: 1,
            87: 1,
            12: 1,
            13: 1,
            14: 1,
            15: 1,
            70: 1,
        },
    }
    live = {52: 17}

    def abort_is_free_and_match_continues(sim) -> str | None:
        index = next((i for i, e in enumerate(sim._events) if e.kind == "abortive_draw"), None)
        if index is None:
            kinds = [e.kind for e in sim._events]
            return f"no abortive_draw event (tail {kinds[-8:]})"
        event = sim._events[index]
        if event.payload.reason != "kyuushu_kyuuhai":
            return f"abort reason {event.payload.reason!r} != 'kyuushu_kyuuhai'"
        delta = next((d.value for d in event.public_delta if list(d.path) == ["scores"]), None)
        if delta != [0, 0, 0, 0]:
            return f"kyuushu abort moved scores by {delta}; Tenhou pays nothing"
        if list(event.payload.scores or ()) != [25000, 25000, 25000, 25000]:
            return f"scores at abort {event.payload.scores} != starting [25000]*4"
        settled_before = [
            e.kind
            for e in sim._events[:index]
            if e.kind in ("ron", "tsumo", "draw_end", "riichi_discard")
        ]
        if settled_before:
            return f"settlement events before the abort: {settled_before}"
        if not any(e.kind == "round_start" for e in sim._events[index:]):
            return "match did not continue after the abortive draw"
        return None

    result = _run(
        "WP04A-09",
        "kyuushu kyuuhai nine-terminal abort",
        ("abortive_draws",),
        (
            "tenhou.net/man RULE L1029-1030: kyuushu needs 9+ terminal/honour "
            "kinds on the first uninterrupted draw; every abortive draw renchains",
            "probe DUT-2: canonical action kind abort_nine_terminals offered at "
            "dealer step 0; adapter maps engine kyushu_kyuhai -> kyuushu_kyuuhai",
            "wall discipline: only consumed live index 52 pinned (17)",
        ),
        hands=hands,
        live_draws=live,
        script=(_do("abort_nine_terminals"),),
        expectations=[
            _expect_abortive("kyuushu_kyuuhai"),
            expect_predicate(
                "kyuushu aborts unpaid and play continues",
                abort_is_free_and_match_continues,
            ),
        ],
        finish_to_terminal=True,
    )
    assert_supported(result, "WP04A-09")


def test_wp04a_10_exhaustive_draw_noten_split() -> None:
    """Wall exhausts under pure tsumogiri auto-drive (fallback policy prefers
    pass at windows and tsumogiri at draws, so nobody ever claims or wins).
    Seat 0 is the ONLY tenpai hand: 123m456p789s CCC + N tanki (verified with
    the engine evaluator: tenpai=True waits=[N]); seats 1-3 verified noten.
    Tenhou noten penalty: tenpai collects 3000, each noten seat pays 1000 ->
    FIRST draw_end carries scores delta [+3000, -1000, -1000, -1000] and the
    following round_end shows [28000, 24000, 24000, 24000].

    Only the FIRST occurrence is asserted: later hands redealt from the wall
    stream have unpinned shapes, and the absolute post-payment snapshot rides
    on round_end because the draw_end envelope's own payload.scores field
    double-applies the delta (adapter observation, reported separately)."""
    hands = {
        0: {
            0: 1,
            4: 1,
            8: 1,
            48: 1,
            52: 1,
            56: 1,
            96: 1,
            100: 1,
            104: 1,
            132: 1,
            133: 1,
            134: 1,
            120: 1,
        },
        1: {
            112: 1,
            115: 1,
            28: 1,
            36: 1,
            44: 1,
            69: 1,
            77: 1,
            85: 1,
            93: 1,
            101: 1,
            109: 1,
            129: 1,
            111: 1,
        },
        2: {
            116: 1,
            118: 1,
            24: 1,
            33: 1,
            41: 1,
            57: 1,
            65: 1,
            73: 1,
            81: 1,
            89: 1,
            97: 1,
            130: 1,
            126: 1,
        },
        3: {
            12: 1,
            20: 1,
            37: 1,
            45: 1,
            60: 1,
            76: 1,
            84: 1,
            92: 1,
            105: 1,
            113: 1,
            127: 1,
            131: 1,
            135: 1,
        },
    }

    def first_exhaustive_split(sim) -> str | None:
        index = next((i for i, e in enumerate(sim._events) if e.kind == "draw_end"), None)
        if index is None:
            return "no draw_end event; exhaustive draw never reached"
        event = sim._events[index]
        if event.payload.reason != "exhaustive_draw":
            return f"first draw_end reason {event.payload.reason!r} != 'exhaustive_draw'"
        delta = next((d.value for d in event.public_delta if list(d.path) == ["scores"]), None)
        if delta != [3000, -1000, -1000, -1000]:
            return (
                f"tenpai/noten split {delta} != authority [3000, -1000, -1000, "
                "-1000] (tenpai +3000 total, noten -1000 each, honba 0)"
            )
        round_end = next((e for e in sim._events[index:] if e.kind == "round_end"), None)
        if round_end is None:
            return "no round_end after the exhaustive draw"
        post = list(round_end.payload.scores or ())
        if post != [28000, 24000, 24000, 24000]:
            return f"post-payment scores {post} != 25000 + split"
        return None

    result = _run(
        "WP04A-10",
        "exhaustive draw tenpai/noten payment split",
        ("all_last_policy",),
        (
            "tenhou.net/man RULE L1028 area: ryuukyoku settles 听牌 3000 / "
            "ノーテン -1000x3; noten dealer rotates, tenpai dealer renchains "
            "(quoted under all_last_policy row of the recon evidence matrix)",
            "tenpai status proven objectively with the engine's own evaluator: "
            "seat0 tenpai waits=[N], seats 1-3 noten",
            "wall note: identity live region left unpinned deliberately - the "
            "outcome is invariant under fallback drive because no seat can "
            "claim or win (pass/tsumogiri preference); N copies beyond seat0's "
            "stay unreachable in the dead wall",
        ),
        hands=hands,
        live_draws={},
        script=(),
        expectations=[
            expect_predicate(
                "first exhaustive draw pays 3000/-1000 split",
                first_exhaustive_split,
            ),
        ],
        finish_to_terminal=True,
    )
    assert_supported(result, "WP04A-10")
