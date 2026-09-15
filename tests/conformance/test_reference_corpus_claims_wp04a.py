"""WP-04A reference corpus: claim-window cases 03-06 (kuikae, furiten, multi-ron).

Call-meld exchange, temporary and permanent furiten, and multi-ron packet
priority replayed through the WP-03A reference adapter; documented engine
deviations resolve through DOCUMENTED_DEVIATIONS.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hydra2.conformance.runner import CaseResult, expect_predicate
from hydra2.conformance.walls import type_id
from tests.conformance.test_reference_corpus_wp04a import (
    _do,
    _expect_scores_delta,
    _neg,
    _run,
    assert_supported,
)

pytestmark = pytest.mark.contract_package("WP-04A")


def test_wp04a_03_kuikae_post_pon_same_meld_swap_barred() -> None:
    """seat1 holds 5p {53,54,55}; dealer tedashis red 5p 52; the pon consumes
    {53,54} leaving copy 55. Discarding 55 immediately would exchange a hand
    copy for the called meld tile (kuikae) - forbidden since 2007-11-29 - so
    the engine MUST NOT offer discard 55 on seat1's post-pon discard decision.
    Positive control: a legal non-meld post-pon discard still flows."""

    def check_pon(sim) -> str | None:
        pon = next((e for e in sim._events if e.kind == "pon"), None)
        if pon is None:
            return "no pon envelope"
        if int(pon.payload.tile) != 52 or int(pon.actor) != 1:
            return f"pon tile/actor {pon.payload.tile}/{int(pon.actor)} != 52/1"
        return None

    def check_first_post_pon_discard(sim) -> str | None:
        seen_pon = False
        for e in sim._events:
            if e.kind == "pon":
                seen_pon = True
                continue
            if seen_pon and e.kind in ("discard", "tsumogiri"):
                t = int(e.payload.tile)
                if type_id(t) == 13:
                    return f"first post-pon discard {t} is still a 5p (kuikae swap executed)"
                return None
        return "no discard envelope after pon"

    result = _run(
        "WP04A-03",
        "kuikae edges: immediate post-pon same-meld swap barred (kuikae_policy=forbidden)",
        ("kuikae_policy",),
        (
            "manifest kuikae_policy=forbidden (configs/rules/tenhou_4p_hanchan_v1.json)",
            "tenhou.net/man: kuikae banned 2007-11-29",
            "adapter rules gate: RiichiEnv hard-forbids kuikae swaps",
            "geometry: seat1 holds 5p {53,54,55}; dealer tedashi red 5p 52; pon consumes "
            "{53,54}, leaving copy 55 as the barred same-turn discard",
        ),
        hands={
            0: {
                52: 1,
                4: 1,
                5: 1,
                8: 1,
                9: 1,
                12: 1,
                13: 1,
                16: 1,
                20: 1,
                21: 1,
                24: 1,
                28: 1,
                32: 1,
            },
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
            2: dict(_HONORS_FIRST_JUN),
            3: dict(_HONORS_SECOND_JUN),
        },
        live_draws={52: 6, 53: 40, 54: 44, 55: 48, 56: 36, 57: 60, 58: 64, 131: 71},
        script=(
            _do("discard", 52),  # dealer tedashi red 5p
            _do("pon", 52),  # seat1 pons (consumes 53+54, keeps 55)
            _neg("discard", 55),  # barred kuikae swap must not be offered
            _do("auto"),  # legal post-pon discard flows
        ),
        expectations=[
            expect_predicate("pon of red 5p 52 by seat1", check_pon),
            expect_predicate(
                "first post-pon discard is not a 5p copy", check_first_post_pon_discard
            ),
        ],
        finish_to_terminal=True,
    )
    assert_supported(result, "WP04A-03")


def test_wp04a_04a_temp_furiten_clears_then_ron_lands() -> None:
    """seat1 waits 9s tanki on a concealed haku triplet hand (fanpai 1han,
    ankoh 8fu + tanki 2fu + menzen 10fu + base 20fu = 40fu). Dealer tedashis
    9s#1: seat1 passes (temporary furiten), then clears it at own tsumogiri.
    Dealer tedashis 9s#2 later: the ron MUST land, paying dealer 2600
    (child-table 1300 doubled because the discarder is the dealer).
    DOCUMENTED DEVIATION: RiichiEnv pays the child value 1300."""

    def check_ron(sim) -> str | None:
        ron = _first_ron(sim)
        if ron is None:
            return "no hora envelope in stream"
        if ron.kind != "ron":
            return f"first hora is {ron.kind}, expected ron"
        if int(ron.actor) != 1 or int(ron.payload.source_seat) != 0 or int(ron.payload.tile) != 105:
            return (
                f"ron actor/src/tile {int(ron.actor)}/{int(ron.payload.source_seat)}"
                f"/{int(ron.payload.tile)} != 1/0/105"
            )
        return None

    result = _run(
        "WP04A-04a",
        "furiten variant: passed ron clears at own discard, later ron lands",
        ("scoring_tables",),
        (
            "manifest furiten_policy="
            "river_only_permanent_after_riichi_miss_same_goaround_temporary",
            "tenhou.net/man furiten: temporary furiten ends at own discard",
            "engine flag missed_agari_doujun maps to 'temporary' (state.py D-WP03A-3)",
            "probe: riichienv.calculate_score(han,fu,is_oya=is WINNER only) pay_ron child=1300 "
            "oya=2000; dealer-discarder x2 multiplier absent in RiichiEnv 0.4.10",
            "shape: seat1 haku ankoh + 234m 678m 123s, 9s tanki; 40fu 1han; dealer pays 2600",
        ),
        hands={
            0: {
                104: 1,
                105: 1,
                0: 1,
                1: 1,
                2: 1,
                3: 1,
                16: 1,
                17: 1,
                18: 1,
                19: 1,
                32: 1,
                33: 1,
                34: 1,
            },
            1: {
                124: 1,
                125: 1,
                126: 1,
                4: 1,
                8: 1,
                12: 1,
                20: 1,
                24: 1,
                28: 1,
                72: 1,
                76: 1,
                80: 1,
                106: 1,
            },
            2: dict(_HONORS_FIRST_JUN),
            3: {
                121: 1,
                122: 1,
                123: 1,
                128: 1,
                129: 1,
                130: 1,
                131: 1,
                132: 1,
                133: 1,
                134: 1,
                135: 1,
                36: 1,
                38: 1,
            },
        },
        live_draws={
            52: 107,
            53: 56,
            54: 60,
            55: 64,
            56: 52,
            57: 44,
            58: 48,
            59: 40,
            60: 39,
            131: 71,
        },
        script=(
            _do("discard", 104),  # dealer tedashi 9s#1 -> seat1 can ron (yakuhai haku)
            _do("pass"),  # seat1 declines: temporary furiten
            _do("tsumogiri"),  # seat1 own discard CLEARS temp furiten
            _do("tsumogiri"),  # seat2
            _do("tsumogiri"),  # seat3
            _do("tsumogiri"),  # dealer
            _do("tsumogiri"),  # seat1
            _do("tsumogiri"),  # seat2
            _do("tsumogiri"),  # seat3
            _do("discard", 105),  # dealer tedashi 9s#2
            _do("ron", 105),  # seat1 rons: furiten cleared, hora lands
        ),
        expectations=[
            expect_predicate("ron by seat1 on dealer's 105", check_ron),
            _expect_scores_delta([-2600, 2600, 0, 0]),
        ],
        finish_to_terminal=True,
    )
    assert_documented_mismatch(result, "WP04A-04a")


def test_wp04a_04b_permanent_furiten_after_riichi_miss() -> None:
    """seat1 riichis waiting sou9 tanki (copy 104); dealer tedashis copy 105;
    seat1 declines (riichi miss => PERMANENT furiten). Copy 107 re-enters the
    river two turns later: the engine must never offer seat1 a ron again (the
    negate step pins the deterministic window; the stream predicate pins the
    whole first hand), while the hand continues to terminal."""

    def check_riichi_anchor(sim) -> str | None:
        acc = next((e for e in sim._events if e.kind == "riichi_accepted"), None)
        if acc is None:
            return "no riichi_accepted event"
        if int(acc.actor) != 1:
            return f"riichi_accepted actor {int(acc.actor)} != 1"
        return None

    def check_no_seat1_win_before_round_end(sim) -> str | None:
        for e in sim._events:
            if e.kind == "round_end":
                break
            if e.kind in ("ron", "tsumo") and int(e.actor) == 1:
                return f"seat1 won ({e.kind}) after the riichi miss: permanent furiten violated"
        return None

    def check_terminal(sim) -> str | None:
        if not sim._terminal:
            return "simulation did not reach terminal state"
        return None

    result = _run(
        "WP04A-04b",
        "furiten variant: riichi player passing the winning tile is permanently furiten",
        ("furiten_policy",),
        (
            "manifest furiten_policy="
            "river_only_permanent_after_riichi_miss_same_goaround_temporary",
            "tenhou.net/man: riichi player declining own winning tile can never ron afterwards",
            "geometry: sou9 copies 104(seat1 tanki) 105(dealer tedashi) 107(live62 re-entry) "
            "106(dead wall, unreachable)",
        ),
        hands={
            0: {
                105: 1,
                0: 1,
                1: 1,
                32: 1,
                33: 1,
                19: 1,
                36: 1,
                37: 1,
                38: 1,
                39: 1,
                41: 1,
                42: 1,
                43: 1,
            },
            1: {
                104: 1,
                4: 1,
                8: 1,
                12: 1,
                20: 1,
                24: 1,
                28: 1,
                40: 1,
                44: 1,
                48: 1,
                52: 1,
                56: 1,
                60: 1,
            },
            2: dict(_HONORS_FIRST_JUN),
            3: dict(_HONORS_SECOND_JUN),
        },
        live_draws={
            52: 89,
            53: 5,
            54: 9,
            55: 10,
            56: 14,
            57: 18,
            58: 22,
            59: 26,
            60: 30,
            61: 74,
            62: 107,
            131: 71,
        },
        dead_wall={132: 106},
        script=(
            _do("tsumogiri"),  # dealer river claim-proof tile
            _do("riichi_discard", 5),  # seat1 riichi: 4 melds + sou9 tanki
            _do("tsumogiri"),  # seat2
            _do("tsumogiri"),  # seat3
            _do("discard", 105),  # dealer tedashi sou9#2 -> seat1 offered ron
            _do("pass"),  # seat1 declines: PERMANENT furiten (riichi miss)
            _do("tsumogiri"),  # seat1 forced
            _do("tsumogiri"),  # seat2
            _do("tsumogiri"),  # seat3
            _do("tsumogiri"),  # dealer
            _do("tsumogiri"),  # seat1 forced
            _do("tsumogiri"),  # seat2 draws 62 = 107: sou9 back on the river
            _neg("ron"),  # violation iff the engine re-offers seat1's ron
        ),
        expectations=[
            expect_predicate("seat1 riichi anchored", check_riichi_anchor),
            expect_predicate(
                "no seat1 hora before first round_end", check_no_seat1_win_before_round_end
            ),
            expect_predicate("game reached terminal", check_terminal),
        ],
        finish_to_terminal=True,
    )
    assert_supported(result, "WP04A-04b")


def test_wp04a_05_double_ron_priority_packets_upstream_first() -> None:
    """seat1+seat2 riichi on SECOND turns (daburi impossible by definition,
    ippatsu alive), both waiting 3s/6s through 4s5s; child seat3 tedashis sou3
    83 within the same go-around. Table: seat1 riichi+ippatsu+tanyao+pinfu+aka
    = 5han30fu mangan 8000; seat2 same without aka = 4han30fu 7700; BOTH sticks
    go UPSTREAM (walking backward from discarder 3: seat2 first) => packet
    heads with seat2 and deltas [0,8000,9700,-15700].
    DOCUMENTED DEVIATION: packet heads with seat1 and seat1 takes both sticks."""

    def check_packet(sim) -> str | None:
        ron = _first_ron(sim)
        if ron is None:
            return "no ron envelope (double ron merged packet missing)"
        if int(ron.actor) != 2:
            return (
                f"merged ron packet actor={int(ron.actor)} != 2 "
                "(upstream winner must head the packet)"
            )
        if int(ron.payload.source_seat) != 3 or int(ron.payload.tile) != 83:
            return f"ron src/tile {int(ron.payload.source_seat)}/{int(ron.payload.tile)} != 3/83"
        return None

    def check_call_resolved(sim) -> str | None:
        ron = _first_ron(sim)
        if ron is None or ron.payload.action_id is None:
            return "no accepted ron action id to anchor call_resolved"
        want = [int(ron.payload.action_id)]
        cr = next(
            (
                e
                for e in sim._events
                if e.kind == "call_resolved" and list(e.payload.accepted_action_ids) == want
            ),
            None,
        )
        if cr is None:
            return f"no call_resolved with accepted_action_ids == {want}"
        offered = list(cr.payload.offered_action_ids)
        if len(offered) < 2 or want[0] not in offered:
            return f"offered ids {offered} lack both ron offers / accepted id"
        return None

    result = _run(
        "WP04A-05",
        "double ron priority packets: upstream-first merge, call_resolved ids, sticks upstream",
        ("multi_ron_resolution", "riichi_stick_rule"),
        (
            "manifest multi-ron: wins establish from the discarder's kamicha side upward",
            "riichi stick rule: kyotaku to the upstream winner on double ron (tenhou.net/man)",
            "upstream derivation: walking backward from discarder seat3 (2->1->0) the first "
            "winner is seat2 => seat2 heads packets and takes both sticks",
            "yaku derivation (second-turn declares: daburi impossible; ippatsu alive): "
            "seat1 riichi+ippatsu+tanyao+pinfu+aka-sou5 = 5han30fu mangan 8000; "
            "seat2 riichi+ippatsu+tanyao+pinfu = 4han30fu 7700",
            "table deltas: [0,8000,9700,-15700]",
        ),
        hands={
            0: dict(_HONORS_FIRST_JUN),
            1: {
                4: 1,
                8: 1,
                12: 1,
                20: 1,
                24: 1,
                28: 1,
                40: 1,
                44: 1,
                48: 1,
                84: 1,
                88: 1,
                60: 1,
                61: 1,
            },
            2: {
                5: 1,
                9: 1,
                13: 1,
                21: 1,
                25: 1,
                29: 1,
                41: 1,
                45: 1,
                49: 1,
                85: 1,
                89: 1,
                64: 1,
                65: 1,
            },
            3: {
                83: 1,
                0: 1,
                1: 1,
                32: 1,
                33: 1,
                16: 1,
                17: 1,
                36: 1,
                37: 1,
                38: 1,
                39: 1,
                72: 1,
                76: 1,
            },
        },
        live_draws={
            52: 123,
            53: 69,
            54: 70,
            55: 124,
            56: 125,
            57: 126,
            58: 127,
            59: 96,
            130: 35,
            131: 71,
        },
        script=(
            _do("tsumogiri"),  # dealer river honor
            _do("tsumogiri"),  # seat1 plain draw
            _do("tsumogiri"),  # seat2 plain draw
            _do("tsumogiri"),  # seat3 plain draw
            _do("tsumogiri"),  # dealer second draw
            _do("riichi_discard", 126),  # seat1 SECOND-turn riichi (ippatsu alive, daburi void)
            _do("riichi_discard", 127),  # seat2 SECOND-turn riichi
            _do("discard", 83),  # seat3 tedashi sou3 -> double ron window
            _do("ron", 83),  # buffered responder steps
            _do("ron", 83),
        ),
        expectations=[
            expect_predicate("merged ron packet headed by upstream winner seat2", check_packet),
            expect_predicate("call_resolved offered>=2 accepted==ron id", check_call_resolved),
            _expect_scores_delta([0, 8000, 9700, -15700]),
        ],
        finish_to_terminal=True,
    )
    assert_documented_mismatch(result, "WP04A-05")


def test_wp04a_06_multi_ron_sticks_upstream_with_dealer_co_winner() -> None:
    """Dealer and seat1 riichi (SECOND turns) and both ron child seat3's sou3
    tedashi. Upstream walk from discarder 3 hits seat1 before the dealer, so
    the packet must head with seat1 and BOTH sticks belong to seat1: dealer
    oya-win 5han30fu (incl aka) = 12000, seat1 4han30fu = 7700 =>
    [12000, 9700, 0, -21700].
    DOCUMENTED DEVIATION: packet heads with the dealer (seat order 0<1) and the
    dealer collects both sticks: observed [14000, 7700, 0, -19700].
    HONBA SCOPE: honba is NOT injectable through the conformance reset surface
    (adapter.reset(rules=,wall=,seat_permutation=) only; _open_hand hardcodes
    honba=0), so this case asserts the sticks-only slice per scope note."""

    def check_packet(sim) -> str | None:
        ron = _first_ron(sim)
        if ron is None:
            return "no ron envelope"
        if int(ron.actor) != 1:
            return (
                f"merged ron packet actor={int(ron.actor)} != 1 "
                "(upstream winner seat1 must head packet)"
            )
        if int(ron.payload.source_seat) != 3 or int(ron.payload.tile) != 82:
            return f"ron src/tile {int(ron.payload.source_seat)}/{int(ron.payload.tile)} != 3/82"
        return None

    result = _run(
        "WP04A-06",
        "multi-ron sticks+honba: child discarder, dealer co-winner, sticks to upstream seat1",
        ("multi_ron_resolution", "riichi_stick_rule"),
        (
            "manifest honba: 300xN per honba; honba NOT injectable via reset "
            "(adapter.reset(rules=,wall=,seat_permutation=) only, _open_hand honba=0 hardcoded)",
            "scope note: sticks-only asserted; honba split deferred until injection surface exists",
            "atamahane: walking backward from discarder seat3 (2->1->0) seat1 precedes the dealer",
            "yaku derivation (second-turn declares: daburi void; ippatsu alive): dealer "
            "riichi+ippatsu+tanyao+pinfu+aka = 5han30fu OYA-ron 12000; seat1 same minus aka "
            "= 4han30fu child-ron 7700",
            "table deltas: [12000,9700,0,-21700]; sticks (2x1000) upstream to seat1",
        ),
        hands={
            0: {
                4: 1,
                8: 1,
                12: 1,
                20: 1,
                24: 1,
                28: 1,
                40: 1,
                44: 1,
                48: 1,
                84: 1,
                88: 1,
                60: 1,
                61: 1,
            },
            1: {
                5: 1,
                9: 1,
                13: 1,
                21: 1,
                25: 1,
                29: 1,
                41: 1,
                45: 1,
                49: 1,
                85: 1,
                89: 1,
                64: 1,
                65: 1,
            },
            2: {
                36: 1,
                37: 1,
                38: 1,
                39: 1,
                2: 1,
                3: 1,
                32: 1,
                33: 1,
                92: 1,
                94: 1,
                96: 1,
                98: 1,
                100: 1,
            },
            3: {
                82: 1,
                16: 1,
                17: 1,
                93: 1,
                97: 1,
                101: 1,
                102: 1,
                103: 1,
                67: 1,
                0: 1,
                1: 1,
                34: 1,
                86: 1,
            },
        },
        live_draws={
            52: 117,
            53: 118,
            54: 119,
            55: 90,
            56: 110,
            57: 111,
            58: 112,
            130: 35,
            131: 71,
        },
        script=(
            _do("tsumogiri"),  # dealer river honor
            _do("tsumogiri"),  # seat1 plain draw
            _do("tsumogiri"),  # seat2 plain draw
            _do("tsumogiri"),  # seat3 plain draw
            _do("riichi_discard", 110),  # dealer SECOND-turn riichi (aka-sou5 ryanmen)
            _do("riichi_discard", 111),  # seat1 SECOND-turn riichi
            _do("tsumogiri"),  # seat2
            _do("discard", 82),  # seat3 tedashi sou3 -> double ron window
            _do("ron", 82),  # pending sorted: dealer buffered first
            _do("ron", 82),  # seat1 buffered second
        ),
        expectations=[
            expect_predicate(
                "merged ron packet headed by seat1 (upstream of child discarder)", check_packet
            ),
            _expect_scores_delta([12000, 9700, 0, -21700]),
        ],
        finish_to_terminal=True,
    )
    assert_documented_mismatch(result, "WP04A-06")


def _first_ron(sim):
    return next((e for e in sim._events if e.kind in ("ron", "tsumo")), None)


def assert_documented_mismatch(result: CaseResult, case_id: str) -> None:
    reason = DOCUMENTED_DEVIATIONS[case_id]
    assert result.status == "mismatch", (
        f"{case_id}: expected documented mismatch ({reason}); got {result.status}"
    )
    assert result.counterexample_path, f"{case_id}: mismatch must persist a counterexample"
    assert Path(result.counterexample_path).exists(), (
        f"{case_id}: counterexample file missing: {result.counterexample_path}"
    )


DOCUMENTED_DEVIATIONS: dict[str, str] = {
    "WP04A-04a": (
        "RiichiEnv 0.4.10 pays the CHILD table value when a child rons off a DEALER "
        "discard (calculate_score models only the WINNER's rank; the dealer-pays-"
        "double channel-hon rule is absent): observed [-1300,1300,0,0] vs Tenhou "
        "40fu 1han dealer-discarder 2600."
    ),
    "WP04A-05": (
        "Multi-ron resolution orders winners by seat number and allocates the "
        "kyotaku sticks to the first-processed winner: walking backward from "
        "discarder seat3 the upstream winner is seat2, yet the packet heads with "
        "seat1 and seat1 takes both sticks."
    ),
    "WP04A-06": (
        "Same seat-order deviation with the dealer as co-winner (seat-order 0,1 "
        "differs from upstream order 1,0): packet heads with the dealer and the "
        "dealer receives both sticks; observed [14000,7700,0,-19700] vs table "
        "[12000,9700,0,-21700]."
    ),
}

_HONORS_FIRST_JUN = {
    108: 1,
    109: 1,
    110: 1,
    111: 1,
    112: 1,
    113: 1,
    114: 1,
    115: 1,
    116: 1,
    117: 1,
    118: 1,
    119: 1,
    120: 1,
}

_HONORS_SECOND_JUN = {
    121: 1,
    122: 1,
    123: 1,
    124: 1,
    125: 1,
    126: 1,
    127: 1,
    128: 1,
    129: 1,
    130: 1,
    131: 1,
    132: 1,
    133: 1,
}
