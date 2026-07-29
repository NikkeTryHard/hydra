"""WP-04A reference conformance corpus (BUILD lines 397-420).

Fourteen edge-case cases replayed through the WP-03A reference adapter against
frozen expectations derived from the rules manifest and Tenhou evidence
quotes. Engine output drives wall construction only; every expected value is a
table-derived constant. The first counterexample of a failing case is
persisted under ``$HYDRA2_ARTIFACT_ROOT/counterexamples/WP-04A/``.

Status (this session): infrastructure + cases 01/02/11 executed. The
chankan case is blocked by the same-type copy-displacement dealing bug
(D-WP04A-FIX5 made pins ID-exact; the remaining blocker is the adapter's
missing chankan response window - RiichiEnv auto-passes responders that are
absent from the step dict). Cases 03-10/12-14 are authored in the ext module.
"""

from __future__ import annotations

import inspect
import json
import time
from functools import cache
from pathlib import Path

import pytest

from hydra2.config import artifact_root
from hydra2.conformance.report import build_intersection_report, write_intersection_report
from hydra2.conformance.runner import (
    CaseResult,
    ReferenceTraceRunner,
    ScriptedDecision,
    expect_predicate,
)
from hydra2.conformance.walls import build_wall, type_id
from hydra2.contracts.common import ContractError
from hydra2.contracts.rules import resolve_final_ranks, rules_manifest_from_payload
from hydra2.contracts.utility import (
    UTILITY_OBJECTIVE,
    UTILITY_TIE_POLICY,
    RawOutcome,
    make_utility_manifest,
    root_scalar,
    utility,
)

pytestmark = pytest.mark.contract_package("WP-04A")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RULES_PAYLOAD = json.loads(
    (_REPO_ROOT / "configs" / "rules" / "tenhou_4p_hanchan_v1.json").read_text()
)
_MANIFEST = rules_manifest_from_payload(_RULES_PAYLOAD["payload"])

DOCUMENTED_UNSUPPORTED: dict[str, str] = {
    "suufon_renda": (
        "RiichiEnv 0.4.8 never emits the four-winds abortive ryukyoku: four "
        "first-turn own-wind discards leave the hand running (probe DUT-1)."
    ),
    "scoring_tables": (
        "owner_decision D1: RiichiEnv 0.4.8 omits the dealer-discarder x2 "
        "payment multiplier (child ron off dealer pays 1300 where Tenhou "
        "tables require 2600); counterexample WP04A-04a.json. Expectations "
        "stay Tenhou-correct; upstream fix or adapter correction layer is "
        "tracked as follow-up debt."
    ),
    "multi_ron_resolution": (
        "owner_decision D2: double-ron packets and riichi-stick attribution "
        "follow seat number instead of kamicha-upstream priority; "
        "counterexamples WP04A-05.json / WP04A-06.json. Payments themselves "
        "match tables; only ordering/attribution deviates."
    ),
    "riichi_stick_rule": (
        "owner_decision D2 (stick slice): on multi-ron the engine hands every "
        "kyotaku stick to the lowest seat number rather than the upstream "
        "winner; see WP04A-05.json / WP04A-06.json."
    ),
    "all_last_policy": (
        "owner_decision D3: RiichiEnv YON_HANCHAN has no West-round entry, so "
        "all-last continuation below return points cannot occur; "
        "counterexample WP04A-14b.json (same class as suufon_renda)."
    ),
    "sudden_death_policy": (
        "owner_decision D3 (sudden-death slice): no sudden-death extension "
        "exists in the pinned engine; see WP04A-14b.json."
    ),
}

T30FU_CHILD_RON = {1: 1000, 2: 2000, 3: 3900, 4: 5200, 5: 8000}


def _pass() -> ScriptedDecision:
    return ScriptedDecision("pass")


def _do(kind: str, tile: int | None = None) -> ScriptedDecision:
    return ScriptedDecision(kind, tile=tile)


def _neg(kind: str, tile: int | None = None) -> ScriptedDecision:
    return ScriptedDecision(kind, tile=tile, negate=True)


def _first_event(sim, kind: str):
    return next((e for e in sim._events if e.kind == kind), None)


def _event_tiles(sim, kind: str) -> list[int]:
    return [
        int(e.payload.tile) for e in sim._events if e.kind == kind and e.payload.tile is not None
    ]


def _hora_scores_delta(sim):
    for envelope in sim._events:
        if envelope.kind in ("ron", "tsumo"):
            for delta in envelope.public_delta:
                if list(delta.path) == ["scores"]:
                    return [int(v) for v in delta.value]
    return None


def _expect_scores_delta(expected):
    def check(sim) -> str | None:
        got = _hora_scores_delta(sim)
        if got is None:
            return "no hora scores delta in stream"
        if got != list(expected):
            return f"hora deltas {got} != table-derived {list(expected)}"
        return None

    return expect_predicate("hora deltas match Tenhou tables", check)


def _expect_dora_revealed(tiles: list[int]):
    def check(sim) -> str | None:
        got = _event_tiles(sim, "dora_revealed")
        if got != tiles:
            return f"dora_revealed {got} != expected {tiles}"
        return None

    return expect_predicate(f"dora_revealed == {tiles}", check)


def _expect_no_ura_in_public():
    def check(sim) -> str | None:
        for envelope in sim._events:
            for delta in envelope.public_delta:
                if "ura" in str(list(delta.path)).lower():
                    return f"ura leaked via public path {list(delta.path)}"
        return None

    return expect_predicate("ura markers hidden until hora", check)


def _expect_abortive(reason: str):
    def check(sim) -> str | None:
        event = _first_event(sim, "abortive_draw")
        if event is None:
            kinds = [e.kind for e in sim._events]
            return f"no abortive_draw event (tail {kinds[-8:]})"
        if event.payload.reason != reason:
            return f"abortive reason {event.payload.reason!r} != {reason!r}"
        return None

    return expect_predicate(f"abortive_draw reason={reason}", check)


_RESULTS: dict[str, CaseResult] = {}


def _run(case_id, title, rule_fields, evidence, **kwargs) -> CaseResult:
    result = _run_case(_runner(), case_id, title, rule_fields, evidence, **kwargs)
    _RESULTS[case_id] = result
    return result


@cache
def _runner() -> ReferenceTraceRunner:
    # Session-shared runner (one per process): construction only binds the
    # manifest. run_case stays stateless across calls (fresh simulator per
    # case, counterexamples persisted per case_id). CaseResults are NEVER
    # cached — _RESULTS/_WAVE_C_RESULTS still record live runs per case.
    return ReferenceTraceRunner(manifest=_MANIFEST)


def _run_case(
    runner: ReferenceTraceRunner,
    case_id: str,
    title: str,
    rule_fields,
    evidence,
    *,
    hands,
    live_draws,
    script,
    expectations,
    dead_wall=None,
    finish_to_terminal=True,
) -> CaseResult:
    wall = build_wall(hands=hands, live_draws=live_draws, dead_wall=dead_wall or {})
    return runner.run_case(
        case_id,
        title,
        tuple(rule_fields),
        tuple(evidence),
        wall,
        tuple(script),
        list(expectations),
        finish_to_terminal=finish_to_terminal,
    )


def assert_supported(result: CaseResult, case_id: str) -> None:
    assert result.status == "supported", (
        f"{case_id}: {result.error_detail}; counterexample={result.counterexample_path}"
    )


# ---------------------------------------------------------------------------
# WP04A-01 fifth dora indicator + kan-dora/ura timing.
# ---------------------------------------------------------------------------


def test_wp04a_01_fifth_dora_and_kan_ura_timing() -> None:
    """Two dealer ankans reveal indicators 129 then 127 immediately after each
    kan; ura slots stay out of public state without a winning riichi hand."""
    hands = {
        0: {
            88: 1,
            89: 1,
            90: 1,
            116: 1,
            117: 1,
            118: 1,
            40: 1,
            41: 1,
            42: 1,
            43: 1,
            67: 1,
            68: 1,
            69: 1,
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
        3: {0: 1, 4: 1, 8: 1, 12: 1, 16: 1, 20: 1, 24: 1, 28: 1, 32: 1, 36: 1, 70: 1, 71: 1, 64: 1},
    }
    live = {
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
    }
    dead_wall = {131: 131, 129: 129, 127: 127}
    script = (
        _do("ankan"),
        _do("tsumogiri"),
        _do("auto"),
        _do("auto"),
        _do("auto"),
        _do("ankan"),
        _do("tsumogiri"),
        _do("auto"),
        _do("auto"),
        _do("auto"),
    )
    result = _run(
        "WP04A-01",
        "fifth dora indicator + kan-dora/ura timing",
        ("kan_dora_reveal_policy", "kan_ura_policy", "rinshan_policy"),
        (
            "tenhou.net/man YAKU L1246",
            "rules manifest kan_dora_reveal_policy=ankan_immediate_open_delayed",
            "probe facts journal: indicators [131]->[131,129]; ura slots indicator-1",
        ),
        hands=hands,
        live_draws=live,
        script=script,
        expectations=[_expect_dora_revealed([129, 127]), _expect_no_ura_in_public()],
        dead_wall=dead_wall,
        finish_to_terminal=True,
    )
    assert_supported(result, "WP04A-01")


# ---------------------------------------------------------------------------
# WP04A-02 chankan + rinshan payout (D-WP04A-FIX1/FIX4b regression).
# ---------------------------------------------------------------------------


def test_wp04a_02_chankan_and_rinshan_payout() -> None:
    """s3 riichis waiting 5s/8s ryanmen WITHOUT holding a copy; s1 ponned 5s
    earlier and later kakens the fourth copy -> chankan window opens for s3.
    Payout: riichi+pinfu+chankan = 4 han 30 fu child-vs-child = 5200 + stick."""
    hands = {
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
    }
    live = {
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
    }
    script = (
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
        _do("ron"),
    )
    expectations = [
        _expect_scores_delta((0, T30FU_CHILD_RON[4] * -1, 0, T30FU_CHILD_RON[4] + 1000)),
        expect_predicate(
            "chankan window opened by kakan",
            lambda sim: (
                None if _first_event(sim, "ron") is not None else "no ron event after kakan"
            ),
        ),
    ]
    result = _run(
        "WP04A-02",
        "chankan + rinshan payout",
        ("chankan_policy", "rinshan_policy", "riichi_stick_allocation"),
        (
            "tenhou.net/man YAKU L1177/L1246",
            "Tenhou scoring table: 4han30fu child ron = 5200; the agari winner "
            "collects every kyotaku stick incl. their own",
            "D-WP04A-FIX1 regression (Main-authorized adapter fix)",
        ),
        hands=hands,
        live_draws=live,
        script=script,
        expectations=expectations,
        finish_to_terminal=True,
    )
    assert_supported(result, "WP04A-02")


# ---------------------------------------------------------------------------
# WP04A-11 suufon_renda: documented-unsupported counterexample.
# ---------------------------------------------------------------------------


def test_wp04a_11_suufon_renda_documented_unsupported() -> None:
    """Four first-turn own-wind discards must abort (suufon_renda is a declared
    manifest rule); RiichiEnv 0.4.8 keeps playing. EXPECTED mismatch resolved
    through DOCUMENTED_UNSUPPORTED."""
    hands = {
        0: {
            108: 1,
            0: 1,
            4: 1,
            8: 1,
            12: 1,
            16: 1,
            20: 1,
            24: 1,
            1: 1,
            5: 1,
            124: 1,
            128: 1,
            132: 1,
        },
        1: {
            112: 1,
            28: 1,
            32: 1,
            36: 1,
            40: 1,
            44: 1,
            48: 1,
            52: 1,
            29: 1,
            33: 1,
            125: 1,
            129: 1,
            133: 1,
        },
        2: {
            116: 1,
            56: 1,
            60: 1,
            64: 1,
            68: 1,
            72: 1,
            76: 1,
            80: 1,
            57: 1,
            61: 1,
            126: 1,
            130: 1,
            134: 1,
        },
        3: {
            120: 1,
            84: 1,
            88: 1,
            92: 1,
            96: 1,
            100: 1,
            104: 1,
            85: 1,
            93: 1,
            97: 1,
            127: 1,
            131: 1,
            135: 1,
        },
    }
    live = {52: 2, 53: 30, 54: 58, 55: 86}
    script = tuple(_do("discard", wind) for wind in (108, 112, 116, 120))
    result = _run(
        "WP04A-11",
        "suufon_renda four-winds abort",
        ("suufon_renda",),
        (
            "tenhou.net/man RULE L1029-1030 (all five abortive draws incl. suufon_renda)",
            "probe DUT-1: engine continues the hand instead of aborting",
        ),
        hands=hands,
        live_draws=live,
        script=script,
        expectations=[_expect_abortive("suufon_renda")],
        finish_to_terminal=False,
    )
    assert result.status == "mismatch", (
        "engine unexpectedly aborted via suufon_renda - refresh the corpus verdict"
    )
    assert result.counterexample_path, "mismatch case must persist its counterexample"


# ---------------------------------------------------------------------------
# Intersection report + package disposition.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Integrated wave deliveries (B: 07-10, A: 03-06, C: 12-14).
# ---------------------------------------------------------------------------


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
            "oya=2000; dealer-discarder x2 multiplier absent in RiichiEnv 0.4.8",
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
        "RiichiEnv 0.4.8 pays the CHILD table value when a child rons off a DEALER "
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


def test_wp04a_12_sanchahou_triple_ron_abort() -> None:
    """p0/p1/p2 declare riichi in sequence; p3's tsumogiri of the fourth East
    completes all three waits. Tenhou RULE L1029-1030 makes 三家和了 an
    abortive draw (manifest ``abortive_draws`` lists 'sanchahou'); the hand
    must abort with NO hora payment and the three posted sticks must sit in
    the kyotaku for the next hand (``riichi_stick_allocation`` abort_carry).
    """
    wall = _sanchaho_wall()
    script = (
        _do("riichi_discard", tile=_CHUN_FIRST),
        _do("riichi_discard", tile=128),
        _do("riichi_discard", tile=124),
        _do("tsumogiri", tile=_E_TILES[3]),
        _do("ron"),
        _do("ron"),
        _do("ron"),
    )
    rule_fields = (
        "abortive_draws:sanchahou",
        "riichi_stick_allocation:end_top_take_abort_carry_dealin_exempt",
        "multiple_ron_policy",
    )
    evidence = (
        "recon-tenhou/evidence.md row23 RULE L1029-1030 (三家和了あり)",
        "riichienv GameRule.default_tenhou().sanchaho_is_draw=True",
    )

    def check_stream(sim) -> str | None:
        events = sim._events
        accepted = [e for e in events if e.kind == "riichi_accepted"]
        if len(accepted) != 3:
            return f"expected 3 riichi_accepted, got {len(accepted)}"
        if sorted(int(e.payload.actor) for e in accepted) != [0, 1, 2]:
            return "declarers must be seats 0,1,2"
        first_round_end = next(
            (i for i, e in enumerate(events) if e.kind == "round_end"), len(events)
        )
        head = events[:first_round_end]
        if any(e.kind in ("ron", "tsumo") for e in head):
            return "hora envelope appeared inside the sanchahou hand; abort must pay nothing"
        abortive = [e for e in head if e.kind == "abortive_draw"]
        if len(abortive) != 1:
            return f"expected exactly 1 abortive_draw before round_end, got {len(abortive)}"
        env = abortive[0]
        if str(env.payload.reason) != "sanchahou":
            return f"abortive reason {env.payload.reason!r} != 'sanchahou'"
        score_deltas = [d.value for d in env.public_delta if tuple(d.path) == ("scores",)]
        if not score_deltas or [int(v) for v in score_deltas[0]] != [-1000, -1000, -1000, 0]:
            return (
                "abortive deltas must move exactly the three posted sticks into the "
                f"pot and pay the feeder nothing, got {score_deltas}"
            )
        nxt = next(
            (e for e in events if e.kind == "round_start" and e.sequence > env.sequence), None
        )
        if nxt is None:
            return "no round_start after the abort"
        sticks = [d.value for d in nxt.public_delta if tuple(d.path) == ("riichi_sticks",)]
        if not sticks or int(sticks[0]) != 3:
            return f"next hand must carry kyotaku=3 sticks, delta={sticks}"
        if tuple(int(s) for s in nxt.payload.scores) != (24000, 24000, 24000, 25000):
            return (
                "net boundary scores must leave each declarer one stick down and the "
                f"feeder untouched, got {tuple(int(s) for s in nxt.payload.scores)}"
            )
        return None

    result = _record(
        _runner().run_case(
            "WP04A-12",
            "sanchahou: triple ron over three riichis aborts unpaid",
            rule_fields,
            evidence,
            wall_tiles=wall,
            script=script,
            expectations=[expect_predicate("sanchahou abort semantics", check_stream)],
            finish_to_terminal=True,
        )
    )
    assert_supported(result, "WP04A-12")


def test_wp04a_13_rank_tie_break_and_uma_utility() -> None:
    """Unit-grade proof that resolve_final_ranks + utility() honour the
    published policy fields (Tenhou L1025/L1013): equal scores place by
    East-1 seat-wind order; uma_by_rank converts through a UtilityManifest;
    tied ranks NEVER reach utility (use_rules_resolved_rank)."""
    assert _MANIFEST.rank_tie_break == "east1_seat_wind_order"
    assert _MANIFEST.placement_conversion_id == "tenhou_rank_sticks_top_uma_v1"
    assert _MANIFEST.uma_by_rank == (20, 10, -10, -20)

    # Strict ordering.
    assert resolve_final_ranks((30000, 25000, 20000, 15000)) == (1, 2, 3, 4)
    # Pair tie at 25000: lower seat index (East-1 wind order) ranks better.
    assert resolve_final_ranks((25000, 25000, 20000, 30000)) == (2, 3, 4, 1)
    # Four-way tie: pure seat-wind order.
    assert resolve_final_ranks((25000, 25000, 25000, 25000)) == (1, 2, 3, 4)

    from hydra2.engines.riichienv.state import rules_identity_hash

    rules_hash = str(rules_identity_hash(_MANIFEST))
    manifest = make_utility_manifest(
        utility_id="wp04a-13-tenhou-uma-v1",
        schema_version="1.0.0",
        rules_id=str(_MANIFEST.rules_id),
        rules_hash=rules_hash,
        objective=UTILITY_OBJECTIVE,
        rank_values=tuple(float(u) * 1000.0 for u in _MANIFEST.uma_by_rank),
        tie_policy=UTILITY_TIE_POLICY,
        value_min=-(10.0**9),
        value_max=10.0**9,
        zero_sum=True,
    )

    scores = (35000, 30000, 25000, 10000)
    start = (_MANIFEST.starting_points,) * 4
    outcome = RawOutcome(
        final_scores=scores,
        ranks=resolve_final_ranks(scores),
        point_deltas=tuple(scores[i] - start[i] for i in range(4)),
        settlements=(),
        rules_id=str(_MANIFEST.rules_id),
        rules_hash=rules_hash,
    )
    vector = utility(outcome, manifest)
    assert vector.values == (20000.0, 10000.0, -10000.0, -20000.0)
    assert str(vector.utility_manifest_hash) == str(manifest.digest)
    assert root_scalar(vector, 2) == -10000.0

    # Tied ranks are rejected upstream of any valuation.
    try:
        RawOutcome(
            final_scores=scores,
            ranks=(1, 1, 3, 4),
            point_deltas=(0, 0, 0, 0),
            settlements=(),
            rules_id=str(_MANIFEST.rules_id),
            rules_hash=rules_hash,
        )
    except ContractError:
        pass
    else:  # pragma: no cover - contract guard
        raise AssertionError("RawOutcome accepted tied ranks; ties must stay unresolved")

    _WAVE_C_RESULTS["WP04A-13"] = CaseResult(
        case_id="WP04A-13",
        title="placement ranks/tie-break/uma utility honour manifest",
        status="supported",
        rule_fields=(
            "rank_tie_break",
            "uma_by_rank",
            "placement_conversion_id",
            "starting_points",
            "return_points",
        ),
        evidence=(
            "recon-tenhou/evidence.md row29 RULE L1025 (東1局風順同点順位)",
            "recon-tenhou/evidence.md row4 RULE L1013,L1058 (ウマ10-20)",
            "contracts.utility tie_policy=use_rules_resolved_rank",
        ),
    )


def test_wp04a_14a_all_last_dealer_tenpai_stop_yame() -> None:
    """All-last agari-yame, tenpai-stop branch: seat3's E1 mangan ron puts the
    future South-4 dealer on 33000 (top, >= return_points). Under neutral
    play every later hand exhausts; at South-4 the dealer is tenpai (zero
    noten payments) while top>=30000, so per agari_yame_policy
    'dealer_top_auto_win_and_tenpai_stop' (man L1023/L1059) the match ends
    immediately: no renchan, no West entry, terminal outcome published."""

    def check_yame(sim) -> str | None:
        events = sim._events
        starts = [e for e in events if e.kind == "round_start"]
        if len(starts) != 8:
            return f"hanchan must open exactly 8 hands (E1..S4), got {len(starts)}"
        dealers = [int(e.payload.actor) for e in starts]
        if dealers != [0, 1, 2, 3, 0, 1, 2, 3]:
            return f"dealer rotation must run E1..S4 without renchan, got {dealers}"
        last = starts[-1]
        # Adapter hand ordinals double-count reopened boundaries (0,2,..,14).
        if int(last.payload.round_index) != 14:
            return f"S4 adapter ordinal must be 14, got {last.payload.round_index}"
        if tuple(int(s) for s in last.payload.scores) != (25000, 17000, 25000, 33000):
            return (
                "S4 entry scores must be mangan-shifted (25000,17000,25000,33000), got "
                f"{tuple(int(s) for s in last.payload.scores)}"
            )
        if int(last.payload.actor) != 3:
            return f"S4 dealer must be seat 3, got {last.payload.actor}"
        end_positions = [i for i, e in enumerate(events) if e.kind == "game_end"]
        if len(end_positions) != 1:
            return f"expected exactly one game_end, got {len(end_positions)}"
        end_at = end_positions[0]
        tail_kinds = [e.kind for e in events[end_at + 1 :]]
        if tail_kinds:
            return f"nothing may follow game_end, saw {tail_kinds[:4]}"
        pre = events[end_at - 1]
        if pre.kind != "round_end":
            return f"event before game_end must be the S4 round_end, got {pre.kind}"
        draw = events[end_at - 2]
        if draw.kind != "draw_end" or str(draw.payload.reason) != "exhaustive_draw":
            return (
                "S4 must end in an exhaustive_draw immediately before the boundary, got "
                f"{draw.kind}:{draw.payload.reason}"
            )
        deltas = [d.value for d in draw.public_delta if tuple(d.path) == ("scores",)]
        if deltas and any(int(v) != 0 for v in deltas[0]):
            return "all-tenpai exhaustive draw must pay nothing; dealer tenpai-stop applies"
        out = sim._raw_outcome
        if out is None:
            return "terminal outcome missing"
        finals = tuple(int(s) for s in out.final_scores)
        if finals[3] != max(finals) or finals[3] < 30000:
            return f"dealer must finish top >= 30000, finals={finals}"
        if tuple(int(r) for r in out.ranks) != resolve_final_ranks(finals):
            return f"ranks {tuple(out.ranks)} disagree with resolve_final_ranks{finals}"
        return None

    result = _record(
        _runner().run_case(
            "WP04A-14a",
            "all-last dealer tenpai-stop yame (top >= return_points)",
            ("agari_yame_policy", "all_last_policy", "return_points"),
            (
                "recon-tenhou/evidence.md row27 RULE L1023,L1059 (自動聴牌止め)",
                "recon-tenhou/evidence.md row26 (ラス親は原点越えのトップを維持すれば終了)",
            ),
            wall_tiles=_yame_wall(),
            script=(
                ScriptedDecision("auto"),  # seat0 junk tsumogiri
                ScriptedDecision("auto"),  # seat1 draws chun mate, feeds
                ScriptedDecision("ron"),  # seat3 mangan ron
            ),
            expectations=[expect_predicate("south-4 dealer tenpai yame", check_yame)],
            finish_to_terminal=True,
        )
    )
    assert_supported(result, "WP04A-14a")


def test_wp04a_14b_west_entry_sudden_death_expected_mismatch() -> None:
    """Sudden-death/West entry: with the top below return_points (30000) at
    the end of South-4 the manifest demands continuation
    (all_last_policy='south_west_entry_renchan_extension',
    sudden_death_policy='ge_return_points_excluding_sticks_dealer_priority',
    man L1019-1021 「サドンデス…30000点(供託未収)以上になった時点で終了」).
    RiichiEnv 0.4.8 YON_HANCHAN instead emits game_end 'hanchan_complete' -
    the same class of missing-rule deviation as WP04A-11 suufon_renda.
    EXPECTED-MISMATCH: the failure IS the documented evidence and must be
    persisted as a counterexample, never weakened."""

    def west_entry_required(sim) -> str | None:
        out = sim._raw_outcome
        if out is None:
            return "no terminal outcome; engine neither continued nor ended cleanly"
        finals = tuple(int(s) for s in out.final_scores)
        top = max(finals)
        if top >= int(_MANIFEST.return_points):
            return None  # clean end is Tenhou-correct here
        ends = [e for e in sim._events if e.kind == "game_end"]
        extra = [
            e
            for e in sim._events
            if e.kind == "round_start" and ends and e.sequence > ends[0].sequence
        ]
        if extra:
            return None  # engine did continue past the nominal end
        starts = [e for e in sim._events if e.kind == "round_start"]
        winds_seen = len(starts)
        return (
            f"manifest requires West entry: top={top} < return_points="
            f"{_MANIFEST.return_points} at match end yet engine emitted "
            f"game_end after {winds_seen} hands with no continuation hand"
        )

    result = _record(
        _runner().run_case(
            "WP04A-14b",
            "west entry required below return points (engine deviates)",
            ("all_last_policy", "sudden_death_policy"),
            (
                "recon-tenhou/evidence.md row28 RULE L1019-1021,L1063 (西入/サドンデス)",
                "recon-tenhou/evidence.md row26 (東南戦の西入あり)",
            ),
            wall_tiles=build_wall(hands={}, live_draws={}),
            script=(),
            expectations=[expect_predicate("west entry below return points", west_entry_required)],
            finish_to_terminal=True,
        )
    )
    _WAVE_C_RESULTS[result.case_id] = result
    assert result.status == "mismatch", (
        f"WP04A-14b: expected the documented engine deviation (mismatch), got {result.status}"
    )
    assert result.counterexample_path, "expected-mismatch case must persist its counterexample"


def test_wp04a_14c_tobi_score_injection_unavailable_blocked() -> None:
    """Tobi (<0 immediate end; tobi_policy='negative_points_immediate_end',
    bankruptcy_threshold=0, man L1022/L1077) cannot be exercised through the
    public simulator surface: reset() sources scores exclusively from
    rules.starting_points, and deterministic neutral play from 25000 never
    bankrupts a seat. BLOCKED(engine limitation) - documented mechanically."""
    from hydra2.engines.riichienv import RiichiEnvExactSimulator

    assert _PAYLOAD["tobi_policy"] == "negative_points_immediate_end"
    assert int(_PAYLOAD["bankruptcy_threshold"]) == 0
    assert int(_PAYLOAD["starting_points"]) == 25000
    params = inspect.signature(RiichiEnvExactSimulator.reset).parameters
    assert "scores" not in params and "starting_scores" not in params, (
        "adapter grew score injection; re-author a real tobi case"
    )
    _WAVE_C_RESULTS["WP04A-14c"] = CaseResult(
        case_id="WP04A-14c",
        title="tobi unreachable: no score injection through reset()",
        status="blocked",
        rule_fields=("tobi_policy", "bankruptcy_threshold", "starting_points"),
        evidence=(
            "recon-tenhou/evidence.md row25 RULE L1022, Q&A L1077 (飛び終了)",
            "RiichiEnvExactSimulator.reset(rules=, wall=, seat_permutation=) - no scores parameter",
        ),
    )


def test_wp04a_wave_c_disposition_summary() -> None:
    """Publishes the wave-C verdict lines: ID | verdict | 1-line evidence."""
    lines = []
    for case_id in ("WP04A-12", "WP04A-13", "WP04A-14a", "WP04A-14b", "WP04A-14c"):
        result = _WAVE_C_RESULTS.get(case_id)
        assert result is not None, f"{case_id} did not run"
        detail = (result.error_detail or "").split(";")[0][:80]
        lines.append(f"{case_id} | {result.status} | {detail}")
    print("\n".join(lines))
    assert len(_WAVE_C_RESULTS) == 5


_WAVE_C_RESULTS: dict[str, CaseResult] = {}

_PAYLOAD = _RULES_PAYLOAD["payload"]


def _record(result: CaseResult) -> CaseResult:
    _WAVE_C_RESULTS[result.case_id] = result
    return result


def _yame_wall() -> tuple[int, ...]:
    """Hand 0: seat3 rons a 5-han (sanshoku + 3 dora) mangan off seat1's
    opening tsumogiri; remaining junk cannot complete under neutral play."""
    hands = {
        0: {
            108: 1,
            109: 1,
            110: 1,
            111: 1,
            116: 1,
            117: 1,
            118: 1,
            120: 1,
            121: 1,
            122: 1,
            124: 1,
            125: 1,
            126: 1,
        },
        1: {
            112: 1,
            113: 1,
            114: 1,
            127: 1,
            128: 1,
            129: 1,
            123: 1,
            58: 1,
            62: 1,
            66: 1,
            102: 1,
            103: 1,
            106: 1,
        },
        2: {
            36: 1,
            37: 1,
            38: 1,
            56: 1,
            57: 1,
            60: 1,
            90: 1,
            91: 1,
            95: 1,
            98: 1,
            105: 1,
            119: 1,
            131: 1,
        },
        3: {
            4: 1,
            8: 1,
            12: 1,
            40: 1,
            44: 1,
            48: 1,
            76: 1,
            80: 1,
            84: 1,
            32: 1,
            33: 1,
            34: 1,
            132: 1,
        },
    }
    live = {52: 107, 53: 133}  # seat1 draws the chun mate and feeds seat3
    dead = {131: 30}  # dora indicator 8m -> 999m = 3 dora
    return build_wall(hands=hands, live_draws=live, dead_wall=dead)


_CHUN_FIRST = 132

_E_TILES = (108, 109, 110, 111)


def _sanchaho_wall() -> tuple[int, ...]:
    """Three closed tanki-on-East hands; seat3 feeds the fourth East copy."""
    hands = {
        0: {
            0: 1,
            4: 1,
            8: 1,
            52: 1,
            56: 1,
            60: 1,
            76: 1,
            80: 1,
            84: 1,
            92: 1,
            96: 1,
            100: 1,
            _E_TILES[0]: 1,
        },
        1: {
            5: 1,
            9: 1,
            12: 1,
            57: 1,
            61: 1,
            65: 1,
            81: 1,
            85: 1,
            89: 1,
            97: 1,
            101: 1,
            105: 1,
            _E_TILES[1]: 1,
        },
        2: {
            17: 1,
            21: 1,
            25: 1,
            36: 1,
            40: 1,
            44: 1,
            72: 1,
            73: 1,
            74: 1,
            93: 1,
            94: 1,
            95: 1,
            _E_TILES[2]: 1,
        },
        3: {
            1: 1,
            2: 1,
            3: 1,
            13: 1,
            14: 1,
            15: 1,
            29: 1,
            30: 1,
            31: 1,
            69: 1,
            70: 1,
            71: 1,
            113: 1,
        },
    }
    # Live: declarer flips are junk honors; seat3's first draw is East #4.
    live = {52: _CHUN_FIRST, 53: 128, 54: 124, 55: _E_TILES[3]}
    return build_wall(hands=hands, live_draws=live)


def test_wp04a_intersection_report_and_disposition() -> None:
    """Publishes the supported-rule report atomically and checks disposition:
    passed only when zero unresolved mismatches remain AFTER
    documented-unsupported resolution."""
    assert _RESULTS or _WAVE_C_RESULTS, "no cases registered - run the corpus tests first"
    merged = dict(_RESULTS)
    for cid, res in _WAVE_C_RESULTS.items():
        merged.setdefault(cid, res)
    results = [merged[cid] for cid in sorted(merged)]
    expected_ids = {
        "WP04A-01",
        "WP04A-02",
        "WP04A-03",
        "WP04A-04a",
        "WP04A-04b",
        "WP04A-05",
        "WP04A-06",
        "WP04A-07",
        "WP04A-08",
        "WP04A-09",
        "WP04A-10",
        "WP04A-11",
        "WP04A-12",
        "WP04A-13",
        "WP04A-14a",
        "WP04A-14b",
    }
    missing = expected_ids - set(merged)
    assert not missing, f"corpus coverage gap: {sorted(missing)}"
    document = build_intersection_report(
        rules_id="tenhou_4p_hanchan_v1",
        rules_manifest_sha256=(
            "sha256:3042a493280224f533d831f371275b1c96585cf1db5a2e5fb86ec259f403286b"
        ),
        results=results,
        documented_unsupported=DOCUMENTED_UNSUPPORTED,
    )
    run_id = time.strftime("%Y%m%dT%H%M%S%fZ", time.gmtime())
    destination = artifact_root() / "reports" / "WP-04A" / run_id / "report.json"
    write_intersection_report(document, destination)
    assert destination.is_file()
    assert document["tally"]["mismatch"] == len(document["unresolved_mismatch_cases"])
    assert document["declared_support"]["verdict"] == (
        "supported" if not document["unresolved_mismatch_cases"] else "blocked"
    )
