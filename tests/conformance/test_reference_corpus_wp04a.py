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

import json
import time
from functools import cache
from pathlib import Path

import pytest

from hydra2.conformance.report import build_intersection_report, write_intersection_report
from hydra2.conformance.runner import (
    CaseResult,
    ReferenceTraceRunner,
    ScriptedDecision,
    expect_predicate,
)
from hydra2.conformance.walls import build_wall
from hydra2.contracts.rules import rules_manifest_from_payload

pytestmark = pytest.mark.contract_package("WP-04A")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RULES_PAYLOAD = json.loads(
    (_REPO_ROOT / "configs" / "rules" / "tenhou_4p_hanchan_v1.json").read_text()
)
_MANIFEST = rules_manifest_from_payload(_RULES_PAYLOAD["payload"])

DOCUMENTED_UNSUPPORTED: dict[str, str] = {
    "suufon_renda": (
        "RiichiEnv 0.4.10 never emits the four-winds abortive ryukyoku: four "
        "first-turn own-wind discards leave the hand running (probe DUT-1)."
    ),
    "scoring_tables": (
        "owner_decision D1: RiichiEnv 0.4.10 omits the dealer-discarder x2 "
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


def _run(case_id, title, rule_fields, evidence, runner=None, **kwargs) -> CaseResult:
    result = _run_case(
        runner if runner is not None else _runner(), case_id, title, rule_fields, evidence, **kwargs
    )
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


def test_wp04a_01_fifth_dora_and_kan_ura_timing(runner=None) -> None:
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
        runner=runner,
    )
    assert_supported(result, "WP04A-01")


# ---------------------------------------------------------------------------
# WP04A-02 chankan + rinshan payout (D-WP04A-FIX1/FIX4b regression).
# ---------------------------------------------------------------------------


def test_wp04a_02_chankan_and_rinshan_payout(runner=None) -> None:
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
        runner=runner,
    )
    assert_supported(result, "WP04A-02")


# ---------------------------------------------------------------------------
# WP04A-11 suufon_renda: documented-unsupported counterexample.
# ---------------------------------------------------------------------------


def test_wp04a_11_suufon_renda_documented_unsupported(runner=None) -> None:
    """Four first-turn own-wind discards must abort (suufon_renda is a declared
    manifest rule); RiichiEnv 0.4.10 keeps playing. EXPECTED mismatch resolved
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
        runner=runner,
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


_WAVE_C_RESULTS: dict[str, CaseResult] = {}

_PAYLOAD = _RULES_PAYLOAD["payload"]


def _record(result: CaseResult) -> CaseResult:
    _WAVE_C_RESULTS[result.case_id] = result
    return result


def test_wp04a_intersection_report_and_disposition(tmp_path) -> None:
    """Publishes the supported-rule report atomically and checks disposition:
    passed only when zero unresolved mismatches remain AFTER
    documented-unsupported resolution."""
    # Self-contained by design: rebuild every case in-process through a
    # worker-local runner (counterexamples persist under tmp_path, never
    # the shared artifact root; report publishes under tmp_path — two
    # workers sharing a second-stamp run_id never clobber). Under xdist
    # loadscope each file pins its own worker so cross-file globals never
    # merge (proven: -n 4 fails with coverage gap). Every case test accepts
    # a runner override forwarded to _run/_record (defaults keep standalone
    # runs on the shared runner, unchanged).
    from tests.conformance import test_reference_corpus_claims_wp04a as _claims
    from tests.conformance import test_reference_corpus_scoring_wp04a as _scoring
    from tests.conformance import test_reference_corpus_terminals_wp04a as _terminals

    _local_runner = ReferenceTraceRunner(manifest=_MANIFEST, artifact_root_path=tmp_path)
    test_wp04a_01_fifth_dora_and_kan_ura_timing(runner=_local_runner)
    test_wp04a_02_chankan_and_rinshan_payout(runner=_local_runner)
    test_wp04a_11_suufon_renda_documented_unsupported(runner=_local_runner)
    _claims.test_wp04a_03_kuikae_post_pon_same_meld_swap_barred(runner=_local_runner)
    _claims.test_wp04a_04a_temp_furiten_clears_then_ron_lands(runner=_local_runner)
    _claims.test_wp04a_04b_permanent_furiten_after_riichi_miss(runner=_local_runner)
    _claims.test_wp04a_05_double_ron_priority_packets_upstream_first(runner=_local_runner)
    _claims.test_wp04a_06_multi_ron_sticks_upstream_with_dealer_co_winner(runner=_local_runner)
    _scoring.test_wp04a_07_red_five_scoring(runner=_local_runner)
    _scoring.test_wp04a_08_pao_liability_split_and_kazoe(runner=_local_runner)
    _scoring.test_wp04a_09_kyuushu_kyuuhai_abort(runner=_local_runner)
    _scoring.test_wp04a_10_exhaustive_draw_noten_split(runner=_local_runner)
    _terminals.test_wp04a_12_sanchahou_triple_ron_abort(runner=_local_runner)
    _terminals.test_wp04a_13_rank_tie_break_and_uma_utility(runner=_local_runner)
    _terminals.test_wp04a_14a_all_last_dealer_tenpai_stop_yame(runner=_local_runner)
    _terminals.test_wp04a_14b_west_entry_sudden_death_expected_mismatch(runner=_local_runner)
    _terminals.test_wp04a_14c_tobi_score_injection_unavailable_blocked(runner=_local_runner)
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
    destination = tmp_path / "reports" / "WP-04A" / run_id / "report.json"
    write_intersection_report(document, destination)
    assert destination.is_file()
    assert document["tally"]["mismatch"] == len(document["unresolved_mismatch_cases"])
    assert document["declared_support"]["verdict"] == (
        "supported" if not document["unresolved_mismatch_cases"] else "blocked"
    )
