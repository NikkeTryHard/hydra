"""WP-04A reference corpus: abortive draws and match end (cases 12-14).

Triple-ron abort, rank and uma accounting, and the all-last continuation
branches replayed through the WP-03A reference adapter; the 12-14
disposition summary lives here with the cases it aggregates.
"""

from __future__ import annotations

import inspect

import pytest

from hydra2.conformance.runner import CaseResult, ScriptedDecision, expect_predicate
from hydra2.conformance.walls import build_wall
from hydra2.contracts.common import ContractError
from hydra2.contracts.rules import resolve_final_ranks
from hydra2.contracts.utility import (
    UTILITY_OBJECTIVE,
    UTILITY_TIE_POLICY,
    RawOutcome,
    make_utility_manifest,
    root_scalar,
    utility,
)
from tests.conformance.test_reference_corpus_wp04a import (
    _MANIFEST,
    _PAYLOAD,
    _WAVE_C_RESULTS,
    _do,
    _record,
    _runner,
    assert_supported,
)

pytestmark = pytest.mark.contract_package("WP-04A")


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
        # Engine 0.4.10 deducts the 1000 stick from scores() at declaration
        # (upstream deposit rework #231/#232; matches Tenhou, where the score
        # display drops the moment riichi is declared). The abort therefore
        # reports zero score movement; the netting is locked here via the
        # abortive payload scores and below via the kyotaku carry + boundary.
        score_deltas = [d.value for d in env.public_delta if tuple(d.path) == ("scores",)]
        if not score_deltas or [int(v) for v in score_deltas[0]] != [0, 0, 0, 0]:
            return (
                "abortive deltas must report no score movement (sticks netted at "
                f"declaration since engine 0.4.10) and pay the feeder nothing, got {score_deltas}"
            )
        if tuple(int(s) for s in env.payload.scores) != (24000, 24000, 24000, 25000):
            return (
                "abortive payload scores must leave each declarer one stick down "
                f"and the feeder untouched, got {tuple(int(s) for s in env.payload.scores)}"
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
    RiichiEnv 0.4.10 YON_HANCHAN instead emits game_end 'hanchan_complete' -
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
