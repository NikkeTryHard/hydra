"""SPEC 5.1 Tenhou rules vocabulary and contract-local canonical JSON.

Owns the evidence-fixed vocabulary every rules manifest builds on: the
``tenhou_4p_hanchan_v1`` identity constants, the closed policy-enum
inventories, the payload field list, and the small field validators shared
by the manifest dataclasses. The RFC 8785 writer below is a thin delegate
to :mod:`hydra2.artifacts.canonical` (THE single authority); tests prove
byte-identity, never a second implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ABORTIVE_DRAW_KINDS",
    "ADAPTER_COMPATIBILITY_STATUSES",
    "AGARI_YAME_POLICIES",
    "ALL_LAST_POLICIES",
    "CHANKAN_POLICIES",
    "FAST_CLOCK_SECONDS",
    "FURITEN_POLICIES",
    "KAN_DORA_REVEAL_POLICIES",
    "KAN_URA_POLICIES",
    "KAZOE_POLICIES",
    "KUIKAE_POLICIES",
    "KUITAN_VALUES",
    "MULTIPLE_RON_POLICIES",
    "OKA_POLICIES",
    "PAO_POLICIES",
    "PLACEMENT_CONVERSION_IDS",
    "RANK_TIE_BREAKS",
    "RED_TILE_IDS",
    "RETURN_POINTS",
    "RIICHI_STICK_ALLOCATIONS",
    "RINSHAN_POLICIES",
    "RULES_ID",
    "RULES_MANIFEST_PAYLOAD_FIELDS",
    "SOURCE_EVIDENCE_KEY",
    "STANDARD_CLOCK_SECONDS",
    "SUDDEN_DEATH_POLICIES",
    "TENHOU_ABORTIVE_DRAWS",
    "TOBI_POLICIES",
    "YAKUMAN_POLICIES",
    "canonical_contract_json_bytes",
    "canonical_contract_json_text",
]

# ---------------------------------------------------------------------------
# Identity constants fixed by SPEC 5.1 for tenhou_4p_hanchan_v1.
# ---------------------------------------------------------------------------

RULES_ID = "tenhou_4p_hanchan_v1"
STARTING_POINTS = 25000
RETURN_POINTS = 30000
#: SPEC §4.1/§5.1: exactly these three physical tile ids are red fives.
RED_TILE_IDS = (16, 52, 88)

#: Standard table clock 「普 5+10秒」 and fast clock 「速 3+5秒」
#: (main think seconds + reserve time-bank seconds, man.html TT L998-999).
STANDARD_CLOCK_SECONDS = (5, 10)
FAST_CLOCK_SECONDS = (3, 5)


def _require_int(value: int, *, name: str, minimum: int, maximum: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum or (maximum is not None and value > maximum):
        raise ContractError(
            f"{name}={value} outside [{minimum}, {maximum if maximum is not None else '∞'}]"
        )
    return value


def _require_str(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a str, got {type(value).__name__}")
    return value


def _require_bool(value: bool, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_enum(value: str, *, name: str, allowed: tuple[str, ...] | frozenset[str]) -> str:
    text = _require_str(value, name=name)
    if text not in allowed:
        raise ContractError(f"{name}={text!r} must be one of {sorted(allowed)}")
    return text


def _require_quad_ints(
    values: Sequence[int], *, name: str, minimum: int, maximum: int | None
) -> tuple[int, int, int, int]:
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        raise ContractError(f"{name} must be a sequence of exactly 4 ints")
    quad = tuple(
        _require_int(item, name=f"{name}[{i}]", minimum=minimum, maximum=maximum)
        for i, item in enumerate(values)
    )
    first, second, third, fourth = quad
    return (first, second, third, fourth)


# ---------------------------------------------------------------------------
# Policy enums established from Tenhou source evidence (WP-02B).
#
# Single-member enums record the unique evidence-fixed behavior; extension
# follows the SPEC §24 contract-change procedure. Where the source page itself
# presents selectable alternatives, both members are declared.
# ---------------------------------------------------------------------------

#: Oka (top-place bonus pool). Ranked Tenhou defines none; owner decision d1.
OKA_POLICIES = ("none", "half_return")
#: Open tanyao. man.html L1016 lists 喰い断なし/あり as selectable; ranked value
#: fixed by owner decision d4 (allowed).
KUITAN_VALUES = ("allowed", "forbidden")
#: Melded-tile swap (kuikae) forbidden since 2007-11-29 (L1015/L1056).
KUIKAE_POLICIES = ("forbidden", "permitted")
#: River-only judgment, permanent furiten after riichi win-refusal,
#: same-go-around temporary furiten after any win refusal incl. chankan
#: (L1040-1042/L1052-1053/Q&A L1088).
FURITEN_POLICIES = ("river_only_permanent_after_riichi_miss_same_goaround_temporary",)
#: Chankan permitted as a yaku (L1177); kokushi cannot ron an ankan tile.
CHANKAN_POLICIES = ("permitted",)
#: Replacement draws come from a 14-tile dead wall (owner decision d2; page
#: states only the 3-player count 「嶺上牌は8枚」 L1098).
RINSHAN_POLICIES = ("dead_wall_14",)
#: Ankan kan-dora immediate (dora = bonus-indicator tiles); open meld/kakan
#: revealed after discard or just before following rinshan draw (L1045).
KAN_DORA_REVEAL_POLICIES = ("ankan_immediate_open_delayed",)
#: Kan ura-dora exists (L1044); reveal timing unstated (near-gap recorded).
KAN_URA_POLICIES = ("present",)
#: Pao for Big Three Dragons and Big Four Winds: tsumo payer pays all, ron
#: half; honba charged to pao bearer; no suukantsu pao (L1035-1036/L1054).
PAO_POLICIES = ("daisangen_daisuishi_tsumo_full_ron_half",)
#: Yakuman compound; upgraded forms (suuankou tanki, kokushi 13-wait) single
#: yakuman (YAKU L1252).
YAKUMAN_POLICIES = ("compound_multiple_upgraded_forms_single",)
#: Counted yakuman at 13+ han (L1254).
KAZOE_POLICIES = ("counted_yakuman_at_13_han",)
#: Double ron: every winner paid (no atamahane), honba AND riichi sticks to
#: winner nearest dealer's left, dealer-included ron renchains (L1033-1034).
MULTIPLE_RON_POLICIES = ("all_winners_paid_sticks_to_dealer_left",)
#: End-of-game remaining sticks to 1st place (L1024); dealing into the riichi
#: declaration tile deposits nothing (L1039); abortive-draw sticks remain
#: deposited into the renchan hand (owner decision d3; refund unstated L1029).
RIICHI_STICK_ALLOCATIONS = ("end_top_take_abort_carry_dealin_exempt",)
ABORTIVE_DRAW_KINDS = frozenset(
    {
        "kyuushu_kyuuhai",
        "suucha_riichi",
        "sanchahou",
        "suukaikan",
        "suufon_renda",
    }
)
#: All five abortive draws exist and EVERY one renchains (L1029-1030); order
#: follows the source listing 九種九牌/四家立直/三家和了/四槓散了/四風連打.
TENHOU_ABORTIVE_DRAWS = (
    "kyuushu_kyuuhai",
    "suucha_riichi",
    "sanchahou",
    "suukaikan",
    "suufon_renda",
)
#: South-entry (tonpuu) / West-entry (hanchan) continuation with renchan
#: extension until dealer-top-over-return or sudden-death cap (L1018-1019,
#: Q&A L1080-1083).
ALL_LAST_POLICIES = ("south_west_entry_renchan_extension",)
#: Last-hand dealer-top auto win-stop and auto tenpai-stop (tenpai stop since
#: 2010-06-01; L1023/L1059).
AGARI_YAME_POLICIES = ("dealer_top_auto_win_and_tenpai_stop",)
#: Tobi (bankruptcy) ends the game the instant a seat goes below zero points;
#: exactly 0 continues and negative totals still appear in results
#: (L1022 「飛び終了あり。点数がマイナスで飛び終了、マイナス点数も集計、0点は続行」,
#: Q&A L1077 「飛びや天辺が発生した場合はその時点で終了します」).
TOBI_POLICIES = ("negative_points_immediate_end",)
#: Sudden death at >= return points excluding deposited riichi sticks; dealer
SUDDEN_DEATH_POLICIES = ("ge_return_points_excluding_sticks_dealer_priority",)
#: Final ties broken by seat-wind order of East-1 (dealer first, clockwise;
#: L1025). Seats align East..North in seat order, so lower seat wins.
RANK_TIE_BREAKS = ("east1_seat_wind_order",)
#: Placement conversion pipeline: rank by raw final score (2022 rounding
#: abolition L1060), leftover sticks to top (L1024), then uma 10-20 applied.
PLACEMENT_CONVERSION_IDS = ("tenhou_rank_sticks_top_uma_v1",)

ADAPTER_COMPATIBILITY_STATUSES = ("supported", "unsupported", "qualified")

#: Payload keys serialized for a RulesManifest plus the declared extra key.
RULES_MANIFEST_PAYLOAD_FIELDS: tuple[str, ...] = (
    "rules_id",
    "source",
    "players",
    "match_length",
    "starting_points",
    "return_points",
    "uma_by_rank",
    "oka_policy",
    "kuitan",
    "red_tile_ids",
    "clocks",
    "kuikae_policy",
    "furiten_policy",
    "chankan_policy",
    "rinshan_policy",
    "kan_dora_reveal_policy",
    "kan_ura_policy",
    "pao_policy",
    "yakuman_policy",
    "kazoe_policy",
    "multiple_ron_policy",
    "riichi_stick_allocation",
    "abortive_draws",
    "nagashi_mangan",
    "bankruptcy_threshold",
    "all_last_policy",
    "agari_yame_policy",
    "tobi_policy",
    "sudden_death_policy",
    "rank_tie_break",
    "placement_conversion_id",
    "adapter_compatibility",
)
SOURCE_EVIDENCE_KEY = "source_evidence"


# ---------------------------------------------------------------------------
# Canonical JSON — thin delegate to hydra2.artifacts.canonical (single authority).
# No second implementation lives here; byte-identity is proven by contract tests.
# Lazy import keeps the contracts layer importable without eagerly binding artifacts.
# ---------------------------------------------------------------------------


def canonical_contract_json_text(value: Any) -> str:
    """RFC 8785 canonical JSON text; delegates to artifacts.canonical."""
    from hydra2.artifacts.canonical import canonicalize as _authority_text

    return _authority_text(value)


def canonical_contract_json_bytes(value: Any) -> bytes:
    """RFC 8785 canonical UTF-8 bytes; identical to artifacts.canonical output."""
    from hydra2.artifacts.canonical import canonical_bytes as _authority_bytes

    return _authority_bytes(value)
