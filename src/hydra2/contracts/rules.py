"""SPEC 5.1 Tenhou rules manifest — WP-02B contract module.

Defines the immutable :class:`RulesManifest` vocabulary for
``tenhou_4p_hanchan_v1`` (decision D-002: tenhou.net/man is the sole initial
rules authority). Every policy string value MUST come from the enum constants
declared here; unknown source behavior blocks manifest publication and engine
defaults NEVER fill fields (BUILD WP-02B checklist).

Evidence base:
- Primary snapshot ``man_curl.html``
  sha256:3de07f2c338caf520f6c4dc2160270adb7b1fbd25c1666bf61d3f351902f210a
  (byte-exact curl capture of https://tenhou.net/man/, retrieved
  2026-08-22T22:31:52Z; rendered fast_fetch_raw copy sha256:e52f1469...).
- Field-by-field quotes: Private evidence (D-002), see attestation metadata.
- Owner decisions (no on-page evidence) are encoded as explicit provenance
  entries in the published payload, never silent defaults.

Contracts import stdlib + contracts only (SPEC §1), so this module also hosts
the contract-local canonical JSON writer used for identity payloads. It mirrors
the RFC 8785 semantics of ``hydra2.artifacts.canonical`` on the JSON domain;
tests prove byte-equality against that authority.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hydra2.contracts.common import (
    CanonicalizationError,
    ContractError,
    DigestText,
    TileId,
    UtcTimestamp,
    make_digest_text,
    make_tile_id,
    make_utc_timestamp,
)

__all__ = [
    "ABORTIVE_DRAW_KINDS",
    "ADAPTER_COMPATIBILITY_STATUSES",
    "AGARI_YAME_POLICIES",
    "ALL_LAST_POLICIES",
    "CHANKAN_POLICIES",
    "FAST_CLOCK",
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
    "STANDARD_CLOCK",
    "STANDARD_CLOCK_SECONDS",
    "SUDDEN_DEATH_POLICIES",
    "TENHOU_ABORTIVE_DRAWS",
    "TOBI_POLICIES",
    "YAKUMAN_POLICIES",
    "AdapterCompatibility",
    "ClockRule",
    "RulesManifest",
    "SourceAuthority",
    "canonical_contract_json_bytes",
    "canonical_contract_json_text",
    "manifest_to_payload",
    "resolve_final_ranks",
    "rules_manifest_from_payload",
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
#: Ankan kan-dora immediate; open meld/kakan revealed after discard or just
#: before following rinshan draw (L1045).
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

_TENHOU_MAN_URL = "https://tenhou.net/man/"


# ---------------------------------------------------------------------------
# Contract-local canonical JSON (RFC 8785 JCS subset over the JSON domain).
# Mirrors hydra2.artifacts.canonical byte-for-byte; contracts may not import
# artifacts (SPEC §1), so the writer lives here and tests prove equivalence.
# ---------------------------------------------------------------------------

_MAX_SAFE_INTEGER = 2**53 - 1

_SHORT_ESCAPES = {
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0C: "\\f",
    0x0D: "\\r",
}


def _es6_number_to_string(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise CanonicalizationError(f"non-finite number {value!r} has no canonical serialization")
    if value == 0:  # -0.0 -> "0" (RFC 8785 Appendix B row 2)
        return "0"
    sign = "-" if value < 0 else ""
    mantissa, _, exponent_text = repr(abs(value)).partition("e")
    exponent10 = int(exponent_text) if exponent_text != "" else 0
    integer_part, _, fraction_part = mantissa.partition(".")
    raw_digits = integer_part + fraction_part
    _stripped = raw_digits.lstrip("0").rstrip("0")
    digits = _stripped if _stripped != "" else "0"
    k = len(digits)
    trailing_stripped = len(raw_digits) - len(raw_digits.rstrip("0"))
    n = k + trailing_stripped + exponent10 - len(fraction_part)
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    scientific_exponent = n - 1
    exponent_form = (
        f"e+{scientific_exponent}" if scientific_exponent >= 0 else f"e{scientific_exponent}"
    )
    head = digits[0] + ("." + digits[1:] if k > 1 else "")
    return sign + head + exponent_form


def _serialize_string(text: str) -> str:
    try:
        _ = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CanonicalizationError(
            f"string contains an unpaired surrogate (invalid Unicode): {text!r}"
        ) from exc
    pieces: list[str] = ['"']
    for char in text:
        code = ord(char)
        if code in _SHORT_ESCAPES:
            pieces.append(_SHORT_ESCAPES[code])
        elif code < 0x20:
            pieces.append(f"\\u{code:04x}")
        elif char == '"':
            pieces.append('\\"')
        elif char == "\\":
            pieces.append("\\\\")
        else:
            pieces.append(char)
    pieces.append('"')
    return "".join(pieces)


def _serialize(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise CanonicalizationError(
                f"integer {value} exceeds the IEEE 754 double-safe range "
                f"(±{_MAX_SAFE_INTEGER}); serialize it as a float or string"
            )
        return str(value)
    if isinstance(value, float):
        return _es6_number_to_string(value)
    if isinstance(value, str):
        return _serialize_string(value)
    if isinstance(value, list):
        return "[" + ",".join(_serialize(item) for item in value) + "]"  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise CanonicalizationError(
                    f"object key {key!r} is not a string; JSON objects are string-keyed only"
                )
        ordered_keys = sorted(value, key=lambda key: key.encode("utf-16-be"))  # RFC 8785 §3.2.3
        members = (f"{_serialize_string(key)}:{_serialize(value[key])}" for key in ordered_keys)  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
        return "{" + ",".join(members) + "}"
    raise CanonicalizationError(
        f"value of type {type(value).__name__} is outside the canonical JSON domain"
    )


def canonical_contract_json_text(value: Any) -> str:
    """RFC 8785 canonical JSON text over the JSON domain (contract-local)."""
    return _serialize(value)


def canonical_contract_json_bytes(value: Any) -> bytes:
    """RFC 8785 canonical UTF-8 bytes; identical to artifacts.canonical output."""
    return canonical_contract_json_text(value).encode("utf-8")


# ---------------------------------------------------------------------------
# SPEC 5.1 dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceAuthority:
    """Reviewed upstream snapshot identity (SPEC 5.1)."""

    url: str
    retrieved_at_utc: UtcTimestamp
    content_sha256: DigestText

    def __post_init__(self) -> None:
        url = _require_str(self.url, name="url")
        if not url.startswith(("http://", "https://")):
            raise ContractError(f"url must be an http(s) URL, got {url!r}")
        object.__setattr__(self, "retrieved_at_utc", make_utc_timestamp(self.retrieved_at_utc))
        object.__setattr__(self, "content_sha256", make_digest_text(self.content_sha256))


@dataclass(frozen=True, slots=True)
class ClockRule:
    """Think-clock: base per-turn seconds + reserve time-bank seconds."""

    base_seconds: int
    increment_seconds: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_seconds",
            _require_int(self.base_seconds, name="base_seconds", minimum=1, maximum=None),
        )
        object.__setattr__(
            self,
            "increment_seconds",
            _require_int(self.increment_seconds, name="increment_seconds", minimum=1, maximum=None),
        )


@dataclass(frozen=True, slots=True)
class AdapterCompatibility:
    """Per-manhash adapter declaration (SPEC 5.1: supported/unsupported/qualified)."""

    adapter_id: str
    status: str
    rules_hash: DigestText

    def __post_init__(self) -> None:
        adapter_id = _require_str(self.adapter_id, name="adapter_id")
        if adapter_id == "":
            raise ContractError("adapter_id must be non-empty")
        object.__setattr__(
            self,
            "status",
            _require_enum(self.status, name="status", allowed=ADAPTER_COMPATIBILITY_STATUSES),
        )
        object.__setattr__(self, "rules_hash", make_digest_text(self.rules_hash))


def _validated_clock_tuple(clocks: Sequence[ClockRule]) -> tuple[ClockRule, ...]:
    if not isinstance(clocks, (tuple, list)) or len(clocks) == 0:
        raise ContractError("clocks must be a non-empty sequence of ClockRule")
    for i, clock in enumerate(clocks):
        if not isinstance(clock, ClockRule):
            raise ContractError(f"clocks[{i}] must be a ClockRule, got {type(clock).__name__}")
    return tuple(clocks)


@dataclass(frozen=True, slots=True, kw_only=True)
class RulesManifest:
    """Complete scoring/match rule set for tenhou_4p_hanchan_v1 (SPEC 5.1).

    Every field is required; strict validation rejects missing flags so engine
    defaults can never fill them. Policy strings must belong to the enums
    declared in this module (Tenhou evidence-backed).
    """

    rules_id: str
    source: SourceAuthority
    players: int
    match_length: str
    starting_points: int
    return_points: int
    uma_by_rank: tuple[int, int, int, int]
    oka_policy: str
    kuitan: bool
    red_tile_ids: tuple[TileId, ...]
    clocks: tuple[ClockRule, ...]
    kuikae_policy: str
    furiten_policy: str
    chankan_policy: str
    rinshan_policy: str
    kan_dora_reveal_policy: str
    kan_ura_policy: str
    pao_policy: str
    yakuman_policy: str
    kazoe_policy: str
    multiple_ron_policy: str
    riichi_stick_allocation: str
    abortive_draws: tuple[str, ...]
    nagashi_mangan: bool
    bankruptcy_threshold: int
    all_last_policy: str
    agari_yame_policy: str
    tobi_policy: str
    sudden_death_policy: str
    rank_tie_break: str
    placement_conversion_id: str
    adapter_compatibility: tuple[AdapterCompatibility, ...]

    def __post_init__(self) -> None:
        if self.rules_id != RULES_ID:
            raise ContractError(f"rules_id must be {RULES_ID!r}, got {self.rules_id!r}")
        if not isinstance(self.source, SourceAuthority):
            raise ContractError("source must be a SourceAuthority")
        object.__setattr__(
            self, "players", _require_int(self.players, name="players", minimum=2, maximum=4)
        )
        if self.players != 4:
            raise ContractError(f"players must be 4 for {RULES_ID}, got {self.players}")
        if self.match_length != "hanchan":
            raise ContractError(f"match_length must be 'hanchan', got {self.match_length!r}")
        object.__setattr__(
            self,
            "starting_points",
            _require_int(self.starting_points, name="starting_points", minimum=0, maximum=None),
        )
        if self.starting_points != STARTING_POINTS:
            raise ContractError(
                f"starting_points must be {STARTING_POINTS}, got {self.starting_points}"
            )
        object.__setattr__(
            self,
            "return_points",
            _require_int(self.return_points, name="return_points", minimum=0, maximum=None),
        )
        if self.return_points != RETURN_POINTS:
            raise ContractError(f"return_points must be {RETURN_POINTS}, got {self.return_points}")
        object.__setattr__(
            self,
            "uma_by_rank",
            _require_quad_ints(
                self.uma_by_rank, name="uma_by_rank", minimum=-(10**9), maximum=10**9
            ),
        )
        object.__setattr__(
            self,
            "oka_policy",
            _require_enum(self.oka_policy, name="oka_policy", allowed=OKA_POLICIES),
        )
        object.__setattr__(self, "kuitan", _require_bool(self.kuitan, name="kuitan"))
        red_ids = tuple(make_tile_id(item) for item in self.red_tile_ids)
        if red_ids != RED_TILE_IDS:
            raise ContractError(f"red_tile_ids must be exactly {RED_TILE_IDS}, got {red_ids}")
        object.__setattr__(self, "red_tile_ids", red_ids)
        clocks = _validated_clock_tuple(self.clocks)
        expected_clocks = (
            ClockRule(*STANDARD_CLOCK_SECONDS),
            ClockRule(*FAST_CLOCK_SECONDS),
        )
        if clocks != expected_clocks:
            raise ContractError(
                f"clocks must be standard {STANDARD_CLOCK_SECONDS} then fast "
                f"{FAST_CLOCK_SECONDS} as (base, increment), got {clocks}"
            )
        object.__setattr__(self, "clocks", clocks)
        object.__setattr__(
            self,
            "kuikae_policy",
            _require_enum(self.kuikae_policy, name="kuikae_policy", allowed=KUIKAE_POLICIES),
        )
        object.__setattr__(
            self,
            "furiten_policy",
            _require_enum(self.furiten_policy, name="furiten_policy", allowed=FURITEN_POLICIES),
        )
        object.__setattr__(
            self,
            "chankan_policy",
            _require_enum(self.chankan_policy, name="chankan_policy", allowed=CHANKAN_POLICIES),
        )
        object.__setattr__(
            self,
            "rinshan_policy",
            _require_enum(self.rinshan_policy, name="rinshan_policy", allowed=RINSHAN_POLICIES),
        )
        object.__setattr__(
            self,
            "kan_dora_reveal_policy",
            _require_enum(
                self.kan_dora_reveal_policy,
                name="kan_dora_reveal_policy",
                allowed=KAN_DORA_REVEAL_POLICIES,
            ),
        )
        object.__setattr__(
            self,
            "kan_ura_policy",
            _require_enum(self.kan_ura_policy, name="kan_ura_policy", allowed=KAN_URA_POLICIES),
        )
        object.__setattr__(
            self,
            "pao_policy",
            _require_enum(self.pao_policy, name="pao_policy", allowed=PAO_POLICIES),
        )
        object.__setattr__(
            self,
            "yakuman_policy",
            _require_enum(self.yakuman_policy, name="yakuman_policy", allowed=YAKUMAN_POLICIES),
        )
        object.__setattr__(
            self,
            "kazoe_policy",
            _require_enum(self.kazoe_policy, name="kazoe_policy", allowed=KAZOE_POLICIES),
        )
        object.__setattr__(
            self,
            "multiple_ron_policy",
            _require_enum(
                self.multiple_ron_policy, name="multiple_ron_policy", allowed=MULTIPLE_RON_POLICIES
            ),
        )
        object.__setattr__(
            self,
            "riichi_stick_allocation",
            _require_enum(
                self.riichi_stick_allocation,
                name="riichi_stick_allocation",
                allowed=RIICHI_STICK_ALLOCATIONS,
            ),
        )
        if not isinstance(self.abortive_draws, (tuple, list)) or len(self.abortive_draws) == 0:
            raise ContractError("abortive_draws must be a non-empty tuple of declared kinds")
        draws = tuple(
            _require_enum(draw, name="abortive_draw", allowed=ABORTIVE_DRAW_KINDS)
            for draw in self.abortive_draws
        )
        if len(set(draws)) != len(draws):
            raise ContractError("abortive_draws must not repeat kinds")
        if draws != TENHOU_ABORTIVE_DRAWS:
            raise ContractError(
                f"abortive_draws must record all five declared kinds in source order "
                f"{TENHOU_ABORTIVE_DRAWS}, got {draws}"
            )
        object.__setattr__(self, "abortive_draws", draws)
        object.__setattr__(
            self, "nagashi_mangan", _require_bool(self.nagashi_mangan, name="nagashi_mangan")
        )
        object.__setattr__(
            self,
            "bankruptcy_threshold",
            _require_int(
                self.bankruptcy_threshold,
                name="bankruptcy_threshold",
                minimum=-(10**12),
                maximum=10**12,
            ),
        )
        object.__setattr__(
            self,
            "all_last_policy",
            _require_enum(self.all_last_policy, name="all_last_policy", allowed=ALL_LAST_POLICIES),
        )
        object.__setattr__(
            self,
            "agari_yame_policy",
            _require_enum(
                self.agari_yame_policy, name="agari_yame_policy", allowed=AGARI_YAME_POLICIES
            ),
        )
        object.__setattr__(
            self,
            "tobi_policy",
            _require_enum(self.tobi_policy, name="tobi_policy", allowed=TOBI_POLICIES),
        )
        object.__setattr__(
            self,
            "sudden_death_policy",
            _require_enum(
                self.sudden_death_policy, name="sudden_death_policy", allowed=SUDDEN_DEATH_POLICIES
            ),
        )
        object.__setattr__(
            self,
            "rank_tie_break",
            _require_enum(self.rank_tie_break, name="rank_tie_break", allowed=RANK_TIE_BREAKS),
        )
        object.__setattr__(
            self,
            "placement_conversion_id",
            _require_enum(
                self.placement_conversion_id,
                name="placement_conversion_id",
                allowed=PLACEMENT_CONVERSION_IDS,
            ),
        )
        adapters: list[AdapterCompatibility] = []
        seen_adapters: set[str] = set()
        if not isinstance(self.adapter_compatibility, (tuple, list)):
            raise ContractError("adapter_compatibility must be a tuple of AdapterCompatibility")
        for i, adapter in enumerate(self.adapter_compatibility):
            if not isinstance(adapter, AdapterCompatibility):
                raise ContractError(
                    f"adapter_compatibility[{i}] must be AdapterCompatibility, "
                    f"got {type(adapter).__name__}"
                )
            if adapter.adapter_id in seen_adapters:
                raise ContractError(f"duplicate adapter_id {adapter.adapter_id!r}")
            seen_adapters.add(adapter.adapter_id)
            adapters.append(adapter)
        object.__setattr__(self, "adapter_compatibility", tuple(adapters))


# ---------------------------------------------------------------------------
# Rank resolution with the encoded tie-break (BUILD checklist item).
# ---------------------------------------------------------------------------


def resolve_final_ranks(final_scores: Sequence[int]) -> tuple[int, int, int, int]:
    """Resolve placement ranks 1..4 from raw final scores.

    Encodes rank_tie_break="east1_seat_wind_order": equal scores are ordered by
    seat wind order of East-1 (man.html L1025 「終了時に同点の場合は東1局の風順で順位を決定」).
    Seats 0..3 align with East/South/West/North winds (SPEC §8), so the lower
    seat index takes the better rank on ties.
    """
    scores = _require_quad_ints(
        final_scores, name="final_scores", minimum=-(10**12), maximum=10**12
    )
    order = sorted(range(4), key=lambda seat: (-scores[seat], seat))  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
    ranks = [0, 0, 0, 0]
    for position, seat in enumerate(order):
        ranks[seat] = position + 1
    return (ranks[0], ranks[1], ranks[2], ranks[3])


# ---------------------------------------------------------------------------
# Payload codec (configs/rules JSON payload <-> manifest).
# ---------------------------------------------------------------------------


def manifest_to_payload(
    manifest: RulesManifest, *, source_evidence: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Serialize a manifest to its JSON payload dict (SPEC 5.1 fields).

    ``source_evidence`` is the only declared extra key; it carries per-field
    authority references and owner decisions so no value is a silent default.
    """
    if source_evidence is not None and not isinstance(source_evidence, Mapping):
        raise ContractError("source_evidence must be a mapping when provided")
    payload: dict[str, Any] = {}
    for name in RULES_MANIFEST_PAYLOAD_FIELDS:
        value: Any = getattr(manifest, name)
        if name == "source":
            value = {
                "url": manifest.source.url,
                "retrieved_at_utc": manifest.source.retrieved_at_utc,
                "content_sha256": manifest.source.content_sha256,
            }
        elif name == "clocks":
            value = [
                {"base_seconds": clock.base_seconds, "increment_seconds": clock.increment_seconds}
                for clock in manifest.clocks
            ]
        elif name == "adapter_compatibility":
            value = [
                {
                    "adapter_id": adapter.adapter_id,
                    "status": adapter.status,
                    "rules_hash": adapter.rules_hash,
                }
                for adapter in manifest.adapter_compatibility
            ]
        elif name == "red_tile_ids":
            value = list(manifest.red_tile_ids)
        elif name == "uma_by_rank":
            value = list(manifest.uma_by_rank)
        elif name == "abortive_draws":
            value = list(manifest.abortive_draws)
        payload[name] = value
    if source_evidence is not None:
        payload[SOURCE_EVIDENCE_KEY] = dict(source_evidence)
    return payload


def _source_from_payload(raw: Any) -> SourceAuthority:
    if not isinstance(raw, Mapping) or set(raw) != {"url", "retrieved_at_utc", "content_sha256"}:
        raise ContractError("payload['source'] must map url/retrieved_at_utc/content_sha256")
    return SourceAuthority(
        url=raw["url"],  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
        retrieved_at_utc=raw["retrieved_at_utc"],  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
        content_sha256=raw["content_sha256"],  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
    )


def _clocks_from_payload(raw: Any) -> tuple[ClockRule, ...]:
    if not isinstance(raw, list) or len(raw) == 0:
        raise ContractError("payload['clocks'] must be a non-empty array")
    clocks: list[ClockRule] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, Mapping) or set(entry) != {"base_seconds", "increment_seconds"}:
            raise ContractError(f"payload['clocks'][{i}] must map base_seconds/increment_seconds")
        clocks.append(
            ClockRule(
                base_seconds=entry["base_seconds"],  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
                increment_seconds=entry["increment_seconds"],  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
            )
        )
    return tuple(clocks)


def _adapters_from_payload(raw: Any) -> tuple[AdapterCompatibility, ...]:
    if not isinstance(raw, list):
        raise ContractError("payload['adapter_compatibility'] must be an array")
    adapters: list[AdapterCompatibility] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, Mapping) or set(entry) != {"adapter_id", "status", "rules_hash"}:
            raise ContractError(
                f"payload['adapter_compatibility'][{i}] must map adapter_id/status/rules_hash"
            )
        adapters.append(
            AdapterCompatibility(
                adapter_id=entry["adapter_id"],  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
                status=entry["status"],  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
                rules_hash=entry["rules_hash"],  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
            )
        )
    return tuple(adapters)


def rules_manifest_from_payload(payload: Mapping[str, Any]) -> RulesManifest:
    """Rebuild a validated :class:`RulesManifest` from its JSON payload.

    Rejects missing SPEC 5.1 fields, undeclared extra keys (only
    ``source_evidence`` is declared), and any invalid enum/range value.
    """
    if not isinstance(payload, Mapping):
        raise ContractError("rules manifest payload must be a JSON object")
    keys = set(payload)
    missing = [name for name in RULES_MANIFEST_PAYLOAD_FIELDS if name not in keys]
    if len(missing) > 0:
        raise ContractError(
            f"rules manifest payload is missing fields {missing}; "
            "engine defaults MUST NOT fill fields"
        )
    extras = sorted(keys - set(RULES_MANIFEST_PAYLOAD_FIELDS) - {SOURCE_EVIDENCE_KEY})
    if len(extras) > 0:
        raise ContractError(
            f"undeclared extra payload keys {extras} (only '{SOURCE_EVIDENCE_KEY}' is declared)"
        )
    fields: dict[str, Any] = {}
    for name in RULES_MANIFEST_PAYLOAD_FIELDS:
        value: Any = payload[name]
        if name == "source":
            fields[name] = _source_from_payload(value)
        elif name == "clocks":
            fields[name] = _clocks_from_payload(value)
        elif name == "adapter_compatibility":
            fields[name] = _adapters_from_payload(value)
        elif name in ("uma_by_rank", "red_tile_ids", "abortive_draws"):
            if not isinstance(value, list):
                raise ContractError(f"payload[{name!r}] must be an array")
            fields[name] = tuple(value)
        else:
            fields[name] = value
    evidence = payload.get(SOURCE_EVIDENCE_KEY)
    if evidence is not None:
        _validate_source_evidence(evidence)
    return RulesManifest(**fields)


def _validate_source_evidence(evidence: Any) -> None:
    """Provenance map: ``authorities`` descriptors plus per-field citations.

    Every cited manifest field MUST reference an authority (snapshot, official
    secondary source, or explicit owner decision) so no value is a silent
    default. The ``authorities`` section itself holds descriptors keyed by the
    authority ids referenced from ``fields``.
    """
    if not isinstance(evidence, Mapping):
        raise ContractError(f"{SOURCE_EVIDENCE_KEY} must be a mapping with 'fields' citations")
    for section, value in evidence.items():
        if not isinstance(section, str) or section == "":
            raise ContractError(f"{SOURCE_EVIDENCE_KEY} keys must be non-empty strings")
        if not isinstance(value, Mapping):
            raise ContractError(f"{SOURCE_EVIDENCE_KEY}[{section!r}] must be a mapping")
        if section == "authorities":
            continue
        for field_name, entry in value.items():
            if not isinstance(field_name, str) or field_name == "":
                raise ContractError(
                    f"{SOURCE_EVIDENCE_KEY}[{section!r}] keys must be non-empty strings"
                )
            if not isinstance(entry, Mapping):
                raise ContractError(
                    f"{SOURCE_EVIDENCE_KEY}[{section!r}][{field_name!r}] must be a mapping "
                    "with an 'authority' reference"
                )
            if "authority" not in entry:
                raise ContractError(
                    f"{SOURCE_EVIDENCE_KEY}[{section!r}][{field_name!r}] lacks 'authority'; "
                    "every manifest value must cite its source or owner decision"
                )

# Convenience instances used by tests and config generation.

STANDARD_CLOCK = ClockRule(
    base_seconds=STANDARD_CLOCK_SECONDS[0], increment_seconds=STANDARD_CLOCK_SECONDS[1]
)
FAST_CLOCK = ClockRule(base_seconds=FAST_CLOCK_SECONDS[0], increment_seconds=FAST_CLOCK_SECONDS[1])
