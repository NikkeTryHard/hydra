"""SPEC 5.1 rules manifest — dataclasses, rank resolution, and payload codec.

Owns the immutable :class:`RulesManifest` vocabulary for
``tenhou_4p_hanchan_v1``: the source/clock/adapter dataclasses with their
strict validators, the seat-wind rank resolver, the JSON payload codec, and
the provenance-evidence checker. Every policy string value MUST come from
the frozen vocabulary in ``hydra2._native.contracts`` (single source);
unknown source behavior blocks manifest publication and engine defaults NEVER fill
fields. ``RULES_ID`` pins the manifest identity across the split.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts._validators import _require_bool as _require_bool
from hydra2.contracts._validators import _require_enum as _require_enum
from hydra2.contracts._validators import _require_int as _require_int
from hydra2.contracts._validators import _require_quad_ints as _require_quad_ints
from hydra2.contracts._validators import _require_str as _require_str
from hydra2.contracts.common import (
    ContractError,
    DigestText,
    TileId,
    UtcTimestamp,
    make_utc_timestamp,
)

#: SPEC 5.1 identity + frozen vocab (single source: ``hydra2._native.contracts``).
RULES_ID: str = _bridge_contracts.RULES_ID
STARTING_POINTS: int = _bridge_contracts.STARTING_POINTS
RETURN_POINTS: int = _bridge_contracts.RETURN_POINTS
RED_TILE_IDS: tuple[int, ...] = _bridge_contracts.RED_TILE_IDS
STANDARD_CLOCK_SECONDS: tuple[int, int] = _bridge_contracts.STANDARD_CLOCK_SECONDS
FAST_CLOCK_SECONDS: tuple[int, int] = _bridge_contracts.FAST_CLOCK_SECONDS
OKA_POLICIES: tuple[str, ...] = _bridge_contracts.OKA_POLICIES
KUITAN_VALUES: tuple[str, ...] = _bridge_contracts.KUITAN_VALUES
KUIKAE_POLICIES: tuple[str, ...] = _bridge_contracts.KUIKAE_POLICIES
FURITEN_POLICIES: tuple[str, ...] = _bridge_contracts.FURITEN_POLICIES
CHANKAN_POLICIES: tuple[str, ...] = _bridge_contracts.CHANKAN_POLICIES
RINSHAN_POLICIES: tuple[str, ...] = _bridge_contracts.RINSHAN_POLICIES
KAN_DORA_REVEAL_POLICIES: tuple[str, ...] = _bridge_contracts.KAN_DORA_REVEAL_POLICIES
KAN_URA_POLICIES: tuple[str, ...] = _bridge_contracts.KAN_URA_POLICIES
PAO_POLICIES: tuple[str, ...] = _bridge_contracts.PAO_POLICIES
YAKUMAN_POLICIES: tuple[str, ...] = _bridge_contracts.YAKUMAN_POLICIES
KAZOE_POLICIES: tuple[str, ...] = _bridge_contracts.KAZOE_POLICIES
MULTIPLE_RON_POLICIES: tuple[str, ...] = _bridge_contracts.MULTIPLE_RON_POLICIES
RIICHI_STICK_ALLOCATIONS: tuple[str, ...] = _bridge_contracts.RIICHI_STICK_ALLOCATIONS
ABORTIVE_DRAW_KINDS: frozenset[str] = _bridge_contracts.ABORTIVE_DRAW_KINDS
TENHOU_ABORTIVE_DRAWS: tuple[str, ...] = _bridge_contracts.TENHOU_ABORTIVE_DRAWS
ALL_LAST_POLICIES: tuple[str, ...] = _bridge_contracts.ALL_LAST_POLICIES
AGARI_YAME_POLICIES: tuple[str, ...] = _bridge_contracts.AGARI_YAME_POLICIES
TOBI_POLICIES: tuple[str, ...] = _bridge_contracts.TOBI_POLICIES
SUDDEN_DEATH_POLICIES: tuple[str, ...] = _bridge_contracts.SUDDEN_DEATH_POLICIES
RANK_TIE_BREAKS: tuple[str, ...] = _bridge_contracts.RANK_TIE_BREAKS
PLACEMENT_CONVERSION_IDS: tuple[str, ...] = _bridge_contracts.PLACEMENT_CONVERSION_IDS
ADAPTER_COMPATIBILITY_STATUSES: tuple[str, ...] = _bridge_contracts.ADAPTER_COMPATIBILITY_STATUSES
RULES_MANIFEST_PAYLOAD_FIELDS: tuple[str, ...] = _bridge_contracts.RULES_MANIFEST_PAYLOAD_FIELDS
SOURCE_EVIDENCE_KEY: str = _bridge_contracts.SOURCE_EVIDENCE_KEY


__all__ = [
    "FAST_CLOCK",
    "STANDARD_CLOCK",
    "AdapterCompatibility",
    "ClockRule",
    "RulesManifest",
    "SourceAuthority",
    "manifest_to_payload",
    "resolve_final_ranks",
    "rules_manifest_from_payload",
]


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
        digest: DigestText = _bridge_contracts.make_digest_text(self.content_sha256)
        object.__setattr__(self, "content_sha256", digest)


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
        rules_digest: DigestText = _bridge_contracts.make_digest_text(self.rules_hash)
        object.__setattr__(self, "rules_hash", rules_digest)


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
        red_ids = tuple(_bridge_contracts.make_tile_id(item) for item in self.red_tile_ids)
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
# Rank resolution with the encoded tie-break.
# ---------------------------------------------------------------------------


def resolve_final_ranks(final_scores: Sequence[int]) -> tuple[int, int, int, int]:
    """Resolve placement ranks 1..4 from raw final scores.

    Encodes rank_tie_break="east1_seat_wind_order": equal scores are ordered by
    seat wind order of East-1 (man.html L1025 「終了時に同点の場合は東1局の風順で順位を決定」).
    Seats 0..3 are the East/South/West/North winds of East-1 in actor-observation
    seat order, so the lower seat index takes the better rank on ties.
    Thin bridge delegate: ``hydra2._native.contracts.resolve_final_ranks``
    decides (same tie-break, same +/-1e12 domain, bool excluded). Evidence:
    value parity on strict/tie/edge quads, 1.06-1.16x faster. Bridge rejects
    surface as ContractError (fail-closed); a missing bridge raises
    ImportError with a build-ext hint.
    """
    try:
        ranks: list[int] = _bridge_contracts.resolve_final_ranks(tuple(final_scores))
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with contracts not importable; "
            "rebuild the bridge with `pixi run build-ext` before resolving ranks"
        ) from exc
    except Exception as exc:
        raise ContractError(f"final_scores rejected: {exc}") from exc
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
        url=raw["url"],  # pyrefly: ignore[unknown-argument-type]  # reason: raw Mapping shape-checked above; ctor validates each field
        retrieved_at_utc=raw["retrieved_at_utc"],  # pyrefly: ignore[unknown-argument-type]  # reason: raw Mapping shape-checked above; ctor validates each field
        content_sha256=raw["content_sha256"],  # pyrefly: ignore[unknown-argument-type]  # reason: raw Mapping shape-checked above; ctor validates each field
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
                base_seconds=entry["base_seconds"],  # pyrefly: ignore[unknown-argument-type]  # reason: entry Mapping shape-checked above; ctor validates each field
                increment_seconds=entry["increment_seconds"],  # pyrefly: ignore[unknown-argument-type]  # reason: entry Mapping shape-checked above; ctor validates each field
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
                adapter_id=entry["adapter_id"],  # pyrefly: ignore[unknown-argument-type]  # reason: entry Mapping shape-checked above; ctor validates each field
                status=entry["status"],  # pyrefly: ignore[unknown-argument-type]  # reason: entry Mapping shape-checked above; ctor validates each field
                rules_hash=entry["rules_hash"],  # pyrefly: ignore[unknown-argument-type]  # reason: entry Mapping shape-checked above; ctor validates each field
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
