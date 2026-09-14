"""SPEC 5.1 Tenhou rules manifest — WP-02B contract module.

Re-export facade over the split modules: :mod:`hydra2.contracts.rules_canonical`
(identity constants, policy enums, field validators, contract-local canonical
JSON) and :mod:`hydra2.contracts.rules_manifest` (manifest dataclasses, rank
resolution, payload codec). Import from this path; it preserves every public
name and ``__all__``.
"""

from __future__ import annotations

from hydra2.contracts.rules_canonical import (
    ABORTIVE_DRAW_KINDS as ABORTIVE_DRAW_KINDS,
)
from hydra2.contracts.rules_canonical import (
    ADAPTER_COMPATIBILITY_STATUSES as ADAPTER_COMPATIBILITY_STATUSES,
)
from hydra2.contracts.rules_canonical import (
    AGARI_YAME_POLICIES as AGARI_YAME_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    ALL_LAST_POLICIES as ALL_LAST_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    CHANKAN_POLICIES as CHANKAN_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    FAST_CLOCK_SECONDS as FAST_CLOCK_SECONDS,
)
from hydra2.contracts.rules_canonical import (
    FURITEN_POLICIES as FURITEN_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    KAN_DORA_REVEAL_POLICIES as KAN_DORA_REVEAL_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    KAN_URA_POLICIES as KAN_URA_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    KAZOE_POLICIES as KAZOE_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    KUIKAE_POLICIES as KUIKAE_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    KUITAN_VALUES as KUITAN_VALUES,
)
from hydra2.contracts.rules_canonical import (
    MULTIPLE_RON_POLICIES as MULTIPLE_RON_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    OKA_POLICIES as OKA_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    PAO_POLICIES as PAO_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    PLACEMENT_CONVERSION_IDS as PLACEMENT_CONVERSION_IDS,
)
from hydra2.contracts.rules_canonical import (
    RANK_TIE_BREAKS as RANK_TIE_BREAKS,
)
from hydra2.contracts.rules_canonical import (
    RED_TILE_IDS as RED_TILE_IDS,
)
from hydra2.contracts.rules_canonical import (
    RETURN_POINTS as RETURN_POINTS,
)
from hydra2.contracts.rules_canonical import (
    RIICHI_STICK_ALLOCATIONS as RIICHI_STICK_ALLOCATIONS,
)
from hydra2.contracts.rules_canonical import (
    RINSHAN_POLICIES as RINSHAN_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    RULES_ID as RULES_ID,
)
from hydra2.contracts.rules_canonical import (
    RULES_MANIFEST_PAYLOAD_FIELDS as RULES_MANIFEST_PAYLOAD_FIELDS,
)
from hydra2.contracts.rules_canonical import (
    SOURCE_EVIDENCE_KEY as SOURCE_EVIDENCE_KEY,
)
from hydra2.contracts.rules_canonical import (
    STANDARD_CLOCK_SECONDS as STANDARD_CLOCK_SECONDS,
)
from hydra2.contracts.rules_canonical import (
    SUDDEN_DEATH_POLICIES as SUDDEN_DEATH_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    TENHOU_ABORTIVE_DRAWS as TENHOU_ABORTIVE_DRAWS,
)
from hydra2.contracts.rules_canonical import (
    TOBI_POLICIES as TOBI_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    YAKUMAN_POLICIES as YAKUMAN_POLICIES,
)
from hydra2.contracts.rules_canonical import (
    canonical_contract_json_bytes as canonical_contract_json_bytes,
)
from hydra2.contracts.rules_canonical import (
    canonical_contract_json_text as canonical_contract_json_text,
)
from hydra2.contracts.rules_manifest import (
    FAST_CLOCK as FAST_CLOCK,
)
from hydra2.contracts.rules_manifest import (
    STANDARD_CLOCK as STANDARD_CLOCK,
)
from hydra2.contracts.rules_manifest import (
    AdapterCompatibility as AdapterCompatibility,
)
from hydra2.contracts.rules_manifest import (
    ClockRule as ClockRule,
)
from hydra2.contracts.rules_manifest import (
    RulesManifest as RulesManifest,
)
from hydra2.contracts.rules_manifest import (
    SourceAuthority as SourceAuthority,
)
from hydra2.contracts.rules_manifest import (
    manifest_to_payload as manifest_to_payload,
)
from hydra2.contracts.rules_manifest import (
    resolve_final_ranks as resolve_final_ranks,
)
from hydra2.contracts.rules_manifest import (
    rules_manifest_from_payload as rules_manifest_from_payload,
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
