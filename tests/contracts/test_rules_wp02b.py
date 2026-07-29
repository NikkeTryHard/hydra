"""WP-02B rules-manifest gate: canonical hash, flag completeness, tie-break, provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydra2.artifacts.canonical import canonical_bytes, loads_canonical
from hydra2.artifacts.digest import sha256_digest
from hydra2.artifacts.registry import artifact_id, envelope_from_json
from hydra2.contracts.common import ContractError
from hydra2.contracts.rules import (
    RULES_MANIFEST_PAYLOAD_FIELDS,
    SOURCE_EVIDENCE_KEY,
    manifest_to_payload,
    resolve_final_ranks,
    rules_manifest_from_payload,
)

pytestmark = pytest.mark.contract_package("WP-02B")

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = _REPO_ROOT / "configs" / "rules" / "tenhou_4p_hanchan_v1.json"

#: Golden identity of the published manifest envelope (RFC 8785 + SHA-256).
GOLDEN_ENVELOPE_DIGEST = "sha256:3042a493280224f533d831f371275b1c96585cf1db5a2e5fb86ec259f403286b"
GOLDEN_SOURCE_SHA256 = "sha256:3de07f2c338caf520f6c4dc2160270adb7b1fbd25c1666bf61d3f351902f210a"

OWNER_DECISION_IDS = (
    "owner_decision:d1_oka_none",
    "owner_decision:d2_dead_wall_14",
    "owner_decision:d3_abortive_riichi_sticks_carried",
    "owner_decision:kuitan_ranked_allowed",
)


@pytest.fixture(scope="module")
def envelope_doc() -> dict:
    return loads_canonical(CONFIG_PATH.read_bytes())


@pytest.fixture(scope="module")
def manifest(envelope_doc):
    return rules_manifest_from_payload(envelope_doc["payload"])


class TestManifestCanonicalHashStable:
    def test_file_is_canonical_and_matches_golden_digest(self, envelope_doc):
        raw = CONFIG_PATH.read_bytes()
        assert sha256_digest(raw) == GOLDEN_ENVELOPE_DIGEST
        assert canonical_bytes(envelope_doc) == raw  # stored bytes are RFC 8785

    def test_envelope_identity_recomputes_independently(self, envelope_doc):
        envelope = envelope_from_json(envelope_doc)
        assert envelope.artifact_type == "hydra2.rules_manifest"
        assert envelope.schema_version == "1.0.0"
        assert envelope.compatibility == "exact"
        assert artifact_id(envelope) == GOLDEN_ENVELOPE_DIGEST

    def test_manifest_round_trip_reproduces_published_payload(self, envelope_doc, manifest):
        source_evidence = envelope_doc["payload"][SOURCE_EVIDENCE_KEY]
        reserialized = manifest_to_payload(manifest, source_evidence=source_evidence)
        assert json.loads(canonical_bytes(reserialized)) == json.loads(
            canonical_bytes(envelope_doc["payload"])
        )

    def test_source_authority_binds_the_reviewed_snapshot(self, manifest):
        assert manifest.source.url == "https://tenhou.net/man/"
        assert manifest.source.content_sha256 == GOLDEN_SOURCE_SHA256
        assert manifest.source.retrieved_at_utc == "2026-08-22T22:31:52Z"


class TestCompleteFlagSet:
    def test_every_spec_field_present_without_extras(self, envelope_doc):
        payload = envelope_doc["payload"]
        missing = [name for name in RULES_MANIFEST_PAYLOAD_FIELDS if name not in payload]
        assert not missing, f"missing SPEC 5.1 flags: {missing}"
        extras = set(payload) - set(RULES_MANIFEST_PAYLOAD_FIELDS) - {SOURCE_EVIDENCE_KEY}
        assert not extras, f"undeclared extras: {extras}"

    def test_selected_values_match_tenhou_evidence(self, manifest):
        assert (manifest.players, manifest.match_length) == (4, "hanchan")
        assert manifest.starting_points == 25000
        assert manifest.return_points == 30000
        assert manifest.uma_by_rank == (20, 10, -10, -20)
        assert manifest.oka_policy == "none"
        assert manifest.kuitan is True
        assert manifest.red_tile_ids == (16, 52, 88)
        assert [(c.base_seconds, c.increment_seconds) for c in manifest.clocks] == [
            (5, 10),
            (3, 5),
        ]
        assert manifest.kuikae_policy == "forbidden"
        assert manifest.chankan_policy == "permitted"
        assert manifest.rinshan_policy == "dead_wall_14"
        assert manifest.kan_dora_reveal_policy == "ankan_immediate_open_delayed"
        assert manifest.kan_ura_policy == "present"
        assert manifest.multiple_ron_policy == "all_winners_paid_sticks_to_dealer_left"
        assert manifest.tobi_policy == "negative_points_immediate_end"
        assert manifest.bankruptcy_threshold == 0
        assert manifest.abortive_draws == (
            "kyuushu_kyuuhai",
            "suucha_riichi",
            "sanchahou",
            "suukaikan",
            "suufon_renda",
        )
        assert manifest.nagashi_mangan is True
        assert manifest.rank_tie_break == "east1_seat_wind_order"
        assert manifest.placement_conversion_id == "tenhou_rank_sticks_top_uma_v1"
        # Authorization/qualification metadata deliberately absent until WP-03A/WP-04B.
        assert manifest.adapter_compatibility == ()

    def test_missing_flag_rejected_engine_defaults_never_fill(self, envelope_doc):
        broken = json.loads(json.dumps(envelope_doc))
        del broken["payload"]["riichi_stick_allocation"]
        with pytest.raises(ContractError, match="engine defaults MUST NOT fill"):
            rules_manifest_from_payload(broken["payload"])

    def test_undeclared_extra_key_rejected(self, envelope_doc):
        broken = json.loads(json.dumps(envelope_doc))
        broken["payload"]["surprise_default"] = True
        with pytest.raises(ContractError, match="undeclared extra payload keys"):
            rules_manifest_from_payload(broken["payload"])

    def test_red_tile_ids_are_exact_triple(self, envelope_doc):
        broken = json.loads(json.dumps(envelope_doc))
        broken["payload"]["red_tile_ids"] = [16, 52]
        with pytest.raises(ContractError, match="red_tile_ids"):
            rules_manifest_from_payload(broken["payload"])

    def test_clocks_must_be_standard_then_fast(self, envelope_doc):
        broken = json.loads(json.dumps(envelope_doc))
        broken["payload"]["clocks"] = [
            {"base_seconds": 3, "increment_seconds": 5},
            {"base_seconds": 5, "increment_seconds": 10},
        ]
        with pytest.raises(ContractError, match="clocks"):
            rules_manifest_from_payload(broken["payload"])

    def test_unknown_policy_enum_value_rejected(self, envelope_doc):
        broken = json.loads(json.dumps(envelope_doc))
        broken["payload"]["oka_policy"] = "oka_pool_half_return"  # undeclared enum value
        with pytest.raises(ContractError, match="oka_policy"):
            rules_manifest_from_payload(broken["payload"])


class TestRankTieBreakEncoded:
    def test_manifest_declares_east1_seat_wind_order(self, manifest):
        assert manifest.rank_tie_break == "east1_seat_wind_order"

    def test_all_equal_scores_rank_by_east1_wind_order(self):
        assert resolve_final_ranks((25000, 25000, 25000, 25000)) == (1, 2, 3, 4)

    def test_dealer_pair_takes_leading_ranks(self):
        # Seats 0 and 1 tie on top; East-1 wind order puts the dealer first.
        assert resolve_final_ranks((30000, 30000, 25000, 25000)) == (1, 2, 3, 4)

    def test_tie_between_south_and_west_prefers_lower_seat(self):
        assert resolve_final_ranks((40000, 28000, 28000, 24000)) == (1, 2, 3, 4)

    def test_late_board_tie_keeps_wind_order_not_score_history(self):
        # Seats 0 and 1 tie for last; seat 0 (East-1 dealer) takes rank 3.
        assert resolve_final_ranks((25000, 25000, 40000, 30000)) == (3, 4, 1, 2)

    def test_strict_scores_dominate_ties(self):
        assert resolve_final_ranks((100, 300, 200, 400)) == (4, 2, 3, 1)


class TestOwnerDecisionsExplicit:
    def test_four_owner_decisions_recorded_with_basis(self, envelope_doc):
        authorities = envelope_doc["payload"][SOURCE_EVIDENCE_KEY]["authorities"]
        for decision_id in OWNER_DECISION_IDS:
            entry = authorities[decision_id]
            assert entry["kind"] == "owner_decision"
            assert entry["decided_by"] == "Main (contract owner)"
            assert entry["basis"]

    def test_every_spec_field_cites_an_authority(self, envelope_doc):
        evidence = envelope_doc["payload"][SOURCE_EVIDENCE_KEY]
        cited = set(evidence["fields"])
        uncited = [name for name in RULES_MANIFEST_PAYLOAD_FIELDS if name not in cited]
        assert not uncited, f"fields without provenance: {uncited}"
        for field_name, entry in evidence["fields"].items():
            assert entry.get("authority"), f"{field_name} lacks an authority citation"

    def test_gap_fields_reference_owner_decision_authorities(self, envelope_doc):
        fields = envelope_doc["payload"][SOURCE_EVIDENCE_KEY]["fields"]
        assert "owner_decision:d1_oka_none" in fields["oka_policy"]["authority"]
        assert "owner_decision:d2_dead_wall_14" in fields["rinshan_policy"]["authority"]
        assert (
            "owner_decision:d3_abortive_riichi_sticks_carried"
            in fields["riichi_stick_allocation"]["authority"]
        )
        assert "owner_decision:kuitan_ranked_allowed" in fields["kuitan"]["authority"]

    def test_secondary_authority_records_url_and_digest(self, envelope_doc):
        secondary = envelope_doc["payload"][SOURCE_EVIDENCE_KEY]["authorities"][
            "secondary:tenhou_blog_38995838"
        ]
        assert secondary["url"].startswith("http://blog.tenhou.net/")
        assert secondary["content_sha256"].startswith("sha256:")
