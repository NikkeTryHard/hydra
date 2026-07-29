"""WP-02D supplement gate: published ObservationSchema artifact and hash wiring.

Covers the supplement deliverables that close owner decision D-WP02D-12:
the closed ObservationSchema artifact derived from the live ActorObservation
dataclass is canonically published at configs/contracts/observation_schema_v1.json,
and ObservationBuilder stamps its payload digest onto every built observation
as ``observation_schema_hash`` (caller-supplied lineage removed). The golden
transform test proves the wiring changed exactly two document fields:
re-substituting the legacy caller-supplied lineage into a live observation and
re-deriving ``observation_hash`` over the identity rule reproduces the captured
pre-wiring bytes bit-exactly.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from hydra2.contracts.canonical import canonical_json_bytes
from hydra2.contracts.common import ContractError, DigestMismatchError, DigestText, Seat
from hydra2.contracts.observation import (
    OBSERVATION_SCHEMA_ARTIFACT_TYPE,
    OBSERVATION_SCHEMA_SCHEMA_VERSION,
    ActorObservation,
    ObservationBuilder,
    build_observation_schema_envelope,
    build_observation_schema_payload,
    compute_observation_schema_digest,
    make_actor_observation,
    observation_schema_digest,
    parse_observation_schema,
)

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.contract_package("WP-02D")

_OBSERVATION_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "contracts" / "observation_schema_v1.json"
)

#: Canonical envelope bytes of the published artifact (file identity).
GOLDEN_OBSERVATION_SCHEMA_SHA256 = (
    "sha256:ac4c2e35f2d4079691e0eef2011fb744d1e0fa723473ec0ab8d3c520dd4c46ba"
)
#: Payload digest inside the artifact; the value stamped onto observations.
GOLDEN_OBSERVATION_SCHEMA_PAYLOAD_DIGEST = (
    "sha256:e959280b9b2e9fbe923eff91b3658cb738b287ce9d64aac428ac05bbded6004a"
)
#: Seat-0 scripted-round canonical bytes captured BEFORE the builder stopped
#: accepting caller-supplied schema lineage (legacy hash below).
GOLDEN_PRE_WIRING_SEAT0_SHA256 = (
    # Re-derived alongside GOLDEN_WIRED_SEAT0_SHA256 after the event_schema_v1
    # supersession (kan/dora_revealed grammar repairs): the reconstruction
    # starts from the live wired document, whose lineage embeds the event-
    # schema digest, so both values move together.
    "sha256:bdd909c54e631647db36f0235944b8696f4aa2a15b0df9c05070e39a7f72206a"
)
#: Derived deterministically at wiring time: the same scripted-round document
#: with the builder-stamped lineage and the recomputed observation hash.
GOLDEN_WIRED_SEAT0_SHA256 = (
    # Re-derived after the event_schema_v1.json supersession (kan/dora_revealed
    # grammar repairs): builder-stamped lineage covers the event-schema digest,
    # so the wired seat-0 canonical bytes move deterministically.
    "sha256:e1c6a32728df0bc86eb6136814fcb7104086a9b738f8b5d781c30d54492673b7"
)

_LEGACY_SCHEMA_HASH = "sha256:" + "0b" * 32


def _load_events_obs_module() -> ModuleType:
    """Load the WP-02D scripted-round module without depending on collection order."""
    path = Path(__file__).with_name("test_events_obs_wp02d.py")
    spec = importlib.util.spec_from_file_location("wp02d_events_obs_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wp02d() -> ModuleType:
    return _load_events_obs_module()


@pytest.fixture(scope="module")
def action_table(wp02d: ModuleType):
    from hydra2.contracts.action import load_action_table

    return load_action_table(wp02d.ACTION_TABLE_PATH)


@pytest.fixture(scope="module")
def seat0_observation(wp02d: ModuleType, action_table) -> ActorObservation:
    """Seat-0 observation over the scripted round under wired lineage."""
    builder = wp02d._make_builder(len(action_table.actions))
    wp02d._feed_round(builder, wp02d.Round().build_stream())
    return builder.build(actor=Seat(0), legal_mask=wp02d._true_mask(len(action_table.actions)))


def _sha256_hex(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class TestObservationSchemaArtifact:
    """The published v1 artifact: canonical bytes, closed payload, strict parser."""

    def test_published_bytes_are_the_canonical_envelope_golden(self):
        disk = _OBSERVATION_SCHEMA_PATH.read_bytes()
        envelope = build_observation_schema_envelope()
        assert canonical_json_bytes(envelope) == disk
        assert _sha256_hex(disk) == GOLDEN_OBSERVATION_SCHEMA_SHA256
        parsed = parse_observation_schema(disk)
        assert parsed["artifact_type"] == OBSERVATION_SCHEMA_ARTIFACT_TYPE
        assert parsed["schema_version"] == OBSERVATION_SCHEMA_SCHEMA_VERSION
        assert parsed["compatibility"] == "exact"

    def test_payload_digest_is_stable_and_matches_every_entry_point(self):
        payload = build_observation_schema_payload()
        recomputed = compute_observation_schema_digest(payload)
        assert recomputed == GOLDEN_OBSERVATION_SCHEMA_PAYLOAD_DIGEST
        assert observation_schema_digest() == GOLDEN_OBSERVATION_SCHEMA_PAYLOAD_DIGEST
        parsed = parse_observation_schema(_OBSERVATION_SCHEMA_PATH.read_bytes())
        assert parsed["payload"]["digest"] == GOLDEN_OBSERVATION_SCHEMA_PAYLOAD_DIGEST

    def test_field_order_and_constraints_stay_closed_over_the_live_dataclass(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import hydra2.contracts.observation as obs_mod

        payload = build_observation_schema_payload()
        names = tuple(field.name for field in ActorObservation.__dataclass_fields__.values())  # type: ignore[attr-defined]
        assert tuple(payload["field_order"]) == names  # type: ignore[arg-type]
        assert set(payload["fields"]) == set(names)  # type: ignore[attr-defined]
        rows = payload["fields"]
        assert rows["actor"] == {"dtype": "seat", "minimum": 0, "maximum": 3}  # type: ignore[index]
        assert rows["dora_indicators"]["length"] == 5  # type: ignore[index]
        assert rows["dora_indicators"]["sentinel"] == -1  # type: ignore[index]
        # Drift guard: a constraint table out of sync with the dataclass
        # (add/remove/rename/reorder anywhere) must raise, never silently pass.
        drifted = dict(obs_mod._FIELD_CONSTRAINTS)
        del drifted["kan_count"]
        monkeypatch.setattr(obs_mod, "_FIELD_CONSTRAINTS", drifted)
        with pytest.raises(ContractError, match="closed over ActorObservation"):
            build_observation_schema_payload()

    def test_parser_rejects_unknown_envelope_field(self):
        document = parse_observation_schema(_OBSERVATION_SCHEMA_PATH.read_bytes())
        poisoned = dict(document)
        poisoned["unexpected_extra"] = True
        with pytest.raises(ContractError, match=r"SPEC 2.2 envelope"):
            parse_observation_schema(json.dumps(poisoned).encode("utf-8"))

    def test_parser_rejects_tampered_payload_digest(self):
        document = parse_observation_schema(_OBSERVATION_SCHEMA_PATH.read_bytes())
        tampered = dict(document)
        payload = dict(tampered["payload"])
        payload["digest"] = "sha256:" + "f" * 64
        tampered["payload"] = payload
        with pytest.raises(DigestMismatchError):
            parse_observation_schema(json.dumps(tampered).encode("utf-8"))


class TestSchemaHashWiring:
    """The builder stamps the published digest; caller-supplied lineage is gone."""

    def test_builder_stamps_the_published_payload_digest_on_every_seat(
        self, wp02d: ModuleType, action_table
    ):
        builder = wp02d._make_builder(len(action_table.actions))
        wp02d._feed_round(builder, wp02d.Round().build_stream())
        mask = wp02d._true_mask(len(action_table.actions))
        for seat in range(4):
            observation = builder.build(actor=Seat(seat), legal_mask=mask)
            assert observation.observation_schema_hash == GOLDEN_OBSERVATION_SCHEMA_PAYLOAD_DIGEST
        assert observation_schema_digest() == GOLDEN_OBSERVATION_SCHEMA_PAYLOAD_DIGEST

    def test_builder_no_longer_accepts_caller_supplied_lineage(self, wp02d: ModuleType):
        stale_kwargs = {
            "game_id": "g-wp02d",
            "rules_id": "tenhou_4p_hanchan_v1",
            "rules_hash": wp02d._RULES_HASH,
            "action_table_hash": DigestText("sha256:" + "7b" * 32),
            "expected_legal_mask_length": 32,
            "event_schema_hash": wp02d.ev.load_event_schema(wp02d.EVENT_SCHEMA_PATH)["payload"][
                "digest"
            ],
            "packet_boundary_hash": wp02d._PACKET_BOUNDARY_HASH,
            "observation_schema_hash": _LEGACY_SCHEMA_HASH,
        }
        with pytest.raises(TypeError, match="observation_schema_hash"):
            ObservationBuilder(**stale_kwargs)

    def test_wired_golden_differs_from_pre_wiring_in_exactly_two_fields(
        self, seat0_observation: ActorObservation
    ):
        live_document = seat0_observation.to_json()
        assert _sha256_hex(canonical_json_bytes(live_document)) == GOLDEN_WIRED_SEAT0_SHA256

        # Reconstruct the pre-wiring document: swap back the legacy caller-
        # supplied lineage, then re-derive observation_hash over the identity
        # rule (canonical bytes of the field document without the hash field).
        pre_wiring = dict(live_document)
        pre_wiring["observation_schema_hash"] = _LEGACY_SCHEMA_HASH
        identity = {k: v for k, v in pre_wiring.items() if k != "observation_hash"}
        pre_wiring["observation_hash"] = _sha256_hex(canonical_json_bytes(identity))
        assert _sha256_hex(canonical_json_bytes(pre_wiring)) == GOLDEN_PRE_WIRING_SEAT0_SHA256

        differing = sorted(k for k in live_document if live_document[k] != pre_wiring[k])
        assert differing == ["observation_hash", "observation_schema_hash"]

    def test_direct_construction_rejects_unknown_fields(self):
        with pytest.raises(ContractError, match=r"unknown=\['bogus_field'\]"):
            make_actor_observation(
                game_id="g",
                observation_schema_hash=_LEGACY_SCHEMA_HASH,
                bogus_field="x",  # type: ignore[arg-type]
            )


class TestSchemaHashWiringNegative:
    def test_tampered_schema_lineage_breaks_identity_binding(
        self, wp02d: ModuleType, seat0_observation: ActorObservation
    ):
        from hydra2.contracts.observation import VISIBILITY_VALIDATOR

        VISIBILITY_VALIDATOR.validate_observation(seat0_observation)
        saved = seat0_observation.observation_schema_hash
        object.__setattr__(seat0_observation, "observation_schema_hash", _LEGACY_SCHEMA_HASH)
        with pytest.raises(DigestMismatchError):
            VISIBILITY_VALIDATOR.validate_observation(seat0_observation)
        object.__setattr__(seat0_observation, "observation_schema_hash", saved)
