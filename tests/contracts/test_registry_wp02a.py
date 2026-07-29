"""Immutable registry: envelope identity, lookup, rejections, compatibility."""

from __future__ import annotations

import hashlib

import pytest

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.artifacts.registry import (
    ArtifactEnvelope,
    ArtifactRegistry,
    MigrationMetadata,
    artifact_id,
    envelope_from_json,
    to_json,
)
from hydra2.contracts.common import (
    ContractError,
    CorruptArtifactError,
    DigestMismatchError,
    IncompatibleSchemaError,
    LineageError,
    SchemaVersion,
)

pytestmark = pytest.mark.contract_package("WP-02A")

HAND_CANONICAL_TEXT = (
    '{"artifact_type":"hydra2.test","compatibility":"exact",'
    '"payload":{"x":1},"schema_version":"1.0.0"}'
)
HAND_DIGEST = "sha256:" + hashlib.sha256(HAND_CANONICAL_TEXT.encode("utf-8")).hexdigest()


def make_envelope(**overrides) -> ArtifactEnvelope:
    fields = {
        "artifact_type": "hydra2.test",
        "schema_version": "1.0.0",
        "compatibility": "exact",
        "payload": {"x": 1},
    }
    fields.update(overrides)
    return ArtifactEnvelope(**fields)


class TestEnvelopeIdentity:
    def test_artifact_id_matches_hand_computed_canonical_digest(self):
        envelope = make_envelope()
        assert artifact_id(envelope) == HAND_DIGEST
        assert to_json(envelope)["schema_version"] == "1.0.0"

    def test_payload_key_order_irrelevant_but_content_changes_identity(self):
        base = make_envelope(payload={"a": 1, "b": 2})
        reordered = make_envelope(payload={"b": 2, "a": 1})
        assert artifact_id(base) == artifact_id(reordered)  # order-insensitive
        assert artifact_id(make_envelope(payload={"a": 1})) != artifact_id(base)
        assert artifact_id(make_envelope(compatibility="backward_read")) != artifact_id(
            make_envelope()
        )

    def test_envelope_validation(self):
        with pytest.raises(ContractError):
            make_envelope(artifact_type="")
        with pytest.raises(ContractError):
            make_envelope(compatibility="forward")
        with pytest.raises(ContractError):
            make_envelope(schema_version="1.0")

    def test_unknown_major_rejected_at_envelope_boundary(self):
        with pytest.raises(IncompatibleSchemaError):
            envelope_from_json(
                {
                    "artifact_type": "t",
                    "schema_version": "9.9.9",
                    "compatibility": "exact",
                    "payload": {},
                },
                supported_majors=frozenset({1}),
            )

    def test_round_trip_through_json_document(self):
        envelope = make_envelope(payload={"k": [None, True, 2.5, "s"], "e": {}})
        rebuilt = envelope_from_json(to_json(envelope))
        assert rebuilt == envelope
        assert artifact_id(rebuilt) == artifact_id(envelope)


class TestPublishAndLookup:
    def test_publish_then_lookup_returns_verified_envelope_and_row(self, tmp_path):
        registry = ArtifactRegistry(tmp_path)
        envelope = make_envelope(payload={"deep": {"list": [1, 2.5, None]}})
        entry = registry.publish(envelope)
        result = registry.lookup(
            artifact_type=envelope.artifact_type,
            schema_version=envelope.schema_version,
            artifact_id=entry.artifact_id,
        )
        assert result.envelope == envelope
        assert result.entry.compatibility == "exact"
        assert result.entry.migration is None
        path = registry.artifact_path(envelope.artifact_type, entry.artifact_id)
        assert path.read_bytes() == canonical_bytes(to_json(envelope))

    def test_publish_is_idempotent_for_identical_bytes(self, tmp_path):
        registry = ArtifactRegistry(tmp_path)
        envelope = make_envelope()
        first = registry.publish(envelope)
        second = registry.publish(envelope)
        assert first.artifact_id == second.artifact_id

    def test_migration_metadata_round_trips_without_changing_identity(self, tmp_path):
        registry = ArtifactRegistry(tmp_path)
        migration = MigrationMetadata(source_schema_version="0.9.0", migration_id="mig-42")
        envelope = make_envelope(compatibility="backward_read", payload={"v": 2})
        entry = registry.publish(envelope, migration=migration)
        assert artifact_id(envelope) == entry.artifact_id  # metadata is not identity
        result = registry.lookup(
            artifact_type=envelope.artifact_type,
            schema_version="1.0.0",
            artifact_id=entry.artifact_id,
        )
        assert result.entry.migration == migration

    def test_corrupted_repair_attempt_rejected_as_overwrite(self, tmp_path):
        registry = ArtifactRegistry(tmp_path)
        envelope = make_envelope()
        entry = registry.publish(envelope)
        path = registry.artifact_path("hydra2.test", entry.artifact_id)
        path.write_bytes(path.read_bytes().replace(b'"x":1', b'"x":2'))
        with pytest.raises(DigestMismatchError):
            registry.publish(envelope)  # cannot silently repair immutable bytes


class TestLookupRejections:
    def test_truncated_file_rejected(self, tmp_path):
        registry = ArtifactRegistry(tmp_path)
        entry = registry.publish(make_envelope())
        path = registry.artifact_path("hydra2.test", entry.artifact_id)
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) // 2])
        with pytest.raises(CorruptArtifactError):
            registry.lookup(
                artifact_type="hydra2.test",
                schema_version="1.0.0",
                artifact_id=entry.artifact_id,
            )

    def test_tampered_valid_json_rejected_by_digest_before_decode(self, tmp_path):
        registry = ArtifactRegistry(tmp_path)
        entry = registry.publish(make_envelope())
        path = registry.artifact_path("hydra2.test", entry.artifact_id)
        tampered = path.read_bytes().replace(b'"x":1', b'"x":999')
        assert tampered != path.read_bytes()
        path.write_bytes(tampered)  # still valid JSON — digest must catch it first
        with pytest.raises(CorruptArtifactError, match="hash"):
            registry.lookup(
                artifact_type="hydra2.test",
                schema_version="1.0.0",
                artifact_id=entry.artifact_id,
            )

    def test_registered_file_deleted_rejected(self, tmp_path):
        registry = ArtifactRegistry(tmp_path)
        entry = registry.publish(make_envelope())
        registry.artifact_path("hydra2.test", entry.artifact_id).unlink()
        with pytest.raises(CorruptArtifactError):
            registry.lookup(
                artifact_type="hydra2.test",
                schema_version="1.0.0",
                artifact_id=entry.artifact_id,
            )

    def test_identity_collision_address_mismatch_detected(self, tmp_path):
        registry = ArtifactRegistry(tmp_path)
        other = make_envelope(payload={"other": True})
        other_entry = registry.publish(other)
        other_bytes = registry.artifact_path("hydra2.test", other_entry.artifact_id).read_bytes()
        entry = registry.publish(make_envelope())
        address = registry.artifact_path("hydra2.test", entry.artifact_id)
        address.write_bytes(other_bytes)  # foreign content stored under wrong id
        with pytest.raises(CorruptArtifactError):
            registry.lookup(
                artifact_type="hydra2.test",
                schema_version="1.0.0",
                artifact_id=entry.artifact_id,
            )

    def test_unknown_major_version_rejected(self, tmp_path):
        registry = ArtifactRegistry(tmp_path)
        entry = registry.publish(make_envelope())
        with pytest.raises(IncompatibleSchemaError):
            registry.lookup(
                artifact_type="hydra2.test",
                schema_version="2.0.0",
                artifact_id=entry.artifact_id,
            )
        with pytest.raises(IncompatibleSchemaError):
            registry.publish(make_envelope(schema_version="3.0.0"))

    def test_supported_majors_override_allows_second_major(self, tmp_path):
        registry = ArtifactRegistry(tmp_path, supported_majors=frozenset({1, 2}))
        envelope = make_envelope(schema_version="2.1.3")
        entry = registry.publish(envelope)
        result = registry.lookup(
            artifact_type="hydra2.test",
            schema_version="2.1.3",
            artifact_id=entry.artifact_id,
        )
        assert result.envelope.schema_version == SchemaVersion("2.1.3")

    def test_noncanonical_digest_argument_rejected(self, tmp_path):
        registry = ArtifactRegistry(tmp_path)
        for bad in ("SHA256:" + "a" * 64, "deadbeef", "sha256:" + "A" * 64, "md5:abc"):
            with pytest.raises((ContractError, LineageError)):
                registry.lookup(
                    artifact_type="hydra2.test",
                    schema_version="1.0.0",
                    artifact_id=bad,
                )

    def test_unknown_triple_is_lineage_error(self, tmp_path):
        registry = ArtifactRegistry(tmp_path)
        with pytest.raises(LineageError):
            registry.lookup(
                artifact_type="hydra2.absent",
                schema_version="1.0.0",
                artifact_id=HAND_DIGEST,
            )


class TestCompatibilityPolicy:
    def test_unknown_optional_field_requires_backward_read_declaration(self):
        exact_doc = {
            "artifact_type": "t",
            "schema_version": "1.0.0",
            "compatibility": "exact",
            "payload": {},
            "extra": 1,
        }
        with pytest.raises(ContractError):
            envelope_from_json(exact_doc)
        accepted = envelope_from_json({**exact_doc, "compatibility": "backward_read"})
        assert accepted.compatibility == "backward_read"

    def test_missing_required_field_rejected_regardless_of_compatibility(self):
        doc = {
            "artifact_type": "t",
            "schema_version": "1.0.0",
            "compatibility": "backward_read",
            "extra": 1,
        }
        with pytest.raises(ContractError):
            envelope_from_json(doc)

    def test_invalid_enum_rejected(self):
        doc = {
            "artifact_type": "t",
            "schema_version": "1.0.0",
            "compatibility": "sideways",
            "payload": {},
        }
        with pytest.raises(ContractError):
            envelope_from_json(doc)
