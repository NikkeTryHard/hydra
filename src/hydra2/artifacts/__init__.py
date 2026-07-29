"""Canonical hashing, digests, atomic publication, immutable registry (WP-02A).

SPEC 1 rule: this package MUST NOT import engines, tensors, training, or
tracking; stdlib + contracts only.
"""

from hydra2.artifacts.atomic import atomic_replace_bytes, publish_atomic
from hydra2.artifacts.canonical import (
    MAX_SAFE_INTEGER,
    canonical_bytes,
    canonicalize,
    es6_number_to_string,
    loads_canonical,
)
from hydra2.artifacts.digest import (
    of_bytes,
    of_canonical,
    require_digest_match,
    sha256_digest,
    sha256_file,
    validate_digest,
)
from hydra2.artifacts.registry import (
    COMPATIBILITY_VALUES,
    INDEX_SCHEMA_VERSION,
    ArtifactEnvelope,
    ArtifactRegistry,
    MigrationMetadata,
    RegistryEntry,
    artifact_id,
    envelope_from_json,
    to_json,
)

__all__ = [
    "COMPATIBILITY_VALUES",
    "INDEX_SCHEMA_VERSION",
    "MAX_SAFE_INTEGER",
    "ArtifactEnvelope",
    "ArtifactRegistry",
    "MigrationMetadata",
    "RegistryEntry",
    "artifact_id",
    "atomic_replace_bytes",
    "canonical_bytes",
    "canonicalize",
    "envelope_from_json",
    "es6_number_to_string",
    "loads_canonical",
    "of_bytes",
    "of_canonical",
    "publish_atomic",
    "require_digest_match",
    "sha256_digest",
    "sha256_file",
    "to_json",
    "validate_digest",
]
