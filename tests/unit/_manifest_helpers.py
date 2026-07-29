"""Test-only manifest-hash helper (WP-05B / WP-11).

Builds deterministic ``sha256:<64 hex>`` digests for the 10 checkpoint
manifest keys required by :class:`SupervisedLoop` and
:class:`ActorLearnerReplay`.  TEST-ONLY: real runs supply frozen spec
digests; this helper exists so hermetic unit tests can satisfy the
constructor gate without fabricating production manifests.
"""

from __future__ import annotations

import hashlib

__all__ = ["REQUIRED_MANIFEST_KEYS", "make_test_manifest_hashes"]

REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "run_spec_hash",
    "model_spec_hash",
    "optimizer_spec_hash",
    "scheduler_spec_hash",
    "environment_hash",
    "rules_hash",
    "utility_manifest_hash",
    "action_schema_hash",
    "observation_schema_hash",
    "dataset_manifest_hash",
)


def make_test_manifest_hashes(prefix: str = "test") -> dict[str, str]:
    """Return deterministic test-only digests for all required manifest keys."""
    return {
        key: "sha256:" + hashlib.sha256(f"{prefix}:{key}".encode()).hexdigest()
        for key in REQUIRED_MANIFEST_KEYS
    }
