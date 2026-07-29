"""WP-02A supersede gate: ``hydra2._canon`` delegates to ``hydra2.artifacts``."""

from __future__ import annotations

import hashlib
import json

import pytest

from hydra2 import _canon
from hydra2.artifacts.atomic import atomic_replace_bytes
from hydra2.artifacts.canonical import canonical_bytes
from hydra2.artifacts.digest import of_canonical

pytestmark = pytest.mark.contract_package("WP-02A")


class TestShimDelegation:
    def test_canonical_json_bytes_is_rfc8785_qualified(self):
        data = {"n": [1.0, -0.0], "\U0001f600": 1, "\ufb33": 2}
        assert _canon.canonical_json_bytes(data) == canonical_bytes(data)
        # RFC 8785 number forms, not the WP-01 stdlib approximation.
        assert _canon.canonical_json_bytes([1.0, -0.0, 5e-7]) == b"[1,0,5e-7]"

    def test_sha256_digest_of_json_delegates(self):
        value = {"z": [1, 2.5], "a": None}
        assert _canon.sha256_digest_of_json(value) == of_canonical(value)

    def test_atomic_write_bytes_behaves_like_replace_writer(self, tmp_path):
        target = tmp_path / "control.json"
        _canon.atomic_write_bytes(target, b"one")
        _canon.atomic_write_bytes(target, b"two")
        assert target.read_bytes() == b"two"
        assert [p.name for p in tmp_path.iterdir()] == ["control.json"]
        # same underlying primitive
        atomic_replace_bytes(target, b"three")
        assert target.read_bytes() == b"three"

    def test_legacy_names_still_importable_for_wp01_callers(self):
        from hydra2._canon import (  # noqa: F401
            require_digest_match,
            sha256_digest,
            sha256_file,
        )


class TestVerifyCompatibility:
    """`hydra2 work-package verify` contract MUST NOT change."""

    def test_string_only_document_hashes_identically_to_bootstrap_algorithm(self):
        doc = {
            "work_package": "WP-01",
            "status": "passed",
            "inputs": [{"id": "a", "sha256": "sha256:" + "0" * 64}],
            "commands": [{"argv": ["pixi", "run", "test"], "exit_code": 0}],
        }
        bootstrap_bytes = json.dumps(
            doc,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert _canon.canonical_json_bytes(doc) == bootstrap_bytes
        expected = "sha256:" + hashlib.sha256(bootstrap_bytes).hexdigest()
        assert _canon.sha256_digest_of_json(doc) == expected
