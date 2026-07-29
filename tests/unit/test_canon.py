"""Bootstrap canonical-serialization surface after the WP-02A supersede.

``hydra2._canon`` now delegates to :mod:`hydra2.artifacts` (RFC 8785); these
tests pin the shim's names and the qualified semantics WP-01 callers rely on.
"""

from __future__ import annotations

import json

import pytest

from hydra2._canon import (
    atomic_write_bytes,
    canonical_json_bytes,
    require_digest_match,
    sha256_digest,
    sha256_digest_of_json,
)
from hydra2.contracts.common import (
    CanonicalizationError,
    DigestMismatchError,
)


class TestCanonicalBytes:
    def test_sorted_keys_no_whitespace(self):
        data = {"b": 1, "a": {"y": [1, 2], "x": None}}
        assert canonical_json_bytes(data) == b'{"a":{"x":null,"y":[1,2]},"b":1}'

    def test_key_order_irrelevant(self):
        assert canonical_json_bytes({"a": 1, "b": 2}) == canonical_json_bytes({"b": 2, "a": 1})

    def test_unicode_preserved_utf8(self):
        assert canonical_json_bytes({"k": "麻"}) == '{"k":"麻"}'.encode()

    def test_rfc8785_number_forms(self):
        assert canonical_json_bytes([1.0, -0.0, 5e-7, 0.000001, 1e21]) == (
            b"[1,0,5e-7,0.000001,1e+21]"
        )

    def test_utf16_code_unit_key_order(self):
        # U+FB33 sorts AFTER the surrogate pair of U+1F600 in UTF-16 order,
        # but BEFORE it in Python code-point order.
        data = {"\ufb33": 1, "\U0001f600": 2}
        assert canonical_json_bytes(data) == '{"😀":2,"\ufb33":1}'.encode()

    def test_round_trip_stable(self):
        value = {"nested": {"list": [1, 2.5, True, None]}, "s": "text"}
        once = canonical_json_bytes(value)
        reparsed = json.loads(once.decode("utf-8"))
        assert canonical_json_bytes(reparsed) == once

    def test_non_finite_float_rejected_typed(self):
        with pytest.raises(CanonicalizationError):
            canonical_json_bytes({"x": float("nan")})
        with pytest.raises(CanonicalizationError):
            canonical_json_bytes({"x": float("inf")})

    def test_unknown_type_rejected(self):
        with pytest.raises(CanonicalizationError):
            canonical_json_bytes({"x": object()})

    def test_integer_beyond_safe_range_rejected(self):
        with pytest.raises(CanonicalizationError):
            canonical_json_bytes(2**53)


class TestDigestHelpers:
    def test_sha256_digest_format(self):
        digest = sha256_digest(b"")
        assert digest == "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_sha256_digest_of_json_matches_manual(self):
        value = {"z": 1, "a": 2}
        expect = sha256_digest(canonical_json_bytes(value))
        assert sha256_digest_of_json(value) == expect

    def test_require_digest_match_passes_on_equal(self):
        require_digest_match(
            recorded=sha256_digest(b"x"), recomputed=sha256_digest(b"x"), subject="t"
        )

    def test_require_digest_match_raises_typed(self):
        with pytest.raises(DigestMismatchError):
            require_digest_match(
                recorded=sha256_digest(b"x"), recomputed=sha256_digest(b"y"), subject="t"
            )

    def test_require_digest_match_rejects_malformed_recorded(self):
        with pytest.raises(DigestMismatchError):
            require_digest_match(recorded="deadbeef", recomputed=None, subject="t")


class TestAtomicWrite:
    def test_write_then_read_exact(self, tmp_path):
        target = tmp_path / "sub" / "artifact.json"
        atomic_write_bytes(target, b"\x00hydra2")
        assert target.read_bytes() == b"\x00hydra2"

    def test_overwrite_is_atomic_and_complete(self, tmp_path):
        target = tmp_path / "artifact.json"
        atomic_write_bytes(target, b"first")
        atomic_write_bytes(target, b"second-version")
        assert target.read_bytes() == b"second-version"
        leftovers = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name]
        assert leftovers == []
