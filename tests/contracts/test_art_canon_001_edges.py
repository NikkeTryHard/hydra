"""Numeric/Unicode edge cases and canonical-domain adversarial rejections."""

from __future__ import annotations

import pytest

from hydra2.artifacts.canonical import (
    MAX_SAFE_INTEGER,
    canonical_bytes,
    canonicalize,
    es6_number_to_string,
    loads_canonical,
)
from hydra2.contracts.common import CanonicalizationError, ContractError

pytestmark = pytest.mark.contract_package("WP-02A")


class TestNumberEdges:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1.0, "1"),
            (-0.0, "0"),
            (0.0, "0"),
            (1e21, "1e+21"),
            (1e20, "100000000000000000000"),
            (5e-7, "5e-7"),
            (1e-6, "0.000001"),
            (1e-7, "1e-7"),
            (100.0, "100"),
            (0.1, "0.1"),
            (1424953923781206.25, "1424953923781206.2"),  # round-to-even (App B note 4)
            (9007199254740992.0, "9007199254740992"),  # 2**53 as a double is legal
        ],
    )
    def test_es6_forms(self, value: float, expected: str):
        assert es6_number_to_string(value) == expected
        assert canonicalize([value]) == "[" + expected + "]"

    def test_integers_within_safe_domain_are_decimal(self):
        assert canonicalize([0, 1, -1, MAX_SAFE_INTEGER, -MAX_SAFE_INTEGER]) == (
            "[0,1,-1,9007199254740991,-9007199254740991]"
        )

    def test_integer_beyond_safe_domain_rejected(self):
        for value in (2**53, -(2**53), 2**64, 9223372036854775807):
            with pytest.raises(CanonicalizationError):
                canonicalize(value)

    def test_bool_is_literal_not_number(self):
        assert canonicalize([True, False]) == "[true,false]"


class TestUnicodeKeyOrderEdges:
    def test_non_bmp_key_orders_by_leading_surrogate(self):
        # U+1F600 encodes as the surrogate pair <D83D DE00>. Its leading unit
        # D83D precedes FB33 and FFFF, so UTF-16 order differs from Python
        # code-point order (which would place U+FB33 before U+1F600).
        doc = {"\ufb33": 1, "\U0001f600": 2, "\uffff": 3}
        assert list(loads_canonical(canonicalize(doc))) == [
            "\U0001f600",
            "\ufb33",
            "\uffff",
        ]

    def test_escape_form_and_literal_form_identify_same_key(self):
        # JSON input escapes vs literal UTF-8 decode to identical keys;
        # duplicate detection happens on decoded keys.
        with pytest.raises(CanonicalizationError):
            loads_canonical('{"\\u20ac":1,"€":2}')

    def test_prefix_rule_shorter_first(self):
        doc = {"ab": 1, "abc": 2, "a": 3}
        assert list(loads_canonical(canonicalize(doc))) == ["a", "ab", "abc"]

    def test_empty_key_sorts_first(self):
        doc = {"a": 1, "": 2}
        assert list(loads_canonical(canonicalize(doc))) == ["", "a"]


class TestContainers:
    def test_empty_containers(self):
        assert canonical_bytes({}) == b"{}"
        assert canonical_bytes([]) == b"[]"
        assert canonical_bytes({"a": {}, "b": []}) == b'{"a":{},"b":[]}'

    def test_nested_arrays_keep_element_order(self):
        value = {"m": [[3, 1, 2], [{"z": 1, "a": 2}], []]}
        assert canonical_bytes(value) == b'{"m":[[3,1,2],[{"a":2,"z":1}],[]]}'

    def test_deep_recursion(self):
        depth = 200
        value: object = []
        for _ in range(depth):
            value = [value]
        levels = depth + 1  # initial [] plus one level per wrap
        assert canonicalize(value) == "[" * levels + "]" * levels


class TestDomainRejections:
    def test_nan_inf_rejected_everywhere(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(CanonicalizationError):
                canonicalize(bad)
            with pytest.raises(CanonicalizationError):
                canonicalize({"x": [bad]})

    def test_non_string_keys_rejected(self):
        with pytest.raises(CanonicalizationError):
            canonicalize({1: "a"})
        with pytest.raises(CanonicalizationError):
            canonicalize({(1, 2): "a"})
        with pytest.raises(CanonicalizationError):
            canonicalize({None: "a"})

    def test_outside_json_domain_rejected(self):
        for bad in ({1, 2}, b"bytes", ("tuple",), 1j, object(), Ellipsis):
            with pytest.raises(CanonicalizationError):
                canonicalize(bad)

    def test_lone_surrogate_rejected_in_values_and_keys_typed(self):
        with pytest.raises(CanonicalizationError):
            canonicalize("\ud800")
        with pytest.raises(CanonicalizationError):
            canonicalize({"k\udfff": 1})

    def test_parse_boundary_duplicate_keys(self):
        with pytest.raises(CanonicalizationError):
            loads_canonical('{"a":1,"a":2}')

    def test_parse_boundary_nan_constants(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with pytest.raises(CanonicalizationError):
                loads_canonical(token)

    def test_parse_boundary_bom_rejected(self):
        with pytest.raises(CanonicalizationError):
            loads_canonical(b"\xef\xbb\xbf{}")
        with pytest.raises(CanonicalizationError):
            loads_canonical("﻿{}")

    def test_parse_boundary_malformed_json_is_contract_error(self):
        with pytest.raises(ContractError):
            loads_canonical("{not json")

    def test_no_trailing_newline_or_whitespace_emitted(self):
        data = canonical_bytes({"k": 1})
        assert not data.endswith(b"\n")
        assert b" " not in data
