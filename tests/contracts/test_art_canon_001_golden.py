"""ART-CANON-001: RFC 8785 golden vectors — keys, numbers, Unicode, exact bytes.

Vectors transcribed from RFC 8785 (§3.2.2 example, §3.2.3 sorting fixture,
§3.2.4 UTF-8 bytes, Appendix B number table).
"""

from __future__ import annotations

import hashlib
import struct

import pytest

from hydra2.artifacts.canonical import (
    canonical_bytes,
    canonicalize,
    es6_number_to_string,
    loads_canonical,
)
from hydra2.artifacts.digest import sha256_digest
from hydra2.contracts.common import CanonicalizationError

pytestmark = pytest.mark.contract_package("WP-02A")

# RFC 8785 §3.2.2 input document and §3.2.2/§3.2.3 canonical output.
RFC_INPUT_JSON = (
    '{"numbers":[333333333.33333329,1E30,4.50,2e-3,0.000000000000000000000000001],'
    '"string":"\\u20ac$\\u000F\\u000aA\'\\u0042\\u0022\\u005c\\\\\\"\\/",'
    '"literals":[null,true,false]}'
)
_STRING_ESCAPES = ["€", "$", "\\u000f", "\\n", "A", "'", "B", '\\"', "\\\\", "\\\\", '\\"', "/"]
RFC_EXPECTED_CANONICAL = (
    '{"literals":[null,true,false],'
    '"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
    '"string":' + '"' + "".join(_STRING_ESCAPES) + '"' + "}"
)
# RFC 8785 §3.2.4: exact UTF-8 bytes of the canonicalized sample.
RFC_EXPECTED_HEX = (
    "7b226c69746572616c73223a5b6e756c6c2c747275652c66616c73655d2c226e756d626572"
    "73223a5b3333333333333333332e333333333333332c31652b33302c342e352c302e303032"
    "2c31652d32375d2c22737472696e67223a22e282ac245c75303030665c6e4127425c225c5c5c"
    "5c5c222f227d"
)

# RFC 8785 Appendix B: IEEE 754 hex -> ECMAScript JSON text.
APPENDIX_B_VECTORS = [
    ("0000000000000000", "0"),
    ("8000000000000000", "0"),
    ("0000000000000001", "5e-324"),
    ("8000000000000001", "-5e-324"),
    ("7fefffffffffffff", "1.7976931348623157e+308"),
    ("ffefffffffffffff", "-1.7976931348623157e+308"),
    ("4340000000000000", "9007199254740992"),
    ("c340000000000000", "-9007199254740992"),
    ("4430000000000000", "295147905179352830000"),
    ("44b52d02c7e14af5", "9.999999999999997e+22"),
    ("44b52d02c7e14af6", "1e+23"),
    ("44b52d02c7e14af7", "1.0000000000000001e+23"),
    ("444b1ae4d6e2ef4e", "999999999999999700000"),
    ("444b1ae4d6e2ef4f", "999999999999999900000"),
    ("444b1ae4d6e2ef50", "1e+21"),
    ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"),
    ("3eb0c6f7a0b5ed8d", "0.000001"),
    ("41b3de4355555553", "333333333.3333332"),
    ("41b3de4355555554", "333333333.33333325"),
    ("41b3de4355555555", "333333333.3333333"),
    ("41b3de4355555556", "333333333.3333334"),
    ("41b3de4355555557", "333333333.33333343"),
    ("becbf647612f3696", "-0.0000033333333333333333"),
    ("43143ff3c1cb0959", "1424953923781206.2"),
]

# RFC 8785 §3.2.3 sorting fixture: raw keys and the required UTF-16 order.
SORT_KEYS_RAW = {
    "\u20ac": "Euro Sign",
    "\r": "Carriage Return",
    "\ufb33": "Hebrew Letter Dalet With Dagesh",
    "1": "One",
    "\U0001f600": "Emoji: Grinning Face",
    "\u0080": "Control",
    "\u00f6": "Latin Small Letter O With Diaeresis",
}
SORT_KEYS_EXPECTED_ORDER = [
    "\r",
    "1",
    "\u0080",
    "\u00f6",
    "\u20ac",
    "\U0001f600",
    "\ufb33",
]


class TestSectionExample:
    def test_example_document_canonical_text(self):
        doc = loads_canonical(RFC_INPUT_JSON)
        assert canonicalize(doc) == RFC_EXPECTED_CANONICAL

    def test_example_document_exact_utf8_bytes(self):
        doc = loads_canonical(RFC_INPUT_JSON)
        assert canonical_bytes(doc).hex() == RFC_EXPECTED_HEX


class TestAppendixBNumbers:
    @pytest.mark.parametrize(("hex754", "expected_text"), APPENDIX_B_VECTORS)
    def test_vector(self, hex754: str, expected_text: str):
        value = struct.unpack(">d", bytes.fromhex(hex754))[0]
        assert es6_number_to_string(value) == expected_text
        assert canonicalize([value]) == "[" + expected_text + "]"

    def test_nan_and_infinity_rejected(self):
        with pytest.raises(CanonicalizationError):
            canonicalize(float("nan"))
        with pytest.raises(CanonicalizationError):
            canonicalize([float("inf"), float("-inf")])


class TestPropertySorting:
    def test_rfc_sort_fixture_order(self):
        ordered_text = canonicalize(SORT_KEYS_RAW)
        parsed_keys = list(loads_canonical(ordered_text))
        assert parsed_keys == SORT_KEYS_EXPECTED_ORDER

    def test_sorting_is_recursive_into_child_objects_and_arrays(self):
        doc = {"z": {"n": 1, "a": 2}, "arr": [{"y": 1, "x": 2}]}
        text = canonicalize(doc)
        assert text.index('"a"') < text.index('"n"')
        assert text.index('"x"') < text.index('"y"')
        # array element order itself is preserved
        assert canonicalize([[2, 1]]) == "[[2,1]]"


class TestStringEscaping:
    def test_control_characters_shortest_form_then_lowercase_unicode(self):
        assert canonicalize("\b\t\n\f\r") == '"\\b\\t\\n\\f\\r"'
        assert canonicalize("\x01\x1f") == '"\\u0001\\u001f"'

    def test_quote_backslash_escaped_slash_and_del_literal(self):
        # "/" stays literal, '"' and "\\" are escaped, DEL (U+007F) stays literal.
        assert canonicalize('/"\\\x7f') == '"/' + '\\"' + "\\\\" + "\x7f" + '"'

    def test_string_round_trip_through_parse(self):
        for text in ["plain", '€\n\t"\\', "麻雀", "\U0001f600"]:
            parsed = loads_canonical(canonicalize(text))
            assert canonicalize(parsed) == canonicalize(text)


class TestIndependentDigestRecomputation:
    def test_section_bytes_digest_matches_hashlib_directly(self):
        doc = loads_canonical(RFC_INPUT_JSON)
        expected = "sha256:" + hashlib.sha256(bytes.fromhex(RFC_EXPECTED_HEX)).hexdigest()
        assert sha256_digest(canonical_bytes(doc)) == expected
