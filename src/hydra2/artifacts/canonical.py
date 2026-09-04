"""RFC 8785 (JCS) canonical JSON — the Hydra2 identity-bytes authority.

SPEC 2.2: identity artifacts hash ``canonical_bytes(value)`` where the bytes
are RFC 8785 JSON Canonicalization Scheme output: UTF-8, no BOM, no
whitespace, ECMAScript ``Number::toString`` numbers, minimal string escaping,
object keys sorted as UTF-16 code-unit arrays.

Domain (closed): null, bool, string, finite number, array, string-keyed
object. NaN/Inf, non-string object keys, integers outside the IEEE 754
double-safe range, lone surrogates, and duplicate keys at the parse boundary
raise :class:`CanonicalizationError`/:class:`ContractError`.
"""

from __future__ import annotations

import json
import math
from typing import Any, cast

from hydra2.contracts.common import CanonicalizationError, ContractError

__all__ = [
    "MAX_SAFE_INTEGER",
    "canonical_bytes",
    "canonicalize",
    "es6_number_to_string",
    "loads_canonical",
]

#: Largest magnitude representable exactly in IEEE 754 double and JSON-safe.
#: Python ``int`` inputs beyond this are rejected as non-canonical-safe; use a
#: float or decimal string per SPEC 2.2 ("Floats in identity artifacts MUST be
#: finite") and RFC 8785 Appendix D for larger values.
MAX_SAFE_INTEGER = 2**53 - 1

_SHORT_ESCAPES = {
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0C: "\\f",
    0x0D: "\\r",
}


def es6_number_to_string(value: float) -> str:
    """Serialize an IEEE 754 double per ECMA-262 ``Number::toString(10)``.

    Shortest round-trip digits (CPython ``repr`` is shortest round-trip)
    reformatted with the ECMAScript placement rules; verified against all
    RFC 8785 Appendix B vectors and V8 ``JSON.stringify`` on random doubles.
    """
    if math.isnan(value) or math.isinf(value):
        raise CanonicalizationError(f"non-finite number {value!r} has no canonical serialization")
    if value == 0:  # covers -0.0 -> "0" (RFC 8785 Appendix B row 2)
        return "0"
    sign = "-" if value < 0 else ""
    mantissa, _, exponent_text = repr(abs(value)).partition("e")
    exponent10 = int(exponent_text) if exponent_text != "" else 0
    integer_part, _, fraction_part = mantissa.partition(".")
    raw_digits = integer_part + fraction_part
    stripped_digits = raw_digits.lstrip("0").rstrip("0")
    digits = stripped_digits if stripped_digits != "" else "0"
    k = len(digits)
    trailing_stripped = len(raw_digits) - len(raw_digits.rstrip("0"))
    # value = int(digits) * 10**(n - k); n = position of the decimal point.
    n = k + trailing_stripped + exponent10 - len(fraction_part)
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    scientific_exponent = n - 1
    exponent_form = (
        f"e+{scientific_exponent}" if scientific_exponent >= 0 else f"e{scientific_exponent}"
    )
    head = digits[0] + ("." + digits[1:] if k > 1 else "")
    return sign + head + exponent_form


def _require_valid_unicode(text: str) -> None:
    try:
        _ = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CanonicalizationError(
            f"string contains an unpaired surrogate (invalid Unicode): {text!r}"
        ) from exc


def _serialize_string(text: str) -> str:
    _require_valid_unicode(text)
    pieces: list[str] = ['"']
    for char in text:
        code = ord(char)
        if code in _SHORT_ESCAPES:
            pieces.append(_SHORT_ESCAPES[code])
        elif code < 0x20:
            pieces.append(f"\\u{code:04x}")
        elif char == '"':
            pieces.append('\\"')
        elif char == "\\":
            pieces.append("\\\\")
        else:
            pieces.append(char)
    pieces.append('"')
    return "".join(pieces)


def _utf16be_key(key: str) -> bytes:
    """UTF-16BE sort key (RFC 8785 §3.2.3); module-level so `sorted` does not
    rebuild a closure per dict."""
    return key.encode("utf-16-be")


def _serialize(value: Any) -> str:
    # Branch order follows measured corpus frequency (identity/observation
    # docs are str/int/list/dict-heavy; bool stays before int since bool
    # subclasses int; float/None are rare).
    if isinstance(value, str):
        return _serialize_string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise CanonicalizationError(
                f"integer {value} exceeds the IEEE 754 double-safe range "
                f"(±{MAX_SAFE_INTEGER}); serialize it as a float or string"
            )
        return str(value)
    if isinstance(value, list):
        list_value: list[Any] = cast("list[Any]", value)
        return "[" + ",".join([_serialize(cast("Any", item)) for item in list_value]) + "]"
    if isinstance(value, dict):
        dict_value: dict[str, Any] = cast("dict[str, Any]", value)
        for key in dict_value:
            if not isinstance(key, str):
                raise CanonicalizationError(
                    f"object key {key!r} is not a string; JSON objects are string-keyed only"
                )
            _require_valid_unicode(key)
        # RFC 8785 §3.2.3: sort by UTF-16 code units of the raw key text.
        ordered_keys: list[str] = sorted(dict_value, key=_utf16be_key)
        members = [
            f"{_serialize_string(key)}:{_serialize(cast('Any', dict_value[key]))}"
            for key in ordered_keys
        ]
        return "{" + ",".join(members) + "}"
    if isinstance(value, float):
        return es6_number_to_string(value)
    if value is None:
        return "null"
    raise CanonicalizationError(
        f"value of type {type(value).__name__} is outside the canonical JSON domain"
    )


def canonicalize(value: Any) -> str:
    """RFC 8785 canonical JSON text for ``value`` (no trailing newline)."""
    return _serialize(value)


def canonical_bytes(value: Any) -> bytes:
    """RFC 8785 canonical UTF-8 bytes for ``value``; SPEC 2.2 identity form."""
    # Every string (values and keys) is validated during serialization, so the
    # final encode cannot fail on unpaired surrogates.
    return canonicalize(value).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in pairs:
        if key in result:
            raise CanonicalizationError(f"duplicate object key {key!r} in input document")
        result[key] = item
    return result


def _reject_json_constant(token: str) -> Any:
    raise CanonicalizationError(
        f"{token} is not part of the canonical JSON domain (finite numbers only)"
    )


def loads_canonical(document: str | bytes | bytearray) -> Any:
    """Parse a JSON document with I-JSON boundary enforcement.

    Rejects UTF-8 BOM, duplicate object keys, and NaN/Infinity literals so a
    re-serialization of the result stays canonical. Output tolerance follows
    RFC 8259; identity always comes from :func:`canonical_bytes`.
    """
    if isinstance(document, (bytes, bytearray)):
        if bytes(document[:3]) == b"\xef\xbb\xbf":
            raise CanonicalizationError("UTF-8 BOM is not permitted in canonical documents")
        text = bytes(document).decode("utf-8")
    else:
        if document.startswith("\ufeff"):
            raise CanonicalizationError("UTF-8 BOM is not permitted in canonical documents")
        text = document
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ContractError(f"input is not valid JSON: {exc}") from exc
