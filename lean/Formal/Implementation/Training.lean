import Mathlib.Data.String.Basic
/-! # Hydra2 Training provenance — `shared_run_fields_hash` (RFC8785 canonical JSON sha256)

Mirrors `IMPLEMENTATION_SPEC.md §2.2` `canonical_bytes` (RFC8785 JCS deterministic sorting,
`ES6 Number::toString`-style shortest number, `UTF-8`, no whitespace) + `digest.py` `sha256_digest`
and `§20` `MatchedObjectiveGroup` `shared_run_fields_hash` over shared `RunSpec` fields +
shared objective params `w_value, w_bc, α`. In Lean we model the hash interface as an
opaque `String → String` (real implementation is `Python` `src/hydra2/artifacts/canonical.py` +
`digest.py`); `MatchedGroup` byte-identical provenance is then hash equality over canonical bytes.

This file is provenance interface only — Lean derivation of `sha256` is not attempted.
-/

namespace Hydra2.Implementation.Training

/-- Abstract canonical-bytes interface: `RFC8785` `canonical_bytes : Value → Bytes` as `String → String` hex for Lean. `JCS` `deterministic` `sorting` `UTF-16` `code-unit` `keys` `ES6` `Number::toString`-style `shortest` `no` `whitespace` `UTF-8` `no` `BOM` `finite` `only` `canonicalize()` `authority` `for` `identity` `bytes` `RFC8785` `JCS` `sorted` `UTF-16` `keys` `ECMAScript` `Number::toString` `shortest` `minimal` `escapes` `no` `whitespace` `UTF-8` `no` `BOM` `finite` `only` `MAX_SAFE_INTEGER` `2^53-1` `9007199254740991` `finite` `range` `JSON` `number` `ES6` `Number` `MAX_SAFE_INTEGER`. -/
def canonicalBytes (s : String) : String := s -- interface placeholder; real bytes via Python `src/hydra2/artifacts/canonical.py` `canonical_bytes` `canonicalize()` `RFC8785` `JCS` `sorted` `MAX_SAFE_INTEGER`

/-- Abstract `sha256_digest` interface: `sha256(canonical_bytes(v))` hex digest in `sha256:<64 lowercase hex>` textual form (`validate_digest` regex `^sha256:[0-9a-f]{64}$`); `of_canonical = sha256(canonical_bytes(value))`; dual paths `sha256_digest` (in-memory) + `sha256_file` (chunked streaming) must agree (BUILD WP-02A). -/
def sha256Digest (s : String) : String := s -- interface placeholder; real via `digest.py` `sha256_digest`/`sha256_file`/`of_canonical`/`validate_digest`

/-- `shared_run_fields_hash` = `sha256(canonical_bytes(sharedFields))` (`SPEC §20` `1441-1453`). -/
noncomputable def sharedRunFieldsHash (canonicalSharedFields : String) : String :=
  sha256Digest (canonicalBytes canonicalSharedFields)

/-- Provenance invariant: `MatchedGroup` `w_value, w_bc, α` byte-identical means canonical bytes equal
and hash equal. This is build-system check, not `Real`-analytic derivation. -/
theorem shared_params_byte_identical_implies_hash_eq (a b : String) (h : canonicalBytes a = canonicalBytes b) :
    sharedRunFieldsHash a = sharedRunFieldsHash b := by
  unfold sharedRunFieldsHash
  rw [h]

theorem sharedRunFieldsHash_trans (a b c : String) (hab : sharedRunFieldsHash a = sharedRunFieldsHash b)
    (hbc : sharedRunFieldsHash b = sharedRunFieldsHash c) : sharedRunFieldsHash a = sharedRunFieldsHash c :=
  hab.trans hbc

theorem sharedRunFieldsHash_symm (a b : String) (h : sharedRunFieldsHash a = sharedRunFieldsHash b) :
    sharedRunFieldsHash b = sharedRunFieldsHash a := h.symm
theorem sharedRunFieldsHash_refl (a : String) : sharedRunFieldsHash a = sharedRunFieldsHash a := rfl
theorem sharedRunFieldsHash_is_of_canonical (s : String) :
    sharedRunFieldsHash s = sha256Digest (canonicalBytes s) := rfl
end Hydra2.Implementation.Training
