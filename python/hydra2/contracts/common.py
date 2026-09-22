"""SPEC 2.1 primitive aliases and SPEC 3 typed failure hierarchy.

This module is the bootstrap subset of the contract layer owned by WP-01.
Full contract modules (rules, utility, tile, action, event, observation)
arrive in WP-02; nothing here anticipates them.

Deletion wave (minimal-Python end-state): the R2 comparator/validator
translators (``is_seat``/``is_tile``/``is_digest``, ``make_seat``/
``make_tile_id``/``make_action_id``/``make_digest_text``) are deleted —
callers import the ``hydra2._native.contracts`` bridge directly
(parity battery + differential probe green on this tree; the bridge raises
``ValueError``/``TypeError`` where the translators raised ``ContractError``).
This module keeps the SPEC 2.1 aliases, the SPEC 3 failure hierarchy, and
the makers with no bridge counterpart.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NewType

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal; oracle branch below decides
    _bridge_contracts = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# SPEC 3 - Failure model (typed errors). All expected Hydra2 failures derive
# from this exact hierarchy; library code raises these and never bare
# RuntimeError for expected conditions.
# ---------------------------------------------------------------------------
# Single source: the Rust SPEC 3 hierarchy
# (``crates/bridge/src/common_errors.rs``, registered on the shared
# ``hydra2._native.contracts`` submodule). The names below alias the bridge
# classes, so MRO / ``except`` / ``str`` / pickling all follow the Rust
# types; ``__module__`` stays ``hydra2.contracts.common`` (set Rust-side),
# which keeps pickle round-trips (xdist workers pickle exceptions) routing
# through this module. A missing/stale ``.so`` falls back to the verbatim
# oracle classes (``getattr``-default probe only, byte-identical behavior).
if getattr(_bridge_contracts, "Hydra2Error", None) is not None:
    # Direct access (not getattr): ruff B009 forbids constant-attribute
    # getattr, and pyrefly types the two-arg default form as None while
    # direct access on the unresolvable native module is Any (callable,
    # catchable, subclassable-with-ignore). The probe above keeps the
    # stale-.so fallback; a partial .so fails fast here, fail closed.
    assert _bridge_contracts is not None
    Hydra2Error: type[Exception] = _bridge_contracts.Hydra2Error
    ContractError: type[Exception] = _bridge_contracts.ContractError
    IncompatibleSchemaError: type[Exception] = _bridge_contracts.IncompatibleSchemaError
    CanonicalizationError: type[Exception] = _bridge_contracts.CanonicalizationError
    DigestMismatchError: type[Exception] = _bridge_contracts.DigestMismatchError
    RulesMismatchError: type[Exception] = _bridge_contracts.RulesMismatchError
    InvalidTileError: type[Exception] = _bridge_contracts.InvalidTileError
    InvalidActionError: type[Exception] = _bridge_contracts.InvalidActionError
    VisibilityViolationError: type[Exception] = _bridge_contracts.VisibilityViolationError
    IllegalActionError: type[Exception] = _bridge_contracts.IllegalActionError
    CorruptArtifactError: type[Exception] = _bridge_contracts.CorruptArtifactError
    LineageError: type[Exception] = _bridge_contracts.LineageError
    QuarantinedError: type[Exception] = _bridge_contracts.QuarantinedError
    UnsupportedRuleError: type[Exception] = _bridge_contracts.UnsupportedRuleError
    DeterminismError: type[Exception] = _bridge_contracts.DeterminismError
    StaleBeliefError: type[Exception] = _bridge_contracts.StaleBeliefError
    PacketPartitionError: type[Exception] = _bridge_contracts.PacketPartitionError
    ProposalSupportError: type[Exception] = _bridge_contracts.ProposalSupportError
    DeadlineExceededError: type[Exception] = _bridge_contracts.DeadlineExceededError
    QualificationRequiredError: type[Exception] = _bridge_contracts.QualificationRequiredError
else:
    # Oracle fallback (stale/unbuilt ``.so``): verbatim HEAD definitions.
    class Hydra2Error(Exception):
        """Base class of every expected Hydra2 failure."""

    class ContractError(Hydra2Error):
        """A value violated a canonical contract (range, enum, schema shape)."""

    class IncompatibleSchemaError(ContractError):
        """Unknown or unsupported major schema version."""

    class CanonicalizationError(ContractError):
        """A value cannot be represented canonically (NaN/Inf/nondeterministic map)."""

    class DigestMismatchError(ContractError):
        """A recomputed digest does not match a recorded digest."""

    class RulesMismatchError(ContractError):
        """Rules identity differs between artifacts or runtime expectations."""

    class InvalidTileError(ContractError):
        """A physical/logical tile id is out of range or otherwise invalid."""

    class InvalidActionError(ContractError):
        """An action id is outside the canonical vocabulary or illegal here."""

    class VisibilityViolationError(ContractError):
        """Actor-visible data leaked hidden-world information."""

    class IllegalActionError(ContractError):
        """An action was taken that the legal mask excludes."""

    class CorruptArtifactError(Hydra2Error):
        """Stored bytes do not match their recorded identity."""

    class LineageError(Hydra2Error):
        """Provenance/lineage chain is missing or inconsistent."""

    class QuarantinedError(Hydra2Error):
        """Tried to consume quarantined data."""

    class UnsupportedRuleError(Hydra2Error):
        """Rule configuration is not supported by this build."""

    class DeterminismError(Hydra2Error):
        """A determinism invariant was violated."""

    class StaleBeliefError(Hydra2Error):
        """Belief state does not match the current epoch."""

    class PacketPartitionError(Hydra2Error):
        """Packet partition bookkeeping is inconsistent."""

    class ProposalSupportError(Hydra2Error):
        """Proposal distribution lacks required support."""

    class DeadlineExceededError(Hydra2Error):
        """Search deadline expired (expected control flow at runner boundary only)."""

    class QualificationRequiredError(Hydra2Error):
        """Path requires a qualification token that is absent."""


# ---------------------------------------------------------------------------
# PR4 diagnostic codes (SPEC 3): bijective code -> error class for the semantic
# search/belief raise sites. Routers match on these codes, never on message text.
# Pure validation keeps byte-identical messages (no code suffix there).
# ---------------------------------------------------------------------------
def _pbrf_from_bridge() -> dict[str, type[Exception]] | None:
    """Bridge table on ``hydra2._native.contracts`` or ``None`` when stale."""
    if _bridge_contracts is None:
        return None
    codes: tuple[str, ...] | None = getattr(_bridge_contracts, "RECORD_PBRF_CODES", None)
    lookup: Callable[[str], str] | None = getattr(
        _bridge_contracts, "record_pbrf_error_class", None
    )
    if codes is None or lookup is None:
        return None
    try:
        names = {code: lookup(code) for code in codes}
    except (ValueError, TypeError, AttributeError):
        return None
    table = {
        "PacketPartitionError": PacketPartitionError,
        "StaleBeliefError": StaleBeliefError,
        "DigestMismatchError": DigestMismatchError,
        "VisibilityViolationError": VisibilityViolationError,
        "ProposalSupportError": ProposalSupportError,
    }
    try:
        return {code: table[name] for code, name in names.items()}
    except KeyError:
        return None


_bridge_pbrf: dict[str, type[Exception]] | None = _pbrf_from_bridge()
if _bridge_pbrf is not None:
    PBRF_ERROR_CODES: dict[str, type[Exception]] = _bridge_pbrf
else:
    PBRF_ERROR_CODES = {
        "PBRF_PARTITION_EMPTY": PacketPartitionError,
        "PBRF_PARTITION_ALIAS": PacketPartitionError,
        "PBRF_PARTITION_MASS": PacketPartitionError,
        "PBRF_PARTITION_CHILD_NORM": PacketPartitionError,
        "PBRF_STALE_EPOCH": StaleBeliefError,
        "PBRF_STALE_TARGET": StaleBeliefError,
        "PBRF_STALE_PARENT": StaleBeliefError,
        "PBRF_STALE_PROVENANCE": StaleBeliefError,
        "PBRF_STALE_WORLDREF": StaleBeliefError,
        "PBRF_DIGEST_DELTA": DigestMismatchError,
        "PBRF_DIGEST_WORLD_ID": DigestMismatchError,
        "PBRF_VIS_TREE_KEY": VisibilityViolationError,
        "PBRF_VIS_TREE_KEY_NESTED": VisibilityViolationError,
        "PBRF_VIS_POLICY_WORLD": VisibilityViolationError,
        "PBRF_VIS_POLICY_HANDS": VisibilityViolationError,
        "PBRF_SUPPORT_REGION": ProposalSupportError,
        "PBRF_SUPPORT_POINT": ProposalSupportError,
    }


# ---------------------------------------------------------------------------
# SPEC 2.1 - Primitive aliases.
# ---------------------------------------------------------------------------

Seat = NewType("Seat", int)  # integer 0..3
SequenceNo = NewType("SequenceNo", int)  # nonnegative, strictly increasing per game
ActionId = NewType("ActionId", int)  # canonical action vocabulary index
TileId = NewType("TileId", int)  # physical tile 0..135
TileType = NewType("TileType", int)  # logical tile 0..33
BeliefEpochId = NewType("BeliefEpochId", int)
ParentId = NewType("ParentId", str)
PacketId = NewType("PacketId", str)
RunId = NewType("RunId", str)
DigestText = NewType("DigestText", str)  # exactly sha256:<64 lowercase hex>
UtcTimestamp = NewType("UtcTimestamp", str)  # RFC 3339 UTC, second or finer precision
SchemaVersion = NewType("SchemaVersion", str)  # MAJOR.MINOR.PATCH


_UTC_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
_SCHEMA_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _require_int(value: int, *, name: str, minimum: int, maximum: int | None) -> int:
    # bool MUST NOT pass integer validation (bool subclasses int).
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f">={minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ContractError(f"{name}={value} out of range ({bound})")
    return value


def _require_str(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a str, got {type(value).__name__}")
    return value


def make_sequence_no(value: int) -> SequenceNo:
    """Thin bridge translator: ``hydra2._native.contracts.make_sequence_no`` decides."""
    gate = getattr(_bridge_contracts, "make_sequence_no", None)
    if gate is not None:
        try:
            raw: int = gate(value)
            return SequenceNo(raw)
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
    return SequenceNo(_require_int(value, name="sequence_no", minimum=0, maximum=None))


def make_tile_type(value: int) -> TileType:
    """Thin bridge translator: ``hydra2._native.contracts.make_tile_type`` decides."""
    gate = getattr(_bridge_contracts, "make_tile_type", None)
    if gate is not None:
        try:
            raw: int = gate(value)
            return TileType(raw)
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
    return TileType(_require_int(value, name="tile_type", minimum=0, maximum=33))


def make_belief_epoch_id(value: int) -> BeliefEpochId:
    """Thin bridge translator: ``hydra2._native.contracts.make_belief_epoch_id`` decides."""
    gate = getattr(_bridge_contracts, "make_belief_epoch_id", None)
    if gate is not None:
        try:
            raw: int = gate(value)
            return BeliefEpochId(raw)
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
    return BeliefEpochId(_require_int(value, name="belief_epoch_id", minimum=0, maximum=None))


def make_parent_id(value: str) -> ParentId:
    """Thin bridge translator: ``hydra2._native.contracts.make_parent_id`` decides."""
    gate = getattr(_bridge_contracts, "make_parent_id", None)
    if gate is not None:
        try:
            raw: str = gate(value)
            return ParentId(raw)
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
    text = _require_str(value, name="parent_id")
    if text == "":
        raise ContractError("parent_id must be non-empty")
    return ParentId(text)


def make_packet_id(value: str) -> PacketId:
    """Thin bridge translator: ``hydra2._native.contracts.make_packet_id`` decides."""
    gate = getattr(_bridge_contracts, "make_packet_id", None)
    if gate is not None:
        try:
            raw: str = gate(value)
            return PacketId(raw)
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
    text = _require_str(value, name="packet_id")
    if text == "":
        raise ContractError("packet_id must be non-empty")
    return PacketId(text)


def make_run_id(value: str) -> RunId:
    """Thin bridge translator: ``hydra2._native.contracts.make_run_id`` decides."""
    gate = getattr(_bridge_contracts, "make_run_id", None)
    if gate is not None:
        try:
            raw: str = gate(value)
            return RunId(raw)
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
    text = _require_str(value, name="run_id")
    if text == "":
        raise ContractError("run_id must be non-empty")
    return RunId(text)


def make_utc_timestamp(value: str) -> UtcTimestamp:
    """Thin bridge translator: ``hydra2._native.contracts.make_utc_timestamp`` decides."""
    gate = getattr(_bridge_contracts, "make_utc_timestamp", None)
    if gate is not None:
        try:
            raw: str = gate(value)
            return UtcTimestamp(raw)
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
    text = _require_str(value, name="utc_timestamp")
    if _UTC_TS_RE.fullmatch(text) is None:
        raise ContractError(
            f"utc_timestamp {text!r} must be RFC 3339 UTC (YYYY-MM-DDTHH:MM:SS[.ffffff]Z)"
        )
    return UtcTimestamp(text)


def make_schema_version(value: str) -> SchemaVersion:
    """Thin bridge translator: ``hydra2._native.contracts.make_schema_version`` decides."""
    gate = getattr(_bridge_contracts, "make_schema_version", None)
    if gate is not None:
        try:
            raw: str = gate(value)
            return SchemaVersion(raw)
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
    text = _require_str(value, name="schema_version")
    if _SCHEMA_VERSION_RE.fullmatch(text) is None:
        raise ContractError(f"schema_version {text!r} must be MAJOR.MINOR.PATCH")
    return SchemaVersion(text)
