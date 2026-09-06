"""SPEC 2.1 primitive aliases and SPEC 3 typed failure hierarchy.

This module is the bootstrap subset of the contract layer owned by WP-01.
Contracts import only the Python standard library and other contract modules.
Full contract modules (rules, utility, tile, action, event, observation)
arrive in WP-02; nothing here anticipates them.
"""

from __future__ import annotations

import re
from typing import NewType, TypeGuard

# ---------------------------------------------------------------------------
# SPEC 3 - Failure model (typed errors). All expected Hydra2 failures derive
# from this exact hierarchy; library code raises these and never bare
# RuntimeError for expected conditions.
# ---------------------------------------------------------------------------


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
PBRF_ERROR_CODES: dict[str, type[Hydra2Error]] = {
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


_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
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


def is_seat(value: object) -> TypeGuard[Seat]:
    """Narrowing predicate: True iff value is a valid Seat (0..3, bool excluded)."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 3


def is_digest(value: object) -> TypeGuard[DigestText]:
    """Narrowing predicate: True iff value matches 'sha256:<64 lowercase hex>'."""
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def is_tile(value: object) -> TypeGuard[TileId]:
    """Narrowing predicate: True iff value is a valid TileId (0..135, bool excluded)."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 135


def make_seat(value: int) -> Seat:
    return Seat(_require_int(value, name="seat", minimum=0, maximum=3))


def make_sequence_no(value: int) -> SequenceNo:
    return SequenceNo(_require_int(value, name="sequence_no", minimum=0, maximum=None))


def make_action_id(value: int) -> ActionId:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"action_id={value!r} must be a nonnegative int")
    return ActionId(value)


def make_tile_id(value: int) -> TileId:
    return TileId(_require_int(value, name="tile_id", minimum=0, maximum=135))


def make_tile_type(value: int) -> TileType:
    return TileType(_require_int(value, name="tile_type", minimum=0, maximum=33))


def make_belief_epoch_id(value: int) -> BeliefEpochId:
    return BeliefEpochId(_require_int(value, name="belief_epoch_id", minimum=0, maximum=None))


def make_parent_id(value: str) -> ParentId:
    text = _require_str(value, name="parent_id")
    if text == "":
        raise ContractError("parent_id must be non-empty")
    return ParentId(text)


def make_packet_id(value: str) -> PacketId:
    text = _require_str(value, name="packet_id")
    if text == "":
        raise ContractError("packet_id must be non-empty")
    return PacketId(text)


def make_run_id(value: str) -> RunId:
    text = _require_str(value, name="run_id")
    if text == "":
        raise ContractError("run_id must be non-empty")
    return RunId(text)


def make_digest_text(value: str) -> DigestText:
    text = _require_str(value, name="digest_text")
    if _DIGEST_RE.fullmatch(text) is None:
        raise ContractError(f"digest_text {text!r} must match 'sha256:<64 lowercase hex>'")
    return DigestText(text)


def make_utc_timestamp(value: str) -> UtcTimestamp:
    text = _require_str(value, name="utc_timestamp")
    if _UTC_TS_RE.fullmatch(text) is None:
        raise ContractError(
            f"utc_timestamp {text!r} must be RFC 3339 UTC (YYYY-MM-DDTHH:MM:SS[.ffffff]Z)"
        )
    return UtcTimestamp(text)


def make_schema_version(value: str) -> SchemaVersion:
    text = _require_str(value, name="schema_version")
    if _SCHEMA_VERSION_RE.fullmatch(text) is None:
        raise ContractError(f"schema_version {text!r} must be MAJOR.MINOR.PATCH")
    return SchemaVersion(text)
