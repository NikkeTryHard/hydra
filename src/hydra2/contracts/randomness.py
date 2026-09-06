"""SPEC 13 semantic randomness — purpose-discriminated stream keys.

Every formal stochastic draw in Hydra2 derives its seed from a fully named
:class:`RandomStreamKey`, never from call order, global counters, or implicit
state. The seed derivation is the canonical payload

    {"protocol": "hydra2_rng_v1", "master_seed": <hex>, "key": <key json>}

hashed with SHA-256 (SPEC 13); the key JSON is RFC 8785 canonical bytes via
:mod:`hydra2.contracts.canonical` so identical keys hash identically on every
platform and process.

RandomStreamSchema — purpose/field matrix
-----------------------------------------

``RandomStreamSchema.validate_key`` enforces, per purpose:

* SPEC-mandated requirements: ``mlmc_level`` requires ``fidelity_level``;
  ``rqmc_scramble`` requires ``scramble_id``; ``smc_propagation`` and
  ``smc_resampling`` are distinct purposes and each requires
  ``population_id``; ``gumbel_root`` requires ``action_id``; retries are a
  new key with an incremented ``attempt_id`` (see :func:`retry_key`);
  every field unused by a purpose MUST be null.
* Exactly-one-of ``case_id`` / ``wall_id`` for game-scoped purposes (owner
  decision D-WP03B-1: belief sampling, policy sampling, root selection,
  transitions, advantages, confirmation, gumbel roots, and coupling all
  address exactly one game instance; environment-level purposes — wall
  generation, evaluation schedules, statistical-method streams, training
  streams — must leave both null because their identity is experiment-wide).
* Distinctness of natural / proposal / actor-policy / root-selection /
  transition / confirmation streams is structural: the purpose literal is
  part of the hashed payload, so two purposes can never collide even with
  otherwise-identical fields.
* Coupling primitives (D-WP03B-2): ``coupling_primitive`` carries the
  branch-independent primitive identity in ``parent_id`` (a canonical string
  label) and FORBIDS ``candidate_id`` — coupled branches must derive the
  byte-identical primitive stream and map it through their own conditional
  laws downstream.
* MLMC fidelity levels may additionally name their SMC population;
  RQMC scrambles may be candidate-specific; SMC streams carry the epoch they
  propagate to / resample at (owner decisions D-WP03B-3/4).

Final-evaluation isolation (SPEC 13 last bullet): :class:`MasterSeedMaterial`
splits master material into a ``selection_training`` scope and a
``final_evaluation`` scope; :func:`authority_stream` refuses to run final
purposes (``confirmation``, ``evaluation_schedule``) on selection/training
material and refuses every other purpose on final material, so training and
candidate-selection code paths cannot touch evaluation seeds.

Contracts modules import stdlib and sibling contracts only (SPEC 1).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Literal

from hydra2.contracts.canonical import canonical_json_bytes
from hydra2.contracts.common import (
    BeliefEpochId,
    ContractError,
    DeterminismError,
    PacketId,
    ParentId,
    Seat,
    make_action_id,
    make_belief_epoch_id,
    make_packet_id,
    make_parent_id,
    make_seat,
)

__all__ = [
    "FINAL_EVALUATION_PURPOSES",
    "RANDOM_PURPOSES",
    "MasterSeedMaterial",
    "RandomPurpose",
    "RandomStream",
    "RandomStreamCheckpoint",
    "RandomStreamKey",
    "RandomStreamSchema",
    "StreamLedger",
    "authority_stream",
    "derive_scope_material",
    "key_to_json",
    "make_random_stream_key",
    "retry_key",
    "semantic_seed",
]

RandomPurpose = Literal[
    "wall",
    "belief_natural_sample",
    "belief_proposal_sample",
    "actor_policy_sample",
    "root_tree_selection",
    "rollout_transition",
    "rollout_advantage",
    "confirmation",
    "coupling_primitive",
    "mlmc_level",
    "rqmc_scramble",
    "smc_propagation",
    "smc_resampling",
    "training_shuffle",
    "training_dropout",
    "evaluation_schedule",
    "gumbel_root",
    "kernel_sample",
]

RANDOM_PURPOSES: tuple[str, ...] = (
    "wall",
    "belief_natural_sample",
    "belief_proposal_sample",
    "actor_policy_sample",
    "root_tree_selection",
    "rollout_transition",
    "rollout_advantage",
    "confirmation",
    "coupling_primitive",
    "mlmc_level",
    "rqmc_scramble",
    "smc_propagation",
    "smc_resampling",
    "training_shuffle",
    "training_dropout",
    "evaluation_schedule",
    "gumbel_root",
    "kernel_sample",
)

#: Purposes whose seeds are final-evaluation material only (SPEC 13: final
#: evaluation seeds remain inaccessible to training and selection paths).
FINAL_EVALUATION_PURPOSES: frozenset[str] = frozenset({"confirmation", "evaluation_schedule"})

_GAME_SCOPED: frozenset[str] = frozenset(
    {
        "belief_natural_sample",
        "belief_proposal_sample",
        "actor_policy_sample",
        "root_tree_selection",
        "rollout_transition",
        "rollout_advantage",
        "confirmation",
        "coupling_primitive",
        "gumbel_root",
        "kernel_sample",
    }
)

# Fields beyond the always-required core (experiment_id, split_id,
# replicate_id, attempt_id). Required entries MUST be non-null; every field
# not listed as required OR optional for a purpose is forbidden (null).
_REQUIRED_BY_PURPOSE: dict[str, tuple[str, ...]] = {
    "wall": ("wall_id",),
    "evaluation_schedule": (),
    "belief_natural_sample": ("belief_epoch", "population_id", "candidate_id"),
    "belief_proposal_sample": ("belief_epoch", "population_id", "candidate_id"),
    "actor_policy_sample": ("candidate_id",),
    "root_tree_selection": ("candidate_id", "parent_id"),
    "rollout_transition": ("candidate_id", "parent_id", "action_id"),
    "rollout_advantage": ("candidate_id", "packet_id"),
    "confirmation": ("candidate_id",),
    # Branch-independent primitive identity label rides in parent_id.
    "coupling_primitive": ("parent_id",),
    "mlmc_level": ("fidelity_level",),
    "rqmc_scramble": ("scramble_id",),
    "smc_propagation": ("population_id", "belief_epoch"),
    "smc_resampling": ("population_id", "belief_epoch"),
    "training_shuffle": (),
    "training_dropout": (),
    "gumbel_root": ("candidate_id", "action_id"),
    "kernel_sample": ("candidate_id", "parent_id", "action_id", "belief_epoch"),
}


_OPTIONAL_BY_PURPOSE: dict[str, frozenset[str]] = {
    "wall": frozenset(),
    "evaluation_schedule": frozenset(),
    "belief_natural_sample": frozenset({"root_seat"}),
    "belief_proposal_sample": frozenset({"root_seat", "parent_id"}),
    "actor_policy_sample": frozenset({"root_seat"}),
    "root_tree_selection": frozenset({"visit_index"}),
    "rollout_transition": frozenset({"packet_id"}),
    "rollout_advantage": frozenset({"action_id"}),
    "confirmation": frozenset({"root_seat"}),
    "coupling_primitive": frozenset(),
    "mlmc_level": frozenset({"population_id", "candidate_id"}),
    "rqmc_scramble": frozenset({"candidate_id"}),
    "smc_propagation": frozenset({"candidate_id"}),
    "smc_resampling": frozenset({"candidate_id"}),
    "training_shuffle": frozenset({"candidate_id"}),
    "training_dropout": frozenset({"candidate_id"}),
    "gumbel_root": frozenset({"visit_index"}),
    "kernel_sample": frozenset(),
}


@dataclass(frozen=True, slots=True)
class RandomStreamKey:
    """SPEC 13 stream key; field order matches the specification verbatim."""

    purpose: RandomPurpose
    experiment_id: str
    split_id: str
    candidate_id: str | None
    case_id: str | None
    wall_id: str | None
    root_seat: Seat | None
    belief_epoch: BeliefEpochId | None
    parent_id: ParentId | None
    action_id: int | None
    packet_id: PacketId | None
    fidelity_level: int | None
    population_id: int | None
    replicate_id: int
    scramble_id: int | None
    visit_index: int | None
    attempt_id: int


_KEY_FIELDS: tuple[str, ...] = tuple(item.name for item in fields(RandomStreamKey))


def key_to_json(key: RandomStreamKey) -> dict[str, object]:
    """Canonical JSON projection of a key (nulls preserved, sorted by JCS)."""
    return {name: getattr(key, name) for name in _KEY_FIELDS}


class RandomStreamSchema:
    """Validation matrix over :class:`RandomStreamKey` purposes."""

    required_by_purpose = _REQUIRED_BY_PURPOSE
    optional_by_purpose = _OPTIONAL_BY_PURPOSE

    @staticmethod
    def validate_key(key: RandomStreamKey) -> RandomStreamKey:
        if key.purpose not in _REQUIRED_BY_PURPOSE:
            raise ContractError(f"purpose {key.purpose!r} is not one of {RANDOM_PURPOSES}")
        if not isinstance(key.experiment_id, str) or key.experiment_id == "":
            raise ContractError("experiment_id must be a nonempty str")
        if not isinstance(key.split_id, str) or key.split_id == "":
            raise ContractError("split_id must be a nonempty str")
        for name in ("replicate_id", "attempt_id"):
            value = getattr(key, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"{name} must be a nonnegative int")

        required = _REQUIRED_BY_PURPOSE[key.purpose]
        optional = _OPTIONAL_BY_PURPOSE[key.purpose]
        allowed = frozenset(required) | optional
        if key.purpose in _GAME_SCOPED:
            # Exactly one of case_id / wall_id governs game scoping; both are
            # admissible here so the dedicated rule below can discriminate.
            allowed = allowed | {"case_id", "wall_id"}

        for name in required:
            if getattr(key, name) is None:
                raise ContractError(f"purpose {key.purpose!r} requires non-null {name}")
        for name in _KEY_FIELDS:
            if name in ("purpose", "experiment_id", "split_id", "replicate_id", "attempt_id"):
                continue
            value = getattr(key, name)
            if name in allowed:
                _validate_field_value(name, value)
                continue
            if value is not None:
                raise ContractError(
                    f"purpose {key.purpose!r} forbids non-null {name} (unused fields must be null)"
                )

        if key.purpose in _GAME_SCOPED:
            if (key.case_id is None) == (key.wall_id is None):
                raise ContractError(
                    f"purpose {key.purpose!r} requires exactly one of case_id / wall_id"
                )
            if key.case_id is not None and (not isinstance(key.case_id, str) or key.case_id == ""):
                raise ContractError("case_id must be a nonempty str")
        return key


def _validate_field_value(name: str, value: object) -> None:
    if value is None:
        return
    if name == "root_seat":
        make_seat(value)  # type: ignore[arg-type]
    elif name == "belief_epoch":
        make_belief_epoch_id(value)  # type: ignore[arg-type]
    elif name == "action_id":
        make_action_id(value)  # type: ignore[arg-type]
    elif name in ("parent_id", "packet_id", "case_id", "candidate_id", "wall_id"):
        if not isinstance(value, str) or value == "":
            raise ContractError(f"{name} must be a nonempty str")
        if name == "parent_id":
            _ = make_parent_id(value)
        elif name == "packet_id":
            _ = make_packet_id(value)
    else:  # fidelity_level, population_id, scramble_id, visit_index
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"{name} must be a nonnegative int")


def make_random_stream_key(**kwargs: object) -> RandomStreamKey:
    """Build and validate a :class:`RandomStreamKey`; missing optionals null."""
    known = {"purpose", "experiment_id", "split_id", "replicate_id", "attempt_id"}
    full: dict[str, object] = {name: None for name in _KEY_FIELDS if name not in known}
    full.update(kwargs)
    unknown = set(kwargs) - set(_KEY_FIELDS)
    if len(unknown) > 0:
        raise ContractError(f"unknown RandomStreamKey fields: {sorted(unknown)}")
    for name in ("experiment_id", "split_id", "replicate_id", "attempt_id"):
        if name not in kwargs:
            raise ContractError(f"missing required field {name}")
    if "purpose" not in kwargs:
        raise ContractError("missing required field purpose")
    key = RandomStreamKey(**full)  # type: ignore[arg-type]
    return RandomStreamSchema.validate_key(key)


def retry_key(key: RandomStreamKey) -> RandomStreamKey:
    """Retries derive a fresh stream by incrementing ``attempt_id`` (SPEC 13)."""
    return make_random_stream_key(**{**key_to_json(key), "attempt_id": key.attempt_id + 1})


def semantic_seed(master_seed: bytes, *, key: RandomStreamKey) -> bytes:
    _ = RandomStreamSchema.validate_key(key)
    if not isinstance(master_seed, (bytes, bytearray)) or len(master_seed) == 0:
        raise ContractError("master_seed must be nonempty bytes")
    payload = canonical_json_bytes(
        {
            "protocol": "hydra2_rng_v1",
            "master_seed": master_seed.hex(),
            "key": key_to_json(key),
        }
    )
    return hashlib.sha256(payload).digest()


@dataclass(frozen=True, slots=True)
class RandomStreamCheckpoint:
    """Opaque checkpoint storing everything needed for exact continuation."""

    seed_hex: str
    cursor: int

    def __post_init__(self) -> None:
        try:
            _ = bytes.fromhex(self.seed_hex)
        except ValueError as exc:
            raise ContractError(f"checkpoint seed_hex invalid: {exc}") from exc
        if isinstance(self.cursor, bool) or not isinstance(self.cursor, int) or self.cursor < 0:
            raise ContractError("checkpoint cursor must be a nonnegative int")


class RandomStream:
    """Counter-based (seekable, replayable) stream over a semantic seed.

    Block ``i`` of output is ``sha256(domain || length(seed) || seed || i)``
    with a fixed domain tag; byte position ``p`` therefore depends only on
    ``(seed, p // 32)`` — no sequential state, no call-order sensitivity.
    """

    _DOMAIN = b"hydra2_ctr_v1\x00"
    _BLOCK = 32

    def __init__(self, seed: bytes, *, cursor: int = 0) -> None:
        if not isinstance(seed, (bytes, bytearray)) or len(seed) == 0:
            raise ContractError("seed must be nonempty bytes")
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ContractError("cursor must be a nonnegative int")
        self._seed = seed
        self._cursor = cursor

    @classmethod
    def from_key(cls, master_seed: bytes, *, key: RandomStreamKey) -> RandomStream:
        return cls(semantic_seed(master_seed, key=key))

    @classmethod
    def restore(cls, checkpoint: RandomStreamCheckpoint) -> RandomStream:
        return cls(bytes.fromhex(checkpoint.seed_hex), cursor=checkpoint.cursor)

    @property
    def cursor(self) -> int:
        return self._cursor

    def jump_to(self, cursor: int) -> None:
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ContractError("cursor must be a nonnegative int")
        self._cursor = cursor

    def checkpoint(self) -> RandomStreamCheckpoint:
        return RandomStreamCheckpoint(seed_hex=self._seed.hex(), cursor=self._cursor)

    def get_bytes(self, count: int) -> bytes:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ContractError("count must be a nonnegative int")
        first_block, offset = divmod(self._cursor, self._BLOCK)
        needed = (offset + count + self._BLOCK - 1) // self._BLOCK
        out = bytearray()
        for index in range(first_block, first_block + needed):
            out += self._block_at(index)
        self._cursor += count
        return bytes(out[offset : offset + count])

    def random_float(self) -> float:
        """Uniform in [0, 1) from the next 64 stream bits."""
        return int.from_bytes(self.get_bytes(8), "big") / float(2**64)

    def random_below(self, bound: int) -> int:
        """Uniform integer in [0, bound) via rejection sampling."""
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 2:
            raise ContractError("bound must be an int >= 2")
        nbytes: int = (bound - 1).bit_length() // 8 + 1
        limit: int = (256**nbytes) // bound * bound
        while True:
            value = int.from_bytes(self.get_bytes(nbytes), "big")
            if value < limit:
                return value % bound

    def _block_at(self, index: int) -> bytes:
        return hashlib.sha256(
            self._DOMAIN
            + len(self._seed).to_bytes(4, "big")
            + self._seed
            + index.to_bytes(8, "big")
        ).digest()


class StreamLedger:
    """Issues streams once per exact key; reissue raises DeterminismError.

    Reusing one stream for distinct purposes is prevented structurally by the
    purpose-discriminated seed, but issuing the SAME key twice (e.g. replaying
    a selection stream inside a confirmation path) would silently correlate
    draws; the ledger makes that reuse loud instead.
    """

    def __init__(self) -> None:
        self._issued: dict[bytes, None] = {}

    def issue(self, master_seed: bytes, key: RandomStreamKey) -> RandomStream:
        identity = canonical_json_bytes(key_to_json(RandomStreamSchema.validate_key(key)))
        if identity in self._issued:
            raise DeterminismError(
                f"stream key already issued (reuse across draws is forbidden): "
                f"{identity.decode('utf-8', 'replace')}"
            )
        self._issued[identity] = None
        return RandomStream.from_key(master_seed, key=key)


@dataclass(frozen=True, slots=True)
class MasterSeedMaterial:
    """Root randomness split into isolated scopes (SPEC 13 final bullet)."""

    material: bytes
    scope: Literal["selection_training", "final_evaluation"]

    def __post_init__(self) -> None:
        if not isinstance(self.material, (bytes, bytearray)) or len(self.material) == 0:
            raise ContractError("material must be nonempty bytes")


def derive_scope_material(root_material: bytes, scope: str) -> MasterSeedMaterial:
    """Derive independent per-scope master material from one root secret."""
    if scope not in ("selection_training", "final_evaluation"):
        raise ContractError(f"unknown scope {scope!r}")
    if not isinstance(root_material, (bytes, bytearray)) or len(root_material) == 0:
        raise ContractError("root_material must be nonempty bytes")
    derived = hashlib.sha256(
        b"hydra2_master_scope_v1\x00" + scope.encode("ascii") + b"\x00" + root_material
    ).digest()
    return MasterSeedMaterial(material=derived, scope=scope)


def authority_stream(domain: MasterSeedMaterial, key: RandomStreamKey) -> RandomStream:
    """Stream from scoped material; cross-scope use fails closed."""
    _ = RandomStreamSchema.validate_key(key)
    key_is_final = key.purpose in FINAL_EVALUATION_PURPOSES
    if domain.scope == "final_evaluation" and not key_is_final:
        raise ContractError(
            f"final-evaluation material cannot feed selection/training purpose {key.purpose!r}"
        )
    if domain.scope == "selection_training" and key_is_final:
        raise ContractError(
            f"selection/training material cannot feed final-evaluation purpose {key.purpose!r}"
        )
    return RandomStream(semantic_seed(domain.material, key=key))
