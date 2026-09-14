"""SPEC 8 actor observation, serialization, and visibility — WP-02D contract module.

Re-export facade over the split modules: :mod:`hydra2.contracts.observation_types`
(dora shape, phase/meld vocabulary, field validators, exposed-meld record),
:mod:`hydra2.contracts.observation_actor` (observation dataclass, identity
hash, closed factory), :mod:`hydra2.contracts.observation_schema` (closed
schema tables and versioned artifact), and
:mod:`hydra2.contracts.observation_assembly` (validator and per-seat
builder). Import from this path; it preserves every public name and
``__all__``.
"""

from hydra2.contracts.observation_actor import (
    _OBSERVATION_FIELDS as _OBSERVATION_FIELDS,
)
from hydra2.contracts.observation_actor import (
    ActorObservation as ActorObservation,
)
from hydra2.contracts.observation_actor import (
    compute_observation_hash as compute_observation_hash,
)
from hydra2.contracts.observation_actor import (
    make_actor_observation as make_actor_observation,
)
from hydra2.contracts.observation_actor import (
    observation_identity_document as observation_identity_document,
)
from hydra2.contracts.observation_assembly import (
    HISTORY_EVENT_CAP as HISTORY_EVENT_CAP,
)
from hydra2.contracts.observation_assembly import (
    VISIBILITY_VALIDATOR as VISIBILITY_VALIDATOR,
)
from hydra2.contracts.observation_assembly import (
    ObservationBuilder as ObservationBuilder,
)
from hydra2.contracts.observation_assembly import (
    VisibilityValidator as VisibilityValidator,
)
from hydra2.contracts.observation_schema import (
    OBSERVATION_SCHEMA_ARTIFACT_TYPE as OBSERVATION_SCHEMA_ARTIFACT_TYPE,
)
from hydra2.contracts.observation_schema import (
    OBSERVATION_SCHEMA_RELPATH as OBSERVATION_SCHEMA_RELPATH,
)
from hydra2.contracts.observation_schema import (
    OBSERVATION_SCHEMA_SCHEMA_VERSION as OBSERVATION_SCHEMA_SCHEMA_VERSION,
)
from hydra2.contracts.observation_schema import (
    build_observation_schema_envelope as build_observation_schema_envelope,
)
from hydra2.contracts.observation_schema import (
    build_observation_schema_payload as build_observation_schema_payload,
)
from hydra2.contracts.observation_schema import (
    compute_observation_schema_digest as compute_observation_schema_digest,
)
from hydra2.contracts.observation_schema import (
    load_observation_schema as load_observation_schema,
)
from hydra2.contracts.observation_schema import (
    observation_schema_digest as observation_schema_digest,
)
from hydra2.contracts.observation_schema import (
    parse_observation_schema as parse_observation_schema,
)
from hydra2.contracts.observation_types import (
    DORA_SENTINEL as DORA_SENTINEL,
)
from hydra2.contracts.observation_types import (
    DORA_SHAPE as DORA_SHAPE,
)
from hydra2.contracts.observation_types import (
    MELD_KINDS as MELD_KINDS,
)
from hydra2.contracts.observation_types import (
    PHASES as PHASES,
)
from hydra2.contracts.observation_types import (
    MeldKind as MeldKind,
)
from hydra2.contracts.observation_types import (
    Phase as Phase,
)
from hydra2.contracts.observation_types import (
    VisibleMeld as VisibleMeld,
)
from hydra2.contracts.observation_types import (
    visible_meld_id as visible_meld_id,
)

__all__ = [
    "DORA_SENTINEL",
    "DORA_SHAPE",
    "HISTORY_EVENT_CAP",
    "OBSERVATION_SCHEMA_ARTIFACT_TYPE",
    "OBSERVATION_SCHEMA_RELPATH",
    "OBSERVATION_SCHEMA_SCHEMA_VERSION",
    "PHASES",
    "VISIBILITY_VALIDATOR",
    "ActorObservation",
    "MeldKind",
    "ObservationBuilder",
    "Phase",
    "VisibilityValidator",
    "build_observation_schema_envelope",
    "build_observation_schema_payload",
    "compute_observation_hash",
    "compute_observation_schema_digest",
    "load_observation_schema",
    "make_actor_observation",
    "observation_identity_document",
    "observation_schema_digest",
    "parse_observation_schema",
    "visible_meld_id",
]

# Names importable from this path before the split that live in the
# submodules now (kept so engine helpers and type-checking imports resolve).
from hydra2.contracts.observation_assembly import (
    _CALL_MELD_KINDS as _CALL_MELD_KINDS,
)
from hydra2.contracts.observation_assembly import (
    _PUBLIC_SNAPSHOT_FIELDS as _PUBLIC_SNAPSHOT_FIELDS,
)
from hydra2.contracts.observation_schema import (
    _FIELD_CONSTRAINTS as _FIELD_CONSTRAINTS,
)
from hydra2.contracts.observation_schema import (
    _OBSERVATION_SCHEMA_DIGEST_CACHE as _OBSERVATION_SCHEMA_DIGEST_CACHE,
)
from hydra2.contracts.observation_types import _FURIETEN_STATES as _FURIETEN_STATES
from hydra2.contracts.observation_types import _RIICHI_STATES as _RIICHI_STATES
from hydra2.contracts.observation_types import _WIND_TILE_TYPES as _WIND_TILE_TYPES
