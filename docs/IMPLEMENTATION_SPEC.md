# Hydra2 Canonical Implementation Specification

**Status:** Normative pre-implementation specification.  
**Execution authority:** [BUILD_EXECUTION_PLAN.md](./BUILD_EXECUTION_PLAN.md)  
**Direction:** [PROJECT_PLAN.md](./PROJECT_PLAN.md)  
**Research equations:** [ALGORITHM_EXPERIMENT_BLUEPRINT.md](./ALGORITHM_EXPERIMENT_BLUEPRINT.md)

<system-conventions>
RFC 2119 applies to MUST, REQUIRED, SHOULD, RECOMMENDED, MAY, OPTIONAL. `NEVER` and `AVOID` mean `MUST NOT` and `SHOULD NOT`.
</system-conventions>

<critical>
This file defines implementation shape. Builders MUST implement these names, fields, semantics, validation rules, and error behavior. Builders MUST NOT research or invent alternate APIs. A required value marked `manifest-supplied`, `pilot-frozen`, or `qualification-produced` has no default: absence blocks the consuming path. Pseudocode is executable design, not permission to omit validation, error paths, persistence, tests, or evidence named by the execution plan.
</critical>

## 1. Scope and Source Layout

Initial Python namespace:

```text
src/hydra2/
  artifacts/{canonical.py,digest.py,registry.py,atomic.py}
  contracts/{common.py,rules.py,utility.py,tile.py,action.py,event.py,observation.py}
  runtime/{protocol.py,plain.py,fabric.py,checkpoint.py,environment.py}
  engines/{protocol.py,riichienv/,mahjax/}
  conformance/{cases.py,runner.py,report.py}
  data/{raw_manifest.py,validate.py,split.py,rows.py,parquet.py,cache.py,loader.py}
  models/{protocol.py,encoder.py,network.py,attention.py,outputs.py}
  belief/{protocol.py,world.py,particle.py,packet.py,epoch.py,tiny_oracle.py}
  search/{common.py,candidate0.py,ismcts.py,despot.py,pbrf.py,modules/,resolving.py,gumbel.py}
  train/{state.py,objective.py,supervised/,distill/,rl/}
  eval/{schedule.py,case.py,runner.py,blocks.py,statistics.py,telemetry.py,promotion.py}
  performance/{candidate.py,qualify.py,ledger.py}
  tracking/{protocol.py,wandb_mirror.py}
```

Rules:

- Contracts MUST import only Python standard library and other contract modules.
- `artifacts` MUST NOT import engines, tensors, training, or tracking.
- Engines depend on contracts; contracts NEVER depend on engines.
- Models receive tensors derived from `ActorObservation`; model APIs NEVER receive `FullWorld`.
- Privileged data types live under `data.rows` and MUST NOT share inference batch constructors.
- Exact simulator remains eager Python/native engine code. Compiled regions begin after actor-visible tensor encoding.
- Every public type below is immutable unless explicitly named mutable state.
- Public APIs use keyword-only arguments where two same-typed IDs or arrays could be swapped.

## 2. Versioning, Canonical Bytes, and IDs

### 2.1 Primitive aliases

```python
from typing import NewType, Literal

Seat = NewType("Seat", int)                 # integer 0..3
SequenceNo = NewType("SequenceNo", int)    # nonnegative, strictly increasing per game
ActionId = NewType("ActionId", int)        # canonical action vocabulary index
TileId = NewType("TileId", int)            # physical tile 0..135
TileType = NewType("TileType", int)        # logical tile 0..33
BeliefEpochId = NewType("BeliefEpochId", int)
ParentId = NewType("ParentId", str)
PacketId = NewType("PacketId", str)
RunId = NewType("RunId", str)
DigestText = NewType("DigestText", str)    # exactly sha256:<64 lowercase hex>
UtcTimestamp = NewType("UtcTimestamp", str)# RFC 3339 UTC, second or finer precision
SchemaVersion = NewType("SchemaVersion", str) # MAJOR.MINOR.PATCH
```

Constructors MUST validate ranges. `bool` MUST NOT pass integer validation.

### 2.2 Canonical artifact envelope

Every identity artifact MUST serialize this envelope:

```python
@dataclass(frozen=True, slots=True)
class ArtifactEnvelope:
    artifact_type: str
    schema_version: SchemaVersion
    compatibility: Literal["exact", "backward_read"]
    payload: Mapping[str, JsonValue]
```

Identity bytes:

```python
def canonical_bytes(value: JsonValue) -> bytes:
    # RFC 8785 JSON Canonicalization Scheme, UTF-8, no BOM/newline.
    ...

def sha256_digest(data: bytes) -> DigestText:
    return DigestText("sha256:" + hashlib.sha256(data).hexdigest())

def artifact_id(envelope: ArtifactEnvelope) -> DigestText:
    return sha256_digest(canonical_bytes(to_json(envelope)))
```

Requirements:

- JSON domain: null, bool, string, finite number, array, string-keyed object only.
- Identity timestamps MUST be payload fields only when semantics require time. Registry insertion time is metadata and MUST NOT change identity.
- Sets MUST serialize as sorted arrays under a field-specific declared order.
- Byte strings MUST serialize as lowercase hex with named byte domain; implicit Base64 is prohibited.
- Floats in identity artifacts MUST be finite. Probability artifacts SHOULD use decimal strings when exact reconstruction matters.
- Reader MUST recompute digest before decoding semantic payload.
- Unknown major version: `IncompatibleSchemaError`.
- Unknown required field, invalid enum, duplicate key, noncanonical digest: hard error.
- Unknown optional field is accepted only when compatibility declaration explicitly allows it.

### 2.3 Atomic publication

```python
def publish_atomic(*, destination: Path, data: bytes, expected: DigestText) -> None:
    require sha256_digest(data) == expected
    require destination.parent exists and is not symlink
    create unique same-directory temp with O_CREAT|O_EXCL
    write all; flush; fsync(file)
    if destination exists:
        require full bytes digest == expected
        delete temp; return
    rename temp -> destination without overwrite race
    fsync(destination.parent)
```

On any error: close and remove owned temporary path where possible; NEVER remove existing destination. Registry row publication happens only after destination verification.

## 3. Failure Model

All expected Hydra2 failures derive from:

```python
class Hydra2Error(Exception): ...
class ContractError(Hydra2Error): ...
class IncompatibleSchemaError(ContractError): ...
class CanonicalizationError(ContractError): ...
class DigestMismatchError(ContractError): ...
class RulesMismatchError(ContractError): ...
class InvalidTileError(ContractError): ...
class InvalidActionError(ContractError): ...
class VisibilityViolationError(ContractError): ...
class IllegalActionError(ContractError): ...
class CorruptArtifactError(Hydra2Error): ...
class LineageError(Hydra2Error): ...
class QuarantinedError(Hydra2Error): ...
class UnsupportedRuleError(Hydra2Error): ...
class DeterminismError(Hydra2Error): ...
class StaleBeliefError(Hydra2Error): ...
class PacketPartitionError(Hydra2Error): ...
class ProposalSupportError(Hydra2Error): ...
class DeadlineExceededError(Hydra2Error): ...
class QualificationRequiredError(Hydra2Error): ...
```

Rules:

- Library code raises typed errors; CLI maps them to nonzero exit and stable error class.
- Errors contain safe identifiers/hashes/event indices, NEVER concealed tiles from another seat or confidential source identity.
- Search deadline expiration is expected control flow only at the runner boundary. Runner invokes frozen fallback and records timeout; planner MUST NOT silently return partial output.
- Corruption/visibility/rules/legality/stale-state errors NEVER fall back silently.
> Caveat 2026-09-04 (non-normative): synthetic-path hardening verified — teacher/loop/replay/completion WP-10..WP-12 paths now raise hard errors, never synthetic fallback. See `## Status 2026-09-04` §A. Normative §§3:156-159 unchanged.

Diagnostic codes (normative, PR4): semantic search/belief raise sites append
a ` [{CODE}]` suffix to their messages from the list below; error CLASSES are unchanged.
Codes exist so routers (ponder, halving/VOC, exclusion-report) branch on failure kind without fragile
substring matching owned anywhere else; the single bijective table `PBRF_ERROR_CODES`
lives beside the search package. Pure caller-input/config validation (the ContractError
long tail) keeps byte-identical messages. Codes: `PBRF_PARTITION_EMPTY`,
`PBRF_PARTITION_ALIAS`, `PBRF_PARTITION_MASS`, `PBRF_PARTITION_CHILD_NORM`,
`PBRF_STALE_EPOCH`, `PBRF_STALE_TARGET`, `PBRF_STALE_PARENT`, `PBRF_STALE_PROVENANCE`,
`PBRF_STALE_WORLDREF`, `PBRF_DIGEST_DELTA`, `PBRF_DIGEST_WORLD_ID`,
`PBRF_VIS_TREE_KEY`, `PBRF_VIS_TREE_KEY_NESTED`, `PBRF_VIS_POLICY_WORLD`,
`PBRF_VIS_POLICY_HANDS`, `PBRF_SUPPORT_REGION`, `PBRF_SUPPORT_POINT`.

## 4. Tile Contract

### 4.1 Physical encoding

```text
TileType 0..8   = 1m..9m
TileType 9..17  = 1p..9p
TileType 18..26 = 1s..9s
TileType 27..33 = East, South, West, North, White, Green, Red
TileId = 4 * TileType + copy, copy in 0..3
```

Red-five identity is manifest-dependent but canonical physical IDs are fixed:

```text
red 5m = TileId(4 * 4  + 0) = 16
red 5p = TileId(4 * 13 + 0) = 52
red 5s = TileId(4 * 22 + 0) = 88
```

For `tenhou_4p_hanchan_v1`, those three IDs are red; no honor is red. Engine adapters MUST map engine-native red representation to these IDs. Logical operations use `tile_id // 4`; discard/call serialization preserves physical IDs.

### 4.2 Tile validation

```python
def validate_tile_multiset(tiles: Iterable[TileId], rules: RulesManifest) -> None:
    require every id in 0..135
    require no physical id repeats
    require red flags equal physical IDs declared by rules
```

A full world MUST contain every enabled physical tile exactly once across hands, live wall, dead wall, visible zones, and consumed/meld zones according to phase. The validator reports first duplicate/missing physical ID internally; actor-facing errors omit hidden identities.

## 5. Rules and Utility

### 5.1 Rules manifest

```python
@dataclass(frozen=True, slots=True)
class SourceAuthority:
    url: str
    retrieved_at_utc: UtcTimestamp
    content_sha256: DigestText

@dataclass(frozen=True, slots=True)
class ClockRule:
    base_seconds: int
    increment_seconds: int

@dataclass(frozen=True, slots=True)
class RulesManifest:
    rules_id: str                           # tenhou_4p_hanchan_v1
    source: SourceAuthority
    players: Literal[4]
    match_length: Literal["hanchan"]
    starting_points: Literal[25000]
    return_points: Literal[30000]
    uma_by_rank: tuple[int, int, int, int] # manifest-supplied; Tenhou 10-20
    oka_policy: str                         # manifest-supplied enum
    red_tile_ids: tuple[TileId, ...]        # exactly (16,52,88)
    clocks: tuple[ClockRule, ...]           # standard and fast
    kuitan: bool
    kuikae_policy: str
    furiten_policy: str
    chankan_policy: str
    rinshan_policy: str
    kan_dora_reveal_policy: str
    kan_ura_policy: str
    pao_policy: str
    yakuman_policy: str
    kazoe_policy: str
    multiple_ron_policy: str
    riichi_stick_allocation: str
    abortive_draws: tuple[str, ...]
    nagashi_mangan: bool
    bankruptcy_threshold: int
    all_last_policy: str
    agari_yame_policy: str
    tobi_policy: str
    sudden_death_policy: str
    rank_tie_break: str
    placement_conversion_id: str
    adapter_compatibility: tuple["AdapterCompatibility", ...]
```

No scoring/match field above is optional. String policy values MUST come from schema enums established by WP-02B using Tenhou source evidence. Unknown source behavior blocks manifest publication. Adapters declare `supported`, `unsupported`, or `qualified` per manifest hash.

WP-02B does not ask a builder to rediscover Tenhou policy values. Before implementation, a reviewed source snapshot and complete `configs/rules/tenhou_4p_hanchan_v1.json` payload MUST be committed by the contract owner. WP-02B validates source digest, schema completeness, and adapter behavior, then publishes the immutable artifact. Missing reviewed bytes block WP-02B.

### 5.2 Outcome and utility

```python
@dataclass(frozen=True, slots=True)
class SettlementFact:
    kind: str
    from_seat: Seat | None
    to_seats: tuple[Seat, ...]
    point_deltas: tuple[int, int, int, int]
    detail: Mapping[str, JsonValue]

@dataclass(frozen=True, slots=True)
class RawOutcome:
    final_scores: tuple[int, int, int, int]
    ranks: tuple[int, int, int, int]       # rank per seat, each 1..4 exactly once
    point_deltas: tuple[int, int, int, int]
    settlements: tuple[SettlementFact, ...]
    rules_id: str
    rules_hash: DigestText

@dataclass(frozen=True, slots=True)
class UtilityManifest:
    utility_id: str
    schema_version: SchemaVersion
    rules_id: str
    rules_hash: DigestText
    objective: Literal["expected_final_placement"]
    rank_values: tuple[float, float, float, float] # indexed rank 1..4
    tie_policy: Literal["use_rules_resolved_rank"]
    value_min: float
    value_max: float
    zero_sum: bool
    digest: DigestText

@dataclass(frozen=True, slots=True)
class UtilityVector:
    values: tuple[float, float, float, float]
    utility_id: str
    utility_manifest_hash: DigestText
    rules_hash: DigestText

def utility(outcome: RawOutcome, manifest: UtilityManifest) -> UtilityVector: ... # sets utility_manifest_hash=manifest.digest
def root_scalar(value: UtilityVector, seat: Seat) -> float: return value.values[seat]
```

Primary `utility_id` is expected final placement. The utility manifest MUST specify exact rank-to-value conversion and tie behavior; no code hard-codes an assumed zero-sum vector. Raw scores/ranks/deltas survive training targets, backups, and logs.
`rank_values`, bounds, and all computed values MUST be finite; `value_min <= rank_values[r] <= value_max`. `utility()` rejects utility/rules hash mismatch, unresolved/tied ranks, non-permutation ranks, and values outside bounds, and returns `utility_manifest_hash=manifest.digest`; consumers compare it to `CandidateSpec.utility_manifest_hash`. Failures raise `RulesMismatchError` or `ContractError`. `zero_sum` is descriptive only and true only when the manifest proves it.

## 6. Canonical Action Vocabulary

### 6.1 Action kinds and stable IDs

`ActionId` identifies a fully parameterized canonical action in a versioned `ActionTable`, not only a kind. Kind ordinals are frozen:

```text
0 PASS
1 DISCARD
2 TSUMOGIRI
3 RIICHI_DISCARD
4 CHI
5 PON
6 DAIMINKAN
7 ANKAN
8 KAKAN
9 RON
10 TSUMO
11 ABORT_NINE_TERMINALS
12 ACCEPT_ABORTIVE_DRAW
```

```python
ActionKind = Literal[
    "pass", "discard", "tsumogiri", "riichi_discard", "chi", "pon",
    "daiminkan", "ankan", "kakan", "ron", "tsumo",
    "abort_nine_terminals", "accept_abortive_draw",
]

@dataclass(frozen=True, slots=True)
class CanonicalAction:
    kind: ActionKind
    actor: Seat
    tile: TileId | None
    called_tile: TileId | None
    consumed_tiles: tuple[TileId, ...]
    source_seat: Seat | None
    declares_riichi: bool
    metadata: tuple[tuple[str, JsonValue], ...]
```

### 6.2 Invariants

- `pass`, `ron`: source seat MAY be set only for an offered discard/call window; no consumed tiles.
- `discard`, `tsumogiri`: exactly `tile`; no source/called/consumed; `declares_riichi=False`.
- `riichi_discard`: exactly `tile`; `declares_riichi=True`.
- `chi`: source is previous seat; one called tile + exactly two distinct physical consumed tiles; same suit; three logical values consecutive; honors forbidden.
- `pon`: one called tile + exactly two same-logical-type consumed tiles; source different actor.
- `daiminkan`: one called tile + exactly three same-logical-type consumed tiles.
- `ankan`: no source/called; exactly four same-logical-type physical consumed tiles.
- `kakan`: `tile` is added physical tile; metadata includes canonical prior-pon meld ID; source/called empty.
- `tsumo`: no source/called/consumed; winning physical tile appears in actor state.
- Abort kinds legal only in exact engine phase.
- Every physical tile in one action is unique.
- Metadata keys sorted, schema-declared, and kind-specific. Arbitrary extension metadata is rejected.

### 6.3 Action table

```python
@dataclass(frozen=True, slots=True)
class CanonicalActionTemplate:
    kind: ActionKind
    tile: TileId | None
    called_tile: TileId | None
    consumed_tiles: tuple[TileId, ...]
    source_offset: Literal[-1, 0, 1, 2] | None # relative modulo four; 0 only when kind permits self
    declares_riichi: bool
    meld_ref_required: bool

@dataclass(frozen=True, slots=True)
class ActionContext:
    actor: Seat
    action_table_hash: DigestText
    phase: "Phase"
    offered_tile: TileId | None
    offered_by: Seat | None
    own_concealed_tiles: tuple[TileId, ...]
    visible_melds: tuple["VisibleMeld", ...]

@dataclass(frozen=True, slots=True)
class ActionTable:
    schema_version: SchemaVersion
    actions: tuple[CanonicalActionTemplate, ...]
    digest: DigestText

class ActionCodec(Protocol):
    def encode(self, action: CanonicalAction, *, table: ActionTable, context: ActionContext) -> ActionId: ...
    def decode(self, action_id: ActionId, *, table: ActionTable, context: ActionContext) -> CanonicalAction: ...
```
`decode` verifies `table.digest == context.action_table_hash`, substitutes `context.actor`, resolves `source_offset`, and validates offered tile/source, owned physical tiles, phase, and required prior meld. `ActionContext` therefore includes `action_table_hash: DigestText`. Invalid table/context raises `InvalidActionError`; no partial action. `encode(decode(id,table,context),table,context) == id` and converse hold for every legal context. Artifact path: `configs/contracts/action_table_v1.json`; WP-02C publishes its digest before adapters/models.

Generation order is lexicographic over `(kind_ordinal, tile, called_tile, consumed_tiles, source_offset, declares_riichi, meld_ref_required)` with `None` before integers. `actor` is contextual and therefore excluded from template identity; `declares_riichi` is included. Table generator enumerates every and only structurally valid template once; canonical bytes and digest freeze indices. Legal masks are `bool[num_actions]` aligned with this table. Engine adapters MUST NOT reorder or compress it.

## 7. Events, Packets, and Visibility

### 7.1 Event envelope

```python
Visibility = Literal["public", "actor_private", "server_private"]
EventKind = Literal[
    "game_start", "round_start", "turn_advance", "draw_tile", "discard",
    "riichi_declared", "riichi_accepted", "call_window", "call_resolved",
    "chi", "pon", "daiminkan", "ankan", "kakan", "dora_revealed",
    "ron", "tsumo", "draw_end", "abortive_draw", "round_end", "game_end",
]

@dataclass(frozen=True, slots=True)
class EventPayload:
    kind: EventKind
    actor: Seat | None
    tile: TileId | None
    action_id: ActionId | None
    source_seat: Seat | None
    consumed_tiles: tuple[TileId, ...]
    offered_action_ids: tuple[ActionId, ...]
    accepted_action_ids: tuple[ActionId, ...]
    round_index: int | None
    scores: tuple[int, int, int, int] | None
    reason: str | None

@dataclass(frozen=True, slots=True)
class PublicStateDelta:
    path: tuple[str | int, ...]
    operation: Literal["set", "append", "increment"]
    value: JsonValue

@dataclass(frozen=True, slots=True)
class EventEnvelope:
    game_id: str
    sequence: SequenceNo
    kind: EventKind
    actor: Seat | None
    visibility: Visibility
    visible_to: tuple[Seat, ...]
    payload: EventPayload
    public_delta: tuple[PublicStateDelta, ...]
    rules_hash: DigestText
    schema_hash: DigestText
```

Validation:

- Public: `visible_to == (0,1,2,3)`.
- Actor-private: exactly one seat; for `draw_tile`, it equals actor.
- Server-private: empty `visible_to`; cannot serialize into any actor history.
- Sequence strictly increases; no duplicate sequence.
- `turn_advance(actor)` public and tile-free.
- `draw_tile(actor,tile)` actor-private and contains exactly physical tile ID.
- Public discard/call contains only revealed physical tiles.
- Call resolution packet encodes all offered responses/priority outcome required for one successor, but actor streams contain only legally visible response facts.
`EventSchema` is a versioned closed artifact at `configs/contracts/event_schema_v1.json`. For each `EventKind`, it lists exact required-null/non-null payload fields, visibility, actor constraint, allowed `PublicStateDelta` paths/operations/value types, and ordering predecessor/successor set. Unknown payload data and undeclared delta paths are rejected. At minimum: `turn_advance` requires actor only; `draw_tile` requires actor+tile and actor-private visibility; discard/call/dora require revealed tile/action composition; start/end events require their declared round/scores/reason fields. The reviewed artifact, not adapter code, is exhaustive authority.

### 7.2 Actor-visible packet

```python
@dataclass(frozen=True, slots=True)
class ActorVisiblePacket:
    packet_id: PacketId
    actor_view: Seat
    source_sequence_start: SequenceNo
    source_sequence_end: SequenceNo
    events: tuple[EventEnvelope, ...]
    public_state_hash_before: DigestText
    public_state_hash_after: DigestText
    observation_hash_after: DigestText
```

`packet_id = sha256(canonical bytes excluding packet_id)`. WP-02D publishes `PacketBoundarySpec` at `configs/contracts/packet_boundary_v1.json`: root actor, start boundary, first subsequent actor-visible decision/update boundary, call/pass priority grouping, and terminal boundary. Packet partitions use this one hash across belief/search; CandidateSpec references it but cannot redefine it. Packets MUST be mutually exclusive/exhaustive and nonempty.

## 8. Actor Observation and Serialization

```python
DORA_SENTINEL = -1
DORA_SHAPE = (5,)

Phase = Literal["round_start", "draw_decision", "discard_response", "kan_response", "round_end", "game_end"]

@dataclass(frozen=True, slots=True)
class VisibleMeld:
    meld_id: str
    kind: Literal["chi", "pon", "daiminkan", "ankan", "kakan"]
    owner: Seat
    source_seat: Seat | None
    called_tile: TileId | None
    tiles: tuple[TileId, ...]

@dataclass(frozen=True, slots=True)
class ActorObservation:
    game_id: str
    decision_id: str
    sequence: SequenceNo
    actor: Seat
    rules_id: str
    rules_hash: DigestText
    action_table_hash: DigestText
    event_schema_hash: DigestText
    observation_schema_hash: DigestText
    packet_boundary_hash: DigestText
    round_index: int
    round_wind: TileType
    hand_number: int
    seat_winds: tuple[TileType, TileType, TileType, TileType]
    honba: int
    riichi_sticks: int
    dealer: Seat
    scores: tuple[int, int, int, int]
    turn_actor: Seat
    phase: Phase
    live_wall_tiles_remaining: int
    kan_count: int
    ippatsu_active: tuple[bool, bool, bool, bool]
    actor_furiten: Literal["none", "temporary", "riichi", "discard"]
    actor_can_tsumo: bool
    actor_can_riichi: bool
    pending_declaration_discard: TileId | None
    concealed_hand: tuple[TileId, ...]
    own_drawn_tile: TileId | None
    visible_discards: tuple[tuple[TileId, ...], ...] # four seats
    visible_melds: tuple[tuple["VisibleMeld", ...], ...]
    riichi_states: tuple[str, str, str, str]
    dora_indicators: tuple[int, int, int, int, int]
    visible_history: tuple[EventEnvelope, ...]
    legal_mask: tuple[bool, ...]
    observation_hash: DigestText
```

Rules:

- Concealed hand sorted by physical TileId for serialization; drawn tile remains separate.
- Each dora slot is `DORA_SENTINEL` or TileId 0..135. Revealed values are contiguous from index 0. Shape exactly five.
- Legal mask length equals action table length and has at least one true value at every decision.
- Observation fields MUST NOT contain wall/dead wall, opponent concealed tiles, unrevealed dora/ura, engine RNG, future events, server-private events, opponent legal masks, or privileged labels.
- Observation identity excludes derived tensor padding and model state.
- `observation_hash = sha256(canonical actor-observation bytes without observation_hash)`.
Seat winds are a permutation of East/South/West/North aligned by seat; round wind/hand number, live-wall count, kan count, ippatsu, furiten, riichi eligibility, and pending declaration discard MUST equal exact actor-visible rules state. `VisibleMeld` preserves every revealed physical tile and source. `ObservationSchema` at `configs/contracts/observation_schema_v1.json` fixes field order, enum/range, and serialization.

```python
class ObservationBuilder(Protocol):
    def append_visible(self, event: EventEnvelope) -> None: ...
    def build(self, *, actor: Seat, legal_mask: tuple[bool, ...]) -> ActorObservation: ...

class VisibilityValidator(Protocol):
    def validate_event_for_actor(self, event: EventEnvelope, actor: Seat) -> None: ...
    def validate_observation(self, observation: ActorObservation) -> None: ...
```

Builder MUST maintain one isolated history/cache per seat. It MUST NOT build a full-state object then rely on field deletion. Debug representations and exceptions obey same visibility boundary.

## 9. Engine Protocol

```python
@dataclass(frozen=True, slots=True)
class EngineIdentity:
    name: str
    version: str
    adapter_version: SchemaVersion
    source_revision: str | None
    environment_hash: DigestText

@dataclass(frozen=True, slots=True)
class WallSchedule:
    schedule_id: str
    physical_tiles: tuple[TileId, ...] # exact engine consumption order
    digest: DigestText

@dataclass(frozen=True, slots=True)
class TransitionResult:
    events: tuple[EventEnvelope, ...]
    next_actor: Seat | None
    terminal: bool
    raw_outcome: RawOutcome | None
    state_digest: DigestText

class ExactSimulator(Protocol):
    @property
    def identity(self) -> EngineIdentity: ...
    def reset(self, *, rules: RulesManifest, wall: WallSchedule, seat_permutation: tuple[Seat, ...]) -> None: ...
    def actor_observation(self, actor: Seat) -> ActorObservation: ...
    def legal_actions(self, actor: Seat) -> tuple[CanonicalAction, ...]: ...
    def legal_mask(self, actor: Seat) -> tuple[bool, ...]: ...
    def apply(self, action: CanonicalAction) -> TransitionResult: ...
    def snapshot(self) -> "SimulatorSnapshot": ...
    def restore(self, snapshot: "SimulatorSnapshot") -> None: ...
    def clone(self) -> "ExactSimulator": ...
```

Invariants:

- `legal_actions` sorted by ActionId and exactly equals true mask indices.
- `apply` accepts one legal action only and emits canonical events in strict sequence.
- Snapshot identity includes rules, engine, full state, and semantic RNG state. Snapshot serialization is privileged.
- A complete `WallSchedule` determines every physical stochastic tile outcome; reset has no seed. Any non-tile engine randomness uses named semantic streams passed to the specific operation and included in snapshot/schedule identity; adapters MUST NOT shuffle or ignore the supplied wall.
- Actor observation derives through per-seat visibility boundary.
- RiichiEnv adapter is reference after conformance; unsupported rules raise `UnsupportedRuleError` before game start.
- MahJax implements protocol only behind a qualification token bound to its complete tuple.

## 10. Runtime and Checkpoint Protocol

```python
PrecisionId = Literal["fp32", "fp16_mixed", "bf16_mixed"]
CompileMode = Literal["eager", "default", "max-autotune-no-cudagraphs", "max-autotune"]

@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    adapter_id: Literal["plain_pytorch", "fabric_2.6.5"]
    device: str
    precision: PrecisionId
    compile_mode: CompileMode
    fullgraph: bool
    dynamic: bool | None
    backward_pass_autocast: Literal["off"] | None

@dataclass(frozen=True, slots=True)
class RuntimeHandle:
    model: object
    optimizer: object
    backward: Callable[[object], None]
    device: object
    runtime_identity: DigestText

class RuntimeAdapter(Protocol):
    def setup(self, *, model: object, optimizer: object, spec: RuntimeSpec) -> RuntimeHandle: ...
    def barrier(self) -> None: ...
    def synchronize(self) -> None: ...
```

Build order:

```python
def build_runtime(*, adapter, model, optimizer, spec):
    require exact supported RuntimeSpec
    def compile_once(m):
        if spec.compile_mode == "eager": return m
        return torch.compile(
            m, backend="inductor", mode=spec.compile_mode,
            fullgraph=spec.fullgraph, dynamic=spec.dynamic,
        )
    if spec.precision != "fp32" and spec.compile_mode != "eager":
        require spec.backward_pass_autocast == "off"
        with torch._functorch.config.patch(backward_pass_autocast="off"):
            model = compile_once(model)
            return adapter.setup(model=model, optimizer=optimizer, spec=spec)
    model = compile_once(model)
    return adapter.setup(model=model, optimizer=optimizer, spec=spec)
```

Fabric setup remains inside patch because Fabric may unwrap/reapply compile. Plain adapter MUST follow identical call order.
`setup` MUST return the exact wrapped/rebound model and optimizer used for subsequent forward/update operations; caller discards pre-setup references. Plain setup moves model and optimizer state to target device. Fabric setup calls `Fabric.setup(model, optimizer)` once and returns Fabric's rebound objects/backward/device. Neither adapter owns loop, checkpoint, optimizer policy, or compilation.

Checkpoint:

```python
@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    checkpoint_version: SchemaVersion
    run_spec_hash: DigestText
    model_spec_hash: DigestText
    model_state_hash: DigestText
    optimizer_spec_hash: DigestText
    optimizer_state_hash: DigestText
    scheduler_spec_hash: DigestText
    scheduler_state_hash: DigestText
    training_state_hash: DigestText
    sampler_state_hash: DigestText
    rng_state_hash: DigestText
    environment_hash: DigestText
    rules_hash: DigestText
    utility_manifest_hash: DigestText
    action_schema_hash: DigestText
    observation_schema_hash: DigestText
    dataset_manifest_hash: DigestText | None
    rollout_artifact_hash: DigestText | None
    parent_checkpoint_hash: DigestText | None
```

Project code owns serialization. Required training state: global update, microstep, epoch, examples seen, optimizer/scheduler/scaler state, semantic RNG counters, sampler cursor, split/manifest IDs. Checkpoint source identity mirrors RunSpec: supervised/distill checkpoints require `dataset_manifest_hash` and null rollout; RL checkpoints require `rollout_artifact_hash` and null dataset. `run_spec_hash` and the selected source hash are verified before mutating runtime objects. Ordinary single-device `torch.save` is initial payload format; manifest identity remains independent of container metadata.

## 11. Model Contract

### 11.1 Tensor batch

```python
@dataclass(frozen=True, slots=True)
class TensorFieldSpec:
    name: str
    dtype: Literal["bool", "int32", "int64", "float32"]
    shape: tuple[str | int, ...] # symbols limited to B,T,A,F
    padding_value: int | float | bool | None
    valid_min: float | None
    valid_max: float | None
    mask_field: str | None

@dataclass(frozen=True, slots=True)
class ModelInputSchema:
    schema_version: SchemaVersion
    fields: tuple[TensorFieldSpec, ...]
    history_bucket_lengths: tuple[int, ...]
    action_count: int
    digest: DigestText
```

```python
@dataclass(frozen=True, slots=True)
class ModelHeadSpec:
    head_id: str
    output_key: str
    target_id: str
    loss_id: str
    parameters: Mapping[str, JsonValue]

@dataclass(frozen=True, slots=True)
class ModelSpec:
    schema_version: SchemaVersion
    input_schema_hash: DigestText
    feature_derivation_hash: DigestText
    architecture_id: str
    architecture_parameters: Mapping[str, JsonValue]
    head_specs: tuple[ModelHeadSpec, ...]
    action_table_hash: DigestText
    observation_schema_hash: DigestText
    utility_manifest_hash: DigestText
    digest: DigestText
```

`feature_derivation_hash` identifies exact actor-observation-to-tensor code/config, including optional derived features. `head_specs` are sorted by `head_id`; output key, target, loss, masking, reduction, and every coefficient live in the named head specification. Unknown architecture/head/loss IDs are rejected by the registry. Model state, caches, checkpoints, exports, and results bind `ModelSpec.digest`; changing any field invalidates them.

```python
@dataclass(frozen=True, slots=True)
class ActorTensorBatch:
    features: Mapping[str, "Tensor"]
    history_mask: "BoolTensor"          # [B,T], True = valid/participate
    legal_mask: "BoolTensor"            # [B,A]
    observation_hashes: tuple[DigestText, ...]
    actor_seats: "IntTensor"             # [B]
```

All shapes/dtypes/ranges live in versioned `ModelInputSchema`. Encoder validates `ActorObservation` first. Padding values never carry semantics without mask. Nonterminal all-false legal row is hard error.
Initial schema artifact is `configs/models/model_input_v1.json`. It MUST enumerate, in canonical order, every public scalar/categorical/tile/history feature produced from §8, plus `history_mask`, `legal_mask`, `actor_seats`, and observation hashes; no arbitrary runtime feature key is accepted. `ModelSpec`, cache identity, and checkpoint manifest include this digest. Changing feature name/order/shape/dtype/range/padding/mask is a schema break.

### 11.2 Optional actor-visible shape features

The baseline input schema excludes derived shape features. An optional arm MAY publish a new `ModelInputSchema` containing only these deterministic actor-visible fields:

```text
own_private_ids = set(concealed_hand) union {own_drawn_tile when present}
public_ids      = union of physical IDs in visible_discards, visible_melds,
                  and revealed dora indicators
own_physical_count[t]   = count(id in own_private_ids where id // 4 == t)
public_physical_count[t] = count(id in public_ids where id // 4 == t)
public_unseen[t] = 4 - own_physical_count[t] - public_physical_count[t]
post_discard_shanten[a] = exact shanten after legal discard-like action a
own_wait[a,t] = 1 when adding logical tile t structurally completes that post-discard hand, else 0
ukeire[a,t] = public_unseen[t] when adding t lowers post-discard shanten, else 0
```

`own_private_ids` contains concealed tiles plus the separate drawn tile only; it excludes own melds. `public_ids` is a set before logical aggregation, so one called physical tile represented in both discard history and a meld counts once. Dora sentinels are excluded. The two sets MUST be disjoint; every count and `public_unseen` MUST be an integer in `0..4`. Counts remain red-aware until physical-ID deduplication and logical aggregation. Non-discard actions use an explicit validity mask and zero padding. Shanten/completion includes the actor's visible open melds and ordinary, chiitoitsu, and kokushi structure; `own_wait` is not a yaku, furiten, or legal-win claim. These fields MUST be pure functions of `ActorObservation`, canonical action, and qualified exact shanten/completion code. They are availability/shape features, not wall probabilities, opponent-hand estimates, yaku-aware win probabilities, expected scores, or values.

Each arm publishes distinct input-schema, model-spec, derivation, checkpoint, and result hashes. Required tests: reference parity on ordinary/chiitoitsu/kokushi/open-hand/red cases; hidden-world permutation invariance; public-count conservation/range; action alignment; cache/full equality. Hydra1-style continuation multipliers, shape bonuses, heuristic multi-draw `tenpai_prob`/`win_prob`, and hand-built `expected_score` are prohibited.

### 11.3 Output

```python
@dataclass(frozen=True, slots=True)
class ModelOutput:
    policy_logits: "FloatTensor"        # [B,A], unnormalized; illegal entries masked by consumer
    placement_logits: "FloatTensor"     # [B,4,4] seat x final-rank distribution
    value_vector: "FloatTensor"         # [B,4], derived under named utility
    event_logits: Mapping[str, "FloatTensor"]
    belief_logits: Mapping[str, "FloatTensor"]
    diagnostics: Mapping[str, "Tensor"]
    utility_id: str
    utility_manifest_hash: DigestText
    model_identity: DigestText

class ActorModel(Protocol):
    def evaluate(self, batch: ActorTensorBatch) -> ModelOutput: ...
```

Requirements:

- `policy_logits.shape[-1] == legal_mask.shape[-1]`.
- Mask before softmax/loss/argmax; illegal probability exactly zero after normalization.
- Value retains four seats. Search selects root seat only at root decision.
- Event/belief head definitions and target support frozen in model spec.
- Diagnostics contain tensor-derived actor-visible values only.
- SDPA boolean mask `True` means participate; evaluation dropout exactly `0.0`.
- Cache/full encoding outputs satisfy frozen numerical tolerance.

Selection:

```python
def masked_policy(logits, legal_mask):
    require any legal per row
    return softmax(logits.masked_fill(~legal_mask, -inf), dim=-1)
```

Tie behavior is CandidateSpec-supplied and deterministic.

## 12. Data and Lineage Schemas

### 12.1 Raw object manifest

```python
@dataclass(frozen=True, slots=True)
class PackagedObjectRow:
    packaged_object_id: DigestText
    source_kind: Literal["raw", "archive_member", "precompressed"]
    source_container_sha256: DigestText | None
    source_member_path: str | None
    source_bytes_sha256: DigestText
    source_bytes_length: int
    compressed_path: str
    compressed_bytes_sha256: DigestText
    compressed_bytes_length: int
    decoded_bytes_sha256: DigestText
    decoded_bytes_length: int
    record_count: int
    canonical_jsonl: bool
    packager_identity: DigestText
    packager_config_hash: DigestText
    created_at_utc: UtcTimestamp

@dataclass(frozen=True, slots=True)
class RawObjectRow:
    object_id: DigestText
    packaged_object_id: DigestText
    confidential_source_id: str
    authorization_attestation_id: str
    permitted_purpose: tuple[str, ...]
    disclosure_class: str
    acquisition_metadata: Mapping[str, JsonValue]
    semantic_state: Literal["unvalidated", "valid", "quarantined"]
    semantic_validation_hash: DigestText | None
    first_error_class: str | None
    first_error_event_index: int | None
    parent_ids: tuple[DigestText, ...]
    created_at_utc: UtcTimestamp
```

Byte domains are literal bytes at named stage. Decoded hash covers complete decompressed payload without normalization. Canonical JSONL is recorded separately; packager MUST NOT rewrite decoded data to make it canonical.
`PackagedObjectRow` is WP-00B transport authority and contains no authorization claim. `RawObjectRow` is WP-04B authority: it references one immutable packaged row and adds attestation plus semantic result. Its `object_id` hashes that join. A missing attestation cannot be represented as an authorized raw row.

### 12.2 Decision row

```python
@dataclass(frozen=True, slots=True)
class DecisionRow:
    game_id: str
    round_id: str
    decision_id: str
    seat: Seat
    source_object_id: DigestText
    split: Literal["train", "validation", "test", "decision_eval", "block_eval"]
    rules_hash: DigestText
    adapter_hash: DigestText
    observation_hash: DigestText
    action_table_hash: DigestText
    derivation_hash: DigestText
    actor_observation: ActorObservation
    chosen_action_id: ActionId
    privileged_label_ref: DigestText | None
```

Privileged rows are separate files/schema and join by opaque decision ID only in authorized training jobs. Inference loaders cannot import privileged row constructors.

### 12.3 RL rollout artifact

```python
@dataclass(frozen=True, slots=True)
class RlRolloutRow:
    row_id: DigestText
    wall_id: str
    trajectory_step: int
    game_id: str
    round_id: str
    decision_id: str
    actor: Seat
    observation_hash: DigestText
    actor_observation: ActorObservation
    action_table_hash: DigestText
    legal_mask: tuple[bool, ...]
    selected_action_id: ActionId
    behavior_policy_checkpoint_hash: DigestText
    old_selected_log_probability: float
    raw_advantage: float
    return_vector: UtilityVector
    bc_reference_checkpoint_hash: DigestText | None
    bc_reference_logits: tuple[float, ...] | None
    transition_provenance_hash: DigestText

@dataclass(frozen=True, slots=True)
class RlRolloutArtifact:
    schema_version: SchemaVersion
    rules_hash: DigestText
    utility_manifest_hash: DigestText
    observation_schema_hash: DigestText
    action_table_hash: DigestText
    behavior_policy_checkpoint_hash: DigestText
    behavior_model_spec_hash: DigestText
    source_wall_ledger_hash: DigestText
    row_derivation_hash: DigestText
    rows: tuple[RlRolloutRow, ...]
    digest: DigestText
```

`row_id = sha256(canonical bytes of the complete row excluding row_id)`. Validator recomputes every ID and rejects duplicates. `trajectory_step` is nonnegative and strictly increases within `(wall_id, game_id)`; artifact row order is the total lexicographic order `(wall_id, game_id, trajectory_step, actor, decision_id, row_id)`. Any reordering is noncanonical and changes/rejects artifact identity. Each row's observation hash, action table, and legal mask MUST equal its embedded actor observation. The selected action MUST be in range and legal. Behavior checkpoint identity MUST equal the artifact header. `raw_advantage` is the acting seat's scalar advantage under the utility manifest; its GAE/return/bootstrap/reward-shaping method and parameters are frozen by `row_derivation_hash`. `return_vector` retains all four seats. `old_selected_log_probability`, advantage, return values, and optional BC logits MUST be finite; exponentiating the old log probability in objective dtype MUST produce `0 < pi_old <= 1`. BC checkpoint/logits are both absent or both present; logits length equals the action table and use the same legal mask. When present, every row BC checkpoint MUST equal the consuming RunSpec's `bc_reference_checkpoint_hash`. Utility/rules identities MUST match throughout. Every `wall_id` MUST belong to `source_wall_ledger_hash`; evaluation walls are rejected. PPO and ACH consume the identical immutable artifact; neither recomputes advantages, returns, behavior probabilities, or BC logits.

### 12.4 Split protocol

```python
class Splitter(Protocol):
    def assign(self, games: Sequence[GameIdentity], spec: SplitSpec) -> SplitManifest: ...
```

- Assign complete games before decisions.
- Group source/player/time according to available attested metadata.
- Exact and near-duplicate components cannot cross split.
- Evaluation wall IDs cannot enter training/selection.
- Split manifest stores algorithm/version/seed/input hashes and every assignment.
- Missing grouping metadata is explicit; NEVER infer fabricated identities.

### 12.5 Loader

Loader verifies dataset manifest, shard hashes, schema hashes, row counts, legal masks, and split membership before yielding. Any corrupt shard aborts the run. Cache key is digest of dataset/split/schema/preprocess/layout/dtype/library identities. Cache miss rebuilds; incompatible cache never reshapes.

## 13. Semantic Randomness

All formal stochastic work uses purpose-discriminated named labels:

```python
@dataclass(frozen=True, slots=True)
class RandomStreamKey:
    purpose: Literal["wall", "belief_natural_sample", "belief_proposal_sample", "actor_policy_sample", "root_tree_selection", "rollout_transition", "rollout_advantage", "confirmation", "coupling_primitive", "mlmc_level", "rqmc_scramble", "smc_propagation", "smc_resampling", "training_shuffle", "training_dropout", "evaluation_schedule", "gumbel_root", "kernel_sample"]
    experiment_id: str
    split_id: str
    candidate_id: str | None
    case_id: str | None
    wall_id: str | None
    root_seat: Seat | None
    belief_epoch: BeliefEpochId | None
    parent_id: ParentId | None
    action_id: ActionId | None
    packet_id: PacketId | None
    fidelity_level: int | None
    population_id: int | None
    replicate_id: int
    scramble_id: int | None
    visit_index: int | None
    attempt_id: int

def semantic_seed(master_seed: bytes, *, key: RandomStreamKey) -> bytes:
    payload = canonical_bytes({"protocol": "hydra2_rng_v1", "master_seed": master_seed.hex(), "key": to_json(key)})
    return hashlib.sha256(payload).digest()
```

Rules:
- `RandomStreamSchema` declares required/forbidden fields for every literal purpose: exactly one of case/wall where relevant; natural/proposal/actor-policy/root-selection/transition/confirmation streams are distinct; coupling uses branch-independent primitive identity; MLMC requires fidelity; RQMC requires scramble; SMC propagation and resampling are distinct and require population; Gumbel requires action; kernel sampling requires candidate/parent/action/epoch; retries increment `attempt_id`; unused fields MUST be null.

- Never derive by call order.
- Never reuse one stream for distinct purposes.
- Coupled branches share only explicitly declared primitive labels; each branch maps primitives through its own conditional law.
- Checkpoint stores counters/semantic cursors needed for exact continuation.
- Final evaluation seeds remain inaccessible to training and selection code paths.

## 14. Belief, World, Proposal, and Packet Kernel

### 14.1 Full world

```python
@dataclass(frozen=True, slots=True)
class FullWorld:
    world_id: DigestText
    simulator_snapshot: "SimulatorSnapshot"
    concealed_hands: tuple[tuple[TileId, ...], ...]
    live_wall: tuple[TileId, ...]
    dead_wall: tuple[TileId, ...]
    latent_state: Mapping[str, JsonValue]
    rules_hash: DigestText
    observation_hash: DigestText
```

Privileged. NEVER serialized into actor model/search keys/logs. Search sandbox may hold worlds behind opaque IDs.

### 14.2 Target and proposal

```python
@dataclass(frozen=True, slots=True)
class BeliefEpoch:
    epoch: BeliefEpochId
    target_id: DigestText
    root_actor: Seat
    observation_hash: DigestText
    rules_hash: DigestText
    belief_model_hash: DigestText
    event_model_hash: DigestText
    proposal_spec_hash: DigestText

@dataclass(frozen=True, slots=True)
class Particle:
    parent_id: ParentId
    world_ref: str
    epoch: BeliefEpochId
    target_id: DigestText
    source: Literal["natural", "proposal"]
    log_target_density: float
    log_proposal_density: float
    proposal_id: DigestText

class Belief(Protocol):
    def begin(self, observation: ActorObservation, *, model_id: DigestText) -> BeliefEpoch: ...
    def sample_natural(self, epoch: BeliefEpoch, *, count: int, rng: "RandomStream") -> tuple[Particle, ...]: ...
    def sample_proposal(self, epoch: BeliefEpoch, *, proposal: "ProposalSpec", count: int, rng: "RandomStream") -> tuple[Particle, ...]: ...
    def condition_for_actor(self, epoch: BeliefEpoch, *, actor_observation: ActorObservation, count: int, rng: "RandomStream") -> tuple[Particle, ...]: ...
    def pushforward_condition(self, epoch: BeliefEpoch, *, action: CanonicalAction, packet: ActorVisiblePacket) -> BeliefEpoch: ...
    def log_density(self, epoch: BeliefEpoch, world_ref: str) -> float: ...
```

Natural sample requires `log_target_density == log_proposal_density` and ratio one. Proposal support: every target-positive sampled/evaluated region must have positive mixture density. Nonfinite density or stale epoch is hard failure.

### 14.3 Transition packet kernel

```python
@dataclass(frozen=True, slots=True)
class PacketSuccessor:
    packet: ActorVisiblePacket
    successor_world_ref: str
    delta_ref: str
    probability: float
    log_physical_probability: float
    log_actor_policy_probability: float

class PacketKernel(Protocol):
    def enumerate_next(self, *, epoch: BeliefEpoch, particle: Particle, action: CanonicalAction, policy_set: "PolicySet") -> tuple[PacketSuccessor, ...]: ...
```

Candidate 3 requires finite exhaustive enumeration. For each parent/action:

- successors pairwise disjoint by complete packet identity;
- probabilities finite/nonnegative;
- sum equals one within CandidateSpec kernel tolerance;
- physical and actor-policy likelihood each applied once;
- successor state follows exact simulator transition;
- packet fields satisfy root actor visibility;
- no parent-only reweight qualifies as successor posterior.
> Caveat 2026-09-04 (non-normative): `PolicySet` verdict — `src/hydra2/belief/natural.py:86-95` remains a WP-07A placeholder (`policies` provenance tuple; `log_prob` returns `0.0` uniform). Kernel supplies deterministic likelihood; no per-policy evaluation. Normative §14.3 enumeration requirements unchanged; a real policy set is future work gated by §24.

### 14.3.1 Sampled kernel mode (non-exhaustive, versioned)

Beside exhaustive enumeration lives mode `natural_trace_sample_v1`: per (parent,
action), a frozen count L of draws from the SAME frame law (zero new policy
semantics), each carrying (packet, successor refs, raw weight, provenance with mode
string + L + seed material). NO finite-sample mass-one claim: raw weighted estimates
only; unsampled children are unobserved support, never zero-probability events
(renormalizing a sampled batch does not recover an exact partition). Exhaustive
`enumerate_next` is untouched and remains the only WP-09A certificate path. Sampled
batches carry the mode string in every downstream key/hash so the modes never mix.
Sampling streams use purpose `kernel_sample` (never a live purpose: ledger
collisions fail closed).

## 15. Candidate Specification and Search API

```python
@dataclass(frozen=True, slots=True)
class ResourceBudget:
    mode: Literal["gameplay_5s", "ponder", "analysis"]
    deadline_ms: int
    fallback_margin_ms: int
    max_model_calls: int | None
    max_transitions: int | None
    max_particles: int | None
    max_memory_bytes: int | None

@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    algorithm: str
    algorithm_version: SchemaVersion
    rules_hash: DigestText
    utility_id: str
    utility_manifest_hash: DigestText
    action_table_hash: DigestText
    observation_schema_hash: DigestText
    packet_boundary_hash: DigestText
    model_hash: DigestText
    belief_model_hash: DigestText | None
    event_model_hash: DigestText | None
    continuation_policy_hashes: tuple[DigestText, ...]
    proposal_spec_hash: DigestText | None
    case_manifest_hash: DigestText
    resource_budget: ResourceBudget
    fallback_candidate_id: Literal["candidate0"]
    tie_break: str
    rng_protocol_hash: DigestText
    random_stream_schema_hash: DigestText
    parameters: Mapping[str, JsonValue]

@dataclass(frozen=True, slots=True)
class SearchRequest:
    observation: ActorObservation
    legal_actions: tuple[CanonicalAction, ...]
    candidate_spec: CandidateSpec
    deadline_monotonic_ns: int
    belief_epoch: BeliefEpoch | None

@dataclass(frozen=True, slots=True)
class SearchResult:
    selected_action: CanonicalAction
    candidate_actions: tuple[CanonicalAction, ...]
    value_vectors: tuple[UtilityVector, ...]
    candidate_spec_hash: DigestText
    telemetry: "ResourceTelemetry"
    evidence_refs: tuple[DigestText, ...]
    completed: bool

class Planner(Protocol):
    def act(self, request: SearchRequest) -> SearchResult: ...
    def observe(self, packet: ActorVisiblePacket) -> None: ...
    def ponder(self, *, deadline_monotonic_ns: int) -> None: ...
```

Runner validates selected action against exact mask. Incomplete/deadline result is discarded and Candidate 0 fallback invoked with reserved margin. Ponder can mutate only planner-owned speculative state; `observe` verifies packet/epoch then commits or rebuilds.
CandidateSpec identity binds exact utility bytes, packet-boundary semantics, and purpose-specific RNG schema. Request observation rules/action/observation/packet hashes, model/checkpoint utility manifest hash, every returned `UtilityVector`, and every promotion record MUST match those fields or raise `ContractError` before search.
> Caveat 2026-09-04 (non-normative): hash-binding-before-cases verified — `CandidateSpec` identity rule above holds; factories bind every hash before cases with candidate0 as descriptor authority (`src/hydra2/search/candidate0.py:235-239,294-311`). Gumbel/PBRF/local-resolving/qualification mirror those descriptors verbatim. See `## Status 2026-09-04` §B. Normative text unchanged.

## 16. Exact Candidate Algorithms

### 16.1 Candidate 0

```python
def candidate0(request, *, model, encoder, action_table, action_codec):
    context = action_context_from(request.observation)
    out = model.evaluate(encoder.encode(request.observation))
    probs = masked_policy(out.policy_logits, request.observation.legal_mask)
    action_id = frozen_choice(probs, out.value_vector, request.candidate_spec.tie_break)
    action = action_codec.decode(action_id, table=action_table, context=context)
    return complete_result(action)
```

One model evaluation only. No belief, particles, search, pondering, online adaptation, or hidden state.

### 16.2 Candidate 1 natural ISMCTS

State keys MUST be canonical information-set hashes for the acting player. Root action statistics shared across every hidden world consistent with root observation. Each simulation:

```text
world <- natural root particle
while within frozen depth and nonterminal:
  actor_obs <- exact simulator actor observation for current actor
  if current actor == root actor:
    info_key <- hash(actor_obs without legal-mask redundancy)
    action <- root tree policy(info_key, exact legal mask, frozen UCT)
    append root information node to backup path
  else:
    action <- sample frozen continuation_policy[current actor](actor_obs, exact legal mask)
  transition exact world
leaf vector <- exact terminal utility or actor-visible model value
backup same four-seat vector through visited root information nodes
```

No world ID/full-state hash in action-selection key. Non-root policy sees that actor's observation only. Root selection indexes root seat after vector aggregation. Re-determinization disabled.

### 16.3 Candidate 2 natural DESPOT

Scenario is `(natural world, semantic random stream)`. Scenario tree branches on actor-visible observations. Bounds/regularization are implementation heuristics unless mathematically proved and named. Blueprint policy is feasible lower estimate only. Arbitrary proposal weighting prohibited.

### 16.4 Candidate 3 PBRF core

```python
def build_pbrf(epoch, parent_count, candidates, policy_set, kernel, rng):
    parents = belief.sample_natural(epoch, count=parent_count, rng=rng)
    frozen_candidates = freeze(candidates(parents))
    children = {}
    for action in frozen_candidates:
        for parent in parents:
            successors = kernel.enumerate_next(
                epoch=epoch, particle=parent, action=action, policy_set=policy_set)
            require_partition(successors)
            for successor in successors:
                key = (action_id(action), successor.packet.packet_id)
                children[key].append(ChildEntry(
                    parent_id=parent.parent_id,
                    successor_world_ref=successor.successor_world_ref,
                    successor_delta=successor.delta_ref,
                    raw_weight=successor.probability / len(parents),
                    target_id=epoch.target_id,
                    epoch=epoch.epoch,
                ))
        require abs(sum(normalizer(children[action,*])) - 1) <= kernel_tolerance
    fixed_allocate(children, frozen_schedule)
    return ImmutableForest(epoch, parents, frozen_candidates, children)
```

Commit:

```python
def commit(forest, action, actual_packet):
    require action was emitted from forest
    authoritative_epoch = belief.pushforward_condition(
        forest.epoch, action=action, packet=actual_packet)
    matching = forest.child(action, actual_packet.packet_id)
    if matching is absent or not target-compatible(authoritative_epoch):
        return fresh_rebuild(authoritative_epoch), CommitDisposition("miss_rebuild")
    promoted = rekey_and_verify(matching, authoritative_epoch)
    carry_logps = conditional_carry_logps(matching)  # log(raw_i / Z); zero-mass -> miss
    squash_all_sibling_values_visits_posteriors_pairings(forest)
    return promoted_carried_forest, CommitDisposition("hit_commit")
```

Sample law (normative): hit-commit promoted parents are CARRIED conditionals sampling `b_{eta,a,e}^+`, never fresh naturals — the emitted action and realized packet selected them, and selection conditions the population. Their densities are the normalized conditional weights `log(raw_i / Z)`, never uniform `-log(N)`; both density fields take this value so no hidden importance ratio is smuggled. Zero-mass conditioning supports no population and takes the miss path. Forests accept parent source in `{natural, carried}`; `carried` requires finite equal densities. Natural-only samplers MUST never consume carried parents as fresh draws.

Ancestry disclosure (normative, PR4): `ChildEntry` and promoted/rebuilt parent
particles carry `ancestors: tuple[str, ...] = ()` (parent-id chain, oldest-first).
Fresh roots carry `()`; commit appends the conditioning parent
(`e.ancestors + (e.parent_id,)`); rekey passes chains through unchanged; miss-path
dummies carry `()` (the provenance break IS the miss semantics). Weight, ESS,
normalizer, and allocation math MUST NOT read ancestry. Ancestry exists so shared-parent
populations can never be mistaken for independent draws: any population-level
variance MUST use covariance form, never ESS/N, wherever ancestors are shared.

A child view computes normalized weights only when normalizer > 0. ESS is diagnostic. Confirmation always fresh natural full fidelity after candidate freeze.
`successor_world_ref` is mandatory and resolves to the exact transitioned world in the privileged sandbox. `successor_delta` is a cache optimization only; reconstruction from parent+delta MUST digest-equal `successor_world_ref` before use.
> Caveat 2026-09-04 (non-normative): PBRF guarded-import verdict — `src/hydra2/search/pbrf.py:50-88` keeps `_HAS_*` import guards, but load-bearing paths fail closed: missing kernel raises `ContractError` (`pbrf.py:524-530`), missing belief/epoch raises (`pbrf.py:539`, `1500`, `1566`), utility-hash derivation raises (`pbrf.py:1062`). Telemetry/RNG mirrors degrade to debug-logged local derivation only, never to silent success. Normative §16.4 unchanged.

### 16.5 Candidate 4 modules

Implement formulas exactly from blueprint §§11.1-11.10. Module interface:

```python
class PbrfModule(Protocol):
    @property
    def module_id(self) -> str: ...
    def validate_spec(self, spec: CandidateSpec) -> None: ...
    def transform(self, context: "PbrfContext") -> "PbrfContext": ...
    def evidence(self) -> tuple[DigestText, ...]: ...
```

One enabled module per qualification CandidateSpec. Required invariants:

- Rao-Blackwell: enumerate declared finite variable, exact conditional weights, charge calls.
- Targeted MIS: `m=(n0*q0+n1*q1)/(n0+n1)`; apply `b*L/m` once; natural floor; no clipping.
- CRN: share primitive uniforms only; branch-specific inverse mapping; measure covariance.
- MLMC: signed complete telescope; pilot-frozen levels/counts; independent groups.
- RQMC: one scramble one dependent replicate; uncertainty across independent scrambles.
- Coreset: nonnegative weights sum one; search only.
- Pruning: only simultaneous one-sided `U_b < L_a` at declared uncertainty unit/multiplicity.
- Controlled SMC: unnormalized estimator; independent populations uncertainty; no unbiased ratio claim.
- Persistence: target-compatible commit only; all siblings squashed.
- VOC: every live child floor, cap, exact total budget, charged overhead; frozen routing.
  Exact routing (normative, PR3): 20% of ponder units to support/recovery work, 20%
  round-robin over retained positive-mass cells (at most one floor unit per allocated
  cell until the pool exhausts), 60% by frozen predicted-value scores; no cell exceeds
  `max(0.25, 1/m)` of total units for m eligible cells; largest-remainder quantization
  with canonical cell-ID ties; infeasible floors/caps take a deterministic predeclared
  relaxation, logged; unallocatable units stay unused, never force-assigned. Overhead
  charged to context budgets. Scores route exploratory work only, never confirmatory
  evidence.

### 16.6 Candidate 5 local resolving

Tables key `(actor, information_node_hash)`, never root world. Every actor update uses only that actor's information set. Return vectors remain four-seat. Subgame horizon, abstraction, leaf model, update, iteration count, and averaging are CandidateSpec fields. Cycle and abstraction failure abort candidate. Output is empirical optimizer result, never equilibrium certificate.
> Caveat 2026-09-04 (non-normative): local_resolving confirmation flag — `src/hydra2/search/local_resolving.py:1300-1304` is the warm-start bias branch (`0.7` on action 0, gated by `parameters["warm_start"]`), not a confirmation gate; the semantic-stream comment naming `actor_policy_sample + confirmation` is at `local_resolving.py:1317-1330`. Hash authority follows candidate0 descriptors (`local_resolving.py:984-990,1044-1045`). Normative §16.6 unchanged.

### 16.7 Candidate 6 Gumbel search

Root Gumbels derive from `(case_id, root_seat, candidate_id, action_id)`. Sequential-halving rounds and visit allocations are CandidateSpec-supplied. Every transition exact. Model supplies priors, opponent policy, belief, leaf vector only. Backup vector; scalarize at root. Matched comparator counts model calls and exact transitions.

Candidate profiles (normative status: provisional priors, PR3): frozen
`CandidateProfile` rows (candidate cap M, horizon H, carry quota, halving rounds;
jobs = 4M·log2M, transitions ≤ jobs×H) with a pure admission gate selecting the
largest profile passing the synchronized cost gate on disjoint pilot states; nothing
fits → Candidate 0 (always reachable; never an error, never a forced profile). Profile
numbers promote to capacities only via the RTX pilot fixture. Gate selection from
held-out win rates is prohibited.
> Caveat 2026-09-04 (non-normative): Gumbel hash authority — `src/hydra2/search/gumbel.py:1677-1714` derives model/utility from the live model (raising on failure at `gumbel.py:1702`) and rng/stream/case verbatim from candidate0 descriptors; `_load_default_hashes` (`gumbel.py:1630-1666`) is file-backed-only. Normative §16.7 unchanged.

### 16.8 Candidate 7 distillation

Teacher identity frozen before generation. Each record stores actor observation hash, legal mask, teacher policy over canonical actions, four-seat return, optional event/belief labels, teacher CandidateSpec hash, resource budget, and trajectory provenance. Student inference has no privileged fields. Replacing teacher invalidates all dependent records/checkpoints/results.

### 16.9 Candidate 8 joint type/world robustness

State is joint particles `(theta, world, weight)`, not independent marginals. Observed opponent action enters packet kernel policy likelihood exactly once. Sequential updates preserve type/world correlation. Uncertainty set member is coherent policy keyed by opponent information sets, legal masks, and same-information equality. `rho`, `epsilon`, divergence direction, support class, rationality rule, feasibility proof, and nominal model are manifest-supplied. Absent held-out calibration blocks use.

## 17. Persistence Factorial

Planner capabilities:

```python
@dataclass(frozen=True, slots=True)
class PersistenceArm:
    id: Literal["B", "F", "R", "P", "C"]
    retain_state: bool
    opponent_time_compute: bool
    own_deadline_ms: int
    extra_wait_allowance_ms: int
    deployable: bool
```

Exact definitions:

- B: Candidate 0; no search/state/ponder.
- F: fresh search each own decision; destroy all search state after action; no opponent-time compute.
- R: retain target-compatible state; zero search work from emitted action until next actor-visible packet.
- P: retain; work only between emitted action and next actor-visible packet; commit through verified packet.
- C: fresh at next own observation; no retained state; receives predeclared own deadline + assigned wait-window allowance; laboratory only.

B/F/R/P share deployable own deadline. Standard table planner deadline <= 5,000 ms minus frozen fallback margin. Every arm logs actual model calls, transitions, synchronized duration, peak memory, and joules when available.

Ponder quota (normative, PR1): P-arm `ponder()` takes optional
`ponder_quota_total: int | None = None`. `None` reproduces legacy behavior bit-for-bit
(fixed 2 units per child, sub-0.5ms fallback to 1). When set (positive int, bool
rejected), at most that many ponder units distribute deterministically across sorted
child ids, round-robin one unit per child until quota exhausts; every counter
(child_stats, ponder_calls, model calls, transitions, budget, joules) charges exactly
the distributed units. Deadline still gates all work. B/F/R/C paths ignore the quota
entirely (R-zero assertion unchanged). Ponder-derived statistics are working
statistics, never confirmatory evidence; population variance uses covariance form,
never ESS/N.

## 18. Evaluation and Statistical Contract

### 18.1 Schedules

```python
@dataclass(frozen=True, slots=True)
class MatchSchedule:
    wall_ids: tuple[str, ...]
    walls_hash: DigestText
    seat_allocations: tuple[tuple[str, str, str, str], ...]
    latency_schedule_hash: DigestText
    rules_hash: DigestText
    seed_protocol_hash: DigestText
```

For each wall:

- symmetric 2-v-2 uses all six placements of A in two seats;
- 1-v-3 diagnostic rotates A through all four seats;
- schedule committed before results;
- divergent games remain members of one wall block, not identical counterfactual paths.

### 18.2 Resource telemetry

```python
@dataclass(frozen=True, slots=True)
class ResourceTelemetry:
    mode: str
    wall_id: str | None
    case_id: str | None
    candidate_spec_hash: DigestText
    hardware_hash: DigestText
    environment_hash: DigestText
    cold_start: bool
    synchronized_elapsed_ms: float
    model_calls: int
    exact_transitions: int
    particles: int
    fallback_used: bool
    timeout: bool
    illegal_action: bool
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None
    host_peak_bytes: int | None
    energy_joules: float | None
    graph_breaks: int | None
    recompiles: int | None
    invalid_reason: str | None
```

Counters include all planner overhead named by resource view. Missing required telemetry invalidates block according to predeclared tolerance; it is never imputed silently.

### 18.3 Metrics and uncertainty

Primary block outcome: declared expected-final-placement contrast. Diagnostics: raw points, first/fourth, deal-in, riichi, call, legality, timeout, fallback, latency, energy.

Independent unit is complete wall block. Bootstrap resamples blocks. Sign-flip flips block contrasts. Clustered model/calibration metrics use game/player groups, never decisions. Fixed sample size:

```text
N = ceil((((z_(1-alpha) + z_(1-beta)) * s) / delta)^2)
```

`alpha`, `beta`, pilot `s`, practical margin `delta`, multiplicity, maximum blocks, and fixed-N versus named time-uniform CS are frozen blind to arm labels. Adaptive peeking without declared sequential design invalidates confirmation.

### 18.4 Promotion record

```python
@dataclass(frozen=True, slots=True)
class PromotionRecord:
    candidate_spec_hash: DigestText
    utility_manifest_hash: DigestText
    comparator_spec_hashes: tuple[DigestText, ...]
    case_manifest_hash: DigestText
    result_table_hash: DigestText
    resource_view: str
    uncertainty_unit: Literal["case", "iid_pair", "wall_block", "smc_population", "rqmc_scramble", "game_cluster"]
    pass_inequality: str
    observed_estimate: float
    confidence_bounds: tuple[float, float]
    gates: Mapping[str, Literal["passed", "failed", "not_applicable"]]
    disposition: Literal["promoted", "rejected", "blocked"]
```

All Candidates 0-6 retain records, including failure/rejection. Search-derived candidate selection and natural confirmation must use disjoint semantic streams.
Use `case` for independent decision cases, `iid_pair` for paired natural confirmations, `wall_block` for duplicate matches, `smc_population` for independent controlled-SMC populations, `rqmc_scramble` for independent scrambles, and `game_cluster` only for held-out model/calibration metrics. CandidateSpec and result schema reject mismatched units.

Additive provenance bindings (normative, PR4): `PromotionRecord` gains three OPTIONAL
fields with defaults — `schedule_hash: DigestText | None = None`,
`environment_hash: DigestText | None = None`,
`excluded_blocks: tuple[ExcludedBlock, ...] = ()`. The factory accepts missing keys
(defaults apply); unknown keys still raise. `record_to_json` includes them
(`None` serializes null); `promotion_digest` covers them. Migration: digests change
going forward; no production record exists (WP-10 blocked by design), so no backfill
is owed. A promotion that cannot name its schedule, machine, and excluded walls is
not reproducible and MUST NOT issue `disposition="promoted"`.

Confirmation sidecar (normative, PR4): pure function `confirmation_sidecar(*,
schedule, blocks: BlockAggregateResult, telemetry_report) -> dict` returning
`{schedule_commitment_hash, excluded: [{wall_id, reason, detail}],
telemetry_completeness_digest}`. Confirmation paths that hand-roll hashes (e.g. the
teacher five-arms) call it BESIDE — never instead of — their existing hashes; their
decision outputs stay byte-identical. Full admission migration belongs to a WP-10-owned change.

## 19. Performance Qualification

Initial semantic oracle: eager FP32, exact simulator eager, plain PyTorch, padded/bucketed histories, SDPA, dense action head, AdamW default, ordinary checkpoint.

Compile ladder order is fixed:

```text
1 eager
2 default, dynamic=None
3 max-autotune-no-cudagraphs, dynamic=False
4 max-autotune, dynamic=False
```

Later arm cannot qualify before earlier arm evidence exists; rejection does not forbid testing later only when CandidateSpec states reason and comparator remains eager. Performance options are separate CandidateSpecs: compile mode, SDPA/FlexAttention, FP16, BF16, TF32, pin/nonblocking, optimizer selection, nested, checkpointing, sparsity, FP8, NVFP4, container. Bundled attribution is prohibited.

Acceptance requires all applicable blueprint §16.3.4 gates. Exact discrete outputs/legal masks must match. Numeric tolerances are pilot-frozen. RTX 5070/A100 identities, caches, tables, and conclusions separate. A missing measurement means unqualified, not neutral.

A100 ledger:

```python
@dataclass(frozen=True, slots=True)
class A100Reservation:
    request_id: str
    rtx_bottleneck_evidence_hash: DigestText
    hypothesis: str
    candidate_spec_hash: DigestText
    corpus_hash: DigestText
    requested_gpu_hours: float
    transfer_plan: str
    compile_amortization_plan: str
    approved_at_utc: UtcTimestamp

@dataclass(frozen=True, slots=True)
class A100Reconciliation:
    request_id: str
    start_utc: UtcTimestamp
    stop_utc: UtcTimestamp
    charged_gpu_hours: float
    retries: int
    failures: int
    output_hashes: tuple[DigestText, ...]
    disposition: str
```

Reserve update is transactional. Failed/aborted time charged. One time-separated A100 is never distributed topology.

## 20. Training State and Objectives

```python
@dataclass(slots=True)
class TrainingState:
    global_update: int
    microstep: int
    epoch: int
    examples_seen: int
    best_selection_metric: float | None
    sampler_cursor: JsonValue
    semantic_rng_state: JsonValue
```

```python
ObjectiveId = Literal["supervised_v1", "distill_v1", "ppo_clipped_v1", "ach_direct_sampled_v1"]

@dataclass(frozen=True, slots=True)
class RunSpec:
    schema_version: SchemaVersion
    run_kind: Literal["supervised", "distill", "rl"]
    model_spec_hash: DigestText
    initial_checkpoint_hash: DigestText
    dataset_manifest_hash: DigestText | None
    rollout_artifact_hash: DigestText | None
    objective_id: ObjectiveId
    objective_parameters: Mapping[str, JsonValue]
    behavior_policy_checkpoint_hash: DigestText | None
    bc_reference_checkpoint_hash: DigestText | None
    optimizer_id: str
    optimizer_parameters: Mapping[str, JsonValue]
    scheduler_id: str
    scheduler_parameters: Mapping[str, JsonValue]
    gradient_clip_norm: float | None
    optimizer_minibatch_size: int
    microbatch_size: int
    accumulation_steps: int
    minibatch_order_hash: DigestText
    max_updates: int
    runtime_spec_hash: DigestText
    resource_ledger_reservation_hash: DigestText
    seed_manifest_hash: DigestText
    selection_metric_id: str
    stopping_rule: Mapping[str, JsonValue]
    checkpoint_frequency_updates: int
    scaler_id: str | None
    digest: DigestText

@dataclass(frozen=True, slots=True)
class MatchedObjectiveGroup:
    group_id: DigestText
    rollout_artifact_hash: DigestText
    initial_checkpoint_hash: DigestText
    minibatch_order_hash: DigestText
    shared_run_fields_hash: DigestText
    ppo_run_spec_hash: DigestText
    ach_run_spec_hash: DigestText
```

Exactly one training source applies: supervised/distill runs require `dataset_manifest_hash` and no rollout; RL requires `rollout_artifact_hash`, behavior checkpoint, and no mutable dataset source. BC reference is required exactly when objective `w_bc > 0`. `optimizer_minibatch_size = microbatch_size * accumulation_steps`; all sizes, update counts, and checkpoint frequency are positive. `minibatch_order_hash` binds the complete ordered row-ID batches for every update. Registered optimizer/scheduler/objective/selection/scaler IDs have closed parameter schemas; unknown or extra keys fail validation. Stopping rule is finite and closed by `run_kind`; RL matched-objective runs MUST use the same rule without adaptive metric peeking. Runtime and resource reservation exist before launch. Actual result records bind RunSpec, runtime, and reconciled resource-ledger hashes.
Matched PPO/ACH RunSpecs MUST have byte-identical canonical values for every field except `objective_id`, objective-specific parameters, and `digest`. Shared objective parameters `w_value`, `w_bc`, and `alpha` MUST be present and byte-identical in both specs; both BC reference checkpoint hashes MUST equal the rollout rows when `w_bc > 0`. `shared_run_fields_hash` hashes all shared fields plus shared objective parameters and MUST match both specs. Any ModelSpec/RunSpec/RlRolloutArtifact field change invalidates dependent caches, checkpoints, exports, result tables, promotion records, and the matched-objective group.

Supervised objective:

```text
L = w_policy * masked_cross_entropy
  + w_placement * placement_distribution_loss
  + sum_h w_event[h] * event_loss[h]
  + sum_h w_belief[h] * belief_loss[h]
```

Every weight/head/target/masking/reduction is model-spec supplied. Zero-weight heads MAY be absent; implicit defaults prohibited. Behavior cloning anchors remain in Candidate 7. Optimizer, scheduler, clipping, accumulation, scaler, selection metric, stopping rule, and checkpoint frequency are RunSpec fields.

AMP:

- autocast forward + loss only;
- FP16 uses GradScaler, unscale before finite check/clipping;
- BF16 no scaler unless separately justified;
- every trainable path receives finite gradient under coverage fixture;
- skipped updates recorded.

RL remains project-owned actor/learner/replay. Replay carries actor-visible input and privileged labels separately. Historical opponents immutable. Evaluation walls never enter replay.

### 20.1 Masked PPO comparator

PPO consumes one frozen `RlRolloutArtifact`. For current legal logits `z`, selected action `a`, stored old selected log probability `log_pi_old`, raw advantage `A`, model value vector `v`, return vector `G`, and frozen finite RunSpec parameters `(clip_eps, eps_std, w_value, w_bc, alpha)`:

```text
pi          = legal_softmax(z)
log_pi_a    = log(pi[a])
ratio       = exp(log_pi_a - log_pi_old)
A_std       = (A - mean_batch(A)) / sqrt(mean_batch((A - mean_batch(A))^2) + eps_std)
surrogate   = min(ratio * stop_gradient(A_std),
                  clamp(ratio, 1-clip_eps, 1+clip_eps) * stop_gradient(A_std))
L_value     = mean_batch_seat((v - G)^2)
L_bc        = mean_batch(KL(pi || pi_bc)) over legal actions
L_PPO       = -mean_batch(surrogate) + w_value * L_value + w_bc * L_bc
              - alpha * mean_batch(entropy(pi))
```

Require `0 < clip_eps < 1`, `eps_std > 0`, and `w_value,w_bc,alpha >= 0`. `pi_bc` is legal softmax of the rollout's frozen BC logits; when `w_bc == 0`, BC fields MUST be absent and `L_bc=0`. PPO uses unclipped four-seat value MSE and no adaptive KL stop. Ratio, clip fraction, legal entropy, legal KL, value loss, finite loss/gradient, and explained variance by seat are reported.

### 20.2 Optional direct-sampled ACH objective

PPO remains mandatory comparator. ACH MAY run only in a `MatchedObjectiveGroup` over identical rollout, initialization, ordered optimizer minibatches, optimizer/scheduler, update count, runtime, seeds, and resource reservation.

For legal logits `z`, selected action `a`, `pi_old = exp(log_pi_old)`, raw advantage `A`, and frozen RunSpec parameters `(eta, eps, l_th, pi_min, eps_A, w_value, w_bc, alpha)`:

```text
c       = mean(z[j] for legal j)
y[j]    = clamp(z[j] - c, -l_th, l_th) for legal j; -inf otherwise
pi      = softmax(y)
rho     = pi[a] / max(pi_old, pi_min)
A_bar   = A / sqrt(mean_batch(A^2) + eps_A)
gate    = (A_bar >= 0 and rho < 1 + eps and y[a] <  l_th)
       or (A_bar <  0 and rho > 1 - eps and y[a] > -l_th)
L_ACH   = -mean_batch(gate * eta * y[a] * stop_gradient(A_bar) / max(pi_old, pi_min))
          + w_value * L_value + w_bc * L_bc - alpha * mean_batch(entropy(pi))
```

Require `eta >= 0`, `eps > 0`, `l_th > 0`, `0 < pi_min <= 1`, `eps_A > 0`, and `w_value,w_bc,alpha >= 0`; every parameter and stored `log_pi_old` is finite, with `0 < pi_old <= 1` in objective dtype. `L_value`, `pi_bc`, and `L_bc` have exactly §20.1 semantics. Zero advantage belongs to the nonnegative gate. Every `mean_batch`/`mean_batch_seat` is over the complete frozen optimizer minibatch before microbatch splitting; accumulation sums exact numerators/counts, so microbatch size cannot change the objective.

Illegal probabilities and illegal-logit gradients MUST be exactly zero. Report old-probability clamp fraction/minimum, gate fractions by advantage sign/legal-action count, ratio bounds/clipping fraction, entropy fraction, legal BC-KL, value loss, and finite loss/gradient. ACH uses the same rollout/checkpoint/runtime path; another backend, residual head, selector, advantage/return recomputation, or rollout law is prohibited. Required fixtures: formula parity, selected-gradient direction, blocked gradients, all-illegal/selected-illegal rejection, all-zero advantage, JSON-finite metrics, real optimizer update, frozen-artifact/direct-call equality, deterministic resume, and PPO matched-group comparison. No held-out match gain means `rejected`, not default.

## 21. Configuration and Validation

Config categories:

```text
RulesManifest, UtilityManifest, ActionTable, EventSchema, ObservationSchema,
EnvironmentManifest, DatasetManifest, ModelSpec, RunSpec, BeliefSpec,
ProposalSpec, CandidateSpec, CaseManifest, MatchSchedule, PerformanceCandidate,
PromotionRecord
```

Each config:

- has schema version and canonical hash;
- resolves all references by hash before run;
- rejects unknown keys by default;
- contains no executable Python/import path unless schema explicitly permits a registered implementation ID;
- contains no environment-dependent implicit path;
- stores secrets only as external secret references, never values;
- passes `pixi run config-check`.

Registry maps registered string IDs to constructors. Arbitrary dynamic imports from config prohibited.
> Caveat 2026-09-04 (non-normative): toolchain hygiene — pyrefly keeps 10 codes at `warn` (triage, non-blocking) vs 16 at `error` (CI-blocking); PERF is advisory for `training`+`models` only with per-file-ignores elsewhere. See `## Status 2026-09-04` §D. Normative §21 `pixi run config-check` gate unchanged.

## 22. Required Golden and Adversarial Fixtures

Permanent minimum fixture IDs:

```text
ART-CANON-001 RFC8785 key/number/unicode
ART-ATOMIC-001 interrupted publication
RULE-TENHOU-001 complete manifest
ACT-ROUNDTRIP-001 every action kind/red identity
OBS-DORA-005 exact five slots
OBS-DRAW-PRIVATE-001 concealed draw split
OBS-HIDDEN-PERM-001 unseen permutation
OBS-CANARY-001 forbidden hidden payload
ENG-TRACE-001 seeded complete game
ENG-CALL-PRIORITY-001 complete call/pass resolution
DATA-CORRUPT-ZSTD-001 valid magic truncated
DATA-SPLIT-LEAK-001 duplicate across partitions
RUNTIME-RESUME-001 identical next update
MODEL-PUBLIC-SHAPE-001 actor-visible shanten/ukeire parity and hidden invariance
RL-ROLLOUT-001 immutable row identity/action-mask/behavior-policy validation
RL-PPO-FORMULA-001 masked clipped-PPO formula/value/BC reduction parity
RL-ACH-FORMULA-001 direct-sampled ACH formula/mask/gradient parity
BELIEF-FINITE-001 exact world normalization
BELIEF-PUSH-001 pushforward equals rebuild
PACKET-PARTITION-001 mass one/no duplicate
MIS-DOUBLE-001 detects second correction
CRN-MARGINAL-001 branch marginals preserved
MLMC-TELESCOPE-001 signed complete telescope
RQMC-SCRAMBLE-001 uncertainty across scrambles
SMC-UNNORM-001 exact unnormalized expectation
PBRF-COMMIT-001 child commit equals rebuild
PBRF-STALE-001 sibling access rejected
SEARCH-NONANTICIPATE-001 same observation same root policy
EVAL-BLOCK-001 block bootstrap/sign flip
PERF-LEGAL-PARITY-001 exact legal/chosen action
```

Builders MAY add fixtures. They MUST NOT rename/remove/weaken these without a contract migration that updates all documents and dependent evidence.

## 23. Definition of Implemented

A module is implemented only when:

1. Public API and invariants in this specification exist.
2. Its `BUILD_EXECUTION_PLAN.md` package checklist passes.
3. Positive, boundary, corruption, and forbidden-path fixtures pass.
4. Artifact identities and completion record are published.
5. Fresh-process smoke path exercises actual integration.
6. No downstream-facing stub, TODO, fake fallback, mock engine, approximate rules transition, or unqualified accelerator remains on enabled path.

Documentation alone never changes implementation status. A type definition without producer, consumer, validation, tests, and evidence is incomplete.

## 24. Contract Change Procedure

Before changing any schema, rule, action ordering, visibility field, model tensor, estimator, runtime/dependency, performance arm, or promotion criterion:

1. State reason and affected requirement.
2. List invalidated environment/contract/policy/run/checkpoint/compile/result hashes.
3. Define compatibility: exact break or backward read.
4. Define migration/rebuild; no padding or semantic shim for incompatible data.
5. Update this spec, execution plan, project plan, and blueprint where affected.
6. Add failure fixture proving old/new distinction.
7. Re-run every invalidated qualification gate.

<critical>
No builder discretion exists to weaken contracts for convenience. Unknown Tenhou value, absent authorization, unavailable dependency, missing pilot threshold, unsupported engine behavior, stale hash, or insufficient held-out data means `blocked`. It NEVER means use engine defaults, approximate a rule, expose hidden state, silently skip data, invent a threshold, claim neutrality, or ship partial behavior.
</critical>

## Status 2026-09-04

> Dated, non-normative addendum. Normative §§1-24 above are preserved byte-for-byte except the inline caveat blockquotes. This section records code truth as observed 2026-09-04 on the uncommitted worktree. `Formal/` is out of scope. `docs/hydra2-human-fetch/` and `docs/hydra2-loop6-suggestions/` are ignored.

### SPEC §-number verification (checkable claims)

- Typed-error rules are SPEC §§3:156-159 exactly: 156 library-raises-typed-errors, 157 safe-identifiers-only, 158 deadline-is-runner-control-flow, 159 corruption-never-silent-fallback. Verified in-file 2026-09-04.
- `§12 Data and Lineage Schemas` header is at line 796 (`### 12.1` at 798); the task note `797?` is off by one — line 797 is blank. Content (packaged/decision/rollout/split/loader, `§§12.1-12.5`) is otherwise as cited.
- Hash-binding-before-cases is SPEC §15:1105 (`CandidateSpec identity binds exact utility bytes ... or raise ContractError before search`) plus factory docstrings: `src/hydra2/search/candidate0.py:235-239` (`All hash fields are bound before cases`), `src/hydra2/search/pbrf.py:1100-1106`, `src/hydra2/search/gumbel.py:1734-1740`.

### A. Synthetic paths hardened to hard errors (teacher/loop/replay/completion, WP-10..WP-12)

- Training loop: `_dummy_digest` removed; `_REQUIRED_MANIFEST_KEYS` (10 digests) at `src/hydra2/training/loop.py:60-71`; missing/empty/malformed raises `ContractError` at `loop.py:294-303` (`manifest_hashes is required`, per-key required, `_require_sha256` at `loop.py:74-77`). Docstring at `loop.py:244-253` states no defaults/fallbacks.
- Replay: identical cutover at `src/hydra2/training/replay.py:67-78` (keys), `replay.py:81-84` (`_require_sha256`), `replay.py:364-373` (all-10-required), plus `dataset must be AuthoritativeParquetDataset` at `replay.py:311` and eval-wall overlap at `replay.py:634`.
- Teacher (canonical `src/hydra2/distillation/teacher.py`, re-exported via PEP 562 at `src/hydra2/distill/teacher.py:10-17`): `_synthetic_digest` and synthetic eligible-gate fallback deleted (diff `distillation/teacher.py` WP-10 block); action-table validation raises at `teacher.py:137-148`; unknown/rejected candidate raises at `teacher.py:250-262`; generation raises `WP-10 blocked` never hash noise at `teacher.py:837-848`; legal-mask/value/seed hardening at `teacher.py:503-518,673-690,844-848`.
- Completion registry: `WP-10` now depends on `WP-09C/D/E` plus `WP-12` at `src/hydra2/completion.py:105-109`; comment at `completion.py:107-108` states missing/ineligible gate raises `ContractError — WP-10 blocked, never synthetic fallback`.
- Residual synthetic allowances are out of the hardened path and still explicit: `require_attestation(allow_synthetic=True)` default at `src/hydra2/data/attestation.py:209-232`, oracle-loader empty-shard synthesize at `src/hydra2/belief/oracle_loader.py:302-307`, DESPOT synthetic world fallback at `src/hydra2/search/despot_natural.py:545-580`. None of these sit behind the WP-05B/WP-10/WP-11 manifest gates.

### B. Candidate0 descriptor authority (gumbel/PBRF/local_resolving/qualification)

- Authority: `make_candidate0_spec` at `src/hydra2/search/candidate0.py:217-356`; RNG/stream/case canonical descriptors at `candidate0.py:294-311` (`counter_based_v1`, `random_stream_v1:["candidate0_tie"]`, empty-manifest case hash).
- Gumbel mirrors authority: `_model_hash_from_identity` via candidate0 import at `src/hydra2/search/gumbel.py:1677-1689`, `_derive_utility_manifest_hash` fail-loud at `gumbel.py:1692-1702`, `_canonical_hashes` verbatim at `gumbel.py:1706-1714`, factory binding note at `gumbel.py:1734-1740`. Determinism fix: salted-sha256 `gumbel_aid_v1` fallback at `gumbel.py:852-858,1364-1369` (replaces `hash()`).
- PBRF mirrors authority: same trio at `src/hydra2/search/pbrf.py:1037-1062` (import/mirror, fail-loud utility) and `pbrf.py:1066-1074` (canonical hashes); factory note at `pbrf.py:1100-1106`; `PLACEHOLDER_*` imports removed (diff `pbrf.py` header).
- Local_resolving mirrors authority: `_model_hash_from_identity` at `src/hydra2/search/local_resolving.py:984-990`, factory note at `local_resolving.py:1044-1045`, `_canonical_hashes`-equivalent RNG block at `local_resolving.py:1147`; `_load_default_hashes` is file-backed-only with `MISSING_HASH` per-key fallback (`local_resolving.py:935-978`).
- Qualification binds the same way and fails closed: `_load_default_hashes_for_spec` at `src/hydra2/analysis/qualification.py:756-814` (file-backed configs + live-model utility/model + candidate0 RNG/stream/case verbatim; derivation failure raises `ContractError` at `qualification.py:784-786,797-799`, never constant hashes per docstring at `qualification.py:762-764`). Gameplay fallback maps candidate ids at `qualification.py:888-919` with `fallback_candidate_id="candidate0"`.
- Confirmation flag note: `local_resolving.py:1300-1304` is the `warm_start` bias branch, not a confirmation gate; the only `confirmation` token in-file is the RNG-purpose comment at `local_resolving.py:1317` (`actor_policy_sample + confirmation`, implemented at `local_resolving.py:1320-1333`). No `confirmation` boolean field exists at that site as of 2026-09-04.

### C. PBRF guarded-imports fail-closed verdict + PolicySet verdict

- Guarded imports at `src/hydra2/search/pbrf.py:44-88` (`_HAS_RANDOM/_HAS_KERNEL/_HAS_BELIEF/_HAS_PACKET/_HAS_TELEMETRY` with `Any` fallback) are import-hygiene only. Load-bearing use is fail-closed: kernel required (`pbrf.py:524-530`), belief+epoch required (`pbrf.py:539`), planner belief required (`pbrf.py:1566`), belief_epoch required (`pbrf.py:1500`), utility-hash derivation raises (`pbrf.py:1062`). Non-load-bearing mirrors (telemetry minimal object at `pbrf.py:1309`, deterministic RNG-from-epoch at `pbrf.py:539-545,667-673`, default `NaturalBelief` adoption at `pbrf.py:1551`) are debug-logged fallbacks confined to planner/test paths, never to forest-commit semantics (commit path at `pbrf.py:1178-1191` per SPEC §16.4).
- `PolicySet` verdict: `src/hydra2/belief/natural.py:86-95` is an explicit WP-07A placeholder (empty provenance tuple, uniform `log_prob → 0.0`); `src/hydra2/belief/kernel.py:153-183` defaults to `PolicySet()` when unset. Packet-kernel likelihood therefore does not evaluate real opponent policies yet. SPEC §14.3 `policy_set` parameter shape is satisfied; policy semantics are future work under §24, not a silent default — callers MUST NOT read the placeholder as a calibrated opponent model.

### D. Pyrefly warn-level hygiene + per-file-ignores rationale + pixi-interpreter pin

- Pyrefly scope is `src` only (`pyproject.toml:132` `project-includes`). 16 codes promoted to `error` (CI-blocking, 319 surfaced: `bad-argument-type` 211 dominant) at `pyproject.toml:148-164`; 10 codes at `warn` (triage, non-blocking, 275 hidden: `implicit-bool` 120 + `unknown-argument-type` 155 dominant) at `pyproject.toml:165-175`; 9 Any-related stay `ignore` (JAX/torch bridge intentional) at `pyproject.toml:176-185`. Warn means hygiene debt to triage per-file, never a gate bypass — errors remain zero-tolerance.
- Per-file-ignores rationale at `pyproject.toml:96-110`: `search`/`tests/search` ignore `E501` (wall/score tables) + `PERF` (noise outside hot path); `tests/**` ignores `E501/S101/N806/PERF` (fixture literals, `S101` test idiom, paper-notation capitals, non-hot path); `eval/runtime/data/contracts/engines/belief` ignore `PERF` so PERF stays advisory for `training`+`models` only (`pyproject.toml:84-88`). `line-length 100` kept (`pyproject.toml:65`).
- Pixi-interpreter pin (root cause, long-term): `python-interpreter-path = ".pixi/envs/default/bin/python"` at `pyproject.toml:146`, with rationale at `pyproject.toml:133-145` — pyrefly auto-discovers empty uv-made `./.venv` instead of the pixi env (64 phantom missing-imports); pin outranks stray `.venv` per discovery order. Upstream `facebook/pyrefly#4432` + PR `#4490` cited in-file; removal condition is vendored `#4490` plus `dump-config` proof. `uv.lock` ban restates sole authority at `pyproject.toml:15-16,145` and `.gitignore:21-22`.

### E. Runtime / pytest / qualification / overfit (new context, relevant only)

- Torch: `torch ==2.14.0` pinned at `pyproject.toml:33`; locked wheel at `pixi.lock:519-521` (`torch-2.14.0-cp312-manylinux_2_28`); live interpreter reports `2.14.0+cu130` (`.pixi/envs/default/bin/python -c 'import torch'`, CUDA string via `src/hydra2/runtime/environment.py:146` `torch.version.cuda`). Pixi is sole authority (`pyproject.toml:15-16`); `uv.lock` absent (banned per above).
- Pytest defaults: `addopts -ra --strict-markers` at `pyproject.toml:114`; markers `gpu/slow/soak` at `pyproject.toml:115-119` (opt-in lanes, never deselected by default); inductor cache `TORCHINDUCTOR_CACHE_DIR` version-keyed default at `tests/conftest.py:31-45` (no other inductor knobs); session `require_cuda` fixture hard-fails without CUDA at `tests/conftest.py:671-676` (no silent CPU fallback).
- Qualification hashes bound to candidate0: see §B (`qualification.py:756-814,817-919`).
- Overfit repeatability proof restored: thresholds `OVERFIT_NLL_THRESHOLD=0.15` / `OVERFIT_TOP1_THRESHOLD=0.90` at `src/hydra2/eval/baseline.py:64-65`; `tiny_shard_overfit` gate raises below threshold at `baseline.py:487-575`; repeatability (same-seed equality, cross-seed variance) asserted at `tests/unit/test_baseline_wp05c.py:122-136`.
- Reported gates 2026-09-04 (worktree truth, main agent revalidates): pyrefly 0 errors, ruff clean, unit 364/0, collect 802. Worktree observed 2026-09-04: `55` porcelain entries (`52` tracked-modified per `git diff --stat`, plus `docs/hydra2-loop6-suggestions/`, `formal/`, `tests/unit/_manifest_helpers.py` untracked); the `51-file` figure in the task note is stale by two — use the observed count at validation time.
