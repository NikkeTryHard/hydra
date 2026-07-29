"""WP-05A model input and spec contracts — SPEC 11.1.

Implements :class:`TensorFieldSpec`, :class:`ModelInputSchema`,
:class:`ModelHeadSpec` and :class:`ModelSpec` with RFC 8785 digest
semantics. Baseline excludes optional shape features (SPEC 11.2); any
arm publishes a new schema identity.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import (
    ContractError,
    DigestText,
    make_digest_text,
)

MODEL_INPUT_SCHEMA_VERSION = "1.0.0"
MODEL_SPEC_SCHEMA_VERSION = "1.0.0"
MODEL_INPUT_ARTIFACT_TYPE = "hydra2.model_input_schema"
MODEL_SPEC_ARTIFACT_TYPE = "hydra2.model_spec"
MODEL_INPUT_RELPATH = Path("configs") / "models" / "model_input_v1.json"

# Canonical JSON safety — 2**53 - 1 is ECMA-262 ``Number.MAX_SAFE_INTEGER`` and
# the RFC 8259 round-trip guarantee for decoder interop. Values outside this
# bound lose integer precision in JS/JSON decoders and are rejected by
# ``_validate_json_value`` below. Single source of truth avoids magic 9007199254740991.
JSON_SAFE_INT_MAX: int = 2**53 - 1

# Baseline bucketing — padded/bucketed histories with explicit ``history_mask``.
# Four power-of-two buckets ``(32, 64, 128, 256)`` are **frozen** by SPEC 11.1;
# changing them is a new schema identity (digest changes). Powers of two
# additionally bound ``torch.compile`` recompilation: each distinct ``T`` would
# recompile up to ``recompile_limit=8`` without ``dynamic=True`` /
# ``isolate_recompiles=True`` (see perf-A §4.1/§8.3 and
# https://docs.pytorch.org/docs/2.13/generated/torch.compile.html). The float
# additive mask in ``models/model.py`` (``_TransformerLayer``) is intentionally
# kept alongside these buckets to guarantee bucket-size invariance — bool-mask
# SDPA showed sequence-length dependent outputs in earlier PyTorch releases
# (ported from perf-A §4.1/§8.2; SDPA bool semantics:
# https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html).
# Device-agnostic: no hardcoded ``sm_120``/``sm_*`` or ``cuda:0`` here; device
# is carried by tensors (``targets.device`` / ``hist_emb.device``).
HISTORY_BUCKET_LENGTHS: tuple[int, ...] = (32, 64, 128, 256)
# Frozen action table v1 size — 6792 legal mahjong actions. Used as canonical
# ``[B, A]`` width for ``legal_mask``/``policy_logits``; tests may inject a
# smaller value for speed but production asserts ``== BASELINE_ACTION_COUNT``
# (see ``model.Hydra2BaselineModel.__init__``). Not a tmpfs/artifact path.
BASELINE_ACTION_COUNT = 6792

# Registry of known identifiers — unknown IDs are rejected.
KNOWN_ARCHITECTURES: frozenset[str] = frozenset(
    {
        "hydra2_baseline_transformer_v1",
        "hydra2_baseline_mlp_v1",
    }
)
KNOWN_LOSS_IDS: frozenset[str] = frozenset(
    {
        "masked_cross_entropy",
        "cross_entropy",
        "mse",
        "cross_entropy_4x4",
    }
)
KNOWN_HEAD_IDS: frozenset[str] = frozenset(
    {
        "belief_next",
        "event_next",
        "placement",
        "policy",
        "value",
    }
)

__all__ = [
    "BASELINE_ACTION_COUNT",
    "HISTORY_BUCKET_LENGTHS",
    "KNOWN_ARCHITECTURES",
    "KNOWN_HEAD_IDS",
    "KNOWN_LOSS_IDS",
    "MODEL_INPUT_ARTIFACT_TYPE",
    "MODEL_INPUT_RELPATH",
    "MODEL_INPUT_SCHEMA_VERSION",
    "MODEL_SPEC_ARTIFACT_TYPE",
    "MODEL_SPEC_SCHEMA_VERSION",
    "ModelHeadSpec",
    "ModelInputSchema",
    "ModelSpec",
    "TensorFieldSpec",
    "build_model_input_schema_envelope",
    "build_model_input_schema_payload",
    "compute_model_input_schema_digest",
    "compute_model_spec_digest",
    "model_input_schema_digest",
    "model_spec_digest_document",
]


@dataclass(frozen=True, slots=True)
class TensorFieldSpec:
    """One tensor field in the versioned input schema (SPEC 11.1)."""

    name: str
    dtype: Literal["bool", "int32", "int64", "float32"]
    shape: tuple[str | int, ...]
    padding_value: int | float | bool | None
    valid_min: float | None
    valid_max: float | None
    mask_field: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name == "":
            raise ContractError(f"TensorFieldSpec name invalid {self.name!r}")
        if self.dtype not in ("bool", "int32", "int64", "float32"):
            raise ContractError(f"dtype {self.dtype!r} unknown")
        if not isinstance(self.shape, tuple) or len(self.shape) == 0:
            raise ContractError(f"shape {self.shape!r} must be non-empty tuple")
        for dim in self.shape:
            if isinstance(dim, int):
                if dim <= 0:
                    raise ContractError(f"shape dim {dim!r} must be positive")
            elif isinstance(dim, str):
                if dim not in ("B", "T", "A", "F"):
                    raise ContractError(f"shape symbol {dim!r} must be B,T,A,F")
            else:
                raise ContractError(f"shape dim {dim!r} invalid")
        if self.mask_field is not None and not isinstance(self.mask_field, str):
            raise ContractError("mask_field must be str or None")


@dataclass(frozen=True, slots=True)
class ModelInputSchema:
    """Versioned input-schema binding (SPEC 11.1)."""

    schema_version: str
    fields: tuple[TensorFieldSpec, ...]
    history_bucket_lengths: tuple[int, ...]
    action_count: int
    digest: DigestText

    def __post_init__(self) -> None:
        _digest: DigestText = make_digest_text(self.digest)
        if len(self.history_bucket_lengths) == 0:
            raise ContractError("history_bucket_lengths must be non-empty")
        if sorted(self.history_bucket_lengths) != list(self.history_bucket_lengths):
            raise ContractError("history_bucket_lengths must be strictly increasing")
        if any(b <= 0 for b in self.history_bucket_lengths):
            raise ContractError("bucket lengths must be positive")
        if self.action_count <= 0:
            raise ContractError("action_count must be positive")
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ContractError(f"field names must be unique, got duplicates in {names}")
        # Verify canonical order — alphabetical for baseline v1.
        if names != sorted(names):
            raise ContractError("fields must be in canonical alphabetical order")
        # Ensure mask fields refer to existing boolean fields.
        field_map = {f.name: f for f in self.fields}
        for f in self.fields:
            if f.mask_field is not None and f.mask_field not in field_map:
                raise ContractError(f"mask_field {f.mask_field!r} not in schema")
            if f.mask_field is not None and field_map[f.mask_field].dtype != "bool":
                raise ContractError(f"mask_field {f.mask_field!r} must be bool dtype")
        computed = compute_model_input_schema_digest(
            build_model_input_schema_payload_without_digest(self)
        )
        if make_digest_text(self.digest) != computed:
            raise ContractError(f"ModelInputSchema digest mismatch {self.digest} != {computed}")


@dataclass(frozen=True, slots=True)
class ModelHeadSpec:
    """Head descriptor — output key, target and loss (SPEC 11.1)."""

    head_id: str
    output_key: str
    target_id: str
    loss_id: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.head_id not in KNOWN_HEAD_IDS:
            raise ContractError(f"unknown head_id {self.head_id!r}")
        if self.output_key == "":
            raise ContractError("output_key must be non-empty")
        if self.target_id == "":
            raise ContractError("target_id must be non-empty")
        if self.loss_id not in KNOWN_LOSS_IDS:
            raise ContractError(f"unknown loss_id {self.loss_id!r}")
        # Parameters must be JSON domain (finite numbers, string keys).
        _validate_json_value(dict(self.parameters), where=f"head {self.head_id} parameters")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Model identity binding (SPEC 11.1)."""

    schema_version: str
    input_schema_hash: DigestText
    feature_derivation_hash: DigestText
    architecture_id: str
    architecture_parameters: Mapping[str, Any]
    head_specs: tuple[ModelHeadSpec, ...]
    action_table_hash: DigestText
    observation_schema_hash: DigestText
    utility_manifest_hash: DigestText
    digest: DigestText

    def __post_init__(self) -> None:
        _sha1: DigestText = make_digest_text(self.input_schema_hash)
        _sha2: DigestText = make_digest_text(self.feature_derivation_hash)
        _sha3: DigestText = make_digest_text(self.action_table_hash)
        _sha4: DigestText = make_digest_text(self.observation_schema_hash)
        _sha5: DigestText = make_digest_text(self.utility_manifest_hash)
        _sha6: DigestText = make_digest_text(self.digest)
        if self.architecture_id not in KNOWN_ARCHITECTURES:
            raise ContractError(f"unknown architecture_id {self.architecture_id!r}")
        # Head specs sorted by head_id.
        ids = [h.head_id for h in self.head_specs]
        if ids != sorted(ids):
            raise ContractError("head_specs must be sorted by head_id")
        if len(ids) != len(set(ids)):
            raise ContractError("head_ids must be unique")
        _validate_json_value(dict(self.architecture_parameters), where="architecture_parameters")
        computed = compute_model_spec_digest(model_spec_digest_document(self))
        if make_digest_text(self.digest) != computed:
            raise ContractError(f"ModelSpec digest mismatch {self.digest} != {computed}")


def _validate_json_value(value: Any, *, where: str) -> None:
    """Validate JSON-domain value (finite, safe-int, str-keyed objects only).

    Bound is ``JSON_SAFE_INT_MAX`` (2**53 - 1) per ECMA-262 / RFC 8259 so that
    payloads round-trip through any JSON decoder without integer precision loss.
    No hardcoded paths or device strings are consulted here; digest derivation
    is device-agnostic.
    """
    import math as _math

    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > JSON_SAFE_INT_MAX:
            raise ContractError(
                f"{where}: integer {value} exceeds safe range (> JSON_SAFE_INT_MAX)"
            )
        return
    if isinstance(value, float):
        if not _math.isfinite(value):
            raise ContractError(f"{where}: non-finite {value!r}")
        return
    if isinstance(value, str):
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):  # type: ignore[unknown-variable-type]
            _validate_json_value(item, where=f"{where}[{i}]")  # pyrefly: ignore[unknown-argument-type]  # value is Any, item inferred as Any
        return
    if isinstance(value, Mapping):
        for key, item in value.items():  # type: ignore[unknown-variable-type]
            if not isinstance(key, str):
                raise ContractError(f"{where}: object keys must be strings")
            _validate_json_value(item, where=f"{where}.{key}")  # pyrefly: ignore[unknown-argument-type]  # Mapping value is Any
        return
    raise ContractError(f"{where}: type {type(value).__name__} outside JSON domain")


# ---------------------------------------------------------------------------
# Canonical field table for model_input_v1 — alphabetical, frozen.
# ---------------------------------------------------------------------------


def _baseline_fields() -> tuple[TensorFieldSpec, ...]:
    fields: list[TensorFieldSpec] = [
        TensorFieldSpec(
            name="actor",
            dtype="int64",
            shape=("B",),
            padding_value=None,
            valid_min=0,
            valid_max=3,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="actor_can_riichi",
            dtype="bool",
            shape=("B",),
            padding_value=None,
            valid_min=None,
            valid_max=None,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="actor_can_tsumo",
            dtype="bool",
            shape=("B",),
            padding_value=None,
            valid_min=None,
            valid_max=None,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="actor_furiten",
            dtype="int64",
            shape=("B",),
            padding_value=None,
            valid_min=0,
            valid_max=3,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="actor_seats",
            dtype="int64",
            shape=("B",),
            padding_value=None,
            valid_min=0,
            valid_max=3,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="concealed_hand_counts",
            dtype="int32",
            shape=("B", 34),
            padding_value=None,
            valid_min=0,
            valid_max=4,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="dealer",
            dtype="int64",
            shape=("B",),
            padding_value=None,
            valid_min=0,
            valid_max=3,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="dora_indicators",
            dtype="int32",
            shape=("B", 5),
            padding_value=-1,
            valid_min=-1,
            valid_max=135,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="hand_number",
            dtype="int32",
            shape=("B",),
            padding_value=None,
            valid_min=0,
            valid_max=None,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="history_event_kind",
            dtype="int64",
            shape=("B", "T"),
            padding_value=0,
            valid_min=0,
            valid_max=19,
            mask_field="history_mask",
        ),
        TensorFieldSpec(
            name="history_mask",
            dtype="bool",
            shape=("B", "T"),
            padding_value=False,
            valid_min=None,
            valid_max=None,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="honba",
            dtype="int32",
            shape=("B",),
            padding_value=None,
            valid_min=0,
            valid_max=None,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="ippatsu_active",
            dtype="bool",
            shape=("B", 4),
            padding_value=None,
            valid_min=None,
            valid_max=None,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="kan_count",
            dtype="int32",
            shape=("B",),
            padding_value=None,
            valid_min=0,
            valid_max=4,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="legal_mask",
            dtype="bool",
            shape=("B", "A"),
            padding_value=False,
            valid_min=None,
            valid_max=None,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="live_wall_tiles_remaining",
            dtype="int32",
            shape=("B",),
            padding_value=None,
            valid_min=0,
            valid_max=None,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="own_drawn_tile",
            dtype="int32",
            shape=("B",),
            padding_value=-1,
            valid_min=-1,
            valid_max=135,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="phase",
            dtype="int64",
            shape=("B",),
            padding_value=None,
            valid_min=0,
            valid_max=5,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="riichi_states",
            dtype="int64",
            shape=("B", 4),
            padding_value=None,
            valid_min=0,
            valid_max=2,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="riichi_sticks",
            dtype="int32",
            shape=("B",),
            padding_value=None,
            valid_min=0,
            valid_max=None,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="round_index",
            dtype="int32",
            shape=("B",),
            padding_value=None,
            valid_min=0,
            valid_max=None,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="round_wind",
            dtype="int64",
            shape=("B",),
            padding_value=None,
            valid_min=0,
            valid_max=3,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="scores",
            dtype="int32",
            shape=("B", 4),
            padding_value=None,
            valid_min=-(10**9),
            valid_max=10**9,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="seat_winds",
            dtype="int64",
            shape=("B", 4),
            padding_value=None,
            valid_min=0,
            valid_max=3,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="turn_actor",
            dtype="int64",
            shape=("B",),
            padding_value=None,
            valid_min=0,
            valid_max=3,
            mask_field=None,
        ),
        TensorFieldSpec(
            name="visible_discards_counts",
            dtype="int32",
            shape=("B", 34),
            padding_value=None,
            valid_min=0,
            valid_max=4,
            mask_field=None,
        ),
    ]
    # Already alphabetical — verify.
    assert [f.name for f in fields] == sorted(f.name for f in fields)
    return tuple(fields)


_BASELINE_FIELDS = _baseline_fields()


def build_model_input_schema_payload_without_digest(
    schema: ModelInputSchema | None = None,
) -> dict[str, Any]:
    fields = _BASELINE_FIELDS if schema is None else schema.fields
    buckets = HISTORY_BUCKET_LENGTHS if schema is None else schema.history_bucket_lengths
    action_count = BASELINE_ACTION_COUNT if schema is None else schema.action_count
    return {
        "schema_version": MODEL_INPUT_SCHEMA_VERSION,
        "fields": [
            {
                "name": f.name,
                "dtype": f.dtype,
                "shape": list(f.shape),
                "padding_value": f.padding_value,
                "valid_min": f.valid_min,
                "valid_max": f.valid_max,
                "mask_field": f.mask_field,
            }
            for f in fields
        ],
        "history_bucket_lengths": list(buckets),
        "action_count": action_count,
    }


def build_model_input_schema_payload() -> dict[str, Any]:
    payload = build_model_input_schema_payload_without_digest()
    payload["digest"] = compute_model_input_schema_digest(payload)
    return payload


def compute_model_input_schema_digest(payload_without_digest: Mapping[str, Any]) -> DigestText:
    identity = canonical_bytes(dict(payload_without_digest))
    return DigestText("sha256:" + hashlib.sha256(identity).hexdigest())


def model_input_schema_digest() -> DigestText:
    return compute_model_input_schema_digest(build_model_input_schema_payload_without_digest())


def build_model_input_schema_envelope() -> dict[str, Any]:
    payload = build_model_input_schema_payload()
    return {
        "artifact_type": MODEL_INPUT_ARTIFACT_TYPE,
        "compatibility": "exact",
        "payload": payload,
        "schema_version": MODEL_INPUT_SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# ModelSpec helpers
# ---------------------------------------------------------------------------


def _default_head_specs() -> tuple[ModelHeadSpec, ...]:
    heads = (
        ModelHeadSpec(
            head_id="belief_next",
            output_key="belief_logits",
            target_id="next_event_kind",
            loss_id="cross_entropy",
            parameters={"num_classes": 20, "mask_field": None},
        ),
        ModelHeadSpec(
            head_id="event_next",
            output_key="event_logits",
            target_id="next_event_kind",
            loss_id="cross_entropy",
            parameters={"num_classes": 20, "mask_field": None},
        ),
        ModelHeadSpec(
            head_id="placement",
            output_key="placement_logits",
            target_id="final_placement",
            loss_id="cross_entropy_4x4",
            parameters={"seats": 4, "ranks": 4},
        ),
        ModelHeadSpec(
            head_id="policy",
            output_key="policy_logits",
            target_id="selected_action",
            loss_id="masked_cross_entropy",
            parameters={"mask_field": "legal_mask"},
        ),
        ModelHeadSpec(
            head_id="value",
            output_key="value_vector",
            target_id="utility_vector",
            loss_id="mse",
            parameters={"seats": 4},
        ),
    )
    return tuple(sorted(heads, key=lambda h: h.head_id))


def model_spec_digest_document(spec: ModelSpec | Mapping[str, Any]) -> dict[str, Any]:
    def get(name: str) -> Any:
        if isinstance(spec, Mapping):
            return spec[name]
        return getattr(spec, name)

    head_specs = get("head_specs")
    if isinstance(head_specs, (list, tuple)):
        serialized_heads = []
        for head in head_specs:
            if isinstance(head, Mapping):
                serialized_heads.append(dict(head))
            else:
                serialized_heads.append(
                    {
                        "head_id": head.head_id,
                        "output_key": head.output_key,
                        "target_id": head.target_id,
                        "loss_id": head.loss_id,
                        "parameters": dict(head.parameters),  # pyrefly: ignore[unknown-argument-type]  # head is Any from Mapping lookup
                    }
                )
        serialized_heads = sorted(serialized_heads, key=lambda h: h["head_id"])  # pyrefly: ignore[unknown-argument-type]  # h is dict[Any]
    else:
        serialized_heads = head_specs

    arch_params = get("architecture_parameters")
    if isinstance(arch_params, Mapping):
        arch_params = dict(arch_params)

    return {
        "action_table_hash": get("action_table_hash"),
        "architecture_id": get("architecture_id"),
        "architecture_parameters": arch_params,
        "feature_derivation_hash": get("feature_derivation_hash"),
        "head_specs": serialized_heads,
        "input_schema_hash": get("input_schema_hash"),
        "observation_schema_hash": get("observation_schema_hash"),
        "schema_version": get("schema_version"),
        "utility_manifest_hash": get("utility_manifest_hash"),
    }


def compute_model_spec_digest(document_without_digest: Mapping[str, Any]) -> DigestText:
    identity = canonical_bytes(dict(document_without_digest))
    return DigestText("sha256:" + hashlib.sha256(identity).hexdigest())


def _feature_derivation_hash() -> DigestText:
    # Derivation binds encoder source and schema field names.
    # Use hash of sorted field names + schema digest + encoder stub id.
    content = {
        "encoder_id": "hydra2.models.encoder.v1",
        "field_names": [f.name for f in _BASELINE_FIELDS],
        "input_schema_digest": model_input_schema_digest(),
    }
    return DigestText("sha256:" + hashlib.sha256(canonical_bytes(content)).hexdigest())
