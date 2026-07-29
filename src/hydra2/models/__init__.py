"""Hydra2 model package — encoder, schema, baseline model."""

from __future__ import annotations

from hydra2.models.encoder import (
    ActorTensorBatch,
    bucket_for_length,
    encode_observations,
    validate_batch_against_schema,
)
from hydra2.models.model import Hydra2BaselineModel, ModelOutput, masked_policy, select_actions
from hydra2.models.schema import (
    ModelHeadSpec,
    ModelInputSchema,
    ModelSpec,
    TensorFieldSpec,
    build_model_input_schema_payload,
    model_input_schema_digest,
)

__all__ = [
    "ActorTensorBatch",
    "Hydra2BaselineModel",
    "ModelHeadSpec",
    "ModelInputSchema",
    "ModelOutput",
    "ModelSpec",
    "TensorFieldSpec",
    "bucket_for_length",
    "build_model_input_schema_payload",
    "encode_observations",
    "masked_policy",
    "model_input_schema_digest",
    "select_actions",
    "validate_batch_against_schema",
]
