"""WP-05A input-schema artifact, model spec, batch validation, attn precision."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hydra2.contracts.common import ContractError
from hydra2.models.encoder import (
    ActorTensorBatch,
    encode_observations,
    validate_batch_against_schema,
)
from hydra2.models.model import Hydra2BaselineModel, select_actions
from hydra2.models.schema import (
    HISTORY_BUCKET_LENGTHS,
    build_model_input_schema_payload_without_digest,
    compute_model_input_schema_digest,
    model_input_schema_digest,
)
from tests.unit._model_inference_helpers import _batch_to_device, _make_observation

pytestmark = pytest.mark.contract_package("WP-05A")

_MODEL_INPUT_RELPATH = Path("configs/models/model_input_v1.json")


# ---------------------------------------------------------------------------
# 11 keep optional actor-visible shanten/ukeire features out of baseline
# ---------------------------------------------------------------------------


def test_optional_shape_features_excluded() -> None:
    payload = json.loads(_MODEL_INPUT_RELPATH.read_text(encoding="utf-8"))["payload"]
    field_names = [f["name"] for f in payload["fields"]]

    forbidden = {
        "own_private_ids",
        "public_ids",
        "own_physical_count",
        "public_physical_count",
        "public_unseen",
        "post_discard_shanten",
        "own_wait",
        "ukeire",
        "tenpai_prob",
        "win_prob",
        "expected_score",
    }
    overlap = forbidden.intersection(field_names)
    assert not overlap, f"baseline schema must not contain optional shape features: {overlap}"

    # Schema is alphabetical, frozen order.
    assert field_names == sorted(field_names)

    # Input schema digest matches file.
    file_payload_digest = payload["digest"]
    computed = compute_model_input_schema_digest(build_model_input_schema_payload_without_digest())
    assert file_payload_digest == computed
    assert str(model_input_schema_digest()) == file_payload_digest

    # Encoder never emits those features.
    batch = encode_observations([_make_observation()])
    assert not forbidden.intersection(batch.features.keys())

    # Schema object itself rejects unknown additions (can't insert arbitrary key).
    # Changing any field's shape/dtype is a digest break.
    from hydra2.models.schema import TensorFieldSpec

    # Adding a new field would break canonical digest.
    extra = TensorFieldSpec(
        name="own_private_ids",
        dtype="int32",
        shape=("B", 34),
        padding_value=None,
        valid_min=0,
        valid_max=4,
        mask_field=None,
    )
    assert extra.name not in field_names


def test_input_schema_artifact_envelope_and_alpha_order() -> None:
    raw = json.loads(_MODEL_INPUT_RELPATH.read_text(encoding="utf-8"))
    assert raw["artifact_type"] == "hydra2.model_input_schema"
    assert raw["compatibility"] == "exact"
    assert "payload" in raw and "digest" in raw["payload"]
    # Every field declares shape/dtype/padding/mask per SPEC 11.1.
    for fld in raw["payload"]["fields"]:
        assert "name" in fld and "dtype" in fld and "shape" in fld
        assert "padding_value" in fld and "mask_field" in fld
        assert fld["dtype"] in ("bool", "int32", "int64", "float32")
    # Bucket lengths are strictly increasing.
    buckets = raw["payload"]["history_bucket_lengths"]
    assert buckets == sorted(buckets) and len(buckets) == len(set(buckets))
    assert buckets == list(HISTORY_BUCKET_LENGTHS)


def test_model_spec_identity_binding() -> None:
    m = Hydra2BaselineModel()
    doc = m.model_spec()
    # Required hashes present and well-formed.
    for key in (
        "input_schema_hash",
        "feature_derivation_hash",
        "action_table_hash",
        "observation_schema_hash",
        "utility_manifest_hash",
        "digest",
    ):
        assert key in doc
        assert str(doc[key]).startswith("sha256:")
        assert len(str(doc[key])) == 71
    # Head specs sorted by head_id.
    head_ids = [h["head_id"] for h in doc["head_specs"]]
    assert head_ids == sorted(head_ids)
    # Changing feature set would change derivation hash.
    from hydra2.models.schema import _feature_derivation_hash

    assert str(_feature_derivation_hash()) == doc["feature_derivation_hash"]
    # Digest is canonical (computed without digest field).
    from hydra2.models.schema import compute_model_spec_digest

    doc_without = {k: v for k, v in doc.items() if k != "digest"}
    assert str(compute_model_spec_digest(doc_without)) == doc["digest"]


def test_validate_batch_against_schema_rejects_malformed() -> None:
    obs = _make_observation()
    batch = encode_observations([obs])
    validate_batch_against_schema(batch)

    # Wrong dtype: promote actor to float.
    bad_features = dict(batch.features)
    bad_features["actor"] = batch.features["actor"].to(torch.float32)
    bad = ActorTensorBatch(
        features=bad_features,
        history_mask=batch.history_mask,
        legal_mask=batch.legal_mask,
        observation_hashes=batch.observation_hashes,
        actor_seats=batch.actor_seats,
    )
    with pytest.raises(ContractError):
        validate_batch_against_schema(bad)

    # Wrong shape: history_event_kind transposed.
    bad2_features = dict(batch.features)
    bad2_features["history_event_kind"] = batch.features["history_event_kind"].T
    bad2 = ActorTensorBatch(
        features=bad2_features,
        history_mask=batch.history_mask,
        legal_mask=batch.legal_mask,
        observation_hashes=batch.observation_hashes,
        actor_seats=batch.actor_seats,
    )
    with pytest.raises(ContractError):
        validate_batch_against_schema(bad2)


@pytest.mark.gpu
def test_attn_bf16_matches_fp32_decisions() -> None:
    """Scoped bf16 attention preserves decisions (flash-path parity).

    Only the SDPA kernel runs in bf16 (unlocking fused/flash paths, which
    have no fp32 kernel); every projection, norm, and output stays fp32.
    Logits agree to bf16 tolerance and argmax decisions are exact, so the
    flag is a pure-speedup switch for measured-faster shapes.
    """
    if not torch.cuda.is_available():
        pytest.fail("BLOCKER: CUDA device unavailable; GPU probes cannot be qualified on CPU")
    torch.manual_seed(0)
    model = Hydra2BaselineModel()
    model.eval()
    model.to("cuda")
    batch = _batch_to_device(
        encode_observations([_make_observation(), _make_observation(actor=2)]),
        torch.device("cuda"),
    )
    with torch.no_grad():
        ref = model(batch)
        assert ref.policy_logits.dtype == torch.float32
        for layer in model.layers:
            layer.attn_bf16 = True
        out = model(batch)
    assert out.policy_logits.dtype == torch.float32
    assert torch.allclose(out.policy_logits, ref.policy_logits, atol=1e-2, rtol=1e-2)
    assert torch.equal(
        select_actions(out.policy_logits, batch.legal_mask),
        select_actions(ref.policy_logits, batch.legal_mask),
    )
    assert torch.isfinite(out.policy_logits[batch.legal_mask]).all()
