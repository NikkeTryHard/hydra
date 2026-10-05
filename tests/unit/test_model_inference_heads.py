"""WP-05A dense policy, four-seat value, and event/belief heads."""

from __future__ import annotations

import pytest
import torch

from hydra2.contracts.event_vocab import EVENT_KINDS
from hydra2.models.encoder import encode_observations
from hydra2.models.model import Hydra2BaselineModel
from hydra2.models.schema import BASELINE_ACTION_COUNT
from tests.unit._model_inference_helpers import _batch_to_device, _make_observation

pytestmark = pytest.mark.contract_package("WP-05A")

# ---------------------------------------------------------------------------
# 5 dense legal policy head
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "precision",
    [
        pytest.param("fp32", id="fp32"),
        pytest.param("bf16", id="bf16", marks=pytest.mark.gpu),
    ],
)
def test_dense_legal_policy_head(precision: str) -> None:
    if precision == "bf16":
        # CUDA-only numerics: same weights, fp32 reference vs bf16 autocast.
        if not torch.cuda.is_available():
            pytest.fail("BLOCKER: CUDA device unavailable; GPU probes cannot be qualified on CPU")
        torch.manual_seed(0)
        model = Hydra2BaselineModel()
        model.eval()
        model.to("cuda")
        batch = _batch_to_device(encode_observations([_make_observation()]), torch.device("cuda"))
        with torch.no_grad():
            ref = model.evaluate(batch)
            assert ref.policy_logits.dtype == torch.float32
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.evaluate(batch)
        # Logits are dense bf16 covering every action; masking happens outside.
        assert out.policy_logits.dtype == torch.bfloat16
        assert out.policy_logits.shape[-1] == BASELINE_ACTION_COUNT
        assert torch.allclose(
            out.policy_logits.float(), ref.policy_logits.float(), atol=1e-2, rtol=1e-2
        )
        # Legal entries remain finite after masked fill.
        illegal = ~batch.legal_mask
        masked = out.policy_logits.float().masked_fill(illegal, float("-inf"))
        # At least one finite per row (since at least one legal).
        assert torch.isfinite(masked[batch.legal_mask]).all()
        return
    model = Hydra2BaselineModel()
    assert model.policy_head.out_features == BASELINE_ACTION_COUNT
    assert model.policy_head.in_features == model.d_model * 2

    batch = encode_observations([_make_observation()])
    out = model.evaluate(batch)
    # Logits are dense floats covering every action; masking happens outside.
    assert out.policy_logits.dtype == torch.float32
    assert out.policy_logits.shape[-1] == BASELINE_ACTION_COUNT
    # Legal entries remain finite after masked fill.
    illegal = ~batch.legal_mask
    masked = out.policy_logits.masked_fill(illegal, float("-inf"))
    # At least one finite per row (since at least one legal).
    assert torch.isfinite(masked[batch.legal_mask]).all()


# ---------------------------------------------------------------------------
# 6 four-seat value distribution/vector head
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "precision",
    [
        pytest.param("fp32", id="fp32"),
        pytest.param("bf16", id="bf16", marks=pytest.mark.gpu),
    ],
)
def test_four_seat_value_distribution_vector_head(precision: str) -> None:
    if precision == "bf16":
        # CUDA-only numerics: same weights, fp32 reference vs bf16 autocast.
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
            ref = model.evaluate(batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.evaluate(batch)
        # Both placement and value retain four seats.
        assert out.placement_logits.shape == (2, 4, 4)
        assert out.value_vector.shape == (2, 4)
        # Value is under named utility; check identity binding.
        assert out.utility_id == model.utility_id
        assert out.utility_manifest_hash == model.utility_manifest_hash
        assert out.model_identity == model.model_identity
        # Placement rows are per-seat rank logits; value vector is per-seat.
        assert out.placement_logits.dtype == torch.bfloat16
        assert out.value_vector.dtype == torch.bfloat16
        assert torch.allclose(
            out.placement_logits.float(), ref.placement_logits.float(), atol=1e-2, rtol=1e-2
        )
        assert torch.allclose(
            out.value_vector.float(), ref.value_vector.float(), atol=1.5e-1, rtol=2e-2
        )
        return
    model = Hydra2BaselineModel()
    batch = encode_observations([_make_observation(), _make_observation(actor=2)])
    out = model.evaluate(batch)
    # Both placement and value retain four seats.
    assert out.placement_logits.shape == (2, 4, 4)
    assert out.value_vector.shape == (2, 4)
    # Value is under named utility; check identity binding.
    assert out.utility_id == model.utility_id
    assert out.utility_manifest_hash == model.utility_manifest_hash
    assert out.model_identity == model.model_identity
    # Placement rows are per-seat rank logits; value vector is per-seat.
    assert out.placement_logits.dtype == torch.float32
    assert out.value_vector.dtype == torch.float32


# ---------------------------------------------------------------------------
# 7 event likelihood heads required by belief model
# ---------------------------------------------------------------------------


def test_event_likelihood_heads_for_belief() -> None:
    model = Hydra2BaselineModel()
    batch = encode_observations([_make_observation()])
    out = model.evaluate(batch)
    # Both heads present, keyed as next_event, with correct width.
    assert "next_event" in out.event_logits
    assert "next_event" in out.belief_logits
    assert out.event_logits["next_event"].shape == (1, len(EVENT_KINDS))
    assert out.belief_logits["next_event"].shape == (1, len(EVENT_KINDS))
    # They are distinct dense logits (different params).
    assert not torch.equal(out.event_logits["next_event"], out.belief_logits["next_event"])
