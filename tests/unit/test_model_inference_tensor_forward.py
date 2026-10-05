"""WP-05A shared tensor_forward entry: parity, grads, slices, identity."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from hydra2.contracts.event_vocab import EVENT_KINDS
from hydra2.models.encoder import ActorTensorBatch, encode_observations
from hydra2.models.model import Hydra2BaselineModel
from hydra2.models.schema import BASELINE_ACTION_COUNT
from tests.unit._model_inference_helpers import (
    _batch_to_device,
    _history_of_length,
    _make_observation,
)

pytestmark = pytest.mark.contract_package("WP-05A")

# ---------------------------------------------------------------------------
# 13 shared tensor_forward entry (dense/packed parity, grads, slices, identity)
# ---------------------------------------------------------------------------


def _packed_from_batch(batch: ActorTensorBatch) -> Any:
    """Pack real history prefixes from an encoded batch (pull order)."""
    from hydra2.models.encoder import PackedHistories

    kind = batch.features["history_event_kind"]
    mask = batch.history_mask
    lengths = [int(v) for v in mask.sum(dim=1).tolist()]
    bounds = [0]
    for ln in lengths:
        bounds.append(bounds[-1] + ln)
    return PackedHistories(
        packed_kind=kind[mask],
        cu_seqlens=torch.tensor(bounds, dtype=torch.int32),
        row_lengths=tuple(lengths),
        max_len=max(lengths) if lengths else 0,
    )


def _batch_with_packed(batch: ActorTensorBatch, packed: Any) -> ActorTensorBatch:
    return ActorTensorBatch(
        features=batch.features,
        history_mask=batch.history_mask,
        legal_mask=batch.legal_mask,
        observation_hashes=batch.observation_hashes,
        actor_seats=batch.actor_seats,
        packed=packed,
    )


def test_tensor_forward_matches_evaluate_dense() -> None:
    """Shared entry vs evaluate: bitwise parity over real planes across B/T."""
    torch.manual_seed(0)
    model = Hydra2BaselineModel()
    model.eval()
    n_events: int = len(EVENT_KINDS)
    cases = [
        [_make_observation(history=())],
        [_make_observation(history=_history_of_length(5)), _make_observation(history=())],
        [
            _make_observation(actor=0, history=_history_of_length(1)),
            _make_observation(actor=2, history=_history_of_length(7)),
            _make_observation(actor=3, history=_history_of_length(10)),
        ],
    ]
    for obs_list in cases:
        batch = encode_observations(list(obs_list))
        with torch.no_grad():
            out = model.evaluate(batch)
            policy, fused = model.tensor_forward(
                batch.features, batch.history_mask, getattr(batch, "packed", None)
            )
        batch_size = batch.history_mask.shape[0]
        assert fused.shape == (batch_size, 20 + 2 * n_events)
        assert torch.equal(policy, out.policy_logits)
        assert torch.equal(
            fused[..., 0:16].contiguous().view(batch_size, 4, 4), out.placement_logits
        )
        assert torch.equal(fused[..., 16:20], out.value_vector)
        assert torch.equal(fused[..., 20 : 20 + n_events], out.event_logits["next_event"])
        assert torch.equal(
            fused[..., 20 + n_events : 20 + 2 * n_events], out.belief_logits["next_event"]
        )


def test_tensor_forward_empty_packed_matches_evaluate() -> None:
    """Empty-packed entry vs packed evaluate: bitwise parity on CPU.

    All-empty histories pack to total 0, so the trunk skips packed attention
    and renders zero-vector pooling without touching CUDA kernels.
    """
    torch.manual_seed(0)
    model = Hydra2BaselineModel()
    model.eval()
    batch = encode_observations([_make_observation(history=()), _make_observation(history=())])
    packed = _packed_from_batch(batch)
    assert int(packed.packed_kind.numel()) == 0
    batch_p = _batch_with_packed(batch, packed)
    with torch.no_grad():
        out = model.evaluate(batch_p)
        policy, fused = model.tensor_forward(batch.features, batch.history_mask, packed)
    assert torch.equal(policy, out.policy_logits)
    assert torch.equal(fused[..., 16:20], out.value_vector)


@pytest.mark.gpu
@pytest.mark.serial
def test_tensor_forward_nonempty_packed_matches_evaluate(require_cuda: torch.device) -> None:
    """Non-empty packed entry vs packed evaluate: bitwise parity on CUDA."""
    _ = require_cuda  # hard-fails without CUDA; silent CPU fallback is forbidden.
    torch.manual_seed(0)
    model = Hydra2BaselineModel()
    model.eval()
    model.to("cuda")
    device = torch.device("cuda")
    batch = _batch_to_device(
        encode_observations(
            [
                _make_observation(history=_history_of_length(5)),
                _make_observation(actor=1, history=_history_of_length(9)),
            ]
        ),
        device,
    )
    packed = _packed_from_batch(batch)
    batch_p = _batch_with_packed(batch, packed)
    with torch.no_grad():
        out = model.evaluate(batch_p)
        policy, fused = model.tensor_forward(batch.features, batch.history_mask, packed)
    # Same packed path on both sides: identical ops over identical inputs.
    assert torch.equal(policy, out.policy_logits)
    assert torch.equal(fused[..., 16:20], out.value_vector)


def test_tensor_forward_gradients_slices_identity() -> None:
    """Grad-enabled entry: evaluate-identical grads, slices, and identity."""
    torch.manual_seed(0)
    model_a = Hydra2BaselineModel()
    model_a.eval()
    torch.manual_seed(0)
    model_b = Hydra2BaselineModel()
    model_b.eval()
    assert model_a.model_identity == model_b.model_identity
    batch = encode_observations(
        [
            _make_observation(history=_history_of_length(3)),
            _make_observation(actor=1, history=_history_of_length(5)),
        ]
    )
    out = model_a.evaluate(batch)
    loss_a = out.policy_logits.sum() + out.value_vector.sum() + out.placement_logits.sum()
    model_a.zero_grad()
    loss_a.backward()
    policy, fused = model_b.tensor_forward(batch.features, batch.history_mask, None)
    batch_size = batch.history_mask.shape[0]
    placement = fused[..., 0:16].contiguous().view(batch_size, 4, 4)
    value = fused[..., 16:20]
    assert torch.equal(placement, out.placement_logits)
    assert torch.equal(value, out.value_vector)
    loss_b = policy.sum() + value.sum() + placement.sum()
    model_b.zero_grad()
    loss_b.backward()
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert param_a.grad is not None and param_b.grad is not None
        assert torch.isfinite(param_a.grad).all() and torch.isfinite(param_b.grad).all()
        assert torch.equal(param_a.grad, param_b.grad)
    assert model_a.policy_head.weight.grad is not None
    assert bool((model_a.policy_head.weight.grad != 0).any().item())
    assert model_a.small_heads.weight.grad is not None
    assert bool((model_a.small_heads.weight.grad != 0).any().item())
    # Identity survives shared-entry calls; supervised adapter path unchanged.
    assert model_a.model_identity == model_b.model_identity
    from hydra2.training.adapters import model_output_to_loss_dict

    loss_dict = model_output_to_loss_dict(out)
    assert loss_dict["policy_logits"].shape == (batch_size, BASELINE_ACTION_COUNT)
    assert loss_dict["value_vector"].shape == (batch_size, 4)
