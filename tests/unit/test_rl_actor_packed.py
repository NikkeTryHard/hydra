"""Packed actor histories cannot depend on neighboring games or unused capacity."""

from __future__ import annotations

import pytest
import torch

from hydra2.models.model import Hydra2BaselineModel
from hydra2.training.rl_actor_packed import packed_actor_forward

pytestmark = [pytest.mark.contract_package("WP-11"), pytest.mark.gpu, pytest.mark.serial]


def test_packed_actor_isolates_rows_and_live_graph_capacity(require_cuda: None) -> None:
    from hydra2.training.rl_checkpoint import _freeze_actor_linear_dtypes

    torch.manual_seed(41)
    actor = (
        Hydra2BaselineModel(
            architecture_id="hydra2_big_transformer_v1",
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=64,
            dropout=0.0,
        )
        .cuda()
        .eval()
    )
    actor.requires_grad_(False)
    _freeze_actor_linear_dtypes(actor=actor)
    actor._build_scalar_features = lambda planes, dtype: planes["scalars"].to(dtype)
    planes = {
        "history_mask": torch.tensor([[True, True, False], [True, False, False]], device="cuda"),
        "scalars": torch.zeros((2, 64), device="cuda"),
    }
    packed = {
        "packed_kind": torch.tensor([1, 2, 3, 0, 0, 0, 0, 0], device="cuda"),
        "cu_seqlens": torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda"),
        "token_row": torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], device="cuda"),
        "token_pos": torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], device="cuda"),
        "valid": torch.tensor([True, True, True, False, False, False, False, False], device="cuda"),
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        first = packed_actor_forward(actor, planes, packed)
        reference = tuple(t.clone() for t in first)
        packed["packed_kind"][2:] = 4
        changed = packed_actor_forward(actor, planes, packed)
        assert torch.equal(reference[0][0], changed[0][0])
        assert torch.equal(reference[1][0], changed[1][0])
        assert not torch.equal(reference[0][1], changed[0][1])
        for _ in range(3):
            packed_actor_forward(actor, planes, packed)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = packed_actor_forward(actor, planes, packed)
        planes["history_mask"][0, 1] = False
        packed["cu_seqlens"].copy_(torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"))
        packed["token_row"].copy_(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], device="cuda"))
        packed["token_pos"].zero_()
        packed["valid"][2] = False
        packed["packed_kind"][1] = 7
        actor.policy_head.weight.add_(0.125)
        expected = packed_actor_forward(actor, planes, packed)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output[0], expected[0])
        assert torch.equal(output[1], expected[1])
        assert not torch.equal(reference[0], output[0])
        assert torch.isfinite(output[0]).all()
        assert torch.isfinite(output[1]).all()
