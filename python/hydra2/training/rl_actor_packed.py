"""Actor-only packed histories; native boundaries keep independent games isolated."""

from __future__ import annotations

from typing import Any

import torch

from hydra2.training.rl_objective import _assert_tensor


def packed_actor_forward(
    actor: Any, planes: dict[str, torch.Tensor], packed: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    cu = packed["cu_seqlens"]
    lengths = cu[1:] - cu[:-1]
    capacity = packed["packed_kind"].numel()
    _assert_tensor(cu[0] == 0, "packed actor boundaries must start at zero")
    _assert_tensor(cu[-1] <= capacity, "packed actor boundaries exceed token storage")
    _assert_tensor((lengths >= 0).all(), "packed actor boundaries must be monotone")
    _assert_tensor((lengths <= 256).all(), "packed actor history exceeds 256 tokens")
    _assert_tensor(
        (lengths == planes["history_mask"].sum(dim=1)).all(),
        "packed actor lengths differ from native real history prefixes",
    )
    # Native positions reset per row; using stream offsets would leak neighboring
    # histories into position embeddings and change the actor-visible model.
    x = actor.history_embedding(packed["packed_kind"]) + actor.pos_embedding(packed["token_pos"])
    for layer in actor.layers:
        x = layer.forward_packed(x, cu, 256)
    x = actor.final_norm(x)
    # Capacity tail is not an attention sequence. Clear it before pooling;
    # multiplying by zero would preserve a NaN from unused attention storage.
    x = torch.where(packed["valid"][:, None], x, 0.0)
    batch = lengths.numel()
    sums = torch.zeros((batch, actor.d_model), dtype=x.dtype, device=x.device)
    sums.index_add_(0, packed["token_row"], x)
    pooled = sums / lengths.to(x.dtype).clamp(min=1).unsqueeze(-1)
    scalar = actor.scalar_proj(actor._build_scalar_features(planes, dtype=x.dtype))
    trunk = torch.cat((pooled, scalar), dim=-1)
    return actor.policy_head(trunk), actor.small_heads(trunk)
