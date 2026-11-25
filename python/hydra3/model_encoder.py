"""Actor-visible rich-state encoder with fused scaled-dot-product attention."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model_schema import (
    HONOR_COUNT,
    RANK_COUNT,
    SEAT_TOKENS,
    SUIT_COUNT,
    TOKEN_WIDTH,
)


class _ResidualBlock(nn.Module):
    """Pre-norm attention and feed-forward residual block."""

    def __init__(self, width: int, heads: int, ff: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_out = nn.Linear(width, width)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, ff),
            nn.GELU(),
            nn.Linear(ff, width),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(self.norm_attention(x))
        qkv = qkv.view(batch, tokens, 3, self.heads, self.head_width)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        x = x + self.attention_out(attended)
        return x + self.ff(self.norm_ff(x))


class _RoleBias(nn.Module):
    """Suit-rank tile positions plus distinct honor, seat and match roles."""

    suit_index: Tensor
    rank_index: Tensor
    honor_index: Tensor
    seat_index: Tensor

    def __init__(self, width: int) -> None:
        super().__init__()
        self.suit = nn.Embedding(SUIT_COUNT, width)
        self.rank = nn.Embedding(RANK_COUNT, width)
        self.honor = nn.Embedding(HONOR_COUNT, width)
        self.seat = nn.Embedding(SEAT_TOKENS, width)
        self.match = nn.Parameter(torch.zeros(width))
        suit_index = torch.arange(SUIT_COUNT).repeat_interleave(RANK_COUNT)
        rank_index = torch.arange(RANK_COUNT).repeat(SUIT_COUNT)
        honor_index = torch.arange(HONOR_COUNT)
        seat_index = torch.arange(SEAT_TOKENS)
        self.register_buffer("suit_index", suit_index, persistent=False)
        self.register_buffer("rank_index", rank_index, persistent=False)
        self.register_buffer("honor_index", honor_index, persistent=False)
        self.register_buffer("seat_index", seat_index, persistent=False)

    def forward(self) -> Tensor:
        suited = self.suit(self.suit_index) + self.rank(self.rank_index)
        honors = self.honor(self.honor_index)
        seats = self.seat(self.seat_index)
        return torch.cat((suited, honors, seats, self.match.unsqueeze(0)), dim=0)


class TokenEncoder(nn.Module):
    """Encodes one rich 39-token state per acting seat."""

    def __init__(self, width: int, depth: int, heads: int, ff: int) -> None:
        super().__init__()
        self.features = nn.Linear(TOKEN_WIDTH, width)
        self.roles = _RoleBias(width)
        self.norm_in = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([_ResidualBlock(width, heads, ff) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(width)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.norm_in(self.features(tokens) + self.roles())
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)
