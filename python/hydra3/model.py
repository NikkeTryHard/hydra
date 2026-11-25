"""Public policy contract: one rich-state encode and one candidate readout."""

import torch
from torch import Tensor, nn

from .model_config import SCORER_EDIT, ModelConfig
from .model_encoder import TokenEncoder
from .model_schema import (
    ACTION_ID,
    ACTION_WIDTH,
    MATCH_TOKEN,
    MAX_OWN_COUNT,
    OWN_COUNT,
    OWN_RED,
    SELF_SEAT_TOKEN,
    TILE_TOKENS,
    TOKEN_COUNT,
    TOKEN_WIDTH,
)
from .model_scorer import EditScorer, PlainScorer

__all__ = ["ModelConfig", "Policy"]


class _ScoreHead(nn.Module):
    """Shared nonlinear readout from one candidate representation to a logit."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, representation: Tensor) -> Tensor:
        return self.net(representation).squeeze(-1)


class _Critic(nn.Module):
    """Actor-only placement critic from the match and self-seat tokens."""

    def __init__(self, width: int) -> None:
        super().__init__()
        critic_width = 2 * width
        self.net = nn.Sequential(
            nn.LayerNorm(critic_width),
            nn.Linear(critic_width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, encoded: Tensor) -> Tensor:
        actor = torch.cat((encoded[:, MATCH_TOKEN], encoded[:, SELF_SEAT_TOKEN]), dim=-1)
        return self.net(actor).squeeze(-1)


class Policy(nn.Module):
    """Scores every legal candidate from one shared encoding of the table."""

    config: ModelConfig
    scorer: EditScorer | PlainScorer

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError(f"config must be ModelConfig, got {type(config)!r}")
        self.config = config
        self.encoder = TokenEncoder(config.width, config.depth, config.heads, config.ff)
        if config.scorer == SCORER_EDIT:
            self.scorer = EditScorer(config.width)
        else:
            self.scorer = PlainScorer(config.width)
        self.score = _ScoreHead(config.width)
        self.critic = _Critic(config.width)

    def forward(self, tokens: Tensor, candidates: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (TOKEN_COUNT, TOKEN_WIDTH):
            raise ValueError(
                f"tokens must be [B, {TOKEN_COUNT}, {TOKEN_WIDTH}], got {tuple(tokens.shape)}"
            )
        if candidates.ndim != 3 or candidates.shape[2] != ACTION_WIDTH:
            raise ValueError(
                f"candidates must be [B, K, {ACTION_WIDTH}], got {tuple(candidates.shape)}"
            )
        if tokens.shape[0] != candidates.shape[0]:
            raise ValueError("tokens and candidates must share a batch size")
        if not tokens.is_floating_point():
            raise TypeError("tokens must be floating point")
        if candidates.is_floating_point() or candidates.is_complex():
            raise TypeError("candidates must be integer")

        encoded = self.encoder(tokens)
        scorer = self.scorer
        if isinstance(scorer, EditScorer):
            own_counts = tokens[:, :TILE_TOKENS, OWN_COUNT].clamp(0, MAX_OWN_COUNT)
            own_reds = tokens[:, :TILE_TOKENS, OWN_RED].clamp(0, 1)
            representation = scorer(encoded, candidates, own_counts, own_reds)
        else:
            representation = scorer(encoded, candidates)

        logits = self.score(representation)
        legal = candidates[..., ACTION_ID] >= 0
        logits = logits.masked_fill(~legal, float("-inf"))
        return logits, self.critic(encoded)
