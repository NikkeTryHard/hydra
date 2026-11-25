"""Compiled, bounded-shape CUDA scoring for one frozen self-play policy."""

from dataclasses import dataclass
from functools import cache

import torch
from torch import Tensor, nn

from .learn_batch import score_policy

# These are execution shapes, never game limits. Wider legal sets or larger cohorts
# use the exact eager scorer rather than dropping an action or a player.
_CANDIDATES = 16
_MIN_ROWS = 64
_MAX_ROWS = 1024


@cache
def _compiled_score():
    # The caller owns graph capture. Inductor must not capture a second graph or
    # reuse its outputs while the rollout is still reading them.
    return torch.compile(
        score_policy, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
    )


@dataclass(slots=True)
class _Bucket:
    tokens: Tensor
    actions: Tensor
    graph: torch.cuda.CUDAGraph
    outputs: tuple[Tensor, Tensor, Tensor]
    last_width: int

    def load(self, tokens: Tensor, actions: Tensor) -> None:
        rows, width = actions.shape[:2]
        self.tokens[:rows].copy_(tokens)
        self.actions[:rows, :width].copy_(actions)
        # The second axis is shared by the candidate scorer. Already-masked tail
        # columns stay masked until a wider call writes them again.
        if width < self.last_width:
            self.actions[:, width : self.last_width].fill_(-1)
        self.last_width = width


class CompiledActor:
    """Own bounded replays for one model during one frozen rollout, never across updates."""

    def __init__(self, device: torch.device) -> None:
        self._stream = torch.cuda.Stream(device=device)
        self._buckets: dict[int, _Bucket] = {}
        self._model: nn.Module | None = None
        self.hits = 0
        self.fallbacks = 0

    def __call__(
        self, model: nn.Module, tokens: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self._model is None:
            self._model = model
        elif self._model is not model:
            raise ValueError("a compiled actor cannot replay a different policy's weights")
        rows, width = actions.shape[:2]
        if rows > _MAX_ROWS or width > _CANDIDATES:
            self.fallbacks += 1
            return score_policy(model, tokens, actions)
        size = max(_MIN_ROWS, 1 << (rows - 1).bit_length())
        bucket = self._buckets.get(size)
        if bucket is None:
            bucket = self._capture(model, tokens, actions, size)
            self._buckets[size] = bucket
        bucket.load(tokens, actions)
        bucket.graph.replay()
        self.hits += 1
        logits, value, mask = bucket.outputs
        # Sampling sees real rows and columns only. Values live in the archive
        # past the next replay, which will overwrite this graph's output buffers.
        return logits[:rows, :width], value[:rows].clone(), mask[:rows, :width]

    def _capture(self, model: nn.Module, tokens: Tensor, actions: Tensor, size: int) -> _Bucket:
        x = tokens.new_zeros((size, *tokens.shape[1:]))
        y = actions.new_full((size, _CANDIDATES, actions.shape[-1]), -1)
        current = torch.cuda.current_stream(tokens.device)
        self._stream.wait_stream(current)
        score = _compiled_score()
        # Compile and initialize the kernels outside capture. Use identical autocast
        # contexts so capture cannot compile again or bind to stale cached casts.
        with (
            torch.cuda.stream(self._stream),
            torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
        ):
            score(model, x, y)
        current.wait_stream(self._stream)
        graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(graph, stream=self._stream, capture_error_mode="thread_local"),
            torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
        ):
            outputs = score(model, x, y)
        return _Bucket(x, y, graph, outputs, _CANDIDATES)
