"""Bounded CUDA graph replay for one frozen actor rollout."""

from collections import Counter
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .learn_batch import score_policy

_CAPTURE_AFTER = 16
_GRAPH_LIMIT = 12


@dataclass(slots=True)
class _Replay:
    graph: torch.cuda.CUDAGraph
    tokens: Tensor
    actions: Tensor
    outputs: tuple[Tensor, Tensor, Tensor]


class ActorGraphs:
    """Capture exact shapes without changing batch membership or sampling."""

    def __init__(self, device: torch.device) -> None:
        self._stream = torch.cuda.Stream(device=device)
        self._seen: Counter[tuple[int, int]] = Counter()
        self._graphs: dict[tuple[int, int], _Replay] = {}
        self.hits = 0

    def __call__(
        self, model: nn.Module, tokens: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        key = (actions.shape[0], actions.shape[1])
        replay = self._graphs.get(key)
        if replay is None:
            if len(self._graphs) == _GRAPH_LIMIT:
                return score_policy(model, tokens, actions)
            self._seen[key] += 1
            if self._seen[key] != _CAPTURE_AFTER:
                return score_policy(model, tokens, actions)
            replay = self._capture(model, tokens, actions)
            self._graphs[key] = replay

        replay.tokens.copy_(tokens)
        replay.actions.copy_(actions)
        replay.graph.replay()
        self.hits += 1
        logits, value, mask = replay.outputs
        # Collection keeps values past the next replay, which overwrites graph outputs.
        return logits, value.clone(), mask

    def _capture(self, model: nn.Module, tokens: Tensor, actions: Tensor) -> _Replay:
        tokens = tokens.clone()
        actions = actions.clone()
        current = torch.cuda.current_stream(tokens.device)
        self._stream.wait_stream(current)
        # Do not bind a graph to an autocast cache owned by an outer context.
        with (
            torch.cuda.stream(self._stream),
            torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
        ):
            score_policy(model, tokens, actions)
        current.wait_stream(self._stream)
        graph = torch.cuda.CUDAGraph()
        # Only this thread submits actor work. XLA can manage independent device
        # buffers on its worker threads; a process-wide capture can be invalidated by them.
        with (
            torch.cuda.graph(graph, stream=self._stream, capture_error_mode="thread_local"),
            torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
        ):
            outputs = score_policy(model, tokens, actions)
        return _Replay(graph, tokens, actions, outputs)
