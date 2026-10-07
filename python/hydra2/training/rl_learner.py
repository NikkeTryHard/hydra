"""Torch-only learner microsteps over native-filled, immutable replay batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hydra2.contracts.common import ContractError
from hydra2.training.rl_gpu import _DTYPES
from hydra2.training.rl_objective import (
    PpoParameters,
    PpoTerms,
    ppo_micro_terms,
    prepare_advantages,
)


@dataclass(slots=True)
class ReplaySlot:
    host_features: torch.Tensor
    host_facts: torch.Tensor
    device_features: torch.Tensor
    device_facts: torch.Tensor
    feature_exporter: Any
    fact_exporter: Any
    ready: torch.cuda.Event | None = None
    consumed: torch.cuda.Event | None = None
    count: int = 0
    bucket_t: int = 0
    planes: dict[str, torch.Tensor] | None = None
    selected: torch.Tensor | None = None
    old_logp: torch.Tensor | None = None
    advantage: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    normalized_advantage: torch.Tensor | None = None
    float_returns: torch.Tensor | None = None


@dataclass(slots=True)
class _LearnerGraph:
    planes: dict[str, torch.Tensor]
    selected: torch.Tensor
    old_logp: torch.Tensor
    advantage: torch.Tensor
    returns: torch.Tensor
    first: torch.cuda.CUDAGraph
    accumulate: torch.cuda.CUDAGraph
    gradients: list[torch.Tensor | None]
    metrics: list[torch.Tensor]
    transfers: dict[tuple[int, int], torch.cuda.CUDAGraph]
    capture_stream: torch.cuda.Stream


class LearnerService:
    """Two bounded replay slots; gradients remain on the sole compute stream.

    Native admission supplies validated real counts and layouts. Each optimizer
    batch is normalized once before slicing, so short tails retain the PPO law.
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        *,
        native: Any,
        compute_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        device: torch.device,
        parameters: PpoParameters,
        microbatch_cap: int,
        optimizer_cap: int,
        gradient_clip_norm: float,
        cache_byte_budget: int,
        compile_mode: str = "eager",
        span_sink: Any = None,
    ) -> None:
        if model.training or device.type != "cuda":
            raise ContractError("RL learner requires eval mode with CUDA autograd")
        if compile_mode not in ("eager", "cuda_graph_replay"):
            raise ContractError("RL learner supports eager or exact CUDA replay only")
        self.compile_mode = compile_mode
        self.graphs: dict[int, _LearnerGraph] = {}
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compute_stream = compute_stream
        self.copy_stream = copy_stream
        self.device = device
        self.parameters = parameters
        self.microbatch_cap = microbatch_cap
        self.optimizer_cap = optimizer_cap
        self.gradient_clip_norm = gradient_clip_norm
        feature_capacity, fact_capacity = native.slab_geometry(256, optimizer_cap)
        if 2 * (feature_capacity + fact_capacity) > cache_byte_budget:
            raise ContractError("paired replay cache exceeds the frozen GPU byte budget")
        self.slots: list[ReplaySlot] = []
        for _ in range(2):
            features = torch.empty(feature_capacity, dtype=torch.uint8, pin_memory=True)
            facts = torch.empty(fact_capacity, dtype=torch.uint8, pin_memory=True)
            self.slots.append(
                ReplaySlot(
                    features,
                    facts,
                    torch.empty_like(features, device=device),
                    torch.empty_like(facts, device=device),
                    features.numpy(),
                    facts.numpy(),
                )
            )
        self.active: ReplaySlot | None = None
        self.offset = 0
        self.metrics: list[torch.Tensor] | None = None
        self.last_update: torch.cuda.Event | None = None
        self.span_sink = span_sink

    def _terms(
        self,
        planes: dict[str, torch.Tensor],
        selected: torch.Tensor,
        old_logp: torch.Tensor,
        advantage: torch.Tensor,
        returns: torch.Tensor,
        count: int,
    ) -> PpoTerms:
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            logits, heads = self.model.tensor_forward(planes, planes["history_mask"])
        return ppo_micro_terms(
            logits,
            heads[:, 16:20],
            planes["legal_mask"],
            selected,
            old_logp,
            advantage,
            returns,
            minibatch_count=count,
            parameters=self.parameters,
        )

    def _capture(self, slot: ReplaySlot) -> _LearnerGraph:
        if self.span_sink is not None:
            self.span_sink(f"graph-capture-start-learner-T{slot.bucket_t}")
        assert slot.planes is not None and slot.selected is not None
        assert slot.old_logp is not None and slot.advantage is not None and slot.returns is not None
        # Completed updates no longer need the previous bucket's private
        # autograd pool. Retaining all buckets accumulates full activation
        # pools and exhausts VRAM when actor collection overlaps learning.
        if len(self.graphs) > 0:
            self.compute_stream.synchronize()
            self.optimizer.zero_grad(set_to_none=True)
            self.graphs.clear()
        span = slice(0, self.microbatch_cap)
        planes = {name: tensor[span].clone() for name, tensor in slot.planes.items()}
        selected, old_logp, advantage, returns = (
            tensor[span].clone()
            for tensor in (slot.selected, slot.old_logp, slot.advantage, slot.returns)
        )

        def terms() -> PpoTerms:
            return self._terms(planes, selected, old_logp, advantage, returns, slot.count)

        # Graph destruction clears cuBLAS workspaces for its capture stream.
        # A private stream prevents retiring this bucket from invalidating
        # live actor graphs; replay still runs on the sole owner stream.
        warm = torch.cuda.Stream(device=self.device)
        warm.wait_stream(self.compute_stream)
        with torch.cuda.stream(warm):
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=True)
                _ = terms().loss.backward()
            done = torch.cuda.Event(blocking=True)
            done.record(warm)
        done.synchronize()
        self.optimizer.zero_grad(set_to_none=True)
        first, accumulate = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        # FIRST owns exactly the autograd-active gradients; keeping unused
        # parameters at None prevents AdamW from decaying unused heads.
        with torch.cuda.graph(first, stream=warm):
            first_terms = terms()
            _ = first_terms.loss.backward()
            metrics = [tensor.detach() for tensor in first_terms]
        gradients = [parameter.grad for parameter in self.model.parameters()]
        # Shared storage is replayed only FIRST then ACCUM*, so live gradients
        # and metrics cannot be overwritten by a different history geometry.
        with torch.cuda.graph(accumulate, pool=first.pool(), stream=warm):
            parts = terms()
            _ = parts.loss.backward()
            for total, part in zip(metrics, parts, strict=True):
                _ = total.add_(part.detach())
        result = _LearnerGraph(
            planes,
            selected,
            old_logp,
            advantage,
            returns,
            first,
            accumulate,
            gradients,
            metrics,
            {},
            warm,
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.graphs[slot.bucket_t] = result
        if self.span_sink is not None:
            self.span_sink(f"graph-capture-end-learner-T{slot.bucket_t}")
        return result

    def fill(
        self, slot_id: int, native: Any, tape: int, replay: int, batch_index: int
    ) -> dict[str, Any]:
        """One native live batched fill, then slot install; never row callbacks."""
        slot = self.slots[slot_id]
        if slot is self.active:
            raise ContractError("cannot overwrite an accumulating replay batch")
        # Learner-stall marks: the consumed sync waits for prior GPU work on
        # this slot, and the native fill encodes rows on Rust workers. Both are
        # host timestamps only and never gate numerics.
        if slot.consumed is not None:
            if self.span_sink is not None:
                self.span_sink("learner-sync-start")
            slot.consumed.synchronize()
            if self.span_sink is not None:
                self.span_sink("learner-sync-end")
        import json

        if self.span_sink is not None:
            self.span_sink("learner-native-fill-start")
        spec = json.loads(
            native.fill_live_slab(
                tape_handle=tape,
                replay_handle=replay,
                batch_idx=batch_index,
                features_buffer=slot.feature_exporter,
                facts_buffer=slot.fact_exporter,
            )
        )
        if self.span_sink is not None:
            self.span_sink("learner-native-fill-end")
        return self.install_filled_slot(slot_id, spec)

    def install_filled_slot(self, slot_id: int, spec: dict[str, Any]) -> dict[str, Any]:
        """Geometry validation, H2D, views, and ready/consumed state for filled bytes.

        Production :meth:`fill` writes native live bytes first; archive
        tooling writes its archive bytes explicitly, then calls this helper.
        """
        slot = self.slots[slot_id]
        count = int(spec["count"])
        feature_bytes, fact_bytes = int(spec["feature_bytes"]), int(spec["fact_bytes"])
        if not 0 < count <= self.optimizer_cap or fact_bytes != count * 48:
            raise ContractError("native replay descriptor violates real-count geometry")
        if feature_bytes > slot.host_features.numel() or fact_bytes > slot.host_facts.numel():
            raise ContractError("native replay descriptor exceeds registered cache capacity")
        slot.count, slot.bucket_t = count, int(spec["t"])
        with torch.cuda.stream(self.copy_stream):
            if self.span_sink is not None:
                self.span_sink("learner-h2d-start")
            _ = slot.device_features[:feature_bytes].copy_(
                slot.host_features[:feature_bytes], non_blocking=True
            )
            _ = slot.device_facts[:fact_bytes].copy_(
                slot.host_facts[:fact_bytes], non_blocking=True
            )
            slot.ready = torch.cuda.Event()
            slot.ready.record(self.copy_stream)
            if self.span_sink is not None:
                self.span_sink("learner-h2d-end")
        planes = {}
        for field in spec["input_layout"]:
            name, shape, offset = field["name"], field["shape"], field["offset"]
            dtype = _DTYPES[field["dtype"]]
            size = 1
            for dimension in shape:
                size *= dimension
            planes[name] = (
                slot.device_features[offset : offset + size * dtype.itemsize]
                .view(dtype)
                .view(shape)
            )
        slot.planes = planes
        slot.selected = slot.device_facts[: 8 * count].view(torch.int64)
        slot.old_logp = slot.device_facts[8 * count : 12 * count].view(torch.float32)
        slot.advantage = slot.device_facts[12 * count : 16 * count].view(torch.float32)
        slot.returns = slot.device_facts[16 * count : 48 * count].view(torch.float64).view(count, 4)
        slot.consumed = None
        return spec

    def begin(self, slot_id: int) -> None:
        slot = self.slots[slot_id]
        if (
            self.active is not None
            or slot.ready is None
            or slot.advantage is None
            or slot.returns is None
        ):
            raise ContractError(
                "optimizer batch admission requires an idle learner and ready replay"
            )
        self.active, self.offset = slot, 0
        self.metrics = None
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(slot.ready)
            self.optimizer.zero_grad(set_to_none=True)
            normalized = prepare_advantages(slot.advantage, eps_std=self.parameters.eps_std)
            returns = slot.returns.to(torch.float32)
            if self.compile_mode == "cuda_graph_replay" and slot.count == self.optimizer_cap:
                if slot.normalized_advantage is None:
                    slot.normalized_advantage = torch.empty_like(normalized)
                    slot.float_returns = torch.empty_like(returns)
                assert slot.float_returns is not None
                _ = slot.normalized_advantage.copy_(normalized)
                _ = slot.float_returns.copy_(returns)
                slot.advantage, slot.returns = slot.normalized_advantage, slot.float_returns
            else:
                slot.advantage, slot.returns = normalized, returns

    def microstep(self) -> bool:
        """Queue exactly one complete forward/loss/backward; return batch completion."""
        slot = self.active
        if (
            slot is None
            or slot.planes is None
            or slot.selected is None
            or slot.old_logp is None
            or slot.advantage is None
            or slot.returns is None
        ):
            raise ContractError("learner microstep requires a fully admitted optimizer batch")
        end = min(self.offset + self.microbatch_cap, slot.count)
        span = slice(self.offset, end)
        with torch.cuda.stream(self.compute_stream):
            planes = {name: tensor[span] for name, tensor in slot.planes.items()}
            if (
                self.compile_mode == "cuda_graph_replay"
                and slot.count == self.optimizer_cap
                and slot.count % self.microbatch_cap == 0
            ):
                captured = self.graphs.get(slot.bucket_t)
                if captured is None:
                    if self.offset != 0:
                        raise ContractError("learner capture requires an optimizer boundary")
                    captured = self._capture(slot)
                transfer_key = (
                    next(i for i, owner in enumerate(self.slots) if owner is slot),
                    self.offset,
                )
                transfer = captured.transfers.get(transfer_key)
                if transfer is None:
                    transfer = torch.cuda.CUDAGraph()
                    # Source slot and normalized buffers keep stable addresses;
                    # offset-keyed transfers cannot read another batch's facts.
                    with torch.cuda.graph(transfer, stream=captured.capture_stream):
                        for name, tensor in captured.planes.items():
                            _ = tensor.copy_(planes[name])
                        for target, source in zip(
                            (
                                captured.selected,
                                captured.old_logp,
                                captured.advantage,
                                captured.returns,
                            ),
                            (slot.selected, slot.old_logp, slot.advantage, slot.returns),
                            strict=True,
                        ):
                            _ = target.copy_(source[span])
                    captured.transfers[transfer_key] = transfer
                transfer.replay()
                if self.offset == 0:
                    for parameter, gradient in zip(
                        self.model.parameters(), captured.gradients, strict=True
                    ):
                        parameter.grad = gradient
                    captured.first.replay()
                    self.metrics = captured.metrics
                else:
                    captured.accumulate.replay()
            else:
                terms = self._terms(
                    planes,
                    slot.selected[span],
                    slot.old_logp[span],
                    slot.advantage[span],
                    slot.returns[span],
                    slot.count,
                )
                _ = terms.loss.backward()
                detached = [tensor.detach() for tensor in terms]
                if self.metrics is None:
                    self.metrics = detached
                else:
                    for total, part in zip(self.metrics, detached, strict=True):
                        _ = total.add_(part)
        self.offset = end
        return end == slot.count

    def finish(self) -> tuple[int, PpoTerms]:
        slot = self.active
        if slot is None or self.offset != slot.count or self.metrics is None:
            raise ContractError("optimizer update cannot commit a partial accumulation")
        with torch.cuda.stream(self.compute_stream):
            _ = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
                error_if_nonfinite=True,
                foreach=True,
            )
            self.optimizer.step()
            self.scheduler.step()
            slot.consumed = torch.cuda.Event(blocking=True)
            slot.consumed.record(self.compute_stream)
            self.last_update = slot.consumed
        count, metrics = slot.count, PpoTerms(*self.metrics)
        self.active, self.metrics = None, None
        return count, metrics

    def boundary(self) -> None:
        if self.active is not None:
            raise ContractError("checkpoint boundary cannot split gradient accumulation")
        if self.last_update is not None:
            self.last_update.synchronize()
