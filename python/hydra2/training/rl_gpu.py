"""Torch/CUDA owner; native leases retain identities, row facts, and scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from hydra2.contracts.common import ContractError
from hydra2.training.rl_objective import _assert_tensor

_DTYPES = {
    "bool": torch.bool,
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "float64": torch.float64,
}


def strict_categorical(
    policy_logits: torch.Tensor, legal_mask: torch.Tensor, uniform: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 policy with normalized FP64 strict-upper-bound categorical sampling."""
    if policy_logits.ndim != 2 or legal_mask.shape != policy_logits.shape:
        raise ContractError("actor logits and legal mask must share [B,A] geometry")
    if min(policy_logits.shape) == 0:
        raise ContractError("actor requires positive real row and action counts")
    if legal_mask.dtype != torch.bool or uniform.dtype != torch.float64:
        raise ContractError("actor requires a bool mask and native binary64 uniforms")
    if uniform.shape != (policy_logits.shape[0],):
        raise ContractError("actor uniforms must have one reserved draw per real row")
    _assert_tensor(
        legal_mask.any(dim=1).all()
        & torch.isfinite(policy_logits).all()
        & torch.isfinite(uniform).all()
        & ((uniform >= 0.0) & (uniform < 1.0)).all(),
        "actor rejects all-illegal, nonfinite, or out-of-domain sampling facts",
    )
    logp = policy_logits.to(torch.float32).masked_fill(~legal_mask, -torch.inf).log_softmax(dim=1)
    cdf = logp.exp().to(torch.float64).cumsum(dim=1)
    cdf = cdf / cdf[:, -1:]
    selected = torch.searchsorted(cdf, uniform.unsqueeze(1).contiguous(), right=True).squeeze(1)
    _assert_tensor(
        (selected < policy_logits.shape[1]).all(), "actor CDF does not cover its uniform"
    )
    selected_logp = logp.gather(1, selected.unsqueeze(1)).squeeze(1)
    _assert_tensor(
        legal_mask.gather(1, selected.unsqueeze(1)).all()
        & torch.isfinite(selected_logp).all()
        & (selected_logp.exp() > 0.0).all(),
        "actor selected an illegal or unrepresentable probability",
    )
    return selected, selected_logp


def _plane_views(raw: torch.Tensor, layout: Any) -> dict[str, torch.Tensor]:
    planes = {}
    for name, dtype_name, shape, offset in layout:
        dtype = _DTYPES[dtype_name]
        count = 1
        for size in shape:
            count *= size
        planes[name] = raw[offset : offset + count * dtype.itemsize].view(dtype).view(shape)
    return planes


def _actor_into_output(
    actor: Any,
    planes: dict[str, torch.Tensor],
    output: torch.Tensor,
    batch: int,
) -> None:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        logits, fused_heads = actor.tensor_forward(planes, planes["history_mask"])
    selected, old_logp = strict_categorical(logits, planes["legal_mask"], planes["uniform"])
    output[: 8 * batch].view(torch.int64).copy_(selected)
    output[8 * batch : 12 * batch].view(torch.float32).copy_(old_logp)
    values = output[12 * batch : 28 * batch].view(torch.float32).view(batch, 4)
    values.copy_(fused_heads[:, 16:20])
    _assert_tensor(torch.isfinite(values).all(), "actor value vector must be finite")


@dataclass(slots=True)
class _Slot:
    bucket_t: int
    host_input: torch.Tensor
    host_output: torch.Tensor
    device_input: torch.Tensor
    device_output: torch.Tensor
    input_exporter: Any
    output_exporter: Any
    views: dict[tuple[int, int], dict[str, torch.Tensor]] = field(default_factory=dict)
    live: bool = False


@dataclass(slots=True)
class _Flight:
    lease: Any
    slot: _Slot
    h2d_done: torch.cuda.Event
    compute_done: torch.cuda.Event
    d2h_done: torch.cuda.Event | None = None


@dataclass(slots=True)
class _ActorGraph:
    input: torch.Tensor
    output: torch.Tensor
    graph: torch.cuda.CUDAGraph


class GpuService:
    """One compute stream, one copy stream, and paired slabs held through capture.

    CPU scheduling, deterministic membership, and policy identity remain native;
    this class only performs operations that require Torch tensors/CUDA events.
    """

    def __init__(
        self,
        pool: Any,
        *,
        device: torch.device,
        compile_mode: str = "eager",
        span_sink: Any = None,
    ) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ContractError("continuous RL requires the configured CUDA device")
        if compile_mode not in ("eager", "cuda_graph_replay"):
            raise ContractError("RL CUDA owner supports eager or exact CUDA replay only")
        self.pool = pool
        self.device = device
        self.compute_stream = torch.cuda.Stream(device=device)
        self.copy_stream = torch.cuda.Stream(device=device)
        self.slots: dict[int, _Slot] = {}
        self.flights: list[_Flight] = []
        self.compile_mode = compile_mode
        self.span_sink: Any = span_sink
        self._actor_graphs: dict[tuple[int, int], _ActorGraph] = {}
        self._graph_actor: Any = None
        self._actor_pool = (
            torch.cuda.graph_pool_handle() if compile_mode == "cuda_graph_replay" else None
        )

    def register_slot(
        self, bucket_t: int, slot_id: int, input_bytes: int, output_bytes: int
    ) -> None:
        if slot_id in self.slots or min(input_bytes, output_bytes) <= 0:
            raise ContractError("CUDA slot geometry is invalid or already registered")
        host_input = torch.empty(input_bytes, dtype=torch.uint8, pin_memory=True)
        host_output = torch.empty(output_bytes, dtype=torch.uint8, pin_memory=True)
        input_exporter = host_input.numpy()
        output_exporter = host_output.numpy()
        self.pool.register_buffers(bucket_t, slot_id, input_exporter, output_exporter)
        self.slots[slot_id] = _Slot(
            bucket_t,
            host_input,
            host_output,
            torch.empty(input_bytes, dtype=torch.uint8, device=self.device),
            torch.empty(output_bytes, dtype=torch.uint8, device=self.device),
            input_exporter,
            output_exporter,
        )

    def replace_pool(self, pool: Any) -> None:
        """Keep CUDA storage/streams; a live lease would make rebinding unsafe."""
        if self.flights or any(slot.live for slot in self.slots.values()):
            raise ContractError("pool replacement requires all issued leases captured")
        # Register on the new pool before swapping: a registration failure
        # leaves this binding intact, and the new pool stays owned by its
        # opener (Collector closes it) — never abandoned open here.
        for slot_id, slot in self.slots.items():
            pool.register_buffers(slot.bucket_t, slot_id, slot.input_exporter, slot.output_exporter)
        old_pool = self.pool
        self.pool = pool
        try:
            old_pool.close()
        except BaseException:
            self.pool = old_pool
            raise

    @staticmethod
    def _input_views(slot: _Slot, lease: Any) -> dict[str, torch.Tensor]:
        key = (lease.bucket_t, lease.n_valid)
        cached = slot.views.get(key)
        if cached is not None:
            return cached
        planes = _plane_views(slot.device_input, lease.input_layout)
        slot.views[key] = planes
        return planes

    def _capture_actor(self, lease: Any, actor: Any, slot: _Slot) -> _ActorGraph:
        if self._graph_actor is not None and actor is not self._graph_actor:
            raise ContractError("CUDA replay requires stable actor module storage")
        self._graph_actor = actor
        if self.span_sink is not None:
            self.span_sink(f"graph-capture-start-actor-T{lease.bucket_t}-B{lease.n_valid}")
        raw = torch.empty(lease.input_bytes, dtype=torch.uint8, device=self.device)
        output = torch.empty(lease.output_bytes, dtype=torch.uint8, device=self.device)
        planes = _plane_views(raw, lease.input_layout)
        raw.copy_(slot.device_input[: lease.input_bytes])
        warm_stream = torch.cuda.Stream(device=self.device)
        warm_stream.wait_stream(self.compute_stream)
        with torch.cuda.stream(warm_stream):
            for _ in range(3):
                _actor_into_output(actor, planes, output, lease.n_valid)
            ready = torch.cuda.Event(blocking=True)
            ready.record(warm_stream)
        ready.synchronize()
        graph = torch.cuda.CUDAGraph()
        # Only transient allocations share this pool; persistent inputs/outputs
        # are external, so another real B/T replay cannot clobber their values.
        with torch.cuda.graph(graph, pool=self._actor_pool, stream=self.compute_stream):
            _actor_into_output(actor, planes, output, lease.n_valid)
        result = _ActorGraph(raw, output, graph)
        self._actor_graphs[lease.bucket_t, lease.n_valid] = result
        if self.span_sink is not None:
            self.span_sink(f"graph-capture-end-actor-T{lease.bucket_t}-B{lease.n_valid}")
        return result

    def issue(self, lease: Any, actor: Any) -> None:
        slot = self.slots[lease.slot_id]
        if slot.live:
            raise ContractError("CUDA slot cannot be reused before native capture")
        if actor.training:
            raise ContractError("the immutable RL actor must remain in eval mode")
        if (
            lease.input_bytes > slot.host_input.numel()
            or lease.output_bytes > slot.host_output.numel()
        ):
            raise ContractError("native active lease exceeds its registered CUDA slab")
        slot.live = True
        h2d_done = torch.cuda.Event()
        compute_done = torch.cuda.Event()
        lease.mark_h2d_issued()
        with torch.cuda.stream(self.copy_stream):
            slot.device_input[: lease.input_bytes].copy_(
                slot.host_input[: lease.input_bytes], non_blocking=True
            )
            h2d_done.record(self.copy_stream)
        planes = self._input_views(slot, lease)
        lease.mark_inference_queued()
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(h2d_done)
            if self.compile_mode == "cuda_graph_replay":
                captured = self._actor_graphs.get((lease.bucket_t, lease.n_valid))
                if captured is None:
                    captured = self._capture_actor(lease, actor, slot)
                elif actor is not self._graph_actor:
                    raise ContractError("CUDA replay requires stable actor module storage")
                captured.input.copy_(slot.device_input[: lease.input_bytes])
                captured.graph.replay()
                slot.device_output[: lease.output_bytes].copy_(captured.output)
            else:
                _actor_into_output(actor, planes, slot.device_output, lease.n_valid)
            compute_done.record(self.compute_stream)
        self.flights.append(_Flight(lease, slot, h2d_done, compute_done))

    def flush_outputs(self, *, keep_latest: bool = False) -> None:
        """Queue D2H after the next H2D, avoiding a copy-stream compute barrier."""
        pending = [flight for flight in self.flights if flight.d2h_done is None]
        if keep_latest:
            pending = pending[:-1]
        for flight in pending:
            event = torch.cuda.Event(blocking=True)
            flight.lease.mark_d2h_issued()
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(flight.compute_done)
                size = flight.lease.output_bytes
                flight.slot.host_output[:size].copy_(
                    flight.slot.device_output[:size], non_blocking=True
                )
                event.record(self.copy_stream)
            flight.d2h_done = event

    def drain(self, *, wait: bool = False) -> int:
        if wait:
            self.flush_outputs()
        captured = 0
        remaining: list[_Flight] = []
        for flight in self.flights:
            if flight.d2h_done is None:
                remaining.append(flight)
                continue
            if wait:
                flight.d2h_done.synchronize()
            if not flight.d2h_done.query():
                remaining.append(flight)
                continue
            flight.lease.mark_host_ready(flight.compute_done, flight.d2h_done)
            self.pool.commit_actions(flight.lease)
            flight.slot.live = False
            captured += 1
        self.flights = remaining
        return captured

    def wait_for_cuda_progress(self) -> None:
        if not self.flights:
            raise ContractError("no admitted CUDA work can provide progress")
        self.flush_outputs()
        event = self.flights[0].d2h_done
        if event is None:
            raise ContractError("CUDA progress requires an issued output transfer")
        event.synchronize()
        self.drain()

    def copy_actor_at_boundary(self, learner: Any, actor: Any) -> None:
        """Copy at a completed optimizer boundary on the same compute stream."""
        if learner.training or actor.training:
            raise ContractError("both RL modules must be eval with learner autograd enabled")
        with torch.cuda.stream(self.compute_stream), torch.no_grad():
            actor.load_state_dict(learner.state_dict(), strict=True, assign=False)
        # Autocast is scoped per invocation; a stale BF16 weight cache must not
        # survive replacement or the actor would serve a different checkpoint.
        torch.clear_autocast_cache()

    def freeze_terminal_advantages(self, native: Any, tape_handle: int) -> dict[str, Any]:
        """One GPU gather/subtraction; native freezes and validates every real row."""
        import json

        descriptor, actors, games, values, utilities = native.terminal_inputs(tape_handle)
        counts = json.loads(descriptor)
        count, game_count = counts["row_count"], counts["game_count"]
        with torch.cuda.stream(self.compute_stream), torch.inference_mode():
            actor = torch.frombuffer(bytearray(actors), dtype=torch.int64).to(self.device)
            game = torch.frombuffer(bytearray(games), dtype=torch.int64).to(self.device)
            behavior = (
                torch.frombuffer(bytearray(values), dtype=torch.float32)
                .view(count, 4)
                .to(self.device)
            )
            utility = (
                torch.frombuffer(bytearray(utilities), dtype=torch.float64)
                .view(game_count, 4)
                .to(self.device)
            )
            returns = utility.to(torch.float32)[game]
            raw = returns.gather(1, actor[:, None]).squeeze(1) - behavior.gather(
                1, actor[:, None]
            ).squeeze(1)
            _assert_tensor(torch.isfinite(raw).all(), "terminal advantages must be finite")
            frozen = raw.cpu().numpy().tobytes()
        return json.loads(native.freeze_terminal_advantages(tape_handle, frozen))

    def quiesce(self) -> None:
        self.pool.quiesce()
        self.drain(wait=True)
        self.compute_stream.synchronize()
        self.copy_stream.synchronize()
        self.pool.drain()

    def close(self) -> None:
        if self.flights or any(slot.live for slot in self.slots.values()):
            raise ContractError("CUDA service close requires owner-driven quiescence")
        self.pool.close()
        self._actor_graphs.clear()
        self._actor_pool = None
        self._graph_actor = None
        self.slots.clear()
