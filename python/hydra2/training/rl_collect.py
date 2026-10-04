"""CUDA-owner dispatch over a native fixed-cohort collector; no row processing."""

from __future__ import annotations

import contextlib
import json
from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.training.rl_gpu import GpuService


class Collector:
    """Native workers own game evolution; this object owns buffer/event dispatch."""

    def __init__(
        self,
        native: Any,
        spec: dict[str, Any],
        derivation: str,
        *,
        gpu: GpuService | None,
        device: Any,
        capacity: int,
        pipeline_spec_id: str,
        compile_mode: str = "eager",
        span_sink: Any = None,
    ) -> None:
        self.native = native
        self.spec = spec
        self._live_binding = json.dumps(
            {
                "pipeline_spec_id": pipeline_spec_id,
                "generation": spec["generation"],
                "behavior_policy_id": spec["policy_id"],
            }
        ).encode("utf-8")
        profile, context = json.dumps(spec["profile"]), json.dumps(spec["context"])
        opened = native.PyNativePool.open_live(profile, context, derivation, self._live_binding)
        self.pool = opened
        try:
            if gpu is None:
                self.gpu = GpuService(
                    self.pool, device=device, compile_mode=compile_mode, span_sink=span_sink
                )
                for bucket_index, bucket in enumerate((32, 64, 128, 256)):
                    input_bytes, output_bytes = native.buffer_geometry(bucket, capacity)
                    for pair in range(2):
                        self.gpu.register_slot(
                            bucket, 2 * bucket_index + pair, input_bytes, output_bytes
                        )
            else:
                gpu.replace_pool(self.pool)
                self.gpu = gpu
        except BaseException:
            # Construction failed after the native pool opened: close the
            # orphan here since no caller ever receives this object.
            with contextlib.suppress(Exception):
                self.pool.close()
            raise
        self.filling: int | None = None
        self.ready_slots: set[int] = set()
        self.pending_lease: Any = None
        self.complete = False
        self.tape: int | None = None
        self.admitting = True
        self.last_status: dict[str, Any] = self.pool.poll()

    def poll(self) -> bool:
        """Harvest native fill completion before capture; admission never waits."""
        if self.filling is not None:
            ticket = self.pool.take_fill(self.filling)
            if ticket is not None:
                if ticket["n_valid"]:
                    self.ready_slots.add(self.filling)
                self.filling = None
        if self.filling is None:
            self.gpu.drain()
        self.last_status = self.pool.poll()
        if self.last_status["poison"]:
            raise ContractError(f"native collector poisoned: {self.last_status['poison']}")
        count = len(self.spec["profile"]["games"])
        if (
            self.last_status["terminal_games"] == count
            and self.last_status["pending_applies"] == 0
            and self.last_status["live_leases"] == 0
            and self.filling is None
            and not self.gpu.flights
        ):
            self.complete = True
            return False
        if self.pending_lease is None:
            self.pending_lease = self.pool.peek_ready()
        return self.pending_lease is not None

    def admit(self) -> None:
        if not self.admitting or self.complete or self.filling is not None:
            return
        free = [
            slot_id
            for slot_id, slot in self.gpu.slots.items()
            if not slot.live and slot_id not in self.ready_slots
        ]
        self.filling = self.pool.admit_next_fill(free)

    def infer(self, actor: Any) -> None:
        lease = self.pending_lease
        if lease is None:
            raise ContractError("collector inference requires a native READY lease")
        self.gpu.issue(lease, actor)
        self.ready_slots.discard(lease.slot_id)
        self.pending_lease = None
        self.gpu.flush_outputs(keep_latest=True)

    def flush_if_no_next_ticket(self) -> None:
        if self.pending_lease is None and not self.ready_slots:
            self.gpu.flush_outputs()

    def wait(self) -> None:
        """Block only when neither useful CUDA class can advance."""
        self.gpu.flush_outputs()
        if self.filling is not None:
            ticket = self.pool.take_fill_blocking(self.filling)
            if ticket["n_valid"]:
                self.ready_slots.add(self.filling)
            self.filling = None
        elif self.gpu.flights:
            self.gpu.wait_for_cuda_progress()
        else:
            self.pool.wait_for_progress()

    def quiesce(self) -> None:
        """Drain issued work without serializing rows; durable consumers snapshot once."""
        self.admitting = False
        if self.pending_lease is not None:
            self.pool.return_ready_lease(self.pending_lease)
            self.pending_lease = None
        if self.filling is not None:
            ticket = self.pool.take_fill_blocking(self.filling)
            if ticket["n_valid"]:
                self.ready_slots.add(self.filling)
            self.filling = None
        self.gpu.quiesce()

    def resume_admission(self) -> None:
        self.pool.resume_admission()
        self.admitting = True

    def export(self) -> int:
        if not self.complete:
            raise ContractError("partial hanchan cohorts cannot become learner artifacts")
        self.quiesce()
        if self.tape is None:
            self.tape = int(self.pool.drain_tape(self._live_binding))
        return self.tape
