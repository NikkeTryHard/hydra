"""Background checkpoint publish: CPU snapshot plus single-worker writer.

A checkpoint save stalls training for ~1.6s per generation on the main
thread (single ``torch.save`` of the ~27MB blob plus fsync; hashing is only
~52ms). :func:`_snapshot_payload_to_cpu` detaches every live tensor to CPU
synchronously (~7ms, the same copies the manifest hasher already makes),
then :class:`_BackgroundCheckpointWriter` publishes bytes on one background
thread while training continues. Invariant: the worker never touches a live
GPU tensor (mutation race with the next update); every tensor it sees is a
detached CPU clone owned by the snapshot. Failure mode: any worker error
re-raises on the calling thread at the next poll/drain (fail closed, never
a silent skip), and mirror hooks fire only after the generation they
announce has landed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.runtime.checkpoint import PAYLOAD_SECTION_KEYS as PAYLOAD_SECTION_KEYS
from hydra2.runtime.checkpoint import StateTreeError as StateTreeError


def _snapshot_payload_to_cpu(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Detach a checkpoint payload into a CPU-only snapshot (main thread).

    Walks the payload exactly like ``state_tree`` in
    :mod:`hydra2.runtime.checkpoint` but preserves values instead of hashing:
    every tensor (CUDA or CPU) becomes a detached CPU clone with ``copy=True``
    so continued training cannot mutate the snapshot through shared storage;
    containers are rebuilt fresh so live mutable lists/dicts cannot alias
    either; immutable scalars and bytes pass through by reference (never
    mutated in place). Dict key types (including optimizer integer param
    indices) and tuple shapes are preserved so the snapshot round-trips
    through ``torch.save`` / ``load_state_dict`` byte-equivalent to the live
    payload. Any leaf that is not a tensor, container, or immutable scalar
    raises :class:`StateTreeError` instead of coercing (a coerced leaf would
    save successfully and then fail closed at resume with a confusing digest
    mismatch; raising here names the type immediately).
    """
    import torch

    def _snap(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            # Detach + copy on the calling thread: the D2H copy is ordered
            # after the finished update on the default stream, so the clone
            # is a point-in-time copy; copy=True also isolates CPU tensors
            # (state_dict aliases live parameters even on CPU).
            return value.detach().to("cpu", copy=True)
        if isinstance(value, dict):
            return {key: _snap(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_snap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_snap(item) for item in value)
        if value is None or isinstance(value, (bool, str, int, float, bytes)):
            return value
        raise StateTreeError(
            f"unsupported state leaf type {type(value).__name__}; convert to tensor/scalar first"
        )

    if not isinstance(payload, Mapping):
        raise ContractError("checkpoint payload must be a mapping")
    missing: list[str] = [key for key in PAYLOAD_SECTION_KEYS if key not in payload]
    if len(missing) != 0:
        raise ContractError(f"checkpoint payload missing sections: {missing}")
    return {key: _snap(payload[key]) for key in payload}


class _PendingSave:
    """One submitted generation: its worker future plus its confirm hook."""

    __slots__ = (
        "description",
        "future",
        "generation",
        "main_spans",
        "on_done",
        "span_sink",
        "worker_timings",
    )

    def __init__(
        self,
        *,
        future: Any,
        generation: int,
        on_done: Any,
        description: str,
        span_sink: Any = None,
        main_spans: tuple[tuple[str, float, float], ...] = (),
        worker_timings: dict[str, float] | None = None,
    ) -> None:
        self.future = future
        self.generation = generation
        self.on_done = on_done
        self.description = description
        # Span sink is duck-typed (``.emit(stage, dur_ms, update)``): the
        # runtime layer never imports tracking, so no cycle can form.
        # ``main_spans`` are caller-measured (stage, t_start_s, dur_ms) on
        # the submitting thread (e.g. snapshot copies); ``worker_timings``
        # is the shared dict the wrapped job stamps on the worker.
        self.span_sink = span_sink
        self.main_spans = main_spans
        self.worker_timings: dict[str, float] = worker_timings if worker_timings is not None else {}


class _BackgroundCheckpointWriter:
    """Single-worker publisher of checkpoint generations (process lifetime).

    Owns one ``max_workers=1`` executor, so generations always land in
    submit order. Submit blocks (backpressure) while an OLDER generation is
    still in flight rather than queueing unbounded snapshots (each snapshot
    holds tens of MB of CPU clones); same-generation submits queue freely
    behind each other. All methods run on the single calling thread that
    owns training (the worker thread only runs the submitted job); the
    writer is not safe for concurrent submit/poll from multiple threads.
    Failure mode: worker exceptions surface on the calling thread via
    :meth:`poll`, :meth:`drain`, :meth:`wait`, or the submit-time
    backpressure wait (fail closed at the update boundary); the failed
    generation's ``on_done`` never fires, so the mirror never announces
    bytes that failed to land.
    """

    def __init__(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt-publish")
        self._pending: list[_PendingSave] = []
        self._closed = False

    def submit(
        self,
        job: Any,
        *,
        generation: int,
        on_done: Any = None,
        description: str = "checkpoint",
        span_sink: Any = None,
        main_spans: tuple[tuple[str, float, float], ...] = (),
    ) -> Any:
        """Queue ``job`` (zero-arg callable) behind its generation's peers.

        Blocks until every OLDER generation has landed (backpressure: never
        hold two generations in flight); raises that older generation's
        worker error instead of accepting new work on a broken writer.
        Returns the worker future; the caller owns firing completed hooks
        via :meth:`poll`/``drain``. ``span_sink`` (plus caller-measured
        ``main_spans``) emits one row per stage when the generation lands;
        without a sink nothing is recorded and behavior is unchanged.
        """
        import time

        if self._closed:
            raise ContractError("background checkpoint writer is closed")
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise ContractError(
                f"checkpoint generation must be a non-negative int, got {generation!r}"
            )
        for older in [pend for pend in self._pending if pend.generation < generation]:
            self._remove_completed(older, wait=True)
        # Shared timings dict (not the pend: the worker can start the job
        # before _PendingSave exists; the dict object is shared either way,
        # so no stamp is lost to the race).
        timings: dict[str, float] = {}

        def _timed_job() -> Any:
            # Bounds the true serialization window (queue wait excluded:
            # with max_workers=1 the job starts when the previous
            # generation lands, so queue time is backpressure, not work).
            timings["start_s"] = time.time()
            try:
                return job()
            finally:
                timings["end_s"] = time.time()

        future = self._executor.submit(_timed_job)
        pend = _PendingSave(
            future=future,
            generation=generation,
            on_done=on_done,
            description=description,
            span_sink=span_sink,
            main_spans=main_spans,
            worker_timings=timings,
        )
        self._pending.append(pend)
        return future

    def poll(self) -> None:
        """Fire hooks for landed generations; raise the first worker error.

        Cheap when idle (a ``done()`` check per pending generation). Raises
        the worker exception on the calling thread without firing that
        generation's hook; later generations stay queued for the next poll.
        """
        for pend in list(self._pending):
            if pend.future.done():
                self._remove_completed(pend, wait=False)

    def wait(self, future: Any) -> None:
        """Block until one submitted future lands, firing its hook on success.

        Removes the generation and runs its ``on_done`` (the file is on
        disk before this returns, so sync callers observe publish-then-hook
        exactly once through this path and never again via poll). Raises
        the worker error without firing the hook; unknown futures raise
        :class:`ContractError` (never silently pass).
        """
        for pend in list(self._pending):
            if pend.future is future:
                self._remove_completed(pend, wait=True)
                return
        raise ContractError("background checkpoint wait on unknown future")

    def drain(self) -> None:
        """Block until every pending generation lands, firing hooks in order."""
        while len(self._pending) > 0:
            self._remove_completed(self._pending[0], wait=True)

    def close(self) -> None:
        """Drain every pending generation, then shut the worker down.

        Idempotent. Raises a pending worker error instead of dropping it
        (a close that swallowed a failed save would resume from a
        generation the mirror announced but the disk never saw).
        """
        if self._closed:
            return
        try:
            self.drain()
        finally:
            self._closed = True
            self._executor.shutdown(wait=True)

    def _remove_completed(self, pend: _PendingSave, *, wait: bool) -> None:
        """Drop one generation from the queue, raising or firing as due."""
        if wait:
            exc = pend.future.exception()
        else:
            if not pend.future.done():
                return
            exc = pend.future.exception()
        try:
            self._pending.remove(pend)
        except ValueError:
            raise ContractError(
                f"background checkpoint generation already settled: {pend.description}"
            ) from None
        if exc is not None:
            raise exc
        if pend.on_done is not None:
            pend.on_done()
        sink = pend.span_sink
        if sink is not None:
            # Spans emit only on success (a failed generation raises above;
            # its cost is visible as the stall, not as spans). Main-thread
            # spans first (submit order), then the worker serialization
            # window. Sink failures must never break training: the spans
            # are observer-only, so a best-effort emit that raises is
            # swallowed here while the save itself already landed.
            try:
                for stage, t_start_s, dur_ms in pend.main_spans:
                    sink.emit(
                        stage=stage, dur_ms=dur_ms, update=pend.generation, t_start_s=t_start_s
                    )
                start_s = pend.worker_timings.get("start_s")
                end_s = pend.worker_timings.get("end_s")
                if start_s is not None and end_s is not None and end_s >= start_s:
                    sink.emit(
                        stage=f"worker:{pend.description}",
                        dur_ms=(end_s - start_s) * 1000.0,
                        update=pend.generation,
                        t_start_s=start_s,
                    )
            except Exception:
                pass
