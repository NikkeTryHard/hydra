from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from types import TracebackType
from typing import Any, ClassVar, Self, cast

import torch

from hydra_learner.data.shard_contracts import ACTION_SPACE, ManifestSummary, PolicyBatch


@dataclass(frozen=True)
class PinnedPolicyBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    legal_mask: torch.Tensor
    value_target: torch.Tensor
    grp_target: torch.Tensor
    oracle_target: torch.Tensor
    oracle_target_mask: torch.Tensor
    tenpai: torch.Tensor
    opp_next: torch.Tensor
    danger: torch.Tensor
    danger_mask: torch.Tensor
    score_pdf: torch.Tensor
    score_cdf: torch.Tensor
    rows: int

    def as_policy_batch(self) -> PolicyBatch:
        rows = self.rows
        return PolicyBatch(
            obs=self.obs[:rows].numpy(),
            actions=self.actions[:rows].numpy(),
            legal_mask=self.legal_mask[:rows].numpy(),
            value_target=self.value_target[:rows].numpy(),
            grp_target=self.grp_target[:rows].numpy(),
            oracle_target=self.oracle_target[:rows].numpy(),
            oracle_target_mask=self.oracle_target_mask[:rows].numpy(),
            tenpai=self.tenpai[:rows].numpy(),
            opp_next=self.opp_next[:rows].numpy(),
            danger=self.danger[:rows].numpy(),
            danger_mask=self.danger_mask[:rows].numpy(),
            score_pdf=self.score_pdf[:rows].numpy(),
            score_cdf=self.score_cdf[:rows].numpy(),
            safety_target=None,
            safety_mask=None,
        )


@dataclass(frozen=True)
class BuildProgress:
    manifest_path: Path | None
    complete: bool
    build_seconds: float
    loaded_games: int = 0
    skipped_games: int = 0
    samples: int = 0
    batches: int = 0
    max_games_reached: bool = False
    max_samples_reached: bool = False


def build_progress_json(progress: BuildProgress) -> dict[str, object]:
    return {
        "manifest_path": None if progress.manifest_path is None else str(progress.manifest_path),
        "complete": progress.complete,
        "build_seconds": progress.build_seconds,
        "loaded_games": progress.loaded_games,
        "skipped_games": progress.skipped_games,
        "samples": progress.samples,
        "batches": progress.batches,
        "max_games_reached": progress.max_games_reached,
        "max_samples_reached": progress.max_samples_reached,
    }


@dataclass(frozen=True)
class RawMjaiBridgeStats:
    open_count: int = 0
    open_scan_plan_ms: float = 0.0
    last_next_fill_ms: float = 0.0
    last_queue_wait_ms: float = 0.0
    last_bytes_filled: int = 0
    last_games_consumed: int = 0


def _bridge_stats_from_object(stats: Any) -> RawMjaiBridgeStats:
    return RawMjaiBridgeStats(
        open_count=int(stats.open_count),
        open_scan_plan_ms=float(stats.open_scan_plan_ms),
        last_next_fill_ms=float(stats.last_next_fill_ms),
        last_queue_wait_ms=float(stats.last_queue_wait_ms),
        last_bytes_filled=int(stats.last_bytes_filled),
        last_games_consumed=int(stats.last_games_consumed),
    )


@dataclass(frozen=True)
class RawMjaiPinnedResult:
    rows: int
    loaded_games: int
    skipped_games: int
    samples: int
    batches: int
    max_games_reached: bool
    max_samples_reached: bool
    stats: RawMjaiBridgeStats


def _pinned_result_from_object(result: Any) -> RawMjaiPinnedResult:
    stats = _bridge_stats_from_object(result.stats)
    return RawMjaiPinnedResult(
        rows=int(result.rows),
        loaded_games=int(result.loaded_games),
        skipped_games=int(result.skipped_games),
        samples=int(result.samples),
        batches=int(result.batches),
        max_games_reached=bool(result.max_games_reached),
        max_samples_reached=bool(result.max_samples_reached),
        stats=stats,
    )


@dataclass(frozen=True)
class RawMjaiPinnedFilled:
    slot_index: int
    batch: PinnedPolicyBatch
    result: RawMjaiPinnedResult
    fill_ms: float


@dataclass(frozen=True)
class RawMjaiPinnedQueueStats:
    ready_wait_ms_total: float = 0.0
    ready_wait_count: int = 0
    producer_fill_ms_total: float = 0.0
    produced_batches: int = 0
    producer_free_wait_ms_total: float = 0.0
    producer_free_wait_count: int = 0
    producer_error: str | None = None
    ready_queue_size: int = 0
    free_queue_size: int = 0

    @property
    def mean_ready_wait_ms(self) -> float:
        return self.ready_wait_ms_total / self.ready_wait_count if self.ready_wait_count else 0.0

    @property
    def mean_producer_fill_ms(self) -> float:
        return self.producer_fill_ms_total / self.produced_batches if self.produced_batches else 0.0

    @property
    def mean_producer_free_wait_ms(self) -> float:
        return (
            self.producer_free_wait_ms_total / self.producer_free_wait_count if self.producer_free_wait_count else 0.0
        )

    def with_queue_sizes(self, *, ready_queue_size: int, free_queue_size: int) -> RawMjaiPinnedQueueStats:
        return RawMjaiPinnedQueueStats(
            ready_wait_ms_total=self.ready_wait_ms_total,
            ready_wait_count=self.ready_wait_count,
            producer_fill_ms_total=self.producer_fill_ms_total,
            produced_batches=self.produced_batches,
            producer_free_wait_ms_total=self.producer_free_wait_ms_total,
            producer_free_wait_count=self.producer_free_wait_count,
            producer_error=self.producer_error,
            ready_queue_size=ready_queue_size,
            free_queue_size=free_queue_size,
        )


def default_raw_mjai_pyo3_library_path() -> Path:
    env_path = os.environ.get("HYDRA_RAW_MJAI_PYO3_LIB")
    if env_path:
        return Path(env_path)
    repo_root = Path(__file__).resolve().parents[6]
    release_path = repo_root / "target" / "release" / "libhydra_raw_mjai_pyo3.so"
    if release_path.exists():
        return release_path
    return repo_root / "target" / "debug" / "libhydra_raw_mjai_pyo3.so"


def _load_raw_mjai_module(path: Path) -> Any:
    if not path.exists():
        raise ImportError(
            "raw MJAI pinned PyO3 extension is required for raw-MJAI Python training but was not found at "
            f"{path}. Build it with `pixi run cargo build -p hydra-raw-mjai-pyo3 --release --quiet`, "
            "set HYDRA_RAW_MJAI_PYO3_LIB, pass --raw-mjai-pyo3-lib, or select "
            "--raw-mjai-transport stdout for the subprocess fallback."
        )
    spec = importlib.util.spec_from_file_location("hydra_raw_mjai_pyo3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load raw MJAI extension from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hydra_raw_mjai_pyo3"] = module
    spec.loader.exec_module(module)
    return module


class RawMjaiPinnedStream:
    _stream_override: ClassVar[Any | None] = None

    @classmethod
    def _set_stream_override_for_tests(cls, stream_cls: Any | None) -> None:
        cls._stream_override = stream_cls

    def __init__(
        self,
        *,
        data_dirs: Sequence[Path],
        batch_size: int,
        queue_bound: int,
        worker_threads: int,
        max_games: int | None,
        max_samples: int | None,
        train_fraction: float,
        augment: bool,
        split: str,
        library_path: Path,
        ring_size: int,
        skip_games: int = 0,
        close_timeout_s: float = 30.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("raw MJAI pinned batch size must be > 0")
        if ring_size < 2:
            raise ValueError("raw MJAI pinned ring size must be >= 2")
        stream_cls = self._stream_override
        if stream_cls is None:
            module = cast(Any, _load_raw_mjai_module(library_path))
            stream_cls = module.RawMjaiStream
        print(
            "raw_mjai_pinned_load "
            f"library={library_path} stream_cls={getattr(stream_cls, '__name__', type(stream_cls).__name__)}",
            file=sys.stderr,
            flush=True,
        )
        self.data_dirs = tuple(data_dirs)
        self._close_timeout_s = close_timeout_s
        self.batch_size = batch_size
        if not self.data_dirs:
            raise ValueError("raw MJAI pinned stream requires at least one data dir")
        self._stream = stream_cls(
            [str(path) for path in self.data_dirs],
            batch_size=batch_size,
            train_fraction=train_fraction,
            worker_threads=worker_threads,
            queue_bound=queue_bound,
            max_games=max_games,
            max_samples=max_samples,
            skip_games=skip_games,
            augment=augment,
            split=split,
        )
        initial_stats = _bridge_stats_from_object(self._stream.stats())
        print(
            "raw_mjai_pinned_open "
            f"dirs={len(self.data_dirs)} batch={batch_size} ring={ring_size} workers={worker_threads} "
            f"queue_bound={queue_bound} max_games={max_games} max_samples={max_samples} "
            f"skip_games={skip_games} split={split} open_scan_plan_ms={initial_stats.open_scan_plan_ms:.3f}",
            file=sys.stderr,
            flush=True,
        )
        self._ring = [_allocate_pinned_policy_batch(batch_size, pin_memory=True) for _ in range(ring_size)]
        for batch in self._ring:
            _validate_pinned_policy_batch(batch)
        self._events: list[torch.cuda.Event | None] = [None for _ in range(ring_size)]
        self._free_slots: Queue[int | None] = Queue(maxsize=ring_size)
        self._ready: Queue[RawMjaiPinnedFilled | BaseException | None] = Queue(maxsize=ring_size)
        for index in range(ring_size):
            self._free_slots.put(index)
        self._stop = threading.Event()
        self._debug = os.environ.get("HYDRA_RAW_MJAI_PINNED_DEBUG") == "1"
        self._progress = BuildProgress(manifest_path=None, complete=False, build_seconds=0.0)
        self._stats = initial_stats
        self._queue_stats = RawMjaiPinnedQueueStats()
        self._producer_thread: threading.Thread | None = threading.Thread(
            target=self._producer_main,
            name="raw-mjai-pyo3-pinned-producer",
            daemon=True,
        )
        self._producer_thread.start()

    @property
    def manifest_path(self) -> Path:
        return self.data_dirs[0]

    @property
    def manifest_summary(self) -> ManifestSummary | None:
        if self._progress.samples == 0:
            return None
        return ManifestSummary(
            path=self.data_dirs[0],
            train_samples=self._progress.samples,
            validation_samples=0,
            shard_count=0,
            record_size=0,
            feature_flags=0,
        )

    def next_batch(self) -> tuple[PinnedPolicyBatch, float]:
        started = time.perf_counter()
        item = self._ready.get()
        fetch_ms = (time.perf_counter() - started) * 1000.0
        self._queue_stats = RawMjaiPinnedQueueStats(
            ready_wait_ms_total=self._queue_stats.ready_wait_ms_total + fetch_ms,
            ready_wait_count=self._queue_stats.ready_wait_count + 1,
            producer_fill_ms_total=self._queue_stats.producer_fill_ms_total,
            produced_batches=self._queue_stats.produced_batches,
            producer_free_wait_ms_total=self._queue_stats.producer_free_wait_ms_total,
            producer_free_wait_count=self._queue_stats.producer_free_wait_count,
            producer_error=self._queue_stats.producer_error,
        )
        if item is None:
            raise StopIteration("raw MJAI pinned stream exhausted")
        if isinstance(item, BaseException):
            print(
                "raw_mjai_pinned_next_error "
                f"wait_ms={fetch_ms:.3f} error={type(item).__name__}: {item} stats={self._queue_stats}",
                file=sys.stderr,
                flush=True,
            )
            raise item
        self._stats = item.result.stats
        print(
            "raw_mjai_pinned_next "
            f"wait_ms={fetch_ms:.3f} rows={item.result.rows} loaded_games={item.result.loaded_games} "
            f"skipped_games={item.result.skipped_games} samples={item.result.samples} batches={item.result.batches} "
            f"fill_ms={item.fill_ms:.3f} bridge_fill_ms={item.result.stats.last_next_fill_ms:.3f} "
            f"queue_wait_ms={item.result.stats.last_queue_wait_ms:.3f} "
            f"games_consumed={item.result.stats.last_games_consumed}",
            file=sys.stderr,
            flush=True,
        )
        self._progress = BuildProgress(
            manifest_path=None,
            complete=False,
            build_seconds=item.result.stats.open_scan_plan_ms / 1000.0,
            loaded_games=item.result.loaded_games,
            skipped_games=item.result.skipped_games,
            samples=item.result.samples,
            batches=item.result.batches,
            max_games_reached=item.result.max_games_reached,
            max_samples_reached=item.result.max_samples_reached,
        )
        return _slice_pinned_policy_batch(item.batch, item.result.rows), fetch_ms

    def mark_inflight(self, batch: PinnedPolicyBatch) -> None:
        base_ptr = _base_data_ptr(batch.obs)
        for index, candidate in enumerate(self._ring):
            if candidate.obs.data_ptr() == base_ptr:
                event = torch.cuda.Event()
                event.record()
                self._events[index] = event
                self._return_slot_when_ready(index)
                return
        raise ValueError("pinned batch does not belong to raw MJAI ring")

    def progress(self) -> BuildProgress:
        return self._progress

    def bridge_stats(self) -> RawMjaiBridgeStats:
        return self._stats

    def queue_stats(self) -> RawMjaiPinnedQueueStats:
        return self._queue_stats.with_queue_sizes(
            ready_queue_size=self._ready.qsize(),
            free_queue_size=self._free_slots.qsize(),
        )

    def _debug_log(self, message: str) -> None:
        if self._debug:
            print(f"[raw-mjai-pyo3-pinned] {message}", file=sys.stderr, flush=True)

    def close(self) -> None:
        self._debug_log("close begin")
        self._stop.set()
        self._free_slots.put(None)
        thread = self._producer_thread
        if thread is not None:
            thread.join(timeout=self._close_timeout_s)
            if thread.is_alive():
                raise RuntimeError(f"raw MJAI pinned producer did not stop within {self._close_timeout_s:g}s")
            self._producer_thread = None
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()
        self._debug_log("close end")

    def _producer_main(self) -> None:
        try:
            while not self._stop.is_set():
                self._debug_log("producer wait free slot")
                wait_started = time.perf_counter()
                slot = self._free_slots.get()
                free_wait_ms = (time.perf_counter() - wait_started) * 1000.0
                if slot is None or self._stop.is_set():
                    return
                batch = self._ring[slot]
                self._debug_log(f"producer fill slot={slot}")
                fill_started = time.perf_counter()
                result = _pinned_result_from_object(
                    self._stream.next_into(
                        batch.obs.data_ptr(),
                        batch.actions.data_ptr(),
                        batch.legal_mask.data_ptr(),
                        batch.value_target.data_ptr(),
                        batch.grp_target.data_ptr(),
                        batch.oracle_target.data_ptr(),
                        batch.oracle_target_mask.data_ptr(),
                        batch.tenpai.data_ptr(),
                        batch.opp_next.data_ptr(),
                        batch.danger.data_ptr(),
                        batch.danger_mask.data_ptr(),
                        batch.score_pdf.data_ptr(),
                        batch.score_cdf.data_ptr(),
                        batch.rows,
                    )
                )
                fill_ms = (time.perf_counter() - fill_started) * 1000.0
                self._debug_log(f"producer filled slot={slot} rows={result.rows} fill_ms={fill_ms:.3f}")
                print(
                    "raw_mjai_pinned_producer_fill "
                    f"slot={slot} rows={result.rows} fill_ms={fill_ms:.3f} "
                    f"loaded_games={result.loaded_games} skipped_games={result.skipped_games} "
                    f"samples={result.samples} batches={result.batches} "
                    f"bridge_fill_ms={result.stats.last_next_fill_ms:.3f} "
                    f"queue_wait_ms={result.stats.last_queue_wait_ms:.3f} "
                    f"games_consumed={result.stats.last_games_consumed}",
                    file=sys.stderr,
                    flush=True,
                )
                self._stats = result.stats
                self._queue_stats = RawMjaiPinnedQueueStats(
                    ready_wait_ms_total=self._queue_stats.ready_wait_ms_total,
                    ready_wait_count=self._queue_stats.ready_wait_count,
                    producer_fill_ms_total=self._queue_stats.producer_fill_ms_total + fill_ms,
                    produced_batches=self._queue_stats.produced_batches + (1 if result.rows else 0),
                    producer_free_wait_ms_total=self._queue_stats.producer_free_wait_ms_total + free_wait_ms,
                    producer_free_wait_count=self._queue_stats.producer_free_wait_count + 1,
                    producer_error=self._queue_stats.producer_error,
                )
                if result.rows == 0:
                    self._ready.put(None)
                    return
                self._ready.put(RawMjaiPinnedFilled(slot, batch, result, fill_ms))
                self._debug_log(f"producer ready slot={slot}")
        except BaseException as exc:
            self._debug_log(f"producer error {type(exc).__name__}: {exc}")
            self._queue_stats = RawMjaiPinnedQueueStats(
                ready_wait_ms_total=self._queue_stats.ready_wait_ms_total,
                ready_wait_count=self._queue_stats.ready_wait_count,
                producer_fill_ms_total=self._queue_stats.producer_fill_ms_total,
                produced_batches=self._queue_stats.produced_batches,
                producer_free_wait_ms_total=self._queue_stats.producer_free_wait_ms_total,
                producer_free_wait_count=self._queue_stats.producer_free_wait_count,
                producer_error=f"{type(exc).__name__}: {exc}",
            )
            self._ready.put(exc)

    def _return_slot_when_ready(self, index: int) -> None:
        event = self._events[index]
        if event is None or event.query():
            self._events[index] = None
            self._free_slots.put(index)
            return
        thread = threading.Thread(
            target=self._wait_event_then_free,
            args=(index, event),
            name="raw-mjai-pyo3-pinned-slot-return",
            daemon=True,
        )
        thread.start()

    def _wait_event_then_free(self, index: int, event: torch.cuda.Event) -> None:
        self._debug_log(f"slot return wait index={index}")
        event.synchronize()
        self._events[index] = None
        if not self._stop.is_set():
            self._debug_log(f"slot returned index={index}")
            self._free_slots.put(index)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, _exc_type: type[BaseException] | None, _exc: BaseException | None, _tb: TracebackType | None
    ) -> None:
        self.close()


def _allocate_pinned_policy_batch(rows: int, *, pin_memory: bool) -> PinnedPolicyBatch:
    return PinnedPolicyBatch(
        obs=torch.empty((rows, 192, 34), dtype=torch.float32, pin_memory=pin_memory),
        actions=torch.empty((rows,), dtype=torch.int64, pin_memory=pin_memory),
        legal_mask=torch.empty((rows, ACTION_SPACE), dtype=torch.bool, pin_memory=pin_memory),
        value_target=torch.empty((rows,), dtype=torch.float32, pin_memory=pin_memory),
        grp_target=torch.empty((rows, 24), dtype=torch.float32, pin_memory=pin_memory),
        oracle_target=torch.empty((rows, 4), dtype=torch.float32, pin_memory=pin_memory),
        oracle_target_mask=torch.empty((rows,), dtype=torch.float32, pin_memory=pin_memory),
        tenpai=torch.empty((rows, 3), dtype=torch.float32, pin_memory=pin_memory),
        opp_next=torch.empty((rows, 102), dtype=torch.float32, pin_memory=pin_memory),
        danger=torch.empty((rows, 102), dtype=torch.float32, pin_memory=pin_memory),
        danger_mask=torch.empty((rows, 102), dtype=torch.float32, pin_memory=pin_memory),
        score_pdf=torch.empty((rows, 64), dtype=torch.float32, pin_memory=pin_memory),
        score_cdf=torch.empty((rows, 64), dtype=torch.float32, pin_memory=pin_memory),
        rows=rows,
    )


def _slice_pinned_policy_batch(batch: PinnedPolicyBatch, rows: int) -> PinnedPolicyBatch:
    return PinnedPolicyBatch(
        obs=batch.obs[:rows],
        actions=batch.actions[:rows],
        legal_mask=batch.legal_mask[:rows],
        value_target=batch.value_target[:rows],
        grp_target=batch.grp_target[:rows],
        oracle_target=batch.oracle_target[:rows],
        oracle_target_mask=batch.oracle_target_mask[:rows],
        tenpai=batch.tenpai[:rows],
        opp_next=batch.opp_next[:rows],
        danger=batch.danger[:rows],
        danger_mask=batch.danger_mask[:rows],
        score_pdf=batch.score_pdf[:rows],
        score_cdf=batch.score_cdf[:rows],
        rows=rows,
    )


def _base_data_ptr(tensor: torch.Tensor) -> int:
    base = tensor
    while isinstance(base._base, torch.Tensor):
        base = base._base
    return base.data_ptr()


def _validate_pinned_tensor(tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, name: str) -> None:
    if tuple(tensor.shape) != shape or tensor.dtype != dtype or not tensor.is_contiguous():
        raise ValueError(f"raw MJAI pinned {name} mismatch: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    if tensor.device.type != "cpu":
        raise ValueError(f"raw MJAI pinned {name} must be a CPU tensor")
    if not tensor.is_pinned():
        raise ValueError(f"raw MJAI pinned {name} must be pinned host memory")


def _validate_pinned_policy_batch(batch: PinnedPolicyBatch) -> None:
    _validate_pinned_tensor(batch.obs, (batch.rows, 192, 34), torch.float32, "obs")
    _validate_pinned_tensor(batch.actions, (batch.rows,), torch.int64, "actions")
    _validate_pinned_tensor(batch.legal_mask, (batch.rows, ACTION_SPACE), torch.bool, "legal_mask")
    _validate_pinned_tensor(batch.value_target, (batch.rows,), torch.float32, "value_target")
    _validate_pinned_tensor(batch.grp_target, (batch.rows, 24), torch.float32, "grp_target")
    _validate_pinned_tensor(batch.oracle_target, (batch.rows, 4), torch.float32, "oracle_target")
    _validate_pinned_tensor(batch.oracle_target_mask, (batch.rows,), torch.float32, "oracle_target_mask")
    _validate_pinned_tensor(batch.tenpai, (batch.rows, 3), torch.float32, "tenpai")
    _validate_pinned_tensor(batch.opp_next, (batch.rows, 102), torch.float32, "opp_next")
    _validate_pinned_tensor(batch.danger, (batch.rows, 102), torch.float32, "danger")
    _validate_pinned_tensor(batch.danger_mask, (batch.rows, 102), torch.float32, "danger_mask")
    _validate_pinned_tensor(batch.score_pdf, (batch.rows, 64), torch.float32, "score_pdf")
    _validate_pinned_tensor(batch.score_cdf, (batch.rows, 64), torch.float32, "score_cdf")
