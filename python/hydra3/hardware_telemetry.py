"""Opt-in, timestamped Linux CPU and NVIDIA board utilization samples."""

import ctypes as c
import json
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from time import perf_counter_ns

_INTERVAL = 0.1
_CPU_STAT = Path("/proc/stat")


class _Utilization(c.Structure):
    _fields_ = [("gpu", c.c_uint), ("memory", c.c_uint)]


def _check(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"NVML {operation} failed (status {status})")


@contextmanager
def _gpu(uuid: str | None) -> Generator[tuple[c.CDLL, c.c_void_p] | None]:
    if uuid is None:
        yield None
        return
    lib = c.CDLL("libnvidia-ml.so.1")
    _check(lib.nvmlInit_v2(), "initialization")
    try:
        handle = c.c_void_p()
        _check(
            lib.nvmlDeviceGetHandleByUUID(uuid.encode("ascii"), c.byref(handle)),
            "device lookup",
        )
        yield lib, handle
    finally:
        _check(lib.nvmlShutdown(), "shutdown")


def _cpu_ticks() -> dict[str, tuple[int, int]]:
    result = {}
    with _CPU_STAT.open() as stat:
        for line in stat:
            if not line.startswith("cpu"):
                break
            name, *fields = line.split(maxsplit=9)
            ticks = [int(value) for value in fields[:8]]
            result[name] = (sum(ticks), ticks[3] + ticks[4])
    return result


def _read_gpu(lib: c.CDLL, handle: c.c_void_p) -> dict[str, int]:
    util = _Utilization()
    _check(lib.nvmlDeviceGetUtilizationRates(handle, c.byref(util)), "utilization")
    return {"busy_pct": util.gpu, "memory_busy_pct": util.memory}


@contextmanager
def sample_hardware(generation: int, gpu_uuid: str | None) -> Generator[None]:
    """Write samples to stderr without inserting CUDA synchronization into PPO."""
    with _gpu(gpu_uuid) as board:
        previous = _cpu_ticks()
        stop = Event()
        failures: list[Exception] = []

        def sample() -> None:
            nonlocal previous
            try:
                while not stop.wait(_INTERVAL):
                    timestamp = perf_counter_ns()
                    current = _cpu_ticks()
                    cpu: dict[str, float | None] = {}
                    for name, (total, idle) in current.items():
                        old_total, old_idle = previous.get(name, (total, idle))
                        elapsed = total - old_total
                        cpu[name] = (
                            round(100 * (1 - (idle - old_idle) / elapsed), 1)
                            if elapsed > 0
                            else None
                        )
                    previous = current
                    event = {
                        "event": "hardware",
                        "generation": generation,
                        "timestamp_ns": timestamp,
                        "cpu_busy_pct": cpu,
                        "gpu": _read_gpu(*board) if board is not None else None,
                    }
                    print(json.dumps(event), file=sys.stderr, flush=True)
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append(exc)

        thread = Thread(target=sample, name="hydra3-hardware")
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
        if failures:
            raise RuntimeError("hardware telemetry stopped") from failures[0]
