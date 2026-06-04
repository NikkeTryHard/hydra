from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

_CLK_TCK = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
_CPU_COUNT = os.cpu_count() or 1


@dataclass(frozen=True)
class ResourceSnapshot:
    perf_counter: float
    process_ticks: int
    total_ticks: int
    read_bytes: int
    write_bytes: int
    gpu_util_percent: int | None
    gpu_mem_used_mb: int | None
    gpu_mem_free_mb: int | None


def sample_resources() -> ResourceSnapshot:
    gpu_util, gpu_used, gpu_free = _read_gpu_telemetry()
    process_ticks = _read_process_ticks()
    total_ticks = _read_total_cpu_ticks()
    read_bytes, write_bytes = _read_process_io_bytes()
    return ResourceSnapshot(
        perf_counter=time.perf_counter(),
        process_ticks=process_ticks,
        total_ticks=total_ticks,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        gpu_util_percent=gpu_util,
        gpu_mem_used_mb=gpu_used,
        gpu_mem_free_mb=gpu_free,
    )


def resource_delta_metrics(
    prefix: str, start: ResourceSnapshot, end: ResourceSnapshot
) -> dict[str, float | int | None]:
    elapsed = max(end.perf_counter - start.perf_counter, 0.0)
    total_delta = end.total_ticks - start.total_ticks
    process_delta = end.process_ticks - start.process_ticks
    cpu_percent = 0.0
    if total_delta > 0:
        cpu_percent = (process_delta / total_delta) * _CPU_COUNT * 100.0
    read_mb_s = 0.0
    write_mb_s = 0.0
    if elapsed > 0.0:
        read_mb_s = max(0, end.read_bytes - start.read_bytes) / (1024.0 * 1024.0 * elapsed)
        write_mb_s = max(0, end.write_bytes - start.write_bytes) / (1024.0 * 1024.0 * elapsed)
    return {
        f"{prefix}/elapsed_ms": elapsed * 1000.0,
        f"{prefix}/cpu_percent": cpu_percent,
        f"{prefix}/disk_read_mb_s": read_mb_s,
        f"{prefix}/disk_write_mb_s": write_mb_s,
        f"{prefix}/gpu_util_percent": end.gpu_util_percent,
        f"{prefix}/gpu_mem_used_mb": end.gpu_mem_used_mb,
        f"{prefix}/gpu_mem_free_mb": end.gpu_mem_free_mb,
    }


def snapshot_metrics(prefix: str, snapshot: ResourceSnapshot) -> dict[str, float | int | None]:
    return {
        f"{prefix}/gpu_util_percent": snapshot.gpu_util_percent,
        f"{prefix}/gpu_mem_used_mb": snapshot.gpu_mem_used_mb,
        f"{prefix}/gpu_mem_free_mb": snapshot.gpu_mem_free_mb,
        f"{prefix}/process_read_bytes": snapshot.read_bytes,
        f"{prefix}/process_write_bytes": snapshot.write_bytes,
    }


def _read_process_ticks() -> int:
    fields = _read_text("/proc/self/stat").split()
    return int(fields[13]) + int(fields[14])


def _read_total_cpu_ticks() -> int:
    line = _read_text("/proc/stat").splitlines()[0]
    return sum(int(value) for value in line.split()[1:])


def _read_process_io_bytes() -> tuple[int, int]:
    read_bytes = 0
    write_bytes = 0
    for line in _read_text("/proc/self/io").splitlines():
        name, value = line.split(":", 1)
        if name == "read_bytes":
            read_bytes = int(value)
        elif name == "write_bytes":
            write_bytes = int(value)
    return read_bytes, write_bytes


def _read_gpu_telemetry() -> tuple[int | None, int | None, int | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None, None
    if result.returncode != 0:
        return None, None, None
    first = result.stdout.splitlines()[0] if result.stdout else ""
    parts = [part.strip() for part in first.split(",")]
    if len(parts) != 3:
        return None, None, None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None, None, None


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")
