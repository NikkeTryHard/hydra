from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from hydra_learner.hydra_logging import JsonlLogger
from hydra_learner.system_telemetry import sample_resources, snapshot_metrics

_CLK_TCK = os.sysconf(os.sysconf_names["SC_CLK_TCK"])


class PhaseTelemetry:
    def __init__(self, path: Path, interval_s: float = 1.0) -> None:
        self._logger = JsonlLogger(path)
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._phase = "startup"
        self._global_step = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="phase-telemetry", daemon=True)
        self._last_thread_ticks = _thread_ticks()
        self._last_time = time.perf_counter()

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._logger.close()

    def set_phase(self, phase: str, global_step: int) -> None:
        with self._lock:
            self._phase = phase
            self._global_step = global_step

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            now = time.perf_counter()
            thread_ticks = _thread_ticks()
            with self._lock:
                phase = self._phase
                global_step = self._global_step
            elapsed = max(now - self._last_time, 1.0e-9)
            deltas = [
                (tid, name, ticks - self._last_thread_ticks.get(tid, (ticks, name))[0])
                for tid, (ticks, name) in thread_ticks.items()
            ]
            deltas.sort(key=lambda item: item[2], reverse=True)
            top_threads = [
                {"tid": tid, "name": name, "cpu_percent": (ticks / _CLK_TCK) * 100.0 / elapsed}
                for tid, name, ticks in deltas[:8]
                if ticks > 0
            ]
            self._last_thread_ticks = thread_ticks
            self._last_time = now
            resources = sample_resources()
            self._logger.write(
                "phase_sample",
                {
                    "perf_counter": now,
                    "phase": phase,
                    "global_step": global_step,
                    "top_threads": top_threads,
                    **snapshot_metrics("resources", resources),
                },
            )


def _thread_ticks() -> dict[int, tuple[int, str]]:
    out: dict[int, tuple[int, str]] = {}
    task_dir = Path("/proc/self/task")
    for path in task_dir.iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text(encoding="utf-8").split()
        except OSError:
            continue
        out[int(path.name)] = (int(fields[13]) + int(fields[14]), fields[1].strip("()"))
    return out
