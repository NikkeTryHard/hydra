"""WP-14 verbose GPU/CPU sampler (opt-in, observer-only).

A daemon thread samples NVML device counters + per-process GPU/memory
shares, per-thread CPU deltas via psutil, and ``torch.cuda.memory_stats``
every 20-50ms and appends one JSON row per tick to
``logs/verbose-telemetry.jsonl`` under the run dir. Rows carry
``(run_id, global_update, microstep)`` join keys plus dual wall/monotonic
clocks (wall = wall-clock epoch seconds, not the mahjong tile wall) so
offline analysis can join them to ``feed-telemetry.jsonl`` rows
and ``train.log`` lines.

Honesty notes (read before analyzing the output):

- NVML utilization percentages refresh at driver cadence (~0.17-1s), so
  consecutive 20-50ms ticks legitimately repeat values. ``tick_ms``
  timestamps are exact; the values are stepped. Derive duty over windows,
  never from single ticks.
- Per-PID ``sm_util``/``mem_util`` are sampled shares, not additive
  accounting; power/energy/temperature/clocks are device-only (NVML
  exposes no per-process split). Per-PID GPU *memory bytes* come from
  the running-processes snapshot and are exact.
- CPU thread IDs are native TIDs; torch intra-op pools and BLAS threads
  are anonymous (no op mapping). DataLoader workers are separate
  processes and appear under ``cpu.children``.
- First-tick CPU deltas are null (baselines prime on the first tick).

Lifecycle mirrors :mod:`hydra2.tracking.clearml_mirror`: lazy SDK
imports inside the sampler thread only, ``HYDRA2_VERBOSE_TELEMETRY=1``
opt-in (default off), ``HYDRA2_VERBOSE_TELEMETRY_DISABLED=1`` kill-switch
wins, tiered warn-once degradation that never raises into training.

Bridge: the interval clamp (``resolve_interval_ms`` explicit arm) plus the
``TRACKING_SAMPLER_SCHEMA_VERSION`` / ``TRACKING_DEFAULT_INTERVAL_MS`` /
``TRACKING_VALID_INTERVALS_MS`` / ``TRACKING_GPU_DROP_AFTER_ERRORS`` tables
live on ``hydra2._native.contracts`` and decide first when built; the bodies
below are the byte-identical stale-``.so`` fallback. SDKs, threads, clocks,
and the warn/env arms stay Python.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal; oracle branch below decides
    _bridge_contracts = None  # type: ignore[assignment]

__all__ = [
    "DEFAULT_INTERVAL_MS",
    "VALID_INTERVALS_MS",
    "VerboseSampler",
    "is_enabled",
    "resolve_interval_ms",
]

#: Row schema version (bumped only on incompatible field changes).
_SCHEMA_RAW: int | None = getattr(_bridge_contracts, "TRACKING_SAMPLER_SCHEMA_VERSION", None)
SCHEMA_VERSION: int = _SCHEMA_RAW if _SCHEMA_RAW is not None else 1

#: Default tick cadence (ms). NVML util refreshes slower; see module notes.
_INTERVAL_RAW: int | None = getattr(_bridge_contracts, "TRACKING_DEFAULT_INTERVAL_MS", None)
DEFAULT_INTERVAL_MS: int = _INTERVAL_RAW if _INTERVAL_RAW is not None else 50

#: Allowed cadences; anything else warns and clamps to the default.
_VALID_RAW: tuple[int, ...] | None = getattr(_bridge_contracts, "TRACKING_VALID_INTERVALS_MS", None)
VALID_INTERVALS_MS: tuple[int, ...] = _VALID_RAW if _VALID_RAW is not None else (20, 50)

#: Consecutive NVML failures before the GPU group is dropped for the run.
_DROP_RAW: int | None = getattr(_bridge_contracts, "TRACKING_GPU_DROP_AFTER_ERRORS", None)
_GPU_DROP_AFTER_ERRORS: int = _DROP_RAW if _DROP_RAW is not None else 100

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def resolve_interval_ms(explicit: int | float | None = None) -> int:
    """Allowed cadence or the default (warns and clamps otherwise)."""
    gate = getattr(_bridge_contracts, "tracking_clamp_interval_ms", None)
    if gate is not None and explicit is not None:
        with contextlib.suppress(TypeError, ValueError):
            clamped_out: tuple[int, bool] = gate(explicit)
            clamped, warned = clamped_out
            if warned:
                warnings.warn(
                    f"verbose sampler interval {explicit!r} not in {VALID_INTERVALS_MS}; "
                    f"using {DEFAULT_INTERVAL_MS}ms",
                    stacklevel=2,
                )
            return clamped
    if explicit is not None:
        try:
            value = int(explicit)
            if value in VALID_INTERVALS_MS:
                return value
        except Exception:
            pass
        warnings.warn(
            f"verbose sampler interval {explicit!r} not in {VALID_INTERVALS_MS}; "
            f"using {DEFAULT_INTERVAL_MS}ms",
            stacklevel=2,
        )
        return DEFAULT_INTERVAL_MS
    try:
        return resolve_interval_ms(int(os.environ.get("HYDRA2_VERBOSE_INTERVAL_MS", "")))
    except Exception:
        return DEFAULT_INTERVAL_MS


def is_enabled(*, explicit: bool | None = None) -> bool:
    """Opt-in only; kill-switch wins. Never raises."""
    try:
        if explicit is False or _env_truthy("HYDRA2_VERBOSE_TELEMETRY_DISABLED"):
            return False
        return explicit is True or _env_truthy("HYDRA2_VERBOSE_TELEMETRY")
    except Exception:
        return False


def _null_row(
    *,
    run_id: str | None,
    run_digest: str | None,
    counters: tuple[int, int],
    backends: dict[str, str],
    errors: dict[str, int],
) -> dict[str, Any]:
    now_wall = time.time()
    return {
        "v": SCHEMA_VERSION,
        "t_wall_s": now_wall,
        "t_mono_ns": time.monotonic_ns(),
        "run_id": run_id,
        "run_digest": run_digest,
        "global_update": counters[0],
        "microstep": counters[1],
        "gpu": None,
        "proc": None,
        "cpu": None,
        "torch": None,
        "backends": backends,
        "errors": errors,
    }


class VerboseSampler:
    """Daemon-thread NVML/psutil/torch sampler; warn-only on failure."""

    def __init__(
        self,
        *,
        enabled: bool,
        sink_path: Path | str,
        interval_ms: int | float | None = None,
        run_id: str | None = None,
        run_digest: str | None = None,
        counters_fn: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        try:
            self._enabled = enabled
        except Exception:
            self._enabled = False
        try:
            self._sink_path = Path(sink_path)
        except Exception:
            self._sink_path = Path("verbose-telemetry.jsonl")
            self._enabled = False
        try:
            self._interval_s = resolve_interval_ms(interval_ms) / 1000.0
        except Exception:
            self._interval_s = DEFAULT_INTERVAL_MS / 1000.0
        self._run_id = run_id if run_id is not None and run_id != "" else None
        self._run_digest = run_digest if run_digest is not None and run_digest != "" else None
        self._counters_fn = counters_fn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: Any = None
        self._warned: set[str] = set()
        self._gpu_errors = 0
        self._gpu_dropped = False
        self._backends: dict[str, str] = {"gpu": "pending", "cpu": "pending", "torch": "pending"}
        self._errors: dict[str, int] = {}
        self._nvml: Any = None
        self._nvml_handle: Any = None
        self._psutil_process: Any = None
        self._cpu_prev: dict[int, tuple[float, float]] | None = None
        self._cpu_prev_at: float | None = None

    # -- lifecycle ------------------------------------------------------

    def start(self) -> bool:
        """Spawn the sampler thread; idempotent. Returns False when off."""
        if not self._enabled or self._thread is not None:
            return False
        try:
            self._sink_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(  # noqa: SIM115 — run-scoped handle, closed in stop()
                self._sink_path, "a", encoding="utf-8", buffering=1
            )
        except Exception as exc:
            self._warn_once("sink", f"sink open failed: {exc.__class__.__name__}")
            self._enabled = False
            return False
        self._stop.clear()
        thread = threading.Thread(target=self._run, name="hydra2-verbose-sampler", daemon=True)
        self._thread = thread
        thread.start()
        return True

    def stop(self) -> None:
        """Stop the thread, flush and close; idempotent, never raises."""
        thread, self._thread = self._thread, None
        try:
            self._stop.set()
            if thread is not None:
                thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            handle, self._handle = self._handle, None
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.flush()
                with contextlib.suppress(Exception):
                    handle.close()
        except Exception:
            pass
        try:
            nvml, self._nvml = self._nvml, None
            self._nvml_handle = None
            if nvml is not None:
                with contextlib.suppress(Exception):
                    nvml.nvmlShutdown()
        except Exception:
            pass

    def close(self) -> None:
        """Alias for :meth:`stop` (mirror-protocol symmetry)."""
        self.stop()

    # -- internals ------------------------------------------------------

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        warnings.warn(f"verbose sampler degraded ({key}): {message}", stacklevel=3)

    def _counters(self) -> tuple[int, int]:
        try:
            if self._counters_fn is None:
                return (0, 0)
            counted: tuple[int, int] = self._counters_fn()
            update, micro = counted
            return (update, micro)
        except Exception:
            return (0, 0)

    def _bump(self, key: str) -> None:
        with contextlib.suppress(Exception):
            self._errors[key] = self._errors.get(key, 0) + 1

    def _run(self) -> None:
        try:
            self._prime()
        except Exception as exc:
            self._warn_once("prime", exc.__class__.__name__)
        while not self._stop.wait(self._interval_s):
            try:
                self._tick()
            except Exception as exc:
                self._warn_once("tick", exc.__class__.__name__)

    def _prime(self) -> None:
        """Lazy SDK imports + baseline snapshots (sampler thread only)."""
        try:
            import pynvml as nvml

            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml = nvml
            self._nvml_handle = handle
            self._backends["gpu"] = "nvml"
        except Exception as exc:
            self._backends["gpu"] = f"absent-{exc.__class__.__name__}"
            self._nvml = None
            self._nvml_handle = None
        try:
            import psutil  # pyrefly: ignore[untyped-import] # no stubs, precise binds below

            proc = psutil.Process()
            proc_cpu: tuple[float, ...] = proc.cpu_times()
            _primed_update: int = len(proc_cpu)
            self._psutil_process = proc
            self._backends["cpu"] = "psutil"
        except Exception as exc:
            self._backends["cpu"] = f"absent-{exc.__class__.__name__}"
            self._psutil_process = None
        try:
            import torch

            if torch.cuda.is_available():
                _ = torch.cuda.memory_stats()
                self._backends["torch"] = "cuda"
            else:
                self._backends["torch"] = "absent-cpu"
        except Exception as exc:
            self._backends["torch"] = f"absent-{exc.__class__.__name__}"

    def _tick(self) -> None:
        """Append one JSON row (gpu/proc/cpu/torch); stop on sink failure."""
        handle = self._handle
        if handle is None:
            return
        counters = self._counters()
        row = _null_row(
            run_id=self._run_id,
            run_digest=self._run_digest,
            counters=counters,
            backends=dict(self._backends),
            errors=dict(self._errors),
        )
        gpu, proc = self._sample_gpu()
        row["gpu"] = gpu
        row["proc"] = proc
        row["cpu"] = self._sample_cpu()
        row["torch"] = self._sample_torch()
        row["backends"] = dict(self._backends)
        row["errors"] = dict(self._errors)
        try:
            import json

            _row_chars: int = handle.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception as exc:
            self._bump("sink_write")
            self._warn_once("sink_write", exc.__class__.__name__)
            self.stop()

    def _sample_gpu(self) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
        """Snapshot device counters + per-PID shares; drop group after 100 errors."""
        nvml = self._nvml
        handle = self._nvml_handle
        if nvml is None or handle is None or self._gpu_dropped:
            return (None, None)
        try:
            mem: Any = nvml.nvmlDeviceGetMemoryInfo(handle)
            util: Any = nvml.nvmlDeviceGetUtilizationRates(handle)
            try:
                power_raw: int = nvml.nvmlDeviceGetPowerUsage(handle)
                power_mw: int | None = power_raw
            except Exception:
                power_mw = None
            try:
                temp_raw: int = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                temp_c: int | None = temp_raw
            except Exception:
                temp_c = None
            try:
                sm_raw: int = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
                clock_sm: int | None = sm_raw
            except Exception:
                clock_sm = None
            try:
                mem_raw: int = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
                clock_mem: int | None = mem_raw
            except Exception:
                clock_mem = None
            try:
                tx_kind: int = nvml.NVML_PCIE_UTIL_TX_BYTES
                rx_kind: int = nvml.NVML_PCIE_UTIL_RX_BYTES
                tx_raw: int = nvml.nvmlDeviceGetPcieThroughput(handle, tx_kind)
                rx_raw: int = nvml.nvmlDeviceGetPcieThroughput(handle, rx_kind)
                pcie_tx: int | None = tx_raw
                pcie_rx: int | None = rx_raw
            except Exception:
                pcie_tx = None
                pcie_rx = None
            mem_used: int = mem.used
            mem_total: int = mem.total
            gpu_pct: int = util.gpu
            mem_pct: int = util.memory
            gpu = {
                "mem_used_mb": mem_used // (1024 * 1024),
                "mem_total_mb": mem_total // (1024 * 1024),
                "util_gpu_pct": gpu_pct,
                "util_mem_pct": mem_pct,
                "power_w": (power_mw / 1000.0) if power_mw is not None else None,
                "temp_c": temp_c,
                "clock_sm_mhz": clock_sm,
                "clock_mem_mhz": clock_mem,
                "pcie_tx_kbs": pcie_tx,
                "pcie_rx_kbs": pcie_rx,
            }
            procs: list[dict[str, Any]] = []
            try:
                running: list[Any] = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except Exception:
                running = []
            running_pids: list[int] = [p.pid for p in running]
            running_mems: list[int] = [p.usedGpuMemory for p in running]
            mem_by_pid: dict[int, int] = dict(zip(running_pids, running_mems, strict=True))
            try:
                samples: list[Any] = nvml.nvmlDeviceGetProcessUtilization(handle, 0)
            except Exception:
                samples = []
            for sample in samples:
                pid: int = sample.pid
                sm_pct: int = sample.smUtil
                mem_pct: int = sample.memUtil
                stamp_ns: int = sample.timeStamp
                procs.append(
                    {
                        "pid": pid,
                        "sm_util_pct": sm_pct,
                        "mem_util_pct": mem_pct,
                        "mem_used_mb": mem_by_pid.get(pid, 0) // (1024 * 1024),
                        "sample_ns": stamp_ns,
                    }
                )
            for pid, used in mem_by_pid.items():
                if all(entry["pid"] != pid for entry in procs):
                    used_mb: int = used // (1024 * 1024)
                    procs.append(
                        {
                            "pid": pid,
                            "sm_util_pct": None,
                            "mem_util_pct": None,
                            "mem_used_mb": used_mb,
                            "sample_ns": None,
                        }
                    )
            self._gpu_errors = 0
            return (gpu, procs)
        except Exception as exc:
            self._bump("gpu")
            self._gpu_errors += 1
            if self._gpu_errors >= _GPU_DROP_AFTER_ERRORS and not self._gpu_dropped:
                self._gpu_dropped = True
                self._backends["gpu"] = "dropped-errors"
                self._warn_once("gpu", f"dropped after {self._gpu_errors} errors")
            else:
                self._warn_once("gpu", exc.__class__.__name__)
            return (None, None)

    def _sample_cpu(self) -> dict[str, Any] | None:
        """Snapshot RSS, per-TID busy%, children, loadavg; None when psutil absent."""
        proc = self._psutil_process
        if proc is None:
            return None
        try:
            import psutil  # pyrefly: ignore[untyped-import] # no stubs, precise binds below

            threads: list[dict[str, Any]] = []
            try:
                proc_threads: list[Any] = proc.threads()
                current: dict[int, tuple[float, float]] = {}
                for thread in proc_threads:
                    tid: int = thread.id
                    user_s: float = thread.user_time
                    sys_s: float = thread.system_time
                    current[tid] = (user_s, sys_s)
            except Exception:
                current = {}
            prev = self._cpu_prev
            prev_at = self._cpu_prev_at
            now_mono = time.monotonic()
            self._cpu_prev = current
            self._cpu_prev_at = now_mono
            gap_s = now_mono - prev_at if prev_at is not None else 0.0
            if prev is not None and gap_s > 0:
                for tid, (user, system) in current.items():
                    old = prev.get(tid)
                    if old is None:
                        continue
                    busy = max(0.0, (user - old[0]) + (system - old[1]))
                    pct = 100.0 * busy / gap_s
                    threads.append({"tid": tid, "busy_pct": round(min(pct, 100.0), 1)})

                def _thread_key(entry: dict[str, Any]) -> float:
                    pct_value: float = entry["busy_pct"]
                    return pct_value

                threads.sort(key=_thread_key, reverse=True)
                threads = threads[:16]
            try:
                mem_info: Any = proc.memory_info()
                mem_rss: int = mem_info.rss
                rss_mb: int | None = mem_rss // (1024 * 1024)
            except Exception:
                rss_mb = None
            children: list[dict[str, Any]] = []
            try:
                child_list: list[Any] = proc.children(recursive=True)
                for child in child_list:
                    try:
                        with child.oneshot():
                            times: Any = child.cpu_times()
                            kid_threads: list[Any] = child.threads()
                            kids: dict[int, tuple[float, float]] = {}
                            for kid in kid_threads:
                                kid_tid: int = kid.id
                                kid_user: float = kid.user_time
                                kid_sys: float = kid.system_time
                                kids[kid_tid] = (kid_user, kid_sys)
                    except Exception:
                        continue
                    times_user: float = times.user
                    times_sys: float = times.system
                    busy_total: float = (times_user + times_sys) if times else 0.0
                    try:
                        child_mem: Any = child.memory_info()
                        child_mem_rss: int = child_mem.rss
                        child_rss: int | None = child_mem_rss // (1024 * 1024)
                    except Exception:
                        child_rss = None
                    child_pid: int = child.pid
                    children.append(
                        {
                            "pid": child_pid,
                            "cpu_s": round(busy_total, 3),
                            "rss_mb": child_rss,
                            "n_threads": len(kids),
                        }
                    )
            except Exception:
                children = []
            try:
                load: list[float] = list(psutil.getloadavg())
            except Exception:
                load = []
            return {
                "rss_mb": rss_mb,
                "threads": threads,
                "threads_primed": prev is not None,
                "children": children,
                "loadavg": load,
            }
        except Exception as exc:
            self._bump("cpu")
            self._warn_once("cpu", exc.__class__.__name__)
            return None

    def _sample_torch(self) -> dict[str, Any] | None:
        """Snapshot CUDA allocator stats; None unless the torch backend is cuda."""
        backend: str = self._backends.get("torch", "")
        if not backend.startswith("cuda"):
            return None
        try:
            import torch

            stats: dict[str, int] = torch.cuda.memory_stats()
            get = stats.get
            scale = 1024 * 1024
            allocated_raw: int | None = get("allocated_bytes.all.current", 0)
            reserved_raw: int | None = get("reserved_bytes.all.current", 0)
            allocated: int = allocated_raw if allocated_raw is not None else 0
            reserved: int = reserved_raw if reserved_raw is not None else 0
            blocks_raw: int | None = get("active_blocks.all.current", 0)
            retries_raw: int | None = get("num_alloc_retries", 0)
            ooms_raw: int | None = get("num_ooms", 0)
            return {
                "allocated_mb": allocated // scale,
                "reserved_mb": reserved // scale,
                "active_blocks": blocks_raw if blocks_raw is not None else 0,
                "alloc_retries": retries_raw if retries_raw is not None else 0,
                "ooms": ooms_raw if ooms_raw is not None else 0,
            }
        except Exception as exc:
            self._bump("torch")
            self._warn_once("torch", exc.__class__.__name__)
            return None
