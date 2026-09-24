"""Observer stage-span log (logging-only wall-clock spans).

Long-tail training stalls hide inside synchronous gaps no counter names:
snapshot copies, background-write queueing, pre-eval drains, serial eval
expansion, first-eval compiles. :class:`StageSpanSink` appends one JSON row
per measured stage to ``logs/stage-spans.jsonl`` so any gap is attributable
by joining on ``update`` with metrics/feed/verbose rows.

Logging-only contract: rows carry wall time (``t_start_s``) for alignment
and MUST NEVER flow into digests, manifests, checkpoint payloads, or
canonical bytes — wall clocks differ run to run, so hashing one breaks
resume identity and parity gates. Call sites add timing around existing
calls; no training math, ordering, or RNG changes. Thread-safe: worker
threads emit through the same sink behind a lock. Failure mode: a sink
I/O error raises to the caller (a blind observer is worse than a loud
one — but callers MUST only construct the sink where logs exist, never
on import or in unit flows without a run dir).
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from pathlib import Path
from typing import Any

__all__ = ["SPAN_FILE", "StageSpanSink"]


#: Log filename under ``<run_dir>/logs`` (observer-only; resume never reads it).
SPAN_FILE = "stage-spans.jsonl"


class StageSpanSink:
    """Thread-safe append sink for stage-span rows (one per run)."""

    def __init__(self, logs_dir: Path | str) -> None:
        self._path = Path(logs_dir) / SPAN_FILE
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Destination file (created on first emit, parent must exist)."""
        return self._path

    def emit(
        self,
        *,
        stage: str,
        dur_ms: float,
        update: int | None = None,
        t_start_s: float | None = None,
    ) -> None:
        """Append one span row (logging-only).

        ``t_start_s`` defaults to now-minus-duration (for spans measured
        with perf_counter on the calling thread); worker-side and
        pre-measured spans pass their own wall start explicitly.
        """
        dur = float(dur_ms)
        start = float(t_start_s) if t_start_s is not None else time.time() - dur / 1000.0
        row = {
            "dur_ms": dur,
            "kind": "span",
            "stage": str(stage),
            "t_start_s": start,
            "update": None if update is None else int(update),
        }
        line = json.dumps(row, sort_keys=True) + "\n"
        with self._lock, open(self._path, "a", encoding="utf-8") as handle:
            _ = handle.write(line)

    def timed(self, stage: str, update: int | None = None) -> Any:
        """Context manager emitting the block's wall duration on exit.

        Emits even when the block raises (a failed stage's cost still
        counts); the exception propagates unchanged.
        """
        sink = self

        class _Timer:
            def __enter__(self) -> _Timer:
                self._t0 = time.perf_counter()
                return self

            def __exit__(self, *exc: Any) -> None:
                sink.emit(
                    stage=stage,
                    update=update,
                    dur_ms=(time.perf_counter() - self._t0) * 1000.0,
                )

        return _Timer()


@contextlib.contextmanager
def maybe_timed(sink: StageSpanSink | None, stage: str, update: int | None = None) -> Any:
    """No-op when ``sink`` is None, else :meth:`StageSpanSink.timed`."""
    if sink is None:
        yield None
    else:
        with sink.timed(stage, update):
            yield None
