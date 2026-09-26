"""Sidecar reporter unit tests (CPU lane, hermetic, no ClearML server).

Covers the termination-flush handler (the data-loss fix: buffered scalars
must flush + task must close on SIGTERM/SIGINT instead of dying silent)
and the row-to-scalar mapping for the newest keys (train top-3/5 flow
into accuracy/* like the loop emits them). The watch heartbeat line is
formatting-only and covered by manual smoke, not pinned here.
"""

from __future__ import annotations

import importlib.util
import json
import signal
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract_package("WP-14")


def _load_sidecar() -> Any:
    path = Path(__file__).resolve().parents[2] / "scripts" / "clearml_sidecar_report.py"
    spec = importlib.util.spec_from_file_location("sidecar_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, str, float, int]] = []
        self.flushed = 0
        self.fail_flush = False

    def report_scalar(self, *, title: str, series: str, value: float, iteration: int) -> None:
        self.scalars.append((title, series, value, iteration))

    def flush(self) -> None:
        self.flushed += 1
        if self.fail_flush:
            raise RuntimeError("injected flush failure")


class _FakeTask:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _FakeOs:
    def __init__(self) -> None:
        self.killed: list[tuple[int, int]] = []

    def getpid(self) -> int:
        return 1234

    def kill(self, pid: int, signum: int) -> None:
        self.killed.append((pid, signum))


def _write_metrics(run: Path) -> None:
    rows = [
        {
            "global_update": 0,
            "total": 2.0,
            "masked_nll": 1.9,
            "top1": 0.4,
            "top3": 0.6,
            "top5": 0.7,
            "lr_now": 0.0003,
        },
        {
            "global_update": 1,
            "total": 1.8,
            "masked_nll": 1.7,
            "top1": 0.45,
            "top3": 0.65,
            "top5": 0.75,
            "lr_now": 0.0003,
        },
    ]
    (run / "logs").mkdir(parents=True, exist_ok=True)
    (run / "logs" / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_termination_handler_flushes_and_closes(monkeypatch: Any) -> None:
    """SIGTERM flushes buffered scalars, closes the task, then exits loud."""
    module = _load_sidecar()
    logger, task = _FakeLogger(), _FakeTask()
    prev_term = signal.getsignal(signal.SIGTERM)
    prev_int = signal.getsignal(signal.SIGINT)
    fake_os = _FakeOs()
    monkeypatch.setattr(module, "os", fake_os)
    try:
        module._install_termination_flush(logger, task)
        assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        signal.signal(signal.SIGINT, prev_int)
    assert logger.flushed == 1
    assert task.closed == 1
    assert fake_os.killed == [(1234, signal.SIGTERM)]


def test_pass_reports_topk_mapping(tmp_path: Path) -> None:
    """_pass maps the loop's top-3/5 keys into accuracy/* (no silent drop)."""
    module = _load_sidecar()
    _write_metrics(tmp_path)
    logger = _FakeLogger()
    offsets: dict[str, Any] = {}
    counts = module._pass(logger, tmp_path, offsets, {})
    assert counts[0] == 2
    acc = {(series, it): v for title, series, v, it in logger.scalars if title == "accuracy"}
    assert acc[("top1", 0)] == 0.4
    assert acc[("top3", 1)] == 0.65
    assert acc[("top5", 1)] == 0.75
    assert logger.flushed == 1


def test_pass_holds_offsets_when_flush_fails(tmp_path: Path) -> None:
    """A failed flush rolls offsets back: rows re-report, never lost."""
    module = _load_sidecar()
    _write_metrics(tmp_path)
    logger = _FakeLogger()
    logger.fail_flush = True
    offsets: dict[str, Any] = {}
    counts = module._pass(logger, tmp_path, offsets, {})
    assert counts[0] == 2
    assert offsets == {}


def test_pass_advances_offsets_when_flush_succeeds(tmp_path: Path) -> None:
    """A clean flush persists consumption past the reported rows."""
    module = _load_sidecar()
    _write_metrics(tmp_path)
    logger = _FakeLogger()
    offsets: dict[str, Any] = {}
    counts = module._pass(logger, tmp_path, offsets, {})
    assert counts[0] == 2
    assert offsets["metrics"] == (tmp_path / "logs" / "metrics.jsonl").stat().st_size
