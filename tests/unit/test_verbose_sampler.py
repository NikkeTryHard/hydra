"""Verbose sampler: opt-in gating, row schema, observer-only lifecycle."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import pytest

from hydra2.tracking.verbose_sampler import (
    DEFAULT_INTERVAL_MS,
    is_enabled,
    resolve_interval_ms,
)
from hydra2.tracking.verbose_sampler_factory import (
    NullVerboseSampler,
    make_verbose_sampler,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract_package("WP-14")

_REQUIRED_ROW_KEYS = frozenset(
    {
        "v",
        "t_wall_s",
        "t_mono_ns",
        "run_id",
        "run_digest",
        "global_update",
        "microstep",
        "gpu",
        "proc",
        "cpu",
        "torch",
        "backends",
        "errors",
    }
)


def test_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env, no explicit flag: sampler stays off (verbose is opt-in)."""
    monkeypatch.delenv("HYDRA2_VERBOSE_TELEMETRY", raising=False)
    monkeypatch.delenv("HYDRA2_VERBOSE_TELEMETRY_DISABLED", raising=False)
    assert is_enabled() is False
    assert isinstance(make_verbose_sampler(sink_path="x.jsonl"), NullVerboseSampler)


def test_kill_switch_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYDRA2_VERBOSE_TELEMETRY", "1")
    monkeypatch.setenv("HYDRA2_VERBOSE_TELEMETRY_DISABLED", "1")
    assert is_enabled(explicit=True) is False


def test_interval_clamp() -> None:
    assert resolve_interval_ms(20) == 20
    assert resolve_interval_ms(50) == 50
    with pytest.warns(UserWarning):
        assert resolve_interval_ms(100) == DEFAULT_INTERVAL_MS
    with pytest.warns(UserWarning):
        assert resolve_interval_ms("fast") == DEFAULT_INTERVAL_MS  # type: ignore[arg-type]


def test_null_sampler_never_spawns(tmp_path: Path) -> None:
    sampler = make_verbose_sampler(sink_path=tmp_path / "v.jsonl")
    assert isinstance(sampler, NullVerboseSampler)
    assert sampler.start() is False
    sampler.stop()
    sampler.close()
    assert not (tmp_path / "v.jsonl").exists()


def test_live_ticks_write_schema_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled sampler appends schema-valid rows; counters join keys flow."""
    monkeypatch.setenv("HYDRA2_VERBOSE_TELEMETRY", "1")
    monkeypatch.delenv("HYDRA2_VERBOSE_TELEMETRY_DISABLED", raising=False)
    sink = tmp_path / "verbose-telemetry.jsonl"
    state = {"update": 3, "micro": 1}
    sampler = make_verbose_sampler(
        sink_path=sink,
        interval_ms=50,
        run_id="run-probe",
        run_digest="sha256:probe",
        counters_fn=lambda: (state["update"], state["micro"]),
    )
    assert sampler.start() is True
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if sink.exists() and len(sink.read_text(encoding="utf-8").splitlines()) >= 2:
                break
            time.sleep(0.02)
    finally:
        sampler.stop()
        sampler.stop()  # idempotent
    rows = [json.loads(line) for line in sink.read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 2
    for row in rows:
        assert set(row) >= _REQUIRED_ROW_KEYS
        assert row["v"] == 1
        assert row["run_id"] == "run-probe"
        assert row["run_digest"] == "sha256:probe"
        assert (row["global_update"], row["microstep"]) == (3, 1)
        assert set(row["backends"]) == {"gpu", "cpu", "torch"}
    assert any(row["cpu"] is not None for row in rows)  # psutil in env


def test_factory_never_raises() -> None:
    """Garbage kwargs degrade to Null instead of raising into training."""
    assert isinstance(make_verbose_sampler(sink_path=None), NullVerboseSampler)  # type: ignore[arg-type]
    assert isinstance(make_verbose_sampler(), NullVerboseSampler)
