"""Stage-span sink invariants (logging-only observer rows)."""

from __future__ import annotations

import json

from hydra2.tracking.stage_spans import SPAN_FILE, StageSpanSink


def test_emit_appends_valid_row(tmp_path):
    sink = StageSpanSink(tmp_path)
    sink.emit(stage="snapshot", dur_ms=12.5, update=7, t_start_s=1000.0)
    lines = (tmp_path / SPAN_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row == {
        "dur_ms": 12.5,
        "kind": "span",
        "stage": "snapshot",
        "t_start_s": 1000.0,
        "update": 7,
    }


def test_timed_measures_block(tmp_path):
    sink = StageSpanSink(tmp_path)
    with sink.timed("eval-expand", 25):
        total = sum(range(1000))
    assert total == 499500
    row = json.loads((tmp_path / SPAN_FILE).read_text(encoding="utf-8").splitlines()[0])
    assert row["stage"] == "eval-expand"
    assert row["update"] == 25
    assert row["dur_ms"] >= 0.0


def test_update_none_allowed(tmp_path):
    sink = StageSpanSink(tmp_path)
    sink.emit(stage="close", dur_ms=1.0)
    row = json.loads((tmp_path / SPAN_FILE).read_text(encoding="utf-8").splitlines()[0])
    assert row["update"] is None
