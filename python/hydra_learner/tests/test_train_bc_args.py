from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import hydra_learner.train_bc as train_bc
from hydra_learner.metrics import EvalStats, StepStats, summarize_eval
from hydra_learner.raw_mjai_stream import BuildProgress, RawMjaiBridgeStats, RawMjaiPinnedQueueStats

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
from hydra_learner.train_bc import (
    PYTHON_VARIANT_DEFAULT,
    ScalarEventWriter,
    add_scalars,
    log_step_scalars,
    log_validation_scalars,
    parse_args,
    raw_mjai_scalar_snapshot,
)


def test_parse_args_defaults_to_compile_max_autotune() -> None:
    with patch.object(sys, "argv", ["python-bc-train"]):
        args = parse_args()

    assert PYTHON_VARIANT_DEFAULT == "compile_max_autotune"
    assert args.variant == "compile_max_autotune"


def test_parse_args_accepts_ux_artifact_flags() -> None:
    with patch.object(
        sys,
        "argv",
        [
            "python-bc-train",
            "--checkpoint-dir",
            "out/checkpoints",
            "--keep-step-checkpoints",
            "--log-dir",
            "out/logs",
            "--log-every-steps",
            "25",
            "--tensorboard-dir",
            "out/tensorboard",
            "--tensorboard-url",
            "http://127.0.0.1:6007/",
        ],
    ):
        args = parse_args()

    assert str(args.checkpoint_dir) == "out/checkpoints"
    assert args.keep_step_checkpoints is True
    assert str(args.log_dir) == "out/logs"
    assert args.log_every_steps == 25
    assert str(args.tensorboard_dir) == "out/tensorboard"
    assert args.tensorboard_url == "http://127.0.0.1:6007/"


def _read_scalar_tags(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    return {record["tag"]: record["value"] for record in records}


def test_scalar_helpers_emit_production_metrics(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    path = tmp_path / "tensorboard"
    monkeypatch.setattr(train_bc, "SummaryWriter", None)
    writer = ScalarEventWriter(path)
    stat = StepStats(
        step_ms=10.0,
        fwd_loss_ms=2.0,
        backward_ms=3.0,
        optimizer_ms=1.0,
        loss=0.5,
        fetch_decode_ms=5.0,
        h2d_wall_ms=1.0,
        train_gpu_ms=10.0,
    )

    log_step_scalars(writer, stat, batch=32, samples_seen=128, global_step=4)
    metrics = summarize_eval(
        [
            EvalStats(1.0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
            EvalStats(3.0, 1.5, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        ]
    )
    log_validation_scalars(writer, metrics, 4, final=False)
    add_scalars(writer, "skip", {"nan": float("nan"), "text": "x", "none": None, "flag": True}, 4)
    writer.close()

    tags = _read_scalar_tags(path / "scalars.jsonl")
    assert tags["train/loss"] == 0.5
    assert tags["train/samples_per_s"] == 3200.0
    assert tags["train/input_pipeline_wall_ms"] == 6.0
    assert tags["train/end_to_end_samples_per_s"] == 2000.0
    assert tags["validation/loss"] == 2.0
    assert tags["validation/policy_accuracy"] == 0.9
    assert tags["skip/flag"] == 1.0
    assert "skip/nan" not in tags
    assert "skip/text" not in tags


def test_scalar_writer_emits_tensorboard_event_file_when_available(tmp_path: Path) -> None:
    path = tmp_path / "tensorboard"
    writer = ScalarEventWriter(path)
    writer.add_scalar("train/loss", 0.5, 4)
    writer.close()

    if train_bc.SummaryWriter is None:
        assert (path / "scalars.jsonl").is_file()
    else:
        assert any(child.name.startswith("events.out.tfevents") for child in path.iterdir())


class _FakePinnedRaw:
    def progress(self) -> BuildProgress:
        return BuildProgress(
            manifest_path=None,
            complete=True,
            build_seconds=1.25,
            loaded_games=3,
            skipped_games=1,
            samples=64,
            batches=2,
            max_games_reached=False,
            max_samples_reached=True,
        )

    def bridge_stats(self) -> RawMjaiBridgeStats:
        return RawMjaiBridgeStats(open_count=1, last_next_fill_ms=2.5, last_bytes_filled=4096)

    def queue_stats(self) -> RawMjaiPinnedQueueStats:
        return RawMjaiPinnedQueueStats(
            ready_wait_ms_total=6.0,
            ready_wait_count=3,
            producer_fill_ms_total=8.0,
            produced_batches=4,
            producer_free_wait_ms_total=10.0,
            producer_free_wait_count=5,
            ready_queue_size=2,
            free_queue_size=1,
        )


def test_raw_mjai_scalar_snapshot_includes_progress_bridge_and_queue() -> None:
    snapshot = raw_mjai_scalar_snapshot(None, _FakePinnedRaw())  # type: ignore[arg-type]

    assert snapshot["progress/complete"] is True
    assert snapshot["progress/samples"] == 64
    assert snapshot["bridge/last_next_fill_ms"] == 2.5
    assert snapshot["queue/mean_ready_wait_ms"] == 2.0
    assert snapshot["queue/mean_producer_fill_ms"] == 2.0
    assert snapshot["queue/mean_producer_free_wait_ms"] == 2.0
