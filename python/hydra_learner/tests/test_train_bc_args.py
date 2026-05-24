from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

import hydra_learner.hydra_logging as hydra_logging
import hydra_learner.train_bc as train_bc
import hydra_learner.validation as validation
from hydra_learner.checkpoint import ResumeState
from hydra_learner.losses import LossWeights
from hydra_learner.metrics import EvalStats, StepStats, summarize_eval, summarize_steps
from hydra_learner.model import HydraPolicyNet
from hydra_learner.raw_mjai import (
    BuildProgress,
    RawMjaiBridgeStats,
    RawMjaiPinnedQueueStats,
    validate_raw_mjai_source_args,
)
from hydra_learner.shard_contracts import PolicyBatch

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
from hydra_learner.checkpointing import RawMjaiResumeOffsets
from hydra_learner.cli import parse_args, validate_args
from hydra_learner.constants import PYTHON_VARIANT_DEFAULT
from hydra_learner.hydra_logging import (
    JsonlLogger,
    ScalarEventWriter,
    add_scalars,
    log_step_scalars,
    log_validation_scalars,
    raw_mjai_scalar_snapshot,
)
from hydra_learner.optim import build_ema_config, ema_weights
from hydra_learner.validation import RawMjaiValidationSource, evaluate_raw_and_ema


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
            "--min-lr",
            "0.000001",
            "--lr-warmup-steps",
            "12",
            "--lr-schedule",
            "constant",
            "--schedule-total-steps",
            "99",
            "--grad-clip-norm",
            "1.5",
        ],
    ):
        args = parse_args()

    assert str(args.checkpoint_dir) == "out/checkpoints"
    assert args.keep_step_checkpoints is True
    assert str(args.log_dir) == "out/logs"
    assert args.log_every_steps == 25
    assert str(args.tensorboard_dir) == "out/tensorboard"
    assert args.tensorboard_url == "http://127.0.0.1:6007/"
    assert args.min_lr == 0.000001
    assert args.lr_warmup_steps == 12
    assert args.lr_schedule == "constant"
    assert args.schedule_total_steps == 99
    assert args.grad_clip_norm == 1.5
    assert args.raw_mjai_validation_augment is False
    assert args.ema_enabled is False
    assert args.ema_decay == 0.999
    assert args.ema_start_step == 0
    assert args.ema_update_every_steps == 1
    assert args.ema_device == "auto"
    assert args.validation_source_mode == "fixed"


def test_parse_args_accepts_recommended_large_model_shape() -> None:
    with patch.object(
        sys,
        "argv",
        [
            "python-bc-train",
            "--hidden",
            "384",
            "--blocks",
            "12",
            "--bottleneck",
            "96",
        ],
    ):
        args = parse_args()

    assert args.hidden == 384
    assert args.blocks == 12
    assert args.bottleneck == 96


def _read_scalar_tags(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    return {record["tag"]: record["value"] for record in records}


def _policy_batch(rows: int, fill: float) -> PolicyBatch:
    return PolicyBatch(
        obs=np.full((rows, 192, 34), fill, dtype=np.float32),
        actions=np.arange(rows, dtype=np.int64) % 46,
        legal_mask=np.ones((rows, 46), dtype=np.bool_),
        value_target=np.full(rows, fill, dtype=np.float32),
        grp_target=np.zeros((rows, 24), dtype=np.float32),
        oracle_target=np.zeros((rows, 4), dtype=np.float32),
        oracle_target_mask=np.zeros(rows, dtype=np.float32),
        tenpai=np.zeros((rows, 3), dtype=np.float32),
        opp_next=np.zeros((rows, 102), dtype=np.float32),
        danger=np.zeros((rows, 102), dtype=np.float32),
        danger_mask=np.ones((rows, 102), dtype=np.float32),
        score_pdf=np.zeros((rows, 64), dtype=np.float32),
        score_cdf=np.zeros((rows, 64), dtype=np.float32),
        safety_target=None,
        safety_mask=None,
    )


def _eval_stats(**overrides: Any) -> EvalStats:
    values = {
        "loss": 0.0,
        "policy": 0.0,
        "value": 0.0,
        "grp": 0.0,
        "tenpai": 0.0,
        "danger": 0.0,
        "opp_next": 0.0,
        "score_pdf": 0.0,
        "score_cdf": 0.0,
        "oracle_critic": 0.0,
        "safety_residual": 0.0,
        "target_coverage": {"policy": {"active": True, "status": "present_positive", "fraction": 1.0}},
        "policy_accuracy": 0.0,
        "policy_top3_accuracy": 0.0,
        "policy_top5_accuracy": 0.0,
        "policy_nll": 0.0,
        "policy_confidence": 0.0,
        "policy_ece": 0.0,
        "samples": 1,
    }
    values.update(overrides)
    return EvalStats(**values)


class _ValidationStreamFixture:
    instances: ClassVar[list[_ValidationStreamFixture]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        self._batches = [(_policy_batch(4, 1.0), 0.1), (_policy_batch(4, 2.0), 0.2)]
        self._index = 0
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def next_batch(self) -> tuple[PolicyBatch, float]:
        if self._index >= len(self._batches):
            raise StopIteration
        batch = self._batches[self._index]
        self._index += 1
        return batch

    def close(self) -> None:
        self.closed = True


def test_scalar_helpers_emit_production_metrics(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    path = tmp_path / "tensorboard"
    monkeypatch.setattr(hydra_logging, "SummaryWriter", None)
    writer = ScalarEventWriter(path)
    stat = StepStats(
        step_ms=10.0,
        fwd_loss_ms=2.0,
        backward_ms=3.0,
        optimizer_ms=1.0,
        loss=0.5,
        head_losses={"policy": 0.25, "value": 0.125},
        target_coverage={"policy": {"active": True, "status": "present_positive", "fraction": 1.0}},
        fetch_decode_ms=5.0,
        h2d_wall_ms=1.0,
        train_gpu_ms=10.0,
        lr=0.001,
        grad_norm=2.0,
        policy_accuracy=0.75,
        policy_nll=1.25,
        policy_top3_accuracy=0.875,
        policy_top5_accuracy=1.0,
        policy_confidence=0.625,
        policy_entropy=2.5,
        policy_target_prob=0.5,
        policy_margin=0.25,
    )

    log_step_scalars(writer, stat, batch=32, samples_seen=128, global_step=4)
    metrics = summarize_eval(
        [
            _eval_stats(
                loss=1.0,
                policy=0.5,
                value=0.1,
                grp=0.2,
                tenpai=0.3,
                danger=0.4,
                opp_next=0.5,
                score_pdf=0.6,
                score_cdf=0.7,
                policy_accuracy=0.8,
                policy_top3_accuracy=0.9,
                policy_top5_accuracy=0.95,
                policy_nll=0.6,
                policy_confidence=0.85,
                policy_ece=0.05,
                samples=4,
            ),
            _eval_stats(
                loss=3.0,
                policy=1.5,
                value=0.3,
                grp=0.4,
                tenpai=0.5,
                danger=0.6,
                opp_next=0.7,
                score_pdf=0.8,
                score_cdf=0.9,
                policy_accuracy=1.0,
                policy_top3_accuracy=0.8,
                policy_top5_accuracy=0.9,
                policy_nll=0.4,
                policy_confidence=0.7,
                policy_ece=0.1,
                samples=4,
            ),
        ]
    )
    log_validation_scalars(writer, metrics, 4, final=False)
    add_scalars(
        writer, "skip", {"nan": float("nan"), "text": "x", "none": None, "flag": True}, 4, include_status_scalars=True
    )
    writer.close()

    tags = _read_scalar_tags(path / "scalars.jsonl")
    assert tags["train/loss"] == 0.5
    assert tags["train/samples_per_s"] == 3200.0
    assert tags["train/input_pipeline_wall_ms"] == 6.0
    assert tags["train/end_to_end_samples_per_s"] == 2000.0
    assert tags["train/lr"] == 0.001
    assert tags["train/grad_norm"] == 2.0
    assert tags["train/loss/policy"] == 0.25
    assert tags["train/loss/value"] == 0.125
    assert tags["train/policy_accuracy"] == 0.75
    assert tags["train/policy_nll"] == 1.25
    assert tags["train/policy_top3_accuracy"] == 0.875
    assert tags["train/policy_top5_accuracy"] == 1.0
    assert tags["train/policy_confidence"] == 0.625
    assert tags["train/policy_entropy"] == 2.5
    assert tags["train/policy_target_prob"] == 0.5
    assert tags["train/policy_margin"] == 0.25
    assert "train/head_losses" not in tags
    assert "train/target_coverage" not in tags
    assert "train/coverage/policy/active" not in tags
    assert "train/coverage/policy/status_code" not in tags
    assert tags["train/coverage/policy/fraction"] == 1.0
    assert tags["validation/loss"] == 2.0
    assert tags["validation/policy_accuracy"] == 0.9
    assert tags["validation/policy_top3_accuracy"] == 0.8500000000000001
    assert tags["validation/policy_ece"] == 0.07500000000000001
    assert tags["validation/policy_top5_accuracy"] == 0.925
    assert tags["validation/policy_nll"] == 0.5
    assert tags["validation/policy_confidence"] == pytest.approx(0.775)
    assert "validation/coverage/policy/status_code" not in tags
    assert tags["validation/samples"] == 8.0
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


def test_validation_metrics_weight_by_samples() -> None:
    metrics = summarize_eval(
        [
            _eval_stats(
                loss=1.0,
                policy_accuracy=0.5,
                policy_nll=1.0,
                samples=2,
            ),
            _eval_stats(
                loss=3.0,
                policy_accuracy=1.0,
                policy_nll=5.0,
                samples=6,
            ),
        ]
    )

    assert metrics["samples"] == 8.0
    assert metrics["loss"] == 2.5
    assert metrics["policy_nll"] == 4.0
    assert metrics["policy_accuracy"] == 0.875


def test_step_summary_includes_train_policy_diagnostics() -> None:
    summary = summarize_steps(
        [
            StepStats(
                step_ms=10.0,
                fwd_loss_ms=2.0,
                backward_ms=3.0,
                optimizer_ms=1.0,
                loss=0.5,
                head_losses={},
                target_coverage={},
                policy_nll=1.0,
                policy_accuracy=0.5,
                policy_top3_accuracy=0.75,
                policy_top5_accuracy=0.8,
                policy_confidence=0.6,
                policy_entropy=2.0,
                policy_target_prob=0.4,
                policy_margin=0.1,
            ),
            StepStats(
                step_ms=20.0,
                fwd_loss_ms=4.0,
                backward_ms=6.0,
                optimizer_ms=2.0,
                loss=1.5,
                head_losses={},
                target_coverage={},
                policy_nll=3.0,
                policy_accuracy=1.0,
                policy_top3_accuracy=1.0,
                policy_top5_accuracy=1.0,
                policy_confidence=0.8,
                policy_entropy=4.0,
                policy_target_prob=0.6,
                policy_margin=0.3,
            ),
        ],
        batch=4,
    )

    assert summary["mean_policy_nll"] == 2.0
    assert summary["mean_policy_accuracy"] == 0.75
    assert summary["mean_policy_top3_accuracy"] == 0.875
    assert summary["mean_policy_top5_accuracy"] == 0.9
    assert summary["mean_policy_confidence"] == 0.7
    assert summary["mean_policy_entropy"] == 3.0
    assert summary["mean_policy_target_prob"] == 0.5
    assert summary["mean_policy_margin"] == 0.2


def test_raw_train_augment_does_not_imply_validation_augment(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _ValidationStreamFixture.instances = []
    monkeypatch.setattr(validation, "RawMjaiDirectStream", _ValidationStreamFixture)
    args = _valid_args(
        raw_mjai_data_dirs=[tmp_path / "raw"],
        raw_mjai_prefetch_batches=2,
        raw_mjai_queue_bound=8,
        raw_mjai_worker_threads=1,
        raw_mjai_max_games=None,
        raw_mjai_train_fraction=0.9,
        raw_mjai_augment=True,
        raw_mjai_validation_augment=False,
        validation_steps=1,
        validation_source_mode="fixed",
    )
    logger = JsonlLogger(tmp_path / "events.jsonl")

    source = RawMjaiValidationSource(args=args, events=logger)
    source.next_batch()
    logger.close()

    assert _ValidationStreamFixture.instances[0].kwargs["augment"] is False
    assert source.info.augment is False
    source.close()


def test_fixed_validation_reuses_same_window(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _ValidationStreamFixture.instances = []
    monkeypatch.setattr(validation, "RawMjaiDirectStream", _ValidationStreamFixture)
    args = _valid_args(
        raw_mjai_data_dirs=[tmp_path / "raw"],
        raw_mjai_prefetch_batches=2,
        raw_mjai_queue_bound=8,
        raw_mjai_worker_threads=1,
        raw_mjai_max_games=None,
        raw_mjai_train_fraction=0.9,
        raw_mjai_augment=True,
        raw_mjai_validation_augment=False,
        validation_steps=2,
        validation_source_mode="fixed",
    )
    logger = JsonlLogger(tmp_path / "events.jsonl")

    source = RawMjaiValidationSource(args=args, events=logger)
    first, _ = source.next_batch()
    second, _ = source.next_batch()
    first_again, _ = source.next_batch()
    logger.close()

    assert source.info.mode == "fixed"
    assert source.info.requested_batches == 2
    assert source.info.actual_batches == 2
    assert source.info.requested_samples is None
    assert source.info.actual_samples == 8
    assert source.info.sample_cap_overrun == 0
    assert source.info.full_batches is True
    assert first.obs[0, 0, 0] == 1.0
    assert second.obs[0, 0, 0] == 2.0
    assert first_again is first
    assert _ValidationStreamFixture.instances[0].closed is True


def test_streaming_validation_source_is_logged(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _ValidationStreamFixture.instances = []
    monkeypatch.setattr(validation, "RawMjaiDirectStream", _ValidationStreamFixture)
    args = _valid_args(
        raw_mjai_data_dirs=[tmp_path / "raw"],
        raw_mjai_prefetch_batches=2,
        raw_mjai_queue_bound=8,
        raw_mjai_worker_threads=1,
        raw_mjai_max_games=None,
        raw_mjai_train_fraction=0.9,
        raw_mjai_augment=False,
        raw_mjai_validation_augment=True,
        validation_steps=2,
        validation_source_mode="streaming",
    )
    log_path = tmp_path / "events.jsonl"
    logger = JsonlLogger(log_path)

    source = RawMjaiValidationSource(args=args, events=logger)
    logger.close()
    source.close()

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == [
        "validation_source_stream_start",
        "validation_source_stream_started",
        "validation_source",
    ]
    source_record = records[-1]
    assert source_record["mode"] == "streaming"
    assert source_record["requested_batches"] == 2
    assert source_record["actual_batches"] == 0
    assert source_record["requested_samples"] is None
    assert source_record["actual_samples"] == 0
    assert source_record["sample_cap_overrun"] == 0
    assert source_record["full_batches"] is True
    assert source_record["augment"] is True


def test_validation_source_disabled_without_raw_dirs(tmp_path: Path) -> None:
    logger = JsonlLogger(tmp_path / "events.jsonl")
    source = RawMjaiValidationSource(args=_valid_args(validation_steps=2), events=logger)
    logger.close()

    assert source.info.actual_batches == 0
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "event": "validation_source_skipped",
            "validation_steps": 2,
            "has_raw_mjai": False,
            "ts": records[0]["ts"],
        }
    ]


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
    snapshot = raw_mjai_scalar_snapshot(None, _FakePinnedRaw(), RawMjaiResumeOffsets(samples=128, batches=4))  # type: ignore[arg-type]

    assert snapshot["progress/complete"] is True
    assert snapshot["progress/stream_local_samples"] == 64
    assert snapshot["progress/stream_local_batches"] == 2
    assert snapshot["progress/resume_total_samples"] == 192
    assert snapshot["progress/resume_total_batches"] == 6
    assert snapshot["bridge/last_next_fill_ms"] == 2.5
    assert snapshot["queue/mean_ready_wait_ms"] == 2.0
    assert snapshot["queue/mean_producer_fill_ms"] == 2.0
    assert snapshot["queue/mean_producer_free_wait_ms"] == 2.0


def test_raw_mjai_resume_offsets_preserve_checkpoint_progress() -> None:
    state = ResumeState(
        global_step=5,
        samples_seen=40,
        raw_mjai_progress={"loaded_games": 7, "skipped_games": 2, "samples": 36, "batches": 9},
    )

    offsets = RawMjaiResumeOffsets.from_resume(state, batch=4)

    assert offsets.loaded_games == 7
    assert offsets.skipped_games == 2
    assert offsets.samples == 36
    assert offsets.batches == 9


def test_raw_mjai_resume_offsets_fallback_to_samples_seen() -> None:
    state = ResumeState(global_step=5, samples_seen=40, raw_mjai_progress={})

    offsets = RawMjaiResumeOffsets.from_resume(state, batch=4)

    assert offsets.loaded_games == 0
    assert offsets.skipped_games == 0
    assert offsets.samples == 40
    assert offsets.batches == 10


def _cpu_tiny_training_objects() -> tuple[HydraPolicyNet, torch.optim.Optimizer, torch.nn.Module, train_bc.LrScheduler]:
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss_step = train_bc.HydraCompiledLossStep(model, "full_base", LossWeights())
    scheduler = train_bc.LrScheduler(
        train_bc.LrSchedulerConfig(
            base_lr=1.0e-3,
            min_lr=0.0,
            warmup_steps=0,
            total_steps=2,
            target_games=None,
            schedule="cosine",
        )
    )
    return model, optimizer, loss_step, scheduler


def _cpu_batch(rows: int = 2) -> tuple[torch.Tensor, train_bc.BaseTargets]:
    obs = torch.full((rows, 192, 34), 0.01, dtype=torch.float32)
    legal = torch.ones((rows, 46), dtype=torch.bool)
    labels = torch.arange(rows, dtype=torch.int64) % 46
    targets = train_bc.synthetic_targets(obs, legal, labels)
    return obs, train_bc.targets_for_compiled_loss(targets, LossWeights())


def _cuda_tiny_training_objects() -> tuple[
    HydraPolicyNet, torch.optim.Optimizer, torch.nn.Module, train_bc.LrScheduler
]:
    model, optimizer, loss_step, scheduler = _cpu_tiny_training_objects()
    model = model.to(torch.device("cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss_step = train_bc.HydraCompiledLossStep(model, "full_base", LossWeights())
    return model, optimizer, loss_step, scheduler


def _cuda_batch(rows: int = 2) -> tuple[torch.Tensor, train_bc.BaseTargets]:
    obs, _targets = _cpu_batch(rows)
    obs = obs.to(torch.device("cuda"))
    legal = torch.ones((rows, 46), dtype=torch.bool, device=torch.device("cuda"))
    labels = torch.arange(rows, dtype=torch.int64, device=torch.device("cuda")) % 46
    targets = train_bc.synthetic_targets(obs, legal, labels)
    return obs, train_bc.targets_for_compiled_loss(targets, LossWeights())


def _state_changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(before[key], after[key]) for key in before)


def test_non_mutating_warmup_restores_model_and_optimizer_state() -> None:
    model, optimizer, loss_step, scheduler = _cpu_tiny_training_objects()
    obs, targets = _cpu_batch()
    before_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
    train_bc.set_optimizer_lr(optimizer, scheduler.lr_for_step(0))

    train_bc.run_non_mutating_train_step(
        loss_step,
        model,
        optimizer,
        obs,
        targets,
        LossWeights(),
        "full_base",
        microbatch=1,
        autocast=False,
        timed=False,
    )

    assert not _state_changed(before_model, model.state_dict())
    assert optimizer.state_dict()["state"] == {}


def test_ema_update_after_mutating_step_not_warmup() -> None:
    model, optimizer, loss_step, scheduler = _cpu_tiny_training_objects()
    obs, targets = _cpu_batch()
    ema = train_bc.EmaTracker(
        model,
        train_bc.EmaConfig(enabled=True, decay=0.5, start_step=1, update_every_steps=1),
        torch.device("cpu"),
    )
    initial_ema = {key: value.clone() for key, value in ema.state.items()}
    train_bc.set_optimizer_lr(optimizer, scheduler.lr_for_step(0))

    train_bc.run_non_mutating_train_step(
        loss_step,
        model,
        optimizer,
        obs,
        targets,
        LossWeights(),
        "full_base",
        microbatch=1,
        autocast=False,
        timed=False,
    )
    assert ema.update_count == 0
    for key, value in initial_ema.items():
        torch.testing.assert_close(ema.state[key], value, rtol=0.0, atol=0.0)

    train_bc.run_step(
        loss_step, model, optimizer, obs, targets, LossWeights(), "full_base", microbatch=1, autocast=False, timed=False
    )
    ema.maybe_update(model, 1)
    assert ema.update_count == 1
    assert _state_changed(initial_ema, ema.state)


def test_ema_weights_context_restores_raw_model() -> None:
    model, _optimizer, _loss_step, _scheduler = _cpu_tiny_training_objects()
    ema = train_bc.EmaTracker(
        model,
        train_bc.EmaConfig(enabled=True, decay=0.5, start_step=0, update_every_steps=1),
        torch.device("cpu"),
    )
    raw_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for value in ema.state.values():
        value.add_(1.0)

    with ema_weights(model, ema):
        assert _state_changed(raw_state, model.state_dict())

    for key, value in raw_state.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0.0, atol=0.0)


class _DummyValidationSource:
    def __init__(self) -> None:
        self.batch = (_policy_batch(2, 1.0), 0.1)

    def next_batch(self) -> tuple[PolicyBatch, float]:
        return self.batch


def test_validation_returns_raw_and_ema_metrics_separately(monkeypatch: MonkeyPatch) -> None:
    model, _optimizer, _loss_step, _scheduler = _cpu_tiny_training_objects()
    ema = train_bc.EmaTracker(
        model,
        train_bc.EmaConfig(enabled=True, decay=0.5, start_step=0, update_every_steps=1),
        torch.device("cpu"),
    )
    ema.update_count = 1
    calls: list[str] = []

    def fake_eval(_batches: list[tuple[PolicyBatch, float]], **kwargs: object) -> dict[str, object]:
        current_model = kwargs["model"]
        assert isinstance(current_model, HydraPolicyNet)
        raw_sum = sum(
            float(tensor.detach().sum()) for tensor in model.state_dict().values() if tensor.is_floating_point()
        )
        calls.append("ema" if raw_sum > 1000.0 else "raw")
        return {"policy_nll": 1.0 if calls[-1] == "raw" else 0.5, "policy_accuracy": 0.25}

    for value in ema.state.values():
        value.add_(100.0)
    monkeypatch.setattr(validation, "evaluate_validation_batches", fake_eval)

    raw, averaged = evaluate_raw_and_ema(
        cast("RawMjaiValidationSource", _DummyValidationSource()),
        steps=1,
        model=model,
        device=torch.device("cpu"),
        weights=LossWeights(),
        autocast=False,
        ema_tracker=ema,
    )

    assert raw["policy_nll"] == 1.0
    assert averaged is not None
    assert averaged["policy_nll"] == 0.5
    assert calls == ["raw", "ema"]


def test_evaluate_raw_and_ema_uses_same_validation_batches(monkeypatch: MonkeyPatch) -> None:
    model, _optimizer, _loss_step, _scheduler = _cpu_tiny_training_objects()
    ema = train_bc.EmaTracker(
        model,
        train_bc.EmaConfig(enabled=True, decay=0.5, start_step=0, update_every_steps=1),
        torch.device("cpu"),
    )
    ema.update_count = 1
    batches = [(_policy_batch(2, 3.0), 0.1), (_policy_batch(2, 4.0), 0.2)]
    seen: list[list[float]] = []

    class Source:
        def __init__(self) -> None:
            self.index = 0

        def next_batch(self) -> tuple[PolicyBatch, float]:
            item = batches[self.index]
            self.index += 1
            return item

    def fake_eval(validation_batches: list[tuple[PolicyBatch, float]], **_kwargs: object) -> dict[str, object]:
        seen.append([float(batch.obs[0, 0, 0]) for batch, _fetch_ms in validation_batches])
        return {"policy_nll": 1.0, "policy_accuracy": 0.25}

    monkeypatch.setattr(validation, "evaluate_validation_batches", fake_eval)

    raw, averaged = evaluate_raw_and_ema(
        cast("RawMjaiValidationSource", Source()),
        steps=2,
        model=model,
        device=torch.device("cpu"),
        weights=LossWeights(),
        autocast=False,
        ema_tracker=ema,
    )

    assert raw["policy_nll"] == 1.0
    assert averaged is not None
    assert seen == [[3.0, 4.0], [3.0, 4.0]]


def test_ema_arg_validation() -> None:
    args = _valid_args(ema_decay=1.0)
    with pytest.raises(ValueError, match="ema-decay"):
        validate_args(args)
    args = _valid_args(ema_update_every_steps=0)
    with pytest.raises(ValueError, match="ema-update-every-steps"):
        validate_args(args)
    assert build_ema_config(_valid_args()) is None
    enabled = build_ema_config(
        _valid_args(ema_enabled=True, ema_decay=0.9, ema_start_step=2, ema_update_every_steps=3, ema_device="cpu")
    )
    assert enabled == train_bc.EmaConfig(enabled=True, decay=0.9, start_step=2, update_every_steps=3, device="cpu")


def test_ema_device_resolution_and_cpu_cuda_rejection() -> None:
    model, _optimizer, _loss_step, _scheduler = _cpu_tiny_training_objects()
    auto = train_bc.EmaTracker(model, train_bc.EmaConfig(enabled=True, device="auto"), torch.device("cpu"))
    assert auto.device.type == "cpu"
    explicit_cpu = train_bc.EmaTracker(model, train_bc.EmaConfig(enabled=True, device="cpu"), torch.device("cpu"))
    assert all(tensor.device.type == "cpu" for tensor in explicit_cpu.state.values())
    with pytest.raises(ValueError, match="requires CUDA training device"):
        train_bc.EmaTracker(model, train_bc.EmaConfig(enabled=True, device="cuda"), torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA EMA device test requires CUDA")
def test_cuda_ema_update_keeps_shadow_on_cuda() -> None:
    model, optimizer, loss_step, _scheduler = _cuda_tiny_training_objects()
    obs, targets = _cuda_batch()
    ema = train_bc.EmaTracker(model, train_bc.EmaConfig(enabled=True, decay=0.5, device="auto"), torch.device("cuda"))
    assert ema.device.type == "cuda"
    train_bc.run_step(
        loss_step, model, optimizer, obs, targets, LossWeights(), "full_base", microbatch=1, autocast=False, timed=False
    )
    ema.maybe_update(model, 1)
    assert ema.update_count == 1
    assert all(tensor.device.type == "cuda" for tensor in ema.state.values())


def test_ema_load_state_restores_runtime_device_and_counters() -> None:
    model, _optimizer, _loss_step, _scheduler = _cpu_tiny_training_objects()
    ema = train_bc.EmaTracker(model, train_bc.EmaConfig(enabled=True, device="cpu"), torch.device("cpu"))
    source = {key: tensor.detach().clone().add(2.0) for key, tensor in ema.state.items()}
    ema.load_state(source, update_count=7, last_update_step=11)
    assert ema.update_count == 7
    assert ema.last_update_step == 11
    assert all(tensor.device.type == "cpu" for tensor in ema.state.values())
    for key, value in source.items():
        torch.testing.assert_close(ema.state[key], value, rtol=0.0, atol=0.0)


def test_optimizer_update_changes_weights_and_must_be_counted() -> None:
    model, optimizer, loss_step, scheduler = _cpu_tiny_training_objects()
    obs, targets = _cpu_batch()
    before_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
    global_step = 0
    samples_seen = 0
    train_bc.set_optimizer_lr(optimizer, scheduler.lr_for_step(global_step))

    train_bc.run_step(
        loss_step, model, optimizer, obs, targets, LossWeights(), "full_base", microbatch=1, autocast=False, timed=False
    )
    global_step += 1
    samples_seen += obs.shape[0]

    assert _state_changed(before_model, model.state_dict())
    assert global_step == 1
    assert samples_seen == 2


def test_bounded_raw_sample_cap_includes_staged_batch() -> None:
    args = _valid_args(raw_mjai_data_dirs=[Path("/raw")], raw_mjai_max_samples=None, full_epoch=False, steps=10)
    raw_train_max_samples = args.raw_mjai_max_samples
    if args.raw_mjai_data_dirs and raw_train_max_samples is None and not args.full_epoch and args.steps is not None:
        raw_train_batches = args.steps + 1
        raw_train_max_samples = raw_train_batches * args.batch

    assert raw_train_max_samples == 44


def test_cosine_scheduler_uses_global_step_after_resume() -> None:
    scheduler = train_bc.LrScheduler(
        train_bc.LrSchedulerConfig(
            base_lr=1.0, min_lr=0.1, warmup_steps=0, total_steps=1000, target_games=None, schedule="cosine"
        )
    )
    resumed_global_step = 400

    resumed_lr = scheduler.lr_for_step(resumed_global_step)
    uninterrupted_lr = scheduler.lr_for_step(400)

    assert resumed_lr == uninterrupted_lr
    assert resumed_lr < 1.0


def test_fresh_bounded_run_starts_cosine_at_step_zero() -> None:
    scheduler = train_bc.LrScheduler(
        train_bc.LrSchedulerConfig(
            base_lr=1.0, min_lr=0.1, warmup_steps=0, total_steps=1000, target_games=None, schedule="cosine"
        )
    )

    assert scheduler.lr_for_step(0) == 1.0


def test_constant_scheduler_ignores_global_resume_step() -> None:
    scheduler = train_bc.LrScheduler(
        train_bc.LrSchedulerConfig(
            base_lr=1.0, min_lr=0.1, warmup_steps=0, total_steps=1000, target_games=None, schedule="constant"
        )
    )

    assert scheduler.lr_for_step(400) == 1.0


def test_target_games_schedule_uses_optimizer_steps_for_warmup() -> None:
    scheduler = train_bc.LrScheduler(
        train_bc.LrSchedulerConfig(
            base_lr=1.0, min_lr=0.1, warmup_steps=1000, total_steps=None, target_games=10_000, schedule="cosine"
        )
    )

    assert scheduler.lr_for_step(10, completed_games=9000) == 0.01
    assert scheduler.lr_for_step(1000, completed_games=0) == 1.0
    assert scheduler.lr_for_step(1000, completed_games=9000) < 0.2


def test_pre_main_accounting_log_fields_mark_non_mutating_warmup() -> None:
    accounting = train_bc.PreMainAccounting(
        compile_dry_run=True,
        warmup_steps_run=3,
        samples_consumed_pre_main=4,
        batches_consumed_pre_main=1,
    )

    fields = accounting.as_log_fields()

    assert fields["compile_dry_run"] is True
    assert fields["warmup_mode"] == train_bc.WARMUP_MODE
    assert fields["warmup_steps_counted"] == 0
    assert fields["samples_consumed_pre_main"] == 4
    assert fields["pre_main_batches_changed_weights"] is False


def _valid_args(**overrides: Any) -> argparse.Namespace:
    values = {
        "batch": 4,
        "microbatch": 2,
        "steps": 10,
        "warmup": 0,
        "checkpoint_every_steps": 0,
        "log_every_steps": 0,
        "validation_steps": 0,
        "validation_every": 0,
        "lr_warmup_steps": 0,
        "schedule_total_steps": 10,
        "schedule_target_games": None,
        "lr": 0.001,
        "min_lr": 0.0,
        "ema_enabled": False,
        "ema_decay": 0.999,
        "ema_start_step": 0,
        "ema_update_every_steps": 1,
        "grad_clip_norm": None,
        "lr_schedule": "cosine",
        "manifest": None,
        "checkpoint_out": None,
        "checkpoint_dir": None,
        "keep_step_checkpoints": False,
        "actions": 46,
        "torch_profiler_start_step": 0,
        "torch_profiler_stop_step": 1,
        "torch_profiler_trace": None,
        "full_epoch": False,
        "raw_mjai_max_samples": None,
        "resume": None,
        "raw_mjai_data_dirs": None,
        "raw_mjai_validation_augment": False,
        "validation_source_mode": "fixed",
        "validation_max_samples": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_raw_full_epoch_resume_without_cursor_is_rejected(tmp_path: Path) -> None:
    args = _valid_args(
        full_epoch=True,
        resume=tmp_path / "checkpoints" / "latest.pt",
        raw_mjai_data_dirs=[tmp_path / "raw"],
        steps=None,
    )
    (tmp_path / "raw").mkdir()

    with pytest.raises(ValueError, match="raw-MJAI resume is unsupported"):
        validate_raw_mjai_source_args(args)


def test_bounded_raw_resume_without_cursor_is_rejected(tmp_path: Path) -> None:
    args = _valid_args(
        resume=tmp_path / "checkpoints" / "latest.pt",
        raw_mjai_data_dirs=[tmp_path / "raw"],
    )
    (tmp_path / "raw").mkdir()

    with pytest.raises(ValueError, match="raw-MJAI resume is unsupported"):
        validate_raw_mjai_source_args(args)


def test_shard_resume_args_unaffected(tmp_path: Path) -> None:
    args = _valid_args(
        manifest=tmp_path / "manifest.json",
        resume=tmp_path / "checkpoints" / "latest.pt",
    )

    validate_args(args)


def test_validation_sample_cap_overrun_is_logged(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _ValidationStreamFixture.instances = []
    monkeypatch.setattr(validation, "RawMjaiDirectStream", _ValidationStreamFixture)
    args = _valid_args(
        batch=4,
        raw_mjai_data_dirs=[tmp_path / "raw"],
        raw_mjai_prefetch_batches=2,
        raw_mjai_queue_bound=8,
        raw_mjai_worker_threads=1,
        raw_mjai_max_games=None,
        raw_mjai_train_fraction=0.9,
        raw_mjai_augment=False,
        raw_mjai_validation_augment=False,
        validation_steps=2,
        validation_max_samples=6,
        validation_source_mode="fixed",
    )
    log_path = tmp_path / "events.jsonl"
    logger = JsonlLogger(log_path)

    source = RawMjaiValidationSource(args=args, events=logger)
    logger.close()
    source.close()

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    source_record = records[-1]
    assert source_record["event"] == "validation_source_deferred"
    assert source_record["requested_samples"] == 6
    assert source_record["actual_samples"] == 0
    assert source_record["sample_cap_overrun"] == 0
    assert source_record["requested_batches"] == 2
    assert source_record["actual_batches"] == 2
