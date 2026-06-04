from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from hydra_learner.data.batches import targets_for_compiled_loss, tensors_from_policy_batch
from hydra_learner.data.raw_mjai import RawMjaiDirectStream
from hydra_learner.data.shard_contracts import PolicyBatch
from hydra_learner.model.optim import ema_weights
from hydra_learner.telemetry.metrics import EvalStats, summarize_eval
from hydra_learner.training.step import evaluate_batch

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    from hydra_learner.model import HydraPolicyNet
    from hydra_learner.model.losses import LossWeights
    from hydra_learner.model.optim import EmaTracker
    from hydra_learner.telemetry.logging import JsonlLogger


@dataclass(frozen=True)
class ValidationSourceInfo:
    mode: str
    requested_batches: int
    actual_batches: int
    requested_samples: int | None
    actual_samples: int
    sample_cap_overrun: int
    full_batches: bool
    augment: bool


CachedValidationBatch = tuple[PolicyBatch, float]


class RawMjaiValidationSource:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        events: JsonlLogger,
    ) -> None:
        self.mode: str = args.validation_source_mode
        self._args = args
        self._events = events
        self._cached: list[CachedValidationBatch] = []
        self._stream: RawMjaiDirectStream | None = None
        self._index = 0
        self._fixed_prepared = False
        self.info = ValidationSourceInfo(
            mode=self.mode,
            requested_batches=0,
            actual_batches=0,
            requested_samples=args.validation_max_samples,
            actual_samples=0,
            sample_cap_overrun=0,
            full_batches=True,
            augment=args.raw_mjai_validation_augment,
        )
        if args.validation_steps <= 0 or not args.raw_mjai_data_dirs:
            events.write(
                "validation_source_skipped",
                {"validation_steps": args.validation_steps, "has_raw_mjai": bool(args.raw_mjai_data_dirs)},
            )
            return
        if self.mode == "fixed":
            self.info = ValidationSourceInfo(
                mode=self.mode,
                requested_batches=args.validation_steps,
                actual_batches=args.validation_steps,
                requested_samples=args.validation_max_samples,
                actual_samples=0,
                sample_cap_overrun=0,
                full_batches=True,
                augment=args.raw_mjai_validation_augment,
            )
            events.write("validation_source_deferred", asdict(self.info))
            return
        self._stream = self._open_stream(max_samples=None)
        self.info = ValidationSourceInfo(
            mode=self.mode,
            requested_batches=args.validation_steps,
            actual_batches=0,
            requested_samples=args.validation_max_samples,
            actual_samples=0,
            sample_cap_overrun=0,
            full_batches=True,
            augment=args.raw_mjai_validation_augment,
        )
        events.write("validation_source", asdict(self.info))

    def _open_stream(self, *, max_samples: int | None) -> RawMjaiDirectStream:
        args = self._args
        self._events.write(
            "validation_source_stream_start",
            {
                "mode": self.mode,
                "requested_batches": args.validation_steps,
                "max_samples": max_samples,
                "transport": "stdout",
            },
        )
        stream = RawMjaiDirectStream(
            data_dirs=args.raw_mjai_data_dirs,
            batch_size=args.batch,
            prefetch_batches=args.raw_mjai_prefetch_batches,
            queue_bound=args.raw_mjai_queue_bound,
            worker_threads=args.raw_mjai_worker_threads,
            max_games=args.raw_mjai_max_games,
            max_samples=max_samples,
            train_fraction=args.raw_mjai_train_fraction,
            augment=args.raw_mjai_validation_augment,
            split="validation",
        )
        stream.start()
        self._events.write("validation_source_stream_started", {})
        return stream

    def _prepare_fixed(self) -> None:
        if self._fixed_prepared:
            return
        args = self._args
        stream = self._open_stream(max_samples=args.validation_max_samples)
        try:
            try:
                for _ in range(args.validation_steps):
                    self._events.write("validation_source_batch_fetch_start", {"batch_index": len(self._cached)})
                    self._cached.append(stream.next_batch())
                    self._events.write(
                        "validation_source_batch_fetch_complete",
                        {
                            "batch_index": len(self._cached) - 1,
                            "rows": self._cached[-1][0].actions.shape[0],
                        },
                    )
            except StopIteration as exc:
                raise ValueError(
                    "raw MJAI fixed validation window exhausted before --validation-steps batches"
                ) from exc
        finally:
            stream.close()
        samples = sum(batch.actions.shape[0] for batch, _fetch_ms in self._cached)
        batches = len(self._cached)
        full_batches = all(batch.actions.shape[0] == args.batch for batch, _fetch_ms in self._cached)
        overrun = 0 if args.validation_max_samples is None else max(0, samples - args.validation_max_samples)
        self.info = ValidationSourceInfo(
            mode=self.mode,
            requested_batches=args.validation_steps,
            actual_batches=batches,
            requested_samples=args.validation_max_samples,
            actual_samples=samples,
            sample_cap_overrun=overrun,
            full_batches=full_batches,
            augment=args.raw_mjai_validation_augment,
        )
        self._fixed_prepared = True
        self._events.write("validation_source", asdict(self.info))

    def next_batch(self) -> CachedValidationBatch:
        if self.mode == "fixed":
            self._prepare_fixed()
            if not self._cached:
                raise StopIteration("fixed validation window is empty")
            item = self._cached[self._index]
            self._index = (self._index + 1) % len(self._cached)
            return item
        if self._stream is None:
            raise StopIteration("streaming validation source is not open")
        return self._stream.next_batch()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def collect_validation_batches(source: RawMjaiValidationSource, *, steps: int) -> list[CachedValidationBatch]:
    return [source.next_batch() for _ in range(steps)]


def evaluate_validation_batches(
    batches: list[CachedValidationBatch],
    *,
    model: nn.Module,
    device: torch.device,
    weights: LossWeights,
    autocast: bool,
) -> dict[str, object]:
    step_eval: list[EvalStats] = []
    for val_batch, val_fetch_ms in batches:
        val_obs, _val_legal, _val_labels, val_targets, _val_input_timing = tensors_from_policy_batch(
            val_batch, device, val_fetch_ms
        )
        val_targets = targets_for_compiled_loss(val_targets, weights)
        step_eval.append(evaluate_batch(model, val_obs, val_targets, weights, autocast))
    return summarize_eval(step_eval)


def evaluate_validation_source(
    source: RawMjaiValidationSource,
    *,
    steps: int,
    model: nn.Module,
    device: torch.device,
    weights: LossWeights,
    autocast: bool,
) -> dict[str, object]:
    batches = collect_validation_batches(source, steps=steps)
    return evaluate_validation_batches(
        batches,
        model=model,
        device=device,
        weights=weights,
        autocast=autocast,
    )


def evaluate_raw_and_ema(
    source: RawMjaiValidationSource,
    *,
    steps: int,
    model: HydraPolicyNet,
    device: torch.device,
    weights: LossWeights,
    autocast: bool,
    ema_tracker: EmaTracker | None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    batches = collect_validation_batches(source, steps=steps)
    raw_metrics = evaluate_validation_batches(
        batches,
        model=model,
        device=device,
        weights=weights,
        autocast=autocast,
    )
    if ema_tracker is None or ema_tracker.update_count == 0:
        return raw_metrics, None
    with ema_weights(model, ema_tracker):
        ema_metrics = evaluate_validation_batches(
            batches,
            model=model,
            device=device,
            weights=weights,
            autocast=autocast,
        )

    return raw_metrics, ema_metrics
