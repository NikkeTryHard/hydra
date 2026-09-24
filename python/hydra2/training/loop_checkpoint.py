"""Supervised-loop checkpoint: sampler, persistence, report, selection.

Owns the local authoritative persistence half of
:class:`SupervisedLoop`: the sampler snapshot/restore helpers, the atomic
checkpoint publish, the verified resume that validates manifest
identities before mutating any runtime object, the best-checkpoint
binding that keeps a restored best tied to its published file, the
frozen-model report, and the gated held-out selection pair. Manifest
digests and checkpoint payloads move through here verbatim so
previously written checkpoints keep loading byte-identical.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.eval.selection import SelectionConfig, score_selection
from hydra2.runtime.checkpoint import (
    build_manifest,
    capture_rng_state,
    load_checkpoint,
    save_checkpoint,
)
from hydra2.runtime.checkpoint_publish import (
    _BackgroundCheckpointWriter as _BackgroundCheckpointWriter,
)
from hydra2.runtime.checkpoint_publish import (
    _snapshot_payload_to_cpu as _snapshot_payload_to_cpu,
)
from hydra2.training.loop_batch import (
    _batch_action_kinds as _batch_action_kinds,
)
from hydra2.training.loop_batch import (
    _model_forward as _model_forward,
)
from hydra2.training.loop_batch import (
    _move_batch_to_device as _move_batch_to_device,
)
from hydra2.training.loop_batch import (
    _validate_batch_no_privileged as _validate_batch_no_privileged,
)
from hydra2.training.loop_state import (
    TrainingState as TrainingState,
)
from hydra2.training.loop_state import _atomic_publish_best as _atomic_publish_best
from hydra2.training.loop_state import (
    _verify_best_ckpt as _verify_best_ckpt,
)
from hydra2.training.objectives_metrics import (
    compute_metrics as compute_metrics,
)
from hydra2.training.objectives_metrics import (
    compute_per_type_metrics as compute_per_type_metrics,
)
from hydra2.training.objectives_metrics import (
    fit_temperature_scaling as fit_temperature_scaling,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from hydra2.eval.blocks import BlockTolerance, WallBlock
    from hydra2.eval.telemetry import ResourceTelemetry


class SupervisedLoopCheckpointMixin:
    """Sampler, persistence, report, and selection for :class:`SupervisedLoop`.

    Split base hosting the sampler helpers (shared by the engine fetch
    path and the persistence path, mirroring the replay split), the
    checkpoint persistence, the frozen-model report, and the gated
    held-out selection pair; the engine subclass adds construction and
    training and no overrides. Attribute access is duck-typed through
    the subclass.
    """

    model: Any
    optimizer: Any
    scheduler: Any | None
    dataset: Any
    config: Any
    device: Any
    checkpoint_dir: Path
    manifest_hashes: dict[str, str]
    state: TrainingState
    _forward_autocast: Any
    _mirror: Any
    _mlflow_mirror: Any

    # ------------------------------------------------------------------
    # Sampler state helpers
    # ------------------------------------------------------------------

    def _sampler_state_snapshot(self) -> dict[str, Any]:
        if hasattr(self.dataset, "get_sampler_state"):
            return self.dataset.get_sampler_state()
        # Fallback: cursor int
        cur = getattr(self.dataset, "cursor", 0)
        tot = len(self.dataset) if hasattr(self.dataset, "__len__") else 0
        return {"offset": int(cur), "seed": self.config.seed, "total": tot, "epoch": 0}

    def _restore_sampler_state(self, state: Any) -> None:
        if hasattr(self.dataset, "set_sampler_state"):
            self.dataset.set_sampler_state(state)
        else:
            # Fallback best-effort: set cursor attribute
            with contextlib.suppress(Exception):
                _raw_offset: object = state.get("offset", 0) if isinstance(state, dict) else 0
                if isinstance(_raw_offset, bool):
                    _offset = int(_raw_offset)
                elif isinstance(_raw_offset, int):
                    _offset = _raw_offset
                else:
                    _offset = int(str(_raw_offset))
                self.dataset.cursor = _offset

    # ------------------------------------------------------------------
    # Checkpointing — local authoritative artifacts
    # ------------------------------------------------------------------

    def save_checkpoint(self, destination: Path | None = None, *, on_published: Any = None) -> Path:
        """Atomically publish a checkpoint (local authoritative).

        An observer mirror (see tracking/), if configured, may ``shutil.copy``
        from this path but MUST NOT overwrite it — no code path in this module
        writes through a mirror.

        The payload is snapshotted to CPU synchronously (~50ms: the same
        copies the manifest hasher already makes), then the manifest build
        plus the single ``torch.save`` publish run on the loop's background
        writer while training continues. Loops that own their writer (the
        default: unit flows and one-shot ``train()`` callers) wait inline,
        so the file is on disk on return exactly as before; loops with an
        attached external writer (streaming) return the reserved path
        immediately and the bytes land asynchronously — those callers MUST
        drain (per-update poll, pre-eval drain, end-of-run close) and the
        ``on_published`` hook fires only after the generation lands, so the
        mirror never announces bytes that failed to land. Failure mode: a
        worker error raises on the calling thread at the next wait/poll
        (fail closed, never a silent skip).

        Returns the destination path published.
        """
        if destination is None:
            destination = self.checkpoint_dir / f"checkpoint-{self.state.global_update:06d}.pt"
        else:
            destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Payload sections required by the checkpoint manifest (6 keys:
        # model_state, optimizer_state, scheduler_state, training_state,
        # sampler_state, rng_state; run_spec_hash and the selected source hash
        # verified before mutating runtime objects)
        # Scheduler state may be empty dict when no scheduler
        sched_state: Any
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            try:
                sched_state = self.scheduler.state_dict()
            except Exception:
                sched_state = {}
        else:
            sched_state = {}

        # Prefetch-aware sampler state: state.sampler_cursor tracks consumed
        # (not live prefetch head) so resume re-fetches at most the 1-deep
        # lookahead deterministically instead of skipping it. Sync path is
        # identical (consumed == live, no outstanding prefetch).
        _consumed = self.state.sampler_cursor
        sampler_state: Any = (
            _consumed if isinstance(_consumed, dict) else self._sampler_state_snapshot()
        )
        payload: dict[str, Any] = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": sched_state,
            "training_state": self.state.to_dict(),
            "sampler_state": sampler_state,
            "rng_state": capture_rng_state(),
        }
        # Snapshot on the calling thread BEFORE submit: state_dict() aliases
        # live parameters and optimizer state, and the next update mutates
        # them; the worker must only ever see detached CPU clones. Timed
        # for the stage log (observer-only wall time, never hashed).
        import time as _time

        _snap_t0 = _time.perf_counter()
        _snap_wall = _time.time()
        snapshot = _snapshot_payload_to_cpu(payload)
        _snap_ms = (_time.perf_counter() - _snap_t0) * 1000.0
        _span_sink = self.__dict__.get("_span_sink")
        _main_spans: tuple[tuple[str, float, float], ...] = (
            (("snapshot", _snap_wall, _snap_ms),) if _span_sink is not None else ()
        )
        hashes = dict(self.manifest_hashes)
        writer = self._checkpoint_writer()
        external = bool(self.__dict__.get("_ckpt_writer_external", False))
        generation = int(self.state.global_update)
        dest = Path(destination)

        def _publish() -> Path:
            manifest = build_manifest(
                run_spec_hash=hashes["run_spec_hash"],
                model_spec_hash=hashes["model_spec_hash"],
                optimizer_spec_hash=hashes["optimizer_spec_hash"],
                scheduler_spec_hash=hashes["scheduler_spec_hash"],
                environment_hash=hashes["environment_hash"],
                rules_hash=hashes["rules_hash"],
                utility_manifest_hash=hashes["utility_manifest_hash"],
                action_schema_hash=hashes["action_schema_hash"],
                observation_schema_hash=hashes["observation_schema_hash"],
                dataset_manifest_hash=hashes["dataset_manifest_hash"],
                rollout_artifact_hash=None,
                payload=snapshot,
            )
            return save_checkpoint(destination=dest, manifest=manifest, payload=snapshot)

        def _done() -> None:
            # dest is resolved above, so the hook fires with the published
            # path whether wait() runs it inline (loop-owned) or poll()
            # runs it later (external): exactly once either way.
            if on_published is not None:
                on_published(dest)

        future = writer.submit(
            _publish,
            generation=generation,
            on_done=_done if on_published is not None else None,
            description=f"loop-checkpoint-{generation:06d}",
            span_sink=_span_sink,
            main_spans=_main_spans,
        )
        if not external:
            # Loop-owned writer: preserve the historical synchronous contract
            # (file on disk on return); wait() fires the hook inline here.
            writer.wait(future)
        return dest

    def attach_checkpoint_writer(self, writer: Any, *, span_sink: Any = None) -> None:
        """Attach a caller-owned background writer (streaming takes over).

        After attach, :meth:`save_checkpoint` returns the reserved path
        immediately and the caller owns draining (per-update poll, pre-eval
        drain, end-of-run close). Refuses to orphan in-flight saves: if the
        current writer still holds pending generations, attaching raises
        instead of abandoning them (abandoned futures would fire mirror
        hooks for a writer nobody polls).
        """
        current = self.__dict__.get("_ckpt_writer")
        if current is not None and getattr(current, "_pending", None):
            raise ContractError("cannot attach a checkpoint writer while saves are still pending")
        self.__dict__["_ckpt_writer"] = writer
        self.__dict__["_ckpt_writer_external"] = True
        self.__dict__["_span_sink"] = span_sink

    def drain_checkpoints(self) -> None:
        """Block until every submitted generation lands, firing due hooks."""
        writer = self.__dict__.get("_ckpt_writer")
        if writer is None:
            return
        writer.drain()

    def _checkpoint_writer(self) -> Any:
        """Loop-owned writer, lazily created (owns sync semantics by default)."""
        writer = self.__dict__.get("_ckpt_writer")
        if writer is None:
            writer = _BackgroundCheckpointWriter()
            self.__dict__["_ckpt_writer"] = writer
            self.__dict__["_ckpt_writer_external"] = False
        return writer

    def _poll_checkpoint_writer(self) -> None:
        """Fire hooks for landed generations; raise the first worker error.

        Cheap when idle. Called once per training update so a failed save
        stops training at the next update boundary (fail closed), and
        confirmed generations announce to the mirror with at most one
        update of delay.
        """
        writer = self.__dict__.get("_ckpt_writer")
        if writer is None:
            return
        writer.poll()

    def resume_from_checkpoint(self, source: Path) -> None:
        """Verified resume: validates manifest before mutating any runtime object.

        Verifies ``run_spec_hash`` and ``dataset_manifest_hash`` and per-section
        hashes before applying.  Any mismatch raises before mutation.

        On success, restores model, optimizer, scheduler, TrainingState,
        sampler cursor and RNG so that training continues bit-identically.

        Drains pending background saves first: a resume cannot start while
        its own checkpoint file is still being written (loading a torn
        file would fail closed on digest mismatch, but waiting is cheaper
        and never reads a partial write).
        """
        self.drain_checkpoints()
        source = Path(source)
        manifest, payload = load_checkpoint(
            source=source,
            expected_run_spec_hash=self.manifest_hashes["run_spec_hash"],
            expected_source_hash=self.manifest_hashes["dataset_manifest_hash"],
        )
        # Manifest is already validated inside load_checkpoint; double-check source
        # identities match this loop's expectations
        if manifest.model_spec_hash != self.manifest_hashes["model_spec_hash"]:
            raise CorruptArtifactError(
                f"checkpoint model_spec_hash {manifest.model_spec_hash} "
                f"!= expected {self.manifest_hashes['model_spec_hash']}"
            )
        # Apply after verification (run_spec_hash and the selected source hash
        # verified before mutating runtime objects; model/optimizer/scheduler
        # restored before training state, sampler, RNG)
        # Restore model / optimizer / scheduler from payload (these are device-agnostic CPU tensors)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if (
            self.scheduler is not None
            and "scheduler_state" in payload
            and payload["scheduler_state"]
        ):
            try:
                self.scheduler.load_state_dict(payload["scheduler_state"])
            except Exception as exc:
                raise CorruptArtifactError(f"scheduler state incompatible: {exc}") from exc
        # Restore training state and sampler/RNG
        raw_training = payload.get("training_state")
        if isinstance(raw_training, dict):
            restored = TrainingState.from_dict(raw_training)
            # Precision regime must match: a fp32 checkpoint resumed under a
            # bf16 loop (or reverse) would silently change numerics.
            if restored.precision != self.config.precision:
                raise CorruptArtifactError(
                    f"checkpoint precision {restored.precision!r} != "
                    f"loop precision {self.config.precision!r}; refusing cross-regime resume"
                )
            self.state = restored
        else:
            raise CorruptArtifactError("training_state missing or malformed in checkpoint")
        sampler_state = payload.get("sampler_state")
        if sampler_state is not None:
            self._restore_sampler_state(sampler_state)
            # Keep state.sampler_cursor in sync with dataset snapshot
            self.state.sampler_cursor = self._sampler_state_snapshot()
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            # capture_rng_state / _restore is via runtime.checkpoint apply path; inline here
            from hydra2.runtime.checkpoint import _restore_rng_state

            _restore_rng_state(rng_state)
        self.model.to(self.device)
        self.model.train()
        # Best-ckpt binding: a restored best must still match its published file.
        _verify_best_ckpt(
            self.checkpoint_dir,
            metric=self.state.best_selection_metric,
            digest=self.state.best_ckpt_digest,
        )
        # Note: optimizer state tensors remain on CPU after load; the adapter's
        # handle would have moved them on setup.  For plain loop we keep CPU
        # and let next step handle device transfer via model's device.

    # ------------------------------------------------------------------
    # Reporting — masked NLL, top-k, calibration, support/confusion, strata
    # ------------------------------------------------------------------

    def evaluate_report(
        self,
        eval_batches: list[dict[str, Any]] | Any,
        *,
        weights: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute report over eval batches (no grad) with the frozen model.

        Returns dict with keys: ``masked_nll``, ``top1``/``top3``/``top5``,
        ``calibration_ece``, ``support_min``/``max``,
        ``legal_uniform_nll``/``legal_uniform_gap``,
        ``legal_uniform_comparison`` (checklist alias),
        ``strata`` (kinds present when batches carry ``"_action_kinds"``)
        and ``confusion`` (legacy ``0.0``), plus flattened
        ``per_type/<kind>/{n,nll,top1,top3,ece,recall,low_support}`` when kinds
        are complete (``recall`` equals ``top1`` by construction;
        ``low_support`` flags ``n<30``, report-only),
        and post-hoc ``temperature``/``calibrated_nll``/``calibrated_ece``
        fit on the pooled eval rows (validation-only, never training).
        """
        self.model.eval()
        total_nll = 0.0
        total_top1 = 0.0
        total_top3 = 0.0
        total_top5 = 0.0
        total_ece = 0.0
        total_uniform_nll = 0.0
        total_uniform_gap = 0.0
        n = 0
        # Pooled eval tensors for per-type scorecards + temperature (Wave 3-B
        # logging): overall means below stay mean-of-batch-means for
        # checklist comparability; per-type/temperature pool rows.
        pooled_logits: list[torch.Tensor] = []
        pooled_targets: list[torch.Tensor] = []
        pooled_masks: list[torch.Tensor] = []
        pooled_kinds: list[str] = []
        kinds_complete = True

        # eval_batches may be iterable of batch dicts or a dataset with iter_batches
        if hasattr(eval_batches, "iter_batches"):
            _iter_batches: Callable[..., Iterable[dict[str, Any]]] = eval_batches.iter_batches
            batches = list(_iter_batches(4, max_batches=5))
        elif isinstance(eval_batches, list):
            batches = eval_batches
        else:
            batches = list(eval_batches)

        with torch.no_grad():
            for raw in batches:
                _validate_batch_no_privileged(raw)
                batch = _move_batch_to_device(raw, self.device)
                # AMP: forward runs under bf16 autocast when enabled; metrics stay fp32.
                with self._forward_autocast():
                    out = _model_forward(self.model, batch)
                eval_logits: torch.Tensor = out["policy_logits"]
                eval_targets: torch.Tensor = batch["chosen_action_id"]
                eval_mask: torch.Tensor = batch["legal_mask"]
                metrics = compute_metrics(eval_logits, eval_targets, eval_mask)
                total_nll += metrics["masked_nll"]
                total_top1 += metrics["top1"]
                total_top3 += metrics["top3"]
                total_top5 += metrics["top5"]
                total_ece += metrics["calibration_ece"]
                total_uniform_nll += metrics["legal_uniform_nll"]
                total_uniform_gap += metrics["legal_uniform_gap"]
                n += 1
                pooled_logits.append(eval_logits.detach().to("cpu"))
                pooled_targets.append(eval_targets.detach().to("cpu"))
                pooled_masks.append(eval_mask.detach().to("cpu"))
                batch_kinds = _batch_action_kinds(batch, eval_targets)
                if batch_kinds is None:
                    kinds_complete = False
                else:
                    pooled_kinds.extend(batch_kinds)

        if n == 0:
            raise ContractError("evaluate_report requires at least one eval batch")

        report: dict[str, Any] = {
            "masked_nll": total_nll / n,
            "top1": total_top1 / n,
            "top3": total_top3 / n,
            "top5": total_top5 / n,
            "calibration_ece": total_ece / n,
            "legal_uniform_nll": total_uniform_nll / n,
            "legal_uniform_gap": total_uniform_gap / n,
            "support_min": 0.0,
            "support_max": 0.0,
            "confusion": 0.0,
            "strata": 0.0,
            "legal_uniform_comparison": total_nll / n,  # alias for checklist
            "num_eval_batches": float(n),
        }
        # Pooled per-type scorecards + post-hoc temperature (logging only;
        # failures degrade to legacy keys, never fail the report).
        try:
            flat_logits = torch.cat(pooled_logits, dim=0)
            flat_targets = torch.cat(pooled_targets, dim=0)
            flat_masks = torch.cat(pooled_masks, dim=0)
            flat_masked = flat_logits.to(torch.float32).masked_fill(~flat_masks, float("-inf"))
            flat_probs = torch.softmax(flat_masked, dim=-1)
            _, flat_pred = flat_probs.max(dim=1)
            counts: dict[int, int] = {}
            pred_list: list[int] = flat_pred.tolist()
            for _p in pred_list:
                _pi = _p
                counts[_pi] = counts.get(_pi, 0) + 1
            if len(counts) > 0:
                report["support_min"] = float(min(counts.values()))
                report["support_max"] = float(max(counts.values()))
            _kinds_ok = kinds_complete and len(pooled_kinds) == flat_targets.shape[0]
            if _kinds_ok and self.config.log_per_type_metrics:
                per_type = compute_per_type_metrics(
                    flat_logits, flat_targets, flat_masks, pooled_kinds
                )
                for _kind in sorted(per_type):
                    _km = per_type[_kind]
                    report[f"per_type/{_kind}/n"] = _km["n"]
                    report[f"per_type/{_kind}/nll"] = _km["nll"]
                    report[f"per_type/{_kind}/top1"] = _km["top1"]
                    report[f"per_type/{_kind}/top3"] = _km["top3"]
                    report[f"per_type/{_kind}/ece"] = _km["ece"]
                    report[f"per_type/{_kind}/recall"] = _km["recall"]
                    report[f"per_type/{_kind}/low_support"] = _km["low_support"]
                report["strata"] = float(len(per_type))
            if self.config.fit_temperature:
                temp = fit_temperature_scaling(flat_logits, flat_targets, flat_masks)
                report["temperature"] = temp["temperature"]
                report["calibrated_nll"] = temp["nll_after"]
                report["calibrated_ece"] = temp["ece_after"]
            else:
                report["temperature"] = 1.0
                report["calibrated_nll"] = report["masked_nll"]
                report["calibrated_ece"] = report["calibration_ece"]
        except Exception:
            _ = report.setdefault("temperature", 1.0)
            _ = report.setdefault("calibrated_nll", report["masked_nll"])
            _ = report.setdefault("calibrated_ece", report["calibration_ece"])
        self.model.train()
        return report

    # ------------------------------------------------------------------
    # Held-out placement selection — best_selection_metric + best-ckpt
    # ------------------------------------------------------------------

    def evaluate_selection(
        self,
        blocks: tuple[WallBlock, ...],
        telemetry_by_game: Mapping[str, ResourceTelemetry],
        config: SelectionConfig,
        peek_index: int,
        tolerance: BlockTolerance | None = None,
    ) -> float:
        """Gated selection score over VALID wall blocks only (lower-better S1).

        Delegates to :func:`score_selection` (telemetry policy + frozen
        peek-discipline guard enforced); returns the metric only. Not called
        by :meth:`train`; the caller scores a wall-disjoint held-out split
        by hand, then promotes by hand via :meth:`maybe_promote_best`.
        """
        metric, _, _ = score_selection(
            blocks, telemetry_by_game, config, peek_index, tolerance=tolerance
        )
        return metric

    def maybe_promote_best(
        self,
        metric: float,
        ckpt: Path,
        *,
        config: SelectionConfig,
        blocks: tuple[WallBlock, ...],
        telemetry_by_game: Mapping[str, ResourceTelemetry],
        peek_index: int,
        tolerance: BlockTolerance | None = None,
    ) -> bool:
        """Atomically publish ``ckpt`` to ``best-ckpt.pt`` iff gated score improves.

        Re-scores ``blocks``/``telemetry_by_game`` through :func:`score_selection`
        (guard + telemetry policy re-enforced) and requires ``metric`` to equal
        the gated score — unguarded promotion is impossible. Best is the
        running minimum. Publish is atomic (temp + ``os.replace`` + fsync)
        and records the published-file digest in
        ``state.best_ckpt_digest``. Never called by :meth:`train`.
        """
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ContractError(f"selection metric must be finite, got {metric!r}")
        if not math.isfinite(float(metric)):
            raise ContractError(f"selection metric must be finite, got {metric!r}")
        if not isinstance(config, SelectionConfig):
            raise ContractError(f"config must be a SelectionConfig, got {type(config).__name__}")
        expected, _, _ = score_selection(
            blocks, telemetry_by_game, config, peek_index, tolerance=tolerance
        )
        if metric != expected:
            raise ContractError(
                f"promotion metric {metric!r} != gated score {expected!r} "
                "for this blocks/telemetry/peek (score first via evaluate_selection)"
            )
        best = self.state.best_selection_metric
        if best is not None and not (metric < best):
            return False
        digest = _atomic_publish_best(Path(ckpt), self.checkpoint_dir / "best-ckpt.pt")
        self.state.best_selection_metric = metric
        self.state.best_ckpt_digest = digest
        return True


__all__ = [
    "SupervisedLoopCheckpointMixin",
]
