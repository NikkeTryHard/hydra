"""Replay checkpoint — best-ckpt helpers plus selection persistence.

Owns the best-checkpoint digest / atomic-publish / verify-closed helpers
and the checkpoint/selection mixin hosting persistence
(``save_checkpoint`` / ``load_checkpoint``), the non-mutating replay peek,
the authorized-data verifier, and the gated held-out selection pair.
Manifest digests and checkpoint payloads move through here verbatim so
previously written checkpoints keep loading byte-identical.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.eval.statistics import SelectionConfig, score_selection
from hydra2.runtime.checkpoint import (
    build_manifest,
    capture_rng_state,
    load_checkpoint,
    save_checkpoint,
)
from hydra2.training.replay_state import (
    FORBIDDEN_REPLAY_KEYS as FORBIDDEN_REPLAY_KEYS,
)
from hydra2.training.replay_state import (
    ReplayState as ReplayState,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hydra2.eval.blocks import BlockTolerance, WallBlock
    from hydra2.eval.telemetry import ResourceTelemetry


def _best_ckpt_digest_for(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _atomic_publish_best(source: Path, dest: Path) -> str:
    """Atomically publish ``source`` bytes to ``dest``; return its digest."""
    src = Path(source)
    dst = Path(dest)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        raise ContractError(f"selection ckpt missing: {src}")
    data = src.read_bytes()
    digest = _best_ckpt_digest_for(data)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as fh:
        _ = fh.write(data)  # intentionally discarded: byte count unneeded after fsync
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dst)
    try:
        dir_fd = os.open(dst.parent, os.O_DIRECTORY)
    except Exception:
        return digest
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return digest


def _verify_best_ckpt(checkpoint_dir: Path, *, metric: float | None, digest: str | None) -> None:
    """Fail closed when a promoted best has no matching ``best-ckpt.pt``."""
    if metric is None and digest is None:
        return
    dest = Path(checkpoint_dir) / "best-ckpt.pt"
    if not dest.is_file():
        raise CorruptArtifactError(
            f"best-ckpt.pt missing for best_selection_metric={metric!r} in {dest.parent}"
        )
    if digest is not None:
        actual = _best_ckpt_digest_for(dest.read_bytes())
        if actual != digest:
            raise CorruptArtifactError(
                f"best-ckpt.pt digest mismatch: expected {digest}, got {actual}"
            )


class ActorLearnerReplayCheckpointMixin:
    """Checkpoint and selection persistence for the actor-learner replay.

    Split host for the persistence half of :class:`ActorLearnerReplay`;
    the engine subclass adds the construction and training halves and no
    overrides. Attribute access is duck-typed through the engine subclass.
    """

    model: Any
    optimizer: Any
    scheduler: Any | None
    dataset: Any
    config: Any
    handle: Any | None
    runtime_spec: Any | None
    checkpoint_dir: Path
    manifest_hashes: dict[str, str]
    state: ReplayState
    loss_history: list[dict[str, float]]
    _global_metrics_history: list[dict[str, float]]
    _historical_opponents: tuple[str, ...]
    privileged_store: Any
    _mirror: Any
    device: Any
    evaluation_wall_ids: frozenset[str]

    def _sampler_state_snapshot(self) -> dict[str, Any]:
        if hasattr(self.dataset, "get_sampler_state"):
            return self.dataset.get_sampler_state()
        cur = getattr(self.dataset, "cursor", 0)
        tot = len(self.dataset) if hasattr(self.dataset, "__len__") else 0
        return {"offset": int(cur), "seed": self.config.seed, "total": tot, "epoch": 0}

    def _restore_sampler_state(self, state: Any) -> None:
        if hasattr(self.dataset, "set_sampler_state"):
            self.dataset.set_sampler_state(state)
        else:
            with contextlib.suppress(Exception):
                dataset_any: Any = self.dataset
                _raw_offset: object = state.get("offset", 0) if isinstance(state, dict) else 0
                if isinstance(_raw_offset, bool):
                    _offset = int(_raw_offset)
                elif isinstance(_raw_offset, int):
                    _offset = _raw_offset
                else:
                    _offset = int(str(_raw_offset))
                dataset_any.cursor = _offset

    def save_checkpoint(self, path: Path | None = None) -> Path:
        dest = (
            Path(path)
            if path is not None
            else self.checkpoint_dir / f"ckpt-{self.state.global_update:06d}.pt"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None and hasattr(self.scheduler, "state_dict")
            else {},
            "training_state": self.state.to_dict(),
            "sampler_state": self._sampler_state_snapshot(),
            "rng_state": capture_rng_state(),
        }
        manifest = build_manifest(
            run_spec_hash=self.manifest_hashes["run_spec_hash"],
            model_spec_hash=self.manifest_hashes["model_spec_hash"],
            optimizer_spec_hash=self.manifest_hashes["optimizer_spec_hash"],
            scheduler_spec_hash=self.manifest_hashes["scheduler_spec_hash"],
            environment_hash=self.manifest_hashes["environment_hash"],
            rules_hash=self.manifest_hashes["rules_hash"],
            utility_manifest_hash=self.manifest_hashes["utility_manifest_hash"],
            action_schema_hash=self.manifest_hashes["action_schema_hash"],
            observation_schema_hash=self.manifest_hashes["observation_schema_hash"],
            dataset_manifest_hash=self.manifest_hashes["dataset_manifest_hash"],
            rollout_artifact_hash=None,
            payload=payload,
        )
        _ = save_checkpoint(destination=dest, manifest=manifest, payload=payload)
        return dest

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        _manifest, payload = load_checkpoint(
            source=path,
            expected_run_spec_hash=self.manifest_hashes["run_spec_hash"],
            expected_source_hash=self.manifest_hashes["dataset_manifest_hash"],
        )
        # Apply before mutating state that would mask errors: load verifies before touching
        from hydra2.runtime.checkpoint import apply_checkpoint

        apply_checkpoint(
            payload,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        # Restore training state and sampler
        raw_ts = payload.get("training_state", {})
        restored = ReplayState.from_dict(dict(raw_ts))
        # Replay is fp32-only: refuse cross-regime resume (and any non-fp32
        # checkpoint, which could only come from a lying writer).
        if restored.precision != "fp32" or self.config.precision != "fp32":
            raise CorruptArtifactError(
                f"checkpoint precision {restored.precision!r} != replay precision 'fp32'; "
                "refusing cross-regime resume"
            )
        self.state = restored
        sampler_state = payload.get("sampler_state")
        if sampler_state is not None:
            self._restore_sampler_state(sampler_state)
        # RNG already restored via apply_checkpoint
        _verify_best_ckpt(
            self.checkpoint_dir,
            metric=self.state.best_selection_metric,
            digest=self.state.best_ckpt_digest,
        )

    def replay_batches(self, *, batch_size: int, num_batches: int) -> list[list[str]]:
        """Deterministic replay: return decision_id order for next num_batches."""
        saved = self._sampler_state_snapshot()
        out: list[list[str]] = []
        for _ in range(num_batches):
            batch = self.dataset.next_batch(batch_size)
            if batch is None:
                break
            dids = batch.get("_decision_ids", [])
            out.append([str(x) for x in list(dids)])
        # Restore cursor so replay_batches is non-mutating peek (deterministic replay)
        self._restore_sampler_state(saved)
        return out

    def verify_authorized(self) -> None:
        """Verify the underlying dataset is authoritative and disjoint from eval walls."""
        # Dataset already verified shards on construction; re-assert cursor invariants
        if len(self.dataset) == 0:
            raise ContractError("authorized dataset is empty")
        for row in getattr(self.dataset, "_rows", []):
            for bad in FORBIDDEN_REPLAY_KEYS:
                if bad in row:
                    raise ContractError(
                        f"privileged field {bad!r} in authorized row {row.get('decision_id')!r}"
                    )
        if len(self.evaluation_wall_ids) > 0:
            for row_any in getattr(self.dataset, "_rows", []):
                row: dict[str, Any] = row_any
                gid: str = str(row.get("game_id", ""))
                if any(w in gid for w in self.evaluation_wall_ids):
                    raise ContractError(f"evaluation wall overlap detected for {gid!r}")

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

        Mirrors :meth:`SupervisedLoop.evaluate_selection`; never called by
        :meth:`train`. Delegates to :func:`score_selection` (telemetry policy
        + frozen peek-discipline guard enforced); returns the metric only.
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
        """Atomically publish ``ckpt`` to ``best-ckpt.pt`` iff gated score improves."""
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
    "ActorLearnerReplayCheckpointMixin",
    "_atomic_publish_best",
    "_best_ckpt_digest_for",
    "_verify_best_ckpt",
]
