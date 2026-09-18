"""Run-config digest and layout: canonical identity plus run directories.

Owns the RFC 8785 stable digest of the resolved config and the
idempotent output layout writer with its ``latest-run`` marker reader.
A differing ``run.yaml`` under an existing run id raises instead of
overwriting, so stale checkpoints fail closed on config drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from hydra2.artifacts.atomic import atomic_replace_bytes as atomic_replace_bytes
from hydra2.artifacts.digest import of_canonical as of_canonical
from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from hydra2.training._rc_sections import RunConfig as RunConfig

__all__ = [
    "_dump_run_yaml",
    "create_run_layout",
    "effective_run_id",
    "read_latest_run",
    "resolve_artifact_root",
    "run_config_digest",
    "run_config_to_dict",
    "run_dir_for",
]


def run_config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Resolved config as a plain YAML/JSON-serializable mapping (run.yaml)."""
    selection = config.selection
    return {
        "run": {
            "id": config.run.run_id,
            "kind": config.run.kind,
            "description": config.run.description,
        },
        "data": {
            "root": config.data.root,
            "scope": config.data.scope,
            "train_split": config.data.train_split,
            "val_split": config.data.val_split,
            "wall_disjoint": config.data.wall_disjoint,
            "leakage_check": config.data.leakage_check,
            "dora_width": config.data.dora_width,
            "actor_privileged_split": config.data.actor_privileged_split,
            "dataset_manifest_hash": config.data.dataset_manifest_hash,
            "shuffle_buffer_size": config.data.shuffle_buffer_size,
            "drop_last": config.data.drop_last,
            "num_workers": config.data.num_workers,
            "world_size": config.data.world_size,
            "replay_backend": config.data.replay_backend,
            "decode_prefetch": config.data.decode_prefetch,
            "expand_batch_games": config.data.expand_batch_games,
            "homogeneous_buckets": config.data.homogeneous_buckets,
        },
        "model": {
            "architecture_id": config.model.architecture_id,
            "action_count": config.model.action_count,
            "parameters": dict(config.model.parameters),
        },
        "weights": {
            "w_policy": config.weights.w_policy,
            "w_placement": config.weights.w_placement,
            "w_value": config.weights.w_value,
            "w_event": None if config.weights.w_event is None else dict(config.weights.w_event),
            "w_belief": None if config.weights.w_belief is None else dict(config.weights.w_belief),
            "privileged_source_hash": config.weights.privileged_source_hash,
            "label_smoothing": config.weights.label_smoothing,
        },
        "optimizer": {
            "id": config.optimizer.name,
            "lr": config.optimizer.lr,
            "betas": [config.optimizer.betas[0], config.optimizer.betas[1]],
            "weight_decay": config.optimizer.weight_decay,
            "head_lr_mult": config.optimizer.head_lr_mult,
        },
        "scheduler": {
            "id": config.scheduler.name,
            "warmup_updates": config.scheduler.warmup_updates,
            "parameters": dict(config.scheduler.parameters),
            "final_factor": config.scheduler.final_factor,
            "warmup_start_factor": config.scheduler.warmup_start_factor,
        },
        "runtime": {
            "adapter_id": config.runtime.adapter_id,
            "device": config.runtime.device,
            "precision": config.runtime.precision,
            "compile_mode": config.runtime.compile_mode,
        },
        "loop": {
            "microbatch_size": config.loop.microbatch_size,
            "accumulation_steps": config.loop.accumulation_steps,
            "gradient_clip_norm": config.loop.gradient_clip_norm,
            "max_updates": config.loop.max_updates,
            "checkpoint_frequency_updates": config.loop.checkpoint_frequency_updates,
            "precision": config.loop.precision,
            "keep_last_checkpoints": config.loop.keep_last_checkpoints,
            "stratified_sampling": config.loop.stratified_sampling,
            "sampling_ratios": None
            if config.loop.sampling_ratios is None
            else dict(config.loop.sampling_ratios),
            "log_per_type_metrics": config.loop.log_per_type_metrics,
            "fit_temperature": config.loop.fit_temperature,
            "fetch_prefetch_depth": config.loop.fetch_prefetch_depth,
        },
        "seeds": {
            "data_seed": config.seeds.data_seed,
            "train_seed": config.seeds.train_seed,
            "selection_seed": config.seeds.selection_seed,
        },
        "selection": {
            "N": selection.N,
            "pilot_s": selection.pilot_s,
            "delta": selection.delta,
            "alpha": selection.alpha,
            "beta": selection.beta,
            "design": selection.design,
            "declared_peeks": list(selection.declared_peeks),
            "margin": selection.margin,
            "resamples": selection.resamples,
            "seed": selection.seed,
        },
        "mirror": {
            "enabled": config.mirror.enabled,
            "project": config.mirror.project,
            "task_name": config.mirror.task_name,
            "offline_dir": config.mirror.offline_dir,
        },
        "eval": {
            "frequency_updates": config.eval.frequency_updates,
            "num_batches": config.eval.num_batches,
            "walls_dir": config.eval.walls_dir,
            "microbatch_size": config.eval.microbatch_size,
        },
        "output": {
            "artifact_root": config.output.artifact_root,
            "run_id": config.output.run_id,
        },
    }


def run_config_digest(config: RunConfig) -> str:
    """Stable ``sha256:<hex>`` identity of the resolved config (RFC 8785)."""
    return str(of_canonical(run_config_to_dict(config)))


# ---------------------------------------------------------------------------
# Output layout under <artifact_root>/runs/<id>/
# ---------------------------------------------------------------------------


def resolve_artifact_root(config: RunConfig) -> Path:
    """Resolved artifact root: ``output.artifact_root`` wins, else the lazy default."""
    if config.output.artifact_root is not None:
        return Path(config.output.artifact_root).resolve()
    from hydra2.config import artifact_root as _artifact_root

    return _artifact_root()


def effective_run_id(config: RunConfig) -> str:
    """``output.run_id`` override wins, else ``run.id``."""
    override = config.output.run_id
    return override if override is not None and len(override) > 0 else config.run.run_id


def run_dir_for(config: RunConfig, *, artifact_root: Path | str | None = None) -> Path:
    """``<artifact_root>/runs/<effective run id>`` (pure; creates nothing)."""
    run_id = effective_run_id(config)
    if artifact_root is not None:
        return (Path(artifact_root).resolve() / "runs" / run_id).resolve()
    return (resolve_artifact_root(config) / "runs" / run_id).resolve()


def _dump_run_yaml(config: RunConfig) -> bytes:
    text: str = yaml.safe_dump(
        run_config_to_dict(config), sort_keys=True, default_flow_style=False, allow_unicode=True
    )
    return text.encode("utf-8")


def create_run_layout(config: RunConfig, *, artifact_root: Path | str | None = None) -> Path:
    """Create the authoritative run directory tree (idempotent, fail closed).

    Layout under ``<artifact_root>/runs/<id>/``::

        run.yaml               resolved config (this module's output)
        manifests/             dataset / model / runtime / environment manifests
        checkpoints/           ckpt-<update>.pt + sidecars; resume source
        best-ckpt.pt           published on promotion only (absent until then)
        logs/train.log         line-oriented training log (created empty)
        logs/metrics.jsonl     one JSON object per line (created empty)
        mirror/                observer-mirror staging (ClearML offline sessions)
        eval/                  duplicate-wall evaluation reports

    Plus the ``<artifact_root>/runs/latest-run`` marker (text file holding
    the latest created run id). Re-creating with a byte-identical
    ``run.yaml`` is a no-op; a differing ``run.yaml`` raises instead of
    overwriting (never silently adopt a new spec under an old id).
    """
    run_dir = run_dir_for(config, artifact_root=artifact_root)
    for subdir in ("manifests", "checkpoints", "logs", "mirror", "eval"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    payload = _dump_run_yaml(config)
    target = run_dir / "run.yaml"
    if target.is_file():
        if target.read_bytes() != payload:
            raise ContractError(
                f"run.yaml mismatch in {run_dir}: refusing to overwrite a different "
                "resolved spec under the same run id (pick a new run id)"
            )
    else:
        atomic_replace_bytes(target, payload)
    for log_name in ("train.log", "metrics.jsonl"):
        log_path = run_dir / "logs" / log_name
        if not log_path.exists():
            log_path.touch(exist_ok=True)
    marker = run_dir.parent / "latest-run"
    atomic_replace_bytes(marker, (effective_run_id(config) + "\n").encode("utf-8"))
    return run_dir


def read_latest_run(artifact_root: Path | str | None = None) -> str | None:
    """Latest created run id from the ``runs/latest-run`` marker, if present."""
    if artifact_root is None:
        from hydra2.config import artifact_root as _artifact_root

        root = _artifact_root()
    else:
        root = Path(artifact_root).resolve()
    marker = root / "runs" / "latest-run"
    if not marker.is_file():
        return None
    text = marker.read_text(encoding="utf-8").strip()
    return text if text != "" else None
