"""PPO control config digest compatibility helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hydra_learner.ppo.config import PpoControlConfig


def _json_config(config: PpoControlConfig) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _config_digest(config: PpoControlConfig) -> str:
    payload = _json_config(config)
    payload["resume"] = None
    return _payload_digest(payload)


def _payload_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_LEGACY_RESUME_RUN_LOCAL_DIGEST_FIELDS = ("output_dir", "tensorboard_dir", "steps")

_CHECKPOINT_RETENTION_DIGEST_FIELDS = ("keep_step_checkpoints",)

_RUN_LOCAL_DIGEST_FIELDS = (
    *_LEGACY_RESUME_RUN_LOCAL_DIGEST_FIELDS,
    *_CHECKPOINT_RETENTION_DIGEST_FIELDS,
)


def _legacy_resume_run_local_payloads(payload: dict[str, object]) -> list[dict[str, object]]:
    init_checkpoint = payload.get("init_checkpoint")
    if not isinstance(init_checkpoint, str):
        return []
    checkpoint_suffix = "/logs/checkpoints/best.pt"
    if not init_checkpoint.endswith(checkpoint_suffix):
        return []
    run_dir = init_checkpoint[: -len(checkpoint_suffix)] + "/stages/T1_ppo_control/runs/latest_run"
    variant = dict(payload)
    variant["output_dir"] = run_dir
    variant["tensorboard_dir"] = run_dir + "/tensorboard"
    variant["steps"] = None
    return [variant]


def _legacy_rollout_backend_payloads(payload: dict[str, object]) -> list[dict[str, object]]:
    if payload.get("rollout_inference") != "mahjax-gpu":
        return []
    variants: list[dict[str, object]] = []
    for pipeline_depth in (0, payload.get("ppo_pipeline_depth")):
        variant = dict(payload)
        variant["rollout_inference"] = "torch-callback"
        variant["ppo_pipeline_depth"] = pipeline_depth
        variant["rollout_device"] = None
        variants.append(variant)
    return variants


def _with_retention_variants(payload: dict[str, object]) -> list[dict[str, object]]:
    variants = [payload]
    for retention_field in _CHECKPOINT_RETENTION_DIGEST_FIELDS:
        if retention_field not in payload:
            continue
        omitted = dict(payload)
        del omitted[retention_field]
        variants.append(omitted)
        disabled = dict(payload)
        disabled[retention_field] = False
        variants.append(disabled)
    return variants


def _add_resume_config_digest_variants(
    digests: set[str],
    payload: dict[str, object],
    *,
    omit_lr_decay_samples: bool,
    omit_legacy_rollout_fields: bool,
    omit_run_local_fields: bool,
) -> None:
    variant = dict(payload)
    if omit_lr_decay_samples:
        variant["lr_decay_samples"] = None
    if omit_legacy_rollout_fields:
        del variant["ppo_pipeline_depth"]
        del variant["rollout_device"]
    if omit_run_local_fields:
        for field in _RUN_LOCAL_DIGEST_FIELDS:
            del variant[field]
    for retention_variant in _with_retention_variants(variant):
        digests.add(_payload_digest(retention_variant))


def _compatible_resume_config_digests(config: PpoControlConfig) -> set[str]:
    payload = _json_config(config)
    payload["resume"] = None
    digests: set[str] = set()
    payloads = [payload]
    payloads.extend(_legacy_resume_run_local_payloads(payload))
    for legacy_payload in _legacy_rollout_backend_payloads(payload):
        payloads.append(legacy_payload)
        payloads.extend(_legacy_resume_run_local_payloads(legacy_payload))
    can_omit_lr_decay_samples = config.lr_decay_samples is not None
    for resume_payload in payloads:
        can_omit_legacy_rollout_fields = (
            resume_payload.get("ppo_pipeline_depth") == 0 and resume_payload.get("rollout_device") is None
        )
        for retention_variant in _with_retention_variants(resume_payload):
            digests.add(_payload_digest(retention_variant))
        for omit_run_local_fields in (False, True):
            if omit_run_local_fields:
                _add_resume_config_digest_variants(
                    digests,
                    resume_payload,
                    omit_lr_decay_samples=False,
                    omit_legacy_rollout_fields=False,
                    omit_run_local_fields=True,
                )
            if can_omit_lr_decay_samples:
                _add_resume_config_digest_variants(
                    digests,
                    resume_payload,
                    omit_lr_decay_samples=True,
                    omit_legacy_rollout_fields=False,
                    omit_run_local_fields=omit_run_local_fields,
                )
            if can_omit_legacy_rollout_fields:
                _add_resume_config_digest_variants(
                    digests,
                    resume_payload,
                    omit_lr_decay_samples=False,
                    omit_legacy_rollout_fields=True,
                    omit_run_local_fields=omit_run_local_fields,
                )
            if can_omit_lr_decay_samples and can_omit_legacy_rollout_fields:
                _add_resume_config_digest_variants(
                    digests,
                    resume_payload,
                    omit_lr_decay_samples=True,
                    omit_legacy_rollout_fields=True,
                    omit_run_local_fields=omit_run_local_fields,
                )
    return digests
