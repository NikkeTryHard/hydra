"""Checkpoint schema and JSON-safe contract validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra_learner.model import OBS_CHANNELS, TILE_WIDTH

if TYPE_CHECKING:
    from hydra_learner.checkpoint import OptimizerConfig
    from hydra_learner.losses import LossWeights

TARGET_CONTRACT_SEMANTICS = {
    "exit": "exit_root_child_visits_v1",
    "delta_q": "delta_q_child_minus_root_v1",
}
TARGET_CONTRACT_PROVENANCE = "search-derived"

CHECKPOINT_SCHEMA_VERSION = 1
ENCODER_SHAPE = (OBS_CHANNELS, TILE_WIDTH)
HEAD_MODE = "base_plus_optional_oracle_safety"


def manifest_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def target_contract_from_manifest(manifest_path: Path | None, weights: LossWeights) -> dict[str, object] | None:
    lanes: list[tuple[str, str]] = []
    if weights.exit > 0.0:
        lanes.append(("exit", "exit_sidecar"))
    if weights.deltaq > 0.0:
        raise ValueError("delta_q_output_contract_missing")
    if not lanes:
        return None
    if manifest_path is None:
        raise ValueError("target_contract metadata requires compact shard manifest")
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, Mapping):
        raise ValueError("target_contract metadata requires manifest object")
    digest = manifest_digest(manifest_path)
    contract: dict[str, object] = {}
    for lane, manifest_key in lanes:
        sidecar = manifest.get(manifest_key)
        if not isinstance(sidecar, Mapping):
            raise ValueError(f"target_contract.{lane} requires {manifest_key} manifest metadata")
        path = sidecar.get("path")
        source_net_hash = sidecar.get("source_net_hash")
        source_version = sidecar.get("source_version")
        if not isinstance(path, str) or path == "":
            raise ValueError(f"target_contract.{lane}.sidecar_path is required")
        if not isinstance(source_net_hash, int):
            raise ValueError(f"target_contract.{lane}.source_net_hash is required")
        if not isinstance(source_version, int):
            raise ValueError(f"target_contract.{lane}.source_version is required")
        lane_contract: dict[str, object] = {
            "lane": lane,
            "sidecar_path": path,
            "source_net_hash": source_net_hash,
            "source_version": source_version,
            "semantics": TARGET_CONTRACT_SEMANTICS[lane],
            "provenance": TARGET_CONTRACT_PROVENANCE,
            "manifest_path": str(manifest_path),
            "manifest_digest_sha256": digest,
            "coverage_fraction": 1.0,
        }
        _validate_target_lane_contract(lane, lane_contract)
        contract[lane] = lane_contract
    return contract


def _validate_checkpoint_root(checkpoint: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "torch_version",
        "cuda_version",
        "device",
        "model_config",
        "loss_weights",
        "optimizer_config",
        "optimizer_state",
        "model_state",
        "rng_state",
        "manifest",
        "global_step",
        "samples_seen",
        "compile",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint missing keys: {sorted(missing)}")
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"checkpoint schema_version mismatch: {checkpoint['schema_version']!r}")
    _normalize_loss_weights(checkpoint["loss_weights"])


def _checkpoint_optimizer_config_for_resume(actual: object, expected: OptimizerConfig) -> object:
    if not isinstance(actual, dict):
        return actual
    normalized = dict(actual)
    expected_dict = asdict(expected)
    normalized.setdefault("target_games", None)
    for key in ("foreach", "fused"):
        if normalized.get(key) is None and expected_dict.get(key) is not None:
            normalized[key] = expected_dict[key]
    for key in ("lr", "min_lr", "grad_clip_norm", "weight_decay", "beta1", "beta2", "eps"):
        expected_value = expected_dict.get(key)
        if (
            key in normalized
            and isinstance(normalized[key], float)
            and isinstance(expected_value, float)
            and abs(normalized[key] - expected_value) <= max(1.0e-12, abs(expected_value) * 1.0e-6)
        ):
            normalized[key] = expected_value
    if normalized.get("lr_schedule") == "constant" and expected_dict.get("lr_schedule") == "cosine":
        normalized["lr_schedule"] = expected_dict["lr_schedule"]
    if normalized.get("lr_schedule") == expected_dict.get("lr_schedule") == "cosine":
        normalized["target_games"] = expected_dict["target_games"]
    return normalized


def _normalize_loss_weights(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint loss_weights must be an object")
    normalized = dict(value)
    normalized.setdefault("exit", 0.0)
    normalized.setdefault("deltaq", 0.0)
    return normalized


def _normalize_expected_loss_weights(weights: LossWeights) -> dict[str, object]:
    expected = asdict(weights)
    expected.setdefault("exit", 0.0)
    expected.setdefault("deltaq", 0.0)
    return expected


def _normalize_target_contract_for_weights(
    value: Mapping[str, object] | None, weights: LossWeights
) -> dict[str, object] | None:
    lanes: list[str] = []
    if weights.exit > 0.0:
        lanes.append("exit")
    if weights.deltaq > 0.0:
        raise ValueError("delta_q_output_contract_missing")
    if not lanes:
        if value is not None:
            _validate_json_payload(value, "target_contract")
        return None if value is None else dict(value)
    if value is None:
        raise ValueError("target_contract metadata is required when target loss weights are positive")
    _validate_json_payload(value, "target_contract")
    contract = dict(value)
    for lane in lanes:
        lane_value = contract.get(lane)
        if not isinstance(lane_value, Mapping):
            raise ValueError(f"target_contract.{lane} metadata is required")
        _validate_target_lane_contract(lane, lane_value)
    return contract


def _validate_target_lane_contract(lane: str, value: Mapping[str, object]) -> None:
    required = {
        "lane",
        "sidecar_path",
        "source_net_hash",
        "source_version",
        "semantics",
        "provenance",
        "manifest_path",
        "manifest_digest_sha256",
        "coverage_fraction",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(f"target_contract.{lane} missing keys: {sorted(missing)}")
    if value["lane"] != lane:
        raise ValueError(f"target_contract.{lane}.lane mismatch")
    if value["semantics"] != TARGET_CONTRACT_SEMANTICS[lane]:
        raise ValueError(f"target_contract.{lane}.semantics mismatch")
    if value["provenance"] != TARGET_CONTRACT_PROVENANCE:
        raise ValueError(f"target_contract.{lane}.provenance mismatch")
    coverage = value["coverage_fraction"]
    if (
        not isinstance(coverage, int | float)
        or not math.isfinite(float(coverage))
        or not (0.0 < float(coverage) <= 1.0)
    ):
        raise ValueError(f"target_contract.{lane}.coverage_fraction must be in (0, 1]")
    _validate_sidecar_tuple(value, lane)


def _validate_sidecar_tuple(value: Mapping[str, object], lane: str) -> None:
    tuple_keys = ("sidecar_path", "source_net_hash", "source_version")
    present = [value.get(key) is not None for key in tuple_keys]
    if any(present) and not all(present):
        raise ValueError(f"target_contract.{lane} sidecar tuple must have path/hash/version all present or all absent")


def _validate_json_payload(value: object, path: str) -> None:
    if isinstance(value, bool | str) or value is None:
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            _validate_json_payload(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _validate_json_payload(item, f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported {type(value).__name__}")


def _expect_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise ValueError(f"checkpoint {name} mismatch: got {actual!r} expected {expected!r}")
