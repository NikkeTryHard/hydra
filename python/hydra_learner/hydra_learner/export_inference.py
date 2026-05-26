"""Export Python BC checkpoints to ONNX artifacts for Rust arena/RL inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast, override

import torch
from safetensors.torch import load_file, save_file

from hydra_learner.checkpoint import CHECKPOINT_SCHEMA_VERSION, ModelConfig, _torch_load, load_checkpoint_init_only
from hydra_learner.drda import DRDA_RESIDUAL_MODE, DRDA_RESIDUAL_OBJECTIVE
from hydra_learner.model import (
    ACTION_SPACE,
    BACKBONE_PROFILE_CONV2D_LOCAL3,
    BASE_LINEAR_HEADS,
    CONV_MEMORY_FORMAT_CONTIGUOUS,
    OBS_CHANNELS,
    RESIDUAL_PROFILE_DEFAULT,
    TILE_WIDTH,
    HydraPolicyNet,
)

EXPORT_SCHEMA_VERSION = 2
ARTIFACT_NAME = "policy.onnx"
METADATA_NAME = "policy.json"
FIXTURE_NAME = "parity_fixture.safetensors"
WeightSource = Literal["raw", "ema"]


@dataclass(frozen=True)
class ExportConfig:
    checkpoint: Path
    weight_source: WeightSource
    output_dir: Path
    fixture_obs: Path | None
    num_fixture_rows: int
    max_batch: int
    opset_version: int


@dataclass(frozen=True)
class ExportResult:
    artifact_path: Path
    metadata_path: Path
    fixture_path: Path
    source_checkpoint_sha256: str
    global_step: int
    samples_seen: int
    weight_source: WeightSource


class PolicyOnly(torch.nn.Module):
    def __init__(self, model: HydraPolicyNet) -> None:
        super().__init__()
        self.model = model

    @override
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs).policy_logits


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Python .pt training checkpoint")
    parser.add_argument("--weight-source", choices=("raw", "ema"), default="raw")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture-obs", type=Path, help="optional torch/safetensors tensor with obs [N,192,34]")
    parser.add_argument("--num-fixture-rows", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=4096, help="maximum dynamic batch accepted by ONNX export")
    parser.add_argument("--opset-version", type=int, default=18)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> ExportConfig:
    if args.num_fixture_rows < 1:
        raise ValueError("--num-fixture-rows must be >= 1")
    if args.max_batch < args.num_fixture_rows:
        raise ValueError("--max-batch must be >= --num-fixture-rows")
    if args.opset_version < 18:
        raise ValueError("--opset-version must be >= 18")
    return ExportConfig(
        checkpoint=args.checkpoint,
        weight_source=cast(WeightSource, args.weight_source),
        output_dir=args.output_dir,
        fixture_obs=args.fixture_obs,
        num_fixture_rows=args.num_fixture_rows,
        max_batch=args.max_batch,
        opset_version=args.opset_version,
    )


def load_export_policy(config: ExportConfig) -> tuple[PolicyOnly, torch.Tensor, Any, ModelConfig, dict[str, Any]]:
    checkpoint = _torch_load(config.checkpoint)
    _reject_drda_checkpoint_export(checkpoint)
    raw_model_config = checkpoint.get("model_config")
    if not isinstance(raw_model_config, dict):
        raise ValueError("checkpoint missing model_config")
    model_config = _model_config_from_checkpoint(raw_model_config)
    _validate_supported_model_config(model_config)
    model = HydraPolicyNet(
        hidden=model_config.hidden,
        blocks=model_config.blocks,
        bottleneck=model_config.bottleneck,
        actions=model_config.actions,
        residual_profile=model_config.residual_profile,
        backbone_profile=model_config.backbone_profile,
        conv_memory_format=model_config.conv_memory_format,
    )
    init = load_checkpoint_init_only(
        config.checkpoint,
        model=model,
        expected_model_config=model_config,
        weight_source=config.weight_source,
    )
    model.eval()
    return (
        PolicyOnly(model).eval(),
        _fixture_obs(config.fixture_obs, config.num_fixture_rows),
        init,
        model_config,
        checkpoint,
    )


def write_exported_policy(
    config: ExportConfig,
    *,
    policy: PolicyOnly,
    obs: torch.Tensor,
    init: Any,
    model_config: ModelConfig,
    checkpoint: dict[str, Any],
) -> ExportResult:
    _reject_drda_checkpoint_export(checkpoint)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = config.output_dir / ARTIFACT_NAME
    metadata_path = config.output_dir / METADATA_NAME
    fixture_path = config.output_dir / FIXTURE_NAME
    source_hash = _sha256_file(config.checkpoint)
    with torch.inference_mode():
        expected = policy(obs).detach().cpu().contiguous()
    _export_onnx(policy, obs, artifact_path, config)
    artifact_hash = _sha256_file(artifact_path)
    metadata = _metadata(
        checkpoint=checkpoint,
        checkpoint_path=config.checkpoint,
        checkpoint_sha256=source_hash,
        artifact_sha256=artifact_hash,
        init=init,
        model_config=model_config,
        config=config,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_file(
        {"obs": obs.cpu().contiguous(), "policy_logits": expected},
        fixture_path,
        metadata={"hydra_fixture_schema_version": str(EXPORT_SCHEMA_VERSION)},
    )
    return ExportResult(
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        fixture_path=fixture_path,
        source_checkpoint_sha256=source_hash,
        global_step=init.global_step,
        samples_seen=init.samples_seen,
        weight_source=init.weight_source,
    )


def export_inference(config: ExportConfig) -> ExportResult:
    policy, obs, init, model_config, checkpoint = load_export_policy(config)
    return write_exported_policy(
        config,
        policy=policy,
        obs=obs,
        init=init,
        model_config=model_config,
        checkpoint=checkpoint,
    )


def _reject_drda_checkpoint_export(checkpoint: dict[str, Any]) -> None:
    training_objective = checkpoint.get("training_objective")
    if not isinstance(training_objective, dict):
        return
    objective = training_objective.get("objective")
    mode = training_objective.get("mode")
    if objective == DRDA_RESIDUAL_OBJECTIVE or mode == DRDA_RESIDUAL_MODE:
        raise ValueError(
            "DRDA residual adapter checkpoints cannot be exported to ONNX/native arena yet; "
            "additive DRDA runtime/export support is not implemented"
        )


def _export_onnx(policy: PolicyOnly, obs: torch.Tensor, artifact_path: Path, config: ExportConfig) -> None:
    try:
        _ensure_onnxscript_torch_api_alias()
        batch_dim = torch.export.Dim("batch", min=1, max=config.max_batch)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*LeafSpec.*is deprecated.*",
                category=FutureWarning,
            )
            torch.onnx.export(
                policy,
                (obs,),
                str(artifact_path),
                input_names=["obs"],
                output_names=["policy_logits"],
                dynamo=True,
                dynamic_shapes={"obs": {0: batch_dim}},
                opset_version=config.opset_version,
                external_data=False,
                optimize=True,
                verify=False,
            )
    except ModuleNotFoundError as exc:
        raise ValueError("ONNX export requires onnx and onnxscript Python packages") from exc


def _ensure_onnxscript_torch_api_alias() -> None:
    try:
        import onnxscript._framework_apis.torch_2_9 as torch_2_9  # noqa: PLC0415 -- optional exporter compatibility shim.
    except ModuleNotFoundError as exc:
        raise ValueError("ONNX export requires onnxscript with PyTorch framework APIs") from exc
    sys.modules.setdefault("onnxscript._framework_apis.torch_2_11", torch_2_9)


def _model_config_from_checkpoint(raw: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        hidden=_config_int(raw, "hidden"),
        blocks=_config_int(raw, "blocks"),
        bottleneck=_config_int(raw, "bottleneck"),
        actions=_config_int(raw, "actions", ACTION_SPACE),
        residual_profile=str(raw.get("residual_profile", RESIDUAL_PROFILE_DEFAULT)),
        backbone_profile=str(raw.get("backbone_profile", BACKBONE_PROFILE_CONV2D_LOCAL3)),
        conv_memory_format=str(raw.get("conv_memory_format", CONV_MEMORY_FORMAT_CONTIGUOUS)),
        head_mode=str(raw.get("head_mode", "base_plus_optional_oracle_safety")),
        encoder_shape=tuple(raw.get("encoder_shape", (OBS_CHANNELS, TILE_WIDTH))),
    )


def _config_int(raw: dict[str, Any], key: str, default: int | None = None) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int):
        raise TypeError(f"model_config {key} must be int")
    return value


def _validate_supported_model_config(config: ModelConfig) -> None:
    if config.encoder_shape != (OBS_CHANNELS, TILE_WIDTH):
        raise ValueError(f"unsupported encoder_shape {config.encoder_shape!r}")
    if config.actions != ACTION_SPACE:
        raise ValueError(f"unsupported action count {config.actions}")
    if config.backbone_profile != BACKBONE_PROFILE_CONV2D_LOCAL3:
        raise ValueError(f"unsupported backbone_profile {config.backbone_profile!r}")
    if config.residual_profile != RESIDUAL_PROFILE_DEFAULT:
        raise ValueError(f"unsupported residual_profile {config.residual_profile!r}")
    if config.conv_memory_format != CONV_MEMORY_FORMAT_CONTIGUOUS:
        raise ValueError(f"unsupported conv_memory_format {config.conv_memory_format!r}")
    if config.head_mode != "base_plus_optional_oracle_safety":
        raise ValueError(f"unsupported head_mode {config.head_mode!r}")


def _metadata(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    artifact_sha256: str,
    init: Any,
    model_config: ModelConfig,
    config: ExportConfig,
) -> dict[str, Any]:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "format": "onnx",
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_sha256,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_global_step": init.global_step,
        "checkpoint_samples_seen": init.samples_seen,
        "weight_source": init.weight_source,
        "model_config": asdict(model_config),
        "encoder_shape": [OBS_CHANNELS, TILE_WIDTH],
        "action_space": ACTION_SPACE,
        "dtype": "float32",
        "artifact": ARTIFACT_NAME,
        "artifact_sha256": artifact_sha256,
        "input_name": "obs",
        "output_name": "policy_logits",
        "input_shape": ["N", OBS_CHANNELS, TILE_WIDTH],
        "output_shape": ["N", ACTION_SPACE],
        "max_batch": config.max_batch,
        "opset_version": config.opset_version,
        "base_heads": {"width": BASE_LINEAR_HEADS, "policy_logits": [0, ACTION_SPACE]},
        "profiles": {
            "backbone_profile": model_config.backbone_profile,
            "residual_profile": model_config.residual_profile,
            "conv_memory_format": model_config.conv_memory_format,
        },
        "torch_version": checkpoint.get("torch_version"),
    }


def _fixture_obs(path: Path | None, num_rows: int) -> torch.Tensor:
    if path is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0x4859445241)
        return torch.randn((num_rows, OBS_CHANNELS, TILE_WIDTH), generator=generator, dtype=torch.float32)
    if path.suffix == ".safetensors":
        tensors = load_file(path)
        raw = tensors.get("obs")
        if raw is None:
            raise ValueError("fixture safetensors missing obs tensor")
    else:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(raw, dict):
            raw = raw.get("obs")
    if not isinstance(raw, torch.Tensor):
        raise ValueError("fixture obs must be a tensor or dict containing obs")
    obs = raw.detach().cpu().to(dtype=torch.float32).contiguous()
    if tuple(obs.shape[1:]) != (OBS_CHANNELS, TILE_WIDTH):
        raise ValueError(f"fixture obs shape must be [N,{OBS_CHANNELS},{TILE_WIDTH}], got {tuple(obs.shape)}")
    if obs.shape[0] < num_rows:
        raise ValueError(f"fixture obs has {obs.shape[0]} rows, requested {num_rows}")
    return obs[:num_rows].contiguous()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    result = export_inference(validate_args(parse_args(argv)))
    print(
        json.dumps(
            {key: str(value) if isinstance(value, Path) else value for key, value in asdict(result).items()},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
