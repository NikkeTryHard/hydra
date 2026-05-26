from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from hydra_learner import export_inference as export_inference_module
from hydra_learner.checkpoint import EmaConfig, ModelConfig, OptimizerConfig, RuntimeConfig, save_checkpoint
from hydra_learner.drda import DrdaResidualConfig, drda_training_objective_metadata
from hydra_learner.export_inference import export_inference, parse_args, validate_args, write_exported_policy
from hydra_learner.losses import LossWeights
from hydra_learner.model import (
    ACTION_SPACE,
    BACKBONE_PROFILE_TILEFORMER_BIAS,
    OBS_CHANNELS,
    TILE_WIDTH,
    HydraPolicyNet,
)


def test_export_parse_args_accepts_required_flags() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "latest.pt",
            "--weight-source",
            "ema",
            "--output-dir",
            "artifact",
            "--num-fixture-rows",
            "3",
        ]
    )

    config = validate_args(args)

    assert config.checkpoint == Path("latest.pt")
    assert config.weight_source == "ema"
    assert config.output_dir == Path("artifact")
    assert config.num_fixture_rows == 3


def test_export_inference_writes_weights_metadata_and_fixture(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path)
    output_dir = tmp_path / "export"

    result = export_inference(
        validate_args(parse_args(["--checkpoint", str(checkpoint), "--output-dir", str(output_dir)]))
    )

    assert result.artifact_path.name == "policy.onnx"
    assert result.artifact_path.exists()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 2
    assert metadata["format"] == "onnx"
    assert metadata["artifact"] == "policy.onnx"
    assert metadata["input_name"] == "obs"
    assert metadata["output_name"] == "policy_logits"
    assert metadata["encoder_shape"] == [OBS_CHANNELS, TILE_WIDTH]
    assert metadata["action_space"] == ACTION_SPACE
    assert metadata["profiles"]["backbone_profile"] == "conv2d_local3"
    assert metadata["profiles"]["residual_profile"] == "mish_se"
    assert metadata["checkpoint_global_step"] == 7
    assert metadata["checkpoint_samples_seen"] == 11
    fixture = load_file(result.fixture_path)
    assert fixture["obs"].shape == (8, OBS_CHANNELS, TILE_WIDTH)
    assert fixture["policy_logits"].shape == (8, ACTION_SPACE)


def test_export_inference_can_choose_ema_weights(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path, with_ema=True)

    result = export_inference(
        validate_args(
            parse_args(
                ["--checkpoint", str(checkpoint), "--weight-source", "ema", "--output-dir", str(tmp_path / "ema")]
            )
        )
    )

    assert result.artifact_path.exists()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["weight_source"] == "ema"


def test_load_export_policy_validates_checkpoint_before_writing(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path)
    config = validate_args(parse_args(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path / "export")]))

    policy, obs, init, model_config, raw_checkpoint = export_inference_module.load_export_policy(config)

    assert tuple(obs.shape) == (8, OBS_CHANNELS, TILE_WIDTH)
    assert init.global_step == 7
    assert model_config.hidden == 8
    assert isinstance(raw_checkpoint["model_config"], dict)
    with torch.inference_mode():
        assert policy(obs).shape == (8, ACTION_SPACE)


def test_write_exported_policy_rejects_drda_checkpoint_before_creating_artifacts(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path)
    config = validate_args(parse_args(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path / "direct")]))
    policy, obs, init, model_config, raw_checkpoint = export_inference_module.load_export_policy(config)
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base checkpoint")
    raw_checkpoint["training_objective"] = drda_training_objective_metadata(
        config=DrdaResidualConfig(),
        base_checkpoint_path=base_checkpoint,
        base_model_config=ModelConfig(hidden=8, blocks=1, bottleneck=4),
    )

    with pytest.raises(ValueError, match="DRDA residual adapter checkpoints cannot be exported"):
        write_exported_policy(
            config,
            policy=policy,
            obs=obs,
            init=init,
            model_config=model_config,
            checkpoint=raw_checkpoint,
        )

    assert not config.output_dir.exists() or not any(config.output_dir.iterdir())


def test_export_rejects_unsupported_profile(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(
        tmp_path,
        model=HydraPolicyNet(hidden=24, blocks=1, bottleneck=4, backbone_profile=BACKBONE_PROFILE_TILEFORMER_BIAS),
        model_config=ModelConfig(hidden=24, blocks=1, bottleneck=4, backbone_profile=BACKBONE_PROFILE_TILEFORMER_BIAS),
    )

    with pytest.raises(ValueError, match="unsupported backbone_profile"):
        export_inference(
            validate_args(parse_args(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path / "bad")]))
        )


def test_export_rejects_drda_residual_checkpoint(tmp_path: Path) -> None:
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base checkpoint")
    checkpoint = _write_checkpoint(
        tmp_path,
        training_objective=drda_training_objective_metadata(
            config=DrdaResidualConfig(),
            base_checkpoint_path=base_checkpoint,
            base_model_config=ModelConfig(hidden=8, blocks=1, bottleneck=4),
        ),
    )

    with pytest.raises(ValueError, match="DRDA residual adapter checkpoints cannot be exported"):
        export_inference(
            validate_args(parse_args(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path / "drda")]))
        )


def _write_checkpoint(
    tmp_path: Path,
    *,
    model: HydraPolicyNet | None = None,
    model_config: ModelConfig | None = None,
    with_ema: bool = False,
    training_objective: dict[str, object] | None = None,
) -> Path:
    actual_model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4) if model is None else model
    optimizer = torch.optim.AdamW(actual_model.parameters(), lr=1.0e-3)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    ema_state = None
    ema_config = None
    if with_ema:
        ema_config = EmaConfig(enabled=True)
        ema_state = {
            key: torch.full_like(value, 3.0)
            for key, value in actual_model.state_dict().items()
            if value.is_floating_point()
        }
    path = tmp_path / "latest.pt"
    save_checkpoint(
        path,
        model=actual_model,
        optimizer=optimizer,
        model_config=model_config or ModelConfig(hidden=8, blocks=1, bottleneck=4),
        optimizer_config=OptimizerConfig(name="AdamW", lr=1.0e-3, min_lr=1.0e-6),
        runtime_config=RuntimeConfig(
            variant="eager_bf16", loss_mode="full_base", precision_mode="bf16_autocast", compile_fullgraph_check=False
        ),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=7,
        samples_seen=11,
        ema_config=ema_config,
        ema_state=ema_state,
        training_objective=training_objective,
    )
    return path
