from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from hydra_learner.checkpoint import (
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    load_checkpoint,
    manifest_digest,
    restore_rng_state,
    save_checkpoint,
)
from hydra_learner.losses import LossWeights
from hydra_learner.model import RESIDUAL_PROFILE_MISH_NO_SE, RESIDUAL_PROFILE_RELU_SE, HydraPolicyNet


def test_save_load_restores_model_params_exactly(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    ckpt = tmp_path / "ckpt.pt"
    manifest = _write_manifest(tmp_path, b"manifest-a")
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=7,
        samples_seen=14,
    )
    loaded_model, loaded_optimizer = _model_optimizer()
    state = load_checkpoint(
        ckpt,
        model=loaded_model,
        optimizer=loaded_optimizer,
        expected_model_config=_model_config(),
        expected_optimizer_config=_optimizer_config(),
        expected_runtime_config=_runtime_config(),
        expected_loss_weights=LossWeights(),
        expected_manifest_path=manifest,
    )
    assert state.global_step == 7
    assert state.samples_seen == 14
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded_model.state_dict()[key], value, rtol=0.0, atol=0.0)


def test_optimizer_resume_matches_uninterrupted_two_steps(tmp_path: Path) -> None:
    torch.manual_seed(123)
    x1 = torch.randn(2, 192, 34)
    x2 = torch.randn(2, 192, 34)
    manifest = _write_manifest(tmp_path, b"manifest-a")

    uninterrupted, uninterrupted_opt = _model_optimizer()
    resumed, resumed_opt = _model_optimizer()
    resumed.load_state_dict(uninterrupted.state_dict())

    _tiny_step(uninterrupted, uninterrupted_opt, x1)
    _tiny_step(resumed, resumed_opt, x1)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        model=resumed,
        optimizer=resumed_opt,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=1,
        samples_seen=2,
    )
    reloaded, reloaded_opt = _model_optimizer()
    load_checkpoint(
        ckpt,
        model=reloaded,
        optimizer=reloaded_opt,
        expected_model_config=_model_config(),
        expected_optimizer_config=_optimizer_config(),
        expected_runtime_config=_runtime_config(),
        expected_loss_weights=LossWeights(),
        expected_manifest_path=manifest,
    )
    _tiny_step(uninterrupted, uninterrupted_opt, x2)
    _tiny_step(reloaded, reloaded_opt, x2)
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value, rtol=1.0e-6, atol=1.0e-7)


def test_rng_restore_reproduces_next_random_values(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    manifest = _write_manifest(tmp_path, b"manifest-a")
    random.seed(77)
    np.random.seed(78)
    torch.manual_seed(79)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=0,
        samples_seen=0,
    )
    expected = (random.random(), np.random.rand(3), torch.randn(3))
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
    restore_rng_state(checkpoint["rng_state"])
    actual = (random.random(), np.random.rand(3), torch.randn(3))
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)


def test_manifest_digest_mismatch_hard_errors(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    manifest = _write_manifest(tmp_path, b"manifest-a")
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=0,
        samples_seen=0,
    )
    manifest.write_bytes(b"manifest-b")
    with pytest.raises(ValueError, match="manifest"):
        load_checkpoint(
            ckpt,
            model=model,
            optimizer=optimizer,
            expected_model_config=_model_config(),
            expected_optimizer_config=_optimizer_config(),
            expected_runtime_config=_runtime_config(),
            expected_loss_weights=LossWeights(),
            expected_manifest_path=manifest,
        )


def test_model_config_mismatch_hard_errors(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    manifest = _write_manifest(tmp_path, b"manifest-a")
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=0,
        samples_seen=0,
    )
    with pytest.raises(ValueError, match="model_config"):
        load_checkpoint(
            ckpt,
            model=model,
            optimizer=optimizer,
            expected_model_config=ModelConfig(hidden=9, blocks=1, bottleneck=4),
            expected_optimizer_config=_optimizer_config(),
            expected_runtime_config=_runtime_config(),
            expected_loss_weights=LossWeights(),
            expected_manifest_path=manifest,
        )


def test_stale_conv1d_checkpoint_weights_hard_error(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    manifest = _write_manifest(tmp_path, b"manifest-a")
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=0,
        samples_seen=0,
    )
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
    checkpoint["model_state"]["backbone.input.weight"] = checkpoint["model_state"]["backbone.input.weight"].squeeze(2)
    torch.save(checkpoint, ckpt)

    with pytest.raises(ValueError, match=r"model_state\[backbone\.input\.weight\] shape mismatch"):
        load_checkpoint(
            ckpt,
            model=model,
            optimizer=optimizer,
            expected_model_config=_model_config(),
            expected_optimizer_config=_optimizer_config(),
            expected_runtime_config=_runtime_config(),
            expected_loss_weights=LossWeights(),
            expected_manifest_path=manifest,
        )


@pytest.mark.parametrize("profile", [RESIDUAL_PROFILE_MISH_NO_SE, RESIDUAL_PROFILE_RELU_SE])
def test_residual_profile_mismatch_hard_errors(tmp_path: Path, profile: str) -> None:
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4, residual_profile=profile)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    manifest = _write_manifest(tmp_path, b"manifest-a")
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=ModelConfig(hidden=8, blocks=1, bottleneck=4, residual_profile=profile),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=0,
        samples_seen=0,
    )
    default_model, default_optimizer = _model_optimizer()

    with pytest.raises(ValueError, match="model_config"):
        load_checkpoint(
            ckpt,
            model=default_model,
            optimizer=default_optimizer,
            expected_model_config=_model_config(),
            expected_optimizer_config=_optimizer_config(),
            expected_runtime_config=_runtime_config(),
            expected_loss_weights=LossWeights(),
            expected_manifest_path=manifest,
        )


def test_checkpoint_contains_state_dict_not_compiled_object(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    manifest = _write_manifest(tmp_path, b"manifest-a")
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=RuntimeConfig(
            variant="compile_default",
            loss_mode="full_base",
            precision_mode="bf16_autocast",
            compile_fullgraph_check=True,
        ),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=0,
        samples_seen=0,
    )
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert isinstance(checkpoint["model_state"], dict)
    assert "compile" in checkpoint
    assert "compiled" not in checkpoint
    assert all(isinstance(value, torch.Tensor) for value in checkpoint["model_state"].values())


def _model_optimizer() -> tuple[HydraPolicyNet, torch.optim.Optimizer]:
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    return model, optimizer


def _tiny_step(model: HydraPolicyNet, optimizer: torch.optim.Optimizer, obs: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = model(obs).policy_logits.square().mean()
    loss.backward()
    optimizer.step()


def _model_config() -> ModelConfig:
    return ModelConfig(hidden=8, blocks=1, bottleneck=4)


def _optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(name="AdamW", lr=1.0e-3)


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        variant="eager_bf16", loss_mode="full_base", precision_mode="bf16_autocast", compile_fullgraph_check=False
    )


def _write_manifest(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "manifest.json"
    path.write_bytes(content)
    assert manifest_digest(path) is not None
    return path
