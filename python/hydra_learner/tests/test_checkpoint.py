from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from hydra_learner.checkpoint import (
    EmaConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    load_checkpoint,
    load_checkpoint_init_only,
    manifest_digest,
    restore_rng_state,
    save_checkpoint,
    target_contract_from_manifest,
)
from hydra_learner.losses import LossWeights
from hydra_learner.model import (
    BACKBONE_PROFILE_CONVNEXT_TILE_K7,
    BACKBONE_PROFILE_GLOBAL_POOL_BIAS,
    BACKBONE_PROFILE_TILEFORMER_BIAS,
    RESIDUAL_PROFILE_MISH_ECA,
    RESIDUAL_PROFILE_MISH_NO_SE,
    RESIDUAL_PROFILE_RELU_SE,
    HydraPolicyNet,
)


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


def test_load_checkpoint_accepts_legacy_optimizer_config_without_target_games(tmp_path: Path) -> None:
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
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    optimizer_config = payload["optimizer_config"]
    assert isinstance(optimizer_config, dict)
    optimizer_config.pop("target_games")
    torch.save(payload, ckpt)

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


def test_load_checkpoint_returns_raw_mjai_progress(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    ckpt = tmp_path / "ckpt.pt"
    manifest = _write_manifest(tmp_path, b"manifest-a")
    progress = {"loaded_games": 12, "skipped_games": 3, "samples": 128, "batches": 32}
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
        raw_mjai_progress=progress,
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
    assert state.raw_mjai_progress == progress


def test_save_load_restores_ema_state_exactly(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    ckpt = tmp_path / "ckpt.pt"
    manifest = _write_manifest(tmp_path, b"manifest-a")
    ema_config = EmaConfig(enabled=True, decay=0.9, start_step=1, update_every_steps=2, device="cpu")
    ema_state = {
        key: tensor.detach().to(dtype=torch.float32).add(1.0)
        for key, tensor in model.state_dict().items()
        if tensor.is_floating_point()
    }
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
        ema_config=ema_config,
        ema_state=ema_state,
        ema_update_count=3,
        ema_last_update_step=5,
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
        expected_ema_config=ema_config,
    )
    assert state.ema is not None
    assert state.ema.update_count == 3
    assert state.ema.last_update_step == 5
    for key, value in ema_state.items():
        torch.testing.assert_close(state.ema.state_dict[key], value, rtol=0.0, atol=0.0)


def test_ema_config_mismatch_hard_errors(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    ckpt = tmp_path / "ckpt.pt"
    manifest = _write_manifest(tmp_path, b"manifest-a")
    ema_config = EmaConfig(enabled=True, decay=0.9, start_step=0, update_every_steps=1, device="auto")
    ema_state = {
        key: tensor.detach().to(dtype=torch.float32).clone()
        for key, tensor in model.state_dict().items()
        if tensor.is_floating_point()
    }
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=1,
        samples_seen=2,
        ema_config=ema_config,
        ema_state=ema_state,
    )
    with pytest.raises(ValueError, match="ema_config"):
        load_checkpoint(
            ckpt,
            model=model,
            optimizer=optimizer,
            expected_model_config=_model_config(),
            expected_optimizer_config=_optimizer_config(),
            expected_runtime_config=_runtime_config(),
            expected_loss_weights=LossWeights(),
            expected_manifest_path=manifest,
            expected_ema_config=EmaConfig(enabled=True, decay=0.9, start_step=0, update_every_steps=1, device="cpu"),
        )


def test_checkpoint_serializes_ema_state_on_cpu(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    ckpt = tmp_path / "ckpt.pt"
    manifest = _write_manifest(tmp_path, b"manifest-a")
    ema_config = EmaConfig(enabled=True, device="auto")
    ema_state = {
        key: tensor.detach().to(dtype=torch.float32).clone()
        for key, tensor in model.state_dict().items()
        if tensor.is_floating_point()
    }
    if torch.cuda.is_available():
        ema_state = {key: tensor.to(device="cuda") for key, tensor in ema_state.items()}
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=1,
        samples_seen=2,
        ema_config=ema_config,
        ema_state=ema_state,
    )
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["ema_state"].values())
    assert all(tensor.dtype == torch.float32 for tensor in checkpoint["ema_state"].values())


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


def test_optimizer_none_foreach_fused_resume_matches_explicit_expected(tmp_path: Path) -> None:
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
    expected = OptimizerConfig(name="AdamW", lr=1.0e-3, min_lr=1.0e-6, foreach=False, fused=True)
    state = load_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        expected_model_config=_model_config(),
        expected_optimizer_config=expected,
        expected_runtime_config=_runtime_config(),
        expected_loss_weights=LossWeights(),
        expected_manifest_path=manifest,
    )

    assert state.global_step == 7


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


def test_tileformer_bias_checkpoint_config_roundtrip(tmp_path: Path) -> None:
    model = HydraPolicyNet(hidden=24, blocks=1, bottleneck=4, backbone_profile=BACKBONE_PROFILE_TILEFORMER_BIAS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    manifest = _write_manifest(tmp_path, b"manifest-a")
    model_config = ModelConfig(hidden=24, blocks=1, bottleneck=4, backbone_profile=BACKBONE_PROFILE_TILEFORMER_BIAS)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=2,
        samples_seen=4,
    )
    loaded = HydraPolicyNet(hidden=24, blocks=1, bottleneck=4, backbone_profile=BACKBONE_PROFILE_TILEFORMER_BIAS)
    loaded_optimizer = torch.optim.AdamW(loaded.parameters(), lr=1.0e-3)
    state = load_checkpoint(
        ckpt,
        model=loaded,
        optimizer=loaded_optimizer,
        expected_model_config=model_config,
        expected_optimizer_config=_optimizer_config(),
        expected_runtime_config=_runtime_config(),
        expected_loss_weights=LossWeights(),
        expected_manifest_path=manifest,
    )
    assert state.global_step == 2
    assert state.samples_seen == 4
    assert loaded.backbone_profile == BACKBONE_PROFILE_TILEFORMER_BIAS


@pytest.mark.parametrize("backbone_profile", [BACKBONE_PROFILE_CONVNEXT_TILE_K7, BACKBONE_PROFILE_GLOBAL_POOL_BIAS])
def test_resnet_family_backbone_checkpoint_config_roundtrip(tmp_path: Path, backbone_profile: str) -> None:
    model = HydraPolicyNet(hidden=16, blocks=1, bottleneck=4, backbone_profile=backbone_profile)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    manifest = _write_manifest(tmp_path, b"manifest-a")
    model_config = ModelConfig(hidden=16, blocks=1, bottleneck=4, backbone_profile=backbone_profile)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=2,
        samples_seen=4,
    )
    loaded = HydraPolicyNet(hidden=16, blocks=1, bottleneck=4, backbone_profile=backbone_profile)
    loaded_optimizer = torch.optim.AdamW(loaded.parameters(), lr=1.0e-3)
    state = load_checkpoint(
        ckpt,
        model=loaded,
        optimizer=loaded_optimizer,
        expected_model_config=model_config,
        expected_optimizer_config=_optimizer_config(),
        expected_runtime_config=_runtime_config(),
        expected_loss_weights=LossWeights(),
        expected_manifest_path=manifest,
    )
    assert state.global_step == 2
    assert state.samples_seen == 4
    assert loaded.backbone_profile == backbone_profile


def test_mish_eca_checkpoint_config_roundtrip(tmp_path: Path) -> None:
    model = HydraPolicyNet(hidden=16, blocks=1, bottleneck=4, residual_profile=RESIDUAL_PROFILE_MISH_ECA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    manifest = _write_manifest(tmp_path, b"manifest-a")
    model_config = ModelConfig(hidden=16, blocks=1, bottleneck=4, residual_profile=RESIDUAL_PROFILE_MISH_ECA)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=3,
        samples_seen=6,
    )
    loaded = HydraPolicyNet(hidden=16, blocks=1, bottleneck=4, residual_profile=RESIDUAL_PROFILE_MISH_ECA)
    loaded_optimizer = torch.optim.AdamW(loaded.parameters(), lr=1.0e-3)
    state = load_checkpoint(
        ckpt,
        model=loaded,
        optimizer=loaded_optimizer,
        expected_model_config=model_config,
        expected_optimizer_config=_optimizer_config(),
        expected_runtime_config=_runtime_config(),
        expected_loss_weights=LossWeights(),
        expected_manifest_path=manifest,
    )
    assert state.global_step == 3
    assert state.samples_seen == 6
    assert loaded.residual_profile == RESIDUAL_PROFILE_MISH_ECA


def test_runtime_accounting_semantics_mismatch_hard_errors(tmp_path: Path) -> None:
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
    incompatible = RuntimeConfig(
        variant="eager_bf16",
        loss_mode="full_base",
        precision_mode="bf16_autocast",
        compile_fullgraph_check=False,
        compile_dry_run_mode="counted_training_step",
        warmup_mode="counted_training_steps",
    )

    with pytest.raises(ValueError, match="compile"):
        load_checkpoint(
            ckpt,
            model=model,
            optimizer=optimizer,
            expected_model_config=_model_config(),
            expected_optimizer_config=_optimizer_config(),
            expected_runtime_config=incompatible,
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


def test_init_only_loader_loads_model_without_optimizer_or_rng(tmp_path: Path) -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    model, optimizer = _model_optimizer()
    _tiny_step(model, optimizer, torch.randn(2, 192, 34))
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
    before_optimizer = loaded_optimizer.state_dict()
    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)
    expected_random = (random.random(), np.random.rand(2), torch.randn(2))
    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)

    state = load_checkpoint_init_only(
        ckpt,
        model=loaded_model,
        expected_model_config=_model_config(),
    )
    actual_random = (random.random(), np.random.rand(2), torch.randn(2))

    assert state.global_step == 7
    assert state.samples_seen == 14
    assert state.weight_source == "raw"
    assert loaded_optimizer.state_dict() == before_optimizer
    assert actual_random[0] == expected_random[0]
    np.testing.assert_array_equal(actual_random[1], expected_random[1])
    torch.testing.assert_close(actual_random[2], expected_random[2], rtol=0.0, atol=0.0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded_model.state_dict()[key], value, rtol=0.0, atol=0.0)


def test_init_only_loader_can_choose_ema_weights(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    ckpt = tmp_path / "ckpt.pt"
    manifest = _write_manifest(tmp_path, b"manifest-a")
    ema_config = EmaConfig(enabled=True, decay=0.9, start_step=0, update_every_steps=1, device="cpu")
    ema_state = {
        key: tensor.detach().to(dtype=torch.float32).add(1.0)
        for key, tensor in model.state_dict().items()
        if tensor.is_floating_point()
    }
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=1,
        samples_seen=2,
        ema_config=ema_config,
        ema_state=ema_state,
    )
    loaded_model, _ = _model_optimizer()

    state = load_checkpoint_init_only(
        ckpt,
        model=loaded_model,
        expected_model_config=_model_config(),
        weight_source="ema",
    )

    assert state.weight_source == "ema"
    for key, value in loaded_model.state_dict().items():
        if value.is_floating_point():
            torch.testing.assert_close(value, ema_state[key].to(dtype=value.dtype), rtol=0.0, atol=0.0)
        else:
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0.0, atol=0.0)


def test_init_only_loader_rejects_model_config_mismatch(tmp_path: Path) -> None:
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
        global_step=0,
        samples_seen=0,
    )

    with pytest.raises(ValueError, match="model_config"):
        load_checkpoint_init_only(
            ckpt,
            model=model,
            expected_model_config=ModelConfig(hidden=9, blocks=1, bottleneck=4),
        )


def _target_contract(manifest_path: Path, *, lane: str = "exit") -> dict[str, object]:
    semantics = "exit_root_child_visits_v1" if lane == "exit" else "delta_q_child_minus_root_v1"
    return {
        lane: {
            "lane": lane,
            "sidecar_path": "/labels/sidecar.jsonl",
            "source_net_hash": 123,
            "source_version": 7,
            "semantics": semantics,
            "provenance": "search-derived",
            "manifest_path": str(manifest_path),
            "manifest_digest_sha256": manifest_digest(manifest_path),
            "coverage_fraction": 1.0,
        }
    }


def test_checkpoint_normalizes_legacy_loss_weights_missing_phase5(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    ckpt = tmp_path / "ckpt.pt"
    manifest = _write_manifest(tmp_path, b"manifest")
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=1,
        samples_seen=2,
    )
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    del checkpoint["loss_weights"]["exit"]
    del checkpoint["loss_weights"]["deltaq"]
    torch.save(checkpoint, ckpt)

    state = load_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        expected_model_config=_model_config(),
        expected_optimizer_config=_optimizer_config(),
        expected_runtime_config=_runtime_config(),
        expected_loss_weights=LossWeights(),
        expected_manifest_path=manifest,
    )
    assert state.global_step == 1


def test_checkpoint_target_contract_required_and_resume_checked(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    ckpt = tmp_path / "ckpt.pt"
    manifest = _write_manifest(tmp_path, b"manifest")
    with pytest.raises(ValueError, match="target_contract metadata"):
        save_checkpoint(
            ckpt,
            model=model,
            optimizer=optimizer,
            model_config=_model_config(),
            optimizer_config=_optimizer_config(),
            runtime_config=_runtime_config(),
            loss_weights=LossWeights(exit=0.1),
            manifest_path=manifest,
            global_step=1,
            samples_seen=2,
        )
    contract = _target_contract(manifest)
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(exit=0.1),
        manifest_path=manifest,
        global_step=1,
        samples_seen=2,
        target_contract=contract,
    )
    load_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        expected_model_config=_model_config(),
        expected_optimizer_config=_optimizer_config(),
        expected_runtime_config=_runtime_config(),
        expected_loss_weights=LossWeights(exit=0.1),
        expected_manifest_path=manifest,
        expected_target_contract=contract,
    )
    mismatched = _target_contract(manifest)
    assert isinstance(mismatched["exit"], dict)
    mismatched["exit"]["source_version"] = 8
    with pytest.raises(ValueError, match="target_contract mismatch"):
        load_checkpoint(
            ckpt,
            model=model,
            optimizer=optimizer,
            expected_model_config=_model_config(),
            expected_optimizer_config=_optimizer_config(),
            expected_runtime_config=_runtime_config(),
            expected_loss_weights=LossWeights(exit=0.1),
            expected_manifest_path=manifest,
            expected_target_contract=mismatched,
        )


def test_target_contract_from_manifest_save_resume_and_mismatch(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    manifest = _write_sidecar_manifest(tmp_path)
    weights = LossWeights(exit=0.1)
    contract = target_contract_from_manifest(manifest, weights)
    assert contract is not None
    ckpt = tmp_path / "ckpt.pt"

    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=weights,
        manifest_path=manifest,
        global_step=3,
        samples_seen=6,
        target_contract=contract,
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert payload["target_contract"] == contract

    loaded_model, loaded_optimizer = _model_optimizer()
    state = load_checkpoint(
        ckpt,
        model=loaded_model,
        optimizer=loaded_optimizer,
        expected_model_config=_model_config(),
        expected_optimizer_config=_optimizer_config(),
        expected_runtime_config=_runtime_config(),
        expected_loss_weights=weights,
        expected_manifest_path=manifest,
        expected_target_contract=contract,
    )
    assert state.global_step == 3

    mismatched_manifest = _write_sidecar_manifest(tmp_path, source_version=8)
    mismatched = target_contract_from_manifest(mismatched_manifest, weights)
    with pytest.raises(ValueError, match="target_contract mismatch"):
        load_checkpoint(
            ckpt,
            model=loaded_model,
            optimizer=loaded_optimizer,
            expected_model_config=_model_config(),
            expected_optimizer_config=_optimizer_config(),
            expected_runtime_config=_runtime_config(),
            expected_loss_weights=weights,
            expected_manifest_path=manifest,
            expected_target_contract=mismatched,
        )


def test_target_contract_from_manifest_rejects_missing_provenance(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, b"{}")
    with pytest.raises(ValueError, match="exit_sidecar manifest metadata"):
        target_contract_from_manifest(manifest, LossWeights(exit=0.1))
    with pytest.raises(ValueError, match="delta_q_output_contract_missing"):
        target_contract_from_manifest(manifest, LossWeights(deltaq=0.1))


def test_checkpoint_target_contract_partial_sidecar_tuple_rejects(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    manifest = _write_manifest(tmp_path, b"manifest")
    contract = _target_contract(manifest)
    assert isinstance(contract["exit"], dict)
    contract["exit"]["source_version"] = None
    with pytest.raises(ValueError, match="path/hash/version"):
        save_checkpoint(
            tmp_path / "ckpt.pt",
            model=model,
            optimizer=optimizer,
            model_config=_model_config(),
            optimizer_config=_optimizer_config(),
            runtime_config=_runtime_config(),
            loss_weights=LossWeights(exit=0.1),
            manifest_path=manifest,
            global_step=1,
            samples_seen=2,
            target_contract=contract,
        )


def test_checkpoint_deltaq_positive_fails_closed_even_with_contract(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    manifest = _write_manifest(tmp_path, b"manifest")
    contract = _target_contract(manifest, lane="delta_q")
    weights = LossWeights(deltaq=0.1)
    with pytest.raises(ValueError, match="delta_q_output_contract_missing"):
        save_checkpoint(
            tmp_path / "ckpt.pt",
            model=model,
            optimizer=optimizer,
            model_config=_model_config(),
            optimizer_config=_optimizer_config(),
            runtime_config=_runtime_config(),
            loss_weights=weights,
            manifest_path=manifest,
            global_step=1,
            samples_seen=2,
            target_contract=contract,
        )


def test_checkpoint_deltaq_positive_load_fails_closed_even_with_contract(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    manifest = _write_manifest(tmp_path, b"manifest")
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
        global_step=1,
        samples_seen=2,
    )
    with pytest.raises(ValueError, match="delta_q_output_contract_missing"):
        load_checkpoint(
            ckpt,
            model=model,
            optimizer=optimizer,
            expected_model_config=_model_config(),
            expected_optimizer_config=_optimizer_config(),
            expected_runtime_config=_runtime_config(),
            expected_loss_weights=LossWeights(deltaq=0.1),
            expected_manifest_path=manifest,
            expected_target_contract=_target_contract(manifest, lane="delta_q"),
        )


def test_checkpoint_deltaq_zero_allows_json_safe_metadata(tmp_path: Path) -> None:
    model, optimizer = _model_optimizer()
    manifest = _write_manifest(tmp_path, b"manifest")
    contract = _target_contract(manifest, lane="delta_q")
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
        global_step=1,
        samples_seen=2,
        target_contract=contract,
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert payload["target_contract"] == contract
    state = load_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        expected_model_config=_model_config(),
        expected_optimizer_config=_optimizer_config(),
        expected_runtime_config=_runtime_config(),
        expected_loss_weights=LossWeights(),
        expected_manifest_path=manifest,
    )
    assert state.global_step == 1


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
    return OptimizerConfig(name="AdamW", lr=1.0e-3, min_lr=1.0e-6)


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        variant="eager_bf16", loss_mode="full_base", precision_mode="bf16_autocast", compile_fullgraph_check=False
    )


def _write_manifest(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "manifest.json"
    path.write_bytes(content)
    assert manifest_digest(path) is not None
    return path


def _write_sidecar_manifest(tmp_path: Path, *, source_net_hash: int = 123, source_version: int = 7) -> Path:
    path = tmp_path / "manifest.json"
    data = {
        "manifest_version": 3,
        "exit_sidecar": {
            "path": "/labels/exit.jsonl",
            "source_net_hash": source_net_hash,
            "source_version": source_version,
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    assert manifest_digest(path) is not None
    return path
