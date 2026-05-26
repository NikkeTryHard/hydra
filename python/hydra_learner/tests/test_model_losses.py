from __future__ import annotations

from pathlib import Path
from typing import override

import pytest
import torch
import torch.nn.functional as F

from hydra_learner.batches import targets_for_compiled_loss, targets_from_policy_batch, validate_base_targets
from hydra_learner.losses import (
    BaseTargets,
    LossBreakdown,
    LossWeights,
    active_loss_heads,
    base_loss,
    danger_focal_bce,
    exit_policy_loss,
    masked_policy_ce,
    masked_policy_ce_indices,
    opp_next_ce,
    oracle_critic_loss,
    safety_residual_loss,
    target_coverage_dict,
    value_mse,
)
from hydra_learner.model import (
    ACTION_SPACE,
    BACKBONE_PROFILE_CONV2D_LOCAL3,
    BACKBONE_PROFILE_CONVNEXT_TILE_K7,
    BACKBONE_PROFILE_GLOBAL_POOL_BIAS,
    BACKBONE_PROFILE_TILEFORMER_BIAS,
    BACKBONE_PROFILES,
    GRP_CLASSES,
    OPPONENTS,
    RESIDUAL_PROFILE_DEFAULT,
    RESIDUAL_PROFILE_MISH_ECA,
    RESIDUAL_PROFILE_RELU_NO_NORM_NO_SE,
    RESIDUAL_PROFILES,
    SCORE_BINS,
    TILE_WIDTH,
    HydraBaseOutput,
    HydraPolicyNet,
)
from hydra_learner.optim import LrScheduler, LrSchedulerConfig
from hydra_learner.shard_reader import BcShardReader
from hydra_learner.step import HydraCompiledLossStep, loss_step_args, policy_diagnostics_from_logits, run_step


def test_model_outputs_base_head_shapes_and_finite() -> None:
    model = HydraPolicyNet(hidden=16, blocks=1, bottleneck=4)
    out = model(torch.randn(2, 192, 34))
    assert out.policy_logits.shape == (2, ACTION_SPACE)
    assert out.value.shape == (2, 1)
    assert out.score_pdf.shape == (2, SCORE_BINS)
    assert out.score_cdf.shape == (2, SCORE_BINS)
    assert out.opp_tenpai.shape == (2, OPPONENTS)
    assert out.grp.shape == (2, GRP_CLASSES)
    assert out.oracle_critic.shape == (2, 4)
    assert out.safety_residual.shape == (2, ACTION_SPACE)
    assert out.opp_next_discard.shape == (2, OPPONENTS, TILE_WIDTH)
    assert out.danger.shape == (2, OPPONENTS, TILE_WIDTH)
    for tensor in (
        out.policy_logits,
        out.value,
        out.score_pdf,
        out.score_cdf,
        out.opp_tenpai,
        out.grp,
        out.oracle_critic,
        out.safety_residual,
        out.opp_next_discard,
        out.danger,
    ):
        assert bool(torch.isfinite(tensor).all())


def test_recommended_large_conv_profile_instantiates_and_keeps_output_contract() -> None:
    model = HydraPolicyNet(hidden=384, blocks=12, bottleneck=96)
    params = sum(parameter.numel() for parameter in model.parameters())

    assert 11_000_000 <= params <= 12_000_000
    with torch.no_grad():
        out = model(torch.randn(1, 192, 34))
    assert out.policy_logits.shape == (1, ACTION_SPACE)
    assert out.opp_next_discard.shape == (1, OPPONENTS, TILE_WIDTH)
    assert out.danger.shape == (1, OPPONENTS, TILE_WIDTH)
    assert bool(torch.isfinite(out.policy_logits).all())


@pytest.mark.parametrize(
    ("hidden", "blocks", "bottleneck"),
    [(384, 12, 96), (512, 16, 128)],
)
def test_large_conv_profiles_are_constructible(hidden: int, blocks: int, bottleneck: int) -> None:
    model = HydraPolicyNet(hidden=hidden, blocks=blocks, bottleneck=bottleneck)

    assert len(model.backbone.blocks) == blocks
    assert model.base_heads.in_features == hidden


@pytest.mark.parametrize("backbone_profile", [BACKBONE_PROFILE_CONVNEXT_TILE_K7, BACKBONE_PROFILE_GLOBAL_POOL_BIAS])
def test_resnet_family_backbone_profiles_keep_output_contract_and_finite(backbone_profile: str) -> None:
    model = HydraPolicyNet(hidden=16, blocks=1, bottleneck=4, backbone_profile=backbone_profile)
    out = model(torch.randn(2, 192, 34))
    assert out.policy_logits.shape == (2, ACTION_SPACE)
    assert out.opp_next_discard.shape == (2, OPPONENTS, TILE_WIDTH)
    assert out.danger.shape == (2, OPPONENTS, TILE_WIDTH)
    assert all(bool(torch.isfinite(tensor).all()) for tensor in (out.policy_logits, out.opp_next_discard, out.danger))


def test_tileformer_bias_outputs_base_head_shapes_and_finite() -> None:
    model = HydraPolicyNet(hidden=24, blocks=1, bottleneck=4, backbone_profile=BACKBONE_PROFILE_TILEFORMER_BIAS)
    out = model(torch.randn(2, 192, 34))
    assert out.policy_logits.shape == (2, ACTION_SPACE)
    assert out.value.shape == (2, 1)
    assert out.score_pdf.shape == (2, SCORE_BINS)
    assert out.score_cdf.shape == (2, SCORE_BINS)
    assert out.opp_tenpai.shape == (2, OPPONENTS)
    assert out.grp.shape == (2, GRP_CLASSES)
    assert out.oracle_critic.shape == (2, 4)
    assert out.safety_residual.shape == (2, ACTION_SPACE)
    assert out.opp_next_discard.shape == (2, OPPONENTS, TILE_WIDTH)
    assert out.danger.shape == (2, OPPONENTS, TILE_WIDTH)
    assert all(
        bool(torch.isfinite(tensor).all())
        for tensor in (
            out.policy_logits,
            out.value,
            out.score_pdf,
            out.score_cdf,
            out.opp_tenpai,
            out.grp,
            out.oracle_critic,
            out.safety_residual,
            out.opp_next_discard,
            out.danger,
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tileformer_bias_forward_works_on_cuda() -> None:
    model = HydraPolicyNet(hidden=24, blocks=1, bottleneck=4, backbone_profile=BACKBONE_PROFILE_TILEFORMER_BIAS).to(
        "cuda"
    )
    out = model(torch.randn(2, 192, 34, device="cuda"))
    assert out.policy_logits.shape == (2, ACTION_SPACE)
    assert out.policy_logits.device.type == "cuda"


def test_singleton_height_conv2d_matches_tile_axis_conv1d() -> None:
    obs = torch.randn(2, 3, 34)
    weight = torch.randn(5, 3, 3)
    expected = F.conv1d(obs, weight, padding=1)
    actual = F.conv2d(obs.unsqueeze(2), weight.unsqueeze(2), padding=(0, 1)).squeeze(2)
    torch.testing.assert_close(actual, expected)


_CONV2D_RESIDUAL_BACKBONE_PROFILES = (BACKBONE_PROFILE_CONV2D_LOCAL3, BACKBONE_PROFILE_GLOBAL_POOL_BIAS)


@pytest.mark.parametrize("profile", RESIDUAL_PROFILES)
@pytest.mark.parametrize("backbone_profile", _CONV2D_RESIDUAL_BACKBONE_PROFILES)
def test_residual_profiles_keep_output_contract_and_expected_parameters(profile: str, backbone_profile: str) -> None:
    model = HydraPolicyNet(
        hidden=16, blocks=1, bottleneck=4, residual_profile=profile, backbone_profile=backbone_profile
    )
    out = model(torch.randn(2, 192, 34))
    assert out.policy_logits.shape == (2, ACTION_SPACE)
    assert out.opp_next_discard.shape == (2, OPPONENTS, TILE_WIDTH)
    assert out.danger.shape == (2, OPPONENTS, TILE_WIDTH)
    state_keys = model.state_dict().keys()
    if profile == RESIDUAL_PROFILE_MISH_ECA:
        assert any("eca_conv" in key for key in state_keys)
        assert not any("se_fc" in key for key in state_keys)
    elif profile.endswith("no_se"):
        assert not any("se_fc" in key for key in state_keys)
    else:
        assert any("se_fc" in key for key in state_keys)
        assert not any("eca_conv" in key for key in state_keys)
    if profile == RESIDUAL_PROFILE_RELU_NO_NORM_NO_SE:
        assert not any("norm" in key for key in state_keys)
    else:
        assert any("norm" in key for key in state_keys)
    assert not any("eca_conv" in key for key in state_keys) or profile == RESIDUAL_PROFILE_MISH_ECA
    assert all(bool(torch.isfinite(tensor).all()) for tensor in (out.policy_logits, out.opp_next_discard, out.danger))


def test_profile_validation_includes_ablation_names() -> None:
    assert BACKBONE_PROFILE_CONVNEXT_TILE_K7 in BACKBONE_PROFILES
    assert BACKBONE_PROFILE_GLOBAL_POOL_BIAS in BACKBONE_PROFILES
    assert RESIDUAL_PROFILE_MISH_ECA in RESIDUAL_PROFILES


def test_mish_eca_residual_profile_keeps_output_contract_and_uses_eca() -> None:
    model = HydraPolicyNet(hidden=16, blocks=1, bottleneck=4, residual_profile=RESIDUAL_PROFILE_MISH_ECA)
    out = model(torch.randn(2, 192, 34))
    state_keys = model.state_dict().keys()
    assert out.policy_logits.shape == (2, ACTION_SPACE)
    assert out.opp_next_discard.shape == (2, OPPONENTS, TILE_WIDTH)
    assert out.danger.shape == (2, OPPONENTS, TILE_WIDTH)
    assert any("eca_conv" in key for key in state_keys)
    assert not any("se_fc" in key for key in state_keys)
    assert all(bool(torch.isfinite(tensor).all()) for tensor in (out.policy_logits, out.opp_next_discard, out.danger))


def test_group_norm_hidden_must_match_hydra_group_contract() -> None:
    with pytest.raises(ValueError, match=r"hidden=48 groups=32"):
        HydraPolicyNet(hidden=48, blocks=1, bottleneck=4)


def test_no_norm_profile_allows_non_divisible_hidden_for_ablation() -> None:
    model = HydraPolicyNet(hidden=48, blocks=1, bottleneck=4, residual_profile=RESIDUAL_PROFILE_RELU_NO_NORM_NO_SE)
    out = model(torch.randn(2, 192, 34))
    assert out.policy_logits.shape == (2, ACTION_SPACE)
    assert not any("norm" in key for key in model.state_dict())
    assert bool(torch.isfinite(out.policy_logits).all())


def test_invalid_backbone_profile_hard_errors() -> None:
    with pytest.raises(ValueError, match="unsupported backbone profile"):
        HydraPolicyNet(hidden=16, blocks=1, bottleneck=4, backbone_profile="bad_profile")


def test_default_profiles_are_conv2d_local3_mish_se() -> None:
    model = HydraPolicyNet(hidden=16, blocks=1, bottleneck=4)
    assert model.backbone_profile == BACKBONE_PROFILE_CONV2D_LOCAL3
    assert model.residual_profile == RESIDUAL_PROFILE_DEFAULT


def test_invalid_residual_profile_hard_errors() -> None:
    with pytest.raises(ValueError, match="unsupported residual profile"):
        HydraPolicyNet(hidden=16, blocks=1, bottleneck=4, residual_profile="bad_profile")


def test_masked_policy_ce_blocks_illegal_action() -> None:
    logits = torch.tensor([[0.0, 100.0, 1.0]])
    target = torch.tensor([[0.0, 0.0, 1.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    actual = masked_policy_ce(logits, target, mask)
    expected = -F.log_softmax(torch.tensor([[0.0, -1.0e9, 1.0]]), dim=1)[0, 2]
    torch.testing.assert_close(actual, expected.reshape(1))


def test_masked_policy_ce_indices_matches_dense_target() -> None:
    logits = torch.tensor([[0.0, 100.0, 1.0]])
    target = torch.tensor([2])
    mask = torch.tensor([[True, False, True]])
    actual = masked_policy_ce_indices(logits, target, mask)
    expected = masked_policy_ce(
        logits, F.one_hot(target, num_classes=3).to(dtype=torch.float32), mask.to(dtype=torch.float32)
    )
    torch.testing.assert_close(actual, expected)


def test_value_half_mse() -> None:
    actual = value_mse(torch.tensor([[0.5], [-0.5]]), torch.tensor([0.0, -1.0]))
    torch.testing.assert_close(actual, torch.tensor([0.125, 0.125]))


def test_soft_ce_terms() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[0.0, 1.0, 0.0]])
    expected = -F.log_softmax(logits, dim=1)[0, 1]
    outputs = _tiny_outputs(grp=logits, score_pdf=logits)
    targets = _tiny_targets(grp=target, score_pdf=target)
    breakdown = base_loss(
        outputs, targets, LossWeights(policy=0.0, value=0.0, grp=1.0, tenpai=0.0, danger=0.0, opp_next=0.0, score=1.0)
    )
    torch.testing.assert_close(breakdown.grp, expected)
    torch.testing.assert_close(breakdown.score_pdf, expected)


def test_bce_terms() -> None:
    outputs = _tiny_outputs(opp_tenpai=torch.tensor([[0.0, 1.0, -1.0]]), score_cdf=torch.tensor([[0.0, 1.0]]))
    targets = _tiny_targets(tenpai=torch.tensor([[1.0, 0.0, 1.0]]), score_cdf=torch.tensor([[1.0, 0.0]]))
    breakdown = base_loss(
        outputs, targets, LossWeights(policy=0.0, value=0.0, grp=0.0, tenpai=1.0, danger=0.0, opp_next=0.0, score=1.0)
    )
    expected_tenpai = F.binary_cross_entropy_with_logits(
        outputs.opp_tenpai, targets.tenpai_target, reduction="none"
    ).mean()
    expected_cdf = F.binary_cross_entropy_with_logits(
        outputs.score_cdf, targets.score_cdf_target, reduction="none"
    ).mean()
    torch.testing.assert_close(breakdown.tenpai, expected_tenpai)
    torch.testing.assert_close(breakdown.score_cdf, expected_cdf)


def test_danger_focal_with_mask() -> None:
    logits = torch.tensor([[[0.0, 2.0], [-2.0, 0.5]]])
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    mask = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
    actual = danger_focal_bce(logits, target, mask)
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = target * p + (1.0 - target) * (1.0 - p)
    expected = (((1.0 - p_t) ** 2.0) * 0.25 * bce * mask).sum(dim=(1, 2))
    torch.testing.assert_close(actual, expected)


def test_opp_next_ce_averages_over_opponents() -> None:
    logits = torch.tensor([[[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]])
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
    actual = opp_next_ce(logits, target)
    expected = torch.stack(
        [
            -F.log_softmax(logits[0, 0], dim=0)[0],
            -F.log_softmax(logits[0, 1], dim=0)[1],
            -F.log_softmax(logits[0, 2], dim=0)[0],
        ]
    ).mean()
    torch.testing.assert_close(actual, expected.reshape(1))


def test_real_parity_batch_full_base_loss_optimizer_step() -> None:
    manifest_path = Path("crates/hydra-bc-shards/target/python-parity-fixture/manifest.json")
    if not manifest_path.exists():
        pytest.skip("run Rust compact_reader_exports_python_parity_fixture first")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with BcShardReader(manifest_path) as reader:
        batch = reader.batch_range(0, 1)
    model = HydraPolicyNet(hidden=16, blocks=1, bottleneck=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    obs = torch.from_numpy(batch.obs).to(device)
    targets = targets_from_policy_batch(batch, device)
    loss = base_loss(model(obs), targets).total
    assert bool(torch.isfinite(loss))
    loss.backward()
    optimizer.step()


def test_oracle_critic_zero_sum_penalty_formula() -> None:
    pred = torch.tensor([[1.0, 2.0, -1.0, 0.0], [0.5, -0.5, 0.25, -0.25]])
    target = torch.tensor([[0.5, 0.0, -0.5, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([1.0, 0.0])
    actual = oracle_critic_loss(pred, target, mask)
    centered = pred - pred.mean(dim=1, keepdim=True)
    mse = ((centered - target) ** 2).mean(dim=1) * 0.5
    penalty = (pred.sum(dim=1) ** 2) * 10.0
    expected = ((mse + penalty) * mask).sum() / mask.sum().clamp(min=1.0)
    torch.testing.assert_close(actual, expected)


def test_safety_residual_masked_half_mse_denominator() -> None:
    pred = torch.tensor([[1.0, 3.0, 9.0], [2.0, 4.0, 8.0]])
    target = torch.tensor([[0.0, 1.0, 0.0], [1.0, 4.0, 9.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    actual = safety_residual_loss(pred, target, mask)
    sq = 0.5 * (pred - target) ** 2
    expected = (sq * mask).sum() / mask.sum().clamp(min=1.0)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(safety_residual_loss(pred, target, torch.zeros_like(mask)), torch.tensor(0.0))


def test_missing_safety_target_positive_weight_hard_errors() -> None:
    with pytest.raises(ValueError, match="safety_target"):
        base_loss(_tiny_outputs(), _tiny_targets(), LossWeights(safety_residual=0.01))


def test_real_parity_batch_safety_loss_positive_when_enabled() -> None:
    manifest_path = Path("crates/hydra-bc-shards/target/python-parity-fixture/manifest.json")
    if not manifest_path.exists():
        pytest.skip("run Rust compact_reader_exports_python_parity_fixture first")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with BcShardReader(manifest_path) as reader:
        batch = reader.batch_range(0, 1)
    if batch.safety_target is None or batch.safety_mask is None:
        pytest.skip("fixture has no safety residual labels")
    model = HydraPolicyNet(hidden=16, blocks=1, bottleneck=4).to(device)
    targets = targets_from_policy_batch(batch, device)
    breakdown = base_loss(model(torch.from_numpy(batch.obs).to(device)), targets, LossWeights(safety_residual=0.01))
    assert bool(torch.isfinite(breakdown.safety_residual))
    assert breakdown.safety_residual.item() > 0.0


def test_compiled_loss_step_matches_base_loss() -> None:
    model = HydraPolicyNet(hidden=16, blocks=1, bottleneck=4)
    batch = 1
    targets = BaseTargets(
        policy_target=torch.zeros(batch, dtype=torch.int64),
        legal_mask=torch.ones(batch, ACTION_SPACE, dtype=torch.bool),
        value_target=torch.zeros(batch),
        grp_target=F.one_hot(torch.zeros(batch, dtype=torch.int64), num_classes=GRP_CLASSES).to(dtype=torch.float32),
        tenpai_target=torch.zeros(batch, OPPONENTS),
        danger_target=torch.zeros(batch, OPPONENTS, TILE_WIDTH),
        danger_mask=torch.ones(batch, OPPONENTS, TILE_WIDTH),
        opp_next_target=F.one_hot(torch.zeros(batch, OPPONENTS, dtype=torch.int64), num_classes=TILE_WIDTH).to(
            dtype=torch.float32
        ),
        score_pdf_target=F.one_hot(torch.zeros(batch, dtype=torch.int64), num_classes=SCORE_BINS).to(
            dtype=torch.float32
        ),
        score_cdf_target=torch.zeros(batch, SCORE_BINS),
        oracle_target=torch.zeros(batch, 4),
        oracle_target_mask=torch.ones(batch),
        safety_target=torch.ones(batch, ACTION_SPACE),
        safety_mask=torch.ones(batch, ACTION_SPACE),
        exit_target=torch.zeros(batch, ACTION_SPACE),
        exit_mask=torch.zeros(batch, ACTION_SPACE),
        deltaq_target=torch.zeros(batch, ACTION_SPACE),
        deltaq_mask=torch.zeros(batch, ACTION_SPACE),
    )
    obs = torch.randn(1, 192, 34)
    weights = LossWeights(safety_residual=0.01, oracle_critic=0.02)
    expected = base_loss(model(obs), targets, weights).total
    actual = HydraCompiledLossStep(model, "full_base", weights)(*loss_step_args(obs, targets, 0, 1))
    torch.testing.assert_close(actual, expected)


def test_tileformer_bias_full_base_loss_optimizer_step() -> None:
    batch = 2
    model = HydraPolicyNet(hidden=24, blocks=1, bottleneck=4, backbone_profile=BACKBONE_PROFILE_TILEFORMER_BIAS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    targets = _cpu_action_targets()
    obs = torch.randn(batch, 192, 34)
    loss = base_loss(model(obs), targets, LossWeights()).total
    assert bool(torch.isfinite(loss))
    loss.backward()
    optimizer.step()


def test_compiled_loss_targets_allocate_optional_fallbacks_once() -> None:
    targets = _tiny_targets()
    prepared = targets_for_compiled_loss(targets, LossWeights())
    assert prepared.oracle_target is not None
    assert prepared.oracle_target_mask is not None
    assert prepared.safety_target is not None
    assert prepared.safety_mask is not None
    assert prepared.oracle_target.shape == (1, 4)
    assert prepared.safety_target.shape == (1, ACTION_SPACE)
    first_safety = prepared.safety_target
    sliced = loss_step_args(torch.zeros(1, 192, 34), prepared, 0, 1)
    assert sliced[-6].data_ptr() == first_safety.data_ptr()


def test_active_loss_heads_policy_only_is_explicit() -> None:
    assert active_loss_heads(LossWeights(), "policy_only") == ("policy",)


def test_optional_positive_weight_missing_label_hard_errors() -> None:
    with pytest.raises(ValueError, match="oracle targets"):
        targets_for_compiled_loss(_tiny_targets(), LossWeights(oracle_critic=0.1))


def test_target_coverage_distinguishes_absent_zero_and_nonzero_optional_masks() -> None:
    absent = target_coverage_dict(_tiny_targets(), LossWeights(safety_residual=0.1))
    assert absent["safety_residual"] == {"active": True, "status": "absent", "fraction": 0.0}

    zero_mask = _tiny_targets_with_optional(safety_mask=torch.zeros(1, 3))
    zero = target_coverage_dict(zero_mask, LossWeights(safety_residual=0.1))
    assert zero["safety_residual"] == {"active": True, "status": "present_zero", "fraction": 0.0}

    nonzero_mask = _tiny_targets_with_optional(safety_mask=torch.tensor([[0.0, 1.0, 0.0]]))
    nonzero = target_coverage_dict(nonzero_mask, LossWeights(safety_residual=0.1))
    assert nonzero["safety_residual"] == {"active": True, "status": "present_positive", "fraction": 1.0}


def _targets_with_action_lane(
    *,
    exit_target: torch.Tensor | None = None,
    exit_mask: torch.Tensor | None = None,
    deltaq_target: torch.Tensor | None = None,
    deltaq_mask: torch.Tensor | None = None,
) -> BaseTargets:
    targets = _tiny_targets()
    return BaseTargets(
        policy_target=torch.zeros(1, dtype=torch.int64),
        legal_mask=torch.ones(1, ACTION_SPACE, dtype=torch.bool),
        value_target=targets.value_target,
        grp_target=targets.grp_target,
        tenpai_target=targets.tenpai_target,
        danger_target=targets.danger_target,
        danger_mask=targets.danger_mask,
        opp_next_target=targets.opp_next_target,
        score_pdf_target=targets.score_pdf_target,
        score_cdf_target=targets.score_cdf_target,
        exit_target=exit_target,
        exit_mask=exit_mask,
        deltaq_target=deltaq_target,
        deltaq_mask=deltaq_mask,
    )


def _action_lane(action: int, value: float) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros(1, ACTION_SPACE, dtype=torch.float32)
    mask = torch.zeros(1, ACTION_SPACE, dtype=torch.float32)
    target[0, action] = value
    mask[0, action] = 1.0
    return target, mask


def _action_outputs() -> HydraBaseOutput:
    out = _tiny_outputs()
    return HydraBaseOutput(
        policy_logits=torch.zeros(1, ACTION_SPACE),
        value=out.value,
        score_pdf=out.score_pdf,
        score_cdf=out.score_cdf,
        opp_tenpai=out.opp_tenpai,
        grp=out.grp,
        oracle_critic=out.oracle_critic,
        safety_residual=torch.zeros(1, ACTION_SPACE),
        opp_next_discard=out.opp_next_discard,
        danger=out.danger,
    )


def test_optional_action_target_boundary_validation_rejects_bad_inputs() -> None:
    good_target, good_mask = _action_lane(1, 1.0)
    validate_base_targets(
        _targets_with_action_lane(exit_target=good_target, exit_mask=good_mask), batch=1, device=None, context="test"
    )
    with pytest.raises(ValueError, match="both be present"):
        validate_base_targets(_targets_with_action_lane(exit_target=good_target), batch=1, device=None, context="test")
    with pytest.raises(ValueError, match="shape"):
        validate_base_targets(
            _targets_with_action_lane(exit_target=good_target[:, :2], exit_mask=good_mask[:, :2]),
            batch=1,
            device=None,
            context="test",
        )
    with pytest.raises(ValueError, match="dtype"):
        validate_base_targets(
            _targets_with_action_lane(exit_target=good_target.to(torch.float64), exit_mask=good_mask),
            batch=1,
            device=None,
            context="test",
        )
    bad_mask = good_mask.clone()
    bad_mask[0, 1] = 0.5
    with pytest.raises(ValueError, match="binary"):
        validate_base_targets(
            _targets_with_action_lane(exit_target=good_target, exit_mask=bad_mask), batch=1, device=None, context="test"
        )
    bad_target = good_target.clone()
    bad_target[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_base_targets(
            _targets_with_action_lane(exit_target=bad_target, exit_mask=good_mask), batch=1, device=None, context="test"
        )
    off_mask_target = good_target.clone()
    off_mask_target[0, 0] = 0.1
    with pytest.raises(ValueError, match="outside target mask"):
        validate_base_targets(
            _targets_with_action_lane(exit_target=off_mask_target, exit_mask=good_mask),
            batch=1,
            device=None,
            context="test",
        )
    illegal_mask = torch.zeros(1, ACTION_SPACE, dtype=torch.float32)
    illegal_target = torch.zeros(1, ACTION_SPACE, dtype=torch.float32)
    illegal_mask[0, 2] = 1.0
    illegal_target[0, 2] = 1.0
    illegal = _targets_with_action_lane(exit_target=illegal_target, exit_mask=illegal_mask)
    illegal = BaseTargets(
        policy_target=illegal.policy_target,
        legal_mask=torch.ones(1, ACTION_SPACE, dtype=torch.bool),
        value_target=illegal.value_target,
        grp_target=illegal.grp_target,
        tenpai_target=illegal.tenpai_target,
        danger_target=illegal.danger_target,
        danger_mask=illegal.danger_mask,
        opp_next_target=illegal.opp_next_target,
        score_pdf_target=illegal.score_pdf_target,
        score_cdf_target=illegal.score_cdf_target,
        exit_target=illegal.exit_target,
        exit_mask=illegal.exit_mask,
    )
    illegal.legal_mask[0, 2] = False
    with pytest.raises(ValueError, match="illegal action"):
        validate_base_targets(illegal, batch=1, device=None, context="test")


def test_zero_weight_exit_deltaq_do_not_change_base_loss() -> None:
    base = base_loss(_action_outputs(), _targets_with_action_lane(), LossWeights())
    target, mask = _action_lane(1, 1.0)
    with_lanes = base_loss(
        _action_outputs(),
        _targets_with_action_lane(exit_target=target, exit_mask=mask, deltaq_target=target, deltaq_mask=mask),
        LossWeights(exit=0.0, deltaq=0.0),
    )
    torch.testing.assert_close(with_lanes.total, base.total)
    assert targets_for_compiled_loss(_tiny_targets(), LossWeights()).exit_target is not None


def test_exit_positive_weight_requires_labels_and_unit_support_mass() -> None:
    with pytest.raises(ValueError, match="exit targets"):
        targets_for_compiled_loss(_tiny_targets(), LossWeights(exit=0.1))
    target, mask = _action_lane(1, 1.0)
    targets = targets_for_compiled_loss(
        _targets_with_action_lane(exit_target=target, exit_mask=mask), LossWeights(exit=0.1)
    )
    breakdown = base_loss(_action_outputs(), targets, LossWeights(exit=0.1))
    assert torch.isfinite(breakdown.exit)
    bad_mass, _ = _action_lane(1, 0.5)
    with pytest.raises(ValueError, match="approximately 1"):
        targets_for_compiled_loss(
            _targets_with_action_lane(exit_target=bad_mass, exit_mask=mask), LossWeights(exit=0.1)
        )


def test_exit_policy_loss_uses_unit_mass_soft_ce_without_support_scaling() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 9.0]])
    target = torch.tensor([[0.25, 0.75, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    legal = torch.tensor([[True, True, True, False]])

    actual = exit_policy_loss(logits, target, mask, legal)
    expected = -(target[:, :2] * F.log_softmax(logits[:, :2], dim=1)).sum(dim=1)
    old_wrong = expected / 2.0

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, old_wrong)


def test_exit_policy_loss_one_hot_target_matches_masked_policy_ce() -> None:
    logits = torch.tensor([[0.0, 1.5, -2.0, 7.0]])
    target = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    legal = torch.tensor([[True, True, True, False]])

    actual = exit_policy_loss(logits, target, mask, legal)
    expected = masked_policy_ce_indices(logits, torch.tensor([1]), legal)

    torch.testing.assert_close(actual, expected)


def test_deltaq_carrier_validates_but_positive_loss_fails_closed() -> None:
    target, mask = _action_lane(1, -0.5)
    validate_base_targets(
        _targets_with_action_lane(deltaq_target=target, deltaq_mask=mask), batch=1, device=None, context="test"
    )
    with pytest.raises(ValueError, match="delta_q_output_contract_missing"):
        targets_for_compiled_loss(
            _targets_with_action_lane(deltaq_target=target, deltaq_mask=mask), LossWeights(deltaq=0.1)
        )
    with pytest.raises(ValueError, match="delta_q_output_contract_missing"):
        base_loss(
            _action_outputs(),
            _targets_with_action_lane(deltaq_target=target, deltaq_mask=mask),
            LossWeights(deltaq=0.1),
        )


def _cuda_event_factory(*_args: object, **_kwargs: object) -> object:
    class _Event:
        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, _other: object) -> float:
            return 0.0

    return _Event()


def test_run_step_rejects_nonfinite_first_microbatch_before_optimizer_step(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Loss(torch.nn.Module):
        calls: int

        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = 0

        @override
        def forward(self, *_args: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            if self.calls == 1:
                return self.param * torch.tensor(float("nan"))
            return self.param * 0.0

    monkeypatch.setattr(torch.cuda, "Event", _cuda_event_factory)
    loss = _Loss()
    model = HydraPolicyNet(hidden=16, blocks=1, bottleneck=4)
    optimizer = torch.optim.SGD(loss.parameters(), lr=1.0)
    targets = targets_for_compiled_loss(_two_row_targets(), LossWeights())
    before = loss.param.detach().clone()
    with pytest.raises(RuntimeError, match="non-finite BC loss"):
        run_step(loss, model, optimizer, torch.zeros(2, 192, 34), targets, LossWeights(), "full_base", 1, False, False)
    torch.testing.assert_close(loss.param.detach(), before)


def test_lr_scheduler_cosine_warmup_and_floor() -> None:
    scheduler = LrScheduler(
        LrSchedulerConfig(base_lr=1.0, min_lr=0.1, warmup_steps=2, total_steps=6, target_games=None, schedule="cosine")
    )

    assert scheduler.lr_for_step(0) == 0.0
    assert scheduler.lr_for_step(2) == 1.0
    assert scheduler.lr_for_step(6) == pytest.approx(0.1)


def test_lr_scheduler_constant_stays_base_lr() -> None:
    scheduler = LrScheduler(
        LrSchedulerConfig(
            base_lr=0.25, min_lr=0.01, warmup_steps=3, total_steps=8, target_games=None, schedule="constant"
        )
    )

    assert scheduler.lr_for_step(0) == 0.25
    assert scheduler.lr_for_step(8) == 0.25


def test_lr_scheduler_target_games_keeps_step_based_warmup() -> None:
    scheduler = LrScheduler(
        LrSchedulerConfig(base_lr=1.0, min_lr=0.1, warmup_steps=2, total_steps=None, target_games=6, schedule="cosine")
    )

    assert scheduler.lr_for_step(0, completed_games=6) == 0.0
    assert scheduler.lr_for_step(1, completed_games=6) == 0.5
    assert scheduler.lr_for_step(2, completed_games=0) == 1.0
    assert scheduler.lr_for_step(2, completed_games=6) == pytest.approx(0.1)


def test_run_step_clips_gradients_and_reports_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Loss(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(10.0))

        @override
        def forward(self, *_args: torch.Tensor) -> torch.Tensor:
            return self.param * self.param

    monkeypatch.setattr(torch.cuda, "Event", _cuda_event_factory)
    loss = _Loss()
    model = HydraPolicyNet(hidden=16, blocks=1, bottleneck=4)
    optimizer = torch.optim.SGD(loss.parameters(), lr=1.0)
    targets = targets_for_compiled_loss(_cpu_action_targets(), LossWeights())

    stat = run_step(
        loss,
        model,
        optimizer,
        torch.zeros(2, 192, 34),
        targets,
        LossWeights(),
        "full_base",
        2,
        False,
        False,
        grad_clip_norm=0.5,
        collect_diagnostics=False,
    )

    assert stat.grad_norm > 0.5
    torch.testing.assert_close(loss.param.detach(), torch.tensor(9.5))


def test_run_step_collects_policy_diagnostics_without_extra_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StaticModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = 0

        @override
        def forward(self, obs: torch.Tensor) -> HydraBaseOutput:
            self.calls += 1
            batch = obs.shape[0]
            logits = obs.new_full((batch, ACTION_SPACE), -8.0)
            logits[0, 0] = 4.0
            logits[0, 1] = 1.0
            logits[1, 1] = 3.0
            logits[1, 0] = 2.0
            logits = logits * self.param
            grp = obs.new_zeros((batch, GRP_CLASSES))
            grp[:, 0] = 1.0
            score_pdf = obs.new_zeros((batch, SCORE_BINS))
            score_pdf[:, 0] = 1.0
            opp_next = obs.new_zeros((batch, OPPONENTS, TILE_WIDTH))
            opp_next[:, :, 0] = 1.0
            return HydraBaseOutput(
                policy_logits=logits,
                value=obs.new_zeros((batch, 1)),
                score_pdf=score_pdf,
                score_cdf=obs.new_zeros((batch, SCORE_BINS)),
                opp_tenpai=obs.new_zeros((batch, OPPONENTS)),
                grp=grp,
                oracle_critic=obs.new_zeros((batch, 4)),
                safety_residual=obs.new_zeros((batch, ACTION_SPACE)),
                opp_next_discard=opp_next,
                danger=obs.new_zeros((batch, OPPONENTS, TILE_WIDTH)),
            )

    class _StaticLossStep(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_outputs: object = None
            self.last_breakdown: LossBreakdown | None = None
            self.inner = _StaticModel()

        # pyrefly: ignore[missing-override-decorator] - Test loss shim changes nn.Module.forward shape.
        def forward(self, obs: torch.Tensor, *_args: torch.Tensor) -> object:
            outputs = self.inner(obs)
            self.last_outputs = outputs
            loss = masked_policy_ce_indices(outputs.policy_logits, targets.policy_target, targets.legal_mask).mean()
            self.last_breakdown = LossBreakdown(
                total=loss,
                policy=loss,
                value=loss * 0.0,
                grp=loss * 0.0,
                tenpai=loss * 0.0,
                danger=loss * 0.0,
                opp_next=loss * 0.0,
                score_pdf=loss * 0.0,
                score_cdf=loss * 0.0,
                oracle_critic=loss * 0.0,
                safety_residual=loss * 0.0,
                exit=loss * 0.0,
                deltaq=loss * 0.0,
            )
            return loss

    monkeypatch.setattr(torch.cuda, "Event", _cuda_event_factory)
    loss_step = _StaticLossStep()
    model = loss_step.inner
    optimizer = torch.optim.SGD(loss_step.parameters(), lr=0.0)
    targets = targets_for_compiled_loss(_cpu_action_targets(), LossWeights())
    obs = torch.zeros(2, 192, 34)

    stat = run_step(
        loss_step,
        model,
        optimizer,
        obs,
        targets,
        LossWeights(),
        "full_base",
        2,
        False,
        False,
        collect_diagnostics=True,
    )
    calls_after_step = model.calls
    expected = policy_diagnostics_from_logits(_StaticModel()(obs).policy_logits, targets)
    calls_after_expected = model.calls

    assert calls_after_step == 1
    assert calls_after_expected == 1
    assert stat.policy_nll == pytest.approx(float(expected["policy_nll"].detach()))
    assert stat.policy_accuracy == pytest.approx(float(expected["policy_accuracy"].detach()))
    assert stat.policy_top3_accuracy == pytest.approx(float(expected["policy_top3_accuracy"].detach()))
    assert stat.policy_top5_accuracy == pytest.approx(float(expected["policy_top5_accuracy"].detach()))
    assert stat.policy_confidence == pytest.approx(float(expected["policy_confidence"].detach()))
    assert stat.policy_entropy == pytest.approx(float(expected["policy_entropy"].detach()))
    assert stat.policy_target_prob == pytest.approx(float(expected["policy_target_prob"].detach()))
    assert stat.policy_margin == pytest.approx(float(expected["policy_margin"].detach()))


def _tiny_outputs(
    grp: torch.Tensor | None = None,
    score_pdf: torch.Tensor | None = None,
    opp_tenpai: torch.Tensor | None = None,
    score_cdf: torch.Tensor | None = None,
) -> HydraBaseOutput:
    return HydraBaseOutput(
        policy_logits=torch.zeros(1, 3),
        value=torch.zeros(1, 1),
        score_pdf=score_pdf if score_pdf is not None else torch.zeros(1, 3),
        score_cdf=score_cdf if score_cdf is not None else torch.zeros(1, 2),
        opp_tenpai=opp_tenpai if opp_tenpai is not None else torch.zeros(1, 3),
        grp=grp if grp is not None else torch.zeros(1, 3),
        oracle_critic=torch.zeros(1, 4),
        safety_residual=torch.zeros(1, 3),
        opp_next_discard=torch.zeros(1, 3, 2),
        danger=torch.zeros(1, 1, 2),
    )


def _tiny_targets(
    grp: torch.Tensor | None = None,
    score_pdf: torch.Tensor | None = None,
    tenpai: torch.Tensor | None = None,
    score_cdf: torch.Tensor | None = None,
) -> BaseTargets:
    return BaseTargets(
        policy_target=torch.tensor([[1.0, 0.0, 0.0]]),
        legal_mask=torch.tensor([[True, True, True]]),
        value_target=torch.zeros(1),
        grp_target=grp if grp is not None else torch.tensor([[1.0, 0.0, 0.0]]),
        tenpai_target=tenpai if tenpai is not None else torch.zeros(1, 3),
        danger_target=torch.zeros(1, 1, 2),
        danger_mask=torch.ones(1, 1, 2),
        opp_next_target=torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]),
        score_pdf_target=score_pdf if score_pdf is not None else torch.tensor([[1.0, 0.0, 0.0]]),
        score_cdf_target=score_cdf if score_cdf is not None else torch.zeros(1, 2),
    )


def _tiny_targets_with_optional(safety_mask: torch.Tensor) -> BaseTargets:
    targets = _tiny_targets()
    return BaseTargets(
        policy_target=targets.policy_target,
        legal_mask=targets.legal_mask,
        value_target=targets.value_target,
        grp_target=targets.grp_target,
        tenpai_target=targets.tenpai_target,
        danger_target=targets.danger_target,
        danger_mask=targets.danger_mask,
        opp_next_target=targets.opp_next_target,
        score_pdf_target=targets.score_pdf_target,
        score_cdf_target=targets.score_cdf_target,
        safety_target=torch.zeros_like(safety_mask),
        safety_mask=safety_mask,
    )


def _cpu_action_targets() -> BaseTargets:
    batch = 2
    actions = ACTION_SPACE
    grp = torch.zeros(batch, GRP_CLASSES)
    grp[:, 0] = 1.0
    score_pdf = torch.zeros(batch, SCORE_BINS)
    score_pdf[:, 0] = 1.0
    opp_next = torch.zeros(batch, OPPONENTS, TILE_WIDTH)
    opp_next[:, :, 0] = 1.0
    return BaseTargets(
        policy_target=torch.arange(batch, dtype=torch.int64),
        legal_mask=torch.ones(batch, actions, dtype=torch.bool),
        value_target=torch.zeros(batch),
        grp_target=grp,
        tenpai_target=torch.zeros(batch, OPPONENTS),
        danger_target=torch.zeros(batch, OPPONENTS, TILE_WIDTH),
        danger_mask=torch.ones(batch, OPPONENTS, TILE_WIDTH),
        opp_next_target=opp_next,
        score_pdf_target=score_pdf,
        score_cdf_target=torch.zeros(batch, SCORE_BINS),
    )


def _two_row_targets() -> BaseTargets:
    target = _tiny_targets()
    return BaseTargets(
        policy_target=target.policy_target.repeat(2, 1),
        legal_mask=target.legal_mask.repeat(2, 1),
        value_target=target.value_target.repeat(2),
        grp_target=target.grp_target.repeat(2, 1),
        tenpai_target=target.tenpai_target.repeat(2, 1),
        danger_target=target.danger_target.repeat(2, 1, 1),
        danger_mask=target.danger_mask.repeat(2, 1, 1),
        opp_next_target=target.opp_next_target.repeat(2, 1, 1),
        score_pdf_target=target.score_pdf_target.repeat(2, 1),
        score_cdf_target=target.score_cdf_target.repeat(2, 1),
    )
