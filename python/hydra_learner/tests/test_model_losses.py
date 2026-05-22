from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from hydra_learner.losses import (
    BaseTargets,
    LossWeights,
    base_loss,
    danger_focal_bce,
    masked_policy_ce,
    opp_next_ce,
    oracle_critic_loss,
    safety_residual_loss,
    value_mse,
)
from hydra_learner.model import (
    ACTION_SPACE,
    GRP_CLASSES,
    OPPONENTS,
    SCORE_BINS,
    TILE_WIDTH,
    HydraBaseOutput,
    HydraPolicyNet,
)
from hydra_learner.shards import BcShardReader
from hydra_learner.train_bc import HydraCompiledLossStep, loss_step_args, targets_from_policy_batch


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


def test_masked_policy_ce_blocks_illegal_action() -> None:
    logits = torch.tensor([[0.0, 100.0, 1.0]])
    target = torch.tensor([[0.0, 0.0, 1.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    actual = masked_policy_ce(logits, target, mask)
    expected = -F.log_softmax(torch.tensor([[0.0, -1.0e9, 1.0]]), dim=1)[0, 2]
    torch.testing.assert_close(actual, expected.reshape(1))


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
        policy_target=F.one_hot(torch.zeros(batch, dtype=torch.int64), num_classes=ACTION_SPACE).to(
            dtype=torch.float32
        ),
        legal_mask=torch.ones(batch, ACTION_SPACE),
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
    )
    obs = torch.randn(1, 192, 34)
    weights = LossWeights(safety_residual=0.01, oracle_critic=0.02)
    expected = base_loss(model(obs), targets, weights).total
    actual = HydraCompiledLossStep(model, "full_base", weights)(*loss_step_args(obs, targets, 0, 1))
    torch.testing.assert_close(actual, expected)


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
        legal_mask=torch.tensor([[1.0, 1.0, 1.0]]),
        value_target=torch.zeros(1),
        grp_target=grp if grp is not None else torch.tensor([[1.0, 0.0, 0.0]]),
        tenpai_target=tenpai if tenpai is not None else torch.zeros(1, 3),
        danger_target=torch.zeros(1, 1, 2),
        danger_mask=torch.ones(1, 1, 2),
        opp_next_target=torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]),
        score_pdf_target=score_pdf if score_pdf is not None else torch.tensor([[1.0, 0.0, 0.0]]),
        score_cdf_target=score_cdf if score_cdf is not None else torch.zeros(1, 2),
    )
