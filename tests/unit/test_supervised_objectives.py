"""WP-05B supervised objectives: loss kernels and validators.

Covers masked behavior cloning (illegal-logit masking with exact-zero
gradients, illegal-target and all-false rejection), explicit auxiliary
weights (placement, event, and value heads with zero-may-be-absent, plus
0-based placement bounds), the real-model output adapter and the
pre-forward legal-mask gate, plus compiled-kernel and validator parity
with the eager loss (bitwise totals, identical error strings, finite
fp32 auxiliary outputs).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn

from hydra2.contracts.common import ContractError, IllegalActionError
from hydra2.training.loop_state import TrainingLoopConfig
from hydra2.training.objectives_loss import (
    compute_supervised_loss,
    masked_cross_entropy,
    supervised_loss_kernel,
    validate_supervised_inputs,
)
from hydra2.training.objectives_metrics import (
    compute_hot_scalars,
    compute_metrics,
    row_eval_primitives,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract_package("WP-05B")


NUM_ACTIONS_SMALL = 16  # small vocab for test speed (real table is 6792)
FEATURE_DIM = 16


# ---------------------------------------------------------------------------
# Stub models (deterministic, actor-visible only)
# ---------------------------------------------------------------------------


class StubPolicyModel(nn.Module):
    """Minimal policy model: linear over synthetic features."""

    def __init__(
        self, feature_dim: int = FEATURE_DIM, num_actions: int = NUM_ACTIONS_SMALL
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_actions)
        # Mirrors the production attention bf16 switch (model.py attn_bf16):
        # plain bool, never a parameter/buffer, so state_dict is unaffected.
        self.attn_bf16: bool = False

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["features"]  # [B,F]
        logits = self.linear(x)  # [B,A]
        return {"policy_logits": logits}


class StubModelWithAux(nn.Module):
    """Policy + placement + event + value heads for auxiliary weight tests."""

    def __init__(
        self, feature_dim: int = FEATURE_DIM, num_actions: int = NUM_ACTIONS_SMALL
    ) -> None:
        super().__init__()
        self.linear_policy = nn.Linear(feature_dim, num_actions)
        self.linear_placement = nn.Linear(feature_dim, 4)
        self.linear_event = nn.Linear(feature_dim, 3)
        self.linear_value = nn.Linear(feature_dim, 4)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["features"]
        return {
            "policy_logits": self.linear_policy(x),
            "placement_logits": self.linear_placement(x),
            "event_logits": {"win_or_not": self.linear_event(x)},
            "value_vector": self.linear_value(x),
        }


class StubModelPerSeat(nn.Module):
    """Policy + per-seat placement [B,4,4] + value [B,4] for oracle-join tests."""

    def __init__(
        self, feature_dim: int = FEATURE_DIM, num_actions: int = NUM_ACTIONS_SMALL
    ) -> None:
        super().__init__()
        self.linear_policy = nn.Linear(feature_dim, num_actions)
        self.linear_placement = nn.Linear(feature_dim, 16)
        self.linear_value = nn.Linear(feature_dim, 4)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["features"]
        return {
            "policy_logits": self.linear_policy(x),
            "placement_logits": self.linear_placement(x).view(x.shape[0], 4, 4),
            "value_vector": self.linear_value(x),
        }


# ---------------------------------------------------------------------------
# 1 Masked BC objective
# ---------------------------------------------------------------------------


def test_masked_ce_cpu_dispatch_matches_reference() -> None:
    """CPU dispatch takes the eager masked-math path (no fused kernel), exactly.

    Guards the fused-CE gate (objectives_loss masked_cross_entropy): on CPU
    (``logits.is_cuda`` False) the Triton custom op never fires and the
    masked-softmax formula matches a hand-rolled legal-only reference.
    A dispatch regression routing CPU rows through the fused op would fail
    here (wrong values or RuntimeError from the missing kernel).
    """
    torch.manual_seed(7)
    logits = torch.randn(4, 8)
    legal_mask = torch.zeros(4, 8, dtype=torch.bool)
    legal_mask[:, :5] = True
    targets = torch.tensor([0, 2, 4, 1], dtype=torch.long)
    assert not logits.is_cuda
    got = masked_cross_entropy(logits, targets, legal_mask, label_smoothing=0.1)
    # Independent legal-only reference: no fused kernel, no repo helper.
    masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
    log_prob = torch.nn.functional.log_softmax(masked_logits.float(), dim=-1)
    legal_counts = legal_mask.sum(dim=1).float()
    smooth = 0.1 / legal_counts
    batch_idx = torch.arange(4)
    legal_logp_sum = log_prob.masked_fill(~legal_mask, 0.0).sum(dim=1)
    ref = -((0.9) * log_prob[batch_idx, targets] + smooth * legal_logp_sum).mean()
    assert torch.equal(got, ref)
    hot = compute_hot_scalars(logits, targets, legal_mask)
    assert set(hot.keys()) == {"masked_nll", "top1", "top3", "top5"}
    assert math.isfinite(hot["masked_nll"]) and 0.0 <= hot["top1"] <= 1.0
    assert 0.0 <= hot["top3"] <= 1.0 and 0.0 <= hot["top5"] <= 1.0
    assert hot["top1"] <= hot["top3"] <= hot["top5"]

    torch.manual_seed(0)
    logits = torch.randn(2, 8)
    # Make illegal actions huge so they'd dominate unmasked softmax
    legal_mask = torch.tensor(
        [
            [True, True, False, False, False, False, False, False],
            [False, True, True, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    targets = torch.tensor([0, 1], dtype=torch.long)
    # Boost illegal logits to large positive
    logits[0, 2] = 100.0
    logits[1, 0] = 100.0
    loss_masked = masked_cross_entropy(logits, targets, legal_mask)
    # Compare to logits where illegal are -1e9: should be identical
    masked_logits = logits.masked_fill(~legal_mask, -1e9)
    expected = torch.nn.functional.cross_entropy(masked_logits, targets)
    assert torch.allclose(loss_masked, expected), f"{loss_masked.item()} vs {expected.item()}"
    # Gradient for illegal logits must be exactly zero
    logits.requires_grad_(True)
    loss = masked_cross_entropy(logits, targets, legal_mask)
    loss.backward()
    # illegal positions gradients ~0
    assert float(logits.grad[0, 2].item()) == pytest.approx(0.0, abs=1e-6)
    assert float(logits.grad[1, 0].item()) == pytest.approx(0.0, abs=1e-6)


def test_masked_bc_rejects_illegal_target_and_all_false(tmp_path: Path) -> None:
    logits = torch.randn(2, 4)
    legal_mask = torch.ones(2, 4, dtype=torch.bool)
    legal_mask[1, 2] = False
    # target illegal
    targets_illegal = torch.tensor([0, 2], dtype=torch.long)
    with pytest.raises(IllegalActionError):
        masked_cross_entropy(logits, targets_illegal, legal_mask)
    # all-false row
    legal_all_false = torch.zeros(2, 4, dtype=torch.bool)
    targets = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ContractError, match="all-false"):
        masked_cross_entropy(logits, targets, legal_all_false)


# ---------------------------------------------------------------------------
# 2 Auxiliary weights explicit, zero may be absent
# ---------------------------------------------------------------------------


def test_auxiliary_weights_explicit(tmp_path: Path) -> None:
    # Need placement and event targets in batch — inject via
    # monkey-patching tensorize? Instead test compute_supervised_loss directly.
    # Day-one: placement is per-seat [B,4,4] vs [B,4] (mean over seats) and
    # value is MSE on [B,4] UtilityVector.values.
    torch.manual_seed(1)
    B, A = 4, NUM_ACTIONS_SMALL
    logits = torch.randn(B, A)
    legal_mask = torch.ones(B, A, dtype=torch.bool)
    targets = torch.randint(0, A, (B,))
    # Make model output with auxiliary heads (per-seat placement + value)
    placement_logits = torch.randn(B, 4, 4)
    value_vector = torch.randn(B, 4)
    event_logits = {"win_or_not": torch.randn(B, 3)}
    batch = {
        "chosen_action_id": targets,
        "legal_mask": legal_mask,
        "placement_target": torch.randint(0, 4, (B, 4)),
        "value_target": torch.randn(B, 4),
        "event_targets": {"win_or_not": torch.randint(0, 3, (B,))},
    }
    model_out = {
        "policy_logits": logits,
        "placement_logits": placement_logits,
        "value_vector": value_vector,
        "event_logits": event_logits,
    }
    weights = {
        "w_policy": 1.0,
        "w_placement": 0.5,
        "w_value": 0.4,
        "w_event": {"win_or_not": 0.3},
    }
    losses = compute_supervised_loss(model_out, batch, weights)
    # total should be weighted sum (per-seat CE then mean over seats)
    policy_loss = masked_cross_entropy(logits, targets, legal_mask)
    place_loss = torch.nn.functional.cross_entropy(
        placement_logits.reshape(-1, 4), batch["placement_target"].reshape(-1)
    )
    val_loss = torch.nn.functional.mse_loss(value_vector, batch["value_target"])
    ev_loss = torch.nn.functional.cross_entropy(
        event_logits["win_or_not"], batch["event_targets"]["win_or_not"]
    )
    expected_total = 1.0 * policy_loss + 0.5 * place_loss + 0.4 * val_loss + 0.3 * ev_loss
    assert torch.allclose(losses["total"], expected_total, atol=1e-6)
    # Legacy [B,4]-vs-[B] class path stays byte-identical.
    legacy_logits = torch.randn(B, 4)
    legacy_batch = {
        "chosen_action_id": targets,
        "legal_mask": legal_mask,
        "placement_target": torch.randint(0, 4, (B,)),
    }
    legacy_out = {"policy_logits": logits, "placement_logits": legacy_logits}
    legacy_losses = compute_supervised_loss(
        legacy_out, legacy_batch, {"w_policy": 1.0, "w_placement": 0.5}
    )
    legacy_expected = policy_loss + 0.5 * torch.nn.functional.cross_entropy(
        legacy_logits, legacy_batch["placement_target"]
    )
    assert torch.allclose(legacy_losses["total"], legacy_expected, atol=1e-6)


def test_auxiliary_zero_weight_head_may_be_absent(tmp_path: Path) -> None:
    torch.manual_seed(2)
    B, A = 2, NUM_ACTIONS_SMALL
    logits = torch.randn(B, A)
    legal_mask = torch.ones(B, A, dtype=torch.bool)
    targets = torch.randint(0, A, (B,))
    batch = {"chosen_action_id": targets, "legal_mask": legal_mask}
    model_out = {"policy_logits": logits}
    # w_placement>0 but missing head should raise
    with pytest.raises(ContractError, match="w_placement"):
        compute_supervised_loss(model_out, batch, {"w_policy": 1.0, "w_placement": 1.0})
    # w_placement==0 missing head is OK
    losses = compute_supervised_loss(model_out, batch, {"w_policy": 1.0, "w_placement": 0.0})
    assert float(losses["total"].item()) == pytest.approx(
        float(masked_cross_entropy(logits, targets, legal_mask).item())
    )
    # w_event zero may be absent
    losses2 = compute_supervised_loss(
        model_out, batch, {"w_policy": 1.0, "w_event": {"win_or_not": 0.0}}
    )
    assert "total" in losses2

    # w_event>0 but missing should raise
    with pytest.raises(ContractError, match="w_event"):
        compute_supervised_loss(model_out, batch, {"w_policy": 1.0, "w_event": {"win_or_not": 0.5}})


def test_auxiliary_missing_placement_target_raises(tmp_path: Path) -> None:
    torch.manual_seed(3)
    B, A = 2, NUM_ACTIONS_SMALL
    logits = torch.randn(B, A)
    legal_mask = torch.ones(B, A, dtype=torch.bool)
    targets = torch.randint(0, A, (B,))
    model_out = {"policy_logits": logits, "placement_logits": torch.randn(B, 4)}
    batch = {"chosen_action_id": targets, "legal_mask": legal_mask}  # no placement_target
    with pytest.raises(ContractError, match="placement_target"):
        compute_supervised_loss(model_out, batch, {"w_policy": 1.0, "w_placement": 1.0})


def test_loop_config_value_defaults_and_bounds() -> None:
    cfg = TrainingLoopConfig()
    assert cfg.w_policy == 1.0
    assert cfg.w_placement == 0.0
    assert cfg.w_value == 0.0
    assert cfg.objective_weights()["w_value"] == 0.0
    cfg.validate()
    with pytest.raises(ContractError):
        TrainingLoopConfig(w_value=-0.1).validate()
    with pytest.raises(ContractError):
        TrainingLoopConfig(w_placement=-1.0).validate()
    with pytest.raises(ContractError):
        TrainingLoopConfig(w_policy=-1.0).validate()


def test_placement_target_zero_based_bounds() -> None:
    torch.manual_seed(11)
    B, A = 2, NUM_ACTIONS_SMALL
    logits = torch.randn(B, A)
    legal_mask = torch.ones(B, A, dtype=torch.bool)
    targets = torch.randint(0, A, (B,))
    pl_logits = torch.randn(B, 4, 4)
    # Boundary values 0 and 3 across all seats; 0-based target is
    # utility() 1..4 rank minus 1 (utility rank 1 -> target 0).
    pl_target = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    batch = {
        "chosen_action_id": targets,
        "legal_mask": legal_mask,
        "placement_target": pl_target,
    }
    model_out = {"policy_logits": logits, "placement_logits": pl_logits}
    losses = compute_supervised_loss(model_out, batch, {"w_policy": 1.0, "w_placement": 1.0})
    expected_place = torch.nn.functional.cross_entropy(
        pl_logits.reshape(-1, 4), pl_target.reshape(-1)
    )
    assert torch.allclose(losses["placement"], expected_place, atol=1e-6)


def test_placement_target_out_of_range_raises() -> None:
    torch.manual_seed(12)
    B, A = 2, NUM_ACTIONS_SMALL
    logits = torch.randn(B, A)
    legal_mask = torch.ones(B, A, dtype=torch.bool)
    targets = torch.randint(0, A, (B,))
    pl_logits = torch.randn(B, 4, 4)
    # 4 is valid under the utility() 1..4 convention but invalid 0-based.
    bad_hi = torch.zeros(B, 4, dtype=torch.long)
    bad_hi[0, 0] = 4
    with pytest.raises(ContractError, match="0-based"):
        compute_supervised_loss(
            {"policy_logits": logits, "placement_logits": pl_logits},
            {
                "chosen_action_id": targets,
                "legal_mask": legal_mask,
                "placement_target": bad_hi,
            },
            {"w_policy": 1.0, "w_placement": 1.0},
        )
    bad_lo = torch.zeros(B, 4, dtype=torch.long)
    bad_lo[1, 2] = -1
    with pytest.raises(ContractError, match="0-based"):
        compute_supervised_loss(
            {"policy_logits": logits, "placement_logits": pl_logits},
            {
                "chosen_action_id": targets,
                "legal_mask": legal_mask,
                "placement_target": bad_lo,
            },
            {"w_policy": 1.0, "w_placement": 1.0},
        )


# ---------------------------------------------------------------------------
# Model-output adapter and pre-forward gates
# ---------------------------------------------------------------------------


def test_model_forward_converts_model_output_dataclass() -> None:
    """Real-model ModelOutput maps onto the loss dict; dict path is identical."""
    from hydra2.models.model import ModelOutput
    from hydra2.models.schema import BASELINE_ACTION_COUNT as _FULL_ACTIONS
    from hydra2.training.loop_batch import _model_forward

    torch.manual_seed(0)
    batch_size = 2

    class StubModelOutputModel(nn.Module):
        def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
            n = int(batch["chosen_action_id"].shape[0])
            return ModelOutput(
                policy_logits=torch.randn(n, _FULL_ACTIONS),
                placement_logits=torch.randn(n, 4, 4),
                value_vector=torch.randn(n, 4),
                event_logits={},
                belief_logits={},
                diagnostics={},
                utility_id="test-utility",
                utility_manifest_hash="sha256:" + "0" * 64,  # type: ignore[arg-type]
                model_identity="sha256:" + "1" * 64,  # type: ignore[arg-type]
            )

    batch = {"chosen_action_id": torch.zeros(batch_size, dtype=torch.long)}
    converted = _model_forward(StubModelOutputModel(), batch)
    assert tuple(converted["policy_logits"].shape) == (batch_size, _FULL_ACTIONS)
    assert tuple(converted["placement_logits"].shape) == (batch_size, 4, 4)
    assert tuple(converted["value_vector"].shape) == (batch_size, 4)

    sentinel: dict[str, torch.Tensor] = {"policy_logits": torch.randn(batch_size, 4)}

    class StubDictModel(nn.Module):
        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return sentinel

    assert _model_forward(StubDictModel(), batch) is sentinel


def test_model_forward_rejects_all_false_legal_before_forward() -> None:
    """Pre-forward gate: all-false legal mask raises before any model call."""
    from hydra2.models.encoder import ActorTensorBatch
    from hydra2.training.loop_batch import _model_forward

    class _NoCallModel(nn.Module):
        action_count = 6792

        def evaluate(self, batch: object) -> object:
            raise AssertionError("must not reach forward")

        def forward(self, batch: object) -> object:  # type: ignore[override]
            raise AssertionError("must not reach forward")

    feats = {"history_event_kind": torch.zeros(2, 32, dtype=torch.int64)}
    bad = ActorTensorBatch(
        features=feats,
        history_mask=torch.ones(2, 32, dtype=torch.bool),
        legal_mask=torch.zeros(2, 6792, dtype=torch.bool),
        observation_hashes=("sha256:" + "0" * 64, "sha256:" + "0" * 64),
        actor_seats=torch.zeros(2, dtype=torch.int64),
    )
    with pytest.raises(ContractError, match="at least one legal"):
        _model_forward(_NoCallModel(), {"actor_batch": bad})


# ---------------------------------------------------------------------------
# Compiled kernel and validator parity with the eager loss
# ---------------------------------------------------------------------------


def test_compiled_supervised_loss_matches_eager_bitwise() -> None:
    """Inductor-fused kernel is bitwise-identical to eager (train-loop wiring).

    Guards the loop's compiled-loss fast path: the loop pre-validates
    eagerly, then runs the check-free kernel, so the test mirrors that
    split (validate once, compare eager kernel vs compiled kernel). Same
    values down to the last bit on both the plain and legal-only-smoothing
    paths, so compiling the loss is a pure speedup. Skipped where inductor
    is unavailable.
    """
    torch = pytest.importorskip("torch")
    try:
        import torch._inductor
    except ImportError:
        pytest.skip("inductor unavailable")
    torch.manual_seed(0)
    batch_size, width, legal = 8, 64, 5
    logits = torch.randn(batch_size, width)
    mask = torch.zeros(batch_size, width, dtype=torch.bool)
    mask[:, :legal] = True
    targets = torch.randint(0, legal, (batch_size,))
    model_out = {"policy_logits": logits}
    batch = {"chosen_action_id": targets, "legal_mask": mask}
    compiled = torch.compile(supervised_loss_kernel, fullgraph=False)
    for smoothing in (0.0, 0.03):
        weights = {"w_policy": 1.0, "label_smoothing": smoothing}
        validate_supervised_inputs(model_out, batch, weights)
        want = compute_supervised_loss(model_out, batch, weights)
        got = compiled(model_out, batch, weights)
        assert torch.equal(got["total"], want["total"])
        assert torch.equal(got["policy"], want["policy"])


def test_loss_validator_matches_public_errors() -> None:
    """Validator raises the identical error as the eager public loss."""
    torch.manual_seed(0)
    batch_size, width, legal = 4, 16, 5
    logits = torch.randn(batch_size, width)
    good_mask = torch.zeros(batch_size, width, dtype=torch.bool)
    good_mask[:, :legal] = True
    good_targets = torch.randint(0, legal, (batch_size,))

    def _check(model_out: dict, batch: dict, weights: dict) -> None:
        with pytest.raises(ContractError) as public_exc:
            compute_supervised_loss(model_out, batch, weights)
        with pytest.raises(ContractError) as validator_exc:
            validate_supervised_inputs(model_out, batch, weights)
        assert str(public_exc.value) == str(validator_exc.value)

    weights = {"w_policy": 1.0}
    base_out = {"policy_logits": logits}
    base_batch = {"chosen_action_id": good_targets, "legal_mask": good_mask}
    # All-false legal row.
    bad_mask = torch.zeros(batch_size, width, dtype=torch.bool)
    _check(base_out, {"chosen_action_id": good_targets, "legal_mask": bad_mask}, weights)
    # Out-of-range target.
    bad_targets = torch.full((batch_size,), width, dtype=torch.long)
    _check(base_out, {"chosen_action_id": bad_targets, "legal_mask": good_mask}, weights)
    # Illegal target (in range, masked out).
    illegal = torch.zeros(batch_size, dtype=torch.long)
    illegal[0] = legal  # row 0: only [0, legal) legal
    _check(base_out, {"chosen_action_id": illegal, "legal_mask": good_mask}, weights)
    # Missing keys.
    _check(base_out, {"legal_mask": good_mask}, weights)
    _check({}, base_batch, weights)


def test_nonfinite_logits_trip_total_gate() -> None:
    """CE-local gate deleted: corrupt logits still fail closed via the total gate."""
    torch.manual_seed(0)
    batch_size, width, legal = 4, 16, 5
    logits = torch.randn(batch_size, width)
    logits[0, 0] = float("inf")
    good_mask = torch.zeros(batch_size, width, dtype=torch.bool)
    good_mask[:, :legal] = True
    good_targets = torch.randint(0, legal, (batch_size,))
    model_out = {"policy_logits": logits}
    batch = {"chosen_action_id": good_targets, "legal_mask": good_mask}
    with pytest.raises(ContractError, match="non-finite"):
        compute_supervised_loss(model_out, batch, {"w_policy": 1.0})


@pytest.mark.gpu
def test_device_assert_trips_live_on_cuda() -> None:
    """Prod device-assert path fires (subprocess isolation: poison cannot leak).

    Runs the trip in a child process with the test gate scrubbed: a real
    violation must abort (never exit 42), with a device-side assert in
    stderr. CPU-semantics tests elsewhere pin the exact typed errors.
    """
    import os
    import subprocess
    import sys

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    code = (
        "import os;",
        "os.environ.pop('HYDRA2_DISABLE_DEVICE_ASSERTS', None);",
        "import torch;",
        "from hydra2.training.objectives_loss import _check_legal_rows;",
        "mask = torch.zeros(2, 4, dtype=torch.bool, device='cuda');",
        "_check_legal_rows(mask);",
        "torch.cuda.synchronize();",
        "raise SystemExit(42)",
    )
    env = {k: v for k, v in os.environ.items() if k != "HYDRA2_DISABLE_DEVICE_ASSERTS"}
    proc = subprocess.run(
        [sys.executable, "-c", "".join(code)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert proc.returncode != 42, f"device assert did not trip; stderr={proc.stderr[-2000:]}"
    assert "cudaErrorAssert" in proc.stderr or "device-side assert" in proc.stderr, (
        f"expected device-assert text; stderr={proc.stderr[-2000:]}"
    )


def test_aux_losses_fp32_output() -> None:
    """Aux losses compute in fp32 from bf16 inputs (CPU-only).

    Fails pre-pin: ``F.cross_entropy`` on bf16 returns bf16 (and the per-seat
    placement path inherits it).  The mse pin documents the exact-widening
    contract (promotion already yields fp32 there).  Error strings are pinned
    unchanged by ``test_loss_validator_matches_public_errors``.
    """
    import torch.nn.functional as functional

    from hydra2.training.objectives_loss import _generic_ce_loss, _generic_mse_loss

    torch.manual_seed(21)
    # Unmasked CE helper: fp32 dtype + finite + exact vs same-input fp32 math.
    ce_logits = (torch.randn(8, 5) * 2).to(torch.bfloat16)
    ce_targets = torch.randint(0, 5, (8,))
    ce_got = _generic_ce_loss(ce_logits, ce_targets, name="event[probe]")
    assert ce_got.dtype == torch.float32, f"expected fp32, got {ce_got.dtype}"
    assert torch.isfinite(ce_got).all()
    ce_ref = functional.cross_entropy(ce_logits.to(torch.float32), ce_targets.long())
    assert torch.equal(ce_got, ce_ref)
    assert torch.allclose(
        ce_got, functional.cross_entropy(ce_logits.float(), ce_targets), atol=1e-2, rtol=1e-2
    )
    # MSE helper: both sides widened exactly to fp32.
    mse_pred = (torch.randn(4, 4)).to(torch.bfloat16)
    mse_target = torch.randn(4, 4)
    mse_got = _generic_mse_loss(mse_pred, mse_target, name="value")
    assert mse_got.dtype == torch.float32, f"expected fp32, got {mse_got.dtype}"
    assert torch.isfinite(mse_got).all()
    assert torch.equal(
        mse_got, functional.mse_loss(mse_pred.to(torch.float32), mse_target.to(torch.float32))
    )
    # Per-seat placement path through the kernel with bf16 logits.
    batch_size, width = 4, 32
    policy_logits = torch.randn(batch_size, width)
    legal_mask = torch.zeros(batch_size, width, dtype=torch.bool)
    legal_mask[:, :8] = True
    batch = {
        "chosen_action_id": torch.zeros(batch_size, dtype=torch.long),
        "legal_mask": legal_mask,
        "placement_target": torch.randint(0, 4, (batch_size, 4)),
    }
    seat_logits = (torch.randn(batch_size, 4, 4)).to(torch.bfloat16)
    seat_losses = compute_supervised_loss(
        {"policy_logits": policy_logits, "placement_logits": seat_logits},
        batch,
        {"w_policy": 0.0, "w_placement": 1.0},
    )
    assert seat_losses["placement"].dtype == torch.float32, (
        f"expected fp32, got {seat_losses['placement'].dtype}"
    )
    assert torch.isfinite(seat_losses["placement"])
    seat_ref = functional.cross_entropy(
        seat_logits.to(torch.float32).reshape(-1, 4), batch["placement_target"].reshape(-1).long()
    )
    assert torch.equal(seat_losses["placement"], seat_ref)
    # Distribution placement path: fp32 softmax pin.
    dist_logits = (torch.randn(batch_size, 6)).to(torch.bfloat16)
    dist_target = functional.softmax(torch.randn(batch_size, 6), dim=-1)
    dist_losses = compute_supervised_loss(
        {"policy_logits": policy_logits, "placement_logits": dist_logits},
        {**batch, "placement_target": dist_target},
        {"w_policy": 0.0, "w_placement": 1.0},
    )
    assert dist_losses["placement"].dtype == torch.float32
    assert torch.isfinite(dist_losses["placement"])
    dist_ref = functional.mse_loss(
        functional.softmax(dist_logits.to(torch.float32), dim=-1), dist_target.float()
    )
    assert torch.allclose(dist_losses["placement"], dist_ref, atol=1e-6, rtol=1e-5)


def test_row_eval_primitives_reproduce_headline_means() -> None:
    """Row means of primitives equal compute_metrics NLL/top-k (exact pooling)."""
    gen = torch.Generator().manual_seed(20260926)
    batch, width = 48, 16
    logits = torch.randn(batch, width, generator=gen)
    targets = torch.randint(0, width, (batch,), generator=gen)
    legal = torch.ones(batch, width, dtype=torch.bool)
    legal[torch.randn(batch, generator=gen) < -1.0, :] = False
    legal[torch.arange(batch), targets] = True
    legal[:, 0] = True
    prim = row_eval_primitives(logits, targets, legal)
    assert set(prim.keys()) == {"row_nll", "hit1", "hit3", "hit5", "conf"}
    for key in ("row_nll", "hit1", "hit3", "hit5", "conf"):
        assert prim[key].device.type == "cpu"
        assert prim[key].shape == (batch,)
        assert torch.isfinite(prim[key]).all()
    ref = compute_metrics(logits, targets, legal)
    assert prim["row_nll"].mean().item() == pytest.approx(ref["masked_nll"])
    assert prim["hit1"].mean().item() == pytest.approx(ref["top1"])
    assert prim["hit3"].mean().item() == pytest.approx(ref["top3"])
    assert prim["hit5"].mean().item() == pytest.approx(ref["top5"])
    assert ((prim["conf"] >= 0.0) & (prim["conf"] <= 1.0)).all()
