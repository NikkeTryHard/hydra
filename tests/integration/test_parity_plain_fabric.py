"""GPU parity probes: one eager forward/backward/AdamW step through the
plain adapter; the SAME seeded step repeated through plain (seeded repeat
parity) within the frozen fp32 tolerance declared in hydra2.config.
Fabric is REMOVED (M3): the two-arm plain-vs-fabric comparison is replaced
by plain-vs-plain seeded repeat + bitwise repeat probes.
"""

from __future__ import annotations

import pytest
import torch

from hydra2.config import PARITY_ABS_TOL, PARITY_REL_TOL
from hydra2.runtime.plain import PlainPytorchAdapter
from hydra2.runtime.protocol import RuntimeSpec, build_runtime
from tests.conftest import (
    assert_states_bitwise_equal,
    capture_grads,
    make_batch,
    make_model_and_optimizer,
    run_supervised_steps,
    state_snapshot,
)

SEED = 20260822


def plain_spec(**overrides) -> RuntimeSpec:
    fields = {
        "adapter_id": "plain_pytorch",
        "device": "cuda",
        "precision": "fp32",
        "compile_mode": "eager",
        "fullgraph": False,
        "dynamic": None,
        "backward_pass_autocast": None,
    }
    fields.update(overrides)
    return RuntimeSpec(**fields)


@pytest.mark.gpu
class TestEagerPlainStep:
    def test_forward_backward_adamw_step_on_cuda(self, require_cuda):
        model, optimizer = make_model_and_optimizer(SEED)
        handle = build_runtime(
            adapter=PlainPytorchAdapter(),
            model=model,
            optimizer=optimizer,
            spec=plain_spec(device="cuda"),
        )
        x, y = make_batch(SEED + 1)
        before_step = next(handle.model.parameters()).detach().clone()
        losses = run_supervised_steps(handle, x, y, steps=1)
        assert len(losses) == 1
        assert losses[0] > 0.0
        # Parameters actually moved: AdamW applied its update on device.
        for param in handle.model.parameters():
            assert param.device.type == "cuda"
        assert not torch.equal(next(handle.model.parameters()).detach(), before_step)

    def test_plain_setup_returns_exact_objects(self, require_cuda):
        model, optimizer = make_model_and_optimizer(SEED)
        handle = build_runtime(
            adapter=PlainPytorchAdapter(),
            model=model,
            optimizer=optimizer,
            spec=plain_spec(device="cuda"),
        )
        assert handle.model is model
        assert handle.optimizer is optimizer


@pytest.mark.gpu
class TestSeededParity:
    def test_same_seed_step_parity_within_frozen_tolerance(self, require_cuda):
        x, y = make_batch(SEED + 2)

        def build() -> object:
            model, optimizer = make_model_and_optimizer(SEED)
            return build_runtime(
                adapter=PlainPytorchAdapter(),
                model=model,
                optimizer=optimizer,
                spec=plain_spec(device="cuda"),
            )

        plain_handle = build()
        repeat_handle = build()

        plain_losses = run_supervised_steps(plain_handle, x, y, steps=4)
        repeat_losses = run_supervised_steps(repeat_handle, x, y, steps=4)

        # Per-step losses parity (frozen tolerance).
        for step, (lp, lf) in enumerate(zip(plain_losses, repeat_losses, strict=True)):
            assert abs(lp - lf) <= PARITY_ABS_TOL + PARITY_REL_TOL * abs(lp), (
                f"loss diverged at step {step}: {lp} vs {lf}"
            )

        # One more backward pass: gradient parity before the final update.
        plain_loss, plain_grads = capture_grads(plain_handle, x, y)
        repeat_loss, repeat_grads = capture_grads(repeat_handle, x, y)
        assert abs(float(plain_loss) - float(repeat_loss)) <= PARITY_ABS_TOL + PARITY_REL_TOL * abs(
            float(plain_loss)
        )
        assert set(plain_grads) == set(repeat_grads)
        for name in plain_grads:
            assert torch.allclose(
                plain_grads[name],
                repeat_grads[name].to(plain_grads[name].device),
                rtol=PARITY_REL_TOL,
                atol=PARITY_ABS_TOL,
            ), f"gradient parity violated for {name}"

        # Final update parity after stepping both handles.
        plain_handle.optimizer.step()
        repeat_handle.optimizer.step()
        plain_state = state_snapshot(plain_handle.model)
        repeat_state = state_snapshot(repeat_handle.model)
        for key in plain_state:
            assert torch.allclose(
                plain_state[key],
                repeat_state[key],
                rtol=PARITY_REL_TOL,
                atol=PARITY_ABS_TOL,
            ), f"update parity violated for {key}"

    def test_identical_seeds_give_bitwise_identical_repeats(self, require_cuda):
        """Same adapter, same seed twice -> bitwise identical trajectory."""
        x, y = make_batch(SEED + 3)
        snapshots = []
        for _ in range(2):
            model, optimizer = make_model_and_optimizer(SEED)
            handle = build_runtime(
                adapter=PlainPytorchAdapter(),
                model=model,
                optimizer=optimizer,
                spec=plain_spec(device="cuda"),
            )
            run_supervised_steps(handle, x, y, steps=3)
            snapshots.append(state_snapshot(handle.model))
        assert_states_bitwise_equal(snapshots[0], snapshots[1], context="plain-repeat")
