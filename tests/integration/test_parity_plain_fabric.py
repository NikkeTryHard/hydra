"""GPU parity probes: one eager forward/backward/AdamW step through the
plain adapter; the SAME seeded step through the standalone Fabric adapter;
model/loss/gradient/update parity within the frozen fp32 tolerance declared
in hydra2.config.
"""

from __future__ import annotations

import pytest
import torch

from hydra2.config import PARITY_ABS_TOL, PARITY_REL_TOL
from hydra2.runtime.fabric import FabricRuntimeAdapter
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


def fabric_spec(**overrides) -> RuntimeSpec:
    fields = {
        "adapter_id": "fabric_2.6.5",
        "device": "cuda",
        "precision": "fp32",
        "compile_mode": "eager",
        "fullgraph": False,
        "dynamic": None,
        "backward_pass_autocast": None,
    }
    fields.update(overrides)
    return RuntimeSpec(**fields)


def plain_spec(**overrides) -> RuntimeSpec:
    spec = fabric_spec(**{"adapter_id": "plain_pytorch", **overrides})
    return spec


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
        after_step = next(handle.model.parameters()).detach().clone()
        assert not torch.equal(after_step.cpu(), before_step.cpu()), (
            "AdamW step did not change the first parameter tensor"
        )

    def test_plain_setup_returns_exact_objects(self, require_cuda):
        model, optimizer = make_model_and_optimizer(SEED)
        handle = build_runtime(
            adapter=PlainPytorchAdapter(),
            model=model,
            optimizer=optimizer,
            spec=plain_spec(device="cuda"),
        )
        assert handle.model is model  # same object rebound to device in-place
        assert handle.optimizer is optimizer


@pytest.mark.gpu
class TestSeededParity:
    def test_same_seed_step_parity_within_frozen_tolerance(self, require_cuda):
        x, y = make_batch(SEED + 2)

        def build(adapter_id):
            cls = PlainPytorchAdapter if adapter_id == "plain" else FabricRuntimeAdapter
            model, optimizer = make_model_and_optimizer(SEED)
            spec = (
                plain_spec(device="cuda") if adapter_id == "plain" else fabric_spec(device="cuda")
            )
            return build_runtime(adapter=cls(), model=model, optimizer=optimizer, spec=spec)

        plain_handle = build("plain")
        fabric_handle = build("fabric")

        plain_losses = run_supervised_steps(plain_handle, x, y, steps=4)
        fabric_losses = run_supervised_steps(fabric_handle, x, y, steps=4)

        # Per-step losses parity (frozen tolerance).
        for step, (lp, lf) in enumerate(zip(plain_losses, fabric_losses, strict=True)):
            assert abs(lp - lf) <= PARITY_ABS_TOL + PARITY_REL_TOL * abs(lp), (
                f"loss diverged at step {step}: {lp} vs {lf}"
            )

        # One more backward pass: gradient parity before the final update.
        plain_loss, plain_grads = capture_grads(plain_handle, x, y)
        fabric_loss, fabric_grads = capture_grads(fabric_handle, x, y)
        assert abs(float(plain_loss) - float(fabric_loss)) <= PARITY_ABS_TOL + PARITY_REL_TOL * abs(
            float(plain_loss)
        )
        assert set(plain_grads) == set(fabric_grads)
        for name in plain_grads:
            assert torch.allclose(
                plain_grads[name],
                fabric_grads[name].to(plain_grads[name].device),
                rtol=PARITY_REL_TOL,
                atol=PARITY_ABS_TOL,
            ), f"gradient parity violated for {name}"

        # Final update parity after stepping both handles.
        plain_handle.optimizer.step()
        fabric_handle.optimizer.step()
        plain_state = state_snapshot(plain_handle.model)
        fabric_state = state_snapshot(_unwrap(fabric_handle.model))
        for key in plain_state:
            assert torch.allclose(
                plain_state[key],
                fabric_state[key],
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
                adapter=FabricRuntimeAdapter(),
                model=model,
                optimizer=optimizer,
                spec=fabric_spec(device="cuda"),
            )
            run_supervised_steps(handle, x, y, steps=3)
            snapshots.append(state_snapshot(_unwrap(handle.model)))
        assert_states_bitwise_equal(snapshots[0], snapshots[1], context="fabric-repeat")


def _unwrap(model):
    """Peek through Fabric's module wrapper if present."""
    return getattr(model, "_fabric_module", getattr(model, "module", model))
